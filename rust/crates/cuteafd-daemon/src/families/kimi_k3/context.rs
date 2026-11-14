//! Once-per-rank context storage. Decode lanes and gathered prefill borrow the
//! same owner; no context allocation belongs to a request or graph variant.
use crate::shared::{memory::{device::{Allocation, Stream}, Tag}, peer_split::{context::ContextExchange, RankDevice}};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::families::kimi_k3::context::ContextLayout;
use std::{collections::BTreeMap, rc::Rc};

pub(crate) struct Programs<'a> {
    partial: Option<([[cuteafd_ffi::KimiMlaPartial<'a>; 2]; 2],
        cuteafd_loader::families::kimi_k3::context::TilePolicy)>,
    pub combine: cuteafd_ffi::programs::BoundProgram<'a>,
    pub gather: Option<cuteafd_ffi::programs::BoundProgram<'a>>,
    pub sms: usize,
}

impl<'a> Programs<'a> {
    pub(crate) fn load(programs: &'a cuteafd_ffi::programs::Programs<'a>,
        format: cuteafd_core::MlaCacheFormat) -> Result<Self> {
        Self::load_variant(programs, format, false)
    }

    pub(crate) fn load_for_config(programs: &'a cuteafd_ffi::programs::Programs<'a>,
        format: cuteafd_core::MlaCacheFormat,
        cfg: &cuteafd_loader::families::kimi_k3::config::KimiK3Config) -> Result<Self> {
        Self::load_variant(programs, format, cfg.dsa.is_some())
    }

    fn load_variant(programs: &'a cuteafd_ffi::programs::Programs<'a>,
        format: cuteafd_core::MlaCacheFormat, dsa: bool) -> Result<Self> {
        if dsa {
            let combine = programs.spec("kimik31_lse_combine2")?;
            ensure!(combine.scalar_kinds == b"i" && combine.geometry["heads"] == 48
                && combine.geometry["has_sink"] == false, "Kimi DSA context combine ABI differs");
            let name = format!("kimik31_paged_staging_gather{}", format.program_suffix());
            let gather = programs.spec(&name)?;
            ensure!(gather.scalar_kinds == b"iii" && gather.geometry["row_bytes"] == format.record_bytes()
                && gather.geometry["page_rows"] == 64 && gather.geometry["page_bytes"] == 64 * format.record_bytes(),
                "Kimi DSA context gather ABI differs");
            return Ok(Self { partial: None,
                combine: programs.bound("kimik31_lse_combine2", &["o0", "l0", "o1", "l1", "sink", "out"] )?,
                gather: Some(programs.bound(&name, &["pool", "table", "staging"] )?),
                sms: programs.library().sm_count()? });
        }
        let tag = format.program_suffix();
        let combine = programs.spec("kimik3_lse_combine2")?;
        ensure!(combine.scalar_kinds == b"i" && combine.geometry["heads"] == 48
            && combine.geometry["has_sink"] == false, "Kimi context combine ABI differs");
        let gather = programs.spec(&format!("kimik3_paged_staging_gather{tag}"))?;
        ensure!(gather.scalar_kinds == b"iii" && gather.geometry["row_bytes"] == format.record_bytes()
            && gather.geometry["page_rows"] == 64 && gather.geometry["page_bytes"] == 64 * format.record_bytes(),
            "Kimi context gather ABI differs");
        let mut tiles = None;
        let partial = [8, 16].map(|tile| -> Result<_> {
            let ranges = [0, 48].map(|begin| -> Result<_> {
                let name = format!("kimik3_mla_partial{tag}_h{begin}_t{tile}_m64");
                let geometry = &programs.spec(&name)?.geometry;
                ensure!(geometry["heads_per_tile"] == tile, "Kimi context partial tile differs");
                let policy = cuteafd_loader::families::kimi_k3::context::TilePolicy::from_geometry(geometry)?;
                ensure!(tiles.as_ref().is_none_or(|expected| *expected == policy),
                    "Kimi context partial tile policies differ");
                tiles = Some(policy);
                Ok(cuteafd_ffi::KimiMlaPartial::load(programs, &name, begin, 64, format)?)
            });
            let [first, second] = ranges;
            Ok([first?, second?])
        });
        let [first, second] = partial;
        Ok(Self {
            partial: Some(([first?, second?], tiles.context("Kimi context partial tile policy missing")?)),
            combine: programs.bound("kimik3_lse_combine2", &["o0", "l0", "o1", "l1", "sink", "out"] )?,
            gather: Some(programs.bound(&format!("kimik3_paged_staging_gather{tag}"), &["pool", "table", "staging"] )?),
            sms: programs.library().sm_count()?,
        })
    }

    pub(crate) fn partial(&self, rows: usize, rank: usize) -> Result<&cuteafd_ffi::KimiMlaPartial<'a>> {
        let (partial, tiles) = self.partial.as_ref().context("Kimi DSA context requires selected-token partials")?;
        partial[tiles.index(rows)?].get(rank).context("Kimi context partial head rank")
    }

    pub(crate) fn layout(programs: &cuteafd_ffi::programs::Programs<'_>,
        format: cuteafd_core::MlaCacheFormat, extent: usize, rows: usize, banks: usize,
        cfg: &cuteafd_loader::families::kimi_k3::config::KimiK3Config) -> Result<ContextLayout> {
        let mut names = cuteafd_loader::families::kimi_k3::context::program_names_for_config(format, cfg);
        names.extend(programs.names().filter(|name| name.starts_with("dense_trellis_k6_trellis_"))
            .map(str::to_owned));
        let entries = names.iter().map(|name| -> Result<_> {
            let spec = programs.spec(name)?;
            Ok(serde_json::json!({"name": name, "geometry": spec.geometry,
                "scratch_bytes_at_capacity": spec.scratch}))
        }).collect::<Result<Vec<_>>>()?;
        ContextLayout::from_manifest_for_config(&serde_json::json!({"programs": entries}), format, extent, rows, banks, cfg)
    }
}

pub(crate) struct Rank<'a> {
    streams: Vec<Rc<Stream<'a>>>,
    workspace: BTreeMap<&'static str, Allocation<'a>>,
    staging: [Allocation<'a>; 2],
    layout: ContextLayout,
}

impl Drop for Rank<'_> {
    fn drop(&mut self) {
        for stream in &self.streams { stream.drain_or_abort("Kimi context storage readers"); }
    }
}

impl<'a> Rank<'a> {
    fn allocate(layout: &ContextLayout, streams: Vec<Rc<Stream<'a>>>) -> Result<Self> {
        let device = streams.first().context("Kimi context needs retained streams")?.device;
        ensure!(streams.iter().all(|s| s.device.id == device.id
            && std::ptr::eq(s.device.library, device.library)), "Kimi context stream device mismatch");
        let workspace = layout.workspace().into_iter().map(|(name, bytes)| -> Result<_> {
            let allocation = Allocation::new(device, bytes, Tag::workspace("kimi/context"))?;
            ensure!(allocation.buffer.bytes == bytes && allocation.buffer.device_id == device.id,
                "Kimi context {name} allocation differs from admission");
            Ok((name, allocation))
        }).collect::<Result<BTreeMap<_, _>>>()?;
        let buffers = layout.buffers();
        let bytes = usize::try_from(buffers.compiled_extent.div_ceil(buffers.staging_unit_rows)
            .checked_mul(buffers.staging_unit_bytes).context("Kimi context staging overflow")?)?;
        let staging = [Allocation::new(device, bytes, Tag::workspace("kimi/context-staging"))?,
            Allocation::new(device, bytes, Tag::workspace("kimi/context-staging"))?];
        let owner = Self { streams, workspace, staging, layout: layout.clone() };
        let (_, stride) = owner.layout.metadata_extent("context slots", 0)?;
        let banks = owner.buffer("context slots")?.bytes / stride;
        for bank in 0..banks {
            for name in ["context slots", "compact page tables", "shard lengths", "context offsets", "gather tables", "commit blocks", "active splits"] {
                let buffer = owner.metadata(name, bank)?;
                for stream in &owner.streams {
                    stream.device.run(|| {
                        // SAFETY: fixed admitted destination and exact stream survive
                        // all graph variants. Reserve pinned upload storage at startup.
                        unsafe { stream.device.library.reserve_h2d_ordered(buffer, stream.raw) }
                    })?;
                }
            }
        }
        if layout.dsa() {
            for bank in 0..banks {
                let buffer = owner.metadata("DSA merged lengths", bank)?;
                for stream in &owner.streams {
                    stream.device.run(|| {
                        // SAFETY: reserve each fixed per-unit destination before
                        // request-time enqueue-only metadata uploads use it.
                        unsafe { stream.device.library.reserve_h2d_ordered(buffer, stream.raw) }
                    })?;
                }
            }
            // Mask candidate tails before the first live selection.
            for (name, bits) in [("DSA global scores", f32::NEG_INFINITY.to_bits()),
                ("DSA global indices", u32::MAX)] {
                let buffer = owner.buffer(name)?;
                let bytes = bits.to_le_bytes().repeat(buffer.bytes / 4);
                for stream in &owner.streams {
                    stream.device.run(|| {
                        // SAFETY: startup owns this fixed destination and ordered
                        // staging retains the host bytes until the upload lands.
                        unsafe { stream.device.library.copy_h2d_ordered(buffer, &bytes, stream.raw) }
                    })?;
                }
            }
            for stream in &owner.streams { stream.drain()?; }
        }
        Ok(owner)
    }

    pub(crate) fn buffer(&self, name: &str) -> Result<CuteafdDeviceBuffer> {
        self.workspace.get(name).map(|a| a.buffer).with_context(|| format!("unknown Kimi context buffer {name}"))
    }

    pub(crate) fn metadata(&self, name: &str, bank: usize) -> Result<CuteafdDeviceBuffer> {
        let (offset, bytes) = self.layout.metadata_extent(name, bank)?;
        crate::shared::prefix::view(self.buffer(name)?, offset, bytes)
    }

    pub(crate) fn upload(&self, name: &str, bank: usize, bytes: &[u8], stream: &Stream<'_>) -> Result<()> {
        ensure!(self.streams.iter().any(|reader| std::ptr::eq(reader.as_ref(), stream)),
            "Kimi context upload stream was not retained at startup");
        let destination = self.metadata(name, bank)?;
        ensure!(bytes.len() <= destination.bytes, "Kimi context metadata exceeds admission");
        stream.device.run(|| {
            // SAFETY: startup reserved this exact destination/stream. Its owner
            // drains queued readers; ordered staging retains the host source.
            unsafe { stream.device.library.copy_h2d_ordered(destination, bytes, stream.raw) }
        })
    }

    pub(crate) fn decode(&self) -> &cuteafd_loader::families::kimi_k3::context::DecodeBuckets { self.layout.decode() }

    pub(crate) fn extent(&self) -> usize { self.layout.buffers().compiled_extent as usize }

    pub(crate) fn compact_width(&self) -> usize { self.extent().div_ceil(64).div_ceil(2) }

    pub(crate) fn check_reader(&self, stream: &Rc<Stream<'a>>) -> Result<()> {
        ensure!(self.streams.iter().any(|reader| Rc::ptr_eq(reader, stream)),
            "Kimi context binding differs from retained compute stream");
        Ok(())
    }

    pub(crate) fn partial(&self, peer: bool, rows: usize) -> Result<crate::shared::peer_split::attention::Partial> {
        ensure!((1..=64).contains(&rows), "Kimi partial output rows");
        let buffer = self.buffer(if peer { "peer-head partial" } else { "own-head partial" })?;
        Ok(crate::shared::peer_split::attention::Partial { values: buffer.ptr,
            lse: crate::shared::prefix::view(buffer, 64 * 48 * 512 * 2, rows * 48 * 4)?.ptr })
    }

    pub(crate) fn staging(&self, slot: usize) -> Result<CuteafdDeviceBuffer> {
        self.staging.get(slot).map(|a| a.buffer).context("Kimi context staging slot")
    }

    fn allocation_bytes(&self) -> [usize; 2] {
        [self.workspace.values().map(|a| a.buffer.bytes).sum(), self.staging.iter().map(|a| a.buffer.bytes).sum()]
    }
}

pub(crate) struct ContextOwner<'a> {
    pub(crate) ranks: [Rc<Rank<'a>>; 2],
    // Must drop before the engine's peer abort words.
    pub(crate) exchange: ContextExchange<'a>,
    caches: Option<[Rc<super::mla::CacheBank<'a>>; 2]>,
}

impl Drop for ContextOwner<'_> {
    fn drop(&mut self) {
        for rank in &self.ranks {
            for stream in &rank.streams { stream.drain_or_abort("Kimi context compute readers"); }
        }
    }
}

impl<'a> ContextOwner<'a> {
    /// Arithmetic and peer flags share each exact rank compute stream, as in
    /// GLM/V4. Joining both computes onto Kimi's ordinary exchange stream from
    /// serial rank callbacks would introduce a cross-rank event cycle.
    pub(crate) fn new(library: &'a NativeLibrary, layout: &ContextLayout,
        compute_streams: [Rc<Stream<'a>>; 2], abort: Option<[*const u32; 2]>) -> Result<Self> {
        ensure!(compute_streams.iter().all(|stream| std::ptr::eq(stream.device.library, library)),
            "Kimi context compute library mismatch");
        ensure!(compute_streams[0].device.id != compute_streams[1].device.id,
            "Kimi context needs distinct rank devices");
        let ranks = compute_streams.each_ref().map(|stream| RankDevice {
            device: stream.device.id, stream: stream.raw });
        let exchange = compute_streams[0].device.run(||
            ContextExchange::new(library, ranks, &layout.buffers(), abort))?;
        let [r0, r1] = compute_streams;
        let owner = Self { ranks: [Rc::new(Rank::allocate(layout, vec![r0])?),
            Rc::new(Rank::allocate(layout, vec![r1])?)], exchange, caches: None };
        owner.check_charges(layout)?;
        Ok(owner)
    }

    pub(crate) fn retain_caches(&mut self, caches: [Rc<super::mla::CacheBank<'a>>; 2]) -> Result<()> {
        ensure!(self.caches.is_none(), "Kimi context caches already bound");
        for rank in 0..2 {
            ensure!(caches[rank].parity() && caches[rank].format() == caches[0].format()
                && caches[rank].pages() == caches[0].pages(), "Kimi context cache ownership differs");
            caches[rank].check_reader(0, &self.ranks[rank].streams[0])?;
        }
        // Drop drains both exact reader streams before either remote source bank
        // can disappear, including when its local lane has already been dropped.
        self.caches = Some(caches);
        Ok(())
    }

    pub(crate) fn cache(&self, rank: usize, layer: usize) -> Result<CuteafdDeviceBuffer> {
        self.caches.as_ref().context("Kimi context cache readers not retained")?
            .get(rank).context("Kimi context cache rank")?.layer(layer)
    }

    /// Check actual native buffer extents, never a second estimate of the formula.
    pub(crate) fn check_charges(&self, layout: &ContextLayout) -> Result<()> {
        let storage = self.exchange.allocation_buffers();
        for rank in 0..2 {
            let [workspace, staging] = self.ranks[rank].allocation_bytes();
            for charge in layout.buffers().demands()?.iter().filter(|d| usize::from(d.gpu) == rank) {
                let actual = match charge.group.as_str() {
                    "context workspace" => workspace,
                    "context staging" => staging,
                    "context exchange" => storage[rank].bytes,
                    _ => anyhow::bail!("unknown Kimi context allocation charge"),
                };
                ensure!(actual as u64 == charge.bytes,
                    "Kimi rank {rank} {} allocation {actual} differs from charge {}", charge.group, charge.bytes);
            }
        }
        Ok(())
    }
}

/// Bounded runtime splits; live history updates metadata, not graph selection.
pub(crate) use cuteafd_loader::families::kimi_k3::context::SPLIT_BUCKETS;

pub(crate) fn split_bucket(owned_pages: usize, rows: usize, sms: usize) -> Result<usize> {
    ensure!((1..=64).contains(&rows) && sms > 0, "invalid Kimi split work geometry");
    let wave = sms.div_ceil(rows * 6);
    let work = owned_pages.div_ceil(8);
    let traversal = owned_pages.div_ceil(64);
    let wanted = wave.max(traversal).min(work).clamp(2, 16);
    Ok(SPLIT_BUCKETS.into_iter().find(|&bucket| bucket >= wanted).unwrap())
}

/// Per-query causal tables contain only local page indices. Even a row whose
/// current token belongs to the peer still attends its own earlier history.
#[derive(Clone)]
pub(crate) struct DecodeRows {
    pub rows: usize,
    pub width: usize,
    pub owned_pages: usize,
    slots: Vec<u8>,
    tables: Vec<u8>,
    lengths: Vec<u8>,
    offsets: Vec<u8>,
}

impl DecodeRows {
    pub(crate) fn new(rank: usize, sequences: &[super::device::SequenceRows<'_>], extent: usize,
        slots: usize, pool: &cuteafd_engine::prefix::RefPagePool) -> Result<Self> {
        Self::with_capacity(rank, sequences, extent, slots, pool, 64)
    }

    pub(crate) fn gathered(rank: usize, sequences: &[super::device::SequenceRows<'_>], extent: usize,
        slots: usize, pool: &cuteafd_engine::prefix::RefPagePool) -> Result<Self> {
        Self::with_capacity(rank, sequences, extent, slots, pool, 4096)
    }

    fn with_capacity(rank: usize, sequences: &[super::device::SequenceRows<'_>], extent: usize,
        slots: usize, pool: &cuteafd_engine::prefix::RefPagePool, capacity: usize) -> Result<Self> {
        use crate::shared::peer_split::context::{local_pages, shard_rows};
        ensure!(rank < 2 && pool.is_parity(), "Kimi compact metadata needs two-rank parity pages");
        let rows = super::device::validate_sequence_pages(sequences, capacity, slots, extent, pool)?;
        let width = extent.div_ceil(64).div_ceil(2);
        let mut result = Self { rows, width, owned_pages: 0, slots: Vec::with_capacity(rows * 8),
            tables: vec![0; rows * width * 4], lengths: Vec::with_capacity(rows * 4),
            offsets: (0..=rows as i32).flat_map(i32::to_le_bytes).collect() };
        let mut row = 0;
        for sequence in sequences {
            let end = sequence.start.checked_add(sequence.rows).context("Kimi compact frontier overflow")?;
            ensure!(sequence.rows > 0 && end <= extent && end.div_ceil(64) <= sequence.pages.len(),
                "Kimi compact sequence outside admitted context/pages");
            let pages = &sequence.pages[..end.div_ceil(64)];
            // Validate both parities even when this rank has no visible pages.
            let all = [local_pages(pages, 0)?, local_pages(pages, 1)?];
            for offset in 0..sequence.rows {
                let position = sequence.start + offset;
                let visible = shard_rows(position + 1, 64, rank);
                let count = visible.div_ceil(64);
                result.owned_pages = result.owned_pages.max(count);
                let physical = if position / 64 % 2 == rank {
                    i64::from(pages[position / 64] >> 1) * 64 + (position % 64) as i64
                } else { -1 };
                result.slots.extend_from_slice(&physical.to_le_bytes());
                result.lengths.extend_from_slice(&(visible as i32).to_le_bytes());
                for (column, page) in all[rank][..count].iter().enumerate() {
                    let at = (row * width + column) * 4;
                    result.tables[at..at + 4].copy_from_slice(&page.to_le_bytes());
                }
                row += 1;
            }
        }
        Ok(result)
    }

    pub(crate) fn pad(&mut self, bucket: usize) -> Result<()> {
        ensure!((self.lengths.len() / 4..=64).contains(&bucket), "Kimi context decode padding extent");
        self.slots.resize(bucket * 8, 0xff);
        self.tables.resize(bucket * self.width * 4, 0);
        self.lengths.resize(bucket * 4, 0);
        self.offsets = (0..=bucket as i32).flat_map(i32::to_le_bytes).collect();
        Ok(())
    }

    pub(crate) fn uploads(&self) -> [(&'static str, &[u8]); 4] {
        [("context slots", &self.slots), ("compact page tables", &self.tables), ("shard lengths", &self.lengths),
            ("context offsets", &self.offsets)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_decode_metadata_is_causal_and_writes_only_its_owner() -> Result<()> {
        use super::super::device::SequenceRows;
        let mut pool = cuteafd_engine::prefix::RefPagePool::parity(16, 64, 0);
        pool.alloc_for(0..16)?;
        let pages = [14, 9, 4, 3, 12];
        let second = [4, 3, 12, 5];
        let sequences = [SequenceRows { slot: 0, start: 62, rows: 4, pages: &pages },
            SequenceRows { slot: 1, start: 190, rows: 4, pages: &second }];
        let decode_i64 = |bytes: &[u8]| bytes.chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        let decode_i32 = |bytes: &[u8]| bytes.chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        let a = DecodeRows::new(0, &sequences, 320, 2, &pool)?;
        let b = DecodeRows::new(1, &sequences, 320, 2, &pool)?;
        assert_eq!(a.width, 3);
        assert_eq!(decode_i64(&a.slots), [510, 511, -1, -1, 446, 447, -1, -1]);
        assert_eq!(decode_i64(&b.slots), [-1, -1, 256, 257, -1, -1, 128, 129]);
        assert_eq!(decode_i32(&a.lengths), [63, 64, 64, 64, 127, 128, 128, 128]);
        assert_eq!(decode_i32(&b.lengths), [0, 0, 1, 2, 64, 64, 65, 66]);
        assert_eq!(&decode_i32(&a.tables)[12..15], [2, 6, 0]);
        assert_eq!(&decode_i32(&b.tables)[18..21], [1, 2, 0]);
        let empty_peer = DecodeRows::new(1, &[SequenceRows { slot: 0, start: 0, rows: 1, pages: &pages }], 320, 2, &pool)?;
        assert_eq!(decode_i32(&empty_peer.lengths), [0]);
        assert!(empty_peer.tables.iter().all(|&b| b == 0));
        assert!(DecodeRows::new(0, &[SequenceRows { slot: 0, start: 0, rows: 1, pages: &[1] }], 320, 2, &pool).is_err());
        let aliases = [SequenceRows { slot: 0, start: 62, rows: 4, pages: &pages },
            SequenceRows { slot: 1, start: 62, rows: 4, pages: &pages }];
        assert!(DecodeRows::new(0, &aliases, 320, 2, &pool).is_err());
        pool.release(&[14]);
        assert!(DecodeRows::new(0, &sequences, 320, 2, &pool).is_err());
        Ok(())
    }

    #[test]
    fn padded_decode_rows_preserve_live_metadata_and_mask_inactive_rows() -> Result<()> {
        use super::super::device::SequenceRows;
        let mut pool = cuteafd_engine::prefix::RefPagePool::parity(4, 64, 0);
        pool.alloc_for(0..4)?;
        let decode = cuteafd_loader::families::kimi_k3::context::DecodeBuckets::from_manifest(
            &serde_json::json!({"programs":[{"name":"dense_trellis_k6_trellis_mla_q_a_mcg",
                "geometry":{"bits":6,"decode_route_thresholds":[8,16,32,48]}}]}))?;
        for rank in 0..2 { for rows in 1..=64 {
            let bucket = decode.bucket(rows)?;
            assert!(bucket >= rows && decode.plain().contains(&bucket));
            let mut metadata = DecodeRows::new(rank,
                &[SequenceRows { slot: 0, start: 0, rows, pages: &[0] }], 256, 1, &pool)?;
            let original = [metadata.slots.clone(), metadata.tables.clone(), metadata.lengths.clone()];
            let owned = metadata.owned_pages;
            metadata.pad(bucket)?;
            assert_eq!(metadata.rows, rows);
            assert_eq!(metadata.owned_pages, owned);
            for ((_, padded), live) in metadata.uploads().into_iter().zip(&original) {
                assert_eq!(&padded[..live.len()], live);
            }
            assert!(metadata.slots[rows * 8..].iter().all(|&byte| byte == 0xff));
            assert!(metadata.tables[rows * metadata.width * 4..].iter().all(|&byte| byte == 0));
            assert!(metadata.lengths[rows * 4..].iter().all(|&byte| byte == 0));
            assert_eq!(metadata.offsets.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>(), (0..=bucket as i32).collect::<Vec<_>>());
            metadata.pad(bucket)?;
            if bucket > rows { assert!(metadata.pad(rows).is_err()); }
            assert!(metadata.pad(65).is_err());
        }}
        assert!(decode.bucket(0).is_err());
        assert!(decode.bucket(65).is_err());
        Ok(())
    }

    #[test]
    fn live_split_work_selects_only_bounded_startup_variants() -> Result<()> {
        assert_eq!(split_bucket(0, 1, 188)?, 2);
        assert_eq!(split_bucket(16, 1, 188)?, 2);
        assert_eq!(split_bucket(128, 1, 48)?, 8);
        assert_eq!(split_bucket(2048, 1, 188)?, 16);
        assert_eq!(split_bucket(2048, 16, 188)?, 16);
        for rows in [1, 16, 64] { for pages in [0, 1, 16, 128, 512, 2048] {
            assert!(SPLIT_BUCKETS.contains(&split_bucket(pages, rows, 170)?));
        }}
        assert!(split_bucket(128, 0, 188).is_err());
        assert!(split_bucket(128, 1, 0).is_err());
        Ok(())
    }

    #[test]
    fn context_native_allocations_equal_planner_charges_on_each_rank() -> Result<()> {
        use cuteafd_core::MlaCacheFormat;
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        let root = std::path::PathBuf::from(std::env::var_os("HOME").context("HOME")?)
            .join(".cache/cuteafd/builds/kimi-context-owner-fixtures");
        let fixture = Fixture::build_in(&root)?;
        for format in [MlaCacheFormat::Bf16, MlaCacheFormat::Fp8] {
            let tag = format.program_suffix();
            let mut programs = Vec::new();
            for head in [0, 48] { for tile in [8, 16] {
                programs.push(serde_json::json!({
                    "name":format!("kimik3_mla_partial{tag}_h{head}_t{tile}_m64"),
                    "geometry":{"num_splits":16,"runtime_splits":true,"device_splits":true,"fixed_split_grid":16,"lse_base":2,
                        "heads_per_tile":tile,"context_heads_per_tile":(1..=64).map(|rows| if rows < 4 {8} else {16}).collect::<Vec<_>>()},
                    "scratch_bytes_at_capacity":{"scratch":64*48*16*(512*2+4)}}));
            }}
            programs.extend([serde_json::json!({"name":"kimik3_lse_combine2"}),
                serde_json::json!({"name":format!("kimik3_paged_staging_gather{tag}")} )]);
            let manifest = serde_json::json!({"programs":programs});
            for extent in [8192, 1_048_576] {
                // Ordered upload reservations live with the library, not the owner.
                // Isolate cases so the fixture's bounded event pool does not accumulate.
                let library = fixture.load()?;
                fixture.extent_only(&library)?;
                let layout = ContextLayout::from_manifest(&manifest, format, extent, 2752, 3)?;
                let streams = [0, 1].map(|id| Rc::new(Stream::new(
                    crate::shared::memory::device::Device { library: &library, id }).unwrap()));
                let before = fixture.mapped_bytes(&library)?;
                let mut owner = ContextOwner::new(&library, &layout, streams.clone(), None)?;
                for rank in 0..2 {
                    assert_eq!(owner.ranks[rank].streams.len(), 1);
                    assert!(Rc::ptr_eq(&owner.ranks[rank].streams[0], &streams[rank]));
                }
                let other = Stream::new(streams[0].device)?;
                assert!(owner.ranks[0].upload("context slots", 0, &[0; 8], &other).is_err());
                owner.check_charges(&layout)?;
                let charged: u64 = layout.buffers().demands()?.iter().map(|d| d.bytes).sum();
                assert_eq!((fixture.mapped_bytes(&library)? - before) as u64, charged,
                    "native allocator extents include exactly both rank charges");
                let cfg = cuteafd_loader::families::kimi_k3::config::KimiK3Config::from_hf(
                    &cuteafd_loader::plan::testing::kimi_k3_config())?;
                let mut placement = super::super::admission::admit(
                    super::super::admission::tests::inputs(&cfg))?.placement;
                placement.pool_tokens = 256;
                for (kind, layer) in cfg.attention.iter().zip(&mut placement.layers) {
                    if *kind == cuteafd_loader::families::kimi_k3::config::KimiAttention::Mla {
                        layer.mode = cuteafd_loader::placement::LayerMode::ContextSplit;
                    }
                }
                let caches = [0, 1].map(|rank| -> Result<_> {
                    Ok(Rc::new(super::super::mla::CacheLayout::new(&cfg, &placement, rank, format)?
                        .allocate(streams[rank].device, vec![streams[rank].clone()])?))
                }).into_iter().collect::<Result<Vec<_>>>()?;
                let caches: [Rc<super::super::mla::CacheBank<'_>>; 2] = caches.try_into()
                    .map_err(|_| anyhow::anyhow!("expected two Kimi cache banks"))?;
                let retained = caches.each_ref().map(Rc::downgrade);
                assert!(owner.retain_caches([caches[1].clone(), caches[0].clone()]).is_err());
                owner.retain_caches(caches.clone())?;
                assert!(owner.retain_caches(caches.clone()).is_err());
                let mla_layer = cfg.attention.iter().position(|kind|
                    *kind == cuteafd_loader::families::kimi_k3::config::KimiAttention::Mla).unwrap();
                for rank in 0..2 {
                    assert_eq!(owner.cache(rank, mla_layer)?.ptr, caches[rank].layer(mla_layer)?.ptr);
                }
                assert!(owner.cache(2, mla_layer).is_err());
                drop(caches);
                assert!(retained.iter().all(|bank| bank.upgrade().is_some()));
                let events = fixture.events()?.len();
                drop(owner);
                let released = fixture.events()?;
                assert!(released[events..].starts_with("SS"),
                    "both exact rank readers must drain before any remote storage release: {}", &released[events..]);
                assert!(retained.iter().all(|bank| bank.upgrade().is_none()));
                assert_eq!(fixture.mapped_bytes(&library)?, before);
            }
        }
        Ok(())
    }
}
