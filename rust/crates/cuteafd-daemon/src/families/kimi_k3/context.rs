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
    fn k31_gathered_metadata_chunks_match_decode_and_replicated_selection() -> Result<()> {
        use super::super::device::SequenceRows;
        let mut pool = cuteafd_engine::prefix::RefPagePool::parity(128, 64, 0);
        pool.alloc_for(0..128)?;
        let pages = (0..80).map(|page| (page % 2 + 2 * (39 - page / 2)) as u32).collect::<Vec<_>>();
        let start = 2041;
        let rows = 131;
        let sequences = [SequenceRows { slot: 0, start, rows, pages: &pages }];
        let i32s = |bytes: &[u8]| bytes.chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        let mut shard_runs = Vec::new();
        for rank in 0..2 {
            let whole = DecodeRows::gathered(rank, &sequences, 5120, 1, &pool)?;
            for first in (0..rows).step_by(64) {
                let count = (rows - first).min(64);
                let chunk = DecodeRows::new(rank, &[SequenceRows { slot: 0, start: start + first,
                    rows: count, pages: &pages }], 5120, 1, &pool)?;
                for ((_, all), (_, part)) in whole.uploads().into_iter().take(3).zip(chunk.uploads()) {
                    let stride = all.len() / rows;
                    assert_eq!(&all[first * stride..(first + count) * stride], part);
                }
            }
            let runs = i32s(&whole.lengths).into_iter().map(|length| {
                let mut candidates = (0..length as usize).map(|local| {
                    let id = (local / 64 * 2 + rank) * 64 + local % 64;
                    // Ties exercise the total (score descending, id ascending) rule.
                    (((id * 37) % 97) as i32, id as i32)
                }).collect::<Vec<_>>();
                candidates.sort_by_key(|&(score, id)| (-score, id));
                candidates.truncate(2048);
                candidates
            }).collect::<Vec<_>>();
            shard_runs.push(runs);
        }
        let select = |range: std::ops::Range<usize>| range.map(|row| {
            let mut merged = [shard_runs[0][row].clone(), shard_runs[1][row].clone()].concat();
            merged.sort_by_key(|&(score, id)| (-score, id));
            merged.truncate(2048);
            merged.into_iter().map(|(_, id)| id).collect::<Vec<_>>()
        }).collect::<Vec<_>>();
        let single = select(0..rows);
        let chunked = (0..rows).step_by(64).flat_map(|first| select(first..(first + 64).min(rows))).collect::<Vec<_>>();
        assert_eq!(chunked, single);
        assert_eq!(select(0..rows), single, "run-to-run ordering");
        assert_eq!(select(rows - 1..rows)[0], single[rows - 1], "last prompt row is decode's rule");
        for (row, ids) in single.iter().enumerate() {
            let mut replicated = (0..start + row + 1).map(|id| (((id * 37) % 97) as i32, id as i32)).collect::<Vec<_>>();
            replicated.sort_by_key(|&(score, id)| (-score, id));
            replicated.truncate(2048);
            assert_eq!(*ids, replicated.into_iter().map(|(_, id)| id).collect::<Vec<_>>());
        }
        Ok(())
    }

    #[test]
    fn k31_gathered_barrier_borrows_only_the_serial_target_query_channel() {
        let device = include_str!("device.rs");
        let engine = include_str!("engine.rs");
        let layers = include_str!("layers.rs");
        let draft = include_str!("draft.rs");
        assert!(device.contains("self.mla.parity() && self.lane_index == 0"));
        assert!(device.contains("owner.ranks[self.rank].check_reader(self.stream())?"));
        for entry in ["fn prefill_packed_heads", "fn step_device_mode"] {
            let body = engine.split(entry).nth(1).unwrap().split("    pub(crate) fn").next().unwrap();
            assert!(body.contains("self.provisional.borrow().is_none()"));
            assert!(body.contains("self.lanes.try_borrow_mut()?"));
            assert!(body.contains("draft.prepare_target()?"));
        }
        assert!(draft.contains("self.target_streams[0].raw, self.consumed.borrow().raw"));
        let pair = layers.split("fn context_pair_with").nth(1).unwrap().split("fn producer_streams").next().unwrap();
        assert!(pair.find("kernels.wait_prior_commit(reader, owner)?").unwrap()
            < pair.find("kernels.attention_commit(rank)?").unwrap());
        let select = device.split("pub(crate) fn context_dsa_chunk").nth(1).unwrap().split("pub(crate) fn context_dsa_dump").next().unwrap();
        assert!(select.find("dsa.merge.launch").unwrap() < select.find("push_query_width").unwrap());
        assert!(select.contains("context.exchange.wait_query(self.rank, slot)?"));
        assert!(!select.contains("drain()") && !select.contains("synchronize"));
        let diagnostic = device.split("pub(crate) fn context_dsa_dump").nth(1).unwrap().split("fn context_dsa_merge").next().unwrap();
        assert!(diagnostic.find("return Ok(())").unwrap() < diagnostic.find("owner.metadata").unwrap());
        assert!(diagnostic.find("return Ok(())").unwrap() < diagnostic.find("record_selected").unwrap());
        let paired = layers.rsplit("context_pair_with(&mut Pair { binding: self, layer })?;").next().unwrap();
        assert!(paired.contains("context_dsa_dump"));
        assert!(paired.find("context_dsa_dump").unwrap() < paired.find("self.publish").unwrap());
        let graph = device.split("fn graph_step(").nth(1).unwrap().split("pub(crate) fn attention(").next().unwrap();
        assert!(graph.find("if self.eager_context_dsa_diagnostic()").unwrap() < graph.find("self.graphs.take()").unwrap());
        assert!(graph.contains("ensure!(self.graph_untouched()"));
        assert!(graph.split("if self.eager_context_dsa_diagnostic()").nth(1).unwrap()
            .split("if !super::graphs::replay_allowed").next().unwrap().contains("return body(self)"));
        let warm = device.split("pub(crate) fn warm_graphs(").nth(1).unwrap().split("pub(crate) fn graph_cycle").next().unwrap();
        assert!(warm.find("if self.eager_context_dsa_diagnostic() { return Ok(0); }").unwrap()
            < warm.find("graphs.prewarm").unwrap());
        assert!(graph.contains("graphs.graph(&stream, key, allow_capture, pinned, || body(self))"),
            "diagnostic-off dispatch remains the ordinary graph path");
    }

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
    #[ignore = "requires K3.1 checkpoint, two RTX GPUs, and matching WIP native programs"]
    fn k31_context_prefill_host_restore_matches_scratch_index_and_mla_on_both_ranks() -> Result<()> {
        use cuteafd_engine::prefix::{After, PrefixCache, PrefixConfig, PrefixFamily, SnapshotKind};
        use cuteafd_hostcache::{config::Config, copy::Stream as CopyStream};
        use std::{path::PathBuf, sync::Arc};
        cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::KIMI_K3)
            .map_err(|_| anyhow::anyhow!("Kimi cache oracle requires an isolated process"))?;
        let snapshot = PathBuf::from(std::env::var("CUTEAFD_GOLDEN_SNAPSHOT")?);
        let checkpoint = cuteafd_loader::plan::Checkpoint::coordinator(&snapshot, false, false)?;
        let full = cuteafd_loader::families::kimi_k3::config::KimiK3Config::from_hf(&checkpoint.config)?;
        ensure!(full.dsa.is_some(), "K3.1 checkpoint required");
        let cfg = super::super::admission::prefix(&full, 4)?;
        let manifest = PathBuf::from(std::env::var("CUTEAFD_PROGRAM_MANIFEST")?);
        let native = PathBuf::from(std::env::var("CUTEAFD_NATIVE_LIB")?);
        // SAFETY: all native owners retain this sealed library through teardown.
        let library = Arc::new(unsafe { NativeLibrary::load(&native)? });
        let programs = library.programs()?.with_manifest(&manifest)?;
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
        runtime.block_on(async {
            let loaded = super::super::startup::load(&library, &programs, &checkpoint, &full, &cfg,
                super::super::startup::Options {
                    role: super::super::startup::Role::Golden { skip_routed_experts: true },
                    expert_usage: false, expert_usage_reader: Default::default(), cache_format: cuteafd_core::MlaCacheFormat::Fp8,
                    devices: [0, 1], attention: Some(cuteafd_loader::placement::AttentionPlacement::Context),
                    native_lib: &native, manifest: &manifest, peers: vec![], rtx_expert_layers: cuteafd_loader::placement::OnboardSetting::Auto,
                    rows: 256, waves: 1, prefill_rows: Some(256), slots: 2, marks: 2, max_context: 8192,
                    context: cuteafd_loader::placement::planner::Context::pinned(8192), pool_tokens: Some(8192),
                    graphs: true, ffn_mode: Default::default(), hosted_layers: None, draft: None, verify_rows: 0, prefix: None,
                })?;
            let engine = loaded.engine;
            let family = engine.prefix_family()?;
            let layout = family.layout();
            let chunk = layout.mark_bytes.max(layout.page_bytes) as u64;
            let copy = crate::shared::prefix::CudaCopyEngine::registered_owned(&library, family.snapshot_owners())?;
            let mut cache = PrefixCache::new(layout, PrefixConfig { entries: 8, mark_slots: 2,
                keep_logits: true, min_tokens: 1 }, Some((Config { bytes: chunk * 6, chunk_bytes: chunk,
                    min_tokens: 1, ..Default::default() }, copy)))?;
            let build = |units| super::super::prefix::Placement { units, slot: 0, len: 0, state_len: 0 };
            let tokens = (0..131).map(|token| 1000 + token).collect::<Vec<u32>>();
            let mut placement = cache.admit(&family, &tokens, 132, false, build)?.placement;
            engine.prefill_packed(&mut [(&mut placement, &tokens)], cache.pool(), &mut |_, _| Ok(Ok(())))?;
            let scratch = family.page_images(&placement)?;
            let digest = family.live_digest(&placement)?;
            assert!(scratch.iter().all(|rank| !rank.is_empty()));
            assert!(cache.capture(&family, SnapshotKind::Prompt, &tokens, &placement, After::from_logits(&[1., 2.], true))?);
            cache.host_engine_mut().context("host prefix copy engine")?.synchronize(CopyStream::Store)?;
            cache.tick();
            cache.release(&family, &placement.units)?;
            cache.clear(&family)?;
            engine.reset_cold_state()?;
            let hit = cache.admit(&family, &tokens, 132, false, build)?;
            assert_eq!(hit.resume, tokens.len());
            assert!(hit.source.as_ref().is_some_and(|source| source.host));
            assert_eq!(family.page_images(&hit.placement)?, scratch, "both ranks: actual prefill index keys/scales and MLA restore exactly");
            assert_eq!(family.live_digest(&hit.placement)?, digest, "actual prefill recurrent state and index/MLA cache bytes");
            cache.release(&family, &hit.placement.units)?;
            cache.clear(&family)?;
            // A new cache has no host hit: recompute with the same shape and
            // compare actual cache bytes, not logits or future greedy text.
            drop(cache);
            let mut cache = PrefixCache::<crate::shared::prefix::CudaCopyEngine<'_>, super::super::prefix::SnapshotMeta>::new(
                layout, PrefixConfig { entries: 8, mark_slots: 2, keep_logits: true, min_tokens: 1 }, None)?;
            engine.reset_cold_state()?;
            let fresh = cache.admit(&family, &tokens, 132, false, build)?;
            assert_eq!(fresh.resume, 0);
            let mut fresh = fresh.placement;
            engine.prefill_packed(&mut [(&mut fresh, &tokens)], cache.pool(), &mut |_, _| Ok(Ok(())))?;
            assert_eq!(family.page_images(&fresh)?, scratch, "scratch prefill index/MLA determinism on each owner");
            cache.release(&family, &fresh.units)?;
            cache.clear(&family)?;
            Ok(())
        })
    }

    #[test]
    #[ignore = "requires two peer-capable CUDA GPUs and the K3.1 WIP native library"]
    fn k31_native_chunked_parity_select_matches_single_shot_and_decode() -> Result<()> {
        use crate::shared::memory::device::Device;
        use cuteafd_ffi::{KimiCandidateBatch, KimiCandidateMerge, KimiIndexMode, KimiIndexSelection, KimiIndexTopK};
        // SAFETY: all programs and GPU owners are destroyed before this WIP library.
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let programs = library.programs()?.with_manifest(std::path::Path::new(&std::env::var("CUTEAFD_PROGRAM_MANIFEST")?))?;
        for id in 0..2 {
            Device { library: &library, id }.run(|| {
                KimiIndexTopK::load(&programs, 64, 16384, KimiIndexMode::Decode, true)?;
                KimiIndexTopK::load(&programs, 4096, 16384, KimiIndexMode::Prefill, true)?;
                KimiCandidateMerge::load(&programs)?;
                Ok(())
            })?;
        }
        let decode = KimiIndexTopK::load(&programs, 64, 16384, KimiIndexMode::Decode, true)?;
        let prefill = KimiIndexTopK::load(&programs, 4096, 16384, KimiIndexMode::Prefill, true)?;
        let merge = KimiCandidateMerge::load(&programs)?;
        let rows = 131;
        let pages = 40;
        let width = pages / 2;
        let start = 2041;
        let streams = [0, 1].map(|id| Stream::new(Device { library: &library, id }).map(Rc::new));
        let [s0, s1] = streams;
        let streams = [s0?, s1?];
        let ranks = streams.each_ref().map(|stream| RankDevice { device: stream.device.id, stream: stream.raw });
        let buffers = cuteafd_loader::placement::ContextBuffers {
            query_row_bytes: 48 * 576 * 2, partial_row_bytes: 48 * (512 * 2 + 4),
            candidate_row_bytes: 2048 * 8, decode_rows: 64, lanes: 1, ..Default::default() };
        let exchange = ContextExchange::new(&library, ranks, &buffers, None)?;
        let upload = |rank: usize, bytes: &[u8]| -> Result<Allocation<'_>> {
            let owner = Allocation::new(streams[rank].device, bytes.len().max(16), crate::shared::memory::TEST)?;
            streams[rank].device.run(|| library.copy_h2d(owner.buffer, bytes))?;
            Ok(owner)
        };
        let read = |buffer: CuteafdDeviceBuffer| -> Result<Vec<u8>> {
            let mut bytes = vec![0; buffer.bytes];
            Device { library: &library, id: buffer.device_id }.run(|| library.copy_d2h(&mut bytes, buffer))?;
            Ok(bytes)
        };
        let ints = |values: Vec<i32>| values.into_iter().flat_map(i32::to_le_bytes).collect::<Vec<_>>();
        let mut full_cache = vec![0; pages * 8448];
        for token in 0..pages * 64 {
            let page = token / 64;
            let row = token % 64;
            full_cache[page * 8448 + row * 128..page * 8448 + (row + 1) * 128].fill(0x38);
            full_cache[page * 8448 + 8192 + row * 4..page * 8448 + 8192 + (row + 1) * 4]
                .copy_from_slice(&((token + 1) as f32 / 4096.0).to_le_bytes());
        }
        let query = upload(0, &vec![0x38; rows * 4096])?;
        let weights = upload(0, &1f32.to_le_bytes().repeat(rows * 32))?;
        let cache = upload(0, &full_cache)?;
        let table = upload(0, &ints((0..pages as i32).collect()))?;
        let frontiers = upload(0, &ints((0..rows).map(|row| (start + row + 1) as i32).collect()))?;
        let reference = upload(0, &vec![0; rows * 2048 * 4])?;
        let reference_scores = upload(0, &vec![0; rows * 2048 * 4])?;
        let scratch = upload(0, &vec![0; prefill.scratch_bytes().max(decode.scratch_bytes())])?;
        streams[0].device.run(|| Ok(prefill.launch(KimiIndexSelection {
            rows, cache_pages: pages, table_width: pages, table_stride: pages, page_stride: 1, page_offset: 0,
            query_fp8: query.buffer, head_weights: weights.buffer, cache: cache.buffer, table: table.buffer,
            lengths: frontiers.buffer, indices: reference.buffer, scores: reference_scores.buffer,
            scratch: scratch.buffer,
        }, streams[0].raw)?))?;
        streams[0].drain()?;
        let expected = read(reference.buffer)?;
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for rank in 0..2 {
            let compact = (0..width).map(|page| (width - page - 1) as i32).collect::<Vec<_>>();
            let mut cache = vec![0; width * 8448];
            for (logical, physical) in compact.iter().enumerate() {
                cache[*physical as usize * 8448..(*physical as usize + 1) * 8448]
                    .copy_from_slice(&full_cache[(2 * logical + rank) * 8448..(2 * logical + rank + 1) * 8448]);
            }
            inputs.push([
                upload(rank, &vec![0x38; rows * 4096])?, upload(rank, &1f32.to_le_bytes().repeat(rows * 32))?,
                upload(rank, &cache)?, upload(rank, &ints(compact.repeat(rows)))?,
                upload(rank, &ints((0..rows).map(|row| crate::shared::peer_split::context::shard_rows(start + row + 1, 64, rank) as i32).collect()))?,
                upload(rank, &vec![0; scratch.buffer.bytes])?,
            ]);
            outputs.push((0..6).map(|i| upload(rank, &vec![0; if i == 5 { 64 * 4 } else { rows * 2048 * 4 }]))
                .collect::<Result<Vec<_>>>()?);
        }
        for repeat in 0..2 {
            for first in (0..rows).step_by(64) {
                let count = (rows - first).min(64);
                for rank in 0..2 {
                    let input = &inputs[rank];
                    let output = &outputs[rank];
                    let part = |i: usize, stride| crate::shared::prefix::view(input[i].buffer, first * stride, count * stride);
                    let scores = crate::shared::prefix::view(output[0].buffer, first * 2048 * 4, count * 2048 * 4)?;
                    let indices = crate::shared::prefix::view(output[1].buffer, first * 2048 * 4, count * 2048 * 4)?;
                    streams[rank].device.run(|| Ok(decode.launch(KimiIndexSelection {
                        rows: count, cache_pages: width, table_width: width, table_stride: width,
                        page_stride: 2, page_offset: rank, query_fp8: part(0, 4096)?, head_weights: part(1, 32 * 4)?,
                        cache: input[2].buffer, table: part(3, width * 4)?, lengths: part(4, 4)?,
                        indices, scores, scratch: input[5].buffer,
                    }, streams[rank].raw)?))?;
                    exchange.push_candidates(rank, 0, scores.ptr, indices.ptr, count)?;
                    let (scores, indices) = exchange.wait_candidates(rank, 0)?;
                    let received = |ptr| CuteafdDeviceBuffer { ptr, bytes: count * 4096 * 4, device_id: rank as i32, flags: 0 };
                    streams[rank].device.run(|| Ok(merge.launch(KimiCandidateBatch {
                        rows: count, table_stride: width, rank, scores: received(scores), indices: received(indices),
                        output_scores: crate::shared::prefix::view(output[2].buffer, first * 2048 * 4, count * 2048 * 4)?,
                        output_indices: crate::shared::prefix::view(output[3].buffer, first * 2048 * 4, count * 2048 * 4)?,
                        table: part(3, width * 4)?, local_slots: output[4].buffer, local_lengths: output[5].buffer,
                    }, streams[rank].raw)?))?;
                    exchange.push_query_width(rank, 0, input[0].buffer.ptr, 16, 1, 16)?;
                    exchange.wait_query(rank, 0)?;
                }
            }
            for stream in &streams { stream.drain()?; }
            for rank in 0..2 {
                assert_eq!(read(outputs[rank][3].buffer)?, expected, "rank {rank}, repeat {repeat}: chunked parity vs single-shot replicated");
                assert_eq!(read(outputs[rank][2].buffer)?, read(reference_scores.buffer)?, "merged scores");
            }
        }
        // The final prefill row goes through exactly the decode producer/selector
        // route; assert its ordered ids, not merely the selected set.
        let last = &expected[(rows - 1) * 2048 * 4..];
        let ids = last.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
        assert_eq!(ids, (start + rows - 2048..start + rows).rev().map(|id| id as i32).collect::<Vec<_>>());
        Ok(())
    }

    #[test]
    #[ignore = "requires two peer-capable CUDA GPUs and the K3.1 WIP native library"]
    fn k31_gathered_query_ack_aborts_without_an_unmatched_peer() -> Result<()> {
        use crate::shared::{memory::device::Device, peer_split::PeerExchange};
        // SAFETY: the matching WIP library remains live through all native owners.
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let streams = [0, 1].map(|id| Stream::new(Device { library: &library, id }).map(Rc::new));
        let [s0, s1] = streams;
        let streams = [s0?, s1?];
        let ranks = streams.each_ref().map(|stream| RankDevice { device: stream.device.id, stream: stream.raw });
        let peer = PeerExchange::new_abortable(&library, ranks, 2, 256, 256)?;
        let buffers = cuteafd_loader::placement::ContextBuffers {
            query_row_bytes: 48 * 576 * 2, partial_row_bytes: 48 * (512 * 2 + 4),
            candidate_row_bytes: 2048 * 8, decode_rows: 64, lanes: 1, ..Default::default() };
        let exchange = ContextExchange::new(&library, ranks, &buffers,
            Some([peer.abort_word(0)?.context("rank zero abort word")?, peer.abort_word(1)?.context("rank one abort word")?]))?;
        // No rank publishes the acknowledgement: terminal abort must release
        // both the first blocked wait and every subsequent queued generation.
        for rank in 0..2 {
            exchange.wait_query(rank, 0)?;
            exchange.wait_query(rank, 0)?;
        }
        peer.publish_abort()?;
        for stream in &streams { stream.drain()?; }
        assert!(library.peer_wait_timeout().is_none(), "abort must release before the watchdog");
        peer.finish_terminal(true);
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
