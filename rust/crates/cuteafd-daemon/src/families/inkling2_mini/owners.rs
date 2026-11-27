//! Per-unit working owners and admitted serving request bank for Inkling 2 Mini (TP1).
use crate::shared::memory::{
    device::{Allocation, Stream},
    receipts::Inventory,
};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::Tag;
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_loader::families::inkling2::execution::UnitMetadata;
use cuteafd_loader::families::inkling2_mini::execution::{
    MiniCacheLayout, MiniUnitLayout, DECODE_TILE_ROWS, PREFILL_TILE_ROWS,
};
use std::{collections::BTreeMap, rc::Rc};

pub(crate) trait UnitBuffers<'a> {
    fn stream(&self) -> &Stream<'a>;
    fn layout(&self) -> &MiniUnitLayout;
    fn buffer(&self, name: &str) -> Result<Buffer>;
    fn decode_attention(&self) -> bool {
        false
    }
    fn convolution_strides(&self) -> Option<(usize, usize, usize)> {
        None
    }
    fn convolution_replay(&self, _field: &str) -> Result<Option<(Buffer, usize)>> {
        Ok(None)
    }
    fn packed(&self) -> Option<(&UnitOwner<'a>, &super::cache::CacheBank<'a>, usize, &[PackedMember])> {
        None
    }
}

#[derive(Clone)]
pub(crate) struct PackedMember {
    pub sequence: usize,
    pub rows: std::ops::Range<usize>,
    pub metadata: UnitMetadata,
}

/// Binds one unit owner to a specific 48-slot cache slot (`0..48`) for one visit.
pub(crate) struct PackedVisitUnit<'u, 'a> {
    pub unit: &'u UnitOwner<'a>,
    pub cache: &'u super::cache::CacheBank<'a>,
    pub cache_slot: usize,
    pub members: &'u [PackedMember],
}

impl<'a> UnitBuffers<'a> for PackedVisitUnit<'_, 'a> {
    fn decode_attention(&self) -> bool {
        self.unit.decode_attention
    }

    fn convolution_replay(&self, field: &str) -> Result<Option<(Buffer, usize)>> {
        let Some((backing, width)) = self.unit.replay.get() else {
            return Ok(None);
        };
        let (range, _) = self.cache.layout.conv.field(0, 0, field)?;
        let replay_rows = cuteafd_loader::families::inkling2_mini::execution::replay_capacity(self.unit.layout.rows);
        let offset = (self.cache_slot * replay_rows * width + range.start / 4) * 4;
        Ok(Some((super::cache::view(backing, offset, backing.bytes - offset)?, width)))
    }

    fn convolution_strides(&self) -> Option<(usize, usize, usize)> {
        Some((
            self.cache.layout.conv.slot_stride,
            self.cache.layout.conv.lag_stride,
            self.cache.layout.sequences,
        ))
    }

    fn stream(&self) -> &Stream<'a> {
        &self.unit.stream
    }
    fn layout(&self) -> &MiniUnitLayout {
        &self.unit.layout
    }

    fn buffer(&self, name: &str) -> Result<Buffer> {
        match name {
            "ring" | "attn_conv_history" | "mlp_conv_history" | "k_conv_history" | "v_conv_history" => {
                self.cache.field_bank(self.cache_slot, name)
            }
            "global_cache" => self.cache.global(self.cache_slot),
            "cache_slots" if self.cache.layout.global_slots.contains(&self.cache_slot) => {
                self.unit.buffer("global_slots")
            }
            _ => self.unit.buffer(name),
        }
    }

    fn packed(&self) -> Option<(&UnitOwner<'a>, &super::cache::CacheBank<'a>, usize, &[PackedMember])> {
        Some((self.unit, self.cache, self.cache_slot, self.members))
    }
}

pub(crate) struct VisitSegment<'u, 'a> {
    unit: &'u UnitOwner<'a>,
    cache: &'u super::cache::CacheBank<'a>,
    sequence: usize,
    cache_slot: usize,
    rows: std::ops::Range<usize>,
}

impl<'a> UnitBuffers<'a> for VisitSegment<'_, 'a> {
    fn convolution_strides(&self) -> Option<(usize, usize, usize)> {
        Some((self.cache.layout.conv.slot_stride, self.cache.layout.conv.lag_stride, 1))
    }
    fn stream(&self) -> &Stream<'a> {
        &self.unit.stream
    }
    fn layout(&self) -> &MiniUnitLayout {
        &self.unit.layout
    }
    fn buffer(&self, name: &str) -> Result<Buffer> {
        match name {
            "ring" | "attn_conv_history" | "mlp_conv_history" | "k_conv_history" | "v_conv_history" => {
                self.cache.field(self.sequence, self.cache_slot, name)
            }
            "global_cache" => self.cache.global(self.cache_slot),
            "cache_slots" if self.cache.layout.global_slots.contains(&self.cache_slot) => {
                self.row_buffer("global_slots")
            }
            "attention_scratch" | "page_table" => self.unit.buffer(name),
            _ => self.row_buffer(name),
        }
    }
}

impl VisitSegment<'_, '_> {
    fn row_buffer(&self, name: &str) -> Result<Buffer> {
        let buffer = self.unit.buffer(name)?;
        let parts = match name {
            "residual_arena" => self.unit.layout.blocks,
            "prev_pass_arena" => 2,
            "global_mixes" | "ctrl_paired" | "ctrl_rank" | "ctrl_fg" | "ctrl_out" => 3,
            "cross_pass_msg" => 4,
            _ => 1,
        };
        let stride = buffer
            .bytes
            .checked_div(self.unit.layout.rows * parts)
            .context("Inkling 2 Mini row-plane stride")?;
        ensure!(
            stride > 0 && buffer.bytes == self.unit.layout.rows * parts * stride,
            "Inkling 2 Mini row plane has fractional stride"
        );
        let offset = self
            .rows
            .start
            .checked_mul(stride)
            .context("Inkling 2 Mini row-plane offset")?;
        let bytes = if parts == 1 {
            self.rows.len() * stride
        } else {
            buffer.bytes - offset
        };
        super::cache::view(buffer, offset, bytes)
    }
}

pub(crate) struct UnitOwner<'a> {
    pub stream: Rc<Stream<'a>>,
    pub layout: MiniUnitLayout,
    buffers: BTreeMap<String, Allocation<'a>>,
    replay: std::cell::Cell<Option<(Buffer, usize)>>,
    pub decode_attention: bool,
}

impl<'a> UnitOwner<'a> {
    pub fn replaying(&self) -> bool {
        self.replay.get().is_some()
    }

    pub fn set_replay(&self, replay: Option<(Buffer, usize)>) -> Result<()> {
        let replay_rows = cuteafd_loader::families::inkling2_mini::execution::replay_capacity(self.layout.rows);
        ensure!(
            replay.is_none_or(|(buffer, width)| {
                width == 6656 && buffer.device_id == self.stream.device.id && buffer.bytes >= 48 * replay_rows * width * 4
            }),
            "Inkling 2 Mini FIR replay backing differs from admitted 48-slot width 6656"
        );
        self.replay.set(replay);
        Ok(())
    }

    pub fn bind<'u>(
        &'u self,
        cache: &'u super::cache::CacheBank<'a>,
        sequence: usize,
        cache_slot: usize,
    ) -> Result<VisitSegment<'u, 'a>> {
        ensure!(
            sequence < cache.layout.sequences && cache_slot < cache.layout.slots.len(),
            "Inkling 2 Mini persistent binding outside 48-slot cache"
        );
        cache.require_reader(&self.stream)?;
        Ok(VisitSegment {
            unit: self,
            cache,
            sequence,
            cache_slot,
            rows: 0..self.layout.rows,
        })
    }

    pub fn segment<'u>(
        &'u self,
        cache: &'u super::cache::CacheBank<'a>,
        sequence: usize,
        cache_slot: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<VisitSegment<'u, 'a>> {
        ensure!(rows.start < rows.end && rows.end <= self.layout.rows, "Mini segment outside unit");
        let mut seg = self.bind(cache, sequence, cache_slot)?;
        seg.rows = rows;
        Ok(seg)
    }

    /// Packs all 9 read-only per-unit metadata planes into one contiguous staging
    /// slice and uploads them once to the unit's `"metadata"` device buffer.
    pub fn packed_metadata(
        &self,
        cache: &super::cache::CacheBank<'a>,
        members: &[PackedMember],
        uploads: &mut crate::shared::token_table::RetainedUploads<'a>,
    ) -> Result<()> {
        ensure!(
            cache.layout.ring_sequence_bytes % cache.layout.ring_record_bytes() == 0,
            "Inkling 2 Mini sequence bank must have an integral record stride"
        );
        ensure!(
            members.len() <= cache.layout.sequences
                && members.iter().all(|m| m.sequence < cache.layout.sequences)
                && members.iter().map(|m| m.sequence).collect::<std::collections::BTreeSet<_>>().len() == members.len(),
            "Inkling 2 Mini packed cache members alias or exceed admitted request slots"
        );
        let rows = members.last().context("Mini packed metadata is empty")?.rows.end;
        ensure!(rows <= self.layout.rows, "Mini packed metadata extent differs");
        let meta = &self.layout.metadata;
        let width = meta.page_table_width;
        let upload_bytes = meta
            .page_table
            .start
            .checked_add(
                members
                    .len()
                    .checked_mul(width)
                    .and_then(|n| n.checked_mul(4))
                    .context("Mini packed page table byte extent overflow")?,
            )
            .context("Mini packed metadata byte extent overflow")?;
        ensure!(upload_bytes <= meta.bytes, "Mini packed metadata exceeds contiguous owner");

        let mut packed = vec![0u8; upload_bytes];
        let write_i64 = |buf: &mut [u8], span: &std::ops::Range<usize>, row: usize, value: i64| {
            let at = span.start + row * 8;
            buf[at..at + 8].copy_from_slice(&value.to_le_bytes());
        };
        let write_i32 = |buf: &mut [u8], span: &std::ops::Range<usize>, idx: usize, value: i32| {
            let at = span.start + idx * 4;
            buf[at..at + 4].copy_from_slice(&value.to_le_bytes());
        };

        let step = if self.decode_attention {
            DECODE_TILE_ROWS
        } else {
            PREFILL_TILE_ROWS
        };
        let mut row_cursor = 0usize;
        let mut tile_cursor = 0usize;
        let ring_stride = cache.layout.ring_stride_records() as i64;
        for (table, member) in members.iter().enumerate() {
            ensure!(
                member.rows.start == row_cursor && member.metadata.rows() == member.rows.len(),
                "Mini packed metadata members do not cover unit rows"
            );
            ensure!(member.metadata.page_table.len() <= width, "Mini member page table exceeds stride");
            let row_offset = member.rows.start as i64;
            let ring_base = (member.sequence as i64) * ring_stride;
            let conv_slot = member.sequence as i64;
            let first_row = member.rows.start as i32;
            let table_idx = table as i32;
            for i in 0..member.rows.len() {
                let r = row_cursor + i;
                write_i64(&mut packed, &meta.positions, r, member.metadata.positions[i]);
                write_i64(&mut packed, &meta.cache_slots, r, member.metadata.local_slots[i] + row_offset);
                write_i64(&mut packed, &meta.global_slots, r, member.metadata.global_slots[i]);
                write_i64(&mut packed, &meta.ring_slots, r, member.metadata.ring_slots[i] + ring_base);
                write_i64(&mut packed, &meta.conv_slots, r, conv_slot);
                write_i32(&mut packed, &meta.first, r, first_row);
                write_i32(&mut packed, &meta.table_indices, r, table_idx);
            }
            for tile_row in member.rows.clone().step_by(step) {
                write_i32(&mut packed, &meta.tile_rows, tile_cursor, tile_row as i32);
                tile_cursor += 1;
            }
            let page_base = meta.page_table.start + table * width * 4;
            for (p, &page) in member.metadata.page_table.iter().enumerate() {
                let at = page_base + p * 4;
                packed[at..at + 4].copy_from_slice(&page.to_le_bytes());
            }
            row_cursor += member.rows.len();
        }

        // SAFETY: one batched H2D copy into the contiguous admitted `"metadata"` owner.
        unsafe {
            uploads.upload(self.buffer("metadata")?, &packed, self.stream.raw)?;
        }
        Ok(())
    }

    pub fn new(stream: Rc<Stream<'a>>, layout: MiniUnitLayout) -> Result<Self> {
        let buffers = layout
            .buffers
            .iter()
            .map(|(name, bytes)| {
                Ok((
                    name.clone(),
                    Allocation::new(stream.device, *bytes, Tag::workspace("inkling2_mini/unit"))?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let owner = Self {
            stream,
            layout,
            buffers,
            replay: std::cell::Cell::new(None),
            decode_attention: false,
        };
        owner.zero()?;
        Ok(owner)
    }

    pub fn buffer(&self, name: &str) -> Result<Buffer> {
        if let Some(span) = self.layout.metadata.span(name) {
            let backing = self
                .buffers
                .get("metadata")
                .context("Inkling 2 Mini unit lacks contiguous metadata owner")?
                .buffer;
            return super::cache::view(backing, span.start, span.len());
        }
        Ok(self
            .buffers
            .get(name)
            .with_context(|| format!("Inkling 2 Mini unit lacks owner {name}"))?
            .buffer)
    }

    pub fn zero(&self) -> Result<()> {
        self.stream.device.run(|| {
            for buffer in self.buffers.values() {
                // SAFETY: fresh private owners written on retained stream.
                unsafe {
                    self.stream
                        .device
                        .library
                        .cuda_zero_bytes_async(buffer.buffer, buffer.buffer.bytes, self.stream.raw)?;
                }
            }
            Ok(())
        })
    }

    pub fn uploads(
        &self,
        cfg: &cuteafd_loader::families::inkling2_mini::Inkling2MiniConfig,
        members: usize,
    ) -> Result<crate::shared::token_table::RetainedUploads<'a>> {
        self.stream.device.run(|| {
            crate::shared::token_table::RetainedUploads::new(self.stream.device, self.layout.upload_shape(cfg, members)?)
        })
    }

    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, owner) in &self.buffers {
            owner.inventory(inventory, &format!("{prefix}.{name}"))?;
        }
        Ok(())
    }
}

impl<'a> UnitBuffers<'a> for UnitOwner<'a> {
    fn stream(&self) -> &Stream<'a> {
        &self.stream
    }
    fn layout(&self) -> &MiniUnitLayout {
        &self.layout
    }
    fn buffer(&self, name: &str) -> Result<Buffer> {
        UnitOwner::buffer(self, name)
    }
}

impl Drop for UnitOwner<'_> {
    fn drop(&mut self) {
        self.stream.drain_or_abort("Inkling 2 Mini unit owners");
    }
}

pub(crate) struct RequestOwners<'a> {
    pub decode_uploads: crate::shared::token_table::RetainedUploads<'a>,
    pub prefill_uploads: Vec<crate::shared::token_table::RetainedUploads<'a>>,
    pub decode: UnitOwner<'a>,
    pub prefill: Vec<UnitOwner<'a>>,
    pub cache: Rc<super::cache::CacheBank<'a>>,
}

impl<'a> RequestOwners<'a> {
    pub fn allocation_receipts(
        &self,
        inventory: &mut Inventory<'_>,
        census: &mut crate::shared::memory::receipts::Census,
        devices: cuteafd_loader::placement::planner::CoordinatorDevices<'_>,
    ) -> Result<()> {
        self.cache.inventory(inventory)?;
        let decode = "inkling2_mini.decode.0";
        self.decode.inventory(inventory, decode)?;
        self.decode_uploads
            .allocation_receipts(inventory, &format!("{decode}.uploads"))?;
        for (slot, (owner, uploads)) in self.prefill.iter().zip(&self.prefill_uploads).enumerate() {
            let prefix = format!("inkling2_mini.unit.0.{slot}");
            owner.inventory(inventory, &prefix)?;
            uploads.allocation_receipts(inventory, &format!("{prefix}.uploads"))?;
        }
        for (name, uploads) in std::iter::once((format!("{decode}.uploads"), &self.decode_uploads)).chain(
            self.prefill_uploads
                .iter()
                .enumerate()
                .map(|(slot, uploads)| (format!("inkling2_mini.unit.0.{slot}.uploads"), uploads)),
        ) {
            let backing = inventory
                .receipts
                .iter()
                .find(|receipt| {
                    receipt.allocation.id.ends_with(&format!(".{name}"))
                        && receipt.allocation.bytes == uploads.mapped_alias().bytes as u64
                        && receipt.allocation.location == cuteafd_loader::placement::planner::Location::PINNED
                })
                .with_context(|| format!("Mini upload alias has no pinned owner {name}"))?
                .allocation
                .id
                .clone();
            census.mapped(format!("{name}.mapped"), backing, uploads.mapped_alias(), devices)?;
        }
        Ok(())
    }

    pub fn new(
        stream: Rc<Stream<'a>>,
        inputs: &cuteafd_loader::placement::families::inkling2_mini::MiniInputs,
        admitted: &cuteafd_loader::placement::families::inkling2_mini::AdmittedMini,
    ) -> Result<Self> {
        use cuteafd_loader::placement::planner::Location;
        let cfg = &inputs.cfg;
        let sequences = usize::try_from(inputs.sequences)?;
        let context = usize::try_from(inputs.context.maximum)?;
        let pages = usize::try_from(admitted.solved.pool_tokens.div_ceil(64))?;
        let cache_layout = MiniCacheLayout::with_format(cfg, sequences, pages, inputs.kv_format)?;
        let mark_slots = inputs.prefix_mark_slots;
        let decode_layout =
            MiniUnitLayout::serving_decode_rows(cfg, inputs.decode_rows as usize, context, inputs.kv_format, sequences)?;
        let prefill_layout =
            MiniUnitLayout::serving(cfg, usize::try_from(inputs.prefill.rows)?, context, inputs.kv_format, sequences)?;
        let owner = |id: &str, bytes: usize, location| -> Result<()> {
            let allocation = admitted
                .solved
                .facts
                .allocations
                .iter()
                .find(|a| a.id == id)
                .with_context(|| format!("Inkling 2 Mini unadmitted runtime owner {id}"))?;
            ensure!(
                allocation.location == location && allocation.bytes == bytes as u64,
                "Inkling 2 Mini runtime owner differs from admission: {id} {bytes}/{}",
                allocation.bytes
            );
            Ok(())
        };
        let device = Location::coordinator(0);
        owner("inkling2_mini.cache.0.active", cache_layout.active_bytes()?, device)?;
        owner("inkling2_mini.cache.0.conv", cache_layout.conv.bytes()?, device)?;
        if mark_slots > 0 {
            owner("inkling2_mini.cache.0.marks", cache_layout.retained_bytes(mark_slots)?, device)?;
        }
        for &slot in &cache_layout.global_slots {
            let bytes = usize::try_from(cuteafd_loader::placement::planner::sm120_request_backing(
                cache_layout.layer_page_bytes()? as u64,
            )?)?;
            owner(&format!("inkling2_mini.kv.{slot}.0.records"), bytes, device)?;
        }
        for (prefix, layout) in std::iter::once(("inkling2_mini.decode.0".to_string(), &decode_layout)).chain(
            (0..inputs.prefill.waves).map(|slot| (format!("inkling2_mini.unit.0.{slot}"), &prefill_layout)),
        ) {
            for (name, bytes) in &layout.buffers {
                owner(&format!("{prefix}.{name}"), *bytes, device)?;
            }
            let uploads = layout.upload_shape(cfg, sequences.min(layout.rows))?;
            owner(&format!("{prefix}.uploads"), uploads.capacity_bytes()?, Location::PINNED)?;
        }
        let cache = Rc::new(super::cache::CacheBank::new(vec![stream.clone()], cache_layout, mark_slots)?);
        let mut decode = UnitOwner::new(stream.clone(), decode_layout)?;
        decode.decode_attention = true;
        let prefill = (0..inputs.prefill.waves)
            .map(|_| UnitOwner::new(stream.clone(), prefill_layout.clone()))
            .collect::<Result<Vec<_>>>()?;
        let decode_uploads = decode.uploads(cfg, sequences.min(decode.layout.rows))?;
        let prefill_uploads = prefill
            .iter()
            .map(|o| o.uploads(cfg, sequences.min(o.layout.rows)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            decode_uploads,
            prefill_uploads,
            decode,
            prefill,
            cache,
        })
    }
}

#[cfg(test)]
mod packed_tests {
    use super::*;
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use cuteafd_loader::families::{
        inkling2::execution::UnitMetadata,
        inkling2_mini::{Inkling2MiniConfig, InklingKvCache},
    };

    fn test_mini_config() -> Result<Inkling2MiniConfig> {
        Inkling2MiniConfig::from_hf(&serde_json::from_str(include_str!(
            "../../../../cuteafd-loader/src/families/inkling2_mini/fixtures/config.json"
        ))?)
    }

    #[test]
    fn packed_metadata_uploads_all_nine_planes_in_one_batched_h2d_copy() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.configure_device_flags(&library, 0)?;
        let device = crate::shared::memory::device::Device {
            library: &library,
            id: 0,
        };
        let stream = Rc::new(Stream::new(device)?);
        let cfg = test_mini_config()?;
        let cache = super::super::cache::CacheBank::new(
            vec![stream.clone()],
            MiniCacheLayout::with_format(&cfg, 2, 4, InklingKvCache::Fp8)?,
            0,
        )?;
        let mut unit = UnitOwner::new(stream.clone(), MiniUnitLayout::serving(&cfg, 4, 128, InklingKvCache::Fp8, 2)?)?;
        let mut uploads = unit.uploads(&cfg, 2)?;
        let members = vec![
            PackedMember {
                sequence: 1,
                rows: 0..1,
                metadata: UnitMetadata::new(&cfg.text, 63..64, &[2], 4)?,
            },
            PackedMember {
                sequence: 0,
                rows: 1..4,
                metadata: UnitMetadata::new(&cfg.text, 65..68, &[0, 3], 4)?,
            },
        ];

        let before = fixture.events()?.chars().filter(|&c| c == 'C').count();
        unit.packed_metadata(&cache, &members, &mut uploads)?;
        stream.drain()?;
        let after = fixture.events()?.chars().filter(|&c| c == 'C').count();
        assert_eq!(after - before, 1, "all 9 metadata planes must land in one batched H2D copy");

        let i64s = |name: &str| -> Result<Vec<i64>> {
            let buffer = unit.buffer(name)?;
            assert_eq!(buffer.ptr as usize % 16, 0, "{name} device view must be 16-byte aligned");
            let mut bytes = vec![0u8; buffer.bytes];
            library.copy_d2h(&mut bytes, buffer)?;
            Ok(bytes.chunks_exact(8).map(|b| i64::from_le_bytes(b.try_into().unwrap())).collect())
        };
        let i32s = |u: &UnitOwner<'_>, name: &str| -> Result<Vec<i32>> {
            let buffer = u.buffer(name)?;
            assert_eq!(buffer.ptr as usize % 16, 0, "{name} device view must be 16-byte aligned");
            let mut bytes = vec![0u8; buffer.bytes];
            library.copy_d2h(&mut bytes, buffer)?;
            Ok(bytes.chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect())
        };

        assert_eq!(i64s("positions")?, [63, 65, 66, 67]);
        assert_eq!(i64s("cache_slots")?, [0, 1, 2, 3]);
        assert_eq!(i64s("global_slots")?, [191, 193, 194, 195]);
        assert_eq!(i64s("conv_slots")?, [1, 0, 0, 0]);
        let stride = cache.layout.ring_stride_records() as i64;
        assert_eq!(i64s("ring_slots")?, [stride + 63, 65, 66, 67]);
        assert_eq!(i32s(&unit, "first")?, [0, 1, 1, 1]);
        assert_eq!(i32s(&unit, "table_indices")?, [0, 1, 1, 1]);
        assert_eq!(&i32s(&unit, "tile_rows")?[..2], [0, 1]);
        assert_eq!(i32s(&unit, "page_table")?, [2, 0, 0, 3]);

        // Decode mode writes 1-row tiles, still in one single batched H2D copy.
        unit.decode_attention = true;
        let before_decode = fixture.events()?.chars().filter(|&c| c == 'C').count();
        unit.packed_metadata(&cache, &members, &mut uploads)?;
        stream.drain()?;
        let after_decode = fixture.events()?.chars().filter(|&c| c == 'C').count();
        assert_eq!(after_decode - before_decode, 1);
        assert_eq!(i32s(&unit, "tile_rows")?, [0, 1, 2, 3]);
        Ok(())
    }
}
