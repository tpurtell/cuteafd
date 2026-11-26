//! Pages-only committed KV and 21-full-layer IndexCache storage for Hy4 (`hy_v4_vl`).
//!
//! Layout per coordinator GPU rank:
//! - `dsa_kv`: `[78, pages, 64, record_bytes]` (RefPagePool COW physical pages)
//! - `dsa_index`: `[21, pages, 8448]` (per-token E4M3+F32 index K cache on the 21
//!   full indexer layers only: `64 * 128` E4M3 keys followed by `64 * 4` FP32 scales)
//!
//! Zero SWA rings, zero pending indexer keys, zero per-slot device prefix marks
//! (`MarkStore::None`, `mark_bytes = 0`). Exact prefix restore is purely pages-only.
pub(super) use crate::families::hy4_common::weights::slice_buf;
use crate::shared::memory::device::{Allocation, Device, Stream};
use crate::shared::memory::receipts::Inventory;
use crate::shared::memory::Tag;
use crate::shared::prefix::cuda_copy::SnapshotOwner;
use crate::shared::prefix::{PrefixCopies, Reader};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_loader::families::hy4::execution::{CacheLayout, INDEX_PAGE_BYTES};
use cuteafd_loader::families::hy4::{Hy4Config, Hy4KvCache};
use cuteafd_loader::families::hy4_common::PAGE_ROWS;
use std::collections::BTreeSet;
use std::rc::Rc;

/// Per-token E4M3 key bytes inside one 8448-byte `dsa_index` page (`64 * 128 = 8192`).
const INDEX_KEYS_PLANE_BYTES: usize = PAGE_ROWS * 128;

/// Committed per-rank pages-only KV/index storage with fenced COW tail copies.
pub(crate) struct Hy4CacheRank<'a> {
    copies: PrefixCopies<'a>,
    readers: Vec<Rc<Stream<'a>>>,
    pub(crate) device: Device<'a>,
    pub(crate) kv_format: Hy4KvCache,
    pub(crate) layout: CacheLayout,
    pub(crate) kv_layers: usize,
    pub(crate) full_index_layers: usize,
    pub(crate) pages: usize,
    pub(crate) sequences: usize,
    /// `[kv_layers, pages, 64, record_bytes]`
    pub(crate) dsa_kv: Allocation<'a>,
    /// `[full_index_layers, pages, 8448]` (`64 * 128` E4M3 keys + `64 * 4` F32 scales)
    pub(crate) dsa_index: Allocation<'a>,
}

impl<'a> Hy4CacheRank<'a> {
    pub(crate) fn new(
        readers: Vec<Rc<Stream<'a>>>,
        cfg: &Hy4Config,
        kv_format: Hy4KvCache,
        pages: usize,
        sequences: usize,
    ) -> Result<Self> {
        Self::new_owned(
            readers,
            kv_format,
            cfg.layers,
            cfg.index_cache.full_count(),
            pages,
            sequences,
        )
    }

    pub(crate) fn new_owned(
        readers: Vec<Rc<Stream<'a>>>,
        kv_format: Hy4KvCache,
        kv_layers: usize,
        full_index_layers: usize,
        pages: usize,
        sequences: usize,
    ) -> Result<Self> {
        let first = readers
            .first()
            .context("Hy4 cache requires at least one reader stream")?;
        let device = first.device;
        ensure!(
            readers
                .iter()
                .all(|s| s.device.id == device.id && std::ptr::eq(s.device.library, device.library)),
            "Hy4 cache reader devices differ"
        );
        ensure!(
            pages > 0 && sequences > 0 && kv_layers > 0 && full_index_layers > 0,
            "Hy4 cache requires positive layers, pages and sequences"
        );
        let layout = CacheLayout::new(kv_format, kv_layers, full_index_layers)?;

        let dsa_kv_bytes = kv_layers
            .checked_mul(pages)
            .and_then(|v| v.checked_mul(layout.kv_page_bytes))
            .context("Hy4 DSA KV pool overflow")?;
        let dsa_index_bytes = full_index_layers
            .checked_mul(pages)
            .and_then(|v| v.checked_mul(layout.index_page_bytes))
            .context("Hy4 DSA index pool overflow")?;

        let dsa_kv = Allocation::new(device, dsa_kv_bytes, Tag::kv("hy4/dsa-kv"))?;
        let dsa_index = Allocation::new(device, dsa_index_bytes, Tag::kv("hy4/dsa-index"))?;

        let copies = PrefixCopies::new(
            device.library,
            vec![readers
                .iter()
                .map(|s| Reader {
                    device: s.device.id,
                    stream: s.raw,
                })
                .collect()],
        )?;

        let owner = Self {
            copies,
            readers,
            device,
            kv_format,
            layout,
            kv_layers,
            full_index_layers,
            pages,
            sequences,
            dsa_kv,
            dsa_index,
        };
        owner.zero_all()?;
        Ok(owner)
    }

    pub(crate) fn zero_all(&self) -> Result<()> {
        self.copies.ordered(0, |stream| {
            // SAFETY: allocations live on `self.device` and readers are fenced on both sides.
            unsafe {
                self.device
                    .library
                    .cuda_zero_bytes_async(self.dsa_kv.buffer, self.dsa_kv.buffer.bytes, stream)?;
                self.device.library.cuda_zero_bytes_async(
                    self.dsa_index.buffer,
                    self.dsa_index.buffer.bytes,
                    stream,
                )?;
            }
            Ok(())
        })?;
        self.copies.drain()
    }

    pub(crate) fn dsa_kv_layer(&self, layer: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(layer < self.kv_layers, "Hy4 KV layer {layer} out of range");
        let layer_bytes = self.pages * self.layout.kv_page_bytes;
        slice_buf(self.dsa_kv.buffer, layer * layer_bytes, layer_bytes)
    }

    pub(crate) fn dsa_kv_page(&self, layer: usize, page: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(page < self.pages, "Hy4 KV page {page} outside pool {}", self.pages);
        let bank = self.dsa_kv_layer(layer)?;
        slice_buf(bank, page * self.layout.kv_page_bytes, self.layout.kv_page_bytes)
    }

    pub(crate) fn dsa_index_layer(&self, full_idx: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(
            full_idx < self.full_index_layers,
            "Hy4 full indexer layer index {full_idx} out of range"
        );
        let layer_bytes = self.pages * self.layout.index_page_bytes;
        slice_buf(self.dsa_index.buffer, full_idx * layer_bytes, layer_bytes)
    }

    pub(crate) fn dsa_index_page(
        &self,
        full_idx: usize,
        page: usize,
    ) -> Result<CuteafdDeviceBuffer> {
        ensure!(
            page < self.pages,
            "Hy4 index page {page} outside pool {}",
            self.pages
        );
        let bank = self.dsa_index_layer(full_idx)?;
        slice_buf(bank, page * self.layout.index_page_bytes, self.layout.index_page_bytes)
    }

    /// Exact COW partial-page copy across all 78 KV layers and all 21 full IndexCache layers.
    ///
    /// Each 8448-byte `dsa_index` page stores `64 * 128` E4M3 keys followed by `64 * 4` FP32
    /// scales, so a partial-page copy of `rows` copies both `[0..rows*128]` and
    /// `[8192..8192+rows*4]`.
    pub(crate) fn copy_page_rows(&self, from: usize, to: usize, rows: usize) -> Result<()> {
        ensure!(
            from != to && (1..=PAGE_ROWS).contains(&rows),
            "Hy4 page tail copy aliases or exceeds page"
        );
        let kv_bytes = rows * self.kv_format.record_bytes();
        let idx_key_bytes = rows * 128;
        let idx_scale_bytes = rows * 4;
        let mut copies = Vec::with_capacity(self.kv_layers + self.full_index_layers * 2);
        for layer in 0..self.kv_layers {
            let src_kv = slice_buf(self.dsa_kv_page(layer, from)?, 0, kv_bytes)?;
            let dst_kv = slice_buf(self.dsa_kv_page(layer, to)?, 0, kv_bytes)?;
            copies.push((0, dst_kv, src_kv));
        }
        for full_idx in 0..self.full_index_layers {
            let src_page = self.dsa_index_page(full_idx, from)?;
            let dst_page = self.dsa_index_page(full_idx, to)?;
            if rows == PAGE_ROWS {
                copies.push((0, dst_page, src_page));
            } else {
                copies.push((
                    0,
                    slice_buf(dst_page, 0, idx_key_bytes)?,
                    slice_buf(src_page, 0, idx_key_bytes)?,
                ));
                copies.push((
                    0,
                    slice_buf(dst_page, INDEX_KEYS_PLANE_BYTES, idx_scale_bytes)?,
                    slice_buf(src_page, INDEX_KEYS_PLANE_BYTES, idx_scale_bytes)?,
                ));
            }
        }
        // SAFETY: `from != to` guarantees disjoint pages; `PrefixCopies` fences all readers.
        unsafe { self.copies.copy_d2d(&copies) }
    }

    pub(crate) fn drain(&self) -> Result<()> {
        self.copies.drain()
    }

    pub(crate) fn page_bytes_total(&self) -> usize {
        self.kv_layers * self.layout.kv_page_bytes
            + self.full_index_layers * self.layout.index_page_bytes
    }

    pub(crate) fn inventory(&self, inventory: &mut Inventory<'_>) -> Result<()> {
        let location = inventory.location(self.device.id)?;
        let cuteafd_loader::placement::planner::Device::Coordinator(gpu) = location.device else {
            anyhow::bail!("Hy4 cache is not coordinator-owned");
        };
        self.dsa_kv
            .inventory(inventory, &format!("hy4.cache.{gpu}.dsa_kv"))?;
        self.dsa_index
            .inventory(inventory, &format!("hy4.cache.{gpu}.dsa_index"))?;
        self.copies
            .allocation_receipts(inventory, &format!("hy4.cache.{gpu}.copies"))
    }

    /// Read back every committed causal state byte for `placement` on this rank for exact-restore verification.
    pub(crate) fn dump_state(
        &self,
        placement: &Hy4Placement,
        parity_rank: Option<usize>,
    ) -> Result<Hy4RankStateSnapshot> {
        self.drain()?;
        for reader in &self.readers {
            reader.drain()?;
        }
        let rec = self.kv_format.record_bytes();
        let len = placement.len;
        let owned_rows = match parity_rank {
            Some(rank) => crate::shared::peer_split::context::shard_rows(len, PAGE_ROWS, rank),
            None => len,
        };
        let mut dsa_kv = vec![0u8; self.kv_layers * owned_rows * rec];
        let mut dsa_index_keys = vec![0u8; self.full_index_layers * owned_rows * 128];
        let mut dsa_index_scales = vec![0u8; self.full_index_layers * owned_rows * 4];

        self.device.run(|| {
            for layer in 0..self.kv_layers {
                let mut row_cursor = 0usize;
                let mut out_cursor = 0usize;
                let mut page_idx = 0usize;
                while row_cursor < len {
                    let take = (len - row_cursor).min(PAGE_ROWS);
                    let include = parity_rank.is_none_or(|rank| {
                        crate::shared::peer_split::context::page_owner(page_idx) == rank
                    });
                    if include {
                        let raw_id = placement.units[page_idx] as usize;
                        let phys = if let Some(rank) = parity_rank {
                            ensure!(
                                raw_id & 1 == rank,
                                "Hy4 parity dump_state page {page_idx} id {raw_id} != rank {rank}"
                            );
                            raw_id >> 1
                        } else {
                            raw_id
                        };
                        let kv_buf = slice_buf(self.dsa_kv_page(layer, phys)?, 0, take * rec)?;
                        let dst_off = (layer * owned_rows + out_cursor) * rec;
                        self.device
                            .library
                            .copy_d2h(&mut dsa_kv[dst_off..dst_off + take * rec], kv_buf)?;
                        out_cursor += take;
                    }
                    row_cursor += take;
                    page_idx += 1;
                }
            }
            for full_idx in 0..self.full_index_layers {
                let mut row_cursor = 0usize;
                let mut out_cursor = 0usize;
                let mut page_idx = 0usize;
                while row_cursor < len {
                    let take = (len - row_cursor).min(PAGE_ROWS);
                    let include = parity_rank.is_none_or(|rank| {
                        crate::shared::peer_split::context::page_owner(page_idx) == rank
                    });
                    if include {
                        let raw_id = placement.units[page_idx] as usize;
                        let phys = if parity_rank.is_some() {
                            raw_id >> 1
                        } else {
                            raw_id
                        };
                        let page_buf = self.dsa_index_page(full_idx, phys)?;
                        let k_buf = slice_buf(page_buf, 0, take * 128)?;
                        let s_buf = slice_buf(page_buf, INDEX_KEYS_PLANE_BYTES, take * 4)?;
                        let k_off = (full_idx * owned_rows + out_cursor) * 128;
                        let s_off = (full_idx * owned_rows + out_cursor) * 4;
                        self.device
                            .library
                            .copy_d2h(&mut dsa_index_keys[k_off..k_off + take * 128], k_buf)?;
                        self.device
                            .library
                            .copy_d2h(&mut dsa_index_scales[s_off..s_off + take * 4], s_buf)?;
                        out_cursor += take;
                    }
                    row_cursor += take;
                    page_idx += 1;
                }
            }
            Ok(())
        })?;

        Ok(Hy4RankStateSnapshot {
            len,
            dsa_kv,
            dsa_index_keys,
            dsa_index_scales,
        })
    }
}

/// Complete committed causal state read back from one coordinator GPU rank for exact restore checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Hy4RankStateSnapshot {
    pub(crate) len: usize,
    pub(crate) dsa_kv: Vec<u8>,
    pub(crate) dsa_index_keys: Vec<u8>,
    pub(crate) dsa_index_scales: Vec<u8>,
}

impl Hy4RankStateSnapshot {
    pub(crate) fn total_bytes(&self) -> usize {
        self.dsa_kv.len() + self.dsa_index_keys.len() + self.dsa_index_scales.len()
    }

    pub(crate) fn diff(&self, other: &Self) -> Vec<String> {
        let mut diffs = Vec::new();
        if self.len != other.len {
            diffs.push(format!("len: {} != {}", self.len, other.len));
        }
        for (name, a, b) in [
            ("dsa_kv", &self.dsa_kv, &other.dsa_kv),
            ("dsa_index_keys", &self.dsa_index_keys, &other.dsa_index_keys),
            ("dsa_index_scales", &self.dsa_index_scales, &other.dsa_index_scales),
        ] {
            if a.len() != b.len() {
                diffs.push(format!("{name}: len {} != {}", a.len(), b.len()));
            } else if let Some(first) = a.iter().zip(b.iter()).position(|(x, y)| x != y) {
                let count = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
                diffs.push(format!(
                    "{name}: {count}/{} bytes differ (first at byte {first}: 0x{:02x} != 0x{:02x})",
                    a.len(),
                    a[first],
                    b[first]
                ));
            }
        }
        diffs
    }
}

impl SnapshotOwner for Hy4CacheRank<'_> {
    fn snapshot_buffers(&self) -> Result<Vec<CuteafdDeviceBuffer>> {
        Ok(vec![self.dsa_kv.buffer, self.dsa_index.buffer])
    }
}

/// Logical COW pages, active slot, causal frontier, M-RoPE coordinates, and
/// bound still-image feature spans for one Hy4 sequence.
#[derive(Clone, Debug)]
pub(crate) struct Hy4Placement {
    pub units: Vec<u32>,
    pub slot: usize,
    pub len: usize,
    pub rope: std::sync::Arc<cuteafd_loader::families::hy4::vision::RopePositions>,
    pub media: Option<std::sync::Arc<cuteafd_engine::media::RequestMedia>>,
}

impl Hy4Placement {
    pub(crate) fn new(units: Vec<u32>, slot: usize) -> Self {
        Self {
            units,
            slot,
            len: 0,
            rope: Default::default(),
            media: None,
        }
    }

    pub(crate) fn rope_at(&self, pos: usize) -> Result<[i64; 3]> {
        self.rope.at(pos)
    }
}

impl crate::shared::qwen::scheduler::Placement for Hy4Placement {
    fn len(&self) -> usize {
        self.len
    }
    fn set_len(&mut self, len: usize) {
        self.len = len;
    }
    fn units(&self) -> &[u32] {
        &self.units
    }
    fn slot(&self) -> i32 {
        self.slot as i32
    }
    fn spans(&self) -> &[cuteafd_core::MediaSpan] {
        self.media.as_deref().map_or(&[], cuteafd_engine::media::RequestMedia::spans)
    }
}

/// Pages-only exact prefix-cache adapter across 1 or 2 Hy4 coordinator ranks.
#[derive(Clone)]
pub(crate) struct Hy4Prefix<'a> {
    ranks: Vec<Rc<Hy4CacheRank<'a>>>,
    parity: bool,
}

impl<'a> Hy4Prefix<'a> {
    pub(crate) fn new(ranks: Vec<Rc<Hy4CacheRank<'a>>>, parity: bool) -> Result<Self> {
        ensure!(!ranks.is_empty(), "Hy4Prefix requires at least one rank");
        ensure!(
            !parity || ranks.len() == 2,
            "Hy4Prefix parity pool requires two coordinator ranks"
        );
        if parity {
            ensure!(
                ranks[0].pages == ranks[1].pages,
                "Hy4Prefix parity ranks must have equal local page capacities ({} != {})",
                ranks[0].pages,
                ranks[1].pages
            );
        }
        Ok(Self { ranks, parity })
    }

    pub(crate) fn snapshot_owners(&self) -> Vec<Rc<dyn SnapshotOwner + 'a>> {
        self.ranks
            .iter()
            .map(|r| r.clone() as Rc<dyn SnapshotOwner>)
            .collect()
    }

    fn validate(&self, placement: &Hy4Placement, len: usize) -> Result<()> {
        let r0 = &self.ranks[0];
        ensure!(
            placement.slot < r0.sequences && len.div_ceil(PAGE_ROWS) <= placement.units.len(),
            "Hy4 prefix placement outside admitted state/pages"
        );
        if self.parity {
            for (j, &id) in placement.units.iter().enumerate() {
                let owner = crate::shared::peer_split::context::page_owner(j);
                let local = (id as usize) >> 1;
                ensure!(
                    (id as usize) & 1 == owner && local < self.ranks[owner].pages,
                    "Hy4 parity page {j} id {id} outside owner {owner} pool {}",
                    self.ranks[owner].pages
                );
            }
        } else {
            ensure!(
                placement.units.iter().all(|&p| (p as usize) < r0.pages),
                "Hy4 prefix placement outside admitted pages"
            );
        }
        ensure!(
            placement.units.iter().copied().collect::<BTreeSet<_>>().len() == placement.units.len(),
            "Hy4 prefix placement aliases physical pages"
        );
        Ok(())
    }
}

impl cuteafd_engine::prefix::PrefixFamily for Hy4Prefix<'_> {
    type Placement = Hy4Placement;

    fn record_format(&self) -> &'static str {
        match self.ranks[0].kv_format {
            Hy4KvCache::Fp8 => "mla_fp8_ds_v1",
            Hy4KvCache::Bf16 => "mla_bf16_v1",
        }
    }

    fn layout(&self) -> cuteafd_engine::prefix::FamilyLayout {
        use cuteafd_engine::prefix::{FamilyLayout, MarkStore, PageOwners, ReuseRule};
        FamilyLayout {
            page_rows: PAGE_ROWS,
            pages: if self.parity {
                self.ranks[0].pages * 2
            } else {
                self.ranks[0].pages
            },
            page_bytes: if self.parity {
                self.ranks[0].page_bytes_total()
            } else {
                self.ranks.iter().map(|r| r.page_bytes_total()).sum()
            },
            mark_bytes: 0,
            draft_bytes: 0,
            rule: ReuseRule::EXACT,
            mark_store: MarkStore::Arena,
            page_owners: if self.parity {
                PageOwners::Parity
            } else {
                PageOwners::Uniform
            },
        }
    }

    fn pages<'p>(&self, placement: &'p Hy4Placement) -> &'p [u32] {
        &placement.units
    }

    fn bind(&self, placement: &mut Hy4Placement) -> Result<(), cuteafd_engine::prefix::BoxError> {
        self.validate(placement, placement.len)?;
        Ok(())
    }

    fn discard(&self, _placement: &mut Hy4Placement) -> Result<(), cuteafd_engine::prefix::BoxError> {
        Ok(())
    }

    fn extend_pages(
        &self,
        placement: &mut Hy4Placement,
        pages: &[u32],
    ) -> Result<(), cuteafd_engine::prefix::BoxError> {
        let mut extended = placement.clone();
        extended.units.extend_from_slice(pages);
        self.validate(&extended, extended.len)?;
        placement.units = extended.units;
        Ok(())
    }

    fn commit_point(&self, placement: &Hy4Placement) -> usize {
        placement.len
    }

    fn capture(
        &self,
        _slot: cuteafd_engine::prefix::MarkSlot,
        placement: &Hy4Placement,
        len: usize,
    ) -> Result<(), cuteafd_engine::prefix::BoxError> {
        self.validate(placement, len)?;
        if len != placement.len {
            return Err("Hy4 snapshot frontier does not match committed length".into());
        }
        Ok(())
    }

    fn restore(
        &self,
        _mark: Option<cuteafd_engine::prefix::MarkSlot>,
        placement: &mut Hy4Placement,
        len: usize,
    ) -> Result<(), cuteafd_engine::prefix::BoxError> {
        self.validate(placement, len)?;
        placement.len = len;
        Ok(())
    }

    fn copy_rows(
        &self,
        copy: cuteafd_engine::prefix::TailCopy,
    ) -> Result<(), cuteafd_engine::prefix::BoxError> {
        if self.parity {
            let Some(owner) = copy.owner else {
                return Err("Hy4 parity tail copy missing owner rank".into());
            };
            let rank = usize::from(owner);
            if rank >= self.ranks.len()
                || (copy.from as usize & 1) != rank
                || (copy.to as usize & 1) != rank
            {
                return Err(format!("Hy4 parity tail copy {copy:?} crosses GPUs").into());
            }
            self.ranks[rank].copy_page_rows(
                (copy.from >> 1) as usize,
                (copy.to >> 1) as usize,
                copy.rows,
            )?;
            return Ok(());
        }
        if copy.owner.is_some() {
            return Err("Hy4 whole-rank pages have no parity owner".into());
        }
        for rank in &self.ranks {
            rank.copy_page_rows(copy.from as usize, copy.to as usize, copy.rows)?;
        }
        Ok(())
    }

    fn drain(&self) -> Result<(), cuteafd_engine::prefix::BoxError> {
        for rank in &self.ranks {
            rank.drain()?;
        }
        Ok(())
    }

    fn page_device(&self, page: u32) -> Option<u8> {
        self.parity.then_some((page & 1) as u8)
    }

    fn page_segments(&self, page: u32) -> Vec<cuteafd_hostcache::copy::DeviceRange> {
        let mut out = Vec::new();
        if self.parity {
            let owner = (page & 1) as usize;
            let local = (page >> 1) as usize;
            let rank = &self.ranks[owner];
            for layer in 0..rank.kv_layers {
                if let Ok(b) = rank.dsa_kv_page(layer, local) {
                    out.push(crate::shared::prefix::cuda_copy::range(b));
                }
            }
            for full_idx in 0..rank.full_index_layers {
                if let Ok(b) = rank.dsa_index_page(full_idx, local) {
                    out.push(crate::shared::prefix::cuda_copy::range(b));
                }
            }
            return out;
        }
        for rank in &self.ranks {
            for layer in 0..rank.kv_layers {
                if let Ok(b) = rank.dsa_kv_page(layer, page as usize) {
                    out.push(crate::shared::prefix::cuda_copy::range(b));
                }
            }
            for full_idx in 0..rank.full_index_layers {
                if let Ok(b) = rank.dsa_index_page(full_idx, page as usize) {
                    out.push(crate::shared::prefix::cuda_copy::range(b));
                }
            }
        }
        out
    }

    fn mark_segments(
        &self,
        _slot: cuteafd_engine::prefix::MarkSlot,
    ) -> Vec<cuteafd_hostcache::copy::DeviceRange> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::prefix::CudaCopyEngine;
    use cuteafd_engine::prefix::{After, PrefixCache, PrefixConfig, PrefixFamily, SnapshotKind};
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use cuteafd_loader::families::hy4::{Hy4IndexerKind, IndexCacheSchedule};

    #[test]
    fn hy4_pages_only_prefix_cache_restores_and_copies_tails_byte_identically() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let device = Device {
            library: &library,
            id: 0,
        };
        let reader = Rc::new(Stream::new(device)?);
        let cfg = Hy4Config {
            vocab_size: 120_832,
            hidden: 6144,
            layers: 4,
            intermediate: 18_432,
            max_position: 1_048_576,
            rms_norm_eps: 1e-5,
            qa_kva_eps: 1e-6,
            hc_mult: 4,
            hc_eps: 1e-6,
            hc_magnitude: 2.0,
            heads: 64,
            q_lora_rank: 2048,
            kv_lora_rank: 512,
            qk_nope_dim: 192,
            qk_rope_dim: 64,
            v_head_dim: 256,
            rope_theta: 10_000_000.0,
            indexer_heads: 32,
            indexer_head_dim: 128,
            indexer_topk: 2048,
            index_cache: IndexCacheSchedule::from_kinds(vec![
                Hy4IndexerKind::Full,
                Hy4IndexerKind::Full,
                Hy4IndexerKind::Shared,
                Hy4IndexerKind::Shared,
            ])?,
            dense_layer: vec![true, false, false, false],
            experts: 256,
            topk: 8,
            moe_intermediate: 2048,
            shared_experts: 1,
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 2.827,
            swiglu_limit: 10.0,
            image_token_id: 120_120,
            video_token_id: 120_124,
            eos: vec![120_025],
        };
        let rank = Rc::new(Hy4CacheRank::new(vec![reader], &cfg, Hy4KvCache::Fp8, 8, 2)?);
        let family = Hy4Prefix::new(vec![rank.clone()], false)?;
        assert_eq!(family.layout().mark_bytes, 0);
        assert_eq!(rank.layout.index_page_bytes, INDEX_PAGE_BYTES);

        let mut cache = PrefixCache::<CudaCopyEngine<'_>>::new(
            family.layout(),
            PrefixConfig {
                entries: 4,
                mark_slots: 0,
                keep_logits: false,
                min_tokens: 1,
            },
            None,
        )?;

        // Write 75 tokens (1 full page + 11 partial rows) with distinct KV and split-plane index bytes.
        let tokens: Vec<u32> = (10..85).collect();
        let mut p0 = cache
            .admit(&family, &tokens, 128, false, |units| Hy4Placement::new(units, 0))?
            .placement;
        p0.len = tokens.len();
        let rec = Hy4KvCache::Fp8.record_bytes();
        for layer in 0..4 {
            for (lp, &phys) in p0.units.iter().enumerate() {
                let rows = if lp == 0 { PAGE_ROWS } else { 11 };
                let kv_data: Vec<u8> = (0..rows * rec)
                    .map(|i| ((layer * 37 + lp * 13 + i) & 0xff) as u8)
                    .collect();
                library.copy_h2d(slice_buf(rank.dsa_kv_page(layer, phys as usize)?, 0, kv_data.len())?, &kv_data)?;
            }
        }
        for full_idx in 0..2 {
            for (lp, &phys) in p0.units.iter().enumerate() {
                let rows = if lp == 0 { PAGE_ROWS } else { 11 };
                let k_data: Vec<u8> = (0..rows * 128)
                    .map(|i| ((full_idx * 53 + lp * 19 + i) & 0xff) as u8)
                    .collect();
                let s_data: Vec<u8> = (0..rows * 4)
                    .map(|i| ((full_idx * 71 + lp * 23 + i) & 0xff) as u8)
                    .collect();
                let page = rank.dsa_index_page(full_idx, phys as usize)?;
                library.copy_h2d(slice_buf(page, 0, k_data.len())?, &k_data)?;
                library.copy_h2d(slice_buf(page, INDEX_KEYS_PLANE_BYTES, s_data.len())?, &s_data)?;
            }
        }
        let expected = rank.dump_state(&p0, None)?;

        let captured = cache.capture(
            &family,
            SnapshotKind::Prompt,
            &tokens,
            &p0,
            After::from_logits(&[1.0, 0.0], true),
        )?;
        assert!(captured);

        // Restore into slot 1 with an extra diverging token while p0 stays pinned so the partial tail page COW-copies.
        let mut extended = tokens.clone();
        extended.push(999);
        let restored = cache.admit(&family, &extended, 128, false, |units| Hy4Placement::new(units, 1))?;
        assert_eq!(restored.resume, tokens.len());
        let p1 = restored.placement;
        assert_eq!(p1.units[0], p0.units[0], "full page must share zero-copy");
        assert_ne!(p1.units[1], p0.units[1], "partial tail page must COW-copy while source stays pinned");
        let actual = rank.dump_state(&p1, None)?;
        assert!(
            expected.diff(&actual).is_empty(),
            "restored state differs: {:?}",
            expected.diff(&actual)
        );
        // Verify 1-byte mutations in dsa_kv and both planes of dsa_index (keys and scales) are detected.
        let mut flip_kv = actual.clone();
        flip_kv.dsa_kv[0] ^= 1;
        assert!(expected.diff(&flip_kv).iter().any(|d| d.starts_with("dsa_kv:")));
        let mut flip_idx_key = actual.clone();
        flip_idx_key.dsa_index_keys[0] ^= 1;
        assert!(expected.diff(&flip_idx_key).iter().any(|d| d.starts_with("dsa_index_keys:")));
        let mut flip_idx_scale = actual.clone();
        *flip_idx_scale.dsa_index_scales.last_mut().unwrap() ^= 1;
        assert!(expected.diff(&flip_idx_scale).iter().any(|d| d.starts_with("dsa_index_scales:")));
        cache.release(&family, &p0.units)?;
        cache.release(&family, &p1.units)?;
        Ok(())
    }

    #[test]
    fn hy4_parity_prefix_cache_exact_restore_and_cow_tail() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let dev0 = Device { library: &library, id: 0 };
        let dev1 = Device { library: &library, id: 1 };
        let r0_stream = Rc::new(Stream::new(dev0)?);
        let r1_stream = Rc::new(Stream::new(dev1)?);
        let cfg = Hy4Config {
            vocab_size: 120_832,
            hidden: 6144,
            layers: 4,
            intermediate: 18_432,
            max_position: 262_144,
            rms_norm_eps: 1e-5,
            qa_kva_eps: 1e-6,
            hc_mult: 4,
            hc_eps: 1e-6,
            hc_magnitude: 2.0,
            q_lora_rank: 2048,
            kv_lora_rank: 512,
            qk_nope_dim: 192,
            qk_rope_dim: 64,
            v_head_dim: 256,
            heads: 64,
            rope_theta: 10_000_000.0,
            indexer_heads: 32,
            indexer_head_dim: 128,
            indexer_topk: 2048,
            index_cache: IndexCacheSchedule::from_kinds(vec![
                Hy4IndexerKind::Full,
                Hy4IndexerKind::Full,
                Hy4IndexerKind::Shared,
                Hy4IndexerKind::Shared,
            ])?,
            dense_layer: vec![true, false, false, false],
            experts: 256,
            topk: 8,
            moe_intermediate: 2048,
            shared_experts: 1,
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 2.827,
            swiglu_limit: 10.0,
            image_token_id: 120_120,
            video_token_id: 120_124,
            eos: vec![120_025],
        };
        let rank0 = Rc::new(Hy4CacheRank::new(vec![r0_stream], &cfg, Hy4KvCache::Fp8, 16, 2)?);
        let rank1 = Rc::new(Hy4CacheRank::new(vec![r1_stream], &cfg, Hy4KvCache::Fp8, 16, 2)?);
        let family = Hy4Prefix::new(vec![rank0.clone(), rank1.clone()], true)?;
        let layout = family.layout();
        assert_eq!(layout.page_rows, PAGE_ROWS);
        assert_eq!(layout.pages, 32);
        assert_eq!(layout.mark_bytes, 0);
        assert_eq!(layout.page_owners, cuteafd_engine::prefix::PageOwners::Parity);

        let mut cache: PrefixCache<CudaCopyEngine<'_>> = PrefixCache::new(
            layout,
            PrefixConfig { entries: 4, mark_slots: 0, keep_logits: false, min_tokens: 1 },
            None,
        )?;
        let scale_plane_bytes = INDEX_PAGE_BYTES - INDEX_KEYS_PLANE_BYTES;

        // Odd-parity tail (67 tokens -> logical page 1 on rank 1) and even-parity tail (131 tokens -> logical page 2 on rank 0).
        for (base, len) in [(100u32, 67usize), (500u32, 131usize)] {
            let tokens: Vec<u32> = (base..base + len as u32).collect();
            let mut src = cache
                .admit(&family, &tokens, len + 64, false, |units| Hy4Placement::new(units, 0))?
                .placement;
            src.len = len;
            for (j, &id) in src.units.iter().enumerate() {
                assert_eq!((id as usize) & 1, j & 1, "logical page {j} must match parity owner");
                let owner = (id as usize) & 1;
                let local = (id as usize) >> 1;
                let rank = if owner == 0 { &rank0 } else { &rank1 };
                for layer in 0..cfg.layers {
                    let kv = rank.dsa_kv_page(layer, local)?;
                    library.copy_h2d(kv, &vec![(id as u8).wrapping_add(layer as u8 * 17 + 0x31); kv.bytes])?;
                }
                for full_idx in 0..cfg.full_indexer_layers().len() {
                    let idx = rank.dsa_index_page(full_idx, local)?;
                    library.copy_h2d(slice_buf(idx, 0, INDEX_KEYS_PLANE_BYTES)?, &vec![(id as u8).wrapping_add(full_idx as u8 * 19 + 0x51); INDEX_KEYS_PLANE_BYTES])?;
                    library.copy_h2d(slice_buf(idx, INDEX_KEYS_PLANE_BYTES, scale_plane_bytes)?, &vec![(id as u8).wrapping_add(full_idx as u8 * 29 + 0x71); scale_plane_bytes])?;
                }
            }
            assert!(cache.capture(&family, SnapshotKind::Prompt, &tokens, &src, After::from_logits(&[1.0, 0.0], true))?);

            let mut ext = tokens.clone();
            ext.push(9999);
            let restored = cache.admit(&family, &ext, len + 64, false, |units| Hy4Placement::new(units, 1))?;
            assert_eq!(restored.resume, len);
            let p1 = restored.placement;
            let tail = len / PAGE_ROWS;
            assert_ne!(p1.units[tail], src.units[tail], "partial tail page must COW-copy while source stays pinned");
            assert_eq!((p1.units[tail] as usize) & 1, tail & 1, "COW tail page must keep parity owner");
            for (r_idx, rank) in [&rank0, &rank1].into_iter().enumerate() {
                let snap_src = rank.dump_state(&src, Some(r_idx))?;
                let snap_dst = rank.dump_state(&p1, Some(r_idx))?;
                assert!(snap_src.diff(&snap_dst).is_empty(), "rank {r_idx} len {len} diffs: {:?}", snap_src.diff(&snap_dst));
            }
            cache.release(&family, &src.units)?;
            cache.release(&family, &p1.units)?;
        }
        Ok(())
    }
}
