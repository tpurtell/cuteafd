//! Committed KV, compressed-indexer group cache, SWA-512 rings and exact prefix marks.
//!
//! Layout per coordinator GPU rank:
//! - `dsa_kv`: `[dsa_layers, pages, 64, record_bytes]` (RefPagePool COW physical pages)
//! - `dsa_index`: `[dsa_layers, pages, 16 * 132]` (RefPagePool COW physical pages, 4-token groups)
//! - `swa_rings`: `[swa_layers, sequences * swa_ring_rows, record_bytes]` (contiguous per layer so
//!   batched `mla_swa_decode`/`mla_swa_extend` attends all sequences in one launch via page offset)
//! - `pending_keys`: `[sequences, dsa_layers, 3, 128 * 4]` (incomplete `<4` token keys per DSA layer)
//! - `marks`: `[mark_slots, mark_bytes]` (exact SWA rings + pending keys; ~16.9 MiB/mark FP8)
use crate::shared::memory::device::{Allocation, Device, Stream};
use crate::shared::memory::receipts::Inventory;
use crate::shared::memory::Tag;
use crate::shared::prefix::cuda_copy::SnapshotOwner;
use crate::shared::prefix::{PrefixCopies, Reader};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_loader::families::hy4_flash::{
    execution::{CacheLayout, PAGE_ROWS, SWA_RING_PAGES},
    Hy4FlashConfig, Hy4KvCache,
};
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;

/// Bytes per sequence per DSA layer reserved for `0..=3` pending incomplete indexer token keys.
pub(crate) const PENDING_KEY_SLOT_BYTES: usize = 3 * 128 * 4;

pub(super) use crate::families::hy4_common::weights::slice_buf;

/// Committed per-rank KV/index/SWA storage with fenced prefix capture/restore and COW tail copies.
pub(crate) struct Hy4CacheRank<'a> {
    copies: PrefixCopies<'a>,
    readers: Vec<Rc<Stream<'a>>>,
    pub(crate) device: Device<'a>,
    pub(crate) kv_format: Hy4KvCache,
    pub(crate) layout: CacheLayout,
    pub(crate) dsa_layers: usize,
    pub(crate) swa_layers: usize,
    pub(crate) pages: usize,
    pub(crate) sequences: usize,
    pub(crate) swa_ring_pages: usize,
    pub(crate) swa_ring_rows: usize,
    pub(crate) swa_ring: cuteafd_core::prefix::AppendRing,
    pub(crate) mark_slots: usize,
    pub(crate) mark_bytes_per_slot: usize,
    /// `[dsa_layers, pages, 64, record_bytes]`
    pub(crate) dsa_kv: Allocation<'a>,
    /// `[dsa_layers, pages, 16 * 132]`
    pub(crate) dsa_index: Allocation<'a>,
    /// `[swa_layers, sequences * swa_ring_rows, record_bytes]`
    pub(crate) swa_rings: Allocation<'a>,
    /// `[sequences, dsa_layers, 3 * 128 * 4]`
    pub(crate) pending_keys: Allocation<'a>,
    /// Optional prefix-mark storage (`mark_slots * mark_bytes_per_slot`).
    pub(crate) marks: Option<Allocation<'a>>,
    /// Per-sequence count of pending token keys (`0..=3`) per DSA layer.
    pub(crate) pending_counts: RefCell<Vec<Vec<usize>>>,
}

impl<'a> Hy4CacheRank<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        readers: Vec<Rc<Stream<'a>>>,
        cfg: &Hy4FlashConfig,
        kv_format: Hy4KvCache,
        pages: usize,
        sequences: usize,
        max_chunk_rows: usize,
        mark_slots: usize,
    ) -> Result<Self> {
        Self::new_owned(
            readers,
            cfg,
            kv_format,
            cfg.sparse_layers().len(),
            cfg.swa_layers().len(),
            pages,
            sequences,
            max_chunk_rows,
            mark_slots,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_owned(
        readers: Vec<Rc<Stream<'a>>>,
        cfg: &Hy4FlashConfig,
        kv_format: Hy4KvCache,
        dsa_layers: usize,
        swa_layers: usize,
        pages: usize,
        sequences: usize,
        max_chunk_rows: usize,
        mark_slots: usize,
    ) -> Result<Self> {
        let first = readers.first().context("Hy4-flash cache requires at least one reader stream")?;
        let device = first.device;
        ensure!(
            readers.iter().all(|s| s.device.id == device.id && std::ptr::eq(s.device.library, device.library)),
            "Hy4-flash cache reader devices differ"
        );
        ensure!(pages > 0 && sequences > 0, "Hy4-flash cache requires positive pages and sequences");
        let layout = CacheLayout::for_unit(cfg, kv_format, sequences, dsa_layers, swa_layers, max_chunk_rows)?;
        let swa_ring_pages = layout.swa_ring_pages;
        let swa_ring_rows = layout.swa_ring_rows;
        let swa_ring = cuteafd_core::prefix::AppendRing {
            window: cfg.sliding_window as u64,
            step: (swa_ring_rows - cfg.sliding_window) as u64,
            rows: swa_ring_rows as u64,
        };
        let swa_rings_bytes = layout.active_swa_ring_bytes;

        let dsa_kv_bytes = dsa_layers
            .checked_mul(pages)
            .and_then(|v| v.checked_mul(layout.dsa_kv_page_bytes))
            .context("Hy4-flash DSA KV pool overflow")?;
        let dsa_index_bytes = dsa_layers
            .checked_mul(pages)
            .and_then(|v| v.checked_mul(layout.dsa_index_page_bytes))
            .context("Hy4-flash DSA index pool overflow")?;
        let pending_bytes = sequences
            .checked_mul(dsa_layers)
            .and_then(|v| v.checked_mul(PENDING_KEY_SLOT_BYTES))
            .context("Hy4-flash DSA pending keys overflow")?;
        // Exact prefix mark retains only the visible `sliding_window` (512) SWA rows per SWA layer
        // plus `3` pending indexer token keys per DSA layer (~9.63 MiB FP8), matching `CacheLayout`.
        let mark_bytes_per_slot = layout.mark_bytes_per_slot;

        let dsa_kv = Allocation::new(device, dsa_kv_bytes, Tag::kv("hy4_flash/dsa-kv"))?;
        let dsa_index = Allocation::new(device, dsa_index_bytes, Tag::kv("hy4_flash/dsa-index"))?;
        let swa_rings = Allocation::new(device, swa_rings_bytes, Tag::kv("hy4_flash/swa-rings"))?;
        let pending_keys = Allocation::new(device, pending_bytes, Tag::kv("hy4_flash/dsa-pending-keys"))?;
        let marks = if mark_slots > 0 {
            let bytes = mark_slots
                .checked_mul(mark_bytes_per_slot)
                .context("Hy4-flash prefix marks overflow")?;
            Some(Allocation::new(device, bytes, Tag::prefix("hy4_flash/prefix-marks"))?)
        } else {
            None
        };

        let copies = PrefixCopies::new(
            device.library,
            vec![readers
                .iter()
                .map(|s| Reader { device: s.device.id, stream: s.raw })
                .collect()],
        )?;

        let owner = Self {
            copies,
            readers,
            device,
            kv_format,
            layout,
            dsa_layers,
            swa_layers,
            pages,
            sequences,
            swa_ring_pages,
            swa_ring_rows,
            swa_ring,
            mark_slots,
            mark_bytes_per_slot,
            dsa_kv,
            dsa_index,
            swa_rings,
            pending_keys,
            marks,
            pending_counts: RefCell::new(vec![vec![0; dsa_layers]; sequences]),
        };
        owner.zero_all()?;
        Ok(owner)
    }

    pub(crate) fn zero_all(&self) -> Result<()> {
        self.copies.ordered(0, |stream| {
            // SAFETY: all allocations live on `self.device` and readers are fenced on both sides.
            unsafe {
                self.device.library.cuda_zero_bytes_async(self.dsa_kv.buffer, self.dsa_kv.buffer.bytes, stream)?;
                self.device.library.cuda_zero_bytes_async(self.dsa_index.buffer, self.dsa_index.buffer.bytes, stream)?;
                self.device.library.cuda_zero_bytes_async(self.swa_rings.buffer, self.swa_rings.buffer.bytes, stream)?;
                self.device.library.cuda_zero_bytes_async(self.pending_keys.buffer, self.pending_keys.buffer.bytes, stream)?;
            }
            Ok(())
        })?;
        for seq in self.pending_counts.borrow_mut().iter_mut() {
            seq.fill(0);
        }
        self.copies.drain()
    }

    pub(crate) fn reset_sequence(&self, seq: usize) -> Result<()> {
        ensure!(seq < self.sequences, "Hy4-flash sequence {seq} out of range");
        let mut ranges = Vec::with_capacity(self.swa_layers + 1);
        for swa_idx in 0..self.swa_layers {
            ranges.push(self.swa_ring(seq, swa_idx)?);
        }
        let pending_seq_bytes = self.dsa_layers * PENDING_KEY_SLOT_BYTES;
        ranges.push(slice_buf(self.pending_keys.buffer, seq * pending_seq_bytes, pending_seq_bytes)?);
        self.copies.ordered(0, |stream| {
            // SAFETY: fenced copy stream orders all arithmetic readers before zeroing this sequence's state.
            unsafe {
                for buf in &ranges {
                    self.device.library.cuda_zero_bytes_async(*buf, buf.bytes, stream)?;
                }
            }
            Ok(())
        })?;
        self.pending_counts.borrow_mut()[seq].fill(0);
        Ok(())
    }

    pub(crate) fn dsa_kv_layer(&self, dsa_idx: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(dsa_idx < self.dsa_layers, "DSA layer index {dsa_idx} out of range");
        let layer_bytes = self.pages * self.layout.dsa_kv_page_bytes;
        slice_buf(self.dsa_kv.buffer, dsa_idx * layer_bytes, layer_bytes)
    }

    pub(crate) fn dsa_kv_page(&self, dsa_idx: usize, page: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(page < self.pages, "DSA KV page {page} outside pool {}", self.pages);
        let layer = self.dsa_kv_layer(dsa_idx)?;
        slice_buf(layer, page * self.layout.dsa_kv_page_bytes, self.layout.dsa_kv_page_bytes)
    }

    pub(crate) fn dsa_index_layer(&self, dsa_idx: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(dsa_idx < self.dsa_layers, "DSA layer index {dsa_idx} out of range");
        let layer_bytes = self.pages * self.layout.dsa_index_page_bytes;
        slice_buf(self.dsa_index.buffer, dsa_idx * layer_bytes, layer_bytes)
    }

    pub(crate) fn dsa_index_page(&self, dsa_idx: usize, page: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(page < self.pages, "DSA index page {page} outside pool {}", self.pages);
        let layer = self.dsa_index_layer(dsa_idx)?;
        slice_buf(layer, page * self.layout.dsa_index_page_bytes, self.layout.dsa_index_page_bytes)
    }

    /// Contiguous `[sequences * swa_ring_rows, record_bytes]` bank for one SWA layer (batched attention).
    pub(crate) fn swa_layer_bank(&self, swa_idx: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(swa_idx < self.swa_layers, "SWA layer index {swa_idx} out of range");
        let layer_bytes = self.sequences * self.swa_ring_rows * self.kv_format.record_bytes();
        slice_buf(self.swa_rings.buffer, swa_idx * layer_bytes, layer_bytes)
    }

    /// One sequence's `[swa_ring_rows, record_bytes]` ring slice inside `swa_layer_bank(swa_idx)`.
    pub(crate) fn swa_ring(&self, seq: usize, swa_idx: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(seq < self.sequences && swa_idx < self.swa_layers, "SWA ring ({seq}, {swa_idx}) out of range");
        let ring_bytes = self.swa_ring_rows * self.kv_format.record_bytes();
        let bank = self.swa_layer_bank(swa_idx)?;
        slice_buf(bank, seq * ring_bytes, ring_bytes)
    }

    pub(crate) fn pending_key_slot(&self, seq: usize, dsa_idx: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(seq < self.sequences && dsa_idx < self.dsa_layers, "pending key ({seq}, {dsa_idx}) out of range");
        let seq_stride = self.dsa_layers * PENDING_KEY_SLOT_BYTES;
        slice_buf(self.pending_keys.buffer, seq * seq_stride + dsa_idx * PENDING_KEY_SLOT_BYTES, PENDING_KEY_SLOT_BYTES)
    }

    pub(crate) fn mark(&self, slot: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(slot < self.mark_slots, "Hy4-flash mark slot {slot} outside arena {}", self.mark_slots);
        let arena = self.marks.as_ref().context("Hy4-flash prefix mark arena absent")?;
        slice_buf(arena.buffer, slot * self.mark_bytes_per_slot, self.mark_bytes_per_slot)
    }

    fn state_copy(&self, seq: usize, mark_slot: usize, len: usize, capture: bool) -> Result<()> {
        let mark = self.mark(mark_slot)?;
        let rec = self.kv_format.record_bytes();
        let swa_mark_layer_bytes = (self.swa_ring.window as usize) * rec;
        let runs = self.swa_ring.visible(len as u64);
        let mut copies = Vec::with_capacity(self.swa_layers * runs.len() + 1);
        for swa_idx in 0..self.swa_layers {
            let live_ring = self.swa_ring(seq, swa_idx)?;
            let mark_layer_base = swa_idx * swa_mark_layer_bytes;
            for run in &runs {
                let bytes = (run.rows as usize) * rec;
                let live = slice_buf(live_ring, (run.slot as usize) * rec, bytes)?;
                let saved = slice_buf(mark, mark_layer_base + (run.offset as usize) * rec, bytes)?;
                copies.push(if capture { (0, saved, live) } else { (0, live, saved) });
            }
        }
        let pending_offset = self.swa_layers * swa_mark_layer_bytes;
        let pending_bytes = self.dsa_layers * PENDING_KEY_SLOT_BYTES;
        let live_pending = slice_buf(self.pending_keys.buffer, seq * pending_bytes, pending_bytes)?;
        let saved_pending = slice_buf(mark, pending_offset, pending_bytes)?;
        copies.push(if capture { (0, saved_pending, live_pending) } else { (0, live_pending, saved_pending) });
        // SAFETY: disjoint active/mark regions fenced across all registered readers.
        unsafe { self.copies.copy_d2d(&copies) }
    }

    pub(crate) fn capture(&self, seq: usize, mark_slot: usize, len: usize) -> Result<()> {
        self.state_copy(seq, mark_slot, len, true)
    }

    pub(crate) fn restore(&self, mark_slot: usize, seq: usize, len: usize) -> Result<()> {
        self.state_copy(seq, mark_slot, len, false)?;
        self.pending_counts.borrow_mut()[seq].fill(len % 4);
        Ok(())
    }

    /// Exact COW partial-page copy: copies `rows` KV rows and `rows / 4` complete indexer groups across all 12 DSA layers.
    pub(crate) fn copy_page_rows(&self, from: usize, to: usize, rows: usize) -> Result<()> {
        ensure!(from != to && (1..=PAGE_ROWS).contains(&rows), "Hy4-flash page tail copy aliases or exceeds page");
        let kv_bytes = rows * self.kv_format.record_bytes();
        let idx_groups = rows / 4;
        let idx_bytes = idx_groups * (self.layout.dsa_index_page_bytes / (PAGE_ROWS / 4));
        let mut copies = Vec::with_capacity(self.dsa_layers * 2);
        for dsa_idx in 0..self.dsa_layers {
            let src_kv = slice_buf(self.dsa_kv_page(dsa_idx, from)?, 0, kv_bytes)?;
            let dst_kv = slice_buf(self.dsa_kv_page(dsa_idx, to)?, 0, kv_bytes)?;
            copies.push((0, dst_kv, src_kv));
            if idx_bytes > 0 {
                let src_idx = slice_buf(self.dsa_index_page(dsa_idx, from)?, 0, idx_bytes)?;
                let dst_idx = slice_buf(self.dsa_index_page(dsa_idx, to)?, 0, idx_bytes)?;
                copies.push((0, dst_idx, src_idx));
            }
        }
        // SAFETY: `from != to` guarantees disjoint pages; `PrefixCopies` fences all readers.
        unsafe { self.copies.copy_d2d(&copies) }
    }

    pub(crate) fn drain(&self) -> Result<()> {
        self.copies.drain()
    }

    pub(crate) fn page_bytes_total(&self) -> usize {
        self.dsa_layers * (self.layout.dsa_kv_page_bytes + self.layout.dsa_index_page_bytes)
    }

    pub(crate) fn inventory(&self, inventory: &mut Inventory<'_>) -> Result<()> {
        let location = inventory.location(self.device.id)?;
        let cuteafd_loader::placement::planner::Device::Coordinator(gpu) = location.device else {
            anyhow::bail!("Hy4-flash cache is not coordinator-owned");
        };
        self.dsa_kv.inventory(inventory, &format!("hy4_flash.cache.{gpu}.dsa_kv"))?;
        self.dsa_index.inventory(inventory, &format!("hy4_flash.cache.{gpu}.dsa_index"))?;
        self.swa_rings.inventory(inventory, &format!("hy4_flash.cache.{gpu}.swa_rings"))?;
        self.pending_keys.inventory(inventory, &format!("hy4_flash.cache.{gpu}.pending_keys"))?;
        if let Some(marks) = &self.marks {
            marks.inventory(inventory, &format!("hy4_flash.cache.{gpu}.marks"))?;
        }
        self.copies.allocation_receipts(inventory, &format!("hy4_flash.cache.{gpu}.copies"))
    }

    pub(crate) const fn page_rows() -> usize {
        PAGE_ROWS
    }

    pub(crate) const fn swa_ring_pages() -> usize {
        SWA_RING_PAGES
    }

    /// Read back every committed causal state byte for `placement` on this rank for exact-restore verification.
    /// Under ContextSplit (`parity_rank = Some(rank)`), reads back only logical pages owned by `rank`.
    pub(crate) fn dump_state(&self, placement: &Hy4Placement, parity_rank: Option<usize>) -> Result<Hy4RankStateSnapshot> {
        self.drain()?;
        for reader in &self.readers {
            reader.drain()?;
        }
        let rec = self.kv_format.record_bytes();
        let len = placement.len;
        let idx_group_bytes = self.layout.dsa_index_page_bytes / (PAGE_ROWS / 4);

        let owned_rows = match parity_rank {
            Some(rank) => crate::shared::peer_split::context::shard_rows(len, PAGE_ROWS, rank),
            None => len,
        };
        let owned_groups = match parity_rank {
            Some(rank) => crate::shared::peer_split::context::shard_rows(len / 4, PAGE_ROWS / 4, rank),
            None => len / 4,
        };
        let mut dsa_kv = vec![0u8; self.dsa_layers * owned_rows * rec];
        let mut dsa_index = vec![0u8; self.dsa_layers * owned_groups * idx_group_bytes];
        let pending_n = len % 4;
        let pending_tok_bytes = pending_n * 128 * 2;
        let mut pending_keys = vec![0u8; self.dsa_layers * pending_tok_bytes];
        let vis_start = len.saturating_sub(self.swa_ring.window as usize);
        let vis_rows = len - vis_start;
        let mut swa_visible = vec![0u8; self.swa_layers * vis_rows * rec];

        self.device.run(|| {
            for d in 0..self.dsa_layers {
                let mut row_cursor = 0usize;
                let mut out_row_cursor = 0usize;
                let mut out_group_cursor = 0usize;
                let mut page_idx = 0usize;
                while row_cursor < len {
                    let take = (len - row_cursor).min(PAGE_ROWS);
                    let take_groups = take / 4;
                    let include = parity_rank.is_none_or(|rank| crate::shared::peer_split::context::page_owner(page_idx) == rank);
                    if include {
                        let raw_id = placement.units[page_idx] as usize;
                        let phys = if let Some(rank) = parity_rank {
                            ensure!(raw_id & 1 == rank, "parity dump_state page {page_idx} id {raw_id} != rank {rank}");
                            raw_id >> 1
                        } else {
                            raw_id
                        };
                        let kv_buf = slice_buf(self.dsa_kv_page(d, phys)?, 0, take * rec)?;
                        let dst_off = (d * owned_rows + out_row_cursor) * rec;
                        self.device.library.copy_d2h(&mut dsa_kv[dst_off..dst_off + take * rec], kv_buf)?;
                        if take_groups > 0 {
                            let idx_buf = slice_buf(self.dsa_index_page(d, phys)?, 0, take_groups * idx_group_bytes)?;
                            let g_off = (d * owned_groups + out_group_cursor) * idx_group_bytes;
                            self.device.library.copy_d2h(
                                &mut dsa_index[g_off..g_off + take_groups * idx_group_bytes],
                                idx_buf,
                            )?;
                        }
                        out_row_cursor += take;
                        out_group_cursor += take_groups;
                    }
                    row_cursor += take;
                    page_idx += 1;
                }
                if pending_tok_bytes > 0 {
                    let p_buf = slice_buf(self.pending_key_slot(placement.slot, d)?, 0, pending_tok_bytes)?;
                    let p_off = d * pending_tok_bytes;
                    self.device.library.copy_d2h(&mut pending_keys[p_off..p_off + pending_tok_bytes], p_buf)?;
                }
            }
            let runs = self.swa_ring.visible(len as u64);
            for s in 0..self.swa_layers {
                let ring = self.swa_ring(placement.slot, s)?;
                for run in &runs {
                    let bytes = (run.rows as usize) * rec;
                    let src = slice_buf(ring, (run.slot as usize) * rec, bytes)?;
                    let dst_off = (s * vis_rows + run.offset as usize) * rec;
                    self.device.library.copy_d2h(&mut swa_visible[dst_off..dst_off + bytes], src)?;
                }
            }
            Ok(())
        })?;

        Ok(Hy4RankStateSnapshot {
            len,
            dsa_kv,
            dsa_index,
            pending_keys,
            pending_counts: self.pending_counts.borrow()[placement.slot].clone(),
            swa_visible,
            engram_tail: placement.engram_tail,
        })
    }
}

/// Complete committed causal state read back from one coordinator GPU rank for byte-exact restore checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Hy4RankStateSnapshot {
    pub(crate) len: usize,
    pub(crate) dsa_kv: Vec<u8>,
    pub(crate) dsa_index: Vec<u8>,
    pub(crate) pending_keys: Vec<u8>,
    pub(crate) pending_counts: Vec<usize>,
    pub(crate) swa_visible: Vec<u8>,
    pub(crate) engram_tail: [u32; 2],
}

impl Hy4RankStateSnapshot {
    pub(crate) fn total_bytes(&self) -> usize {
        self.dsa_kv.len() + self.dsa_index.len() + self.pending_keys.len() + self.swa_visible.len()
    }

    pub(crate) fn diff(&self, other: &Self) -> Vec<String> {
        let mut diffs = Vec::new();
        if self.len != other.len {
            diffs.push(format!("len: {} != {}", self.len, other.len));
        }
        if self.engram_tail != other.engram_tail {
            diffs.push(format!("engram_tail: {:?} != {:?}", self.engram_tail, other.engram_tail));
        }
        if self.pending_counts != other.pending_counts {
            diffs.push(format!("pending_counts: {:?} != {:?}", self.pending_counts, other.pending_counts));
        }
        for (name, a, b) in [
            ("dsa_kv", &self.dsa_kv, &other.dsa_kv),
            ("dsa_index", &self.dsa_index, &other.dsa_index),
            ("pending_keys", &self.pending_keys, &other.pending_keys),
            ("swa_visible", &self.swa_visible, &other.swa_visible),
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
        let mut out = vec![self.dsa_kv.buffer, self.dsa_index.buffer];
        if let Some(marks) = &self.marks {
            out.push(marks.buffer);
        }
        Ok(out)
    }
}

/// Logical COW pages, active slot, causal frontier and Engram rolling tail for one Hy4-flash sequence.
#[derive(Clone, Debug)]
pub(crate) struct Hy4Placement {
    pub units: Vec<u32>,
    pub slot: usize,
    pub len: usize,
    pub state_len: usize,
    /// Last 2 committed token IDs (`[t_{len-2}, t_{len-1}]`, left-padded with `engram_pad_id`)
    /// so Engram 3-gram hashing resumes byte-identically from any prefix mark or decode step.
    pub engram_tail: [u32; 2],
    /// Interleaved M-RoPE coordinates; text-only requests keep equal axes.
    pub rope: std::sync::Arc<cuteafd_loader::families::hy4_flash::vision::RopePositions>,
    /// Image spans and their cached features, injected into prompt rows.
    pub media: Option<std::sync::Arc<cuteafd_engine::media::RequestMedia>>,
}

impl Hy4Placement {
    pub(crate) fn new(units: Vec<u32>, slot: usize, pad_id: u32) -> Self {
        Self {
            units,
            slot,
            len: 0,
            state_len: 0,
            engram_tail: [pad_id, pad_id],
            rope: Default::default(),
            media: None,
        }
    }

    /// Rope coordinates of a sequence row (prompt, decode, verify or draft).
    pub(crate) fn rope_at(&self, row: usize) -> anyhow::Result<[i64; 3]> {
        self.rope.at(row)
    }

    pub(crate) fn set_engram_tail_from_prefix(&mut self, prefix: &[u32], pad_id: u32) {
        self.engram_tail = match prefix.len() {
            0 => [pad_id, pad_id],
            1 => [pad_id, prefix[0]],
            n => [prefix[n - 2], prefix[n - 1]],
        };
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
        self.media.as_ref().map_or(&[], |media| media.spans())
    }
}

/// Exact prefix-cache adapter across 1 or 2 Hy4-flash coordinator ranks.
/// When `parity` is true (2-GPU ContextSplit), logical page `j` lives on GPU `j & 1`
/// at local physical page `id >> 1` using `RefPagePool::parity(2 * local_pages, 64, 0)`.
#[derive(Clone)]
pub(crate) struct Hy4Prefix<'a> {
    ranks: Vec<Rc<Hy4CacheRank<'a>>>,
    pad_id: u32,
    parity: bool,
}

impl<'a> Hy4Prefix<'a> {
    pub(crate) fn new(ranks: Vec<Rc<Hy4CacheRank<'a>>>, pad_id: u32, parity: bool) -> Result<Self> {
        ensure!(!ranks.is_empty(), "Hy4Prefix requires at least one rank");
        ensure!(!parity || ranks.len() == 2, "Hy4Prefix parity pool requires two coordinator ranks");
        if parity {
            ensure!(
                ranks[0].pages == ranks[1].pages,
                "Hy4Prefix parity ranks must have equal local page capacities ({} != {})",
                ranks[0].pages,
                ranks[1].pages
            );
        }
        Ok(Self { ranks, pad_id, parity })
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
            "Hy4-flash prefix placement outside admitted state/pages"
        );
        if self.parity {
            for (j, &id) in placement.units.iter().enumerate() {
                let owner = crate::shared::peer_split::context::page_owner(j);
                let local = (id as usize) >> 1;
                ensure!(
                    (id as usize) & 1 == owner && local < self.ranks[owner].pages,
                    "Hy4-flash parity page {j} id {id} outside owner {owner} pool {}",
                    self.ranks[owner].pages
                );
            }
        } else {
            ensure!(
                placement.units.iter().all(|&p| (p as usize) < r0.pages),
                "Hy4-flash prefix placement outside admitted pages"
            );
        }
        ensure!(
            placement.units.iter().copied().collect::<BTreeSet<_>>().len() == placement.units.len(),
            "Hy4-flash prefix placement aliases physical pages"
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
            pages: if self.parity { self.ranks[0].pages * 2 } else { self.ranks[0].pages },
            page_bytes: if self.parity {
                self.ranks[0].page_bytes_total()
            } else {
                self.ranks.iter().map(|r| r.page_bytes_total()).sum()
            },
            mark_bytes: self.ranks.iter().map(|r| r.mark_bytes_per_slot).sum(),
            draft_bytes: 0,
            rule: ReuseRule::EXACT,
            mark_store: MarkStore::Arena,
            page_owners: if self.parity { PageOwners::Parity } else { PageOwners::Uniform },
        }
    }

    fn pages<'p>(&self, placement: &'p Hy4Placement) -> &'p [u32] {
        &placement.units
    }

    fn bind(&self, placement: &mut Hy4Placement) -> Result<(), cuteafd_engine::prefix::BoxError> {
        self.validate(placement, placement.len)?;
        for rank in &self.ranks {
            rank.reset_sequence(placement.slot)?;
        }
        placement.state_len = 0;
        placement.engram_tail = [self.pad_id, self.pad_id];
        Ok(())
    }

    fn discard(&self, placement: &mut Hy4Placement) -> Result<(), cuteafd_engine::prefix::BoxError> {
        for rank in &self.ranks {
            rank.reset_sequence(placement.slot)?;
        }
        placement.state_len = 0;
        placement.engram_tail = [self.pad_id, self.pad_id];
        Ok(())
    }

    fn extend_pages(&self, placement: &mut Hy4Placement, pages: &[u32]) -> Result<(), cuteafd_engine::prefix::BoxError> {
        let mut extended = placement.clone();
        extended.units.extend_from_slice(pages);
        self.validate(&extended, extended.len)?;
        placement.units = extended.units;
        Ok(())
    }

    fn commit_point(&self, placement: &Hy4Placement) -> usize {
        placement.len.min(placement.state_len)
    }

    fn capture(
        &self,
        slot: cuteafd_engine::prefix::MarkSlot,
        placement: &Hy4Placement,
        len: usize,
    ) -> Result<(), cuteafd_engine::prefix::BoxError> {
        self.validate(placement, len)?;
        if len != placement.len || len != placement.state_len {
            return Err("Hy4-flash snapshot frontier does not match committed causal state".into());
        }
        for rank in &self.ranks {
            rank.capture(placement.slot, slot.0 as usize, len)?;
        }
        Ok(())
    }

    fn restore(
        &self,
        mark: Option<cuteafd_engine::prefix::MarkSlot>,
        placement: &mut Hy4Placement,
        len: usize,
    ) -> Result<(), cuteafd_engine::prefix::BoxError> {
        self.validate(placement, len)?;
        let slot = mark.context("Hy4-flash exact restore requires its SWA/pending-key mark")?;
        for rank in &self.ranks {
            rank.restore(slot.0 as usize, placement.slot, len)?;
        }
        placement.len = len;
        placement.state_len = len;
        Ok(())
    }

    fn copy_rows(&self, copy: cuteafd_engine::prefix::TailCopy) -> Result<(), cuteafd_engine::prefix::BoxError> {
        if self.parity {
            let Some(owner) = copy.owner else {
                return Err("Hy4-flash parity tail copy missing owner rank".into());
            };
            let rank = usize::from(owner);
            if rank >= self.ranks.len() || (copy.from as usize & 1) != rank || (copy.to as usize & 1) != rank {
                return Err(format!("Hy4-flash parity tail copy {copy:?} crosses GPUs").into());
            }
            self.ranks[rank].copy_page_rows((copy.from >> 1) as usize, (copy.to >> 1) as usize, copy.rows)?;
            return Ok(());
        }
        if copy.owner.is_some() {
            return Err("Hy4-flash whole-rank pages have no parity owner".into());
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
            for d in 0..rank.dsa_layers {
                if let Ok(b) = rank.dsa_kv_page(d, local) {
                    out.push(crate::shared::prefix::cuda_copy::range(b));
                }
                if let Ok(b) = rank.dsa_index_page(d, local) {
                    out.push(crate::shared::prefix::cuda_copy::range(b));
                }
            }
            return out;
        }
        for rank in &self.ranks {
            for d in 0..rank.dsa_layers {
                if let Ok(b) = rank.dsa_kv_page(d, page as usize) {
                    out.push(crate::shared::prefix::cuda_copy::range(b));
                }
                if let Ok(b) = rank.dsa_index_page(d, page as usize) {
                    out.push(crate::shared::prefix::cuda_copy::range(b));
                }
            }
        }
        out
    }

    fn mark_segments(&self, slot: cuteafd_engine::prefix::MarkSlot) -> Vec<cuteafd_hostcache::copy::DeviceRange> {
        self.ranks
            .iter()
            .filter_map(|rank| rank.mark(slot.0 as usize).ok())
            .map(crate::shared::prefix::cuda_copy::range)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::prefix::CudaCopyEngine;
    use cuteafd_engine::prefix::{After, PrefixCache, PrefixConfig, PrefixFamily, SnapshotKind};
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;

    #[test]
    fn hy4_exact_prefix_cache_restores_swa_pending_and_dsa_pages_byte_identically() -> Result<()> {
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        let device = Device { library: &library, id: 0 };
        let reader = Rc::new(Stream::new(device)?);
        let cfg = Hy4FlashConfig {
            vocab_size: 120_832,
            hidden: 3072,
            layers: 2,
            intermediate: 13_824,
            max_position: 262_144,
            rms_norm_eps: 1e-5,
            hc_mult: 4,
            hc_eps: 1e-6,
            hc_magnitude: 2.0,
            q_lora_rank: 1024,
            kv_lora_rank: 512,
            qk_nope_dim: 192,
            qk_rope_dim: 64,
            v_head_dim: 256,
            sparse_heads: 32,
            swa_heads: 64,
            swa_qk_nope_dim: 192,
            swa_qk_rope_dim: 64,
            swa_v_head_dim: 256,
            sliding_window: 512,
            sparse_rope_theta: 10_000.0,
            swa_rope_theta: 10_000.0,
            sparse_layer: vec![true, false],
            indexer_heads: 16,
            indexer_head_dim: 128,
            indexer_compress_ratio: 4,
            indexer_group_topk: 512,
            indexer_candidate_topk: 2048,
            dense_layer: vec![true, false],
            experts: 512,
            topk: 16,
            moe_intermediate: 1536,
            routed_expert_hidden: 1536,
            shared_experts: 1,
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
            swiglu_limit: 10.0,
            engram_layers: Vec::new(),
            engram_vocab_sizes: std::collections::BTreeMap::new(),
            engram_max_ngram_size: 3,
            engram_n_embed_per_ngram: 768,
            engram_n_head_per_ngram: 8,
            engram_pad_id: 120_002,
            engram_seed: 0,
            engram_fp8_embedding: true,
            engram_lookup_table_size: 120_818,
            image_token_id: 120_120,
            video_token_id: 120_124,
            eos: vec![120_025],
        };
        let rank = Rc::new(Hy4CacheRank::new(vec![reader], &cfg, Hy4KvCache::Fp8, 32, 2, 64, 3)?);
        assert_eq!(rank.mark_bytes_per_slot, rank.layout.mark_bytes_per_slot);
        let family = Hy4Prefix::new(vec![rank.clone()], cfg.engram_pad_id, false)?;
        let mut cache: PrefixCache<CudaCopyEngine<'_>> = PrefixCache::new(
            family.layout(),
            PrefixConfig { entries: 3, mark_slots: 3, keep_logits: true, min_tokens: 1 },
            None,
        )?;
        let rec = Hy4KvCache::Fp8.record_bytes();

        // Case 1: non-wrapped frontier (67 tokens -> 2 pages, partial tail = 3 rows, pending = 3).
        let tokens: Vec<u32> = (10..77).collect();
        let mut src = cache
            .admit(&family, &tokens, 128, false, |units| Hy4Placement::new(units, 0, cfg.engram_pad_id))?
            .placement;
        let swa0 = rank.swa_ring(0, 0)?;
        let swa_bytes: Vec<u8> = (0..swa0.bytes).map(|i| ((i / rec) as u8).wrapping_mul(13).wrapping_add(7)).collect();
        library.copy_h2d(swa0, &swa_bytes)?;
        let pend0 = rank.pending_key_slot(0, 0)?;
        let pend_bytes = vec![0xA5u8; pend0.bytes];
        library.copy_h2d(pend0, &pend_bytes)?;
        for &pg in &src.units {
            let kv = rank.dsa_kv_page(0, pg as usize)?;
            library.copy_h2d(kv, &vec![(pg as u8).wrapping_add(0x21); kv.bytes])?;
            let idx = rank.dsa_index_page(0, pg as usize)?;
            library.copy_h2d(idx, &vec![(pg as u8).wrapping_add(0x41); idx.bytes])?;
        }
        src.len = tokens.len();
        src.state_len = tokens.len();
        assert!(cache.capture(&family, SnapshotKind::Prompt, &tokens, &src, After::from_logits(&[1.0, 0.0], true))?);
        cache.release(&family, &src.units)?;

        let mut ext_tokens = tokens.clone();
        ext_tokens.push(999);
        let restored = cache.admit(&family, &ext_tokens, 128, false, |units| {
            Hy4Placement::new(units, 1, cfg.engram_pad_id)
        })?;
        assert_eq!(restored.resume, tokens.len());
        assert_eq!(restored.placement.len, tokens.len());
        assert_eq!(rank.pending_counts.borrow()[1][0], tokens.len() % 4);
        let mut got_swa = vec![0u8; swa0.bytes];
        library.copy_d2h(&mut got_swa, rank.swa_ring(1, 0)?)?;
        assert_eq!(&got_swa[..tokens.len() * rec], &swa_bytes[..tokens.len() * rec]);
        assert!(got_swa[tokens.len() * rec..].iter().all(|&b| b == 0));
        let mut got_pend = vec![0u8; pend0.bytes];
        library.copy_d2h(&mut got_pend, rank.pending_key_slot(1, 0)?)?;
        assert_eq!(got_pend, pend_bytes);
        cache.release(&family, &restored.placement.units)?;

        // Case 2: wrapped frontier (> swa_ring_rows) restores all 512 visible rows across the wrap boundary.
        let wrap_len = rank.swa_ring_rows + 137;
        let wrap_tokens: Vec<u32> = (0..wrap_len as u32).map(|i| 1000 + i).collect();
        let mut wrap_src = cache
            .admit(&family, &wrap_tokens, wrap_len + 64, false, |units| Hy4Placement::new(units, 0, cfg.engram_pad_id))?
            .placement;
        library.copy_h2d(rank.swa_ring(0, 0)?, &swa_bytes)?;
        library.copy_h2d(rank.pending_key_slot(0, 0)?, &pend_bytes)?;
        rank.pending_counts.borrow_mut()[0].fill(wrap_len % 4);
        wrap_src.len = wrap_len;
        wrap_src.state_len = wrap_len;
        assert!(cache.capture(&family, SnapshotKind::Prompt, &wrap_tokens, &wrap_src, After::from_logits(&[0.0, 1.0], true))?);
        cache.release(&family, &wrap_src.units)?;

        let wrap_restored = cache.admit(&family, &wrap_tokens, wrap_len + 64, false, |units| {
            Hy4Placement::new(units, 1, cfg.engram_pad_id)
        })?;
        assert_eq!(wrap_restored.resume, wrap_len);
        assert_eq!(rank.pending_counts.borrow()[1][0], wrap_len % 4);
        library.copy_d2h(&mut got_swa, rank.swa_ring(1, 0)?)?;
        for pos in (wrap_len - cfg.sliding_window)..wrap_len {
            let slot = pos % rank.swa_ring_rows;
            let lo = slot * rec;
            assert_eq!(&got_swa[lo..lo + rec], &swa_bytes[lo..lo + rec], "wrapped visible SWA row {pos} (slot {slot})");
        }
        // State snapshot oracle: restored state matches captured state with 0 differing bytes,
        // and any 1-byte mutation in any component is detected.
        let mut wrap_restored_p = wrap_restored.placement.clone();
        wrap_src.set_engram_tail_from_prefix(&wrap_tokens, cfg.engram_pad_id);
        wrap_restored_p.set_engram_tail_from_prefix(&wrap_tokens, cfg.engram_pad_id);
        let snap_src = rank.dump_state(&wrap_src, None)?;
        let snap_dst = rank.dump_state(&wrap_restored_p, None)?;
        assert!(snap_src.diff(&snap_dst).is_empty(), "diffs: {:?}", snap_src.diff(&snap_dst));
        let mut flipped = snap_dst.clone();
        flipped.swa_visible[0] ^= 1;
        assert!(snap_src.diff(&flipped).iter().any(|d| d.starts_with("swa_visible:")));
        let mut flipped_p = snap_dst.clone();
        flipped_p.pending_keys[0] ^= 1;
        assert!(snap_src.diff(&flipped_p).iter().any(|d| d.starts_with("pending_keys:")));
        let mut flipped_e = snap_dst.clone();
        flipped_e.engram_tail[0] ^= 1;
        assert!(snap_src.diff(&flipped_e).iter().any(|d| d.starts_with("engram_tail:")));
        cache.release(&family, &wrap_restored.placement.units)?;
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
        let cfg = Hy4FlashConfig {
            vocab_size: 120_832,
            hidden: 3072,
            layers: 2,
            intermediate: 13_824,
            max_position: 262_144,
            rms_norm_eps: 1e-5,
            hc_mult: 4,
            hc_eps: 1e-6,
            hc_magnitude: 2.0,
            q_lora_rank: 1024,
            kv_lora_rank: 512,
            qk_nope_dim: 192,
            qk_rope_dim: 64,
            v_head_dim: 256,
            sparse_heads: 32,
            swa_heads: 64,
            swa_qk_nope_dim: 192,
            swa_qk_rope_dim: 64,
            swa_v_head_dim: 256,
            sliding_window: 512,
            sparse_rope_theta: 10_000.0,
            swa_rope_theta: 10_000.0,
            sparse_layer: vec![true, false],
            indexer_heads: 16,
            indexer_head_dim: 128,
            indexer_compress_ratio: 4,
            indexer_group_topk: 512,
            indexer_candidate_topk: 2048,
            dense_layer: vec![true, false],
            experts: 512,
            topk: 16,
            moe_intermediate: 1536,
            routed_expert_hidden: 1536,
            shared_experts: 1,
            n_group: 1,
            topk_group: 1,
            norm_topk_prob: true,
            routed_scaling_factor: 1.0,
            swiglu_limit: 10.0,
            engram_layers: Vec::new(),
            engram_vocab_sizes: std::collections::BTreeMap::new(),
            engram_max_ngram_size: 3,
            engram_n_embed_per_ngram: 768,
            engram_n_head_per_ngram: 8,
            engram_pad_id: 120_002,
            engram_seed: 0,
            engram_fp8_embedding: true,
            engram_lookup_table_size: 120_818,
            image_token_id: 120_120,
            video_token_id: 120_124,
            eos: vec![120_025],
        };
        let rank0 = Rc::new(Hy4CacheRank::new(vec![r0_stream], &cfg, Hy4KvCache::Fp8, 16, 2, 64, 3)?);
        let rank1 = Rc::new(Hy4CacheRank::new(vec![r1_stream], &cfg, Hy4KvCache::Fp8, 16, 2, 64, 3)?);
        let family = Hy4Prefix::new(vec![rank0.clone(), rank1.clone()], cfg.engram_pad_id, true)?;
        let layout = family.layout();
        assert_eq!(layout.page_rows, PAGE_ROWS);
        assert_eq!(layout.pages, 32);
        assert_eq!(layout.page_owners, cuteafd_engine::prefix::PageOwners::Parity);
        assert_eq!(family.page_device(0), Some(0));
        assert_eq!(family.page_device(1), Some(1));
        // Parity page segments target only the single owning rank (2 segments per DSA layer: kv + index).
        assert_eq!(family.page_segments(0).len(), 2);
        assert_eq!(family.page_segments(1).len(), 2);

        let mut cache: PrefixCache<CudaCopyEngine<'_>> = PrefixCache::new(
            layout,
            PrefixConfig { entries: 3, mark_slots: 3, keep_logits: true, min_tokens: 1 },
            None,
        )?;

        // Exercise both odd-parity tail (67 tokens -> logical page 1 on rank 1) and even-parity tail
        // (131 tokens -> logical page 2 on rank 0) to verify COW TailCopy stays on the owning GPU.
        for (base, len) in [(100u32, 67usize), (500u32, 131usize)] {
            let tokens: Vec<u32> = (base..base + len as u32).collect();
            let mut src = cache
                .admit(&family, &tokens, len + 64, false, |units| {
                    Hy4Placement::new(units, 0, cfg.engram_pad_id)
                })?
                .placement;
            for (j, &id) in src.units.iter().enumerate() {
                assert_eq!((id as usize) & 1, j & 1, "logical page {j} must match parity owner");
                let owner = (id as usize) & 1;
                let local = (id as usize) >> 1;
                let rank = if owner == 0 { &rank0 } else { &rank1 };
                let kv = rank.dsa_kv_page(0, local)?;
                library.copy_h2d(kv, &vec![(id as u8).wrapping_add(0x31); kv.bytes])?;
                let idx = rank.dsa_index_page(0, local)?;
                library.copy_h2d(idx, &vec![(id as u8).wrapping_add(0x51); idx.bytes])?;
            }
            for (r_idx, rank) in [&rank0, &rank1].into_iter().enumerate() {
                let swa = rank.swa_ring(0, 0)?;
                library.copy_h2d(swa, &vec![0x11u8.wrapping_add(r_idx as u8); swa.bytes])?;
                let pend = rank.pending_key_slot(0, 0)?;
                library.copy_h2d(pend, &vec![0xA0u8.wrapping_add(r_idx as u8); pend.bytes])?;
                rank.pending_counts.borrow_mut()[0].fill(len % 4);
            }
            src.len = len;
            src.state_len = len;
            src.set_engram_tail_from_prefix(&tokens, cfg.engram_pad_id);
            assert!(cache.capture(&family, SnapshotKind::Prompt, &tokens, &src, After::from_logits(&[1.0, 0.0], true))?);

            // Restore with an extra token forcing a COW TailCopy on the partial tail page (owner = (len/64)&1).
            let mut ext = tokens.clone();
            ext.push(9999);
            let restored = cache.admit(&family, &ext, len + 64, false, |units| {
                Hy4Placement::new(units, 1, cfg.engram_pad_id)
            })?;
            assert_eq!(restored.resume, len);
            let mut restored_p = restored.placement.clone();
            restored_p.set_engram_tail_from_prefix(&tokens, cfg.engram_pad_id);
            assert_ne!(
                restored_p.units[len / PAGE_ROWS],
                src.units[len / PAGE_ROWS],
                "partial tail page must COW-copy while source stays pinned"
            );
            assert_eq!(
                (restored_p.units[len / PAGE_ROWS] as usize) & 1,
                (len / PAGE_ROWS) & 1,
                "COW tail page must keep parity owner"
            );
            for (r_idx, rank) in [&rank0, &rank1].into_iter().enumerate() {
                let snap_src = rank.dump_state(&src, Some(r_idx))?;
                let snap_dst = rank.dump_state(&restored_p, Some(r_idx))?;
                assert!(snap_src.diff(&snap_dst).is_empty(), "rank {r_idx} len {len} diffs: {:?}", snap_src.diff(&snap_dst));
            }
            cache.release(&family, &src.units)?;
            cache.release(&family, &restored.placement.units)?;
        }
        Ok(())
    }
}
