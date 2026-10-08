//! MiMo V2 (Flash, V2.6 Pro) as a prefix-cache family (`cuteafd_engine::prefix`).
//!
//! Paged state: the full-attention layers' BF16 records, 64 rows per page; one page index
//! spans every full layer (Flash 9 x 64 x 2,560 B = 1.47 MB, Pro 10 x 64 x 5,120 B = 3.28 MB).
//! Full pages are shared by reference; nobody writes them again (every sequence appends past
//! its own length), so only a partial tail page is copied.
//!
//! Mark: the sliding-window layers keep a 256-slot ring per sequence that the next step
//! overwrites, and their attention reads keys `p - window + 1 ..= p`, so a sequence continuing
//! at `P` needs ring rows `P - window + 1 .. P`; the mark keeps the last `window` rows of
//! every SWA layer in position order (Flash 39 x 128 x 5,120 B = 25.6 MB, Pro 60 layers =
//! 39.3 MB). With MTP (Flash) it also keeps the last `window + stages + 1` rows of the MTP
//! hidden ring: `mtp_reset` at the restored length makes every stage recompute its own ring
//! from them, exactly as after a full prefill (~1 MiB). Rows go back to the same `position %
//! 256` slots of the new sequence's ring, so the restored ring reads exactly as the captured
//! one. DFlash restores cold by default. `--mimo-prefix-draft` also retains its K/V rings
//! and a conservative valid-context floor, excluding rows speculative writes may have
//! overwritten. Drafts only steer speculation; target state restores exactly.
//!
//! Restores are exact: the deepest retained point whose tokens prefix the request wins
//! (`ReuseRule::EXACT`), and intermediate points (message boundaries, periodic chunk ends; see
//! `cuteafd_engine::prefix::plan_points`) keep one close. A point may lie up to
//! `capture_reach` rows behind the committed length: the rings still hold its window there
//! (256 slots, minus a verify step's 64 rows that may be written past the commit point, minus
//! the rows a mark keeps).
//!
//! Opt-in (`--prefix-partial on`, off by default): V4.1-style partial reuse. A partial match
//! shares the pages below `aligned common - window`, restarts the positional state there and
//! replays the window. The kernels have no per-sequence key floor, so "restarts" means the
//! ring rows before the restart point are zeroed (zero keys and values: an extra zero-logit
//! sink for the first replayed rows); approximate by design, not byte-exact.
//!
//! Every copy is enqueued on the engine stream, in order with the forward passes, and `drain`
//! synchronizes it. Under a head split each GPU keeps its own KV heads: its pages and ring rows
//! are copied on that GPU's stream into its own mark arena (the same slot on both GPUs).
use super::engine::{MimoEngine, MimoPlacement, DECODE_ROWS, PAGE_ROWS, RING_ROWS};
use super::mtp::HIDDEN_ROWS;
use crate::shared::memory::device::{Allocation, Device};
use anyhow::{ensure, Result};
use cuteafd_engine::prefix::{BoxError, FamilyLayout, MarkSlot, MarkStore, PrefixFamily, ReuseRule, TailCopy};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_hostcache::copy::DeviceRange;
use cuteafd_loader::families::mimo_v2::MimoAttention;
use std::rc::Rc;
use std::cell::RefCell;

/// One ring-structured state the mark keeps rows of: `rows` rows before the frontier, each
/// `row` bytes, ring `r` position `p` at `(r * ring_rows + p % ring_rows) * row`.
#[derive(Clone, Copy)]
struct RingState {
    /// The head-split rank whose GPU holds the ring (0 without a split).
    rank: usize,
    buffer: CuteafdDeviceBuffer,
    row: usize,
    ring_rows: usize,
    rows: usize,
    /// Byte offset of this state's rows inside a mark.
    offset: usize,
    draft: bool,
}

pub(crate) struct MimoPrefix<'e, 'a> {
    engine: &'e MimoEngine<'a>,
    /// Full-attention record pools: (rank, buffer, record bytes).
    full: Vec<(usize, CuteafdDeviceBuffer, usize)>,
    states: Vec<RingState>,
    /// Bytes of one mark on each rank's GPU, and their sum.
    rank_mark_bytes: Vec<usize>,
    mark_bytes: usize,
    /// Per rank: its marks' arena.
    arenas: Vec<Rc<Allocation<'a>>>,
    slots: usize,
    partial: bool,
    draft_slots: RefCell<Vec<Option<usize>>>,
    draft_floors: RefCell<Vec<usize>>,
    draft_metadata: Option<Rc<Allocation<'a>>>,
    draft_offset: usize,
}

/// A byte range inside `buffer` (bounds-checked; pointer arithmetic only).
fn view(buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    ensure!(offset.checked_add(bytes).is_some_and(|end| end <= buffer.bytes),
        "view {offset}+{bytes} past a {}-byte buffer", buffer.bytes);
    Ok(CuteafdDeviceBuffer { ptr: buffer.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..buffer })
}

impl<'e, 'a> MimoPrefix<'e, 'a> {
    /// The family over `engine`'s buffers with a device arena of `slots(mark_bytes)` marks;
    /// `partial` opts into V4.1-style partial reuse (approximate).
    pub fn new(engine: &'e MimoEngine<'a>, slots: impl FnOnce(usize) -> usize, partial: bool) -> Result<Self> {
        Self::new_with_draft(engine, slots, partial, false)
    }

    pub fn new_with_draft(engine: &'e MimoEngine<'a>, slots: impl FnOnce(usize) -> usize,
        partial: bool, warm_draft: bool) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("prefix");
        let (mut full, mut states) = (Vec::new(), Vec::new());
        let mut offsets = vec![0usize; engine.ranks()];
        let window = engine.cfg.window;
        ensure!(window > 0 && window <= RING_ROWS, "SWA window {window} does not fit the {RING_ROWS}-slot ring");
        for rank in 0..engine.ranks() {
            for layer in 0..engine.weights.layers.len() {
                let (attention, buffer, record) = engine.kv_layer_on(rank, layer);
                match attention {
                    MimoAttention::Full => full.push((rank, buffer, record)),
                    MimoAttention::Sliding => {
                        states.push(RingState { rank, buffer, row: record, ring_rows: RING_ROWS, rows: window,
                            offset: offsets[rank], draft: false });
                        offsets[rank] += window * record;
                    }
                }
            }
        }
        if let Some(mtp) = &engine.mtp {
            let rows = window + mtp.stages.len() + 1;
            ensure!(rows <= HIDDEN_ROWS, "MTP catch-up of {rows} rows exceeds the {HIDDEN_ROWS}-row hidden ring");
            let row = engine.cfg.hidden * 2;
            states.push(RingState { rank: 0, buffer: mtp.hidden.buffer, row, ring_rows: HIDDEN_ROWS, rows,
                offset: offsets[0], draft: false });
            offsets[0] += rows * row;
        }
        let draft_offset = offsets[0];
        let draft_metadata = if let Some(drafter) = engine.drafter.as_ref().filter(|_| warm_draft) {
            // Leave room for speculative writes and the target mark's capture reach.
            let rows = draft_mark_rows(window);
            let row = drafter.cfg.kv_width() * 2;
            for buffer in drafter.context_rings() {
                states.push(RingState { rank: 0, buffer, row, ring_rows: super::dflash::RING,
                    rows, offset: offsets[0], draft: true });
                offsets[0] += super::dflash::RING * row;
            }
            offsets[0] += 8;
            debug_assert_eq!((offsets[0] - draft_offset) as u64,
                cuteafd_loader::families::mimo_v2::draft_representation::mimo_draft_mark_bytes(
                    drafter.cfg.layers as u64, drafter.cfg.kv_width() as u64)?);
            Some(Rc::new(Allocation::new(Device { library: engine.library, id: engine.kv_layer_on(0, 0).1.device_id }, engine.rings * 8)?))
        } else { None };
        let mark_bytes: usize = offsets.iter().sum();
        let slots = slots(mark_bytes);
        let arenas = if slots > 0 && mark_bytes > 0 {
            offsets.iter().enumerate().map(|(rank, &bytes)| {
                let device = Device { library: engine.library, id: engine.kv_layer_on(rank, 0).1.device_id };
                Allocation::new(device, (slots * bytes).max(256)).map(Rc::new)
            }).collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        ensure!(!partial || window % PAGE_ROWS == 0, "partial reuse replays a whole number of pages");
        Ok(Self { engine, full, states, rank_mark_bytes: offsets, mark_bytes, arenas, slots, partial,
            draft_slots: RefCell::new(if draft_metadata.is_some() { vec![None; engine.rings] } else { Vec::new() }),
            draft_floors: RefCell::new(if draft_metadata.is_some() { vec![0; engine.rings] } else { Vec::new() }),
            draft_metadata, draft_offset })
    }

    pub fn bind_drafter(&self, ring: i32, slot: Option<usize>) {
        self.draft_slots.borrow_mut()[ring as usize] = slot;
        self.draft_floors.borrow_mut()[ring as usize] = 0;
    }

    pub fn draft_from(&self, ring: i32) -> usize {
        self.draft_floors.borrow()[ring as usize]
    }

    fn state_ring(&self, state: &RingState, ring: usize) -> Option<usize> {
        if state.draft { self.draft_slots.borrow()[ring] } else { Some(ring) }
    }

    fn draft_metadata_offset(&self) -> usize {
        self.states.iter().filter(|s| s.draft).map(|s| s.ring_rows * s.row).sum::<usize>() + self.draft_offset
    }

    pub fn mark_bytes(&self) -> usize {
        self.mark_bytes
    }

    pub fn page_bytes(&self) -> usize {
        self.full.iter().map(|(_, _, record)| PAGE_ROWS * record).sum()
    }

    pub fn slots(&self) -> usize {
        if self.arenas.is_empty() { 0 } else { self.slots }
    }

    /// Stable allocations the host snapshot tier may copy. Full pages and
    /// per-rank SWA/MTP mark arenas are registered on their actual devices;
    /// the live rings are captured into those arenas before host submission.
    pub fn host_owners(&self) -> Vec<Rc<Allocation<'a>>> {
        self.engine.full_kv_owners().into_iter().chain(self.arenas.iter().cloned())
            .chain(self.draft_metadata.iter().cloned()).collect()
    }

    /// Component-only positional storage; no attention/model state is claimed.
    #[cfg(test)]
    pub(super) fn terminal_fixture(engine: &'e MimoEngine<'a>) -> Result<Self> {
        let arenas = (0..engine.ranks()).map(|rank| engine.on(rank, || {
            let device = Device { library: engine.library, id: engine.library.cuda_get_device()? };
            Allocation::new(device, 4096).map(Rc::new)
        })).collect::<Result<Vec<_>>>()?;
        Ok(Self { engine, full: Vec::new(), states: Vec::new(),
            rank_mark_bytes: vec![4096; engine.ranks()], mark_bytes: 4096 * engine.ranks(),
            arenas, slots: 1, partial: false, draft_slots: RefCell::new(vec![None; engine.rings]),
            draft_floors: RefCell::new(vec![0; engine.rings]), draft_metadata: None, draft_offset: 0 })
    }

    /// Zero every state's rows before `len` (a partial restore's empty window).
    fn empty_window(&self, ring: usize, len: usize) -> Result<()> {
        for state in &self.states {
            let Some(ring) = self.state_ring(state, ring) else { continue };
            for (ring_row, _, rows) in runs(ring, state.ring_rows, state.rows, len) {
                let target = view(state.buffer, ring_row * state.row, rows * state.row)?;
                // SAFETY: the view lies inside a live allocation of the state's GPU; ordered on its stream.
                self.engine.on(state.rank, || unsafe {
                    self.engine.library.cuda_zero_bytes_async(target, target.bytes, self.engine.stream_of(state.rank))
                })?;
            }
        }
        Ok(())
    }

    /// A copy between two views on rank `rank`'s GPU.
    fn copy(&self, rank: usize, dst: CuteafdDeviceBuffer, src: CuteafdDeviceBuffer) -> Result<()> {
        debug_assert_eq!(dst.bytes, src.bytes);
        // SAFETY: both views lie inside live allocations of that GPU (checked by `view`); the copy
        // is ordered on its stream with every forward pass that reads or writes them.
        self.engine.on(rank, || unsafe {
            self.engine.library.copy_d2d_async(dst, src, src.bytes, self.engine.stream_of(rank))
        })
    }

    /// Copy every state's last rows before `len` between ring `ring` and mark `slot`.
    fn move_mark(&self, slot: MarkSlot, ring: usize, len: usize, capture: bool) -> Result<()> {
        ensure!(!self.arenas.is_empty(), "no mark arena");
        ensure!((slot.0 as usize) < self.slots, "mark slot {} of {}", slot.0, self.slots);
        for state in &self.states {
            let arena = self.arenas[state.rank].buffer;
            let base = slot.0 as usize * self.rank_mark_bytes[state.rank];
            let Some(state_ring) = self.state_ring(state, ring) else { continue };
            for (ring_row, mark_row, rows) in runs(state_ring, state.ring_rows, state.rows, len) {
                let ring_view = view(state.buffer, ring_row * state.row, rows * state.row)?;
                let mark_view = view(arena, base + state.offset + mark_row * state.row, rows * state.row)?;
                if capture {
                    self.copy(state.rank, mark_view, ring_view)?;
                } else {
                    self.copy(state.rank, ring_view, mark_view)?;
                }
            }
        }
        Ok(())
    }
}

fn draft_mark_rows(window: usize) -> usize {
    super::dflash::RING - DECODE_ROWS - RING_ROWS.saturating_sub(DECODE_ROWS + window)
}

/// The last `min(len, rows)` positions before `len` of ring `ring` as runs whose ring slots do
/// not wrap: (ring row, mark row, rows), mark rows in position order from the first kept one.
fn runs(ring: usize, ring_rows: usize, rows: usize, len: usize) -> Vec<(usize, usize, usize)> {
    let first = len - len.min(rows);
    let mut out = Vec::new();
    let mut position = first;
    while position < len {
        let at = position % ring_rows;
        let run = (len - position).min(ring_rows - at);
        out.push((ring * ring_rows + at, position - first, run));
        position += run;
    }
    out
}

impl PrefixFamily for MimoPrefix<'_, '_> {
    type Placement = MimoPlacement;

    fn layout(&self) -> FamilyLayout {
        FamilyLayout {
            page_rows: PAGE_ROWS,
            pages: self.engine.pages,
            page_bytes: self.page_bytes(),
            mark_bytes: self.mark_bytes,
            draft_bytes: 0,
            rule: if self.partial {
                ReuseRule { align: PAGE_ROWS, replay: Some(self.engine.cfg.window) }
            } else {
                ReuseRule::EXACT
            },
            mark_store: MarkStore::Arena,
        }
    }

    fn capture_reach(&self) -> usize {
        let kept = self.states.iter().filter(|s| !s.draft).map(|s| s.rows).max().unwrap_or(0);
        RING_ROWS.saturating_sub(DECODE_ROWS + kept)
    }

    fn pages<'p>(&self, placement: &'p MimoPlacement) -> &'p [u32] {
        &placement.pages
    }

    fn commit_point(&self, placement: &MimoPlacement) -> usize {
        placement.len
    }

    fn capture(&self, slot: MarkSlot, placement: &MimoPlacement, len: usize) -> Result<(), BoxError> {
        if len > placement.len || placement.len - len > self.capture_reach() {
            return Err(format!("capture at {len} is out of reach of the committed {} rows", placement.len).into());
        }
        self.move_mark(slot, placement.ring as usize, len, true)?;
        if let Some(metadata) = &self.draft_metadata {
            let ring = placement.ring as usize;
            let rows = self.states.iter().find(|s| s.draft).map_or(0, |s| s.rows);
            let floor = if self.draft_slots.borrow()[ring].is_some() {
                self.draft_floors.borrow()[ring].max(len.saturating_sub(rows))
            } else { len };
            let source = view(metadata.buffer, ring * 8, 8)?;
            // Synchronous host writes must not race a previous snapshot's async read.
            self.drain()?;
            self.engine.library.copy_h2d(source, &(floor as u64).to_le_bytes())?;
            let target = view(self.arenas[0].buffer,
                slot.0 as usize * self.rank_mark_bytes[0] + self.draft_metadata_offset(), 8)?;
            self.copy(0, target, source)?;
        }
        Ok(())
    }

    fn restore(&self, mark: Option<MarkSlot>, placement: &mut MimoPlacement, len: usize) -> Result<(), BoxError> {
        if len.div_ceil(PAGE_ROWS) > placement.pages.len() {
            return Err(format!("restore of {len} rows into {} pages", placement.pages.len()).into());
        }
        match mark {
            Some(slot) => self.move_mark(slot, placement.ring as usize, len, false)?,
            None if self.partial => self.empty_window(placement.ring as usize, len)?,
            None => return Err("MiMo restores exact snapshots with their positional mark".into()),
        }
        let ring = placement.ring as usize;
        if self.draft_metadata.is_some() { self.draft_floors.borrow_mut()[ring] = len; }
        if let (Some(slot), Some(metadata)) = (mark, &self.draft_metadata) {
            let source = view(self.arenas[0].buffer,
                slot.0 as usize * self.rank_mark_bytes[0] + self.draft_metadata_offset(), 8)?;
            let target = view(metadata.buffer, ring * 8, 8)?;
            self.copy(0, target, source)?;
            self.drain()?;
            let mut bytes = [0u8; 8];
            self.engine.library.copy_d2h(&mut bytes, target)?;
            self.draft_floors.borrow_mut()[ring] = usize::try_from(u64::from_le_bytes(bytes))?;
        }
        placement.len = len;
        Ok(())
    }

    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError> {
        for &(rank, buffer, record) in &self.full {
            let page = PAGE_ROWS * record;
            self.copy(rank, view(buffer, copy.to as usize * page, copy.rows * record)?,
                view(buffer, copy.from as usize * page, copy.rows * record)?)?;
        }
        Ok(())
    }

    fn drain(&self) -> Result<(), BoxError> {
        for rank in 0..self.engine.ranks() {
            self.engine.on(rank, || {
                // SAFETY: the engine owns this stream on the selected rank.
                unsafe { self.engine.library.cuda_stream_synchronize(self.engine.stream_of(rank)) }
            })?;
        }
        Ok(())
    }

    fn page_segments(&self, page: u32) -> Vec<DeviceRange> {
        self.full.iter().map(|&(_, buffer, record)| {
            let bytes = PAGE_ROWS * record;
            DeviceRange { addr: buffer.ptr as u64 + (page as usize * bytes) as u64, bytes }
        }).collect()
    }

    fn mark_segments(&self, slot: MarkSlot) -> Vec<DeviceRange> {
        self.arenas.iter().zip(&self.rank_mark_bytes).map(|(arena, &bytes)| DeviceRange {
            addr: arena.buffer.ptr as u64 + (slot.0 as usize * bytes) as u64,
            bytes,
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{draft_mark_rows, runs, DECODE_ROWS, RING_ROWS};

    #[test]
    fn draft_marks_exclude_speculative_and_target_reach_overwrites() {
        let ring = super::super::dflash::RING;
        for window in [64, 128, 192, 256] {
            let rows = draft_mark_rows(window);
            let reach = RING_ROWS.saturating_sub(DECODE_ROWS + window);
            assert!(rows + DECODE_ROWS + reach <= ring);
            assert_eq!(runs(2, ring, rows, 4096).iter().map(|r| r.2).sum::<usize>(), rows);
        }
        assert_eq!(draft_mark_rows(128), 896);
        assert_eq!(runs(3, ring, 896, 64), vec![(3 * ring, 0, 64)]);
    }

    #[test]
    fn mark_rows_follow_positions_across_the_ring_wrap() {
        // Positions 172..300 of ring 2: slots 172..255, then 0..43.
        assert_eq!(runs(2, 256, 128, 300), vec![(512 + 172, 0, 84), (512, 84, 44)]);
        // A short sequence keeps what it has; an aligned one is one run.
        assert_eq!(runs(0, 256, 128, 50), vec![(0, 0, 50)]);
        assert_eq!(runs(1, 256, 128, 256), vec![(256 + 128, 0, 128)]);
        assert!(runs(3, 256, 128, 0).is_empty());
        // The MTP hidden ring keeps window + stages + 1 rows.
        assert_eq!(runs(0, 256, 132, 1000), vec![(100, 0, 132)]);
        assert_eq!(runs(0, 256, 132, 1100), vec![(200, 0, 56), (0, 56, 76)]);
    }
}
