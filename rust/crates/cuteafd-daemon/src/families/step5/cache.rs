//! Committed state only. Proposal workspaces never alias these planes.
use crate::shared::memory::device::{Allocation, Device, Stream};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_loader::serving_capacity::step5::{ByteSpan, Step5CacheLayout};
use std::{cell::{Cell, RefCell}, rc::{Rc, Weak}};

pub(super) use cuteafd_loader::plan::families::step5::admission::Reservation;

/// Maintenance owns its stream; transactions use registered lane streams.
/// Boundary operations join every live lane before copying shared state.
pub(super) struct CacheRank<'a> {
    pub stream: Stream<'a>,
    lane_streams: RefCell<Vec<Weak<Stream<'a>>>>,
    active: Allocation<'a>,
    marks: Option<Rc<Allocation<'a>>>,
    pages: Option<Rc<Allocation<'a>>>,
    page_count: u64,
    layout: Step5CacheLayout,
    rank: usize,
    sequences: u64,
    slots: u64,
    marks_at: Vec<Option<u64>>,
    pending_bank: Vec<Cell<bool>>,
    predictor_evidence: Vec<Vec<Cell<u64>>>,
    pub(super) poisoned: Cell<bool>,
    pub(super) busy: Vec<Cell<bool>>,
    reservation: Reservation,
}

fn view(buffer: CuteafdDeviceBuffer, span: ByteSpan) -> Result<CuteafdDeviceBuffer> {
    let offset = usize::try_from(span.offset)?;
    let bytes = usize::try_from(span.bytes)?;
    ensure!(offset.checked_add(bytes).is_some_and(|end| end <= buffer.bytes),
        "Step-5 cache view outside allocation");
    Ok(CuteafdDeviceBuffer { ptr: buffer.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..buffer })
}

impl<'a> CacheRank<'a> {
    pub fn new(device: Device<'a>, layout: Step5CacheLayout, rank: usize,
        sequences: u64, slots: u64, pages: u64, available: usize) -> Result<Self> {
        let reservation = Reservation::new(&layout, rank, sequences, slots, pages, available)?;
        let marks_at = vec![None; usize::try_from(slots)?];
        let stream = Stream::new(device)?;
        // Allocate all storage before enqueueing initialization: failed allocation
        // cannot release an earlier plane still being initialized asynchronously.
        let active = Allocation::new(device, reservation.active)?;
        let marks = if reservation.marks > 0 { Some(Rc::new(Allocation::new(device, reservation.marks)?)) } else { None };
        let page_count = pages;
        let pages = if reservation.pages > 0 { Some(Rc::new(Allocation::new(device, reservation.pages)?)) } else { None };
        let predictor_evidence = (0..sequences).map(|_|
            (0..layout.ranks[rank].mtp_lengths.len()).map(|_| Cell::new(0)).collect()).collect();
        let owner = Self { stream, lane_streams: RefCell::new(Vec::new()), active, marks, pages, page_count, layout, rank, sequences, slots,
            marks_at, pending_bank: vec![Cell::new(false); usize::try_from(sequences)?], predictor_evidence,
            poisoned: Cell::new(false), busy: vec![Cell::new(false); usize::try_from(sequences)?], reservation };
        let mut guard = owner.stream.cancellation_guard("Step-5 cache initialization");
        device.run(|| {
            // SAFETY: the active allocation is live through completion on the owned stream.
            unsafe { device.library.cuda_zero_bytes_async(owner.active.buffer,
                owner.active.buffer.bytes, owner.stream.raw) }
        })?;
        owner.stream.drain()?;
        guard.complete = true;
        drop(guard);
        Ok(owner)
    }

    pub fn register_lane(&self, stream: &Rc<Stream<'a>>) -> Result<()> {
        ensure!(stream.device.id == self.stream.device.id
            && std::ptr::eq(stream.device.library, self.stream.device.library),
            "Step-5 cache lane stream device mismatch");
        let mut streams = self.lane_streams.borrow_mut();
        streams.retain(|stream| stream.strong_count() > 0);
        ensure!(streams.iter().all(|other| !other.ptr_eq(&Rc::downgrade(stream))),
            "Step-5 duplicate cache lane stream");
        streams.push(Rc::downgrade(stream));
        Ok(())
    }

    pub fn drain(&self) -> Result<()> {
        super::execution::collect_lane_results(self.lane_streams.borrow().iter()
            .filter_map(Weak::upgrade).map(|stream| stream.drain())
            .chain(std::iter::once_with(|| self.stream.drain())))
    }

    pub(super) fn validate_reset(&self, sequence: u64) -> Result<()> {
        self.check_idle(sequence)
    }

    /// Old KV and summaries remain masked by the zero frontier. Only canonical
    /// lengths and both pending banks must be initialized before slot reuse.
    pub(super) fn reset_sequence(&self, sequence: u64) -> Result<()> {
        self.validate_reset(sequence)?;
        self.drain()?;
        let state = &self.layout.ranks[self.rank];
        let spans = [state.length, state.next_length].into_iter()
            .chain(state.mtp_lengths.iter().copied())
            .chain(state.regions.iter().flat_map(|region| [region.pending_key,
                region.pending_z, region.next_key, region.next_z]));
        let mut guard = self.stream.cancellation_guard("Step-5 cold sequence reset");
        for span in spans {
            let target = self.request_component(span, sequence)?;
            self.stream.device.run(|| {
                // SAFETY: an idle private sequence owns these nonpaged banks;
                // the guard drains errors before any state can be reused.
                unsafe { self.stream.device.library.cuda_zero_bytes_async(
                    target, target.bytes, self.stream.raw) }
            })?;
        }
        self.stream.drain()?;
        self.pending_bank[sequence as usize].set(false);
        for stage in &self.predictor_evidence[sequence as usize] { stage.set(0); }
        guard.complete = true;
        Ok(())
    }

    pub fn bytes(&self) -> usize { self.reservation.total }
    pub fn identity(&self) -> usize { self.active.buffer.ptr as usize }
    pub fn page_pool(&self) -> CuteafdDeviceBuffer {
        self.pages.as_ref().map_or(CuteafdDeviceBuffer {
            ptr: std::ptr::null_mut(), bytes: 0, ..self.active.buffer
        }, |pages| pages.buffer)
    }
    /// Complete registered snapshot inventory, not just one slot or page view.
    pub(super) fn snapshot_owners(&self) -> Vec<Rc<Allocation<'a>>> {
        self.marks.iter().chain(self.pages.iter()).cloned().collect()
    }
    pub fn layout(&self) -> &Step5CacheLayout { &self.layout }
    pub fn rank(&self) -> usize { self.rank }
    pub fn sequences(&self) -> u64 { self.sequences }
    pub fn slots(&self) -> u64 { self.slots }
    pub fn page_count(&self) -> u64 { self.page_count }

    /// Native selected-GQA planes: [physical_page,64,local_kv_heads,192].
    pub fn csa_planes(&self, layer: usize) -> Result<(CuteafdDeviceBuffer, CuteafdDeviceBuffer)> {
        let at = self.layout.ranks[self.rank].regions.iter().position(|r| r.layer == layer)
            .context("Step-5 layer is not CSA")?;
        let regions = &self.layout.ranks[self.rank].regions;
        let bytes = regions[at].page_bytes.checked_mul(self.page_count)
            .context("Step-5 CSA plane overflow")?;
        let offset = regions[..at].iter().try_fold(0u64, |sum, r|
            sum.checked_add(r.page_bytes.checked_mul(2)?.checked_mul(self.page_count)?))
            .context("Step-5 CSA plane offset overflow")?;
        let pool = self.pages.as_ref().context("Step-5 rank has no CSA storage")?.buffer;
        Ok((view(pool, ByteSpan { offset, bytes })?,
            view(pool, ByteSpan { offset: offset.checked_add(bytes).context("Step-5 V plane overflow")?, bytes })?))
    }

    pub fn page_views(&self, page: u32) -> Result<Vec<CuteafdDeviceBuffer>> {
        ensure!(u64::from(page) < self.page_count(), "Step-5 CSA page outside pool");
        let mut result = Vec::new();
        for region in &self.layout.ranks[self.rank].regions {
            let (key, value) = self.csa_planes(region.layer)?;
            for plane in [key, value] {
                let bytes = plane.bytes as u64 / self.page_count();
                result.push(view(plane, ByteSpan { offset: u64::from(page) * bytes, bytes })?);
            }
        }
        Ok(result)
    }

    pub fn mark_view(&self, slot: u32) -> Result<CuteafdDeviceBuffer> {
        ensure!(u64::from(slot) < self.slots, "Step-5 mark outside arena");
        let bytes = self.layout.ranks[self.rank].bytes;
        view(self.marks.as_ref().context("Step-5 prefix retention is disabled")?.buffer,
            ByteSpan { offset: u64::from(slot).checked_mul(bytes).context("Step-5 mark offset overflow")?, bytes })
    }

    pub fn copy_page_rows(&self, from: u32, to: u32, rows: usize) -> Result<()> {
        ensure!(rows <= cuteafd_loader::serving_capacity::step5::PAGE_ROWS as usize,
            "Step-5 CSA tail longer than a page");
        ensure!(from != to, "Step-5 CSA COW requires distinct pages");
        self.drain()?;
        let source = self.page_views(from)?;
        let target = self.page_views(to)?;
        let mut guard = self.stream.cancellation_guard("Step-5 CSA tail copy");
        for (source, target) in source.into_iter().zip(target) {
            let bytes = source.bytes / 64 * rows;
            if bytes > 0 {
                self.copy(view(target, ByteSpan { offset: 0, bytes: bytes as u64 })?,
                    view(source, ByteSpan { offset: 0, bytes: bytes as u64 })?)?;
            }
        }
        self.stream.drain()?;
        guard.complete = true;
        Ok(())
    }

    /// Full contiguous request plane for one declared component; summaries use
    /// full admitted capacity even when a snapshot retains fewer complete rows.
    pub fn component(&self, span: ByteSpan) -> Result<CuteafdDeviceBuffer> {
        let state = &self.layout.ranks[self.rank];
        let declared = span == state.length || span == state.next_length || state.mtp_hidden == Some(span)
            || state.mtp_lengths.contains(&span) || state.mtp_stage_hidden.contains(&span)
            || state.sliding.iter().any(|s| s.key == span || s.value == span)
            || state.regions.iter().any(|r| r.pending_key == span || r.pending_z == span || r.next_key == span || r.next_z == span || r.summaries == Some(span));
        ensure!(declared, "Step-5 component is not part of this rank's cache layout");
        view(self.active.buffer, ByteSpan {
            offset: span.offset.checked_mul(self.sequences).context("Step-5 component offset overflow")?,
            bytes: span.bytes.checked_mul(self.sequences).context("Step-5 component size overflow")?,
        })
    }

    pub fn request_component(&self, span: ByteSpan, sequence: u64) -> Result<CuteafdDeviceBuffer> {
        ensure!(sequence < self.sequences, "Step-5 sequence outside cache");
        view(self.component(span)?, ByteSpan { offset: sequence * span.bytes, bytes: span.bytes })
    }

    pub fn pending(&self, layer: usize, sequence: u64, next: bool)
        -> Result<(CuteafdDeviceBuffer, CuteafdDeviceBuffer)> {
        ensure!(sequence < self.sequences, "Step-5 sequence outside cache");
        let region = self.layout.ranks[self.rank].regions.iter().find(|r| r.layer == layer)
            .context("Step-5 layer is not CSA")?;
        let bank = self.pending_bank[sequence as usize].get() ^ next;
        let (key, z) = if bank { (region.next_key, region.next_z) }
            else { (region.pending_key, region.pending_z) };
        Ok((self.request_component(key, sequence)?, self.request_component(z, sequence)?))
    }

    pub fn length(&self, sequence: u64, next: bool) -> Result<CuteafdDeviceBuffer> {
        ensure!(sequence < self.sequences, "Step-5 sequence outside cache");
        let state = &self.layout.ranks[self.rank];
        let bank = self.pending_bank[sequence as usize].get() ^ next;
        self.request_component(if bank { state.next_length } else { state.length }, sequence)
    }

    /// Submission evidence only; the authoritative length remains in device state.
    /// Restore refreshes this evidence once at the prefix boundary, never per step.
    pub(super) fn predictor_frontier(&self, sequence: u64, stage: usize) -> Result<u64> {
        self.validate_reset(sequence)?;
        Ok(self.predictor_evidence[sequence as usize].get(stage)
            .context("Step-5 predictor stage evidence missing")?.get())
    }

    pub(super) fn record_predictor_frontier(&self, sequence: u64, stage: usize, end: u64) -> Result<()> {
        self.predictor_evidence.get(sequence as usize).and_then(|stages| stages.get(stage))
            .context("Step-5 predictor publication evidence missing")?.set(end);
        Ok(())
    }

    fn predictor_frontiers(&self, sequence: u64, slot: Option<u64>) -> Result<Vec<u64>> {
        let mark = slot.map(|slot| self.mark_view(u32::try_from(slot)?)).transpose()?;
        self.layout.ranks[self.rank].mtp_lengths.iter().map(|&span| {
            ensure!(span.bytes == 8, "Step-5 predictor frontier width");
            let source = if let Some(mark) = mark { view(mark, span)? }
                else { self.request_component(span, sequence)? };
            let mut encoded = [0; 8];
            self.stream.device.run(|| self.stream.device.library.copy_d2h(&mut encoded, source))?;
            Ok(u64::from_le_bytes(encoded))
        }).collect()
    }

    fn snapshot_copies(&self, sequence: u64, slot: u64, frontier: u64, stages: &[u64])
        -> Result<Vec<cuteafd_loader::serving_capacity::step5::SnapshotCopy>> {
        let mut copies = self.layout.snapshot_copies(self.rank, frontier, stages, self.sequences,
            sequence, self.slots, slot)?;
        if self.pending_bank[sequence as usize].get() {
            let state = &self.layout.ranks[self.rank];
            let pairs = std::iter::once((state.length, state.next_length)).chain(state.regions.iter()
                .flat_map(|r| [(r.pending_key, r.next_key), (r.pending_z, r.next_z)]));
            for (canonical, selected) in pairs {
                let at = canonical.offset * self.sequences + sequence * canonical.bytes;
                if let Some(copy) = copies.iter_mut().find(|c| c.active.offset == at) {
                    copy.active.offset = selected.offset * self.sequences + sequence * selected.bytes;
                }
            }
        }
        Ok(copies)
    }

    /// Called only after all ranks have drained, with exclusive step ownership.
    pub fn publish_bank(&self, sequence: u64) {
        let bank = &self.pending_bank[sequence as usize];
        bank.set(!bank.get());
    }

    /// CSA page references/COW are handled by the shared prefix pool separately.
    /// This captures every nonpaged byte at the exact committed frontier only.
    pub fn capture(&mut self, sequence: u64, slot: u64, frontier: u64) -> Result<()> {
        self.check_idle(sequence)?;
        self.drain()?;
        let stages = self.predictor_frontiers(sequence, None)?;
        let copies = self.snapshot_copies(sequence, slot, frontier, &stages)?;
        let marks = self.marks.as_ref().context("Step-5 prefix retention is disabled")?.buffer;
        self.marks_at[usize::try_from(slot)?] = None;
        let mut guard = self.stream.cancellation_guard("Step-5 cache capture");
        let mark = self.mark_view(u32::try_from(slot)?)?;
        self.stream.device.run(|| {
            // SAFETY: this owned slot is unpublished, and the stream drains below.
            // Host snapshots store the full slot, including unused summary capacity.
            unsafe { self.stream.device.library.cuda_zero_bytes_async(mark, mark.bytes, self.stream.raw) }
        })?;
        for copy in copies {
            let source = view(self.active.buffer, copy.active)?;
            let target = view(marks, copy.mark)?;
            self.copy(target, source)?;
        }
        self.stream.drain()?;
        guard.complete = true;
        self.marks_at[usize::try_from(slot)?] = Some(frontier);
        Ok(())
    }

    pub fn restore(&mut self, sequence: u64, slot: u64, frontier: u64) -> Result<()> {
        self.check_idle(sequence)?;
        self.drain()?;
        let stages = self.predictor_frontiers(sequence, Some(slot))?;
        let copies = self.snapshot_copies(sequence, slot, frontier, &stages)?;
        ensure!(self.marks_at[usize::try_from(slot)?] == Some(frontier),
            "Step-5 mark is absent or belongs to a different frontier");
        self.restore_copies(copies)?;
        for (stage, end) in stages.into_iter().enumerate() {
            self.record_predictor_frontier(sequence, stage, end)?;
        }
        Ok(())
    }

    /// Validate canonical bytes, not arena-local capture bookkeeping, before
    /// any rank installs a host-promoted snapshot into active state.
    pub fn validate_retained(&self, sequence: u64, slot: u64, frontier: u64) -> Result<()> {
        self.check_idle(sequence)?;
        self.drain()?;
        let stages = self.predictor_frontiers(sequence, Some(slot))?;
        self.snapshot_copies(sequence, slot, frontier, &stages)?;
        let mark = self.mark_view(u32::try_from(slot)?)?;
        let length = self.layout.ranks[self.rank].length;
        ensure!(length.bytes == 8, "Step-5 canonical mark length width");
        let source = view(mark, length)?;
        let mut encoded = [0; 8];
        self.stream.device.run(|| self.stream.device.library.copy_d2h(&mut encoded, source))?;
        validate_mark_length(encoded, frontier)
    }

    /// Caller validates every rank with validate_retained before installing any
    /// rank. The shared family retains the arena slot across both phases.
    pub fn restore_retained(&self, sequence: u64, slot: u64, frontier: u64) -> Result<()> {
        self.check_idle(sequence)?;
        self.drain()?;
        let stages = self.predictor_frontiers(sequence, Some(slot))?;
        let copies = self.snapshot_copies(sequence, slot, frontier, &stages)?;
        self.restore_copies(copies)?;
        for (stage, end) in stages.into_iter().enumerate() {
            self.record_predictor_frontier(sequence, stage, end)?;
        }
        Ok(())
    }

    fn check_idle(&self, sequence: u64) -> Result<()> {
        ensure!(!self.poisoned.get() && sequence < self.sequences
            && !self.busy[sequence as usize].get(), "Step-5 cache sequence unavailable");
        Ok(())
    }

    fn restore_copies(&self, copies: Vec<cuteafd_loader::serving_capacity::step5::SnapshotCopy>) -> Result<()> {
        let marks = self.marks.as_ref().context("Step-5 prefix retention is disabled")?.buffer;
        let mut guard = self.stream.cancellation_guard("Step-5 cache restore");
        for copy in copies {
            let source = view(marks, copy.mark)?;
            let target = view(self.active.buffer, copy.active)?;
            self.copy(target, source)?;
        }
        self.stream.drain()?;
        guard.complete = true;
        Ok(())
    }

    fn copy(&self, target: CuteafdDeviceBuffer, source: CuteafdDeviceBuffer) -> Result<()> {
        self.stream.device.run(|| {
            // SAFETY: both checked views belong to this owner on the stream's device;
            // capture/restore drain (including error paths) before returning or freeing.
            unsafe { self.stream.device.library.copy_d2d_async(target, source, source.bytes, self.stream.raw) }
        })
    }
}

fn validate_mark_length(encoded: [u8; 8], frontier: u64) -> Result<()> {
    ensure!(u64::from_le_bytes(encoded) == frontier,
        "Step-5 canonical mark length differs from retained frontier");
    Ok(())
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(super) fn layout() -> Step5CacheLayout { layout_at(65) }

    pub(in crate::families::step5) fn layout_at(context: u64) -> Step5CacheLayout {
        layout_for(context, 1, 3, 64)
    }

    pub(in crate::families::step5) fn layout_for(context: u64, ranks: usize, mtp: usize, step: u64) -> Step5CacheLayout {
        let config = cuteafd_loader::plan::families::step5::Config {
            norm_eps: 1e-5, hidden: 4096, vocab: 128896, layers: 92, mtp: 3,
            dense: 13824, experts: 352, intermediate: 1536, shared: 1536, topk: 8,
            heads: 64, groups: 4, dim: 192, window: 512, max_context: 1 << 20,
            full: (0..95).map(|l| l < 92 && l % 4 == 3).collect(),
            moe: Vec::new(), theta: Vec::new(), rotary: Vec::new(),
            rope_scaling: None, scale_full: false, scale_sliding: false,
            activation: Vec::new(), shared_activation: Vec::new(),
        };
        Step5CacheLayout::new(&config, ranks, context, mtp, step).unwrap()
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and CUDA"]
    fn native_cold_slot_reset_clears_both_banks_without_touching_other_slots() -> Result<()> {
        // SAFETY: the selected library implements the native CUDA ABI.
        let library = unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let device = crate::shared::memory::device::cache_test_device(&library)?;
        let layout = layout_for(65, 1, 3, 64);
        let cache = CacheRank::new(device, layout.clone(), 0, 2, 1, 2, usize::MAX)?;
        let initial = vec![0xa5; cache.active.buffer.bytes];
        device.run(|| library.copy_h2d(cache.active.buffer, &initial))?;
        cache.publish_bank(1);
        cache.reset_sequence(1)?;
        let mut actual = vec![0; initial.len()];
        device.run(|| library.copy_d2h(&mut actual, cache.active.buffer))?;
        let mut expected = initial;
        let state = &layout.ranks[0];
        for span in [state.length, state.next_length].into_iter()
            .chain(state.mtp_lengths.iter().copied())
            .chain(state.regions.iter().flat_map(|region| [region.pending_key,
                region.pending_z, region.next_key, region.next_z])) {
            let offset = usize::try_from(span.offset * 2 + span.bytes)?;
            expected[offset..offset + usize::try_from(span.bytes)?].fill(0);
        }
        assert_eq!(actual, expected);
        assert!(!cache.pending_bank[1].get());
        cache.busy[1].set(true);
        assert!(cache.reset_sequence(1).is_err());
        cache.busy[1].set(false);
        assert!(cache.reset_sequence(2).is_err());
        Ok(())
    }

    #[test]
    fn canonical_mark_length_requires_exact_encoded_frontier() {
        for frontier in [0u64, 1, 63, 64, 65, 1 << 20, u64::MAX] {
            validate_mark_length(frontier.to_le_bytes(), frontier).unwrap();
            assert!(validate_mark_length(frontier.wrapping_add(1).to_le_bytes(), frontier).is_err());
            assert!(validate_mark_length(frontier.wrapping_sub(1).to_le_bytes(), frontier).is_err());
        }
    }

    #[test]
    fn cache_reservation_charges_all_storage_before_allocation() {
        let layout = layout();
        let state = layout.ranks[0].bytes as usize;
        let page = layout.geometry.ranks[0].persistent_unit_bytes as usize;
        let expected = state * 3 + state * 34 + page * 2;
        let r = Reservation::new(&layout, 0, 3, 34, 2, expected).unwrap();
        assert_eq!(r, Reservation { active: state * 3, marks: state * 34, pages: page * 2, total: expected });
        assert!(Reservation::new(&layout, 0, 3, 34, 2, expected - 1).is_err());
        assert!(Reservation::new(&layout, 1, 3, 34, 2, usize::MAX).is_err());
        assert!(Reservation::new(&layout, 0, u64::MAX, 34, 2, usize::MAX).is_err());
        assert!(Reservation::new(&layout, 0, 3, u64::MAX, 2, usize::MAX).is_err());
        assert!(Reservation::new(&layout, 0, 3, 34, u64::MAX, usize::MAX).is_err());
        assert!(Reservation::new(&layout, 0, 0, 34, 2, usize::MAX).is_err());
        assert!(Reservation::new(&layout, 0, 3, 34, 0, usize::MAX).is_err());
        assert_eq!(Reservation::new(&layout, 0, 3, 0, 2, usize::MAX).unwrap().marks, 0);
    }

    #[test]
    fn peer_without_csa_reserves_no_physical_pages() {
        let mut layout = layout_for(1024, 2, 0, 64);
        layout.ranks[1].regions.clear();
        layout.geometry.ranks[1].persistent_unit_bytes = 0;
        let state = layout.ranks[1].bytes as usize;
        let reservation = Reservation::new(&layout, 1, 3, 2, 16, state * 5).unwrap();
        assert_eq!(reservation.pages, 0);
        assert_eq!(reservation.total, state * 5);
        layout.geometry.ranks[1].persistent_unit_bytes = 1;
        assert!(Reservation::new(&layout, 1, 3, 2, 16, usize::MAX).is_err());
    }

    #[test]
    fn byte_views_reject_truncation_and_overflow_without_pointer_dereference() {
        let buffer = CuteafdDeviceBuffer { ptr: std::ptr::null_mut(), bytes: 256, device_id: 0, flags: 0 };
        assert_eq!(view(buffer, ByteSpan { offset: 16, bytes: 32 }).unwrap().bytes, 32);
        assert!(view(buffer, ByteSpan { offset: 255, bytes: 2 }).is_err());
        assert!(view(buffer, ByteSpan { offset: u64::MAX, bytes: 2 }).is_err());
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and CUDA"]
    fn native_cache_components_restore_exact_committed_bytes() -> Result<()> {
        // SAFETY: the explicitly selected native library implements the CUDA ABI.
        let library = unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let device = crate::shared::memory::device::cache_test_device(&library)?;
        let layout = layout();
        let mut cache = CacheRank::new(device, layout.clone(), 0, 2, 2, 1, usize::MAX)?;
        assert_eq!(cache.bytes(), Reservation::new(&layout, 0, 2, 2, 1, usize::MAX)?.total);
        assert!(cache.page_pool().bytes > 0);
        let copies = layout.snapshot_copies(0, 65, &[0; 3], 2, 0, 2, 0)?;
        for (i, copy) in copies.iter().enumerate() {
            let plane = view(cache.active.buffer, copy.active)?;
            let data: Vec<u8> = (0..plane.bytes).map(|n| (n.wrapping_mul(37) + i) as u8).collect();
            device.run(|| library.copy_h2d(plane, &data))?;
        }
        for &span in &layout.ranks[0].mtp_lengths {
            device.run(|| library.copy_h2d(cache.request_component(span, 0)?, &0u64.to_le_bytes()))?;
        }
        assert!(cache.restore(0, 0, 65).is_err());
        cache.capture(0, 0, 65)?;
        assert!(cache.restore(1, 0, 64).is_err());
        device.run(|| library.copy_h2d(cache.active.buffer, &vec![0xa5; cache.active.buffer.bytes]))?;
        cache.restore(1, 0, 65)?;
        let restored = layout.snapshot_copies(0, 65, &[0; 3], 2, 1, 2, 0)?;
        let mut actual = vec![0; cache.active.buffer.bytes];
        device.run(|| library.copy_d2h(&mut actual, cache.active.buffer))?;
        let mut expected = vec![0xa5; actual.len()];
        for (i, copy) in restored.iter().enumerate() {
            let at = copy.active.offset as usize;
            let predictor_length = layout.ranks[0].mtp_lengths.iter().any(|s|
                copy.mark.offset == s.offset);
            for n in 0..copy.active.bytes as usize {
                expected[at + n] = if predictor_length { 0 } else { (n.wrapping_mul(37) + i) as u8 };
            }
        }
        assert_eq!(actual, expected);
        assert!(cache.capture(2, 0, 65).is_err());
        assert!(cache.capture(0, 2, 65).is_err());
        Ok(())
    }
}
