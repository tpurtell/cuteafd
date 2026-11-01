//! Dense-summary Step-5 storage adapter for the shared exact prefix pool.
//! The executor still owns admission, page-table publication and committed lengths.
use super::{cache::CacheRank, transaction::Pending};
use anyhow::{ensure, Result};
use cuteafd_engine::prefix::{BoxError, FamilyLayout, Mark, MarkSlot, MarkStore, PageOwners,
    PrefixFamily, RestoreContext, RestorePlan, ReuseRule, TailCopy};
use cuteafd_hostcache::copy::DeviceRange;
use std::cell::RefCell;

#[derive(Debug, Clone)]
pub(super) struct Placement {
    pub pages: Vec<u32>,
    pub sequence: u64,
    pub committed: usize,
}

impl crate::shared::qwen::scheduler::Placement for Placement {
    fn len(&self) -> usize { self.committed }
    fn set_len(&mut self, len: usize) { self.committed = len; }
    fn units(&self) -> &[u32] { &self.pages }
    fn slot(&self) -> i32 { self.sequence as i32 }
    fn spans(&self) -> &[cuteafd_core::MediaSpan] { &[] }
}

/// Carried by both device and host entries. Actual stage lengths are in the mark.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct Frontier {
    committed: usize,
    logical_pages: usize,
    predictor_frontiers: Vec<u64>,
}

impl Frontier {
    fn at(committed: usize) -> Self {
        Self { committed, logical_pages: committed.div_ceil(64), predictor_frontiers: Vec::new() }
    }

    pub(super) fn supports_predictors(&self, frontier: usize, depth: usize) -> bool {
        self.committed == frontier && self.predictor_frontiers.len() == depth
            && self.predictor_frontiers.iter().enumerate().all(|(stage, &end)|
                end == frontier.saturating_sub(stage + 1) as u64)
    }

    fn validate_restore(&self, placement: &Placement, plan: &RestorePlan) -> Result<()> {
        ensure!(plan.exact() && plan.check(plan.snapshot_end).is_ok(),
            "Step-5 does not reconstruct approximate prefix state");
        ensure!(placement.committed == 0, "Step-5 restore requires a fresh destination");
        ensure!(self.committed == plan.resume() && self.logical_pages == plan.resume().div_ceil(64)
                && self.logical_pages <= placement.pages.len(),
            "Step-5 retained mark frontier mismatch");
        Ok(())
    }
}

pub(super) struct Step5Prefix<'e, 'a, T = ()> {
    ranks: Vec<&'e RefCell<CacheRank<'a>>>,
    layout: FamilyLayout,
    kv_format: cuteafd_loader::families::qwen4::Qwen4KvCache,
    sequences: u64,
    context: u64,
    pending: Pending<T>,
}

fn range(buffer: cuteafd_ffi::CuteafdDeviceBuffer) -> DeviceRange {
    DeviceRange { addr: buffer.ptr as u64, bytes: buffer.bytes }
}

impl<'e, 'a> Step5Prefix<'e, 'a> {
    pub fn new(ranks: Vec<&'e RefCell<CacheRank<'a>>>) -> Result<Self> {
        Self::with_pending(ranks, Pending::default())
    }
}

impl<'e, 'a, T> Step5Prefix<'e, 'a, T> {
    pub(super) fn with_pending(ranks: Vec<&'e RefCell<CacheRank<'a>>>, pending: Pending<T>) -> Result<Self> {
        ensure!(!ranks.is_empty(), "Step-5 prefix requires cache ranks");
        let first = ranks[0].borrow();
        let shape = first.layout().clone();
        ensure!(ranks.len() == shape.ranks.len(), "Step-5 prefix rank count");
        let sequences = first.sequences();
        let slots = first.slots();
        let pages = first.page_count();
        let context = shape.context;
        drop(first);
        let mut page_bytes = 0usize;
        let mut mark_bytes = 0usize;
        for (rank, owner) in ranks.iter().enumerate() {
            let cache = owner.borrow();
            ensure!(cache.rank() == rank && cache.layout() == &shape,
                "Step-5 prefix ranks disagree on layout");
            ensure!(cache.sequences() == sequences && cache.slots() == slots && cache.page_count() == pages,
                "Step-5 prefix ranks disagree on capacity");
            let geometry = &shape.geometry.ranks[rank];
            page_bytes = page_bytes.checked_add(usize::try_from(geometry.persistent_unit_bytes)?)
                .ok_or_else(|| anyhow::anyhow!("Step-5 page size overflow"))?;
            mark_bytes = mark_bytes.checked_add(usize::try_from(shape.ranks[rank].bytes)?)
                .ok_or_else(|| anyhow::anyhow!("Step-5 mark size overflow"))?;
        }
        ensure!(pages <= u32::MAX as u64 && slots <= u32::MAX as u64,
            "Step-5 shared pool index overflow");
        let layout = FamilyLayout { page_rows: 64, pages: usize::try_from(pages)?, page_bytes,
            mark_bytes, draft_bytes: 0, rule: ReuseRule::EXACT,
            mark_store: MarkStore::Arena, page_owners: PageOwners::Uniform };
        Ok(Self { ranks, layout, kv_format: shape.kv_format, sequences, context, pending })
    }

    pub(super) fn slots(&self) -> usize { self.ranks[0].borrow().slots() as usize }

    pub(super) fn snapshot_owners(&self) -> Vec<std::rc::Rc<crate::shared::memory::device::Allocation<'a>>> {
        self.ranks.iter().flat_map(|rank| rank.borrow().snapshot_owners()).collect()
    }

    fn validate(&self, placement: &Placement, len: usize) -> Result<()> {
        ensure!(placement.sequence < self.sequences && len as u64 <= self.context,
            "Step-5 placement outside admitted capacity");
        ensure!(len.div_ceil(64) <= placement.pages.len(), "Step-5 placement lacks CSA pages");
        ensure!(placement.pages.iter().all(|p| (*p as usize) < self.layout.pages),
            "Step-5 placement page outside pool");
        Ok(())
    }

    fn restore_mark(&self, slot: MarkSlot, placement: &mut Placement, len: usize) -> Result<()> {
        self.validate(placement, len)?;
        // Validate every rank before any active state is overwritten. Host
        // promotion can reuse an arena slot without local capture bookkeeping.
        for owner in &self.ranks {
            owner.borrow().validate_retained(placement.sequence, u64::from(slot.0), len as u64)?;
        }
        for owner in &self.ranks {
            owner.borrow().restore_retained(placement.sequence, u64::from(slot.0), len as u64)?;
        }
        placement.committed = len;
        Ok(())
    }
}

impl<T> PrefixFamily<Frontier> for Step5Prefix<'_, '_, T> {
    type Placement = Placement;

    fn layout(&self) -> FamilyLayout { self.layout }
    fn record_format(&self) -> &'static str {
        match self.kv_format {
            cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16 => "step5_bf16_csa64_dense_e4m3_marks_v1",
            cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8 => "step5_fp8_head_scales_csa64_dense_e4m3_marks_v1",
        }
    }
    fn pages<'p>(&self, placement: &'p Placement) -> &'p [u32] { &placement.pages }
    /// Each wave uploads the placement's pages; growth happens between rounds only.
    fn extend_pages(&self, placement: &mut Placement, pages: &[u32]) -> Result<(), BoxError> {
        placement.pages.extend_from_slice(pages);
        self.validate(placement, placement.committed)?;
        Ok(())
    }
    fn commit_point(&self, placement: &Placement) -> usize { placement.committed }

    fn bind(&self, placement: &mut Placement) -> Result<(), BoxError> {
        self.pending.retire();
        self.validate(placement, placement.committed)?;
        if placement.committed == 0 {
            for owner in &self.ranks { owner.borrow().validate_reset(placement.sequence)?; }
            super::execution::collect_lane_results(self.ranks.iter()
                .map(|owner| owner.borrow().reset_sequence(placement.sequence)))?;
        }
        Ok(())
    }

    fn capture_meta(&self, placement: &Placement, len: usize) -> Result<Frontier, BoxError> {
        self.validate(placement, len)?;
        if len != placement.committed {
            return Err("Step-5 capture must be at the committed frontier".into());
        }
        let mut frontier = Frontier::at(len);
        for rank in &self.ranks {
            let rank = rank.borrow();
            for stage in 0..rank.layout().ranks[rank.rank()].mtp_lengths.len() {
                frontier.predictor_frontiers.push(rank.predictor_frontier(placement.sequence, stage)?);
            }
        }
        Ok(frontier)
    }

    fn capture(&self, slot: MarkSlot, placement: &Placement, len: usize) -> Result<(), BoxError> {
        self.capture_meta(placement, len)?;
        for owner in &self.ranks {
            owner.borrow_mut().capture(placement.sequence, u64::from(slot.0), len as u64)?;
        }
        Ok(())
    }

    fn restore(&self, mark: Option<MarkSlot>, placement: &mut Placement, len: usize) -> Result<(), BoxError> {
        self.pending.retire();
        let slot = mark.ok_or("Step-5 exact restore requires a positional mark")?;
        self.validate(placement, len)?;
        // Direct callers use local capture bookkeeping. Host-tier restores use
        // restore_with and the entry's self-describing frontier instead.
        for owner in &self.ranks {
            owner.borrow_mut().restore(placement.sequence, u64::from(slot.0), len as u64)?;
        }
        placement.committed = len;
        Ok(())
    }

    fn restore_with(&self, mark: Option<&Mark>, saved: Option<&Frontier>, placement: &mut Placement,
        plan: &RestorePlan, _context: &RestoreContext<'_>) -> Result<(), BoxError> {
        self.pending.retire();
        let saved = saved.ok_or("Step-5 retained mark frontier absent")?;
        saved.validate_restore(placement, plan)?;
        let Some(Mark::Slot(slot)) = mark else {
            return Err("Step-5 exact restore requires an arena mark".into());
        };
        let mut stage = 0;
        for owner in &self.ranks {
            let rank = owner.borrow();
            let count = rank.layout().ranks[rank.rank()].mtp_lengths.len();
            let expected = saved.predictor_frontiers.get(stage..stage + count)
                .ok_or("Step-5 retained predictor metadata count mismatch")?;
            rank.validate_predictor_metadata(placement.sequence, u64::from(slot.0), expected)?;
            stage += count;
        }
        if stage != saved.predictor_frontiers.len() {
            return Err("Step-5 retained predictor metadata count mismatch".into());
        }
        self.restore_mark(*slot, placement, plan.resume())?;
        Ok(())
    }

    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError> {
        if copy.owner.is_some() { return Err("Step-5 uses uniform head-split pages".into()); }
        for owner in &self.ranks {
            owner.borrow().copy_page_rows(copy.from, copy.to, copy.rows)?;
        }
        Ok(())
    }

    fn drain(&self) -> Result<(), BoxError> {
        super::execution::collect_lane_results(self.ranks.iter().map(|owner| owner.borrow().drain()))?;
        Ok(())
    }

    fn page_segments(&self, page: u32) -> Vec<DeviceRange> {
        self.ranks.iter().flat_map(|owner| owner.borrow().page_views(page)
            .expect("shared prefix pool passed an invalid Step-5 page").into_iter().map(range)).collect()
    }

    fn mark_segments(&self, slot: MarkSlot) -> Vec<DeviceRange> {
        self.ranks.iter().map(|owner| range(owner.borrow().mark_view(slot.0)
            .expect("shared prefix arena passed an invalid Step-5 slot"))).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::memory::device::cache_test_device;
    use cuteafd_engine::prefix::{After, PrefixCache, PrefixConfig, SnapshotKind};
    use cuteafd_hostcache::copy::StubCopyEngine;

    #[test]
    fn reservation_top_up_past_16k_stalls_and_resumes_with_live_page_tables() -> Result<()> {
        use crate::shared::kv_reserve::{Reserve, Reserving};
        use cuteafd_engine::reserve::{Lease, ReserveConfig};
        struct Request { placement: Placement, lease: Lease, produced: usize, initial_rows: usize }
        impl Reserving for Request {
            type Placement = Placement;
            fn lease(&self) -> &Lease { &self.lease }
            fn committed(&self) -> usize { self.placement.committed }
            fn produced(&self) -> usize { self.produced }
            fn placement(&self) -> &Placement { &self.placement }
            fn placement_mut(&mut self) -> &mut Placement { &mut self.placement }
            fn closed(&self) -> bool { false }
        }
        // No CUDA owners are needed to exercise the production placement hook.
        // A nonzero committed prompt avoids resetting a device sequence at bind.
        let family = Step5Prefix { ranks: Vec::new(), sequences: 3, context: 1 << 20,
            kv_format: cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16,
            layout: FamilyLayout { page_rows: 64, pages: 849, page_bytes: 64,
                mark_bytes: 0, draft_bytes: 0, rule: ReuseRule::EXACT,
                mark_store: MarkStore::Arena, page_owners: PageOwners::Uniform },
            pending: Pending::<()>::default() };
        let mut cache = PrefixCache::<StubCopyEngine, Frontier>::new(family.layout(), PrefixConfig {
            entries: 0, mark_slots: 0, keep_logits: false, min_tokens: 64 }, None)?;
        let mut reserve = Reserve::new(ReserveConfig::new(640, 0, -1, 1 << 20)?, 64);
        let (mut active, mut stalled) = (Vec::<Request>::new(), Vec::<Request>::new());
        for sequence in 0..3 {
            let lease = reserve.lease(16_320, 3000, 4, Reserve::limit(&cache, &family, 1 << 20))?;
            let rows = reserve.admit(&cache, &family, &active, &stalled, &[], &lease).unwrap();
            let placement = cache.admit_cold(&family, 16_320, rows,
                |pages| Placement { pages, sequence, committed: 16_320 })?.placement;
            let initial_rows = placement.pages.len() * 64;
            active.push(Request { placement, lease, produced: 1, initial_rows });
        }
        let mut waited = std::collections::BTreeSet::new();
        let (mut saw_resume, mut wrote_past_initial) = (false, false);
        for _ in 0..4000 {
            reserve.round(&mut cache, &family, &mut active, &mut stalled, &[], |_, _| panic!("no early end"))?;
            for request in &stalled { waited.insert(request.placement.sequence); }
            let descriptors = active.iter().enumerate().map(|(index, request)| {
                let p = &request.placement;
                saw_resume |= waited.contains(&p.sequence);
                wrote_past_initial |= p.committed + 4 > request.initial_rows;
                super::super::transaction::SequenceRows { sequence: p.sequence, start: index * 4,
                    rows: 4, base: p.committed, pages: &p.pages }
            }).collect::<Vec<_>>();
            if !descriptors.is_empty() {
                super::super::transaction::validate_wave(&descriptors, cache.pool(), 3, 12, 1 << 20, 849)?;
            }
            for request in &mut active {
                request.placement.committed += 4;
                request.produced += 4;
            }
            let (done, running): (Vec<_>, Vec<_>) = active.drain(..)
                .partition(|request| request.placement.committed >= request.lease.end());
            active = running;
            for request in done { cache.release(&family, &request.placement.pages)?; }
            if active.is_empty() && stalled.is_empty() { break; }
        }
        assert!(active.is_empty() && stalled.is_empty());
        assert!(saw_resume && wrote_past_initial && !waited.is_empty());
        assert!(reserve.stats().top_ups > 0 && reserve.stats().stalls > 0);
        assert_eq!(cache.pool().free(), 849);
        Ok(())
    }

    #[test]
    fn drafting_restore_requires_every_canonical_predictor_frontier() {
        for end in [1usize, 2, 3, 63, 64, 65, 255, 256, 257, 511, 512, 513, 1492] {
            for depth in 1..=3 {
                let mut saved = Frontier::at(end);
                saved.predictor_frontiers = (0..depth).map(|stage| end.saturating_sub(stage + 1) as u64).collect();
                assert!(saved.supports_predictors(end, depth));
                assert!(!saved.supports_predictors(end + 1, depth));
                assert!(!saved.supports_predictors(end, depth + 1));
                for stage in 0..depth {
                    let expected = saved.predictor_frontiers[stage];
                    saved.predictor_frontiers[stage] = expected + 1;
                    assert!(!saved.supports_predictors(end, depth));
                    saved.predictor_frontiers[stage] = expected;
                }
                saved.predictor_frontiers.fill(0);
                assert_eq!(saved.supports_predictors(end, depth), end <= 1);
            }
        }
    }

    #[test]
    fn prefix_entry_points_retire_pending_before_validation() -> Result<(), BoxError> {
        use std::{cell::Cell, rc::Rc};
        struct Evidence(Rc<Cell<usize>>);
        impl Drop for Evidence {
            fn drop(&mut self) { self.0.set(self.0.get() + 1); }
        }
        let retired = Rc::new(Cell::new(0));
        let pending = Pending::default();
        let mut family = Step5Prefix { ranks: Vec::new(), sequences: 0, context: 0,
            kv_format: cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16,
            layout: FamilyLayout { page_rows: 64, pages: 1, page_bytes: 64,
                mark_bytes: 0, draft_bytes: 0, rule: ReuseRule::EXACT,
                mark_store: MarkStore::Arena, page_owners: PageOwners::Uniform },
            pending: pending.clone() };
        let bf16_identity = family.record_format();
        family.kv_format = cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8;
        assert_ne!(family.record_format(), bf16_identity);
        family.kv_format = cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16;
        assert_eq!(family.record_format(), bf16_identity);
        let mut placement = Placement { pages: vec![0], sequence: 0, committed: 1 };
        pending.verify(|| Ok((Evidence(retired.clone()), ())))?;
        assert!(family.bind(&mut placement).is_err());
        assert_eq!(retired.get(), 1);
        pending.verify(|| Ok((Evidence(retired.clone()), ())))?;
        assert!(family.restore(None, &mut placement, 1).is_err());
        assert_eq!(retired.get(), 2);
        pending.verify(|| Ok((Evidence(retired.clone()), ())))?;
        let plan = RestorePlan { snapshot_end: 1, target_end: 1,
            lag: cuteafd_engine::prefix::LaggedState { source_end: 1, replay_start: 1 },
            fidelity: cuteafd_engine::prefix::RestoreFidelity::Exact };
        let context = RestoreContext { native_tokens: &[1], media: &[] };
        assert!(family.restore_with(None, None, &mut placement, &plan, &context).is_err());
        assert_eq!(retired.get(), 3);
        Ok(())
    }

    #[test]
    fn retained_frontier_requires_exact_plan_and_fresh_logical_destination() {
        use cuteafd_engine::prefix::{LaggedState, RestoreFidelity};
        for len in [0usize, 1, 63, 64, 65, 1024] {
            let frontier = Frontier::at(len);
            let plan = RestorePlan { snapshot_end: len, target_end: len + 1,
                lag: LaggedState { source_end: len, replay_start: len }, fidelity: RestoreFidelity::Exact };
            let placement = Placement { pages: vec![37; len.div_ceil(64)], sequence: 1, committed: 0 };
            frontier.validate_restore(&placement, &plan).unwrap();
            let mut remapped = placement.clone();
            remapped.pages.fill(19);
            frontier.validate_restore(&remapped, &plan).unwrap();
            remapped.committed = 1;
            assert!(frontier.validate_restore(&remapped, &plan).is_err());
            let mut wrong = frontier.clone();
            wrong.logical_pages += 1;
            assert!(wrong.validate_restore(&placement, &plan).is_err());
            wrong = frontier.clone();
            wrong.committed += 1;
            assert!(wrong.validate_restore(&placement, &plan).is_err());
            let mut invalid = plan;
            invalid.fidelity = RestoreFidelity::ApproximateReplay;
            assert!(frontier.validate_restore(&placement, &invalid).is_err());
            invalid = plan;
            invalid.lag.source_end += 1;
            assert!(frontier.validate_restore(&placement, &invalid).is_err());
            invalid = plan;
            invalid.target_end = len.saturating_sub(1);
            if len > 0 { assert!(frontier.validate_restore(&placement, &invalid).is_err()); }
            if len > 0 {
                remapped = placement.clone();
                remapped.pages.pop();
                assert!(frontier.validate_restore(&remapped, &plan).is_err());
                invalid = plan;
                invalid.lag.replay_start -= 1;
                assert!(frontier.validate_restore(&placement, &invalid).is_err());
            }
        }
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and CUDA"]
    fn native_restore_validates_all_rank_lengths_before_active_writes() -> Result<(), BoxError> {
        // SAFETY: the selected library implements the native CUDA ABI.
        let library = unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let device = cache_test_device(&library)?;
        let plan = RestorePlan { snapshot_end: 65, target_end: 66,
            lag: cuteafd_engine::prefix::LaggedState { source_end: 65, replay_start: 65 },
            fidelity: cuteafd_engine::prefix::RestoreFidelity::Exact };
        let tokens: Vec<u32> = (0..66).collect();
        let context = RestoreContext { native_tokens: &tokens, media: &[] };
        for depth in 0..=3 {
            let shape = super::super::cache::tests::layout_for(66, 2, depth, 64);
            let ranks = (0..2).map(|rank| CacheRank::new(device, shape.clone(), rank, 1, 1, 2, usize::MAX)
                .map(RefCell::new)).collect::<Result<Vec<_>>>()?;
            let family = Step5Prefix::new(ranks.iter().collect())?;
            let mut saved = Frontier::at(65);
            saved.predictor_frontiers = (0..depth).map(|stage| 64 - stage as u64).collect();
            let mut counters = Vec::new();
            let mut stage = 0;
            for (rank, owner) in ranks.iter().enumerate() {
                let mark = owner.borrow().mark_view(0)?;
                let mut bytes = vec![0; mark.bytes];
                let state = &shape.ranks[rank];
                for (span, value) in std::iter::once((state.length, 65)).chain(
                    state.mtp_lengths.iter().copied().map(|span| {
                        let value = saved.predictor_frontiers[stage];
                        stage += 1;
                        (span, value)
                    })) {
                    let at = span.offset as usize;
                    bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
                    counters.push((rank, span, value));
                }
                device.run(|| library.copy_h2d(mark, &bytes))?;
            }
            assert_eq!(stage, depth);
            let pristine = || -> Result<Vec<Vec<u8>>, BoxError> {
                counters.iter().map(|&(rank, span, _)| {
                    let mut bytes = vec![0; span.bytes as usize];
                    device.run(|| library.copy_d2h(&mut bytes, ranks[rank].borrow().request_component(span, 0)?))?;
                    Ok(bytes)
                }).collect()
            };
            let before = pristine()?;
            let mut destination = Placement { pages: vec![0, 1], sequence: 0, committed: 0 };
            // Corrupt both target ranks and every predictor stage independently.
            for &(rank, span, value) in &counters {
                let mark = ranks[rank].borrow().mark_view(0)?;
                let encoded = cuteafd_ffi::CuteafdDeviceBuffer { ptr: mark.ptr
                    .wrapping_byte_add(span.offset as usize), bytes: 8, ..mark };
                device.run(|| library.copy_h2d(encoded, &(value - 1).to_le_bytes()))?;
                assert!(family.restore_with(Some(&Mark::Slot(MarkSlot(0))), Some(&saved),
                    &mut destination, &plan, &context).is_err());
                assert_eq!(destination.committed, 0);
                assert_eq!(pristine()?, before, "no rank may publish before every retained counter validates");
                device.run(|| library.copy_h2d(encoded, &value.to_le_bytes()))?;
            }
            for stage in 0..depth {
                let mut wrong = saved.clone();
                wrong.predictor_frontiers[stage] -= 1;
                assert!(family.restore_with(Some(&Mark::Slot(MarkSlot(0))), Some(&wrong),
                    &mut destination, &plan, &context).is_err());
                assert_eq!(pristine()?, before);
            }
            family.restore_with(Some(&Mark::Slot(MarkSlot(0))), Some(&saved),
                &mut destination, &plan, &context)?;
            assert_eq!(destination.committed, 65);
            for ((_, _, value), bytes) in counters.iter().zip(pristine()?) {
                assert_eq!(bytes, value.to_le_bytes());
            }
            println!("Step-5 retained counter validation depth={depth} all_ranks_before_write=true");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and CUDA"]
    fn native_shared_prefix_restores_marks_and_csa_partial_pages() -> std::result::Result<(), BoxError> {
        // SAFETY: the selected library implements the native CUDA ABI.
        let library = unsafe { cuteafd_ffi::NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let device = cache_test_device(&library)?;
        let shape = super::super::cache::tests::layout_at(66);
        let owner = RefCell::new(CacheRank::new(device, shape.clone(), 0, 2, 2, 8, usize::MAX)?);
        let family = Step5Prefix::new(vec![&owner])?;
        let config = PrefixConfig { entries: 2, mark_slots: 2, keep_logits: false, min_tokens: 1 };
        let mut shared = PrefixCache::<StubCopyEngine, Frontier>::new(family.layout(), config, None)?;
        let tokens: Vec<u32> = (0..65).collect();
        let mut source = shared.admit(&family, &tokens, 65, false,
            |pages| Placement { pages, sequence: 0, committed: 0 })?.placement;
        source.committed = 65;
        assert!(family.capture(MarkSlot(0), &source, 64).is_err());
        assert!(family.restore(None, &mut source, 65).is_err());
        let first = source.pages[0];
        let tail = source.pages[1];
        let set = |buffer: cuteafd_ffi::CuteafdDeviceBuffer, byte: u8| -> Result<()> {
            device.run(|| library.copy_h2d(buffer, &vec![byte; buffer.bytes]))
        };
        for (i, region) in shape.ranks[0].regions.iter().enumerate() {
            let (key, value) = owner.borrow().csa_planes(region.layer)?;
            assert_eq!(key.bytes, 8 * 64 * 4 * 192 * 2);
            assert_eq!(value.bytes, key.bytes);
            assert_ne!(key.ptr, value.ptr);
            assert_eq!(i, shape.ranks[0].regions.iter().position(|r| r.layer == region.layer).unwrap());
        }
        for page in 0..8 {
            for plane in owner.borrow().page_views(page)? { set(plane, 0xa5)?; }
        }
        for rows in [0, 1, 7, 8, 63, 64] {
            for plane in owner.borrow().page_views(7)? { set(plane, 0xa5)?; }
            for plane in owner.borrow().page_views(first)? { set(plane, 0x19)?; }
            family.copy_rows(TailCopy { from: first, to: 7, rows, owner: None })?;
            for plane in owner.borrow().page_views(7)? {
                let mut bytes = vec![0; plane.bytes];
                device.run(|| library.copy_d2h(&mut bytes, plane))?;
                let retained = rows * plane.bytes / 64;
                assert!(bytes[..retained].iter().all(|b| *b == 0x19));
                assert!(bytes[retained..].iter().all(|b| *b == 0xa5));
            }
        }
        for plane in owner.borrow().page_views(tail)? { set(plane, 0x29)?; }
        let spans = shape.snapshot_spans(0, 65, &[0; 3])?;
        let components = |span: &cuteafd_loader::serving_capacity::step5::ByteSpan| {
            let state = &shape.ranks[0];
            std::iter::once(state.length).chain(state.mtp_lengths.iter().copied()).chain(state.mtp_hidden)
                .chain(state.mtp_stage_hidden.iter().copied())
                .chain(state.sliding.iter().flat_map(|r| [r.key, r.value]))
                .chain(state.regions.iter().flat_map(|r| [Some(r.pending_key), Some(r.pending_z), r.summaries].into_iter().flatten()))
                .find(|s| span.offset >= s.offset && span.offset + span.bytes <= s.offset + s.bytes).unwrap()
        };
        for span in &spans {
            let full = components(span);
            set(owner.borrow().component(full)?, 0x39)?;
        }
        device.run(|| library.copy_h2d(owner.borrow().length(0, false)?, &65u64.to_le_bytes()))?;
        for &span in &shape.ranks[0].mtp_lengths {
            device.run(|| library.copy_h2d(owner.borrow().request_component(span, 0)?, &0u64.to_le_bytes()))?;
        }
        let captured = family.capture_meta(&source, 65)?;
        assert!(shared.capture(&family, SnapshotKind::Prompt, &tokens, &source, After::default())?);
        // Simulate host-tier promotion into a different arena slot: its bytes
        // and entry metadata are valid, but no local capture ever used slot 1.
        let mark = owner.borrow().mark_view(0)?;
        let mut saved = vec![0; mark.bytes];
        device.run(|| library.copy_d2h(&mut saved, mark))?;
        device.run(|| library.copy_h2d(owner.borrow().mark_view(1)?, &saved))?;
        let plan = RestorePlan { snapshot_end: 65, target_end: 66,
            lag: cuteafd_engine::prefix::LaggedState { source_end: 65, replay_start: 65 },
            fidelity: cuteafd_engine::prefix::RestoreFidelity::Exact };
        let context = RestoreContext { native_tokens: &tokens, media: &[] };
        let mut promoted = source.clone();
        promoted.sequence = 1;
        promoted.committed = 0;
        assert!(family.restore(Some(MarkSlot(1)), &mut promoted, 65).is_err());
        assert!(family.restore_with(Some(&Mark::Slot(MarkSlot(1))), Some(&Frontier::at(64)),
            &mut promoted, &plan, &context).is_err());
        let length = shape.ranks[0].length;
        let encoded = cuteafd_ffi::CuteafdDeviceBuffer { ptr: owner.borrow().mark_view(1)?.ptr
            .wrapping_byte_add(length.offset as usize), bytes: 8, ..mark };
        device.run(|| library.copy_h2d(encoded, &64u64.to_le_bytes()))?;
        assert!(family.restore_with(Some(&Mark::Slot(MarkSlot(1))), Some(&captured),
            &mut promoted, &plan, &context).is_err());
        assert_eq!(promoted.committed, 0);
        device.run(|| library.copy_h2d(encoded, &65u64.to_le_bytes()))?;
        family.restore_with(Some(&Mark::Slot(MarkSlot(1))), Some(&captured),
            &mut promoted, &plan, &context)?;
        // Only the source's partial page is mutable; the full page remains shared.
        for plane in owner.borrow().page_views(tail)? { set(plane, 0x49)?; }
        for span in &spans {
            let full = components(span);
            set(owner.borrow().component(full)?, 0xa5)?;
        }
        let mut continuation = tokens.clone();
        continuation.push(66);
        let resumed = shared.admit(&family, &continuation, 66, false,
            |pages| Placement { pages, sequence: 1, committed: 0 })?;
        assert_eq!(resumed.resume, 65);
        assert_eq!(resumed.placement.committed, 65);
        assert_eq!(resumed.placement.pages[0], first);
        assert_ne!(resumed.placement.pages[1], tail);
        for plane in owner.borrow().page_views(resumed.placement.pages[1])? {
            let mut bytes = vec![0; plane.bytes];
            device.run(|| library.copy_d2h(&mut bytes, plane))?;
            assert!(bytes[..plane.bytes / 64].iter().all(|b| *b == 0x29));
        }
        for span in &spans {
            let full = components(span);
            let plane = owner.borrow().component(full)?;
            let mut bytes = vec![0; plane.bytes];
            device.run(|| library.copy_d2h(&mut bytes, plane))?;
            let stride = full.bytes as usize;
            assert!(bytes[..stride].iter().all(|b| *b == 0xa5));
            let offset = (span.offset - full.offset) as usize;
            let retained = &bytes[stride + offset..stride + offset + span.bytes as usize];
            if *span == shape.ranks[0].length {
                assert_eq!(retained, &65u64.to_le_bytes());
            } else if shape.ranks[0].mtp_lengths.contains(span) {
                assert_eq!(retained, &0u64.to_le_bytes());
            } else {
                assert!(retained.iter().all(|b| *b == 0x39));
            }
        }
        assert!(owner.borrow().page_views(8).is_err());
        assert!(owner.borrow().copy_page_rows(0, 1, 65).is_err());
        shared.release(&family, &source.pages)?;
        shared.release(&family, &resumed.placement.pages)?;
        Ok(())
    }
}
