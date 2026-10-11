//! DeepSeek V4.1 as a prefix-cache family (`cuteafd_engine::prefix`).
//!
//! **Pages.** One engine page is a 512-token unit: one page (256 source rows) of each ratio-two
//! source (layers 2, 8, 14) and two pages of the ratio-one source (layer 20), 5 x 91,136 B =
//! 455,680 B (~890 B/token). A request's units are allocated for its whole declared lifetime at
//! admission (V4.1 admits by declared lifetime) and bound to every source's page table, so the
//! decode path never allocates; a restore shares the snapshot's full units and copies its
//! partial tail unit once (one eager copy per restore and per capture, PLAN decision 4).
//!
//! **Mark.** Arena slots hold the positional state: every window's ring rows and each source's
//! odd-frontier carry (`BackboneMark::BYTES`, 2,720,064 B, on the prefix-copy device) and, with
//! dSpark, the three drafter rings (`DRAFT_MARK_BYTES`, 202,752 B, on the drafter's
//! device): one combined mark per snapshot, so a restored request drafts warm (MiMo
//! precedent). [`V41Meta`] describes it as plain data: window spans, the frontier, the Engram
//! lookback and the drafter's frontier.
//!
//! **Restore plans.** Exact frontiers restore the mark byte-exactly. An exact ancestor followed
//! by a suffix of at least 128 tokens resumes as an encoder continuation (encoder rings and carry
//! restored, decoder rings rebuilt by the suffix's final replay): still exact. A partial match is
//! `ApproximateReplay`: sources are shared through the even-aligned common prefix
//! (`source_end`), encoder windows restart empty 128 tokens earlier (`replay_start`) and Engram
//! history is rebuilt there from the prompt's native ids.
use super::*;
use super::scores::RetainedScores;
use crate::families::deepseek_v41::v41_backbone_cache::{BackboneMark, CacheLease, UNIT_BYTES, UNIT_TOKENS};
use crate::families::deepseek_v41::v41_requests::RequestMark;
use crate::shared::memory::device::{Allocation, Device};
use crate::shared::prefix::{cuda_copy::range, CudaCopyEngine};
use cuteafd_engine::prefix::{
    After, BoxError, CaptureTicket, Captured, Eviction, FamilyLayout, LaggedState, Mark, MarkSlot, MarkStore,
    PrefixCache, PrefixConfig, PrefixFamily, RestoreCandidate, RestoreContext, RestoreFidelity, RestorePlan,
    ReuseRule, TailCopy,
};
use cuteafd_hostcache::copy::DeviceRange;
use serde::{Deserialize, Serialize};
use std::cell::{Cell, RefCell};
pub(crate) use cuteafd_engine::media::MediaSpan;
/// A prompt's media-keyed token copy for the radix and its image spans (engine `MediaKeys`).
pub(crate) type ImageKeys = cuteafd_engine::media::MediaKeys;
pub(crate) use cuteafd_engine::prefix::SnapshotKind;

/// A V4.1 snapshot's metadata: what its mark's bytes mean. Plain, serializable data.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct V41Meta {
    pub request: RequestMark,
    /// The drafter rings' frontier, when the mark carries them.
    pub draft_end: Option<u64>,
}

/// One admitted request's place in the cache: its units and its lease.
pub(crate) struct V41Placement {
    pub units: Vec<u32>,
    pub lease: CacheLease,
    /// Draft request identity when dSpark runs.
    pub draft: Option<u64>,
    /// Tokens the placement holds committed state for (the commit point).
    pub len: usize,
    /// The restore plan that produced this placement, if any.
    pub plan: Option<RestorePlan>,
}

/// The family's device side, borrowing the request bank and the drafter for one call.
pub(crate) struct V41Prefix<'r, 'a, 'w, C: DraftChain<'a>> {
    requests: RefCell<&'r mut Requests<'a>>,
    draft: RefCell<Option<&'r mut DraftRuntime<'w, 'a, C>>>,
    arena: &'r Arenas<'a>,
    units: usize,
    partial: bool,
    /// Capture lane of the next queued capture (`queue_capture`'s ticket is the lane).
    lane: Cell<usize>,
}

/// The mark arenas, allocated before serving: target marks on the backbone's prefix-copy
/// device, draft marks on the drafter's.
pub(crate) struct Arenas<'a> {
    target: Option<std::rc::Rc<Allocation<'a>>>,
    draft: Option<std::rc::Rc<Allocation<'a>>>,
    slots: usize,
}

impl<'a> Arenas<'a> {
    pub fn new(target: Device<'a>, draft: Option<Device<'a>>, slots: usize) -> Result<Self> {
        let _scope = cuteafd_ffi::memory_ledger::scope("prefix/snapshot");
        let target = (slots > 0).then(|| Allocation::new(target, slots * BackboneMark::BYTES).map(std::rc::Rc::new))
            .transpose()?;
        let draft = match draft {
            Some(device) if slots > 0 => Some(std::rc::Rc::new(Allocation::new(device, slots * super::speculative::DRAFT_MARK_BYTES)?)),
            _ => None,
        };
        Ok(Self { target, draft, slots })
    }
    pub fn device_bytes(slots: usize, draft: bool) -> usize {
        slots * (BackboneMark::BYTES + if draft { super::speculative::DRAFT_MARK_BYTES } else { 0 })
    }
    fn target(&self, slot: MarkSlot) -> Result<cuteafd_ffi::CuteafdDeviceBuffer> {
        let arena = self.target.as_ref().context("no prefix mark arena")?;
        ensure!((slot.0 as usize) < self.slots, "mark slot {} of {}", slot.0, self.slots);
        Ok(view(arena.buffer, slot.0 as usize * BackboneMark::BYTES, BackboneMark::BYTES))
    }
    fn draft(&self, slot: MarkSlot) -> Result<Option<cuteafd_ffi::CuteafdDeviceBuffer>> {
        Ok(match &self.draft {
            Some(arena) => {
                ensure!((slot.0 as usize) < self.slots, "mark slot {} of {}", slot.0, self.slots);
                Some(view(arena.buffer, slot.0 as usize * super::speculative::DRAFT_MARK_BYTES, super::speculative::DRAFT_MARK_BYTES))
            }
            None => None,
        })
    }
    pub fn owners(&self) -> Vec<std::rc::Rc<Allocation<'a>>> {
        self.target.iter().chain(&self.draft).cloned().collect()
    }
}

fn view(buffer: cuteafd_ffi::CuteafdDeviceBuffer, offset: usize, bytes: usize) -> cuteafd_ffi::CuteafdDeviceBuffer {
    debug_assert!(offset + bytes <= buffer.bytes);
    cuteafd_ffi::CuteafdDeviceBuffer { ptr: unsafe { buffer.ptr.cast::<u8>().add(offset).cast() }, bytes, ..buffer }
}

fn boxed(error: anyhow::Error) -> BoxError {
    format!("{error:#}").into()
}

/// V4.1's geometry for the prefix engine over `units` 512-token units.
pub(crate) fn layout(units: usize, draft: bool, partial: bool) -> FamilyLayout {
    FamilyLayout {
        page_rows: UNIT_TOKENS,
        pages: units,
        page_bytes: UNIT_BYTES,
        mark_bytes: BackboneMark::BYTES + if draft { super::speculative::DRAFT_MARK_BYTES } else { 0 },
        draft_bytes: 0,
        rule: if partial { ReuseRule::V41 } else { ReuseRule::EXACT },
        mark_store: MarkStore::Arena,
    }
}

/// The automatic host budget: logical capacity above `entries * max_context` tokens (as
/// before), in this layout's slabs: one unit slab per 512 tokens plus one mark per snapshot.
pub(crate) fn host_budget(layout: FamilyLayout, device_tokens: u64, entries: u32, max_context: u32, chunk: u64)
    -> Result<cuteafd_hostcache::budget::Budget> {
    let mut budget = cuteafd_hostcache::budget::plan(device_tokens, entries, max_context, chunk, false)?;
    if budget.pinned_bytes == 0 { return Ok(budget); }
    let unit = UNIT_TOKENS as u64;
    let snapshots = 2 * u64::from(entries) + 2;
    let units = budget.host_tokens / unit + budget.staging_tokens / unit + snapshots;
    let chunks = |slabs: u64, bytes: usize| -> Result<u64> {
        let per = chunk / bytes as u64;
        ensure!(per > 0, "host cache chunk is smaller than a V4.1 snapshot slab");
        Ok(slabs.div_ceil(per))
    };
    budget.pinned_bytes = (chunks(units, layout.page_bytes)? + chunks(snapshots, layout.mark_bytes)?)
        .checked_mul(chunk).context("host cache budget overflow")?;
    Ok(budget)
}

/// Where V4.1 resumes a snapshot: the default plan, except that an exact ancestor followed by
/// at least 128 new tokens continues its encoder (exact), and a partial match shares sources
/// through the even-aligned, media-safe common prefix and replays 128 tokens before it.
pub(crate) fn plan(hit: RestoreCandidate<'_>, partial: bool) -> RestorePlan {
    if hit.common == hit.snapshot_end {
        return RestorePlan::under(ReuseRule::V41, hit);
    }
    let source_end = if partial { hit.media_safe(hit.common / 2 * 2) } else { 0 };
    let replay_start = hit.media_safe(source_end.saturating_sub(128));
    RestorePlan {
        snapshot_end: hit.snapshot_end,
        target_end: hit.target_end,
        lag: LaggedState { source_end: if replay_start == 0 { 0 } else { source_end }, replay_start },
        fidelity: RestoreFidelity::ApproximateReplay,
    }
}

impl<'r, 'a, 'w, C: DraftChain<'a>> V41Prefix<'r, 'a, 'w, C> {
    pub fn new(requests: &'r mut Requests<'a>, draft: Option<&'r mut DraftRuntime<'w, 'a, C>>, arena: &'r Arenas<'a>,
        partial: bool) -> Self {
        let units = requests.cache().unit_capacity();
        Self { requests: RefCell::new(requests), draft: RefCell::new(draft), arena, units, partial, lane: Cell::new(0) }
    }
    /// Capture lane for the next queued capture.
    pub fn on_lane(&self, lane: usize) -> &Self {
        self.lane.set(lane);
        self
    }
    fn bind_units(&self, placement: &V41Placement, rows: usize) -> Result<()> {
        let mut requests = self.requests.borrow_mut();
        let pages = requests.cache().unit_pages(&placement.units);
        let rows = requests.cache().source_rows(rows);
        let pages = [&pages[0][..], &pages[1][..], &pages[2][..], &pages[3][..]];
        requests.bind_units(placement.lease, placement.units.clone(), pages, rows)
    }
}

impl<'a, C: DraftChain<'a>> PrefixFamily<V41Meta> for V41Prefix<'_, 'a, '_, C> {
    type Placement = V41Placement;

    fn layout(&self) -> FamilyLayout {
        layout(self.units, self.arena.draft.is_some(), self.partial)
    }
    fn eviction(&self) -> Eviction {
        Eviction::FreesPages
    }
    fn plan_restore(&self, hit: RestoreCandidate<'_>) -> RestorePlan {
        plan(hit, self.partial)
    }
    fn capture_meta(&self, placement: &V41Placement, len: usize) -> Result<V41Meta, BoxError> {
        let requests = self.requests.borrow();
        let backbone = requests.cache().mark_meta(placement.lease).map_err(boxed)?;
        if backbone.end as usize != len {
            return Err(format!("V4.1 captures its committed frontier only ({} != {len})", backbone.end).into());
        }
        let (position, recent) = requests.engram_lookback(placement.lease).map_err(boxed)?;
        let draft_end = match (placement.draft, self.draft.borrow().as_ref()) {
            (Some(id), Some(draft)) => Some(draft.committed_end(id).map_err(boxed)?),
            _ => None,
        };
        Ok(V41Meta { request: RequestMark { backbone, engram_position: position, engram_recent: recent }, draft_end })
    }
    fn bind(&self, placement: &mut V41Placement) -> Result<(), BoxError> {
        self.bind_units(placement, 0).map_err(boxed)
    }
    fn discard(&self, placement: &mut V41Placement) -> Result<(), BoxError> {
        // The engine releases the placement's units itself: take them off the request so its
        // release (here or by the scheduler) does not unreference them a second time. A
        // failed restore may already have released the lease; then they were queued for
        // reclaim instead, and are taken back from there.
        let mut requests = self.requests.borrow_mut();
        if requests.take_units(placement.lease).is_empty() {
            requests.forget_released_units(&placement.units);
        }
        Ok(())
    }
    fn restore_with(&self, mark: Option<&Mark>, saved: Option<&V41Meta>, placement: &mut V41Placement,
        plan: &RestorePlan, context: &RestoreContext<'_>) -> Result<(), BoxError> {
        // The forked rows through `source_end` are initialized in the bound pages.
        self.bind_units(placement, plan.lag.source_end).map_err(boxed)?;
        let mut requests = self.requests.borrow_mut();
        // Tail copies of forked units were enqueued on the prefix stream; restores below read
        // the pages only through later kernels on the same stream or after its drain.
        if plan.exact() {
            let (Some(Mark::Slot(slot)), Some(saved)) = (mark, saved) else {
                return Err("an exact V4.1 restore needs its arena mark and metadata".into());
            };
            let continuation = (plan.target_end - plan.snapshot_end >= 128).then_some(plan.target_end as u64);
            let target = self.arena.target(*slot).map_err(boxed)?;
            requests.restore_mark(placement.lease, &saved.request, target, continuation).map_err(boxed)?;
            if continuation.is_none() {
                let mut draft = self.draft.borrow_mut();
                match (placement.draft, draft.as_deref_mut(), saved.draft_end, self.arena.draft(*slot).map_err(boxed)?) {
                    (Some(id), Some(draft), Some(end), Some(ring)) => draft.restore_mark(id, end, ring).map_err(boxed)?,
                    (None, None, None, _) | (None, None, _, None) => {}
                    _ => return Err("retained execution mode differs".into()),
                }
            }
            placement.len = plan.snapshot_end;
        } else {
            let start = requests.restore_encoder_prefix(placement.lease, plan.lag.source_end, context.native_tokens)
                .map_err(boxed)?;
            if start != plan.lag.replay_start {
                return Err(format!("encoder replay starts at {start}, plan {}", plan.lag.replay_start).into());
            }
            placement.len = start;
        }
        placement.plan = Some(*plan);
        Ok(())
    }
    fn pages<'p>(&self, placement: &'p V41Placement) -> &'p [u32] {
        &placement.units
    }
    fn commit_point(&self, placement: &V41Placement) -> usize {
        placement.len
    }
    fn capture(&self, slot: MarkSlot, placement: &V41Placement, len: usize) -> Result<(), BoxError> {
        let target = self.arena.target(slot).map_err(boxed)?;
        let saved = self.requests.borrow_mut().capture_mark(placement.lease, target).map_err(boxed)?;
        if saved.backbone.end as usize != len {
            return Err("V4.1 captures its committed frontier only".into());
        }
        if let (Some(id), Some(draft), Some(ring)) = (placement.draft, self.draft.borrow_mut().as_deref_mut(),
            self.arena.draft(slot).map_err(boxed)?) {
            draft.capture_mark(id, len as u64, ring).map_err(boxed)?;
        }
        Ok(())
    }
    fn queue_capture(&self, slot: MarkSlot, placement: &V41Placement, len: usize, tail: Option<TailCopy>)
        -> Result<Option<CaptureTicket>, BoxError> {
        let lane = self.lane.get();
        if let Some(tail) = tail { self.copy_rows(tail)?; }
        // The tail copy runs on the prefix stream; order the lane's copies after it.
        self.requests.borrow().cache().drain_prefix_stream().map_err(boxed)?;
        let target = self.arena.target(slot).map_err(boxed)?;
        let saved = self.requests.borrow_mut().queue_mark(lane, placement.lease, target).map_err(boxed)?;
        if saved.backbone.end as usize != len {
            let _ = self.requests.borrow_mut().abort_mark(lane);
            return Err("V4.1 captures its committed frontier only".into());
        }
        if let (Some(id), Some(draft), Some(ring)) = (placement.draft, self.draft.borrow_mut().as_deref_mut(),
            self.arena.draft(slot).map_err(boxed)?) {
            if let Err(error) = draft.queue_mark(lane, id, len as u64, ring) {
                let _ = self.requests.borrow_mut().abort_mark(lane);
                return Err(boxed(error));
            }
        }
        Ok(Some(CaptureTicket(lane as u32)))
    }
    fn capture_ready(&self, ticket: CaptureTicket) -> Result<bool, BoxError> {
        let lane = ticket.0 as usize;
        let mut requests = self.requests.borrow_mut();
        let mut draft = self.draft.borrow_mut();
        // Poll both before publishing: a capture is complete only when target and drafter
        // copies both landed.
        let target = if requests.mark_pending(lane) { requests.mark_ready(lane).map_err(boxed)? } else { true };
        let rings = match draft.as_deref_mut() {
            Some(draft) if draft.mark_pending(lane) => draft.mark_ready(lane).map_err(boxed)?,
            _ => true,
        };
        Ok(target && rings)
    }
    fn abort_capture(&self, ticket: CaptureTicket) -> Result<(), BoxError> {
        let lane = ticket.0 as usize;
        let target = self.requests.borrow_mut().abort_mark(lane);
        let rings = self.draft.borrow_mut().as_deref_mut().map_or(Ok(()), |draft| draft.abort_mark(lane));
        target.and(rings).map_err(boxed)
    }
    fn host_restored(&self, pages: &[u32]) -> Result<(), BoxError> {
        self.requests.borrow().cache().publish_units(pages).map_err(boxed)
    }
    fn restore(&self, _: Option<MarkSlot>, _: &mut V41Placement, _: usize) -> Result<(), BoxError> {
        Err("V4.1 restores through its restore plan".into())
    }
    fn copy_rows(&self, copy: TailCopy) -> Result<(), BoxError> {
        let requests = self.requests.borrow();
        requests.cache().copy_unit_rows(copy.from, copy.to, copy.rows).map_err(boxed)?;
        // Replicated layouts mirror the copied tail before any placement reads it.
        requests.cache().publish_units(&[copy.to]).map_err(boxed)
    }
    fn drain(&self) -> Result<(), BoxError> {
        self.requests.borrow().cache().drain_prefix_stream().map_err(boxed)
    }
    fn page_segments(&self, page: u32) -> Vec<DeviceRange> {
        self.requests.borrow().cache().unit_segments(page).into_iter().map(range).collect()
    }
    fn mark_segments(&self, slot: MarkSlot) -> Vec<DeviceRange> {
        let mut segments = Vec::with_capacity(2);
        if let Ok(target) = self.arena.target(slot) { segments.push(range(target)); }
        if let Ok(Some(ring)) = self.arena.draft(slot) { segments.push(range(ring)); }
        segments
    }
}

/// The scheduler's prefix cache: the engine's [`PrefixCache`] with V4.1's metadata, plus the
/// family's arenas and the capture lanes' pending tickets.
pub(crate) struct V41Cache<'a> {
    cache: PrefixCache<CudaCopyEngine<'a>, V41Meta>,
    arenas: Arenas<'a>,
    partial: bool,
    pending: [Option<CaptureTicket>; 2],
    capture_session: Option<String>,
}

/// What a V4.1 admission restored: the resume point and, for an exact full-prompt hit, the
/// retained scores of the first token.
pub(crate) struct Restored {
    pub cached: usize,
    pub scores: Option<RetainedScores>,
    pub plan: Option<RestorePlan>,
}

impl<'a> V41Cache<'a> {
    pub fn new(arenas: Arenas<'a>, config: PrefixConfig, partial: bool,
        host: Option<(cuteafd_hostcache::config::Config, CudaCopyEngine<'a>)>, layout: FamilyLayout) -> Result<Self> {
        let cache = PrefixCache::new(layout, config, host).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Self { cache, arenas, partial, pending: [None, None], capture_session: None })
    }
    pub fn arenas(&self) -> &Arenas<'a> { &self.arenas }
    /// A cache that retains nothing (component fixtures).
    #[cfg(test)]
    pub fn disabled() -> Self {
        Self::configured(0)
    }
    /// A cache of `entries` per bank with no arena or host tier (CPU fixtures: nothing is
    /// captured).
    #[cfg(test)]
    pub fn configured(entries: usize) -> Self {
        let config = PrefixConfig { entries, mark_slots: 0, keep_logits: true, min_tokens: 1 };
        Self::new(Arenas { target: None, draft: None, slots: 0 }, config, true, None, layout(1, false, true))
            .expect("a fixture prefix cache")
    }
    pub fn partial(&self) -> bool { self.partial }
    pub fn engine(&self) -> &PrefixCache<CudaCopyEngine<'a>, V41Meta> { &self.cache }
    pub fn family<'r, 'w, C: DraftChain<'a>>(&'r self, requests: &'r mut Requests<'a>,
        draft: Option<&'r mut DraftRuntime<'w, 'a, C>>) -> V41Prefix<'r, 'a, 'w, C> {
        V41Prefix::new(requests, draft, &self.arenas, self.partial)
    }
    pub fn capture_session(&mut self, session: Option<String>) {
        self.capture_session = session.clone();
        self.cache.capture_session(session);
    }
    pub fn restored_session(&self) -> Option<&str> { self.cache.restored_session() }
    pub fn turn_bank_enabled(&self) -> bool { self.cache.turn_bank_enabled() }
    pub fn host_config(&self) -> Option<&cuteafd_hostcache::config::Config> { self.cache.host_config() }
    pub fn host_metrics(&self) -> Option<cuteafd_hostcache::metrics::Snapshot> { self.cache.stats().host }
    pub fn stats(&self) -> cuteafd_engine::prefix::PrefixStats { self.cache.stats() }
    pub fn prefill_hold(&mut self) -> Result<()> { self.cache.prefill_hold() }

    /// Unreference released requests' units (after the family drained) and poll the host tier.
    pub fn tick<C: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>, draft: Option<&mut DraftRuntime<'_, 'a, C>>)
        -> Result<()> {
        self.reclaim(requests, draft)?;
        self.cache.tick();
        Ok(())
    }
    fn reclaim<C: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>, draft: Option<&mut DraftRuntime<'_, 'a, C>>)
        -> Result<()> {
        let released = requests.take_released_units();
        if released.is_empty() { return Ok(()); }
        let family = V41Prefix::new(requests, draft, &self.arenas, self.partial);
        for units in released {
            self.cache.release(&family, &units).map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        Ok(())
    }

    /// Admit a request leased at `lease` (fresh, its draft request admitted as `draft_id`):
    /// look `keys` up (media-keyed copy of `native`), allocate units for `capacity` tokens
    /// (prompt plus its declared output), restore. Pool pressure is `SourcePoolExhausted`; a
    /// restore that fails is a miss.
    #[allow(clippy::too_many_arguments)]
    pub fn admit<C: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>, mut draft: Option<&mut DraftRuntime<'_, 'a, C>>,
        lease: CacheLease, draft_id: Option<u64>, keys: &[u32], native: &[u32], media: &[MediaSpan], capacity: usize,
        cold: bool) -> Result<Restored> {
        self.reclaim(requests, draft.as_deref_mut())?;
        let family = V41Prefix::new(requests, draft, &self.arenas, self.partial);
        let build = |units: Vec<u32>| V41Placement { units, lease, draft: draft_id, len: 0, plan: None };
        let admitted = if cold || !self.cache.enabled() {
            self.cache.admit_cold(&family, keys.len(), capacity, build)
        } else {
            self.cache.admit_native(&family, keys, native, media, capacity, true, build)
        };
        let admitted = match admitted {
            Ok(admitted) => admitted,
            Err(cuteafd_engine::prefix::PrefixError::Pages(pages)) => {
                return Err(crate::families::deepseek_v41::v41_compressor::SourcePoolExhausted {
                    work_index: 0, needed: pages.needed, available: pages.free }.into());
            }
            Err(error) => return Err(anyhow::anyhow!("{error}")),
        };
        let scores = match &admitted.after {
            Some(after) => {
                let logits = after.logits.as_ref().context("exact prefix has no retained logits")?;
                let bytes = logits.iter().flat_map(|v| v.to_ne_bytes()).collect();
                Some(RetainedScores::new(super::scores::VOCAB, bytes)?)
            }
            None => None,
        };
        let plan = admitted.placement.plan;
        Ok(Restored { cached: admitted.resume, scores, plan })
    }

    /// Whether `work` (each request's remaining declared tokens) fits the units bound to it.
    pub fn fits(&self, requests: &Requests<'a>, work: &[(CacheLease, u32)]) -> Result<bool> {
        match requests.cache().check_append_capacity(work) {
            Ok(()) => Ok(true),
            Err(error) if exhausted(&error) => Ok(false),
            Err(error) => Err(error),
        }
    }

    /// Give back the units of an idle request past what `keep_tokens` needs (its output
    /// allowance shrank to fit the pool).
    pub fn trim<C: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>, draft: Option<&mut DraftRuntime<'_, 'a, C>>,
        lease: CacheLease, keep_tokens: usize) -> Result<()> {
        let mut units = requests.units(lease)?.to_vec();
        let keep = keep_tokens.div_ceil(UNIT_TOKENS).max(1);
        if units.len() <= keep { return Ok(()); }
        let dropped = units.split_off(keep);
        let (pages, rows) = {
            let cache = requests.cache();
            let committed = cache.committed_end(lease)? as usize;
            (cache.unit_pages(&units), cache.source_rows(committed))
        };
        requests.rebind_shrunk(lease, units, [&pages[0], &pages[1], &pages[2], &pages[3]], rows)?;
        let family = V41Prefix::new(requests, draft, &self.arenas, self.partial);
        self.cache.release(&family, &dropped).map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Units a pool of `free` units plus every snapshot-only unit could still give.
    pub fn could_fit(&self, units: usize) -> bool { self.cache.could_free(units) }

    /// Retain a snapshot of `lease` at its committed frontier (synchronously).
    #[allow(clippy::too_many_arguments)]
    pub fn retain<C: DraftChain<'a>>(&mut self, requests: &mut Requests<'a>, draft: Option<&mut DraftRuntime<'_, 'a, C>>,
        kind: SnapshotKind, keys: &[u32], media: &[MediaSpan], next: &RetainedScores, lease: CacheLease, draft_id: Option<u64>)
        -> Result<()> {
        if !self.cache.enabled() { return Ok(()); }
        let end = requests.cache().committed_end(lease)? as usize;
        ensure!(end > 0 && end <= keys.len(), "retained token frontier differs");
        let units = requests.units(lease)?.to_vec();
        let placement = V41Placement { units, lease, draft: draft_id, len: end, plan: None };
        let after = After::from_logits(&next.logits()?, true);
        let family = V41Prefix::new(requests, draft, &self.arenas, self.partial);
        self.cache.capture_media(&family, kind, &keys[..end], media, &placement, after)
            .map(drop).map_err(|e| anyhow::anyhow!("{e}"))
    }
    /// [`V41Cache::retain`] with its copies queued on capture lane `lane`; poll it with
    /// [`V41Cache::poll_retain`]. `Ok(false)` when nothing was queued.
    #[allow(clippy::too_many_arguments)]
    pub fn queue_retain<C: DraftChain<'a>>(&mut self, lane: usize, requests: &mut Requests<'a>,
        draft: Option<&mut DraftRuntime<'_, 'a, C>>, kind: SnapshotKind, keys: &[u32], media: &[MediaSpan],
        next: &RetainedScores, lease: CacheLease, draft_id: Option<u64>) -> Result<bool> {
        ensure!(self.pending.get(lane).context("invalid retention lane")?.is_none(), "retention lane occupied");
        if !self.cache.enabled() { return Ok(false); }
        let end = requests.cache().committed_end(lease)? as usize;
        ensure!(end > 0 && end <= keys.len(), "retained token frontier differs");
        let units = requests.units(lease)?.to_vec();
        let placement = V41Placement { units, lease, draft: draft_id, len: end, plan: None };
        let after = After::from_logits(&next.logits()?, true);
        let family = V41Prefix::new(requests, draft, &self.arenas, self.partial);
        family.on_lane(lane);
        match self.cache.queue_capture(&family, kind, &keys[..end], media, &placement, after)
            .map_err(|e| anyhow::anyhow!("{e}"))? {
            Captured::Queued(ticket) => {
                self.pending[lane] = Some(ticket);
                Ok(true)
            }
            Captured::Done | Captured::Skipped => Ok(false),
        }
    }
    pub fn poll_retain<C: DraftChain<'a>>(&mut self, lane: usize, requests: &mut Requests<'a>,
        draft: Option<&mut DraftRuntime<'_, 'a, C>>) -> Result<bool> {
        let ticket = self.pending.get(lane).copied().flatten().context("retention is not pending")?;
        let family = V41Prefix::new(requests, draft, &self.arenas, self.partial);
        let done = self.cache.poll_capture(&family, ticket).map_err(|e| anyhow::anyhow!("{e}"))?;
        if done { self.pending[lane] = None; }
        Ok(done)
    }
    pub fn abort_retain<C: DraftChain<'a>>(&mut self, lane: usize, requests: &mut Requests<'a>,
        draft: Option<&mut DraftRuntime<'_, 'a, C>>) -> Result<()> {
        let Some(ticket) = self.pending.get_mut(lane).and_then(Option::take) else { return Ok(()) };
        let family = V41Prefix::new(requests, draft, &self.arenas, self.partial);
        self.cache.abort_capture(&family, ticket).map_err(|e| anyhow::anyhow!("{e}"))
    }
}

/// The radix key of a request's `tokens` (its prompt then its generated tokens): the prompt's
/// media-keyed copy, then the generated tokens' native ids (images occur only in prompts).
pub(crate) fn keyed<'t>(keys: &ImageKeys, tokens: &'t [u32]) -> std::borrow::Cow<'t, [u32]> {
    if keys.spans().is_empty() { return std::borrow::Cow::Borrowed(tokens); }
    let prompt = keys.tokens();
    let mut out = prompt[..prompt.len().min(tokens.len())].to_vec();
    out.extend_from_slice(&tokens[out.len()..]);
    std::borrow::Cow::Owned(out)
}

/// The draft request identity of request `id` when a drafter runs.
pub(crate) fn draft_id<T>(draft: &Option<T>, id: u64) -> Option<u64> {
    draft.is_some().then_some(id)
}

pub(crate) fn exhausted(error: &anyhow::Error) -> bool {
    error.downcast_ref::<crate::families::deepseek_v41::v41_compressor::SourcePoolExhausted>().is_some()
}

/// The engine's media spans of a V4.1 prompt's images (full 256-bit identities) and the prompt's
/// media-keyed token copy for the radix.
pub(crate) fn media_keys(tokens: &[u32], images: &[cuteafd_loader::V41ImageSpan])
    -> Result<cuteafd_engine::media::MediaKeys> {
    let spans: Vec<MediaSpan> = images.iter().map(|span| MediaSpan {
        start: span.start,
        len: span.image.grid().tokens(),
        key: cuteafd_core::ImageKey(*span.image.identity()).into(),
    }).collect();
    cuteafd_engine::media::MediaKeys::new(tokens, super::scores::VOCAB as u32, &spans)
        .map_err(|e| anyhow::anyhow!("image cache keys: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_core::ImageKey;

    fn candidate<'m>(common: usize, snapshot_end: usize, target_end: usize, media: &'m [MediaSpan]) -> RestoreCandidate<'m> {
        RestoreCandidate { common, snapshot_end, target_end, resume: ReuseRule::V41.skipped(common, snapshot_end),
            media, saved_media: media }
    }

    #[test]
    fn exact_frontiers_restore_exactly_and_partial_ones_replay_from_an_even_source_frontier() {
        let exact = plan(candidate(1001, 1001, 1001, &[]), true);
        assert_eq!((exact.fidelity, exact.lag), (RestoreFidelity::Exact, LaggedState { source_end: 1001, replay_start: 1001 }));
        // Odd frontier, long suffix: still exact (encoder continuation is the family's choice).
        let ancestor = plan(candidate(777, 777, 2000, &[]), true);
        assert!(ancestor.exact() && ancestor.resume() == 777);
        // A partial match keeps sources through the even common prefix, replays 128 earlier.
        let partial = plan(candidate(1001, 1500, 1600, &[]), true);
        assert_eq!((partial.fidelity, partial.lag),
            (RestoreFidelity::ApproximateReplay, LaggedState { source_end: 1000, replay_start: 872 }));
        assert!(partial.check(1001).is_ok());
        // Too short to replay a window: nothing reused; partial reuse off: nothing either.
        assert_eq!(plan(candidate(100, 500, 600, &[]), true).resume(), 0);
        assert_eq!(plan(candidate(1001, 1500, 1600, &[]), false).resume(), 0);
    }

    #[test]
    fn partial_frontiers_never_land_inside_an_image() {
        let image = [MediaSpan { start: 900, len: 200, key: ImageKey([7; 32]).into() }];
        // Common prefix inside the image: sources stop at its first row, replay before that.
        let inside = plan(candidate(1001, 1500, 1600, &image), true);
        assert_eq!(inside.lag, LaggedState { source_end: 900, replay_start: 772 });
        // Replay start inside an image rounds down to its first row.
        let after = [MediaSpan { start: 800, len: 100, key: ImageKey([7; 32]).into() }];
        let rounded = plan(candidate(1001, 1500, 1600, &after), true);
        assert_eq!(rounded.lag, LaggedState { source_end: 1000, replay_start: 800 });
        assert!(rounded.check(1001).is_ok());
    }

    #[test]
    fn metadata_is_plain_serializable_data() {
        let meta = V41Meta {
            request: RequestMark { backbone: BackboneMark { end: 513, windows: vec![(385, 513); 40] },
                engram_position: 513, engram_recent: vec![Some(5), None, Some(7)] },
            draft_end: Some(513),
        };
        let text = serde_json::to_string(&meta).unwrap();
        assert_eq!(serde_json::from_str::<V41Meta>(&text).unwrap(), meta);
    }
}
