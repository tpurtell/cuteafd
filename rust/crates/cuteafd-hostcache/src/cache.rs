//! The cache facade (packet HC-5): the six calls the engine's scheduler makes, all on its own
//! thread, none of which blocks except the three whose whole purpose is to wait a bounded
//! time (`before_device_evict`, `restore`, `prefill_hold`).
//!
//! The engine attaches a payload `P` to every store (its host-side descriptors: image keys,
//! Engram history, logits, window and compressor metadata); the cache hands it back on a hit
//! and drops it when the snapshot is evicted, so the engine keeps no side table.
//!
//! Life of a snapshot: the engine retains it → `store` plans slabs and enqueues device→host
//! copies on the store stream, returning a ticket → the engine keeps the device snapshot alive
//! while the ticket is pending → `tick` reports completion and the snapshot becomes
//! lookup-visible → the engine may evict it from the device (`before_device_evict` confirms the
//! copy is done or waits within budget) → a later device-bank miss consults `lookup` → on a hit
//! the engine reserves device memory and calls `restore`, which copies host→device into the
//! engine's destinations and waits within budget → the engine applies its reservation and
//! inserts the rebuilt snapshot into its bank.
//!
//! With `StoreMode::OnEvict`, `store` only records the snapshot and the copy is issued by
//! `before_device_evict`, which then waits within the copy budget.
use crate::config::{Config, StoreMode};
use crate::copy::{coalesce, coalesce_restore, CopyEngine, DeviceRange, Event, Stream};
use crate::metrics::{Metrics, Snapshot as MetricsSnapshot};
use crate::pool::{HostRange, Layout, SlabPool};
use crate::snapshot::{
    DevicePageId, EvictionOrder, Hit, HostSnapshot, Key, PageRef, SnapshotMeta, Snapshots, StorePlan,
};
use cuteafd_core::prefix::ReuseRule;
use crate::{SnapshotKind, PAGE_BANKS};
use serde::Serialize;
use std::collections::HashMap;

/// One device page as the engine addresses it: identity plus where its bytes are. The engine
/// keeps a page's rows in several device buffers (packed index, index scales, KV values, KV
/// scales), so a page is a list of segments whose lengths sum to the layout's page size; the
/// host slab holds them concatenated in this order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DevicePage {
    pub id: DevicePageId,
    pub segments: Vec<DeviceRange>,
}

/// A retained snapshot as it sits on the device. Tail and draft are segment lists like pages
/// (the draft is three dSpark rings of varying length); each list's bytes must fit its slab and
/// is stored concatenated. `scores` is empty when the layout's scores class is zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceSnapshot {
    pub meta: SnapshotMeta,
    pub pages: [Vec<DevicePage>; PAGE_BANKS],
    pub tail: Vec<DeviceRange>,
    pub draft: Option<Vec<DeviceRange>>,
    pub scores: Vec<DeviceRange>,
}

/// Where a restore writes: the engine's fresh destinations, in the same shape and with the same
/// segment lengths as the stored snapshot. Pages carry their new device identities so the cache
/// records them as shared after a successful restore (a later store of the same snapshot then
/// copies nothing). An empty full-draft page bank explicitly requests a target-only restore;
/// the caller must bind the drafter cold when its optional attachment cannot be admitted.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RestoreTarget {
    pub pages: [Vec<DevicePage>; PAGE_BANKS],
    pub tail: Vec<DeviceRange>,
    pub draft: Option<Vec<DeviceRange>>,
    pub scores: Vec<DeviceRange>,
}

/// Handle for a store in flight. The engine keeps the device snapshot alive (it is retained
/// anyway) until `tick` lists the ticket as completed or failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct StoreTicket(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum SkipReason {
    TooSmall,
    TooLarge,
    KindOff,
    /// A part's segments do not fit the slab class the layout gives it.
    Malformed,
    Exhausted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum StoreOutcome {
    Issued(StoreTicket),
    /// `OnEvict` mode: recorded, nothing copied yet.
    Deferred(StoreTicket),
    Skipped(SkipReason),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TickReport {
    pub completed: Vec<StoreTicket>,
    /// Exact resident identities committed by the completed tickets.
    pub committed: Vec<(StoreTicket, Key)>,
    pub failed: Vec<StoreTicket>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum EvictDecision {
    /// The host copy is complete: the device snapshot may be dropped.
    Clean,
    /// Waited `ns` within the copy budget and the copy completed.
    WaitedClean { ns: u64 },
    /// The budget ran out or the copy failed: the snapshot leaves the device uncached.
    DroppedUncached,
    /// Exceptional draining failed. The device snapshot and host slabs remain
    /// held; the caller must retain its pages/mark and retry or report failure.
    Held,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum RestoreOutcome {
    /// Every part is on the device; the engine applies its reservation.
    Done {
        ns: u64,
        bytes: u64,
    },
    /// The restore budget ran out; the engine cancels its reservation and prefills.
    TimedOut,
    Failed,
}

/// The cache. Invariants: `metrics().bytes_used <= quota`; a snapshot with a restore in flight
/// is never evicted; every ticket is reported exactly once; a `lookup` hit is a `Retention` hit
/// over the resident snapshots; when disabled every call is a no-op returning the neutral value.
pub struct HostCache<E: CopyEngine, P> {
    config: Config,
    layout: Layout,
    engine: E,
    /// `None` exactly when the cache is disabled, so no pool is ever allocated through the
    /// engine for a disabled cache.
    snapshots: Option<Snapshots>,
    metrics: Metrics,
    /// The engine's payload per resident snapshot, dropped with the snapshot.
    payloads: HashMap<Key, P>,
    /// Stores in flight, in issue order; `tick` polls the issued ones in that order.
    pending: Vec<PendingStore<P>>,
    next_ticket: u64,
}

/// Where a store in flight is in its life.
enum PendingKind {
    /// Copies are on the store stream; `event` completes when they land. `issued_ns` is the
    /// engine clock when the copies were enqueued: the store-latency histogram's start.
    Issued {
        plan: StorePlan,
        bytes: u64,
        event: Event,
        issued_ns: u64,
    },
    /// `OnEvict`: the device snapshot is recorded and nothing is planned until the engine
    /// evicts it.
    Deferred { snapshot: DeviceSnapshot },
    /// An issue failed; the plan's slabs were already released or held, and `tick` reports the
    /// ticket.
    Failed,
}

impl PendingKind {
    /// Take the deferred device snapshot, leaving `Failed`; `None` when the store is not
    /// deferred.
    fn take_deferred(&mut self) -> Option<DeviceSnapshot> {
        match std::mem::replace(self, PendingKind::Failed) {
            PendingKind::Deferred { snapshot } => Some(snapshot),
            other => {
                *self = other;
                None
            }
        }
    }
}

/// A store in flight: what it is doing and the payload that becomes resident at commit.
struct PendingStore<P> {
    ticket: StoreTicket,
    kind: PendingKind,
    payload: P,
}

/// The result of issuing a store's copies.
enum IssueOutcome {
    /// Every copy is on the stream; the event completes when they land.
    Issued(Event),
    /// A copy failed to issue and the already-issued copies drained within the budget, so the
    /// plan's slabs may be released.
    FailedDrained,
    /// A copy failed to issue and the already-issued copies did not drain within the budget, so
    /// the plan's slabs must stay held.
    FailedHeld,
}

/// The result of planning and issuing a store.
enum StoreIssue {
    /// The copies are on the store stream.
    Issued {
        plan: Box<StorePlan>,
        bytes: u64,
        event: Event,
    },
    /// A copy failed to issue; the plan's slabs were released or held by the issuer.
    Failed,
    /// Every unpinned snapshot was evicted and a class is still exhausted.
    Exhausted,
}

/// The device ranges a store copies, captured from the plan and the device snapshot so the
/// copies can be issued.
struct StoreParts {
    /// Segments per copied page, in `StorePlan::copies` order.
    pages: Vec<Vec<DeviceRange>>,
    tail: Vec<DeviceRange>,
    draft: Option<Vec<DeviceRange>>,
    scores: Vec<DeviceRange>,
}

impl StoreParts {
    /// Capture the ranges `plan` copies from `snapshot`; shared pages are not in `plan.copies`
    /// and so are not captured.
    fn capture(plan: &StorePlan, snapshot: &DeviceSnapshot) -> Self {
        Self {
            pages: plan
                .copies
                .iter()
                .map(|&(compressor, index, _, _)| {
                    snapshot.pages[compressor as usize][index as usize]
                        .segments
                        .clone()
                })
                .collect(),
            tail: snapshot.tail.clone(),
            draft: snapshot.draft.clone(),
            scores: snapshot.scores.clone(),
        }
    }

    /// The bytes this store copies to host.
    fn bytes(&self) -> u64 {
        let pages: usize = self
            .pages
            .iter()
            .map(|segments| segment_bytes(segments))
            .sum();
        let draft = self.draft.as_deref().map_or(0, segment_bytes);
        (pages + segment_bytes(&self.tail) + draft + segment_bytes(&self.scores)) as u64
    }
}

/// A restore's copy list: the host range of every part and the target segments it feeds, plus
/// the `(device identity, host page)` pairs to register as shared on success.
struct RestorePlan<'a> {
    pages: Vec<(HostRange, &'a [DeviceRange])>,
    tail: (HostRange, &'a [DeviceRange]),
    draft: Option<(HostRange, &'a [DeviceRange])>,
    scores: Option<(HostRange, &'a [DeviceRange])>,
    shared: Vec<(DevicePageId, PageRef)>,
    bytes: u64,
}

impl<'a> RestorePlan<'a> {
    /// Match `target` against the resident `snapshot`; `None` when a part's shape or size does
    /// not mirror the stored one. Invariant: every returned host range names a live slab.
    fn build(
        snapshots: &Snapshots,
        snapshot: &HostSnapshot,
        target: &'a RestoreTarget,
        layout: &Layout,
    ) -> Option<Self> {
        Self::build_parts(snapshots, snapshot, target, layout, false)
    }

    fn build_parts(snapshots: &Snapshots, snapshot: &HostSnapshot, target: &'a RestoreTarget,
        layout: &Layout, mark_only: bool) -> Option<Self> {
        let mut pages = Vec::new();
        let mut shared = Vec::new();
        let mut bytes = 0u64;
        for (compressor, list) in snapshot.pages.iter().enumerate() {
            if mark_only || (compressor == crate::DRAFT_PAGE_BANK && target.pages[compressor].is_empty()) { continue; }
            let targets = target.pages.get(compressor)?;
            if targets.len() != list.len() {
                return None;
            }
            for (page, device_page) in list.iter().zip(targets) {
                let host = snapshots.page_location(*page)?;
                let total = segment_bytes(&device_page.segments);
                let bank = device_page.id.compressor as usize;
                if (bank != compressor && !(compressor < crate::COMPRESSORS && bank == compressor + 0x80))
                    || total == 0 || total > layout.page_size(compressor) {
                    return None;
                }
                bytes += total as u64;
                pages.push((host, device_page.segments.as_slice()));
                shared.push((device_page.id, *page));
            }
        }
        let tail_total = segment_bytes(&target.tail);
        if tail_total == 0 || tail_total > layout.tail {
            return None;
        }
        bytes += tail_total as u64;
        let draft = match (if mark_only { None } else { snapshot.draft }, target.draft.as_deref()) {
            (None, None) | (None, Some([])) => None,
            (Some(slab), Some(segments)) => {
                let total = segment_bytes(segments);
                if total == 0 || total > layout.draft {
                    return None;
                }
                bytes += total as u64;
                Some((snapshots.location(slab), segments))
            }
            _ => return None,
        };
        let scores = match (if mark_only { None } else { snapshot.scores }, target.scores.as_slice()) {
            (None, []) => None,
            (Some(slab), segments) => {
                let total = segment_bytes(segments);
                if total == 0 || total > layout.scores {
                    return None;
                }
                bytes += total as u64;
                Some((snapshots.location(slab), segments))
            }
            _ => return None,
        };
        Some(Self {
            pages,
            tail: (snapshots.location(snapshot.tail), target.tail.as_slice()),
            draft,
            scores,
            shared,
            bytes,
        })
    }
}

impl<E: CopyEngine, P> HostCache<E, P> {
    /// Validates `config`, allocates the pinned pool through `engine` (nothing when disabled).
    /// V4.1's reuse rule with least-recent eviction, the order the daemon and the generic
    /// engine serve with.
    pub fn new(config: Config, layout: Layout, engine: E) -> anyhow::Result<Self> {
        Self::with_rule(config, layout, engine, ReuseRule::V41, EvictionOrder::LeastRecent)
    }

    /// A cache whose lookups follow a generic family's reuse `rule` and whose eviction follows
    /// `order`; [`HostCache::new`] is `ReuseRule::V41` with `EvictionOrder::LeastRecent`.
    pub fn with_rule(config: Config, layout: Layout, engine: E, rule: ReuseRule, order: EvictionOrder)
        -> anyhow::Result<Self> {
        config.validate()?;
        let mut engine = engine;
        let snapshots = if config.enabled() {
            let chunk_bytes = usize::try_from(config.chunk_bytes)
                .map_err(|_| anyhow::anyhow!("chunk_bytes exceeds usize"))?;
            let pool = SlabPool::new(config.bytes, chunk_bytes, layout, &mut engine)?;
            Some(Snapshots::with_rule(pool, rule, order))
        } else {
            None
        };
        let mut cache = Self {
            config,
            layout,
            engine,
            snapshots,
            metrics: Metrics::default(),
            payloads: HashMap::new(),
            pending: Vec::new(),
            next_ticket: 0,
        };
        cache.refresh_gauges();
        Ok(cache)
    }

    pub fn enabled(&self) -> bool {
        self.config.enabled()
    }

    /// Plan and issue the store; `payload` travels with the snapshot until it is evicted. Under
    /// `OnEvict` nothing is planned or copied here: the snapshot is recorded and the copy is
    /// issued by `before_device_evict`.
    pub fn store(&mut self, snapshot: &DeviceSnapshot, payload: P) -> StoreOutcome {
        if !self.enabled() {
            return StoreOutcome::Skipped(SkipReason::KindOff);
        }
        if let Some(reason) = self.skip_reason(snapshot) {
            self.metrics.get_mut().stores_skipped += 1;
            return StoreOutcome::Skipped(reason);
        }
        let kind = match self.config.store {
            StoreMode::OnRetain => match self.plan_and_issue(snapshot) {
                StoreIssue::Issued { plan, bytes, event } => PendingKind::Issued {
                    plan: *plan,
                    bytes,
                    event,
                    issued_ns: self.engine.now_ns(),
                },
                StoreIssue::Failed => PendingKind::Failed,
                StoreIssue::Exhausted => {
                    self.metrics.get_mut().stores_skipped += 1;
                    return StoreOutcome::Skipped(SkipReason::Exhausted);
                }
            },
            StoreMode::OnEvict => PendingKind::Deferred {
                snapshot: snapshot.clone(),
            },
        };
        let ticket = StoreTicket(self.next_ticket);
        self.next_ticket += 1;
        self.metrics.get_mut().stores_issued += 1;
        let deferred = matches!(kind, PendingKind::Deferred { .. });
        self.pending.push(PendingStore {
            ticket,
            kind,
            payload,
        });
        self.refresh_gauges();
        if deferred {
            StoreOutcome::Deferred(ticket)
        } else {
            StoreOutcome::Issued(ticket)
        }
    }

    /// The engine's payload for a resident snapshot.
    pub fn payload(&self, key: Key) -> Option<&P> {
        self.payloads.get(&key)
    }

    /// Poll the store stream; completed stores become lookup-visible.
    pub fn tick(&mut self) -> TickReport {
        let mut report = TickReport::default();
        if !self.enabled() {
            return report;
        }
        let now = self.engine.now_ns();
        let mut index = 0;
        while index < self.pending.len() {
            let event = match &self.pending[index].kind {
                PendingKind::Deferred { .. } => {
                    index += 1;
                    continue;
                }
                PendingKind::Failed => {
                    if self.engine.release_barrier(Stream::Store).is_err() { break; }
                    let pending = self.pending.remove(index);
                    self.metrics.get_mut().stores_failed += 1;
                    report.failed.push(pending.ticket);
                    continue;
                }
                PendingKind::Issued { event, .. } => *event,
            };
            match self.engine.completed(event) {
                Ok(true) => {
                    let pending = self.pending.remove(index);
                    let ticket = pending.ticket;
                    if let Some(key) = self.commit_pending(pending, now) {
                        report.committed.push((ticket, key));
                    }
                    report.completed.push(ticket);
                }
                Ok(false) => break,
                Err(_) => {
                    if self.engine.release_barrier(Stream::Store).is_err() { break; }
                    let pending = self.pending.remove(index);
                    let ticket = pending.ticket;
                    self.release_pending(pending);
                    self.metrics.get_mut().stores_failed += 1;
                    report.failed.push(ticket);
                }
            }
        }
        self.refresh_gauges();
        report
    }

    /// The engine is about to drop a device snapshot. With a pending ticket, wait within the
    /// copy budget (under `OnEvict`, plan and issue the copy first from the device snapshot
    /// recorded at store time, whose ranges are still valid because the snapshot is still
    /// alive); without a ticket, `Clean`.
    pub fn before_device_evict(&mut self, ticket: Option<StoreTicket>) -> EvictDecision {
        self.before_device_evict_with_key(ticket).0
    }

    /// Finish a pending write and return its exact resident identity, if committed.
    pub fn before_device_evict_with_key(&mut self, ticket: Option<StoreTicket>) -> (EvictDecision, Option<Key>) {
        let mut key = None;
        let decision = self.decide_evict(ticket, &mut key);
        self.refresh_gauges();
        (decision, key)
    }

    /// The body of [`before_device_evict`](Self::before_device_evict), so the gauges are
    /// refreshed on every exit.
    fn decide_evict(&mut self, ticket: Option<StoreTicket>, key: &mut Option<Key>) -> EvictDecision {
        if !self.enabled() {
            return EvictDecision::Clean;
        }
        // One counted device eviction per consultation, before any budget wait and whatever
        // the outcome; the disabled case above never reaches here.
        self.metrics.get_mut().device_evictions += 1;
        let Some(ticket) = ticket else {
            return EvictDecision::Clean;
        };
        let Some(index) = self
            .pending
            .iter()
            .position(|pending| pending.ticket == ticket)
        else {
            return EvictDecision::Clean;
        };
        if let Some(snapshot) = self.pending[index].kind.take_deferred() {
            let mut pending = self.pending.remove(index);
            match self.plan_and_issue(&snapshot) {
                StoreIssue::Issued { plan, bytes, event } => {
                    pending.kind = PendingKind::Issued {
                        plan: *plan,
                        bytes,
                        event,
                        issued_ns: self.engine.now_ns(),
                    };
                    self.pending.insert(index, pending);
                }
                StoreIssue::Failed | StoreIssue::Exhausted => {
                    if self.engine.release_barrier(Stream::Store).is_err() {
                        self.pending.insert(index, pending);
                        return EvictDecision::Held;
                    }
                    self.metrics.get_mut().stores_failed += 1;
                    self.metrics.get_mut().evict_drops_uncached += 1;
                    return EvictDecision::DroppedUncached;
                }
            }
        }
        let event = match &self.pending[index].kind {
            PendingKind::Issued { event, .. } => *event,
            _ => {
                if self.engine.release_barrier(Stream::Store).is_err() { return EvictDecision::Held; }
                let pending = self.pending.remove(index);
                self.release_pending(pending);
                self.metrics.get_mut().stores_failed += 1;
                self.metrics.get_mut().evict_drops_uncached += 1;
                return EvictDecision::DroppedUncached;
            }
        };
        let start = self.engine.now_ns();
        let completed = self
            .engine
            .wait(event, self.config.copy_budget_ns)
            .unwrap_or(false);
        let ns = self.engine.now_ns().saturating_sub(start);
        let metrics = self.metrics.get_mut();
        metrics.evict_waits += 1;
        metrics.evict_wait_ns += ns;
        if completed {
            let pending = self.pending.remove(index);
            let now = self.engine.now_ns();
            *key = self.commit_pending(pending, now);
            EvictDecision::WaitedClean { ns }
        } else {
            // The bounded wait expired; correctness still requires completion
            // before either host slabs or the caller's device pages are reused.
            if self.engine.release_barrier(Stream::Store).is_err() { return EvictDecision::Held; }
            let pending = self.pending.remove(index);
            self.release_pending(pending);
            self.metrics.get_mut().stores_failed += 1;
            self.metrics.get_mut().evict_drops_uncached += 1;
            EvictDecision::DroppedUncached
        }
    }

    /// The scheduler's prefill pacing guard (packet HC-9), called at each prefill chunk
    /// boundary — the full contract lives here; the daemon's wrappers delegate. If the
    /// oldest in-flight store copy has been outstanding longer than `store_pace_ns`, wait
    /// on that store's event for at most `store_pace_ns` more, then return regardless.
    /// Invariants: never blocks longer than `store_pace_ns` — on engines that poll between
    /// probes, a timed-out wait is bounded by the budget plus at most one poll quantum,
    /// since the inter-poll sleep is clipped to the remaining budget; with the knob at 0,
    /// or when the cache is disabled, or when nothing is in flight, the pacing hold itself
    /// is a no-op — no clock read, no hold metric moves (the daemon's wrapper observes the
    /// store stream with `tick` on every prefill chunk regardless of the knob, so store
    /// metrics may still move at that level); it never fails the request (a wait error is
    /// counted as a timeout); a store that completes during the hold is left for `tick` to
    /// commit, so its latency — which already includes the hold time — lands in the
    /// store-latency histogram exactly once and is never double-counted here.
    pub fn prefill_hold(&mut self) -> anyhow::Result<()> {
        let pace = self.config.store_pace_ns;
        if !self.enabled() || pace == 0 {
            return Ok(());
        }
        let oldest = self
            .pending
            .iter()
            .filter_map(|pending| match &pending.kind {
                PendingKind::Issued {
                    event, issued_ns, ..
                } => Some((*event, *issued_ns)),
                _ => None,
            })
            .min_by_key(|&(_, issued_ns)| issued_ns);
        let Some((event, issued_ns)) = oldest else {
            return Ok(());
        };
        // One clock read serves both the age check and the hold's start.
        let now = self.engine.now_ns();
        if now.saturating_sub(issued_ns) <= pace {
            return Ok(());
        }
        let completed = self.engine.wait(event, pace).unwrap_or(false);
        let ns = self.engine.now_ns().saturating_sub(now);
        self.metrics.record_prefill_hold(ns, !completed);
        Ok(())
    }

    /// Non-mutating lookup. Pending stores are not yet visible.
    pub fn peek(&self, tokens: &[u32]) -> Option<Hit> {
        self.snapshots.as_ref()?.peek(tokens)
    }

    /// Resident snapshots with their engine descriptors. This cold-path iterator lets the
    /// engine apply full media identity verification before allocating or restoring anything.
    pub fn resident_payloads(&self) -> impl Iterator<Item = (&SnapshotMeta, crate::snapshot::Key, &P)> {
        self.payloads.iter().filter_map(|(&key, payload)| {
            Some((&self.snapshots.as_ref()?.get(key)?.meta, key, payload))
        })
    }

    /// The key-space tokens of a resident snapshot (the sequence its radix entry is keyed by).
    pub fn snapshot_tokens(&self, key: Key) -> Option<&[u32]> {
        self.snapshots
            .as_ref()
            .and_then(|snapshots| snapshots.get(key))
            .map(|snapshot| snapshot.meta.tokens.as_slice())
    }

    pub fn device_page_freed(&mut self, id: DevicePageId) {
        if let Some(snapshots) = self.snapshots.as_mut() {
            snapshots.device_page_freed(id);
        }
    }

    /// The engine's reuse rule over host snapshots; `None` falls through to prefill.
    pub fn lookup(&mut self, tokens: &[u32]) -> Option<Hit> {
        if !self.enabled() {
            return None;
        }
        self.metrics.get_mut().lookups += 1;
        let now = self.engine.now_ns();
        let hit = self
            .snapshots
            .as_mut()
            .and_then(|snapshots| snapshots.lookup(tokens, now));
        if hit.is_some() {
            self.metrics.get_mut().host_hits += 1;
        }
        hit
    }

    /// Account for an engine-selected, fully verified media hit (or a miss) without rerunning
    /// radix selection. Refreshes the timestamp used by LeastRecent eviction, unlike peek.
    pub fn lookup_verified(&mut self, key: Option<Key>) -> bool {
        if !self.enabled() { return false; }
        self.metrics.get_mut().lookups += 1;
        let now = self.engine.now_ns();
        let hit = key.is_some_and(|key| self.snapshots.as_mut().is_some_and(|s| s.touch(key, now)));
        if hit { self.metrics.get_mut().host_hits += 1; }
        hit
    }

    /// Pin, copy every part into `target`, wait within the restore budget, unpin. On `TimedOut`
    /// the copies may still land later, so the engine must not reuse `target` until it has
    /// synchronized the restore stream itself; the host snapshot stays resident either way.
    pub fn restore(&mut self, key: Key, target: &RestoreTarget) -> RestoreOutcome {
        self.restore_parts(key, target, false)
    }

    /// Copy only positional state when immutable KV pages are still device resident.
    ///
    /// The caller already holds the key (the device entry's host mark), so no lookup runs
    /// on this path. A completed mark restore is still the host tier serving a request from
    /// this snapshot: it refreshes the clock `LeastRecent` evicts by and counts a host hit,
    /// as `lookup_verified` does for a promotion. A restore that does not complete does
    /// neither. The full `restore` is unchanged: its lookup already did both.
    pub fn restore_mark(&mut self, key: Key, tail: Vec<DeviceRange>) -> RestoreOutcome {
        let target = RestoreTarget { pages: Default::default(), tail, draft: None, scores: Vec::new() };
        let outcome = self.restore_parts(key, &target, true);
        if matches!(outcome, RestoreOutcome::Done { .. }) {
            let now = self.engine.now_ns();
            if self.snapshots.as_mut().is_some_and(|snapshots| snapshots.touch(key, now)) {
                self.metrics.get_mut().host_hits += 1;
            }
        }
        outcome
    }

    fn restore_parts(&mut self, key: Key, target: &RestoreTarget, mark_only: bool) -> RestoreOutcome {
        if !self.enabled() {
            return RestoreOutcome::Failed;
        }
        let plan = {
            let Some(snapshots) = self.snapshots.as_ref() else {
                return RestoreOutcome::Failed;
            };
            let Some(snapshot) = snapshots.get(key) else {
                return RestoreOutcome::Failed;
            };
            match RestorePlan::build_parts(snapshots, snapshot, target, &self.layout, mark_only) {
                Some(plan) => plan,
                None => {
                    self.metrics.get_mut().restore_failures += 1;
                    return RestoreOutcome::Failed;
                }
            }
        };
        if let Some(snapshots) = self.snapshots.as_mut() {
            snapshots.pin(key);
        }
        let start = self.engine.now_ns();
        let event = match self.issue_restore(&plan) {
            IssueOutcome::Issued(event) => event,
            IssueOutcome::FailedDrained => {
                self.unpin(key);
                self.metrics.get_mut().restore_failures += 1;
                return RestoreOutcome::Failed;
            }
            IssueOutcome::FailedHeld => {
                if self.engine.release_barrier(Stream::Restore).is_ok() { self.unpin(key); }
                self.metrics.get_mut().restore_timeouts += 1;
                return RestoreOutcome::TimedOut;
            }
        };
        let completed = self
            .engine
            .wait(event, self.config.restore_budget_ns)
            .unwrap_or(false);
        let ns = self.engine.now_ns().saturating_sub(start);
        // A timed-out restore still reads host slabs. Keep its pin if the
        // exceptional barrier fails; callers must also retain their targets.
        if completed || self.engine.release_barrier(Stream::Restore).is_ok() { self.unpin(key); }
        self.refresh_gauges();
        if completed {
            if let Some(snapshots) = self.snapshots.as_mut() {
                for &(id, page) in &plan.shared {
                    snapshots.register_device_page(id, page);
                }
            }
            self.metrics.record_restore(ns, plan.bytes);
            RestoreOutcome::Done {
                ns,
                bytes: plan.bytes,
            }
        } else {
            self.metrics.get_mut().restore_timeouts += 1;
            RestoreOutcome::TimedOut
        }
    }

    pub fn metrics(&self) -> MetricsSnapshot {
        self.metrics.snapshot()
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Read retained copy-provider storage without polling or mutating the cache.
    pub fn engine(&self) -> &E {
        &self.engine
    }

    pub fn engine_mut(&mut self) -> &mut E {
        &mut self.engine
    }

    #[doc(hidden)]
    pub fn metrics_mut(&mut self) -> &mut Metrics {
        &mut self.metrics
    }

    /// Add one pin to a resident snapshot: it is never evicted until [`HostCache::release_pin`].
    /// The engine pins a snapshot that holds the only copy of a live device entry's positional
    /// mark, so host pressure cannot orphan the entry's pages. An unknown key is a no-op.
    pub fn pin(&mut self, key: Key) {
        if let Some(snapshots) = self.snapshots.as_mut() {
            snapshots.pin(key);
        }
        self.refresh_gauges();
    }

    /// Drop a pin taken with [`HostCache::pin`]; an unknown key is a no-op.
    pub fn release_pin(&mut self, key: Key) {
        self.unpin(key);
        self.refresh_gauges();
    }

    /// Remove a resident snapshot regardless of pins. Test support only: it violates the pin
    /// contract on purpose, to exercise the engine's dead-mark path.
    #[doc(hidden)]
    pub fn remove_for_test(&mut self, key: Key) -> bool {
        self.payloads.remove(&key);
        let removed = self.snapshots.as_mut().is_some_and(|s| s.remove(key));
        self.refresh_gauges();
        removed
    }

    /// Whether `key` is resident and pinned (test support and invariant checks).
    pub fn pinned(&self, key: Key) -> bool {
        self.snapshots.as_ref().and_then(|s| s.get(key)).is_some_and(|s| s.pins > 0)
    }

    /// Only the engine retention pin remains; an outstanding restore DMA can hold another.
    pub fn singly_pinned(&self, key: Key) -> bool {
        self.snapshots.as_ref().and_then(|s| s.get(key)).is_some_and(|s| s.pins == 1)
    }

    /// Can the snapshot's slabs ever fit the quota, even with an empty pool?
    pub fn fits_empty(&self, snapshot: &DeviceSnapshot) -> bool {
        use crate::pool::Class;
        let chunk = self.config.chunk_bytes;
        if chunk == 0 { return false; }
        let mut counts = [0u64; Class::ALL.len()];
        for (bank, pages) in snapshot.pages.iter().enumerate() {
            counts[Layout::page_class(bank).index()] += pages.len() as u64;
        }
        counts[Class::Tail.index()] = 1;
        counts[Class::Draft.index()] = u64::from(snapshot.meta.has_draft && self.layout.draft > 0);
        counts[Class::Scores.index()] = u64::from(self.layout.scores > 0);
        Class::ALL.into_iter().zip(counts).try_fold(0u64, |total, (class, count)| {
            if count == 0 { return Some(total); }
            let per_chunk = chunk.checked_div(self.layout.size(class) as u64)?;
            if per_chunk == 0 { return None; }
            total.checked_add(count.div_ceil(per_chunk))
        }).is_some_and(|chunks| chunks <= self.config.bytes / chunk)
    }

    /// Why `snapshot` is not cached, or `None` when it is.
    fn skip_reason(&self, snapshot: &DeviceSnapshot) -> Option<SkipReason> {
        let tokens = snapshot.meta.tokens.len();
        if tokens < self.config.min_tokens as usize {
            return Some(SkipReason::TooSmall);
        }
        if tokens > self.config.max_tokens as usize {
            return Some(SkipReason::TooLarge);
        }
        if !self.kind_enabled(snapshot.meta.kind) {
            return Some(SkipReason::KindOff);
        }
        if !self.parts_fit(snapshot) {
            return Some(SkipReason::Malformed);
        }
        None
    }

    /// Whether the configured kinds include `kind`.
    fn kind_enabled(&self, kind: SnapshotKind) -> bool {
        match kind {
            SnapshotKind::Prompt => self.config.kinds.prompt,
            SnapshotKind::Turn => self.config.kinds.turn,
        }
    }

    /// Whether every part's segments fit the slab class the layout gives it and the snapshot's
    /// shape is consistent: draft present exactly when the snapshot has it and the class is
    /// enabled, tail non-empty, scores non-empty when its class is enabled.
    fn parts_fit(&self, snapshot: &DeviceSnapshot) -> bool {
        let bytes = |segments: &[DeviceRange]| segment_bytes(segments);
        let draft_expected = snapshot.meta.has_draft && self.layout.draft > 0;
        let draft_ok = match snapshot.draft.as_deref() {
            Some(draft) => draft_expected && bytes(draft) > 0 && bytes(draft) <= self.layout.draft,
            None => !draft_expected,
        };
        let scores_ok = if self.layout.scores > 0 {
            bytes(&snapshot.scores) > 0 && bytes(&snapshot.scores) <= self.layout.scores
        } else {
            snapshot.scores.is_empty()
        };
        snapshot.pages.iter().enumerate().all(|(bank, list)| {
            list.iter()
                .all(|page| (page.id.compressor as usize == bank
                    || (bank < crate::COMPRESSORS && page.id.compressor as usize == bank + 0x80))
                    && bytes(&page.segments) > 0 && bytes(&page.segments) <= self.layout.page_size(bank))
        }) && bytes(&snapshot.tail) > 0
            && bytes(&snapshot.tail) <= self.layout.tail
            && draft_ok
            && scores_ok
    }

    /// Issue every copy `plan` owes on the store stream. On a mid-issue failure, drain the
    /// copies already issued within the copy budget before returning; the caller must keep the
    /// plan's slabs held when the drain did not complete.
    fn issue_store(&mut self, plan: &StorePlan, parts: &StoreParts) -> IssueOutcome {
        let Some(snapshots) = self.snapshots.as_ref() else {
            return IssueOutcome::FailedDrained;
        };
        match issue_store_copies(&mut self.engine, snapshots, plan, parts) {
            Ok(()) => match self.engine.record(Stream::Store) {
                Ok(event) => IssueOutcome::Issued(event),
                Err(_) => if self.engine.release_barrier(Stream::Store).is_ok() {
                    IssueOutcome::FailedDrained
                } else { IssueOutcome::FailedHeld },
            },
            Err(_) => {
                if self.drain(Stream::Store) {
                    IssueOutcome::FailedDrained
                } else {
                    IssueOutcome::FailedHeld
                }
            }
        }
    }

    /// Issue every copy a restore owes on the restore stream. On a mid-issue failure, drain the
    /// copies already issued within the copy budget; a failed drain is a timeout, because the
    /// target may still be written.
    fn issue_restore(&mut self, plan: &RestorePlan) -> IssueOutcome {
        match issue_restore_copies(&mut self.engine, plan) {
            Ok(()) => match self.engine.record(Stream::Restore) {
                Ok(event) => IssueOutcome::Issued(event),
                Err(_) => if self.engine.release_barrier(Stream::Restore).is_ok() {
                    IssueOutcome::FailedDrained
                } else { IssueOutcome::FailedHeld },
            },
            Err(_) => {
                if self.drain(Stream::Restore) {
                    IssueOutcome::FailedDrained
                } else {
                    IssueOutcome::FailedHeld
                }
            }
        }
    }

    /// Normal event waits remain bounded. Failed/expired cleanup waits use
    /// the explicit release barrier before returning either kind of storage.
    fn drain(&mut self, stream: Stream) -> bool {
        let completed = match self.engine.record(stream) {
            Ok(event) => self.engine.wait(event, self.config.copy_budget_ns).unwrap_or(false),
            Err(_) => false,
        };
        completed || self.engine.release_barrier(stream).is_ok()
    }

    /// Make a completed store resident and drop the payload it replaced; admission already
    /// reserved its slabs. Invariant: the payload is resident exactly while its snapshot is. Records the
    /// store-latency histogram entry exactly once, here, wherever the completion was observed
    /// (`tick` or the evict path's bounded wait).
    fn commit_pending(&mut self, pending: PendingStore<P>, now: u64) -> Option<Key> {
        let PendingStore { kind, payload, .. } = pending;
        let PendingKind::Issued {
            plan,
            bytes,
            issued_ns,
            ..
        } = kind
        else {
            return None;
        };
        let copied = plan.copies.len() as u64;
        let total: u64 = plan.pages.iter().map(|list| list.len() as u64).sum();
        let before = self.snapshots.as_ref().map_or(0, Snapshots::len);
        let key = match self.snapshots.as_mut() {
            Some(snapshots) => snapshots.commit_store(plan, now),
            None => return None,
        };
        let after = self.snapshots.as_ref().map_or(0, Snapshots::len);
        let replaced = before + 1 - after;
        if replaced > 0 {
            let Self {
                payloads,
                snapshots,
                ..
            } = self;
            if let Some(snapshots) = snapshots.as_ref() {
                payloads.retain(|key, _| snapshots.get(*key).is_some());
            }
        }
        self.payloads.insert(key, payload);
        let metrics = self.metrics.get_mut();
        metrics.stores_completed += 1;
        metrics.stores_replaced += replaced as u64;
        metrics.pages_copied += copied;
        metrics.pages_shared += total - copied;
        metrics.store_bytes += bytes;
        self.metrics.record_store(now.saturating_sub(issued_ns));
        Some(key)
    }

    /// Release a pending store's plan, if it still holds one.
    fn release_pending(&mut self, pending: PendingStore<P>) {
        if let PendingKind::Issued { plan, .. } = pending.kind {
            self.release_plan(plan);
        }
    }

    /// Return a plan's slabs to the pool; its copies have drained or never landed.
    fn release_plan(&mut self, plan: StorePlan) {
        if let Some(snapshots) = self.snapshots.as_mut() {
            snapshots.abort_store(plan);
        }
    }

    /// Drop a plan whose copies may still be in flight: its slabs stay held so no later store
    /// can reuse memory a live copy writes into. Counted so the leak is visible.
    fn hold_plan(&mut self, plan: StorePlan) {
        drop(plan);
        self.metrics.get_mut().store_drain_timeouts += 1;
    }

    /// Synchronous host work at store planning (deferred until device eviction in OnEvict mode),
    /// never at copy completion.
    fn plan_store_evicting(&mut self, meta: &SnapshotMeta,
        device_pages: &[Vec<DevicePageId>; PAGE_BANKS]) -> Option<StorePlan> {
        let snapshots = self.snapshots.as_mut()?;
        let started = std::time::Instant::now();
        let (victims, freed) = snapshots.make_room(meta, device_pages, self.config.eviction_headroom);
        if !victims.is_empty() {
            for key in &victims { self.payloads.remove(key); }
            let metrics = self.metrics.get_mut();
            metrics.host_evictions += victims.len() as u64;
            metrics.host_evicted_bytes += freed;
            let eviction_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
            metrics.host_eviction_ns += eviction_ns;
            tracing::debug!(target: "cuteafd::hostcache", victims = victims.len(), freed_bytes = freed,
                host_evictions = metrics.host_evictions, host_evicted_bytes = metrics.host_evicted_bytes,
                eviction_ns, "host cache store-pressure eviction");
        }
        snapshots.plan_store(meta.clone(), device_pages).ok()
    }

    /// Plan and issue a store's copies, evicting unpinned snapshots if a class is exhausted.
    fn plan_and_issue(&mut self, snapshot: &DeviceSnapshot) -> StoreIssue {
        let device_pages: [Vec<DevicePageId>; PAGE_BANKS] =
            std::array::from_fn(|c| snapshot.pages[c].iter().map(|page| page.id).collect());
        let Some(plan) = self.plan_store_evicting(&snapshot.meta, &device_pages) else {
            return StoreIssue::Exhausted;
        };
        let parts = StoreParts::capture(&plan, snapshot);
        let bytes = parts.bytes();
        match self.issue_store(&plan, &parts) {
            IssueOutcome::Issued(event) => StoreIssue::Issued {
                plan: Box::new(plan),
                bytes,
                event,
            },
            IssueOutcome::FailedDrained => {
                self.release_plan(plan);
                StoreIssue::Failed
            }
            IssueOutcome::FailedHeld => {
                self.hold_plan(plan);
                StoreIssue::Failed
            }
        }
    }

    /// Drop one pin from `key`; an unknown key is a no-op.
    fn unpin(&mut self, key: Key) {
        if let Some(snapshots) = self.snapshots.as_mut() {
            snapshots.unpin(key);
        }
    }

    /// Publish the current pool state as the metrics gauges.
    fn refresh_gauges(&mut self) {
        let (resident, bytes) = match self.snapshots.as_ref() {
            Some(snapshots) => (snapshots.len() as u64, snapshots.bytes_used()),
            None => (0, 0),
        };
        let quota = self.config.bytes;
        let submissions = self.engine.submission_count();
        let metrics = self.metrics.get_mut();
        metrics.resident_snapshots = resident;
        metrics.bytes_used = bytes;
        metrics.quota_bytes = quota;
        metrics.copy_submissions = submissions;
    }
}

/// The bytes of a segment list.
fn segment_bytes(segments: &[DeviceRange]) -> usize {
    segments.iter().map(|segment| segment.bytes).sum()
}

/// `host` sliced per segment: the window each segment of `segments` occupies when they are
/// concatenated at `host`, in order.
fn host_windows(host: HostRange, segments: &[DeviceRange]) -> impl Iterator<Item = HostRange> + '_ {
    let mut offset = host.offset;
    segments.iter().map(move |segment| {
        let window = HostRange {
            chunk: host.chunk,
            offset,
            bytes: segment.bytes,
        };
        offset += segment.bytes;
        window
    })
}

/// Append a store's copies for `segments` to `pairs`: each segment with the window it occupies
/// at `host`, in order.
fn append_store_pairs(
    pairs: &mut Vec<(DeviceRange, HostRange)>,
    segments: &[DeviceRange],
    host: HostRange,
) {
    pairs.extend(segments.iter().copied().zip(host_windows(host, segments)));
}

/// Append a restore's copies for `segments` to `pairs`: each segment's window at `host` with
/// the segment, in order.
fn append_restore_pairs(
    pairs: &mut Vec<(HostRange, DeviceRange)>,
    segments: &[DeviceRange],
    host: HostRange,
) {
    pairs.extend(host_windows(host, segments).zip(segments.iter().copied()));
}

/// Issue every copy `plan` owes on the store stream as one coalesced batch, concatenated in
/// order: adjacent copies merge (see [`coalesce`]), so a snapshot allocated from a fresh pool
/// costs a handful of submissions instead of thousands.
fn issue_store_copies<E: CopyEngine>(
    engine: &mut E,
    snapshots: &Snapshots,
    plan: &StorePlan,
    parts: &StoreParts,
) -> anyhow::Result<()> {
    let mut pairs = Vec::new();
    for (index, &(_, _, _, slab)) in plan.copies.iter().enumerate() {
        append_store_pairs(&mut pairs, &parts.pages[index], snapshots.location(slab));
    }
    append_store_pairs(&mut pairs, &parts.tail, snapshots.location(plan.tail));
    if let (Some(slab), Some(segments)) = (plan.draft, parts.draft.as_deref()) {
        append_store_pairs(&mut pairs, segments, snapshots.location(slab));
    }
    if let (Some(slab), segments) = (plan.scores, parts.scores.as_slice()) {
        append_store_pairs(&mut pairs, segments, snapshots.location(slab));
    }
    let merged = coalesce(&pairs);
    engine.d2h_many(Stream::Store, &merged)
}

/// Issue every copy a restore owes on the restore stream as one coalesced batch,
/// concatenated in order (see [`issue_store_copies`]).
fn issue_restore_copies<E: CopyEngine>(engine: &mut E, plan: &RestorePlan) -> anyhow::Result<()> {
    let mut pairs = Vec::new();
    for (host, segments) in &plan.pages {
        append_restore_pairs(&mut pairs, segments, *host);
    }
    append_restore_pairs(&mut pairs, plan.tail.1, plan.tail.0);
    if let Some((host, segments)) = plan.draft {
        append_restore_pairs(&mut pairs, segments, host);
    }
    if let Some((host, segments)) = plan.scores {
        append_restore_pairs(&mut pairs, segments, host);
    }
    let merged = coalesce_restore(&pairs);
    engine.h2d_many(Stream::Restore, &merged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::testing::layout;
    use crate::snapshot::testing::{id, meta, pages, snapshots};
    use crate::SnapshotKind;

    #[test]
    fn segment_bytes_sums_lengths() {
        assert_eq!(segment_bytes(&[]), 0);
        assert_eq!(
            segment_bytes(&[
                DeviceRange { addr: 0, bytes: 3 },
                DeviceRange { addr: 3, bytes: 4 },
            ]),
            7
        );
    }

    #[test]
    fn target_only_restore_omits_optional_draft_pages_without_registering_them() {
        let layout = layout();
        let mut store = crate::snapshot::testing::snapshots_with_layout(1 << 30, layout);
        let mut stored = pages(&[id(1)]);
        let draft_id = DevicePageId { compressor: crate::DRAFT_PAGE_BANK as u8,
            page: 7, generation: 3 };
        stored[crate::DRAFT_PAGE_BANK].push(draft_id);
        let plan = store.plan_store(meta(SnapshotKind::Turn, &[1, 2], false), &stored).unwrap();
        let key = store.commit_store(plan, 0);
        let snapshot = store.get(key).unwrap();
        let mut target = RestoreTarget {
            pages: Default::default(),
            tail: vec![DeviceRange { addr: 0, bytes: layout.tail }],
            draft: None,
            scores: vec![DeviceRange { addr: 0, bytes: layout.scores }],
        };
        target.pages[0].push(DevicePage { id: id(9),
            segments: vec![DeviceRange { addr: 0, bytes: layout.page }] });
        let cold = RestorePlan::build(&store, snapshot, &target, &layout).unwrap();
        assert_eq!(cold.pages.len(), 1);
        assert_eq!(cold.shared.len(), 1);
        assert_eq!(cold.bytes, (layout.page + layout.tail + layout.scores) as u64);
        let cold_bytes = cold.bytes;
        drop(cold);
        target.pages[crate::DRAFT_PAGE_BANK].push(DevicePage { id: draft_id,
            segments: vec![DeviceRange { addr: 0, bytes: layout.draft_page }] });
        let warm = RestorePlan::build(&store, snapshot, &target, &layout).unwrap();
        assert_eq!(warm.pages.len(), 2);
        assert_eq!(warm.bytes, cold_bytes + layout.draft_page as u64);
        target.pages[crate::DRAFT_PAGE_BANK][0].id.compressor = 0x80;
        assert!(RestorePlan::build(&store, snapshot, &target, &layout).is_none());
    }

    #[test]
    fn restore_plan_matches_the_stored_shape() {
        let mut store = snapshots(1 << 30);
        let plan = store
            .plan_store(meta(SnapshotKind::Turn, &[1, 2], false), &pages(&[id(1)]))
            .expect("plan");
        let key = store.commit_store(plan, 0);
        let snapshot = store.get(key).expect("snapshot");
        let layout = layout();
        let host = store.page_location(snapshot.pages[0][0]).expect("page");
        let target = RestoreTarget {
            pages: std::array::from_fn(|compressor| {
                if compressor == 0 {
                    vec![DevicePage {
                        id: id(9),
                        segments: vec![DeviceRange {
                            addr: 0,
                            bytes: layout.page,
                        }],
                    }]
                } else {
                    Vec::new()
                }
            }),
            tail: vec![DeviceRange {
                addr: 0,
                bytes: layout.tail,
            }],
            draft: None,
            scores: vec![DeviceRange {
                addr: 0,
                bytes: layout.scores,
            }],
        };
        let plan = RestorePlan::build(&store, snapshot, &target, &layout).expect("plan");
        assert_eq!(plan.pages.len(), 1);
        assert_eq!(plan.pages[0].0, host);
        assert_eq!(
            plan.bytes,
            (layout.page + layout.tail + layout.scores) as u64
        );
        assert_eq!(plan.shared, vec![(id(9), snapshot.pages[0][0])]);

        let mut wrong_bank = target.clone();
        wrong_bank.pages[0][0].id.compressor = crate::DRAFT_PAGE_BANK as u8;
        assert!(RestorePlan::build(&store, snapshot, &wrong_bank, &layout).is_none());
        // A target with the wrong page count is rejected.
        let mut short = target.clone();
        short.pages[0].clear();
        assert!(RestorePlan::build(&store, snapshot, &short, &layout).is_none());
        // A target whose page does not fit the class is rejected.
        let mut oversized = target.clone();
        oversized.pages[0][0].segments = vec![DeviceRange {
            addr: 0,
            bytes: layout.page + 1,
        }];
        assert!(RestorePlan::build(&store, snapshot, &oversized, &layout).is_none());
        // A part the stored snapshot has must receive bytes: a zero-length page, tail or scores
        // list is rejected so a `Done` never registers a page that received nothing.
        let mut empty_page = target.clone();
        empty_page.pages[0][0].segments.clear();
        assert!(RestorePlan::build(&store, snapshot, &empty_page, &layout).is_none());
        let mut empty_tail = target.clone();
        empty_tail.tail.clear();
        assert!(RestorePlan::build(&store, snapshot, &empty_tail, &layout).is_none());
        let mut empty_scores = target.clone();
        empty_scores.scores.clear();
        assert!(RestorePlan::build(&store, snapshot, &empty_scores, &layout).is_none());
    }
}
