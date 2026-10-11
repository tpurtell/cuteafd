//! The generic prefix cache (PLAN.md "Prefix cache for every family"): one per generic engine,
//! owning the device page pool, the mark arena, the retained snapshots and the optional host
//! tier. Every call runs on the scheduler thread. A family whose marks live in pool pages
//! ([`MarkStore::Pool`]) has no arena: a capture or promotion takes the mark's pages from the
//! pool beside the snapshot's rows, evicting least recently used snapshots for both (to the host
//! tier when it is on), and an eviction releases both.
//!
//! Life of a request: [`PrefixCache::admit`] looks the prompt up (device first, then the host
//! tier, whose hit is promoted to the device), forks the snapshot's pages into the new placement
//! (full pages shared, the partial tail copied), restores its mark and returns where prefill
//! resumes, with the first token's logits when the whole prompt was retained. At prompt end the
//! scheduler captures a `Prompt` snapshot (unless the prompt was a total hit); at a cacheable
//! completion (EOS or max_tokens, client still there) a `Turn` snapshot; a prefill cancelled at
//! a chunk boundary is parked as a `Prompt` snapshot; a decode cancelled is not retained. Then
//! [`PrefixCache::release`] drops the placement's references.
//!
//! Selection is the most computation saved under the family's [`ReuseRule`](cuteafd_core::prefix::ReuseRule)
//! (`cuteafd_core::prefix::Retention`, prompts and turns in bounded banks). Eviction is least
//! recently used across both banks, prompt before turn at equal use ([`victim`]); `make_room`
//! runs before every admission and capture that needs pages. A restore that cannot complete is a
//! cache miss, never a request error.
use super::chain::{content_id, page_chain_media};
use crate::media::{round_frontier, snapshot_media, verify_media, MediaError, MediaSpan};
use super::entry::{victim, After, Entry, EntryId, Mark};
use super::family::{BoxError, CaptureTicket, Eviction, FamilyLayout, MarkStore, PrefixFamily, RestoreCandidate,
    RestoreContext, RestorePlan};
use super::marks::MarkArena;
use super::pages::{PoolExhausted, RefPagePool};
use cuteafd_core::prefix::{Retention, SnapshotKind};
use cuteafd_hostcache::cache::{
    DevicePage, DeviceSnapshot, EvictDecision, HostCache, RestoreOutcome, RestoreTarget, StoreOutcome,
};
use cuteafd_hostcache::copy::{CopyEngine, DeviceRange, Stream};
use cuteafd_hostcache::snapshot::{DevicePageId, EvictionOrder, SnapshotMeta};
use serde::Serialize;
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PrefixError {
    #[error(transparent)]
    Pages(#[from] PoolExhausted),
    #[error("family {what} failed: {source}")]
    Family {
        what: &'static str,
        #[source]
        source: BoxError,
    },
    #[error("snapshot of {tokens} tokens, but the placement committed {committed} (reach {reach})")]
    Frontier { tokens: usize, committed: usize, reach: usize },
    #[error(transparent)]
    Media(#[from] MediaError),
    #[error("host tier: {0}")]
    Host(String),
    #[error("family layout: {0}")]
    Layout(&'static str),
}

/// Knobs of the device tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PrefixConfig {
    /// Entries per bank (prompts, turns); 0 disables retention (the pool still allocates).
    pub entries: usize,
    /// Device mark slots ([`MarkArena::slots_for`]); unused when marks live in pool pages.
    pub mark_slots: usize,
    /// Keep the last logit row with each snapshot, so sampled exact-length hits need no forward.
    pub keep_logits: bool,
    /// Shortest snapshot worth retaining.
    pub min_tokens: usize,
}

/// A reusable snapshot for a prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hit {
    pub id: EntryId,
    pub kind: SnapshotKind,
    /// Rows the reuse rule saves (the ranking); the family's [`RestorePlan`] decides where prefill
    /// actually resumes.
    pub resume: usize,
    /// Tokens the prompt shares with the snapshot (media-verified up to `resume`).
    pub common: usize,
    pub frontier: usize,
}

/// Where a restored request came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Source {
    pub session: Option<String>,
    pub kind: SnapshotKind,
    pub frontier: usize,
    /// Promoted from the host tier by this admission.
    pub host: bool,
    /// A partial match: the positional state restarts empty at `resume`, which lies the rule's
    /// replay window before the aligned common prefix (approximate; exact frontiers are exact).
    pub partial: bool,
    /// The family's restore plan.
    pub plan: RestorePlan,
}

/// An admitted request: its placement, the rows already in it, and the first token's source
/// when the whole prompt was retained.
pub struct Admitted<P> {
    pub placement: P,
    pub resume: usize,
    pub after: Option<After>,
    pub source: Option<Source>,
}

/// What travels with a host snapshot besides its device bytes: plain host data (no device
/// handles), so an entry is self-describing given the family's layout.
pub struct HostPayload<M = ()> {
    pub session: Option<String>,
    pub after: After,
    pub media: Vec<MediaSpan>,
    /// The family's snapshot metadata ([`PrefixFamily::capture_meta`]).
    pub family: M,
}

/// Counters and gauges for `/v1/stats`.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct PrefixStats {
    pub lookups: u64,
    /// Rejected radix matches whose full image identities differ.
    pub media_key_collisions: u64,
    pub hits: u64,
    /// Hits that restored the whole prompt (first token from the snapshot's logits).
    pub exact_hits: u64,
    /// Hits on a partial match (replay window, approximate positional state).
    pub partial_hits: u64,
    pub hit_tokens: u64,
    pub promotions: u64,
    pub restore_failures: u64,
    /// Hits dropped because the pool could not hold the fork.
    pub fork_no_room: u64,
    pub captures_prompt: u64,
    pub captures_turn: u64,
    /// Prompt snapshots parked by a prefill cancelled at a chunk boundary.
    pub parked: u64,
    pub capture_skips: u64,
    pub evictions: u64,
    pub cow_copies: u64,
    pub host_store_skips: u64,
    pub host_evict_waits: u64,
    pub host_evict_uncached: u64,
    /// Queued captures published, aborted, and quarantined after a failed drain.
    pub captures_queued: u64,
    pub captures_aborted: u64,
    pub quarantined: u64,
    /// Snapshots page pressure left in place because evicting them frees no page
    /// ([`Eviction::FreesPages`]).
    pub eviction_skips: u64,
    pub entries_prompt: usize,
    pub entries_turn: usize,
    pub pages: usize,
    pub pages_free: usize,
    pub pages_shared: usize,
    /// Distinct pages held by retained snapshots; when no request runs, every used page is one.
    pub pages_retained: usize,
    pub mark_slots: usize,
    /// Retained marks, in arena slots or pool pages.
    pub marks_in_use: usize,
    /// Pool pages held by marks (counted in `pages_retained` too).
    pub mark_pages: usize,
    pub host: Option<cuteafd_hostcache::metrics::Snapshot>,
}

/// A capture whose copies a family queued on its own stream ([`PrefixCache::queue_capture`]).
/// It owns the snapshot's pages and mark until the copies land; nothing can evict or reuse them.
struct PendingCapture<M> {
    ticket: CaptureTicket,
    kind: SnapshotKind,
    entry: Entry<M>,
}

/// A capture whose storage was taken and whose copies are the caller's to enqueue.
struct Prepared<M> {
    entry: Entry<M>,
    copy: Option<super::pages::TailCopy>,
}

/// What [`PrefixCache::queue_capture`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Captured {
    /// Not retained (disabled, too short, no room).
    Skipped,
    /// Captured synchronously and retained.
    Done,
    /// Copies queued; poll with this ticket.
    Queued(CaptureTicket),
}

/// Storage whose queued copies could not be proven drained: never handed out again.
struct Quarantined {
    _pages: Vec<u32>,
    _mark: Option<Mark>,
}

pub struct PrefixCache<E: CopyEngine, M = ()> {
    layout: FamilyLayout,
    config: PrefixConfig,
    retained: Retention<EntryId>,
    entries: BTreeMap<EntryId, Entry<M>>,
    pending: Vec<PendingCapture<M>>,
    quarantine: Vec<Quarantined>,
    pool: RefPagePool,
    arena: MarkArena,
    host: Option<HostCache<E, HostPayload<M>>>,
    clock: u64,
    next_id: EntryId,
    /// Family copies were enqueued since the family last drained.
    dirty: bool,
    capture_session: Option<String>,
    /// Where the last admission's restore came from (provenance only).
    restored_session: Option<String>,
    stats: PrefixStats,
}

impl<E: CopyEngine, M: Clone + Default> PrefixCache<E, M> {
    /// `host`: the pinned host tier's knobs and copy engine (`None`, or a zero `bytes`, keeps it off).
    pub fn new(layout: FamilyLayout, config: PrefixConfig, host: Option<(cuteafd_hostcache::config::Config, E)>)
        -> Result<Self, PrefixError> {
        // Before the host tier pins its memory.
        if let MarkStore::Pool { pages, reserved } = layout.mark_store {
            if layout.mark_bytes == 0 || pages == 0 || pages > layout.pages.saturating_sub(reserved) {
                return Err(PrefixError::Layout("pool-page marks take at least one page and at most the pool's \
                    unreserved pages"));
            }
        }
        let host = match host {
            Some((host_config, engine)) if config.entries > 0 && host_config.enabled() => Some(
                HostCache::with_rule(host_config, layout.host_layout(), engine, layout.rule, EvictionOrder::LeastRecent)
                    .map_err(|e| PrefixError::Host(format!("{e:#}")))?,
            ),
            _ => None,
        };
        let arena = config.entries > 0 && layout.mark_bytes > 0 && layout.mark_store == MarkStore::Arena;
        let slots = if arena { config.mark_slots } else { 0 };
        Ok(Self {
            retained: Retention::with_rule(config.entries, layout.rule),
            entries: BTreeMap::new(),
            pending: Vec::new(),
            quarantine: Vec::new(),
            pool: RefPagePool::with_reserved(layout.pages, layout.page_rows, layout.mark_store.reserved()),
            arena: MarkArena::new(slots, layout.mark_bytes),
            host,
            clock: 0,
            next_id: 0,
            dirty: false,
            capture_session: None,
            restored_session: None,
            stats: PrefixStats::default(),
            layout,
            config,
        })
    }

    /// The session the last admission restored from, if it restored.
    pub fn restored_session(&self) -> Option<&str> {
        self.restored_session.as_deref()
    }

    /// Whether completed-turn snapshots are kept at all (a disabled cache takes no frontier).
    pub fn turn_bank_enabled(&self) -> bool {
        self.enabled()
    }

    /// The host tier's configuration, when it is on.
    pub fn host_config(&self) -> Option<&cuteafd_hostcache::config::Config> {
        self.host.as_ref().map(HostCache::config)
    }

    /// Once per prefill chunk: observe the host tier's store stream, then hold the chunk for at
    /// most the configured store pace while the oldest store copy is overdue
    /// ([`HostCache::prefill_hold`]). A no-op without a host tier.
    pub fn prefill_hold(&mut self) -> anyhow::Result<()> {
        match &mut self.host {
            Some(host) => {
                host.tick();
                host.prefill_hold()
            }
            None => Ok(()),
        }
    }

    /// Set only immediately before a capture; provenance never retains request handles.
    pub fn capture_session(&mut self, session: Option<String>) { self.capture_session = session; }

    pub fn enabled(&self) -> bool {
        self.config.entries > 0
    }
    pub fn layout(&self) -> FamilyLayout {
        self.layout
    }
    pub fn pool(&self) -> &RefPagePool {
        &self.pool
    }
    pub fn arena(&self) -> &MarkArena {
        &self.arena
    }
    pub fn host_engine_mut(&mut self) -> Option<&mut E> {
        self.host.as_mut().map(HostCache::engine_mut)
    }

    /// Admit a request of `tokens` that may grow to `capacity` tokens. `build` turns a page list
    /// into the family's placement (called once per attempt; a failed restore retries cold).
    pub fn admit<F: PrefixFamily<M, Placement = P>, P>(&mut self, family: &F, tokens: &[u32], capacity: usize,
        sampled: bool, build: impl FnMut(Vec<u32>) -> P) -> Result<Admitted<P>, PrefixError> {
        self.admit_native(family, tokens, tokens, &[], capacity, sampled, build)
    }

    /// `tokens` are MediaKeys' keyed copy, never the model's native ids. Verify identities
    /// before forking pages. Encoding waiters call peek_media first, without owning a placement.
    pub fn admit_media<F: PrefixFamily<M, Placement = P>, P>(&mut self, family: &F, tokens: &[u32],
        media: &[MediaSpan], capacity: usize, sampled: bool, build: impl FnMut(Vec<u32>) -> P)
        -> Result<Admitted<P>, PrefixError> {
        self.admit_native(family, tokens, tokens, media, capacity, sampled, build)
    }

    /// [`PrefixCache::admit_media`] with the prompt's native token ids beside its keyed copy, for
    /// a family whose restore rebuilds token-derived state ([`RestoreContext`]).
    #[allow(clippy::too_many_arguments)]
    pub fn admit_native<F: PrefixFamily<M, Placement = P>, P>(&mut self, family: &F, tokens: &[u32],
        native: &[u32], media: &[MediaSpan], capacity: usize, sampled: bool, mut build: impl FnMut(Vec<u32>) -> P)
        -> Result<Admitted<P>, PrefixError> {
        crate::media::keys::validate_spans(media)?;
        if media.last().is_some_and(|s| s.checked_end().unwrap() > tokens.len()) || native.len() != tokens.len() {
            return Err(MediaError::Spans.into());
        }
        self.restored_session = None;
        let total = self.pool.pages_for(capacity.max(tokens.len()));
        if self.enabled() && !tokens.is_empty() {
            self.stats.lookups += 1;
            let mut promoted = false;
            let mut hit = self.lookup_media(tokens, media, sampled);
            if hit.is_none() {
                hit = self.promote(family, tokens, media, sampled)?;
                promoted = hit.is_some();
            }
            if let Some(hit) = hit {
                let context = RestoreContext { native_tokens: native, media };
                match self.restore_hit(family, &hit, tokens, &context, total, promoted, &mut build) {
                    Ok(Some(admitted)) => return Ok(admitted),
                    Ok(None) => self.stats.fork_no_room += 1,
                    Err(PrefixError::Family { what, source }) => {
                        tracing::warn!(target: "cuteafd::prefix", what, error = %source, "prefix restore abandoned; prefilling");
                        self.stats.restore_failures += 1;
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        self.admit_cold(family, tokens.len(), capacity, build)
    }

    /// Bypass prefix reuse after repeated peek/admission races. The caller has materialized
    /// the entire prompt's media before invoking this allocator.
    pub fn admit_cold<F: PrefixFamily<M, Placement = P>, P>(&mut self, family: &F, tokens: usize,
        capacity: usize, build: impl FnOnce(Vec<u32>) -> P) -> Result<Admitted<P>, PrefixError> {
        let total = self.pool.pages_for(capacity.max(tokens));
        self.make_room(family, total, None)?;
        let pages = self.pool.alloc(total)?;
        let mut placement = build(pages.clone());
        if let Err(source) = family.bind(&mut placement) {
            self.release(family, &pages)?;
            return Err(PrefixError::Family { what: "bind", source });
        }
        Ok(Admitted { placement, resume: 0, after: None, source: None })
    }

    /// Device lookup under the family's rule. An exact-length match whose snapshot cannot
    /// produce the first token gives way to the longest shorter one.
    fn lookup(&mut self, tokens: &[u32], sampled: bool) -> Option<Hit> {
        let mut query = tokens;
        loop {
            let (common, frontier, &id) = self.retained.lookup_reusable(query)?;
            let resume = self.layout.rule.skipped(common, frontier);
            let entry = self.entries.get_mut(&id)?;
            if resume == tokens.len() && !entry.after.serves(sampled) {
                if query.len() < tokens.len() || tokens.len() < 2 {
                    return None;
                }
                query = &tokens[..tokens.len() - 1];
                continue;
            }
            if !verify_media(resume, &entry.media, &[]) { return None; }
            self.clock += 1;
            entry.last_use = self.clock;
            return Some(Hit { id, kind: entry.kind, resume, common: common.min(query.len()), frontier });
        }
    }

    /// Non-mutating lookup across both tiers. No copies, allocation, pins, counters or LRU
    /// touches: it is only an encoding hint, and normal admission rechecks it afterwards.
    pub fn peek(&self, tokens: &[u32], sampled: bool) -> usize {
        self.peek_media(tokens, &[], sampled)
    }

    pub fn peek_media(&self, tokens: &[u32], media: &[MediaSpan], sampled: bool) -> usize {
        if !self.enabled() { return 0; }
        let device = if media.is_empty() {
            let mut query = tokens;
            loop {
                let Some((common, frontier, &id)) = self.retained.peek_reusable(query) else { break 0 };
                let Some(entry) = self.entries.get(&id) else { break 0 };
                let resume = self.layout.rule.skipped(common, frontier);
                if !verify_media(resume, &entry.media, media) { break 0; }
                if resume == tokens.len() && !entry.after.serves(sampled) {
                    if query.len() < tokens.len() || tokens.len() < 2 { break 0; }
                    query = &tokens[..tokens.len() - 1];
                } else { break resume; }
            }
        } else {
            self.device_media_hit(tokens, media, sampled).0.map_or(0, |h| h.resume)
        };
        let host = self.host.as_ref().and_then(|host| {
            if media.is_empty() {
                let hit = host.peek(tokens)?;
                let payload = host.payload(hit.key)?;
                let resume = self.layout.rule.skipped(hit.common, hit.frontier);
                (verify_media(resume, &payload.media, media)
                    && (resume != tokens.len() || payload.after.serves(sampled))).then_some(resume)
            } else { self.host_media_hit(tokens, media, sampled).0.map(|h| h.1) }
        }).unwrap_or(0);
        device.max(host)
    }

    fn device_media_hit(&self, tokens: &[u32], media: &[MediaSpan], sampled: bool) -> (Option<Hit>, u64) {
        let mut best = None;
        let mut rank = (0, false, 0);
        let mut collisions = 0;
        for (&id, entry) in &self.entries {
            let (resume, common, collision) = media_resume(self.layout.rule, tokens, media, &entry.tokens,
                &entry.media, &entry.after, sampled);
            collisions += u64::from(collision);
            let candidate = (resume, entry.kind == SnapshotKind::Turn, entry.last_use);
            if resume > 0 && candidate > rank {
                rank = candidate;
                best = Some(Hit { id, kind: entry.kind, resume, common, frontier: entry.len() });
            }
        }
        (best, collisions)
    }

    fn lookup_media(&mut self, tokens: &[u32], media: &[MediaSpan], sampled: bool) -> Option<Hit> {
        if media.is_empty() { return self.lookup(tokens, sampled); }
        let (hit, collisions) = self.device_media_hit(tokens, media, sampled);
        self.stats.media_key_collisions += collisions;
        if let Some(hit) = hit {
            self.clock += 1;
            self.entries.get_mut(&hit.id)?.last_use = self.clock;
        }
        hit
    }

    fn host_media_hit(&self, tokens: &[u32], media: &[MediaSpan], sampled: bool)
        -> (Option<(cuteafd_hostcache::snapshot::Hit, usize)>, u64) {
        let Some(host) = &self.host else { return (None, 0) };
        let mut best = None;
        let mut rank = (0, false, 0);
        let mut collisions = 0;
        for (meta, key, payload) in host.resident_payloads() {
            let (resume, _, collision) = media_resume(self.layout.rule, tokens, media, &meta.tokens,
                &payload.media, &payload.after, sampled);
            collisions += u64::from(collision);
            let candidate = (resume, meta.kind == SnapshotKind::Turn, key);
            if resume > 0 && candidate > rank {
                rank = candidate;
                let common = common_prefix(tokens, &meta.tokens);
                best = Some((cuteafd_hostcache::snapshot::Hit {
                    key, kind: meta.kind, common, frontier: meta.tokens.len(),
                }, resume));
            }
        }
        (best, collisions)
    }

    #[allow(clippy::too_many_arguments)]
    fn restore_hit<F: PrefixFamily<M, Placement = P>, P>(&mut self, family: &F, hit: &Hit, tokens: &[u32],
        context: &RestoreContext<'_>, total: usize, promoted: bool, build: &mut impl FnMut(Vec<u32>) -> P)
        -> Result<Option<Admitted<P>>, PrefixError> {
        let saved_media = self.entries[&hit.id].media.clone();
        let plan = family.plan_restore(RestoreCandidate { common: hit.common, snapshot_end: hit.frontier,
            target_end: tokens.len(), resume: hit.resume, media: context.media, saved_media: &saved_media });
        // Media atomicity holds at both frontiers: shared rows never end inside an image, and a
        // replay never starts inside one. An exact plan is verified at the snapshot's frontier.
        let media_safe = |at: usize| round_frontier(at, context.media) == at && round_frontier(at, &saved_media) == at;
        let checked = plan.check(hit.common).map_err(|e| PrefixError::Family { what: "plan", source: e.into() })
            .and_then(|()| if plan.exact() != (hit.common == hit.frontier) || hit.common < plan.lag.source_end
                || !media_safe(plan.lag.source_end) || !media_safe(plan.lag.replay_start)
                || !verify_media(plan.lag.source_end, &self.entries[&hit.id].media, context.media) {
                Err(PrefixError::Family { what: "plan", source: "restore plan crosses a media span or claims exactness \
                    for a partial match".into() })
            } else { Ok(()) });
        checked?;
        let (source_end, resume) = (plan.lag.source_end, plan.resume());
        if resume == 0 {
            return Ok(None);
        }
        let need = self.pool.fork_cost(source_end, total);
        if !self.make_room(family, need, Some(hit.id))? {
            return Ok(None);
        }
        let entry = self.entries.get(&hit.id).expect("the hit is kept while making room");
        let fork = self.pool.fork(&entry.pages, source_end, total)?;
        let exact = plan.exact();
        let (mark, meta) = if exact { (entry.mark.clone(), Some(entry.meta.clone())) } else { (None, None) };
        let after = (resume == tokens.len()).then(|| entry.after.clone());
        let session = entry.session.clone();
        let pages = fork.pages.clone();
        let mut placement = build(fork.pages);
        self.dirty = true;
        let restored = family.bind(&mut placement).map_err(|e| ("bind", e))
            .and_then(|()| match fork.copy {
                Some(copy) => {
                    self.stats.cow_copies += 1;
                    family.copy_rows(copy).map_err(|e| ("copy", e))
                }
                None => Ok(()),
            })
            .and_then(|()| family.restore_with(mark.as_ref(), meta.as_ref(), &mut placement, &plan, context)
                .map_err(|e| ("restore", e)));
        if let Err((what, source)) = restored {
            // Drain before the pages can be handed out again; the family resets what it bound.
            self.drain(family)?;
            if let Err(error) = family.discard(&mut placement) {
                tracing::error!(target: "cuteafd::prefix", %error, "discarding a failed restore");
            }
            self.release(family, &pages)?;
            return Err(PrefixError::Family { what, source });
        }
        self.stats.hits += 1;
        self.stats.hit_tokens += resume as u64;
        self.stats.exact_hits += u64::from(after.is_some());
        self.stats.partial_hits += u64::from(!exact);
        self.restored_session = session.clone();
        Ok(Some(Admitted {
            placement,
            resume,
            after,
            source: Some(Source { session, kind: hit.kind, frontier: hit.frontier, host: promoted, partial: !exact, plan }),
        }))
    }

    /// Retain `placement`'s first `tokens.len()` rows as a `kind` snapshot; `after` is what
    /// follows them. The snapshot point may lie up to `family.capture_reach()` rows before the
    /// placement's commit point (an intermediate point). Ok(false) when it was not retained
    /// (disabled, too short, no room).
    pub fn capture<F: PrefixFamily<M>>(&mut self, family: &F, kind: SnapshotKind, tokens: &[u32],
        placement: &F::Placement, after: After) -> Result<bool, PrefixError> {
        self.capture_media(family, kind, tokens, &[], placement, after)
    }

    /// A chunk/message/cancel frontier inside an image rounds down to its first row.
    /// If the family's mark can no longer reach that row, skip instead of capturing wrong state.
    pub fn capture_media<F: PrefixFamily<M>>(&mut self, family: &F, kind: SnapshotKind, tokens: &[u32],
        media: &[MediaSpan], placement: &F::Placement, after: After) -> Result<bool, PrefixError> {
        let Some(prepared) = self.prepare_capture(family, kind, tokens, media, placement, after)? else {
            return Ok(false);
        };
        let Prepared { entry, copy } = prepared;
        let len = entry.len();
        let captured = match copy {
            Some(copy) => family.copy_rows(copy),
            None => Ok(()),
        }
        .and_then(|()| match &entry.mark {
            Some(Mark::Pages(pages)) => family.capture_pages(pages, placement, len),
            Some(Mark::Slot(slot)) => family.capture(*slot, placement, len),
            None => Ok(()),
        });
        if let Err(source) = captured {
            self.release(family, &entry.pages)?;
            self.give_back(family, entry.mark)?;
            return Err(PrefixError::Family { what: "capture", source });
        }
        self.publish(family, kind, entry)?;
        Ok(true)
    }

    /// [`PrefixCache::capture_media`] for a family that copies its snapshots on a stream of its
    /// own ([`PrefixFamily::queue_capture`]): the storage is taken and the copies queued now; the
    /// snapshot is published by [`PrefixCache::poll_capture`] once they landed. Until then the
    /// pending capture owns its pages and mark; nothing evicts or reuses them. A family without
    /// queued captures captures synchronously and gets `Ok(Captured::Done)`.
    pub fn queue_capture<F: PrefixFamily<M>>(&mut self, family: &F, kind: SnapshotKind, tokens: &[u32],
        media: &[MediaSpan], placement: &F::Placement, after: After) -> Result<Captured, PrefixError> {
        if !self.arena_marks() {
            return self.capture_media(family, kind, tokens, media, placement, after)
                .map(|done| if done { Captured::Done } else { Captured::Skipped });
        }
        let Some(Prepared { entry, copy }) = self.prepare_capture(family, kind, tokens, media, placement, after)? else {
            return Ok(Captured::Skipped);
        };
        let Some(Mark::Slot(slot)) = entry.mark else { unreachable!("arena marks take a slot") };
        match family.queue_capture(slot, placement, entry.len(), copy) {
            Ok(Some(ticket)) => {
                self.pending.push(PendingCapture { ticket, kind, entry });
                self.stats.captures_queued += 1;
                Ok(Captured::Queued(ticket))
            }
            Ok(None) => {
                // The family captures synchronously: do it now on its own stream.
                let len = entry.len();
                let captured = match copy {
                    Some(copy) => family.copy_rows(copy),
                    None => Ok(()),
                }.and_then(|()| family.capture(slot, placement, len));
                if let Err(source) = captured {
                    self.release(family, &entry.pages)?;
                    self.give_back(family, entry.mark)?;
                    return Err(PrefixError::Family { what: "capture", source });
                }
                self.publish(family, kind, entry)?;
                Ok(Captured::Done)
            }
            Err(source) => {
                // A partial enqueue may still write the storage: drain it through the family,
                // or keep it forever.
                match family.drain() {
                    Ok(()) => {
                        self.release(family, &entry.pages)?;
                        self.give_back(family, entry.mark)?;
                    }
                    Err(error) => {
                        tracing::error!(target: "cuteafd::prefix", %error, "capture enqueue failed and did not drain; quarantining");
                        self.quarantine.push(Quarantined { _pages: entry.pages, _mark: entry.mark });
                        self.stats.quarantined += 1;
                    }
                }
                Err(PrefixError::Family { what: "capture", source })
            }
        }
    }

    /// Publish a queued capture whose copies landed: `Ok(true)` once it is retained (or was
    /// dropped because its bank no longer wants it), `Ok(false)` while copies are in flight.
    pub fn poll_capture<F: PrefixFamily<M>>(&mut self, family: &F, ticket: CaptureTicket) -> Result<bool, PrefixError> {
        let Some(index) = self.pending.iter().position(|p| p.ticket == ticket) else {
            return Err(PrefixError::Layout("no pending capture with this ticket"));
        };
        let ready = match family.capture_ready(ticket) {
            Ok(ready) => ready,
            Err(source) => {
                self.abort_capture(family, ticket)?;
                return Err(PrefixError::Family { what: "capture poll", source });
            }
        };
        if !ready {
            return Ok(false);
        }
        let PendingCapture { kind, entry, .. } = self.pending.swap_remove(index);
        // Another capture may have filled the bank or replaced this frontier meanwhile.
        if let Some(old) = self.retained.bank_mut(kind).remove_exact(&entry.tokens) {
            self.evict(family, old)?;
        }
        self.publish(family, kind, entry)?;
        Ok(true)
    }

    /// Abandon a queued capture: its storage returns to the cache once the family proved its
    /// copies drained, and is quarantined (never handed out again) otherwise.
    pub fn abort_capture<F: PrefixFamily<M>>(&mut self, family: &F, ticket: CaptureTicket) -> Result<(), PrefixError> {
        let Some(index) = self.pending.iter().position(|p| p.ticket == ticket) else { return Ok(()) };
        let PendingCapture { entry, .. } = self.pending.swap_remove(index);
        self.stats.captures_aborted += 1;
        match family.abort_capture(ticket) {
            Ok(()) => {
                self.release(family, &entry.pages)?;
                self.give_back(family, entry.mark)
            }
            Err(error) => {
                tracing::error!(target: "cuteafd::prefix", %error, "queued capture did not drain; quarantining its storage");
                self.quarantine.push(Quarantined { _pages: entry.pages, _mark: entry.mark });
                self.stats.quarantined += 1;
                Ok(())
            }
        }
    }

    /// Validate a capture and take its storage: bank room, a mark, the forked pages (the tail
    /// copy is the caller's to run). `None` when it is not retained (disabled, too short, no room).
    fn prepare_capture<F: PrefixFamily<M>>(&mut self, family: &F, kind: SnapshotKind, tokens: &[u32],
        media: &[MediaSpan], placement: &F::Placement, mut after: After) -> Result<Option<Prepared<M>>, PrefixError> {
        crate::media::keys::validate_spans(media)?;
        let rounded = round_frontier(tokens.len(), media);
        let changed = rounded != tokens.len();
        let tokens = &tokens[..rounded];
        if changed { after = After::default(); }
        if !self.enabled() || tokens.len() < self.config.min_tokens.max(1) {
            return Ok(None);
        }
        let committed = family.commit_point(placement);
        let reach = family.capture_reach();
        if tokens.len() > committed || committed - tokens.len() > reach {
            if changed { self.stats.capture_skips += 1; return Ok(None); }
            return Err(PrefixError::Frontier { tokens: tokens.len(), committed, reach });
        }
        let len = tokens.len();
        let meta = family.capture_meta(placement, len)
            .map_err(|source| PrefixError::Family { what: "capture metadata", source })?;
        // Replace the same snapshot, or keep the bank within its bound (pending captures count),
        // before taking storage.
        if let Some(old) = self.retained.bank_mut(kind).remove_exact(tokens) {
            self.evict(family, old)?;
        }
        let pending = self.pending.iter().filter(|p| p.kind == kind).count();
        while self.retained.bank(kind).entries() + pending >= self.config.entries {
            let oldest = victim(self.entries.iter().filter(|(_, e)| e.kind == kind).map(|(&id, e)| (id, e.kind, e.last_use)));
            match oldest {
                Some(id) => self.evict(family, id)?,
                None => break,
            }
        }
        if self.retained.bank(kind).entries() + pending >= self.config.entries {
            self.stats.capture_skips += 1;
            return Ok(None);
        }
        let slot = if self.arena_marks() {
            loop {
                match self.arena.take() {
                    Ok(slot) => break Some(slot),
                    Err(_) => match self.victim(None) {
                        Some(id) => self.evict(family, id)?,
                        None => {
                            self.stats.capture_skips += 1;
                            return Ok(None);
                        }
                    },
                }
            }
        } else {
            None
        };
        let total = self.pool.pages_for(len);
        // The rows a fork copies and, for pool-page marks, the mark's own pages.
        let mark_pages = self.layout.mark_store.pages();
        if !self.make_room(family, self.pool.fork_cost(len, total) + mark_pages, None)? {
            self.give_back(family, slot.map(Mark::Slot))?;
            self.stats.capture_skips += 1;
            return Ok(None);
        }
        let mark = match slot {
            Some(slot) => Some(Mark::Slot(slot)),
            None => self.take_mark_pages()?,
        };
        let fork = match self.pool.fork(family.pages(placement), len, total) {
            Ok(fork) => fork,
            Err(error) => {
                self.give_back(family, mark)?;
                return Err(error.into());
            }
        };
        self.dirty = true;
        if !self.config.keep_logits {
            after.logits = None;
        }
        let entry = Entry { session: self.capture_session.clone(), tokens: tokens.to_vec(),
            media: snapshot_media(len, media), kind, pages: fork.pages, mark, after, meta, last_use: 0, ticket: None };
        Ok(Some(Prepared { entry, copy: fork.copy }))
    }

    /// Retain a captured snapshot whose copies were enqueued (synchronous families) or landed.
    fn publish<F: PrefixFamily<M>>(&mut self, family: &F, kind: SnapshotKind, mut entry: Entry<M>)
        -> Result<(), PrefixError> {
        self.clock += 1;
        entry.last_use = self.clock;
        let id = self.next_id;
        self.next_id += 1;
        let tokens = entry.tokens.clone();
        self.entries.insert(id, entry);
        if let Some(evicted) = self.retained.bank_mut(kind).insert(&tokens, id) {
            self.evict(family, evicted)?;
        }
        match kind {
            SnapshotKind::Prompt => self.stats.captures_prompt += 1,
            SnapshotKind::Turn => self.stats.captures_turn += 1,
        }
        self.host_store(family, id)
    }

    /// A prefill cancelled at a chunk boundary: keep what it computed as a prompt snapshot
    /// (the client usually retries the same prompt).
    pub fn park<F: PrefixFamily<M>>(&mut self, family: &F, tokens: &[u32], placement: &F::Placement)
        -> Result<bool, PrefixError> {
        let parked = self.capture(family, SnapshotKind::Prompt, tokens, placement, After::default())?;
        self.stats.parked += u64::from(parked);
        Ok(parked)
    }

    pub fn park_media<F: PrefixFamily<M>>(&mut self, family: &F, tokens: &[u32], media: &[MediaSpan],
        placement: &F::Placement) -> Result<bool, PrefixError> {
        let parked = self.capture_media(family, SnapshotKind::Prompt, tokens, media, placement, After::default())?;
        self.stats.parked += u64::from(parked);
        Ok(parked)
    }

    /// Drop a placement's page references (after the family's queued work drained).
    pub fn release<F: PrefixFamily<M>>(&mut self, family: &F, pages: &[u32]) -> Result<(), PrefixError> {
        self.drain(family)?;
        let freed = self.pool.release(pages);
        if let Some(host) = &mut self.host {
            for page in freed {
                host.device_page_freed(DevicePageId { compressor: 0, page: page.index, generation: page.generation });
            }
        }
        Ok(())
    }

    /// Evict snapshots (never `keep`) until `pages` pages are free; false when nothing is left to
    /// evict and the pool is still short. The family's [`Eviction`] picks the victims:
    /// least recently used, or ([`Eviction::FreesPages`]) only snapshots whose eviction frees a
    /// page, least recently used first, so a snapshot whose pages running requests hold stays
    /// reusable.
    pub fn make_room<F: PrefixFamily<M>>(&mut self, family: &F, pages: usize, keep: Option<EntryId>)
        -> Result<bool, PrefixError> {
        let frees = family.eviction() == Eviction::FreesPages;
        while self.pool.free() < pages {
            let victim = if frees { self.freeing_victim(keep) } else { self.victim(keep) };
            match victim {
                Some(id) => self.evict(family, id)?,
                None => {
                    if frees {
                        self.stats.eviction_skips += self.entries.len().saturating_sub(usize::from(keep.is_some())) as u64;
                    }
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Whether `pages` more pages could be had, evicting only what [`PrefixCache::make_room`]
    /// would. Changes nothing.
    pub fn could_free(&self, pages: usize) -> bool {
        // Under either eviction order, the pages that come back are those only retained
        // snapshots hold (a page a placement holds never frees).
        self.pool.free() >= pages || self.pool.free() + self.snapshot_only_pages().count() >= pages
    }

    /// Pages every reference to which a retained snapshot holds.
    fn snapshot_only_pages(&self) -> impl Iterator<Item = u32> + '_ {
        let mut held: BTreeMap<u32, u32> = BTreeMap::new();
        for entry in self.entries.values() {
            for &page in entry.pages.iter().chain(entry.mark_pages()) {
                *held.entry(page).or_default() += 1;
            }
        }
        held.into_iter().filter(|&(page, n)| self.pool.refs(page) == n).map(|(page, _)| page)
    }

    /// Give back a placement's pages past `keep` (an idle request whose output allowance shrank
    /// to what the pool holds). Only pages the placement alone holds and nothing wrote may go.
    pub fn trim<F: PrefixFamily<M>>(&mut self, family: &F, pages: &mut Vec<u32>, keep: usize)
        -> Result<(), PrefixError> {
        if pages.len() <= keep { return Ok(()); }
        let tail = pages.split_off(keep);
        self.release(family, &tail)
    }

    /// Extend a live lease before a forward pass. Only inactive snapshots can
    /// be evicted; existing live pages keep their references on failed growth.
    pub fn grow<F: PrefixFamily<M>>(&mut self, family: &F, pages: &mut Vec<u32>, tokens: usize)
        -> Result<(), PrefixError> {
        let additional = self.pool.pages_for(tokens).saturating_sub(pages.len());
        if additional == 0 { return Ok(()); }
        self.make_room(family, additional, None)?;
        pages.extend(self.pool.alloc(additional)?);
        Ok(())
    }

    /// Poll the host tier's store copies; once per scheduler step.
    pub fn tick(&mut self) {
        if let Some(host) = &mut self.host {
            host.tick();
        }
    }

    pub fn stats(&self) -> PrefixStats {
        let mut stats = self.stats.clone();
        stats.entries_prompt = self.retained.bank(SnapshotKind::Prompt).entries();
        stats.entries_turn = self.retained.bank(SnapshotKind::Turn).entries();
        stats.pages = self.pool.capacity();
        stats.pages_free = self.pool.free();
        stats.pages_shared = self.pool.shared();
        let mut retained: Vec<u32> = self.entries.values()
            .flat_map(|e| e.pages.iter().chain(e.mark_pages()).copied()).collect();
        retained.sort_unstable();
        retained.dedup();
        stats.pages_retained = retained.len();
        stats.mark_slots = self.arena.slots();
        stats.mark_pages = self.entries.values().map(|e| e.mark_pages().len()).sum();
        stats.marks_in_use = self.arena.in_use()
            + self.entries.values().filter(|e| matches!(e.mark, Some(Mark::Pages(_)))).count();
        stats.host = self.host.as_ref().map(HostCache::metrics);
        stats
    }

    /// Evict every snapshot (tests and shutdown).
    pub fn clear<F: PrefixFamily<M>>(&mut self, family: &F) -> Result<(), PrefixError> {
        while let Some(id) = self.victim(None) {
            self.evict(family, id)?;
        }
        Ok(())
    }

    fn victim(&self, keep: Option<EntryId>) -> Option<EntryId> {
        victim(self.entries.iter().filter(|(&id, _)| Some(id) != keep).map(|(&id, e)| (id, e.kind, e.last_use)))
    }

    /// [`Eviction::FreesPages`]: the least recently used snapshot (prompt first at equal use)
    /// holding a page, or mark page, that only it references. Evicting a snapshot all of whose
    /// pages a running request or another snapshot also holds frees nothing. Once every snapshot
    /// sharing such pages with no live holder has gone, their pages free with the last one, so
    /// a page held only by snapshots counts as freeable when every holder is a snapshot.
    fn freeing_victim(&self, keep: Option<EntryId>) -> Option<EntryId> {
        let freeable: std::collections::BTreeSet<u32> = self.snapshot_only_pages().collect();
        victim(self.entries.iter()
            .filter(|(&id, e)| Some(id) != keep && e.pages.iter().chain(e.mark_pages()).any(|p| freeable.contains(p)))
            .map(|(&id, e)| (id, e.kind, e.last_use)))
    }

    fn drain<F: PrefixFamily<M>>(&mut self, family: &F) -> Result<(), PrefixError> {
        if self.dirty {
            family.drain().map_err(|source| PrefixError::Family { what: "drain", source })?;
            self.dirty = false;
        }
        Ok(())
    }

    /// Whether marks live in the arena (a family with marks and no pool-page store).
    fn arena_marks(&self) -> bool {
        self.layout.mark_bytes > 0 && self.layout.mark_store == MarkStore::Arena
    }

    /// A pool-page mark's pages (the caller made room), in ascending order so a family's
    /// segments over consecutive pages coalesce; `None` for arena or mark-less families.
    fn take_mark_pages(&mut self) -> Result<Option<Mark>, PrefixError> {
        match self.layout.mark_store {
            MarkStore::Pool { pages, .. } if self.layout.mark_bytes > 0 => {
                let mut pages = self.pool.alloc(pages)?;
                pages.sort_unstable();
                Ok(Some(Mark::Pages(pages)))
            }
            _ => Ok(None),
        }
    }

    /// Return a mark's storage once the family's queued copies drained: its arena slot, or its
    /// pool pages.
    fn give_back<F: PrefixFamily<M>>(&mut self, family: &F, mark: Option<Mark>) -> Result<(), PrefixError> {
        match mark {
            Some(Mark::Slot(slot)) => {
                self.drain(family)?;
                self.arena.give_back(slot);
            }
            Some(Mark::Pages(pages)) => self.release(family, &pages)?,
            None => {}
        }
        Ok(())
    }

    /// The device ranges of `mark` for the host tier (a mark-less family's stand-in tail).
    fn mark_segments<F: PrefixFamily<M>>(family: &F, mark: Option<&Mark>) -> Result<Vec<DeviceRange>, PrefixError> {
        Ok(match mark {
            Some(&Mark::Slot(slot)) => family.mark_segments(slot),
            Some(Mark::Pages(pages)) => family.mark_page_segments(pages)
                .map_err(|source| PrefixError::Family { what: "mark segments", source })?,
            None => family.host_tail(),
        })
    }

    /// Drop a device snapshot: its host copy finishes within budget first, then its storage goes
    /// back once the family's queued copies drained.
    fn evict<F: PrefixFamily<M>>(&mut self, family: &F, id: EntryId) -> Result<(), PrefixError> {
        let Some(entry) = self.entries.get(&id) else { return Ok(()) };
        if let Some(host) = &mut self.host {
            match host.before_device_evict(entry.ticket) {
                EvictDecision::Clean => {}
                EvictDecision::WaitedClean { .. } => self.stats.host_evict_waits += 1,
                EvictDecision::DroppedUncached => self.stats.host_evict_uncached += 1,
                EvictDecision::Held => return Err(PrefixError::Host(
                    "host snapshot copy did not drain; retaining device pages and positional mark".into())),
            }
        }
        let entry = self.entries.remove(&id).expect("checked before release barrier");
        self.retained.bank_mut(entry.kind).remove_exact(&entry.tokens);
        self.stats.evictions += 1;
        self.release(family, &entry.pages)?;
        self.give_back(family, entry.mark)
    }

    /// Host identities of an entry's pages: full pages by content (hash chain over their tokens),
    /// the partial tail by device allocation.
    fn identities(&self, tokens: &[u32], media: &[MediaSpan], pages: &[u32]) -> Vec<DevicePageId> {
        let chain = page_chain_media(tokens, media, self.layout.page_rows);
        pages
            .iter()
            .enumerate()
            .map(|(i, &page)| match chain.get(i) {
                Some(&id) => content_id(0, id),
                None => DevicePageId { compressor: 0, page, generation: self.pool.generation(page) },
            })
            .collect()
    }

    /// Issue the write-behind host copy of a new snapshot (after its device copies drained).
    fn host_store<F: PrefixFamily<M>>(&mut self, family: &F, id: EntryId) -> Result<(), PrefixError> {
        if self.host.is_none() {
            return Ok(());
        }
        self.drain(family)?;
        let entry = self.entries.get(&id).expect("stored entry is retained");
        let ids = self.identities(&entry.tokens, &entry.media, &entry.pages);
        let mut pages: [Vec<DevicePage>; cuteafd_hostcache::COMPRESSORS] = Default::default();
        pages[0] = entry.pages.iter().zip(ids).map(|(&page, id)| DevicePage { id, segments: family.page_segments(page) }).collect();
        let snapshot = DeviceSnapshot {
            meta: SnapshotMeta { kind: entry.kind, tokens: entry.tokens.clone(), end: entry.len() as u32, has_draft: false },
            pages,
            tail: Self::mark_segments(family, entry.mark.as_ref())?,
            draft: None,
            scores: Vec::new(),
        };
        let payload = HostPayload { session: entry.session.clone(), after: entry.after.clone(), media: entry.media.clone(),
            family: entry.meta.clone() };
        let host = self.host.as_mut().expect("checked");
        let ticket = match host.store(&snapshot, payload) {
            StoreOutcome::Issued(ticket) | StoreOutcome::Deferred(ticket) => Some(ticket),
            StoreOutcome::Skipped(_) => {
                self.stats.host_store_skips += 1;
                None
            }
        };
        self.entries.get_mut(&id).expect("checked").ticket = ticket;
        Ok(())
    }

    /// On a device miss: rebuild the best host snapshot on the device so the device path finds
    /// it. Anything short of a completed restore is a miss.
    fn promote<F: PrefixFamily<M>>(&mut self, family: &F, tokens: &[u32], media: &[MediaSpan], sampled: bool) -> Result<Option<Hit>, PrefixError> {
        let selected = if media.is_empty() {
            self.host.as_mut().and_then(|host| host.lookup(tokens))
        } else {
            let (selected, collisions) = self.host_media_hit(tokens, media, sampled);
            self.stats.media_key_collisions += collisions;
            let key = selected.as_ref().map(|(hit, _)| hit.key);
            if !self.host.as_mut().is_some_and(|host| host.lookup_verified(key)) {
                return Ok(None);
            }
            selected.map(|(hit, _)| hit)
        };
        let Some(hit) = selected else { return Ok(None) };
        let host = self.host.as_mut().expect("selected host snapshot");
        let (Some(snapshot_tokens), Some(payload)) = (host.snapshot_tokens(hit.key), host.payload(hit.key)) else {
            return Ok(None);
        };
        let snapshot_tokens = snapshot_tokens.to_vec();
        let after = payload.after.clone();
        let saved_media = payload.media.clone();
        let session = payload.session.clone();
        let meta = payload.family.clone();
        let (resume, _, collision) = media_resume(self.layout.rule, tokens, media, &snapshot_tokens,
            &saved_media, &after, sampled);
        if resume == 0 {
            self.stats.media_key_collisions += u64::from(collision);
            return Ok(None);
        }
        let len = snapshot_tokens.len();
        if len == tokens.len() && !after.serves(sampled) {
            return Ok(None);
        }
        let need = self.pool.pages_for(len);
        let mark_pages = self.layout.mark_store.pages();
        if !self.make_room(family, need + mark_pages, None)? {
            return Ok(None);
        }
        let slot = if self.arena_marks() {
            loop {
                match self.arena.take() {
                    Ok(slot) => break Some(slot),
                    Err(_) => match self.victim(None) {
                        Some(id) => self.evict(family, id)?,
                        None => return Ok(None),
                    },
                }
            }
        } else {
            None
        };
        if self.pool.free() < need + mark_pages {
            self.give_back(family, slot.map(Mark::Slot))?;
            return Ok(None);
        }
        let pages = self.pool.alloc(need)?;
        let mark = match slot {
            Some(slot) => Some(Mark::Slot(slot)),
            None => self.take_mark_pages()?,
        };
        // The restore stream writes these pages and the mark: nothing queued may still use them.
        self.drain(family)?;
        let ids = self.identities(&snapshot_tokens, &saved_media, &pages);
        let mut target_pages: [Vec<DevicePage>; cuteafd_hostcache::COMPRESSORS] = Default::default();
        target_pages[0] = pages.iter().zip(ids).map(|(&page, id)| DevicePage { id, segments: family.page_segments(page) }).collect();
        let tail = match Self::mark_segments(family, mark.as_ref()) {
            Ok(tail) => tail,
            Err(error) => {
                self.release(family, &pages)?;
                self.give_back(family, mark)?;
                return Err(error);
            }
        };
        let target = RestoreTarget {
            pages: target_pages,
            tail,
            draft: None,
            scores: Vec::new(),
        };
        let outcome = self.host.as_mut().expect("checked").restore(hit.key, &target);
        let restored = match outcome {
            RestoreOutcome::Done { .. } => family.host_restored(&pages)
                .map_err(|error| { tracing::warn!(target: "cuteafd::prefix", %error, "host restore publication failed"); }),
            outcome => {
                tracing::warn!(target: "cuteafd::prefix", ?outcome, tokens = len, "host restore did not complete");
                Err(())
            }
        };
        match restored {
            Ok(()) => {}
            Err(()) => {
                // Both timeout and submission errors may leave copies queued.
                // On barrier failure keep the pool refs/mark allocated and the
                // host snapshot pinned, instead of handing live targets back.
                let host = self.host.as_mut().expect("checked");
                host.engine_mut().release_barrier(Stream::Restore)
                    .map_err(|error| PrefixError::Host(format!("host restore release barrier: {error:#}")))?;
                tracing::warn!(target: "cuteafd::prefix", tokens = len, "host restore abandoned; prefilling");
                self.stats.restore_failures += 1;
                self.release(family, &pages)?;
                self.give_back(family, mark)?;
                return Ok(None);
            }
        }
        self.clock += 1;
        let id = self.next_id;
        self.next_id += 1;
        self.entries.insert(id, Entry { session, tokens: snapshot_tokens.clone(), media: saved_media, kind: hit.kind, pages, mark, after,
            meta, last_use: self.clock, ticket: None });
        if let Some(evicted) = self.retained.bank_mut(hit.kind).insert(&snapshot_tokens, id) {
            self.evict(family, evicted)?;
        }
        self.stats.promotions += 1;
        Ok(self.lookup_media(tokens, media, sampled))
    }
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(a, b)| a == b).count()
}

fn media_resume(rule: cuteafd_core::prefix::ReuseRule, tokens: &[u32], media: &[MediaSpan],
    saved_tokens: &[u32], saved_media: &[MediaSpan], after: &After, sampled: bool) -> (usize, usize, bool) {
    let mut common = common_prefix(tokens, saved_tokens);
    let mut resume = rule.skipped(common, saved_tokens.len());
    if resume == tokens.len() && !after.serves(sampled) {
        common = common.saturating_sub(1);
        resume = rule.skipped(common, saved_tokens.len());
    }
    // Apply the family's alignment/replay first, then image atomicity. A rule's partial
    // restore starts empty, so rounding further down is safe (never claim an exact mark).
    // `common` stays raw: the family's plan picks its frontiers from it, and `restore_hit`
    // verifies both against the media of either prompt before forking a row.
    resume = round_frontier(round_frontier(resume, media), saved_media);
    if !verify_media(resume, saved_media, media) { return (0, common, true); }
    (resume, common, false)
}
