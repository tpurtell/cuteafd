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
//! (`cuteafd_core::prefix::Retention`, prompts and turns in bounded banks), on both tiers: the
//! candidate that leaves the fewest rows to prefill or replay wins, an exact restore on a tie.
//!
//! A host-enabled mark family keeps a retained entry's positional mark only on the host
//! (`Entry::host_mark`) once its copy landed. That host snapshot is pinned while the device entry
//! lives, so host pressure evicts host-only snapshots and never orphans live device pages; the
//! pinned bytes stay inside the host quota. If pinned snapshots fill the quota, older idle
//! host-backed device entries release their pins so the host tier can make room. If none can
//! be retired, the new entry keeps its device staging mark until the staging ring needs it.
//! A mark-less entry (staging reaped an uncached mark, or, as an invariant violation counted in
//! `PrefixStats::dead_marks`, its host mark is gone) is handled by the rule: under an approximate
//! partial rule (V4.1, MiMo `--prefix-partial on`) its exact pages stay and it restores by bounded
//! replay from just short of its frontier (`markless_replays`); otherwise it is evicted and
//! selection retries the next exact candidate.
//!
//! Approximate lineage is scoped to a row ([`Entry::approximate_from`]): a placement restored by
//! approximate replay, or from a snapshot that carries such lineage, may differ from an exact
//! prefill from that row on (and in all of its positional state). Pages wholly before it keep
//! token-content host identities; a later partial restore that shares only pages before it is
//! not tainted by it. Eviction is least
//! recently used across both banks, prompt before turn at equal use ([`victim`]); `make_room`
//! runs before every admission and capture that needs pages. A restore that cannot complete is a
//! cache miss, never a request error.
use super::chain::{content_id, page_chain_media};
use crate::media::{round_frontier, snapshot_media, verify_media, MediaError, MediaSpan};
use super::entry::{victim, After, Entry, EntryId, Mark};
use super::family::{BoxError, CaptureTicket, Eviction, FamilyLayout, MarkStore, PageOwners, PrefixFamily, RestoreCandidate,
    RestoreContext, RestoreFidelity, RestorePlan};
use super::marks::MarkArena;
use super::pages::{Need, PoolExhausted, RefPagePool};
use cuteafd_core::prefix::{Retention, SnapshotKind};
use cuteafd_hostcache::cache::{
    DevicePage, DeviceSnapshot, EvictDecision, HostCache, RestoreOutcome, RestoreTarget, SkipReason, StoreOutcome,
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
    #[error("draft {0}")]
    DraftPages(PoolExhausted),
    #[error("family {what} failed: {source}")]
    Family {
        what: &'static str,
        #[source]
        source: BoxError,
    },
    #[error("prefix copy drain failed: {source}")]
    Drain { #[source] source: BoxError },
    #[error("prefix restore rollback failed: {source}")]
    Rollback { #[source] source: BoxError },
    #[error("snapshot of {tokens} tokens, but the placement committed {committed} (reach {reach})")]
    Frontier { tokens: usize, committed: usize, reach: usize },
    #[error(transparent)]
    Media(#[from] MediaError),
    #[error("host tier: {0}")]
    Host(String),
    #[error("family layout: {0}")]
    Layout(&'static str),
    /// A selected entry had no mark on either tier; it was evicted (admission retries selection).
    #[error("retained snapshot lost its positional mark")]
    DeadMark,
}

fn draft_exhaustion(mut error: &(dyn std::error::Error + 'static)) -> Option<PoolExhausted> {
    loop {
        if let Some(super::DraftPageError::Exhausted(pages)) = error.downcast_ref() { return Some(*pages); }
        error = error.source()?;
    }
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
    /// Restored pages or positional state from the host tier by this admission.
    pub host: bool,
    /// A divergent match, not a retained ancestor. Numerical fidelity is separate.
    pub partial: bool,
    /// The family's restore plan; its fidelity covers inherited lineage too.
    pub plan: RestorePlan,
    /// First row whose restored state may differ from an exact prefill (positional state is then
    /// approximate throughout); `None` for an exact restore.
    pub approximate_from: Option<usize>,
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
#[derive(Clone)]
pub struct HostPayload<M = ()> {
    pub record_format: &'static str,
    pub session: Option<String>,
    pub after: After,
    pub media: Vec<MediaSpan>,
    /// The family's snapshot metadata ([`PrefixFamily::capture_meta`]).
    pub family: M,
    /// The snapshot's approximate lineage ([`PrefixFamily::approximate_from`]).
    pub approximate_from: Option<usize>,
    pub draft: Option<super::DraftPageAttachment>,
}

/// Counters and gauges for `/v1/stats`.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct PrefixStats {
    pub lookups: u64,
    /// Rejected radix matches whose full image identities differ.
    pub media_key_collisions: u64,
    pub hits: u64,
    /// Hits that restored the whole prompt with exact fidelity.
    pub exact_hits: u64,
    /// Hits on a divergent match, including exactly reusable paged state.
    pub partial_hits: u64,
    /// Restores whose own plan replays approximately (a partial match under a replay rule).
    pub approximate_replay_hits: u64,
    /// Restores that replay nothing themselves but reuse state descended from an approximate
    /// replay (pages from its taint row on, or its positional mark); labelled ApproximateReplay.
    pub approximate_lineage_hits: u64,
    pub hit_tokens: u64,
    /// Host-tier restores: a lookup that re-inserted the entry from the host cache
    /// (`host_mark: Some(..)`). Not [`Self::mark_promotions`], which counts host mark
    /// copies that do not copy resident pages.
    ///
    /// The wire key stays `promotions` so existing `/v1/stats` consumers do not break.
    #[serde(rename = "promotions")]
    pub host_promotions: u64,
    pub device_hits: u64,
    pub host_hits: u64,
    /// Host mark copies without copying already-resident KV pages.
    pub mark_promotions: u64,
    pub capture_stalls: u64,
    pub capture_stall_ns: u64,
    pub restore_failures: u64,
    /// Hits dropped because the pool could not hold the fork.
    pub fork_no_room: u64,
    /// Device entries found holding no mark anywhere (their host-only mark was gone). An
    /// invariant violation: host marks of live entries are pinned. The entry is evicted and
    /// selection retried, never served approximately.
    pub dead_marks: u64,
    /// Restores of a retained entry without a positional mark by bounded replay from its own
    /// pages (approximate partial rules); counted in `approximate_replay_hits` too.
    pub markless_replays: u64,
    /// Uncached staging marks dropped while their entry's pages stayed (approximate partial rules).
    pub marks_dropped: u64,
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
    /// Allocated pages per GPU half (parity pools only).
    pub pages_used_by_gpu: Option<[usize; 2]>,
    pub pages_shared: usize,
    /// Distinct pages held by retained snapshots; when no request runs, every used page is one.
    pub pages_retained: usize,
    pub mark_slots: usize,
    /// Retained marks, in arena slots or pool pages.
    pub marks_in_use: usize,
    /// Pool pages held by marks (counted in `pages_retained` too).
    pub mark_pages: usize,
    /// Retained snapshots whose mark lives only on the host tier (its device staging slot was
    /// handed back once the host copy landed). At idle, a mark family's
    /// `marks_in_use + marks_on_host` equals its retained entries.
    pub marks_on_host: usize,
    /// The family's snapshots carry a positional mark (pages-only families: none).
    pub mark_family: bool,
    /// Retained device entries whose only mark is a host snapshot that is no longer resident.
    /// Always 0 while host marks of live entries are pinned.
    pub marks_lost: usize,
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
    copies: Vec<super::pages::TailCopy>,
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
    _draft: Option<super::DraftPageAttachment>,
}

pub struct PrefixCache<E: CopyEngine, M = ()> {
    layout: FamilyLayout,
    config: PrefixConfig,
    retained: Retention<EntryId>,
    entries: BTreeMap<EntryId, Entry<M>>,
    pending: Vec<PendingCapture<M>>,
    completed_captures: Vec<CaptureTicket>,
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
    serving_detail: std::cell::RefCell<Option<(std::time::Instant, [usize; 5])>>,
}

impl<E: CopyEngine, M: Clone + Default> PrefixCache<E, M> {
    /// `host`: the pinned host tier's knobs and copy engine (`None`, or a zero `bytes`, keeps it off).
    pub fn new(layout: FamilyLayout, config: PrefixConfig, host: Option<(cuteafd_hostcache::config::Config, E)>)
        -> Result<Self, PrefixError> {
        Self::new_with_draft_pages(layout, config, host, 0)
    }

    pub fn new_with_draft_pages(layout: FamilyLayout, config: PrefixConfig,
        host: Option<(cuteafd_hostcache::config::Config, E)>, draft_page_bytes: usize)
        -> Result<Self, PrefixError> {
        // Before the host tier pins its memory.
        if layout.page_owners == PageOwners::Parity && (layout.pages % 2 != 0 || layout.mark_store != MarkStore::Arena) {
            return Err(PrefixError::Layout("parity half pools need an even page count and arena marks"));
        }
        if let MarkStore::Pool { pages, reserved } = layout.mark_store {
            if layout.mark_bytes == 0 || pages == 0 || pages > layout.pages.saturating_sub(reserved) {
                return Err(PrefixError::Layout("pool-page marks take at least one page and at most the pool's \
                    unreserved pages"));
            }
        }
        let host = match host {
            Some((mut host_config, engine)) if config.entries > 0 && host_config.enabled() => Some({
                // A host-enabled mark family keeps only staging slots on the device
                // (`HOST_MARK_STAGING_SLOTS`): a retained mark lives on the host or nowhere. A
                // snapshot the host tier refused as too short would lose its slot, and so the whole
                // snapshot, to the next capture (#2582: a 260-token shared base gone after one
                // warm wave). Such a family's host tier keeps everything its device keeps.
                if layout.mark_bytes > 0 {
                    host_config.min_tokens = host_config.min_tokens.min(u32::try_from(config.min_tokens.max(1)).unwrap_or(u32::MAX));
                }
                HostCache::with_rule(host_config, layout.host_layout().with_draft_pages(draft_page_bytes), engine, layout.rule, EvictionOrder::LeastRecent)
                    .map_err(|e| PrefixError::Host(format!("{e:#}")))?
            }),
            _ => None,
        };
        let arena = config.entries > 0 && layout.mark_bytes > 0 && layout.mark_store == MarkStore::Arena;
        let slots = if arena { config.mark_slots } else { 0 };
        Ok(Self {
            retained: Retention::with_rule(config.entries, layout.rule),
            entries: BTreeMap::new(),
            pending: Vec::new(),
            completed_captures: Vec::new(),
            quarantine: Vec::new(),
            pool: match layout.page_owners {
                PageOwners::Uniform => RefPagePool::with_reserved(layout.pages, layout.page_rows, layout.mark_store.reserved()),
                PageOwners::Parity => RefPagePool::parity(layout.pages, layout.page_rows, 0),
            },
            arena: MarkArena::new(slots, layout.mark_bytes),
            host,
            clock: 0,
            next_id: 0,
            dirty: false,
            capture_session: None,
            restored_session: None,
            stats: PrefixStats::default(),
            serving_detail: std::cell::RefCell::new(None),
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
        self.tick();
        match &mut self.host {
            Some(host) => {
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
    /// Host keys holding live entries' only marks, oldest entry first (test support).
    #[doc(hidden)]
    pub fn host_mark_keys(&self) -> Vec<cuteafd_hostcache::snapshot::Key> {
        self.entries.values().filter(|e| e.mark.is_none()).filter_map(|e| e.host_mark).collect()
    }
    /// Drop every device-resident mark as staging pressure does for uncached writes (test support).
    #[doc(hidden)]
    pub fn drop_marks_for_test<F: PrefixFamily<M>>(&mut self, family: &F) -> usize {
        let ids: Vec<_> = self.entries.iter().filter(|(_, e)| e.mark.is_some()).map(|(&id, _)| id).collect();
        for &id in &ids {
            let mark = self.entries.get_mut(&id).and_then(|e| e.mark.take());
            self.give_back(family, mark).expect("drained");
            self.stats.marks_dropped += 1;
        }
        ids.len()
    }
    #[doc(hidden)]
    pub fn host_mut(&mut self) -> Option<&mut HostCache<E, HostPayload<M>>> { self.host.as_mut() }

    /// Read the already-present host tier; absence never initializes one.
    pub fn host_engine(&self) -> Option<&E> {
        self.host.as_ref().map(HostCache::engine)
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
        native: &[u32], media: &[MediaSpan], capacity: usize, sampled: bool, build: impl FnMut(Vec<u32>) -> P)
        -> Result<Admitted<P>, PrefixError> {
        self.admit_native_compatible(family, tokens, native, media, capacity, sampled, None, build)
    }

    /// Request-local state requirements are checked before either tier restores bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn admit_native_compatible<F: PrefixFamily<M, Placement = P>, P>(&mut self, family: &F,
        tokens: &[u32], native: &[u32], media: &[MediaSpan], capacity: usize, sampled: bool,
        compatible: Option<&dyn Fn(&M, usize) -> bool>, mut build: impl FnMut(Vec<u32>) -> P)
        -> Result<Admitted<P>, PrefixError> {
        crate::media::keys::validate_spans(media)?;
        if media.last().is_some_and(|s| s.checked_end().unwrap() > tokens.len()) || native.len() != tokens.len() {
            return Err(MediaError::Spans.into());
        }
        self.restored_session = None;
        self.tick();
        self.reconcile_draft(family)?;
        let total = self.pool.pages_for(capacity.max(tokens.len()));
        let mut lookups = if self.enabled() && !tokens.is_empty() { 1 + self.entries.len() } else { 0 };
        if lookups > 0 { self.stats.lookups += 1; }
        // A dead mark evicts its entry and selects again (bounded by the retained entries).
        while lookups > 0 {
            lookups -= 1;
            let mut promoted = false;
            let mut hit = if let Some(compatible) = compatible {
                self.device_compatible_hit(tokens, media, sampled, compatible)
            } else { self.lookup_media(tokens, media, sampled) };
            // A selected device entry without a mark restores by replay (or not at all).
            if let Some(h) = hit.as_ref() {
                match self.check_mark(family, h.id) {
                    Err(PrefixError::DeadMark) => continue,
                    result => result?,
                }
            }
            hit = hit.and_then(|h| self.markless_hit(h, tokens, media));
            let rule = self.layout.rule;
            let device_rank = hit.as_ref().map_or((0, false), |h| rule.rank(h.common, h.frontier, h.resume));
            // A host snapshot that ranks above the device hit under the same rule (rows saved,
            // with the ancestor slack of an approximate partial rule) is promoted. Exact-only
            // families keep their device hit unless a compatibility check applies.
            let host_better = (compatible.is_some() || rule.approximate_partial()) && hit.is_some()
                && self.host_verified_hit(tokens, media, sampled, compatible)
                    .is_some_and(|(host, resume)| rule.rank(host.common, host.frontier, resume) > device_rank);
            if hit.is_none() || host_better {
                let device = hit.take();
                let restored = self.promote(family, tokens, media, sampled, compatible)?;
                promoted = restored.is_some();
                // Promotion can evict the previous device candidate before failing: fall back to
                // it only while it is still retained.
                hit = if promoted { restored }
                else if let Some(compatible) = compatible {
                    self.device_compatible_hit(tokens, media, sampled, compatible)
                } else { device.filter(|h| self.entries.contains_key(&h.id)) };
            }
            if let Some(hit) = hit {
                if compatible.is_some() {
                    self.clock += 1;
                    if let Some(entry) = self.entries.get_mut(&hit.id) { entry.last_use = self.clock; }
                }
                let context = RestoreContext { native_tokens: native, media, approximate_from: None };
                let result = self.restore_hit(family, &hit, tokens, &context, total, capacity.max(tokens.len()), promoted, &mut build);
                if promoted {
                    let mark = self.entries.get_mut(&hit.id).and_then(|e| e.mark.take());
                    self.give_back(family, mark)?;
                }
                match result {
                    Ok(Some(admitted)) => return Ok(admitted),
                    Ok(None) => {},
                    Err(PrefixError::DeadMark) => continue,
                    Err(PrefixError::Family { what, source }) => {
                        tracing::warn!(target: "cuteafd::prefix", what, error = %source, "prefix restore abandoned; prefilling");
                        self.stats.restore_failures += 1;
                    }
                    Err(error) => return Err(error),
                }
            }
            break;
        }
        self.admit_cold(family, tokens.len(), capacity, build)
    }

    /// Bypass prefix reuse after repeated peek/admission races. The caller has materialized
    /// the entire prompt's media before invoking this allocator.
    pub fn admit_cold<F: PrefixFamily<M, Placement = P>, P>(&mut self, family: &F, tokens: usize,
        capacity: usize, build: impl FnOnce(Vec<u32>) -> P) -> Result<Admitted<P>, PrefixError> {
        let total = self.pool.pages_for(capacity.max(tokens));
        self.make_room_for(family, self.pool.need_for(0..total), None)?;
        let pages = self.pool.alloc_for(0..total)?;
        let mut placement = build(pages.clone());
        let admitted = family.bind(&mut placement)
            .map_err(|source| PrefixError::Family { what: "bind", source })
            .and_then(|()| self.admit_draft(family, &mut placement, capacity.max(tokens)));
        if let Err(error) = admitted {
            // A failing bind may already have queued writes; successful cold admission
            // retains the existing no-copy/no-extra-drain hot path.
            self.dirty = true;
            self.rollback(family, &mut placement, pages, None, error)?;
            unreachable!("rollback always returns the admission error");
        }
        Ok(Admitted { placement, resume: 0, after: None, source: None })
    }

    /// Undo every stage of a failed admission before retrying cold. Only the original family
    /// operation is recoverable; an unproved drain or reset quarantines the references forever.
    fn rollback<F: PrefixFamily<M>>(&mut self, family: &F, placement: &mut F::Placement,
        pages: Vec<u32>, temporary_mark: Option<Mark>, error: PrefixError) -> Result<(), PrefixError> {
        let drained = if matches!(error, PrefixError::Drain { .. }) { Err(error) }
            else { family.drain_rollback().map(|()| { self.dirty = false; error })
                .map_err(|source| PrefixError::Drain { source }) };
        let (error, failure) = match drained {
            Ok(error) => (Some(error), family.discard(placement).err()),
            Err(error) => (None, Some(Box::new(error) as BoxError)),
        };
        if let Some(source) = failure {
            tracing::warn!(target: "cuteafd::prefix", slot = ?family.placement_slot(placement),
                pages = pages.len(), cause = %source, "prefix rollback quarantined until restart");
            family.quarantine_draft_pages(placement);
            family.quarantine(placement);
            self.quarantine.push(Quarantined { _pages: pages, _mark: temporary_mark, _draft: None });
            self.stats.quarantined += 1;
            return Err(PrefixError::Rollback { source });
        }
        self.release(family, &pages)?;
        self.give_back(family, temporary_mark)?;
        Err(error.expect("successful drain preserved the admission error"))
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
        let rule = self.layout.rule;
        // (rank, resume): the better-ranked tier and the rows it saves.
        let none = ((0, false), 0);
        let device = if media.is_empty() {
            let mut query = tokens;
            loop {
                let Some((common, frontier, &id)) = self.retained.peek_reusable(query) else { break none };
                let Some(entry) = self.entries.get(&id) else { break none };
                let resume = self.layout.rule.skipped(common, frontier);
                if !verify_media(resume, &entry.media, media) { break none; }
                if resume == tokens.len() && !entry.after.serves(sampled) {
                    if query.len() < tokens.len() || tokens.len() < 2 { break none; }
                    query = &tokens[..tokens.len() - 1];
                } else { break (rule.rank(common, frontier, resume), resume); }
            }
        } else {
            self.device_media_hit(tokens, media, sampled).0.map_or(none, |h| (rule.rank(h.common, h.frontier, h.resume), h.resume))
        };
        let host = self.host_verified_hit(tokens, media, sampled, None)
            .map_or(none, |(h, resume)| (rule.rank(h.common, h.frontier, resume), resume));
        device.max(host).1
    }

    /// A compatible ancestor may be shallower than an incompatible token match.
    pub fn peek_media_compatible(&self, tokens: &[u32], media: &[MediaSpan], sampled: bool,
        compatible: &dyn Fn(&M, usize) -> bool) -> usize {
        if !self.enabled() { return 0; }
        let rule = self.layout.rule;
        let none = ((0, false), 0);
        let device = self.device_compatible_hit(tokens, media, sampled, compatible)
            .map_or(none, |h| (rule.rank(h.common, h.frontier, h.resume), h.resume));
        let host = self.host_compatible_hit(tokens, media, sampled, compatible)
            .map_or(none, |(h, resume)| (rule.rank(h.common, h.frontier, resume), resume));
        device.max(host).1
    }

    fn device_compatible_hit(&self, tokens: &[u32], media: &[MediaSpan], sampled: bool,
        compatible: &dyn Fn(&M, usize) -> bool) -> Option<Hit> {
        self.entries.iter().filter_map(|(&id, entry)| {
            let (resume, common, _) = media_resume(self.layout.rule, tokens, media, &entry.tokens,
                &entry.media, &entry.after, sampled);
            (resume > 0 && compatible(&entry.meta, entry.len())).then_some((
                (self.layout.rule.rank(common, entry.len(), resume), entry.kind == SnapshotKind::Turn, entry.last_use),
                Hit { id, kind: entry.kind, resume, common, frontier: entry.len() }))
        }).max_by_key(|(rank, _)| *rank).map(|(_, hit)| hit)
    }

    fn host_compatible_hit(&self, tokens: &[u32], media: &[MediaSpan], sampled: bool,
        compatible: &dyn Fn(&M, usize) -> bool) -> Option<(cuteafd_hostcache::snapshot::Hit, usize)> {
        self.host.as_ref()?.resident_payloads().filter_map(|(meta, key, payload)| {
            let (resume, common, _) = media_resume(self.layout.rule, tokens, media, &meta.tokens,
                &payload.media, &payload.after, sampled);
            (resume > 0 && compatible(&payload.family, meta.tokens.len())).then_some((
                (self.layout.rule.rank(common, meta.tokens.len(), resume), meta.kind == SnapshotKind::Turn, key),
                (cuteafd_hostcache::snapshot::Hit { key, kind: meta.kind, common,
                    frontier: meta.tokens.len() }, resume)))
        }).max_by_key(|(rank, _)| *rank).map(|(_, hit)| hit)
    }

    fn device_media_hit(&self, tokens: &[u32], media: &[MediaSpan], sampled: bool) -> (Option<Hit>, u64) {
        let mut best = None;
        let mut rank = ((0, false), false, 0);
        let mut collisions = 0;
        for (&id, entry) in &self.entries {
            let (resume, common, collision) = media_resume(self.layout.rule, tokens, media, &entry.tokens,
                &entry.media, &entry.after, sampled);
            collisions += u64::from(collision);
            let candidate = (self.layout.rule.rank(common, entry.len(), resume), entry.kind == SnapshotKind::Turn, entry.last_use);
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
        let mut rank = ((0, false), false, 0);
        let mut collisions = 0;
        for (meta, key, payload) in host.resident_payloads() {
            let (resume, common, collision) = media_resume(self.layout.rule, tokens, media, &meta.tokens,
                &payload.media, &payload.after, sampled);
            collisions += u64::from(collision);
            let candidate = (self.layout.rule.rank(common, meta.tokens.len(), resume), meta.kind == SnapshotKind::Turn, key);
            if resume > 0 && candidate > rank {
                rank = candidate;
                best = Some((cuteafd_hostcache::snapshot::Hit {
                    key, kind: meta.kind, common, frontier: meta.tokens.len(),
                }, resume));
            }
        }
        (best, collisions)
    }

    fn host_verified_hit(&self, tokens: &[u32], media: &[MediaSpan], sampled: bool,
        compatible: Option<&dyn Fn(&M, usize) -> bool>) -> Option<(cuteafd_hostcache::snapshot::Hit, usize)> {
        if let Some(compatible) = compatible {
            return self.host_compatible_hit(tokens, media, sampled, compatible);
        }
        if !media.is_empty() { return self.host_media_hit(tokens, media, sampled).0; }
        let host = self.host.as_ref()?;
        let hit = host.peek(tokens)?;
        let payload = host.payload(hit.key)?;
        let resume = self.layout.rule.skipped(hit.common, hit.frontier);
        (resume > 0 && verify_media(resume, &payload.media, media)
            && (resume != tokens.len() || payload.after.serves(sampled))).then_some((hit, resume))
    }

    #[allow(clippy::too_many_arguments)]
    fn restore_hit<F: PrefixFamily<M, Placement = P>, P>(&mut self, family: &F, hit: &Hit, tokens: &[u32],
        context: &RestoreContext<'_>, total: usize, draft_capacity: usize, promoted: bool, build: &mut impl FnMut(Vec<u32>) -> P)
        -> Result<Option<Admitted<P>>, PrefixError> {
        let saved_media = self.entries[&hit.id].media.clone();
        let mut plan = family.plan_restore(RestoreCandidate { common: hit.common, snapshot_end: hit.frontier,
            target_end: tokens.len(), resume: hit.resume, media: context.media, saved_media: &saved_media });
        // Media atomicity holds at both frontiers: shared rows never end inside an image, and a
        // replay never starts inside one. An exact plan is verified at the snapshot's frontier.
        let media_safe = |at: usize| round_frontier(at, context.media) == at && round_frontier(at, &saved_media) == at;
        let checked = plan.check(hit.common).map_err(|e| PrefixError::Family { what: "plan", source: e.into() })
            .and_then(|()| if plan.restores_mark() != (hit.common == hit.frontier)
                || (!plan.restores_mark() && self.layout.rule.replay.is_some() && plan.exact())
                || hit.common < plan.lag.source_end
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
        let writable = family.write_frontier(source_end);
        let need = self.pool.fork_need_writable(source_end, total, writable);
        if !self.make_room_for(family, need, Some(hit.id))? {
            self.stats.fork_no_room += 1;
            return Ok(None);
        }
        let entry = self.entries.get(&hit.id).expect("the hit is kept while making room");
        let (fork_pages, copies) = self.pool.fork_writable(&entry.pages, source_end, total, writable)?;
        let ancestor = plan.restores_mark();
        // Scoped lineage: an ancestor restore reuses the snapshot's positional state, so it
        // inherits any lineage; a partial restore inherits it only when it shares a page row
        // at or past the snapshot's taint row. Its own replay taints rows from `source_end`.
        let replays = !plan.exact();
        let inherited = entry.approximate_from.filter(|&from| ancestor || from < source_end);
        let approximate_from = inherited.into_iter().chain(replays.then_some(source_end)).min();
        if approximate_from.is_some() { plan.fidelity = RestoreFidelity::ApproximateReplay; }
        let (mut mark, meta) = if ancestor { (entry.mark.clone(), Some(entry.meta.clone())) } else { (None, None) };
        let host_mark = if ancestor && self.layout.mark_bytes > 0 && mark.is_none() { entry.host_mark } else { None };
        let after = (resume == tokens.len()).then(|| entry.after.clone());
        let session = entry.session.clone();
        let draft = ancestor.then(|| entry.draft.clone()).flatten()
            .filter(|draft| family.draft_pages_valid(draft));
        let pages = fork_pages.clone();
        let temporary = host_mark.is_some();
        if let Some(key) = host_mark {
            mark = match self.restore_mark(family, key, hit.id) {
                Ok(mark) => mark,
                Err(error) => { self.release(family, &pages)?; return Err(error); }
            };
            if mark.is_none() {
                self.release(family, &pages)?;
                if !self.host.as_ref().is_some_and(|h| h.payload(key).is_some()) {
                    // The pinned host mark is gone: an invariant violation (pins prevent it).
                    // Under a replay rule the entry restores from its pages; otherwise drop it so
                    // selection finds the next exact one.
                    debug_assert!(false, "a live device entry lost its pinned host mark");
                    tracing::error!(target: "cuteafd::prefix", tokens = hit.frontier, "device entry lost its host mark");
                    self.stats.dead_marks += 1;
                    if let Some(entry) = self.entries.get_mut(&hit.id) { entry.host_mark = None; }
                    if !self.layout.rule.approximate_partial() { self.evict(family, hit.id)?; }
                    return Err(PrefixError::DeadMark);
                }
                return Ok(None);
            }
        }
        let mut placement = build(fork_pages);
        self.dirty = true;
        let restored = self.check_owners(family, &pages).map_err(|e| ("owners", e))
            .and_then(|()| family.bind(&mut placement).map_err(|e| ("bind", e)))
            .and_then(|()| copies.into_iter().try_for_each(|copy| {
                self.stats.cow_copies += 1;
                family.copy_rows(copy).map_err(|e| ("copy", e))
            }))
            .and_then(|()| family.restore_with(mark.as_ref(), meta.as_ref(), &mut placement, &plan,
                &RestoreContext { approximate_from, ..*context })
                .map_err(|e| ("restore", e)))
            .and_then(|()| match family.restore_draft_pages(draft.as_ref(), &mut placement, resume) {
                Err(error) if draft.is_some() && draft_exhaustion(error.as_ref()).is_some() =>
                    family.restore_draft_pages(None, &mut placement, resume)
                        .map_err(|e| ("restore draft pages cold", e)),
                result => result.map_err(|e| ("restore draft pages", e)),
            });
        let restored = restored.map_err(|(what, source)| PrefixError::Family { what, source });
        let restored = restored.and_then(|()| {
            let mut admitted = self.admit_draft(family, &mut placement, draft_capacity);
            if matches!(admitted, Err(PrefixError::DraftPages(_))) && draft.is_some() {
                // Optional retained draft history must not reject an otherwise admissible hit.
                self.drain(family)?;
                family.restore_draft_pages(None, &mut placement, resume).map_err(|source|
                    PrefixError::Family { what: "drop unadmitted draft history", source })?;
                admitted = self.admit_draft(family, &mut placement, draft_capacity);
            }
            admitted?;
            if self.host.is_some() && mark.is_some() { self.drain(family)?; }
            Ok(())
        });
        if let Err(error) = restored {
            self.rollback(family, &mut placement, pages, if temporary { mark } else { None }, error)?;
            unreachable!("rollback always returns the admission error");
        }
        if temporary { self.give_back(family, mark)?; }
        let host_hit = promoted || temporary;
        self.stats.host_hits += u64::from(host_hit);
        self.stats.device_hits += u64::from(!host_hit);
        self.stats.hits += 1;
        self.stats.hit_tokens += resume as u64;
        self.stats.exact_hits += u64::from(after.is_some() && plan.exact());
        self.stats.partial_hits += u64::from(!ancestor);
        self.stats.approximate_replay_hits += u64::from(replays);
        self.stats.markless_replays += u64::from(replays && hit.common + 1 == hit.frontier
            && self.entries.get(&hit.id).is_some_and(|e| e.mark.is_none() && e.host_mark.is_none()));
        self.stats.approximate_lineage_hits += u64::from(!replays && approximate_from.is_some());
        self.restored_session = session.clone();
        Ok(Some(Admitted {
            placement,
            resume,
            after,
            source: Some(Source { session, kind: hit.kind, frontier: hit.frontier, host: host_hit, partial: !ancestor,
                plan, approximate_from }),
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
        let Prepared { entry, copies } = prepared;
        let len = entry.len();
        let captured = copies.into_iter().try_for_each(|copy| family.copy_rows(copy))
        .and_then(|()| match &entry.mark {
            Some(Mark::Pages(pages)) => family.capture_pages(pages, placement, len),
            Some(Mark::Slot(slot)) => family.capture(*slot, placement, len),
            None => Ok(()),
        });
        if let Err(source) = captured {
            self.release(family, &entry.pages)?;
            self.release_draft(family, entry.draft.as_ref())?;
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
        let Some(Prepared { entry, copies }) = self.prepare_capture(family, kind, tokens, media, placement, after)? else {
            return Ok(Captured::Skipped);
        };
        let Some(Mark::Slot(slot)) = entry.mark else { unreachable!("arena marks take a slot") };
        // Existing queued families accept one tail. Multiple writable retained units use
        // their ordinary ordered capture stream without changing the asynchronous contract.
        let queued = if copies.len() <= 1 {
            family.queue_capture(slot, placement, entry.len(), copies.first().copied())
        } else { Ok(None) };
        match queued {
            Ok(Some(ticket)) => {
                self.pending.push(PendingCapture { ticket, kind, entry });
                self.stats.captures_queued += 1;
                Ok(Captured::Queued(ticket))
            }
            Ok(None) => {
                // The family captures synchronously: do it now on its own stream.
                let len = entry.len();
                let captured = copies.into_iter().try_for_each(|copy| family.copy_rows(copy)).and_then(|()| family.capture(slot, placement, len));
                if let Err(source) = captured {
                    self.release(family, &entry.pages)?;
                    self.release_draft(family, entry.draft.as_ref())?;
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
                        self.release_draft(family, entry.draft.as_ref())?;
                        self.give_back(family, entry.mark)?;
                    }
                    Err(error) => {
                        tracing::error!(target: "cuteafd::prefix", %error, "capture enqueue failed and did not drain; quarantining");
                        self.quarantine.push(Quarantined { _pages: entry.pages, _mark: entry.mark, _draft: entry.draft });
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
        if let Some(index) = self.completed_captures.iter().position(|&t| t == ticket) {
            self.completed_captures.swap_remove(index);
            return Ok(true);
        }
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
        let PendingCapture { kind, entry, .. } = self.pending.remove(index);
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
        self.completed_captures.retain(|&done| done != ticket);
        let Some(index) = self.pending.iter().position(|p| p.ticket == ticket) else { return Ok(()) };
        let PendingCapture { entry, .. } = self.pending.remove(index);
        self.stats.captures_aborted += 1;
        match family.abort_capture(ticket) {
            Ok(()) => {
                self.release(family, &entry.pages)?;
                self.release_draft(family, entry.draft.as_ref())?;
                self.give_back(family, entry.mark)
            }
            Err(error) => {
                tracing::error!(target: "cuteafd::prefix", %error, "queued capture did not drain; quarantining its storage");
                self.quarantine.push(Quarantined { _pages: entry.pages, _mark: entry.mark, _draft: entry.draft });
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
        // A snapshot no longer than its taint row still holds approximate positional state.
        let approximate_from = family.approximate_from(placement)
            .map_err(|source| PrefixError::Family { what: "capture lineage", source })?.map(|from| from.min(len));
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
        self.reap_marks(family)?;
        let slot = if self.arena_marks() {
            loop {
                match self.arena.take() {
                    Ok(slot) => break Some(slot),
                    Err(_) => match self.mark_victim() {
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
        let mark_pages = self.pool.need_for(0..self.layout.mark_store.pages());
        let writable = family.write_frontier(len);
        if !self.make_room_for(family, self.pool.fork_need_writable(len, total, writable) + mark_pages, None)? {
            self.give_back(family, slot.map(Mark::Slot))?;
            self.stats.capture_skips += 1;
            return Ok(None);
        }
        let mark = match slot {
            Some(slot) => Some(Mark::Slot(slot)),
            None => self.take_mark_pages()?,
        };
        let (fork_pages, copies) = match self.pool.fork_writable(family.pages(placement), len, total, writable) {
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
        self.reconcile_draft(family)?;
        let draft = match self.capture_draft(family, placement, len) {
            Ok(draft) => draft,
            Err(source) => {
                self.release(family, &fork_pages)?;
                self.give_back(family, mark)?;
                return Err(PrefixError::Family { what: "capture draft pages", source });
            }
        };
        let entry = Entry { session: self.capture_session.clone(), tokens: tokens.to_vec(),
            media: snapshot_media(len, media), kind, pages: fork_pages, mark, after, meta, approximate_from, draft,
            last_use: 0, ticket: None, host_mark: None };
        Ok(Some(Prepared { entry, copies }))
    }

    /// Retain a captured snapshot whose copies were enqueued (synchronous families) or landed.
    fn publish<F: PrefixFamily<M>>(&mut self, family: &F, kind: SnapshotKind, mut entry: Entry<M>)
        -> Result<(), PrefixError> {
        // Target and draft copies become lookup-visible as one retired transaction.
        if entry.draft.is_some() {
            if let Err(error) = self.drain(family) {
                self.quarantine.push(Quarantined { _pages: entry.pages, _mark: entry.mark, _draft: entry.draft });
                self.stats.quarantined += 1;
                return Err(error);
            }
        }
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
        self.host_store(family, id)?;
        self.reconcile_draft(family)
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
        self.make_room_for(family, self.pool.need_for(0..pages), keep)
    }

    /// [`PrefixCache::make_room`] for fresh pages per half (both halves of a parity pool must fit).
    pub fn make_room_for<F: PrefixFamily<M>>(&mut self, family: &F, need: Need, keep: Option<EntryId>)
        -> Result<bool, PrefixError> {
        let frees = family.eviction() == Eviction::FreesPages;
        while !self.pool.fits(need) {
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
        let positions = pages.len()..self.pool.pages_for(tokens).max(pages.len());
        if positions.is_empty() { return Ok(()); }
        self.make_room_for(family, self.pool.need_for(positions.clone()), None)?;
        pages.extend(self.pool.alloc_for(positions)?);
        Ok(())
    }

    /// Grow a live placement's reservation to `rows` rows (and its drafter's credit with it),
    /// evicting only inactive snapshots. `Ok(false)`, with the target pages unchanged, when the
    /// pools cannot supply them now: the request waits (stalls) and retries.
    pub fn extend<F: PrefixFamily<M>>(&mut self, family: &F, placement: &mut F::Placement, rows: usize)
        -> Result<bool, PrefixError> {
        // Draft credit first: a raised credit with no target growth is only an early reservation.
        match self.admit_draft(family, placement, rows) {
            Ok(()) => {}
            Err(PrefixError::DraftPages(_)) => return Ok(false),
            Err(error) => return Err(error),
        }
        let have = family.pages(placement).len();
        let total = self.pool.pages_for(rows);
        if total <= have { return Ok(true); }
        let need = self.pool.need_for(have..total);
        if !self.make_room_for(family, need, None)? { return Ok(false); }
        let fresh = self.pool.alloc_for(have..total)?;
        if let Err(source) = family.extend_pages(placement, &fresh) {
            self.pool.release(&fresh);
            return Err(PrefixError::Family { what: "extend pages", source });
        }
        Ok(true)
    }

    /// Capacity a reservation may still take, in the pool's own units: free pages plus pages
    /// only retained snapshots hold (what [`PrefixCache::extend`] and admission can evict), per
    /// half of a parity pool, and the drafter's reservable pages.
    pub fn room<F: PrefixFamily<M>>(&self, family: &F) -> crate::reserve::Room {
        let room = self.target_room();
        match family.draft_reservable() { Some(free) => room.with_draft(free), None => room }
    }

    /// [`PrefixCache::room`] for a family without full-context draft pages.
    pub fn target_room(&self) -> crate::reserve::Room {
        let mut halves = self.pool.free_halves();
        for page in self.snapshot_only_pages() {
            halves[if self.pool.is_parity() { (page & 1) as usize } else { 0 }] += 1;
        }
        crate::reserve::Room { halves, parity: self.pool.is_parity(), draft: None }
    }

    /// Whether every retained page is visible to [`PrefixCache::reservable`]: no capture is
    /// still queued (its pages come back as an evictable snapshot when it lands).
    pub fn settled(&self) -> bool {
        self.pending.is_empty()
    }

    /// Private append lease: share complete pages and COW the partial tail.
    /// The caller either replaces its placement and releases the old lease, or
    /// releases this lease on abort. No committed placement changes here.
    pub fn fork_step<F: PrefixFamily<M>>(&mut self, family: &F, pages: &[u32],
        frontier: usize, end: usize) -> Result<Vec<u32>, PrefixError> {
        if end < frontier || pages.len() < self.pool.pages_for(frontier) {
            return Err(PrefixError::Family { what: "fork step", source: "invalid append frontier".into() });
        }
        let total = self.pool.pages_for(end);
        let writable = family.write_frontier(frontier);
        self.make_room_for(family, self.pool.fork_need_writable(frontier, total, writable), None)?;
        let (fork_pages, copies) = self.pool.fork_writable(pages, frontier, total, writable)?;
        if !copies.is_empty() {
            if let Err(source) = copies.into_iter().try_for_each(|copy| family.copy_rows(copy)).and_then(|()| family.drain()) {
                // Draining also covers a family that failed after enqueueing.
                if let Err(source) = family.drain() {
                    self.quarantine.push(Quarantined { _pages: fork_pages, _mark: None, _draft: None });
                    self.stats.quarantined += 1;
                    return Err(PrefixError::Family { what: "fork step drain", source });
                }
                self.pool.release(&fork_pages);
                return Err(PrefixError::Family { what: "fork step copy", source });
            }
        }
        Ok(fork_pages)
    }

    /// Poll the host stream and release staging marks only after their last reader completes.
    pub fn tick(&mut self) {
        if let Some(host) = &mut self.host {
            let report = host.tick();
            for entry in self.entries.values_mut() {
                let Some(ticket) = entry.ticket else { continue };
                if report.completed.contains(&ticket) {
                    entry.ticket = None;
                    let key = report.committed.iter().find_map(|&(done, key)|
                        (done == ticket && host.payload(key).is_some()).then_some(key));
                    if let Some(key) = key.filter(|_| entry.mark.is_some()) {
                        // The host copy now holds the entry's only mark: pin it for the
                        // entry's lifetime, so host pressure cannot orphan its pages.
                        host.pin(key);
                        entry.host_mark = Some(key);
                        match entry.mark.take() {
                            Some(Mark::Slot(slot)) => self.arena.give_back(slot),
                            Some(Mark::Pages(pages)) => { self.pool.release(&pages); }
                            None => {}
                        }
                    }
                } else if report.failed.contains(&ticket) { entry.ticket = None; }
            }
        }
    }

    /// Whether `entry` can restore its positional mark (on the device, or a resident host copy).
    fn has_mark(&self, entry: &Entry<M>) -> bool {
        self.layout.mark_bytes == 0 || entry.mark.is_some()
            || entry.host_mark.is_some_and(|key| self.host.as_ref().is_some_and(|h| h.payload(key).is_some()))
    }

    /// A device entry whose host-only mark is no longer resident: an invariant violation (pins
    /// prevent it), counted and asserted. A replay rule keeps the pages (the entry replays);
    /// otherwise the entry is evicted so it cannot shadow an exact candidate.
    fn check_mark<F: PrefixFamily<M>>(&mut self, family: &F, id: EntryId) -> Result<(), PrefixError> {
        let Some(entry) = self.entries.get(&id) else { return Ok(()) };
        let Some(key) = entry.host_mark.filter(|_| entry.mark.is_none()) else { return Ok(()) };
        if self.host.as_ref().is_some_and(|h| h.payload(key).is_some()) { return Ok(()); }
        debug_assert!(false, "a live device entry lost its pinned host mark");
        tracing::error!(target: "cuteafd::prefix", tokens = entry.len(), "device entry lost its host mark");
        self.stats.dead_marks += 1;
        if self.layout.rule.approximate_partial() {
            self.entries.get_mut(&id).expect("checked").host_mark = None;
            Ok(())
        } else {
            self.evict(family, id)?;
            Err(PrefixError::DeadMark)
        }
    }

    /// An entry selected as an exact ancestor but holding no mark restores by bounded replay from
    /// its own pages: rank and plan it as a partial match one row short of its frontier.
    fn markless_hit(&self, mut hit: Hit, tokens: &[u32], media: &[MediaSpan]) -> Option<Hit> {
        let entry = self.entries.get(&hit.id)?;
        if hit.common < hit.frontier || self.has_mark(entry) { return Some(hit); }
        if !self.layout.rule.approximate_partial() { return None; }
        let common = hit.frontier - 1;
        let plan = RestorePlan::under(self.layout.rule, RestoreCandidate { common, snapshot_end: hit.frontier,
            target_end: tokens.len(), resume: self.layout.rule.skipped(common, hit.frontier), media, saved_media: &entry.media });
        let resume = plan.resume();
        (resume > 0 && verify_media(plan.lag.source_end, &entry.media, media))
            .then(|| { hit.common = common; hit.resume = resume; hit })
    }

    fn mark_victim(&self) -> Option<EntryId> {
        victim(self.entries.iter().filter(|(_, e)| e.mark.is_some()).map(|(&id, e)| (id, e.kind, e.last_use)))
    }

    fn staging_full(&self) -> bool {
        if self.arena_marks() { return self.arena.free() == 0; }
        self.layout.mark_bytes > 0 && self.entries.values().filter(|e| e.mark.is_some()).count()
            + self.pending.len() >= cuteafd_core::prefix::HOST_MARK_STAGING_SLOTS as usize
    }

    /// Only a full staging ring waits. Complete the oldest write without dropping its KV.
    fn reap_marks<F: PrefixFamily<M>>(&mut self, family: &F) -> Result<(), PrefixError> {
        self.tick();
        if self.host.is_none() || !self.staging_full() { return Ok(()); }
        let start = std::time::Instant::now();
        let mut waited = false;
        while self.staging_full() {
            if let Some(id) = self.entries.iter().filter(|(_, e)| e.mark.is_some() && e.ticket.is_some())
                .min_by_key(|(_, e)| e.ticket.unwrap().0).map(|(&id, _)| id) {
                waited = true;
                let ticket = self.entries[&id].ticket;
                let host = self.host.as_mut().expect("host enabled");
                let (decision, key) = host.before_device_evict_with_key(ticket);
                match decision {
                    EvictDecision::Clean | EvictDecision::WaitedClean { .. } => {
                        let entry = self.entries.get_mut(&id).unwrap();
                        entry.ticket = None;
                        if let Some(key) = key {
                            host.pin(key);
                            entry.host_mark = Some(key);
                            let mark = entry.mark.take();
                            self.give_back(family, mark)?;
                        }
                    }
                    EvictDecision::DroppedUncached => { self.entries.get_mut(&id).unwrap().ticket = None; }
                    EvictDecision::Held => return Err(PrefixError::Host("staging write did not drain".into())),
                }
            } else if !self.pending.is_empty() {
                waited = true;
                let ticket = self.pending[0].ticket;
                family.wait_capture(ticket).map_err(|source| PrefixError::Family { what: "capture wait", source })?;
                let PendingCapture { kind, entry, .. } = self.pending.remove(0);
                if let Some(old) = self.retained.bank_mut(kind).remove_exact(&entry.tokens) { self.evict(family, old)?; }
                self.publish(family, kind, entry)?;
                self.completed_captures.push(ticket);
            } else {
                // Uncacheable/failed host writes retain their mark only until staging pressure.
                let Some(id) = self.mark_victim() else { break };
                if self.layout.rule.approximate_partial() {
                    // Keep the exact pages: the entry restores by bounded replay from them.
                    let mark = self.entries.get_mut(&id).and_then(|e| e.mark.take());
                    self.give_back(family, mark)?;
                    self.stats.marks_dropped += 1;
                } else {
                    self.evict(family, id)?;
                }
            }
        }
        if waited {
            self.stats.capture_stalls += 1;
            self.stats.capture_stall_ns += start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        }
        Ok(())
    }

    fn restore_mark<F: PrefixFamily<M>>(&mut self, family: &F, key: cuteafd_hostcache::snapshot::Key, keep: EntryId)
        -> Result<Option<Mark>, PrefixError> {
        if !self.host.as_ref().is_some_and(|h| h.payload(key).is_some()) { return Ok(None); }
        self.reap_marks(family)?;
        let mark = if self.arena_marks() {
            while self.arena.free() == 0 {
                let Some(id) = self.mark_victim() else { return Ok(None) };
                self.evict(family, id)?;
            }
            Some(Mark::Slot(self.arena.take().expect("free slot")))
        } else {
            let need = self.pool.need_for(0..self.layout.mark_store.pages());
            if !self.make_room_for(family, need, Some(keep))? { return Ok(None); }
            self.take_mark_pages()?
        };
        let tail = match Self::mark_segments(family, mark.as_ref()) {
            Ok(tail) => tail,
            Err(error) => { self.give_back(family, mark)?; return Err(error); }
        };
        let result = self.host.as_mut().expect("host enabled").restore_mark(key, tail);
        if !matches!(result, RestoreOutcome::Done { .. }) {
            if let Err(error) = self.host.as_mut().unwrap().engine_mut().release_barrier(Stream::Restore) {
                self.quarantine.push(Quarantined { _pages: Vec::new(), _mark: mark, _draft: None });
                self.stats.quarantined += 1;
                return Err(PrefixError::Host(format!("mark restore barrier: {error:#}")));
            }
            self.give_back(family, mark)?;
            self.stats.restore_failures += 1;
            return Ok(None);
        }
        if let Some(Mark::Slot(slot)) = &mark {
            let replicas = family.mark_replicas(*slot);
            if !replicas.is_empty() {
                self.dirty = true;
                if let Err(source) = family.copy_replicas(&replicas) {
                    self.give_back(family, mark)?;
                    return Err(PrefixError::Family { what: "mark replicas", source });
                }
            }
        }
        self.stats.mark_promotions += 1;
        Ok(mark)
    }

    pub fn stats(&self) -> PrefixStats {
        self.stats_with_detail(self.occupancy_detail())
    }

    /// Live counters and constant-time gauges every round; entry/page walks at
    /// most once a second. `stats()` remains exact for correctness callers.
    pub fn serving_stats(&self) -> PrefixStats {
        let mut detail = self.serving_detail.borrow_mut();
        let now = std::time::Instant::now();
        if detail.as_ref().is_none_or(|(last, _)| now.duration_since(*last) >= std::time::Duration::from_secs(1)) {
            *detail = Some((now, self.occupancy_detail()));
        }
        self.stats_with_detail(detail.as_ref().unwrap().1)
    }

    /// Settled final occupancy must not remain stale after the scheduler parks.
    pub fn refresh_serving_stats(&self) {
        *self.serving_detail.borrow_mut() = Some((std::time::Instant::now(), self.occupancy_detail()));
    }

    fn occupancy_detail(&self) -> [usize; 5] {
        let mut retained: Vec<u32> = self.entries.values()
            .flat_map(|e| e.pages.iter().chain(e.mark_pages()).copied()).collect();
        retained.sort_unstable();
        retained.dedup();
        [self.pool.shared(), retained.len(),
            self.entries.values().map(|e| e.mark_pages().len()).sum(),
            self.entries.values().filter(|e| matches!(e.mark, Some(Mark::Pages(_)))).count(),
            self.entries.values().filter(|e| e.mark.is_none() && e.host_mark.is_some()).count()]
    }

    fn stats_with_detail(&self, detail: [usize; 5]) -> PrefixStats {
        let mut stats = self.stats.clone();
        stats.entries_prompt = self.retained.bank(SnapshotKind::Prompt).entries();
        stats.entries_turn = self.retained.bank(SnapshotKind::Turn).entries();
        stats.pages = self.pool.capacity();
        stats.pages_free = self.pool.free();
        stats.pages_used_by_gpu = self.pool.is_parity().then(|| self.pool.used_halves());
        stats.pages_shared = detail[0];
        stats.pages_retained = detail[1];
        stats.mark_slots = self.arena.slots();
        stats.mark_pages = detail[2];
        stats.marks_in_use = self.arena.in_use() + detail[3];
        stats.marks_on_host = detail[4];
        stats.mark_family = self.layout.mark_bytes > 0;
        stats.marks_lost = self.host.as_ref().map_or(0, |host| self.entries.values()
            .filter(|e| e.mark.is_none() && e.host_mark.is_some_and(|key| host.payload(key).is_none())).count());
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
            family.drain().map_err(|source| PrefixError::Drain { source })?;
            self.dirty = false;
        }
        Ok(())
    }

    /// The family places each page where the pool says it lives (parity pools: GPU `id & 1`).
    fn check_owners<F: PrefixFamily<M>>(&self, family: &F, pages: &[u32]) -> Result<(), BoxError> {
        match pages.iter().find(|&&page| family.page_device(page) != self.pool.owner(page)) {
            Some(&page) => Err(format!("page {page} lives on {:?} for the family, {:?} in the pool",
                family.page_device(page), self.pool.owner(page)).into()),
            None => Ok(()),
        }
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
                let mut pages = self.pool.alloc_for(0..pages)?;
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
        if let (Some(key), Some(host)) = (entry.host_mark, self.host.as_mut()) { host.release_pin(key); }
        self.retained.bank_mut(entry.kind).remove_exact(&entry.tokens);
        self.stats.evictions += 1;
        self.release(family, &entry.pages)?;
        self.release_draft(family, entry.draft.as_ref())?;
        self.give_back(family, entry.mark)
    }

    fn reconcile_draft<F: PrefixFamily<M>>(&mut self, family: &F) -> Result<(), PrefixError> {
        for entry in self.entries.values_mut() {
            let Some(draft) = &entry.draft else { continue };
            if !family.draft_pages_valid(draft) {
                entry.draft = None;
            } else if entry.ticket.is_none() {
                family.retain_draft_pages(draft).map_err(|source|
                    PrefixError::Family { what: "retain draft pages", source })?;
            }
        }
        Ok(())
    }

    /// Draft-only eviction keeps the target snapshot and its host payload reusable cold.
    fn evict_draft<F: PrefixFamily<M>>(&mut self, family: &F) -> Result<bool, PrefixError> {
        self.reconcile_draft(family)?;
        let Some(id) = self.entries.iter().filter(|(_, e)| e.ticket.is_none()
            && e.draft.as_ref().is_some_and(|d| family.draft_pages_revocable(d)))
            .min_by_key(|(_, e)| e.last_use).map(|(&id, _)| id) else { return Ok(false) };
        let draft = self.entries[&id].draft.clone();
        self.release_draft(family, draft.as_ref())?;
        self.entries.get_mut(&id).expect("selected").draft = None;
        Ok(true)
    }

    fn admit_draft<F: PrefixFamily<M>>(&mut self, family: &F, placement: &mut F::Placement,
        capacity: usize) -> Result<(), PrefixError> {
        self.reconcile_draft(family)?;
        loop {
            match family.admit_draft_pages(placement, capacity) {
                Ok(()) => { self.reconcile_draft(family)?; return Ok(()); }
                Err(source) if draft_exhaustion(source.as_ref()).is_some() => {
                    if !self.evict_draft(family)? {
                        return Err(PrefixError::DraftPages(draft_exhaustion(source.as_ref()).expect("matched")));
                    }
                }
                Err(source) => return Err(PrefixError::Family { what: "admit draft pages", source }),
            }
        }
    }

    fn capture_draft<F: PrefixFamily<M>>(&mut self, family: &F, placement: &F::Placement,
        len: usize) -> Result<Option<super::DraftPageAttachment>, BoxError> {
        loop {
            match family.capture_draft_pages(placement, len) {
                Err(source) if draft_exhaustion(source.as_ref()).is_some() => {
                    if !self.evict_draft(family).map_err(|e| Box::new(e) as BoxError)? { return Ok(None); }
                }
                result => return result,
            }
        }
    }

    fn release_draft<F: PrefixFamily<M>>(&mut self, family: &F,
        draft: Option<&super::DraftPageAttachment>) -> Result<(), PrefixError> {
        if let Some(draft) = draft.filter(|d| family.draft_pages_valid(d)) {
            self.drain(family)?;
            family.release_draft_pages(draft).map_err(|source|
                PrefixError::Family { what: "release draft pages", source })?;
        }
        Ok(())
    }

    /// Token-only page identities are safe only when all paged state follows the target
    /// frontier. Predictor-inclusive pages can differ for the same tokens (a no-spec writer),
    /// so use allocation identities rather than deduplicating their canonical state. Pages at or
    /// past an approximate lineage's taint row hold rows an exact prefill may not reproduce:
    /// only the pages wholly before it are named by content.
    pub(super) fn identities<F: PrefixFamily<M>>(&self, family: &F, tokens: &[u32], media: &[MediaSpan], pages: &[u32],
        approximate_from: Option<usize>) -> Vec<DevicePageId> {
        let chain = if family.write_frontier(tokens.len()) == tokens.len() {
            let exact = approximate_from.unwrap_or(tokens.len()).min(tokens.len());
            page_chain_media(&tokens[..exact], media, self.layout.page_rows)
        } else { Vec::new() };
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
        self.check_owners(family, &entry.pages).map_err(|source| PrefixError::Family { what: "page owners", source })?;
        let ids = self.identities(family, &entry.tokens, &entry.media, &entry.pages, entry.approximate_from);
        let mut pages: [Vec<DevicePage>; cuteafd_hostcache::PAGE_BANKS] = Default::default();
        pages[0] = entry.pages.iter().zip(ids).map(|(&page, id)| DevicePage { id, segments: family.page_segments(page) }).collect();
        if let Some(draft) = &entry.draft {
            pages[cuteafd_hostcache::DRAFT_PAGE_BANK] = draft.pages.iter().map(|&page| DevicePage {
                id: DevicePageId { compressor: cuteafd_hostcache::DRAFT_PAGE_BANK as u8,
                    page: page.index, generation: page.generation },
                segments: family.draft_page_segments(page),
            }).collect();
        }
        let snapshot = DeviceSnapshot {
            meta: SnapshotMeta { kind: entry.kind, tokens: entry.tokens.clone(), end: entry.len() as u32, has_draft: false },
            pages,
            tail: Self::mark_segments(family, entry.mark.as_ref())?,
            draft: None,
            scores: Vec::new(),
        };
        let payload = HostPayload { record_format: family.record_format(), session: entry.session.clone(), after: entry.after.clone(), media: entry.media.clone(),
            family: entry.meta.clone(), approximate_from: entry.approximate_from, draft: entry.draft.clone() };
        let ticket = loop {
            let host = self.host.as_mut().expect("checked");
            match host.store(&snapshot, payload.clone()) {
                StoreOutcome::Issued(ticket) | StoreOutcome::Deferred(ticket) => break Some(ticket),
                StoreOutcome::Skipped(SkipReason::Exhausted) if self.layout.mark_bytes > 0
                    && host.fits_empty(&snapshot) => {
                    let Some(victim) = self.host_pressure_victim(id) else {
                        self.stats.host_store_skips += 1;
                        break None;
                    };
                    self.evict(family, victim)?;
                }
                StoreOutcome::Skipped(_) => {
                    self.stats.host_store_skips += 1;
                    break None;
                }
            }
        };
        self.entries.get_mut(&id).expect("checked").ticket = ticket;
        Ok(())
    }

    /// Host pressure must not strand a mark or displace a shared base with its
    /// own follow-up. Retire an idle host-backed entry, not its host allocation
    /// directly: the host cache then reclaims unpinned slabs in its normal order.
    fn host_pressure_victim(&self, incoming: EntryId) -> Option<EntryId> {
        let host = self.host.as_ref()?;
        let tokens = &self.entries.get(&incoming)?.tokens;
        // A page reference not owned by a retained or queued entry belongs to
        // a live placement (or quarantine); conservatively leave it alone.
        let mut held = BTreeMap::<u32, u32>::new();
        for entry in self.entries.values().chain(self.pending.iter().map(|p| &p.entry)) {
            for &page in &entry.pages { *held.entry(page).or_default() += 1; }
        }
        self.entries.iter().filter(|(&id, entry)| id != incoming
            && entry.ticket.is_none() && entry.mark.is_none()
            && entry.host_mark.is_some_and(|key| host.singly_pinned(key))
            && !tokens.starts_with(&entry.tokens)
            && entry.pages.iter().all(|page| self.pool.refs(*page) == held[page]))
            .min_by_key(|(&id, entry)| {
                let ancestor = self.entries.values().any(|other| other.len() > entry.len()
                    && other.tokens.starts_with(&entry.tokens));
                (ancestor, entry.last_use, entry.kind == SnapshotKind::Turn, id)
            }).map(|(&id, _)| id)
    }

    /// On a device miss: rebuild the best host snapshot on the device so the device path finds
    /// it. Anything short of a completed restore is a miss.
    fn promote<F: PrefixFamily<M>>(&mut self, family: &F, tokens: &[u32], media: &[MediaSpan], sampled: bool,
        compatible: Option<&dyn Fn(&M, usize) -> bool>) -> Result<Option<Hit>, PrefixError> {
        let selected = if let Some(compatible) = compatible {
            let selected = self.host_compatible_hit(tokens, media, sampled, compatible);
            let key = selected.as_ref().map(|(hit, _)| hit.key);
            if !self.host.as_mut().is_some_and(|host| host.lookup_verified(key)) { return Ok(None); }
            selected.map(|(hit, _)| hit)
        } else if media.is_empty() {
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
        if payload.record_format != family.record_format() {
            return Ok(None);
        }
        let snapshot_tokens = snapshot_tokens.to_vec();
        let after = payload.after.clone();
        let saved_media = payload.media.clone();
        let session = payload.session.clone();
        let meta = payload.family.clone();
        let approximate_from = payload.approximate_from;
        let saved_draft = payload.draft.clone();
        let (resume, common, collision) = media_resume(self.layout.rule, tokens, media, &snapshot_tokens,
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
        let fresh = self.pool.need_for(0..need) + self.pool.need_for(0..self.layout.mark_store.pages());
        if !self.make_room_for(family, fresh, None)? {
            return Ok(None);
        }
        self.reap_marks(family)?;
        let slot = if self.arena_marks() {
            loop {
                match self.arena.take() {
                    Ok(slot) => break Some(slot),
                    Err(_) => match self.mark_victim() {
                        Some(id) => self.evict(family, id)?,
                        None => return Ok(None),
                    },
                }
            }
        } else {
            None
        };
        if !self.pool.fits(fresh) {
            self.give_back(family, slot.map(Mark::Slot))?;
            return Ok(None);
        }
        let pages = self.pool.alloc_for(0..need)?;
        let mark = match slot {
            Some(slot) => Some(Mark::Slot(slot)),
            None => self.take_mark_pages()?,
        };
        // The restore stream writes these pages and the mark: nothing queued may still use them.
        self.drain(family)?;
        if let Err(source) = self.check_owners(family, &pages) {
            tracing::warn!(target: "cuteafd::prefix", error = %source, "host restore abandoned; prefilling");
            self.stats.restore_failures += 1;
            self.release(family, &pages)?;
            self.give_back(family, mark)?;
            return Ok(None);
        }
        let ids = self.identities(family, &snapshot_tokens, &saved_media, &pages, approximate_from);
        let mut target_pages: [Vec<DevicePage>; cuteafd_hostcache::PAGE_BANKS] = Default::default();
        target_pages[0] = pages.iter().zip(ids).map(|(&page, id)| DevicePage { id, segments: family.page_segments(page) }).collect();
        let tail = match Self::mark_segments(family, mark.as_ref()) {
            Ok(tail) => tail,
            Err(error) => {
                self.release(family, &pages)?;
                self.give_back(family, mark)?;
                return Err(error);
            }
        };
        let draft = match saved_draft.as_ref().map(|saved| family.promote_draft_pages(saved)).transpose() {
            Ok(draft) => draft,
            Err(source) if draft_exhaustion(source.as_ref()).is_some() => None,
            Err(source) => {
                self.release(family, &pages)?;
                self.give_back(family, mark)?;
                return Err(PrefixError::Family { what: "reserve host draft pages", source });
            }
        };
        if let Some(draft) = &draft {
            target_pages[cuteafd_hostcache::DRAFT_PAGE_BANK] = draft.pages.iter().map(|&page| DevicePage {
                id: DevicePageId { compressor: cuteafd_hostcache::DRAFT_PAGE_BANK as u8,
                    page: page.index, generation: page.generation },
                segments: family.draft_page_segments(page),
            }).collect();
        }
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
                if let Err(error) = host.engine_mut().release_barrier(Stream::Restore) {
                    self.quarantine.push(Quarantined { _pages: pages, _mark: mark, _draft: draft });
                    self.stats.quarantined += 1;
                    return Err(PrefixError::Host(format!("host restore release barrier: {error:#}")));
                }
                tracing::warn!(target: "cuteafd::prefix", tokens = len, "host restore abandoned; prefilling");
                self.stats.restore_failures += 1;
                self.release(family, &pages)?;
                self.release_draft(family, draft.as_ref())?;
                self.give_back(family, mark)?;
                return Ok(None);
            }
        }
        // Replicated bytes came back to GPU0 only; copy them to GPU1 before anything reads them.
        let mut replicas: Vec<_> = pages.iter().flat_map(|&page| family.page_replicas(page)).collect();
        if let Some(Mark::Slot(slot)) = &mark {
            replicas.extend(family.mark_replicas(*slot));
        }
        if !replicas.is_empty() {
            self.dirty = true;
            if let Err(source) = family.copy_replicas(&replicas) {
                tracing::warn!(target: "cuteafd::prefix", error = %source, tokens = len, "host restore replicas failed; prefilling");
                self.stats.restore_failures += 1;
                self.release(family, &pages)?;
                self.release_draft(family, draft.as_ref())?;
                self.give_back(family, mark)?;
                return Ok(None);
            }
        }
        // Restore copies and any replicas must retire before either attachment becomes visible.
        if draft.is_some() {
            if let Err(error) = self.drain(family) {
                self.quarantine.push(Quarantined { _pages: pages, _mark: mark, _draft: draft });
                self.stats.quarantined += 1;
                return Err(error);
            }
        }
        self.clock += 1;
        let id = self.next_id;
        self.next_id += 1;
        self.entries.insert(id, Entry { session, tokens: snapshot_tokens.clone(), media: saved_media, kind: hit.kind, pages, mark, after,
            meta, approximate_from, draft, last_use: self.clock, ticket: None, host_mark: Some(hit.key) });
        self.host.as_mut().expect("promoted from the host").pin(hit.key);
        if let Some(evicted) = self.retained.bank_mut(hit.kind).insert(&snapshot_tokens, id) {
            self.evict(family, evicted)?;
        }
        self.stats.host_promotions += 1;
        Ok(Some(Hit { id, kind: hit.kind, resume, common, frontier: len }))
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
    let hit = RestoreCandidate { common, snapshot_end: saved_tokens.len(), target_end: tokens.len(),
        resume, media, saved_media };
    let verified = if rule == cuteafd_core::prefix::ReuseRule::V41 {
        let plan = RestorePlan::under(rule, hit);
        resume = plan.resume();
        plan.lag.source_end
    } else {
        resume = hit.media_safe(resume);
        resume
    };
    if !verify_media(verified, saved_media, media) { return (0, common, true); }
    (resume, common, false)
}
