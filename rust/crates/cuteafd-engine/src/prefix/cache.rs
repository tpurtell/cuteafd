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
use super::family::{BoxError, FamilyLayout, MarkStore, PageOwners, PrefixFamily};
use super::marks::MarkArena;
use super::pages::{Need, PoolExhausted, RefPagePool};
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
    /// Rows restored; prefill resumes here.
    pub resume: usize,
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
}

/// An admitted request: its placement, the rows already in it, and the first token's source
/// when the whole prompt was retained.
pub struct Admitted<P> {
    pub placement: P,
    pub resume: usize,
    pub after: Option<After>,
    pub source: Option<Source>,
}

/// What travels with a host snapshot besides its device bytes.
pub struct HostPayload {
    pub session: Option<String>,
    pub after: After,
    pub media: Vec<MediaSpan>,
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
    pub host: Option<cuteafd_hostcache::metrics::Snapshot>,
}

pub struct PrefixCache<E: CopyEngine> {
    layout: FamilyLayout,
    config: PrefixConfig,
    retained: Retention<EntryId>,
    entries: BTreeMap<EntryId, Entry>,
    pool: RefPagePool,
    arena: MarkArena,
    host: Option<HostCache<E, HostPayload>>,
    clock: u64,
    next_id: EntryId,
    /// Family copies were enqueued since the family last drained.
    dirty: bool,
    capture_session: Option<String>,
    stats: PrefixStats,
}

impl<E: CopyEngine> PrefixCache<E> {
    /// `host`: the pinned host tier's knobs and copy engine (`None`, or a zero `bytes`, keeps it off).
    pub fn new(layout: FamilyLayout, config: PrefixConfig, host: Option<(cuteafd_hostcache::config::Config, E)>)
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
            stats: PrefixStats::default(),
            layout,
            config,
        })
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
    pub fn admit<F: PrefixFamily<Placement = P>, P>(&mut self, family: &F, tokens: &[u32], capacity: usize,
        sampled: bool, build: impl FnMut(Vec<u32>) -> P) -> Result<Admitted<P>, PrefixError> {
        self.admit_media(family, tokens, &[], capacity, sampled, build)
    }

    /// `tokens` are MediaKeys' keyed copy, never the model's native ids. Verify identities
    /// before forking pages. Encoding waiters call peek_media first, without owning a placement.
    pub fn admit_media<F: PrefixFamily<Placement = P>, P>(&mut self, family: &F, tokens: &[u32],
        media: &[MediaSpan], capacity: usize, sampled: bool, mut build: impl FnMut(Vec<u32>) -> P)
        -> Result<Admitted<P>, PrefixError> {
        crate::media::keys::validate_spans(media)?;
        if media.last().is_some_and(|s| s.checked_end().unwrap() > tokens.len()) {
            return Err(MediaError::Spans.into());
        }
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
                match self.restore_hit(family, &hit, tokens, total, promoted, &mut build) {
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
        self.make_room_for(family, self.pool.need_for(0..total), None)?;
        let pages = self.pool.alloc_for(0..total)?;
        Ok(Admitted { placement: build(pages), resume: 0, after: None, source: None })
    }

    /// Bypass prefix reuse after repeated peek/admission races. The caller has materialized
    /// the entire prompt's media before invoking this allocator.
    pub fn admit_cold<F: PrefixFamily<Placement = P>, P>(&mut self, family: &F, tokens: usize,
        capacity: usize, build: impl FnOnce(Vec<u32>) -> P) -> Result<Admitted<P>, PrefixError> {
        let total = self.pool.pages_for(capacity.max(tokens));
        self.make_room_for(family, self.pool.need_for(0..total), None)?;
        let pages = self.pool.alloc_for(0..total)?;
        Ok(Admitted { placement: build(pages), resume: 0, after: None, source: None })
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
            return Some(Hit { id, kind: entry.kind, resume, frontier });
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
            let (resume, collision) = media_resume(self.layout.rule, tokens, media, &entry.tokens,
                &entry.media, &entry.after, sampled);
            collisions += u64::from(collision);
            let candidate = (resume, entry.kind == SnapshotKind::Turn, entry.last_use);
            if resume > 0 && candidate > rank {
                rank = candidate;
                best = Some(Hit { id, kind: entry.kind, resume, frontier: entry.len() });
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
            let (resume, collision) = media_resume(self.layout.rule, tokens, media, &meta.tokens,
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

    fn restore_hit<F: PrefixFamily<Placement = P>, P>(&mut self, family: &F, hit: &Hit, tokens: &[u32], total: usize,
        promoted: bool, build: &mut impl FnMut(Vec<u32>) -> P) -> Result<Option<Admitted<P>>, PrefixError> {
        let need = self.pool.fork_need(hit.resume, total);
        if !self.make_room_for(family, need, Some(hit.id))? {
            return Ok(None);
        }
        let entry = self.entries.get(&hit.id).expect("the hit is kept while making room");
        let fork = self.pool.fork(&entry.pages, hit.resume, total)?;
        let partial = hit.resume != entry.len();
        let mark = if partial { None } else { entry.mark.clone() };
        let after = (hit.resume == tokens.len()).then(|| entry.after.clone());
        let pages = fork.pages.clone();
        let mut placement = build(fork.pages);
        self.dirty = true;
        let restored = self.check_owners(family, &pages).and_then(|()| match fork.copy {
            Some(copy) => {
                self.stats.cow_copies += 1;
                family.copy_rows(copy)
            }
            None => Ok(()),
        })
        .and_then(|()| match &mark {
            Some(Mark::Pages(pages)) => family.restore_pages(pages, &mut placement, hit.resume),
            Some(Mark::Slot(slot)) => family.restore(Some(*slot), &mut placement, hit.resume),
            None => family.restore(None, &mut placement, hit.resume),
        });
        if let Err(source) = restored {
            self.release(family, &pages)?;
            return Err(PrefixError::Family { what: "restore", source });
        }
        self.stats.hits += 1;
        self.stats.hit_tokens += hit.resume as u64;
        self.stats.exact_hits += u64::from(after.is_some());
        self.stats.partial_hits += u64::from(partial);
        Ok(Some(Admitted {
            placement,
            resume: hit.resume,
            after,
            source: Some(Source { session: self.entries[&hit.id].session.clone(), kind: hit.kind, frontier: hit.frontier, host: promoted, partial }),
        }))
    }

    /// Retain `placement`'s first `tokens.len()` rows as a `kind` snapshot; `after` is what
    /// follows them. The snapshot point may lie up to `family.capture_reach()` rows before the
    /// placement's commit point (an intermediate point). Ok(false) when it was not retained
    /// (disabled, too short, no room).
    pub fn capture<F: PrefixFamily>(&mut self, family: &F, kind: SnapshotKind, tokens: &[u32],
        placement: &F::Placement, after: After) -> Result<bool, PrefixError> {
        self.capture_media(family, kind, tokens, &[], placement, after)
    }

    /// A chunk/message/cancel frontier inside an image rounds down to its first row.
    /// If the family's mark can no longer reach that row, skip instead of capturing wrong state.
    pub fn capture_media<F: PrefixFamily>(&mut self, family: &F, kind: SnapshotKind, tokens: &[u32],
        media: &[MediaSpan], placement: &F::Placement, mut after: After) -> Result<bool, PrefixError> {
        crate::media::keys::validate_spans(media)?;
        let rounded = round_frontier(tokens.len(), media);
        let changed = rounded != tokens.len();
        let tokens = &tokens[..rounded];
        if changed { after = After::default(); }
        if !self.enabled() || tokens.len() < self.config.min_tokens.max(1) {
            return Ok(false);
        }
        let committed = family.commit_point(placement);
        let reach = family.capture_reach();
        if tokens.len() > committed || committed - tokens.len() > reach {
            if changed { self.stats.capture_skips += 1; return Ok(false); }
            return Err(PrefixError::Frontier { tokens: tokens.len(), committed, reach });
        }
        // Replace the same snapshot, or keep the bank within its bound, before taking storage.
        if let Some(old) = self.retained.bank_mut(kind).remove_exact(tokens) {
            self.evict(family, old)?;
        }
        while self.retained.bank(kind).entries() >= self.config.entries {
            let oldest = victim(self.entries.iter().filter(|(_, e)| e.kind == kind).map(|(&id, e)| (id, e.kind, e.last_use)));
            match oldest {
                Some(id) => self.evict(family, id)?,
                None => break,
            }
        }
        let slot = if self.arena_marks() {
            loop {
                match self.arena.take() {
                    Ok(slot) => break Some(slot),
                    Err(_) => match self.victim(None) {
                        Some(id) => self.evict(family, id)?,
                        None => {
                            self.stats.capture_skips += 1;
                            return Ok(false);
                        }
                    },
                }
            }
        } else {
            None
        };
        let len = tokens.len();
        let total = self.pool.pages_for(len);
        // The rows a fork copies and, for pool-page marks, the mark's own pages.
        let mark_pages = self.pool.need_for(0..self.layout.mark_store.pages());
        if !self.make_room_for(family, self.pool.fork_need(len, total) + mark_pages, None)? {
            self.give_back(family, slot.map(Mark::Slot))?;
            self.stats.capture_skips += 1;
            return Ok(false);
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
        let captured = match fork.copy {
            Some(copy) => family.copy_rows(copy),
            None => Ok(()),
        }
        .and_then(|()| match &mark {
            Some(Mark::Pages(pages)) => family.capture_pages(pages, placement, len),
            Some(Mark::Slot(slot)) => family.capture(*slot, placement, len),
            None => Ok(()),
        });
        if let Err(source) = captured {
            self.release(family, &fork.pages)?;
            self.give_back(family, mark)?;
            return Err(PrefixError::Family { what: "capture", source });
        }
        if !self.config.keep_logits {
            after.logits = None;
        }
        self.clock += 1;
        let id = self.next_id;
        self.next_id += 1;
        self.entries.insert(id, Entry { session: self.capture_session.clone(), tokens: tokens.to_vec(), media: snapshot_media(tokens.len(), media), kind, pages: fork.pages, mark, after,
            last_use: self.clock, ticket: None });
        if let Some(evicted) = self.retained.bank_mut(kind).insert(tokens, id) {
            self.evict(family, evicted)?;
        }
        match kind {
            SnapshotKind::Prompt => self.stats.captures_prompt += 1,
            SnapshotKind::Turn => self.stats.captures_turn += 1,
        }
        self.host_store(family, id)?;
        Ok(true)
    }

    /// A prefill cancelled at a chunk boundary: keep what it computed as a prompt snapshot
    /// (the client usually retries the same prompt).
    pub fn park<F: PrefixFamily>(&mut self, family: &F, tokens: &[u32], placement: &F::Placement)
        -> Result<bool, PrefixError> {
        let parked = self.capture(family, SnapshotKind::Prompt, tokens, placement, After::default())?;
        self.stats.parked += u64::from(parked);
        Ok(parked)
    }

    pub fn park_media<F: PrefixFamily>(&mut self, family: &F, tokens: &[u32], media: &[MediaSpan],
        placement: &F::Placement) -> Result<bool, PrefixError> {
        let parked = self.capture_media(family, SnapshotKind::Prompt, tokens, media, placement, After::default())?;
        self.stats.parked += u64::from(parked);
        Ok(parked)
    }

    /// Drop a placement's page references (after the family's queued work drained).
    pub fn release<F: PrefixFamily>(&mut self, family: &F, pages: &[u32]) -> Result<(), PrefixError> {
        self.drain(family)?;
        let freed = self.pool.release(pages);
        if let Some(host) = &mut self.host {
            for page in freed {
                host.device_page_freed(DevicePageId { compressor: 0, page: page.index, generation: page.generation });
            }
        }
        Ok(())
    }

    /// Evict least recently used snapshots (never `keep`) until `pages` pages are free; false when
    /// nothing is left to evict and the pool is still short.
    pub fn make_room<F: PrefixFamily>(&mut self, family: &F, pages: usize, keep: Option<EntryId>)
        -> Result<bool, PrefixError> {
        self.make_room_for(family, self.pool.need_for(0..pages), keep)
    }

    /// [`PrefixCache::make_room`] for fresh pages per half (both halves of a parity pool must fit).
    pub fn make_room_for<F: PrefixFamily>(&mut self, family: &F, need: Need, keep: Option<EntryId>)
        -> Result<bool, PrefixError> {
        while !self.pool.fits(need) {
            match self.victim(keep) {
                Some(id) => self.evict(family, id)?,
                None => return Ok(false),
            }
        }
        Ok(true)
    }

    /// Extend a live lease before a forward pass. Only inactive snapshots can
    /// be evicted; existing live pages keep their references on failed growth.
    pub fn grow<F: PrefixFamily>(&mut self, family: &F, pages: &mut Vec<u32>, tokens: usize)
        -> Result<(), PrefixError> {
        let positions = pages.len()..self.pool.pages_for(tokens).max(pages.len());
        if positions.is_empty() { return Ok(()); }
        self.make_room_for(family, self.pool.need_for(positions.clone()), None)?;
        pages.extend(self.pool.alloc_for(positions)?);
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
        stats.pages_used_by_gpu = self.pool.is_parity().then(|| self.pool.used_halves());
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
    pub fn clear<F: PrefixFamily>(&mut self, family: &F) -> Result<(), PrefixError> {
        while let Some(id) = self.victim(None) {
            self.evict(family, id)?;
        }
        Ok(())
    }

    fn victim(&self, keep: Option<EntryId>) -> Option<EntryId> {
        victim(self.entries.iter().filter(|(&id, _)| Some(id) != keep).map(|(&id, e)| (id, e.kind, e.last_use)))
    }

    fn drain<F: PrefixFamily>(&mut self, family: &F) -> Result<(), PrefixError> {
        if self.dirty {
            family.drain().map_err(|source| PrefixError::Family { what: "drain", source })?;
            self.dirty = false;
        }
        Ok(())
    }

    /// The family places each page where the pool says it lives (parity pools: GPU `id & 1`).
    fn check_owners<F: PrefixFamily>(&self, family: &F, pages: &[u32]) -> Result<(), BoxError> {
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
    fn give_back<F: PrefixFamily>(&mut self, family: &F, mark: Option<Mark>) -> Result<(), PrefixError> {
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
    fn mark_segments<F: PrefixFamily>(family: &F, mark: Option<&Mark>) -> Result<Vec<DeviceRange>, PrefixError> {
        Ok(match mark {
            Some(&Mark::Slot(slot)) => family.mark_segments(slot),
            Some(Mark::Pages(pages)) => family.mark_page_segments(pages)
                .map_err(|source| PrefixError::Family { what: "mark segments", source })?,
            None => family.host_tail(),
        })
    }

    /// Drop a device snapshot: its host copy finishes within budget first, then its storage goes
    /// back once the family's queued copies drained.
    fn evict<F: PrefixFamily>(&mut self, family: &F, id: EntryId) -> Result<(), PrefixError> {
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
    fn host_store<F: PrefixFamily>(&mut self, family: &F, id: EntryId) -> Result<(), PrefixError> {
        if self.host.is_none() {
            return Ok(());
        }
        self.drain(family)?;
        let entry = self.entries.get(&id).expect("stored entry is retained");
        self.check_owners(family, &entry.pages).map_err(|source| PrefixError::Family { what: "page owners", source })?;
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
        let payload = HostPayload { session: entry.session.clone(), after: entry.after.clone(), media: entry.media.clone() };
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
    fn promote<F: PrefixFamily>(&mut self, family: &F, tokens: &[u32], media: &[MediaSpan], sampled: bool) -> Result<Option<Hit>, PrefixError> {
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
        let (resume, collision) = media_resume(self.layout.rule, tokens, media, &snapshot_tokens,
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
        let host = self.host.as_mut().expect("checked");
        match host.restore(hit.key, &target) {
            RestoreOutcome::Done { .. } => {}
            outcome => {
                // Both timeout and submission errors may leave copies queued.
                // On barrier failure keep the pool refs/mark allocated and the
                // host snapshot pinned, instead of handing live targets back.
                host.engine_mut().release_barrier(Stream::Restore)
                    .map_err(|error| PrefixError::Host(format!("host restore release barrier: {error:#}")))?;
                tracing::warn!(target: "cuteafd::prefix", ?outcome, tokens = len, "host restore abandoned; prefilling");
                self.stats.restore_failures += 1;
                self.release(family, &pages)?;
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
                self.give_back(family, mark)?;
                return Ok(None);
            }
        }
        self.clock += 1;
        let id = self.next_id;
        self.next_id += 1;
        self.entries.insert(id, Entry { session, tokens: snapshot_tokens.clone(), media: saved_media, kind: hit.kind, pages, mark, after,
            last_use: self.clock, ticket: None });
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
    saved_tokens: &[u32], saved_media: &[MediaSpan], after: &After, sampled: bool) -> (usize, bool) {
    let mut common = common_prefix(tokens, saved_tokens);
    let mut resume = rule.skipped(common, saved_tokens.len());
    if resume == tokens.len() && !after.serves(sampled) {
        common = common.saturating_sub(1);
        resume = rule.skipped(common, saved_tokens.len());
    }
    // Apply the family's alignment/replay first, then image atomicity. A rule's partial
    // restore starts empty, so rounding further down is safe (never claim an exact mark).
    resume = round_frontier(round_frontier(resume, media), saved_media);
    if !verify_media(resume, saved_media, media) { return (0, true); }
    (resume, false)
}
