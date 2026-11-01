//! KV reservation by decode chunks with a drain-safe stall rule (issue #951), shared by every
//! serve loop.
//!
//! A request reserves its prefill requirement plus one decode chunk at admission and tops the
//! reservation up by one chunk whenever its next round would write past it. A top-up the pool
//! cannot grant safely stalls the request: it keeps its pages and skips decode rounds, and with
//! the default options it is never truncated.
//!
//! Safety is the banker's rule over every in-flight request: after each admission and grant
//! there is an order in which every request reaches its guaranteed end from the available pages
//! plus what the requests before it release. With the default drain buffer (0) the guaranteed
//! end is the whole lifetime (prompt + maximum output), so the pool never deadlocks and nothing
//! is cut. A nonzero buffer guarantees only that many output tokens; past it, a request that
//! would stall ends with `length`, so its pages count as releasable.
//!
//! Candidates for a top-up are served in ascending order of what they still need (equal to the
//! potential allocation at the default buffer); each is granted only if the drain rule still
//! holds afterwards, so when grants conflict the requests with the most potential allocation
//! remaining are the ones that stall, and the one that can finish from what is free always
//! runs. A refused candidate does not block later ones (a parity pool's other half may still
//! fit them).
//!
//! Everything here is host arithmetic over page counts in the pool's own units ([`Room`]): a
//! parity pool's two halves separately (logical page `j` comes from half `j % 2`), and the
//! drafter's page pool when the family reserves draft pages with target pages. Free means free
//! pages plus pages only inactive snapshots hold (those `PrefixCache::extend` may evict). One
//! [`KvReserve::round`] drives every serve loop through the [`Ledger`] trait.
use serde::Serialize;
use thiserror::Error;

/// Default `--kv-decode-reserve-chunk` (tokens).
pub const DEFAULT_CHUNK: usize = 16_384;

/// The global options, validated against the server's effective maximum output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ReserveConfig {
    chunk: usize,
    /// Guaranteed output tokens per request under pressure; `None` (option 0) = all of them.
    drain_buffer: Option<usize>,
    /// Output tokens after which a request that would stall ends with `length` instead.
    termination: Option<usize>,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ReserveConfigError {
    #[error("--kv-decode-reserve-chunk must be positive")]
    Chunk,
    #[error("--kv-reserve-drain-buffer {buffer} must be below the effective maximum output {max_output} (0 = the whole maximum output)")]
    Buffer { buffer: usize, max_output: usize },
    #[error("--kv-early-termination-min-output-tokens must be -1 (off) or a token count, not {0}")]
    Threshold(i64),
    #[error("--kv-reserve-drain-buffer {buffer} is below --kv-early-termination-min-output-tokens {threshold}: the drain allowance must cover the early-termination threshold")]
    BufferBelowThreshold { buffer: usize, threshold: usize },
}

impl ReserveConfig {
    /// `drain_buffer` 0 guarantees each request's whole remaining output; `early_termination`
    /// -1 is off. `max_output` is the server's effective maximum output (the context when
    /// `--max-output-tokens` is not positive).
    pub fn new(chunk: usize, drain_buffer: usize, early_termination: i64, max_output: usize)
        -> Result<Self, ReserveConfigError> {
        if chunk == 0 { return Err(ReserveConfigError::Chunk); }
        let drain_buffer = match drain_buffer {
            0 => None,
            buffer if buffer < max_output => Some(buffer),
            buffer => return Err(ReserveConfigError::Buffer { buffer, max_output }),
        };
        let explicit = match early_termination {
            -1 => None,
            value if value >= 0 => Some(value as usize),
            value => return Err(ReserveConfigError::Threshold(value)),
        };
        if let Some(threshold) = explicit {
            let buffer = drain_buffer.unwrap_or(max_output);
            if buffer < threshold {
                return Err(ReserveConfigError::BufferBelowThreshold { buffer, threshold });
            }
        }
        // A nonzero buffer with no explicit threshold terminates at the buffer.
        Ok(Self { chunk, drain_buffer, termination: explicit.or(drain_buffer) })
    }

    pub fn chunk(&self) -> usize { self.chunk }
    pub fn drain_buffer(&self) -> Option<usize> { self.drain_buffer }
    pub fn termination_threshold(&self) -> Option<usize> { self.termination }
}

/// A request's reservation bounds, fixed at admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lease {
    prompt: usize,
    /// Rows one round may write past the committed frontier (verify rows).
    step: usize,
    /// Rows the request may commit (its logical capacity): prompt + maximum output, within the
    /// limit. It finishes with `length` there.
    end: usize,
    /// Rows its reservation may reach: the end plus one round's rows (a verify writes proposal
    /// rows past the last kept one), within the limit.
    lifetime: usize,
    /// Rows guaranteed under the drain buffer (the lifetime with the default buffer).
    allowance: usize,
}

impl Lease {
    pub fn prompt(&self) -> usize { self.prompt }
    pub fn step(&self) -> usize { self.step }
    /// The request's logical capacity: it finishes with `length` here.
    pub fn end(&self) -> usize { self.end }
    /// Rows the reservation may reach.
    pub fn lifetime(&self) -> usize { self.lifetime }
    /// Rows the next round needs reserved, from `committed` rows.
    pub fn round_rows(&self, committed: usize) -> usize {
        committed.saturating_add(self.step).min(self.lifetime).max(committed)
    }
    /// Output tokens the lease allows.
    pub fn output(&self) -> usize { self.end - self.prompt }
}

#[derive(Debug, Error, PartialEq, Eq)]
#[error("prompt of {prompt} tokens leaves no room for output within {limit} KV rows")]
pub struct NoRoom { pub prompt: usize, pub limit: usize }

/// Free reservation capacity in the pool's own units. A parity pool hands logical page `j` of a
/// sequence a page of half `j % 2` (both GPUs of a context-split layout); a uniform pool keeps
/// everything in `[0]`. A family with a full-context drafter also draws from the draft page pool
/// (64-row pages, its own capacity) whenever its target reservation grows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Room {
    pub halves: [usize; 2],
    pub parity: bool,
    /// Free draft pages, when the family reserves drafter pages with its target pages.
    pub draft: Option<usize>,
}

impl Room {
    pub fn uniform(free: usize) -> Self { Self { halves: [free, 0], parity: false, draft: None } }
    pub fn parity(halves: [usize; 2]) -> Self { Self { halves, parity: true, draft: None } }
    pub fn with_draft(self, free: usize) -> Self { Self { draft: Some(free), ..self } }
    /// Target pages for logical positions `from..to` of one sequence, per half.
    pub fn need(&self, from: usize, to: usize) -> [usize; 2] {
        let n = to.saturating_sub(from);
        if !self.parity { return [n, 0]; }
        // Even positions in from..to: those below `to` less those below `from`.
        let even = to.div_ceil(2).saturating_sub(from.div_ceil(2)).min(n);
        [even, n - even]
    }
    fn fits(&self, need: [usize; 2], draft: usize) -> bool {
        need[0] <= self.halves[0] && need[1] <= self.halves[1] && self.draft.is_none_or(|free| draft <= free)
    }
    fn take(&mut self, need: [usize; 2], draft: usize) {
        self.halves[0] -= need[0];
        self.halves[1] -= need[1];
        if let Some(free) = &mut self.draft { *free -= draft; }
    }
    fn give(&mut self, back: [usize; 2], draft: usize) {
        self.halves[0] += back[0];
        self.halves[1] += back[1];
        if let Some(free) = &mut self.draft { *free += draft; }
    }
    /// Total free target pages.
    pub fn total(&self) -> usize { self.halves[0] + self.halves[1] }
}

/// A request's draft page reservation (`DraftPagePool::admit`): its live growth from `origin`
/// in 64-row pages plus `headroom` pages (a partial page and a proposal block).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftDemand {
    pub origin: usize,
    pub headroom: usize,
    /// Draft pages its credit holds now.
    pub held: usize,
}

/// Rows one request's reservation can ever reach in a draft pool of `pages` pages whose
/// drafter proposes `block` rows: its credit adds `block / 64 + 1` pages to the rows it holds.
pub fn draft_lifetime_rows(pages: usize, block: usize) -> usize {
    pages.saturating_sub(block.div_ceil(crate::prefix::DRAFT_PAGE_ROWS) + 1) * crate::prefix::DRAFT_PAGE_ROWS
}

/// Draft pages one reservation of `rows` rows holds.
pub fn draft_pages(rows: usize, draft: &DraftDemand) -> usize {
    rows.saturating_sub(draft.origin).div_ceil(crate::prefix::DRAFT_PAGE_ROWS) + draft.headroom
}

/// One in-flight request as the policy sees it.
#[derive(Clone, Copy, Debug)]
pub struct Demand<'a> {
    pub lease: Lease,
    /// Pages its placement holds.
    pub held: usize,
    /// Their ids, when other requests may share some (forks of one snapshot): a shared page
    /// returns only with its last live holder. Empty: all `held` pages are its own.
    pub pages: &'a [u32],
    /// Rows already in its KV (a prefilling request counts its whole prompt: its reservation
    /// covers it).
    pub committed: usize,
    /// Output tokens produced.
    pub produced: usize,
    /// Decoding (a top-up candidate). Prefilling requests only hold pages.
    pub decoding: bool,
    /// Its drafter's page credit, when the family reserves draft pages with target pages.
    pub draft: Option<DraftDemand>,
}

/// What one request does this round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// Run with the pages it holds.
    Run,
    /// Grow to this many pages, then run.
    Grow(usize),
    /// Skip the round, keeping its pages.
    Stall,
    /// End with `finish_reason=length` (early termination, or a broken deadlock).
    Terminate,
}

/// Lifetime counters for `/v1/stats` (the stalled gauge is the scheduler's).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ReserveStats {
    /// Top-ups applied (a grant the pool then refused is not one).
    pub top_ups: u64,
    pub top_up_pages: u64,
    /// Transitions into the stalled state.
    pub stalls: u64,
    pub early_terminations: u64,
    /// Every request stalled with nothing able to return a page (a policy defect: the request
    /// with the most potential ended with `length` so the server keeps going).
    pub deadlock_breaks: u64,
    /// Granted top-ups the pool could not supply (the plan and the pool disagreed).
    pub refused_grants: u64,
    /// Admissions that waited for the drain rule.
    pub admission_waits: u64,
}

/// The scheduler side of one round, so the round's order is the same in every serve loop: the
/// policy plans, ended requests release their pages, and grants are applied against the pool
/// as it then stands.
pub trait Ledger {
    /// Free capacity now.
    fn room(&self) -> Room;
    /// No pages come back by themselves (a queued snapshot capture): an all-stalled round is a
    /// real deadlock.
    fn settled(&self) -> bool;
    /// The in-flight requests, `stalled` flags of the decoding ones in the same order.
    fn demands(&self) -> (Vec<Demand<'_>>, Vec<bool>);
    /// End request `index` with `length` and release its pages now.
    fn end(&mut self, index: usize);
    /// Grow request `index` to `pages` target pages (and its draft credit); `false` when the
    /// pool cannot supply them now (nothing changed).
    fn grow(&mut self, index: usize, pages: usize) -> bool;
    fn set_stalled(&mut self, index: usize, stalled: bool);
}

pub struct KvReserve {
    config: ReserveConfig,
    page_rows: usize,
    stats: ReserveStats,
}

impl KvReserve {
    pub fn new(config: ReserveConfig, page_rows: usize) -> Self {
        assert!(page_rows > 0, "pages hold at least one row");
        Self { config, page_rows, stats: ReserveStats::default() }
    }

    pub fn config(&self) -> ReserveConfig { self.config }
    pub fn page_rows(&self) -> usize { self.page_rows }
    pub fn stats(&self) -> ReserveStats { self.stats }

    pub fn pages_for(&self, rows: usize) -> usize { rows.div_ceil(self.page_rows).max(1) }

    /// Bounds for a request of `prompt` tokens and `output` maximum output tokens whose rounds
    /// write up to `step` rows past the committed frontier, within `limit` rows (the context,
    /// and the pool's capacity: a lifetime the pool cannot hold ends with `length` at it).
    pub fn lease(&self, prompt: usize, output: usize, step: usize, limit: usize) -> Result<Lease, NoRoom> {
        let step = step.max(1);
        let end = prompt.saturating_add(output).min(limit);
        if output == 0 || end <= prompt { return Err(NoRoom { prompt, limit }); }
        let lifetime = end.saturating_add(step).min(limit);
        let allowance = match self.config.drain_buffer {
            None => lifetime,
            Some(buffer) => prompt.saturating_add(buffer).saturating_add(step).min(lifetime),
        };
        Ok(Lease { prompt, step, end, lifetime, allowance })
    }

    /// Rows a reservation grows to from `base` rows when it must hold `need`: the next chunk
    /// rounded down to a page, or the whole lifetime when less than half a chunk would be left.
    fn target_rows(&self, base: usize, need: usize, lifetime: usize) -> usize {
        let chunk = self.config.chunk;
        let target = (base.saturating_add(chunk) / self.page_rows * self.page_rows).max(need);
        if target >= lifetime || lifetime - target < chunk / 2 { lifetime.max(need) } else { target }
    }

    /// Rows an admission reserves: the prompt and its first round, plus one chunk.
    pub fn admission_rows(&self, lease: &Lease) -> usize {
        self.target_rows(lease.prompt, lease.round_rows(lease.prompt), lease.lifetime)
    }

    pub fn admission_pages(&self, lease: &Lease) -> usize { self.pages_for(self.admission_rows(lease)) }

    /// Whether this request must grow before its next round.
    pub fn needs_top_up(&self, demand: &Demand) -> bool {
        demand.decoding && demand.lease.round_rows(demand.committed) > demand.held * self.page_rows
    }

    /// Past its drain allowance: it ends at its next refused top-up, releasing everything.
    fn terminable(&self, demand: &Demand) -> bool {
        self.config.drain_buffer.is_some_and(|buffer| demand.produced >= buffer)
    }

    /// Target pages its guaranteed end holds (the banker's goal); its current pages when it
    /// would end instead.
    fn goal(&self, demand: &Demand) -> usize {
        if self.terminable(demand) { return demand.held; }
        self.pages_for(demand.lease.allowance.max(demand.lease.round_rows(demand.committed))).max(demand.held)
    }

    /// Pages still to come before its guaranteed end.
    pub fn need(&self, demand: &Demand) -> usize { self.goal(demand) - demand.held }

    /// Pages still to come before the end of its life.
    pub fn potential(&self, demand: &Demand) -> usize {
        self.pages_for(demand.lease.lifetime).saturating_sub(demand.held)
    }

    /// Draft pages a reservation of `pages` target pages holds.
    fn draft_held(&self, draft: &DraftDemand, pages: usize) -> usize {
        draft_pages(pages * self.page_rows, draft).max(draft.held)
    }

    /// What growing `demand` to `pages` target pages takes from the pool: its target pages per
    /// half, and the draft credit beyond what it holds now (`draft.held`, 0 before admission).
    fn charge(&self, room: &Room, demand: &Demand, pages: usize) -> ([usize; 2], usize) {
        let draft = demand.draft.map_or(0, |draft| self.draft_held(&draft, pages) - draft.held.min(self.draft_held(&draft, pages)));
        (room.need(demand.held, pages), draft)
    }

    /// What releasing `demand` returns: its own pages per half (logical position `j` lives in
    /// half `j % 2`), and its draft credit.
    fn release(&self, room: &Room, demand: &Demand, shared: &std::collections::HashMap<u32, u32>) -> ([usize; 2], usize) {
        // Positions past its listed ids (a trial grant) are its own.
        let mut back = room.need(demand.pages.len().min(demand.held), demand.held);
        for (j, page) in demand.pages.iter().enumerate().take(demand.held) {
            if shared.get(page).is_none_or(|&refs| refs <= 1) { back[if room.parity { j % 2 } else { 0 }] += 1; }
        }
        (back, demand.draft.map_or(0, |draft| self.draft_held(&draft, demand.held)))
    }

    /// Banker's safety: some order lets every request reach its guaranteed end. Releasing only
    /// returns capacity, so any request that can reach its goal now may go first: repeat until
    /// all have or none can.
    pub fn safe<'a>(&self, room: Room, demands: impl IntoIterator<Item = Demand<'a>>) -> bool {
        let demands: Vec<Demand<'a>> = demands.into_iter().collect();
        let mut live: std::collections::HashMap<u32, u32> = Default::default();
        for d in &demands {
            for &page in d.pages { *live.entry(page).or_default() += 1; }
        }
        let mut work = room;
        let mut done = vec![false; demands.len()];
        let mut left = demands.len();
        while left > 0 {
            let mut progressed = false;
            for (i, d) in demands.iter().enumerate() {
                if done[i] { continue; }
                let (need, draft) = self.charge(&work, d, self.goal(d));
                if !work.fits(need, draft) { continue; }
                // It reaches its goal (borrowing `need`, which comes back with it), then gives
                // back the pages it holds now that no other live request shares, and its
                // whole draft credit.
                let (back, draft_back) = self.release(&work, d, &live);
                work.give(back, draft_back);
                for &page in d.pages {
                    let refs = live.get_mut(&page).expect("counted");
                    *refs -= 1;
                }
                done[i] = true;
                left -= 1;
                progressed = true;
            }
            if !progressed { return false; }
        }
        true
    }

    /// Pages to reserve for a new request, or `None` while it waits: a request is stalled (it
    /// goes first), the pool lacks the pages, or the drain rule would not hold afterwards.
    /// `draft` is its drafter's credit before admission (origin at its restored frontier).
    pub fn admit(&mut self, room: Room, demands: &[Demand<'_>], lease: &Lease, draft: Option<DraftDemand>,
        stalled: bool) -> Option<usize> {
        let pages = self.admission_pages(lease);
        let fresh = Demand { lease: *lease, held: 0, pages: &[], committed: lease.prompt, produced: 0,
            decoding: false, draft };
        let (need, draft_need) = self.charge(&room, &fresh, pages);
        let fits = !stalled && room.fits(need, draft_need) && {
            let mut after = room;
            after.take(need, draft_need);
            let admitted = Demand { held: pages, draft: draft.map(|d| DraftDemand { held: self.draft_held(&d, pages), ..d }),
                ..fresh };
            self.safe(after, demands.iter().copied().chain([admitted]))
        };
        if fits { Some(pages) } else {
            self.stats.admission_waits += 1;
            None
        }
    }

    /// The grant for candidate `d` against `room` and the other requests: the next chunk, else
    /// just this round's rows, whichever keeps the drain rule.
    fn grant(&self, room: &Room, demands: &[Demand<'_>], index: usize) -> Option<usize> {
        let d = demands[index];
        let round = d.lease.round_rows(d.committed);
        let minimal = self.pages_for(round).max(d.held);
        let full = self.pages_for(self.target_rows(d.held * self.page_rows, round, d.lease.lifetime)).max(minimal);
        [full, minimal].into_iter().find(|&pages| {
            let (need, draft) = self.charge(room, &d, pages);
            room.fits(need, draft) && {
                let mut after = *room;
                after.take(need, draft);
                let grown = Demand { held: pages, draft: d.draft.map(|dd| DraftDemand { held: self.draft_held(&dd, pages), ..dd }), ..d };
                self.safe(after, demands.iter().enumerate().map(|(j, s)| if j == index { grown } else { *s }))
            }
        })
    }

    /// One round: plan; end the refused requests past the early-termination threshold (their
    /// pages return before anything is granted, then plan again); grant top-ups least need
    /// first against the pool as it then stands, skipping (stalling) any the drain rule or the
    /// pool's halves refuse. A grant the pool refuses stalls the rest for this round and is retried
    /// against the real pool next round; a round where every decoding request stalls and
    /// nothing else can return a page ends the request with the most potential (the deadlock
    /// break), so no round repeats forever.
    pub fn round(&mut self, ledger: &mut impl Ledger) {
        loop {
            let (demands, was) = ledger.demands();
            let decoding = was.len();
            debug_assert!(demands[..decoding].iter().all(|d| d.decoding) && demands[decoding..].iter().all(|d| !d.decoding));
            let mut candidates: Vec<usize> = (0..decoding).filter(|&i| self.needs_top_up(&demands[i])).collect();
            candidates.sort_by_key(|&i| (self.need(&demands[i]), self.potential(&demands[i]), i));
            let mut work = ledger.room();
            let mut state = demands.clone();
            let mut plan = Vec::new();
            let mut refused = Vec::new();
            // Least need first. A refused candidate is skipped, not a barrier: in a parity pool
            // a later candidate may need the other half, and the drain rule (`safe`) already
            // keeps the ones with the most potential waiting when grants really conflict.
            for &i in &candidates {
                match self.grant(&work, &state, i) {
                    Some(pages) => {
                        let (need, draft) = self.charge(&work, &state[i], pages);
                        work.take(need, draft);
                        state[i].held = pages;
                        if let Some(dd) = &mut state[i].draft { dd.held = self.draft_held(dd, pages); }
                        plan.push((i, pages));
                    }
                    None => refused.push(i),
                }
            }
            let mut ending: Vec<usize> = refused.iter().copied()
                .filter(|&i| self.config.termination.is_some_and(|t| demands[i].produced >= t)).collect();
            let held: Vec<usize> = demands.iter().map(|d| d.held).collect();
            let settled = ledger.settled() && decoding == demands.len();
            let idle = settled && plan.is_empty() && refused.len() == decoding && decoding > 0;
            drop((demands, state));
            if ending.is_empty() && idle {
                // Every request waits for capacity and none can come back: end the largest.
                let (demands, _) = ledger.demands();
                let victim = (0..decoding).max_by_key(|&i| (self.potential(&demands[i]), i)).expect("decoding");
                drop(demands);
                ending.push(victim);
                self.stats.deadlock_breaks += 1;
            } else {
                self.stats.early_terminations += ending.len() as u64;
            }
            if !ending.is_empty() {
                ending.sort_unstable();
                for &i in ending.iter().rev() { ledger.end(i); }
                continue;
            }
            let mut stalled_now: Vec<usize> = refused;
            for (k, &(i, pages)) in plan.iter().enumerate() {
                let before = held[i];
                if ledger.grow(i, pages) {
                    self.stats.top_ups += 1;
                    self.stats.top_up_pages += (pages - before) as u64;
                } else {
                    // The pool and the plan disagree (a capture still queued): wait a round.
                    self.stats.refused_grants += 1;
                    stalled_now.extend(plan[k..].iter().map(|&(i, _)| i));
                    break;
                }
            }
            // A settled pool refused a grant and nothing else runs: the plan cannot be met,
            // so the same decision would repeat. Break it like a deadlock.
            if settled && decoding > 0 && stalled_now.len() == decoding && stalled_now.len() > 0
                && plan.iter().all(|&(i, _)| stalled_now.contains(&i)) && !plan.is_empty() {
                let (demands, _) = ledger.demands();
                let victim = (0..decoding).max_by_key(|&i| (self.potential(&demands[i]), i)).expect("decoding");
                drop(demands);
                self.stats.deadlock_breaks += 1;
                ledger.end(victim);
                continue;
            }
            for i in 0..decoding {
                let stall = stalled_now.contains(&i);
                if stall && !was[i] { self.stats.stalls += 1; }
                if stall != was[i] { ledger.set_stalled(i, stall); }
            }
            return;
        }
    }

}

#[cfg(test)]
#[path = "reserve_tests.rs"]
mod tests;
