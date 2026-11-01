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
//! potential allocation at the default buffer), and the first one refused stalls together with
//! every candidate after it: the requests that stall are always those with the most potential
//! allocation remaining, and the one that can finish from what is free always runs.
//!
//! Everything here is host arithmetic over page counts. The scheduler counts `available` as
//! free pages plus pages only inactive snapshots hold (those `PrefixCache::grow` may evict).
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
    /// Rows the request may commit: prompt + maximum output, within the context and the pool.
    /// Rounds never write past it (their verify rows are clamped to it).
    end: usize,
    /// Rows guaranteed under the drain buffer (`end` with the default buffer).
    allowance: usize,
}

impl Lease {
    pub fn prompt(&self) -> usize { self.prompt }
    pub fn step(&self) -> usize { self.step }
    /// The request's logical capacity: it finishes with `length` here.
    pub fn end(&self) -> usize { self.end }
    /// Rows the reservation may reach.
    pub fn lifetime(&self) -> usize { self.end }
    /// Rows the next round needs reserved, from `committed` rows.
    pub fn round_rows(&self, committed: usize) -> usize { committed.saturating_add(self.step).min(self.end).max(committed) }
    /// Output tokens the lease allows.
    pub fn output(&self) -> usize { self.end - self.prompt }
}

#[derive(Debug, Error, PartialEq, Eq)]
#[error("prompt of {prompt} tokens leaves no room for output within {limit} KV rows")]
pub struct NoRoom { pub prompt: usize, pub limit: usize }

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
    pub top_ups: u64,
    pub top_up_pages: u64,
    /// Transitions into the stalled state.
    pub stalls: u64,
    pub early_terminations: u64,
    /// Every request stalled with nothing able to return a page (a policy defect: the request
    /// with the most potential ended with `length` so the server keeps going).
    pub deadlock_breaks: u64,
    /// Admissions that waited for the drain rule.
    pub admission_waits: u64,
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
        let allowance = match self.config.drain_buffer {
            None => end,
            Some(buffer) => prompt.saturating_add(buffer).saturating_add(step).min(end),
        };
        Ok(Lease { prompt, step, end, allowance })
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
        self.target_rows(lease.prompt, lease.round_rows(lease.prompt), lease.end)
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

    /// Pages still to come before its guaranteed end (the banker's need).
    pub fn need(&self, demand: &Demand) -> usize {
        if self.terminable(demand) { return 0; }
        self.pages_for(demand.lease.allowance.max(demand.lease.round_rows(demand.committed))).saturating_sub(demand.held)
    }

    /// Pages still to come before the end of its life.
    pub fn potential(&self, demand: &Demand) -> usize {
        self.pages_for(demand.lease.end).saturating_sub(demand.held)
    }

    /// Banker's safety: some order lets every request reach its guaranteed end. Releasing a
    /// request only adds pages and what a set of releases returns does not depend on their
    /// order, so taking the smallest need first finds an order whenever one exists.
    pub fn safe<'a>(&self, available: usize, demands: impl IntoIterator<Item = Demand<'a>>) -> bool {
        let demands: Vec<Demand<'a>> = demands.into_iter().collect();
        let mut live: std::collections::HashMap<u32, u32> = Default::default();
        for d in &demands {
            for &page in d.pages { *live.entry(page).or_default() += 1; }
        }
        let mut order: Vec<(usize, usize)> = demands.iter().enumerate().map(|(i, d)| (self.need(d), i)).collect();
        order.sort_unstable();
        let mut work = available;
        for (need, i) in order {
            if need > work { return false; }
            let d = &demands[i];
            // Pages beyond its listed ids (a trial grant) are its own.
            work += d.held.saturating_sub(d.pages.len());
            for &page in d.pages {
                let refs = live.get_mut(&page).expect("counted");
                *refs -= 1;
                if *refs == 0 { work += 1; }
            }
        }
        true
    }

    /// Pages to reserve for a new request, or `None` while it waits: a request is stalled (it
    /// goes first), the pool lacks the pages, or the drain rule would not hold afterwards.
    pub fn admit(&mut self, available: usize, demands: &[Demand<'_>], lease: &Lease, stalled: bool) -> Option<usize> {
        let pages = self.admission_pages(lease);
        let new = Demand { lease: *lease, held: pages, pages: &[], committed: lease.prompt, produced: 0, decoding: false };
        let fits = !stalled && pages <= available
            && self.safe(available - pages, demands.iter().copied().chain([new]));
        if fits { Some(pages) } else {
            self.stats.admission_waits += 1;
            None
        }
    }

    /// One round's actions for every in-flight request. `stalled` marks requests stalled last
    /// round (for the transition counter); `settled` says nothing outside these requests holds
    /// pages that will come back by themselves (a queued snapshot capture), so a round in which
    /// every request stalls is a real deadlock rather than a transient one.
    pub fn plan(&mut self, available: usize, demands: &[Demand<'_>], stalled: &[bool], settled: bool) -> Vec<Action> {
        let mut actions = vec![Action::Run; demands.len()];
        let mut state: Vec<Demand<'_>> = demands.to_vec();
        let mut gone = vec![false; demands.len()];
        let mut available = available;
        let mut candidates: Vec<usize> = (0..demands.len()).filter(|&i| self.needs_top_up(&demands[i])).collect();
        candidates.sort_by_key(|&i| (self.need(&demands[i]), self.potential(&demands[i]), i));
        let mut refused = false;
        for &i in &candidates {
            let d = state[i];
            if !refused {
                let round = d.lease.round_rows(d.committed);
                let minimal = self.pages_for(round).saturating_sub(d.held);
                let full = self.pages_for(self.target_rows(d.held * self.page_rows, round, d.lease.end))
                    .saturating_sub(d.held).max(minimal);
                let grant = [full, minimal].into_iter().find(|&grant| grant <= available && {
                    let mut trial = d;
                    trial.held += grant;
                    self.safe(available - grant, state.iter().enumerate()
                        .filter(|&(j, _)| !gone[j]).map(|(j, s)| if j == i { trial } else { *s }))
                });
                if let Some(grant) = grant {
                    // `pages` stays: the grant's fresh pages are counted as its own.
                    available -= grant;
                    state[i].held += grant;
                    actions[i] = Action::Grow(state[i].held);
                    self.stats.top_ups += 1;
                    self.stats.top_up_pages += grant as u64;
                    continue;
                }
            }
            if self.config.termination.is_some_and(|threshold| d.produced >= threshold) {
                actions[i] = Action::Terminate;
                gone[i] = true;
                // Only its own pages come back now; shared ones wait for their other holders.
                available += d.held.saturating_sub(d.pages.len()) + d.pages.iter().filter(|&&page|
                    !state.iter().enumerate().any(|(j, s)| j != i && !gone[j] && s.pages.contains(&page))).count();
                self.stats.early_terminations += 1;
                continue;
            }
            refused = true;
            actions[i] = Action::Stall;
            if !stalled.get(i).copied().unwrap_or(false) { self.stats.stalls += 1; }
        }
        // Every request waits for pages and none can come back: end the largest.
        if settled && !actions.is_empty() && actions.iter().all(|a| *a == Action::Stall) {
            let victim = (0..demands.len()).max_by_key(|&i| (self.potential(&demands[i]), i)).expect("nonempty");
            actions[victim] = Action::Terminate;
            self.stats.deadlock_breaks += 1;
        }
        actions
    }
}

#[cfg(test)]
#[path = "reserve_tests.rs"]
mod tests;
