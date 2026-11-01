use super::*;

const ROWS: usize = 64;

fn policy(chunk: usize, buffer: usize, early: i64, max_output: usize) -> KvReserve {
    KvReserve::new(ReserveConfig::new(chunk, buffer, early, max_output).unwrap(), ROWS)
}

#[test]
fn options_validate_against_the_effective_maximum_output() {
    let max = 131_072;
    assert_eq!(ReserveConfig::new(0, 0, -1, max), Err(ReserveConfigError::Chunk));
    assert_eq!(ReserveConfig::new(16_384, max, -1, max), Err(ReserveConfigError::Buffer { buffer: max, max_output: max }));
    assert_eq!(ReserveConfig::new(16_384, 0, -2, max), Err(ReserveConfigError::Threshold(-2)));
    // The startup refusal: an explicit threshold above the drain allowance.
    assert_eq!(ReserveConfig::new(16_384, 4096, 8192, max),
        Err(ReserveConfigError::BufferBelowThreshold { buffer: 4096, threshold: 8192 }));
    assert!(ReserveConfig::new(16_384, 0, max as i64 + 1, max).is_err());
    let default = ReserveConfig::new(16_384, 0, -1, max).unwrap();
    assert_eq!((default.drain_buffer(), default.termination_threshold()), (None, None));
    // A nonzero buffer with no explicit threshold terminates at the buffer.
    let buffered = ReserveConfig::new(16_384, 8192, -1, max).unwrap();
    assert_eq!((buffered.drain_buffer(), buffered.termination_threshold()), (Some(8192), Some(8192)));
    let both = ReserveConfig::new(16_384, 8192, 4096, max).unwrap();
    assert_eq!(both.termination_threshold(), Some(4096));
    let early = ReserveConfig::new(16_384, 0, 4096, max).unwrap();
    assert_eq!((early.drain_buffer(), early.termination_threshold()), (None, Some(4096)));
}

#[test]
fn admission_reserves_the_prompt_plus_a_chunk_rounded_down_to_a_page() {
    let p = policy(16_384, 0, -1, 131_072);
    // 1000 + 16384 = 17384 rows, rounded down to 17344 (271 pages).
    let lease = p.lease(1000, 131_072, 8, 1 << 20).unwrap();
    assert_eq!(p.admission_rows(&lease), 17_344);
    assert_eq!(p.admission_pages(&lease), 271);
    // Never below the prompt's own pages plus its first round.
    let tiny = policy(64, 0, -1, 131_072);
    let lease = tiny.lease(1000, 131_072, 8, 1 << 20).unwrap();
    assert_eq!(tiny.admission_rows(&lease), 1024);
    assert!(tiny.admission_pages(&lease) * ROWS >= 1000 + 8);
}

#[test]
fn a_tail_under_half_a_chunk_reserves_the_whole_lifetime() {
    let p = policy(16_384, 0, -1, 131_072);
    // Lifetime 1000 + 20000 + 8: the chunk target 17344 leaves 3664 < 8192, so all of it.
    let lease = p.lease(1000, 20_000, 8, 1 << 20).unwrap();
    assert_eq!(p.admission_rows(&lease), 21_008);
    // A tail of at least half a chunk keeps the chunk.
    let lease = p.lease(1000, 30_000, 8, 1 << 20).unwrap();
    assert_eq!(p.admission_rows(&lease), 17_344);
    // Short outputs reserve their lifetime at once.
    let lease = p.lease(1000, 100, 8, 1 << 20).unwrap();
    assert_eq!(p.admission_rows(&lease), 1108);
}

#[test]
fn lease_respects_the_context_and_refuses_no_room() {
    let p = policy(16_384, 0, -1, 131_072);
    let lease = p.lease(1000, 131_072, 8, 4096).unwrap();
    assert_eq!((lease.end(), lease.lifetime(), lease.output()), (4096, 4096, 3096));
    assert_eq!(p.lease(1000, 2000, 8, 4096).unwrap().lifetime(), 3008);
    // The last round's rows stay within the limit.
    assert_eq!((lease.round_rows(4090), lease.round_rows(4096)), (4096, 4096));
    assert_eq!(p.lease(4096, 10, 8, 4096), Err(NoRoom { prompt: 4096, limit: 4096 }));
    assert!(p.lease(10, 0, 8, 4096).is_err());
}

fn decoding(p: &KvReserve, prompt: usize, output: usize, held: usize, committed: usize) -> Demand<'static> {
    let lease = p.lease(prompt, output, 8, 1 << 20).unwrap();
    Demand { lease, held, pages: &[], committed, produced: committed - prompt, decoding: true, draft: None }
}

// --- A model pool behind the Ledger -------------------------------------------------------

/// A target page pool tracked page by page (a parity pool hands logical page `j` a page of half
/// `j % 2`), a draft pool with `DraftPagePool::admit`'s credit rule, and pages a queued capture
/// holds for a while (the pool is unsettled until they come back). Its accounting is its own:
/// the policy's arithmetic is checked against it, not reused.
#[derive(Clone)]
struct Pool {
    free: [usize; 2],
    parity: bool,
    /// Free draft pages (credit already deducted), when a full-context drafter runs.
    draft_free: Option<usize>,
    draft_block: usize,
    /// Pages a queued capture holds, per half, and the steps until it lands.
    pending: Option<([usize; 2], usize)>,
}

impl Pool {
    fn capacity_rows(&self) -> usize { (self.free[0] + self.free[1]) * ROWS }
    fn take(&mut self, from: usize, to: usize) -> bool {
        let mut need = [0; 2];
        for j in from..to { need[if self.parity { j % 2 } else { 0 }] += 1; }
        if need[0] > self.free[0] || need[1] > self.free[1] { return false; }
        self.free[0] -= need[0];
        self.free[1] -= need[1];
        true
    }
    fn give(&mut self, from: usize, to: usize) {
        for j in from..to { self.free[if self.parity { j % 2 } else { 0 }] += 1; }
    }
    /// `DraftPagePool::admit`: the credit for `rows` from `origin` plus `block / 64 + 1` pages.
    fn credit(&self, rows: usize, origin: usize) -> usize {
        rows.saturating_sub(origin).div_ceil(crate::prefix::DRAFT_PAGE_ROWS) + self.draft_block.div_ceil(64) + 1
    }
}

#[derive(Clone)]
struct Live {
    spec: usize,
    lease: Lease,
    held: usize,
    draft_credit: usize,
    committed: usize,
    produced: usize,
    prefill_left: usize,
    stalled: bool,
}

/// The serve loop's side of a round over the model pool.
struct Sim<'s> {
    pool: &'s mut Pool,
    live: &'s mut Vec<Live>,
    finished: &'s mut Vec<Option<(Finish, usize)>>,
}

impl Sim<'_> {
    fn order(&self) -> (Vec<usize>, Vec<usize>) {
        let decoding = (0..self.live.len()).filter(|&i| self.live[i].prefill_left == 0).collect();
        let holding = (0..self.live.len()).filter(|&i| self.live[i].prefill_left > 0).collect();
        (decoding, holding)
    }
}

fn room(pool: &Pool) -> Room {
    let room = if pool.parity { Room::parity(pool.free) } else { Room::uniform(pool.free[0]) };
    match pool.draft_free { Some(free) => room.with_draft(free), None => room }
}

fn demand(l: &Live, pool: &Pool) -> Demand<'static> {
    let decoding = l.prefill_left == 0;
    Demand { lease: l.lease, held: l.held, pages: &[], committed: if decoding { l.committed } else { l.lease.prompt() },
        produced: l.produced, decoding,
        draft: pool.draft_free.map(|_| DraftDemand { origin: 0, headroom: pool.draft_block.div_ceil(64) + 1,
            held: l.draft_credit }) }
}

impl Ledger for Sim<'_> {
    fn room(&self) -> Room { room(self.pool) }
    fn settled(&self) -> bool { self.pool.pending.is_none() }
    fn demands(&self) -> (Vec<Demand<'_>>, Vec<bool>) {
        let (decoding, holding) = self.order();
        let demands = decoding.iter().chain(&holding).map(|&i| demand(&self.live[i], self.pool)).collect();
        (demands, decoding.iter().map(|&i| self.live[i].stalled).collect())
    }
    fn end(&mut self, index: usize) {
        let i = self.order().0[index];
        let l = self.live.remove(i);
        self.pool.give(0, l.held);
        if let Some(free) = &mut self.pool.draft_free { *free += l.draft_credit; }
        self.finished[l.spec] = Some((Finish::Early, l.produced));
    }
    fn grow(&mut self, index: usize, pages: usize) -> bool {
        let i = self.order().0[index];
        let (held, credit) = (self.live[i].held, self.live[i].draft_credit);
        let new_credit = self.pool.credit(pages * ROWS, 0).max(credit);
        if let Some(free) = self.pool.draft_free {
            if new_credit - credit > free { return false; }
        }
        if !self.pool.take(held, pages) { return false; }
        if let Some(free) = &mut self.pool.draft_free { *free -= new_credit - credit; }
        self.live[i].held = pages;
        self.live[i].draft_credit = new_credit;
        true
    }
    fn set_stalled(&mut self, index: usize, stalled: bool) {
        let i = self.order().0[index];
        self.live[i].stalled = stalled;
    }
}

#[derive(Clone, Copy)]
struct Spec {
    arrival: usize,
    prompt: usize,
    max_output: usize,
    /// Output it produces before EOS (≥ max_output: it runs to max_tokens).
    actual: usize,
    step: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Finish { Stop, Length, Early }

struct Outcome {
    finished: Vec<Option<(Finish, usize)>>,
    /// Output tokens each request's lease allowed (its max_tokens, within the context and pools).
    allowed: Vec<usize>,
    stats: ReserveStats,
    peak_concurrency: usize,
    pressure_rounds: usize,
}

const CONTEXT: usize = 1 << 20;

/// Rounds of a serve loop under the policy: FIFO admission, chunked prefill, then the shared
/// round (which ends requests and releases their pages inside it), then a decode step that
/// accepts 1..=step tokens. Completed requests release after the step; now and then a finished
/// request leaves a capture that holds its pages a few steps (an unsettled pool).
fn simulate(p: &mut KvReserve, pool: Pool, specs: &[Spec], rng: &mut Rng) -> Outcome {
    let mut pool = pool;
    let capacity = pool.capacity_rows();
    let draft_capacity = pool.draft_free;
    // Rows one request can ever hold: the target pool, and the draft pool less its headroom.
    let limit = draft_capacity.map_or(capacity, |draft| capacity.min(draft_lifetime_rows(draft, pool.draft_block)));
    let mut queue: std::collections::VecDeque<usize> = Default::default();
    let mut live: Vec<Live> = Vec::new();
    let mut finished = vec![None; specs.len()];
    let mut allowed = vec![0; specs.len()];
    let mut order: Vec<usize> = (0..specs.len()).collect();
    order.sort_by_key(|&i| (specs[i].arrival, i));
    let mut next = 0;
    let (mut round, mut idle, mut peak, mut pressure) = (0, 0, 0, 0);
    while finished.iter().any(Option::is_none) {
        round += 1;
        assert!(round < 5_000_000, "simulation did not finish");
        if let Some((held, steps)) = &mut pool.pending {
            *steps -= 1;
            if *steps == 0 {
                let held = *held;
                pool.free[0] += held[0];
                pool.free[1] += held[1];
                pool.pending = None;
            }
        }
        while next < order.len() && specs[order[next]].arrival <= round {
            queue.push_back(order[next]);
            next += 1;
        }
        // Admission, FIFO, against the pool as it stands.
        while let Some(&head) = queue.front() {
            let s = specs[head];
            let lease = p.lease(s.prompt, s.max_output, s.step, CONTEXT.min(limit)).unwrap();
            let demands: Vec<_> = live.iter().map(|l| demand(l, &pool)).collect();
            let stalled = live.iter().any(|l| l.stalled);
            let draft = pool.draft_free.map(|_| DraftDemand { origin: 0, headroom: pool.draft_block.div_ceil(64) + 1, held: 0 });
            let Some(pages) = p.admit(room(&pool), &demands, &lease, draft, stalled) else { break };
            let credit = pool.credit(pages * ROWS, 0);
            assert!(pool.draft_free.is_none_or(|free| credit <= free), "admission over the draft pool");
            assert!(pool.take(0, pages), "admission over the target pool");
            if let Some(free) = &mut pool.draft_free { *free -= credit; }
            queue.pop_front();
            allowed[head] = lease.output();
            live.push(Live { spec: head, lease, held: pages, draft_credit: credit, committed: 0, produced: 0,
                prefill_left: s.prompt, stalled: false });
        }
        peak = peak.max(live.len());
        if live.is_empty() { idle = 0; continue; }
        // Prefill one chunk per prefilling request (inside its admission reservation).
        let mut progressed = false;
        for l in live.iter_mut().filter(|l| l.prefill_left > 0) {
            let rows = l.prefill_left.min(8192);
            l.prefill_left -= rows;
            l.committed += rows;
            assert!(l.committed <= l.held * ROWS, "prefill wrote past its reservation");
            progressed = true;
        }
        let before = live.len();
        let stalls_before = p.stats().stalls;
        p.round(&mut Sim { pool: &mut pool, live: &mut live, finished: &mut finished });
        if live.len() < before || p.stats().stalls > stalls_before { pressure += 1; progressed |= live.len() < before; }
        let held: usize = live.iter().map(|l| l.held).sum();
        assert!(held + pool.free[0] + pool.free[1] + pool.pending.map_or(0, |(h, _)| h[0] + h[1]) == capacity / ROWS,
            "page accounting");
        if let (Some(cap), Some(free)) = (draft_capacity, pool.draft_free) {
            let credit: usize = live.iter().map(|l| l.draft_credit).sum();
            assert_eq!(credit + free, cap, "draft accounting");
        }
        let mut done = Vec::new();
        for (i, l) in live.iter_mut().enumerate() {
            if l.stalled || l.prefill_left > 0 || l.committed < l.lease.prompt() { continue; }
            assert!(l.lease.round_rows(l.committed) <= l.held * ROWS, "a round would write past the reservation");
            let s = specs[l.spec];
            let accepted = rng.range(1, l.lease.step());
            let left = s.actual.min(l.lease.output()) - l.produced;
            let take = accepted.min(left).min(l.lease.end() - l.committed);
            l.produced += take;
            l.committed += take;
            progressed |= take > 0;
            if l.produced >= s.actual.min(l.lease.output()) {
                done.push((i, if s.actual < l.lease.output() { Finish::Stop } else { Finish::Length }));
            } else if l.committed >= l.lease.end() {
                done.push((i, Finish::Length));
            }
        }
        for &(i, finish) in done.iter().rev() {
            let l = live.remove(i);
            // A turn snapshot sometimes holds the request's pages for a few steps.
            if pool.pending.is_none() && rng.range(0, 3) == 0 {
                let mut held = [0; 2];
                for j in 0..l.held { held[if pool.parity { j % 2 } else { 0 }] += 1; }
                pool.pending = Some((held, rng.range(1, 4)));
            } else {
                pool.give(0, l.held);
            }
            if let Some(free) = &mut pool.draft_free { *free += l.draft_credit; }
            finished[l.spec] = Some((finish, l.produced));
            progressed = true;
        }
        if progressed { idle = 0 } else {
            idle += 1;
            assert!(idle < 8, "no progress: livelock with {} live requests", live.len());
        }
    }
    Outcome { finished, allowed, stats: p.stats(), peak_concurrency: peak, pressure_rounds: pressure }
}

fn uniform(pages: usize) -> Pool { Pool { free: [pages, 0], parity: false, draft_free: None, draft_block: 0, pending: None } }
fn parity(pages: usize) -> Pool { Pool { free: [pages / 2, pages / 2], parity: true, draft_free: None, draft_block: 0, pending: None } }
fn drafted(mut pool: Pool, block: usize) -> Pool {
    // The real draft pool: as many 64-row pages as the target pool's rows, no headroom.
    pool.draft_free = Some(pool.capacity_rows().div_ceil(64));
    pool.draft_block = block;
    pool
}

fn random_specs(rng: &mut Rng, n: usize, max_output: usize) -> Vec<Spec> {
    (0..n).map(|_| {
        let max = if rng.range(0, 3) == 0 { rng.range(1, max_output) } else { max_output };
        Spec { arrival: rng.range(0, 400), prompt: rng.range(1, 40_000), max_output: max,
            actual: if rng.range(0, 4) == 0 { usize::MAX } else { rng.range(1, max_output) }, step: rng.range(1, 16) }
    }).collect()
}

fn check_complete(specs: &[Spec], outcome: &Outcome) {
    for ((s, finished), &allowed) in specs.iter().zip(&outcome.finished).zip(&outcome.allowed) {
        let (finish, produced) = finished.expect("every request finishes");
        assert_ne!(finish, Finish::Early, "no early termination by default");
        assert!(allowed <= s.max_output);
        assert_eq!(produced, s.actual.min(allowed), "no truncation below the lease's output");
    }
    assert_eq!(outcome.stats.deadlock_breaks, 0, "the default buffer never deadlocks");
    assert_eq!(outcome.stats.early_terminations, 0);
}

/// The pool shapes every simulator case runs on: uniform, parity (context-split TP2), each with
/// and without a full-context drafter.
fn shapes(pages: usize) -> [Pool; 4] {
    [uniform(pages), parity(pages), drafted(uniform(pages), 16), drafted(parity(pages), 64)]
}

#[test]
fn simulator_default_buffer_never_deadlocks_or_truncates() {
    for seed in 1..=24u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let max_output = 131_072;
        let specs = random_specs(&mut rng, 32, max_output);
        // Pools from barely one full lifetime (odd page counts too) up to plenty.
        let pages = (40_000 + max_output + 16).div_ceil(ROWS) * [1, 2, 4, 16][seed as usize % 4] + seed as usize % 2 * 2;
        for pool in shapes(pages) {
            let mut p = policy([4096, 16_384, 65_536][seed as usize % 3], 0, -1, max_output);
            let outcome = simulate(&mut p, pool, &specs, &mut rng);
            check_complete(&specs, &outcome);
        }
    }
}

#[test]
fn simulator_chunks_raise_concurrency_over_whole_lifetime_reservation() {
    let mut rng = Rng(7);
    // Sixteen requests omit max_tokens (131072) but stop after a few thousand tokens.
    let specs: Vec<Spec> = (0..16).map(|_| Spec { arrival: 0, prompt: rng.range(500, 4000), max_output: 131_072,
        actual: rng.range(1000, 6000), step: 8 }).collect();
    let pages = 4 * (131_072 + 4000 + 8) / ROWS;
    for pool in shapes(pages) {
        let mut chunked = policy(16_384, 0, -1, 131_072);
        let outcome = simulate(&mut chunked, pool, &specs, &mut Rng(9));
        check_complete(&specs, &outcome);
        // Lifetime reservation would fit 4 at once; the drain rule admits until only one
        // lifetime of headroom is left — the hardware gate's C16 case.
        assert!(outcome.peak_concurrency >= 12, "peak {}", outcome.peak_concurrency);
    }
}

#[test]
fn simulator_adversarial_everyone_runs_to_max_tokens_at_once() {
    // Every request runs to max_tokens, all arrive together, the pool holds ~2.5 lifetimes.
    for (chunk, step) in [(4096, 1), (16_384, 16), (1024, 8)] {
        let max_output = 60_000;
        let specs: Vec<Spec> = (0..24).map(|i| Spec { arrival: 0, prompt: 1000 + 997 * i, max_output,
            actual: usize::MAX, step }).collect();
        for pool in shapes((5 * (24_000 + max_output + 16) / 2).div_ceil(ROWS) | 1) {
            let mut p = policy(chunk, 0, -1, max_output);
            let outcome = simulate(&mut p, pool, &specs, &mut Rng(11));
            check_complete(&specs, &outcome);
            assert!(outcome.stats.stalls > 0 && outcome.pressure_rounds > 0, "the schedule is under pressure");
        }
    }
    // Long runners admitted first, then a flood of short ones that must not starve them.
    let mut specs = vec![Spec { arrival: 0, prompt: 30_000, max_output: 200_000, actual: usize::MAX, step: 8 }; 3];
    specs.extend((0..40).map(|i| Spec { arrival: 1 + i, prompt: 2000, max_output: 200_000, actual: 3000, step: 8 }));
    for pool in shapes((2 * (30_000 + 200_000 + 8usize)).div_ceil(ROWS)) {
        let mut p = policy(16_384, 0, -1, 200_000);
        check_complete(&specs, &simulate(&mut p, pool, &specs, &mut Rng(13)));
    }
}

#[test]
fn simulator_drain_buffer_terminates_only_past_the_buffer_and_only_under_pressure() {
    let max_output = 100_000;
    let buffer = 8192;
    for seed in 1..=8u64 {
        let mut rng = Rng(seed * 31 + 5);
        let specs = random_specs(&mut rng, 32, max_output);
        // Small pool: room for only a few buffers' worth beyond the prompts.
        for pool in shapes((40_000 + buffer + 16usize).div_ceil(ROWS) * 3) {
            let mut p = policy(4096, buffer, -1, max_output);
            let outcome = simulate(&mut p, pool, &specs, &mut rng);
            assert_eq!(outcome.stats.deadlock_breaks, 0);
            for ((s, finished), &allowed) in specs.iter().zip(&outcome.finished).zip(&outcome.allowed) {
                let (finish, produced) = finished.unwrap();
                match finish {
                    Finish::Early => assert!(produced >= buffer, "terminated at {produced} < buffer"),
                    _ => assert_eq!(produced, s.actual.min(allowed)),
                }
            }
        }
    }
    // No pressure, no termination.
    let mut rng = Rng(3);
    let specs = random_specs(&mut rng, 16, max_output);
    for pool in shapes(16 * (40_000 + max_output + 16) / ROWS) {
        let mut p = policy(4096, buffer, -1, max_output);
        let outcome = simulate(&mut p, pool, &specs, &mut rng);
        check_complete(&specs, &outcome);
    }
}

#[test]
fn simulator_early_termination_happens_only_past_the_threshold_where_a_stall_would_occur() {
    let max_output = 100_000;
    let threshold = 4096;
    for seed in 1..=8u64 {
        let mut rng = Rng(seed * 77 + 1);
        let specs = random_specs(&mut rng, 32, max_output);
        for pool in shapes((40_000 + max_output + 16).div_ceil(ROWS) * 2) {
            let mut p = policy(4096, 0, threshold as i64, max_output);
            let outcome = simulate(&mut p, pool, &specs, &mut rng);
            assert_eq!(outcome.stats.deadlock_breaks, 0);
            let mut early = 0;
            for ((s, finished), &allowed) in specs.iter().zip(&outcome.finished).zip(&outcome.allowed) {
                let (finish, produced) = finished.unwrap();
                if finish == Finish::Early {
                    assert!(produced >= threshold);
                    early += 1;
                } else {
                    assert_eq!(produced, s.actual.min(allowed));
                }
            }
            assert!(early == 0 || outcome.pressure_rounds > 0);
        }
    }
    let mut rng = Rng(5);
    let specs = random_specs(&mut rng, 16, max_output);
    for pool in shapes(16 * (40_000 + max_output + 16) / ROWS) {
        let mut p = policy(4096, 0, threshold as i64, max_output);
        let outcome = simulate(&mut p, pool, &specs, &mut rng);
        check_complete(&specs, &outcome);
    }
}

// --- Regressions for the PR #954 review (each failed at e0be6814e) -------------------------

/// Finding 1: three admissions of 257 pages (odd) into a 964-page parity pool. A scalar
/// `2 * min(halves)` lost a page per odd placement, so the last admission left one page too few
/// and the default buffer ended a request through the deadlock break.
#[test]
fn parity_pools_charge_odd_allocations_per_half_and_never_deadlock() {
    let specs = vec![Spec { arrival: 0, prompt: 64, max_output: 28_540, actual: usize::MAX, step: 64 }; 3];
    let mut p = policy(16_384, 0, -1, 28_540);
    let outcome = simulate(&mut p, parity(964), &specs, &mut Rng(1));
    check_complete(&specs, &outcome);
    // The exact per-half charge: 257 pages from position 0 take 129 even, 128 odd.
    assert_eq!(Room::parity([0, 0]).need(0, 257), [129, 128]);
    assert_eq!(Room::parity([0, 0]).need(257, 449), [96, 96]);
}

/// Re-review finding 2: candidates are tried least need first, but a refusal does not block
/// later ones. Parity pool of 6 pages, room [0, 1] after admissions: A needs half 0 (refused),
/// B needs half 1 (fits). B used to be skipped, the round looked all-stalled and settled, and
/// the deadlock break ended a request with `length` under the default buffer.
#[test]
fn a_refused_candidate_does_not_block_one_the_other_half_can_hold() {
    let mut p = policy(64, 0, -1, 1 << 20);
    let a = p.lease(64, 100, 1, 6 * ROWS).unwrap();
    let b = p.lease(128, 96, 1, 6 * ROWS).unwrap();
    // A holds 2 pages (positions 0-1: [1, 1]) at 128 rows; B holds 3 (positions 0-2: [2, 1]).
    let a = Demand { lease: a, held: 2, pages: &[], committed: 128, produced: 64, decoding: true, draft: None };
    let b = Demand { lease: b, held: 3, pages: &[], committed: 192, produced: 64, decoding: true, draft: None };
    assert!(p.needs_top_up(&a) && p.needs_top_up(&b));
    assert_eq!((Room::parity([0, 1]).need(2, 3), Room::parity([0, 1]).need(3, 4)), ([1, 0], [0, 1]));
    struct Two { live: Vec<(Demand<'static>, bool)>, room: Room, ended: usize }
    impl Ledger for Two {
        fn room(&self) -> Room { self.room }
        fn settled(&self) -> bool { true }
        fn demands(&self) -> (Vec<Demand<'_>>, Vec<bool>) {
            (self.live.iter().map(|(d, _)| *d).collect(), self.live.iter().map(|(_, s)| *s).collect())
        }
        fn end(&mut self, index: usize) { self.live.remove(index); self.ended += 1; }
        fn grow(&mut self, index: usize, pages: usize) -> bool {
            let need = self.room.need(self.live[index].0.held, pages);
            assert!(need[0] <= self.room.halves[0] && need[1] <= self.room.halves[1]);
            self.room.halves = [self.room.halves[0] - need[0], self.room.halves[1] - need[1]];
            self.live[index].0.held = pages;
            true
        }
        fn set_stalled(&mut self, index: usize, stalled: bool) { self.live[index].1 = stalled; }
    }
    let mut ledger = Two { live: vec![(a, false), (b, false)], room: Room::parity([0, 1]), ended: 0 };
    p.round(&mut ledger);
    assert_eq!((ledger.ended, p.stats().deadlock_breaks), (0, 0), "no truncation");
    assert_eq!((ledger.live[0].1, ledger.live[1].1, ledger.live[1].0.held), (true, false, 4));
    // The simulator reaches the same state with these two requests in that pool, end to end.
    let specs = vec![Spec { arrival: 0, prompt: 64, max_output: 100, actual: usize::MAX, step: 1 },
        Spec { arrival: 0, prompt: 128, max_output: 96, actual: usize::MAX, step: 1 }];
    let mut p = policy(64, 0, -1, 1 << 20);
    check_complete(&specs, &simulate(&mut p, parity(6), &specs, &mut Rng(3)));
}

/// Finding 2: the drafter's credit carries `block / 64 + 1` pages beyond the target pages, and
/// the draft pool has no such headroom. A request whose lifetime reaches the pool's capacity
/// used to be granted the same refused top-up every round (a livelock).
#[test]
fn draft_pool_headroom_is_charged_so_the_last_top_up_never_livelocks() {
    // One request runs to a lifetime equal to the target pool.
    let pages = 300;
    let specs = vec![Spec { arrival: 0, prompt: 64, max_output: pages * ROWS, actual: usize::MAX, step: 16 }];
    for block in [16, 64] {
        let mut p = policy(4096, 0, -1, pages * ROWS);
        let outcome = simulate(&mut p, drafted(uniform(pages), block), &specs, &mut Rng(2));
        // The pool cannot hold the drafter's headroom at the very end: the policy ends it at
        // its reservation's limit with `length`, never loops and never breaks a deadlock.
        check_complete(&specs, &outcome);
        let (finish, produced) = outcome.finished[0].unwrap();
        // It runs to its lease's end: the pool's rows less the drafter's headroom pages.
        let headroom = block.div_ceil(64) + 1;
        assert_eq!((finish, produced + 64), (Finish::Length, (pages - headroom) * ROWS));
    }
    // Several requests near the limit with drafters: all finish.
    let mut rng = Rng(4);
    let specs = random_specs(&mut rng, 24, 60_000);
    for block in [16, 64] {
        let mut p = policy(4096, 0, -1, 60_000);
        let outcome = simulate(&mut p, drafted(uniform(1100), block), &specs, &mut rng);
        check_complete(&specs, &outcome);
    }
}

/// A grant the pool refuses (the plan and the pool disagree) is not counted as a top-up, stalls
/// the request for the round, and a settled pool that keeps refusing ends the largest request
/// instead of repeating the same round forever.
#[test]
fn refused_grants_are_not_counted_and_cannot_repeat_forever() {
    struct Stubborn { live: Vec<(Demand<'static>, bool)>, ended: usize, settled: bool }
    impl Ledger for Stubborn {
        fn room(&self) -> Room { Room::uniform(10_000) }
        fn settled(&self) -> bool { self.settled }
        fn demands(&self) -> (Vec<Demand<'_>>, Vec<bool>) {
            (self.live.iter().map(|(d, _)| *d).collect(), self.live.iter().map(|(_, s)| *s).collect())
        }
        fn end(&mut self, index: usize) { self.live.remove(index); self.ended += 1; }
        fn grow(&mut self, _: usize, _: usize) -> bool { false }
        fn set_stalled(&mut self, index: usize, stalled: bool) { self.live[index].1 = stalled; }
    }
    let p0 = policy(16_384, 0, -1, 131_072);
    let d = decoding(&p0, 1000, 131_072, 271, 17_340);
    // Unsettled (a capture still queued): stall and retry next round, nothing counted or ended.
    let mut p = policy(16_384, 0, -1, 131_072);
    let mut ledger = Stubborn { live: vec![(d, false)], ended: 0, settled: false };
    p.round(&mut ledger);
    assert_eq!((p.stats().top_ups, p.stats().refused_grants, ledger.ended, ledger.live[0].1), (0, 1, 0, true));
    // Settled: the same refusal would repeat, so the round ends it.
    ledger.settled = true;
    p.round(&mut ledger);
    assert_eq!((ledger.ended, p.stats().deadlock_breaks, p.stats().top_ups), (1, 1, 0));
}

/// Finding 4: a request that ends early returns its pages before anything is granted, so the
/// candidate after it grows in the same round (no false stall) and counters match the pool.
#[test]
fn ended_requests_release_before_grants_and_counters_match_the_pool() {
    let pages = 2 * 271 + 10;
    let mut pool = uniform(pages);
    let mut p = policy(16_384, 0, 1000, 131_072);
    // Leases are capped at what the pool can hold (`Reserve::limit`).
    let lease = p.lease(1000, 131_072, 8, pages * ROWS).unwrap();
    let mut live: Vec<Live> = (0..2).map(|spec| Live { spec, lease, held: 271, draft_credit: 0,
        committed: 17_340, produced: if spec == 0 { 16_340 } else { 500 }, prefill_left: 0, stalled: false }).collect();
    assert!(pool.take(0, 271) && pool.take(0, 271));
    let mut finished = vec![None; 2];
    p.round(&mut Sim { pool: &mut pool, live: &mut live, finished: &mut finished });
    // Spec 0 (past the threshold, most potential) ended; spec 1 grew in the same round.
    assert_eq!(finished[0].map(|(f, _)| f), Some(Finish::Early));
    assert_eq!((live.len(), live[0].stalled), (1, false));
    assert!(live[0].held > 271);
    assert_eq!((p.stats().top_ups, p.stats().refused_grants), (1, 0));
    // The counter adds the pages the grant added, not the size it grew to.
    assert_eq!(p.stats().top_up_pages as usize, live[0].held - 271);
}

#[test]
fn the_request_with_the_most_potential_stalls_first() {
    let p0 = policy(4096, 0, -1, 1 << 20);
    // Only the short one can finish from what is free: it grows, the two long ones stall.
    let short = decoding(&p0, 1000, 9000, 80, 5115);
    let long = decoding(&p0, 1000, 100_000, 1500, 95_995);
    let longer = decoding(&p0, 1000, 120_000, 1800, 115_195);
    let available = p0.need(&short);
    assert!(p0.safe(Room::uniform(available), [longer, short, long]));
    struct Fixed { live: Vec<(Demand<'static>, bool)>, free: usize, grown: Vec<usize> }
    impl Ledger for Fixed {
        fn room(&self) -> Room { Room::uniform(self.free) }
        fn settled(&self) -> bool { true }
        fn demands(&self) -> (Vec<Demand<'_>>, Vec<bool>) {
            (self.live.iter().map(|(d, _)| *d).collect(), self.live.iter().map(|(_, s)| *s).collect())
        }
        fn end(&mut self, _: usize) { panic!("nothing ends") }
        fn grow(&mut self, index: usize, pages: usize) -> bool {
            self.free -= pages - self.live[index].0.held;
            self.live[index].0.held = pages;
            self.grown.push(index);
            true
        }
        fn set_stalled(&mut self, index: usize, stalled: bool) { self.live[index].1 = stalled; }
    }
    let mut p = policy(4096, 0, -1, 1 << 20);
    let mut ledger = Fixed { live: vec![(longer, false), (short, false), (long, false)], free: available, grown: Vec::new() };
    p.round(&mut ledger);
    assert_eq!(ledger.grown, vec![1]);
    assert_eq!((ledger.live[0].1, ledger.live[1].1, ledger.live[2].1), (true, false, true));
    assert_eq!(ledger.live[1].0.held, p.pages_for(10_008));
    // With room for everyone, nobody stalls.
    let mut p = policy(4096, 0, -1, 1 << 20);
    let mut ledger = Fixed { live: vec![(longer, false), (short, false), (long, false)], free: available + 200,
        grown: Vec::new() };
    p.round(&mut ledger);
    assert_eq!(ledger.grown.len(), 3);
}

#[test]
fn admission_waits_while_a_request_is_stalled_or_the_drain_rule_would_break() {
    let mut p = policy(16_384, 0, -1, 131_072);
    let running = decoding(&p, 1000, 131_072, 271, 2000);
    let lease = p.lease(1000, 131_072, 8, 1 << 20).unwrap();
    // The running request needs ~1800 more pages; a newcomer taking 271 of 2000 breaks the rule.
    assert_eq!(p.admit(Room::uniform(2000), &[running], &lease, None, false), None);
    assert_eq!(p.admit(Room::uniform(100_000), &[running], &lease, None, true), None);
    assert_eq!(p.admit(Room::uniform(100_000), &[running], &lease, None, false), Some(271));
    assert_eq!(p.stats().admission_waits, 2);
}

#[test]
fn shared_pages_return_only_with_their_last_holder() {
    let p = policy(4096, 0, -1, 1 << 20);
    let lease = p.lease(640, 639, 1, 1 << 20).unwrap();
    // Two forks of one 10-page snapshot, each needing 10 more pages; 10 free.
    let a: &'static [u32] = (0..10).collect::<Vec<u32>>().leak();
    let b: &'static [u32] = (0..10).collect::<Vec<u32>>().leak();
    let fork = |pages: &'static [u32]| Demand { lease, held: 10, pages, committed: 640, produced: 0, decoding: true, draft: None };
    // Shared pages free nothing when the first finishes: the second still finds 10.
    assert!(p.safe(Room::uniform(10), [fork(a), fork(b)]));
    // With 5 free neither can finish, and the shared pages do not count as released.
    assert!(!p.safe(Room::uniform(5), [fork(a), fork(b)]));
    let private = |held| Demand { lease, held, pages: &[], committed: 640, produced: 0, decoding: true, draft: None };
    assert!(p.safe(Room::uniform(10), [private(10), private(10)]));
}

// --- Simulator support --------------------------------------------------------------------

/// Deterministic xorshift64*.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn range(&mut self, low: usize, high: usize) -> usize { low + (self.next() % (high - low + 1) as u64) as usize }
}
