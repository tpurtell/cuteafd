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
    // Lifetime 1000 + 20000: the chunk target 17344 leaves 3656 < 8192, so all of it.
    let lease = p.lease(1000, 20_000, 8, 1 << 20).unwrap();
    assert_eq!(p.admission_rows(&lease), 21_000);
    // A tail of at least half a chunk keeps the chunk.
    let lease = p.lease(1000, 30_000, 8, 1 << 20).unwrap();
    assert_eq!(p.admission_rows(&lease), 17_344);
    // Short outputs reserve their lifetime at once.
    let lease = p.lease(1000, 100, 8, 1 << 20).unwrap();
    assert_eq!(p.admission_rows(&lease), 1100);
}

#[test]
fn lease_respects_the_context_and_refuses_no_room() {
    let p = policy(16_384, 0, -1, 131_072);
    let lease = p.lease(1000, 131_072, 8, 4096).unwrap();
    assert_eq!((lease.end(), lease.lifetime(), lease.output()), (4096, 4096, 3096));
    // The last round's rows stay within the end.
    assert_eq!((lease.round_rows(4090), lease.round_rows(4096)), (4096, 4096));
    assert_eq!(p.lease(4096, 10, 8, 4096), Err(NoRoom { prompt: 4096, limit: 4096 }));
    assert!(p.lease(10, 0, 8, 4096).is_err());
}

fn decoding(p: &KvReserve, prompt: usize, output: usize, held: usize, committed: usize) -> Demand<'static> {
    let lease = p.lease(prompt, output, 8, 1 << 20).unwrap();
    Demand { lease, held, pages: &[], committed, produced: committed - prompt, decoding: true }
}

#[test]
fn top_up_grows_one_chunk_when_the_next_round_would_pass_the_reservation() {
    let mut p = policy(16_384, 0, -1, 131_072);
    let d = decoding(&p, 1000, 131_072, 271, 17_340);
    assert!(p.needs_top_up(&d));
    assert!(!p.needs_top_up(&decoding(&p, 1000, 131_072, 271, 17_336)));
    // 17344 + 16384 = 33728 rows = 527 pages.
    assert_eq!(p.plan(10_000, &[d], &[false], true), vec![Action::Grow(527)]);
    // The last top-up takes the tail when under half a chunk would remain.
    let d = decoding(&p, 1000, 40_000, 527, 33_725);
    assert_eq!(p.plan(10_000, &[d], &[false], true), vec![Action::Grow(p.pages_for(41_000))]);
    assert_eq!(p.stats().top_ups, 2);
}

#[test]
fn a_refused_top_up_stalls_and_resumes_once_pages_return() {
    let mut p = policy(16_384, 0, -1, 131_072);
    let small = decoding(&p, 1000, 131_072, 271, 17_340);
    let other = Demand { decoding: false, ..decoding(&p, 1000, 4000, 80, 1000) };
    // 100 free pages: the full chunk is unsafe; the minimal page is granted only if the drain
    // rule still holds, and it does not (the big request cannot finish from 100 + 80).
    let actions = p.plan(100, &[small, other], &[false, false], true);
    assert_eq!(actions, vec![Action::Stall, Action::Run]);
    assert_eq!(p.stats().stalls, 1);
    // Still stalled next round: no new transition.
    assert_eq!(p.plan(100, &[small, other], &[true, false], true)[0], Action::Stall);
    assert_eq!(p.stats().stalls, 1);
    // The other finished and the pool has room again: resume.
    let actions = p.plan(100 + 80 + 2000, &[small], &[true], true);
    assert!(matches!(actions[0], Action::Grow(_)));
}

#[test]
fn the_request_with_the_most_potential_stalls_first() {
    let mut p = policy(4096, 0, -1, 1 << 20);
    // Only the short one can finish from what is free: it grows, the two long ones stall.
    let short = decoding(&p, 1000, 9000, 80, 5115);
    let long = decoding(&p, 1000, 100_000, 1500, 95_995);
    let longer = decoding(&p, 1000, 120_000, 1800, 115_195);
    let demands = [longer, short, long];
    let available = p.need(&short);
    assert!(p.safe(available, demands));
    let actions = p.plan(available, &demands, &[false; 3], true);
    assert_eq!(actions, vec![Action::Stall, Action::Grow(p.pages_for(10_000)), Action::Stall]);
    // With room for everyone, nobody stalls.
    let actions = p.plan(available + 200, &demands, &[false; 3], true);
    assert!(actions.iter().all(|a| matches!(a, Action::Grow(_))), "{actions:?}");
}

#[test]
fn admission_waits_while_a_request_is_stalled_or_the_drain_rule_would_break() {
    let mut p = policy(16_384, 0, -1, 131_072);
    let running = decoding(&p, 1000, 131_072, 271, 2000);
    let lease = p.lease(1000, 131_072, 8, 1 << 20).unwrap();
    // The running request needs ~1800 more pages; a newcomer taking 271 of 2000 breaks the rule.
    assert_eq!(p.admit(2000, &[running], &lease, false), None);
    assert_eq!(p.admit(100_000, &[running], &lease, true), None);
    assert_eq!(p.admit(100_000, &[running], &lease, false), Some(271));
    assert_eq!(p.stats().admission_waits, 2);
}

#[test]
fn early_termination_ends_a_request_past_the_threshold_instead_of_stalling() {
    let mut p = policy(4096, 0, 2000, 1 << 20);
    let past = decoding(&p, 1000, 100_000, 80, 5115);
    let young = decoding(&p, 4000, 100_000, 80, 5115);
    let actions = p.plan(0, &[past, young], &[false, false], true);
    // `past` produced 4115 ≥ 2000: ends; `young` produced 1115: stalls.
    assert_eq!(actions, vec![Action::Terminate, Action::Stall]);
    assert_eq!(p.stats().early_terminations, 1);
}

// --- Simulator ---------------------------------------------------------------------------

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

struct Live {
    spec: usize,
    lease: Lease,
    held: usize,
    committed: usize,
    produced: usize,
    prefill_left: usize,
    stalled: bool,
}

struct Outcome {
    finished: Vec<Option<(Finish, usize)>>,
    stats: ReserveStats,
    peak_concurrency: usize,
    pressure_rounds: usize,
}

const CONTEXT: usize = 1 << 20;

/// Rounds of a serve loop under the policy: FIFO admission, chunked prefill, decode rounds that
/// accept 1..=step tokens. Asserts every invariant each round.
fn simulate(p: &mut KvReserve, pool: usize, specs: &[Spec], rng: &mut Rng) -> Outcome {
    let mut queue: std::collections::VecDeque<usize> = Default::default();
    let mut live: Vec<Live> = Vec::new();
    let mut finished = vec![None; specs.len()];
    let mut order: Vec<usize> = (0..specs.len()).collect();
    order.sort_by_key(|&i| (specs[i].arrival, i));
    let mut next = 0;
    let (mut round, mut idle, mut peak, mut pressure) = (0, 0, 0, 0);
    while finished.iter().any(Option::is_none) {
        round += 1;
        assert!(round < 5_000_000, "simulation did not finish");
        while next < order.len() && specs[order[next]].arrival <= round {
            queue.push_back(order[next]);
            next += 1;
        }
        let held: usize = live.iter().map(|l| l.held).sum();
        assert!(held <= pool, "reserved {held} pages > pool {pool}");
        let demands = |live: &[Live]| live.iter().map(|l| Demand { lease: l.lease, held: l.held, pages: &[],
            committed: if l.prefill_left > 0 { l.lease.prompt() } else { l.committed }, produced: l.produced,
            decoding: l.prefill_left == 0 }).collect::<Vec<_>>();
        // Admission, FIFO.
        while let Some(&head) = queue.front() {
            let s = specs[head];
            let lease = p.lease(s.prompt, s.max_output, s.step, CONTEXT.min(pool * ROWS)).unwrap();
            let available = pool - live.iter().map(|l| l.held).sum::<usize>();
            let stalled = live.iter().any(|l| l.stalled);
            let Some(pages) = p.admit(available, &demands(&live), &lease, stalled) else { break };
            assert!(p.safe(available - pages, demands(&live).into_iter().chain([Demand { lease, held: pages, pages: &[],
                committed: s.prompt, produced: 0, decoding: false }])));
            queue.pop_front();
            live.push(Live { spec: head, lease, held: pages, committed: 0, produced: 0,
                prefill_left: s.prompt, stalled: false });
        }
        peak = peak.max(live.len());
        if live.is_empty() { continue; }
        // Prefill one chunk per prefilling request (inside its admission reservation).
        let mut progressed = false;
        for l in live.iter_mut().filter(|l| l.prefill_left > 0) {
            let rows = l.prefill_left.min(8192);
            l.prefill_left -= rows;
            l.committed += rows;
            assert!(l.committed <= l.held * ROWS, "prefill wrote past its reservation");
            progressed = true;
        }
        let available = pool - live.iter().map(|l| l.held).sum::<usize>();
        let demands_now = demands(&live);
        let was: Vec<bool> = live.iter().map(|l| l.stalled).collect();
        let actions = p.plan(available, &demands_now, &was, true);
        // The stall victims are those with the most potential: none granted has more.
        let granted = (0..live.len()).filter(|&i| matches!(actions[i], Action::Grow(_)))
            .map(|i| p.potential(&demands_now[i])).max();
        for i in 0..live.len() {
            if actions[i] == Action::Stall {
                if let Some(granted) = granted {
                    if p.config().drain_buffer().is_none() {
                        assert!(p.potential(&demands_now[i]) >= granted, "a request with less potential stalled");
                    }
                }
            }
        }
        if actions.iter().any(|a| matches!(a, Action::Stall | Action::Terminate)) { pressure += 1; }
        let mut done = Vec::new();
        for (i, (l, action)) in live.iter_mut().zip(&actions).enumerate() {
            match *action {
                Action::Grow(pages) => { assert!(pages > l.held); l.held = pages; }
                Action::Stall => { l.stalled = true; continue; }
                Action::Terminate => { done.push((i, Finish::Early)); continue; }
                Action::Run => {}
            }
            l.stalled = false;
            if l.prefill_left > 0 || l.committed < l.lease.prompt() { continue; }
            assert!(l.lease.round_rows(l.committed) <= l.held * ROWS, "a round would write past the reservation");
            let s = specs[l.spec];
            let accepted = rng.range(1, l.lease.step());
            let left = s.actual.min(s.max_output) - l.produced;
            let take = accepted.min(left).min(l.lease.end() - l.committed);
            l.produced += take;
            l.committed += take;
            progressed |= take > 0;
            if l.produced >= s.actual.min(s.max_output) {
                done.push((i, if s.actual < s.max_output { Finish::Stop } else { Finish::Length }));
            } else if l.committed >= l.lease.end() {
                done.push((i, Finish::Length));
            }
        }
        for &(i, finish) in done.iter().rev() {
            let l = live.remove(i);
            finished[l.spec] = Some((finish, l.produced));
            progressed = true;
        }
        if progressed { idle = 0 } else {
            idle += 1;
            assert!(idle < 3, "no progress: deadlock with {} live requests", live.len());
        }
    }
    Outcome { finished, stats: p.stats(), peak_concurrency: peak, pressure_rounds: pressure }
}

fn random_specs(rng: &mut Rng, n: usize, max_output: usize) -> Vec<Spec> {
    (0..n).map(|_| {
        let max = if rng.range(0, 3) == 0 { rng.range(1, max_output) } else { max_output };
        Spec { arrival: rng.range(0, 400), prompt: rng.range(1, 40_000), max_output: max,
            actual: if rng.range(0, 4) == 0 { usize::MAX } else { rng.range(1, max_output) }, step: rng.range(1, 16) }
    }).collect()
}

fn check_complete(specs: &[Spec], outcome: &Outcome) {
    for (s, finished) in specs.iter().zip(&outcome.finished) {
        let (finish, produced) = finished.expect("every request finishes");
        assert_ne!(finish, Finish::Early, "no early termination by default");
        assert_eq!(produced, s.actual.min(s.max_output), "no truncation");
    }
    assert_eq!(outcome.stats.deadlock_breaks, 0);
    assert_eq!(outcome.stats.early_terminations, 0);
}

#[test]
fn simulator_default_buffer_never_deadlocks_or_truncates() {
    for seed in 1..=40u64 {
        let mut rng = Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let max_output = 131_072;
        let specs = random_specs(&mut rng, 48, max_output);
        // Pools from barely one full lifetime up to plenty.
        let pool = (40_000 + max_output + 16).div_ceil(ROWS) * [1, 2, 4, 16][seed as usize % 4];
        let mut p = policy([4096, 16_384, 65_536][seed as usize % 3], 0, -1, max_output);
        let outcome = simulate(&mut p, pool, &specs, &mut rng);
        check_complete(&specs, &outcome);
        if seed % 4 == 0 { assert!(outcome.pressure_rounds > 0 || outcome.peak_concurrency > 1); }
    }
}

#[test]
fn simulator_chunks_raise_concurrency_over_whole_lifetime_reservation() {
    let mut rng = Rng(7);
    // Sixteen requests omit max_tokens (131072) but stop after a few thousand tokens.
    let specs: Vec<Spec> = (0..16).map(|_| Spec { arrival: 0, prompt: rng.range(500, 4000), max_output: 131_072,
        actual: rng.range(1000, 6000), step: 8 }).collect();
    let pool = 4 * (131_072 + 4000 + 8) / ROWS;
    let mut chunked = policy(16_384, 0, -1, 131_072);
    let outcome = simulate(&mut chunked, pool, &specs, &mut Rng(9));
    check_complete(&specs, &outcome);
    // Lifetime reservation would fit 4 at once; the drain rule admits until only one lifetime
    // of headroom is left (~ pool / chunk pages) — the hardware gate's C16 case.
    assert!(outcome.peak_concurrency >= 12, "peak {}", outcome.peak_concurrency);
}

#[test]
fn simulator_adversarial_everyone_runs_to_max_tokens_at_once() {
    // Every request runs to max_tokens, all arrive together, the pool holds ~2.5 lifetimes.
    for (chunk, step) in [(4096, 1), (16_384, 16), (1024, 8)] {
        let max_output = 60_000;
        let specs: Vec<Spec> = (0..24).map(|i| Spec { arrival: 0, prompt: 1000 + 997 * i, max_output,
            actual: usize::MAX, step }).collect();
        let pool = (5 * (24_000 + max_output + 16) / 2).div_ceil(ROWS);
        let mut p = policy(chunk, 0, -1, max_output);
        let outcome = simulate(&mut p, pool, &specs, &mut Rng(11));
        check_complete(&specs, &outcome);
        assert!(outcome.stats.stalls > 0 && outcome.pressure_rounds > 0, "the schedule is under pressure");
    }
    // Long runners admitted first, then a flood of short ones that must not starve them.
    let mut specs = vec![Spec { arrival: 0, prompt: 30_000, max_output: 200_000, actual: usize::MAX, step: 8 }; 3];
    specs.extend((0..40).map(|i| Spec { arrival: 1 + i, prompt: 2000, max_output: 200_000, actual: 3000, step: 8 }));
    let pool = (2 * (30_000 + 200_000 + 8usize)).div_ceil(ROWS);
    let mut p = policy(16_384, 0, -1, 200_000);
    check_complete(&specs, &simulate(&mut p, pool, &specs, &mut Rng(13)));
}

#[test]
fn simulator_drain_buffer_terminates_only_past_the_buffer_and_only_under_pressure() {
    let max_output = 100_000;
    let buffer = 8192;
    for seed in 1..=12u64 {
        let mut rng = Rng(seed * 31 + 5);
        let specs = random_specs(&mut rng, 32, max_output);
        // Small pool: room for only a few buffers' worth beyond the prompts.
        let pool = (40_000 + buffer + 16usize).div_ceil(ROWS) * 3;
        let mut p = policy(4096, buffer, -1, max_output);
        let outcome = simulate(&mut p, pool, &specs, &mut rng);
        assert_eq!(outcome.stats.deadlock_breaks, 0);
        for (s, finished) in specs.iter().zip(&outcome.finished) {
            let (finish, produced) = finished.unwrap();
            match finish {
                Finish::Early => assert!(produced >= buffer, "terminated at {produced} < buffer"),
                _ => assert_eq!(produced, s.actual.min(s.max_output)),
            }
        }
    }
    // No pressure, no termination.
    let mut rng = Rng(3);
    let specs = random_specs(&mut rng, 16, max_output);
    let mut p = policy(4096, buffer, -1, max_output);
    let outcome = simulate(&mut p, 16 * (40_000 + max_output + 16) / ROWS, &specs, &mut rng);
    check_complete(&specs, &outcome);
}

#[test]
fn simulator_early_termination_happens_only_past_the_threshold_where_a_stall_would_occur() {
    let max_output = 100_000;
    let threshold = 4096;
    for seed in 1..=12u64 {
        let mut rng = Rng(seed * 77 + 1);
        let specs = random_specs(&mut rng, 32, max_output);
        let pool = (40_000 + max_output + 16).div_ceil(ROWS) * 2;
        let mut p = policy(4096, 0, threshold as i64, max_output);
        let outcome = simulate(&mut p, pool, &specs, &mut rng);
        assert_eq!(outcome.stats.deadlock_breaks, 0);
        let mut early = 0;
        for (s, finished) in specs.iter().zip(&outcome.finished) {
            let (finish, produced) = finished.unwrap();
            if finish == Finish::Early {
                assert!(produced >= threshold);
                early += 1;
            } else {
                assert_eq!(produced, s.actual.min(s.max_output));
            }
        }
        assert!(early == 0 || outcome.pressure_rounds > 0);
    }
    let mut rng = Rng(5);
    let specs = random_specs(&mut rng, 16, max_output);
    let mut p = policy(4096, 0, threshold as i64, max_output);
    let outcome = simulate(&mut p, 16 * (40_000 + max_output + 16) / ROWS, &specs, &mut rng);
    check_complete(&specs, &outcome);
}

#[test]
fn shared_pages_return_only_with_their_last_holder() {
    let p = policy(4096, 0, -1, 1 << 20);
    let lease = p.lease(640, 640, 1, 1 << 20).unwrap();
    // Two forks of one 10-page snapshot, each needing 10 more pages; 10 free.
    let a: Vec<u32> = (0..10).collect();
    let b: Vec<u32> = (0..10).collect();
    let fork = |pages: &'static [u32]| Demand { lease, held: 10, pages, committed: 640, produced: 0, decoding: true };
    let (a, b): (&'static [u32], &'static [u32]) = (a.leak(), b.leak());
    // Shared pages free nothing when the first finishes: the second still finds 10.
    assert!(p.safe(10, [fork(a), fork(b)]));
    // With 5 free neither can finish, and the shared pages do not count as released.
    assert!(!p.safe(5, [fork(a), fork(b)]));
    // Counted as private they would (wrongly) look safe after the first finished.
    let private = |held| Demand { lease, held, pages: &[], committed: 640, produced: 0, decoding: true };
    assert!(p.safe(10, [private(10), private(10)]));
}
