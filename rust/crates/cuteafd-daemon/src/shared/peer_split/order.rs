//! Cross-exchange ordering for two-GPU executors (placement P3).
//!
//! Each GPU runs one in-order stream. A wait on flag `f` blocks that stream
//! until the other GPU's matching push on `f` has run. Pushes never block.
//! Exchanges that share a stream (the attention/FFN exchange, the expert
//! exchange, hops) can therefore deadlock even when every wait has its push
//! queued: GPU0 waits on B, which GPU1 pushes only after it waits on A, which
//! GPU0 pushes only after its wait on B.
//!
//! [`check`] runs a recorded schedule (both streams' push/wait sequences, in
//! the order the host queued them) to completion on the CPU, or names the
//! two blocked waits. A per-layer executor (P7, S4c) records its schedule
//! for every lane interleaving it can produce, and asserts it in tests
//! before any hardware run.
//!
//! The rule that makes executors pass: a wait for a peer's result is queued
//! only after this GPU has queued every push the peer can reach before
//! producing that result. The hop primitive splits send from land so the
//! executor can place the land after those pushes (for V4's P1 cycle, after
//! the next lane's attention all-reduce push).
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

/// A flag both GPUs agree on: which exchange and which slot (or hop index).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Flag {
    pub exchange: &'static str,
    pub slot: usize,
    /// Destination buffer identity (lane workspace for DIRECT transfers).
    pub target: usize,
}

/// One queued operation on a GPU's stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Op {
    Push(Flag),
    Wait(Flag),
}

/// Both GPUs' streams in queue order, recorded with a label per op.
#[derive(Debug, Default, Clone)]
pub(crate) struct Schedule {
    pub streams: [Vec<(Op, String)>; 2],
    requests: Vec<[usize; 2]>,
}

impl Schedule {
    pub fn end_request(&mut self) {
        self.requests.push(self.streams.each_ref().map(Vec::len));
    }

    pub fn push(&mut self, gpu: usize, exchange: &'static str, slot: usize, label: impl Into<String>) {
        self.streams[gpu].push((Op::Push(Flag { exchange, slot, target: 0 }), label.into()));
    }

    pub fn wait(&mut self, gpu: usize, exchange: &'static str, slot: usize, label: impl Into<String>) {
        self.streams[gpu].push((Op::Wait(Flag { exchange, slot, target: 0 }), label.into()));
    }
    pub fn push_to(&mut self, gpu: usize, exchange: &'static str, slot: usize, target: usize, label: impl Into<String>) {
        self.streams[gpu].push((Op::Push(Flag { exchange, slot, target }), label.into()));
    }

    pub fn wait_for(&mut self, gpu: usize, exchange: &'static str, slot: usize, target: usize, label: impl Into<String>) {
        self.streams[gpu].push((Op::Wait(Flag { exchange, slot, target }), label.into()));
    }
}

/// The schedule's two streams blocked on each other, or a wait no push ever
/// matches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Deadlock {
    /// Per GPU: the label of the op it is blocked on (`None`: finished).
    pub blocked: [Option<String>; 2],
}

impl fmt::Display for Deadlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let side = |gpu: usize| self.blocked[gpu].as_deref().unwrap_or("done").to_string();
        write!(f, "cross-exchange deadlock: gpu0 blocked at {}, gpu1 blocked at {}", side(0), side(1))
    }
}

/// Native publication sequences belong to an exchange slot, not a destination.
/// DIRECT can publish several distinct lane buffers before any wait, but those
/// waits must consume destinations in publication order. Buffer reuse remains
/// forbidden until that destination's earlier publication has been consumed.
#[derive(Default)]
struct Publications {
    fifo: [HashMap<(&'static str, usize), VecDeque<usize>>; 2],
    outstanding: [HashSet<Flag>; 2],
}

impl Publications {
    /// False means a wait's next native sequence has not been published yet.
    fn advance(&mut self, gpu: usize, op: Op) -> Result<bool, String> {
        match op {
            Op::Push(flag) => {
                if !self.outstanding[gpu].insert(flag) {
                    return Err("unconsumed slot reuse".into());
                }
                self.fifo[gpu].entry((flag.exchange, flag.slot)).or_default().push_back(flag.target);
            }
            Op::Wait(flag) => {
                let fifo = self.fifo[1 - gpu].entry((flag.exchange, flag.slot)).or_default();
                let Some(&target) = fifo.front() else { return Ok(false) };
                if target != flag.target {
                    return Err(format!("crossed destination wait: expected target {target}, got {}", flag.target));
                }
                fifo.pop_front();
                self.outstanding[1 - gpu].remove(&flag);
            }
        }
        Ok(true)
    }
}

/// Runs `schedule` with each wait consuming the next per-slot sequence from
/// the other GPU, including its destination identity, and each stream in order.
/// Ok only when both streams drain and every publication is consumed.
pub(crate) fn check(schedule: &Schedule) -> Result<(), Deadlock> {
    let mut start = [0; 2];
    for &end in &schedule.requests {
        let request = Schedule { streams: [0, 1].map(|rank|
            schedule.streams[rank][start[rank]..end[rank]].to_vec()), requests: Vec::new() };
        check_terminal(&request, false)?;
        start = end;
    }
    let tail = Schedule { streams: [0, 1].map(|rank| schedule.streams[rank][start[rank]..].to_vec()),
        requests: Vec::new() };
    check_terminal(&tail, false)
}

/// Terminal abort releases pending waits; unfinished publications are discarded
/// only on this retired-engine path, never on a healthy request boundary.
pub(crate) fn check_terminal(schedule: &Schedule, aborted: bool) -> Result<(), Deadlock> {
    let mut at = [0usize; 2];
    let mut publications = Publications::default();
    loop {
        let mut progressed = false;
        for gpu in 0..2 {
            while let Some((op, _)) = schedule.streams[gpu].get(at[gpu]) {
                if !aborted {
                    match publications.advance(gpu, *op) {
                        Ok(true) => {}
                        Ok(false) => break,
                        Err(reason) => return Err(Deadlock { blocked: [0, 1].map(|rank| {
                            if rank == gpu { Some(format!("{reason}: {}", schedule.streams[gpu][at[gpu]].1)) }
                            else { schedule.streams[rank].get(at[rank]).map(|(_, label)| label.clone()) }
                        }) }),
                    }
                }
                at[gpu] += 1;
                progressed = true;
                break;
            }
        }
        let done = (0..2).all(|gpu| at[gpu] == schedule.streams[gpu].len());
        if done {
            if !aborted {
                for gpu in 0..2 {
                    if let Some(flag) = publications.outstanding[gpu].iter().next() {
                        return Err(Deadlock { blocked: [0, 1].map(|rank| {
                            (rank == gpu).then(|| format!("unconsumed publication: {flag:?}"))
                        }) });
                    }
                }
            }
            return Ok(());
        }
        if !progressed {
            return Err(Deadlock { blocked: [0, 1].map(|gpu| schedule.streams[gpu].get(at[gpu]).map(|(_, l)| l.clone())) });
        }
    }
}

/// Exercise the shared submission boundary at every recorded enqueue prefix.
/// The other stream may already be fully queued when its producer fails.
/// Healthy requests remain strictly balanced; only terminal abort discards
/// unfinished generations and releases waits for producers never enqueued.
#[cfg(test)]
pub(crate) fn check_submission_faults(schedule: &Schedule) -> Result<(), Deadlock> {
    use crate::shared::experts::rtx::pair::{submit, teardown, TerminalStepError, TestSubmission};
    use std::cell::{Cell, RefCell};
    check(schedule)?;
    for rank in 0..2 {
        for at in 0..=schedule.streams[rank].len() {
            let mut prefix = schedule.clone();
            prefix.requests.clear();
            prefix.streams[rank].truncate(at);
            for cancellation in [false, true] {
                let owners = RefCell::new(TestSubmission::new(Some(&prefix)));
                if cancellation {
                    let cancelled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _: anyhow::Result<()> = submit(
                            || owners.borrow().require_live(),
                            || teardown(&mut *owners.borrow_mut()),
                            || panic!("rank {rank} enqueue {at} dropped body"),
                        );
                    }));
                    assert!(cancelled.is_err());
                } else {
                    let result: anyhow::Result<()> = submit(
                        || owners.borrow().require_live(),
                        || teardown(&mut *owners.borrow_mut()),
                        || anyhow::bail!("rank {rank} enqueue {at}"),
                    );
                    assert!(result.unwrap_err().downcast_ref::<TerminalStepError>().is_some());
                }
                owners.borrow().verify_retired().unwrap();
                let replayed = Cell::new(false);
                let replay = submit(
                    || owners.borrow().require_live(),
                    || panic!("retired engine drained twice"),
                    || { replayed.set(true); Ok(()) },
                );
                assert!(replay.is_err() && !replayed.get());
            }
        }
    }
    Ok(())
}

/// GLM Flash layer-major TP2/Spark exchange fixture, including route broadcast.
#[cfg(test)]
pub(crate) fn glmf_tp2_fixture(layers: usize, local: std::ops::Range<usize>, lanes: usize, broadcast: bool,
    independent_first: bool) -> Schedule {
    let mut s = Schedule::default();
    let attn = |s: &mut Schedule, gpu, layer, lane| {
        s.push(gpu, "main", 4 * lane + 2 * (layer % 2), "attention push");
        s.wait(gpu, "main", 4 * lane + 2 * (layer % 2), "attention wait");
    };
    let front = |s: &mut Schedule, layer, lane| {
        attn(s, 1, layer, lane);
        if !local.contains(&layer) { s.push(1, "main", 4 * lane + 2 * (layer % 2) + 1, "shared push"); }
    };
    for lane in 0..lanes { s.push_to(0, "main", usize::MAX, lane, "stream push"); }
    for lane in 0..lanes {
        s.wait_for(1, "main", usize::MAX, lane, "stream wait");
        front(&mut s, 0, lane);
    }
    let units: Vec<_> = (0..layers).flat_map(|l| (0..lanes).map(move |k| (l, k))).collect();
    attn(&mut s, 0, 0, 0);
    for (i, &(layer, lane)) in units.iter().enumerate() {
        if local.contains(&layer) {
            if broadcast {
                s.push(0, "routes", 2 * lane + layer % 2, "canonical routes push");
                s.wait(1, "routes", 2 * lane + layer % 2, "canonical routes wait");
            }
            s.push(1, "main", 4 * lane + 2 * (layer % 2) + 1, "fused TP2 push");
        }
        if layer + 1 < layers {
            s.wait(1, "main", 4 * lane + 2 * (layer % 2) + 1, "FFN wait");
            front(&mut s, layer + 1, lane);
        }
        let next = units.get(i + 1).copied();
        let early = independent_first && next.is_some_and(|(_, k)| k != lane);
        if early { let (l, k) = next.unwrap(); attn(&mut s, 0, l, k); }
        if layer + 1 < layers { s.push(0, "main", 4 * lane + 2 * (layer % 2) + 1, "FFN push"); }
        s.wait(0, "main", 4 * lane + 2 * (layer % 2) + 1, "FFN wait");
        if !early { if let Some((l, k)) = next { attn(&mut s, 0, l, k); } }
    }
    // All lanes' FFN producers precede the terminal peer waits. Each payload
    // is rank 0's already-rounded mHC output, in the existing lane workspace.
    for lane in 0..lanes {
        s.push_to(0, "main", super::DIRECT, lane, "terminal residual push after mHC");
        s.wait_for(1, "main", super::DIRECT, lane, "terminal residual wait");
    }
    s
}


#[cfg(test)]
pub(crate) mod fixtures;
#[cfg(test)]
pub(crate) mod host;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_cannot_consume_a_previous_requests_publication() {
        let mut s = Schedule::default();
        s.push(0, "main", 0, "request one output");
        s.end_request();
        s.wait(1, "main", 0, "request two stale input");
        s.end_request();
        assert!(check(&s).is_err());
    }

    #[test]
    fn destination_identity_never_permits_same_buffer_overwrite() {
        let mut same = Schedule::default();
        same.push_to(0, "main", super::super::DIRECT, 7, "first payload");
        same.push_to(0, "main", super::super::DIRECT, 7, "overwrite");
        same.wait_for(1, "main", super::super::DIRECT, 7, "first received");
        same.wait_for(1, "main", super::super::DIRECT, 7, "second received");
        // Block the consumer so the second publication really overtakes it.
        same.streams[1].insert(0, (Op::Wait(Flag { exchange: "late", slot: 0, target: 0 }), "late consumer".into()));
        same.push(0, "late", 0, "consumer release");
        assert!(check(&same).is_err());
        let mut distinct = Schedule::default();
        for target in 0..4 { distinct.push_to(0, "main", super::super::DIRECT, target, "lane payload"); }
        for target in 0..4 { distinct.wait_for(1, "main", super::super::DIRECT, target, "lane received"); }
        assert_eq!(check(&distinct), Ok(()));
        distinct.streams[1].pop();
        assert!(check(&distinct).is_err());
    }

    #[test]
    fn direct_waits_cannot_cross_destination_sequences() {
        for producer in 0..2 {
            for crossed_after in 0..3 {
                let mut s = Schedule::default();
                for target in 0..4 {
                    s.push_to(producer, "main", super::super::DIRECT, target, "publish lane");
                }
                let mut targets: Vec<_> = (0..4).collect();
                targets.swap(crossed_after, crossed_after + 1);
                for target in targets {
                    s.wait_for(1 - producer, "main", super::super::DIRECT, target, "consume wrong lane");
                }
                let error = check(&s).unwrap_err();
                assert!(error.blocked[1 - producer].as_ref().unwrap().contains("crossed destination wait"));
            }
        }
    }

    #[test]
    fn destination_can_be_reused_after_its_fifo_wait() {
        let mut s = Schedule::default();
        for _ in 0..3 {
            for target in 0..2 {
                s.push_to(0, "main", super::super::DIRECT, target, "publish lane");
                s.wait_for(1, "main", super::super::DIRECT, target, "consume lane");
            }
        }
        assert_eq!(check_submission_faults(&s), Ok(()));
    }

    const MAIN: &str = "main";
    const EXPERTS: &str = "experts";

    /// V4's FFN exchange slot (engine.rs `slot`) and expert slot.
    fn ffn(layer: usize, lane: usize) -> usize { 4 * lane + 2 * (layer % 2) + 1 }
    fn attn(layer: usize, lane: usize) -> usize { 4 * lane + 2 * (layer % 2) }
    fn expert(layer: usize, lane: usize) -> usize { 2 * lane + layer % 2 }

    /// Rank 0's attention unit (front + all-reduce) and rank 1's front for the same unit.
    fn attention(s: &mut Schedule, gpu: usize, layer: usize, lane: usize) {
        s.push(gpu, MAIN, attn(layer, lane), format!("gpu{gpu} attn push L{layer} lane{lane}"));
        s.wait(gpu, MAIN, attn(layer, lane), format!("gpu{gpu} attn wait L{layer} lane{lane}"));
    }

    /// Rank 1's `peer_front` tail: its shared half out on the FFN slot.
    fn peer_front(s: &mut Schedule, layer: usize, lane: usize) {
        attention(s, 1, layer, lane);
        s.push(1, MAIN, ffn(layer, lane), format!("gpu1 shared push L{layer} lane{lane}"));
    }

    /// Rank 1's GPU1-range experts for a unit: packed routes in, result out.
    fn peer_experts(s: &mut Schedule, layer: usize, lane: usize) {
        s.wait(1, EXPERTS, expert(layer, lane), format!("gpu1 routes wait L{layer} lane{lane}"));
        s.push(1, EXPERTS, expert(layer, lane), format!("gpu1 result push L{layer} lane{lane}"));
    }

    /// Rank 1's `peer_post`.
    fn peer_post(s: &mut Schedule, layer: usize, lane: usize) {
        s.wait(1, MAIN, ffn(layer, lane), format!("gpu1 ffn wait L{layer} lane{lane}"));
    }

    /// Rank 0's `post_split` (the FFN all-reduce).
    fn post(s: &mut Schedule, layer: usize, lane: usize) {
        s.push(0, MAIN, ffn(layer, lane), format!("gpu0 ffn push L{layer} lane{lane}"));
        s.wait(0, MAIN, ffn(layer, lane), format!("gpu0 ffn wait L{layer} lane{lane}"));
    }

    /// Rank 0's routes out for a GPU1-range layer; the result wait is separate.
    fn routes(s: &mut Schedule, layer: usize, lane: usize) {
        s.push(0, EXPERTS, expert(layer, lane), format!("gpu0 routes push L{layer} lane{lane}"));
    }

    fn result(s: &mut Schedule, layer: usize, lane: usize) {
        s.wait(0, EXPERTS, expert(layer, lane), format!("gpu0 result wait L{layer} lane{lane}"));
    }

    /// V4's two-lane prefill as the host queues it (engine.rs `step`, units
    /// layer-major, rank 1 one unit ahead), with every layer `>= first_peer`
    /// on GPU1's expert range. `deferred` is the executor-owned order: rank
    /// 0's result wait moves out of `local_experts` into the unit's post, and
    /// the post follows the next unit's attention when that unit is on the
    /// other lane (as the Spark-pipelined path already orders its posts).
    fn v4_prefill(layers: usize, first_peer: usize, deferred: bool) -> Schedule {
        let mut s = Schedule::default();
        let units: Vec<(usize, usize)> = (0..layers).flat_map(|l| (0..2).map(move |n| (l, n))).collect();
        // Step start: residual streams to rank 1 (DIRECT), then rank 1's layer-0 fronts.
        for lane in 0..2 {
            s.push_to(0, MAIN, usize::MAX, lane, format!("gpu0 streams push lane{lane}"));
        }
        for lane in 0..2 {
            s.wait_for(1, MAIN, usize::MAX, lane, format!("gpu1 streams wait lane{lane}"));
            peer_front(&mut s, 0, lane);
        }
        attention(&mut s, 0, units[0].0, units[0].1);
        for (index, &(layer, lane)) in units.iter().enumerate() {
            let peer = layer >= first_peer;
            if peer { routes(&mut s, layer, lane); }
            if peer && !deferred { result(&mut s, layer, lane); }
            // peer_next: rank 1's experts, its post, its next front.
            if peer { peer_experts(&mut s, layer, lane); }
            peer_post(&mut s, layer, lane);
            if layer + 1 < layers { peer_front(&mut s, layer + 1, lane); }
            let next = units.get(index + 1).copied();
            // The next unit's input is this unit's post only on the same lane.
            let independent = deferred && next.is_some_and(|(_, next_lane)| next_lane != lane);
            if independent {
                let (nl, nn) = next.unwrap();
                attention(&mut s, 0, nl, nn);
            }
            if peer && deferred { result(&mut s, layer, lane); }
            post(&mut s, layer, lane);
            if !independent {
                if let Some((nl, nn)) = next { attention(&mut s, 0, nl, nn); }
            }
        }
        s
    }

    #[test]
    fn matched_exchanges_on_one_lane_drain() {
        let mut s = Schedule::default();
        s.push(0, MAIN, 0, "a0");
        s.wait(1, MAIN, 0, "b0");
        s.push(1, MAIN, 1, "b1");
        s.wait(0, MAIN, 1, "a1");
        assert_eq!(check(&s), Ok(()));
        // A wait no push matches is reported where it blocks.
        s.wait(0, MAIN, 2, "orphan");
        assert_eq!(check(&s), Err(Deadlock { blocked: [Some("orphan".into()), None] }));
    }

    #[test]
    fn head_split_prefill_without_gpu1_experts_drains() {
        // Today's default: every routed layer on GPU0 or the Sparks; only the
        // main exchange crosses (rank 1 waits on attention and FFN slots).
        assert_eq!(check(&v4_prefill(6, usize::MAX, false)), Ok(()));
    }

    #[test]
    fn gpu1_expert_wait_inside_the_unit_is_the_p1_two_lane_cycle() {
        // P1 bug 2 (Pro EXL3 max, RTX_EXPERT_PEER=on, 1,243-token prompt):
        // rank 0 waits for GPU1's expert result inside `local_experts` while
        // rank 1 waits in the next lane's attention for data rank 0 pushes
        // only after that wait.
        let deadlock = check(&v4_prefill(6, 2, false)).unwrap_err();
        assert_eq!(deadlock.blocked, [Some("gpu0 result wait L2 lane0".into()),
            Some("gpu1 attn wait L2 lane1".into())]);
        assert!(deadlock.to_string().contains("cross-exchange deadlock"));
    }

    #[test]
    fn executor_owned_order_lands_the_result_after_the_peers_input() {
        // The same layers with the result landed at post time, after the next
        // unit's attention: the schedule drains for any first GPU1 layer.
        for first in [0, 1, 2, 5] {
            assert_eq!(check(&v4_prefill(6, first, true)), Ok(()), "first GPU1 layer {first}");
        }
    }
    #[test]
    fn glmf_tp2_two_lane_local_post_requires_independent_attention_first() {
        for local in [0..6, 1..4, 4..6] {
            for broadcast in [false, true] {
                assert_eq!(check(&glmf_tp2_fixture(6, local.clone(), 2, broadcast, true)), Ok(()));
            }
        }
        assert!(check(&glmf_tp2_fixture(6, 0..6, 2, true, false)).is_err());
    }

    #[test]
    fn glmf_terminal_replication_drains_across_lanes_and_request_boundaries() {
        // Four lanes remain legal for Spark-only; mixed TP2 admits at most two.
        for lanes in 1..=4 {
            let ranges = if lanes <= 2 { vec![0..0, 0..5, 1..4, 4..5] } else { vec![0..0] };
            for local in ranges {
                let request = glmf_tp2_fixture(5, local, lanes, true, true);
                assert_eq!(check_submission_faults(&request), Ok(()));
                let mut repeated = Schedule::default();
                for _ in 0..3 {
                    for (stream, next) in repeated.streams.iter_mut().zip(&request.streams) {
                        stream.extend(next.iter().cloned());
                    }
                    repeated.end_request();
                }
                assert_eq!(check(&repeated), Ok(()));
                for rank in 0..2 {
                    let mut missing = request.clone();
                    missing.streams[rank].pop();
                    assert!(check(&missing).is_err(), "missing terminal rank {rank} operation");
                }
            }
        }
    }

    fn glmf_remote_draft_round(s: &mut Schedule, reused: bool, input_ack_before_head: bool) {
        if reused { s.wait(0, "draft-taps-ack", 0, "previous tap projection consumed"); }
        s.push(0, "draft-taps", 0, "target tap pack final");
        s.wait(1, "draft-taps", 0, "remote tap pack arrival");
        s.push(1, "draft-taps-ack", 0, "remote tap projection consumed");
        if reused { s.wait(0, "draft-input-ack", 0, "previous draft input consumed"); }
        s.push(0, "draft-input", 0, "target embedding gather final");
        s.wait(1, "draft-input", 0, "remote embedding rows arrival");
        if reused { s.wait(1, "draft-hidden-ack", 0, "previous target head consumed"); }
        s.push(1, "draft-hidden", 0, "remote full-R normalized rows final");
        // Waiting for the entire draft to finish before launching its target
        // head is a cycle: the draft itself waits for the head's logits.
        if input_ack_before_head { s.wait(0, "draft-input-ack", 0, "premature input consumed wait"); }
        s.wait(0, "draft-hidden", 0, "target normalized rows arrival");
        s.push(0, "draft-hidden-ack", 0, "target full-R head consumed");
        if reused { s.wait(0, "draft-logits-ack", 0, "previous draft logits consumed"); }
        s.push(0, "draft-logits", 0, "target full-R logits final");
        s.wait(1, "draft-logits", 0, "remote logits arrival");
        s.push(1, "draft-logits-ack", 0, "remote sampler consumed");
        s.push(1, "draft-input-ack", 0, "remote draft input consumed");
    }

    #[test]
    fn glmf_tp2_then_remote_draft_ack_schedule_drains_reused_slots() {
        // Design fixture only: the remote executor must record this ordering
        // from its real enqueue paths before enabling GPU1 placement.
        for lanes in [1, 2] {
            for broadcast in [false, true] {
                let mut schedule = Schedule::default();
                for round in 0..3 {
                    let target = glmf_tp2_fixture(6, 1..4, lanes, broadcast, true);
                    for (stream, next) in schedule.streams.iter_mut().zip(target.streams) { stream.extend(next); }
                    glmf_remote_draft_round(&mut schedule, round > 0, false);
                }
                schedule.wait(0, "draft-taps-ack", 0, "request final context acknowledgement");
                schedule.wait(0, "draft-input-ack", 0, "request final input acknowledgement");
                schedule.wait(1, "draft-hidden-ack", 0, "request final hidden acknowledgement");
                schedule.wait(0, "draft-logits-ack", 0, "request final logits acknowledgement");
                schedule.end_request();
                assert_eq!(check(&schedule), Ok(()));
            }
        }
        let mut schedule = glmf_tp2_fixture(6, 1..4, 2, true, true);
        glmf_remote_draft_round(&mut schedule, false, true);
        assert_eq!(check(&schedule).unwrap_err().blocked,
            [Some("premature input consumed wait".into()), Some("remote logits arrival".into())]);
    }

    #[test]
    fn glmf_tp2_three_lane_front_is_not_an_admitted_schedule() {
        assert!(check(&glmf_tp2_fixture(5, 1..4, 3, false, true)).is_err());
    }

}
