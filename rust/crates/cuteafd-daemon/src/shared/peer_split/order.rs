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
use std::collections::HashMap;
use std::fmt;

/// A flag both GPUs agree on: which exchange and which slot (or hop index).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Flag {
    pub exchange: &'static str,
    pub slot: usize,
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
}

impl Schedule {
    pub fn push(&mut self, gpu: usize, exchange: &'static str, slot: usize, label: impl Into<String>) {
        self.streams[gpu].push((Op::Push(Flag { exchange, slot }), label.into()));
    }

    pub fn wait(&mut self, gpu: usize, exchange: &'static str, slot: usize, label: impl Into<String>) {
        self.streams[gpu].push((Op::Wait(Flag { exchange, slot }), label.into()));
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

/// Runs `schedule` with each wait on GPU `g` consuming the next push on the
/// same flag from the other GPU (the device-side sequences), and each stream
/// in order. Ok when both streams drain.
pub(crate) fn check(schedule: &Schedule) -> Result<(), Deadlock> {
    let mut at = [0usize; 2];
    // Pushes published by GPU g per flag, and waits GPU g has consumed per flag.
    let mut pushed: [HashMap<Flag, usize>; 2] = Default::default();
    let mut waited: [HashMap<Flag, usize>; 2] = Default::default();
    loop {
        let mut progressed = false;
        for gpu in 0..2 {
            while let Some((op, _)) = schedule.streams[gpu].get(at[gpu]) {
                match *op {
                    Op::Push(flag) => *pushed[gpu].entry(flag).or_default() += 1,
                    Op::Wait(flag) => {
                        let have = pushed[1 - gpu].get(&flag).copied().unwrap_or(0);
                        let used = waited[gpu].entry(flag).or_default();
                        if *used >= have { break; }
                        *used += 1;
                    }
                }
                at[gpu] += 1;
                progressed = true;
            }
        }
        let done = (0..2).all(|gpu| at[gpu] == schedule.streams[gpu].len());
        if done { return Ok(()); }
        if !progressed {
            return Err(Deadlock { blocked: [0, 1].map(|gpu| schedule.streams[gpu].get(at[gpu]).map(|(_, l)| l.clone())) });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            s.push(0, MAIN, usize::MAX, format!("gpu0 streams push lane{lane}"));
        }
        for lane in 0..2 {
            s.wait(1, MAIN, usize::MAX, format!("gpu1 streams wait lane{lane}"));
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
}
