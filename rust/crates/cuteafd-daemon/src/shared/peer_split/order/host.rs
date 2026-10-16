//! Host ordering for two-GPU executors. [`super::check`] proves the device
//! flags of both streams drain once everything is queued. An executor that
//! also blocks the host while it queues (GLM 5.3: a stream synchronize before
//! each Spark expert wave, `moe_stage`) can still hang: while the host waits on
//! GPU `g`, nothing more is queued anywhere, so `g` must be able to finish
//! everything queued on it from the pushes already queued on the other GPU.
//!
//! [`check_host`] replays the host's issue order: each queued op joins its
//! GPU's stream; at a [`HostOp::Sync`] both streams run as far as they can with
//! what is queued, and the sync must find its GPU's stream empty.
use super::{check, Deadlock, Flag, Op, Publications, Schedule};

/// One step of the host's issue order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HostOp {
    /// Queue a push or wait on `gpu`'s stream.
    Queue { gpu: usize, op: Op, label: String },
    /// The host blocks until `gpu`'s stream has run everything queued on it.
    Sync { gpu: usize, label: String },
}

/// The host's issue order of a two-GPU step.
#[derive(Debug, Default, Clone)]
pub(crate) struct HostSchedule {
    pub ops: Vec<HostOp>,
}

impl HostSchedule {
    pub fn push(&mut self, gpu: usize, exchange: &'static str, slot: usize, label: impl Into<String>) {
        self.ops.push(HostOp::Queue { gpu, op: Op::Push(Flag { exchange, slot, target: 0 }), label: label.into() });
    }

    pub fn wait(&mut self, gpu: usize, exchange: &'static str, slot: usize, label: impl Into<String>) {
        self.ops.push(HostOp::Queue { gpu, op: Op::Wait(Flag { exchange, slot, target: 0 }), label: label.into() });
    }

    pub fn push_to(&mut self, gpu: usize, exchange: &'static str, slot: usize, target: usize, label: impl Into<String>) {
        self.ops.push(HostOp::Queue { gpu, op: Op::Push(Flag { exchange, slot, target }), label: label.into() });
    }

    pub fn wait_for(&mut self, gpu: usize, exchange: &'static str, slot: usize, target: usize, label: impl Into<String>) {
        self.ops.push(HostOp::Queue { gpu, op: Op::Wait(Flag { exchange, slot, target }), label: label.into() });
    }

    pub fn sync(&mut self, gpu: usize, label: impl Into<String>) {
        self.ops.push(HostOp::Sync { gpu, label: label.into() });
    }

    /// The device streams alone (host syncs dropped), for [`super::check`].
    pub fn streams(&self) -> Schedule {
        let mut out = Schedule::default();
        for op in &self.ops {
            if let HostOp::Queue { gpu, op, label } = op {
                out.streams[*gpu].push((*op, label.clone()));
            }
        }
        out
    }
}

/// The host blocked at `sync` for good: the streams stopped at `blocked`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HostDeadlock {
    pub sync: String,
    pub blocked: [Option<String>; 2],
}

impl std::fmt::Display for HostDeadlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let side = |gpu: usize| self.blocked[gpu].as_deref().unwrap_or("done").to_string();
        write!(f, "host blocked at {} for good: gpu0 at {}, gpu1 at {}", self.sync, side(0), side(1))
    }
}

/// Device streams as queued so far, run as far as their flags allow.
#[derive(Default)]
struct Streams {
    queued: [Vec<(Op, String)>; 2],
    at: [usize; 2],
    publications: Publications,
}

impl Streams {
    fn run(&mut self) -> Result<(), String> {
        loop {
            let mut progressed = false;
            for gpu in 0..2 {
                while let Some((op, _)) = self.queued[gpu].get(self.at[gpu]) {
                    if !self.publications.advance(gpu, *op)? { break; }
                    self.at[gpu] += 1;
                    progressed = true;
                    break;
                }
            }
            if !progressed { return Ok(()); }
        }
    }

    fn blocked(&self) -> [Option<String>; 2] {
        [0, 1].map(|gpu| self.queued[gpu].get(self.at[gpu]).map(|(_, label)| label.clone()))
    }
}

/// Ok when every host sync completes and both streams then drain.
pub(crate) fn check_host(schedule: &HostSchedule) -> Result<(), HostDeadlock> {
    let mut streams = Streams::default();
    for op in &schedule.ops {
        match op {
            HostOp::Queue { gpu, op, label } => streams.queued[*gpu].push((*op, label.clone())),
            HostOp::Sync { gpu, label } => {
                streams.run().map_err(|reason| HostDeadlock {
                    sync: format!("{label}: {reason}"), blocked: streams.blocked(),
                })?;
                if streams.at[*gpu] < streams.queued[*gpu].len() {
                    return Err(HostDeadlock { sync: label.clone(), blocked: streams.blocked() });
                }
            }
        }
    }
    check(&schedule.streams()).map_err(|Deadlock { blocked }| HostDeadlock { sync: "end of step".into(), blocked })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_sync_rejects_crossed_direct_destinations() {
        let mut s = HostSchedule::default();
        s.push_to(0, "main", super::super::super::DIRECT, 0, "lane zero");
        s.push_to(0, "main", super::super::super::DIRECT, 1, "lane one");
        s.wait_for(1, "main", super::super::super::DIRECT, 1, "crossed lane one");
        s.sync(1, "consume lane one");
        s.wait_for(1, "main", super::super::super::DIRECT, 0, "crossed lane zero");
        assert!(check_host(&s).unwrap_err().sync.contains("crossed destination wait"));
    }

    #[test]
    fn a_sync_waits_for_progress_the_queued_pushes_allow() {
        let mut s = HostSchedule::default();
        s.wait(0, "x", 0, "gpu0 waits x");
        s.push(1, "x", 0, "gpu1 pushes x");
        s.sync(0, "sync gpu0");
        assert_eq!(check_host(&s), Ok(()));
        // The push is queued only after the host blocked on GPU0: it never comes.
        let mut s = HostSchedule::default();
        s.wait(0, "x", 0, "gpu0 waits x");
        s.sync(0, "sync gpu0");
        s.push(1, "x", 0, "gpu1 pushes x");
        assert_eq!(check_host(&s), Err(HostDeadlock { sync: "sync gpu0".into(),
            blocked: [Some("gpu0 waits x".into()), None] }));
        // Device flags alone would drain it.
        assert_eq!(check(&s.streams()), Ok(()));
    }

    #[test]
    fn a_sync_on_one_gpu_lets_the_other_lag() {
        let mut s = HostSchedule::default();
        s.wait(1, "y", 0, "gpu1 waits y");
        s.push(0, "x", 0, "gpu0 pushes x");
        s.sync(0, "sync gpu0");
        s.push(0, "y", 0, "gpu0 pushes y");
        s.wait(1, "x", 0, "gpu1 consumes x");
        assert_eq!(check_host(&s), Ok(()));
        let mut stuck = s.clone();
        stuck.wait(0, "z", 0, "gpu0 waits z");
        assert_eq!(check_host(&stuck), Err(HostDeadlock { sync: "end of step".into(),
            blocked: [Some("gpu0 waits z".into()), None] }));
    }
}

/// GLM 5.3 host issue orders (`glm5::engine`), heads or context, for the host checker.
#[cfg(test)]
pub(crate) mod glm {
    use super::HostSchedule;
    use crate::shared::peer_split::DIRECT;

    pub const LAYERS: usize = 78;
    pub const FIRST_MOE: usize = 3;

    pub fn full_indexer(layer: usize) -> bool {
        layer < 3 || (layer >= 6 && (layer - 6) % 4 == 0)
    }

    /// `engine::slot`: attention (`ffn` false) or FFN exchange slot of a layer and lane.
    fn slot(layer: usize, ffn: bool, lane: usize) -> usize {
        4 * lane + 2 * (layer % 2) + usize::from(ffn)
    }

    /// Executor orders the host checker must refuse.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Fault {
        None,
        /// Prefill: rank 1's next attention is queued only after a lane's wave lands
        /// (`peer_next` after `land`): the device flags drain, and so does the host.
        PeerAfterLand,
        /// Rank 1's work is queued only after rank 0's whole step: the host syncs GPU0 at
        /// the first MoE stage while rank 0 waits for rank 1's layer-0 pushes.
        PeerAfterStep,
    }

    /// One layer's attention on `gpu` (`attention_unit` + the attention all-reduce).
    fn attention(s: &mut HostSchedule, gpu: usize, layer: usize, lane: usize, context: bool) {
        if context {
            let cs = 2 * lane + layer % 2;
            s.push(gpu, "context-q", cs, format!("gpu{gpu} q push L{layer}"));
            if full_indexer(layer) {
                s.push(gpu, "context-candidates", cs, format!("gpu{gpu} candidates push L{layer}"));
                s.wait(gpu, "context-candidates", cs, format!("gpu{gpu} candidates wait L{layer}"));
            }
            s.wait(gpu, "context-q", cs, format!("gpu{gpu} q wait L{layer}"));
            s.push(gpu, "context-partial", cs, format!("gpu{gpu} partial push L{layer}"));
            s.wait(gpu, "context-partial", cs, format!("gpu{gpu} partial wait L{layer}"));
        }
        s.push(gpu, "main", slot(layer, false, lane), format!("gpu{gpu} attn push L{layer} lane{lane}"));
        s.wait(gpu, "main", slot(layer, false, lane), format!("gpu{gpu} attn wait L{layer} lane{lane}"));
    }

    /// Rank 1's `peer_segment(index)` (decode): previous FFN in, attention, FFN half out.
    fn peer_segment(s: &mut HostSchedule, index: usize, context: bool) {
        if index == 0 {
            s.wait(1, "main", DIRECT, "gpu1 direct wait");
        } else {
            s.wait(1, "main", slot(index - 1, true, 0), format!("gpu1 ffn wait L{}", index - 1));
        }
        attention(s, 1, index, 0, context);
        s.push(1, "main", slot(index, true, 0), format!("gpu1 ffn push L{index}"));
    }

    /// `decode_layers` of one decode/verify step, as the host issues it: rank 0's
    /// segment, rank 1's next segment, then (MoE) `moe_stage`'s GPU0 sync and the
    /// Spark wave; the last segment is the final norm and head.
    pub fn decode(context: bool, fault: Fault) -> HostSchedule {
        let mut s = HostSchedule::default();
        // `step_tables`: both streams drained before the tables are written.
        s.sync(0, "tables gpu0");
        s.sync(1, "tables gpu1");
        let mut previous_moe = false;
        for index in 0..=LAYERS {
            // Rank 0's segment `index`.
            if index == 0 {
                s.push(0, "main", DIRECT, "gpu0 direct push");
            } else {
                let before = index - 1;
                if previous_moe && index < LAYERS {
                    s.push(0, "main", slot(before, true, 0), format!("gpu0 ffn push L{before}"));
                }
                s.wait(0, "main", slot(before, true, 0), format!("gpu0 ffn wait L{before}"));
            }
            if index == LAYERS {
                break;
            }
            attention(&mut s, 0, index, 0, context);
            let dense = index < FIRST_MOE;
            if dense {
                s.push(0, "main", slot(index, true, 0), format!("gpu0 ffn push L{index}"));
            }
            if fault != Fault::PeerAfterStep {
                if index == 0 {
                    peer_segment(&mut s, 0, context);
                }
                if index + 1 < LAYERS {
                    peer_segment(&mut s, index + 1, context);
                }
            }
            if !dense {
                s.sync(0, format!("moe_stage gpu0 L{index}"));
            }
            previous_moe = !dense;
        }
        if fault == Fault::PeerAfterStep {
            for index in 0..LAYERS {
                peer_segment(&mut s, index, context);
            }
        }
        s
    }

    /// `step_lanes`: rank 1's layer-0 attention for every lane, then units layer-major;
    /// each MoE unit stages (GPU0 sync) and sends before rank 1's `peer_next`, and its
    /// wave lands (a host wait on the Sparks, no GPU flag) before the lane's next attention.
    pub fn prefill(lanes: usize, fault: Fault) -> HostSchedule {
        let mut s = HostSchedule::default();
        s.sync(0, "tables gpu0");
        s.sync(1, "tables gpu1");
        for lane in 0..lanes {
            s.push_to(0, "main", DIRECT, lane, format!("gpu0 direct push lane{lane}"));
        }
        let peer_attention = |s: &mut HostSchedule, index: usize, lane: usize| {
            if index == 0 {
                s.wait_for(1, "main", DIRECT, lane, format!("gpu1 direct wait lane{lane}"));
            }
            attention(s, 1, index, lane, false);
            s.push(1, "main", slot(index, true, lane), format!("gpu1 ffn push L{index} lane{lane}"));
        };
        let peer_next = |s: &mut HostSchedule, index: usize, lane: usize| {
            if index + 1 < LAYERS {
                s.wait(1, "main", slot(index, true, lane), format!("gpu1 ffn wait L{index} lane{lane}"));
                peer_attention(s, index + 1, lane);
            }
        };
        if fault != Fault::PeerAfterStep {
            for lane in 0..lanes {
                peer_attention(&mut s, 0, lane);
            }
        }
        let post = |s: &mut HostSchedule, index: usize, lane: usize, moe: bool| {
            if moe && index + 1 < LAYERS {
                s.push(0, "main", slot(index, true, lane), format!("gpu0 ffn push L{index} lane{lane}"));
            }
            s.wait(0, "main", slot(index, true, lane), format!("gpu0 ffn wait L{index} lane{lane}"));
        };
        let units: Vec<(usize, usize)> = (0..LAYERS).flat_map(|l| (0..lanes).map(move |k| (l, k))).collect();
        let mut inflight: std::collections::VecDeque<(usize, usize)> = Default::default();
        let attention0 = |s: &mut HostSchedule, (index, lane): (usize, usize)| {
            attention(s, 0, index, lane, false);
            if index < FIRST_MOE {
                s.push(0, "main", slot(index, true, lane), format!("gpu0 ffn push L{index} lane{lane}"));
            }
        };
        attention0(&mut s, units[0]);
        let late = fault == Fault::PeerAfterLand;
        let deferred = fault == Fault::PeerAfterStep;
        for (position, &(index, lane)) in units.iter().enumerate() {
            if index < FIRST_MOE {
                if !deferred {
                    peer_next(&mut s, index, lane);
                }
                post(&mut s, index, lane, false);
            } else {
                s.sync(0, format!("moe_stage gpu0 L{index} lane{lane}"));
                if !late && !deferred {
                    peer_next(&mut s, index, lane);
                }
                inflight.push_back((index, lane));
            }
            match units.get(position + 1).copied() {
                Some(next) => {
                    while inflight.iter().any(|u| u.1 == next.1) {
                        let (i, k) = inflight.pop_front().unwrap();
                        post(&mut s, i, k, true);
                        if late {
                            peer_next(&mut s, i, k);
                        }
                    }
                    attention0(&mut s, next);
                }
                None => {
                    while let Some((i, k)) = inflight.pop_front() {
                        post(&mut s, i, k, true);
                        if late {
                            peer_next(&mut s, i, k);
                        }
                    }
                }
            }
        }
        if deferred {
            for &(index, lane) in &units {
                if index == 0 {
                    peer_attention(&mut s, 0, lane);
                }
                peer_next(&mut s, index, lane);
            }
        }
        s
    }
}

#[cfg(test)]
mod glm_tests {
    use super::glm::{decode, prefill, Fault};
    use super::*;

    #[test]
    fn glm_decode_heads_and_context_drain_with_host_syncs() {
        for context in [false, true] {
            assert_eq!(check_host(&decode(context, Fault::None)), Ok(()), "context {context}");
        }
    }

    #[test]
    fn glm_prefill_lanes_drain_with_host_syncs() {
        for lanes in 1..=4 {
            assert_eq!(check_host(&prefill(lanes, Fault::None)), Ok(()), "{lanes} lanes");
        }
    }

    #[test]
    fn glm_decode_peer_queued_after_the_step_hangs_the_host_at_the_first_moe_stage() {
        // Rank 0's layer-0 attention all-reduce (and, under context, its q wait) needs rank
        // 1's segment 0: queued after the step, the device flags drain, but the host never
        // returns from the first MoE layer's stage sync.
        for context in [false, true] {
            let schedule = decode(context, Fault::PeerAfterStep);
            assert_eq!(check(&schedule.streams()), Ok(()), "device flags alone drain");
            let error = check_host(&schedule).unwrap_err();
            assert_eq!(error.sync, "moe_stage gpu0 L3", "context {context}: {error}");
            assert_eq!(error.blocked[1], None);
            // Layer 0 has a full indexer: under context rank 0 stops at its candidate wait.
            let first = if context { "gpu0 candidates wait L0" } else { "gpu0 attn wait L0 lane0" };
            assert_eq!(error.blocked[0].as_deref(), Some(first), "context {context}");
        }
    }

    #[test]
    fn glm_prefill_peer_after_landing_drains_but_after_the_step_hangs() {
        for lanes in [1, 3] {
            // Late but before the lane's next attention: still safe.
            assert_eq!(check_host(&prefill(lanes, Fault::PeerAfterLand)), Ok(()), "{lanes} lanes");
            let schedule = prefill(lanes, Fault::PeerAfterStep);
            let error = check_host(&schedule).unwrap_err();
            assert_eq!(error.sync, "moe_stage gpu0 L3 lane0", "{lanes} lanes: {error}");
            assert_eq!(error.blocked[0].as_deref(), Some("gpu0 attn wait L0 lane0"), "{error}");
        }
    }
}
