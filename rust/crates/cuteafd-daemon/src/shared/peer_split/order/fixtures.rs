//! CPU schedule builders for K3 context decode and K4 two-lane executors.
use super::Schedule;
use crate::shared::peer_split::DIRECT;

#[derive(Clone, Copy, Debug)]
pub(crate) enum PeerQueue {
    OneAhead,
    Lockstep,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum ContextFault {
    None,
    PartialWaitBeforePush { layer: usize },
    CandidatesWaitBeforePush { gpu: usize, layer: usize },
    CandidatesPushAfterPartialWait { gpu: usize, layer: usize },
}

fn exchange_slot(layer: usize, ffn: bool, lane: usize) -> usize {
    4 * lane + 2 * (layer % 2) + usize::from(ffn)
}

fn context_attention(
    s: &mut Schedule,
    gpu: usize,
    layer: usize,
    lane: usize,
    full_indexer: bool,
    fault: ContextFault,
) {
    let slot = 2 * lane + layer % 2;
    s.push(
        gpu,
        "context-q",
        slot,
        format!("gpu{gpu} q push L{layer} lane{lane}"),
    );
    let defer_candidates = full_indexer
        && matches!(fault,
        ContextFault::CandidatesPushAfterPartialWait { gpu: bad_gpu, layer: bad_layer }
            if gpu == bad_gpu && layer == bad_layer);
    if full_indexer {
        let wait_first = defer_candidates
            || matches!(fault,
            ContextFault::CandidatesWaitBeforePush { gpu: bad_gpu, layer: bad_layer }
                if gpu == bad_gpu && layer == bad_layer);
        if wait_first {
            s.wait(
                gpu,
                "context-candidates",
                slot,
                format!("gpu{gpu} candidates wait L{layer} lane{lane}"),
            );
        }
        if !defer_candidates {
            s.push(
                gpu,
                "context-candidates",
                slot,
                format!("gpu{gpu} candidates push L{layer} lane{lane}"),
            );
        }
        if !wait_first {
            s.wait(
                gpu,
                "context-candidates",
                slot,
                format!("gpu{gpu} candidates wait L{layer} lane{lane}"),
            );
        }
    }
    s.wait(
        gpu,
        "context-q",
        slot,
        format!("gpu{gpu} q wait L{layer} lane{lane}"),
    );
    let wait_first = matches!(fault,
        ContextFault::PartialWaitBeforePush { layer: bad_layer } if layer == bad_layer);
    if wait_first {
        s.wait(
            gpu,
            "context-partial",
            slot,
            format!("gpu{gpu} partial wait L{layer} lane{lane}"),
        );
    }
    // Compute the peer's heads, publish their partial, then compute our heads
    // before combining. Computation has no flag operations in this checker.
    s.push(
        gpu,
        "context-partial",
        slot,
        format!("gpu{gpu} partial push L{layer} lane{lane}"),
    );
    if !wait_first {
        s.wait(
            gpu,
            "context-partial",
            slot,
            format!("gpu{gpu} partial wait L{layer} lane{lane}"),
        );
    }
    if defer_candidates {
        // Unlike an immediate wait-then-push, stranding the candidate push
        // behind combine creates a cycle with the peer's candidate merge.
        s.push(
            gpu,
            "context-candidates",
            slot,
            format!("gpu{gpu} candidates push L{layer} lane{lane}"),
        );
    }
    let slot = exchange_slot(layer, false, lane);
    s.push(
        gpu,
        "main",
        slot,
        format!("gpu{gpu} attn push L{layer} lane{lane}"),
    );
    s.wait(
        gpu,
        "main",
        slot,
        format!("gpu{gpu} attn wait L{layer} lane{lane}"),
    );
}

/// GLM `decode_layers` / `peer_segment`, including DIRECT and the previous
/// FFN wait at each segment's start. A dense FFN push and a deferred MoE push
/// precede the same next-segment wait, so record exactly one push per layer.
pub(crate) fn context_decode(
    layers: usize,
    lane: usize,
    full_indexer: impl Fn(usize) -> bool,
    peer_queue: PeerQueue,
    fault: ContextFault,
) -> Schedule {
    let mut s = Schedule::default();
    let segment = |s: &mut Schedule, gpu: usize, layer: usize| {
        if layer == 0 {
            if gpu == 0 {
                s.push(gpu, "main", DIRECT, format!("gpu0 direct push lane{lane}"));
            } else {
                s.wait(gpu, "main", DIRECT, format!("gpu1 direct wait lane{lane}"));
            }
        } else {
            let before = layer - 1;
            s.wait(
                gpu,
                "main",
                exchange_slot(before, true, lane),
                format!("gpu{gpu} ffn wait L{before} lane{lane}"),
            );
        }
        context_attention(s, gpu, layer, lane, full_indexer(layer), fault);
        s.push(
            gpu,
            "main",
            exchange_slot(layer, true, lane),
            format!("gpu{gpu} ffn push L{layer} lane{lane}"),
        );
    };
    for layer in 0..layers {
        match peer_queue {
            PeerQueue::OneAhead => {
                segment(&mut s, 0, layer);
                // The real executor queues peer L0 after rank 0 L0, and then
                // peer L(i+1) immediately after rank 0 Li.
                if layer == 0 {
                    segment(&mut s, 1, 0);
                }
                if layer + 1 < layers {
                    segment(&mut s, 1, layer + 1);
                }
            }
            PeerQueue::Lockstep => {
                segment(&mut s, 1, layer);
                segment(&mut s, 0, layer);
            }
        }
    }
    // decode_layers has a final norm/head segment consuming the last FFN.
    if let Some(last) = layers.checked_sub(1) {
        s.wait(
            0,
            "main",
            exchange_slot(last, true, lane),
            format!("gpu0 ffn wait L{last} lane{lane}"),
        );
    }
    s
}

/// Lane A runs group t, lane B group t-1. Broadcasts precede both GPUs'
/// FFN halves; only the per-slot FFN lane order varies on GPU1.
pub(crate) fn layers_split(
    groups: usize,
    owner: impl Fn(usize) -> usize,
    swap_gpu1: bool,
) -> Schedule {
    let mut s = Schedule::default();
    for t in 0..=groups {
        let mut work = Vec::with_capacity(2);
        if t < groups {
            work.push((0, t));
        }
        if t > 0 {
            work.push((1, t - 1));
        }
        for &(lane, group) in &work {
            let gpu = owner(group);
            assert!(gpu < 2);
            let name = if lane == 0 { "A" } else { "B" };
            let slot = 2 * lane + group % 2;
            // Only the owner computes attention, then sends the residual.
            s.push(
                gpu,
                "bcast",
                slot,
                format!("gpu{gpu} bcast push lane {name} group {group}"),
            );
            s.wait(
                1 - gpu,
                "bcast",
                slot,
                format!("gpu{} bcast wait lane {name} group {group}", 1 - gpu),
            );
        }
        for gpu in 0..2 {
            let mut order = work.clone();
            if gpu == 1 && swap_gpu1 {
                order.reverse();
            }
            for (lane, group) in order {
                let name = if lane == 0 { "A" } else { "B" };
                let slot = 2 * lane + group % 2;
                s.push(
                    gpu,
                    "ffn",
                    slot,
                    format!("gpu{gpu} ffn push lane {name} group {group}"),
                );
                s.wait(
                    gpu,
                    "ffn",
                    slot,
                    format!("gpu{gpu} ffn wait lane {name} group {group}"),
                );
            }
        }
    }
    s
}

/// Owner FFN: each GPU queues its owned groups, lane A then lane B per
/// group, with hops only across ownership boundaries (prototype layers_owner).
pub(crate) fn layers_owner(
    groups: usize,
    lanes: usize,
    owner: impl Fn(usize) -> usize,
) -> Schedule {
    let mut s = Schedule::default();
    for group in 0..groups {
        let gpu = owner(group);
        assert!(gpu < 2);
        for lane in 0..lanes {
            if group > 0 && owner(group - 1) != gpu {
                s.wait(
                    gpu,
                    "hop",
                    2 * lane + group % 2,
                    format!("gpu{gpu} hop wait group {group} lane{lane}"),
                );
            }
            // Attention and the whole FFN run on the owner before sending.
            if group + 1 < groups && owner(group + 1) != gpu {
                s.push(
                    gpu,
                    "hop",
                    2 * lane + (group + 1) % 2,
                    format!("gpu{gpu} hop push group {} lane{lane}", group + 1),
                );
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::peer_split::order::{check, Flag, Op};

    fn glm_full(layer: usize) -> bool {
        layer < 3 || (layer >= 6 && (layer - 6) % 4 == 0)
    }

    fn v4_full(layer: usize) -> bool {
        layer >= 2 && layer % 2 == 0
    }

    fn assert_blocked(s: &Schedule, labels: [&str; 2]) {
        assert_eq!(
            check(s).unwrap_err().blocked,
            labels.map(|label| Some(label.into()))
        );
    }

    #[test]
    fn context_decode_glm53_one_ahead_and_lockstep_drain() {
        assert_eq!((0..78).filter(|&layer| glm_full(layer)).count(), 21);
        for queue in [PeerQueue::OneAhead, PeerQueue::Lockstep] {
            for lane in 0..2 {
                let s = context_decode(78, lane, glm_full, queue, ContextFault::None);
                assert_eq!(check(&s), Ok(()), "{queue:?} lane{lane}");
                // Recycled flags, not unique layer flags: cover all 21 indexers
                // and the terminal FFN consumption on rank 0.
                assert_eq!(s.streams[0].len(), 8 * 78 + 2 * 21 + 1);
                assert_eq!(s.streams[1].len(), 8 * 78 + 2 * 21);
            }
        }
    }

    #[test]
    fn context_decode_v4_61_layers_drain() {
        assert_eq!((0..61).filter(|&layer| v4_full(layer)).count(), 30);
        for queue in [PeerQueue::OneAhead, PeerQueue::Lockstep] {
            for lane in 0..2 {
                let s = context_decode(61, lane, v4_full, queue, ContextFault::None);
                assert_eq!(check(&s), Ok(()), "{queue:?} lane{lane}");
                assert_eq!(s.streams[0].len(), 8 * 61 + 2 * 30 + 1);
                assert_eq!(s.streams[1].len(), 8 * 61 + 2 * 30);
            }
        }
    }

    #[test]
    fn context_decode_records_segment_and_attention_order() {
        let s = context_decode(8, 1, glm_full, PeerQueue::OneAhead, ContextFault::None);
        for gpu in 0..2 {
            let expected = [
                (
                    Op::Push(Flag {
                        exchange: "context-q",
                        slot: 2,
                    }),
                    format!("gpu{gpu} q push L6 lane1"),
                ),
                (
                    Op::Push(Flag {
                        exchange: "context-candidates",
                        slot: 2,
                    }),
                    format!("gpu{gpu} candidates push L6 lane1"),
                ),
                (
                    Op::Wait(Flag {
                        exchange: "context-candidates",
                        slot: 2,
                    }),
                    format!("gpu{gpu} candidates wait L6 lane1"),
                ),
                (
                    Op::Wait(Flag {
                        exchange: "context-q",
                        slot: 2,
                    }),
                    format!("gpu{gpu} q wait L6 lane1"),
                ),
                (
                    Op::Push(Flag {
                        exchange: "context-partial",
                        slot: 2,
                    }),
                    format!("gpu{gpu} partial push L6 lane1"),
                ),
                (
                    Op::Wait(Flag {
                        exchange: "context-partial",
                        slot: 2,
                    }),
                    format!("gpu{gpu} partial wait L6 lane1"),
                ),
                (
                    Op::Push(Flag {
                        exchange: "main",
                        slot: 4,
                    }),
                    format!("gpu{gpu} attn push L6 lane1"),
                ),
                (
                    Op::Wait(Flag {
                        exchange: "main",
                        slot: 4,
                    }),
                    format!("gpu{gpu} attn wait L6 lane1"),
                ),
                (
                    Op::Push(Flag {
                        exchange: "main",
                        slot: 5,
                    }),
                    format!("gpu{gpu} ffn push L6 lane1"),
                ),
                (
                    Op::Wait(Flag {
                        exchange: "main",
                        slot: 5,
                    }),
                    format!("gpu{gpu} ffn wait L6 lane1"),
                ),
            ];
            assert!(s.streams[gpu]
                .windows(expected.len())
                .any(|ops| ops == expected));
            let direct = Flag {
                exchange: "main",
                slot: DIRECT,
            };
            assert_eq!(
                s.streams[gpu][0].0,
                if gpu == 0 {
                    Op::Push(direct)
                } else {
                    Op::Wait(direct)
                }
            );
        }
    }

    #[test]
    fn context_partial_wait_before_push_deadlocks() {
        for queue in [PeerQueue::OneAhead, PeerQueue::Lockstep] {
            for lane in 0..2 {
                let s = context_decode(
                    78,
                    lane,
                    glm_full,
                    queue,
                    ContextFault::PartialWaitBeforePush { layer: 7 },
                );
                assert_blocked(
                    &s,
                    [
                        &format!("gpu0 partial wait L7 lane{lane}"),
                        &format!("gpu1 partial wait L7 lane{lane}"),
                    ],
                );
            }
        }
    }

    #[test]
    fn context_candidates_immediate_wait_then_push_drains() {
        for queue in [PeerQueue::OneAhead, PeerQueue::Lockstep] {
            let s = context_decode(
                78,
                0,
                glm_full,
                queue,
                ContextFault::CandidatesWaitBeforePush { gpu: 0, layer: 6 },
            );
            assert_eq!(check(&s), Ok(()));
        }
    }

    #[test]
    fn context_candidates_push_after_partial_wait_deadlocks() {
        for queue in [PeerQueue::OneAhead, PeerQueue::Lockstep] {
            let s = context_decode(
                78,
                0,
                glm_full,
                queue,
                ContextFault::CandidatesPushAfterPartialWait { gpu: 0, layer: 6 },
            );
            // GPU0 can consume GPU1's list, but GPU1 needs GPU0's missing
            // list before publishing the partial that GPU0 now waits on.
            assert_blocked(
                &s,
                [
                    "gpu0 partial wait L6 lane0",
                    "gpu1 candidates wait L6 lane0",
                ],
            );
        }
    }

    #[test]
    fn two_lane_split_ffn_alternating_owners_drain() {
        assert_eq!(check(&layers_split(22, |group| group % 2, false)), Ok(()));
    }

    #[test]
    fn two_lane_split_ffn_swapped_lanes_deadlock() {
        assert_blocked(
            &layers_split(22, |group| group % 2, true),
            [
                "gpu0 ffn wait lane A group 1",
                "gpu1 ffn wait lane B group 0",
            ],
        );
    }

    #[test]
    fn two_lane_split_ffn_one_cutover_drain() {
        assert_eq!(
            check(&layers_split(22, |group| usize::from(group >= 11), false)),
            Ok(())
        );
    }

    #[test]
    fn two_lane_split_ffn_one_cutover_swapped_lanes_deadlock() {
        assert_blocked(
            &layers_split(22, |group| usize::from(group >= 11), true),
            [
                "gpu0 ffn wait lane A group 1",
                "gpu1 ffn wait lane B group 0",
            ],
        );
    }

    #[test]
    fn two_lane_owner_ffn_with_hops_drain() {
        for groups in [21, 22] {
            assert_eq!(check(&layers_owner(groups, 2, |group| group % 2)), Ok(()));
            assert_eq!(
                check(&layers_owner(groups, 2, |group| usize::from(group >= 11))),
                Ok(())
            );
        }
    }
}
