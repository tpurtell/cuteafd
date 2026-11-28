//! Rank-local decode graph admission for full Hy4, distinct from Hy4-flash.
use crate::placement::{
    graph_reserve::{self, Error, Resolved, WorkingSet},
    AttentionPlacement,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    SingleSpark,
    DualContextSpark,
}

impl Layout {
    pub fn resolve(gpus: usize, sparks: usize, attention: AttentionPlacement) -> Option<Self> {
        match (gpus, sparks, attention) {
            (1, 4, AttentionPlacement::Heads) => Some(Self::SingleSpark),
            (2, 6, AttentionPlacement::Context) => Some(Self::DualContextSpark),
            _ => None,
        }
    }
}

// K8 MC, FP8 KV, no drafter (hy4-gr-base and hy4-gr-ab-retry,
// 2026-11-26). Short/32K C1 alone measured 32/40/40 MiB. Use the
// full-card charged high-water as the working set instead; capture batches
// reached 34/42/40 MiB before ARC calibration (observed margin / 2).
const CENSUS: &[(Layout, usize, WorkingSet)] = &[
    (Layout::SingleSpark, 0, WorkingSet { w1: 160 << 20, step_growth: 34 << 20, capture_growth: 34 << 20 }),
    (Layout::DualContextSpark, 0, WorkingSet { w1: 164 << 20, step_growth: 42 << 20, capture_growth: 42 << 20 }),
    (Layout::DualContextSpark, 1, WorkingSet { w1: 158 << 20, step_growth: 40 << 20, capture_growth: 40 << 20 }),
];

/// The exact census entry, else the Hy4 × ranks envelope; never 0 with graphs on.
pub fn reserve_policy(layout: Option<Layout>, gpus: usize, rank: usize) -> Result<Resolved, Error> {
    graph_reserve::current_census(rank, layout, CENSUS, graph_reserve::GraphFamily::Hy4, gpus)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_card_census_resolves_soft_and_hard_caps() {
        for (layout, rank, w1, soft, hard) in [
            (Layout::SingleSpark, 0, 160, 256, 324),
            (Layout::DualContextSpark, 0, 164, 260, 344),
            (Layout::DualContextSpark, 1, 158, 254, 334),
        ] {
            let measured = graph_reserve::census_working_set(rank, Some(layout), CENSUS).unwrap();
            assert_eq!(measured.w1, w1 << 20);
            let policy = graph_reserve::resolve_working_set(None, None, None, rank, measured).unwrap();
            assert_eq!(policy.base_bytes, soft << 20);
            assert_eq!(policy.bytes, hard << 20);
        }
    }

    #[test]
    fn unsupported_shapes_do_not_borrow_a_census() {
        for (gpus, sparks, attention) in [
            (1, 0, AttentionPlacement::Heads),
            (2, 0, AttentionPlacement::Context),
            (2, 6, AttentionPlacement::Heads),
            (3, 6, AttentionPlacement::Context),
        ] {
            assert_eq!(Layout::resolve(gpus, sparks, attention), None);
        }
        assert_eq!(graph_reserve::census_working_set(0, None, CENSUS), None);
        assert_eq!(graph_reserve::census_working_set(1, Some(Layout::SingleSpark), CENSUS), None);
        assert_eq!(graph_reserve::census_working_set(2, Some(Layout::DualContextSpark), CENSUS), None);
        cuteafd_core::memory_policy::MemoryPolicy::default().scoped(|| {
            for (gpus, sparks, attention) in [(1, 0, AttentionPlacement::Heads), (2, 6, AttentionPlacement::Heads)] {
                for rank in 0..gpus {
                    assert_eq!(reserve_policy(Layout::resolve(gpus, sparks, attention), gpus, rank).unwrap().bytes,
                        graph_reserve::envelope_bytes(graph_reserve::GraphFamily::Hy4, gpus, rank));
                }
            }
        });
    }
}
