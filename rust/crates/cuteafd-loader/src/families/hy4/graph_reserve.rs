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

// Short and 32K C1, K8 MC, FP8 KV, no drafter (hy4-gr-base, dev
// 43a4f41f1b, 2026-11-26). The first event-completed capture batch
// grew the arena by the entire W1; use that physical batch for both
// step and capture headroom rather than dividing by executable count.
const CENSUS: &[(Layout, usize, WorkingSet)] = &[
    (Layout::SingleSpark, 0, WorkingSet { w1: 32 << 20, step_growth: 32 << 20, capture_growth: 32 << 20 }),
    (Layout::DualContextSpark, 0, WorkingSet { w1: 40 << 20, step_growth: 40 << 20, capture_growth: 40 << 20 }),
    (Layout::DualContextSpark, 1, WorkingSet { w1: 40 << 20, step_growth: 40 << 20, capture_growth: 40 << 20 }),
];

pub fn reserve_policy(layout: Option<Layout>, rank: usize) -> Result<Resolved, Error> {
    graph_reserve::current_census(rank, layout, CENSUS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_c1_census_resolves_soft_and_hard_caps() {
        for (layout, rank, w1, soft, hard) in [
            (Layout::SingleSpark, 0, 32, 128, 192),
            (Layout::DualContextSpark, 0, 40, 136, 216),
            (Layout::DualContextSpark, 1, 40, 136, 216),
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
    }
}
