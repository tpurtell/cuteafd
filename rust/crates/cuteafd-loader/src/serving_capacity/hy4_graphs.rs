//! Lazy rank-local decode graphs and per-rank C1 working-set admission.

use crate::placement::graph_reserve::{self, Error, Resolved, WorkingSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hy4GraphLayout {
    Exl3Min,
    Exl3Max,
    Exl3MinSpark,
    Exl3MaxSpark,
}

impl Hy4GraphLayout {
    pub fn resolve(ranks: usize, sparks: usize) -> Option<Self> {
        match (ranks, sparks) {
            (1, 0) => Some(Self::Exl3Min),
            (2, 0) => Some(Self::Exl3Max),
            (1, s) if s > 0 => Some(Self::Exl3MinSpark),
            (2, s) if s > 0 => Some(Self::Exl3MaxSpark),
            _ => None,
        }
    }

}

/// Short and 32K C1 physical graph-arena censuses on the reference cards.
/// All measured cards use DFly2. The single-GPU Spark C1 census is the maximum
/// across 1/4 Sparks with zero and auto expert onboarding (K8 compact):
/// zero-onboard W1 130 MiB on both, auto W1 90/78 MiB; target verify
/// width growth <= 24 MiB, draft growth <= 2 MiB, single capture <= 4 MiB.
/// The TP2 Spark C1 census (116/108 MiB) and wider census (1130/1058 MiB)
/// are intentionally not defaults: both graph reserves slowed matched decode
/// versus eager. TP1 retains its C1-derived cap; the wider 1340 MiB working
/// set also regressed throughput despite eliminating donor churn.
const CENSUS: &[(Hy4GraphLayout, usize, WorkingSet)] = &[
    (Hy4GraphLayout::Exl3Min, 0, WorkingSet { w1: 88 << 20, step_growth: 18 << 20, capture_growth: 4 << 20 }),
    (Hy4GraphLayout::Exl3Max, 0, WorkingSet { w1: 100 << 20, step_growth: 20 << 20, capture_growth: 4 << 20 }),
    (Hy4GraphLayout::Exl3Max, 1, WorkingSet { w1: 98 << 20, step_growth: 18 << 20, capture_growth: 4 << 20 }),
    (Hy4GraphLayout::Exl3MinSpark, 0, WorkingSet { w1: 130 << 20, step_growth: 26 << 20, capture_growth: 4 << 20 }),
];

#[cfg(test)]
fn working_set(layout: Option<Hy4GraphLayout>, rank: usize) -> Option<WorkingSet> {
    graph_reserve::census_working_set(rank, layout, CENSUS)
}

/// Use identical inputs for admission, provenance, and runtime. Layouts
/// without a qualified census default to eager; explicit reserves still work.
pub fn reserve_policy(layout: Option<Hy4GraphLayout>, rank: usize) -> Result<Resolved, Error> {
    graph_reserve::current_census(rank, layout, CENSUS)
}

/// Short and 32K C1 physical graph-arena census for split DFly2 on `Exl3Max`
/// (2x RTX, 0 Sparks, shipping `CORE_VIEW=original`, default `TP2_DFLASH=on`):
/// measured `100 MiB` on GPU0 and `98 MiB` on GPU1 (zero captures/misses/eager
/// runs on both timed C1 bands).
// The split-census rows are identical to the current base table's local TP2
// rows; this predicate preserves the split-only query used by the tests.
#[cfg(test)]
fn split_dfly2_working_set(layout: Option<Hy4GraphLayout>, rank: usize) -> Option<WorkingSet> {
    working_set(layout.filter(|key| *key == Hy4GraphLayout::Exl3Max), rank)
}

/// Split DFly2 and the ordinary policy currently have the same measured
/// local TP2 rows. Preserve all layout fallbacks through the common table.
pub fn reserve_policy_for(
    layout: Option<Hy4GraphLayout>,
    rank: usize,
    _tp2_dfly2: bool,
) -> Result<Resolved, Error> {
    reserve_policy(layout, rank)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c1_census_sets_rank_local_soft_and_hard_caps() {
        use Hy4GraphLayout::*;
        for (layout, rank, w1, step, capture, soft, hard) in [
            (Exl3Min, 0, 88, 18, 4, 184, 192),
            (Exl3Max, 0, 100, 20, 4, 196, 204),
            (Exl3Max, 1, 98, 18, 4, 194, 202),
            (Exl3MinSpark, 0, 130, 26, 4, 226, 234),
        ] {
            let census = working_set(Some(layout), rank).unwrap();
            assert_eq!(census.w1, w1 << 20);
            assert_eq!(census.step_growth, step << 20);
            assert_eq!(census.capture_growth, capture << 20);
            let policy = graph_reserve::resolve_working_set(None, None, None, rank, census).unwrap();
            assert_eq!(policy.base_bytes, soft << 20);
            assert_eq!(policy.bytes, hard << 20);
            assert_eq!(policy.growth_headroom, Some((2 * capture) << 20));
        }
    }

    #[test]
    fn unmeasured_ranks_do_not_borrow_graph_working_sets() {
        assert_eq!(working_set(Hy4GraphLayout::resolve(1, 0), 1), None);
        assert_eq!(working_set(Hy4GraphLayout::resolve(2, 0), 2), None);
        assert_eq!(working_set(Hy4GraphLayout::resolve(2, 6), 2), None);
        for sparks in [1, 4, 6] {
            for rank in [0, 1] {
                assert_eq!(working_set(Hy4GraphLayout::resolve(2, sparks), rank), None);
            }
        }
        assert_eq!(working_set(Hy4GraphLayout::resolve(3, 0), 0), None);
        assert_eq!(working_set(Hy4GraphLayout::resolve(1, 1), 0),
                   working_set(Hy4GraphLayout::resolve(1, 4), 0));
    }

    #[test]
    fn split_dfly2_delegates_to_shared_working_set_resolver() {
        let max = Some(Hy4GraphLayout::Exl3Max);
        for (rank, expected_w1, expected_soft, expected_hard) in [
            (0, 100, 196, 204),
            (1, 98, 194, 202),
        ] {
            let census = split_dfly2_working_set(max, rank).unwrap();
            assert_eq!(census.w1, expected_w1 << 20);
            let policy = graph_reserve::resolve_working_set(None, None, None, rank, census).unwrap();
            assert_eq!(policy.base_bytes, expected_soft << 20);
            assert_eq!(policy.bytes, expected_hard << 20);
        }
        assert_eq!(split_dfly2_working_set(Some(Hy4GraphLayout::Exl3Min), 0), None);
        assert_eq!(split_dfly2_working_set(Some(Hy4GraphLayout::Exl3MaxSpark), 0), None);
    }
}
