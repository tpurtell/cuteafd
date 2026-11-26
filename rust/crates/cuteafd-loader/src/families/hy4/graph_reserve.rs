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

// Populate only from a physical C1 census of this family.
const CENSUS: &[(Layout, usize, WorkingSet)] = &[];

pub fn reserve_policy(layout: Option<Layout>, rank: usize) -> Result<Resolved, Error> {
    graph_reserve::current_census(rank, layout, CENSUS)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }
}
