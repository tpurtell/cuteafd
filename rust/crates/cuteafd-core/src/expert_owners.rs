//! Checkpoint-independent whole-expert ownership across Spark ranks (expert
//! parallel). Each rank holds complete experts, so any per-expert nonlinear
//! step after the down projection (Muse Spark's `post_expert_norm`) stays exact
//! on the rank, and each rank returns one weighted partial sum over its owned
//! routes: the exchange, reply size and coordinator reduce match TP.
//!
//! The owner map is data. `staggered` builds it from integer arithmetic only;
//! a measured popularity table may replace it later (the planner, the workers
//! and the coordinator all read the same map and seal it by fingerprint).
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ExpertOwnersError {
    #[error("whole-expert ownership needs 1..=8 ranks, at least one expert per rank and at least one layer")]
    Geometry,
    #[error("whole-expert owner table layer {layer} expert {expert} names rank {rank} outside world {world}")]
    Rank { layer: usize, expert: usize, rank: usize, world: usize },
    #[error("whole-expert owner table layer {layer} leaves rank {rank} without experts")]
    Empty { layer: usize, rank: usize },
    #[error("whole-expert owner table is ragged: layer {layer} has {found} experts, expected {expected}")]
    Ragged { layer: usize, found: usize, expected: usize },
}

/// The owning rank of every routed expert in every routed layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpertOwners {
    pub world: usize,
    /// `owners[layer][expert]` = rank.
    pub owners: Vec<Vec<u8>>,
}

impl ExpertOwners {
    /// Round-robin by expert id, rotated by layer: rank `(e + layer) % world`.
    /// Equal-size ranks when `world` divides the experts; otherwise the
    /// remainder rotates across ranks layer by layer, so no rank is the
    /// largest in every layer.
    pub fn staggered(layers: usize, experts: usize, world: usize) -> Result<Self, ExpertOwnersError> {
        if !(1..=8).contains(&world) || experts < world || layers == 0 {
            return Err(ExpertOwnersError::Geometry);
        }
        let owners = (0..layers).map(|layer| (0..experts).map(|expert| ((expert + layer) % world) as u8).collect())
            .collect();
        Ok(Self { world, owners })
    }

    /// A measured table (e.g. popularity-balanced), validated.
    pub fn from_table(world: usize, owners: Vec<Vec<u8>>) -> Result<Self, ExpertOwnersError> {
        if !(1..=8).contains(&world) || owners.is_empty() || owners[0].len() < world {
            return Err(ExpertOwnersError::Geometry);
        }
        let experts = owners[0].len();
        for (layer, row) in owners.iter().enumerate() {
            if row.len() != experts {
                return Err(ExpertOwnersError::Ragged { layer, found: row.len(), expected: experts });
            }
            let mut held = vec![false; world];
            for (expert, &rank) in row.iter().enumerate() {
                let rank = usize::from(rank);
                if rank >= world { return Err(ExpertOwnersError::Rank { layer, expert, rank, world }); }
                held[rank] = true;
            }
            if let Some(rank) = held.iter().position(|held| !held) {
                return Err(ExpertOwnersError::Empty { layer, rank });
            }
        }
        Ok(Self { world, owners })
    }

    pub fn layers(&self) -> usize { self.owners.len() }
    pub fn experts(&self) -> usize { self.owners.first().map_or(0, Vec::len) }

    /// Owned global expert ids of `rank` in `layer`, ascending.
    pub fn owned(&self, layer: usize, rank: usize) -> Vec<usize> {
        self.owners.get(layer).map_or_else(Vec::new, |row| row.iter().enumerate()
            .filter_map(|(expert, &owner)| (usize::from(owner) == rank).then_some(expert)).collect())
    }

    /// The rank-local map the expert kernel reads: global id -> local slot, -1
    /// for experts owned elsewhere (their routes are skipped on this rank).
    pub fn global_to_local(&self, layer: usize, rank: usize) -> Vec<i32> {
        let mut map = vec![-1; self.experts()];
        for (local, expert) in self.owned(layer, rank).into_iter().enumerate() {
            map[expert] = local as i32;
        }
        map
    }

    /// The largest owned count of any rank in any layer (the compiled extent).
    pub fn max_owned(&self) -> usize {
        (0..self.layers()).flat_map(|layer| (0..self.world).map(move |rank| (layer, rank)))
            .map(|(layer, rank)| self.owned(layer, rank).len()).max().unwrap_or(0)
    }

    /// Seal shared by planner, coordinator and every worker.
    pub fn fingerprint(&self) -> String {
        let mut hash = Sha256::new();
        hash.update(b"cuteafd.expert-owners.v1");
        hash.update((self.world as u64).to_le_bytes());
        hash.update((self.layers() as u64).to_le_bytes());
        for row in &self.owners {
            hash.update((row.len() as u64).to_le_bytes());
            hash.update(row);
        }
        format!("{:x}", hash.finalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staggered_ownership_is_a_partition_balanced_across_layers() {
        for world in [4usize, 6] {
            let owners = ExpertOwners::staggered(62, 256, world).unwrap();
            let mut totals = vec![0usize; world];
            for layer in 0..62 {
                let mut seen = vec![false; 256];
                for rank in 0..world {
                    let owned = owners.owned(layer, rank);
                    assert!(owned.len() == 256 / world || owned.len() == 256 / world + 1);
                    totals[rank] += owned.len();
                    let map = owners.global_to_local(layer, rank);
                    for (local, &expert) in owned.iter().enumerate() {
                        assert!(!seen[expert]);
                        seen[expert] = true;
                        assert_eq!(map[expert], local as i32);
                    }
                    assert_eq!(map.iter().filter(|&&slot| slot >= 0).count(), owned.len());
                }
                assert!(seen.into_iter().all(|seen| seen));
            }
            // The remainder rotates: totals differ by at most one layer's remainder share.
            assert!(totals.iter().max().unwrap() - totals.iter().min().unwrap() <= 62);
            assert_eq!(owners.max_owned(), 256usize.div_ceil(world));
        }
    }

    #[test]
    fn tables_are_validated_and_sealed() {
        let owners = ExpertOwners::staggered(2, 8, 4).unwrap();
        let table = ExpertOwners::from_table(4, owners.owners.clone()).unwrap();
        assert_eq!(table.fingerprint(), owners.fingerprint());
        let mut rotated = owners.owners.clone();
        rotated[1].rotate_left(1);
        assert_ne!(ExpertOwners::from_table(4, rotated).unwrap().fingerprint(), owners.fingerprint());
        assert!(matches!(ExpertOwners::from_table(4, vec![vec![0, 1, 2, 4]]), Err(ExpertOwnersError::Rank { .. })));
        assert!(matches!(ExpertOwners::from_table(4, vec![vec![0, 1, 2, 2]]), Err(ExpertOwnersError::Empty { rank: 3, .. })));
        assert!(matches!(ExpertOwners::from_table(2, vec![vec![0, 1], vec![0]]), Err(ExpertOwnersError::Ragged { .. })));
        assert!(ExpertOwners::staggered(1, 3, 4).is_err());
        assert!(ExpertOwners::staggered(1, 256, 9).is_err());
    }
}
