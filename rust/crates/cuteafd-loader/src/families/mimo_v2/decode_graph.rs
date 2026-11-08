//! Storage contract for MiMo's retained per-layer decode graph executables.
//! Bytes are the measured whole-capture retention of a 64-shape Pro launch,
//! scaled by executable count with the shared estimate arithmetic: no
//! per-executable envelope and no separate driver margin.
use crate::serving_capacity::CacheGeometryError;
use cuteafd_core::serving_capacity::MemoryReservation;

pub const MIMO_DECODE_ROWS: usize = 64;
/// Both full-row logits and retained subset logits are reachable in serving.
pub const MIMO_DECODE_TAIL_HEAD_VARIANTS: [bool; 2] = [true, false];

/// Measured retained storage of the 64-shape Pro capture on SM120
/// (driver/runtime 13020, CUDA 13.2, 2026-11-08): 616 MiB over 4544 lead
/// executables (70 layers plus one tail-head variant per shape) and 566 MiB
/// over 4480 peer executables (70 layers per shape). The lead set now carries
/// the second, headless tail variant; the measured maximum per-shape increment
/// was 12 MiB, charged once here rather than spread over every executable.
/// [`crate::placement::inventory::GraphRank::estimate`] scales each rank's
/// observation by executable count, rounded once and without a margin.
pub const MIMO_GRAPH_LEAD_OBSERVED: (u64, u64) = (4608, (616 + 12) << 20);
pub const MIMO_GRAPH_PEER_OBSERVED: (u64, u64) = (4480, 566 << 20);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MimoDecodeGraphPlan {
    pub row_shapes: usize,
    /// Lead rank has each layer plus both full/partial-head tail variants.
    /// A peer has each attention layer and no vocabulary tail.
    pub segments_per_rank: Vec<usize>,
    pub executables_per_rank: Vec<usize>,
}

impl MimoDecodeGraphPlan {
    pub fn new(
        layers: usize,
        ranks: usize,
        row_shapes: usize,
        enabled: bool,
    ) -> Result<Self, CacheGeometryError> {
        if !(1..=2).contains(&ranks) || layers == 0 || !(1..=MIMO_DECODE_ROWS).contains(&row_shapes)
        {
            return Err(CacheGeometryError::Unsupported {
                family: "mimo_v2",
                what: "segmented decode graph layer/rank/row geometry",
            });
        }
        if !enabled {
            return Ok(Self {
                row_shapes: 0,
                segments_per_rank: vec![0; ranks],
                executables_per_rank: vec![0; ranks],
            });
        }
        let lead = layers
            .checked_add(MIMO_DECODE_TAIL_HEAD_VARIANTS.len())
            .ok_or(CacheGeometryError::Overflow(
                "MiMo decode graph tail segments",
            ))?;
        let mut segments_per_rank = vec![lead];
        if ranks == 2 {
            segments_per_rank.push(layers);
        }
        let executables_per_rank = segments_per_rank
            .iter()
            .map(|&segments| {
                segments
                    .checked_mul(row_shapes)
                    .ok_or(CacheGeometryError::Overflow(
                        "MiMo decode graph executable count",
                    ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            row_shapes,
            segments_per_rank,
            executables_per_rank,
        })
    }

    /// The measured whole-capture observation `(executables, retained bytes)`
    /// this physical rank's set is priced from. Rank 0 carries the head tails.
    pub fn observation(&self, rank: usize) -> Result<(u64, u64), CacheGeometryError> {
        match self.executables_per_rank.get(rank) {
            Some(_) if rank == 0 => Ok(MIMO_GRAPH_LEAD_OBSERVED),
            Some(_) if rank == 1 => Ok(MIMO_GRAPH_PEER_OBSERVED),
            _ => Err(CacheGeometryError::Unsupported {
                family: "mimo_v2",
                what: "decode graph physical rank",
            }),
        }
    }

    /// The admitted startup graph set: one measured owner per physical rank.
    pub fn graph_set(&self) -> Result<crate::placement::inventory::GraphSet, CacheGeometryError> {
        use crate::placement::inventory::{GraphRank, GraphSet, Lifetime};
        let mut ranks = Vec::with_capacity(self.executables_per_rank.len());
        for (rank, &count) in self.executables_per_rank.iter().enumerate() {
            ranks.push(if count == 0 {
                GraphRank::estimated(0, 0)
            } else {
                let (observed, bytes) = self.observation(rank)?;
                GraphRank::estimate(count as u64, bytes, observed).map_err(|_| {
                    CacheGeometryError::Unsupported {
                        family: "mimo_v2",
                        what: "decode graph executable byte conversion",
                    }
                })?
            });
        }
        Ok(GraphSet {
            ranks,
            lifetime: Lifetime::Startup,
            shapes: self.row_shapes as u64,
        })
    }

    /// The retained startup graph owners of one physical rank. Nothing is
    /// reserved for an unlisted shape: late capture is refused, so a shape
    /// outside the startup set allocates no graph storage.
    pub fn reservations(
        &self,
        rank: usize,
    ) -> Result<Vec<MemoryReservation>, CacheGeometryError> {
        let &count =
            self.executables_per_rank
                .get(rank)
                .ok_or(CacheGeometryError::Unsupported {
                    family: "mimo_v2",
                    what: "decode graph physical rank",
                })?;
        if count == 0 {
            return Ok(Vec::new());
        }
        let bytes = self.graph_set()?.bytes(rank);
        Ok(vec![MemoryReservation {
            name: "runtime.decode_graph_exec_storage_bound".into(),
            bytes,
        }])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measured_observation_prices_both_head_tails_and_actual_physical_ranks() {
        let plan = MimoDecodeGraphPlan::new(70, 2, 64, true).unwrap();
        assert_eq!(plan.segments_per_rank, vec![72, 70]);
        assert_eq!(plan.executables_per_rank, vec![4608, 4480]);
        let lead = plan.reservations(0).unwrap();
        assert_eq!(lead.len(), 1);
        assert_eq!(lead[0].bytes, 628 << 20);
        assert_eq!(plan.reservations(1).unwrap()[0].bytes, 566 << 20);
        assert_eq!(
            MimoDecodeGraphPlan::new(70, 1, 1, true)
                .unwrap()
                .executables_per_rank,
            vec![72]
        );
        assert_eq!(
            MimoDecodeGraphPlan::new(70, 1, 64, true).unwrap().reservations(0).unwrap()[0].bytes,
            628 << 20
        );
    }

    #[test]
    fn the_graph_set_carries_no_margin() {
        let set = MimoDecodeGraphPlan::new(70, 2, 64, true).unwrap().graph_set().unwrap();
        assert_eq!(set.shapes, 64);
        assert!(set.ranks.iter().all(|rank| rank.margin == 0));
        assert_eq!(set.at_ready(0) + set.growth(0), 628 << 20);
        assert_eq!(set.at_ready(1) + set.growth(1), 566 << 20);
    }

    #[test]
    fn other_row_counts_scale_the_measured_observation() {
        let plan = MimoDecodeGraphPlan::new(70, 2, 1, true).unwrap();
        assert_eq!(plan.executables_per_rank, vec![72, 70]);
        assert_eq!(plan.reservations(0).unwrap()[0].bytes, (628u64 << 20).div_ceil(64));
        assert_eq!(plan.reservations(1).unwrap()[0].bytes, (566u64 << 20).div_ceil(64));
    }

    #[test]
    fn disabled_graphs_reserve_no_executables_or_storage() {
        let plan = MimoDecodeGraphPlan::new(70, 2, 64, false).unwrap();
        assert_eq!(plan.executables_per_rank, vec![0, 0]);
        assert!(plan.reservations(0).unwrap().is_empty());
        assert!(plan.graph_set().unwrap().ranks.iter().all(|rank| rank.bytes == 0));
    }

    #[test]
    fn invalid_or_overflowing_graph_geometry_is_named() {
        for (layers, ranks, rows) in [
            (0, 1, 64),
            (70, 0, 64),
            (70, 3, 64),
            (70, 1, 0),
            (70, 1, 65),
        ] {
            assert!(matches!(
                MimoDecodeGraphPlan::new(layers, ranks, rows, true),
                Err(CacheGeometryError::Unsupported { .. })
            ));
        }
        assert!(matches!(
            MimoDecodeGraphPlan::new(usize::MAX, 1, 64, true),
            Err(CacheGeometryError::Overflow(_))
        ));
        let plan = MimoDecodeGraphPlan::new(70, 2, 64, true).unwrap();
        assert!(plan.reservations(2).is_err());
    }
}
