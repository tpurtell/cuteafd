//! Storage contract for MiMo's retained per-layer decode graph executables.
//! The byte envelope is calibrated on the current segmented Pro program;
//! it is a conservative admission bound, not a portable CUDA allocation size.
use crate::serving_capacity::CacheGeometryError;
use cuteafd_core::serving_capacity::MemoryReservation;

pub const MIMO_DECODE_ROWS: usize = 64;
/// Both full-row logits and retained subset logits are reachable in serving.
pub const MIMO_DECODE_TAIL_HEAD_VARIANTS: [bool; 2] = [true, false];
pub const MIMO_GRAPH_EXEC_BOUND_BYTES: u64 = 192 << 10;
pub const MIMO_GRAPH_DRIVER_MARGIN_BYTES: u64 = 64 << 20;

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

    pub fn graph_set(&self, bytes_per_exec: u64, driver_margin: u64) -> crate::placement::inventory::GraphSet {
        use crate::placement::inventory::{GraphRank, GraphSet, Lifetime};
        GraphSet { ranks: self.executables_per_rank.iter().map(|&count| GraphRank {
            executables: count as u64, margin: if count == 0 { 0 } else { driver_margin },
            bytes: if count == 0 { 0 } else { (count as u64).saturating_mul(bytes_per_exec).saturating_add(driver_margin) },
        }).collect(), lifetime: Lifetime::Startup, shapes: self.row_shapes as u64 }
    }

    /// A measured 64-shape Pro capture consumed 616/566 MiB on lead/peer.
    /// Its largest shape increment was 12 MiB per 70/71 executables. 192 KiB
    /// per executable exceeds that observed increment; the separate 64 MiB
    /// margin bounds driver/rounding variation without calling it measured.
    /// Modules, other libraries, sampling and grammar are reserved separately.
    pub fn reservations(
        &self,
        rank: usize,
        bytes_per_exec: u64,
        driver_margin: u64,
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
        if bytes_per_exec == 0 {
            return Err(CacheGeometryError::Unsupported {
                family: "mimo_v2",
                what: "positive decode graph executable reservation",
            });
        }
        let bytes =
            (count as u64)
                .checked_mul(bytes_per_exec)
                .ok_or(CacheGeometryError::Overflow(
                    "MiMo decode graph executable bytes",
                ))?;
        bytes
            .checked_add(driver_margin)
            .ok_or(CacheGeometryError::Overflow(
                "MiMo decode graph storage bound",
            ))?;
        Ok(vec![
            MemoryReservation {
                name: "runtime.decode_graph_exec_storage_bound".into(),
                bytes,
            },
            MemoryReservation {
                name: "runtime.decode_graph_driver_margin".into(),
                bytes: driver_margin,
            },
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pro_counts_both_head_tails_and_actual_physical_ranks() {
        let plan = MimoDecodeGraphPlan::new(70, 2, 64, true).unwrap();
        assert!(plan.reservations(0, 0, 0).is_err());
        assert_eq!(plan.segments_per_rank, vec![72, 70]);
        assert_eq!(plan.executables_per_rank, vec![4608, 4480]);
        let lead = plan
            .reservations(
                0,
                MIMO_GRAPH_EXEC_BOUND_BYTES,
                MIMO_GRAPH_DRIVER_MARGIN_BYTES,
            )
            .unwrap();
        assert_eq!(lead[0].bytes, 864 << 20);
        assert_eq!(lead[1].bytes, 64 << 20);
        assert_eq!(
            MimoDecodeGraphPlan::new(70, 1, 1, true)
                .unwrap()
                .executables_per_rank,
            vec![72]
        );
    }

    #[test]
    fn disabled_graphs_reserve_no_executables_or_driver_margin() {
        let plan = MimoDecodeGraphPlan::new(70, 2, 64, false).unwrap();
        assert_eq!(plan.executables_per_rank, vec![0, 0]);
        assert!(plan.reservations(0, 0, u64::MAX).unwrap().is_empty());
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
        assert!(plan.reservations(0, u64::MAX, 0).is_err());
        assert!(plan.reservations(0, 1, u64::MAX).is_err());
        assert!(plan.reservations(2, 1, 0).is_err());
    }
}
