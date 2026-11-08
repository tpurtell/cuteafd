//! BF16 coordinator receive planes shared by Spark-backed families, and the
//! shared prefill unit shape (PLAN "Prefill contract").
use cuteafd_core::serving_capacity::spark_intake_allocation_bytes;

/// One independently allocated BF16 partial plane per rank and lane.
pub fn spark_intake_bytes(lanes: u64, ranks: u64, rows: u64, hidden: u64) -> u64 {
    spark_intake_requests(lanes, ranks, rows, hidden).into_iter().sum()
}

/// The exact per-plane requests behind [`spark_intake_bytes`]: `SparkIntake`
/// allocates one backing per rank and lane.
pub fn spark_intake_requests(lanes: u64, ranks: u64, rows: u64, hidden: u64) -> Vec<u64> {
    let bytes = spark_intake_allocation_bytes(rows * hidden * 2).expect("Spark intake backing overflow");
    vec![bytes; usize::try_from(lanes * ranks).expect("Spark intake lane count")]
}

/// One allocation per rank containing every outstanding wave's logical slot.
/// Round the whole reserve once, matching `SparkIntake::with_waves`.
pub fn prefill_return_bytes(waves: u64, unit_rows: u64, ranks: u64, hidden: u64) -> u64 {
    spark_intake_bytes(1, ranks, waves * unit_rows, hidden)
}

/// BF16 fallback capacity also holds packed replies; fused reduction needs no scratch.
pub fn prefill_return_bytes_with_format(waves: u64, unit_rows: u64, ranks: u64,
    hidden: u64, format: cuteafd_core::SparkReplyFormat) -> anyhow::Result<u64> {
    use anyhow::Context;
    if ranks == 0 { return Ok(0); }
    let rows = waves.checked_mul(unit_rows).context("Spark return rows overflow")?;
    let logical = rows.checked_mul(hidden).and_then(|n| n.checked_mul(2))
        .context("Spark return plane overflow")?;
    let planes = spark_intake_allocation_bytes(logical)?
        .checked_mul(ranks).context("Spark return planes overflow")?;
    anyhow::ensure!(format.row_bytes(usize::try_from(hidden)?).is_some(), "unsupported Spark combined reply width");
    Ok(planes)
}

/// The exact per-rank requests behind [`prefill_return_bytes_with_format`]:
/// one combined return plane per rank (`SparkIntake::with_waves`).
pub fn prefill_return_requests_with_format(waves: u64, unit_rows: u64, ranks: u64,
    hidden: u64, format: cuteafd_core::SparkReplyFormat) -> anyhow::Result<Vec<u64>> {
    use anyhow::Context;
    let bytes = prefill_return_bytes_with_format(waves, unit_rows, ranks, hidden, format)?;
    if bytes == 0 { return Ok(Vec::new()); }
    Ok(vec![bytes / ranks; usize::try_from(ranks).context("Spark return ranks")?])
}

/// Prefill row units and how many are in flight on one lane and transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillShape {
    pub unit_rows: u64,
    pub waves: u64,
}

impl PrefillShape {
    pub fn validate(self) -> anyhow::Result<Self> {
        anyhow::ensure!((64..=4096).contains(&self.unit_rows) && self.unit_rows % 64 == 0
            && (1..=4).contains(&self.waves), "prefill unit {} rows x {} waves outside 64..=4096 (page aligned) x 1..=4",
            self.unit_rows, self.waves);
        Ok(self)
    }

    /// Rows of every slot-owned plane: one unit per wave in flight.
    pub fn slot_rows(self) -> u64 { self.unit_rows * self.waves }

}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partial_reply_plan_charges_one_padded_reserve_per_rank() {
        for ranks in [4, 6] {
            let logical = 3 * 2752 * 34_864;
            assert_eq!(prefill_return_bytes(3, 2752, ranks, 34_864 / 2),
                ranks * spark_intake_allocation_bytes(logical).unwrap());
        }
        let shared = 4 * spark_intake_allocation_bytes(3 * 64 * 16 * 2).unwrap();
        let independent = 12 * spark_intake_allocation_bytes(64 * 16 * 2).unwrap();
        assert_eq!(prefill_return_bytes(3, 64, 4, 16), shared);
        assert_eq!(spark_intake_bytes(3, 4, 64, 16), independent);
        assert!(shared < independent);
    }

    #[test]
    fn aligned_intake_charge_keeps_one_page_of_alignment_slack() {
        for lanes in [1, 2, 4] { for ranks in [0, 4, 6] { for rows in [1024, 4096] {
            assert_eq!(spark_intake_bytes(lanes, ranks, rows, 6144),
                lanes * ranks * spark_intake_allocation_bytes(rows * 6144 * 2).unwrap());
        } } }
    }

    #[test]
    fn two_2048_row_return_slots_cost_one_4096_row_intake() {
        assert_eq!(prefill_return_bytes(2, 2048, 6, 3584), spark_intake_bytes(1, 6, 4096, 3584));
        assert_eq!(PrefillShape { unit_rows: 2048, waves: 2 }.slot_rows(), 4096);
    }

    #[test]
    fn combined_reply_charge_retains_bf16_planes_without_codec_scratch() -> anyhow::Result<()> {
        use cuteafd_core::SparkReplyFormat::{Bf16, Fp8, Nvfp4};
        for hidden in [2560, 3584, 4096, 5120, 6144, 7168] {
            for (waves, rows) in [(1, 64), (2, 4096), (3, 2752), (4, 2048)] {
                for ranks in [1, 2, 4, 6, 8] {
                    let planes = ranks * spark_intake_allocation_bytes(waves * rows * hidden * 2)?;
                    for format in [Bf16, Fp8, Nvfp4] {
                        assert_eq!(prefill_return_bytes_with_format(waves, rows, ranks, hidden, format)?,
                            planes);
                    }
                }
            }
        }
        assert_eq!(prefill_return_bytes_with_format(u64::MAX, u64::MAX, 0, 17, Fp8)?, 0);
        for format in [Fp8, Nvfp4] {
            assert!(prefill_return_bytes_with_format(1, 64, 1, 17, format).is_err());
            assert!(prefill_return_bytes_with_format(u64::MAX, 2, 1, 4096, format).is_err());
            assert!(prefill_return_bytes_with_format(1, u64::MAX, 1, 4096, format).is_err());
            assert!(prefill_return_bytes_with_format(1, 64, u64::MAX, 4096, format).is_err());
        }
        Ok(())
    }

    #[test]
    fn shapes_reject_unaligned_units_and_excess_depth() {
        assert!(PrefillShape { unit_rows: 100, waves: 2 }.validate().is_err());
        assert!(PrefillShape { unit_rows: 2048, waves: 5 }.validate().is_err());
        assert!(PrefillShape { unit_rows: 2752, waves: 3 }.validate().is_ok());
    }
}
