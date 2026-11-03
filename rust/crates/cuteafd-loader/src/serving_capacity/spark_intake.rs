//! BF16 coordinator receive planes shared by Spark-backed families, and the
//! shared prefill unit shape (PLAN "Prefill contract").
use cuteafd_core::serving_capacity::spark_intake_allocation_bytes;

/// One independently allocated BF16 partial plane per rank and lane.
pub fn spark_intake_bytes(lanes: u64, ranks: u64, rows: u64, hidden: u64) -> u64 {
    lanes * ranks * spark_intake_allocation_bytes(rows * hidden * 2).expect("Spark intake backing overflow")
}

/// One allocation per rank containing every outstanding wave's logical slot.
/// Round the whole reserve once, matching `SparkIntake::with_waves`.
pub fn prefill_return_bytes(waves: u64, unit_rows: u64, ranks: u64, hidden: u64) -> u64 {
    spark_intake_bytes(1, ranks, waves * unit_rows, hidden)
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
    fn shapes_reject_unaligned_units_and_excess_depth() {
        assert!(PrefillShape { unit_rows: 100, waves: 2 }.validate().is_err());
        assert!(PrefillShape { unit_rows: 2048, waves: 5 }.validate().is_err());
        assert!(PrefillShape { unit_rows: 2752, waves: 3 }.validate().is_ok());
    }
}
