//! Shared prefill admission. Every Spark family prefills at a fixed depth of
//! three units cut from one reserve (TJ 2026-10-28, "just do fixed 3"): the
//! standard 8192 rows where the card holds it, else the largest 64-row
//! multiple that fits (the planner's fit cap), units roundup64(reserve / 3).
//! Every measured layout ran best at three; the adaptive controller (#762)
//! was removed. An explicit PREFILL_WAVES / PREFILL_ROWS pin is plain config
//! for measurement arms. A family may cap the depth below three where its
//! executor holds fewer units (hard capacity, not a preference).

/// The standard prefill reserve.
pub const RESERVE_ROWS: u64 = 8192;
pub const MAX_WAVES: u64 = 4;
/// The fixed prefill depth.
pub const DEPTH: u64 = 3;
pub const MIN_FILL: f64 = 0.5;
/// Prefill rows are 64-row pages.
pub const ROW_PAGE: u64 = 64;

/// The unit of a `waves`-way split of `reserve` rows: reserve / k rounded up
/// to a 64-row page (one and two waves both run reserve / 2). It runs as live
/// rows on the smallest compiled capacity at least that wide.
pub fn unit_rows_in(waves: u64, reserve: u64) -> u64 {
    reserve.div_ceil(waves.clamp(2, MAX_WAVES)).div_ceil(ROW_PAGE) * ROW_PAGE
}

/// The standard reserve's units: 4096 / 2752 / 2048 at two, three and four waves.
pub fn unit_rows(waves: u64) -> u64 { unit_rows_in(waves, RESERVE_ROWS) }

/// Hard executor capacities, consumed identically by plan, serve and golden.
/// Memory admission supplies the available arena extent, not a scheduling row cap.
#[derive(Debug, Clone, Copy)]
pub struct PrefillLimits {
    pub max_unit_rows: u64,
    pub max_span_rows: u64,
    /// The deepest depth the family's executor holds; the depth is
    /// [`DEPTH`] capped by it.
    pub max_waves: u64,
    pub capacities: &'static [u32],
}

impl Default for PrefillLimits {
    fn default() -> Self {
        Self { max_unit_rows: 4096, max_span_rows: RESERVE_ROWS + 64,
            max_waves: MAX_WAVES, capacities: &cuteafd_core::EXL3_CAPACITIES }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PrefillError {
    #[error("unsupported prefill shape: {waves} waves x {unit_rows} rows; units must be page-aligned 64..=4096 rows and waves 1..=4")]
    InvalidShape { unit_rows: u64, waves: u64 },
    #[error("unsupported prefill shape: {waves} waves x {unit_rows} rows; executor supports at most {max_waves} waves x {max_unit_rows} rows")]
    ExecutorCapacity { unit_rows: u64, waves: u64, max_unit_rows: u64, max_waves: u64 },
    #[error("unsupported prefill shape: {waves} waves x {unit_rows} rows exceeds the admitted {reserve_rows}-row live-slot reserve")]
    ReserveExceeded { unit_rows: u64, waves: u64, reserve_rows: u64 },
    #[error("unsupported prefill span: {span_rows} rows must equal {slot_rows} live slot rows within the {max_span_rows}-row shared arena")]
    IntakeSpan { span_rows: u64, slot_rows: u64, max_span_rows: u64 },
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PrefillOverrides {
    pub unit_rows: Option<u64>,
    pub waves: Option<u64>,
    pub span_rows: Option<u64>,
    /// A reserve below the standard one (TJ's reserve rule: the planner's
    /// fit cap, never shrunk for KV or onboarded layers); the depth cuts it.
    pub reserve_rows: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillPlan {
    pub unit_rows: u64,
    pub waves: u64,
    /// Rows of every slot-owned plane: `waves * unit_rows`.
    pub span_rows: u64,
}

impl PrefillPlan {
    pub fn units(self) -> super::PrefillShape {
        super::PrefillShape { unit_rows: self.unit_rows, waves: self.waves }
    }
}

/// The depth with Sparks: [`DEPTH`] within the executor's capacity; without
/// Sparks, one unit (no exchange to overlap; local executors run serial).
pub fn depth(spark_ranks: usize, limits: PrefillLimits) -> u64 {
    if spark_ranks == 0 { 1 } else { DEPTH.min(limits.max_waves).min(MAX_WAVES).max(1) }
}

pub fn prefill_shape(spark_ranks: usize, limits: PrefillLimits,
    explicit: PrefillOverrides) -> Result<PrefillPlan, PrefillError> {
    let reserve = explicit.reserve_rows.unwrap_or(RESERVE_ROWS);
    if reserve == 0 || reserve > RESERVE_ROWS || reserve % ROW_PAGE != 0 {
        return Err(PrefillError::IntakeSpan { span_rows: reserve, slot_rows: reserve, max_span_rows: RESERVE_ROWS });
    }
    let waves = explicit.waves.unwrap_or_else(|| depth(spark_ranks, limits));
    let rows = explicit.unit_rows.unwrap_or_else(|| {
        let budget = unit_rows_in(waves, reserve).min(limits.max_unit_rows)
            .min(limits.max_span_rows.min(RESERVE_ROWS + 64) / waves.max(1));
        // A fit-capped reserve runs its own cut; the standard one the widest
        // compiled capacity inside it (the shipped 4096 / 2752 / 2048).
        if explicit.reserve_rows.is_some() { budget } else {
            limits.capacities.iter().copied().map(u64::from)
                .filter(|&rows| rows >= 64 && rows % 64 == 0 && rows <= budget).max().unwrap_or(0)
        }
    });
    validate_shape(rows, waves, limits, reserve, explicit.span_rows)
}

fn validate_shape(rows: u64, waves: u64, limits: PrefillLimits, reserve: u64,
    explicit_span: Option<u64>) -> Result<PrefillPlan, PrefillError> {
    let units = super::PrefillShape { unit_rows: rows, waves };
    if units.validate().is_err() {
        return Err(PrefillError::InvalidShape { unit_rows: rows, waves });
    }
    if waves > limits.max_waves || rows > limits.max_unit_rows {
        return Err(PrefillError::ExecutorCapacity { unit_rows: rows, waves,
            max_unit_rows: limits.max_unit_rows, max_waves: limits.max_waves });
    }
    let reserve_rows = reserve.div_ceil(ROW_PAGE * waves) * (ROW_PAGE * waves);
    let slot_rows = units.slot_rows();
    if slot_rows > reserve_rows {
        return Err(PrefillError::ReserveExceeded { unit_rows: rows, waves, reserve_rows });
    }
    let span_rows = explicit_span.unwrap_or(slot_rows);
    let max_span_rows = limits.max_span_rows.min(RESERVE_ROWS + 64);
    if span_rows != slot_rows || span_rows > max_span_rows {
        return Err(PrefillError::IntakeSpan { span_rows, slot_rows, max_span_rows });
    }
    Ok(PrefillPlan { unit_rows: rows, waves, span_rows })
}

/// Use fewer waves when a short prompt cannot keep each unit half full.
pub fn waves_used(rows: u64, waves: u64, unit: u64) -> u64 {
    if rows == 0 || unit == 0 || waves <= 1 { return 1; }
    let cover = rows.div_ceil(unit);
    let filled = (rows as f64 / (MIN_FILL * unit as f64)).floor() as u64;
    cover.max(waves.min(filled)).min(waves).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// TJ 2026-10-28: three units of the reserve with Sparks, one without;
    /// an explicit depth still pins (measurement arms).
    #[test]
    fn the_depth_is_three_with_sparks_and_one_without() {
        for ranks in 1..=8 {
            assert_eq!(prefill_shape(ranks, PrefillLimits::default(), Default::default()).unwrap(),
                PrefillPlan { unit_rows: 2752, waves: 3, span_rows: 8256 });
        }
        assert_eq!(prefill_shape(0, PrefillLimits::default(), Default::default()).unwrap(),
            PrefillPlan { unit_rows: 4096, waves: 1, span_rows: 4096 });
        for (waves, rows) in [(1, 4096), (2, 4096), (3, 2752), (4, 2048)] {
            let pinned = prefill_shape(4, PrefillLimits::default(), PrefillOverrides { waves: Some(waves), ..Default::default() }).unwrap();
            assert_eq!((pinned.unit_rows, pinned.waves, pinned.span_rows), (rows, waves, rows * waves));
        }
    }

    /// A card that cannot hold the standard reserve cuts its capped reserve
    /// into three units, never smaller (TJ's reserve rule).
    #[test]
    fn a_fit_capped_reserve_is_cut_in_three() {
        for (reserve, unit) in [(7104, 2368), (5376, 1792), (3264, 1088), (768, 256)] {
            let plan = prefill_shape(4, PrefillLimits::default(), PrefillOverrides { reserve_rows: Some(reserve), ..Default::default() }).unwrap();
            assert_eq!((plan.unit_rows, plan.waves, plan.span_rows), (unit, 3, 3 * unit), "{reserve}");
        }
        for reserve in [0, 8256, 100] {
            assert!(matches!(prefill_shape(4, PrefillLimits::default(), PrefillOverrides { reserve_rows: Some(reserve), ..Default::default() }),
                Err(PrefillError::IntakeSpan { .. })), "{reserve}");
        }
    }

    #[test]
    fn units_are_automatic_splits_of_the_reserve() {
        assert_eq!((2..=4).map(|k| unit_rows_in(k, RESERVE_ROWS)).collect::<Vec<_>>(), [4096, 2752, 2048]);
        for reserve in (64..=RESERVE_ROWS).step_by(64) {
            for k in 2..=MAX_WAVES {
                let unit = unit_rows_in(k, reserve);
                assert!(unit % ROW_PAGE == 0 && k * unit >= reserve && unit <= reserve, "{reserve}/{k}: {unit}");
            }
        }
        assert_eq!((2..=4).map(|k| unit_rows_in(k, 3840)).collect::<Vec<_>>(), [1920, 1280, 960]);
    }

    #[test]
    fn units_are_the_exact_capacities_inside_the_reserve() {
        for waves in 2..=MAX_WAVES {
            let unit = unit_rows(waves);
            assert!(cuteafd_core::EXL3_CAPACITIES.contains(&(unit as u32)));
            assert!(unit * waves <= RESERVE_ROWS + 64 * waves && unit * waves >= RESERVE_ROWS);
        }
    }

    #[test]
    fn executor_capacity_caps_the_depth_and_the_unit() {
        let shallow = PrefillLimits { max_waves: 2, ..Default::default() };
        assert_eq!(prefill_shape(6, shallow, Default::default()).unwrap(),
            PrefillPlan { unit_rows: 4096, waves: 2, span_rows: 8192 });
        assert!(matches!(prefill_shape(6, shallow, PrefillOverrides { waves: Some(3), ..Default::default() }),
            Err(PrefillError::ExecutorCapacity { waves: 3, max_waves: 2, .. })));
        let narrow = PrefillLimits { max_unit_rows: 1024, max_span_rows: 2048, ..Default::default() };
        assert_eq!(prefill_shape(6, narrow, Default::default()).unwrap(),
            PrefillPlan { unit_rows: 256, waves: 3, span_rows: 768 });
        assert!(matches!(prefill_shape(6, PrefillLimits { max_unit_rows: 1024, ..Default::default() },
            PrefillOverrides { unit_rows: Some(2048), waves: Some(2), ..Default::default() }),
            Err(PrefillError::ExecutorCapacity { unit_rows: 2048, .. })));
        let tight = PrefillLimits { max_span_rows: 128, capacities: &[256, 1024, 4096], ..Default::default() };
        assert!(prefill_shape(6, tight, Default::default()).is_err());
    }

    #[test]
    fn explicit_shapes_cannot_grow_the_shared_live_reserve() {
        for (waves, rows) in [(1, 4096), (2, 4096), (3, 2752), (4, 2048), (4, 512)] {
            let shape = prefill_shape(6, PrefillLimits::default(), PrefillOverrides {
                waves: Some(waves), unit_rows: Some(rows), ..Default::default()
            }).unwrap();
            assert_eq!((shape.waves, shape.unit_rows, shape.span_rows), (waves, rows, waves * rows));
        }
        for (waves, rows) in [(3, 2816), (3, 4096), (4, 2112), (4, 4096)] {
            assert!(matches!(prefill_shape(6, PrefillLimits::default(), PrefillOverrides {
                waves: Some(waves), unit_rows: Some(rows), ..Default::default()
            }), Err(PrefillError::ReserveExceeded { .. })));
        }
    }

    #[test]
    fn intake_is_exactly_the_shared_live_slot_extent() {
        let limits = PrefillLimits { max_span_rows: u64::MAX, ..Default::default() };
        assert_eq!(prefill_shape(6, limits, Default::default()).unwrap(),
            PrefillPlan { unit_rows: 2752, waves: 3, span_rows: 8256 });
        for span in [0, 8192, 8257, 16384] {
            assert!(matches!(prefill_shape(6, limits, PrefillOverrides {
                waves: Some(3), span_rows: Some(span), ..Default::default()
            }), Err(PrefillError::IntakeSpan { .. })));
        }
    }

    #[test]
    fn explicit_shape_errors_are_typed_and_never_clamped() {
        for (rows, waves) in [(0, 2), (65, 2), (8192, 1), (4096, 0), (2048, 5)] {
            assert_eq!(prefill_shape(6, PrefillLimits::default(), PrefillOverrides {
                unit_rows: Some(rows), waves: Some(waves), ..Default::default()
            }).unwrap_err(), PrefillError::InvalidShape { unit_rows: rows, waves });
        }
    }

    #[test]
    fn qwen_token_tables_follow_explicit_shared_depth() {
        let limits = crate::families::qwen4::prefill_limits(false);
        assert_eq!(prefill_shape(6, limits, Default::default()).unwrap(),
            PrefillPlan { unit_rows: 2752, waves: 3, span_rows: 8256 });
        for waves in 2..=4 {
            assert_eq!(prefill_shape(6, limits, PrefillOverrides { waves: Some(waves), ..Default::default() }).unwrap().waves, waves);
        }
        assert_eq!(prefill_shape(0, crate::families::qwen4::prefill_limits(true), Default::default()).unwrap().waves, 1);
    }

    #[test]
    fn short_prompts_keep_units_at_least_half_full() {
        let (k, unit) = (3, 2752);
        assert_eq!(waves_used(1, k, unit), 1);
        assert_eq!(waves_used(2752, k, unit), 2);
        assert_eq!(waves_used(2 * 1376 - 1, k, unit), 1);
        assert_eq!(waves_used(3 * 1376 - 1, k, unit), 2);
        assert_eq!(waves_used(3 * 1376, k, unit), 3);
        assert_eq!(waves_used(8256, k, unit), 3);
        assert_eq!(waves_used(65536, k, unit), 3);
        assert_eq!(waves_used(65536, 1, unit), 1);
        assert_eq!(waves_used(4095, 2, 4096), 1);
        assert_eq!(waves_used(4096, 2, 4096), 2);
        assert_eq!(waves_used(4095, 4, 2048), 3);
        assert_eq!(waves_used(0, 3, 2752), 1);
    }
}
