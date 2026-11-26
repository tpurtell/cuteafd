//! The decode-lane contract every family shares (PLAN "Two decode lanes").
//!
//! A decode lane is an independent decode batch with its own attention step,
//! its own expert exchange on its own decode link, and its own mutable
//! workspaces, graphs, drafter scratch and sampler. Weights, KV pools and
//! request-indexed rings stay shared. Every byte a second lane owns is a
//! family planner charge made before allocation.
use super::PlanError;

/// Most decode lanes one coordinator runs.
pub const MAX_DECODE_LANES: u8 = 2;

/// The reason of a descriptor with no decode loop of its own (a drafter
/// checkpoint, which drafts inside its target family's lane).
pub const DRAFTER: &str = "a drafter has no decode loop of its own: it drafts inside its target family's lane";

/// What a family's executor supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneSupport {
    /// One decode batch only; `reason` records why a second lane is not
    /// wired (a measurement or the predicted cost), and the planner reports it.
    One { reason: &'static str },
    /// One or two lanes; defaults by coordinator RTX GPU count (`single_rtx` on
    /// 1× RTX, `dual_rtx` on 2× RTX) until an explicit override is passed.
    OneOrTwo { single_rtx: u8, dual_rtx: u8 },
}

impl LaneSupport {
    /// Most lanes the executor can run (each owns a Spark decode execution).
    pub fn max_lanes(self) -> u8 {
        match self { Self::One { .. } => 1, Self::OneOrTwo { .. } => MAX_DECODE_LANES }
    }
    /// Lanes a launch without `--decode-lanes` requests for `rtx_gpus` coordinator GPUs.
    pub fn default_lanes(self, rtx_gpus: usize) -> u8 {
        match self {
            Self::One { .. } => 1,
            Self::OneOrTwo { single_rtx, dual_rtx } => if rtx_gpus >= 2 { dual_rtx } else { single_rtx },
        }
    }
}

/// Resolve the decode-lane count before layout or any CUDA allocation.
///
/// A family that admits one or two resolves its layout default (`rtx_gpus`),
/// honors an explicit override
/// (`requested`), and resolves one lane at configured concurrency one: an idle
/// second lane would cost memory for nothing.
pub fn resolve(family: &dyn super::Family, requested: Option<u8>, concurrency: u64, rtx_gpus: usize) -> Result<u8, PlanError> {
    let support = family.decode_lanes();
    let default = support.default_lanes(rtx_gpus);
    let lanes = requested.unwrap_or(default);
    if !(1..=MAX_DECODE_LANES).contains(&lanes) {
        return Err(PlanError::InvalidOption { option: "--decode-lanes", reason: "must be 1 or 2".into() });
    }
    let refusal = match (support, lanes) {
        (LaneSupport::One { reason }, 2) => Some(reason),
        _ => None,
    };
    if let Some(reason) = refusal {
        return Err(PlanError::UnsupportedOption { option: "--decode-lanes", value: lanes, family: family.id().into(), reason });
    }
    Ok(match support {
        LaneSupport::OneOrTwo { .. } if concurrency <= 1 => 1,
        _ => lanes,
    })
}

/// The lane a newly admitted request joins among `lanes`: the lighter one,
/// lane zero on a tie (always lane zero with one lane). A drained cohort is
/// then rebalanced to within one request.
pub fn assign(loads: [usize; MAX_DECODE_LANES as usize], lanes: u8) -> usize {
    usize::from(lanes > 1 && loads[1] < loads[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_family_resolves_its_default_and_refuses_what_it_cannot_run() {
        for family in super::super::family::registry() {
            for rtx_gpus in [1usize, 2] {
                let default = family.decode_lanes().default_lanes(rtx_gpus);
                assert_eq!(resolve(*family, None, 16, rtx_gpus).unwrap(), default, "{} {rtx_gpus}xRTX", family.id());
                for lanes in [1, 2] {
                    let supported = match family.decode_lanes() {
                        LaneSupport::One { .. } => lanes == 1,
                        LaneSupport::OneOrTwo { .. } => true,
                    };
                    let result = resolve(*family, Some(lanes), 16, rtx_gpus);
                    if supported { assert_eq!(result.unwrap(), lanes); } else {
                        assert!(matches!(result, Err(PlanError::UnsupportedOption { option: "--decode-lanes", value, .. })
                            if value == lanes), "{} {lanes}", family.id());
                    }
                }
                for lanes in [0, 3, 255] {
                    assert!(matches!(resolve(*family, Some(lanes), 16, rtx_gpus), Err(PlanError::InvalidOption { .. })));
                }
            }
        }
        // V4.1 runs one or two lanes: 1 lane by default on 1×RTX, 2 lanes by default
        // on 2×RTX, explicit 1|2 override on either layout, and 1 at concurrency one
        // (an idle second lane would hold workspaces, graphs and a Spark decode execution for nothing).
        let v41 = &crate::plan::families::deepseek::DEEPSEEK_V41;
        assert_eq!(super::super::Family::decode_lanes(v41), LaneSupport::OneOrTwo { single_rtx: 1, dual_rtx: 2 });
        for (rtx_gpus, requested, concurrency, expected) in [
            (1, None, 16, 1), (1, Some(1), 16, 1), (1, Some(2), 16, 2),
            (1, None, 1, 1), (1, Some(2), 1, 1), (1, None, 32, 1), (1, Some(1), 32, 1),
            (1, None, 33, 1), (1, None, 64, 1), (1, Some(1), 64, 1),
            (2, None, 64, 2), (2, Some(1), 64, 1),
            (2, None, 16, 2), (2, Some(2), 16, 2), (2, Some(1), 16, 1),
            (2, None, 1, 1), (2, Some(2), 1, 1), (2, Some(1), 1, 1), (2, Some(1), 32, 1),
        ] {
            assert_eq!(resolve(v41, requested, concurrency, rtx_gpus).unwrap(), expected,
                "{rtx_gpus}xRTX {requested:?} C{concurrency}");
        }
    }

    #[test]
    fn one_lane_families_record_their_reason() {
        for id in ["kimi_k3", "step5", "qwen4", "qwen3_5", "mimo_v2", "glm5", "glm5_flash", "deepseek_v4"] {
            let family = crate::plan::spark_executions::descriptor(id).unwrap();
            let LaneSupport::One { reason } = family.decode_lanes() else { panic!("{id} is not One") };
            assert!(reason != DRAFTER && reason.contains("#744"), "{id}: {reason}");
            assert!(matches!(resolve(family, Some(2), 16, 2),
                Err(PlanError::UnsupportedOption { reason: refused, .. }) if refused == reason), "{id}");
        }
    }

    /// Every target family runs decode on the shared lane owners (#744): only
    /// drafter descriptors keep the default.
    #[test]
    fn only_drafters_keep_the_default_reason() {
        for family in super::super::family::registry() {
            let drafter = matches!(family.id(), "dflash2" | "dspark");
            let default = family.decode_lanes() == (LaneSupport::One { reason: DRAFTER });
            assert_eq!(default, drafter, "{}", family.id());
        }
    }

    #[test]
    fn assignment_fills_the_lighter_lane_and_ties_go_to_lane_zero() {
        assert_eq!([assign([0, 0], 2), assign([1, 0], 2), assign([1, 1], 2), assign([3, 5], 2)], [0, 1, 0, 0]);
        let mut loads = [0usize; 2];
        for _ in 0..17 {
            loads[assign(loads, 2)] += 1;
            assert!(loads[0].abs_diff(loads[1]) <= 1);
        }
        // One lane admits every request to lane zero.
        let mut loads = [0usize; 2];
        for _ in 0..16 { loads[assign(loads, 1)] += 1; }
        assert_eq!(loads, [16, 0]);
    }
}
