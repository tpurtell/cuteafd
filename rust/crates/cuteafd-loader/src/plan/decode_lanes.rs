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
    /// One or two; `default` until a hardware A/B qualifies another default.
    OneOrTwo { default: u8 },
    /// A fixed pair (the executor has no one-lane path).
    FixedTwo,
}

impl LaneSupport {
    /// Most lanes the executor can run (each owns a Spark decode execution).
    pub fn max_lanes(self) -> u8 {
        match self { Self::One { .. } => 1, Self::OneOrTwo { .. } | Self::FixedTwo => MAX_DECODE_LANES }
    }
}

/// Resolve the decode-lane count before layout or any CUDA allocation.
///
/// A family that admits one or two resolves one lane at configured
/// concurrency one: an idle second lane would cost memory for nothing.
pub fn resolve(family: &dyn super::Family, requested: Option<u8>, concurrency: u64) -> Result<u8, PlanError> {
    let support = family.decode_lanes();
    let lanes = requested.unwrap_or(match support {
        LaneSupport::One { .. } => 1,
        LaneSupport::OneOrTwo { default } => default,
        LaneSupport::FixedTwo => 2,
    });
    if !(1..=MAX_DECODE_LANES).contains(&lanes) {
        return Err(PlanError::InvalidOption { option: "--decode-lanes", reason: "must be 1 or 2".into() });
    }
    let refusal = match (support, lanes) {
        (LaneSupport::FixedTwo, 1) => Some("the executor has a fixed pair of decode lanes"),
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

/// The lane a newly admitted request joins: the lighter one, lane zero on a
/// tie. A drained cohort is then rebalanced to within one request.
pub fn assign(loads: [usize; MAX_DECODE_LANES as usize]) -> usize {
    usize::from(loads[1] < loads[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_family_resolves_its_default_and_refuses_what_it_cannot_run() {
        for family in super::super::family::registry() {
            let default = match family.decode_lanes() {
                LaneSupport::One { .. } => 1, LaneSupport::OneOrTwo { default } => default, LaneSupport::FixedTwo => 2 };
            assert_eq!(resolve(*family, None, 16).unwrap(), default, "{}", family.id());
            for lanes in [1, 2] {
                let supported = match family.decode_lanes() {
                    LaneSupport::One { .. } => lanes == 1,
                    LaneSupport::OneOrTwo { .. } => true,
                    LaneSupport::FixedTwo => lanes == 2,
                };
                let result = resolve(*family, Some(lanes), 16);
                if supported { assert_eq!(result.unwrap(), lanes); } else {
                    assert!(matches!(result, Err(PlanError::UnsupportedOption { option: "--decode-lanes", value, .. })
                        if value == lanes), "{} {lanes}", family.id());
                }
            }
            for lanes in [0, 3, 255] {
                assert!(matches!(resolve(*family, Some(lanes), 16), Err(PlanError::InvalidOption { .. })));
            }
        }
        // The fixed pair holds at concurrency one.
        let v41 = &crate::plan::families::deepseek::DEEPSEEK_V41;
        assert_eq!(resolve(v41, Some(2), 1).unwrap(), 2);
        assert_eq!(super::super::Family::decode_lanes(v41), LaneSupport::FixedTwo);
    }

    #[test]
    fn one_lane_families_record_their_reason() {
        for id in ["kimi_k3", "step5", "qwen4", "qwen3_5", "mimo_v2", "glm5", "deepseek_v4"] {
            let family = crate::plan::spark_executions::descriptor(id).unwrap();
            let LaneSupport::One { reason } = family.decode_lanes() else { panic!("{id} is not One") };
            assert!(reason != DRAFTER && reason.contains("#744"), "{id}: {reason}");
            assert!(matches!(resolve(family, Some(2), 16),
                Err(PlanError::UnsupportedOption { reason: refused, .. }) if refused == reason), "{id}");
        }
        let glmf = &crate::plan::families::glm::GLM5_FLASH;
        assert_eq!(super::super::Family::decode_lanes(glmf), LaneSupport::OneOrTwo { default: 1 });
        assert_eq!(resolve(glmf, None, 16).unwrap(), 1);
        assert_eq!(resolve(glmf, Some(2), 16).unwrap(), 2);
        assert_eq!(resolve(glmf, Some(2), 1).unwrap(), 1);
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
        assert_eq!([assign([0, 0]), assign([1, 0]), assign([1, 1]), assign([3, 5])], [0, 1, 0, 0]);
        let mut loads = [0usize; 2];
        for _ in 0..17 {
            loads[assign(loads)] += 1;
            assert!(loads[0].abs_diff(loads[1]) <= 1);
        }
    }
}
