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

/// What a family's executor supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneSupport {
    /// One decode batch only.
    One,
    /// One or two; `default` until a hardware A/B qualifies another default.
    /// A second lane at configured concurrency one costs memory for nothing.
    OneOrTwo { default: u8 },
    /// One or two, both at every concurrency: the executor allocates both
    /// lanes' owners either way, so one lane only routes every request to
    /// lane zero (V4.1, whose default pair predates the shared contract).
    Routed { default: u8 },
}

/// The family's lane support. A family moves from `One` when its port onto
/// the shared lane owners lands (with its planner charges).
pub fn support(family: &str) -> LaneSupport {
    match family {
        "deepseek_v41" => LaneSupport::Routed { default: 2 },
        _ => LaneSupport::One,
    }
}

/// Resolve the decode-lane count before layout or any CUDA allocation.
///
/// A family that admits one or two resolves one lane at configured
/// concurrency one: an idle second lane would cost memory for nothing.
pub fn resolve(family: &str, requested: Option<u8>, concurrency: u64) -> Result<u8, PlanError> {
    let support = support(family);
    let lanes = requested.unwrap_or(match support {
        LaneSupport::One => 1,
        LaneSupport::OneOrTwo { default } | LaneSupport::Routed { default } => default,
    });
    if !(1..=MAX_DECODE_LANES).contains(&lanes) {
        return Err(PlanError::InvalidOption { option: "--decode-lanes", reason: "must be 1 or 2".into() });
    }
    let refusal = match (support, lanes) {
        (LaneSupport::One, 2) => Some("this family has no independent second decode lane"),
        _ => None,
    };
    if let Some(reason) = refusal {
        return Err(PlanError::UnsupportedOption { option: "--decode-lanes", value: lanes, family: family.into(), reason });
    }
    Ok(match support {
        LaneSupport::OneOrTwo { .. } if concurrency <= 1 => 1,
        _ => lanes,
    })
}

/// The lane a newly admitted request joins: the lighter one, lane zero on a
/// tie. A drained cohort is then rebalanced to within one request. With one
/// lane every request joins lane zero.
pub fn assign(lanes: u8, loads: [usize; MAX_DECODE_LANES as usize]) -> usize {
    if lanes == 1 { 0 } else { usize::from(loads[1] < loads[0]) }
}

/// Independent dSpark draft chains a coordinator owns: one per decode lane,
/// and one at configured concurrency one. Each chain holds
/// `requests.div_ceil(chains)` requests; planner and engine size from here.
pub fn draft_chains(requests: u64, lanes: u8) -> u64 {
    if requests == 1 { 1 } else { u64::from(lanes) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_family_resolves_its_default_and_refuses_what_it_cannot_run() {
        for (family, default) in [("deepseek_v41", 2), ("mimo_v2", 1), ("glm5", 1), ("glm5_flash", 1),
            ("deepseek_v4", 1), ("qwen4", 1), ("qwen3_5", 1), ("kimi_k3", 1), ("step5", 1)] {
            assert_eq!(resolve(family, None, 16).unwrap(), default, "{family}");
            for lanes in [1, 2] {
                let supported = match support(family) {
                    LaneSupport::One => lanes == 1,
                    LaneSupport::OneOrTwo { .. } | LaneSupport::Routed { .. } => true,
                };
                let result = resolve(family, Some(lanes), 16);
                if supported { assert_eq!(result.unwrap(), lanes); } else {
                    assert!(matches!(result, Err(PlanError::UnsupportedOption { option: "--decode-lanes", value, .. })
                        if value == lanes), "{family} {lanes}");
                }
            }
            for lanes in [0, 3, 255] {
                assert!(matches!(resolve(family, Some(lanes), 16), Err(PlanError::InvalidOption { .. })));
            }
        }
        // A routed pair keeps the requested count at concurrency one; an
        // allocated pair drops to one lane there.
        assert_eq!(resolve("deepseek_v41", Some(2), 1).unwrap(), 2);
        assert_eq!(resolve("deepseek_v41", None, 1).unwrap(), 2);
        assert_eq!(resolve("deepseek_v41", Some(1), 16).unwrap(), 1);
    }

    #[test]
    fn draft_chains_follow_decode_lanes_except_at_concurrency_one() {
        assert_eq!([draft_chains(1, 2), draft_chains(2, 2), draft_chains(16, 2)], [1, 2, 2]);
        assert_eq!([draft_chains(1, 1), draft_chains(16, 1)], [1, 1]);
    }

    #[test]
    fn assignment_fills_the_lighter_lane_and_ties_go_to_lane_zero() {
        assert_eq!([assign(2, [0, 0]), assign(2, [1, 0]), assign(2, [1, 1]), assign(2, [3, 5])], [0, 1, 0, 0]);
        let mut loads = [0usize; 2];
        for _ in 0..17 {
            loads[assign(2, loads)] += 1;
            assert!(loads[0].abs_diff(loads[1]) <= 1);
        }
        assert!([[0, 0], [1, 0], [5, 3], [0, 4]].into_iter().all(|loads| assign(1, loads) == 0));
    }
}
