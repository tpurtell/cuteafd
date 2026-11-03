//! Host-side draft acceptance, independent of target sampling.
use crate::TargetSamplingParams;

/// Target distribution statistics at a drafted position, after any logit masks.
/// Probabilities and entropy use the untempered target distribution.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AcceptanceStats {
    pub draft_probability: f32,
    pub argmax_probability: f32,
    pub entropy: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
pub enum SpeculativeAcceptance {
    #[default]
    Exact,
    MinP { min_p: f32 },
    Typical { epsilon: f32, delta: f32 },
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AcceptanceError {
    #[error("speculative_acceptance must be an object")]
    InvalidObject,
    #[error("relaxed speculative acceptance unsupported: {0}; supported: single-GPU V4.1 device token selection without constraints")]
    UnsupportedPath(&'static str),
    #[error("unknown speculative_acceptance mode")]
    UnknownMode,
    #[error("missing speculative_acceptance parameter: {0}")]
    MissingParameter(&'static str),
    #[error("invalid speculative_acceptance parameter: {0}")]
    InvalidParameter(&'static str),
    #[error("unexpected speculative_acceptance parameter")]
    UnexpectedParameter,
    #[error("relaxed speculative acceptance requires greedy target sampling")]
    RelaxedSampling,
    #[error("relaxed speculative acceptance is disabled by operator policy")]
    Disabled,
    #[error("speculative_acceptance.min_p is below the operator minimum")]
    MinPBelowPolicy,
    #[error("speculative_acceptance.epsilon exceeds the operator maximum")]
    EpsilonAbovePolicy,
}

impl SpeculativeAcceptance {
    pub fn is_exact(self) -> bool { matches!(self, Self::Exact) }

    pub fn validate(self, sampling: TargetSamplingParams) -> Result<Self, AcceptanceError> {
        let unit = |value: f32| value.is_finite() && (0.0..=1.0).contains(&value);
        match self {
            Self::MinP { min_p } if !unit(min_p) => return Err(AcceptanceError::InvalidParameter("min_p")),
            Self::Typical { epsilon, .. } if !unit(epsilon) => return Err(AcceptanceError::InvalidParameter("epsilon")),
            Self::Typical { delta, .. } if !delta.is_finite() || delta <= 0.0 => return Err(AcceptanceError::InvalidParameter("delta")),
            _ => {}
        }
        if !self.is_exact() && !sampling.is_greedy() {
            return Err(AcceptanceError::RelaxedSampling);
        }
        Ok(self)
    }

    /// Exact mode requires token equality instead of distribution statistics.
    /// The verifier owns that comparison; this predicate only handles relaxed modes.
    pub fn accepts_draft(self, stats: AcceptanceStats) -> bool {
        match self {
            Self::Exact => false,
            Self::MinP { min_p } => stats.draft_probability >= min_p * stats.argmax_probability,
            Self::Typical { epsilon, delta } => stats.draft_probability > epsilon.min(delta * (-stats.entropy).exp()),
        }
    }
}

/// Deployment bounds never clamp requests silently: a violation is a typed error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AcceptancePolicy {
    pub allow_relaxed: bool,
    pub min_min_p: f32,
    pub max_epsilon: f32,
}

impl Default for AcceptancePolicy {
    fn default() -> Self { Self { allow_relaxed: true, min_min_p: 0.0, max_epsilon: 1.0 } }
}

impl AcceptancePolicy {
    pub fn new(allow_relaxed: bool, min_min_p: f32, max_epsilon: f32) -> Result<Self, AcceptanceError> {
        for (name, value) in [("min_min_p", min_min_p), ("max_epsilon", max_epsilon)] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(AcceptanceError::InvalidParameter(name));
            }
        }
        Ok(Self { allow_relaxed, min_min_p, max_epsilon })
    }

    pub fn validate(self, acceptance: SpeculativeAcceptance, sampling: TargetSamplingParams)
        -> Result<SpeculativeAcceptance, AcceptanceError> {
        Self::new(self.allow_relaxed, self.min_min_p, self.max_epsilon)?;
        acceptance.validate(sampling)?;
        if acceptance.is_exact() { return Ok(acceptance); }
        if !self.allow_relaxed { return Err(AcceptanceError::Disabled); }
        match acceptance {
            SpeculativeAcceptance::MinP { min_p } if min_p < self.min_min_p => Err(AcceptanceError::MinPBelowPolicy),
            SpeculativeAcceptance::Typical { epsilon, .. } if epsilon > self.max_epsilon => Err(AcceptanceError::EpsilonAbovePolicy),
            _ => Ok(acceptance),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(p: f32) -> AcceptanceStats {
        AcceptanceStats { draft_probability: p, argmax_probability: 0.5, entropy: 0.0 }
    }

    #[test]
    fn boundary_rules_are_inclusive_min_p_and_strict_typical() {
        let min_p = SpeculativeAcceptance::MinP { min_p: 0.5 };
        assert!(min_p.accepts_draft(stats(0.25)));
        assert!(!min_p.accepts_draft(stats(f32::from_bits(0.25f32.to_bits() - 1))));
        let typical = SpeculativeAcceptance::Typical { epsilon: 0.25, delta: 1.0 };
        assert!(!typical.accepts_draft(stats(0.25)));
        assert!(typical.accepts_draft(stats(f32::from_bits(0.25f32.to_bits() + 1))));
        let high_entropy = AcceptanceStats { entropy: 10.0, ..stats(0.01) };
        assert!(typical.accepts_draft(high_entropy));
        assert!(!typical.accepts_draft(stats(0.01)));
        assert!(SpeculativeAcceptance::MinP { min_p: 0.0 }.accepts_draft(stats(0.0)));
    }

    #[test]
    fn typed_validation_and_policy_caps() {
        let greedy = TargetSamplingParams::greedy();
        let sampled = TargetSamplingParams::new(0.7, 1.0, None, 0.0, 1).unwrap();
        let policy = AcceptancePolicy::new(true, 0.5, 0.2).unwrap();
        assert_eq!(AcceptancePolicy::default().validate(SpeculativeAcceptance::Exact, sampled), Ok(SpeculativeAcceptance::Exact));
        assert_eq!(AcceptancePolicy::new(false, 0.0, 1.0).unwrap().validate(SpeculativeAcceptance::MinP { min_p: 1.0 }, greedy), Err(AcceptanceError::Disabled));
        assert_eq!(policy.validate(SpeculativeAcceptance::MinP { min_p: 0.5 }, sampled), Err(AcceptanceError::RelaxedSampling));
        assert!(policy.validate(SpeculativeAcceptance::MinP { min_p: 0.5 }, greedy).is_ok());
        assert_eq!(policy.validate(SpeculativeAcceptance::MinP { min_p: 0.49 }, greedy), Err(AcceptanceError::MinPBelowPolicy));
        assert!(policy.validate(SpeculativeAcceptance::Typical { epsilon: 0.2, delta: 2.0 }, greedy).is_ok());
        assert_eq!(policy.validate(SpeculativeAcceptance::Typical { epsilon: 0.21, delta: 2.0 }, greedy), Err(AcceptanceError::EpsilonAbovePolicy));
        for value in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
            assert_eq!(SpeculativeAcceptance::MinP { min_p: value }.validate(greedy), Err(AcceptanceError::InvalidParameter("min_p")));
            assert_eq!(SpeculativeAcceptance::Typical { epsilon: value, delta: 1.0 }.validate(greedy), Err(AcceptanceError::InvalidParameter("epsilon")));
            assert!(AcceptancePolicy::new(true, value, 1.0).is_err());
            assert!(AcceptancePolicy::new(true, 0.0, value).is_err());
        }
        for delta in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_eq!(SpeculativeAcceptance::Typical { epsilon: 0.1, delta }.validate(greedy), Err(AcceptanceError::InvalidParameter("delta")));
        }
    }
}
