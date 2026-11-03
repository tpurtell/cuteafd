//! Strict parsing and deployment policy for greedy draft acceptance.
use cuteafd_core::{AcceptanceError, AcceptancePolicy, SpeculativeAcceptance, TargetSamplingParams};
use serde_json::Value;

static POLICY: std::sync::OnceLock<AcceptancePolicy> = std::sync::OnceLock::new();

/// Install once at startup, before constructing any model profiles.
pub fn set_acceptance_policy(policy: AcceptancePolicy) -> Result<(), AcceptanceError> {
    AcceptancePolicy::new(policy.allow_relaxed, policy.min_min_p, policy.max_epsilon)?;
    POLICY.set(policy).map_err(|_| AcceptanceError::InvalidParameter("operator policy already installed"))
}

pub(super) fn default_policy() -> AcceptancePolicy { POLICY.get().copied().unwrap_or_default() }

pub(super) fn request_acceptance(body: &Value, sampling: TargetSamplingParams, policy: AcceptancePolicy)
    -> Result<SpeculativeAcceptance, AcceptanceError> {
    let Some(value) = body.get("speculative_acceptance") else { return Ok(SpeculativeAcceptance::Exact); };
    let object = value.as_object().ok_or(AcceptanceError::InvalidObject)?;
    let mode = object.get("mode").ok_or(AcceptanceError::MissingParameter("mode"))?
        .as_str().ok_or(AcceptanceError::InvalidParameter("mode"))?;
    let number = |name: &'static str| -> Result<f32, AcceptanceError> {
        let value = object.get(name).ok_or(AcceptanceError::MissingParameter(name))?
            .as_f64().ok_or(AcceptanceError::InvalidParameter(name))?;
        if !value.is_finite() || (name == "delta" && value <= 0.0)
            || (name != "delta" && !(0.0..=1.0).contains(&value)) {
            return Err(AcceptanceError::InvalidParameter(name));
        }
        Ok(value as f32)
    };
    let (acceptance, fields): (_, &[&str]) = match mode {
        "exact" => (SpeculativeAcceptance::Exact, &["mode"]),
        "min_p" => (SpeculativeAcceptance::MinP { min_p: number("min_p")? }, &["mode", "min_p"]),
        "typical" => (SpeculativeAcceptance::Typical { epsilon: number("epsilon")?, delta: number("delta")? }, &["mode", "epsilon", "delta"]),
        _ => return Err(AcceptanceError::UnknownMode),
    };
    if object.keys().any(|field| !fields.contains(&field.as_str())) {
        return Err(AcceptanceError::UnexpectedParameter);
    }
    policy.validate(acceptance, sampling)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(value: Value) -> Result<SpeculativeAcceptance, AcceptanceError> {
        request_acceptance(&json!({"speculative_acceptance": value}), TargetSamplingParams::greedy(),
            AcceptancePolicy::new(true, 0.5, 0.2).unwrap())
    }

    #[test]
    fn missing_unknown_invalid_and_policy_errors_are_typed() {
        assert_eq!(parse(json!(null)), Err(AcceptanceError::InvalidObject));
        assert_eq!(parse(json!({})), Err(AcceptanceError::MissingParameter("mode")));
        assert_eq!(parse(json!({"mode": 1})), Err(AcceptanceError::InvalidParameter("mode")));
        assert_eq!(parse(json!({"mode":"other"})), Err(AcceptanceError::UnknownMode));
        assert_eq!(parse(json!({"mode":"min_p"})), Err(AcceptanceError::MissingParameter("min_p")));
        assert_eq!(parse(json!({"mode":"min_p","min_p":"0.5"})), Err(AcceptanceError::InvalidParameter("min_p")));
        assert_eq!(parse(json!({"mode":"min_p","min_p":1.1})), Err(AcceptanceError::InvalidParameter("min_p")));
        assert_eq!(parse(json!({"mode":"min_p","min_p":0.49})), Err(AcceptanceError::MinPBelowPolicy));
        assert_eq!(parse(json!({"mode":"typical","epsilon":0.1})), Err(AcceptanceError::MissingParameter("delta")));
        assert_eq!(parse(json!({"mode":"typical","epsilon":0.3,"delta":1.0})), Err(AcceptanceError::EpsilonAbovePolicy));
        assert_eq!(parse(json!({"mode":"typical","epsilon":0.1,"delta":0})), Err(AcceptanceError::InvalidParameter("delta")));
        assert_eq!(parse(json!({"mode":"exact","min_p":0.5})), Err(AcceptanceError::UnexpectedParameter));
        assert_eq!(parse(json!({"mode":"min_p","min_p":0.5,"epsilon":0.1})), Err(AcceptanceError::UnexpectedParameter));
        assert_eq!(parse(json!({"mode":"min_p","min_p":0.5})), Ok(SpeculativeAcceptance::MinP { min_p: 0.5 }));
        assert_eq!(parse(json!({"mode":"typical","epsilon":0.2,"delta":2})), Ok(SpeculativeAcceptance::Typical { epsilon: 0.2, delta: 2.0 }));
    }
}

#[cfg(test)]
mod path_tests {
    use super::*;
    use crate::openai::{build, ModelEncoding, ModelProfile, NativeLimits};
    use serde_json::json;

    #[test]
    fn unsupported_runtime_family_and_constraint_paths_reject_before_admission() {
        let request = json!({"messages":[{"role":"user","content":"hi"}],
            "speculative_acceptance":{"mode":"min_p","min_p":0.8}});
        let mut profile = ModelProfile::default();
        profile.relaxed_acceptance_supported = false;
        let error = build(&profile, NativeLimits::default(), &Default::default(), request.clone()).err().unwrap();
        assert!(error.message.contains("unsupported"));
        assert!(error.message.contains("V4.1"));
        assert!(!ModelProfile::new("v4", ModelEncoding::DeepseekV4).relaxed_acceptance_supported);
        let mut constrained = request;
        constrained["response_format"] = json!({"type":"json_object"});
        let error = build(&ModelProfile::default(), NativeLimits::default(), &Default::default(), constrained).err().unwrap();
        assert!(error.message.contains("constrained decoding"), "{}", error.message);
    }
}
