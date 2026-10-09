//! Drafter-specific selector priors and deployment-local online refinement.
use super::draft_policy::{Calibration, DraftHistory};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;

pub(crate) const FEATURE_ORDER: [&str; 6] = ["history_logit", "log1p_margin", "candidate_probability_logit",
    "candidate_entropy", "log1p_unary_rank", "position_over_7"];

/// Six logistic feature coefficients plus an intercept and normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SelectorFit {
    pub schema: String,
    pub key: String,
    pub feature_order: [String; 6],
    pub mean: [f64; 6],
    pub scale: [f64; 6],
    pub intercept: f64,
    pub coefficients: [f64; 6],
    #[serde(default)]
    pub training: serde_json::Value,
}

impl Default for SelectorFit {
    fn default() -> Self {
        // glmrt's 32-request fixed-K7 GLM-5.3 K4 fit (6d34604f).
        Self {
            schema: "cuteafd-draft-confidence-v1".into(), key: "generic".into(),
            feature_order: FEATURE_ORDER.map(String::from),
            mean: [1.0337757155740168, 1.5042334365117003, 4.105818737585014, 0.5404882231666059,
                0.18709320564567553, 0.36878205711857714],
            scale: [0.6869594087005682, 0.8183292270430992, 3.7913657744791154, 0.6412433617632998,
                0.4768998480274627, 0.23888156204918426],
            intercept: 1.3954686667753908,
            coefficients: [0.05344836366609415, -0.3418558408836222, 1.7687177070514315,
                -0.5006880248156605, -0.21189320574726964, -0.09773211967803144],
            training: serde_json::Value::Null,
        }
    }
}

fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-4, 1.0 - 1e-4);
    (p / (1.0 - p)).ln()
}

pub(crate) fn features(prior: f64, position: usize, [margin, probability, entropy, rank]: [f32; 4])
    -> Option<[f64; 6]> {
    let valid = prior.is_finite() && (0.0..=1.0).contains(&prior) && (1..=7).contains(&position)
        && [margin, probability, entropy, rank].iter().all(|x| x.is_finite())
        && margin >= 0.0 && probability > 0.0 && probability <= 1.0
        && (-1e-5..=2.80).contains(&entropy) && (0.0..16.0).contains(&rank) && rank.fract() == 0.0;
    valid.then(|| [logit(prior), f64::from(margin).ln_1p(), logit(f64::from(probability)),
        f64::from(entropy), f64::from(rank).ln_1p(), position as f64 / 7.0])
}

impl SelectorFit {
    fn validate(&self, key: &str) -> Result<()> {
        ensure!(self.schema == "cuteafd-draft-confidence-v1", "unknown draft confidence schema");
        ensure!(self.key == key, "draft confidence key {} does not match {key}", self.key);
        ensure!(self.feature_order == FEATURE_ORDER.map(String::from), "draft confidence feature order differs");
        ensure!(self.mean.iter().chain(&self.coefficients).chain([&self.intercept]).all(|v| v.is_finite())
            && self.scale.iter().all(|s| s.is_finite() && *s > 0.0), "invalid draft confidence coefficients or normalization");
        Ok(())
    }

    pub fn confidence(&self, history: &[f64], selector: &[[f32; 4]]) -> Option<Vec<f64>> {
        if history.is_empty() || history.len() > 7 || selector.len() < history.len() { return None; }
        history.iter().zip(selector).enumerate().map(|(index, (&prior, &selector))| {
            let x = features(prior, index + 1, selector)?;
            let z = (self.intercept + (0..6).map(|i| self.coefficients[i] * (x[i] - self.mean[i])
                / self.scale[i]).sum::<f64>()).clamp(-40.0, 40.0);
            Some(1.0 / (1.0 + (-z).exp()))
        }).collect()
    }

    pub fn load(directory: &Path, key: &str) -> Result<Self> {
        let components: Vec<_> = key.split('/').collect();
        ensure!(components.len() == 3 && components.iter().all(|part| !part.is_empty()
            && part.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')),
            "draft confidence key must be family/drafter/numerics");
        let path = directory.join(format!("draft-confidence.{}.json", components[2]));
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                tracing::info!(key, path = %path.display(), "draft confidence key absent; using generic GLM-5.3 selector fit");
                return Ok(Self::default());
            }
            Err(error) => return Err(error).with_context(|| format!("read draft confidence {}", path.display())),
        };
        let fit = match serde_json::from_slice::<Self>(&bytes).map_err(anyhow::Error::from)
            .and_then(|fit| { fit.validate(key)?; Ok(fit) }) {
            Ok(fit) => fit,
            Err(error) => {
                tracing::warn!(key, path = %path.display(), %error, "invalid/mismatched draft confidence; using generic GLM-5.3 selector fit");
                return Ok(Self::default());
            }
        };
        tracing::info!(key, path = %path.display(), "loaded drafter confidence prior");
        Ok(fit)
    }
}

/// The keyed logistic fit is the prior; the existing forgotten, slope-clamped
/// online correction absorbs deployment/target-quant shifts without per-quant files.
pub(crate) struct ConfidencePolicy {
    pub key: String,
    selector: Option<SelectorFit>,
    online: Calibration,
    enabled: bool,
    trace: Option<std::io::BufWriter<std::fs::File>>,
}

impl ConfidencePolicy {
    pub fn load(directory: &Path, key: String) -> Result<Self> {
        let enabled = super::draft_policy::enabled("CUTEAFD_DRAFT_CONFIDENCE");
        let fit = if enabled { SelectorFit::load(directory, &key)? } else { SelectorFit::default() };
        let selector = (enabled || !key.starts_with("mimo_v2/")).then_some(fit);
        Self::new(key, selector)
    }

    /// Native MTP has history rates but no top-16 selector features.
    pub fn history(key: String) -> Result<Self> { Self::new(key, None) }

    fn new(key: String, selector: Option<SelectorFit>) -> Result<Self> {
        let trace = super::draft_policy::speculation_trace_path("CUTEAFD_DRAFT_CONFIDENCE_TRACE")
            .map(|(_, path)| std::fs::OpenOptions::new().create(true).append(true).open(path)
                .map(std::io::BufWriter::new)).transpose()?;
        let enabled = super::draft_policy::enabled("CUTEAFD_DRAFT_CONFIDENCE");
        // Enabled selector loading already logs the selected key or fallback once.
        if !enabled || selector.is_none() {
            tracing::info!(key, enabled, selector_prior = selector.as_ref().map(|fit| fit.key.as_str()),
                "draft confidence policy initialized");
        }
        Ok(Self { key, selector, online: Calibration::default(), enabled, trace })
    }

    /// Unrefined rates are retained for a causally correct online observation.
    pub fn prior(&self, history: &DraftHistory, selector: Option<&[[f32; 4]]>, width: usize) -> Vec<f64> {
        let rates = history.conditional(width);
        self.selector.as_ref().zip(selector).and_then(|(fit, selector)| fit.confidence(&rates, selector)).unwrap_or(rates)
    }

    pub fn apply(&self, rates: &[f64]) -> Vec<f64> {
        if self.enabled { rates.iter().map(|&rate| self.online.apply(rate)).collect() } else { rates.to_vec() }
    }

    pub fn observe(&mut self, history: &DraftHistory, selector: Option<&[[f32; 4]]>, prior: &[f64],
        proposed: usize, accepted: usize, finished: bool, request: u64) {
        // A terminal token/client cutoff censors the next position rather than rejecting it.
        let observed = proposed.min(accepted.saturating_add(usize::from(!finished)));
        if self.enabled {
            for (i, &rate) in prior.iter().enumerate().take(observed) { self.online.observe(rate, i < accepted); }
        }
        if let (Some(trace), Some(selector)) = (&mut self.trace, selector) {
            let record = serde_json::json!({"type": "draft_confidence", "key": self.key, "request": request,
                "history": history.conditional(selector.len()), "features": selector, "prior": prior,
                "proposed": proposed, "accepted": accepted, "observed": observed,
                "online": self.online.fitted()});
            if let Err(error) = writeln!(trace, "{record}").and_then(|_| trace.flush()) {
                tracing::warn!(%error, "draft confidence trace failed");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyed_fit_validates_numerics_and_normalization() {
        let mut fit = SelectorFit { key: "glm5_flash/dflash2/fp8-w8a8-r1".into(), ..SelectorFit::default() };
        fit.validate(&fit.key).unwrap();
        assert!(fit.validate("glm5_flash/dflash2/bf16-r1").is_err());
        fit.scale[2] = 0.0;
        assert!(fit.validate(&fit.key).is_err());
        fit.scale[2] = 1.0;
        fit.coefficients[1] = f64::NAN;
        assert!(fit.validate(&fit.key).is_err());
    }

    #[test]
    fn keyed_loader_uses_matching_fit_and_falls_back_once_on_invalid_data() {
        let directory = tempfile::tempdir().unwrap();
        let key = "glm5_flash/dflash2/fp8-w8a8-r1";
        let path = directory.path().join("draft-confidence.fp8-w8a8-r1.json");
        assert_eq!(SelectorFit::load(directory.path(), key).unwrap().key, "generic");
        let mut fit = SelectorFit { key: key.into(), intercept: 0.25, ..SelectorFit::default() };
        std::fs::write(&path, serde_json::to_vec(&fit).unwrap()).unwrap();
        assert_eq!(SelectorFit::load(directory.path(), key).unwrap().intercept, 0.25);
        fit.key = "mimo_v2/dflash/fp8-w8a8-r1".into();
        std::fs::write(&path, serde_json::to_vec(&fit).unwrap()).unwrap();
        assert_eq!(SelectorFit::load(directory.path(), key).unwrap().key, "generic");
        fit.key = key.into();
        fit.scale[0] = 0.0;
        std::fs::write(&path, serde_json::to_vec(&fit).unwrap()).unwrap();
        assert_eq!(SelectorFit::load(directory.path(), key).unwrap().key, "generic");
        std::fs::write(&path, b"{malformed").unwrap();
        assert_eq!(SelectorFit::load(directory.path(), key).unwrap().key, "generic");
        assert!(SelectorFit::load(directory.path(), "../dflash2/fp8").is_err());
    }

    #[test]
    fn features_match_replay_tool_and_reject_invalid_selector_rows() {
        let x = features(0.75, 7, [1.0, 0.5, 1.0, 2.0]).unwrap();
        let expected = [3.0f64.ln(), 2.0f64.ln(), 0.0, 1.0, 3.0f64.ln(), 1.0];
        for (actual, expected) in x.into_iter().zip(expected) { assert!((actual - expected).abs() < 1e-12); }
        assert!(features(0.75, 8, [1.0, 0.5, 1.0, 2.0]).is_none());
        assert!(features(0.75, 1, [1.0, 0.0, 1.0, 2.0]).is_none());
        assert!(features(0.75, 1, [1.0, 0.5, 1.0, 2.5]).is_none());
        assert!(features(f64::NAN, 1, [1.0, 0.5, 1.0, 2.0]).is_none());
    }

    #[test]
    fn disabled_policy_keeps_frozen_glm_and_history_only_mimo_baselines() {
        let history = DraftHistory::default();
        let selector = [[1.0, 0.5, 1.0, 2.0]; 7];
        let mut policy = ConfidencePolicy { key: "glm5/dflash2/bf16-r1".into(), selector: Some(SelectorFit::default()),
            online: Calibration::default(), enabled: false, trace: None };
        let prior = policy.prior(&history, Some(&selector), 7);
        assert_eq!(prior, SelectorFit::default().confidence(&history.conditional(7), &selector).unwrap());
        policy.observe(&history, Some(&selector), &prior, 7, 0, false, 0);
        assert_eq!(policy.apply(&prior), prior);
        policy.selector = None;
        assert_eq!(policy.prior(&history, Some(&selector), 7), history.conditional(7));
    }

    #[test]
    fn online_state_is_per_key_and_terminal_outcomes_are_censored() {
        let mut policy = ConfidencePolicy { key: "mimo_v2/dflash/bf16-r1".into(), selector: Some(SelectorFit::default()),
            online: Calibration::default(), enabled: true, trace: None };
        let history = DraftHistory::default();
        for _ in 0..200 { policy.observe(&history, None, &[0.8; 7], 7, 0, false, 0); }
        assert!(policy.apply(&[0.8])[0] < 0.1);
        let other = Calibration::default();
        assert_eq!(other.apply(0.8), 0.8);
        let before = policy.online.fitted();
        policy.observe(&history, None, &[0.8; 7], 7, 0, true, 0);
        assert_eq!(policy.online.fitted(), before);
    }
}
