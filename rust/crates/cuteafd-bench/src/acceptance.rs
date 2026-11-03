//! Greedy verifier policies and exact-arm continuations on sealed fidelity prompts.
use crate::{client::Client, report::{Check, CheckStatus, ServerInfo}, runner::RunRequest};
use anyhow::{ensure, Context, Result};
use cuteafd_api::openai::probe::ProbeSpec;
use cuteafd_core::SpeculativeAcceptance;
use serde::{Deserialize, Serialize};
use serde_json::json;

const TOKENS: usize = 128;

fn parameter(value: f32) -> String {
    let plain = value.to_string();
    let scientific = format!("{value:e}");
    if scientific.len() < plain.len() { scientific } else { plain }
}

pub fn label(policy: SpeculativeAcceptance) -> String {
    match policy {
        SpeculativeAcceptance::Exact => "exact".into(),
        SpeculativeAcceptance::MinP { min_p } => format!("min_p {}", parameter(min_p)),
        SpeculativeAcceptance::Typical { epsilon, delta } => format!("typical epsilon {} delta {}", parameter(epsilon), parameter(delta)),
    }
}

pub fn from_cli(mode: &str, min_p: Option<f32>, epsilon: Option<f32>, delta: Option<f32>)
    -> std::result::Result<SpeculativeAcceptance, String> {
    let policy = match (mode, min_p, epsilon, delta) {
        ("exact", None, None, None) => SpeculativeAcceptance::Exact,
        ("min_p", Some(min_p), None, None) => SpeculativeAcceptance::MinP { min_p },
        ("typical", None, Some(epsilon), Some(delta)) => SpeculativeAcceptance::Typical { epsilon, delta },
        _ => return Err("exact takes no parameters; min_p needs --acceptance-min-p; typical needs --acceptance-epsilon and --acceptance-delta".into()),
    };
    validate_policy(policy)?;
    Ok(policy)
}

fn validate_policy(policy: SpeculativeAcceptance) -> std::result::Result<(), String> {
    policy.validate(cuteafd_core::TargetSamplingParams::greedy()).map(|_| ()).map_err(|error| error.to_string())
}

pub(crate) fn validate(request: &RunRequest) -> std::result::Result<(), String> {
    validate_policy(request.speculative_acceptance)?;
    if request.save_generated_reference && !matches!(request.speculative_acceptance, SpeculativeAcceptance::Exact) {
        return Err("only the exact arm can save generated references".into());
    }
    if request.save_generated_reference && request.generated_reference.is_some() {
        return Err("choose save_generated_reference or generated_reference".into());
    }
    if (request.save_generated_reference || request.generated_reference.is_some()) && request.baseline == Some(false) {
        return Err("generated references require the baseline report".into());
    }
    if let Some(saved) = &request.generated_reference {
        validate_reference_session(saved).map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn validate_reference_session(saved: &GeneratedReference) -> Result<()> {
    ensure!(saved.schema == "cuteafd.bench.generated-reference/1" && saved.server_session == session(),
        "exact and relaxed arms must target the same running server");
    ensure!(matches!(saved.speculative_acceptance, SpeculativeAcceptance::Exact), "reference is not an exact arm");
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedReference {
    pub schema: String,
    /// A process-local identity: all arms must run on the same live server.
    pub server_session: String,
    pub exact_fingerprint: String,
    pub reference_sha256: String,
    pub max_tokens: usize,
    pub speculative_acceptance: SpeculativeAcceptance,
    pub cases: Vec<GeneratedCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedCase {
    pub id: String,
    pub prompt_ids: Vec<u32>,
    pub generated: Vec<u32>,
}

fn session() -> &'static str {
    static SESSION: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    SESSION.get_or_init(|| uuid::Uuid::new_v4().to_string())
}

#[derive(Debug, Serialize)]
struct Match {
    id: String,
    matched: usize,
    positions: usize,
    /// Zero-based generated-token position; a length mismatch also diverges.
    first_divergence: Option<usize>,
    exact_tokens: usize,
    candidate_tokens: usize,
}

fn compare(id: &str, exact: &[u32], candidate: &[u32]) -> Match {
    let positions = exact.len().max(candidate.len());
    Match { id: id.into(), matched: exact.iter().zip(candidate).filter(|(a, b)| a == b).count(), positions,
        first_divergence: (0..positions).find(|&i| exact.get(i) != candidate.get(i)),
        exact_tokens: exact.len(), candidate_tokens: candidate.len() }
}

pub(crate) fn measure(client: &Client, info: &ServerInfo, saved: Option<&GeneratedReference>) -> Result<Check> {
    let selection = crate::fidelity_match::resolve(&info.checkpoint(), info.configuration.snapshot.as_deref().map(std::path::Path::new))?;
    let cache = std::env::var_os("CUTEAFD_FIDELITY_CACHE").map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/cuteafd/fidelity"));
    let (reference, digest, _) = crate::fidelity_dataset::download(&ureq::agent(), &cache,
        crate::fidelity_dataset::REPOSITORY, &selection.revision, &selection.config)?;
    let windows = crate::fidelity_dataset::quick_windows(&reference)?;
    let mut exact_info = info.clone();
    exact_info.configuration.speculative_acceptance = SpeculativeAcceptance::Exact;
    let fingerprint = crate::server::fingerprint(&exact_info);
    let prompts: Vec<_> = windows.iter().map(|window| {
        ensure!(window.media.is_empty(), "generated match requires text fidelity prompts");
        // Context-only anchors continue after the full sealed window; mixed windows
        // continue from their first generated-token boundary.
        let prompt_end = window.roles.iter().position(|role| role == "gen").unwrap_or(window.tokens.len());
        ensure!(prompt_end > 0 && prompt_end <= window.tokens.len(), "invalid fidelity prompt boundary");
        Ok((window.id.clone(), window.tokens[..prompt_end].to_vec()))
    }).collect::<Result<_>>()?;
    if let Some(saved) = saved {
        validate_reference_session(saved)?;
        ensure!(saved.exact_fingerprint == fingerprint && saved.reference_sha256 == digest && saved.max_tokens == TOKENS,
            "generated reference configuration/dataset differs");
        ensure!(saved.cases.len() == prompts.len() && saved.cases.iter().zip(&prompts)
            .all(|(case, (id, ids))| &case.id == id && &case.prompt_ids == ids && !case.generated.is_empty()),
            "generated reference does not cover the fixed fidelity prompts");
    }
    let mut cases = Vec::new();
    for (id, prompt_ids) in prompts {
        let chat = client.chat(crate::baseline::plain(client, "fixed fidelity continuation", TOKENS as u64),
            Some(ProbeSpec { cold: true, pair_exempt: true, prompt_ids: Some(prompt_ids.clone()),
                record_rows: 1, ..ProbeSpec::default() }))?;
        let probe = chat.probe.context("generated match needs probe generated IDs")?;
        ensure!(probe.error.is_none() && probe.engine.is_some(), "generated probe failed: {:?}", probe.error);
        ensure!(probe.cold && probe.cached_tokens == 0, "generated continuation reused prefix cache");
        ensure!(probe.prompt_ids == prompt_ids && !probe.generated.is_empty(), "generated probe prompt/coverage differs");
        cases.push(GeneratedCase { id, prompt_ids, generated: probe.generated });
    }
    let mut check = Check::new("greedy_match", "Greedy match against exact arm");
    check.status = CheckStatus::Info;
    if let Some(saved) = saved {
        let matches: Vec<_> = saved.cases.iter().zip(&cases).map(|(exact, candidate)|
            compare(&exact.id, &exact.generated, &candidate.generated)).collect();
        let matched: usize = matches.iter().map(|m| m.matched).sum();
        let positions: usize = matches.iter().map(|m| m.positions).sum();
        ensure!(positions > 0, "no generated positions to compare");
        let rate = matched as f64 / positions as f64;
        check.set("match_rate", rate);
        check.set("matched", matched as u64);
        check.set("positions", positions as u64);
        check.set("per_prompt", json!(matches));
        check.summary = format!("greedy match {:.1}% ({matched}/{positions} positional tokens)", rate * 100.0);
    } else {
        check.summary = "saved exact-arm generated tokens on the fixed quick fidelity prompts".into();
        check.set("generated_reference", json!(GeneratedReference { schema: "cuteafd.bench.generated-reference/1".into(),
            server_session: session().into(), exact_fingerprint: fingerprint, reference_sha256: digest,
            max_tokens: TOKENS, speculative_acceptance: SpeculativeAcceptance::Exact, cases: cases.clone() }));
    }
    if saved.is_some() { check.set("candidate_generated", json!(cases)); }
    check.set("speculative_acceptance", json!(client.configuration.speculative_acceptance));
    Ok(check)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn positional_match_counts_reconvergence_and_length_difference() {
        let score = compare("fixed", &[1, 2, 3, 4], &[1, 9, 3]);
        assert_eq!((score.matched, score.positions, score.first_divergence), (2, 4, Some(1)));
        assert_eq!(compare("fixed", &[1], &[1, 2]).first_divergence, Some(1));
        assert_eq!(compare("fixed", &[1, 2], &[1, 2]).first_divergence, None);
    }
    #[test]
    fn cli_parameters_are_explicit_and_validated() {
        assert!(from_cli("exact", None, None, None).is_ok());
        assert!(from_cli("min_p", Some(0.3), None, None).is_ok());
        assert!(from_cli("typical", None, Some(0.1), Some(0.2)).is_ok());
        assert!(from_cli("exact", Some(0.3), None, None).is_err());
        assert!(from_cli("min_p", Some(f32::NAN), None, None).is_err());
        assert!(from_cli("typical", None, Some(-0.1), Some(0.2)).is_err());
        assert!(validate(&RunRequest { speculative_acceptance: SpeculativeAcceptance::MinP { min_p: 0.3 },
            save_generated_reference: true, ..Default::default() }).is_err());
    }
}

#[cfg(test)]
mod policy_tests {
    use super::*;
    #[test]
    fn generated_reference_cannot_cross_server_sessions() {
        let mut saved = GeneratedReference { schema: "cuteafd.bench.generated-reference/1".into(),
            server_session: session().into(), exact_fingerprint: "fixed".into(), reference_sha256: "dataset".into(),
            max_tokens: TOKENS, speculative_acceptance: SpeculativeAcceptance::Exact, cases: vec![] };
        let request = |saved| RunRequest { generated_reference: Some(saved), ..Default::default() };
        validate(&request(saved.clone())).unwrap();
        saved.server_session = "different-process".into();
        assert!(validate(&request(saved)).unwrap_err().contains("same running server"));
    }

    #[test]
    fn typed_request_roundtrip_preserves_parameters() {
        for policy in [SpeculativeAcceptance::Exact, SpeculativeAcceptance::MinP { min_p: 0.3 },
            SpeculativeAcceptance::Typical { epsilon: 0.1, delta: 2.0 }] {
            let request: RunRequest = serde_json::from_value(json!({"speculative_acceptance": policy})).unwrap();
            assert_eq!(json!(request.speculative_acceptance), json!(policy));
            validate(&request).unwrap();
        }
        assert!(from_cli("typical", None, Some(1.1), Some(1.0)).is_err());
        assert!(from_cli("typical", None, Some(0.1), Some(0.0)).is_err());
    }

    #[test]
    fn acceptance_fingerprint_separates_all_parameters() {
        let mut info = ServerInfo { model: "fixed".into(), ..Default::default() };
        let policies = [SpeculativeAcceptance::Exact, SpeculativeAcceptance::MinP { min_p: 0.3 },
            SpeculativeAcceptance::MinP { min_p: 0.4 }, SpeculativeAcceptance::Typical { epsilon: 0.1, delta: 1.0 },
            SpeculativeAcceptance::Typical { epsilon: 0.2, delta: 1.0 }, SpeculativeAcceptance::Typical { epsilon: 0.1, delta: 2.0 }];
        let keys: std::collections::HashSet<_> = policies.iter().map(|&policy| {
            info.configuration.speculative_acceptance = policy;
            crate::server::fingerprint(&info)
        }).collect();
        assert_eq!(keys.len(), policies.len());
    }
}
