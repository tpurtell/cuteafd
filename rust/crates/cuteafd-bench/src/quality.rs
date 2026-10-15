//! The rule that turns a fidelity check's measured numbers into a status.
//!
//! A `report.json` is measured data, not a verdict: what a check's numbers
//! mean can change after the report was written (a calibration rule, a new
//! checkpoint's intended arithmetic), so the publisher and the offline
//! exporter recompute every fidelity status from the metrics the check
//! recorded and the served checkpoint's own declared numerics, never from the
//! status stored in the file.
use crate::report::{Check, CheckStatus, Report, ServerInfo};
use cuteafd_loader::formats::modelopt::ModelOpt;
use cuteafd_loader::plan::checkpoint::read_json;
use std::path::Path;

/// Why a fidelity check is informational rather than gated.
pub const CALIBRATED_W4A4: &str = "calibrated W4A4";

/// Stored publication lead for a fidelity miss. A caption paints this only
/// when [`shown`] still classifies that panel as a failure.
const STORED_FAIL_LEAD: &str = "FAIL: not recommended.";

/// What one surface shows for a panel under the policy in force now.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shown {
    pub status: CheckStatus,
    /// Share-card badge, or the check's detail line for a single panel.
    pub verdict: String,
    /// `failed: …` when the quality gate fails; empty otherwise.
    pub fail_line: String,
    /// Image caption. Empty when the report stores none.
    pub caveat: String,
    /// Prefix of [`Self::caveat`] that a caption emphasizes.
    pub caveat_lead: Option<String>,
    /// Why an informational check is not gated.
    pub note: Option<String>,
}

/// The displayed verdict and caption for `panel`.
///
/// `panel` is a check id (`fidelity`, `cache_exact`, …) or `quality` for the
/// whole baseline gate. The stored check status and `qualification.caveat`
/// are measurements: a calibrated W4A4 fidelity miss is informational, and a
/// stored `FAIL: not recommended.` lead follows that classification. Every
/// other panel keeps the stored outcome. The card, the report page and the
/// README caption all call this; none of them read the stored caveat.
pub fn shown(report: &Report, panel: &str) -> Shown {
    let checks = policy_checks(report);
    let fidelity = checks.iter().find(|check| check.id == "fidelity").cloned();
    let (caveat, caveat_lead) = display_caveat(report.server.setting("qualification.caveat"), fidelity.as_ref());
    if panel == "quality" || panel == "baseline" {
        if report.baseline.is_none() {
            return Shown { status: CheckStatus::Pending, verdict: "quality not verified".into(),
                fail_line: String::new(), caveat, caveat_lead, note: None };
        }
        let mut quality = crate::report::Quality { checks, status: CheckStatus::Pending };
        quality.settle();
        let fail_line = fail_line(&quality.checks);
        return Shown { status: quality.status, verdict: quality.badge(), fail_line, caveat, caveat_lead,
            note: fidelity.and_then(|check| check.note) };
    }
    let Some(check) = checks.into_iter().find(|check| check.id == panel) else {
        return Shown { status: CheckStatus::Pending, verdict: String::new(), fail_line: String::new(),
            caveat, caveat_lead, note: None };
    };
    let verdict = check.detail();
    Shown { status: check.status, verdict, fail_line: String::new(), caveat, caveat_lead, note: check.note }
}

/// Applies the fidelity rule to `check` in place. Live scoring and
/// [`reconcile`] both go through [`shown`].
pub fn apply_fidelity(info: &ServerInfo, check: &mut Check) {
    let view = shown(&fidelity_view(info, check), "fidelity");
    if matches!(check.status, CheckStatus::Unsupported | CheckStatus::Skipped | CheckStatus::Pending) {
        // Nothing was measured (no reference, no server hook): the check keeps
        // its own outcome.
        return;
    }
    check.status = view.status;
    check.note = view.note;
}

/// Recomputes a stored report's quality status under the rule in force now,
/// so a report written before the rule can be published or re-rendered without
/// editing it. The export on disk is left as measured.
pub fn reconcile(report: &mut Report) {
    let view = shown(report, "fidelity");
    let Some(baseline) = report.baseline.as_mut() else { return };
    if let Some(check) = baseline.quality.checks.iter_mut().find(|check| check.id == "fidelity") {
        if !matches!(check.status, CheckStatus::Unsupported | CheckStatus::Skipped | CheckStatus::Pending) {
            check.status = view.status;
            check.note = view.note;
        }
    }
    baseline.quality.settle();
}

/// A one-check report so [`apply_fidelity`] can ask [`shown`] for the outcome.
fn fidelity_view(info: &ServerInfo, check: &Check) -> Report {
    Report {
        schema: String::new(), id: String::new(), created: String::new(), finished: None,
        status: crate::report::RunStatus::Imported, profile: String::new(), plan: Vec::new(),
        server: info.clone(), fingerprint: String::new(),
        baseline: Some(crate::report::Baseline {
            fingerprint: String::new(), run_id: String::new(), created: String::new(),
            card: crate::report::BasicCard::default(),
            quality: crate::report::Quality { status: CheckStatus::Pending, checks: vec![check.clone()] },
            seconds: 0.0, thermal: None, fabric: Vec::new(),
        }),
        panels: Vec::new(), error: None, fabric: Vec::new(),
    }
}

/// Checks with the fidelity panel classified by the current numerics rule.
fn policy_checks(report: &Report) -> Vec<Check> {
    let Some(baseline) = &report.baseline else { return Vec::new() };
    baseline.quality.checks.iter().map(|check| {
        if check.id != "fidelity" {
            return check.clone();
        }
        let (status, note) = fidelity_outcome(&report.server, check);
        Check { status, note, ..check.clone() }
    }).collect()
}

/// NVIDIA's NVFP4 exports quantize the routed experts' activations as well as
/// their weights (W4A4) and the checkpoint is calibrated for that arithmetic,
/// so its absolute fidelity against a BF16 golden reference — measured and
/// shown — is not a gate: a miss is informational, and the gates for such a
/// card are the exact prefix-cache restore and lossless speculation. Every
/// other checkpoint keeps the calibrated bar.
fn fidelity_outcome(info: &ServerInfo, check: &Check) -> (CheckStatus, Option<String>) {
    if matches!(check.status, CheckStatus::Unsupported | CheckStatus::Skipped | CheckStatus::Pending) {
        return (check.status, check.note.clone());
    }
    if calibrated_w4a4(info) {
        return (CheckStatus::Info, Some(CALIBRATED_W4A4.into()));
    }
    let mut status = check.status;
    if let Some(met) = bar_met(check) {
        status = if met { CheckStatus::Pass } else { CheckStatus::Fail };
    }
    (status, None)
}

fn fail_line(checks: &[Check]) -> String {
    let failed: Vec<&str> = checks.iter().filter(|check| check.status == CheckStatus::Fail)
        .map(|check| check.title.as_str()).collect();
    if failed.is_empty() { String::new() } else { format!("failed: {}", failed.join(", ")) }
}

/// A stored `FAIL: not recommended.` lead is the fidelity panel's verdict.
/// When that panel is informational, the lead follows; every other note is
/// publication text and stays as stored.
fn display_caveat(stored: Option<&str>, fidelity: Option<&Check>) -> (String, Option<String>) {
    let Some(stored) = stored.filter(|note| !note.is_empty()) else { return (String::new(), None) };
    let Some(rest) = stored.strip_prefix(STORED_FAIL_LEAD) else { return (stored.to_string(), None) };
    if fidelity.is_some_and(|check| check.status == CheckStatus::Info) {
        let why = fidelity.and_then(|check| check.note.as_deref()).unwrap_or("informational");
        let lead = format!("Informational (info: {why}).");
        return (format!("{lead}{rest}"), Some(lead));
    }
    (stored.to_string(), Some(STORED_FAIL_LEAD.to_string()))
}

/// Whether the served checkpoint declares W4A4 NVFP4 routed experts. Read from
/// the checkpoint's own `config.json` / `hf_quant_config.json`, the way the
/// loader reads them — never from its name.
pub fn calibrated_w4a4(info: &ServerInfo) -> bool {
    checkpoint_w4a4(&info.checkpoint(), info.configuration.snapshot.as_deref().map(Path::new))
}

/// Whether the checkpoint `model` (served from `snapshot`, in whatever path
/// spelling the server saw) is a W4A4 NVFP4 export.
pub fn checkpoint_w4a4(model: &str, snapshot: Option<&Path>) -> bool {
    crate::fidelity_match::served_snapshot_path(model, snapshot).is_some_and(|path| declared_w4a4(&path))
}

/// A checkpoint without ModelOpt metadata, or whose metadata does not read, is
/// not a W4A4 export: the bar stands.
fn declared_w4a4(snapshot: &Path) -> bool {
    let Ok(config) = read_json(&snapshot.join("config.json")) else { return false };
    ModelOpt::read(snapshot, &config).is_ok_and(|meta| meta.is_some_and(|meta| meta.experts_w4a4()))
}

/// Whether the check's recorded run met the calibrated bar, decided by the
/// verdict rather than read from a stored status. `None` for a report written
/// before the run was stored: there is nothing to recompute from.
fn bar_met(check: &Check) -> Option<bool> {
    let run: crate::fidelity::Run = serde_json::from_value(check.metrics.get("run")?.clone()).ok()?;
    Some(crate::fidelity::verdict(&run).pass)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::{CheckStatus, Report};
    use serde_json::json;

    /// A served snapshot carrying `config` as its own `config.json`.
    fn configured(dir: &Path, config: serde_json::Value) -> std::path::PathBuf {
        let path = dir.join("models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/2061e0b0");
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("config.json"), config.to_string()).unwrap();
        path
    }

    fn experts(algo: &str) -> serde_json::Value {
        json!({"quantization_config": {"quant_method": "modelopt", "quant_algo": algo, "group_size": 16,
            "quantized_layers": {"model.language_model.layers.0.mlp.experts": {"quant_algo": algo}}}})
    }

    #[test]
    fn w4a4_experts_are_read_from_the_checkpoint_not_its_name() {
        let tmp = tempfile::tempdir().unwrap();
        let name = "nvidia/Qwen3.8-Flash-Next-NVFP4";
        assert!(checkpoint_w4a4(name, Some(&configured(tmp.path(), experts("NVFP4")))));
        // The same name in a checkpoint whose experts are weight-only.
        assert!(!checkpoint_w4a4(name, Some(&configured(tmp.path(), experts("W4A16_NVFP4")))));
        // A checkpoint that is not ModelOpt at all.
        assert!(!checkpoint_w4a4(name, Some(&configured(tmp.path(), json!({"quant_method": "fp8"})))));
        // No metadata on disk (a report re-rendered away from its checkpoint).
        assert!(!checkpoint_w4a4(name, Some(&tmp.path().join("elsewhere"))));
    }

    /// The Qwen 3.8 Flash Next NVFP4 card as it was written: KL 0.039 and
    /// top-1 93.7% against a bar of 0.040 and 94.0%.
    fn nvfp4_card(dir: &Path, config: serde_json::Value) -> Report {
        let mut report = crate::sample::report(true);
        report.server.model = "nvidia/Qwen3.8-Flash-Next-NVFP4".into();
        report.server.family = Some("qwen4".into());
        report.server.configuration.snapshot = Some(configured(dir, config).display().to_string());
        let check = report.baseline.as_mut().unwrap().quality.checks.iter_mut()
            .find(|c| c.id == "fidelity").unwrap();
        check.set("kl", 0.03866);
        check.set("top1", 0.93719);
        report
    }

    #[test]
    fn calibrated_w4a4_fidelity_is_informational_and_the_quality_status_passes() {
        let tmp = tempfile::tempdir().unwrap();
        let mut report = nvfp4_card(tmp.path(), experts("NVFP4"));
        assert!(report.quality_failed(), "the stored report failed the bar");
        reconcile(&mut report);
        let quality = &report.baseline.as_ref().unwrap().quality;
        assert_eq!(quality.check("fidelity").unwrap().status, CheckStatus::Info);
        assert_eq!(quality.badge(),
            "KL 0.039 · top-1 93.7% (info: calibrated W4A4) ✓ exact cache ✓ lossless spec");
        assert_eq!(quality.status, CheckStatus::Pass);
        assert!(!report.quality_failed());
        let card = crate::render::card::card_svg(&report);
        assert!(card.contains("top-1 93.7% (info: calibrated W4A4)"), "{card}");
        assert!(!card.contains("QUALITY GATE FAILED"));
        // Reconciling again, or a report that was already informational, is a no-op.
        let mut twice = report.clone();
        reconcile(&mut twice);
        assert_eq!(crate::render::card::card_svg(&twice), card);
    }

    #[test]
    fn the_bar_still_gates_a_checkpoint_without_w4a4_experts() {
        let tmp = tempfile::tempdir().unwrap();
        let mut report = nvfp4_card(tmp.path(), experts("W4A16_NVFP4"));
        reconcile(&mut report);
        assert_eq!(report.baseline.as_ref().unwrap().quality.status, CheckStatus::Fail);
        assert!(report.quality_failed());
        assert!(crate::render::card::card_svg(&report).contains("QUALITY GATE FAILED"));
        // A checkpoint this host cannot read keeps the bar too: the claim that
        // its experts are W4A4 is what makes the numbers informational.
        let mut missing = nvfp4_card(tmp.path(), experts("NVFP4"));
        missing.server.configuration.snapshot = Some(tmp.path().join("gone").display().to_string());
        reconcile(&mut missing);
        assert!(missing.quality_failed());
    }

    #[test]
    fn the_publisher_recomputes_a_stored_report() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let dir = root.join("benchmarks/qwen4/2026-10-14-smoke-qwen38-nvfp4-max");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("report.json"),
            serde_json::to_string(&nvfp4_card(&root.join("hub"), experts("NVFP4"))).unwrap()).unwrap();
        let placed = crate::publish::scan(root).unwrap();
        let html = crate::publish::results(&placed);
        assert!(html.contains("top-1 93.7% (info: calibrated W4A4)"), "{html}");
        assert!(!html.contains("FAILED"), "{html}");
        assert!(!crate::publish::index(&placed).contains("quality gate failed"));
    }

    #[test]
    fn a_report_without_the_stored_run_keeps_the_status_it_was_written_with() {
        let tmp = tempfile::tempdir().unwrap();
        // `sample::report` stores kl/top1 only: there is no run to recompute from,
        // so a non-W4A4 report is left exactly as published.
        let mut report = crate::sample::report(false);
        report.server.configuration.snapshot = Some(configured(tmp.path(), json!({})).display().to_string());
        let before = crate::render::card::card_svg(&report);
        reconcile(&mut report);
        assert_eq!(report.baseline.as_ref().unwrap().quality.status, CheckStatus::Pass);
        assert_eq!(crate::render::card::card_svg(&report), before);
    }

    fn fail_note(report: &mut Report) {
        report.server.configuration.settings.push(crate::report::Setting {
            name: "qualification.caveat".into(),
            value: Some("FAIL: not recommended. Fails the calibrated check.".into()),
            default: None, source: "publication".into(),
        });
    }

    #[test]
    fn a_stored_fail_nvfp4_export_renders_informational_on_card_row_and_caption() {
        let tmp = tempfile::tempdir().unwrap();
        let mut report = nvfp4_card(tmp.path(), experts("NVFP4"));
        assert!(report.quality_failed(), "the export stored a fidelity FAIL");
        fail_note(&mut report);
        let shown = shown(&report, "quality");
        assert_eq!(shown.status, CheckStatus::Pass);
        assert!(shown.verdict.contains("info: calibrated W4A4"), "{}", shown.verdict);
        assert!(shown.caveat.contains("info: calibrated W4A4"), "{}", shown.caveat);
        assert!(!shown.caveat.contains("FAIL:"));
        let card = crate::render::card::card_svg(&report);
        assert!(card.contains("info: calibrated W4A4"), "{card}");
        assert!(!card.contains("QUALITY GATE FAILED"), "{card}");
        let html = crate::publish::results(&[crate::publish::Placed {
            dir: "benchmarks/qwen4/stored-fail-nvfp4".into(), report,
        }]);
        assert!(html.contains("info: calibrated W4A4"), "{html}");
        assert!(html.contains("Informational"), "{html}");
        assert!(!html.contains("FAILED"), "{html}");
        assert!(!html.contains("FAIL: not recommended"), "{html}");
    }

    #[test]
    fn a_stored_fail_without_w4a4_experts_still_renders_fail() {
        let tmp = tempfile::tempdir().unwrap();
        let mut report = nvfp4_card(tmp.path(), experts("W4A16_NVFP4"));
        assert!(report.quality_failed());
        fail_note(&mut report);
        let shown = shown(&report, "quality");
        assert_eq!(shown.status, CheckStatus::Fail);
        assert!(shown.caveat.starts_with("FAIL: not recommended."));
        let card = crate::render::card::card_svg(&report);
        assert!(card.contains("QUALITY GATE FAILED"), "{card}");
        assert!(!card.contains("info: calibrated W4A4"), "{card}");
        let html = crate::publish::results(&[crate::publish::Placed {
            dir: "benchmarks/qwen4/stored-fail-w4a16".into(), report,
        }]);
        assert!(html.contains("FAILED"), "{html}");
        assert!(html.contains("FAIL: not recommended"), "{html}");
        assert!(!html.contains("info: calibrated W4A4"), "{html}");
    }
}
