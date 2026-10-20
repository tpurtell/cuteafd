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

/// The verdict sentence a report carries when its own gates failed. It is the
/// lead of a publication note written while the panel failed, so a surface
/// showing it derives it from the report's quality under the rule in force now
/// rather than from the note's stored text.
pub const NOT_RECOMMENDED: &str = "FAIL: not recommended.";

/// Applies the fidelity rule to `check` in place.
///
/// NVIDIA's NVFP4 exports quantize the routed experts' activations as well as
/// their weights (W4A4) and the checkpoint is calibrated for that arithmetic,
/// so its absolute fidelity against a BF16 golden reference — measured and
/// shown — is not a gate: a miss is informational, and the gates for such a
/// card are the exact prefix-cache restore and lossless speculation. Every
/// other checkpoint keeps the calibrated bar.
pub fn apply_fidelity(info: &ServerInfo, check: &mut Check) {
    if matches!(check.status, CheckStatus::Unsupported | CheckStatus::Skipped | CheckStatus::Pending) {
        // Nothing was measured (no reference, no server hook): the check keeps
        // its own outcome.
        return;
    }
    if calibrated_w4a4(info) {
        check.status = CheckStatus::Info;
        check.note = Some(CALIBRATED_W4A4.into());
        return;
    }
    check.note = None;
    if let Some(met) = bar_met(check) {
        check.status = if met { CheckStatus::Pass } else { CheckStatus::Fail };
    }
}

/// Recomputes a stored report's quality status under the rule in force now,
/// so a report written before the rule can be published or re-rendered without
/// editing it.
pub fn reconcile(report: &mut Report) {
    let info = &report.server;
    let Some(baseline) = report.baseline.as_mut() else { return };
    for check in baseline.quality.checks.iter_mut().filter(|c| c.id == "fidelity") {
        apply_fidelity(info, check);
    }
    baseline.quality.settle();
}

/// `report` as the rule in force now reads it: the same measured data, with
/// every verdict this rule reaches. Every surface that shows a verdict renders
/// from this, so a report written under an earlier rule cannot present a stale
/// one, and the stored file is never edited to fix it.
pub fn under_current_rule(report: &Report) -> Report {
    let mut current = report.clone();
    reconcile(&mut current);
    current
}

/// The verdict `report`'s quality earns under the rule in force now, if any:
/// the sentence a surface shows when the gates failed, and nothing when the
/// same numbers are informational (a calibrated W4A4 checkpoint's fidelity) or
/// pass.
pub fn warning(report: &Report) -> Option<&'static str> {
    report.quality_failed().then_some(NOT_RECOMMENDED)
}

/// `note` without a stored verdict sentence: the prose a publication note
/// still carries once the verdict itself is taken from the report's quality.
pub fn without_verdict(note: &str) -> &str {
    note.strip_prefix(NOT_RECOMMENDED).map_or(note, str::trim_start)
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
    // Archived flat-era runs did not record a precision tier. Do not invent one
    // from today's cache or rewrite their published status while re-rendering.
    run.precision.as_ref()?;
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
    fn failed_nvfp4_tier_verdict_keeps_calibrated_w4a4_info_status() {
        let tmp = tempfile::tempdir().unwrap();
        let mut report = nvfp4_card(tmp.path(), experts("NVFP4"));
        let panel = crate::sample::full_report();
        let mut run: crate::fidelity::Run = serde_json::from_value(
            panel.panel("fidelity").unwrap().latest().unwrap()["decode"].clone()).unwrap();
        run.precision = Some(crate::fidelity_precision::Precision::new(4.0, "nvfp4", false, "test").unwrap());
        for p in &mut run.score.records { p.kl = 0.0655; }
        let verdict = crate::fidelity::verdict(&run);
        assert!(!verdict.pass);
        assert!(verdict.reasons.iter().any(|s| s.contains("KL above 3 to <6 bpw tier bar 0.05")));
        let check = report.baseline.as_mut().unwrap().quality.checks.iter_mut().find(|c| c.id == "fidelity").unwrap();
        check.set("run", json!(run)); check.set("verdict", json!(verdict));
        check.set("precision", json!(run.precision));
        check.status = CheckStatus::Fail;
        reconcile(&mut report);
        let check = report.baseline.as_ref().unwrap().quality.check("fidelity").unwrap();
        assert_eq!(check.status, CheckStatus::Info);
        assert_eq!(check.note.as_deref(), Some(CALIBRATED_W4A4));
        assert_eq!(check.metrics["verdict"]["pass"], false);
        assert!(crate::render::card::card_svg(&report).contains("4.0 bpw"));
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
        let html = crate::publish::results(&placed, &crate::publish::one_row_of(&placed[0])).unwrap();
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
}
