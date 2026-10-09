//! Published fidelity tiers share the CLI scorer, inside the dashboard's held slot.
use super::{Ctx, Panel, Progress, Rates};
use crate::client::Client;
use crate::fidelity::{compare, verdict, Run};
use crate::fidelity_cli::{run_with, RunArgs};
use crate::report::ServerInfo;
use anyhow::{Context, Result};
use cuteafd_api::openai::probe::ProbeSpec;
use serde_json::{json, Value};
use std::path::PathBuf;

pub static STANDARD: FidelityPanel = FidelityPanel { full: false };
pub static FULL: FidelityPanel = FidelityPanel { full: true };
pub struct FidelityPanel { full: bool }

pub fn prefill_unavailable(info: &ServerInfo) -> Option<String> {
    let admitted = crate::fidelity_dataset::prefill_admitted(&info.configuration.settings);
    (!admitted).then(|| "Prefill scoring not admitted at launch; restart with FULL_PREFILL_LOGITS=on (--full-prefill-logits).".into())
}

/// Runtime scratch rows are consumed window-by-window, then the private run directory is removed.
pub fn score(client: &Client, info: &ServerInfo, tier: &str, path: &str, progress: &Progress,
    start: f64, span: f64, max_context: u64) -> Result<Run> {
    let root = std::env::var_os("CUTEAFD_PROBE_DUMP_ROOT").map(PathBuf::from).unwrap_or_else(||
        PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/cuteafd/fidelity/rows"));
    std::fs::create_dir_all(&root)?;
    let scratch = tempfile::tempdir_in(root)?;
    let args = RunArgs { url: client.base.clone(), tier: tier.into(), arm: "dashboard".into(),
        out: scratch.path().join("run.json"), reference: None,
        media_root: std::env::var_os("CUTEAFD_FIDELITY_MEDIA_ROOT").map(PathBuf::from),
        dataset: None, dataset_revision: None, dataset_config: None, dataset_cache: None,
        rows: None, dump_dir: Some(scratch.path().to_path_buf()), score_path: path.into(),
        verify_rows: None, api_key: None };
    let mut model_record = client.model_record()?;
    model_record["max_context"] = json!(max_context);
    model_record["full_prefill_logits"] = json!(prefill_unavailable(info).is_none());
    model_record["snapshot"] = json!(info.configuration.snapshot);
    let checkpoint = info.checkpoint();
    // The checkpoint, not an aliased served name, binds the reference and history.
    run_with(&args, &checkpoint, &model_record, |request| {
        client.check()?;
        let spec: ProbeSpec = serde_json::from_value(request["spec"].clone())?;
        let chat = client.chat(request["body"].clone(), Some(spec))?;
        let probe = chat.probe.context("no fidelity probe record")?;
        Ok(json!({"probe": probe, "server": {"model": checkpoint, "family": info.family,
            "snapshot": info.configuration.snapshot, "settings": info.configuration.settings, "build": info.build}}))
    }, |done, total, run| {
        progress.step(start + span * done as f64 / total as f64, format!("{tier} {path}: {done}/{total} windows"));
        if tier != "quick" {
            let mut partial = record(run, None);
            if path == "prefill" {
                if let Some(previous) = progress.get().partial {
                    for key in ["decode", "verdict", "per_window"] {
                        if let Some(value) = previous.get(key) { partial[key] = value.clone(); }
                    }
                }
            }
            progress.partial(partial);
        }
        // The qualified compact scorer has consumed these multi-gigabyte rows.
        // Dashboard passes retain scores, not full-vocabulary scratch dumps.
        let name = if tier == "standard" { format!("window-{path}-{:03}", done - 1) }
            else { format!("window-{:03}", done - 1) };
        let _ = std::fs::remove_dir_all(scratch.path().join(name));
    }).and_then(|run| {
        anyhow::ensure!(run.score.records.iter().all(|p| p.position < max_context as usize), "fidelity exceeds server context");
        Ok(run)
    })
}

pub fn record(run: &Run, previous: Option<&Run>) -> Value {
    let paired = previous.map(|previous| match compare(run, previous, 0.005, 0.005, 1000, 20260829) {
        Ok(comparison) => json!({"earlier_arm": previous.arm, "earlier_score": summary(&crate::reference::Fidelity::from_records(previous.score.records.iter().filter(|p| p.role == "gen").cloned().collect())), "comparison": comparison}),
        Err(error) => json!({"unavailable": error.to_string()}),
    });
    let (path, verdict_key, window_key) = if run.path_shape == "prefill-shaped" {
        ("prefill", "prefill_verdict", "prefill_per_window")
    } else { ("decode", "verdict", "per_window") };
    let mut value = json!({"tier": run.tier, "dataset": run.dataset, "paired": paired});
    if let Some(selection) = &run.reference_selection {
        for (key, field) in serde_json::to_value(selection).unwrap().as_object().unwrap() {
            value[key] = field.clone();
        }
    }
    if run.tier == crate::fidelity_dataset::STANDARD_TIER {
        value["mode"] = run.dataset.as_ref().map(|d| d["standard_subset"]["mode"].clone()).unwrap_or(Value::Null);
    }
    value[path] = json!(run);
    value[verdict_key] = json!(verdict(run));
    value[window_key] = json!(window_summaries(run));
    value
}

fn summary(score: &crate::reference::Fidelity) -> crate::reference::Fidelity {
    let mut score = score.clone(); score.records.clear(); score
}

pub fn window_summaries(run: &Run) -> std::collections::BTreeMap<String, crate::reference::Fidelity> {
    run.score.groups("window").into_iter().map(|(id, f)| (id, summary(&f))).collect()
}

impl Panel for FidelityPanel {
    fn id(&self) -> &'static str { if self.full { "fidelity_full" } else { "fidelity" } }
    fn title(&self) -> &'static str { if self.full { "Fidelity · Full" } else { "Fidelity" } }
    fn description(&self) -> &'static str {
        if self.full { "64 sealed windows, decode + prefill; calibrated top-1, KL and tripwires. Requires launch-admitted prefill logits." }
        else { "Standard-v2: sealed, balanced 32 decode / 32 prefill windows; separate calibrated verdicts. Without prefill admission: all 64 decode only. Optional paired history comparison." }
    }
    fn estimate_s(&self, rates: &Rates, info: &ServerInfo) -> f64 {
        let decode = 32768.0 / rates.decode_tok_s + 350000.0 / rates.prefill_tok_s;
        let prefill = 380000.0 / rates.prefill_tok_s;
        if self.full { decode + prefill }
        else if prefill_unavailable(info).is_none() { (decode + prefill) / 2.0 }
        else { decode }
    }
    fn unavailable(&self, info: &ServerInfo) -> Option<String> {
        if info.checkpoint().is_empty() {
            return Some("Discovering server configuration; run the basic card first".into());
        }
        crate::fidelity_match::resolve(&info.checkpoint(), info.configuration.snapshot.as_deref().map(std::path::Path::new))
            .err().map(|e| e.to_string())
            .or_else(|| if self.full { prefill_unavailable(info) } else { None })
    }
    fn run(&self, ctx: &Ctx<'_>) -> Result<Value> {
        let tier = if self.full { "full" } else { "standard" };
        let both_paths = self.full || prefill_unavailable(ctx.info).is_none();
        let decode = score(ctx.client, ctx.info, tier, "decode", ctx.progress, 0.0,
            if both_paths { 0.5 } else { 1.0 }, ctx.max_context)?;
        // History stays optional: a missing/incompatible earlier run never masquerades as a pass.
        let mut result = record(&decode, None);
        if both_paths {
            let prefill = score(ctx.client, ctx.info, tier, "prefill", ctx.progress, 0.5, 0.5, ctx.max_context)?;
            if !self.full {
                anyhow::ensure!(decode.engine == prefill.engine && decode.settings == prefill.settings
                    && decode.dataset == prefill.dataset && decode.reference_sha256 == prefill.reference_sha256,
                    "Standard scoring paths use different references or server settings");
            }
            result["prefill_verdict"] = json!(verdict(&prefill));
            result["prefill_per_window"] = json!(window_summaries(&prefill));
            result["prefill"] = json!(prefill);
        }
        ctx.progress.partial(result.clone());
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_requires_launch_admission_and_standard_does_not() {
        let mut info = ServerInfo {model:"Qwen/Qwen3.8-Flash-Next-FP8".into(),..Default::default()};
        assert!(STANDARD.unavailable(&info).is_none());
        assert!(FULL.unavailable(&info).unwrap().contains("FULL_PREFILL_LOGITS=on"));
        info.configuration.settings.push(crate::report::Setting {name:"full-prefill-logits".into(),
            value:Some("true".into()),default:Some("false".into()),source:"cli".into()});
        assert!(FULL.unavailable(&info).is_none());
        for model in ["XiaomiMiMo/MiMo-V2.6-Flash-MOPD", "XiaomiMiMo/MiMo-V2.6-Pro-MOPD"] {
            info.model = model.into();
            assert!(STANDARD.unavailable(&info).is_none());
            assert!(FULL.unavailable(&info).is_none());
        }
        info.configuration.settings.clear();
        assert!(FULL.unavailable(&info).unwrap().contains("FULL_PREFILL_LOGITS=on"));
    }
}
