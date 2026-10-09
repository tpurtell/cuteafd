//! Paired fidelity reports and window-clustered non-inferiority statistics.
//! Statistics ported from Hugh Madden's MIT glm53f-afd v1.1.0 harness/klgate.py.
use crate::reference::{Fidelity, Position};
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    pub schema: String,
    pub arm: String,
    pub checkpoint: String,
    pub set_sha256: String,
    pub reference_sha256: String,
    pub tier: String,
    /// Never compare decode and prefill rows, or call a compact KL full-vocabulary.
    pub path_shape: String,
    pub kl_kind: String,
    pub verify_rows: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<serde_json::Value>,
    #[serde(default, flatten)]
    pub reference_selection: Option<crate::fidelity_match::Resolution>,
    /// Informational split balance, separate from the comparison-stable dataset identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standard_balance: Option<serde_json::Value>,
    pub engine: String,
    pub settings: serde_json::Value,
    pub seconds: f64,
    pub score: Fidelity,
    pub floor_top1: f64,
    pub floor_kl: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tripwire_expect: Option<crate::reference::TripwireExpect>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Comparison {
    pub pass: bool,
    pub positions: usize,
    pub windows: usize,
    pub baseline_only: usize,
    pub candidate_only: usize,
    pub discordance: f64,
    pub top1_loss: f64,
    pub top1_upper95: f64,
    pub top1_se_bootstrap: f64,
    pub kl_delta: f64,
    pub kl_upper95: f64,
    pub kl_se_clustered: Option<f64>,
    pub mcnemar_p: f64,
    pub top1_margin: f64,
    pub kl_margin: f64,
    pub bootstrap: usize,
    pub seed: u64,
    pub absolute_pass: bool,
    pub tripwires: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibrated_tripwires: Option<CalibratedTripwires>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripwireMetric {
    pub baseline: f64,
    pub candidate: f64,
    pub loss: f64,
    pub lower95: f64,
    pub upper95: f64,
    pub gross_margin: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibratedTripwires {
    pub confident_top1: Option<TripwireMetric>,
    pub top3: TripwireMetric,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullComparison {
    pub pass: bool,
    pub decode: Comparison,
    pub prefill: Comparison,
}

/// A full precision decision requires both scoring shapes from the same arms.
pub fn compare_full(
    a_decode: &Run, b_decode: &Run, a_prefill: &Run, b_prefill: &Run,
    bootstrap: usize, seed: u64,
) -> Result<FullComparison> {
    for (decode, prefill) in [(a_decode, a_prefill), (b_decode, b_prefill)] {
        ensure!(decode.tier == "full" && prefill.tier == "full", "both paths must use the full tier");
        ensure!(decode.path_shape == "decode-shaped" && prefill.path_shape == "prefill-shaped",
            "full decision requires decode and prefill scoring paths");
        ensure!(decode.arm == prefill.arm && decode.checkpoint == prefill.checkpoint
            && decode.set_sha256 == prefill.set_sha256 && decode.reference_sha256 == prefill.reference_sha256
            && decode.engine == prefill.engine && decode.settings == prefill.settings && decode.dataset == prefill.dataset,
            "scoring paths use different arms, references or server settings");
        ensure!(decode.tripwire_expect == prefill.tripwire_expect,
            "scoring paths use different calibrated tripwire expectations");
    }
    let decode = compare(a_decode, b_decode, 0.005, 0.005, bootstrap, seed)?;
    let prefill = compare(a_prefill, b_prefill, 0.005, 0.005, bootstrap, seed)?;
    Ok(FullComparison { pass: decode.pass && prefill.pass, decode, prefill })
}

/// Cluster-robust SE of a ratio-of-sums token mean, using whole windows as clusters.
pub fn clustered_se(sums: &[f64], counts: &[usize]) -> Option<f64> {
    if sums.len() < 2 || sums.len() != counts.len() { return None; }
    let n = counts.iter().sum::<usize>() as f64;
    if n == 0.0 { return None; }
    let mean = sums.iter().sum::<f64>() / n;
    let residual = sums.iter().zip(counts).map(|(t, &c)| (t - c as f64 * mean).powi(2)).sum::<f64>();
    let g = sums.len() as f64;
    Some((g / (g - 1.0) * residual).sqrt() / n)
}

pub fn quantile(xs: &[f64], q: f64) -> f64 {
    let mut sorted = xs.to_vec();
    sorted.sort_by(f64::total_cmp);
    let h = (sorted.len() - 1) as f64 * q;
    let lo = h.floor() as usize;
    sorted[lo] + (sorted[(lo + 1).min(sorted.len() - 1)] - sorted[lo]) * (h - lo as f64)
}

// SplitMix64; reproducible independent of platform and without a serving RNG dependency.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn index(&mut self, n: usize) -> usize {
        let n = n as u64;
        let threshold = n.wrapping_neg() % n;
        loop { let x = self.next(); if x >= threshold { return (x % n) as usize; } }
    }
}

/// Hugh's ratio-of-sums window-block bootstrap; a sampled window keeps every row.
pub fn block_bootstrap(stats: &[[f64; 2]], counts: &[usize], b: usize, seed: u64) -> Vec<[f64; 2]> {
    let mut rng = Rng(seed);
    (0..b).map(|_| {
        let (mut sums, mut n) = ([0.0; 2], 0usize);
        for _ in stats {
            let i = rng.index(stats.len());
            n += counts[i];
            sums[0] += stats[i][0]; sums[1] += stats[i][1];
        }
        [sums[0] / n as f64, sums[1] / n as f64]
    }).collect()
}

fn erfc(x: f64) -> f64 {
    // Numerical Recipes approximation (absolute error < 1.3e-7).
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let value = t * (-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418
        + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587
        + t * (-0.82215223 + t * 0.17087277))))))))).exp();
    if x < 0.0 { 2.0 - value } else { value.min(1.0) }
}

fn pairs(run: &Run) -> Result<BTreeMap<(String, usize), &Position>> {
    ensure!(run.schema == "cuteafd.fidelity.run/2", "unknown run schema");
    ensure!(run.score.missing == 0 && run.score.non_finite == 0, "run has missing or nonfinite rows");
    ensure!(run.score.positions == run.score.records.len(), "run position count differs from records");
    let mut out = BTreeMap::new();
    for p in &run.score.records {
        ensure!(p.finite && p.kl.is_finite() && p.nll.is_finite() && p.ref_nll.is_finite(), "nonfinite position");
        ensure!(p.role == "gen" || p.role == "ctx", "unknown role");
        ensure!(out.insert((p.window.clone(), p.position), p).is_none(), "duplicate position");
    }
    ensure!(!out.is_empty(), "no scored rows");
    Ok(out)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Verdict {
    pub pass: bool,
    pub generated: Fidelity,
    pub top1_min: f64,
    pub kl_max: f64,
    pub confident_top1_min: f64,
    pub top3_min: f64,
    pub reasons: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

/// One calibrated absolute verdict shared by the card, CLI and dashboard.
/// Bounds apply to generated rows; context/code rows remain in the report.
pub fn verdict(run: &Run) -> Verdict {
    let mut generated = Fidelity::from_records(run.score.records.iter().filter(|p| p.role == "gen").cloned().collect());
    let (top1_min, kl_max) = (run.floor_top1.max(0.90), run.floor_kl.min(0.06));
    let (confident_top1_min, top3_min) = run.tripwire_expect.as_ref()
        .map_or((0.98, 0.99), |e| (e.confident_top1_min, e.top3_min));
    let mut reasons = Vec::new();
    if run.score.missing > 0 || run.score.non_finite > 0 { reasons.push("missing or nonfinite rows".into()); }
    if generated.positions == 0 { reasons.push("no generated rows".into()); }
    if !run.floor_top1.is_finite() || !run.floor_kl.is_finite() || !top1_min.is_finite() || !kl_max.is_finite() || !confident_top1_min.is_finite() || !top3_min.is_finite() {
        reasons.push("nonfinite expectation".into());
    }
    if generated.top1 + 1e-12 < top1_min { reasons.push("generated top-1 below expect/floor".into()); }
    if !generated.kl.is_finite() || generated.kl > kl_max { reasons.push("generated KL above expect/floor".into()); }
    if generated.confident_top1.is_some_and(|v| v + 1e-12 < confident_top1_min) {
        reasons.push("confident top-1 below calibrated tripwire".into());
    }
    if generated.top3_contained + 1e-12 < top3_min { reasons.push("top-3 below calibrated tripwire".into()); }
    if generated.groups("window").values().any(|w| w.top1 + 1e-12 < 0.80) {
        reasons.push("generated window top-1 below 80% floor".into());
    }
    if let Some(dataset) = &run.dataset {
        if let Err(error) = crate::fidelity_dataset::ensure_valid_publication(
            dataset["repository"].as_str().unwrap_or(""), dataset["revision"].as_str().unwrap_or(""),
            dataset["config"].as_str().unwrap_or("")) {
            reasons.push(error.to_string());
        }
    }
    generated.records.clear();
    let label = (run.tier == crate::fidelity_dataset::STANDARD_TIER).then(||
        run.dataset.as_ref().and_then(|d| d["standard_subset"]["mode"].as_str())
            .unwrap_or("standard-v2").to_string());
    Verdict { pass: reasons.is_empty(), generated, top1_min, kl_max, confident_top1_min, top3_min, reasons, label }
}

fn absolute(run: &Run) -> bool {
    let score = Fidelity::from_records(run.score.records.iter().filter(|p| p.role == "gen").cloned().collect());
    score.positions > 0 && score.top1 + 1e-12 >= run.floor_top1.max(0.90) && score.kl <= run.floor_kl.min(0.06)
        && score.groups("window").values().all(|w| w.top1 + 1e-12 >= 0.80)
}

fn comparison_settings(server: &serde_json::Value) -> Result<serde_json::Value> {
    let mut normalized = server.clone();
    ensure!(normalized.is_object(), "missing server settings");
    if let Some(settings) = normalized.get_mut("settings") {
        let mut names = std::collections::BTreeSet::new();
        let mut fixed = BTreeMap::new();
        for setting in settings.as_array().context("server settings must be an array")? {
            let name = setting.get("name").and_then(|v| v.as_str()).context("setting name missing")?;
            ensure!(!name.is_empty() && names.insert(name), "empty or duplicate server setting {name}");
            let value = setting.get("value").context("setting value missing")?;
            // Only precision switches may differ; scheduling, layout and unknown knobs stay fixed.
            // GLM 5.3 Flash's kda-state (its KDA recurrent state's storage) and target-head (the
            // target LM head's GEMM) change numerics only.
            if !matches!(name, "CUTEAFD_V41_FP8_HEAD" | "fp8-head" | "kda-fp8" | "fp8-prefill"
                | "fp8-decode" | "mtp-fp8-head" | "kv-cache" | "expert-input" | "kda-state" | "target-head") {
                fixed.insert(name.to_owned(), value.clone());
            }
        }
        *settings = serde_json::to_value(fixed)?;
    }
    Ok(normalized)
}

pub fn compare(a: &Run, b: &Run, top1_margin: f64, kl_margin: f64, bootstrap: usize, seed: u64) -> Result<Comparison> {
    ensure!(top1_margin.is_finite() && top1_margin > 0.0 && kl_margin.is_finite() && kl_margin > 0.0,
        "margins must be positive and finite");
    ensure!(bootstrap >= 100, "at least 100 bootstrap replicates required");
    for run in [a, b] {
        if let Some(dataset) = &run.dataset {
            crate::fidelity_dataset::ensure_valid_publication(
                dataset["repository"].as_str().unwrap_or(""),
                dataset["revision"].as_str().unwrap_or(""),
                dataset["config"].as_str().unwrap_or(""))?;
        }
    }
    ensure!(!a.checkpoint.is_empty() && !a.set_sha256.is_empty() && !a.reference_sha256.is_empty(), "missing provenance");
    ensure!(a.checkpoint == b.checkpoint && a.set_sha256 == b.set_sha256 && a.reference_sha256 == b.reference_sha256,
        "runs use different checkpoints, sets or references");
    ensure!(a.tier == b.tier && a.path_shape == b.path_shape && a.kl_kind == b.kl_kind && a.verify_rows == b.verify_rows && a.dataset == b.dataset,
        "runs use different tiers, scoring shapes, KL estimators or verify widths");
    ensure!(a.engine == b.engine && a.settings.get("build") == b.settings.get("build")
        && a.settings.get("snapshot") == b.settings.get("snapshot"), "runs use different engines, builds or snapshots");
    ensure!(comparison_settings(&a.settings)? == comparison_settings(&b.settings)?,
        "runs use different nonprecision server settings");
    ensure!(matches!(a.path_shape.as_str(), "decode-shaped" | "prefill-shaped"), "unknown scoring shape");
    ensure!(matches!(a.tier.as_str(), "full" | "standard-v2") || a.path_shape == "decode-shaped",
        "Quick and legacy Standard must be decode-shaped");
    ensure!(matches!(a.tier.as_str(), "quick" | "standard" | "standard-v2" | "full"), "unknown tier");
    if a.tier != "quick" {
        ensure!(a.kl_kind == "full-vocabulary" || (a.kl_kind == "qualified-top1024-plus-tail"
            && a.dataset.as_ref().is_some_and(|d| d["revision"].as_str().is_some_and(|r|
                r.len() == 40 && r.bytes().all(|c| c.is_ascii_hexdigit())))),
            "full tier requires full-vocabulary or revision-pinned qualified top1024 KL");
    }
    let (pa, pb) = (pairs(a)?, pairs(b)?);
    ensure!(pa.keys().collect::<Vec<_>>() == pb.keys().collect::<Vec<_>>(), "runs score different rows");
    let mut by_window: BTreeMap<String, ([f64; 2], usize)> = BTreeMap::new();
    let (mut a_only, mut b_only) = (0usize, 0usize);
    for (key, x) in &pa {
        let y = pb[key];
        ensure!(x.role == y.role && x.block == y.block && x.bucket == y.bucket && x.reference_argmax == y.reference_argmax
            && x.ref_nll == y.ref_nll && x.confident == y.confident, "row metadata differ at {key:?}");
        if x.role != "gen" { continue; }
        a_only += usize::from(x.agree && !y.agree);
        b_only += usize::from(!x.agree && y.agree);
        let (sums, count) = by_window.entry(key.0.clone()).or_default();
        sums[0] += f64::from(y.agree) - f64::from(x.agree);
        sums[1] += x.kl - y.kl;
        *count += 1;
    }
    ensure!(!by_window.is_empty(), "no assistant-generated positions");
    let stats: Vec<_> = by_window.values().map(|v| v.0).collect();
    let counts: Vec<_> = by_window.values().map(|v| v.1).collect();
    let n = counts.iter().sum::<usize>();
    let loss = (b_only as f64 - a_only as f64) / n as f64;
    let kl_delta = stats.iter().map(|s| s[1]).sum::<f64>() / n as f64;
    let reps = block_bootstrap(&stats, &counts, bootstrap, seed);
    let top_reps: Vec<_> = reps.iter().map(|s| s[0]).collect();
    let kl_reps: Vec<_> = reps.iter().map(|s| s[1]).collect();
    let mean = top_reps.iter().sum::<f64>() / bootstrap as f64;
    let se = (top_reps.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (bootstrap - 1) as f64).sqrt();
    let top_upper = loss + 1.6448536269514722 * se;
    let kl_upper = quantile(&kl_reps, 0.95);
    let discordant = a_only + b_only;
    let chi2 = if discordant > 0 { ((a_only.abs_diff(b_only) as f64 - 1.0).max(0.0)).powi(2) / discordant as f64 } else { 0.0 };
    let absolute_pass = absolute(a) && absolute(b);
    let mut tripwires = Vec::new();
    ensure!(a.tripwire_expect == b.tripwire_expect, "runs use different calibrated tripwire expectations");
    let calibrated_tripwires = if let Some(expect) = &a.tripwire_expect {
        ensure!([expect.confident_top1_min, expect.top3_min].iter().all(|x| x.is_finite() && (0.0..=1.0).contains(x))
            && expect.confident_drop_margin == 0.01 && expect.top3_drop_margin == 0.005,
            "invalid calibrated tripwire thresholds or gross margins");
        for (name, r) in [("candidate", a), ("baseline", b)] {
            let f = Fidelity::from_records(r.score.records.iter().filter(|p| p.role == "gen").cloned().collect());
            if f.confident_top1.is_some_and(|v| v + 1e-12 < expect.confident_top1_min) {
                tripwires.push(format!("{name}: confident top1 below family calibrated minimum"));
            }
            if f.top3_contained + 1e-12 < expect.top3_min {
                tripwires.push(format!("{name}: top3 containment below family calibrated minimum"));
            }
        }
        let metric = |confident_only: bool, margin: f64| -> Option<TripwireMetric> {
            let mut windows: BTreeMap<&str, ([f64; 2], usize)> = BTreeMap::new();
            let (mut baseline, mut candidate) = (0usize, 0usize);
            for (key, x) in &pa {
                if x.role != "gen" || (confident_only && !x.confident) { continue; }
                let y = pb[key];
                let (av, bv) = if confident_only { (x.agree, y.agree) } else { (x.top3_contained, y.top3_contained) };
                baseline += usize::from(bv); candidate += usize::from(av);
                let (sum, count) = windows.entry(&key.0).or_default();
                sum[0] += f64::from(bv) - f64::from(av);
                *count += 1;
            }
            if windows.is_empty() { return None; }
            let stats: Vec<_> = windows.values().map(|v| v.0).collect();
            let counts: Vec<_> = windows.values().map(|v| v.1).collect();
            let count = counts.iter().sum::<usize>() as f64;
            let samples: Vec<_> = block_bootstrap(&stats, &counts, bootstrap, seed).iter().map(|s| s[0]).collect();
            Some(TripwireMetric { baseline: baseline as f64 / count, candidate: candidate as f64 / count,
                loss: (baseline as f64 - candidate as f64) / count,
                lower95: quantile(&samples, 0.05), upper95: quantile(&samples, 0.95), gross_margin: margin })
        };
        let confident = metric(true, expect.confident_drop_margin);
        let top3 = metric(false, expect.top3_drop_margin).context("missing generated top3 rows")?;
        if confident.as_ref().is_some_and(|m| m.lower95 > m.gross_margin) {
            tripwires.push("paired confident top1 drop exceeds 1 point beyond noise".into());
        }
        if top3.lower95 > top3.gross_margin {
            tripwires.push("paired top3 containment drop exceeds 0.5 point beyond noise".into());
        }
        Some(CalibratedTripwires { confident_top1: confident, top3 })
    } else {
        // Frozen legacy runs retain their original constants and verdicts.
        for (name, r) in [("candidate", a), ("baseline", b)] {
            let f = Fidelity::from_records(r.score.records.iter().filter(|p| p.role == "gen").cloned().collect());
            if f.confident_top1.is_some_and(|v| v < 0.98) { tripwires.push(format!("{name}: confident top1 <98%")); }
            if f.top3_contained < 0.99 { tripwires.push(format!("{name}: reference top3 containment <99%")); }
        }
        None
    };
    let code = |r: &Run| Fidelity::from_records(r.score.records.iter().filter(|p| p.block == "C").cloned().collect());
    let (ac, bc) = (code(a), code(b));
    if ac.positions > 0 && ac.nll - bc.nll > 0.01 { tripwires.push("human code NLL increases >0.01 nat".into()); }
    // One cluster cannot estimate window uncertainty. Never certify by silently treating it as zero.
    let enough_windows = by_window.len() >= 2;
    if !enough_windows { tripwires.push("insufficient windows for clustered uncertainty".into()); }
    Ok(Comparison { pass: enough_windows && absolute_pass && tripwires.is_empty() && top_upper < top1_margin && kl_upper < kl_margin,
        positions: n, windows: by_window.len(), baseline_only: b_only, candidate_only: a_only,
        discordance: discordant as f64 / n as f64, top1_loss: loss, top1_upper95: top_upper,
        top1_se_bootstrap: se, kl_delta, kl_upper95: kl_upper,
        kl_se_clustered: clustered_se(&stats.iter().map(|s| s[1]).collect::<Vec<_>>(), &counts),
        mcnemar_p: if discordant == 0 { 1.0 } else { erfc((chi2 / 2.0).sqrt()) },
        top1_margin, kl_margin, bootstrap, seed, absolute_pass, tripwires, calibrated_tripwires })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn run(windows: usize, rows: usize) -> Run {
        let records = (0..windows).flat_map(|w| (0..rows).map(move |p| Position {
            window: w.to_string(), block: "A".into(), bucket: "0-2K".into(), role: "gen".into(), position: p,
            agree: true, confident: true, top3_contained: true, agree_text: true, finite: true,
            kl: 0.01, nll: 0.5, ref_nll: 0.5, argmax: 1, reference_argmax: 1,
        })).collect();
        Run { schema: "cuteafd.fidelity.run/2".into(), arm: "test".into(), checkpoint: "checkpoint".into(),
            set_sha256: "set".into(), reference_sha256: "reference".into(), tier: "full".into(),
            path_shape: "decode-shaped".into(), kl_kind: "full-vocabulary".into(), verify_rows: Some(8), dataset: None, reference_selection: None, standard_balance: None,
            engine: "test".into(), settings: serde_json::json!({}), seconds: 0.0,
            score: Fidelity::from_records(records), floor_top1: 0.9, floor_kl: 0.06, tripwire_expect: None }
    }
    #[test]
    fn standard_shares_full_decode_verdict_and_rejects_prefill() {
        let full = run(8,512);
        let mut standard = full.clone(); standard.tier = "standard".into();
        assert_eq!(verdict(&full).pass, verdict(&standard).pass);
        assert!(compare(&standard,&standard,0.005,0.005,100,1).unwrap().pass);
        standard.path_shape = "prefill-shaped".into();
        assert!(compare(&standard,&standard,0.005,0.005,100,1).is_err());
        let mut bad = full.clone(); bad.floor_top1 = f64::NAN;
        assert!(!verdict(&bad).pass);
        bad = full.clone(); bad.score.missing = 1; assert!(!verdict(&bad).pass);
        bad = full; bad.dataset = Some(serde_json::json!({"repository":crate::fidelity_dataset::REPOSITORY,
            "revision":crate::fidelity_dataset::RETIRED_FLASH_REVISION,"config":crate::fidelity_dataset::RETIRED_FLASH_CONFIG}));
        let rejected = verdict(&bad);
        assert!(!rejected.pass);
        assert!(rejected.reasons.iter().any(|reason| reason.contains(crate::fidelity_dataset::FLASH_CONFIG)));
        assert!(compare(&bad, &bad, 0.005, 0.005, 100, 1).is_err());
    }

    #[test]
    fn standard_v2_labels_fallback_and_refuses_legacy_or_other_mode_pairing() {
        let mut old = run(8,512); old.tier = "standard".into();
        let mut split = old.clone(); split.tier = crate::fidelity_dataset::STANDARD_TIER.into();
        assert!(compare(&split,&old,0.005,0.005,100,1).is_err());
        split.dataset = Some(serde_json::json!({"standard_subset":{"mode":"32 decode / 32 prefill"}}));
        assert!(compare(&split,&split,0.005,0.005,100,1).unwrap().pass);
        split.path_shape = "prefill-shaped".into();
        assert!(compare(&split,&split,0.005,0.005,100,1).unwrap().pass);
        let mut regressed = split.clone();
        for row in &mut regressed.score.records { row.kl = 0.2; }
        assert!(!verdict(&regressed).pass);
        assert!(verdict(&split).pass);
        let mut fallback = split.clone(); fallback.path_shape = "decode-shaped".into();
        fallback.dataset.as_mut().unwrap()["standard_subset"]["mode"] = serde_json::json!(crate::fidelity_dataset::STANDARD_FALLBACK);
        let verdict = verdict(&fallback);
        assert!(verdict.pass);
        assert_eq!(verdict.label.as_deref(),Some(crate::fidelity_dataset::STANDARD_FALLBACK));
        let report = crate::panels::fidelity::record(&fallback,None);
        assert_eq!(report["mode"],crate::fidelity_dataset::STANDARD_FALLBACK);
        assert_eq!(report["verdict"]["label"],crate::fidelity_dataset::STANDARD_FALLBACK);
        split.path_shape = "decode-shaped".into();
        assert!(compare(&split,&fallback,0.005,0.005,100,1).is_err());
    }

    #[test]
    fn standard_balance_is_optional_and_preserves_saved_run_comparability() {
        let mut legacy = run(8,512);
        for tier in ["quick", "standard", "full", crate::fidelity_dataset::STANDARD_TIER] {
            legacy.tier = tier.into();
            let saved = serde_json::to_value(&legacy).unwrap();
            assert!(saved.get("standard_balance").is_none());
            assert!(serde_json::from_value::<Run>(saved).unwrap().standard_balance.is_none());
        }
        legacy.dataset = Some(serde_json::json!({"standard_subset":{"mode":"32 decode / 32 prefill"}}));
        let mut current = legacy.clone();
        current.standard_balance = Some(serde_json::json!({
            "decode":{"context_buckets":{"0-2K":13,"2-8K":10,"8-16K":9},"generated_positions":8786},
            "prefill":{"context_buckets":{"0-2K":16,"2-8K":7,"8-16K":9},"generated_positions":7832}}));
        let saved = serde_json::to_value(&current).unwrap();
        assert_eq!(saved["standard_balance"], current.standard_balance.as_ref().unwrap().clone());
        assert!(saved["dataset"]["standard_subset"].get("balance").is_none());
        assert_eq!(serde_json::to_vec(&current.dataset).unwrap(), serde_json::to_vec(&legacy.dataset).unwrap());
        let restored: Run = serde_json::from_value(saved).unwrap();
        assert_eq!(restored.standard_balance, current.standard_balance);
        assert!(compare(&legacy, &restored, 0.005, 0.005, 100, 1).unwrap().pass);
    }

    #[test]
    fn fixed_mimo_publication_uses_calibrated_bounds() {
        let mut fixed = run(8, 512);
        fixed.dataset = Some(serde_json::json!({"repository":crate::fidelity_dataset::REPOSITORY,
            "revision":crate::fidelity_dataset::FLASH_REVISION,"config":crate::fidelity_dataset::FLASH_CONFIG}));
        fixed.floor_top1 = 0.93;
        fixed.floor_kl = 0.04;
        fixed.tripwire_expect = Some(crate::reference::TripwireExpect {
            confident_top1_min:0.96,top3_min:0.97,confident_drop_margin:0.01,top3_drop_margin:0.005 });
        let result = verdict(&fixed);
        assert!(result.pass);
        assert_eq!((result.top1_min,result.kl_max,result.confident_top1_min,result.top3_min),
            (0.93,0.04,0.96,0.97));
        assert!(compare(&fixed, &fixed, 0.005, 0.005, 100, 1).unwrap().pass);
    }

    #[test]
    fn full_decision_requires_both_paths_and_fixed_arms() {
        let decode = run(12, 512);
        let mut prefill = decode.clone();
        prefill.path_shape = "prefill-shaped".into();
        prefill.verify_rows = None;
        assert!(compare_full(&decode, &decode, &prefill, &prefill, 100, 1).unwrap().pass);
        assert!(compare_full(&decode, &decode, &decode, &prefill, 100, 1).is_err());
        let mut quick = decode.clone(); quick.tier = "quick".into();
        assert!(compare_full(&quick, &decode, &prefill, &prefill, 100, 1).is_err());
        let mut changed = prefill.clone(); changed.settings = serde_json::json!({"head": "different"});
        assert!(compare_full(&decode, &decode, &changed, &prefill, 100, 1).is_err());
        changed = prefill.clone(); changed.arm = "different".into();
        assert!(compare_full(&decode, &decode, &changed, &prefill, 100, 1).is_err());
        let mut regressed = prefill.clone();
        for p in &mut regressed.score.records { if p.position < 6 { p.agree = false; } }
        let result = compare_full(&decode, &decode, &regressed, &prefill, 100, 1).unwrap();
        assert!(result.decode.pass); assert!(!result.prefill.pass); assert!(!result.pass);
    }

    #[test]
    fn clustered_se_by_hand() {
        let wanted = (1.5f64 * (1.5625 + 0.0625 + 2.25)).sqrt() / 4.0;
        assert!((clustered_se(&[1.0, 2.0, 6.0], &[1, 1, 2]).unwrap() - wanted).abs() < 1e-15);
    }
    #[test]
    fn identical_runs_pass_with_zero_delta() {
        let r = run(12, 512);
        let c = compare(&r, &r, 0.005, 0.005, 5000, 20260829).unwrap();
        assert!(c.pass); assert_eq!(c.top1_loss, 0.0); assert_eq!(c.kl_upper95, 0.0);
    }
    #[test]
    fn one_point_regression_fails_full_gate_and_small_panel_cannot_certify_it() {
        for windows in [64, 1] {
            let b = run(windows, 512);
            let mut a = b.clone();
            for p in &mut a.score.records { if p.position < 6 { p.agree = false; } }
            let c = compare(&a, &b, 0.005, 0.005, 5000, 1).unwrap();
            assert!(!c.pass); assert!(c.top1_loss > 0.01);
        }
    }
    #[test]
    fn pairing_rejects_missing_duplicates_and_changed_provenance() {
        let b = run(12, 8);
        let mut a = b.clone(); a.checkpoint = "other".into();
        assert!(compare(&a, &b, 0.01, 0.01, 100, 1).is_err());
        a = b.clone(); a.score.records.pop();
        assert!(compare(&a, &b, 0.01, 0.01, 100, 1).is_err());
        a = b.clone(); a.score.records[0] = a.score.records[1].clone();
        assert!(compare(&a, &b, 0.01, 0.01, 100, 1).is_err());
    }
    #[test]
    fn pairing_requires_same_build_and_quick_decode_shape() {
        let b = run(3, 8);
        let mut a = b.clone(); a.settings = serde_json::json!({"build": {"commit": "other"}});
        assert!(compare(&a, &b, 0.01, 0.01, 100, 1).is_err());
        a = b.clone(); a.path_shape = "unknown".into();
        assert!(compare(&a, &a, 0.01, 0.01, 100, 1).is_err());
        a = b.clone(); a.tier = "quick".into(); a.path_shape = "prefill-shaped".into();
        assert!(compare(&a, &a, 0.01, 0.01, 100, 1).is_err());
    }
    #[test]
    fn pairing_allows_only_precision_switches_to_change() {
        let mut b = run(3, 8);
        b.settings = serde_json::json!({"model": "checkpoint", "settings": [
            {"name": "concurrency", "value": "1", "source": "cli"},
            {"name": "CUTEAFD_V41_FP8_HEAD", "value": "off", "source": "env"}]});
        let mut a = b.clone();
        a.settings["settings"][1]["value"] = serde_json::json!("all");
        assert!(compare(&a, &b, 0.005, 0.005, 100, 1).unwrap().pass);
        a.settings["settings"].as_array_mut().unwrap().reverse();
        assert!(compare(&a, &b, 0.005, 0.005, 100, 1).unwrap().pass);
        a.settings["settings"][1]["source"] = serde_json::json!("default");
        assert!(compare(&a, &b, 0.005, 0.005, 100, 1).unwrap().pass);
        a.settings["settings"][1]["value"] = serde_json::json!("4");
        assert!(compare(&a, &b, 0.005, 0.005, 100, 1).is_err());
        a = b.clone();
        a.settings["settings"].as_array_mut().unwrap().push(serde_json::json!({"name": "CUTEAFD_UNKNOWN_FP8", "value": "on"}));
        assert!(compare(&a, &b, 0.005, 0.005, 100, 1).is_err());
        for name in ["concurrency", "CUTEAFD_V41_FP8_HEAD"] {
            a = b.clone();
            a.settings["settings"].as_array_mut().unwrap().push(serde_json::json!({"name": name, "value": "1"}));
            assert!(compare(&a, &a, 0.005, 0.005, 100, 1).is_err());
        }
        a = b.clone(); a.settings["settings"][0].as_object_mut().unwrap().remove("value");
        assert!(compare(&a, &a, 0.005, 0.005, 100, 1).is_err());
        a = b.clone(); a.settings["settings"] = serde_json::json!({});
        assert!(compare(&a, &a, 0.005, 0.005, 100, 1).is_err());
        a = b.clone(); a.settings["family"] = serde_json::json!("other");
        assert!(compare(&a, &b, 0.005, 0.005, 100, 1).is_err());
    }

    #[test]
    fn pairing_accepts_glm_flash_kda_state_and_target_head_but_not_layout_or_scheduling() {
        // A GLM 5.3 Flash baseline at checkpoint precision: FP32 KDA state, exact target head.
        let mut b = run(3, 8);
        b.settings = serde_json::json!({"model": "checkpoint", "settings": [
            {"name": "concurrency", "value": "4", "source": "cli"},
            {"name": "index-cache", "value": "keys", "source": "default"},
            {"name": "kda-state", "value": "f32", "source": "default"},
            {"name": "target-head", "value": "exact", "source": "default"}]});
        let with = |kda: &str, head: &str| {
            let mut a = b.clone();
            a.settings["settings"][2] = serde_json::json!({"name": "kda-state", "value": kda, "source": "cli"});
            a.settings["settings"][3] = serde_json::json!({"name": "target-head", "value": head, "source": "cli"});
            a
        };
        // Either precision switch, or both, may differ, on each scoring shape and in a full decision.
        for (kda, head) in [("bf16", "exact"), ("bf16-tile", "exact"), ("f32", "tensor"), ("bf16", "tensor")] {
            let a = with(kda, head);
            assert!(compare(&a, &b, 0.005, 0.005, 100, 1).unwrap().pass, "{kda} / {head}");
            let (mut a_prefill, mut b_prefill) = (a.clone(), b.clone());
            for r in [&mut a_prefill, &mut b_prefill] {
                r.path_shape = "prefill-shaped".into();
                r.verify_rows = None;
            }
            assert!(compare_full(&a, &b, &a_prefill, &b_prefill, 100, 1).unwrap().pass, "{kda} / {head}");
        }
        // A nonprecision difference beside them is still refused: a cache layout ...
        let mut a = with("bf16", "tensor");
        a.settings["settings"][1]["value"] = serde_json::json!("compact");
        let error = compare(&a, &b, 0.005, 0.005, 100, 1).unwrap_err().to_string();
        assert!(error.contains("different nonprecision server settings"), "{error}");
        // ... or scheduling.
        a = with("bf16", "exact");
        a.settings["settings"][0]["value"] = serde_json::json!("16");
        let error = compare(&a, &b, 0.005, 0.005, 100, 1).unwrap_err().to_string();
        assert!(error.contains("different nonprecision server settings"), "{error}");
        // The precision settings themselves still need a value, once.
        a = with("bf16", "exact");
        a.settings["settings"][2].as_object_mut().unwrap().remove("value");
        assert!(compare(&a, &b, 0.005, 0.005, 100, 1).is_err());
        a = with("bf16", "exact");
        a.settings["settings"].as_array_mut().unwrap().push(serde_json::json!({"name": "kda-state", "value": "f32"}));
        assert!(compare(&a, &b, 0.005, 0.005, 100, 1).is_err());
    }

    #[test]
    fn tripwires_cannot_pass_a_numerically_identical_pair() {
        let mut r = run(3, 100);
        for p in r.score.records.iter_mut().filter(|p| p.position < 3) { p.agree = false; }
        r.score = Fidelity::from_records(r.score.records);
        let c = compare(&r, &r, 0.005, 0.005, 100, 1).unwrap();
        assert!(c.absolute_pass); assert_eq!(c.top1_loss, 0.0); assert!(!c.pass);
        assert!(c.tripwires.iter().any(|s| s.contains("confident")));
    }
    fn calibrated_run() -> Run {
        let mut r = run(4, 1000);
        r.tripwire_expect = Some(crate::reference::TripwireExpect {
            confident_top1_min: 0.95, top3_min: 0.97,
            confident_drop_margin: 0.01, top3_drop_margin: 0.005,
        });
        r
    }

    #[test]
    fn family_tripwires_allow_qualified_regimes_below_legacy_constants() {
        let mut r = calibrated_run();
        for p in &mut r.score.records {
            p.agree = p.position >= 30;
            p.top3_contained = p.position >= 20;
        }
        let c = compare(&r, &r, 0.005, 0.005, 100, 1).unwrap();
        assert!(c.pass);
        let metrics = c.calibrated_tripwires.unwrap();
        let confident = metrics.confident_top1.unwrap();
        assert_eq!(confident.baseline, 0.97);
        assert_eq!(confident.candidate, 0.97);
        assert_eq!(confident.loss, 0.0);
        assert_eq!(confident.lower95, 0.0);
        assert_eq!(confident.upper95, 0.0);
        assert_eq!(metrics.top3.baseline, 0.98);
        r.tripwire_expect = None;
        let legacy = compare(&r, &r, 0.005, 0.005, 100, 1).unwrap();
        assert!(!legacy.pass);
        assert!(legacy.calibrated_tripwires.is_none());
        assert!(legacy.tripwires.iter().any(|s| s.contains("<98%")));
        assert!(legacy.tripwires.iter().any(|s| s.contains("<99%")));
    }

    #[test]
    fn family_tripwires_reject_gross_not_merely_detectable_drops() {
        let baseline = calibrated_run();
        for (flips, gross) in [(4, false), (12, true)] {
            let mut candidate = baseline.clone();
            for p in &mut candidate.score.records {
                p.agree = p.position >= flips;
                p.top3_contained = p.position >= flips;
            }
            let c = compare(&candidate, &baseline, 0.005, 0.005, 100, 1).unwrap();
            assert_eq!(c.tripwires.iter().any(|s| s.contains("paired confident")), gross);
            assert_eq!(c.tripwires.iter().any(|s| s.contains("paired top3")), gross);
            assert_eq!(c.pass, !gross);
            let m = c.calibrated_tripwires.unwrap().confident_top1.unwrap();
            assert!((m.lower95 - flips as f64 / 1000.0).abs() < 1e-12);
            assert!((m.upper95 - m.lower95).abs() < 1e-12);
        }
        let mut top3 = baseline.clone();
        for p in &mut top3.score.records { p.top3_contained = p.position >= 6; }
        let c = compare(&top3, &baseline, 0.005, 0.005, 100, 1).unwrap();
        assert!(!c.pass);
        assert!(c.tripwires.iter().any(|s| s.contains("paired top3")));
        assert!(!c.tripwires.iter().any(|s| s.contains("paired confident")));
    }

    #[test]
    fn calibrated_runs_report_absent_confident_subset_and_legacy_json_roundtrip() {
        let mut r = calibrated_run();
        for p in &mut r.score.records { p.confident = false; }
        let c = compare(&r, &r, 0.005, 0.005, 100, 1).unwrap();
        assert!(c.pass);
        assert!(c.calibrated_tripwires.unwrap().confident_top1.is_none());
        r.tripwire_expect = None;
        let value = serde_json::to_value(&r).unwrap();
        assert!(value.get("tripwire_expect").is_none());
        let legacy: Run = serde_json::from_value(value).unwrap();
        assert!(legacy.tripwire_expect.is_none());
        let c = compare(&legacy, &legacy, 0.005, 0.005, 100, 1).unwrap();
        assert!(serde_json::to_value(c).unwrap().get("calibrated_tripwires").is_none());
    }

    #[test]
    fn family_tripwire_calibration_must_match_and_be_valid() {
        let r = calibrated_run();
        let mut other = r.clone();
        other.tripwire_expect.as_mut().unwrap().top3_min = 0.96;
        assert!(compare(&r, &other, 0.005, 0.005, 100, 1).is_err());
        for value in [f64::NAN, -0.1, 1.1] {
            other = r.clone();
            other.tripwire_expect.as_mut().unwrap().confident_top1_min = value;
            assert!(compare(&other, &other, 0.005, 0.005, 100, 1).is_err());
        }
        other = r.clone();
        other.tripwire_expect.as_mut().unwrap().confident_drop_margin = 0.0;
        assert!(compare(&other, &other, 0.005, 0.005, 100, 1).is_err());
        let mut prefill = r.clone(); prefill.path_shape = "prefill-shaped".into();
        prefill.tripwire_expect = None;
        assert!(compare_full(&r, &r, &prefill, &prefill, 100, 1).is_err());
        other = r.clone();
        for p in &mut other.score.records { p.top3_contained = p.position >= 31; }
        let c = compare(&other, &other, 0.005, 0.005, 100, 1).unwrap();
        assert!(c.tripwires.iter().any(|s| s.contains("family calibrated minimum")));
        assert!(!c.pass);
    }

    #[test]
    fn bootstrap_is_reproducible_and_preserves_unequal_counts() {
        let stats = [[1.0, 2.0], [4.0, 8.0], [6.0, 12.0]];
        let a = block_bootstrap(&stats, &[1, 2, 3], 5000, 7);
        assert_eq!(a, block_bootstrap(&stats, &[1, 2, 3], 5000, 7));
        assert!(a.iter().all(|s| s[1] == 2.0 * s[0]));
        assert_eq!(quantile(&[0.0, 10.0], 0.95), 9.5);
    }
}
