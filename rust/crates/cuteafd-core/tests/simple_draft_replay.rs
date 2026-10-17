//! Offline replay of a recorded draft-round trace (`SPECULATION_TRACE`
//! `draft_round` records, written by the default shared policy) through the
//! simple draft rules, against what the default policy chose.
//!
//! Each round, `simple` and `simple-derived` choose lengths from the recorded
//! raw confidences, then observe the round as it executed (calibration, `r`
//! and `c` evolve as they would have from the same evidence). Their choices
//! are scored with:
//! - accepted tokens: exact where the recorded verification covers the chosen
//!   length (a chosen length past a recorded rejection keeps the recorded
//!   accepted count; one short of it accepts every chosen draft); past the
//!   recorded length when every recorded draft was accepted, each extra
//!   position adds the policy's own calibrated survival (reported apart as
//!   `censored_tokens`);
//! - round time: `T = a0 + a1 * C + r * rows`, least squares over the whole
//!   trace's recorded rounds (its residual is reported).
//!
//! Output: one JSON summary per concurrency group and arm on stderr.
//! Run: `CUTEAFD_SIMPLE_REPLAY=trace.jsonl [CUTEAFD_SIMPLE_REPLAY_ROWS=64] cargo test -p cuteafd-core --release
//! --test simple_draft_replay -- --ignored --nocapture`
use cuteafd_core::{DraftCandidate, DraftMode, DraftPolicy, DraftRoundObservation, LayerResource, ObservedDraftRequest,
    PolicyGeometry, PolicySeed, ResourceClass};
use serde_json::Value;
use std::io::BufRead;

fn geometry(record: &Value, max_positions: usize, native: usize) -> PolicyGeometry {
    let layers: Vec<LayerResource> = record["geometry"].as_array().unwrap().iter().map(|l| LayerResource {
        class: l["class"].as_u64().map(|c| c as u8),
        slice_bytes: l["slice_bytes"].as_f64().unwrap(),
        group_rows: l["group_rows"].as_u64().unwrap() as u8,
        timed: l["timed"].as_bool().unwrap(),
    }).collect();
    let topk = record["route_layout"][2].as_u64().unwrap() as u8;
    PolicyGeometry { layers, experts: 1024, topk, max_requests: 64, max_positions, widths: vec![native],
        classes: vec![ResourceClass::rtx(), ResourceClass::spark()], group_rows_by_rows: Vec::new(), regimes: 2 }
}

/// Least squares `T = a0 + a1 * C + r * rows` and its relative RMS residual.
fn cost_truth(rounds: &[(f64, f64, f64)]) -> ([f64; 3], f64) {
    let mut m = [[0.; 3]; 3];
    let mut y = [0.; 3];
    for &(c, rows, t) in rounds {
        let x = [1., c, rows];
        for i in 0..3 { for j in 0..3 { m[i][j] += x[i] * x[j]; } y[i] += x[i] * t; }
    }
    m[1][1] += 1e-6;
    // Gaussian elimination, no pivoting needed for a normal matrix.
    for col in 0..3 {
        for row in 0..3 {
            if row == col { continue; }
            let f = m[row][col] / m[col][col];
            for k in 0..3 { m[row][k] -= f * m[col][k]; }
            y[row] -= f * y[col];
        }
    }
    let theta = [y[0] / m[0][0], y[1] / m[1][1], y[2] / m[2][2]];
    let (mut err, mut sum) = (0., 0.);
    for &(c, rows, t) in rounds {
        let p = theta[0] + theta[1] * c + theta[2] * rows;
        err += (p - t) * (p - t);
        sum += t * t;
    }
    (theta, (err / sum).sqrt())
}

#[derive(Clone, Copy, Default, Debug)]
struct Score { rounds: u64, rounds_requests: f64, tokens: f64, censored: f64, rows: f64, time_us: f64, drafts: f64 }

fn group(requests: usize) -> usize { match requests { 0..=1 => 0, 2..=4 => 1, 5..=8 => 2, _ => 3 } }

#[test]
#[ignore]
fn replay_simple_rules_against_recorded_choices() {
    let Ok(path) = std::env::var("CUTEAFD_SIMPLE_REPLAY") else { return };
    let max_rows: usize = std::env::var("CUTEAFD_SIMPLE_REPLAY_ROWS").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    let records: Vec<Value> = std::io::BufReader::new(std::fs::File::open(&path).unwrap()).lines()
        .map_while(Result::ok).filter_map(|l| serde_json::from_str::<Value>(&l).ok())
        .filter(|r| r["type"] == "draft_round").collect();
    let positions = records.iter().flat_map(|r| r["requests"].as_array().unwrap().iter())
        .filter_map(|q| q["raw_confidence"].as_array().map(Vec::len)).max().unwrap_or(5).max(1);
    let native = records.iter().filter_map(|r| r["draft_width"].as_u64()).max().unwrap_or(positions as u64) as usize;
    let truth: Vec<(f64, f64, f64)> = records.iter().filter_map(|r| r["total_us"].as_f64()
        .map(|t| (r["requests"].as_array().unwrap().len() as f64, r["rows"].as_f64().unwrap(), t))).collect();
    let (theta, residual) = cost_truth(&truth);
    eprintln!("{path}: {} rounds, {positions} positions, native {native}; cost truth T = {:.0} + {:.0} C + {:.0} rows µs \
        (relative RMS residual {residual:.3})", records.len(), theta[0], theta[1], theta[2]);
    let time = |c: usize, rows: usize| theta[0] + theta[1] * c as f64 + theta[2] * rows as f64;
    let mut arms: Vec<(&str, Option<DraftPolicy>)> = vec![("recorded", None)];
    for (label, mode) in [("simple", DraftMode::Simple), ("simple-rc", DraftMode::SimpleRc),
        ("simple-derived", DraftMode::SimpleDerived)] {
        let mut policy = DraftPolicy::new(geometry(&records[0], positions, native), PolicySeed::weak(native), false).unwrap();
        policy.set_draft_mode(mode);
        arms.push((label, Some(policy)));
    }
    let mut scores = vec![[Score::default(); 4]; arms.len()];
    let mut cutoffs: Vec<[Vec<f64>; 4]> = vec![Default::default(); arms.len()];
    let mut row_token: Vec<[Vec<(f64, f64)>; 4]> = vec![Default::default(); arms.len()];
    for record in &records {
        let requests = record["requests"].as_array().unwrap();
        let ids: Vec<u64> = requests.iter().map(|q| q["id"].as_u64().unwrap()).collect();
        let raw: Vec<Vec<f64>> = requests.iter().map(|q| if q["source"] == "Neural" {
            q["raw_confidence"].as_array().map_or_else(Vec::new, |c| c.iter().map(|v| v.as_f64().unwrap()).collect())
        } else { Vec::new() }).collect();
        let verified: Vec<usize> = requests.iter().map(|q| q["verified_rows"].as_u64().unwrap() as usize).collect();
        let accepted: Vec<usize> = requests.iter().map(|q| q["accepted_rows"].as_u64().unwrap() as usize).collect();
        let censored: Vec<bool> = requests.iter().map(|q| !q["censor"].is_null()).collect();
        let copied = requests.iter().any(|q| q["source"] == "Copy");
        // Score only rounds where every request's drafts came from the drafter.
        let scorable = !copied && record["total_us"].is_number() && record["draft_width"].is_u64();
        let g = group(ids.len());
        for (arm, (_, policy)) in arms.iter_mut().enumerate() {
            let lengths: Vec<usize> = match policy {
                None => verified.iter().map(|v| v - 1).collect(),
                Some(policy) => {
                    let candidates: Vec<_> = ids.iter().zip(&raw).map(|(&id, c)| DraftCandidate { id, confidence: c }).collect();
                    let selection = policy.select_drafted_within(false, &candidates, native, max_rows, ids.len()).unwrap();
                    if let Some(c) = policy.simple_cutoff(ids.len()) {
                        cutoffs[arm][g].push(c.value);
                        if let (Some(r), Some(t)) = (c.row_us, c.token_us) { row_token[arm][g].push((r, t)); }
                    }
                    selection.map_or_else(|| vec![0; ids.len()], |s| s.lengths)
                }
            };
            if !scorable { continue; }
            let score = &mut scores[arm][g];
            score.rounds += 1;
            score.rounds_requests += ids.len() as f64;
            let rows = ids.len() + lengths.iter().sum::<usize>();
            score.rows += rows as f64;
            score.drafts += lengths.iter().sum::<usize>() as f64;
            score.time_us += time(ids.len(), rows);
            for i in 0..ids.len() {
                let (drafts, hits) = (verified[i] - 1, accepted[i] - 1);
                let length = lengths[i].min(raw[i].len().max(drafts));
                if hits < drafts || censored[i] || length <= drafts {
                    // The recorded outcome decides: first rejection at hits + 1.
                    score.tokens += 1. + length.min(hits) as f64;
                } else {
                    // Every recorded draft was accepted; extend by calibrated survival.
                    score.tokens += 1. + hits as f64;
                    let p = arms_probabilities(policy, ids.len(), ids[i], &raw[i]);
                    let mut survival = 1.;
                    for k in drafts..length {
                        survival *= p.get(k).copied().unwrap_or(0.);
                        score.tokens += survival;
                        score.censored += survival;
                    }
                }
            }
        }
        // Every simple arm learns from the round as it executed.
        let confidences: Vec<Option<&[f64]>> = requests.iter().zip(&raw).zip(&accepted).map(|((q, c), &a)|
            (!c.is_empty()).then(|| if q["censor"].is_null() { &c[..] } else { &c[..c.len().min(a - 1)] })).collect();
        let observed: Vec<_> = ids.iter().zip(&verified).zip(&accepted).zip(&confidences)
            .map(|(((&id, &rows), &accepted), &confidence)| ObservedDraftRequest { id, rows, accepted, confidence }).collect();
        let rows: usize = verified.iter().sum();
        let layers = record["route_layout"][0].as_u64().unwrap() as usize;
        let topk = record["route_layout"][2].as_u64().unwrap() as usize;
        let routes = vec![0u16; layers * rows * topk];
        let layer_us = vec![None; layers];
        for (_, policy) in arms.iter_mut() {
            let Some(policy) = policy else { continue };
            let observation = DraftRoundObservation { shared: false, requests: &observed, routes: &routes,
                layer_us: &layer_us, total_us: record["total_us"].as_f64().unwrap_or(f64::NAN), draft_us: f64::NAN,
                drafted_requests: None, wide: false, predicted_us: None };
            policy.observe(observation).unwrap();
        }
    }
    let labels = ["c1", "c2-4", "c5-8", "c9+"];
    for (arm, (label, _)) in arms.iter().enumerate() {
        for (g, score) in scores[arm].iter().enumerate().filter(|(_, s)| s.rounds > 0) {
            let mut sorted = cutoffs[arm][g].clone();
            sorted.sort_by(f64::total_cmp);
            let median = sorted.get(sorted.len() / 2).copied();
            let quantile = |q: f64| sorted.get(((sorted.len() as f64 * q) as usize).min(sorted.len().saturating_sub(1))).copied();
            let n = row_token[arm][g].len().max(1) as f64;
            let (r, c) = row_token[arm][g].iter().fold((0., 0.), |(a, b), (r, c)| (a + r / n, b + c / n));
            let base = scores[0][g];
            let tok_s = score.tokens / score.time_us * 1e6;
            eprintln!("{}", serde_json::json!({"arm": label, "group": labels[g], "rounds": score.rounds,
                "tok_per_s": tok_s, "vs_recorded": tok_s / (base.tokens / base.time_us * 1e6),
                "tokens_per_round": score.tokens / score.rounds as f64,
                "drafts_per_round": score.drafts / score.rounds as f64,
                "rows_per_round": score.rows / score.rounds as f64,
                "tokens_per_row": score.tokens / score.rows,
                // Extrapolated tokens counted as none: a lower bound.
                "tok_per_s_floor": (score.tokens - score.censored) / score.time_us * 1e6,
                "vs_recorded_floor": (score.tokens - score.censored) / score.time_us * 1e6 / (base.tokens / base.time_us * 1e6),
                "wasted_drafts_per_round": (score.drafts - (score.tokens - score.rounds_requests)) / score.rounds as f64,
                "censored_share": score.censored / score.tokens, "median_cutoff": median,
                "cutoff_p10": quantile(0.1), "cutoff_p90": quantile(0.9), "mean_row_us": r, "mean_token_us": c}));
        }
    }
}

fn arms_probabilities(policy: &Option<DraftPolicy>, requests: usize, id: u64, raw: &[f64]) -> Vec<f64> {
    match policy {
        Some(policy) => policy.probabilities(requests, DraftCandidate { id, confidence: raw }),
        None => Vec::new(),
    }
}
