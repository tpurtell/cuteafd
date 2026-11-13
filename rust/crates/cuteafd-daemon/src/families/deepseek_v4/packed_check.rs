//! Reference-aligned diagnostic through the real segmented serving executor.
use super::{engine::Engine, pool::PoolAllocator, prefix, SparkLink};
use crate::shared::prefill_packing::{self, PackingMode};
use anyhow::{ensure, Result};
use cuteafd_api::openai::probe::{Probe, ProbeSpec};
use cuteafd_bench::reference::{Fidelity, Window};

struct Observed { logits: Vec<f32>, state: prefix::PackedState, frontier: usize }

fn widths(case: usize, member: usize) -> usize {
    match case { 0 => [29, 31, 23][member % 3], _ => [200, 127, 64][member % 3] }
}

fn arm(engine: &Engine<'_>, links: &mut [SparkLink<'_>], runtime: &tokio::runtime::Runtime,
    prompts: &[Vec<u32>], starts: &[usize], case: usize, packed: bool) -> Result<(Vec<Observed>, usize)> {
    let mut allocator = PoolAllocator::for_engine(engine)?;
    let mut placements = prompts.iter().map(|p| allocator.admit(p.len() + 1)).collect::<Result<Vec<_>>>()?;
    for placement in &placements { prefix::packed_state(engine, placement, true)?; }
    ensure!(starts.len() == prompts.len(), "diagnostic start count differs");
    for ((p, tokens), &start) in placements.iter_mut().zip(prompts).zip(starts) {
        ensure!(start < tokens.len(), "diagnostic start past prompt");
        if start > 0 { engine.prefill(p, &tokens[..start], links, runtime, 0, None)?; }
    }
    let mut logits = vec![Vec::new(); prompts.len()];
    let mut actual_packs = 0;
    while placements.iter().zip(prompts).any(|(p, t)| p.len < t.len()) {
        let active = placements.iter().zip(prompts).enumerate()
            .filter_map(|(i, (p, t))| (p.len < t.len()).then_some(i)).collect::<Vec<_>>();
        let chunks = active.iter().map(|&i| (placements[i].len,
            widths(case, i).min(prompts[i].len() - placements[i].len))).collect::<Vec<_>>();
        let groups = if packed { prefill_packing::groups(active.len(), |r| engine.packed_plan(&chunks[r]).is_ok()) }
            else { (0..active.len()).map(|i| i..i + 1).collect() };
        for range in groups {
            if range.len() == 1 {
                let i = active[range.start];
                let (start, rows) = chunks[range.start];
                let consumer = crate::shared::probe::CallbackRows::new(|position, values: &[f32]| {
                    ensure!(position == starts[i] + logits[i].len() / engine.cfg.vocab_size + 1,
                        "packed singleton scoring row order differs");
                    logits[i].extend_from_slice(values);
                    Ok(())
                });
                engine.prefill_consumer(&mut placements[i], &prompts[i][start..start + rows],
                    links, None, crate::shared::probe::ScoringRows { consumer: &consumer, position: start + 1 })?;
            } else {
                actual_packs += 1;
                let ids = &active[range.clone()];
                // Active indices are ordered; collect disjoint mutable placements without unsafe.
                let mut members = placements.iter_mut().enumerate().filter_map(|(i, p)| {
                    ids.iter().position(|&id| id == i).map(|j| {
                        let (start, rows) = chunks[range.start + j];
                        (p, &prompts[i][start..start + rows])
                    })
                }).collect::<Vec<_>>();
                let ends = ids.iter().enumerate().map(|(j, &i)| {
                    let (start, rows) = chunks[range.start + j]; start + rows == prompts[i].len()
                }).collect::<Vec<_>>();
                for result in engine.prefill_packed_check(&mut members, links, &ends, &mut |j, _, rows| {
                    logits[ids[j]].extend(rows.to_host(engine.library)?);
                    Ok(())
                })? { result?; }
            }
        }
    }
    let observed = placements.iter().zip(logits).map(|(p, logits)| Ok(Observed {
        logits, state: prefix::packed_state(engine, p, false)?, frontier: p.len,
    })).collect::<Result<Vec<_>>>()?;
    Ok((observed, actual_packs))
}

fn equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}

fn require_identical(a: &Observed, b: &Observed, vocab: usize, label: &str) -> Result<()> {
    let logit = a.logits.iter().zip(&b.logits).position(|(x, y)| x.to_bits() != y.to_bits());
    if let Some(i) = logit {
        println!("packed byte divergence {label}: logits row {} token {} singleton_bits {:08x} packed_bits {:08x}",
            i / vocab, i % vocab, a.logits[i].to_bits(), b.logits[i].to_bits());
    }
    let state = a.state.bytes.iter().zip(&b.state.bytes).position(|(x, y)| x != y);
    if let Some(i) = state {
        let (name, range) = b.state.tensors.iter().find(|(_, range)| range.contains(&i))
            .ok_or_else(|| anyhow::anyhow!("packed state offset {i} has no tensor"))?;
        println!("packed byte divergence {label}: state tensor {name} byte {} singleton {:02x} packed {:02x}",
            i - range.start, a.state.bytes[i], b.state.bytes[i]);
    }
    ensure!(a.logits.len() == b.logits.len() && a.state.bytes.len() == b.state.bytes.len()
        && a.state.tensors == b.state.tensors && logit.is_none() && state.is_none(),
        "{label} differs from singleton bytes (logits/state)");
    Ok(())
}

fn pair_metrics(a: &[f32], b: &[f32], vocab: usize) -> (f64, f64) {
    let normalized = |row: &[f32]| {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = max + row.iter().map(|&x| (f64::from(x) - max).exp()).sum::<f64>().ln();
        row.iter().map(|&x| f64::from(x) - lse).collect::<Vec<_>>()
    };
    let argmax = |row: &[f32]| row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let mut kl = 0.0;
    let mut same = 0;
    for (a, b) in a.chunks_exact(vocab).zip(b.chunks_exact(vocab)) {
        same += usize::from(argmax(a) == argmax(b));
        let (a, b) = (normalized(a), normalized(b));
        kl += a.iter().zip(b).map(|(a, b)| a.exp() * (a - b)).sum::<f64>();
    }
    let rows = (a.len() / vocab) as f64;
    (kl.max(0.0) / rows, same as f64 / rows)
}

fn reference_score(window: &Window, observed: &Observed, vocab: usize) -> Fidelity {
    let probe = Probe::new(ProbeSpec { score_from: Some(window.score_from),
        score_positions: Some(window.positions.iter().map(|p| p.pos).collect()),
        want: window.want(), top_k: 32, ..Default::default() });
    for (row, logits) in observed.logits.chunks_exact(vocab).enumerate() {
        probe.row(window.score_from + row, logits);
    }
    window.score(&probe.record().rows)
}

fn report_reference(control: &Fidelity, packed: &Fidelity, label: &str) {
    let delta_kl = packed.kl - control.kl;
    let delta_top1 = packed.top1 - control.top1;
    println!("packed reference {label}: singleton KL {:.8}/top1 {:.6}, packed KL {:.8}/top1 {:.6}, delta KL {delta_kl:.8}/top1 {delta_top1:.6}, positions {}",
        control.kl, control.top1, packed.kl, packed.top1, packed.positions);
}

fn valid_score(score: &Fidelity, positions: usize) -> bool {
    score.positions == positions && positions > 0 && score.missing == 0
        && score.non_finite == 0 && score.kl.is_finite() && score.top1.is_finite()
}

fn noninferior_with_noise(control: &Fidelity, repeat: &Fidelity, packed: &Fidelity, label: &str) -> Result<()> {
    report_reference(control, repeat, &format!("{label} singleton A vs B noise"));
    report_reference(control, packed, label);
    let noise_kl = repeat.kl - control.kl;
    let noise_top1 = repeat.top1 - control.top1;
    let delta_kl = packed.kl - control.kl;
    let delta_top1 = packed.top1 - control.top1;
    println!("packed noise-adjusted {label}: excess delta KL {:.8}/top1 {:.6}, margins .005/-.01",
        delta_kl - noise_kl, delta_top1 - noise_top1);
    ensure!(valid_score(control, control.positions) && valid_score(repeat, control.positions)
        && valid_score(packed, control.positions)
        && delta_kl <= noise_kl + 0.005 + 1e-12 && delta_top1 >= noise_top1 - 0.01 - 1e-12,
        "{label} failed noise-controlled singleton non-inferiority");
    Ok(())
}

fn exact_with_noise(control: &Observed, repeat: &Observed, packed: &Observed, vocab: usize, label: &str) -> Result<()> {
    if require_identical(control, repeat, vocab, &format!("{label} singleton A vs B")).is_err() {
        println!("packed exact {label}: byte gate bounded by #886 singleton run-to-run nondeterminism; not byte-qualified");
        // Preserve packed mismatch diagnostics even when the singleton control is unstable.
        let _ = require_identical(control, packed, vocab, &format!("{label} singleton A vs packed"));
        Ok(())
    } else {
        require_identical(control, packed, vocab, label)
    }
}

pub(super) fn run(engine: &Engine<'_>, links: &mut [SparkLink<'_>], runtime: &tokio::runtime::Runtime,
    windows: &[Window], members: usize, mode: PackingMode) -> Result<()> {
    ensure!((2..=16).contains(&members) && engine.shape.sequences >= members,
        "packed check needs 2 to 16 admitted sequence slots");
    ensure!(engine.prefill_rows >= 200, "packed check requires >=200 prefill rows");
    let selected = windows.iter().filter(|w| w.positions.len() == 512).take(members + 1).collect::<Vec<_>>();
    ensure!(selected.len() == members + 1, "packed check needs distinct sealed windows and a replacement partner");
    for (i, window) in selected.iter().enumerate() {
        ensure!(window.score_from > 0 && window.tokens.len() > window.score_from
            && window.positions.iter().enumerate().all(|(j, p)| p.pos == window.score_from + j)
            && window.tokens.len() - window.score_from == 512,
            "packed check window {} needs 512 contiguous scored positions", window.id);
        ensure!(selected[..i].iter().all(|other| other.id != window.id && other.tokens != window.tokens),
            "packed check windows must have distinct identities and token sequences");
    }
    let vocab = engine.cfg.vocab_size;
    for case in 0..2 {
        for replacement in [false, true] {
            let mut references = selected[..members].to_vec();
            if replacement { references[1] = selected[members]; }
            let starts = references.iter().map(|w| w.score_from - 1).collect::<Vec<_>>();
            let prompts = references.iter().map(|w| w.tokens[..w.tokens.len() - 1].to_vec()).collect::<Vec<_>>();
            let (alone, _) = arm(engine, links, runtime, &prompts, &starts, case, false)?;
            let (repeat, _) = arm(engine, links, runtime, &prompts, &starts, case, false)?;
            let (packed, packs) = arm(engine, links, runtime, &prompts, &starts, case, true)?;
            if mode == PackingMode::Unrestricted { ensure!(packs > 0, "packed check exercised no multi-member unit"); }
            let mut control_records = Vec::new();
            let mut repeat_records = Vec::new();
            let mut packed_records = Vec::new();
            for (i, (((a, noise), b), window)) in alone.iter().zip(&repeat).zip(&packed).zip(&references).enumerate() {
                let rows = prompts[i].len() - starts[i];
                ensure!([a, noise, b].iter().all(|o| o.frontier == prompts[i].len()
                    && o.logits.len() == rows * vocab && !o.state.bytes.is_empty()
                    && o.logits.iter().all(|v| v.is_finite()))
                    && a.state.bytes.len() == noise.state.bytes.len() && a.state.bytes.len() == b.state.bytes.len()
                    && a.state.tensors == noise.state.tensors && a.state.tensors == b.state.tensors,
                    "invalid packed observation");
                let (noise_kl, noise_top1) = pair_metrics(&a.logits, &noise.logits, vocab);
                println!("packed singleton noise case {case} replacement {replacement} member {i}: KL {noise_kl:.8}, top1 {noise_top1:.6}, logits byte-identical {}, state byte-identical {}",
                    equal(&a.logits, &noise.logits), a.state.bytes == noise.state.bytes);
                let (kl, top1) = pair_metrics(&a.logits, &b.logits, vocab);
                println!("packed case {case} replacement {replacement} member {i} window {}: singleton KL {kl:.8}, top1 {top1:.6}, logits byte-identical {}, state byte-identical {}",
                    window.id, equal(&a.logits, &b.logits), a.state.bytes == b.state.bytes);
                let control = reference_score(window, a, vocab);
                let noise_score = reference_score(window, noise, vocab);
                let score = reference_score(window, b, vocab);
                ensure!([&control, &noise_score, &score].iter().all(|s| valid_score(s, 512)),
                    "invalid or incomplete member score");
                report_reference(&control, &noise_score, &format!("case {case} replacement {replacement} member {i} singleton A vs B noise"));
                report_reference(&control, &score, &format!("case {case} replacement {replacement} member {i}"));
                if mode == PackingMode::Exact {
                    exact_with_noise(a, noise, b, vocab, &format!("case {case} replacement {replacement} member {i} mode {mode:?}"))?;
                }
                control_records.extend(control.records);
                repeat_records.extend(noise_score.records);
                packed_records.extend(score.records);
            }
            let (control, noise, packed_score) = (Fidelity::from_records(control_records),
                Fidelity::from_records(repeat_records), Fidelity::from_records(packed_records));
            let label = format!("case {case} replacement {replacement} pooled");
            if mode == PackingMode::Unrestricted {
                noninferior_with_noise(&control, &noise, &packed_score, &label)?;
            } else {
                report_reference(&control, &noise, &format!("{label} singleton A vs B noise"));
                report_reference(&control, &packed_score, &label);
            }
            println!("packed check case {case} replacement {replacement}: {members} members, {packs} actual multi-member units, mode {mode:?}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noninferior(control: &Fidelity, packed: &Fidelity, label: &str) -> Result<()> {
        noninferior_with_noise(control, control, packed, label)
    }

    #[test]
    #[ignore = "requires the sealed V4 Flash coordinator and Spark expert packages"]
    fn live_noise_controlled_packed_reference() -> Result<()> {
        use clap::Parser;
        #[derive(Parser)]
        struct Cli { #[command(flatten)] args: super::super::GoldenArgs }
        let snapshot = std::env::var("CUTEAFD_GOLDEN_SNAPSHOT")?;
        let peers = std::env::var("CUTEAFD_GOLDEN_PEERS")?;
        let fixtures = std::env::var("CUTEAFD_GOLDEN_FIXTURES")?;
        let manifest = std::env::var("CUTEAFD_PROGRAM_MANIFEST")?;
        let mode = std::env::var("CUTEAFD_PACKED_CHECK_MODE")?;
        let args = Cli::try_parse_from(["v4", "--snapshot", &snapshot, "--peers", &peers,
            "--manifest", &manifest, "--golden", &fixtures, "--full-prefill-logits",
            "--pool-tokens", "262144", "--rtx-expert-layers", "2", "--dspark",
            "--prefill-rows", "2752", "--prefill-waves", "3", "--max-sequences", "8",
            "--packed-check", "3", "--prefill-packing", &mode])?.args;
        super::super::golden(args)
    }

    #[test]
    fn noise_control_bounds_regression_and_retains_finite_checks() {
        let score = |kl, top1| Fidelity { positions: 1536, kl, top1, ..Default::default() };
        let a = score(0.02, 0.97);
        let b = score(0.024, 0.96);
        assert!(noninferior_with_noise(&a, &b, &score(0.029, 0.95), "boundary").is_ok());
        assert!(noninferior_with_noise(&a, &b, &score(0.029001, 0.95), "KL").is_err());
        assert!(noninferior_with_noise(&a, &b, &score(0.029, 0.949999), "top1").is_err());
        assert!(noninferior_with_noise(&a, &score(f64::NAN, 0.96), &a, "NaN noise").is_err());
        assert!(noninferior_with_noise(&a, &Fidelity { missing: 1, ..b }, &a, "missing noise").is_err());
        // Improving singleton B tightens, rather than widens, the permitted delta.
        assert!(noninferior_with_noise(&a, &score(0.01, 0.99), &a, "signed noise").is_err());
    }

    #[test]
    fn exact_bytes_only_bounded_when_matching_singleton_is_unstable() {
        let observed = |logit| Observed { logits: vec![logit, 0.0], frontier: 1,
            state: prefix::PackedState { bytes: vec![0], tensors: vec![("state".into(), 0..1)] } };
        let a = observed(1.0);
        assert!(exact_with_noise(&a, &observed(1.0), &observed(1.0), 2, "stable").is_ok());
        assert!(exact_with_noise(&a, &observed(1.0), &observed(2.0), 2, "packing only").is_err());
        assert!(exact_with_noise(&a, &observed(2.0), &observed(3.0), 2, "#886").is_ok());
    }

    #[test]
    fn byte_identity_rejects_logit_and_named_state_differences() {
        let observed = || Observed { logits: vec![1.0, 0.0], frontier: 1,
            state: prefix::PackedState { bytes: vec![0, 1, 2, 3],
                tensors: vec![("rank0/layer2/unit0/compressed".into(), 0..2),
                    ("rank0/layer2/compressor_state0".into(), 2..4)] } };
        let a = observed();
        assert!(require_identical(&a, &observed(), 2, "equal").is_ok());
        let mut b = observed();
        b.logits[1] = -0.0;
        assert!(require_identical(&a, &b, 2, "logits").is_err());
        let mut b = observed();
        b.state.bytes[2] = 9;
        assert!(require_identical(&a, &b, 2, "state").is_err());
        let mut b = observed();
        b.state.bytes.pop();
        assert!(require_identical(&a, &b, 2, "extent").is_err());
    }

    #[test]
    fn noninferiority_checks_boundaries_missing_and_pooled_regressions() {
        let score = |kl, top1| Fidelity { positions: 512, kl, top1, ..Default::default() };
        let control = score(0.2, 0.8);
        assert!(noninferior(&control, &score(0.205, 0.79), "boundary").is_ok());
        assert!(noninferior(&control, &score(0.205001, 0.8), "kl").is_err());
        assert!(noninferior(&control, &score(0.2, 0.789999), "top1").is_err());
        for bad in [Fidelity { missing: 1, ..score(0.2, 0.8) },
            Fidelity { non_finite: 1, ..score(0.2, 0.8) },
            Fidelity { positions: 0, ..score(0.2, 0.8) }, score(f64::NAN, 0.8)] {
            assert!(noninferior(&control, &bad, "invalid").is_err());
        }
    }

    #[test]
    fn pooled_gate_allows_member_drift_but_rejects_aggregate_regression() {
        use cuteafd_bench::reference::Position;
        let record = |kl| Position { window: "member".into(), block: String::new(),
            bucket: String::new(), role: String::new(), position: 0, agree: true,
            confident: true, top3_contained: true, agree_text: true, finite: true,
            kl, nll: 1.0, ref_nll: 1.0, argmax: 0, reference_argmax: 0 };
        let controls = vec![record(0.2), record(0.2)];
        let packed = vec![record(0.21), record(0.19)];
        assert!(noninferior(&Fidelity::from_records(vec![controls[0].clone()]),
            &Fidelity::from_records(vec![packed[0].clone()]), "member informational").is_err());
        assert!(noninferior(&Fidelity::from_records(controls.clone()),
            &Fidelity::from_records(packed), "pooled").is_ok());
        assert!(noninferior(&Fidelity::from_records(controls),
            &Fidelity::from_records(vec![record(0.21), record(0.21)]), "pooled regression").is_err());
    }

    #[test]
    fn diagnostic_widths_cover_router_boundary_and_pair_metrics_are_informational() {
        assert_eq!((widths(0, 0), widths(0, 1), widths(0, 2)), (29, 31, 23));
        assert_eq!((widths(1, 0), widths(1, 1), widths(1, 2)), (200, 127, 64));
        let (kl, top1) = pair_metrics(&[1.0, 0.0], &[0.0, 1.0], 2);
        assert!(kl > 0.0);
        assert_eq!(top1, 0.0);
        assert!(!equal(&[0.0], &[-0.0]));
    }
}
