//! Test-only real-checkpoint Context oracle; never enables a serving speculator.
use super::*;
use super::super::{engine::Engine, prefix::Placement};
use cuteafd_engine::prefix::{PrefixFamily, RefPagePool};
use std::{cell::RefCell, collections::BTreeMap, path::PathBuf, sync::Arc};

const CONTEXT: usize = 4224;
const WIDTHS: [usize; 5] = [1, 2, 5, 7, 8];
const HISTORIES: [usize; 4] = [63, 2040, 4095, CONTEXT - 8];
type Selections = BTreeMap<usize, Vec<Vec<u8>>>;
thread_local! {
    static SELECTED: RefCell<Option<Selections>> = const { RefCell::new(None) };
}

pub(in super::super) fn recording() -> bool { SELECTED.with(|s| s.borrow().is_some()) }

pub(in super::super) fn record(stream: &Stream<'_>, layer: usize, rows: usize,
    indices: CuteafdDeviceBuffer, scores: CuteafdDeviceBuffer) -> Result<()> {
    stream.drain()?;
    let mut values = Vec::new();
    for buffer in [indices, scores] {
        let mut bytes = vec![0; rows * 2048 * 4];
        let buffer = crate::shared::prefix::view(buffer, 0, bytes.len())?;
        stream.copy_d2h(&mut bytes, buffer)?;
        values.push(bytes);
    }
    SELECTED.with(|selected| {
        let mut selected = selected.borrow_mut();
        let output = selected.as_mut().context("oracle recording absent")?.entry(layer).or_default();
        for row in 0..rows {
            let span = row * 8192..(row + 1) * 8192;
            output.push([&values[0][span.clone()], &values[1][span]].concat());
        }
        Ok(())
    })
}

fn capture<T>(body: impl FnOnce() -> Result<T>) -> Result<(T, Selections)> {
    SELECTED.with(|s| { *s.borrow_mut() = Some(BTreeMap::new()); });
    let result = body();
    let selected = SELECTED.with(|s| s.borrow_mut().take().unwrap());
    Ok((result?, selected))
}

#[derive(Debug, serde::Serialize)]
struct Difference { max_abs: f64, kl: f64, top1: bool }
fn difference(reference: &[f32], actual: &[f32]) -> Result<Difference> {
    ensure!(!reference.is_empty() && reference.len() == actual.len()
        && reference.iter().chain(actual).all(|x| x.is_finite()), "invalid oracle logits");
    let argmax = |values: &[f32]| values.iter().enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0))).unwrap().0;
    let log_z = |values: &[f32]| {
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        max + values.iter().map(|&x| (x as f64 - max).exp()).sum::<f64>().ln()
    };
    let (zr, za) = (log_z(reference), log_z(actual));
    Ok(Difference {
        max_abs: reference.iter().zip(actual).map(|(&r, &a)| (r as f64 - a as f64).abs()).fold(0., f64::max),
        kl: reference.iter().zip(actual).map(|(&r, &a)| {
            let logp = r as f64 - zr;
            logp.exp() * (logp - (a as f64 - za))
        }).sum::<f64>().max(0.),
        top1: argmax(reference) == argmax(actual),
    })
}

fn read_logits(engine: &Engine<'_>, logits: crate::shared::token_io::DeviceLogits) -> Result<Vec<f32>> {
    engine.library().cuda_set_device(1)?;
    logits.to_host(engine.library())
}

fn prepare(engine: &Engine<'_>, histories: &[usize], tokens: &[Vec<u32>])
    -> Result<(RefPagePool, Vec<Placement>)> {
    engine.reset_cold_state()?;
    let family = engine.prefix_family()?;
    let mut pool = super::super::prefix::diagnostic_pool(family.layout());
    let mut placements = Vec::new();
    for (slot, &history) in histories.iter().enumerate() {
        let mut placement = Placement { media_spans: Vec::new(), units: pool.alloc((history + 8).div_ceil(64))?,
            slot, len: 0, state_len: 0 };
        family.bind(&mut placement).map_err(|e| anyhow::anyhow!("{e}"))?;
        for chunk in tokens[slot][..history].chunks(256) {
            engine.step_device(&mut [(&mut placement, chunk)], &pool, 1, false)?;
        }
        placements.push(placement);
    }
    Ok((pool, placements))
}

fn dispose(engine: &Engine<'_>, placements: &mut [Placement]) -> Result<()> {
    let family = engine.prefix_family()?;
    family.drain().map_err(|e| anyhow::anyhow!("{e}"))?;
    for placement in placements { family.discard(placement).map_err(|e| anyhow::anyhow!("{e}"))?; }
    Ok(())
}

fn digests(engine: &Engine<'_>, placements: &[Placement]) -> Result<Vec<String>> {
    let family = engine.prefix_family()?;
    placements.iter().map(|placement| family.live_digest(placement)).collect()
}

fn accept(engine: &Engine<'_>, placements: &mut [Placement], tokens: &[Vec<u32>],
    histories: &[usize], count: usize, keep: usize) -> Result<()> {
    let mut states = (0..placements.len()).map(|_| super::super::backend::DraftState::default()).collect::<Vec<_>>();
    let mut verified = placements.iter_mut().zip(&mut states).enumerate().map(|(slot, (placement, seq))| {
        let start = placement.len;
        placement.len += keep;
        crate::shared::qwen::scheduler::Verified {
            placement, seq, start, history: (), rows: &tokens[slot][histories[slot]..histories[slot] + count],
            first_row: slot * count, kept: Some((keep, 0)), finished: false, neural: false,
        }
    }).collect::<Vec<_>>();
    engine.accept_provisional(&mut verified)
}

fn case(engine: &Engine<'_>, histories: &[usize], tokens: &[Vec<u32>], n: usize,
    label: &str, failures: &mut usize) -> Result<()> {
    let (pool, mut placements) = prepare(engine, histories, tokens)?;
    let mut sequential = vec![Vec::<Vec<f32>>::new(); histories.len()];
    let mut states = Vec::new();
    let mut selection = BTreeMap::<usize, Vec<Vec<Vec<u8>>>>::new();
    for row in 0..n {
        let (values, selected) = capture(|| {
            let mut batch = placements.iter_mut().enumerate().map(|(s, p)|
                (p, &tokens[s][histories[s] + row..histories[s] + row + 1])).collect::<Vec<_>>();
            read_logits(engine, engine.step_device(&mut batch, &pool, histories.len(), false)?)
        })?;
        ensure!(engine.target_view().config()?["model_type"] != "kimi_k31" || !selected.is_empty(),
            "oracle did not observe a Context DSA selector");
        let vocab = values.len() / histories.len();
        for (s, logits) in values.chunks_exact(vocab).enumerate() { sequential[s].push(logits.to_vec()); }
        for (layer, rows) in selected { selection.entry(layer).or_default().push(rows); }
        states.push(digests(engine, &placements)?);
    }
    dispose(engine, &mut placements)?;
    // Same-token decode versus ordinary gathered prefill is the numerical floor.
    let (pool, mut placements) = prepare(engine, histories, tokens)?;
    let mut floor = vec![Vec::<f32>::new(); histories.len()];
    let mut batch = placements.iter_mut().enumerate().map(|(s, p)|
        (p, &tokens[s][histories[s]..histories[s] + n])).collect::<Vec<_>>();
    let outcomes = engine.prefill_packed_all_rows(&mut batch, &pool, &mut |slot, _, logits| {
        floor[slot].extend(read_logits(engine, logits)?);
        Ok(Ok(()))
    })?;
    for outcome in outcomes { outcome?; }
    let packed_state_equal = digests(engine, &placements)? == states[n - 1];
    let k31 = engine.target_view().config()?["model_type"] == "kimi_k31";
    let vocab = sequential[0][0].len();
    for s in 0..histories.len() { for r in 0..n {
        println!("KIMI_PACKED_PREFILL {}", serde_json::json!({"view":label,"k31":k31,
            "histories":histories,"n":n,"slot":s,"row":r,
            "difference":difference(&sequential[s][r], &floor[s][r * vocab..(r + 1) * vocab])?,
            "committed_state_identical":packed_state_equal}));
    }}
    dispose(engine, &mut placements)?;
    if !k31 { return Ok(()); }
    for keep in [n, n / 2].into_iter().filter(|&k| k > 0) {
        let (pool, mut placements) = prepare(engine, histories, tokens)?;
        let (values, selected) = capture(|| {
            let mut batch = placements.iter_mut().enumerate().map(|(s, p)|
                (p, &tokens[s][histories[s]..histories[s] + n])).collect::<Vec<_>>();
            read_logits(engine, engine.verify_device(&mut batch, &pool, false)?)
        })?;
        let selections_equal = selection.iter().all(|(layer, steps)| {
            selected.get(layer).is_some_and(|rows| rows.len() == histories.len() * n
                && (0..histories.len()).all(|s| (0..n).all(|r| rows[s * n + r] == steps[r][s])))
        }) && selected.len() == selection.len();
        accept(engine, &mut placements, tokens, histories, n, keep)?;
        let state_equal = digests(engine, &placements)? == states[keep - 1];
        let vocab = sequential[0][0].len();
        for s in 0..histories.len() { for r in 0..n {
            let verify = difference(&sequential[s][r], &values[(s * n + r) * vocab..(s * n + r + 1) * vocab])?;
            let floor = difference(&sequential[s][r], &floor[s][r * vocab..(r + 1) * vocab])?;
            println!("K31_MULTIROW {}", serde_json::json!({"view":label,"histories":histories,"n":n,
                "keep":keep,"slot":s,"row":r,"verify":verify,"prefill_floor":floor,
                "selections_and_scores_identical":selections_equal,"committed_state_identical":state_equal}));
        }}
        *failures += usize::from(!selections_equal || !state_equal);
        if keep < n {
            let replay_histories = histories.iter().map(|h| h + keep).collect::<Vec<_>>();
            let mut batch = placements.iter_mut().enumerate().map(|(s, p)|
                (p, &tokens[s][replay_histories[s]..histories[s] + n])).collect::<Vec<_>>();
            let (_, replay) = capture(|| read_logits(engine, engine.verify_device(&mut batch, &pool, false)?))?;
            let replay_equal = selection.iter().all(|(layer, steps)| replay.get(layer).is_some_and(|rows|
                rows.len() == histories.len() * (n - keep) && (0..histories.len()).all(|s|
                    (keep..n).all(|r| rows[s * (n - keep) + r - keep] == steps[r][s]))));
            accept(engine, &mut placements, tokens, &replay_histories, n - keep, n - keep)?;
            let replay_state = digests(engine, &placements)? == states[n - 1];
            *failures += usize::from(!replay_equal || !replay_state);
            println!("K31_MULTIROW_REPLAY {}", serde_json::json!({"view":label,"histories":histories,
                "n":n,"keep":keep,"selections_and_scores_identical":replay_equal,
                "committed_state_identical":replay_state}));
        }
        dispose(engine, &mut placements)?;
    }
    Ok(())
}

#[test]
fn metrics_and_required_matrix() -> Result<()> {
    let identical = difference(&[1., 2., -3.], &[1., 2., -3.])?;
    assert_eq!((identical.max_abs, identical.kl, identical.top1), (0., 0., true));
    let shifted = difference(&[1., 2., -3.], &[4., 5., 0.])?;
    assert!(shifted.kl < 1e-12 && shifted.top1);
    assert!(!difference(&[1., 2.], &[3., 2.])?.top1);
    assert!(difference(&[f32::NAN], &[0.]).is_err());
    assert_eq!(WIDTHS.iter().max().unwrap() * HISTORIES.len(), 32);
    assert_eq!(HISTORIES[3] + WIDTHS[4], CONTEXT);
    assert!(HISTORIES.contains(&4095)); // Crosses a 512-token rank-local selector tile.
    Ok(())
}

#[test]
#[ignore = "requires K3.1 MC checkpoint, two RTX GPUs and sealed WIP programs"]
fn k31_context_multirow_verify_matches_sequential() -> Result<()> { run_checkpoint(true) }

#[test]
#[ignore = "requires K3 K6 checkpoint, two RTX GPUs and sealed WIP programs"]
fn k3_context_packed_prefill_matches_sequential() -> Result<()> { run_checkpoint(false) }

fn run_checkpoint(k31: bool) -> Result<()> {
    use cuteafd_loader::plan::checkpoint::CoreView;
    let views = if k31 { &[CoreView::Original, CoreView::K6Compact][..] } else { &[CoreView::Original][..] };
    let mut failures = 0;
    for &view in views {
        with_engine(k31, view, |engine, tokens, label| {
            for n in WIDTHS {
                for history in HISTORIES { case(engine, &[history], tokens, n, label, &mut failures)?; }
                case(engine, &HISTORIES, tokens, n, label, &mut failures)?;
            }
            Ok(())
        })?;
    }
    ensure!(failures == 0, "K3.1 multirow verify: {failures} selection/score/committed-state defects; retain speculation refusal");
    Ok(())
}

/// #2543: a step whose error path already published the abort (rank 0 waiting
/// for a push rank 1 never queued) retires through the engine teardown's second
/// publication with every drain: the abort, not the watchdog, releases the
/// wait, the engine ends Drained and the library is not quarantined.
#[test]
#[ignore = "requires K3.1 MC checkpoint, two RTX GPUs and sealed WIP programs"]
fn k31_double_abort_publication_drains_cleanly() -> Result<()> {
    use cuteafd_loader::plan::checkpoint::CoreView;
    with_engine(true, CoreView::Original, |engine, tokens, _| {
        let (pool, mut placements) = prepare(engine, &[63], tokens)?;
        super::super::layers::fault::arm_half_issued();
        let mut batch = placements.iter_mut().map(|p| (p, &tokens[0][63..64])).collect::<Vec<_>>();
        let error = engine.step_device(&mut batch, &pool, 1, false).err().context("injected step succeeded")?;
        println!("KIMI_DOUBLE_ABORT error={error:#}");
        ensure!(format!("{error:#}").contains("injected Kimi half-issued step failure"), "unexpected failure: {error:#}");
        match error.downcast_ref::<crate::shared::experts::rtx::pair::StepFailure>() {
            Some(crate::shared::experts::rtx::pair::StepFailure::Terminal { drain, .. }) =>
                ensure!(drain == "queued streams drained; engine cannot be reused", "teardown retained owners: {drain}"),
            _ => anyhow::bail!("injected failure was not a terminal step failure: {error:#}"),
        }
        ensure!(engine.terminal_state() == crate::shared::peer_split::TerminalState::Drained,
            "engine ended {:?}", engine.terminal_state());
        ensure!(engine.library().peer_wait_timeout().is_none(), "abort must release the wait before the watchdog");
        ensure!(!engine.library().is_quarantined_after_failed_drain(), "clean drain quarantined the module");
        ensure!(engine.step_device(&mut batch, &pool, 1, false).is_err(), "terminal engine accepted a step");
        println!("KIMI_DOUBLE_ABORT drained=true quarantined=false watchdog=false");
        Ok(())
    })
}

/// Loads layers 0..3 of the checkpoint (routed experts skipped) on both RTX GPUs.
fn with_engine(k31: bool, view: cuteafd_loader::plan::checkpoint::CoreView,
    body: impl FnOnce(&Engine<'_>, &[Vec<u32>], &str) -> Result<()>) -> Result<()> {
    use cuteafd_loader::plan::checkpoint::CheckpointView;
    cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::KIMI_K3)
        .map_err(|_| anyhow::anyhow!("oracle requires an isolated process"))?;
    let snapshot = PathBuf::from(std::env::var("CUTEAFD_GOLDEN_SNAPSHOT")?);
    let manifest = PathBuf::from(std::env::var("CUTEAFD_PROGRAM_MANIFEST")?);
    let native = PathBuf::from(std::env::var("CUTEAFD_NATIVE_LIB")?);
    let library = Arc::new(cuteafd_ffi::test_support::load_sealed_library(&native)?);
    let programs = library.programs()?.with_manifest(&manifest)?;
    let tokenizer = cuteafd_loader::LoadedTokenizer::from_snapshot(&snapshot)?;
    let code = include_str!("engine.rs");
    let mut tokens = Vec::new();
    for slot in 0..4 {
        let mut prompt = format!("Review this Rust inference engine and fix error handling. Case {slot}.\n");
        while tokenizer.encode_ids(&prompt)?.len() < CONTEXT + 8 { prompt.push_str(code); }
        tokens.push(tokenizer.encode_ids(&prompt)?);
    }
    let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
    let view = CheckpointView::select(&snapshot, view)?;
    let checkpoint = cuteafd_loader::plan::Checkpoint::coordinator_view(&view, false, false)?;
    let full = cuteafd_loader::families::kimi_k3::config::KimiK3Config::from_hf(&checkpoint.config)?;
    ensure!(full.dsa.is_some() == k31, "oracle checkpoint family differs");
    let cfg = super::super::admission::prefix(&full, 4)?;
    use cuteafd_loader::families::kimi_k3::config::KimiAttention;
    // Attention geometry and DSA semantics are global checkpoint fields.
    // Actual layers 0..3 span every attention class (three KDA, one DSA).
    for kind in &full.attention { ensure!(cfg.attention.contains(kind), "uncovered attention class: {kind:?}"); }
    ensure!(cfg.attention.contains(&KimiAttention::Kda)
        && cfg.attention.contains(&KimiAttention::Mla), "oracle must include recurrent KDA and DSA");
    let label = format!("{:?}", view.mode());
    runtime.block_on(async {
        let loaded = super::super::startup::load(&library, &programs, &checkpoint, &full, &cfg,
            super::super::startup::Options { media: None,
                role: super::super::startup::Role::Golden { skip_routed_experts: true },
                expert_usage: false, expert_usage_reader: Default::default(), cache_format: cuteafd_core::MlaCacheFormat::Fp8,
                devices: [0, 1], attention: Some(cuteafd_loader::placement::AttentionPlacement::Context),
                native_lib: &native, manifest: &manifest, peers: vec![], rtx_expert_layers: cuteafd_loader::placement::OnboardSetting::Auto,
                rows: 256, waves: 1, prefill_rows: Some(256), slots: 4, marks: 0, max_context: CONTEXT,
                context: cuteafd_loader::placement::planner::Context::pinned(CONTEXT as u64), pool_tokens: Some(4 * CONTEXT as u64),
                graphs: true, ffn_mode: Default::default(), hosted_layers: None, draft: None, verify_rows: 8, decode_rows: 64, prefix: None,
                prove_serve_admission: false,
            })?;
        println!("K31_MULTIROW_LAYOUT view={label} layers=4 routed=skipped ranks=2 attention=Context context={CONTEXT} context_graphs=on optional_kda_ffn_graphs=off");
        body(&loaded.engine, &tokens, &label)
    })
}
