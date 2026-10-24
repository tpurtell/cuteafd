//! Shared-engine residual-prefix diagnostics and all-layer fidelity scoring.
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{net::SocketAddr, path::{Path, PathBuf}, sync::Arc};
use super::{admission, engine::Engine};
use crate::shared::inventory::{InventoryPhase, RuntimeInventory};
use cuteafd_core::ExpertGeometry;
#[cfg(test)]
use cuteafd_core::memory_layout::Basis;
use cuteafd_loader::families::kimi_k3::config::KimiK3Config;
#[cfg(test)]
use cuteafd_loader::placement::GpuBudget;

const HIDDEN: usize = 7168;
const REVISION: &str = "f831ab66814297da540d832a5235f8e904f29d06";
const SOURCE: [(&str, &str); 3] = [
    ("configuration_kimi_k3_sha256", "735eb9ebe593e17d231e08e1df7f7be9b5ee0e079f511aa201f9572077b416ae"),
    ("modeling_kimi_linear_sha256", "9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a"),
    ("modeling_kimi_k3_sha256", "b9171c96726eda55234c92ac8dfae7e24c512fda68968ae8f2c3782b42665ea2"),
];

#[derive(Debug, clap::Args)]
pub(crate) struct GoldenArgs {
    #[arg(long, default_value = "fp8")] kv_cache: cuteafd_core::MlaCacheFormat,
    /// Prefill unit rows for scoring chunks; unset keeps whole chunks as one unit.
    #[arg(long)] prefill_rows: Option<usize>,
    /// Prefill units in flight while scoring (N=1 against N=2 at the same
    /// units is byte-identical).
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(1..=4))] prefill_waves: u8,
    #[arg(long)] snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")] native_lib: Option<PathBuf>,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")] manifest: PathBuf,
    /// Read-only cuteafd.kimi.layer-state/1 export, never a qualified reference panel.
    #[arg(long, required_unless_present = "nll")] reference_state: Option<PathBuf>,
    /// Original token panel whose set and window hashes bind the residual state.
    #[arg(long, required_unless_present = "nll")] windows: Option<PathBuf>,
    /// Prefix window, or one quick-subset window for a full-model smoke.
    #[arg(long, required_unless_present = "nll")] window: Option<String>,
    /// Inclusive zero-based decoder layer; inventory is truncated before admission.
    #[arg(long, required_unless_present = "nll")] last_layer: Option<usize>,
    /// Immutable published F32 reference directory, consumed by shared fidelity code.
    #[arg(long, requires = "nll")] golden: Option<PathBuf>,
    /// New output directory for the shared probe row dumps and fidelity metrics.
    #[arg(long, requires = "nll")] save_logits: Option<PathBuf>,
    #[arg(long, value_delimiter = ',')] peers: Vec<SocketAddr>,
    #[arg(long, default_value_t = 0)] device: i32,
    #[arg(long, default_value_t = 1)] split_device: i32,
    #[arg(long)] check_headers: bool,
    /// Routed layers split across both coordinator GPUs (rtx-tp2).
    #[arg(long, default_value_t = 0)] rtx_expert_layers: usize,
    /// Complete per-rank allocation ceiling; still an offline estimate.
    #[arg(long)] spark_budget_gib: Option<f64>,
    /// Informational F32 comparison only; direct gate/up is the speed-first default.
    #[arg(long, env = "CUTEAFD_KIMI_COMPENSATED_FFN")]
    compensated_ffn: bool,
    /// All-layer logits scored against the immutable F32 reference quick subset.
    #[arg(long, requires_all = ["golden", "save_logits"], conflicts_with_all = ["reference_state", "last_layer", "windows"])] nll: bool,
    /// Score the windows as teacher-forced speculative verify cycles of this
    /// many rows (2..=64; all rows accepted) instead of prefill-shaped chunks:
    /// the C1 serving verify shape, with the prompt prefilled up to score_from.
    #[arg(long, requires = "nll", value_parser = clap::value_parser!(u16).range(2..=64))] verify_rows: Option<u16>,
    /// With --verify-rows: replay startup verify graphs (on) or run eagerly (off).
    #[arg(long, requires = "verify_rows", default_value = "on", value_parser = ["on", "off"])] graphs: String,
    /// Grouped K6 KDA q/k/v/g input launch (on) or the four standalone launches
    /// (off; eager scoring only, since startup graphs capture the grouped launch).
    #[arg(long, default_value = "on", value_parser = ["on", "off"])] grouped_kda: String,
    /// With --verify-rows: score only the first N verify rows of each window.
    #[arg(long, requires = "verify_rows")] verify_limit: Option<usize>,
}

struct Reference { tokens: Vec<u32>, hidden: Vec<f32> }

impl GoldenArgs {
    fn ffn_mode(&self) -> cuteafd_ffi::KimiFfnMode {
        if self.compensated_ffn { cuteafd_ffi::KimiFfnMode::Compensated }
        else { cuteafd_ffi::KimiFfnMode::default() }
    }

    fn provenance(&self) -> Value {
        serde_json::json!({"schema": "cuteafd.kimi.run/1", "ffn_mode": self.ffn_mode().name(), "kv_cache": self.kv_cache.label(),
            "gate_up_programs": ([3072, 16896].map(|width| self.ffn_mode().gate_up_stem(width))),
            "rtx_expert_layers":self.rtx_expert_layers, "window": self.window, "last_layer": self.last_layer, "diagnostic_only": !self.nll,
            "verify_rows": self.verify_rows, "graphs": self.verify_rows.map(|_| &self.graphs), "grouped_kda": self.grouped_kda,
            "verify_limit": self.verify_limit, "prefill_rows": self.prefill_rows, "prefill_waves": self.prefill_waves})
    }
}

fn digest(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
fn sha(value: &Value) -> bool {
    value.as_str().is_some_and(|s| s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit()))
}

fn reference(root: &Path, panel: &Value, window: &str, last: usize) -> Result<Reference> {
    ensure!(!window.is_empty() && window.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-'),
        "Kimi unsafe reference window ID");
    let meta: Value = serde_json::from_slice(&std::fs::read(root.join("latest.json"))?)?;
    ensure!(meta["schema"] == "cuteafd.kimi.layer-state/1" && meta["layer"].as_u64() == Some(last as u64)
        && meta["diagnostic_prefix"].as_u64() == Some(last as u64),
        "Kimi requires a flagged diagnostic state retaining exactly --last-layer");
    let binding = &meta["binding"];
    ensure!(binding["state_dtype"] == "float32" && panel["family"] == "kimi_k3"
        && sha(&panel["set_sha256"]) && binding["set_sha256"] == panel["set_sha256"],
        "Kimi reference dtype, family or set binding differs");
    ensure!(binding["arithmetic"] == serde_json::json!({"dtype":"float32", "matmul_precision":"highest",
        "torch_matmul_tf32":false, "cudnn_tf32":false, "triton_dot_precision":"ieee",
        "kda_solve_tril_precision":"ieee"}), "Kimi reference F32 arithmetic policy differs");
    let identity = &binding["snapshot_identity"];
    ensure!(identity["snapshot_revision"] == REVISION && sha(&identity["config_sha256"])
        && sha(&identity["tokenizer_sha256"]), "Kimi reference snapshot identity differs");
    if let Some(arm) = panel.get("reference_snapshot") {
        for key in ["snapshot_revision", "config_sha256", "tokenizer_sha256", "checkpoint", "precision"] {
            if let Some(expected) = arm.get(key) {
                ensure!(identity.get(key) == Some(expected), "Kimi panel reference identity differs: {key}");
            }
        }
    }
    if let Some(expected) = panel.get("tokenizer_sha256") {
        ensure!(&identity["tokenizer_sha256"] == expected, "Kimi panel tokenizer identity differs");
    }
    let pins = &binding["source_pin"];
    ensure!(pins["revision"] == REVISION, "Kimi reference official revision differs");
    for (key, expected) in SOURCE { ensure!(pins[key] == expected, "Kimi reference source pin differs: {key}"); }
    for key in ["fla_source_sha256", "transformers_source_sha256", "compressed_tensors_source_sha256",
        "reference_adapter_sha256", "reference_runner_sha256"] {
        ensure!(sha(&pins[key]), "Kimi missing reference runtime/source digest: {key}");
    }
    let matches = panel["windows"].as_array().context("Kimi panel has no windows")?.iter()
        .filter(|w| w["id"] == window).collect::<Vec<_>>();
    ensure!(matches.len() == 1, "Kimi window absent or duplicated");
    ensure!(matches[0]["media"].as_array().is_none_or(Vec::is_empty), "Kimi diagnostic is text-only");
    let ids = matches[0]["tokens"].as_array().context("Kimi window has no tokens")?;
    let tokens = ids.iter().map(|id| id.as_u64().filter(|id| *id < 163840)
        .map(|id| id as u32).context("Kimi token outside vocabulary")).collect::<Result<Vec<_>>>()?;
    ensure!((1..=4096).contains(&tokens.len()), "Kimi cold diagnostic window must fit one 4096-row lane");
    ensure!(binding["windows"][window] == digest(&serde_json::to_vec(&tokens)?),
        "Kimi window token hash differs");
    let blob = &meta["windows"][window]["hidden"];
    let name = format!("layer{last:03}/{window}.bin");
    ensure!(blob["path"] == name && sha(&blob["sha256"]), "Kimi residual path or SHA256 differs");
    let root = root.canonicalize()?;
    let path = root.join(&name);
    ensure!(path.canonicalize()?.starts_with(&root) && !path.symlink_metadata()?.file_type().is_symlink(),
        "Kimi residual path escapes state directory");
    let expected = tokens.len().checked_mul(HIDDEN * 4).context("Kimi residual extent overflow")?;
    ensure!(blob["bytes"].as_u64() == Some(expected as u64) && path.metadata()?.len() == expected as u64,
        "Kimi reference residual shape differs");
    let bytes = std::fs::read(path)?;
    ensure!(blob["sha256"] == digest(&bytes), "Kimi residual file SHA256 differs");
    let hidden = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>();
    ensure!(hidden.iter().all(|v| v.is_finite()), "Kimi non-finite reference residual");
    Ok(Reference { tokens, hidden })
}

fn compare(actual: &[f32], expected: &[f32]) -> Result<(f64, f64)> {
    ensure!(!actual.is_empty() && actual.len() == expected.len()
        && actual.iter().chain(expected).all(|v| v.is_finite()), "Kimi residual comparison shape/finite failure");
    let (mut aa, mut bb, mut ab, mut dd) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in actual.iter().zip(expected) {
        let (a, b) = (f64::from(a), f64::from(b));
        aa += a * a; bb += b * b; ab += a * b; dd += (a - b) * (a - b);
    }
    ensure!(aa > 0.0 && bb > 0.0, "Kimi zero-norm residual comparison");
    Ok((ab / (aa.sqrt() * bb.sqrt()), (dd / bb).sqrt()))
}

fn hosted_layers(value: &str) -> Result<Vec<usize>> {
    if value.is_empty() { return Ok(vec![]); }
    value.split(',').map(|s| s.parse::<usize>().context("Kimi hosted layer ID")).collect()
}

pub(crate) async fn run_golden(args: GoldenArgs) -> Result<()> {
    on_engine_thread(move || run(args)).await
}

async fn on_engine_thread(work: impl FnOnce() -> Result<()> + Send + 'static) -> Result<()> {
    // Like serving, the golden owns its CUDA state and transport runtime on a blocking worker.
    tokio::task::spawn_blocking(work).await?
}

fn run(args: GoldenArgs) -> Result<()> {
    if args.nll { return full_run(args); }
    tracing::info!(ffn_mode = args.ffn_mode().name(), "Kimi FFN numerics selection");
    println!("Kimi run provenance: {}", args.provenance());
    let last = args.last_layer.context("prefix requires --last-layer")?;
    let panel: Value = serde_json::from_slice(&std::fs::read(args.windows.as_ref().context("prefix requires --windows")?)?)?;
    let reference = reference(args.reference_state.as_deref().context("prefix requires --reference-state")?,
        &panel, args.window.as_deref().context("prefix requires --window")?, last)?;
    let checkpoint = cuteafd_loader::plan::Checkpoint::coordinator(&args.snapshot, false, false)?;
    let full = KimiK3Config::from_hf(&checkpoint.config)?;
    full.release_geometry()?;
    let cfg = admission::prefix(&full, last.checked_add(1).context("Kimi last-layer overflow")?)?;
    let hosted = hosted_layers(&std::env::var("CUTEAFD_GOLDEN_EXPERT_LAYERS")?)?;
    ensure!(args.peers.len() == 6 && hosted == cfg.dense.iter().enumerate()
        .filter_map(|(layer, dense)| (!dense).then_some(layer)).skip(args.rtx_expert_layers).collect::<Vec<_>>(),
        "Kimi diagnostic needs exactly the selected routed layers on six real ranks");
    if args.check_headers {
        println!("Kimi diagnostic reference/header binding valid: layers=0..{} rows={}; no numerical qualification",
            last, reference.tokens.len());
        return Ok(());
    }
    native(&args, &checkpoint, &full, &cfg, reference, None)
}

fn full_reference(args: &GoldenArgs) -> Result<Vec<cuteafd_bench::reference::Window>> {
    use cuteafd_bench::fidelity_dataset;
    let root = args.golden.as_deref().context("--nll requires --golden")?;
    let bytes = std::fs::read(root.join("manifest.json"))?;
    ensure!(digest(&bytes) == fidelity_dataset::KIMI_K3_MANIFEST_SHA256,
        "Kimi F32 reference manifest differs from immutable publication");
    let manifest: Value = serde_json::from_slice(&bytes)?;
    ensure!(manifest["config"] == fidelity_dataset::KIMI_K3_CONFIG && manifest["vocab"] == 163840,
        "Kimi F32 reference config/vocabulary differs");
    let reference = fidelity_dataset::load(root, &manifest)?;
    let mut windows = fidelity_dataset::quick_windows(&reference)?;
    if let Some(id) = &args.window {
        windows.retain(|window| &window.id == id);
        ensure!(windows.len() == 1, "Kimi smoke window must belong to the sealed quick subset");
    }
    ensure!(windows.iter().all(|window| window.media.is_empty() && window.tokens.len() <= 16384
        && window.score_from > 0 && window.score_from < window.tokens.len()),
        "Kimi full-forward window needs text-only bounded causal scoring");
    Ok(windows)
}

fn full_run(args: GoldenArgs) -> Result<()> {
    let windows = full_reference(&args)?;
    let checkpoint = cuteafd_loader::plan::Checkpoint::coordinator(&args.snapshot, false, false)?;
    let cfg = KimiK3Config::from_hf(&checkpoint.config)?;
    cfg.release_geometry()?;
    let hosted = hosted_layers(&std::env::var("CUTEAFD_GOLDEN_EXPERT_LAYERS")?)?;
    ensure!(args.peers.len() == 6 && hosted == cfg.dense.iter().enumerate()
        .filter_map(|(layer, dense)| (!dense).then_some(layer)).skip(args.rtx_expert_layers).collect::<Vec<_>>(),
        "Kimi full-forward needs the planned Spark routed subset on six real ranks");
    println!("Kimi all-layer run provenance: {}", args.provenance());
    println!("Kimi F32 reference: config={} revision={} windows={} scored_rows={}",
        cuteafd_bench::fidelity_dataset::KIMI_K3_CONFIG, cuteafd_bench::fidelity_dataset::KIMI_K3_REVISION,
        windows.len(), windows.iter().map(|w| w.positions.len()).sum::<usize>());
    if args.check_headers { return Ok(()); }
    let output = args.save_logits.as_deref().context("--nll requires --save-logits")?;
    ensure!(!output.try_exists()? && output.parent().is_some_and(Path::is_dir),
        "Kimi logits output needs an existing parent and a new directory");
    native(&args, &checkpoint, &cfg, &cfg, Reference { tokens: vec![], hidden: vec![] }, Some(&windows))
}

fn forward_chunks(tokens: usize, capacity: usize) -> Result<Vec<std::ops::Range<usize>>> {
    ensure!((2..=16384).contains(&tokens)
        && (1..=cuteafd_loader::families::kimi_k3::allocation::LANE_ROWS_MAX).contains(&capacity),
        "Kimi causal window/chunk extent differs");
    let count = tokens - 1;
    Ok((0..count).step_by(capacity).map(|first| first..(first + capacity).min(count)).collect())
}

fn score_windows(args: &GoldenArgs, engine: &Engine<'_>, pool: &mut cuteafd_engine::prefix::RefPagePool,
    windows: &[cuteafd_bench::reference::Window], capacity: usize) -> Result<()> {
    use cuteafd_api::openai::probe::{Probe, ProbeSpec};
    use cuteafd_bench::{fidelity_rows, reference::Fidelity};
    use std::io::Write;
    let root = args.save_logits.as_deref().context("--nll requires --save-logits")?;
    std::fs::create_dir(root).context("Kimi logits output must be a new directory")?;
    let mut records = Vec::new();
    engine.reset_cold_state()?;
    for window in windows {
        let chunks = forward_chunks(window.tokens.len(), capacity)?;
        let count = window.tokens.len() - 1;
        let pages = pool.alloc(count.div_ceil(cuteafd_ffi::MLA_PAGE_ROWS))?;
        let dump = root.join(&window.id);
        let probe = Probe::new(ProbeSpec { cold: true, no_speculation: true,
            prompt_ids: Some(window.tokens.clone()), score_from: Some(window.score_from),
            score_positions: Some(window.positions.iter().map(|p| p.pos).collect()),
            top_k: 32, want: window.want(), dump_rows: Some(dump.clone()), ..Default::default() });
        probe.admitted("kimi_k3", &window.tokens, 0);
        probe.selected_score_path("prefill");
        probe.provenance(args.provenance());
        for chunk in chunks {
            let (first, end) = (chunk.start, chunk.end);
            let sequences = vec![vec![super::device::SequenceRows {
                slot: 0, start: first, rows: end - first, pages: &pages }]];
            engine.forward(&[window.tokens[first..end].to_vec()], &sequences, pool,
                |lane, offset, rows, logits| {
                    ensure!(lane == 0 && logits.len() == rows * 163840, "Kimi full-forward logits shape differs");
                    for (row, logits) in logits.chunks_exact(163840).enumerate() {
                        let position = first + offset + row + 1;
                        if position >= window.score_from { probe.row(position, logits); }
                    }
                    Ok(())
                })?;
            probe.scored_prefill_chunk(first + 1, end - first);
        }
        let record = probe.record();
        ensure!(record.error.is_none(), "Kimi row dump failed: {:?}", record.error);
        let mut score = window.score(&record.rows);
        ensure!(score.missing == 0 && score.non_finite == 0 && score.records.len() == window.positions.len(),
            "Kimi full-forward did not cover every finite scored row");
        fidelity_rows::score_compact(163840, window, &dump, &mut score)?;
        println!("Kimi window {}: rows={} top-1={:.4}% mean NLL={:.8} mean KL(top1024+tail)={:.8}",
            window.id, score.positions, score.top1 * 100.0, score.nll, score.kl);
        let mut out = std::fs::OpenOptions::new().write(true).create_new(true).open(root.join(format!("{}.json", window.id)))?;
        serde_json::to_writer(&mut out, &serde_json::json!({"probe":record,"score":score}))?;
        out.write_all(b"\n")?;
        records.extend(score.records);
        // Forward's terminal head readback joins both producer ranks. Reset
        // joins each exact lane before reusing slot zero and its logical pages.
        engine.reset_cold_state()?;
        pool.release(&pages);
    }
    let score = Fidelity::from_records(records);
    println!("Kimi prefill logits: rows={} top-1 agreement {:.4}% | mean NLL engine {:.8} | mean KL(top1024+tail) {:.8}",
        score.positions, score.top1 * 100.0, score.nll, score.kl);
    let out = std::fs::OpenOptions::new().write(true).create_new(true).open(root.join("fidelity.json"))?;
    serde_json::to_writer(out, &serde_json::json!({"schema":"cuteafd.kimi.golden/1",
        "reference_config":cuteafd_bench::fidelity_dataset::KIMI_K3_CONFIG,
        "reference_revision":cuteafd_bench::fidelity_dataset::KIMI_K3_REVISION,
        "reference_manifest_sha256":cuteafd_bench::fidelity_dataset::KIMI_K3_MANIFEST_SHA256,
        "path_shape":"prefill-shaped","kl_kind":"top1024+tail","provenance":args.provenance(),"score":score}))?;
    Ok(())
}

/// Decode-shaped scoring: prefill each window to score_from, then verify
/// `verify` teacher-forced rows per cycle and accept them all, exactly the
/// serving speculative path (replay records, verify MLA, KDA commit). With
/// startup graphs on, every exact width up to `verify` replays; the final
/// shorter cycle uses its own exact width.
fn score_verify(args: &GoldenArgs, engine: &Engine<'_>, pool: &mut cuteafd_engine::prefix::RefPagePool,
    windows: &[cuteafd_bench::reference::Window], capacity: usize, verify: usize, graphs: bool) -> Result<()> {
    use cuteafd_api::openai::probe::{Probe, ProbeSpec};
    use cuteafd_bench::{fidelity_rows, reference::Fidelity};
    use cuteafd_engine::prefix::PrefixFamily;
    use std::io::Write;
    let root = args.save_logits.as_deref().context("--nll requires --save-logits")?;
    std::fs::create_dir(root).context("Kimi logits output must be a new directory")?;
    let family = engine.prefix_family()?;
    let library = engine.library();
    let mut records = Vec::new();
    let captures = cuteafd_ffi::graph_captures();
    for window in windows {
        let count = window.tokens.len() - 1;
        let end = args.verify_limit.map_or(count, |limit| count.min(window.score_from - 1 + limit));
        let units = pool.alloc(window.tokens.len().div_ceil(cuteafd_ffi::MLA_PAGE_ROWS))?;
        let mut placement = super::prefix::Placement { units: units.clone(), slot: 0, len: 0, state_len: 0 };
        family.bind(&mut placement).map_err(|error| anyhow::anyhow!("{error}"))?;
        let dump = root.join(&window.id);
        let probe = Probe::new(ProbeSpec { cold: true, no_speculation: true,
            prompt_ids: Some(window.tokens.clone()), score_from: Some(window.score_from),
            score_positions: Some(window.positions.iter().map(|p| p.pos)
                .filter(|&pos| pos <= end).collect()),
            top_k: 32, want: window.want(), dump_rows: Some(dump.clone()), ..Default::default() });
        probe.admitted("kimi_k3", &window.tokens, 0);
        probe.selected_score_path(if graphs { "verify-graphs" } else { "verify-eager" });
        probe.provenance(args.provenance());
        // Prompt prefix: rows 0..score_from-1, chunked at the lane capacity.
        let prefix = window.score_from - 1;
        for first in (0..prefix).step_by(capacity) {
            let chunk = &window.tokens[first..(first + capacity).min(prefix)];
            engine.step_device(&mut [(&mut placement, chunk)], pool, 1, true)?;
        }
        let mut first = prefix;
        while first < end {
            let rows = &window.tokens[first..(first + verify).min(end)];
            let start = placement.len;
            let logits = engine.verify_device(&mut [(&mut placement, rows)], pool, graphs)?;
            library.cuda_set_device(args.split_device)?;
            let values = logits.to_host(library)?;
            for (row, logits) in values.chunks_exact(163840).enumerate() {
                let position = first + row + 1;
                if position >= window.score_from { probe.row(position, logits); }
            }
            let mut seq = super::backend::DraftState::default();
            placement.len = start + rows.len();
            engine.accept_provisional::<_, ()>(&mut [crate::shared::qwen::scheduler::Verified {
                placement: &mut placement, seq: &mut seq, start, history: (), rows,
                first_row: 0, kept: Some((rows.len(), 0)), finished: false, neural: false }])?;
            first += rows.len();
        }
        let record = probe.record();
        ensure!(record.error.is_none(), "Kimi row dump failed: {:?}", record.error);
        let mut scored = window.clone();
        scored.positions.retain(|p| p.pos <= end);
        let mut score = scored.score(&record.rows);
        ensure!(score.missing == 0 && score.non_finite == 0 && score.records.len() == scored.positions.len(),
            "Kimi verify scoring did not cover every finite scored row");
        fidelity_rows::score_compact(163840, &scored, &dump, &mut score)?;
        println!("Kimi window {} verify={verify} graphs={graphs}: rows={} top-1={:.4}% mean NLL={:.8} mean KL(top1024+tail)={:.8}",
            window.id, score.positions, score.top1 * 100.0, score.nll, score.kl);
        let mut out = std::fs::OpenOptions::new().write(true).create_new(true).open(root.join(format!("{}.json", window.id)))?;
        serde_json::to_writer(&mut out, &serde_json::json!({"probe":record,"score":score}))?;
        out.write_all(b"\n")?;
        records.extend(score.records);
        family.drain().map_err(|error| anyhow::anyhow!("{error}"))?;
        family.discard(&mut placement).map_err(|error| anyhow::anyhow!("{error}"))?;
        pool.release(&units);
    }
    let steady = cuteafd_ffi::graph_captures() - captures;
    ensure!(steady == 0, "Kimi verify scoring captured {steady} graphs after startup");
    let replays = engine.graph_replays()?;
    let score = Fidelity::from_records(records);
    println!("Kimi verify logits: verify={verify} graphs={graphs} replays={replays} steady_captures={steady} rows={} top-1 agreement {:.4}% | mean NLL engine {:.8} | mean KL(top1024+tail) {:.8}",
        score.positions, score.top1 * 100.0, score.nll, score.kl);
    let out = std::fs::OpenOptions::new().write(true).create_new(true).open(root.join("fidelity.json"))?;
    serde_json::to_writer(out, &serde_json::json!({"schema":"cuteafd.kimi.golden/1",
        "reference_config":cuteafd_bench::fidelity_dataset::KIMI_K3_CONFIG,
        "reference_revision":cuteafd_bench::fidelity_dataset::KIMI_K3_REVISION,
        "reference_manifest_sha256":cuteafd_bench::fidelity_dataset::KIMI_K3_MANIFEST_SHA256,
        "path_shape":"verify-shaped","kl_kind":"top1024+tail","provenance":args.provenance(),
        "graph_replays":replays,"score":score}))?;
    Ok(())
}

fn native(args: &GoldenArgs, checkpoint: &cuteafd_loader::plan::Checkpoint,
    full: &KimiK3Config, cfg: &KimiK3Config, reference: Reference,
    scoring: Option<&[cuteafd_bench::reference::Window]>) -> Result<()> {
    cuteafd_core::set_expert_geometry(ExpertGeometry::KIMI_K3)
        .map_err(|_| anyhow::anyhow!("Kimi golden needs an isolated process"))?;
    super::startup::preflight(&args.manifest, args.native_lib.as_deref().context("--native-lib required")?)?;
    // SAFETY: sealed runner supplies the DSO; all borrowing owners drop before it.
    let library = Arc::new(unsafe { cuteafd_ffi::NativeLibrary::load(args.native_lib.as_ref().context("--native-lib required")?)? });
    let programs = library.programs()?.with_manifest(&args.manifest)?;
    // Lane rows: the longest scored chunk, at most `waves` units (an explicit
    // unit) or one 4096-row unit.
    use cuteafd_loader::families::kimi_k3::allocation::UNIT_ROWS_MAX;
    let lane = args.prefill_rows.map_or(UNIT_ROWS_MAX, |unit| unit * usize::from(args.prefill_waves));
    let rows = scoring.map_or(reference.tokens.len(), |windows| windows.iter().map(|w| w.tokens.len() - 1)
        .max().unwrap_or(1).min(lane));
    let verify = args.verify_rows.map(usize::from);
    let graphs = verify.is_some() && args.graphs == "on";
    let super::startup::Loaded { engine, admission_free, admitted_bytes } = super::startup::load(
        &library, &programs, checkpoint, full, cfg, super::startup::Options {
            cache_format: args.kv_cache,
            devices: [args.device, args.split_device], native_lib: args.native_lib.as_deref().context("--native-lib required")?,
            peers: args.peers.clone(), rtx_expert_layers: cuteafd_loader::placement::Onboard::Layers(args.rtx_expert_layers),
            spark_budget_gib: args.spark_budget_gib,
            rows, waves: usize::from(args.prefill_waves), prefill_rows: args.prefill_rows,
            slots: 1, marks: 1, max_context: cuteafd_loader::families::kimi_k3::COMPILED_CONTEXT, pool_tokens: Some(1_048_576), graphs,
            draft: None, ffn_mode: args.ffn_mode(), hosted_layers: Some(hosted_layers(&std::env::var("CUTEAFD_GOLDEN_EXPERT_LAYERS")?)?),
            verify_rows: verify.unwrap_or(0),
        })?;
    // Diagnostic A/B: the four standalone launches; refused once graphs are captured.
    if args.grouped_kda == "off" { engine.set_grouped_kda(false)?; }
    use cuteafd_engine::prefix::PrefixFamily;
    let pages = engine.prefix_family()?.layout().pages;
    let loaded = RuntimeInventory::measure(&library, &[args.device, args.split_device], InventoryPhase::PostLoad, |_| Ok(()))?;
    loaded.record(if scoring.is_some() { "kimi_k3-full-loaded" } else { "kimi_k3-prefix-loaded" }, &admission_free);
    for (rank, sample) in loaded.gpus.iter().enumerate() {
        tracing::info!(rank, device = sample.device, admitted_bytes = admitted_bytes[rank],
            measured_growth_bytes = admission_free[rank].saturating_sub(sample.free_bytes),
            "Kimi diagnostic loaded usage against admission");
    }
    let mut pool = cuteafd_engine::prefix::RefPagePool::new(pages, cuteafd_ffi::MLA_PAGE_ROWS);
    let actual = if let Some(windows) = scoring {
        if let Some(verify) = verify {
            score_verify(args, &engine, &mut pool, windows, rows, verify, graphs)?;
        } else {
            score_windows(args, &engine, &mut pool, windows, rows)?;
        }
        None
    } else {
        let allocated = pool.alloc(rows.div_ceil(cuteafd_ffi::MLA_PAGE_ROWS))?;
        Some(engine.prefix_residual(&reference.tokens, &allocated, &pool)?)
    };
    let steady = RuntimeInventory::measure(&library, &[args.device, args.split_device], InventoryPhase::PostLoad, |_| Ok(()))?;
    steady.record(if scoring.is_some() { "kimi_k3-full-steady" } else { "kimi_k3-prefix-steady" }, &admission_free);
    for (rank, sample) in steady.gpus.iter().enumerate() {
        tracing::info!(rank, device = sample.device, admitted_bytes = admitted_bytes[rank],
            measured_growth_bytes = admission_free[rank].saturating_sub(sample.free_bytes),
            "Kimi diagnostic steady usage against admission");
    }
    let Some(actual) = actual else { return Ok(()); };
    let (cosine, relative_l2) = compare(&actual, &reference.hidden)?;
    println!("Kimi diagnostic only: layer={} rows={} cosine={cosine:.12} relative_l2={relative_l2:.12}; no NLL/KL or qualification",
        args.last_layer.unwrap(), rows);
    Ok(())
}

#[cfg(test)]
use super::startup::{code_demand, exchange_demands};

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn golden_engine_thread_drives_its_transport_runtime_outside_async_command() -> Result<()> {
        for runtime in [
            tokio::runtime::Builder::new_current_thread().enable_all().build()?,
            tokio::runtime::Builder::new_multi_thread().worker_threads(1).enable_all().build()?,
        ] {
            runtime.block_on(async {
                let command_thread = std::thread::current().id();
                on_engine_thread(move || {
                    assert_ne!(std::thread::current().id(), command_thread);
                    let transport = tokio::runtime::Builder::new_current_thread().enable_all().build()?;
                    transport.block_on(async {
                        tokio::task::yield_now().await;
                        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                    });
                    Ok(())
                }).await?;
                let error = on_engine_thread(|| anyhow::bail!("engine completion failed"))
                    .await.unwrap_err();
                assert_eq!(error.to_string(), "engine completion failed");
                Ok::<_, anyhow::Error>(())
            })?;
        }
        Ok(())
    }

    fn fixture(root: &Path) -> Result<Value> {
        std::fs::create_dir_all(root.join("layer003"))?;
        let bytes = vec![0u8; 2 * HIDDEN * 4];
        std::fs::write(root.join("layer003/w.bin"), &bytes)?;
        let mut pins = json!({"revision":REVISION});
        for (key, value) in SOURCE { pins[key] = json!(value); }
        for key in ["fla_source_sha256", "transformers_source_sha256", "compressed_tensors_source_sha256",
            "reference_adapter_sha256", "reference_runner_sha256"] { pins[key] = json!("a".repeat(64)); }
        let panel = json!({"family":"kimi_k3", "set_sha256":"b".repeat(64), "windows":[{"id":"w","tokens":[1,2]}]});
        let state = json!({"schema":"cuteafd.kimi.layer-state/1", "layer":3, "diagnostic_prefix":3,
            "binding":{"source_pin":pins,"state_dtype":"float32","set_sha256":panel["set_sha256"],
                "arithmetic":{"dtype":"float32","matmul_precision":"highest","torch_matmul_tf32":false,
                    "cudnn_tf32":false,"triton_dot_precision":"ieee","kda_solve_tril_precision":"ieee"},
                "snapshot_identity":{"snapshot_revision":REVISION,"config_sha256":"c".repeat(64),
                    "tokenizer_sha256":"d".repeat(64)},"windows":{"w":digest(b"[1,2]")}},
            "windows":{"w":{"hidden":{"path":"layer003/w.bin","bytes":bytes.len(),"sha256":digest(&bytes)},"blocks":[]}}});
        std::fs::write(root.join("latest.json"), serde_json::to_vec(&state)?)?;
        Ok(panel)
    }

    #[test]
    fn flagged_reference_validates_native_source_tokens_extent_and_hash() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let panel = fixture(tmp.path())?;
        let loaded = reference(tmp.path(), &panel, "w", 3)?;
        assert_eq!(loaded.tokens, [1, 2]);
        assert_eq!(loaded.hidden.len(), 2 * HIDDEN);
        assert!(reference(tmp.path(), &panel, "w", 2).is_err());
        assert!(reference(tmp.path(), &panel, "../w", 3).is_err());
        let mut changed = panel.clone(); changed["windows"][0]["tokens"][0] = json!(3);
        assert!(reference(tmp.path(), &changed, "w", 3).is_err());
        let original = std::fs::read(tmp.path().join("latest.json"))?;
        for pointer in ["/diagnostic_prefix", "/binding/state_dtype", "/binding/set_sha256",
            "/binding/source_pin/revision", "/binding/source_pin/modeling_kimi_linear_sha256",
            "/binding/source_pin/reference_runner_sha256", "/binding/windows/w",
            "/binding/arithmetic/torch_matmul_tf32", "/binding/arithmetic/triton_dot_precision",
            "/binding/snapshot_identity/snapshot_revision", "/binding/snapshot_identity/config_sha256",
            "/windows/w/hidden/bytes", "/windows/w/hidden/path", "/windows/w/hidden/sha256"] {
            let mut state: Value = serde_json::from_slice(&original)?;
            *state.pointer_mut(pointer).unwrap() = Value::Null;
            std::fs::write(tmp.path().join("latest.json"), serde_json::to_vec(&state)?)?;
            assert!(reference(tmp.path(), &panel, "w", 3).is_err(), "{pointer}");
        }
        std::fs::write(tmp.path().join("latest.json"), original)?;
        let mut bytes = std::fs::read(tmp.path().join("layer003/w.bin"))?; bytes[0] = 1;
        std::fs::write(tmp.path().join("layer003/w.bin"), bytes)?;
        assert!(reference(tmp.path(), &panel, "w", 3).is_err());
        Ok(())
    }

    #[test]
    fn diagnostic_cli_requires_reference_and_layer_and_refuses_scores() -> Result<()> {
        use clap::Parser;
        let args = ["cuteafd", "kimi-golden", "--snapshot", "/checkpoint", "--windows", "/panel",
            "--reference-state", "/state", "--window", "w", "--last-layer", "3"];
        let parsed = crate::cli::Cli::try_parse_from(args)?;
        let crate::cli::Commands::KimiGolden(parsed) = parsed.command else { panic!("Kimi command"); };
        assert_eq!(parsed.last_layer, Some(3));
        assert_eq!(parsed.kv_cache, cuteafd_core::MlaCacheFormat::Fp8);
        let crate::cli::Commands::KimiGolden(bf16) = crate::cli::Cli::try_parse_from(
            args.into_iter().chain(["--kv-cache", "bf16"]))?.command else { panic!("Kimi command"); };
        assert_eq!(bf16.kv_cache, cuteafd_core::MlaCacheFormat::Bf16);
        assert_eq!(parsed.reference_state.as_deref(), Some(Path::new("/state")));
        for flag in ["--last-layer", "--reference-state"] {
            let at = args.iter().position(|arg| *arg == flag).unwrap();
            assert!(crate::cli::Cli::try_parse_from(args.iter().enumerate()
                .filter_map(|(index, arg)| (!(at..at + 2).contains(&index)).then_some(*arg))).is_err());
        }
        assert!(crate::cli::Cli::try_parse_from(args.into_iter().chain(["--nll"])).is_err());
        Ok(())
    }

    #[test]
    fn full_cli_requires_published_reference_and_new_logits_output() -> Result<()> {
        use clap::Parser;
        let args = ["cuteafd", "kimi-golden", "--snapshot", "/checkpoint", "--nll",
            "--golden", "/reference", "--save-logits", "/output"];
        let parsed = crate::cli::Cli::try_parse_from(args)?;
        let crate::cli::Commands::KimiGolden(parsed) = parsed.command else { panic!("Kimi command"); };
        assert!(parsed.nll && parsed.last_layer.is_none() && parsed.reference_state.is_none());
        assert_eq!(parsed.provenance()["diagnostic_only"], false);
        for flag in ["--golden", "--save-logits"] {
            let at = args.iter().position(|arg| *arg == flag).unwrap();
            assert!(crate::cli::Cli::try_parse_from(args.iter().enumerate()
                .filter_map(|(index, arg)| (!(at..at + 2).contains(&index)).then_some(*arg))).is_err());
        }
        for tail in [["--reference-state", "/state"], ["--last-layer", "3"], ["--windows", "/panel"]] {
            assert!(crate::cli::Cli::try_parse_from(args.into_iter().chain(tail)).is_err());
        }
        assert!(crate::cli::Cli::try_parse_from(args.into_iter().chain(["--window", "legacy"])).is_ok());
        let crate::cli::Commands::KimiGolden(verify) = crate::cli::Cli::try_parse_from(args.into_iter()
            .chain(["--verify-rows", "8", "--graphs", "off", "--verify-limit", "64"]))?.command else { panic!("Kimi command"); };
        assert_eq!((verify.verify_rows, verify.graphs.as_str(), verify.verify_limit), (Some(8), "off", Some(64)));
        assert_eq!(verify.provenance()["graphs"], "off");
        assert_eq!(parsed.provenance()["graphs"], serde_json::Value::Null);
        for bad in [&["--verify-rows", "1"][..], &["--verify-rows", "65"], &["--graphs", "off"], &["--verify-limit", "4"],
            &["--grouped-kda", "maybe"]] {
            assert!(crate::cli::Cli::try_parse_from(args.into_iter().chain(bad.iter().copied())).is_err(), "{bad:?}");
        }
        Ok(())
    }

    #[test]
    fn full_chunks_cover_causal_positions_without_truncation() -> Result<()> {
        let lengths = [(513, 1), (1835, 1323), (1823, 1311), (4669, 4157),
            (10827, 10315), (1024, 512), (1827, 1315), (1816, 1304)];
        let mut scored = 0;
        for (tokens, score_from) in lengths {
            let chunks = forward_chunks(tokens, 4096)?;
            let positions = chunks.iter().flat_map(|chunk| chunk.clone().map(|row| row + 1)).collect::<Vec<_>>();
            assert_eq!(positions, (1..tokens).collect::<Vec<_>>());
            assert!(chunks.iter().all(|chunk| !chunk.is_empty() && chunk.len() <= 4096));
            assert_eq!(chunks.len(), (tokens - 1).div_ceil(4096));
            let count = positions.iter().filter(|&&pos| pos >= score_from).count();
            assert_eq!(count, 512);
            scored += count;
        }
        assert_eq!(scored, 4096);
        for (tokens, capacity) in [(1, 1), (16385, 4096), (513, 0), (513, 8193)] {
            assert!(forward_chunks(tokens, capacity).is_err());
        }
        Ok(())
    }

    #[test]
    fn cold_window_reuse_retains_one_slot_and_releases_completed_pages() {
        let device = include_str!("device.rs");
        let reset = device.split("pub(crate) fn reset_cold_state").nth(1).unwrap()
            .split("pub(crate) fn stage_attention").next().unwrap();
        assert!(reset.contains("self.kda.slot(layer, 0)?") && reset.contains("KimiAttention::Kda"));
        assert!(reset.contains("cuda_zero_bytes_async(buffer, buffer.bytes, self.stream().raw)"));
        let engine = include_str!("engine.rs");
        let reset = engine.split("pub(crate) fn reset_cold_state").nth(1).unwrap()
            .split("pub(crate) fn forward").next().unwrap();
        assert!(reset.contains("lanes.local.len() == 1"));
        assert!(reset.find("cuda_stream_synchronize").unwrap() < reset.find("local.reset_cold_state()?").unwrap());
        assert!(!reset.contains("shutdown") && !reset.contains("drain_compute"));
        let source = include_str!("golden.rs");
        let score = source.split("fn score_windows").nth(1).unwrap().split("fn native").next().unwrap();
        assert!(score.find("engine.forward").unwrap() < score.rfind("engine.reset_cold_state()?").unwrap());
        assert!(score.rfind("engine.reset_cold_state()?").unwrap() < score.find("pool.release(&pages)").unwrap());
        assert!(score.contains("slot: 0") && score.contains("fidelity_rows::score_compact"));
    }

    #[test]
    #[ignore = "requires immutable Kimi F32 reference readback; set CUTEAFD_KIMI_F32_REFERENCE"]
    fn mounted_full_reference_uses_shared_quick_subset() -> Result<()> {
        use clap::Parser;
        let root = std::env::var("CUTEAFD_KIMI_F32_REFERENCE")?;
        let parsed = crate::cli::Cli::try_parse_from(["cuteafd", "kimi-golden", "--snapshot", "/checkpoint",
            "--nll", "--golden", &root, "--save-logits", "/output"])?;
        let crate::cli::Commands::KimiGolden(mut args) = parsed.command else { panic!("Kimi command"); };
        let windows = full_reference(&args)?;
        assert_eq!(windows.len(), 8);
        assert_eq!(windows.iter().map(|w| w.positions.len()).sum::<usize>(), 4096);
        for id in cuteafd_bench::fidelity_dataset::QUICK_WINDOWS {
            let window = windows.iter().find(|window| window.id == id).unwrap();
            let positions = forward_chunks(window.tokens.len(), 4096)?.into_iter()
                .flat_map(|chunk| chunk.map(|row| row + 1)).collect::<std::collections::BTreeSet<_>>();
            assert!(window.positions.iter().all(|p| positions.contains(&p.pos)));
        }
        args.window = Some("legacy".into());
        assert_eq!(full_reference(&args)?.len(), 1);
        args.window = Some("a01".into());
        assert!(full_reference(&args).is_err());
        Ok(())
    }

    #[test]
    fn ffn_cli_and_env_opt_in_are_explicit_and_recorded() -> Result<()> {
        use clap::{CommandFactory, FromArgMatches};
        let args = ["cuteafd", "kimi-golden", "--snapshot", "/checkpoint", "--windows", "/panel",
            "--reference-state", "/state", "--window", "w", "--last-layer", "3"];
        // Remove the inherited environment from the command, without mutating
        // process-global state shared with other CPU tests.
        for (value, tail, expected) in [(None, vec![], "direct"),
            (Some("true"), vec![], "compensated"),
            (Some("false"), vec![], "direct"),
            (Some("false"), vec!["--compensated-ffn"], "compensated")] {
            let command = crate::cli::Cli::command().mut_subcommand("kimi-golden", |command|
                command.mut_arg("compensated_ffn", |arg|
                    arg.env(None::<&str>).default_value(value.unwrap_or("false"))));
            let matches = command.try_get_matches_from(args.into_iter().chain(tail))?;
            let parsed = crate::cli::Cli::from_arg_matches(&matches)?;
            let crate::cli::Commands::KimiGolden(parsed) = parsed.command else { panic!("Kimi command"); };
            assert_eq!(parsed.ffn_mode().name(), expected);
            assert_eq!(parsed.provenance()["ffn_mode"], expected);
            let suffix = if expected == "compensated" { "_comp" } else { "" };
            assert_eq!(parsed.provenance()["gate_up_programs"], serde_json::json!([
                format!("kimik3_ffn_gate_up_i3072{suffix}"), format!("kimik3_ffn_gate_up_i16896{suffix}")]));
        }
        let command = crate::cli::Cli::command();
        let sub = command.get_subcommands().find(|command| command.get_name() == "kimi-golden").unwrap();
        let arg = sub.get_arguments().find(|arg| arg.get_id() == "compensated_ffn").unwrap();
        assert_eq!(arg.get_env().unwrap(), "CUTEAFD_KIMI_COMPENSATED_FFN");
        Ok(())
    }

    #[test]
    fn terminal_prefix_path_never_applies_output_mix_or_head() {
        let source = include_str!("device.rs");
        let terminal = source.split("pub(crate) fn terminal_residual").nth(1).unwrap()
            .split("pub(crate) fn residual").next().unwrap();
        assert!(terminal.contains("8 * self.lane.rows() * 7168 * 2"));
        assert!(terminal.contains("self.programs.add.launch(prefix, delta, output"));
        assert!(!terminal.contains("MixSite::Output"));
        let source = include_str!("engine.rs");
        let terminal = source.split("pub(crate) fn prefix_residual").nth(1).unwrap()
            .split("pub(crate) fn forward").next().unwrap();
        assert!(terminal.contains("super::layers::Output::Residual"));
        assert!(!terminal.contains(".project("));
        assert!(!terminal.contains("MixSite::Output"));
    }

    #[test]
    fn selected_hosted_layers_allow_dense_only_but_reject_missing_ids() -> Result<()> {
        assert!(hosted_layers("")?.is_empty());
        assert_eq!(hosted_layers("1,2,3")?, [1, 2, 3]);
        for invalid in [",", "1,", ",1", "1,,2", "-1", "a"] {
            assert!(hosted_layers(invalid).is_err(), "{invalid}");
        }
        Ok(())
    }

    #[test]
    fn reference_identity_cannot_be_rebound_by_the_panel() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let panel = fixture(tmp.path())?;
        for (key, value) in [("snapshot_revision", "wrong"), ("config_sha256", "e"),
            ("tokenizer_sha256", "f"), ("checkpoint", "wrong"), ("precision", "bf16")] {
            let mut changed = panel.clone();
            changed["reference_snapshot"] = serde_json::json!({key:value});
            assert!(reference(tmp.path(), &changed, "w", 3).is_err(), "{key}");
        }
        let mut changed = panel.clone();
        changed["tokenizer_sha256"] = serde_json::json!("e".repeat(64));
        assert!(reference(tmp.path(), &changed, "w", 3).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires sealed Kimi checkpoint and coordinator artifact metadata; CPU only"]
    fn mounted_whole_expert_coordinator_demand() -> Result<()> {
        let snapshot = PathBuf::from(std::env::var_os("CUTEAFD_KIMI_EXL3_SNAPSHOT").context("snapshot")?);
        let artifacts = PathBuf::from(std::env::var_os("CUTEAFD_KIMI_COORDINATOR_ARTIFACTS").context("artifacts")?);
        let checkpoint = cuteafd_loader::plan::Checkpoint::coordinator(&snapshot, false, false)
            .with_context(|| format!("read coordinator checkpoint {}", snapshot.display()))?;
        let cfg = KimiK3Config::from_hf(&checkpoint.config)?;
        let catalog = cuteafd_loader::read_expert_catalog(&snapshot)
            .with_context(|| format!("read expert catalog {}", snapshot.display()))?;
        let path = artifacts.join("PROGRAMS.json");
        let manifest: Value = serde_json::from_slice(&std::fs::read(&path)
            .with_context(|| format!("read coordinator programs {}", path.display()))?)
            .with_context(|| format!("parse coordinator programs {}", path.display()))?;
        let programs = manifest["programs"].as_array().context("programs")?;
        let scratch = |prefix: &str| -> Result<usize> {
            programs.iter().filter(|p| p["name"].as_str().is_some_and(|name| name.starts_with(prefix)))
                .map(|p| p["scratch_bytes_at_capacity"]["scratch"].as_u64()
                    .map(|bytes| bytes as usize).context("sealed scratch extent"))
                .collect::<Result<Vec<_>>>()?.into_iter().max().context("missing sealed scratch")
        };
        let native_scratch = super::super::device::NativeScratch { kda: scratch("kimik3_kda_m")?,
            mla: [scratch("kimik3_mla_producer_")?, scratch("kimik3_dense_mla_")?, scratch("kimik3_mla_output_")?] };
        let rows = 4096;
        let arch = cuteafd_loader::placement::ArchContext::for_device("sm_120", 96 << 30);
        let names = programs.iter().filter_map(|p| p["name"].as_str()).collect::<Vec<_>>();
        let selected = cuteafd_loader::placement::ProgramSet::from_names(&["kimik3"], names);
        let mut fixed = exchange_demands(rows, false,false)?;
        for rank in 0..2 {
            let code = cuteafd_loader::placement::inventory::loaded_code_allowance("kimik3", "exl3", true,
                rank as u8, arch, Some(&selected));
            fixed.push(code_demand(rank, code.bytes, arch.cublas_bytes, 1)?);
        }
        // No CUDA baseline is claimed: conservatively reserve context plus a
        // separate 2 GiB module allowance in this CPU-only forecast.
        let reserve = arch.context_bytes + (2u64 << 30);
        let slots = 8;
        let mark_bytes = super::super::state::KdaLayout::new(&cfg, slots, 1, 1)?.mark_bytes() * 2;
        let marks = crate::shared::prefix::MarkRule { lanes: slots, entries: 20,
            budget_bytes: 2 << 30, host: true }.slots(mark_bytes);
        let package = artifacts.join("exl3").join(crate::shared::experts::exl3::package_name("kimik3",
            catalog.exl3().context("EXL3 manifest")?.decoder_tiers())).join("rtx-tp2");
        let mut rtx = admission::plan_rtx(&cfg, &catalog, package.clone(), rows, 92)
            .with_context(|| format!("read TP2 expert inventory {}", package.display()))?;
        rtx.layers = cuteafd_loader::placement::Onboard::Auto;
        let workspace = rtx.workspace;
        let working = cuteafd_loader::placement::SparkWorkingSet { rtx_layers: vec![],
            spark_layers: (1..93).collect(), spark_ranks: 6, prefill_lanes: 1 };
        let spark = cuteafd_loader::placement::SparkBudgetInventory::exl3(&catalog, &working,
            &[cuteafd_core::SparkEndpoint { ring_depth: None, capacity: rows as u32, ingress: cuteafd_core::SparkIngress::Fp8K32, execution_lane: 0 }],
            crate::shared::placement::spark_rings(cuteafd_transport::TcpTransportConfig::default().max_frame_bytes)?,
            cuteafd_loader::families::kimi_k3::SPARK_DEFAULT_BUDGET_BYTES,
            cuteafd_loader::families::kimi_k3::SPARK_ESTIMATED_RESERVE_BYTES)?;
        let graph_set = super::super::graphs::owned_inventory(&cfg, 1, rows, slots, 0)?;
        let admitted = admission::admit(admission::Inputs { cfg: &cfg, core: Default::default(), cache_format: cuteafd_core::MlaCacheFormat::Bf16, gpus: [GpuBudget {
            capacity_bytes: 96 << 30, headroom_bytes: 2 << 30,
            baseline: cuteafd_loader::placement::Baseline::Measured { free_bytes: (96 << 30) - reserve } }; 2],
            devices: [0, 1], peer_access: true, spark_ranks: 6, attention: None,
            requested_pool: Some(1_048_576), state_slots: slots, prefix_marks: marks, rows, lanes: 1, waves: 2, unit_rows: None,
            lane_layouts: vec![std::array::from_fn(|_| super::super::device::LaneLayout::new(&cfg, rows, (rows).min(4096),
                1048576, native_scratch).unwrap())], fixed,
            startup: (0..2).map(|rank| super::super::graphs::demand(&graph_set.inventory, rank)).collect(),
            pool_bounded: vec![], rtx: Some(rtx), spark_budget: Some(spark.clone()), draft: None, verify_rows: 0 })?;
        let onboard = admitted.placement.onboard_layers;
        assert!(onboard >= spark.minimum_rtx_layers()?);
        assert_eq!(admitted.working_set().rtx_layers, (1..=onboard).collect::<Vec<_>>());
        assert_eq!(admitted.working_set().spark_layers, (onboard + 1..93).collect::<Vec<_>>());
        let page_bytes = cfg.mla_layers() as u64 * 64 * cuteafd_ffi::MLA_RECORD_BYTES as u64;
        let host = cuteafd_core::prefix::HostCacheGeometry { unit_rows: 64, page_bytes,
            mark_bytes: mark_bytes as u64, draft_bytes: 0 };
        let chunk = host.chunk_bytes();
        let required = cuteafd_core::prefix::host_cache_required_bytes(1 << 20, slots as u64,
            64, page_bytes, mark_bytes as u64, 0, chunk).context("host required quota")?;
        let host_bytes = host.resolve(None, 1 << 20, slots as u64, 128 << 30, chunk).context("host quota")?;
        eprintln!("Kimi CPU host forecast concurrency={slots} context=1048576 mark_bytes={mark_bytes} page_bytes={page_bytes} chunk={chunk} required={required} resolved={host_bytes} marks={} legacy_marks={marks}", admitted.mark_selection.slots);
        for rank in 0..6 {
            eprintln!("Kimi CPU Spark forecast rank={rank} retained={} budget={} reserve={} estimated_until_W9_census=true",
                spark.rank_bytes(rank, onboard)?, spark.budget, cuteafd_loader::families::kimi_k3::SPARK_ESTIMATED_RESERVE_BYTES);
        }
        for rank in 0..2 {
            let items = &admitted.placement.items[rank];
            let total = items.iter().map(|i| i.bytes).sum::<u64>();
            eprintln!("Kimi CPU coordinator forecast rank={rank} rows={rows} onboard={onboard} module_context_reserve={reserve} expert_workspace={workspace} admitted={total} expert_arena={} pool={} margin={}",
                admitted.placement.tp2.as_ref().context("TP2 arena forecast")?.peak_bytes[rank], admitted.placement.pool_tokens, (96u64 << 30) - total - reserve - (2u64 << 30));
            for item in items { eprintln!("Kimi CPU coordinator rank={rank} category={:?} group={} bytes={}",
                item.category, item.group, item.bytes); }
            assert!(total + reserve + (2u64 << 30) < 96u64 << 30);
        }
        Ok(())
    }

    #[test]
    fn shared_code_allowance_and_each_private_head_are_reserved() -> Result<()> {
        use cuteafd_loader::placement::{ArchContext, ProgramSet, inventory::loaded_code_allowance};
        let arch = ArchContext::for_device("sm_120", 96 << 30);
        let programs = ProgramSet::from_names(&["kimik3"], ["kimik3_kda_m64", "other_m64"]);
        for rank in 0..2 {
            let code = loaded_code_allowance("kimik3", "exl3", true, rank as u8, arch, Some(&programs));
            assert_eq!(code.basis, Basis::Formula);
            assert_eq!(code.bytes, arch.cublas_bytes + programs.module_bytes() + arch.peer_context_bytes);
            for lanes in [1, 2] {
                let demand = code_demand(rank, code.bytes, arch.cublas_bytes, lanes)?;
                assert_eq!(demand.basis, Basis::Formula);
                assert_eq!(demand.bytes, code.bytes + if rank == 1 { lanes as u64 * arch.cublas_bytes } else { 0 });
            }
        }
        assert!(code_demand(2, 0, 0, 1).is_err());
        assert!(code_demand(1, u64::MAX, 1, 1).is_err());
        Ok(())
    }

    #[test]
    fn peer_exchange_fixed_charges_exclude_solver_owned_hop_planes() -> Result<()> {
        for rows in [1, 64, 4096] {
            for diagnostics in [false, true] {
                let demands = exchange_demands(rows, diagnostics,false)?;
                let progress = if diagnostics {
                    (2 * std::mem::size_of::<cuteafd_ffi::peer_exchange::PeerProgress>()).max(256)
                } else { 0 };
                for (rank, demand) in demands.iter().enumerate() {
                    assert_eq!(demand.gpu as usize, rank);
                    assert_eq!(demand.basis, Basis::Exact);
                    assert_eq!(demand.bytes as usize, (rows * HIDDEN * 2).max(256)
                        + 256 + 256 + 256 + progress);
                }
            }
        }
        assert!(exchange_demands(0, false,false).is_err());
        assert!(exchange_demands(4097, false,false).is_err());
        Ok(())
    }

    #[test]
    fn residual_metrics_are_finite_unscored_diagnostics() -> Result<()> {
        let (cos, l2) = compare(&[1.0, 2.0], &[1.0, 2.0])?;
        assert!((cos - 1.0).abs() < 1e-14 && l2 == 0.0);
        assert!(compare(&[f32::NAN], &[1.0]).is_err());
        assert!(compare(&[1.0], &[1.0, 2.0]).is_err());
        assert!(compare(&[0.0], &[0.0]).is_err());
        Ok(())
    }
}
