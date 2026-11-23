//! Hy4-flash coordinator golden runner (`hy4f-golden`) for raptor RTX GPUs (#2478).
//!
//! Compares against the published Hy4-flash fidelity set (`0b1a914773`,
//! `hy4-flash-v1_20261118_own-text-bf16-f32`, agg. NLL `1.283684105386`) and
//! `python/reference/families/hy4_flash`, reading FP8 Engram shards only and
//! verifying unit `N=1` vs chunked byte-exactness and exact prefix restore.
use super::execution::Hy4Engine;
use crate::shared::golden_scores::{scored_nll, ReferenceRows};
use crate::shared::memory::device::Device;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::hy4_flash::{Hy4FlashConfig, Hy4KvCache};
use cuteafd_loader::plan::checkpoint::{Checkpoint, CheckpointView};
use cuteafd_loader::plan::families::hy4_flash as hy4_plan;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct GoldenArgs {
    #[command(flatten)]
    pub(crate) attention: crate::cli::CoreViewArgs,
    /// KV cache storage format (matches serving default `fp8`).
    #[arg(long, default_value = "fp8")]
    pub(crate) kv_cache: Hy4KvCache,
    /// Checkpoint snapshot directory (e.g. Hy4-flash-EXL3-K4-MC).
    #[arg(long)]
    pub(crate) snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    pub(crate) native_lib: Option<PathBuf>,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")]
    pub(crate) manifest: PathBuf,
    /// Directory containing `tokens.bin` (and optional `logits.bin` reference export).
    #[arg(long)]
    pub(crate) golden: Option<PathBuf>,
    /// Optional window subdirectory under `--golden`.
    #[arg(long)]
    pub(crate) window: Option<String>,
    /// Last target layer inclusive (0 for layer-0 dense check; 41 for full logits).
    #[arg(long, default_value_t = 41)]
    pub(crate) last_layer: usize,
    /// Primary CUDA device index (0 or 1 on raptor).
    #[arg(long, default_value_t = 0)]
    pub(crate) device: i32,
    /// Optional second CUDA device for 2x RTX TP2 coordinator execution.
    #[arg(long)]
    pub(crate) split_device: Option<i32>,
    /// Validate checkpoint headers, MC views, Engram FP8 shards, and target arenas without CUDA.
    #[arg(long)]
    pub(crate) check_headers: bool,
    /// Compute NLL/KL/top-1 against reference tokens and optional logits.
    #[arg(long)]
    pub(crate) nll: bool,
    /// Chunk size (rows per unit) for unit-exactness verification (`N=1` vs chunked).
    #[arg(long)]
    pub(crate) prefill_units: Option<usize>,
    /// Serving-shaped N=1 decode parity: prefill `0..score_from-1` in chunks, then step 1 token at a time.
    #[arg(long)]
    pub(crate) decode: bool,
    /// Enable bounded decode CUDA graphs during `--decode` or `--resume-at` verification.
    #[arg(long)]
    pub(crate) decode_graphs: bool,
    /// Diagnostic only: isolate one kind of split attention in an otherwise matched golden.
    #[arg(long, value_parser = ["both", "dsa", "swa", "off"], default_value = "both")]
    pub(crate) mla_split_probe: String,
    /// Compare one score position's captured attention with eager single/split replay.
    /// Writes to --dump-dir without disabling graph capture.
    #[arg(long, requires_all = ["dump_dir", "decode_graphs"])]
    pub(crate) mla_witness_position: Option<usize>,
    /// Capture layer boundaries on these scored decode positions (TP1 only).
    #[arg(long, value_delimiter = ',', requires_all = ["dump_dir", "decode_graphs"], conflicts_with = "mla_witness_position")]
    pub(crate) mla_layer_trace_positions: Vec<usize>,
    /// Save FP32 logits (`logits.bin`) to this directory.
    #[arg(long, hide = true)]
    pub(crate) save_logits: Option<PathBuf>,
    /// Dump per-layer and layer 0..3 sub-block F32 tensors for bisection against python/reference.
    #[arg(long)]
    pub(crate) dump_dir: Option<PathBuf>,
    /// Diagnostic: zero routed experts to isolate non-expert backbone (iHC, Engram, MLA, indexer, shared FFN).
    #[arg(long, hide = true)]
    pub(crate) skip_routed_experts: bool,
    /// Optional path to `hy4f:exl3-k45-mul1` routed expert package directory.
    #[arg(long)]
    pub(crate) expert_package: Option<PathBuf>,
    /// Exact prefix-restore gate: prefill `tokens[..P]`, capture, restore into a second slot,
    /// assert 0 state bytes differ at P, and compare 1-token decode continuations.
    #[arg(long, value_delimiter = ',')]
    pub(crate) resume_at: Vec<usize>,
    /// Number of decode tokens to compare past each `--resume-at` point.
    #[arg(long, default_value_t = 16)]
    pub(crate) resume_rows: usize,
    /// Exercise host-tier mark offload + promotion during `--resume-at`.
    #[arg(long)]
    pub(crate) resume_host: bool,
}

fn read_u32_bin(path: &Path) -> Result<Vec<u32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    ensure!(bytes.len() % 4 == 0 && !bytes.is_empty(), "invalid u32 binary {}", path.display());
    Ok(bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn digest_f32(values: &[f32]) -> String {
    let mut h = Sha256::new();
    for v in values {
        h.update(v.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

fn load_published_hy4_reference(root: &Path, vocab: usize) -> Result<cuteafd_bench::reference::Reference> {
    use cuteafd_bench::fidelity_dataset;
    let bytes = std::fs::read(root.join("manifest.json"))
        .with_context(|| format!("reading {}", root.join("manifest.json").display()))?;
    let manifest: serde_json::Value = serde_json::from_slice(&bytes)?;
    let digest = format!("{:x}", Sha256::digest(&bytes));
    ensure!(
        manifest["config"].as_str() == Some(fidelity_dataset::HY4_CONFIG)
            && digest == fidelity_dataset::HY4_MANIFEST_SHA256,
        "Hy4-flash fidelity manifest mismatch: config={:?} sha256={digest}",
        manifest["config"]
    );
    let reference = fidelity_dataset::load(root, &manifest)?;
    ensure!(reference.vocab == vocab, "Hy4-flash fidelity vocab {} != {vocab}", reference.vocab);
    Ok(reference)
}

fn build_probe_row(position: usize, logits: &[f32], wanted_ids: &[u32]) -> cuteafd_api::openai::probe::ProbeRow {
    let finite = logits.iter().all(|v| v.is_finite());
    let mut max_v = f32::NEG_INFINITY;
    let mut argmax = 0u32;
    for (i, &v) in logits.iter().enumerate() {
        if v > max_v {
            max_v = v;
            argmax = i as u32;
        }
    }
    let max_f64 = f64::from(max_v);
    let sum_exp: f64 = logits.iter().map(|&v| (f64::from(v) - max_f64).exp()).sum();
    let lse = max_f64 + sum_exp.ln();
    let wanted: Vec<(u32, f32)> = wanted_ids
        .iter()
        .map(|&id| (id, (f64::from(logits[id as usize]) - lse) as f32))
        .collect();
    let mut fnv = 0xcbf2_9ce4_8422_2325u64;
    for &v in logits {
        for b in v.to_le_bytes() {
            fnv ^= u64::from(b);
            fnv = fnv.wrapping_mul(0x0100_0000_01b3);
        }
    }
    cuteafd_api::openai::probe::ProbeRow {
        position,
        hash: format!("{fnv:016x}"),
        argmax,
        top: Vec::new(),
        wanted,
        finite,
    }
}

fn check_headers_only(args: &GoldenArgs) -> Result<()> {
    let view = args.attention.core_view;
    let selected = CheckpointView::select(&args.snapshot, view)?;
    let checkpoint = Checkpoint::open_view(&selected)?;
    let cfg = Hy4FlashConfig::read(&args.snapshot)?;
    let model = hy4_plan::Model::open(&checkpoint)?;
    let gpus = if args.split_device.is_some() { 2 } else { 1 };
    let modes = cuteafd_loader::placement::families::hy4_flash::layer_modes(
        &cfg,
        gpus,
        cuteafd_loader::placement::AttentionPlacement::Heads,
    );
    let arenas = cuteafd_loader::families::hy4_flash::weight_layout::target_arenas(
        &checkpoint,
        &cfg,
        &model.core,
        &modes,
        gpus,
    )?;
    let total_arena_bytes: usize = arenas.iter().map(|a| a.arena.bytes()).sum();
    println!(
        "hy4f-golden check-headers OK: view={view:?} gpus={} layers={} dsa={} swa={} engram_fp8=required target_arena_bytes={}",
        gpus,
        cfg.layers,
        cfg.sparse_layers().len(),
        cfg.swa_layers().len(),
        total_arena_bytes,
    );
    Ok(())
}

pub(crate) async fn run_golden(args: GoldenArgs) -> Result<()> {
    if args.check_headers {
        return check_headers_only(&args);
    }
    let view = args.attention.core_view;
    let selected = CheckpointView::select(&args.snapshot, view)?;
    let checkpoint = Checkpoint::open_view(&selected)?;
    let cfg = Hy4FlashConfig::read(&args.snapshot)?;

    let golden_dir = args
        .golden
        .as_ref()
        .context("hy4f-golden requires --golden <DIR> (unless --check-headers)")?;

    // Two reference layouts: published cuteafd-fidelity dataset (manifest.json + *.safetensors)
    // or raw window directory (tokens.bin + optional logits.bin).
    let published = if golden_dir.join("manifest.json").is_file() {
        Some(load_published_hy4_reference(golden_dir, cfg.vocab_size)?)
    } else {
        None
    };
    let (selected_windows, raw_tokens, raw_window_dir) = if let Some(ref_set) = &published {
        let wins = match &args.window {
            Some(w) if w.trim() == "all-quick" => {
                let mut ws = Vec::with_capacity(ref_set.quick_windows.len());
                for id in &ref_set.quick_windows {
                    let win = ref_set
                        .windows
                        .iter()
                        .find(|w| w.id == *id)
                        .with_context(|| format!("Hy4-flash fidelity window {id:?} not found"))?;
                    ws.push(win.clone());
                }
                ws
            }
            Some(w) => {
                let wanted: Vec<String> = w
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                let mut ws = Vec::with_capacity(wanted.len());
                for id in &wanted {
                    let win = ref_set
                        .windows
                        .iter()
                        .find(|w| w.id == *id)
                        .with_context(|| format!("Hy4-flash fidelity window {id:?} not found"))?;
                    ws.push(win.clone());
                }
                ws
            }
            None => cuteafd_bench::fidelity_dataset::quick_windows(ref_set)?,
        };
        ensure!(!wins.is_empty(), "Hy4-flash fidelity run selected zero windows");
        (Some(wins), Vec::new(), golden_dir.clone())
    } else {
        let window_dir = match &args.window {
            Some(w) => golden_dir.join(w),
            None => golden_dir.clone(),
        };
        let tokens = read_u32_bin(&window_dir.join("tokens.bin"))?;
        (None, tokens, window_dir)
    };

    let max_resume = args
        .resume_at
        .iter()
        .copied()
        .max()
        .map(|p| p + args.resume_rows)
        .unwrap_or(0);
    let max_seq = selected_windows
        .as_ref()
        .map(|ws| ws.iter().map(|w| w.tokens.len()).max().unwrap_or(1))
        .unwrap_or(raw_tokens.len())
        .max(max_resume);
    let unit = args.prefill_units.unwrap_or(max_seq).min(4096);
    let max_pages = if !args.resume_at.is_empty() {
        (max_seq.div_ceil(64) * 3 + 16).max(64)
    } else {
        max_seq.div_ceil(64).max(16)
    };

    // Set process-wide routed-expert geometry BEFORE NativeLibrary::load so C++
    // `cuteafd_set_expert_hidden(1536)` configures `exl3_wire` and routed kernels.
    if let Err(prev) = cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::HY4_FLASH) {
        ensure!(
            prev == cuteafd_core::ExpertGeometry::HY4_FLASH,
            "expert geometry already set to {prev:?}"
        );
    }

    let lib_path = args
        .native_lib
        .as_deref()
        .unwrap_or_else(|| Path::new("/opt/cuteafd/lib/libcuteafd_native.so"));
    // SAFETY: trusted native library built by this repository's toolchain.
    let library = unsafe { NativeLibrary::load(lib_path)? };
    let programs = library.programs()?.with_manifest(&args.manifest)?;

    let mut device_ids = vec![args.device];
    if let Some(second) = args.split_device {
        ensure!(second != args.device, "--split-device must differ from --device");
        device_ids.push(second);
    }
    let devices: Vec<Device<'_>> = device_ids.iter().map(|&id| Device { library: &library, id }).collect();
    ensure!(
        args.expert_package.is_none() || !args.skip_routed_experts,
        "--expert-package conflicts with --skip-routed-experts"
    );
    let onboarded: Vec<Option<crate::shared::experts::rtx::exl3::Exl3Tp2<'_>>> =
        if let Some(pkg) = &args.expert_package {
            use crate::shared::experts::rtx::{exl3::Exl3Tp2, RtxSelectionPolicy};
            let catalog = cuteafd_loader::read_expert_catalog_view(
                checkpoint.view.as_ref().context("Hy4-flash checkpoint view required for routed experts")?,
            )?;
            let layers: Vec<usize> = cfg
                .routed_layers()
                .into_iter()
                .filter(|&l| l <= args.last_layer)
                .collect();
            let budget = 80usize << 30;
            if devices.len() == 1 {
                let w = Exl3Tp2::load_whole_weights(
                    devices[0],
                    &catalog,
                    pkg,
                    layers,
                    unit,
                    budget,
                    RtxSelectionPolicy::FirstCapacity,
                )?;
                vec![Some(w.execution()?)]
            } else {
                let [w0, w1] = Exl3Tp2::load_pair_weights(
                    [devices[0], devices[1]],
                    &catalog,
                    cuteafd_core::ExpertGeometry::HY4_FLASH,
                    [pkg.as_path(), pkg.as_path()],
                    layers,
                    unit,
                    [budget, budget],
                    RtxSelectionPolicy::FirstCapacity,
                )?;
                vec![Some(w0.execution()?), Some(w1.execution()?)]
            }
        } else {
            (0..devices.len()).map(|_| None).collect()
        };

    let mut engine = if !args.resume_at.is_empty() {
        Hy4Engine::new_serving(
            &devices,
            &programs,
            &checkpoint,
            cfg.clone(),
            args.kv_cache,
            unit,
            unit.min(cuteafd_loader::placement::families::hy4_flash::DECODE_ROWS as usize),
            1,
            max_pages,
            2,
            max_pages * 64,
            if args.resume_host { 2 } else { 4 },
            args.skip_routed_experts,
            onboarded,
        )?
    } else {
        Hy4Engine::new(
            &devices,
            &programs,
            &checkpoint,
            cfg.clone(),
            args.kv_cache,
            unit,
            max_pages,
            args.skip_routed_experts,
            onboarded,
        )?
    };
    engine.golden_mla_splits(matches!(args.mla_split_probe.as_str(), "both" | "dsa"),
        matches!(args.mla_split_probe.as_str(), "both" | "swa"));
    if let Some(position) = args.mla_witness_position {
        ensure!(args.decode && args.resume_at.is_empty() && !args.skip_routed_experts,
            "--mla-witness-position requires normal --decode with expert weights");
        engine.mla_witness = Some((position, args.dump_dir.clone().context("witness dump directory")?));
    } else if !args.mla_layer_trace_positions.is_empty() {
        ensure!(args.decode && args.resume_at.is_empty() && !args.skip_routed_experts && devices.len() == 1,
            "--mla-layer-trace-positions requires TP1 --decode with expert weights");
        engine.mla_layer_trace = Some((args.mla_layer_trace_positions.clone(),
            args.dump_dir.clone().context("layer trace dump directory")?));
    } else {
        engine.dump_dir = args.dump_dir.clone();
    }
    if args.decode_graphs {
        let layout = cuteafd_loader::serving_capacity::hy4_graphs::Hy4GraphLayout::resolve(devices.len(), 0);
        engine.configure_graphs(layout)?;
    }

    if !args.resume_at.is_empty() {
        let mut tokens = if let Some(ws) = &selected_windows {
            let mut joined = Vec::new();
            for w in ws {
                joined.extend_from_slice(&w.tokens);
                if joined.len() >= max_resume {
                    break;
                }
            }
            joined
        } else {
            raw_tokens
        };
        if tokens.len() < max_resume && !tokens.is_empty() {
            let base = tokens.clone();
            while tokens.len() < max_resume {
                tokens.extend_from_slice(&base);
            }
            tokens.truncate(max_resume);
        }
        return run_resume_check(&mut engine, &tokens, &args.resume_at, args.resume_rows, unit, args.resume_host);
    }

    let vocab = cfg.vocab_size;
    if let Some(windows) = selected_windows {
        let mut all_records = Vec::new();
        for win in &windows {
            engine.reset()?;
            let mut full_logits: Option<Vec<f32>> = None;
            let mut scored_slices: Vec<(usize, Vec<f32>)> = Vec::with_capacity(win.positions.len());
            if args.decode {
                let prefix = win.score_from.saturating_sub(1);
                for chunk in win.tokens[..prefix].chunks(unit) {
                    let _ = engine.forward_chunk_opts(chunk, args.last_layer, false)?;
                }
                let wanted_set: std::collections::BTreeSet<usize> =
                    win.positions.iter().map(|p| p.pos).collect();
                for pos in prefix..(win.tokens.len() - 1) {
                    let next_pos = pos + 1;
                    let need = wanted_set.contains(&next_pos);
                    let row = if args.decode_graphs {
                        engine.forward_decode_step(win.tokens[pos], args.last_layer, need)?
                    } else {
                        engine.forward_chunk_opts(&win.tokens[pos..pos + 1], args.last_layer, need)?
                    };
                    if need {
                        scored_slices.push((next_pos, row));
                    }
                }
            } else {
                let mut logits = Vec::with_capacity(win.tokens.len() * vocab);
                for chunk in win.tokens.chunks(unit) {
                    let part = engine.forward_chunk(chunk, args.last_layer)?;
                    logits.extend_from_slice(&part);
                }
                for pos in &win.positions {
                    let row_idx = pos.pos - 1;
                    scored_slices.push((pos.pos, logits[row_idx * vocab..(row_idx + 1) * vocab].to_vec()));
                }
                full_logits = Some(logits);
            }
            let want = win.want();
            let mut rows = Vec::with_capacity(win.positions.len());
            let mut scored_flat = Vec::with_capacity(win.positions.len() * vocab);
            for (pos_id, slice) in &scored_slices {
                rows.push(build_probe_row(*pos_id, slice, &want[pos_id]));
                scored_flat.extend_from_slice(slice);
            }
            let mut score = win.score(&rows);
            // Recompute each row's KL and NLL using direct engine tail in f64 (matching
            // `cuteafd_bench::fidelity_rows::score_compact`, avoiding `1 - q_mass` cancellation).
            for ((rec, pos), (_, slice)) in score.records.iter_mut().zip(&win.positions).zip(&scored_slices) {
                let max_v = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let max_f64 = f64::from(max_v);
                let lse = max_f64 + slice.iter().map(|&v| (f64::from(v) - max_f64).exp()).sum::<f64>().ln();
                let mut chosen = vec![false; vocab];
                let mut kl = 0.0f64;
                for t in &pos.top {
                    let id = t.id as usize;
                    chosen[id] = true;
                    let q_lp = f64::from(slice[id]) - lse;
                    kl += t.lp.exp() * (t.lp - q_lp);
                }
                if pos.tail_lp.is_finite() {
                    let mut peak = f64::NEG_INFINITY;
                    for (id, &v) in slice.iter().enumerate() {
                        if !chosen[id] {
                            let lp = f64::from(v) - lse;
                            if lp > peak {
                                peak = lp;
                            }
                        }
                    }
                    let tail_sum: f64 = slice
                        .iter()
                        .enumerate()
                        .filter(|&(id, _)| !chosen[id])
                        .map(|(_, &v)| ((f64::from(v) - lse) - peak).exp())
                        .sum();
                    let tail = peak + tail_sum.ln();
                    kl += pos.tail_lp.exp() * (pos.tail_lp - tail);
                }
                rec.kl = kl.max(0.0);
                rec.nll = -(f64::from(slice[pos.next as usize]) - lse);
            }
            score = cuteafd_bench::reference::Fidelity::from_records(score.records);
            let gen_win = cuteafd_bench::reference::Fidelity::from_records(
                score.records.iter().filter(|r| r.role == "gen").cloned().collect(),
            );
            let ctx_win = cuteafd_bench::reference::Fidelity::from_records(
                score.records.iter().filter(|r| r.role == "ctx").cloned().collect(),
            );
            let mode = if args.decode { "decode" } else { "prefill" };
            let digest_str = full_logits
                .as_deref()
                .map(digest_f32)
                .unwrap_or_else(|| digest_f32(&scored_flat));
            let scored_digest = digest_f32(&scored_flat);
            println!(
                "hy4f-golden mode={} window={} rows={} digest={} scored_digest={} mean NLL={:.6} (ref {:.6}) mean KL={:.6} (gen={:.6}/{:.2}%/{} ctx={:.6}/{:.2}%/{}) top-1={:.2}% top-3={:.2}%",
                mode,
                win.id,
                score.positions,
                digest_str,
                scored_digest,
                score.nll,
                score.ref_nll,
                score.kl,
                if gen_win.positions > 0 { gen_win.kl } else { 0.0 },
                if gen_win.positions > 0 { gen_win.top1 * 100.0 } else { 0.0 },
                gen_win.positions,
                if ctx_win.positions > 0 { ctx_win.kl } else { 0.0 },
                if ctx_win.positions > 0 { ctx_win.top1 * 100.0 } else { 0.0 },
                ctx_win.positions,
                score.top1 * 100.0,
                score.top3_contained * 100.0,
            );
            if score.kl > 0.05 {
                let mut worst = score.records.clone();
                worst.sort_by(|a, b| b.kl.total_cmp(&a.kl));
                for r in worst.iter().take(12) {
                    let prev_tok = win.tokens.get(r.position.saturating_sub(1)).copied().unwrap_or(0);
                    let next_tok = win.tokens.get(r.position).copied().unwrap_or(0);
                    println!(
                        "  worst pos={} role={} prev={} next={} kl={:.4} nll={:.3} ref_nll={:.3} argmax={} ref_argmax={}",
                        r.position, r.role, prev_tok, next_tok, r.kl, r.nll, r.ref_nll, r.argmax, r.reference_argmax,
                    );
                }
                println!("  first 10 ctx rows:");
                for r in score.records.iter().filter(|r| r.role == "ctx").take(10) {
                    let prev_tok = win.tokens.get(r.position.saturating_sub(1)).copied().unwrap_or(0);
                    let next_tok = win.tokens.get(r.position).copied().unwrap_or(0);
                    println!(
                        "    pos={} prev={} next={} kl={:.4} nll={:.3} ref_nll={:.3} argmax={} ref_argmax={}",
                        r.position, prev_tok, next_tok, r.kl, r.nll, r.ref_nll, r.argmax, r.reference_argmax,
                    );
                }
            }
            if let Some(out_dir) = &args.save_logits {
                if windows.len() == 1 {
                    std::fs::create_dir_all(out_dir)?;
                    let to_save = full_logits.as_deref().unwrap_or(&scored_flat);
                    let out_bytes: Vec<u8> = to_save.iter().flat_map(|v| v.to_le_bytes()).collect();
                    std::fs::write(out_dir.join("logits.bin"), out_bytes)?;
                }
            }
            all_records.extend(score.records);
        }
        let gen_records: Vec<_> = all_records.iter().filter(|r| r.role == "gen").cloned().collect();
        let ctx_records: Vec<_> = all_records.iter().filter(|r| r.role == "ctx").cloned().collect();
        let agg = cuteafd_bench::reference::Fidelity::from_records(all_records);
        let gen_agg = cuteafd_bench::reference::Fidelity::from_records(gen_records);
        let ctx_agg = cuteafd_bench::reference::Fidelity::from_records(ctx_records);
        println!(
            "hy4f-golden aggregate ({} windows, {} rows): mean NLL={:.6} (ref {:.6}, delta {:+.6}) mean KL={:.6} top-1={:.2}% confident-top-1={:.2}% top-3={:.2}%",
            windows.len(),
            agg.positions,
            agg.nll,
            agg.ref_nll,
            agg.nll - agg.ref_nll,
            agg.kl,
            agg.top1 * 100.0,
            agg.confident_top1.unwrap_or(0.0) * 100.0,
            agg.top3_contained * 100.0,
        );
        if gen_agg.positions > 0 {
            println!(
                "hy4f-golden generated-only ({} rows, verdict subset): mean NLL={:.6} (ref {:.6}, delta {:+.6}) mean KL={:.6} top-1={:.2}% confident-top-1={:.2}% top-3={:.2}%",
                gen_agg.positions,
                gen_agg.nll,
                gen_agg.ref_nll,
                gen_agg.nll - gen_agg.ref_nll,
                gen_agg.kl,
                gen_agg.top1 * 100.0,
                gen_agg.confident_top1.unwrap_or(0.0) * 100.0,
                gen_agg.top3_contained * 100.0,
            );
        }
        if ctx_agg.positions > 0 {
            println!(
                "hy4f-golden context-only ({} rows): mean NLL={:.6} (ref {:.6}, delta {:+.6}) mean KL={:.6} top-1={:.2}% confident-top-1={:.2}% top-3={:.2}%",
                ctx_agg.positions,
                ctx_agg.nll,
                ctx_agg.ref_nll,
                ctx_agg.nll - ctx_agg.ref_nll,
                ctx_agg.kl,
                ctx_agg.top1 * 100.0,
                ctx_agg.confident_top1.unwrap_or(0.0) * 100.0,
                ctx_agg.top3_contained * 100.0,
            );
        }
        if args.decode_graphs {
            for snap in crate::shared::decode_graph::device::snapshots() {
                let hits: u64 = snap.cycles.values().map(|c| c.hits).sum();
                let captures: u64 = snap.cycles.values().map(|c| c.captures).sum();
                let eager: u64 = snap.cycles.values().map(|c| c.eager_runs).sum();
                println!(
                    "hy4f-golden decode-graphs device={}: reserve={} charged={} captures={} hits={} eager_runs={}",
                    snap.device, snap.reserve, snap.charged, captures, hits, eager,
                );
            }
        }
        return Ok(());
    }

    let tokens = raw_tokens;
    let window_dir = raw_window_dir;
    let mut logits = Vec::new();
    for chunk in tokens.chunks(unit) {
        let part = engine.forward_chunk(chunk, args.last_layer)?;
        logits.extend_from_slice(&part);
    }

    if args.last_layer + 1 < cfg.layers {
        let res = engine.read_residual_f32(tokens.len().min(unit))?;
        println!(
            "hy4f-golden layer {} complete: rows={} residual_digest={}",
            args.last_layer,
            tokens.len().min(unit),
            digest_f32(&res)
        );
        return Ok(());
    }

    let digest = digest_f32(&logits);
    println!(
        "hy4f-golden logits digest={} rows={} vocab={} unit={} engram=engram_fp8",
        digest,
        tokens.len(),
        vocab,
        unit,
    );

    if let Some(out_dir) = &args.save_logits {
        std::fs::create_dir_all(out_dir)?;
        let out_bytes: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(out_dir.join("logits.bin"), out_bytes)?;
    }

    if args.nll {
        let reference = ReferenceRows::open(&window_dir.join("logits.bin"), tokens.len(), vocab, true)?;
        let mean_nll = if tokens.len() > 1 {
            scored_nll(&logits[..(tokens.len() - 1) * vocab], &tokens[1..], vocab)?
        } else {
            0.0
        };
        let nll_rows = tokens.len().saturating_sub(1);
        let mut top1_matches = 0usize;
        let mut ref_rows = 0usize;
        if let Some(ref_reader) = &reference {
            for row in 0..tokens.len() {
                let slice = &logits[row * vocab..(row + 1) * vocab];
                let ref_row = ref_reader.row(row)?;
                let got_top = slice
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                let ref_top = ref_row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                if got_top == ref_top {
                    top1_matches += 1;
                }
                ref_rows += 1;
            }
        }
        println!(
            "hy4f-golden mean NLL={mean_nll:.6} ({nll_rows} scored rows); top-1 agreement={top1_matches}/{ref_rows}"
        );
    }
    Ok(())
}

fn run_resume_check(
    engine: &mut Hy4Engine<'_>,
    tokens: &[u32],
    points: &[usize],
    resume_rows: usize,
    unit_rows: usize,
    use_host: bool,
) -> Result<()> {
    use super::cache::Hy4Placement;
    use super::execution::HeadMode;
    use crate::shared::prefill_pipeline::{sequence_units, Unit, UnitSegment};
    use crate::shared::prefix::CudaCopyEngine;
    use cuteafd_engine::prefix::{After, PrefixCache, PrefixConfig, PrefixFamily, SnapshotKind};
    use cuteafd_loader::formats::mapped_table::ReadPriority;

    ensure!(resume_rows > 0, "--resume-rows must be positive");
    let library = engine.library();
    let family = engine.prefix()?;
    let layout = family.layout();
    let host = if use_host {
        let hcfg = cuteafd_hostcache::config::Config {
            bytes: 2u64 << 30,
            chunk_bytes: (256u64 << 20).max(layout.page_bytes as u64).max(layout.mark_bytes as u64),
            min_tokens: 1,
            max_tokens: u32::try_from(engine.max_context)?,
            ..Default::default()
        };
        hcfg.validate()?;
        let copy = CudaCopyEngine::registered_retained(library, vec![], family.snapshot_owners())?;
        Some((hcfg, copy))
    } else {
        None
    };
    let mut cache: PrefixCache<CudaCopyEngine<'_>> = PrefixCache::new(
        layout,
        PrefixConfig {
            entries: 4,
            mark_slots: engine.mark_slots(),
            keep_logits: true,
            min_tokens: 1,
        },
        host,
    )?;
    let pad_id = engine.cfg.engram_pad_id;
    let layers = engine.cfg.layers;
    let vocab = engine.cfg.vocab_size;

    let run_prefill = |engine: &mut Hy4Engine<'_>, placement: &mut Hy4Placement, slice: &[u32]| -> Result<Vec<f32>> {
        let start = placement.len;
        let mut units = sequence_units(slice.len(), unit_rows, 1, layers)?;
        let last_idx = units.len() - 1;
        for (i, u) in units.iter_mut().enumerate() {
            let u_start = start + u.rows.start;
            let u_end = start + u.rows.end;
            u.rows = u_start..u_end;
            u.segments = vec![UnitSegment {
                sequence: 0,
                rows: 0..u.rows.len(),
                tokens: u_start..u_end,
                prompt_end: i == last_idx,
                snapshot: false,
                head: i == last_idx,
            }];
        }
        let mut next = [placement.clone()];
        let mut last_logits = Vec::new();
        engine.run_units(
            &mut next,
            &[slice],
            &units,
            ReadPriority::Prefill,
            |u| if u.index == last_idx { HeadMode::LastPerSeq } else { HeadMode::None },
            &mut |_, _, tile| {
                last_logits = tile.to_host(library)?;
                Ok(())
            },
        )?;
        *placement = next.into_iter().next().unwrap();
        Ok(last_logits)
    };

    let run_decode_step = |engine: &mut Hy4Engine<'_>, placement: &mut Hy4Placement, tok: u32| -> Result<Vec<f32>> {
        let pos = placement.len;
        let unit = Unit {
            index: 0,
            sequence: 0,
            segments: vec![UnitSegment {
                sequence: 0,
                rows: 0..1,
                tokens: pos..pos + 1,
                prompt_end: false,
                snapshot: false,
                head: true,
            }],
            rows: 0..1,
            layers: 0..layers,
            slot: 0,
            barrier: false,
        };
        let mut next = [placement.clone()];
        let src = [tok];
        let mut logits = Vec::new();
        engine.run_units(
            &mut next,
            &[&src],
            &[unit],
            ReadPriority::Decode,
            |_| HeadMode::Suffix(0),
            &mut |_, _, tile| {
                logits = tile.to_host(library)?;
                Ok(())
            },
        )?;
        *placement = next.into_iter().next().unwrap();
        Ok(logits)
    };

    for &at in points {
        ensure!(at > 0 && at + resume_rows <= tokens.len(), "--resume-at {at} + {resume_rows} exceeds {} tokens", tokens.len());
        let cap = at + resume_rows + 16;
        // 1) Straight prefill A of tokens[..at] on slot 0.
        let mut p_a = cache
            .admit_cold(&family, at, cap, |units| Hy4Placement::new(units, 0, pad_id))?
            .placement;
        let logits_p = run_prefill(engine, &mut p_a, &tokens[..at])?;
        ensure!(logits_p.len() == vocab, "last-row logits len mismatch");
        let state_a = engine.dump_state(&p_a)?;
        ensure!(
            cache.capture(&family, SnapshotKind::Prompt, &tokens[..at], &p_a, After::from_logits(&logits_p, true))?,
            "--resume-at {at}: capture skipped"
        );
        if use_host {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while cache.stats().marks_in_use > 0 && std::time::Instant::now() < deadline {
                cache.tick();
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }

        // 2) Restore B from PrefixCache into slot 1 (different slot + COW tail copy + mark restore).
        let admitted_b = cache.admit(&family, &tokens[..at + resume_rows], cap, false, |units| {
            Hy4Placement::new(units, 1, pad_id)
        })?;
        ensure!(admitted_b.resume == at, "--resume-at {at}: expected resume={at}, got {}", admitted_b.resume);
        let mut p_b = admitted_b.placement;
        p_b.set_engram_tail_from_prefix(&tokens[..at], pad_id);

        // 3) Byte-exact state comparison across all ranks (DSA KV, DSA index, pending keys/counts, SWA-512 ring, Engram tail).
        let state_b = engine.dump_state(&p_b)?;
        let mut total_bytes = 0usize;
        for (rank_idx, (sa, sb)) in state_a.iter().zip(&state_b).enumerate() {
            total_bytes += sa.total_bytes();
            let diffs = sa.diff(sb);
            ensure!(
                diffs.is_empty(),
                "hy4f-golden --resume-at {at} rank {rank_idx} STATE MISMATCH: {}",
                diffs.join("; ")
            );
        }

        // 4) 1-token-at-a-time decode continuation comparison: A (straight) vs B (restored).
        let mut max_abs_diff = 0.0f32;
        let mut bitwise_eq = 0usize;
        let mut top1_eq = 0usize;
        let mut first_diff_step: Option<(usize, f32)> = None;
        for step in 0..resume_rows {
            let tok = tokens[at + step];
            let la = run_decode_step(engine, &mut p_a, tok)?;
            let lb = run_decode_step(engine, &mut p_b, tok)?;
            if step == 0 {
                let after_a = engine.dump_state(&p_a)?;
                let after_b = engine.dump_state(&p_b)?;
                for (rank_idx, (sa, sb)) in after_a.iter().zip(&after_b).enumerate() {
                    let diffs = sa.diff(sb);
                    if at + 1 <= 2048 {
                        ensure!(
                            diffs.is_empty(),
                            "hy4f-golden --resume-at {at} rank {rank_idx} POST-STEP-1 STATE MISMATCH: {}",
                            diffs.join("; ")
                        );
                    } else if !diffs.is_empty() {
                        println!(
                            "hy4f-golden --resume-at {at} (host={use_host}) rank {rank_idx} post-step-1 state diffs (>2048 DSA): {}",
                            diffs.join("; ")
                        );
                    }
                }
            }
            let mut step_max = 0.0f32;
            let mut step_bits = true;
            for (&a, &b) in la.iter().zip(&lb) {
                if a.to_bits() != b.to_bits() {
                    step_bits = false;
                }
                let d = (a - b).abs();
                if d > step_max {
                    step_max = d;
                }
            }
            if step_max > max_abs_diff {
                max_abs_diff = step_max;
            }
            if step_bits {
                bitwise_eq += 1;
            } else if first_diff_step.is_none() {
                first_diff_step = Some((step, step_max));
            }
            let arg_a = la.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map(|(i, _)| i).unwrap_or(0);
            let arg_b = lb.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map(|(i, _)| i).unwrap_or(0);
            if arg_a == arg_b {
                top1_eq += 1;
            }
        }
        let st = cache.stats();
        println!(
            "hy4f-golden --resume-at {at} (host={use_host}, mark_promotions={}): state_bytes_diff=0/{total_bytes} cont_rows={resume_rows} max_abs_diff={max_abs_diff:.6e} bitwise_equal={bitwise_eq}/{resume_rows} top1={top1_eq}/{resume_rows} first_diff={first_diff_step:?}",
            st.mark_promotions
        );
        if at + resume_rows <= 2048 {
            ensure!(
                max_abs_diff == 0.0,
                "hy4f-golden --resume-at {at} (<2048 dense DSA): continuation not bit-exact (max_abs_diff={max_abs_diff:.6e}, first_diff={first_diff_step:?})"
            );
        }
        cache.release(&family, &p_a.units)?;
        cache.release(&family, &p_b.units)?;
        let idle = cache.stats();
        ensure!(
            idle.pages - idle.pages_free == idle.pages_retained,
            "hy4f-golden --resume-at {at}: idle page leak ({idle:?})"
        );
    }
    println!("hy4f-golden --resume-at: PASS");
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    #[test]
    fn hy4f_golden_cli_parses() {
        let cli = crate::cli::Cli::try_parse_from([
            "cuteafd",
            "hy4f-golden",
            "--snapshot",
            "/snapshot",
            "--golden",
            "/golden",
            "--nll",
            "--split-device",
            "1",
            "--skip-routed-experts",
        ])
        .expect("hy4f-golden CLI parse");
        let crate::cli::Commands::Hy4fGolden(args) = cli.command else {
            panic!("expected Hy4fGolden");
        };
        assert!(args.nll);
        assert_eq!(args.split_device, Some(1));
        assert!(args.skip_routed_experts);
        assert_eq!(args.last_layer, 41);
        assert!(
            crate::cli::Cli::try_parse_from([
                "cuteafd",
                "hy4f-golden",
                "--snapshot",
                "/snapshot",
                "--engram-bf16",
            ])
            .is_err(),
            "--engram-bf16 must be rejected (FP8 Engram only)"
        );
    }

    #[test]
    #[ignore = "requires published Hy4-flash fidelity dataset in ~/.cache/cuteafd/fidelity"]
    fn installed_hy4_published_fidelity_dataset_validates() -> anyhow::Result<()> {
        let root = std::path::PathBuf::from(std::env::var("HOME")?)
            .join(".cache/cuteafd/fidelity/wrldsuksgo2mars--cuteafd-fidelity")
            .join(cuteafd_bench::fidelity_dataset::HY4_REVISION)
            .join(cuteafd_bench::fidelity_dataset::HY4_CONFIG);
        let reference = super::load_published_hy4_reference(&root, 120_832)?;
        assert_eq!(reference.windows.len(), 70);
        assert!(reference.quick_windows.iter().any(|w| w == "a00"));
        let mut nll_sum = 0.0f64;
        let mut nll_rows = 0usize;
        for w in &reference.windows {
            for p in &w.positions {
                nll_sum += -p.next_lp;
                nll_rows += 1;
            }
        }
        assert_eq!(nll_rows, 35_840);
        assert!((nll_sum / nll_rows as f64 - 1.283684105385722).abs() < 1e-6);
        Ok(())
    }
}
