//! Hy4-flash coordinator golden runner (`hy4f-golden`) for raptor RTX GPUs (#2478).
//!
//! Compares against the published Hy4-flash fidelity set (`0b1a914773`,
//! `hy4-flash-v1_20261118_own-text-bf16-f32`, agg. NLL `1.283684105386`) and
//! `python/reference/families/hy4_flash`, reading FP8 Engram shards only and
//! verifying unit `N=1` vs chunked byte-exactness and exact prefix restore.
use crate::families::hy4_common::golden::{
    digest_f32, load_published_reference, read_u32_bin, score_window_direct_tail,
};
use super::execution::Hy4Engine;
use crate::shared::golden_scores::{scored_nll, ReferenceRows};
use crate::shared::memory::device::Device;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::hy4_flash::{Hy4FlashConfig, Hy4KvCache};
use cuteafd_loader::plan::checkpoint::{Checkpoint, CheckpointView};
use cuteafd_loader::plan::families::hy4_flash as hy4_plan;
use sha2::{Digest, Sha256};
use std::{net::SocketAddr, path::{Path, PathBuf}, sync::Arc};

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
    /// Attention placement on 2 GPUs (`auto` -> `context`, or explicit `heads` / `context`).
    #[arg(long, default_value = "auto", value_parser = ["auto", "heads", "context"])]
    pub(crate) attention_placement: String,
    /// Spark worker peer endpoints (`host:port,...`) for routed experts.
    #[arg(long, value_delimiter = ',')]
    pub(crate) peers: Vec<SocketAddr>,
    /// Validate checkpoint headers, MC views, Engram FP8 shards, and target arenas without CUDA.
    #[arg(long)]
    pub(crate) check_headers: bool,
    /// Compute NLL/KL/top-1 against reference tokens and optional logits.
    #[arg(long)]
    pub(crate) nll: bool,
    /// Chunk size (rows per unit) for unit-exactness verification (`N=1` vs chunked).
    #[arg(long)]
    pub(crate) prefill_units: Option<usize>,
    /// Prefill wave slots in flight (`1..=3`) for multi-wave byte-identity verification.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(1..=3))]
    pub(crate) prefill_waves: u8,
    #[arg(long, default_value = "auto")]
    pub(crate) decode_rows: cuteafd_loader::plan::decode_rows::DecodeRowsSetting,
    /// Serving-shaped N=1 decode parity: prefill `0..score_from-1` in chunks, then step 1 token at a time.
    #[arg(long)]
    pub(crate) decode: bool,
    /// Enable bounded decode CUDA graphs during `--decode` or `--resume-at` verification.
    #[arg(long)]
    pub(crate) decode_graphs: bool,
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
    /// Prove multi-wave single-sequence and concurrent two-sequence byte-identity
    /// (prompt A in wave k, prompt B in wave k+1) and disjoint per-wave Engram buffers.
    #[arg(long)]
    pub(crate) verify_waves: bool,
    /// Qualify the bundled drafter with the exact resume check and a rejected tail.
    #[arg(long, requires = "resume_at")]
    pub(crate) resume_dfly2: bool,
    /// Split DFly2 attention heads and MLP neurons across both RTX GPUs.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    pub(crate) tp2_dflash: bool,
    /// Compare serial greedy continuation to forced multi-row verify with those same tokens.
    #[arg(long, requires = "resume_at", conflicts_with = "resume_dfly2")]
    pub(crate) resume_verify: bool,
}

fn load_published_hy4_reference(root: &Path, vocab: usize) -> Result<cuteafd_bench::reference::Reference> {
    use cuteafd_bench::fidelity_dataset;
    load_published_reference(
        root,
        vocab,
        fidelity_dataset::HY4_CONFIG,
        fidelity_dataset::HY4_MANIFEST_SHA256,
        "Hy4-flash",
    )
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
    } else if args.verify_waves {
        (max_seq.div_ceil(64) * 2 + 16).max(64)
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
    let library = Arc::new(unsafe { NativeLibrary::load(lib_path)? });
    let programs = library.programs()?.with_manifest(&args.manifest)?;

    let mut device_ids = vec![args.device];
    if let Some(second) = args.split_device {
        ensure!(second != args.device, "--split-device must differ from --device");
        device_ids.push(second);
    }
    let devices: Vec<Device<'_>> = device_ids.iter().map(|&id| Device { library: &library, id }).collect();
    for dev in &devices {
        dev.run(|| crate::shared::inventory::preload_native_modules(&library))?;
    }
    let decode_rows = cuteafd_loader::plan::resolve_decode_rows("hy4_flash",
        cuteafd_loader::plan::decode_rows::DecodeRowsRequest {
            requested: args.decode_rows.as_option(),
            concurrency: if args.resume_at.is_empty() { 1 } else { 2 },
            decode_lanes: 1, rtx_gpus: devices.len(), draft_length: None,
        })? as usize;

    ensure!(
        args.expert_package.is_none() || !args.skip_routed_experts,
        "--expert-package conflicts with --skip-routed-experts"
    );
    ensure!(args.peers.is_empty() || (args.expert_package.is_none() && !args.skip_routed_experts),
        "--peers conflicts with --expert-package or --skip-routed-experts");
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
                    unit.max(decode_rows),
                    budget,
                    RtxSelectionPolicy::CanonicalDecode,
                )?;
                vec![Some(w.execution()?)]
            } else {
                let [w0, w1] = Exl3Tp2::load_pair_weights(
                    [devices[0], devices[1]],
                    &catalog,
                    cuteafd_core::ExpertGeometry::HY4_FLASH,
                    [pkg.as_path(), pkg.as_path()],
                    layers,
                    unit.max(decode_rows),
                    [budget, budget],
                    RtxSelectionPolicy::CanonicalDecode,
                )?;
                vec![Some(w0.execution()?), Some(w1.execution()?)]
            }
        } else {
            (0..devices.len()).map(|_| None).collect()
        };

    let prefill_waves = usize::from(args.prefill_waves);
    if prefill_waves > 1 {
        ensure!(
            unit % 64 == 0,
            "hy4f-golden --prefill-waves {prefill_waves} requires 64-row-aligned --prefill-units (got {unit})"
        );
    }
    let (_, attention) = super::serve::resolve_attention_placement(devices.len(), args.attention_placement.as_str())?;
    let context_split = attention == cuteafd_loader::placement::AttentionPlacement::Context;
    let local_pages = if context_split { max_pages.div_ceil(2).max(16) } else { max_pages };
    let draft_tp = if args.resume_dfly2 && args.tp2_dflash { 2 } else { 1 };
    ensure!(
        !args.tp2_dflash || (args.resume_dfly2 && devices.len() == 2),
        "hy4f-golden --tp2-dflash requires --resume-dfly2 and --split-device"
    );
    let mut engine = if !args.resume_at.is_empty() || args.verify_waves {
        let marks = if !args.resume_at.is_empty() {
            if args.resume_host { 2 } else { 4 }
        } else {
            0
        };
        Hy4Engine::new_serving(
            &devices,
            &programs,
            None,
            &checkpoint,
            cfg.clone(),
            args.kv_cache,
            attention,
            unit,
            decode_rows,
            prefill_waves,
            local_pages,
            2,
            local_pages * if context_split { 128 } else { 64 },
            local_pages * if context_split { 128 } else { 64 },
            marks,
            args.skip_routed_experts,
            onboarded,
            if draft_tp == 2 { 2 * 6 } else { 0 },
            Default::default(),
        )?
    } else {
        Hy4Engine::new(
            &devices,
            &programs,
            &checkpoint,
            cfg.clone(),
            args.kv_cache,
            attention,
            unit,
            decode_rows,
            prefill_waves,
            local_pages,
            args.skip_routed_experts,
            onboarded,
        )?
    };

    if !args.peers.is_empty() {
        let link = super::super::hy4_common::golden::spark_link(
            &library, devices[0], &selected, cuteafd_core::ExpertGeometry::HY4_FLASH,
            cfg.routed_layers(), &args.peers, unit, decode_rows, prefill_waves,
        )?;
        engine.attach_spark(link, Vec::new());
    }

    if args.resume_dfly2 {
        use cuteafd_loader::families::hy4_flash::{draft, Hy4DflyConfig};
        let model = hy4_plan::Model::open(&checkpoint)?;
        let config = Hy4DflyConfig::from_hf(&checkpoint.config, &cfg)?.context("no bundled DFly2")?;
        let layout = draft::Admission::new_tp_with_context(
            &checkpoint,
            &cfg,
            &config,
            &model.core,
            args.kv_cache,
            2,
            decode_rows,
            draft_tp,
            context_split,
        )?;
        let drafter = super::draft::Drafter::new(
            &devices[..draft_tp],
            &programs,
            &checkpoint,
            &cfg,
            config,
            layout,
            args.kv_cache,
            2,
            decode_rows,
        )?;
        engine.attach_drafter(drafter)?;
    }
    if args.resume_verify { engine.allocate_verify_keys()?; }
    engine.dump_dir = args.dump_dir.clone();
    if args.decode_graphs {
        let layout = cuteafd_loader::serving_capacity::hy4_graphs::Hy4GraphLayout::resolve(devices.len(), args.peers.len());
        engine.configure_graphs(layout)?;
    }

    if args.verify_waves {
        ensure!(prefill_waves >= 2, "--verify-waves requires --prefill-waves >= 2");
        let (prompt_a, prompt_b) = if let Some(ws) = &selected_windows {
            ensure!(ws.len() >= 2, "--verify-waves needs at least 2 distinct windows (e.g. --window a00,a01)");
            (ws[0].tokens.clone(), ws[1].tokens.clone())
        } else {
            ensure!(raw_tokens.len() >= unit * 4, "--verify-waves on raw tokens needs >= 4 * unit tokens");
            let half = (raw_tokens.len() / 2 / 64) * 64;
            (raw_tokens[..half].to_vec(), raw_tokens[half..half * 2].to_vec())
        };
        let proof = engine.verify_concurrent_two_seq_waves(&prompt_a, &prompt_b, unit, args.last_layer)?;
        for line in &proof.buffer_summary {
            println!("hy4f-golden --verify-waves buffer: {line}");
        }
        println!(
            "hy4f-golden --verify-waves PASS: gpus={} waves={} unit={} prompt_a={}({}u) digest_a={} tail_a={:?} prompt_b={}({}u) digest_b={} tail_b={:?} single_mw_diffs=0 conc_a_diffs=0 conc_b_diffs=0",
            proof.gpus,
            proof.waves,
            proof.unit_rows,
            proof.prompt_a_len,
            proof.units_a,
            digest_f32(&proof.logits_a_solo),
            proof.tail_a,
            proof.prompt_b_len,
            proof.units_b,
            digest_f32(&proof.logits_b_solo),
            proof.tail_b,
        );
        return Ok(());
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
        return run_resume_check(&mut engine, &tokens, &args.resume_at, args.resume_rows, unit,
            args.resume_host, args.resume_dfly2, args.resume_verify, args.decode_graphs, args.save_logits.as_deref());
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
                let logits = if prefill_waves > 1 {
                    engine.forward_span(&win.tokens, unit, args.last_layer)?
                } else {
                    let mut logits = Vec::with_capacity(win.tokens.len() * vocab);
                    for chunk in win.tokens.chunks(unit) {
                        let part = engine.forward_chunk(chunk, args.last_layer)?;
                        logits.extend_from_slice(&part);
                    }
                    logits
                };
                for pos in &win.positions {
                    let row_idx = pos.pos - 1;
                    scored_slices.push((pos.pos, logits[row_idx * vocab..(row_idx + 1) * vocab].to_vec()));
                }
                full_logits = Some(logits);
            }
            let mut scored_flat = Vec::with_capacity(win.positions.len() * vocab);
            for (_, slice) in &scored_slices {
                scored_flat.extend_from_slice(slice);
            }
            let score = score_window_direct_tail(win, &scored_slices, vocab);
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
                "hy4f-golden mode={} waves={} window={} rows={} digest={} scored_digest={} engram_tail={:?} mean NLL={:.6} (ref {:.6}) mean KL={:.6} (gen={:.6}/{:.2}%/{} ctx={:.6}/{:.2}%/{}) top-1={:.2}% top-3={:.2}%",
                mode,
                prefill_waves,
                win.id,
                score.positions,
                digest_str,
                scored_digest,
                engine.golden_engram_tail(),
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
    let logits = if prefill_waves > 1 {
        engine.forward_span(&tokens, unit, args.last_layer)?
    } else {
        let mut logits = Vec::new();
        for chunk in tokens.chunks(unit) {
            let part = engine.forward_chunk(chunk, args.last_layer)?;
            logits.extend_from_slice(&part);
        }
        logits
    };

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
    dfly2: bool,
    forced_verify: bool,
    graph_parity: bool,
    output: Option<&std::path::Path>,
) -> Result<()> {
    use super::cache::Hy4Placement;
    use super::execution::HeadMode;
    use crate::shared::prefill_pipeline::{sequence_units, Unit, UnitSegment};
    use crate::shared::prefix::CudaCopyEngine;
    use cuteafd_engine::prefix::{After, PrefixCache, PrefixConfig, PrefixFamily, SnapshotKind};
    use cuteafd_loader::formats::mapped_table::ReadPriority;

    ensure!(resume_rows > 0, "--resume-rows must be positive");
    if forced_verify {
        ensure!((2..=engine.decode_rows).contains(&resume_rows), "forced verify needs 2..={} rows", engine.decode_rows);
    }
    let dump_root = forced_verify.then(|| engine.dump_dir.take()).flatten();
    let mut verify_exact = true;
    let mut parity_logits = Vec::new();
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
        let mut units = sequence_units(slice.len(), unit_rows, engine.prefill_waves(), layers)?;
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

    let run_verify = |engine: &mut Hy4Engine<'_>, placement: &mut Hy4Placement, input: &[u32], keep: usize| -> Result<Vec<f32>> {
        let pos = placement.len;
        let rows = input.len();
        let unit = Unit {
            index: 0,
            sequence: 0,
            segments: vec![UnitSegment {
                sequence: 0,
                rows: 0..rows,
                tokens: pos..pos + rows,
                prompt_end: false,
                snapshot: false,
                head: true,
            }],
            rows: 0..rows,
            layers: 0..layers,
            slot: 0,
            barrier: false,
        };
        let mut next = [placement.clone()];
        let mut logits = Vec::new();
        engine.set_provisional(rows > 1)?;
        let result = engine.run_units(
            &mut next,
            &[input],
            &[unit],
            ReadPriority::Decode,
            |_| HeadMode::Suffix(0),
            &mut |_, _, tile| {
                logits = tile.to_host(library)?;
                Ok(())
            },
        );
        engine.set_provisional(false)?;
        result?;
        let mut next = next.into_iter().next().unwrap();
        if rows > 1 {
            engine.commit_provisional_keys(placement.slot, pos, 0, keep)?;
            engine.accept_draft(placement.slot, keep - 1, pos + keep)?;
            next.len = pos + keep;
            next.state_len = next.len;
            next.engram_tail = cuteafd_loader::families::hy4_flash::advance_tail(placement.engram_tail, input, keep);
        }
        *placement = next;
        logits.truncate(keep * vocab);
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

        if forced_verify {
            let greedy = |logits: &[f32]| -> Result<u32> {
                ensure!(logits.iter().all(|f| f.is_finite()), "nonfinite target logits in forced verify");
                let mut best = 0;
                for i in 1..logits.len() { if logits[i] > logits[best] { best = i; } }
                Ok(best as u32)
            };
            let mut next = greedy(&logits_p)?;
            let mut input = Vec::with_capacity(resume_rows);
            let mut serial = Vec::with_capacity(resume_rows);
            for j in 0..resume_rows {
                input.push(next);
                engine.dump_dir = if j == 0 { dump_root.as_ref().map(|p| p.join(format!("{at}/singleton"))) } else { None };
                let logits = run_verify(engine, &mut p_a, &[next], 1)?;
                next = greedy(&logits)?;
                serial.push(logits);
            }
            engine.dump_dir = dump_root.as_ref().map(|p| p.join(format!("{at}/batch")));
            let batched = run_verify(engine, &mut p_b, &input, input.len())?;
            engine.dump_dir = None;
            let mut exact = 0;
            let mut matching_ids = 0;
            for (j, (a, b)) in serial.iter().zip(batched.chunks_exact(vocab)).enumerate() {
                let mismatches = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                let max_abs = a.iter().zip(b).map(|(x, y)| (x-y).abs()).fold(0.0f32, f32::max);
                let ids_equal = greedy(a)? == greedy(b)?;
                exact += usize::from(mismatches == 0);
                matching_ids += usize::from(ids_equal);
                println!("hy4f-golden forced verify prefix={at} row={j} hash_match={} mismatches={mismatches} max_abs={max_abs:.8e} argmax_match={ids_equal}", mismatches == 0);
            }
            let left = engine.dump_state(&p_a)?;
            let right = engine.dump_state(&p_b)?;
            let state_exact = left.iter().zip(&right).all(|(a, b)| a.diff(b).is_empty());
            println!("hy4f-golden forced verify prefix={at} exact={exact}/{resume_rows} argmax={matching_ids}/{resume_rows} state_exact={state_exact}");
            verify_exact &= exact == resume_rows && state_exact;
            cache.release(&family, &p_a.units)?;
            cache.release(&family, &p_b.units)?;
            continue;
        }

        if dfly2 {
            // No physical recall pointers live in the mark. The restored
            // request must recover them from its first ordinary target step.
            engine.invalidate_draft(p_b.slot);
            let started = std::time::Instant::now();
            let candidates = engine.draft_propose(&p_a, tokens[at], 5)?;
            let draft_us = started.elapsed().as_micros();
            ensure!(!candidates.is_empty(), "DFly2 fresh request produced no proposals");
            ensure!(engine.draft_propose(&p_b, tokens[at], 5)?.is_empty(),
                "DFly2 restored request proposed from stale taps");
            println!("hy4f-golden DFly2 resume={at}: restored request skips one proposal; fresh draft_us={draft_us} proposed={} ids={candidates:?}", candidates.len());
        }

        // 4) 1-token-at-a-time decode continuation comparison: A (straight) vs B (restored).
        let mut max_abs_diff = 0.0f32;
        let mut bitwise_eq = 0usize;
        let mut top1_eq = 0usize;
        let mut first_diff_step: Option<(usize, f32)> = None;
        for step in 0..resume_rows {
            let tok = tokens[at + step];
            let verify = if dfly2 && step == 1 {
                // Submit the same accepted token with deliberately rejected
                // tails on both paths; a following ordinary step exercises
                // all four Engram layers using the recovered history.
                vec![tok, (tok + 1) % vocab as u32, (tok + 2) % vocab as u32, (tok + 3) % vocab as u32]
            } else { vec![tok] };
            let la = run_verify(engine, &mut p_a, &verify, 1)?;
            let lb = run_verify(engine, &mut p_b, &verify, 1)?;
            if output.is_some() {
                parity_logits.extend(la.iter().flat_map(|logit| logit.to_le_bytes()));
            }
            if dfly2 {
                ensure!(engine.draft_valid(&p_b), "DFly2 restore did not rejoin after one target row");
                let expected = &tokens[..at + step + 1];
                let tail = &expected[expected.len().saturating_sub(2)..];
                ensure!(p_a.engram_tail.ends_with(tail) && p_b.engram_tail.ends_with(tail),
                    "DFly2 rejected rows advanced Engram state");
                if step == 0 {
                    // Compare the same committed taps, in singleton and two-member
                    // schedules, with distinct anchors and a different depth per member.
                    let members = [&p_a, &p_b].into_iter().enumerate().map(|(i, p)| {
                        super::draft::Member { slot: p.slot, pos: p.len,
                            bonus: if i == 0 { tok } else { (tok + 1) % vocab as u32 }, depth: 5 - i * 2,
                            rope_delta: p.rope.delta() }
                    }).collect::<Vec<_>>();
                    let separate = members.iter().map(|m| engine.draft_batch(std::slice::from_ref(m))
                        .map(|mut batch| batch.pop().expect("singleton DFly2 output")))
                        .collect::<Result<Vec<_>>>()?;
                    let together = engine.draft_batch(&members)?;
                    if graph_parity {
                        let eager = engine.without_graphs(|engine| engine.draft_batch(&members))?;
                        // One eager first-touch, then replay the warmed exact member count.
                        for _ in 0..2 {
                            engine.graph_completed()?;
                            let graphed = engine.draft_batch(&members)?;
                            ensure!(graphed == eager, "DFly2 eager and graphed proposal IDs differ at {at}");
                        }
                        engine.graph_completed()?;
                        ensure!(engine.retained_draft_graph_rows(members.len()) > 0,
                            "DFly2 parity did not retain graphs at {} live members", members.len());
                    }
                    let matches = together == separate;
                    if at + 1 <= 2048 {
                        ensure!(matches, "DFly2 batched proposals differ from singleton: {together:?} vs {separate:?}");
                    }
                    println!("hy4f-golden DFly2 batch at {at}: proposal ids match={matches} together={together:?} separate={separate:?}");
                }
            }
            if step == 0 || (dfly2 && step <= 2) {
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
    if dfly2 && graph_parity {
        for rows in [1, 4] {
            ensure!(engine.retained_graph_rows(rows).iter().all(|&(_, count)| count > 0),
                "DFly2 parity did not retain target graphs at {rows} rows");
        }
    }
    if let Some(output) = output {
        ensure!(!parity_logits.is_empty(), "resume check did not produce logits to compare");
        std::fs::create_dir_all(output)?;
        std::fs::write(output.join("logits.bin"), parity_logits)?;
    }
    ensure!(verify_exact, "Hy4 forced target verify differs from serial greedy decode");
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
    fn hy4f_golden_cli_parses_peers() -> anyhow::Result<()> {
        let crate::cli::Commands::Hy4fGolden(args) = crate::cli::Cli::try_parse_from([
            "cuteafd", "hy4f-golden", "--snapshot", "/snapshot", "--golden", "/golden",
            "--peers", "127.0.0.1:19441,127.0.0.1:19442", "--decode", "--decode-graphs",
        ])?.command else { anyhow::bail!("expected Hy4fGolden") };
        assert_eq!(args.peers.len(), 2);
        assert!(args.decode && args.decode_graphs);
        Ok(())
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
