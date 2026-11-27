//! Hy4 (`hy_v4_vl`) coordinator golden runner (`hy4-golden`) for raptor RTX + Sparks (#2550).
//!
//! Compares against the published Hy4 fidelity set (`hy4-v1_20261122_own-text-bf16-f32`,
//! `HY4_FULL_MANIFEST_SHA256`), printing both all-rows and **GENERATED-rows-only** verdicts,
//! and verifying exact pages-only prefix restore and graphed vs eager byte-identity.
use super::cache::Hy4Placement;
use super::execution::Hy4Engine;
use crate::families::hy4_common::golden::{
    digest_f32, load_published_reference, read_u32_bin, score_window_direct_tail,
};
use crate::shared::golden_scores::{scored_nll, ReferenceRows};
use crate::shared::memory::device::Device;
use anyhow::{ensure, Context, Result};
use cuteafd_engine::prefix::{After, PrefixCache, PrefixConfig, PrefixFamily, SnapshotKind};
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::families::hy4::{Hy4Config, Hy4KvCache};
use cuteafd_loader::plan::checkpoint::{Checkpoint, CheckpointView};
use cuteafd_loader::plan::families::hy4 as hy4_plan;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug, Clone, clap::Args)]
pub(crate) struct GoldenArgs {
    #[command(flatten)]
    pub(crate) attention: crate::cli::CoreViewArgs,
    /// KV cache storage format (matches serving default `fp8`).
    #[arg(long, default_value = "fp8")]
    pub(crate) kv_cache: Hy4KvCache,
    /// Checkpoint snapshot directory (e.g. Hy4-EXL3-K4-MC).
    #[arg(long)]
    pub(crate) snapshot: PathBuf,
    #[arg(long, env = "CUTEAFD_NATIVE_LIB")]
    pub(crate) native_lib: Option<PathBuf>,
    #[arg(long, default_value = "/opt/cuteafd/share/PROGRAMS.json")]
    pub(crate) manifest: PathBuf,
    /// Directory containing published fidelity set (`manifest.json`) or `tokens.bin`.
    #[arg(long)]
    pub(crate) golden: Option<PathBuf>,
    /// Optional window id(s) (comma-separated, or `all-quick`).
    #[arg(long)]
    pub(crate) window: Option<String>,
    /// Last target layer inclusive (0 for layer-0 dense check; 77 for full logits).
    #[arg(long, default_value_t = 77)]
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
    /// Validate checkpoint headers, MC views, and target arenas without CUDA.
    #[arg(long)]
    pub(crate) check_headers: bool,
    /// Compute NLL/KL/top-1 against reference tokens and optional logits.
    #[arg(long)]
    pub(crate) nll: bool,
    /// Chunk size (rows per unit) for prefill execution.
    #[arg(long)]
    pub(crate) prefill_units: Option<usize>,
    /// Prefill wave slots in flight (`1..=3`).
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(1..=3))]
    pub(crate) prefill_waves: u8,
    #[arg(long, default_value = "auto")]
    pub(crate) decode_rows: cuteafd_loader::plan::decode_rows::DecodeRowsSetting,
    /// Serving-shaped N=1 decode parity: prefill `0..score_from-1`, then step 1 token at a time.
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
    /// Diagnostic: zero routed experts to isolate non-expert backbone (iHC, MLA, IndexCache, shared FFN).
    #[arg(long, hide = true)]
    pub(crate) skip_routed_experts: bool,
    /// Optional path to `exl3-hy4-k45-mul1/rtx-tp{1,2}` routed expert package directory.
    #[arg(long)]
    pub(crate) expert_package: Option<PathBuf>,
    /// Number of leading routed MoE layers to onboard on RTX (hybrid RTX+Spark parity).
    #[arg(long)]
    pub(crate) rtx_expert_layers: Option<usize>,
    /// Exact pages-only prefix-restore gate: prefill `tokens[..P]`, capture, restore into slot 1,
    /// assert 0 state bytes differ at P, and compare 1-token decode continuations.
    #[arg(long, value_delimiter = ',')]
    pub(crate) resume_at: Vec<usize>,
    /// Number of decode tokens to compare past each `--resume-at` point.
    #[arg(long, default_value_t = 16)]
    pub(crate) resume_rows: usize,
    /// Exercise bundled 5-layer DFly2 during `--resume-at` (invalidation, rejoin, batch & graph parity).
    #[arg(long)]
    pub(crate) resume_dfly2: bool,
    /// Split DFly2 across both RTX GPUs during `--resume-dfly2` (default on 2x RTX).
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    pub(crate) tp2_dflash: bool,
    #[arg(skip)]
    pub(crate) tp2_dflash_explicit: bool,
    /// Exercise forced multi-row speculative verify during `--resume-at` (bit-exact vs serial decode).
    #[arg(long)]
    pub(crate) resume_verify: bool,
}

fn load_published_hy4_reference(
    root: &Path,
    vocab: usize,
) -> Result<cuteafd_bench::reference::Reference> {
    use cuteafd_bench::fidelity_dataset;
    load_published_reference(
        root,
        vocab,
        fidelity_dataset::HY4_FULL_CONFIG,
        fidelity_dataset::HY4_FULL_MANIFEST_SHA256,
        "Hy4",
    )
}

fn check_headers_only(args: &GoldenArgs) -> Result<()> {
    let view = args.attention.core_view;
    let selected = CheckpointView::select(&args.snapshot, view)?;
    let checkpoint = Checkpoint::open_view(&selected)?;
    let cfg = Hy4Config::read(&args.snapshot)?;
    let model = hy4_plan::Model::open(&checkpoint)?;
    let gpus = if args.split_device.is_some() { 2 } else { 1 };
    let modes = cuteafd_loader::placement::families::hy4::layer_modes(
        &cfg,
        gpus,
        cuteafd_loader::placement::AttentionPlacement::Context,
    );
    let arenas = cuteafd_loader::families::hy4::weight_layout::target_arenas(
        &checkpoint,
        &cfg,
        &model.core,
        &modes,
        gpus,
    )?;
    let total_arena_bytes: usize = arenas.iter().map(|a| a.arena.bytes()).sum();
    println!(
        "hy4-golden check-headers OK: view={view:?} gpus={} layers={} full_indexers={} shared_indexers={} target_arena_bytes={}",
        gpus,
        cfg.layers,
        cfg.index_cache.full_count(),
        cfg.index_cache.shared_count(),
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
    let cfg = Hy4Config::read(&args.snapshot)?;
    cfg.release_geometry()?;

    let golden_dir = args
        .golden
        .as_ref()
        .context("hy4-golden requires --golden <DIR> (unless --check-headers)")?;

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
                        .with_context(|| format!("Hy4 fidelity window {id:?} not found"))?;
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
                        .with_context(|| format!("Hy4 fidelity window {id:?} not found"))?;
                    ws.push(win.clone());
                }
                ws
            }
            None => cuteafd_bench::fidelity_dataset::quick_windows(ref_set)?,
        };
        ensure!(!wins.is_empty(), "Hy4 fidelity run selected zero windows");
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

    if let Err(prev) = cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::HY4) {
        ensure!(
            prev == cuteafd_core::ExpertGeometry::HY4,
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
    let devices: Vec<Device<'_>> = device_ids
        .iter()
        .map(|&id| Device {
            library: &library,
            id,
        })
        .collect();
    for dev in &devices {
        dev.run(|| crate::shared::inventory::preload_native_modules(&library))?;
    }
    let decode_rows = cuteafd_loader::plan::resolve_decode_rows(
        "hy4",
        cuteafd_loader::plan::decode_rows::DecodeRowsRequest {
            requested: args.decode_rows.as_option(),
            concurrency: if args.resume_at.is_empty() { 1 } else { 2 },
            decode_lanes: 1,
            rtx_gpus: devices.len(),
            draft_length: if args.resume_dfly2 || args.resume_verify {
                None
            } else {
                Some(0)
            },
        },
    )? as usize;

    let routed_layers: Vec<usize> = cfg
        .routed_layers()
        .into_iter()
        .filter(|&l| l <= args.last_layer)
        .collect();
    let onboard_count = match (args.rtx_expert_layers, &args.expert_package) {
        (Some(n), _) => {
            ensure!(
                n <= routed_layers.len(),
                "--rtx-expert-layers {n} exceeds {} routed layers",
                routed_layers.len()
            );
            n
        }
        (None, Some(_)) => routed_layers.len(),
        (None, None) => 0,
    };
    ensure!(
        onboard_count == 0 || !args.skip_routed_experts,
        "--rtx-expert-layers / --expert-package conflicts with --skip-routed-experts"
    );
    let onboarded_layers = routed_layers[..onboard_count].to_vec();
    let spark_layers = routed_layers[onboard_count..].to_vec();

    let onboarded: Vec<Option<crate::shared::experts::rtx::exl3::Exl3Tp2<'_>>> =
        if !onboarded_layers.is_empty() {
            use crate::shared::experts::rtx::{exl3::Exl3Tp2, RtxSelectionPolicy};
            let catalog = cuteafd_loader::read_expert_catalog_view(
                checkpoint
                    .view
                    .as_ref()
                    .context("Hy4 checkpoint view required for routed experts")?,
            )?;
            let pkg = super::serve::resolve_expert_package(
                lib_path,
                args.expert_package.as_deref(),
                devices.len(),
                &catalog,
            )?;
            let budget = 80usize << 30;
            if devices.len() == 1 {
                let w = Exl3Tp2::load_whole_weights(
                    devices[0],
                    &catalog,
                    &pkg,
                    onboarded_layers.clone(),
                    unit.max(decode_rows),
                    budget,
                    RtxSelectionPolicy::CanonicalDecode,
                )?;
                vec![Some(w.execution()?)]
            } else {
                let [w0, w1] = Exl3Tp2::load_pair_weights(
                    [devices[0], devices[1]],
                    &catalog,
                    cuteafd_core::ExpertGeometry::HY4,
                    [pkg.as_path(), pkg.as_path()],
                    onboarded_layers.clone(),
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
    let (_, attention) =
        super::serve::resolve_attention_placement(devices.len(), args.attention_placement.as_str())?;
    let context_split = attention == cuteafd_loader::placement::AttentionPlacement::Context;
    let local_pages = if context_split {
        max_pages.div_ceil(2).max(16)
    } else {
        max_pages
    };
    let manifest_opt = args.manifest.is_file().then_some(args.manifest.as_path());
    let draft_admission = if args.resume_dfly2 {
        ensure!(
            !args.resume_at.is_empty(),
            "--resume-dfly2 requires --resume-at"
        );
        let dfly_cfg = cuteafd_loader::families::hy4::Hy4DflyConfig::from_hf(
            &checkpoint.config,
            "hy4",
            cfg.layers,
            cfg.vocab_size,
        )?
        .context("Hy4 --resume-dfly2 requires a checkpoint with bundled DFly2")?;
        let draft_tp = if args.tp2_dflash && devices.len() == 2 {
            2
        } else {
            1
        };
        let model = hy4_plan::Model::open(&checkpoint)?;
        let adm = cuteafd_loader::families::hy4::draft::Admission::new_tp_with_context(
            &checkpoint,
            &cfg,
            &dfly_cfg,
            &model.core,
            args.kv_cache,
            2,
            decode_rows,
            draft_tp,
            context_split,
        )?;
        Some((dfly_cfg, adm))
    } else {
        None
    };
    let draft_peer_rows = draft_admission
        .as_ref()
        .filter(|(_, adm)| adm.tp == 2)
        .map_or(0, |(_, adm)| adm.draft_rows());
    let mut engine = if !args.resume_at.is_empty() {
        Hy4Engine::new_serving(
            &devices,
            &programs,
            None,
            None,
            manifest_opt,
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
            args.skip_routed_experts,
            onboarded,
            draft_peer_rows,
            Default::default(),
        )?
    } else {
        Hy4Engine::new(
            &devices,
            &programs,
            manifest_opt,
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
    if let Some((dfly_cfg, adm)) = draft_admission {
        let draft_tp = adm.tp;
        let drafter = super::draft::Drafter::new(
            &devices[..draft_tp],
            &programs,
            &checkpoint,
            &cfg,
            dfly_cfg,
            adm,
            args.kv_cache,
            2,
            decode_rows,
        )?;
        engine.attach_drafter(drafter)?;
    }

    if !args.peers.is_empty() && !args.skip_routed_experts && !spark_layers.is_empty() {
        let link = super::super::hy4_common::golden::spark_link(
            &library,
            devices[0],
            &selected,
            cuteafd_core::ExpertGeometry::HY4,
            onboarded_layers.clone(),
            spark_layers,
            &args.peers,
            unit,
            decode_rows,
            prefill_waves,
        )?;
        engine.attach_spark(link, onboarded_layers);
    }

    engine.dump_dir = args.dump_dir.clone();
    if args.decode_graphs {
        use cuteafd_loader::families::hy4::graph_reserve;
        let layout = graph_reserve::Layout::resolve(devices.len(), args.peers.len(), attention);
        let policies: Vec<_> = (0..devices.len())
            .map(|rank| graph_reserve::reserve_policy(layout, rank))
            .collect::<std::result::Result<_, _>>()?;
        engine.configure_graphs(&policies)?;
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
        }
        for &point in &args.resume_at {
            ensure!(
                point > 0 && point + args.resume_rows <= tokens.len(),
                "--resume-at {point} + {} exceeds token length {}",
                args.resume_rows,
                tokens.len()
            );
            let family = engine.prefix()?;
            let mut cache = PrefixCache::<crate::shared::prefix::CudaCopyEngine<'_>>::new(
                family.layout(),
                PrefixConfig {
                    entries: 4,
                    mark_slots: 0,
                    keep_logits: false,
                    min_tokens: 1,
                },
                None,
            )?;
            let cap_tokens = point + args.resume_rows + 64;
            let prefix_toks = &tokens[..point];
            let tail_toks = &tokens[point..point + args.resume_rows];
            let mut p0 = cache
                .admit(&family, prefix_toks, cap_tokens, false, |units| Hy4Placement::new(units, 0))?
                .placement;
            let vocab = cfg.vocab_size;

            // Fresh prefill of prefix into slot 0, capture pages-only snapshot + last-row logits.
            let mut units = Vec::new();
            for (idx, off) in (0..prefix_toks.len()).step_by(unit).enumerate() {
                let len = (prefix_toks.len() - off).min(unit);
                let is_last = off + len == prefix_toks.len();
                units.push(crate::shared::prefill_pipeline::Unit {
                    index: idx,
                    sequence: 0,
                    segments: vec![crate::shared::prefill_pipeline::UnitSegment {
                        sequence: 0,
                        rows: 0..len,
                        tokens: off..off + len,
                        prompt_end: is_last,
                        snapshot: false,
                        head: is_last,
                    }],
                    rows: off..off + len,
                    layers: 0..cfg.layers,
                    slot: idx % prefill_waves,
                    barrier: false,
                });
            }
            let last_idx = units.len() - 1;
            let lib = engine.library();
            let mut p0_arr = [p0.clone()];
            let mut logits_p = Vec::new();
            engine.run_units(
                &mut p0_arr,
                &[prefix_toks],
                &units,
                cuteafd_loader::formats::mapped_table::ReadPriority::Prefill,
                |u| {
                    if u.index == last_idx {
                        super::execution::HeadMode::LastPerSeq
                    } else {
                        super::execution::HeadMode::None
                    }
                },
                &mut |_, _, tile| {
                    logits_p = tile.to_host(lib)?;
                    Ok(())
                },
            )?;
            p0 = p0_arr.into_iter().next().unwrap();
            let snap_before = engine.dump_state(&p0)?;
            ensure!(
                cache.capture(
                    &family,
                    SnapshotKind::Prompt,
                    prefix_toks,
                    &p0,
                    After::from_logits(&logits_p, true),
                )?,
                "Hy4 prefix capture at {point} returned false"
            );

            // Restore into slot 1 and verify 0 state bytes differ.
            let restored = cache.admit(
                &family,
                &tokens[..point + args.resume_rows],
                cap_tokens,
                false,
                |u| Hy4Placement::new(u, 1),
            )?;
            ensure!(restored.resume == point, "restored prefix_len {} != {point}", restored.resume);
            let mut p1 = restored.placement;
            let snap_after = engine.dump_state(&p1)?;
            for (r, (a, b)) in snap_before.iter().zip(&snap_after).enumerate() {
                let diffs = a.diff(b);
                ensure!(diffs.is_empty(), "Hy4 prefix restore state diff on rank {r} at P={point}: {diffs:?}");
            }

            let run_verify_step = |engine: &mut Hy4Engine<'_>,
                                   placement: &mut Hy4Placement,
                                   input: &[u32],
                                   keep: usize|
             -> Result<Vec<f32>> {
                let pos = placement.len;
                let rows = input.len();
                let u = crate::shared::prefill_pipeline::Unit {
                    index: 0,
                    sequence: 0,
                    segments: vec![crate::shared::prefill_pipeline::UnitSegment {
                        sequence: 0,
                        rows: 0..rows,
                        tokens: pos..pos + rows,
                        prompt_end: false,
                        snapshot: false,
                        head: true,
                    }],
                    rows: 0..rows,
                    layers: 0..cfg.layers,
                    slot: 0,
                    barrier: false,
                };
                let mut p_arr = [placement.clone()];
                let mut out = Vec::new();
                engine.set_provisional(rows > 1)?;
                let res = engine.run_units(
                    &mut p_arr,
                    &[input],
                    &[u],
                    cuteafd_loader::formats::mapped_table::ReadPriority::Decode,
                    |_| super::execution::HeadMode::Suffix(0),
                    &mut |_, _, tile| {
                        out.extend_from_slice(&tile.to_host(lib)?);
                        Ok(())
                    },
                );
                engine.set_provisional(false)?;
                res?;
                let mut next = p_arr.into_iter().next().unwrap();
                if rows > 1 {
                    engine.accept_draft(placement.slot, keep - 1, pos + keep)?;
                    next.len = pos + keep;
                }
                *placement = next;
                out.truncate(keep * vocab);
                Ok(out)
            };

            if args.resume_verify {
                ensure!(
                    (2..=decode_rows).contains(&args.resume_rows),
                    "--resume-verify needs --resume-rows in 2..={decode_rows}"
                );
                let greedy = |logits: &[f32]| -> Result<u32> {
                    ensure!(
                        logits.iter().all(|f| f.is_finite()),
                        "nonfinite target logits in Hy4 forced verify"
                    );
                    let mut best = 0;
                    for i in 1..logits.len() {
                        if logits[i] > logits[best] {
                            best = i;
                        }
                    }
                    Ok(best as u32)
                };
                let mut next = greedy(&logits_p)?;
                let mut input = Vec::with_capacity(args.resume_rows);
                let mut serial = Vec::with_capacity(args.resume_rows);
                for _ in 0..args.resume_rows {
                    input.push(next);
                    let logits = run_verify_step(&mut engine, &mut p0, &[next], 1)?;
                    next = greedy(&logits)?;
                    serial.push(logits);
                }
                let batched = run_verify_step(&mut engine, &mut p1, &input, input.len())?;
                let mut exact = 0usize;
                let mut argmax_eq = 0usize;
                for (j, (a, b)) in serial.iter().zip(batched.chunks_exact(vocab)).enumerate() {
                    let mismatches = a
                        .iter()
                        .zip(b)
                        .filter(|(x, y)| x.to_bits() != y.to_bits())
                        .count();
                    let max_abs = a
                        .iter()
                        .zip(b)
                        .map(|(x, y)| (x - y).abs())
                        .fold(0.0f32, f32::max);
                    let ids_eq = greedy(a)? == greedy(b)?;
                    exact += usize::from(mismatches == 0);
                    argmax_eq += usize::from(ids_eq);
                    println!(
                        "hy4-golden forced verify prefix={point} row={j} exact={} mismatches={mismatches} max_abs={max_abs:.6e} argmax_match={ids_eq}",
                        mismatches == 0
                    );
                }
                let left = engine.dump_state(&p0)?;
                let right = engine.dump_state(&p1)?;
                let state_exact = left.iter().zip(&right).all(|(a, b)| a.diff(b).is_empty());
                println!(
                    "hy4-golden forced verify prefix={point}: exact={exact}/{} argmax={argmax_eq}/{} state_exact={state_exact}",
                    args.resume_rows, args.resume_rows
                );
                cache.release(&family, &p0.units)?;
                cache.release(&family, &p1.units)?;
                ensure!(
                    argmax_eq == args.resume_rows && state_exact,
                    "Hy4 forced verify mismatch at P={point}: argmax={argmax_eq}/{}, state_exact={state_exact}",
                    args.resume_rows
                );
                continue;
            }

            if args.resume_dfly2 {
                engine.invalidate_draft(p1.slot);
                let started = std::time::Instant::now();
                let candidates = engine.draft_propose(&p0, tokens[point], 6)?;
                let draft_us = started.elapsed().as_micros();
                ensure!(
                    candidates.len() == 6,
                    "Hy4 DFly2 fresh request produced {} proposals, expected 6",
                    candidates.len()
                );
                ensure!(
                    engine.draft_propose(&p1, tokens[point], 6)?.is_empty(),
                    "Hy4 DFly2 restored request proposed from stale taps"
                );
                println!(
                    "hy4-golden DFly2 resume={point}: restored skips 1 proposal; fresh draft_us={draft_us} proposed={} ids={candidates:?}",
                    candidates.len()
                );
            }

            // Decode `resume_rows` steps from both p0 (fresh) and p1 (restored) and compare logits.
            let mut logits_fresh = Vec::with_capacity(tail_toks.len() * vocab);
            let mut logits_restored = Vec::with_capacity(tail_toks.len() * vocab);
            for (step, &tok) in tail_toks.iter().enumerate() {
                let verify = if args.resume_dfly2 && step == 1 {
                    vec![
                        tok,
                        (tok + 1) % vocab as u32,
                        (tok + 2) % vocab as u32,
                        (tok + 3) % vocab as u32,
                    ]
                } else {
                    vec![tok]
                };
                let la = run_verify_step(&mut engine, &mut p0, &verify, 1)?;
                let lb = run_verify_step(&mut engine, &mut p1, &verify, 1)?;
                logits_fresh.extend_from_slice(&la);
                logits_restored.extend_from_slice(&lb);
                if args.resume_dfly2 {
                    ensure!(
                        engine.draft_valid(&p1),
                        "Hy4 DFly2 restore did not rejoin after one target step at P={point}"
                    );
                    if step == 0 {
                        let members = [&p0, &p1]
                            .into_iter()
                            .enumerate()
                            .map(|(i, p)| super::draft::Member {
                                slot: p.slot,
                                pos: p.len,
                                bonus: if i == 0 { tok } else { (tok + 1) % vocab as u32 },
                                depth: 6 - i * 2,
                                rope_delta: 0,
                            })
                            .collect::<Vec<_>>();
                        let separate = members
                            .iter()
                            .map(|m| {
                                engine
                                    .draft_batch(std::slice::from_ref(m))
                                    .map(|mut batch| batch.pop().expect("singleton Hy4 DFly2 output"))
                            })
                            .collect::<Result<Vec<_>>>()?;
                        let together = engine.draft_batch(&members)?;
                        if args.decode_graphs {
                            let eager = engine.without_graphs(|e| e.draft_batch(&members))?;
                            for _ in 0..2 {
                                engine.graph_completed()?;
                                let graphed = engine.draft_batch(&members)?;
                                ensure!(
                                    graphed == eager,
                                    "Hy4 DFly2 eager and graphed proposal IDs differ at P={point}: {graphed:?} vs {eager:?}"
                                );
                            }
                            engine.graph_completed()?;
                            ensure!(
                                engine.retained_draft_graph_rows(members.len()) > 0,
                                "Hy4 DFly2 parity did not retain draft graphs at {} members",
                                members.len()
                            );
                        }
                        let matches = together == separate;
                        if point + 1 <= 2048 {
                            ensure!(
                                matches,
                                "Hy4 DFly2 batched proposals differ from singleton at P={point}: {together:?} vs {separate:?}"
                            );
                        }
                        println!(
                            "hy4-golden DFly2 batch at {point}: proposal_ids_match={matches} together={together:?} separate={separate:?}"
                        );
                    }
                }
            }
            cache.release(&family, &p0.units)?;
            cache.release(&family, &p1.units)?;
            let mut max_diff = 0f32;
            let mut byte_diffs = 0usize;
            for (&a, &b) in logits_fresh.iter().zip(&logits_restored) {
                if a.to_bits() != b.to_bits() {
                    byte_diffs += 1;
                }
                max_diff = max_diff.max((a - b).abs());
            }
            println!(
                "hy4-golden --resume-at {point} PASS: state_bytes={} state_diffs=0 resume_rows={} logit_byte_diffs={byte_diffs} max_abs_diff={max_diff:.6e} digest={}",
                snap_before.iter().map(|s| s.total_bytes()).sum::<usize>(),
                args.resume_rows,
                digest_f32(&logits_restored),
            );
            if point + args.resume_rows <= 2048 {
                ensure!(
                    byte_diffs == 0,
                    "Hy4 restored decode logits differed from fresh at P={point}"
                );
            }
        }
        return Ok(());
    }

    if let Some(windows) = &selected_windows {
        let vocab = cfg.vocab_size;
        let mut all_records = Vec::new();
        for win in windows {
            engine.reset_golden()?;
            let mut scored_slices = Vec::with_capacity(win.positions.len());
            let mut full_logits = None;
            if args.decode && win.score_from > 1 && win.score_from < win.tokens.len() {
                let split = win.score_from - 1;
                let _ = engine.forward_golden(&win.tokens[..split], 0, args.last_layer, unit, false)?;
                let tail = engine.forward_golden(&win.tokens[split..], split, args.last_layer, 1, true)?;
                for pos in &win.positions {
                    let row_idx = pos.pos - 1;
                    ensure!(row_idx >= split, "Hy4 decode window {} scored pos {} < split {split}", win.id, pos.pos);
                    let local = row_idx - split;
                    scored_slices.push((pos.pos, tail[local * vocab..(local + 1) * vocab].to_vec()));
                }
            } else {
                let logits = engine.forward_golden(&win.tokens, 0, args.last_layer, unit, args.decode)?;
                for pos in &win.positions {
                    let row_idx = pos.pos - 1;
                    scored_slices.push((pos.pos, logits[row_idx * vocab..(row_idx + 1) * vocab].to_vec()));
                }
                full_logits = Some(logits);
            }
            if let Some(save) = &args.save_logits {
                if windows.len() == 1 {
                    std::fs::create_dir_all(save)?;
                    let mut flat = Vec::new();
                    let to_save = if let Some(full) = &full_logits {
                        full.as_slice()
                    } else {
                        for (_, s) in &scored_slices {
                            flat.extend_from_slice(s);
                        }
                        flat.as_slice()
                    };
                    let bytes: Vec<u8> = to_save.iter().flat_map(|f| f.to_le_bytes()).collect();
                    std::fs::write(save.join("logits.bin"), bytes)?;
                }
            }
            if args.last_layer == cfg.layers - 1 {
                let score = score_window_direct_tail(win, &scored_slices, vocab);
                let gen_win = cuteafd_bench::reference::Fidelity::from_records(
                    score.records.iter().filter(|r| r.role == "gen").cloned().collect(),
                );
                let ctx_win = cuteafd_bench::reference::Fidelity::from_records(
                    score.records.iter().filter(|r| r.role == "ctx").cloned().collect(),
                );
                let mode = if args.decode { "decode" } else { "prefill" };
                let digest_str = full_logits.as_deref().map(digest_f32).unwrap_or_default();
                println!(
                    "hy4-golden mode={} waves={} window={} rows={} digest={} mean NLL={:.6} (ref {:.6}) mean KL={:.6} (gen={:.6}/{:.2}%/{} ctx={:.6}/{:.2}%/{}) top-1={:.2}% top-3={:.2}%",
                    mode,
                    prefill_waves,
                    win.id,
                    score.positions,
                    digest_str,
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
                all_records.extend(score.records);
            }
        }
        if args.last_layer == cfg.layers - 1 && !all_records.is_empty() {
            let gen_records: Vec<_> = all_records.iter().filter(|r| r.role == "gen").cloned().collect();
            let ctx_records: Vec<_> = all_records.iter().filter(|r| r.role == "ctx").cloned().collect();
            let agg = cuteafd_bench::reference::Fidelity::from_records(all_records);
            let gen_agg = cuteafd_bench::reference::Fidelity::from_records(gen_records);
            let ctx_agg = cuteafd_bench::reference::Fidelity::from_records(ctx_records);
            println!(
                "hy4-golden aggregate ({} windows, {} rows): mean NLL={:.6} (ref {:.6}, delta {:+.6}) mean KL={:.6} top-1={:.2}% top-3={:.2}%",
                windows.len(), agg.positions, agg.nll, agg.ref_nll, agg.nll - agg.ref_nll,
                agg.kl, agg.top1 * 100.0, agg.top3_contained * 100.0,
            );
            if gen_agg.positions > 0 {
                println!(
                    "hy4-golden generated-only ({} rows, verdict subset): mean NLL={:.6} (ref {:.6}, delta {:+.6}) mean KL={:.6} top-1={:.2}% top-3={:.2}% graphs_retained={}",
                    gen_agg.positions, gen_agg.nll, gen_agg.ref_nll, gen_agg.nll - gen_agg.ref_nll,
                    gen_agg.kl, gen_agg.top1 * 100.0, gen_agg.top3_contained * 100.0,
                    engine.retained_graph_rows(1),
                );
            }
            if ctx_agg.positions > 0 {
                println!(
                    "hy4-golden context-only ({} rows): mean NLL={:.6} (ref {:.6}, delta {:+.6}) mean KL={:.6} top-1={:.2}% top-3={:.2}%",
                    ctx_agg.positions, ctx_agg.nll, ctx_agg.ref_nll, ctx_agg.nll - ctx_agg.ref_nll,
                    ctx_agg.kl, ctx_agg.top1 * 100.0, ctx_agg.top3_contained * 100.0,
                );
            }
        }
        if args.decode_graphs {
            for snap in crate::shared::decode_graph::device::snapshots() {
                let hits: u64 = snap.cycles.values().map(|c| c.hits).sum();
                let captures: u64 = snap.cycles.values().map(|c| c.captures).sum();
                let eager: u64 = snap.cycles.values().map(|c| c.eager_runs).sum();
                println!(
                    "hy4-golden decode-graphs device={}: reserve={} charged={} captures={} hits={} eager_runs={}",
                    snap.device, snap.reserve, snap.charged, captures, hits, eager,
                );
            }
        }
        return Ok(());
    }

    // Raw window directory (`tokens.bin` + optional `logits.bin`).
    let logits = engine.forward_golden(&raw_tokens, 0, args.last_layer, unit, args.decode)?;
    if let Some(save) = &args.save_logits {
        std::fs::create_dir_all(save)?;
        let bytes: Vec<u8> = logits.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(save.join("logits.bin"), bytes)?;
    }
    if args.nll && args.last_layer == cfg.layers - 1 {
        let vocab = cfg.vocab_size;
        let reference = ReferenceRows::open(&raw_window_dir.join("logits.bin"), raw_tokens.len(), vocab, true)?;
        let mean_nll = if raw_tokens.len() > 1 {
            scored_nll(&logits[..(raw_tokens.len() - 1) * vocab], &raw_tokens[1..], vocab)?
        } else {
            0.0
        };
        let mut top1_matches = 0usize;
        let mut ref_rows = 0usize;
        if let Some(ref_reader) = &reference {
            for row in 0..raw_tokens.len() {
                let slice = &logits[row * vocab..(row + 1) * vocab];
                let ref_row = ref_reader.row(row)?;
                let got_top = slice.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0);
                let ref_top = ref_row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0);
                if got_top == ref_top {
                    top1_matches += 1;
                }
                ref_rows += 1;
            }
        }
        println!(
            "hy4-golden mean NLL={mean_nll:.6} ({} scored rows); top-1 agreement={top1_matches}/{ref_rows} digest={}",
            raw_tokens.len().saturating_sub(1),
            digest_f32(&logits),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    #[test]
    fn hy4_golden_cli_parses_peers_and_resume() -> anyhow::Result<()> {
        let argv = [
            "cuteafd",
            "hy4-golden",
            "--snapshot",
            "/model",
            "--device",
            "0",
            "--split-device",
            "1",
            "--peers",
            "10.55.0.1:19441,10.55.0.2:19441,10.55.0.3:19441,10.55.0.4:19441",
            "--core-view",
            "k8-compact",
            "--kv-cache",
            "fp8",
            "--decode",
            "--decode-graphs",
            "--resume-at",
            "64,128",
        ];
        let crate::cli::Commands::Hy4Golden(args) = crate::cli::Cli::try_parse_from(argv)?.command
        else {
            anyhow::bail!("Hy4 golden command not parsed");
        };
        assert_eq!(args.last_layer, 77);
        assert_eq!(args.peers.len(), 4);
        assert_eq!(args.resume_at, vec![64, 128]);
        assert!(args.decode && args.decode_graphs);
        Ok(())
    }
}
