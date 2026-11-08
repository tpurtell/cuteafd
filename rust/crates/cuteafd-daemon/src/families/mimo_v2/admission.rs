//! Pre-allocation descriptions shared with MiMo's allocation path. CUDA
//! manifest queries read shape/scratch metadata before loading any weights.
use anyhow::{ensure, Context, Result};
use cuteafd_loader::families::mimo_v2::admission::{tensor_bytes, draft_prefix_reservations, prefill_shape, workspace_scratch, draft_reservations_with, draft_rank_reservations_with, DraftReservations, DraftScratch};

use cuteafd_core::serving_capacity::MemoryReservation;
use cuteafd_ffi::programs::Programs;
use cuteafd_loader::families::mimo_v2::{MimoKvCache, MimoKvFormats, MimoPrefillOutput, MimoV2Config};

/// Exact tensor admission against the measured post-module baseline.
pub(super) struct Preflight {
    pub expert_usage: Option<cuteafd_core::expert_usage::Geometry>,
    pub capacity: cuteafd_core::serving_capacity::ResolvedCapacity,
    pub graph_plan: cuteafd_loader::families::mimo_v2::decode_graph::MimoDecodeGraphPlan,
    pub graph_bound_bytes: Vec<u64>,
    pub host_config: Option<cuteafd_hostcache::config::Config>,
    pub local_expert_budget: usize,
    pub placement: cuteafd_loader::placement::Placement,
    pub expert_directory: Option<std::path::PathBuf>,
    pub working_set: cuteafd_loader::placement::SparkWorkingSet,
    pub prefill_rows: usize,
}

fn routed_costs(opened: &super::Opened, args: &super::EngineArgs, layers: usize, ranks: usize)
    -> Result<(Vec<Option<cuteafd_loader::placement::ExpertCost>>, u64, [u64; 2], Option<std::path::PathBuf>)> {
    use crate::shared::experts::fp8::bf16_sibling;
    use crate::shared::experts::rtx::fp8moe::Fp8MoeTp2;
    use cuteafd_loader::placement::{Bytes2, ExpertCost, families::mimo_v2::{expert_layer_bytes, package_directory, package_manifest}};
    use cuteafd_loader::formats::fp8_experts::Slicing;
    if args.skip_experts || args.expert_window.is_some() || opened.cfg.dense.iter().take(layers).all(|&dense| dense) {
        ensure!(ranks == 1 || args.expert_window.is_none(), "MiMo streamed TP1 experts require one RTX");
        return Ok((vec![None; layers], 0, [0; 2], None));
    }
    let tensors = opened.catalog.as_ref().and_then(|catalog| catalog.fp8())
        .context("MiMo routed expert headers are required for shared placement")?;
    let rows = args.prefill_rows.max(super::engine::DECODE_ROWS);
    let mut directory = match &args.fp8_package {
        Some(path) => path.clone(),
        None => package_directory(args.native_lib.parent().context("MiMo native library parent")?,
            &opened.cfg, tensors.format(), ranks)?,
    };
    if bf16_sibling(&directory).is_some_and(|path| path.is_dir()) {
        directory = bf16_sibling(&directory).context("MiMo BF16 package sibling")?;
    }
    let routed = opened.cfg.dense.iter().take(layers).filter(|&&dense| !dense).count();
    let onboard = args.onboard.resolve(routed);
    let zero_local = onboard.layers(routed) == Some(0);
    let metadata = if zero_local { None } else { package_manifest(&directory, None)? };
    if zero_local || (metadata.is_none()
        && onboard == cuteafd_loader::placement::Onboard::Auto && args.fp8_package.is_none()) {
        ensure!(args.peers.is_some(), "MiMo zero local experts requires Spark peers");
        let costs = opened.cfg.dense.iter().take(layers).enumerate().map(|(layer, &dense)| -> Result<_> {
            if dense { return Ok(None); }
            let view = tensors.for_coordinator().for_layer_metadata(layer)?;
            let supported = view.format() == tensors.format();
            Ok(Some(ExpertCost { whole: Bytes2 { resident: expert_layer_bytes(&view, 1, 0, Slicing::Padded)?, staging: 0 },
                half: [Bytes2::default(); 2], tp2: ranks == 2 && supported, whole_supported: supported, spark_ok: true }))
        }).collect::<Result<Vec<_>>>()?;
        return Ok((costs, 0, [0; 2], None));
    }
    let metadata = metadata.context("MiMo local experts need a generated BF16-input package manifest")?;
    let scratch = |path: &std::path::Path| -> Result<u64> {
        let layout = path.file_name().and_then(|name| name.to_str()).context("MiMo package layout")?;
        cuteafd_loader::placement::inventory::fp8moe_scratch_bytes(&metadata, layout, rows as u64)
            .context("MiMo expert manifest lacks selected capacity")
    };
    let (workspace, tp2_workspace) = if ranks == 2 {
        let plans = [Fp8MoeTp2::plan(tensors, &directory, &[tensors.shape().first_layer], 0, rows)?, Fp8MoeTp2::plan(tensors, &directory, &[tensors.shape().first_layer], 1, rows)?];
        for plan in &plans {
            ensure!(scratch(&plan.directory)? == plan.scratch_bytes as u64,
                "MiMo expert manifest scratch differs from native package ABI");
        }
        (0, plans.each_ref().map(|plan| plan.workspace_bytes as u64))
    } else {
        // SAFETY: this reads only the trusted generated package's static ABI.
        let metadata = unsafe { cuteafd_ffi::fp8_moe::Fp8MoeMetadata::read(&directory) }?;
        ensure!(scratch(&directory)? == metadata.scratch_for(rows)?.max(256) as u64,
            "MiMo expert manifest scratch differs from native package ABI");
        ensure!(!metadata.info.wire_input && metadata.info.tp == 1 && metadata.info.hidden == opened.cfg.hidden
            && metadata.info.experts == opened.cfg.experts && metadata.info.topk == opened.cfg.topk
            && metadata.info.intermediate == opened.cfg.moe_intermediate,
            "MiMo local package does not match BF16-input checkpoint geometry: {:?}", metadata.info);
        (metadata.scratch_for(rows)?.max(256) as u64, [0; 2])
    };
    let costs = opened.cfg.dense.iter().take(layers).enumerate().map(|(layer, &dense)| -> Result<_> {
        if dense { return Ok(None); }
        let view = tensors.for_coordinator().for_layer_metadata(layer)?;
        let supported = view.format() == tensors.format();
        let whole = Bytes2 { resident: expert_layer_bytes(&view, 1, 0, Slicing::Padded)?, staging: 0 };
        let half = if ranks == 2 {
            let bytes = |rank| -> Result<_> {
                let (_, slicing) = cuteafd_loader::placement::families::mimo_v2::exact_package(&directory, &view, 2, rank);
                Ok(Bytes2 { resident: expert_layer_bytes(&view, 2, rank, slicing)?, staging: 0 })
            };
            [bytes(0)?, bytes(1)?]
        } else { [Bytes2::default(); 2] };
        Ok(Some(ExpertCost { whole, half, tp2: ranks == 2 && supported,
            whole_supported: supported, spark_ok: args.peers.is_some() }))
    }).collect::<Result<Vec<_>>>()?;
    Ok((costs, workspace, tp2_workspace, Some(directory)))
}

fn resolve_pool_context(capacity: &mut cuteafd_core::serving_capacity::ResolvedCapacity, context: cuteafd_loader::placement::planner::Context, unit_rows: u64) -> Result<()> {
    capacity.effective_max_context_tokens = crate::shared::context::pool_context("mimo_v2",
        context, usize::try_from(capacity.allocated_gpu_kv_tokens)?)? as u64;
    let sequence_units = capacity.effective_max_context_tokens.div_ceil(unit_rows);
    capacity.active_max_context_sequences = (capacity.allocated_gpu_kv_tokens / unit_rows / sequence_units)
        .min(u64::from(capacity.concurrency)) as u32;
    Ok(())
}

fn aggregate_mark_bytes(target: u64, draft: u64, enabled: bool) -> Result<u64> {
    target.checked_add(if enabled { draft } else { 0 }).context("MiMo mark size overflow")
}


#[cfg(test)]
mod draft_prefix_tests {
    use super::*;

    #[test]
    fn planner_marks_equal_admission_at_c4_c16_warm_on_and_off() {
        use clap::Parser;
        use cuteafd_loader::plan::{plan, layout::LayoutOptions, testing::*, PlanOptions};
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            prefix: crate::shared::prefix::PrefixArgs,
        }
        let dir = tempfile::tempdir().unwrap();
        let mut config = mimo_pro_config();
        config["max_position_embeddings"] = serde_json::json!(1 << 20);
        write_snapshot(dir.path(), &config, &mimo_pro_tensors(), Some(8));
        std::fs::create_dir(dir.path().join("dflash")).unwrap();
        std::fs::write(dir.path().join("dflash/config.json"),
            r#"{"num_hidden_layers":5,"num_key_value_heads":8,"head_dim":128}"#).unwrap();
        let cfg = MimoV2Config::from_hf(&config).unwrap();
        assert_eq!(LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), ..Default::default() }.mimo_swa_kv, MimoKvFormats::default().sliding);
        for sliding in [MimoKvCache::Bf16, MimoKvCache::Int8, MimoKvCache::Fp8] {
            let cache = cuteafd_loader::serving_capacity::mimo_cache_geometry(
                &cfg, cfg.layers, 1, MimoKvFormats { full: MimoKvCache::Int8, sliding }, 0).unwrap();
            let target: u64 = cache.ranks.iter().map(|r| r.retained_mark_bytes).sum();
            for draft_kv_cache in [cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16,
                cuteafd_loader::families::qwen4::Qwen4KvCache::Fp8] {
                let draft = cuteafd_loader::families::mimo_v2::draft_representation::mimo_draft_mark_bytes(5, 8, 128, draft_kv_cache).unwrap();
                for concurrency in [4, 16] {
                    for enabled in [false, true] {
                        // Include a budget-tight case to catch the concurrency floor and
                        // a warm mark changing the number of affordable retained slots.
                        for mib in [0, 1024, 2048] {
                            let cli = Cli::parse_from(["serve", "--prefix-cache-mark-mib", &mib.to_string()]);
                            let slots = mark_slots(&cli.prefix, concurrency,
                                aggregate_mark_bytes(target, draft, enabled).unwrap() as usize).unwrap() as u64;
                            let layout = plan(dir.path(), &PlanOptions {
                                layout: Some(LayoutOptions { host_memory: Some((128 << 30, 128 << 30)),
                                    concurrency: concurrency as u64, mimo_prefix_draft: enabled, draft_kv_cache,
                                    mimo_swa_kv: sliding,
                                    mimo_prefix_mark_bytes: mib << 20, host_cache_bytes: Some(0), ..Default::default()
                                }), ..Default::default()
                            }).unwrap().memory_layout.unwrap();
                            let bytes = |name| layout.devices[0].items.iter().filter(|i| i.group == name).map(|i| i.bytes).sum::<u64>();
                            assert_eq!(bytes("marks"), target * slots, "C{concurrency}, warm={enabled}, MiB={mib}: {:?}", layout.notes);
                            let reservations = draft_prefix_reservations(enabled, draft, slots, concurrency as u64).unwrap();
                            assert_eq!(bytes("DFlash context marks"), reservations.first().map_or(0, |r| r.bytes));
                            assert_eq!(bytes("DFlash valid-floor transfer"), reservations.get(1).map_or(0, |r| r.bytes));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn shared_sampler_contract_matches_allocator() {
        for vocab in [64, 152576, 153600] {
            assert_eq!(cuteafd_loader::families::mimo_v2::admission::sampling_wave_bytes(64, vocab),
                crate::shared::sampler::TargetSamplingWave::device_bytes(64, vocab));
        }
    }

    #[test]
    fn disabled_warm_marks_preserve_target_slot_and_host_mark_geometry() {
        let target = 256u64 << 20;
        let draft = 20u64 << 20;
        let bytes = |enabled| aggregate_mark_bytes(target, draft + 8, enabled).unwrap();
        let slots = |enabled| cuteafd_engine::prefix::MarkArena::slots_for(16, 24, bytes(enabled) as usize, 12usize << 30, false);
        assert_eq!(slots(false), cuteafd_engine::prefix::MarkArena::slots_for(16, 24, target as usize, 12usize << 30, false));
        assert!(slots(true) <= slots(false));
        assert_eq!(bytes(false), target);
        assert!(draft_prefix_reservations(false, draft + 8, 42, 18).unwrap().is_empty());
        let enabled = draft_prefix_reservations(true, draft + 8, 42, 18).unwrap();
        assert_eq!(enabled[0].bytes, (draft + 8) * 42);
        assert_eq!(enabled[1].bytes, 18 * 8);
        assert_eq!(bytes(true), target + draft + 8);
        assert!(draft_prefix_reservations(true, draft + 8, u64::MAX, 18).is_err());
    }
}

pub(super) fn resolved_prefill_shape(spark_ranks: usize, rows: usize, waves: usize)
    -> Result<cuteafd_loader::serving_capacity::PrefillShape> {
    use cuteafd_loader::serving_capacity::prefill_waves::{prefill_shape, PrefillLimits, PrefillOverrides};
    Ok(prefill_shape(spark_ranks, PrefillLimits::default(), PrefillOverrides {
        unit_rows: (rows > 0).then_some(rows as u64),
        waves: (waves > 0).then_some(waves as u64), ..Default::default()
    })?.units())
}

/// Resolve physical reservations after module preload, before model allocations.
/// The same profile can be serialized by a planner; explicit benchmark pool
/// and context arguments are always preserved.
pub(super) fn preflight(
    opened: &super::Opened,
    args: &super::EngineArgs,
    programs: &Programs<'_>,
    split_device: Option<i32>,
    serving: Option<(&crate::shared::prefix::PrefixArgs, usize)>,
    prefill_output: MimoPrefillOutput,
    prefix_draft: bool,
    serving_context: cuteafd_loader::placement::planner::Context,
) -> Result<Preflight> {
    preflight_shape(opened, args, programs, split_device, serving,
        prefill_output, prefix_draft, serving_context)
}

fn preflight_shape(
    opened: &super::Opened,
    args: &super::EngineArgs,
    programs: &Programs<'_>,
    split_device: Option<i32>,
    serving: Option<(&crate::shared::prefix::PrefixArgs, usize)>,
    prefill_output: MimoPrefillOutput,
    prefix_draft: bool,
    serving_context: cuteafd_loader::placement::planner::Context,
) -> Result<Preflight> {
    use cuteafd_core::serving_capacity::{CapacityPolicy, DeviceMemory};
    use cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE;
    use cuteafd_loader::families::mimo_v2::capacity::{
        mimo_capacity_profiles, MimoCapacityOptions, MimoRankRuntime,
    };
    use cuteafd_loader::families::mimo_v2::resident::MimoResidentOptions;
    use cuteafd_loader::serving_capacity::{checkpoint_context_limit, mimo_cache_geometry};

    let cfg = &opened.cfg;
    let library = &opened.library;
    let layers = args.layers.unwrap_or(cfg.layers).min(cfg.layers);
    ensure!(
        layers > 0 && args.rings > 0 && args.prefill_rows > 0 && args.max_context > 0,
        "MiMo needs positive layers, rings, prefill rows and context"
    );
    ensure!(
        args.prefill_rows <= programs.capacities().prefill_rows.unwrap_or(4096),
        "MiMo prefill rows exceed the compiled program capacity"
    );
    let devices: Vec<i32> = std::iter::once(args.device).chain(split_device).collect();
    let ranks = devices.len();
    ensure!(!args.tp2_dflash || (ranks == 2 && args.draft.is_some()),
        "TP2 DFlash requires a drafter and two target GPUs");
    let concurrency = serving.map_or(1, |(_, c)| c);
    ensure!(
        (1..=super::engine::DECODE_ROWS).contains(&concurrency) && concurrency <= args.rings,
        "MiMo concurrency {concurrency} needs that many rings and at most 64 decode rows"
    );
    let context = checkpoint_context_limit(&opened.checkpoint.config)?
        .context("MiMo checkpoint has no max_position_embeddings; specify its context capability in config.json")?;
    let kv = args.kv_formats();
    let cache = mimo_cache_geometry(cfg, layers, ranks, kv, args.mtp)?;
    let draft_mark_bytes = args.draft.as_deref().filter(|_| prefix_draft).map(super::dflash::drafter_dir)
        .map(|dir| -> Result<u64> {
            let draft = super::dflash::DflashConfig::read(&dir)?;
            draft.mark_bytes(if args.tp2_dflash { 2 } else { 1 }, args.draft_kv_cache)
        }).transpose()?.unwrap_or(0);
    let target_mark_bytes = cache
        .ranks
        .iter()
        .try_fold(0u64, |sum, r| sum.checked_add(r.retained_mark_bytes))
        .context("MiMo mark size overflow")?;
    let mark_bytes = aggregate_mark_bytes(target_mark_bytes, draft_mark_bytes, prefix_draft)?;
    let mark_slots = match serving {
        Some((prefix, _)) => mark_slots(prefix, concurrency, usize::try_from(mark_bytes)?)? as u64,
        None => 0,
    };
    // Registered snapshot owners restore every physical rank. The host tier
    // receives one aggregate logical page/mark and adds no active GPU KV.
    let host_config = match serving {
        Some((prefix, _)) => {
            let page_bytes = cache
                .ranks
                .iter()
                .try_fold(0u64, |sum, r| sum.checked_add(r.persistent_unit_bytes))
                .context("MiMo aggregate page bytes")?;
            let layout = cuteafd_engine::prefix::FamilyLayout {
                page_rows: super::engine::PAGE_ROWS,
                // An automatic pool is at most the common target; the host tier sizes for that.
                pages: if args.pool_tokens == 0 {
                    usize::try_from(cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS)?
                } else { args.pool_tokens }.div_ceil(super::engine::PAGE_ROWS),
                page_bytes: usize::try_from(page_bytes)?,
                mark_bytes: usize::try_from(mark_bytes)?,
                draft_bytes: 0,
                rule: cuteafd_engine::prefix::ReuseRule::EXACT,
                mark_store: cuteafd_engine::prefix::MarkStore::Arena,
                page_owners: Default::default(),
            };
            let scoring_rows = if args.full_prefill_logits {
                super::engine::prefill_span_rows(args.prefill_span_rows, args.prefill_rows)?
            } else { 64 };
            prefix.with_scoring(scoring_rows, scoring_rows.max(64), cfg.vocab_size)?
                .resolve_host_config(layout, args.max_context, concurrency)?
        }
        None => None,
    };
    let host_bytes = host_config.as_ref().map_or(0, |config| config.bytes);
    let routed_layers = cfg
        .dense
        .iter()
        .take(layers)
        .filter(|&&dense| !dense)
        .count();
    let backend = expert_backend(
        routed_layers,
        args.skip_experts,
        args.local_experts,
        args.peers.is_some(),
    );
    let graph_plan = cuteafd_loader::families::mimo_v2::decode_graph::MimoDecodeGraphPlan::new(
        cfg.layers,
        ranks,
        super::engine::DECODE_ROWS,
        args.decode_graphs
            && layers == cfg.layers
            && (routed_layers == 0 || backend != ExpertBackend::None),
    )?;
    // Retained startup graph owners: measured whole-capture bytes per rank,
    // charged before KV sizing. No driver margin or per-executable envelope.
    let graph_reservations = (0..ranks)
        .map(|rank| graph_plan.reservations(rank))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let graph_bound_bytes = graph_reservations
        .iter()
        .map(|costs| costs.iter().map(|r| r.bytes).sum::<u64>())
        .collect::<Vec<_>>();
    let transport_lanes = resolved_prefill_shape(usize::from(args.peers.is_some()),
        args.prefill_rows, args.prefill_waves)?.waves as usize;
    let spark_ranks = match args
        .peers
        .as_deref()
        .filter(|_| backend == ExpertBackend::Spark)
    {
        Some(peers) => {
            let peers = peers
                .split(',')
                .map(str::parse)
                .collect::<std::result::Result<Vec<std::net::SocketAddr>, _>>()?;
            ensure!(
                (1..=crate::shared::spark_intake::MAX_INTAKE_RANKS).contains(&peers.len()),
                "MiMo needs 1–6 Spark ranks"
            );
            peers.len()
        }
        None => 0,
    };
    let (expert_costs, expert_workspace, tp2_workspace, expert_directory) = routed_costs(opened, args, layers, ranks)?;
    let mut runtime = Vec::with_capacity(ranks);
    let mut memory = Vec::with_capacity(ranks);
    let mut post_kv_loading_additional = Vec::with_capacity(ranks);
    let mut draft_packing = vec![vec![]; ranks];
    for (rank, &device) in devices.iter().enumerate() {
        let device_id = u32::try_from(device).context("negative CUDA device id")?;
        let (free, total) =
            crate::shared::peer_split::on_device(library, device, || {
                library.cuda_memory_info()
            })?;
        memory.push(DeviceMemory {
            device: device_id,
            total_bytes: total as u64,
            baseline_free_bytes: free as u64,
        });
        // Selected program code only; BLAS pools have separate named owners.
        let pending = crate::shared::inventory::pending_code(library, device, rank, ranks == 2,
            cfg.program_family()?, "*", &cuteafd_loader::placement::ProgramSet::from_names(
                &[cfg.program_family()?, &format!("{}2", cfg.program_family()?)], programs.names()))?;
        let code = MemoryReservation { name: "runtime.loaded_code".into(), bytes: pending };
        let mut additional = vec![code.clone()];
        let mut resident_weights = cuteafd_loader::placement::ResidentWeights::default();
        let mut weight_arenas = Vec::new();
        additional.extend(cuteafd_loader::families::mimo_v2::admission::transport_reservations(
            cfg, ranks, rank, args.prefill_rows, transport_lanes, 0, cuteafd_core::spark_reply::requested().prefill(false))?);
        // Only modules and the already-attached peer are live before drafter
        // packing. Spark transport, samplers, marks, graphs and reusable target
        // workspaces are constructed later and do not consume this phase.
        post_kv_loading_additional.push(additional.clone());
        if rank == 0 {
            additional.extend(cuteafd_loader::families::mimo_v2::admission::transport_reservations(
                cfg, 1, 0, args.prefill_rows, transport_lanes, spark_ranks, cuteafd_core::spark_reply::requested().prefill(false))?);
            if serving.is_some() {
                additional.extend(cuteafd_loader::families::mimo_v2::admission::sampling_reservations(cfg.vocab_size)?);
            }
            if backend == ExpertBackend::Local && args.expert_window.is_some() {
                let (reservations, weights) = local_expert_reservations(opened, args, layers)?;
                additional.extend(reservations);
                resident_weights = resident_weights.checked_add(weights)?;
            }
        }
        if rank == 0 || args.tp2_dflash {
            if let Some(dir) = args.draft.as_deref().map(super::dflash::drafter_dir) {
                let draft = super::dflash::DflashConfig::read(&dir)?;
                ensure!(
                    draft.hidden == cfg.hidden
                        && draft.vocab == cfg.vocab_size
                        && draft.taps.iter().all(|&l| l < layers),
                    "DFlash geometry or taps do not fit the selected target layers"
                );
                let reservations = draft_rank_reservations(
                    library,
                    &dir,
                    &draft,
                    opened.weight_formats.draft,
                    args.draft_capacity(draft.block)?,
                    rank,
                    if args.tp2_dflash { 2 } else { 1 },
                )?;
                resident_weights = resident_weights.checked_add(reservations.resident_weights)?;
                additional.extend(reservations.steady);
                weight_arenas.push(reservations.arena);
                let local = draft.for_tp(if args.tp2_dflash { 2 } else { 1 })?;
                let mark = cuteafd_loader::families::mimo_v2::draft_representation::mimo_draft_mark_bytes(
                    local.layers as u64, local.kv_heads as u64, local.head_dim as u64, args.draft_kv_cache)?;
                additional.extend(draft_prefix_reservations(prefix_draft,
                    if rank == 0 { mark } else { mark - 8 }, mark_slots,
                    if rank == 0 { args.rings as u64 } else { 0 })?);
                draft_packing[rank] = reservations.packing;
            }
        }
        additional.extend(graph_reservations[rank].iter().cloned());
        let workspaces = workspace_options(
            cfg,
            args,
            programs,
            ranks,
            rank,
            transport_lanes,
            spark_ranks > 0,
            VOCABULARY_HEAD_WORKSPACE as u64,
            prefill_output,
            opened.weight_formats.any_fp8_output(),
        )?;
        runtime.push(MimoRankRuntime {
            weight_arenas,
            resident_weights,
            device: device_id,
            workspaces,
            additional,
            loading_additional: vec![code],
        });
    }
    let profiles_for = |selected_spark_ranks: usize| -> Result<_> {
        let lanes = transport_lanes;
        let mut selected_runtime = runtime.clone();
        for (rank, costs) in selected_runtime.iter_mut().enumerate() {
            costs.additional.retain(|cost| !cost.name.starts_with("transport."));
            costs.additional.extend(cuteafd_loader::families::mimo_v2::admission::transport_reservations(
                cfg, ranks, rank, args.prefill_rows, lanes, selected_spark_ranks, cuteafd_core::spark_reply::requested().prefill(false))?);
            costs.workspaces = workspace_options(cfg, args, programs, ranks, rank, lanes,
                selected_spark_ranks > 0, VOCABULARY_HEAD_WORKSPACE as u64, prefill_output,
                opened.weight_formats.any_fp8_output())?;
        }
        Ok(mimo_capacity_profiles(
        &opened.checkpoint,
        cfg,
        MimoResidentOptions {
            layers,
            coordinator_ranks: ranks,
            checkpoint_tp: cuteafd_loader::families::mimo_v2::checkpoint_tp(&args.snapshot)?,
            native_mtp_layers: args.mtp,
            gpu_embedding: args.token_io.embed_placement
                .context("MiMo embedding placement must resolve before admission")?
                == crate::shared::token_io::EmbedPlacement::Gpu,
            head_format: opened.weight_formats.head,
            output_formats: opened.weight_formats.output.clone(),
        },
        &MimoCapacityOptions {
            checkpoint_max_context_tokens: context,
            max_context_tokens: args.max_context as u64,
            rings: args.rings as u64,
            mark_slots,
            kv_cache: kv,
            ranks: selected_runtime,
            host_prefix_bytes: host_bytes,
        },
    )?)
    };
    let spark_profiles = profiles_for(spark_ranks)?;
    let no_spark_profiles = if spark_ranks == 0 { spark_profiles.clone() } else { profiles_for(0)? };
    let mut resident_weights = spark_profiles.resident_weights.clone();
    cuteafd_loader::placement::families::mimo_v2::add_local_media_weights(&mut resident_weights,
        &devices, args.planner_media_weights, crate::shared::memory::budget::resident_weight_limit())?;
    use cuteafd_loader::placement::families::mimo_v2::admission::{self as contract, ProductionInput};
    use cuteafd_loader::placement::planner::{Baseline, EmbeddingHome, Operation, Phase, PrefillShape};
    let maximum_pool_tokens = contract::maximum_pool_tokens(
        &memory.iter().map(|sample| sample.total_bytes).collect::<Vec<_>>(),
        args.max_context as u64, (args.pool_tokens > 0).then_some(args.pool_tokens as u64),
        spark_profiles.steady.pool_unit_rows);
    let policy = CapacityPolicy {
        concurrency: u32::try_from(concurrency)?,
        target_pool_tokens: maximum_pool_tokens,
        max_context_tokens: Some(args.max_context as u64),
        pool_tokens: Some(maximum_pool_tokens),
        ..CapacityPolicy::default()
    };
    let report = reservation_report(&spark_profiles.steady, &memory, policy)?;
    let intake_probe_bytes = cuteafd_loader::families::mimo_v2::admission::spark_intake_probe_bytes(
        spark_ranks > 0, std::env::var("CUTEAFD_SPARK_INTAKE").ok().as_deref())?;
    let prefill = PrefillShape {
        rows: args.prefill_rows.max(64) as u64,
        waves: transport_lanes as u64,
    };
    let embedding = if args.token_io.embed_placement
        .context("MiMo embedding placement must resolve before admission")?
        == crate::shared::token_io::EmbedPlacement::Host
    {
        EmbeddingHome::HostPinned
    } else {
        EmbeddingHome::Device(0)
    };
    let operation = if args.full_prefill_logits { Operation::Scoring } else { Operation::Serving };
    let budgets = contract::production_budgets(
        &memory.iter().map(|sample| (sample.total_bytes, sample.total_bytes)).collect::<Vec<_>>(),
        &(0..memory.len()).map(|rank| contract::device_headroom(rank, intake_probe_bytes)).collect::<Vec<_>>(),
        memory.iter().map(|sample| Baseline::Sampled {
            free: sample.baseline_free_bytes, phase: Phase::Load,
            source: "measured post-module free memory".into() }).collect(),
        cuteafd_loader::formats::mapped_preload::host_memory_before_process()?,
        cuteafd_loader::placement::reserve::spark_cap(), spark_ranks)?;
    let request = contract::production_request(
        serving_context, (args.pool_tokens > 0).then_some(args.pool_tokens as u64),
        args.onboard.resolve(routed_layers), prefill, operation,
        crate::shared::memory::budget::resident_weight_limit(), cuteafd_core::spark_reply::requested());
    let prepared = ProductionInput {
        expert_usage: args.expert_usage_enabled()?,
        blas_head: crate::shared::target_head::TargetHead::configured().blas_head(
            opened.weight_formats.head == cuteafd_loader::families::mimo_v2::projection::MimoProjectionRepresentation::Bf16),
        draft_head: args.draft.is_some(),
        cfg, layers, kv, mtp_stages: args.mtp, context: args.max_context as u64,
        maximum_pool_tokens, prefill, lanes: transport_lanes as u64, embedding, gpus: ranks,
        spark_ranks, experts: expert_costs, expert_workspace, tp2_workspace,
        spark: spark_profiles.clone(), no_spark: no_spark_profiles.clone(),
        post_kv_additional: post_kv_loading_additional, draft_packing,
        host: contract::HostOwners { expert_input: args.expert_input.into(), output: prefill_output,
            serving: serving.is_some(), scoring_rows: if args.full_prefill_logits {
                super::engine::prefill_span_rows(args.prefill_span_rows, args.prefill_rows)? } else { 64 },
            diagnostic_rows: if std::env::var("CUTEAFD_ROUTE_CHECK").ok().and_then(|s| s.parse::<usize>().ok()).unwrap_or(0) > 0 {
                super::engine::prefill_span_rows(args.prefill_span_rows, args.prefill_rows)?.max(64) } else { args.prefill_rows.min(256) },
            prefix_bytes: host_bytes, media_bytes: 0,
            headroom: serving.map_or(0, |(prefix, _)| prefix.host_cache_headroom_bytes) },
        spark_reply_formats: cuteafd_core::spark_reply::requested(),
        resident_weights, operation, budgets,
    }.prepare()?;
    let admitted = prepared.admit(request).with_context(|| format!(
        "MiMo shared placement; configured-Spark reservation contract {report}"))?;
    let mut capacity = contract::resolved_capacity(&admitted,
        u32::try_from(concurrency)?, u32::try_from(args.rings)?)?;
    let working = admitted.spark;
    let placement = admitted.placement;
    let profiles = if working.spark_ranks == 0 { &no_spark_profiles } else { &spark_profiles };
    let intake_probe_bytes = if working.spark_ranks > 0 { intake_probe_bytes } else { 0 };
    tracing::info!(phases = %serde_json::to_string(&admitted.solved.phases)?,
        "MiMo admitted phase ledger before allocations");
    cuteafd_loader::placement::reserve::log_plan(&admitted.solved, "mimo_v2");
    tracing::info!(placement = %placement.summary(), "MiMo shared admission solve");
    let pool_tokens = capacity.allocated_gpu_kv_tokens;
    let shortfall_tokens = (args.max_context as u64).saturating_sub(pool_tokens);
    tracing::info!(pool_tokens, context_tokens=args.max_context, full_context_sequences=pool_tokens / args.max_context as u64,
        slots=concurrency, shortfall_tokens, "MiMo full-context pool admission before allocation");
    resolve_pool_context(&mut capacity, serving_context, profiles.steady.pool_unit_rows)?;
    let local_expert_budget = capacity.devices[0]
        .reservations
        .iter()
        .filter(|reservation| reservation.name.starts_with("experts."))
        .try_fold(0u64, |bytes, reservation| {
            bytes.checked_add(reservation.bytes)
        })
        .context("MiMo admitted local expert size overflow")?;
    tracing::info!(checkpoint_max_context = capacity.checkpoint_max_context_tokens,
        max_context = capacity.effective_max_context_tokens, requested_pool = args.pool_tokens,
        allocated_pool = capacity.allocated_gpu_kv_tokens, requested_default = capacity.requested_kv_floor_tokens,
        requested_state_slots = capacity.state_slots, allocated_rings = args.rings,
        host_prefix_bytes = capacity.host_prefix_bytes, mark_slots, intake_probe_bytes,
        ?graph_plan, ?graph_bound_bytes,
        reservations = %serde_json::to_string(&capacity.devices)?,
        "MiMo pre-allocation capacity against measured post-module baseline");
    Ok(Preflight {
        expert_usage: prepared.recipe.expert_usage.clone(),
        capacity,
        graph_plan,
        graph_bound_bytes,
        host_config,
        local_expert_budget: usize::try_from(local_expert_budget)?,
        placement, expert_directory, working_set: working,
        prefill_rows: args.prefill_rows,
    })
}

/// Summarize all fixed categories plus the selected aligned pool even when
/// admission fails. A generic "pool too large" must not conceal the real
/// workspace/drafter/mark cost or imply that host snapshots add active KV.
fn reservation_report(
    profile: &cuteafd_core::serving_capacity::CapacityProfile,
    memory: &[cuteafd_core::serving_capacity::DeviceMemory],
    policy: cuteafd_core::serving_capacity::CapacityPolicy,
) -> Result<String> {
    let tokens = policy.pool_tokens.unwrap_or(policy.target_pool_tokens);
    let units = tokens.div_ceil(profile.pool_unit_rows);
    let mut devices = Vec::with_capacity(profile.devices.len());
    for costs in &profile.devices {
        let sample = memory
            .iter()
            .find(|m| m.device == costs.device)
            .context("missing physical GPU sample")?;
        let non_engine = sample
            .total_bytes
            .checked_sub(sample.baseline_free_bytes)
            .context("invalid physical GPU memory sample")?;
        let minimum_free = cuteafd_loader::placement::families::mimo_v2::admission::device_headroom(
            usize::try_from(costs.device).unwrap_or(usize::MAX), 0);
        let budget = sample.total_bytes.saturating_sub(minimum_free).saturating_sub(non_engine);
        let mut fixed = std::collections::BTreeMap::<&str, u64>::new();
        for cost in &costs.reservations {
            let category = cost
                .name
                .split('.')
                .next()
                .context("empty MiMo reservation category")?;
            let bytes = fixed.entry(category).or_default();
            *bytes = bytes
                .checked_add(cost.bytes)
                .context("MiMo reservation category overflow")?;
        }
        let fixed_bytes = fixed
            .values()
            .try_fold(0u64, |sum, &bytes| sum.checked_add(bytes))
            .context("MiMo fixed reservation overflow")?;
        let pool_bytes = units
            .checked_mul(costs.pool_unit_bytes)
            .context("MiMo requested pool bytes overflow")?;
        devices.push(serde_json::json!({
            "device": costs.device, "total_bytes": sample.total_bytes,
            "non_engine_bytes": non_engine, "engine_budget_bytes": budget,
            "minimum_free_bytes": minimum_free,
            "fixed_categories": fixed, "fixed_bytes": fixed_bytes,
            "requested_pool_bytes": pool_bytes,
            "complete_requested_bytes": fixed_bytes.checked_add(pool_bytes).context("MiMo complete requested bytes overflow")?,
        }));
    }
    Ok(serde_json::to_string(&serde_json::json!({
        "requested_pool_tokens": units.checked_mul(profile.pool_unit_rows).context("MiMo aligned pool overflow")?,
        "checkpoint_max_context_tokens": profile.context.checkpoint_max_tokens,
        "effective_max_context_tokens": policy.max_context_tokens,
        "host_inactive_prefix_bytes": profile.host_prefix_bytes,
        "devices": devices,
    }))?)
}

pub(super) fn mark_slots(
    prefix: &crate::shared::prefix::PrefixArgs,
    concurrency: usize,
    mark_bytes: usize,
) -> Result<usize> {
    ensure!(concurrency <= super::engine::DECODE_ROWS,
        "MiMo mark concurrency exceeds decode capacity");
    (prefix.prefix_cache_mark_mib as u64).checked_mul(1 << 20)
        .context("prefix mark byte budget overflow")?;
    Ok(prefix.mark_rule(concurrency).slots(mark_bytes))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpertBackend {
    None,
    Skip,
    Local,
    Spark,
}

/// Match `Opened::experts`: unused CLI sources must not reserve their storage.
fn expert_backend(routed_layers: usize, skip: bool, local: bool, peers: bool) -> ExpertBackend {
    if routed_layers == 0 {
        ExpertBackend::None
    } else if skip {
        ExpertBackend::Skip
    } else if local {
        ExpertBackend::Local
    } else if peers {
        ExpertBackend::Spark
    } else {
        ExpertBackend::None
    }
}

fn workspace_options(
    cfg: &MimoV2Config,
    args: &super::EngineArgs,
    programs: &Programs<'_>,
    ranks: usize,
    rank: usize,
    transport_lanes: usize,
    spark: bool,
    head_workspace_bytes: u64,
    prefill_output: MimoPrefillOutput,
    fp8_output: bool,
) -> Result<
    Vec<(
        String,
        cuteafd_loader::families::mimo_v2::MimoWorkspaceOptions,
    )>,
> {
    let _ = head_workspace_bytes;
    cuteafd_loader::families::mimo_v2::admission::workspace_options(cfg, ranks, rank,
        args.prefill_rows, args.max_context, transport_lanes, spark, args.expert_input.into(), args.kv_formats(), args.mtp,
        prefill_output, fp8_output, |name| Ok(programs.spec(name)?.scratch.get("scratch").copied().unwrap_or(0)))
}

fn local_package_manifest(directory: &std::path::Path) -> Result<serde_json::Value> {
    cuteafd_loader::placement::families::mimo_v2::package_manifest(directory, None)?
        .with_context(|| format!("MiMo local package {} needs manifest.json in its package root", directory.display()))
}

fn local_expert_reservations(
    opened: &super::Opened,
    args: &super::EngineArgs,
    layers: usize,
) -> Result<(Vec<MemoryReservation>, cuteafd_loader::placement::ResidentWeights)> {
    use cuteafd_ffi::fp8_moe::{Fp8MoeMetadata, Fp8MoeWeights};
    use cuteafd_loader::formats::fp8_experts::ExpertFormat;
    let count = opened
        .cfg
        .dense
        .iter()
        .take(layers)
        .filter(|&&dense| !dense)
        .count();
    if count == 0 {
        return Ok((Vec::new(), Default::default()));
    }
    let tensors = opened
        .catalog
        .as_ref()
        .context("MiMo local expert catalog was not admitted")?
        .fp8()
        .context("MiMo local experts need native FP8/MXFP4 tensors")?;
    let directory = args.fp8_package.clone().unwrap_or_else(|| {
        crate::shared::experts::fp8::package_directory(&args.native_lib, 1, tensors.format())
    });
    // SAFETY: this is the same trusted package selected by the runtime load;
    // metadata reads its static contract without creating CUDA state.
    let metadata = unsafe { Fp8MoeMetadata::read(&directory) }?;
    let info = &metadata.info;
    let format_matches = matches!(
        (tensors.format(), info.weights),
        (ExpertFormat::Fp8Block128, Fp8MoeWeights::Fp8)
            | (ExpertFormat::Mxfp4, Fp8MoeWeights::Mxfp4)
            | (ExpertFormat::Nvfp4, Fp8MoeWeights::Nvfp4)
    );
    ensure!(
        info.tp == 1
            && !info.wire_input
            && info.hidden == opened.cfg.hidden
            && info.experts == opened.cfg.experts
            && info.topk == opened.cfg.topk
            && info.intermediate == opened.cfg.moe_intermediate
            && info.slice == tensors.slice(1)?
            && format_matches,
        "MiMo local package and checkpoint geometry/format disagree"
    );
    let resident_layers = args.expert_window.map_or(count, |window| window.min(count));
    ensure!(resident_layers > 0, "MiMo expert window must be positive");
    let layer_bytes = crate::shared::experts::fp8::Fp8Layer::bytes(tensors, 1)?;
    let weights = tensor_bytes("MiMo local expert store", &[resident_layers, layer_bytes])?;
    Ok((vec![
        MemoryReservation {
            name: "experts.local_resident_weights".into(),
            bytes: weights,
        },
        MemoryReservation {
            name: "experts.local_scratch".into(),
            bytes: {
                let manifest = local_package_manifest(&directory)?;
                let capacity = super::engine::expert_capacity(args.prefill_rows);
                let bytes = cuteafd_loader::placement::inventory::fp8moe_scratch_bytes(&manifest, "tp1", capacity as u64)
                    .context("MiMo local package lacks tp1 scratch at selected capacity")?;
                ensure!(bytes == metadata.scratch_for(capacity)?.max(256) as u64,
                    "MiMo local package JSON/native scratch contracts disagree");
                bytes
            },
        },
    ], cuteafd_loader::placement::ResidentWeights(weights)))
}

pub(super) fn draft_reservations(
    library: &cuteafd_ffi::NativeLibrary,
    directory: &std::path::Path,
    cfg: &super::dflash::DflashConfig,
    mode: cuteafd_loader::families::mimo_v2::draft_representation::MimoDraftRepresentation,
    capacity: cuteafd_loader::families::mimo_v2::draft_representation::MimoDraftCapacity,
) -> Result<DraftReservations> {
    draft_rank_reservations(library, directory, cfg, mode, capacity, 0, 1)
}

pub(super) fn draft_rank_reservations(
    library: &cuteafd_ffi::NativeLibrary,
    directory: &std::path::Path,
    cfg: &super::dflash::DflashConfig,
    mode: cuteafd_loader::families::mimo_v2::draft_representation::MimoDraftRepresentation,
    capacity: cuteafd_loader::families::mimo_v2::draft_representation::MimoDraftCapacity,
    rank: usize,
    ranks: usize,
) -> Result<DraftReservations> {
    let headers = cuteafd_loader::read_safetensors_metadata(
        &directory.join("dflash_draft_model.safetensors"),
    )?;
    draft_rank_reservations_with(cfg, &headers, mode, capacity, rank, ranks, |shape| match shape {
        DraftScratch::Fp8 { rows, k, n } => library.fp8_w8a16_workspace(rows, k, n),
        DraftScratch::Attention {
            sequences,
            heads,
            kv_heads,
            block,
            keys,
        } => library.mimo_dflash_attention_workspace(sequences, heads, kv_heads, block, keys),
        DraftScratch::Topk { rows } => library.glm_dflash_topk_workspace(rows),
    })
}


/// The same reusable scratch allocation serves every target/MTP program at
/// this workspace shape. Rank0 retains the unsplit family too under TP2;
/// native MTP uses those programs even when all target layers are split.

pub(super) fn workspace_native_scratch(
    cfg: &MimoV2Config,
    programs: &Programs<'_>,
    ranks: usize,
    rank: usize,
    decode: bool,
    kv: MimoKvFormats,
    fp8_output: bool,
) -> Result<u64> {
    workspace_scratch(cfg, ranks, rank, decode, kv, fp8_output, |name| {
        Ok(programs
            .spec(name)?
            .scratch
            .get("scratch")
            .copied()
            .unwrap_or(0))
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn local_admission_reads_manifest_from_sealed_package_root() {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("tp1");
        std::fs::create_dir(&directory).unwrap();
        std::fs::write(directory.join("libcuteafd_fp8moe.so"), b"layout fixture").unwrap();
        let manifest = serde_json::json!({"layouts": {"tp1": {"scratch": 4096}}});
        std::fs::write(root.path().join("manifest.json"), manifest.to_string()).unwrap();
        assert_eq!(super::local_package_manifest(&directory).unwrap(), manifest);
        std::fs::remove_file(root.path().join("manifest.json")).unwrap();
        let error = super::local_package_manifest(&directory).unwrap_err().to_string();
        assert!(error.contains("manifest.json in its package root"));
        std::fs::write(root.path().join("manifest.json"), b"invalid JSON").unwrap();
        assert!(super::local_package_manifest(&directory).is_err());
    }

    #[test]
    fn prefill_uses_shared_fixed_depth() {
        let shape = super::prefill_shape(6, None).unwrap();
        assert_eq!((shape.unit_rows, shape.waves), (2752, 3));
        assert_eq!(super::prefill_shape(6, Some(1024)).unwrap().waves, 3);
        assert!(super::prefill_shape(6, Some(4096)).is_err());
        let local = super::prefill_shape(0, None).unwrap();
        assert_eq!((local.unit_rows, local.waves), (4096, 1));
    }

    #[test]
    fn explicit_prefill_pins_match_shared_planner_without_fallback() {
        use cuteafd_loader::serving_capacity::prefill_waves::{prefill_shape, PrefillLimits, PrefillOverrides};
        for ranks in [0, 2, 6] { for (rows, waves) in [(0, 0), (1024, 0), (4096, 2), (2752, 3), (2048, 4), (4096, 3)] {
            let runtime = super::resolved_prefill_shape(ranks, rows, waves);
            let planner = prefill_shape(ranks, PrefillLimits::default(), PrefillOverrides {
                unit_rows: (rows > 0).then_some(rows as u64),
                waves: (waves > 0).then_some(waves as u64), ..Default::default()
            });
            match planner {
                Ok(shape) => assert_eq!(runtime.unwrap(), shape.units()),
                Err(_) => assert!(runtime.is_err(), "invalid pin must not shrink"),
            }
        } }
    }

    #[test]
    fn reservations_follow_the_executed_expert_backend() {
        use super::{expert_backend, ExpertBackend};
        // Dense-only and skip diagnostics allocate neither a local store nor
        // Spark intake even when inherited launch flags name both sources.
        assert_eq!(expert_backend(0, false, true, true), ExpertBackend::None);
        assert_eq!(expert_backend(1, true, true, true), ExpertBackend::Skip);
        assert_eq!(expert_backend(1, false, true, true), ExpertBackend::Local);
        assert_eq!(expert_backend(1, false, false, true), ExpertBackend::Spark);
        assert_eq!(expert_backend(1, false, false, false), ExpertBackend::None);
    }

    use super::*;
    use cuteafd_loader::families::mimo_v2::admission::workspace_shapes;

    #[test]
    fn automatic_context_clamp_updates_aligned_full_sequence_capacity() {
        use cuteafd_core::serving_capacity::*;
        let profile = CapacityProfile {
            context: ContextLimits { checkpoint_max_tokens: 1048576, compiled_index_max_tokens: None },
            pool_unit_rows: 64,
            devices: vec![DeviceCosts { device: 0, pool_unit_bytes: 64, reservations: vec![], reserve_bytes: 0 }],
            host_prefix_bytes: 0,
        };
        let policy = CapacityPolicy { pool_tokens: Some(962560), ..CapacityPolicy::default() };
        let memory = [DeviceMemory { device: 0, total_bytes: 2000000, baseline_free_bytes: 2000000 }];
        let mut capacity = resolve_capacity(policy, &profile, &memory).unwrap();
        assert_eq!(capacity.active_max_context_sequences, 0);
        assert!(resolve_pool_context(&mut capacity.clone(), cuteafd_loader::placement::planner::Context::pinned(1 << 20), 64).is_err());
        resolve_pool_context(&mut capacity, cuteafd_loader::placement::planner::Context::automatic(1 << 20), 64).unwrap();
        assert_eq!(capacity.effective_max_context_tokens, 962560);
        assert_eq!(capacity.active_max_context_sequences, 1);
    }

    #[test]
    fn prefill_units_admit_one_scratch_and_one_head() {
        for waves in 1..=4 {
            assert_eq!(workspace_shapes(4096, true, true, waves, MimoPrefillOutput::LastRow, 0),
                [("prefill", false, 4096, true), ("decode", true, 64, true)]);
        }
    }

    #[test]
    fn failed_pool_report_includes_complete_physical_costs_without_host_capacity() {
        use cuteafd_core::serving_capacity::{
            CapacityPolicy, CapacityProfile, ContextLimits, DeviceCosts, DeviceMemory,
        };
        let padding = cuteafd_core::memory_policy::MemoryPolicy::current().release_padding();
        let profile = CapacityProfile {
            context: ContextLimits {
                checkpoint_max_tokens: 32,
                compiled_index_max_tokens: None,
            },
            pool_unit_rows: 64,
            host_prefix_bytes: 99 << 30,
            devices: vec![DeviceCosts {
                device: 1,
                pool_unit_bytes: 3200,
                reserve_bytes: padding,
                reservations: [
                    ("model.layers.0.values", 1000),
                    ("draft.weights", 1000),
                    ("state.rings", 500),
                    ("runtime.provisional_bound", 1024),
                    ("prefill.shadow", 2000),
                ]
                .into_iter()
                .map(|(name, bytes)| MemoryReservation {
                    name: name.into(),
                    bytes,
                })
                .collect(),
            }],
        };
        let memory = [DeviceMemory {
            device: 1,
            total_bytes: 10000 + padding,
            baseline_free_bytes: 9000 + padding,
        }];
        let policy = CapacityPolicy {
            max_context_tokens: Some(32),
            pool_tokens: Some(127),
            ..CapacityPolicy::default()
        };
        let report: serde_json::Value =
            serde_json::from_str(&reservation_report(&profile, &memory, policy).unwrap()).unwrap();
        assert_eq!(report["requested_pool_tokens"], 128);
        let device = &report["devices"][0];
        // The sample includes 100MiB release padding; subtract it and sampled
        // occupancy (1000), with no 97% ceiling, before this tiny pool test.
        assert_eq!(device["minimum_free_bytes"], padding);
        assert_eq!(device["engine_budget_bytes"], 9000);
        assert_eq!(device["fixed_bytes"], 5524);
        assert_eq!(device["requested_pool_bytes"], 6400);
        assert_eq!(device["complete_requested_bytes"], 11924);
        assert_eq!(device["fixed_categories"]["draft"], 1000);
        assert_eq!(device["fixed_categories"]["prefill"], 2000);
        assert!(
            cuteafd_core::serving_capacity::resolve_capacity(policy, &profile, &memory).is_err()
        );
    }

    #[test]
    fn serving_mark_arena_honors_zero_budget_and_host_staging() {
        use clap::Parser;
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            prefix: crate::shared::prefix::PrefixArgs,
        }
        let cli = Cli::parse_from(["serve", "--prefix-cache-mark-mib", "0"]);
        // Device-only retention is disabled when no complete mark fits.
        assert_eq!(mark_slots(&cli.prefix, 16, 37_500_000).unwrap(), 0);
        let hosted = Cli::parse_from(["serve", "--prefix-cache-mark-mib", "0", "--host-cache-bytes", "auto"]);
        assert_eq!(mark_slots(&hosted.prefix, 16, 37_500_000).unwrap(), 2);
        let disabled = Cli::parse_from(["serve", "--prefix-cache-entries", "0"]);
        assert_eq!(mark_slots(&disabled.prefix, 16, 37_500_000).unwrap(), 0);
        assert!(mark_slots(&cli.prefix, 65, 37_500_000).is_err());
    }

    fn draft_fixture() -> (
        super::super::dflash::DflashConfig,
        Vec<cuteafd_loader::SafetensorsTensorMetadata>,
    ) {
        use cuteafd_core::DType;
        let cfg = super::super::dflash::DflashConfig {
            hidden: 128,
            intermediate: 128,
            layers: 1,
            heads: 1,
            kv_heads: 1,
            head_dim: 128,
            rope_dim: 64,
            theta: 1e6,
            eps: 1e-6,
            block: 8,
            mask_token: 0,
            taps: vec![0],
            vocab: 256,
            window: 1024,
            v_scale: 1.0,
            sinks: true,
        };
        let headers = [
            ("fc.weight", vec![128, 128]),
            ("hidden_norm.weight", vec![128]),
            ("norm.weight", vec![128]),
            ("layers.0.input_layernorm.weight", vec![128]),
            ("layers.0.post_attention_layernorm.weight", vec![128]),
            ("layers.0.self_attn.q_proj.weight", vec![128, 128]),
            ("layers.0.self_attn.k_proj.weight", vec![128, 128]),
            ("layers.0.self_attn.v_proj.weight", vec![128, 128]),
            ("layers.0.self_attn.q_norm.weight", vec![128]),
            ("layers.0.self_attn.k_norm.weight", vec![128]),
            ("layers.0.self_attn.attention_sink_bias", vec![1]),
            ("layers.0.self_attn.o_proj.weight", vec![128, 128]),
            ("layers.0.mlp.gate_proj.weight", vec![128, 128]),
            ("layers.0.mlp.up_proj.weight", vec![128, 128]),
            ("layers.0.mlp.down_proj.weight", vec![128, 128]),
        ]
        .into_iter()
        .map(|(name, shape)| cuteafd_loader::SafetensorsTensorMetadata {
            name: name.into(),
            dtype: DType::Bf16,
            byte_offset: 0,
            byte_length: 2 * shape.iter().product::<usize>() as u64,
            shape,
        })
        .collect();
        (cfg, headers)
    }

    #[test]
    fn draft_profile_distinguishes_context_slots_from_batch_workspace() {
        use cuteafd_loader::families::mimo_v2::draft_representation::{
            MimoDraftCapacity, MimoDraftRepresentation,
        };
        let (cfg, headers) = draft_fixture();
        let capacity = MimoDraftCapacity::new(20, 16, cfg.block).unwrap();
        let mode = MimoDraftRepresentation::Fp8Only;
        let layout = cfg.runtime_layout(mode, capacity).unwrap();
        let reservations = draft_reservations_with(&cfg, &headers, mode, capacity, |shape| {
            match shape {
                DraftScratch::Fp8 { rows, .. } => {
                    assert_eq!(rows, super::super::dflash::TAP_ROWS)
                }
                DraftScratch::Attention {
                    sequences, keys, ..
                } => {
                    assert_eq!((sequences, keys), (16, 1032));
                }
                DraftScratch::Topk { rows } => assert_eq!(rows, 16 * 7),
            }
            Ok(2048)
        })
        .unwrap();
        let costs = reservations.steady;
        let bytes = |name: &str| costs.iter().find(|r| r.name == name).unwrap().bytes;
        assert_eq!(bytes("draft.layer0.k_ring"), 20 * 1024 * 128 * 2);
        assert_eq!(bytes("draft.layer0.v_ring"), 20 * 1024 * 128 * 2);
        assert_eq!(bytes("draft.workspace.logits"), 16 * 8 * 256 * 4);
        assert_eq!(reservations.arena.layout.payload_bytes() as u64,
            layout.weights.resident_bytes().unwrap());
        assert_eq!(reservations.resident_weights.0, reservations.arena.layout.bytes() as u64);
        assert!(!costs
            .iter()
            .any(|cost| cost.name == "draft.weights.bf16_values"
                || cost.name.starts_with("draft.weights.head_fp8")));
        assert_eq!(
            reservations
                .packing
                .iter()
                .find(|cost| cost.name == "loading.draft_source_matrix")
                .unwrap()
                .bytes,
            layout.weights.max_load_staging
        );
        assert!(!reservations
            .packing
            .iter()
            .any(|cost| cost.name.contains("ring") || cost.name.contains("workspace")));
        assert!(!costs.iter().any(|cost| cost.name.starts_with("loading.")));
    }

    #[test]
    fn tp2_draft_admission_validates_full_sources_and_charges_local_owners() {
        use cuteafd_loader::families::{mimo_v2::draft_representation::{MimoDraftCapacity,
            MimoDraftRepresentation, mimo_draft_mark_bytes}, qwen4::Qwen4KvCache};
        let (mut cfg, mut headers) = draft_fixture();
        cfg.heads = 4;
        cfg.kv_heads = 2;
        cfg.intermediate = 256;
        for header in &mut headers {
            let name = header.name.as_str();
            header.shape = if name.ends_with("attention_sink_bias") { vec![4] }
                else if name.ends_with("q_proj.weight") { vec![512, 128] }
                else if name.ends_with("k_proj.weight") || name.ends_with("v_proj.weight") { vec![256, 128] }
                else if name.ends_with("o_proj.weight") { vec![128, 512] }
                else if name.ends_with("gate_proj.weight") || name.ends_with("up_proj.weight") { vec![256, 128] }
                else if name.ends_with("down_proj.weight") { vec![128, 256] }
                else { header.shape.clone() };
            header.byte_length = 2 * header.shape.iter().product::<usize>() as u64;
        }
        for mode in [MimoDraftRepresentation::Bf16Only, MimoDraftRepresentation::Fp8Only] {
            for kv in [Qwen4KvCache::Bf16, Qwen4KvCache::Fp8] {
                let capacity = MimoDraftCapacity { kv_cache: kv,
                    ..MimoDraftCapacity::new(20, 16, cfg.block).unwrap() };
                let whole = draft_reservations_with(&cfg, &headers, mode, capacity, |_| Ok(2048)).unwrap();
                let local = draft_rank_reservations_with(&cfg, &headers, mode, capacity, 0, 2, |shape| {
                    if let DraftScratch::Attention { heads, kv_heads, .. } = shape {
                        assert_eq!((heads, kv_heads), (2, 1));
                    }
                    Ok(2048)
                }).unwrap();
                let bytes = |owners: &[cuteafd_core::serving_capacity::MemoryReservation], name: &str|
                    owners.iter().find(|owner| owner.name == name).unwrap().bytes;
                for ring in ["draft.layer0.k_ring", "draft.layer0.v_ring"] {
                    let local_record = kv.plane_record_bytes(1, 128) as u64;
                    assert_eq!(bytes(&local.steady, ring), 20 * 1024 * local_record);
                    if kv == Qwen4KvCache::Bf16 {
                        assert_eq!(2 * bytes(&local.steady, ring), bytes(&whole.steady, ring));
                    }
                }
                for owner in ["draft.taps", "draft.fused", "draft.fused_norm", "draft.workspace.h",
                    "draft.workspace.n", "draft.workspace.logits", "draft.workspace.head_workspace"] {
                    assert_eq!(bytes(&local.steady, owner), bytes(&whole.steady, owner), "{owner}");
                }
                assert_eq!(bytes(&local.steady, "draft.workspace.reduced"), capacity.block_rows as u64 * 128 * 2);
                let peer = draft_rank_reservations_with(&cfg, &headers, mode, capacity, 1, 2,
                    |shape| { assert!(!matches!(shape, DraftScratch::Topk { .. })); Ok(2048) }).unwrap();
                for name in ["head_workspace", "logits", "unary", "candidates", "topk_workspace"] {
                    assert!(!peer.steady.iter().any(|owner| owner.name == format!("draft.workspace.{name}")));
                }
                assert_eq!(peer.resident_weights.0, local.resident_weights.0);
                assert_eq!(local.resident_weights.0, local.arena.layout.bytes() as u64);
                assert_eq!(local.arena.layout.payload_bytes() as u64,
                    cfg.for_tp(2).unwrap().runtime_layout(mode, capacity).unwrap().weights.resident_bytes().unwrap());
                assert_eq!(local.arena.lifetime, cuteafd_loader::placement::planner::Lifetime {
                    first: cuteafd_loader::placement::planner::Phase::DraftLoad,
                    last: cuteafd_loader::placement::planner::Phase::Growth,
                });
                assert!(local.packing.iter().all(|owner| !owner.name.contains("ring") && !owner.name.contains("workspace")));
                let whole_mark = mimo_draft_mark_bytes(1, 2, 128, kv).unwrap();
                let local_mark = mimo_draft_mark_bytes(1, 1, 128, kv).unwrap();
                assert_eq!(cfg.mark_bytes(2, kv).unwrap(), 2 * local_mark - 8);
                if kv == Qwen4KvCache::Bf16 { assert_eq!(whole_mark, 2 * local_mark - 8); }
                else { assert!(cfg.mark_bytes(2, kv).unwrap() > whole_mark); }
            }
        }
        headers.iter_mut().find(|t| t.name.ends_with("q_proj.weight")).unwrap().shape[0] /= 2;
        assert!(draft_rank_reservations_with(&cfg, &headers, MimoDraftRepresentation::Bf16Only,
            MimoDraftCapacity::new(20, 16, cfg.block).unwrap(), 0, 2, |_| panic!("native query before source validation")).is_err());
    }

    #[test]
    fn compact_draft_rings_preserve_every_other_reservation() {
        use cuteafd_loader::families::{mimo_v2::draft_representation::{MimoDraftCapacity,
            MimoDraftRepresentation}, qwen4::Qwen4KvCache};
        let (cfg, headers) = draft_fixture();
        for mode in [MimoDraftRepresentation::Bf16Only, MimoDraftRepresentation::Fp8Only] {
            let capacity = MimoDraftCapacity::new(20, 16, cfg.block).unwrap();
            let bf16 = draft_reservations_with(&cfg, &headers, mode, capacity, |_| Ok(2048)).unwrap();
            let compact = MimoDraftCapacity { kv_cache: Qwen4KvCache::Fp8, ..capacity };
            let fp8 = draft_reservations_with(&cfg, &headers, mode, compact, |_| Ok(2048)).unwrap();
            assert_eq!(bf16.steady.len(), fp8.steady.len());
            for (index, (a, b)) in bf16.steady.iter().zip(&fp8.steady).enumerate() {
                assert_eq!(a.name, b.name);
                if a.name.ends_with("_ring") {
                    assert_eq!(a.bytes, 20 * 1024 * 128 * 2);
                    assert_eq!(b.bytes, 20 * 1024 * 144);
                } else if a.name == "draft.backing.tail" {
                    // The slack follows the ring widths, so only its shape is shared.
                    for (reservations, bytes) in [(&bf16, a.bytes), (&fp8, b.bytes)] {
                        assert_eq!(bytes,
                            cuteafd_loader::placement::planner::independent_backing_slack(
                                reservations.steady[..index].iter().map(|cost| cost.bytes)).unwrap());
                    }
                } else {
                    assert_eq!(a.bytes, b.bytes, "{}", a.name);
                }
            }
            assert_eq!(bf16.packing.iter().map(|r| r.bytes).sum::<u64>(),
                fp8.packing.iter().map(|r| r.bytes).sum::<u64>());
        }
    }

    #[test]
    fn unsupported_draft_source_fails_before_native_workspace_queries() {
        use cuteafd_loader::families::mimo_v2::draft_representation::{
            MimoDraftCapacity, MimoDraftRepresentation,
        };
        let (cfg, mut headers) = draft_fixture();
        headers
            .iter_mut()
            .find(|t| t.name == "layers.0.mlp.down_proj.weight")
            .unwrap()
            .dtype = cuteafd_core::DType::F8E4M3;
        let error = draft_reservations_with(
            &cfg,
            &headers,
            MimoDraftRepresentation::Fp8Only,
            MimoDraftCapacity::new(20, 16, cfg.block).unwrap(),
            |_| panic!("source validation must precede every native query"),
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("layers.0.mlp.down_proj.weight"));
        assert!(error.to_string().contains("expected BF16"));
    }

    #[test]
    fn bf16_draft_profile_does_not_query_or_charge_fp8_programs() {
        use cuteafd_loader::families::mimo_v2::draft_representation::{
            MimoDraftCapacity, MimoDraftRepresentation,
        };
        let (cfg, headers) = draft_fixture();
        let capacity = MimoDraftCapacity::new(20, 16, cfg.block).unwrap();
        let reservations = draft_reservations_with(
            &cfg,
            &headers,
            MimoDraftRepresentation::Bf16Only,
            capacity,
            |shape| {
                assert!(!matches!(shape, DraftScratch::Fp8 { .. }));
                Ok(2048)
            },
        )
        .unwrap();
        assert!(reservations.steady.iter().all(|r| !r.name.contains("fp8")));
        assert!(reservations.steady.iter().all(|r| !r.name.starts_with("draft.weights.")));
        assert!(reservations.arena.layout.views().keys().all(|name| !name.ends_with(".payload")));
        assert!(reservations.packing.is_empty());
        let invalid = MimoDraftCapacity {
            kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache::Bf16,
            context_slots: 16,
            max_batch_sequences: 20,
            block_rows: 160,
        };
        assert!(draft_reservations_with(
            &cfg,
            &headers,
            MimoDraftRepresentation::Bf16Only,
            invalid,
            |_| Ok(0)
        )
        .is_err());
    }

    #[test]
    fn lead_reserves_unsplit_mtp_programs_and_peer_only_its_share() {
        let cfg = MimoV2Config::from_hf(&cuteafd_loader::plan::testing::mimo_pro_config()).unwrap();
        let mut lead = Vec::new();
        let bytes = workspace_scratch(&cfg, 2, 0, true, MimoKvFormats::default(), false, |name| {
            lead.push(name.to_string());
            Ok(if name.starts_with("mimop_") {
                4096
            } else {
                2048
            })
        })
        .unwrap();
        assert_eq!(bytes, 4096);
        assert!(lead.iter().any(|n| n == "mimop_router_scores"));
        assert!(lead.iter().any(|n| n.starts_with("mimop2_")));
        assert!(lead.iter().any(|n| n.starts_with("mimop_full_attention")));
        let mut peer = Vec::new();
        let bytes = workspace_scratch(&cfg, 2, 1, false, MimoKvFormats::default(), false, |name| {
            peer.push(name.to_string());
            Ok(2048)
        })
        .unwrap();
        assert_eq!(bytes, 2048);
        assert!(peer.iter().any(|n| n == "mimop_router_scores"));
        assert!(peer.iter().all(|n| n == "mimop_router_scores"
            || (n.starts_with("mimop2_") && n.ends_with("m4096"))));
    }

    #[test]
    fn selected_swa_format_queries_target_and_unsplit_mtp_programs() {
        let cfg = MimoV2Config::from_hf(&cuteafd_loader::plan::testing::mimo_pro_config()).unwrap();
        for sliding in [MimoKvCache::Bf16, MimoKvCache::Int8, MimoKvCache::Fp8] {
            for (rank, decode) in [(0, true), (0, false), (1, true), (1, false)] {
                let mut queried = Vec::new();
                let kv = MimoKvFormats { full: MimoKvCache::Int8, sliding };
                let bytes = workspace_scratch(&cfg, 2, rank, decode, kv, false, |name| {
                    queried.push(name.to_owned());
                    Ok(if name.contains("swa_attention") { 8192 } else { 4096 })
                }).unwrap();
                assert_eq!(bytes, 8192);
                let (mode, cap) = if decode { ("decode", "m64") } else { ("prefill", "m4096") };
                for family in if rank == 0 { vec!["mimop2", "mimop"] } else { vec!["mimop2"] } {
                    assert!(queried.contains(&format!("{family}_swa_producer{}_{}", sliding.program_tag(), cap)));
                    assert!(queried.contains(&format!("{family}_swa_attention{}_{mode}_{cap}", sliding.program_tag())));
                    assert!(queried.contains(&format!("{family}_full_attention_kvint8_{mode}_{cap}")));
                }
            }
        }
    }

    #[test]
    fn missing_selected_native_program_fails_before_workspace_allocation() {
        let cfg =
            MimoV2Config::from_hf(&cuteafd_loader::plan::testing::mimo_flash_config()).unwrap();
        let error = workspace_scratch(&cfg, 1, 0, true, MimoKvFormats { full: MimoKvCache::Bf16, sliding: MimoKvCache::Bf16 }, false, |name| {
            if name == "mimo_full_attention_decode_m64" {
                anyhow::bail!("missing {name}")
            } else {
                Ok(0)
            }
        })
        .unwrap_err();
        assert!(error.to_string().contains("mimo_full_attention_decode_m64"));
    }
    #[test]
    fn selected_fp8_output_reserves_each_rank_family_scratch_and_requires_export() {
        let cfg = MimoV2Config::from_hf(&cuteafd_loader::plan::testing::mimo_pro_config()).unwrap();
        let mut queried = Vec::new();
        let bytes = workspace_scratch(&cfg, 2, 0, false, MimoKvFormats::default(), true, |name| {
            queried.push(name.to_owned());
            Ok(match name {
                "mimop_o_w8_m4096" => 69_206_016,
                "mimop2_o_w8_m4096" => 34_603_008,
                _ => 0,
            })
        })
        .unwrap();
        assert_eq!(bytes, 69_206_016);
        assert!(queried.iter().any(|name| name == "mimop2_o_w8_m4096"));
        let peer = workspace_scratch(&cfg, 2, 1, false, MimoKvFormats::default(), true, |name| {
            Ok(if name == "mimop2_o_w8_m4096" {
                34_603_008
            } else {
                0
            })
        })
        .unwrap();
        assert_eq!(peer, 34_603_008);
        let error = workspace_scratch(&cfg, 2, 1, true, MimoKvFormats::default(), true, |name| {
            if name == "mimop2_o_w8_m64" {
                anyhow::bail!("missing {name}");
            }
            Ok(0)
        })
        .unwrap_err();
        assert!(error.to_string().contains("mimop2_o_w8_m64"));
    }
}
