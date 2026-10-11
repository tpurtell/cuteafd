//! Pre-allocation descriptions shared with MiMo's allocation path. CUDA
//! manifest queries read shape/scratch metadata before loading any weights.
use anyhow::{ensure, Context, Result};
use cuteafd_loader::families::mimo_v2::admission::{tensor_bytes, draft_prefix_reservations, resolve_transport_lanes, workspace_scratch, draft_reservations_with, DraftReservations, DraftScratch};

use cuteafd_core::serving_capacity::MemoryReservation;
use cuteafd_ffi::programs::Programs;
use cuteafd_loader::families::mimo_v2::{MimoKvCache, MimoPrefillOutput, MimoV2Config};

/// Exact tensor admission against the measured post-module baseline.
pub(super) struct Preflight {
    pub capacity: cuteafd_core::serving_capacity::ResolvedCapacity,
    pub graph_plan: cuteafd_loader::families::mimo_v2::decode_graph::MimoDecodeGraphPlan,
    pub graph_bound_bytes: Vec<u64>,
    pub host_config: Option<cuteafd_hostcache::config::Config>,
    pub local_expert_budget: usize,
}

fn resolve_pool_context(capacity: &mut cuteafd_core::serving_capacity::ResolvedCapacity, automatic: bool, unit_rows: u64) -> Result<()> {
    capacity.effective_max_context_tokens = crate::shared::context::pool_context("mimo_v2",
        usize::try_from(capacity.effective_max_context_tokens)?, automatic,
        usize::try_from(capacity.allocated_gpu_kv_tokens)?, super::engine::PAGE_ROWS)? as u64;
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
        let cache = cuteafd_loader::serving_capacity::mimo_cache_geometry(
            &cfg, cfg.layers, 1, MimoKvCache::Int8, 0).unwrap();
        let target: u64 = cache.ranks.iter().map(|r| r.retained_mark_bytes).sum();
        let draft = cuteafd_loader::families::mimo_v2::draft_representation::mimo_draft_mark_bytes(5, 1024).unwrap();
        for concurrency in [4, 16] {
            for enabled in [false, true] {
                // Include a budget-tight case to catch the concurrency floor and
                // a warm mark changing the number of affordable retained slots.
                for mib in [0, 1024, 2048] {
                    let cli = Cli::parse_from(["serve", "--prefix-cache-mark-mib", &mib.to_string()]);
                    let slots = mark_slots(&cli.prefix, concurrency,
                        aggregate_mark_bytes(target, draft, enabled).unwrap() as usize).unwrap() as u64;
                    let layout = plan(dir.path(), &PlanOptions {
                        layout: Some(LayoutOptions {
                            concurrency: concurrency as u64, mimo_prefix_draft: enabled,
                            mimo_prefix_mark_bytes: mib << 20, ..Default::default()
                        }), ..Default::default()
                    }).unwrap().memory_layout.unwrap();
                    let bytes = |name| layout.devices[0].items.iter().filter(|i| i.group == name).map(|i| i.bytes).sum::<u64>();
                    assert_eq!(bytes("marks"), target * slots, "C{concurrency}, warm={enabled}, MiB={mib}: {:?}", layout.notes);
                    let reservations = draft_prefix_reservations(enabled, draft, slots, 16).unwrap();
                    assert_eq!(bytes("DFlash context marks"), reservations.first().map_or(0, |r| r.bytes));
                    assert_eq!(bytes("DFlash valid-floor transfer"), reservations.get(1).map_or(0, |r| r.bytes));
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
        let slots = |enabled| cuteafd_engine::prefix::MarkArena::slots_for(16, 24, bytes(enabled) as usize, 12usize << 30);
        assert_eq!(slots(false), cuteafd_engine::prefix::MarkArena::slots_for(16, 24, target as usize, 12usize << 30));
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
    automatic_context: bool,
) -> Result<Preflight> {
    use cuteafd_core::serving_capacity::{
        admit_device_reservations_with_headroom, resolve_capacity_with_startup_peaks, CapacityPolicy, DeviceMemory,
        small_card_headroom_bytes,
    };
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
    let concurrency = serving.map_or(1, |(_, c)| c);
    ensure!(
        (1..=super::engine::DECODE_ROWS).contains(&concurrency) && concurrency <= args.rings,
        "MiMo concurrency {concurrency} needs that many rings and at most 64 decode rows"
    );
    let context = checkpoint_context_limit(&opened.checkpoint.config)?
        .context("MiMo checkpoint has no max_position_embeddings; specify its context capability in config.json")?;
    let kv = args.kv_cache.into();
    let cache = mimo_cache_geometry(cfg, layers, ranks, kv, args.mtp)?;
    let draft_mark_bytes = args.draft.as_deref().filter(|_| prefix_draft).map(super::dflash::drafter_dir)
        .map(|dir| -> Result<u64> {
            let draft = super::dflash::DflashConfig::read(&dir)?;
            let width = (draft.kv_heads as u64).checked_mul(draft.head_dim as u64)
                .context("DFlash KV width overflow")?;
            Ok(cuteafd_loader::families::mimo_v2::draft_representation::mimo_draft_mark_bytes(draft.layers as u64, width)?)
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
            prefix.host_config(layout, args.max_context)?
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
    let graph_exec_bytes = tensor_bytes(
        "MiMo decode graph executable bound",
        &[args.decode_graph_reserve_kib, 1024],
    )?;
    let graph_driver_margin = tensor_bytes(
        "MiMo decode graph driver margin",
        &[args.decode_graph_driver_reserve_mib, 1 << 20],
    )?;
    let graph_reservations = (0..ranks)
        .map(|rank| graph_plan.reservations(rank, graph_exec_bytes, graph_driver_margin))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let graph_bound_bytes = graph_reservations
        .iter()
        .map(|costs| costs.iter().map(|r| r.bytes).sum::<u64>())
        .collect::<Vec<_>>();
    let transport_lanes = transport_lanes(backend == ExpertBackend::Spark, cfg)?;
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
    let mut runtime = Vec::with_capacity(ranks);
    let mut memory = Vec::with_capacity(ranks);
    let mut post_kv_loading_additional = Vec::with_capacity(ranks);
    let mut draft_packing = Vec::new();
    for (rank, &device) in devices.iter().enumerate() {
        let device_id = u32::try_from(device).context("negative CUDA device id")?;
        let (free, total) =
            crate::shared::peer_split::on_device(library, device, args.device, || {
                library.cuda_memory_info()
            })?;
        memory.push(DeviceMemory {
            device: device_id,
            total_bytes: total as u64,
            baseline_free_bytes: free as u64,
        });
        // The code still to load before ready: the serving thread's cuBLAS handle, lazily loaded
        // functions and transport mappings (`placement::inventory::LOADED_CODE`, which the planner
        // charges in its baseline), instead of another context/module envelope.
        let info = library.cuda_device_info(device)?;
        let arch = format!("sm_{}{}", info.compute_capability_major, info.compute_capability_minor);
        let pending = crate::shared::inventory::pending_code(library, device, rank, ranks == 2,
            cfg.program_family()?, "*")?;
        let cublas = MemoryReservation {
            name: "runtime.loaded_code".into(),
            bytes: if cuteafd_loader::placement::loaded_code(cfg.program_family()?, "*", ranks == 2, rank as u8).is_some() {
                pending
            } else {
                cuteafd_loader::placement::inventory::ArchContext::for_device(&arch, total as u64).cublas_bytes
            },
        };
        let mut additional = vec![cublas.clone()];
        additional.extend(cuteafd_loader::families::mimo_v2::admission::transport_reservations(
            cfg, ranks, rank, args.prefill_rows, transport_lanes, 0)?);
        // Only modules and the already-attached peer are live before drafter
        // packing. Spark transport, samplers, marks, graphs and reusable target
        // workspaces are constructed later and do not consume this phase.
        post_kv_loading_additional.push(additional.clone());
        if rank == 0 {
            additional.extend(cuteafd_loader::families::mimo_v2::admission::transport_reservations(
                cfg, 1, 0, args.prefill_rows, transport_lanes, spark_ranks)?);
            if serving.is_some() {
                additional.extend(cuteafd_loader::families::mimo_v2::admission::sampling_reservations(cfg.vocab_size)?);
            }
            if let Some(dir) = args.draft.as_deref().map(super::dflash::drafter_dir) {
                let draft = super::dflash::DflashConfig::read(&dir)?;
                ensure!(
                    draft.hidden == cfg.hidden
                        && draft.vocab == cfg.vocab_size
                        && draft.taps.iter().all(|&l| l < layers),
                    "DFlash geometry or taps do not fit the selected target layers"
                );
                let reservations = draft_reservations(
                    library,
                    &dir,
                    &draft,
                    opened.weight_formats.draft,
                    args.draft_capacity(draft.block)?,
                )?;
                additional.extend(reservations.steady);
                additional.extend(draft_prefix_reservations(prefix_draft, draft_mark_bytes, mark_slots,
                    args.rings as u64)?);
                if cuteafd_loader::families::mimo_v2::admission::prefill_lane_taps(
                    transport_lanes, args.prefill_rows, args.mtp, prefill_output) {
                    additional.push(MemoryReservation {
                        name: "draft.prefill_first_lane_taps".into(),
                        bytes: draft.prefill_lane_tap_bytes()? as u64,
                    });
                }
                draft_packing = reservations.packing;
            }
            if backend == ExpertBackend::Local {
                additional.extend(local_expert_reservations(opened, args, layers)?);
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
            device: device_id,
            workspaces,
            additional,
            loading_additional: vec![cublas],
        });
    }
    let profiles = mimo_capacity_profiles(
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
            ranks: runtime,
            host_prefix_bytes: host_bytes,
        },
    )?;
    let policy = CapacityPolicy {
        concurrency: u32::try_from(concurrency)?,
        small_card_headroom: true,
        target_pool_tokens: (if memory.iter().any(|m| m.total_bytes <= 32u64 << 30) { 1u64 << 20 }
            else { cuteafd_core::serving_capacity::DEFAULT_GPU_KV_TOKENS }).max(args.max_context as u64),
        max_context_tokens: Some(args.max_context as u64),
        // 0: the largest pool every GPU admits after all fixed costs, up to the target.
        pool_tokens: (args.pool_tokens > 0).then_some(args.pool_tokens as u64),
        ..CapacityPolicy::default()
    };
    let report = reservation_report(&profiles.steady, &memory, policy)?;
    tracing::info!(reservations = %report, "MiMo steady allocation contract after module preload, before weight loads");
    for (loading, &sample) in profiles.loading.iter().zip(&memory) {
        admit_device_reservations_with_headroom(policy.gpu_occupancy_percent, sample,
            &loading.reservations, small_card_headroom_bytes(sample.total_bytes))
            .with_context(|| format!("MiMo target-loading admission; steady contract {report}"))?;
    }
    let intake_probe_bytes = cuteafd_loader::families::mimo_v2::admission::spark_intake_probe_bytes(
        spark_ranks > 0, std::env::var("CUTEAFD_SPARK_INTAKE").ok().as_deref())?;
    for sample in &memory {
        debug_assert_eq!(cuteafd_loader::families::mimo_v2::admission::headroom_bytes(sample.total_bytes, false),
            sample.total_bytes - cuteafd_core::serving_capacity::admission_ceiling(sample.total_bytes, 97,
                small_card_headroom_bytes(sample.total_bytes))?);
    }
    let mut capacity = resolve_capacity_with_startup_peaks(policy, &profiles.steady, &memory,
        &[(memory[0].device, intake_probe_bytes)]).with_context(|| {
        format!("MiMo steady admission; complete per-GPU reservation contract {report}")
    })?;
    let pool_tokens = capacity.allocated_gpu_kv_tokens;
    let shortfall_tokens = (args.max_context as u64).saturating_sub(pool_tokens);
    tracing::info!(pool_tokens, context_tokens=args.max_context, full_context_sequences=pool_tokens / args.max_context as u64,
        slots=concurrency, shortfall_tokens, "MiMo full-context pool admission before allocation");
    resolve_pool_context(&mut capacity, automatic_context, profiles.steady.pool_unit_rows)?;
    if !draft_packing.is_empty() {
        let units = capacity.allocated_gpu_kv_tokens / profiles.steady.pool_unit_rows;
        for (rank, phase) in profiles.post_target_kv.iter().enumerate() {
            let mut costs = phase.reservations.clone();
            costs.extend(post_kv_loading_additional[rank].iter().cloned());
            costs.push(MemoryReservation {
                name: "kv.target_records_before_draft".into(),
                bytes: units
                    .checked_mul(phase.pool_unit_bytes)
                    .context("MiMo packing-phase target KV bytes")?,
            });
            if rank == 0 {
                costs.extend(draft_packing.iter().cloned());
            }
            admit_device_reservations_with_headroom(policy.gpu_occupancy_percent, memory[rank],
                &costs, small_card_headroom_bytes(memory[rank].total_bytes))
                .context("MiMo selected drafter packing admission after target KV allocation")?;
            tracing::info!(rank, reservations = %serde_json::to_string(&costs)?,
                "MiMo selected drafter packing phase before any model allocation");
        }
    }
    let local_expert_budget = capacity.devices[0]
        .reservations
        .iter()
        .filter(|reservation| reservation.name.starts_with("experts.local_"))
        .try_fold(0u64, |bytes, reservation| {
            bytes.checked_add(reservation.bytes)
        })
        .context("MiMo admitted local expert size overflow")?;
    if intake_probe_bytes > 0 {
        // The GPU-landing and H2D probes each release their 64 MiB before
        // transport storage is created. Admit their maximum, not their sum,
        // as the startup peak reserved before automatic pool sizing.
        // On small cards the probe uses floor slack released before serving:
        // charge max(probe, floor), not probe + floor, only for this phase.
        let lead = &capacity.devices[0];
        let mut costs = lead.reservations.clone();
        costs.push(MemoryReservation {
            name: "kv.logical_pool".into(),
            bytes: lead.pool_bytes,
        });
        costs.push(MemoryReservation {
            name: "startup.spark_intake_probe_temporary".into(),
            bytes: intake_probe_bytes,
        });
        admit_device_reservations_with_headroom(policy.gpu_occupancy_percent, memory[0],
            &costs, small_card_headroom_bytes(memory[0].total_bytes).saturating_sub(intake_probe_bytes))?;
    }
    tracing::info!(checkpoint_max_context = capacity.checkpoint_max_context_tokens,
        max_context = capacity.effective_max_context_tokens, requested_pool = args.pool_tokens,
        allocated_pool = capacity.allocated_gpu_kv_tokens, requested_default = capacity.requested_kv_floor_tokens,
        requested_state_slots = capacity.state_slots, allocated_rings = args.rings,
        host_prefix_bytes = capacity.host_prefix_bytes, mark_slots, intake_probe_bytes,
        ?graph_plan, ?graph_bound_bytes,
        reservations = %serde_json::to_string(&capacity.devices)?,
        "MiMo pre-allocation capacity against measured post-module baseline");
    Ok(Preflight {
        capacity,
        graph_plan,
        graph_bound_bytes,
        host_config,
        local_expert_budget: usize::try_from(local_expert_budget)?,
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
        let minimum_free = if policy.small_card_headroom {
            cuteafd_core::serving_capacity::small_card_headroom_bytes(sample.total_bytes)
        } else { 0 };
        let budget = cuteafd_core::serving_capacity::admission_ceiling(
            sample.total_bytes, policy.gpu_occupancy_percent, minimum_free)?
            .saturating_sub(non_engine);
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
    let budget = prefix
        .prefix_cache_mark_mib
        .checked_mul(1 << 20)
        .context("MiMo mark budget overflows")?;
    ensure!(
        concurrency <= super::engine::DECODE_ROWS,
        "MiMo mark concurrency exceeds decode capacity"
    );
    Ok(cuteafd_engine::prefix::MarkArena::slots_for(
        concurrency,
        prefix.prefix_cache_entries,
        mark_bytes,
        budget,
    ))
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

pub(super) fn transport_lanes(spark: bool, cfg: &MimoV2Config) -> Result<usize> {
    if !spark {
        return Ok(1);
    }
    let configured = std::env::var("CUTEAFD_MIMO_PREFILL_LANES").ok();
    resolve_transport_lanes(spark, cfg.program_family()?, configured.as_deref())
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
        args.prefill_rows, args.max_context, transport_lanes, spark, args.kv_cache.into(), args.mtp,
        prefill_output, fp8_output, |name| Ok(programs.spec(name)?.scratch.get("scratch").copied().unwrap_or(0)))
}

fn local_expert_reservations(
    opened: &super::Opened,
    args: &super::EngineArgs,
    layers: usize,
) -> Result<Vec<MemoryReservation>> {
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
        return Ok(Vec::new());
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
            | (ExpertFormat::Nvfp4, Fp8MoeWeights::Nvfp4 { .. })
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
    Ok(vec![
        MemoryReservation {
            name: "experts.local_resident_weights".into(),
            bytes: tensor_bytes("MiMo local expert store", &[resident_layers, layer_bytes])?,
        },
        MemoryReservation {
            name: "experts.local_scratch".into(),
            bytes: {
                let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(directory.join("manifest.json"))?)?;
                let capacity = super::engine::expert_capacity(args.prefill_rows);
                let bytes = cuteafd_loader::placement::inventory::fp8moe_scratch_bytes(&manifest, "tp1", capacity as u64)
                    .context("MiMo local package lacks tp1 scratch at selected capacity")?;
                ensure!(bytes == metadata.scratch_for(capacity)?.max(256) as u64,
                    "MiMo local package JSON/native scratch contracts disagree");
                bytes
            },
        },
    ])
}

pub(super) fn draft_reservations(
    library: &cuteafd_ffi::NativeLibrary,
    directory: &std::path::Path,
    cfg: &super::dflash::DflashConfig,
    mode: cuteafd_loader::families::mimo_v2::draft_representation::MimoDraftRepresentation,
    capacity: cuteafd_loader::families::mimo_v2::draft_representation::MimoDraftCapacity,
) -> Result<DraftReservations> {
    let headers = cuteafd_loader::read_safetensors_metadata(
        &directory.join("dflash_draft_model.safetensors"),
    )?;
    draft_reservations_with(cfg, &headers, mode, capacity, |shape| match shape {
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
    kv: MimoKvCache,
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
    fn prefill_lane_defaults_follow_family_and_preserve_explicit_overrides() {
        for (family, default) in [("mimo", 2), ("mimo2", 2), ("mimof", 2), ("mimof2", 2), ("mimop", 3), ("mimop2", 3)] {
            assert_eq!(super::resolve_transport_lanes(true, family, None).unwrap(), default);
            for lanes in 1..=4 {
                let configured = lanes.to_string();
                assert_eq!(super::resolve_transport_lanes(true, family, Some(&configured)).unwrap(), lanes);
            }
            assert!(super::resolve_transport_lanes(true, family, Some("5")).is_err());
            assert_eq!(super::resolve_transport_lanes(false, family, Some("5")).unwrap(), 1);
        }
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
            devices: vec![DeviceCosts { device: 0, pool_unit_bytes: 64, reservations: vec![] }],
            host_prefix_bytes: 0,
        };
        let policy = CapacityPolicy { pool_tokens: Some(962560), ..CapacityPolicy::default() };
        let memory = [DeviceMemory { device: 0, total_bytes: 2000000, baseline_free_bytes: 2000000 }];
        let mut capacity = resolve_capacity(policy, &profile, &memory).unwrap();
        assert_eq!(capacity.active_max_context_sequences, 0);
        assert!(resolve_pool_context(&mut capacity.clone(), false, 64).is_err());
        resolve_pool_context(&mut capacity, true, 64).unwrap();
        assert_eq!(capacity.effective_max_context_tokens, 962496);
        assert_eq!(capacity.active_max_context_sequences, 1);
    }

    #[test]
    fn independent_prefill_admits_two_lead_heads_and_no_peer_head() {
        let last = MimoPrefillOutput::LastRow;
        assert_eq!(workspace_shapes(4096, true, true, 2, last, 0), [
            ("prefill", false, 4096, true), ("decode", true, 64, true),
            ("prefill_first_lane", false, 4096, true),
        ]);
        assert!(workspace_shapes(4096, false, true, 2, last, 0).iter().all(|shape| !shape.3));
        assert!(!workspace_shapes(4096, true, true, 2, MimoPrefillOutput::AllRows, 0)[2].3);
        assert!(!workspace_shapes(4096, true, true, 2, last, 1)[2].3);
        for (rows, spark, lanes) in [(4096, false, 2), (4096, true, 1), (1024, true, 2)] {
            assert_eq!(workspace_shapes(rows, true, spark, lanes, last, 0).len(), 2);
        }
    }

    #[test]
    fn paired_tap_bank_admits_only_one_additional_bf16_activation_extent() {
        let (mut cfg, _) = draft_fixture();
        cfg.hidden = 4096;
        cfg.taps = vec![9, 19, 29, 39, 49];
        assert_eq!(cfg.prefill_lane_tap_bytes().unwrap(), 1024 * 5 * 4096 * 2);
        cfg.hidden = usize::MAX;
        assert!(cfg.prefill_lane_tap_bytes().is_err());
    }

    #[test]
    fn failed_pool_report_includes_complete_physical_costs_without_host_capacity() {
        use cuteafd_core::serving_capacity::{
            CapacityPolicy, CapacityProfile, ContextLimits, DeviceCosts, DeviceMemory,
        };
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
            total_bytes: 10000,
            baseline_free_bytes: 9000,
        }];
        let policy = CapacityPolicy {
            max_context_tokens: Some(32),
            pool_tokens: Some(63),
            ..CapacityPolicy::default()
        };
        let report: serde_json::Value =
            serde_json::from_str(&reservation_report(&profile, &memory, policy).unwrap()).unwrap();
        assert_eq!(report["requested_pool_tokens"], 64);
        let device = &report["devices"][0];
        assert_eq!(device["engine_budget_bytes"], 8700);
        assert_eq!(device["fixed_bytes"], 5524);
        assert_eq!(device["requested_pool_bytes"], 3200);
        assert_eq!(device["complete_requested_bytes"], 8724);
        assert_eq!(device["fixed_categories"]["draft"], 1000);
        assert_eq!(device["fixed_categories"]["prefill"], 2000);
        assert!(
            cuteafd_core::serving_capacity::resolve_capacity(policy, &profile, &memory).is_err()
        );
    }

    #[test]
    fn serving_mark_arena_reserves_actual_concurrency_even_above_nominal_budget() {
        use clap::Parser;
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            prefix: crate::shared::prefix::PrefixArgs,
        }
        let cli = Cli::parse_from(["serve", "--prefix-cache-mark-mib", "0"]);
        // Sixteen live requests need capture/restore slots even when retained
        // entry marks do not fit the nominal mark-cache budget.
        assert_eq!(mark_slots(&cli.prefix, 16, 37_500_000).unwrap(), 34);
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
        assert_eq!(bytes("draft.weights.fp8_values"), layout.weights.fp8_values);
        assert_eq!(bytes("draft.weights.fp8_scales"), layout.weights.fp8_scales);
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
        assert!(reservations
            .steady
            .iter()
            .any(|r| r.name == "draft.weights.bf16_values"));
        assert!(reservations.packing.is_empty());
        let invalid = MimoDraftCapacity {
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
        let bytes = workspace_scratch(&cfg, 2, 0, true, MimoKvCache::Int8, false, |name| {
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
        let bytes = workspace_scratch(&cfg, 2, 1, false, MimoKvCache::Int8, false, |name| {
            peer.push(name.to_string());
            Ok(2048)
        })
        .unwrap();
        assert_eq!(bytes, 2048);
        assert!(peer
            .iter()
            .all(|n| n.starts_with("mimop2_") && !n.contains("router")));
        assert!(peer.iter().all(|n| n.ends_with("m4096")));
    }

    #[test]
    fn missing_selected_native_program_fails_before_workspace_allocation() {
        let cfg =
            MimoV2Config::from_hf(&cuteafd_loader::plan::testing::mimo_flash_config()).unwrap();
        let error = workspace_scratch(&cfg, 1, 0, true, MimoKvCache::Bf16, false, |name| {
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
        let bytes = workspace_scratch(&cfg, 2, 0, false, MimoKvCache::Int8, true, |name| {
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
        let peer = workspace_scratch(&cfg, 2, 1, false, MimoKvCache::Int8, true, |name| {
            Ok(if name == "mimop2_o_w8_m4096" {
                34_603_008
            } else {
                0
            })
        })
        .unwrap();
        assert_eq!(peer, 34_603_008);
        let error = workspace_scratch(&cfg, 2, 1, true, MimoKvCache::Int8, true, |name| {
            if name == "mimop2_o_w8_m64" {
                anyhow::bail!("missing {name}");
            }
            Ok(0)
        })
        .unwrap_err();
        assert!(error.to_string().contains("mimop2_o_w8_m64"));
    }
}
