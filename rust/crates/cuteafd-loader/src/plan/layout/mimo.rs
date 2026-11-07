//! MiMo's planner consumes the same allocation descriptions as runtime admission.
use super::*;
use crate::families::mimo_v2::{admission::*, capacity::*, resident::MimoResidentOptions,
    projection::MimoProjectionRepresentation as R, weight_policy::*, *};
use anyhow::{Context, Result};

pub(crate) fn layout(report: &mut PlanReport, model: &dyn super::super::FamilyModel,
    input: &super::super::PlanInput, options: &mut LayoutOptions) -> MemoryLayout {
    let spark_ranks = match report.placement { ExpertPlacement::Sparks { ranks } => ranks, _ => 0 };
    let routed = model.spec().moe_layers();
    let all_onboarded = routed > 0
        && options.resolve_onboard(routed) == Some(crate::placement::Onboard::Layers(routed));
    let expert_world = if all_onboarded { 0 } else { spark_ranks };
    // MiMo's external chunk is independent of the shared live-slot arena.
    let external_span = options.prefill_span_rows.take().unwrap_or(16384);
    let selected = super::resolve_prefill(input, report, options, expert_world,
        crate::serving_capacity::prefill_waves::PrefillLimits::default());
    options.prefill_span_rows = Some(external_span);
    if let Err(error) = selected { return unsupported_prefill(report, error); }
    report.prefill_span_rows = Some(external_span);
    let shape = Shape::resolve(input, report, model, options,
        ShapePolicy { context: ContextPolicy::Checkpoint, ..ShapePolicy::manifest_bounded() });
    let (concurrency, context, manifest) = (shape.concurrency, shape.context_tokens,
        shape.manifest.as_deref());
    options.native_mtp_layers = shape.native_layers;
    let checkpoint = input.checkpoint();
    let ranks = if options.head_split { options.rtx_bytes.len().clamp(1, 2) } else { 1 };
    let mut notes = vec![format!("resolved context {context} tokens"), "MiMo attention placement heads".into()];
    if options.tp2_dflash && (ranks != 2 || mimo_dflash_dir(checkpoint, options).is_none()) {
        report.refuse_placement("TP2 DFlash requires two target GPUs and DFlash".into(),
            "Select --rtx 2 with DFlash or disable --tp2-dflash.");
        return MemoryLayout { devices: vec![], pool_tokens: 0, waste: vec![], notes };
    }
    if options.host_embedding || (!options.force_gpu_embedding && options.rtx_bytes[0] <= 32 * GIB) {
        match checkpoint.require_untied_embedding("model.embed_tokens.weight") {
            Ok(()) => if let Some(tensor) = checkpoint.tensors.iter().find(|t| t.meta.name == "model.embed_tokens.weight") {
                let bytes = tensor.meta.byte_length;
                notes.push(format!("embedding placement: host (single pinned mapped copy, {bytes} bytes backing, {bytes} device bytes freed)"));
            },
            Err(error) => notes.push(format!("host embedding ineligible: {error:#}; GPU residency retained")),
        }
    }
    let mut devices = options.rtx_bytes.iter().take(ranks).enumerate().map(|(rank, &total)| {
        let cfg = MimoV2Config::from_hf(&checkpoint.config).ok();
        let base = cfg.as_ref().and_then(|cfg| cfg.program_family().ok()).unwrap_or("mimo");
        let families = std::iter::once(base.to_string()).chain((ranks == 2).then(|| format!("{base}2"))).collect::<Vec<_>>();
        let mut runtime = runtime_item(manifest, &families, "*", ranks == 2, rank, total);
        if ranks == 1 && base == "mimof" {
            runtime.bytes = crate::placement::families::mimo_v2::admission::FLASH_MIN_BASELINE_BYTES;
            runtime.basis = Basis::Calibrated;
        }
        DeviceLayout { kind: DeviceKind::Rtx, index: rank as u32, capacity_bytes: total, worker_budget_bytes: 0,
            items: vec![runtime], kv_tokens: 0 }
    }).collect::<Vec<_>>();
    let mut sparks = (0..expert_world).map(|rank| DeviceLayout { kind: DeviceKind::Spark,
        index: rank as u32, capacity_bytes: options.spark_bytes, worker_budget_bytes: options.spark_bytes, kv_tokens: 0,
        items: vec![cuteafd_core::memory_layout::spark_worker_process_resident_item(),
            Item::new(Category::Runtime, "context+modules", "", 512 * MIB, Basis::Calibrated)] }).collect::<Vec<_>>();
    if all_onboarded {
        for rank in 0..spark_ranks {
            ensure_tower_spark(&mut sparks, rank, options);
        }
    }
    // Media sees fixed target/pool demands and mandatory experts, not the
    // optional Auto prefix. Spark-free admission must reserve every expert.
    if report.encoder.is_none() {
        let towers = mimo_towers(report, checkpoint, model.spec().hidden as u64);
        let mut media_options = options.clone();
        if expert_world > 0 && options.onboard.unwrap_or_default()
            == crate::placement::OnboardSetting::Auto {
            media_options.onboard = Some(crate::placement::OnboardSetting::Layers(0));
        }
        let mut admission = devices.clone();
        let no_media = vec![crate::placement::ResidentWeights::default(); ranks];
        match apply_resolved(input, &mut media_options, &mut admission, &no_media, 0, expert_world, concurrency, context, manifest) {
            Ok((placement, working, ..)) => {
                if !working.spark_layers.is_empty() {
                    if let Err(error) = spark_experts(input, options, &mut sparks, placement.onboard_layers) {
                        report.placement_supported = false; notes.push(format!("MiMo Spark slices: {error:#}"));
                    }
                    // The tower's Spark headroom is what the worker has left after its
                    // rank I/O; the final solve re-charges it for its own working set.
                    spark_io(report, model, options, &working, &mut sparks, &mut notes);
                }
                for device in &mut sparks {
                    if cuteafd_core::memory_layout::charge_spark_runtime(device).is_none() {
                        return unsupported_prefill(report, anyhow::anyhow!("Spark arena reserve overflow"));
                    }
                }
                resolve_encoder(report, &towers, &mut admission, &mut sparks,
                    &vec![0; ranks], options, &mut notes);
                for (device, admitted) in devices.iter_mut().zip(admission) {
                    device.items.extend(admitted.items.into_iter().filter(|item|
                        item.group.starts_with("vision") || item.group.starts_with("audio")));
                }
                for spark in &mut sparks { spark.items.retain(|item| item.group != "routed_expert"); }
            }
            Err(error) => {
                notes.push(format!("MiMo media admission: {error:#}"));
                report.refuse_placement(format!("MiMo media admission: {error:#}"),
                    "Select a fitting Spark layout or a smaller explicit pool.");
                // A mandatory target failure must not leave media unresolved or
                // advertise its unallocated RTX headroom as available.
                for device in &mut admission {
                    device.items.push(Item::new(Category::Reserved, "unavailable target admission", "",
                        device.capacity_bytes, Basis::Formula));
                }
                if expert_world > 0 {
                    let _ = spark_experts(input, options, &mut sparks, 0);
                }
                for device in &mut sparks {
                    if cuteafd_core::memory_layout::charge_spark_runtime(device).is_none() {
                        return unsupported_prefill(report, anyhow::anyhow!("Spark arena reserve overflow"));
                    }
                }
                resolve_encoder(report, &towers, &mut admission, &mut sparks,
                    &vec![0; ranks], options, &mut notes);
                for spark in &mut sparks { spark.items.retain(|item| item.group != "routed_expert"); }
            }
        }
    }
    let mut pool_tokens = 0;
    let mut waste = Vec::new();
    let media_weights = [report.encoder.as_ref(), report.audio_encoder.as_ref()].into_iter()
        .flatten().filter_map(|encoder| match encoder.kind {
            super::super::encoder::EncoderKind::Rtx { gpu } => Some((gpu as i32,
                crate::placement::ResidentWeights(encoder.weights))),
            _ => None,
        }).collect::<Vec<_>>();
    let mut loaded_media = vec![crate::placement::ResidentWeights::default(); ranks];
    let media_devices = (0..ranks as i32).collect::<Vec<_>>();
    let media_admission = media_weights.into_iter().try_for_each(|owner|
        crate::placement::families::mimo_v2::add_local_media_weights(&mut loaded_media,
            &media_devices, Some(owner), crate::placement::ResidentWeights(options.weight_budget_bytes)));
    // The embedding cache serve carves beside an admitted encoder (`with_media_headroom`).
    let media_cache = if [report.encoder.as_ref(), report.audio_encoder.as_ref()].into_iter().flatten()
        .any(|encoder| encoder.kind != super::super::encoder::EncoderKind::Off) {
        let total = super::recipe_host_memory(options).map_or(0, |host| host.0);
        options.media_cache_bytes.unwrap_or((8u64 << 30).min(total / 20))
    } else { 0 };
    let selected = media_admission.map_err(anyhow::Error::from).and_then(|()|
        apply_resolved(input, options, &mut devices, &loaded_media, media_cache, expert_world, concurrency, context, manifest));
    match selected {
        Ok((placement, working, ledger, host)) => {
            report.phase_ledger = Some(ledger);
            report.host_budget_bytes = host.host_budget_bytes;
            report.host_ledger = Some(host);
            report.prefill_span_rows = Some(external_span);
            report.prefill_transports = Some(u64::from(!working.spark_layers.is_empty()));
            report.prefill_unit_rows = options.prefill_unit_rows;
            report.prefill_waves = Some(options.prefill_waves);
            if !working.spark_layers.is_empty() {
                spark_io(report, model, options, &working, &mut sparks, &mut notes);
            }
            waste = replicated_weights(&placement);
            report.solver_placement = Some(placement.clone());
            pool_tokens = placement.pool_tokens;
            notes.push(format!("placement: {}", placement.summary()));
            if let Some(tp2) = placement.tp2 {
                notes.push(format!("rtx0/rtx1: {} TP2 expert layer halves ({}..{})", tp2.layers, tp2.first, tp2.first + tp2.layers));
            }
            if !working.spark_layers.is_empty() {
                let expert_bytes = spark_experts(input, options, &mut sparks, placement.onboard_layers);
                if let Err(error) = expert_bytes { report.placement_supported = false; notes.push(format!("MiMo Spark slices: {error:#}")); }
            }
        }
        Err(error) => {
            notes.push(format!("MiMo placement: {error:#}"));
            report.refuse_placement(format!("MiMo placement: {error:#}"),
                "Supply matching generated BF16-input expert and workspace manifests, or select a fitting Spark layout/pool.");
        }
    }
    if options.mimo_prefix_draft && mimo_dflash_dir(checkpoint, options).is_some_and(|dir| dir.join("config.json").is_file()) {
        let slots = draft_representation::mimo_draft_context_slots(options.mimo_draft_sequences().max(concurrency),
            options.mimo_rings(), options.draft_context_slots);
        notes.push(format!("MiMo DFlash context slots {slots}"));
    }
    if manifest.is_none() { notes.push("MiMo native scratch needs --workspace-manifest; native scratch is not qualified without it".into()); }
    if all_onboarded {
        let media_ranks: std::collections::BTreeSet<usize> = [report.encoder.as_ref(), report.audio_encoder.as_ref()]
            .into_iter().flatten().flat_map(|encoder| encoder.spark_ranks()).collect();
        sparks.retain(|device| media_ranks.contains(&(device.index as usize)));
    }
    devices.extend(sparks);
    MemoryLayout { devices, pool_tokens, waste, notes }
}

/// The non-I/O residual of a MiMo Spark wave at 4096 capacity rows (P11
/// replaces it with the worker's own measurement).
const SPARK_WAVE_BYTES: u64 = gib(51);

/// Each selected Spark rank's exact intake, host exchange, row indices and
/// rings for the working set the solve selected, in MiMo's prefill wire
/// format (`--expert-input`), beside the non-I/O wave residual.
fn spark_io(report: &mut PlanReport, model: &dyn super::super::FamilyModel, options: &LayoutOptions,
    working: &crate::placement::SparkWorkingSet, sparks: &mut [DeviceLayout], notes: &mut Vec<String>) {
    let rings = cuteafd_core::SparkRings { max_frame_bytes: 64 << 20, depth: 8,
        slot_bytes: 8 << 20, alignment: 4096 };
    for device in sparks {
        let io = (|| -> Result<(), crate::placement::PlacementError> {
            let geometry = spark_geometry(model).ok_or(crate::placement::PlacementError::Inventory("Spark experts need routed geometry"))?;
            let capacity = u32::try_from(options.spark_capacity_rows)
                .map_err(|_| crate::placement::PlacementError::Overflow("Spark capacity"))?;
            let endpoints = crate::families::mimo_v2::admission::spark_endpoints(capacity as usize, options.mimo_expert_input)
                .map_err(|_| crate::placement::PlacementError::Inventory("MiMo Spark endpoint geometry"))?;
            charge_spark_io(device, working, geometry, capacity, options.mimo_expert_input.bf16(false),
                spark_ring_geometry(1), SPARK_WAVE_BYTES * options.spark_capacity_rows / 4096)?;
            let exact = working.rank_endpoints_io(device.index as usize, geometry, &endpoints, rings)?;
            for item in &mut device.items {
                match item.group.as_str() {
                    "expert intake" => item.bytes = exact.intake,
                    "host expert exchange" => item.bytes = exact.host_exchange,
                    "expert row indices" => item.bytes = exact.row_indices,
                    "rdma rings" => item.bytes = exact.mapped_rings,
                    "endpoint response prefixes" => item.bytes = exact.endpoint_staging,
                    _ => {},
                }
            }
            Ok(())
        })();
        if let Err(error) = io {
            report.placement_supported = false;
            notes.push(format!("{} I/O admission: {error}", device.name()));
        }
    }
}

/// GQA/SWA cache records and projection shards are partitioned, but the
/// head-split executor still keeps a whole router and layer norms per rank.
fn replicated_weights(placement: &crate::placement::Placement) -> Vec<Waste> {
    let Some(peer) = placement.items.get(1) else { return Vec::new(); };
    let bytes = peer.iter().filter(|item| item.category == Category::Weights
        && (item.group.contains("layernorm") || item.group.contains(".mlp.gate."))
        && placement.items[0].contains(item)).map(|item| item.bytes).sum();
    if bytes == 0 { Vec::new() } else {
        vec![Waste { device: "rtx1".into(), what: "router and layer norms replicated under the head split".into(), bytes }]
    }
}

fn spark_experts(input: &super::super::PlanInput, _options: &LayoutOptions,
    sparks: &mut [DeviceLayout], onboard: usize) -> Result<()> {
    if sparks.is_empty() { return Ok(()); }
    let checkpoint = input.checkpoint();
    let catalog = input.expert_catalog()?;
    let tensors = catalog.fp8().context("MiMo Spark expert headers")?;
    let cfg = MimoV2Config::from_hf(&checkpoint.config)?;
    let routed = cfg.dense.iter().enumerate().filter_map(|(layer, &dense)| (!dense).then_some(layer))
        .collect::<Vec<_>>();
    anyhow::ensure!(onboard <= routed.len(), "MiMo onboard exceeds routed layers");
    let remote = &routed[onboard..];
    let ranks = sparks.len();
    use crate::formats::fp8_experts::ExpertFormat;
    let family = match (cfg.program_family()?, tensors.format()) {
        ("mimo", ExpertFormat::Mxfp4) => "mimof",
        (family, _) => family,
    };
    let format = match tensors.format() { ExpertFormat::Nvfp4 => "nvfp4", _ => "fp8" };
    // Coordinator images do not carry Spark packages; support comes from the
    // shared qualified recipe, never absence in the coordinator's local tree.
    let recipe = super::super::experts::spark_recipe(&format!("{family}:{format}"),
        tensors.shape().intermediate, ranks)?;
    for spark in sparks {
        let slicing = recipe.slicing;
        let bytes = remote.iter().try_fold(0u64, |total, &layer| {
            let view = tensors.for_layer_metadata(layer)?;
            let bytes = crate::placement::families::mimo_v2::expert_layer_bytes(&view,
                ranks, spark.index as usize, slicing)?;
            total.checked_add(bytes).context("MiMo Spark expert bytes overflow")
        })?;
        let format = match slicing { crate::formats::fp8_experts::Slicing::Padded => "native padded blocks",
            crate::formats::fp8_experts::Slicing::Blocks(_) => "native exact blocks" };
        spark.items.push(Item::new(Category::Experts, "routed_expert", format, bytes, Basis::Exact));
    }
    Ok(())
}

/// The bundled DFlash drafter directory, only when this launch serves it.
pub(super) fn mimo_dflash_dir(checkpoint: &super::super::Checkpoint, options: &LayoutOptions)
    -> Option<std::path::PathBuf> {
    let dir = checkpoint.snapshot.join("dflash");
    (!options.mimo_drafter_disabled && dir.is_dir()).then_some(dir)
}

fn use_host_embedding(checkpoint: &super::super::Checkpoint, options: &LayoutOptions) -> bool {
    (options.host_embedding || (!options.force_gpu_embedding && options.rtx_bytes[0] <= 32 * GIB))
        && checkpoint.require_untied_embedding("model.embed_tokens.weight").is_ok()
}

pub(super) fn profile(checkpoint: &super::super::Checkpoint, options: &LayoutOptions,
    ranks: usize, spark_ranks: usize, rows: u64, concurrency: u64, context: u64,
    manifest: Option<&serde_json::Value>) -> Result<MimoCapacityProfiles> {
    let cfg = MimoV2Config::from_hf(&checkpoint.config)?;
    let mtp = selected_mtp(checkpoint, options);
    let qualified = default_policy(checkpoint, &cfg) == MimoDefaultPolicy::Fp8;
    let source = |name: &str| checkpoint.tensors.iter().find(|t| t.meta.name == name)
        .map(|t| if t.meta.dtype == cuteafd_core::DType::Bf16 && !qualified { R::Bf16 } else { R::Fp8 });
    let output_formats: std::collections::BTreeMap<_, _> = checkpoint.tensors.iter()
        .filter(|tensor| tensor.meta.name.ends_with(".self_attn.o_proj.weight"))
        .filter_map(|tensor| source(&tensor.meta.name).map(|format| (tensor.meta.name.clone(), format)))
        .collect();
    let fp8_output = output_formats.values().any(|&r| r == R::Fp8);
    let lanes = if options.prefill_lanes > 0 { options.prefill_lanes } else { prefill_shape(spark_ranks, None)?.waves } as usize;
    let rings = options.mimo_rings();
    let cache = crate::serving_capacity::mimo_cache_geometry(&cfg, cfg.layers, ranks, MimoKvFormats { full: MimoKvCache::Int8, sliding: options.mimo_swa_kv }, selected_mtp(checkpoint, options))?;
    let dir = mimo_dflash_dir(checkpoint, options).unwrap_or_default();
    let draft = if mimo_dflash_dir(checkpoint, options).is_some() && dir.join("dflash_draft_model.safetensors").is_file() {
        Some(draft_config::DflashConfig::read(&dir)?)
    } else { None };
    let draft_mark = if options.mimo_prefix_draft && mimo_dflash_dir(checkpoint, options).is_some() {
        if let Some(draft) = &draft { draft.mark_bytes(if options.tp2_dflash { 2 } else { 1 }, options.draft_kv_cache)? }
        else if dir.join("config.json").is_file() {
            let config = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
            super::mimo_draft_prefix_bytes_tp(&config, 1, rings, options.draft_kv_cache, if options.tp2_dflash { 2 } else { 1 }).map_err(anyhow::Error::msg)?.0
        } else { 0 }
    } else { 0 };
    let mark = cache.ranks.iter().map(|r| r.retained_mark_bytes).sum::<u64>() + draft_mark;
    let marks = options.prefix_slots.unwrap_or_else(|| cuteafd_core::prefix::mark_slots_with_host(
        concurrency, options.prefix_entries("mimo_v2"), mark, options.mimo_prefix_mark_bytes, options.host_cache_bytes != Some(0)));
    let graphs = decode_graph::MimoDecodeGraphPlan::new(cfg.layers, ranks, 64, options.mimo_decode_graphs)?;
    let mut runtime = Vec::new();
    for rank in 0..ranks {
        let mut additional = Vec::new();
        let mut resident_weights = crate::placement::ResidentWeights::default();
        additional.extend(transport_reservations(&cfg, ranks, rank, rows as usize, lanes, spark_ranks, options.resolved_spark_reply_format(false))?);
        if rank == 0 { additional.extend(sampling_reservations(cfg.vocab_size)?); }
        if rank == 0 || options.tp2_dflash {
            if let Some(draft) = &draft {
                let sequences = options.mimo_draft_sequences().max(concurrency);
                let slots = draft_representation::mimo_draft_context_slots(sequences, rings, options.draft_context_slots);
                let mut capacity = draft_representation::MimoDraftCapacity::new(slots as usize, sequences as usize, draft.block)?;
                capacity.kv_cache = options.draft_kv_cache;
                let headers = crate::read_safetensors_metadata(&dir.join("dflash_draft_model.safetensors"))?;
                let costs = draft_rank_reservations_with(draft, &headers,
                    if options.mimo_draft_bf16 { draft_representation::MimoDraftRepresentation::Bf16Only }
                        else { draft_representation::MimoDraftRepresentation::Fp8Only }, capacity, rank,
                    if options.tp2_dflash { 2 } else { 1 }, |shape| draft_scratch_bytes(shape, options.physical_sms.unwrap_or_else(||
                        crate::placement::inventory::ArchContext::for_device("sm_120", options.rtx_bytes[0]).sms) as usize))?;
                resident_weights = resident_weights.checked_add(costs.resident_weights)?;
                additional.extend(costs.steady);

            }
            let local_mark = if options.tp2_dflash && draft_mark > 0 {
                (draft_mark - 8) / 2 + if rank == 0 { 8 } else { 0 }
            } else { draft_mark };
            additional.extend(draft_prefix_reservations(options.mimo_prefix_draft, local_mark, marks,
                if rank == 0 { rings } else { 0 })?);
        }
        let workspaces = workspace_options(&cfg, ranks, rank, rows as usize, context as usize,
            lanes, spark_ranks > 0, options.mimo_expert_input, MimoKvFormats { full: MimoKvCache::Int8, sliding: options.mimo_swa_kv }, mtp,
            if options.full_prefill_logits { MimoPrefillOutput::AllRows } else { MimoPrefillOutput::LastRow },
            fp8_output, |name| {
                let Some(manifest) = manifest else { return Ok(0); };
                let program = manifest["programs"].as_array().context("PROGRAMS.json lacks programs")?
                    .iter().find(|p| p["name"].as_str() == Some(name))
                    .with_context(|| format!("MiMo requires program {name}"))?;
                Ok(program["scratch_bytes_at_capacity"]["scratch"].as_u64().unwrap_or(0))
            })?;
        additional.extend(graphs.reservations(rank, decode_graph::MIMO_GRAPH_EXEC_BOUND_BYTES,
            decode_graph::MIMO_GRAPH_DRIVER_MARGIN_BYTES)?);
        debug_assert_eq!(graphs.graph_set(decode_graph::MIMO_GRAPH_EXEC_BOUND_BYTES,
            decode_graph::MIMO_GRAPH_DRIVER_MARGIN_BYTES).bytes(rank), additional.iter()
                .filter(|r| r.name.starts_with("runtime.decode_graph")).map(|r| r.bytes).sum::<u64>());
        runtime.push(MimoRankRuntime { resident_weights, device: rank as u32, workspaces,
            loading_additional: Vec::new(), additional });
    }
    let host_embedding = use_host_embedding(checkpoint, options);
    Ok(mimo_capacity_profiles(checkpoint, &cfg, MimoResidentOptions {
        layers: cfg.layers, coordinator_ranks: ranks, checkpoint_tp: checkpoint_tp(&checkpoint.snapshot)?,
        native_mtp_layers: mtp, gpu_embedding: !host_embedding, head_format: source("lm_head.weight").unwrap_or(R::Bf16), output_formats,
    }, &MimoCapacityOptions { checkpoint_max_context_tokens:
        crate::serving_capacity::checkpoint_context_limit(&checkpoint.config)?
            .context("MiMo checkpoint lacks max_position_embeddings")?,
        max_context_tokens: context, rings, mark_slots: marks, kv_cache: MimoKvFormats { full: MimoKvCache::Int8, sliding: options.mimo_swa_kv },
        ranks: runtime, host_prefix_bytes: 0 })?)
}

fn selected_mtp(checkpoint: &super::super::Checkpoint, options: &LayoutOptions) -> usize {
    let available = checkpoint.tensors.iter().filter_map(|tensor|
        tensor.meta.name.strip_prefix("model.mtp.layers.")?.split('.').next()?.parse::<usize>().ok())
        .max().map_or(0, |stage| stage + 1);
    options.native_mtp_layers.min(available)
}

fn offline_draft_packing(checkpoint: &super::super::Checkpoint, options: &LayoutOptions)
    -> Result<Vec<Vec<cuteafd_core::serving_capacity::MemoryReservation>>> {
    let ranks = if options.head_split { options.rtx_bytes.len().clamp(1, 2) } else { 1 };
    let Some(dir) = mimo_dflash_dir(checkpoint, options)
        .filter(|dir| dir.join("dflash_draft_model.safetensors").is_file()) else { return Ok(vec![vec![]; ranks]); };
    let draft = draft_config::DflashConfig::read(&dir)?;
    let sequences = options.mimo_draft_sequences().max(options.resolved_concurrency("mimo_v2"));
    let slots = draft_representation::mimo_draft_context_slots(sequences, options.mimo_rings(), options.draft_context_slots);
    let mut capacity = draft_representation::MimoDraftCapacity::new(slots as usize, sequences as usize, draft.block)?;
    capacity.kv_cache = options.draft_kv_cache;
    let packing = draft_rank_reservations_with(&draft, &crate::read_safetensors_metadata(&dir.join("dflash_draft_model.safetensors"))?,
        if options.mimo_draft_bf16 { draft_representation::MimoDraftRepresentation::Bf16Only }
            else { draft_representation::MimoDraftRepresentation::Fp8Only }, capacity, 0,
        if options.tp2_dflash { 2 } else { 1 }, |shape| draft_scratch_bytes(shape, options.physical_sms.unwrap_or_else(||
            crate::placement::inventory::ArchContext::for_device("sm_120", options.rtx_bytes[0]).sms) as usize))?.packing;
    Ok((0..ranks).map(|rank| if rank == 0 || options.tp2_dflash { packing.clone() } else { vec![] }).collect())
}

#[allow(clippy::too_many_arguments)]
fn offline_host(checkpoint: &super::super::Checkpoint, options: &LayoutOptions, cfg: &MimoV2Config,
    ranks: usize, rows: u64, concurrency: u64, context: u64, media_cache: u64)
    -> Result<crate::placement::families::mimo_v2::admission::HostOwners> {
    let cache = crate::serving_capacity::mimo_cache_geometry(cfg, cfg.layers, ranks,
        MimoKvFormats { full: MimoKvCache::Int8, sliding: options.mimo_swa_kv }, selected_mtp(checkpoint, options))?;
    let mut mark_bytes = cache.ranks.iter().map(|rank| rank.retained_mark_bytes).sum::<u64>();
    if let Some(dir) = mimo_dflash_dir(checkpoint, options).filter(|_| options.mimo_prefix_draft) {
        if dir.join("config.json").is_file() {
            let config = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
            mark_bytes += super::mimo_draft_prefix_bytes_tp(&config, 1, options.mimo_rings(), options.draft_kv_cache, if options.tp2_dflash { 2 } else { 1 })
                .map_err(anyhow::Error::msg)?.0;
        }
    }
    let geometry = cuteafd_core::prefix::HostCacheGeometry { unit_rows: cache.logical_unit_rows,
        page_bytes: cache.ranks.iter().map(|rank| rank.persistent_unit_bytes).sum(), mark_bytes, draft_bytes: 0 };
    let host = super::recipe_host_memory(options)?;
    let prefix_bytes = if options.prefix_entries("mimo_v2") == 0 || options.host_cache_bytes == Some(0) { 0 }
        else { geometry.resolve(options.host_cache_bytes, context, concurrency, host.0, geometry.chunk_bytes())
            .context("invalid MiMo host prefix geometry or quota")? };
    Ok(crate::placement::families::mimo_v2::admission::HostOwners {
        expert_input: options.mimo_expert_input,
        output: if options.full_prefill_logits { MimoPrefillOutput::AllRows } else { MimoPrefillOutput::LastRow },
        serving: true, scoring_rows: if options.full_prefill_logits { options.prefill_span_rows.unwrap_or(rows) as usize } else { 64 },
        diagnostic_rows: rows.min(256) as usize, prefix_bytes, media_bytes: 0,
        // Serve adds the media quota to the base headroom (`with_media_headroom`).
        headroom: (8u64 << 30).max(host.0 / 10).checked_add(media_cache).context("MiMo media headroom overflow")?,
    })
}

fn expert_costs(input: &super::super::PlanInput, cfg: &MimoV2Config,
    options: &LayoutOptions, ranks: usize, spark_ranks: usize, rows: u64)
    -> Result<(Vec<Option<crate::placement::ExpertCost>>, u64, [u64; 2], crate::placement::Onboard)> {
    use crate::placement::{Bytes2, ExpertCost, Onboard};
    use crate::placement::families::mimo_v2::{expert_layer_bytes, exact_package, package_directory, package_manifest};
    use crate::formats::fp8_experts::Slicing;
    let catalog = input.expert_catalog()?;
    let routed = cfg.dense.iter().filter(|&&dense| !dense).count();
    let onboard = options.resolve_onboard(routed).unwrap_or(Onboard::Auto);
    let tensors = catalog.fp8().context("MiMo needs native FP8/MXFP4/NVFP4 expert headers")?;
    let lib = crate::placement::inventory::image_lib(options.workspace_manifest.as_deref());
    let directory = package_directory(&lib, cfg, tensors.format(), ranks)?;
    let explicit = options.mimo_expert_manifest.as_deref();
    let path = explicit.map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| directory.parent().expect("package parent").join("manifest.json"));
    let metadata = if onboard.layers(routed) == Some(0) { None }
        else { package_manifest(&directory, explicit)? };
    let mut workspace = 0;
    let mut tp2_workspace = [0; 2];
    if let Some(metadata) = metadata.as_ref() {
        for rank in 0..ranks {
            let (path, slicing) = exact_package(&directory, tensors, ranks, rank);
            let layout = path.file_name().and_then(|name| name.to_str()).context("MiMo package layout")?;
            let info = &metadata["layouts"][layout];
            let expected_weights = match tensors.format() {
                crate::formats::fp8_experts::ExpertFormat::Fp8Block128 => "fp8",
                crate::formats::fp8_experts::ExpertFormat::Mxfp4 => "mxfp4",
                crate::formats::fp8_experts::ExpertFormat::Bf16 => "bf16",
                crate::formats::fp8_experts::ExpertFormat::Nvfp4 => "nvfp4",
            };
            anyhow::ensure!(info["input"] == "bf16" && info["tp"].as_u64() == Some(ranks as u64)
                && info["hidden"].as_u64() == Some(cfg.hidden as u64)
                && info["experts"].as_u64() == Some(cfg.experts as u64)
                && info["top_k"].as_u64() == Some(cfg.topk as u64)
                && info["intermediate"].as_u64() == Some(cfg.moe_intermediate as u64)
                && info["slice"].as_u64() == Some(tensors.rank_width(ranks, rank, slicing)? as u64)
                && info["weights"].as_str().is_some_and(|weights| weights.starts_with(expected_weights)),
                "MiMo expert manifest does not serve the checkpoint BF16-input geometry: {layout}");
            let scratch = crate::placement::inventory::fp8moe_scratch_bytes(metadata, layout, rows.max(64))
                .context("MiMo expert manifest lacks selected capacity")?;
            if ranks == 2 {
                let output = rows.max(64).checked_mul(cfg.hidden as u64).and_then(|bytes| bytes.checked_mul(2))
                    .context("MiMo TP2 output overflow")?;
                tp2_workspace[rank] = scratch.checked_add(output).context("MiMo TP2 workspace overflow")?;
            } else { workspace = scratch; }
        }
    } else {
        anyhow::ensure!(spark_ranks > 0 && (onboard.layers(routed) == Some(0)
            || (onboard == Onboard::Auto && explicit.is_none())),
            "MiMo local experts need matching BF16 package manifest {}", path.display());
    }
    let costs = cfg.dense.iter().enumerate().map(|(layer, &dense)| -> anyhow::Result<_> {
        if dense { return Ok(None); }
        let view = tensors.for_coordinator().for_layer_metadata(layer)?;
        let supported = view.format() == tensors.format();
        let whole = Bytes2 { resident: expert_layer_bytes(&view, 1, 0, Slicing::Padded)?, staging: 0 };
        let half = if ranks == 2 {
            let bytes = |rank| -> anyhow::Result<_> {
                let (_, slicing) = exact_package(&directory, &view, 2, rank);
                Ok(Bytes2 { resident: expert_layer_bytes(&view, 2, rank, slicing)?, staging: 0 })
            };
            [bytes(0)?, bytes(1)?]
        } else { [Bytes2::default(); 2] };
        Ok(Some(ExpertCost { whole, half, tp2: ranks == 2 && supported,
            whole_supported: supported, spark_ok: spark_ranks > 0 }))
    }).collect::<anyhow::Result<Vec<_>>>()?;
    Ok((costs, workspace, tp2_workspace, onboard))
}

/// Admit the planner-selected rows exactly; only the shared reserve search retries.
#[allow(clippy::too_many_arguments)]
fn apply_resolved(input: &super::super::PlanInput, options: &mut LayoutOptions,
    devices: &mut Vec<DeviceLayout>, media_weights: &[crate::placement::ResidentWeights],
    media_cache: u64, spark_ranks: usize, concurrency: u64,
    context: u64, manifest: Option<&serde_json::Value>)
    -> Result<(crate::placement::Placement, crate::placement::SparkWorkingSet,
        crate::plan::SolvedPhaseLedger, cuteafd_core::memory_layout::HostLayout)> {
    let rows = options.prefill_unit_rows.context("MiMo prefill shape must be resolved before admission")?;
    apply_selected(input, options, devices, media_weights, media_cache, spark_ranks, rows, concurrency, context, manifest)
}

#[cfg(test)]
fn apply(input: &super::super::PlanInput, options: &LayoutOptions,
    devices: &mut [DeviceLayout], spark_ranks: usize, rows: u64, concurrency: u64,
    context: u64, manifest: Option<&serde_json::Value>) -> Result<crate::placement::Placement> {
    let media = vec![crate::placement::ResidentWeights::default(); devices.len()];
    apply_selected(input, options, devices, &media, 0, spark_ranks, rows, concurrency, context, manifest).map(|(placement, ..)| placement)
}

/// [`apply`], also returning the working set the solve selected (the Spark
/// layers, ranks and prefill lanes whose I/O the workers allocate).
#[allow(clippy::too_many_arguments)]
fn apply_selected(input: &super::super::PlanInput, options: &LayoutOptions,
    devices: &mut [DeviceLayout], media_weights: &[crate::placement::ResidentWeights], media_cache: u64,
    spark_ranks: usize, rows: u64, concurrency: u64,
    context: u64, manifest: Option<&serde_json::Value>)
    -> Result<(crate::placement::Placement, crate::placement::SparkWorkingSet,
        crate::plan::SolvedPhaseLedger, cuteafd_core::memory_layout::HostLayout)> {
    use crate::placement::families::mimo_v2::admission::{self, ProductionInput};
    use crate::placement::planner::{Baseline, EmbeddingHome, Operation, PrefillShape};
    use crate::placement::ResidentWeights;
    let checkpoint = input.checkpoint();
    let profiles = profile(checkpoint, options, devices.len(), spark_ranks, rows, concurrency, context, manifest)?;
    let no_spark = if spark_ranks == 0 { profiles.clone() }
        else { profile(checkpoint, options, devices.len(), 0, rows, concurrency, context, manifest)? };
    let probe = spark_intake_probe_bytes(spark_ranks > 0, std::env::var("CUTEAFD_SPARK_INTAKE").ok().as_deref())?;
    for device in devices.iter_mut() {
        device.items.retain(|i| matches!(i.category, Category::Tables | Category::Reserved | Category::Runtime)
            || i.group.starts_with("vision") || i.group.starts_with("audio"));
    }
    let cfg = MimoV2Config::from_hf(&checkpoint.config)?;
    let ranks = devices.len();
    anyhow::ensure!(media_weights.len() == ranks, "MiMo media owner/rank geometry");
    let resident_weights = profiles.resident_weights.iter().zip(media_weights)
        .map(|(&target, &media)| target.checked_add(media)).collect::<Result<Vec<_>, _>>()?;
    let (experts, expert_workspace, tp2_workspace, onboard) = expert_costs(input, &cfg, options, ranks, spark_ranks, rows)?;
    let lanes = if options.prefill_lanes > 0 { options.prefill_lanes } else { prefill_shape(spark_ranks, None)?.waves } as usize;
    let kv = MimoKvFormats { full: MimoKvCache::Int8, sliding: options.mimo_swa_kv };
    let maximum_pool_tokens = admission::maximum_pool_tokens(&options.rtx_bytes[..ranks], context,
        options.pool_tokens, profiles.steady.pool_unit_rows, spark_ranks == 0);
    let prefill = PrefillShape { rows, waves: lanes as u64 };
    let operation = if options.full_prefill_logits { Operation::Scoring } else { Operation::Serving };
    let embedding = if use_host_embedding(checkpoint, options) { EmbeddingHome::HostPinned }
        else { EmbeddingHome::Device(0) };
    let budgets = admission::production_budgets(
        &options.rtx_bytes[..ranks].iter().map(|&bytes| (bytes, bytes)).collect::<Vec<_>>(),
        &(0..ranks).map(|rank| admission::device_headroom(rank, probe)).collect::<Vec<_>>(),
        devices.iter().map(|device| Baseline::Forecast { occupied: device.used_bytes(),
            source: "retained checkpoint and media owners before the shared MiMo contract".into(),
            assumed: false }).collect(),
        super::recipe_host_memory(options)?,
        options.spark_bytes, spark_ranks)?;
    let request = admission::production_request(context, options.pool_tokens, onboard, prefill,
        operation, ResidentWeights(options.weight_budget_bytes), options.spark_reply_formats);
    let prepared = ProductionInput {
        expert_usage: options.expert_usage,
        cfg: &cfg, layers: cfg.layers, kv, mtp_stages: selected_mtp(checkpoint, options), context, maximum_pool_tokens,
        prefill, lanes: lanes as u64, embedding, gpus: ranks, spark_ranks,
        experts, expert_workspace, tp2_workspace,
        spark: profiles.clone(), no_spark: no_spark.clone(),
        post_kv_additional: vec![vec![]; ranks], draft_packing: offline_draft_packing(checkpoint, options)?,
        host: offline_host(checkpoint, options, &cfg, ranks, rows, concurrency, context, media_cache)?,
        spark_reply_formats: options.spark_reply_formats,
        resident_weights, operation, budgets,
    }.prepare()?;
    let admitted = match prepared.admit(request) {
        Ok(admitted) => admitted,
        Err(error) => {
            // Failed plans still name the full fixed contract, not fictitious
            // free space left by returning before its storage was described.
            for (rank, device) in devices.iter_mut().enumerate() {
                device.capacity_bytes = options.rtx_bytes[rank].saturating_sub(admission::device_headroom(rank, probe));
                for cost in &profiles.steady.devices[rank].reservations {
                    device.items.push(Item::new(reservation_category(&cost.name),
                        crate::placement::families::mimo_v2::reservation_group(&cost.name), "", cost.bytes, Basis::Formula));
                }
            }
            return Err(error.into());
        }
    };

    let ledger = crate::plan::SolvedPhaseLedger::from(&admitted.solved);
    let host = host_view(&admitted.solved, options, media_cache)?;
    let placement = admitted.placement;
    let working = admitted.spark;
    for (rank, device) in devices.iter_mut().enumerate() {
        device.capacity_bytes = options.rtx_bytes[rank].saturating_sub(admission::device_headroom(rank, probe));
        device.items.extend(placement.items[rank].iter().cloned());
        device.kv_tokens = placement.pool_tokens;
    }
    Ok((placement, working, ledger, host))
}

/// The admitted host owners as the report's host ledger: a view of the one
/// solve, never a second admission. Serve carves the media cache from headroom.
fn host_view(solved: &crate::placement::planner::ResolvedPlan, options: &LayoutOptions, media_cache: u64)
    -> Result<cuteafd_core::memory_layout::HostLayout> {
    use crate::placement::planner::Location;
    use cuteafd_core::memory_layout::ItemLifetime;
    let mut items = solved.items(Location::PINNED).map_err(anyhow::Error::from)?.into_iter()
        .filter(|item| item.category != Category::Prefix)
        .map(|item| item.with_lifetime(ItemLifetime::PostReadyReserve)).collect::<Vec<_>>();
    if media_cache > 0 {
        items.push(Item::new(Category::Embedding, "media/embedding-cache", "host quota", media_cache, Basis::Formula)
            .with_lifetime(ItemLifetime::PostReadyReserve));
    }
    let prefix = solved.items(Location::PINNED).map_err(anyhow::Error::from)?.iter()
        .filter(|item| item.category == Category::Prefix).map(|item| item.bytes).sum::<u64>();
    let (total, available) = super::recipe_host_memory(options)?;
    let budget = solved.budgets.iter().find(|budget| budget.locations.contains(&Location::PINNED))
        .context("MiMo host budget")?;
    let reserve = budget.headroom.saturating_sub(media_cache);
    cuteafd_core::memory_layout::HostLayout::new(total, available, reserve, prefix, items)
        .context("MiMo host ledger overflow")
}

use crate::placement::families::mimo_v2::reservation_category;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::{checkpoint::Checkpoint, testing::*, PlanInput};

    #[test]
    fn planner_equals_runtime_admission_flash_pro_one_two_rtx() {
        for pro in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let mut config = if pro { mimo_pro_config() } else { mimo_flash_mopd_config() };
            config["max_position_embeddings"] = serde_json::json!(1 << 20);
            write_snapshot(dir.path(), &config, &if pro { mimo_pro_tensors() } else { mimo_flash_mopd_tensors() }, Some(if pro { 8 } else { 4 }));
            let checkpoint = Checkpoint::open(dir.path()).unwrap();
            let input = PlanInput::from_checkpoint(checkpoint);
            let checkpoint = input.checkpoint();
            let cfg = MimoV2Config::from_hf(&config).unwrap();
            for swa in [MimoKvCache::Bf16, MimoKvCache::Int8, MimoKvCache::Fp8] {
                let kv = MimoKvFormats { full: MimoKvCache::Int8, sliding: swa };
                for ranks in [1, 2] {
                    for (graphs, expert_input) in [false, true].into_iter().flat_map(|graphs|
                        [MimoExpertInput::Fp8, MimoExpertInput::Bf16, MimoExpertInput::Bf16Decode]
                            .map(|input| (graphs, input))) {
                        let opts = LayoutOptions { native_mtp_layers: 0, host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![95 * GIB; ranks], concurrency: 4,
                            pool_tokens: Some(32768), mimo_decode_graphs: graphs, mimo_swa_kv: swa,
                            mimo_expert_input: expert_input, ..Default::default() };
                        let planned = profile(&checkpoint, &opts, ranks, 4, 4096, 4, 32768, None).unwrap();
                        let lanes = prefill_shape(6, None).unwrap().waves as usize;
                        let cache = crate::serving_capacity::mimo_cache_geometry(&cfg, cfg.layers, ranks, kv, 0).unwrap();
                        let mark_bytes = cache.ranks.iter().map(|r| r.retained_mark_bytes).sum();
                        let slots = cuteafd_core::prefix::mark_slots_with_host(4, opts.prefix_entries("mimo_v2"), mark_bytes, opts.mimo_prefix_mark_bytes, opts.host_cache_bytes != Some(0));
                        let graph = decode_graph::MimoDecodeGraphPlan::new(cfg.layers, ranks, 64, graphs).unwrap();
                        let runtime = (0..ranks).map(|rank| {
                            let mut additional = transport_reservations(&cfg, ranks, rank, 4096, lanes, 4, cuteafd_core::SparkReplyFormat::Bf16).unwrap();
                            if rank == 0 { additional.extend(sampling_reservations(cfg.vocab_size).unwrap()); }
                            additional.extend(graph.reservations(rank, decode_graph::MIMO_GRAPH_EXEC_BOUND_BYTES,
                                decode_graph::MIMO_GRAPH_DRIVER_MARGIN_BYTES).unwrap());
                            MimoRankRuntime { resident_weights: Default::default(), device: rank as u32, loading_additional: vec![], additional,
                                workspaces: workspace_options(&cfg, ranks, rank, 4096, 32768, lanes, true,
                                    opts.mimo_expert_input, kv, 0, MimoPrefillOutput::LastRow, pro, |_| Ok(0)).unwrap() }
                        }).collect();
                        let admitted = mimo_capacity_profiles(&checkpoint, &cfg, MimoResidentOptions {
                            layers: cfg.layers, coordinator_ranks: ranks, checkpoint_tp: if pro { 8 } else { 4 },
                            native_mtp_layers: 0, gpu_embedding: true, head_format: if pro { R::Fp8 } else { R::Bf16 },
                            output_formats: (0..cfg.layers).map(|l| (format!("model.layers.{l}.self_attn.o_proj.weight"),
                                if pro { R::Fp8 } else { R::Bf16 })).collect(),
                        }, &MimoCapacityOptions { checkpoint_max_context_tokens: 1 << 20, max_context_tokens: 32768,
                            rings: opts.mimo_rings(), mark_slots: slots, kv_cache: kv, ranks: runtime, host_prefix_bytes: 0 }).unwrap();
                        let mut devices: Vec<_> = (0..ranks).map(|rank| DeviceLayout {
                            kind: DeviceKind::Rtx, index: rank as u32, capacity_bytes: 95 * GIB, worker_budget_bytes: 0,
                            items: vec![], kv_tokens: 0,
                        }).collect();
                        let placement = apply(&input, &opts, &mut devices, 4, 4096, 4, 32768, None).unwrap();
                        assert_eq!(placement.pool_tokens, 32768);
                        let waste = replicated_weights(&placement);
                        if ranks == 1 {
                            assert!(waste.is_empty());
                        } else {
                            let replica: u64 = admitted.steady.devices[1].reservations.iter()
                                .filter(|r| r.name.contains("layernorm") || r.name.contains(".mlp.gate."))
                                .map(|r| r.bytes).sum();
                            assert!(replica > 0);
                            assert_eq!(waste.len(), 1);
                            assert_eq!(waste[0].bytes, replica);
                            assert_eq!(waste[0].device, "rtx1");
                        }
                        for rank in 0..ranks {
                            let total: u64 = admitted.steady.devices[rank].reservations.iter().map(|r| r.bytes).sum::<u64>()
                                + (32768 / admitted.steady.pool_unit_rows) * admitted.steady.devices[rank].pool_unit_bytes
                                + crate::placement::families::mimo_v2::admission::POST_READY_RUNTIME_BYTES;
                            assert_eq!(devices[rank].used_bytes(), total, "mapped planner items equal admission total");
                            assert_eq!(planned.steady.devices[rank].reservations, admitted.steady.devices[rank].reservations,
                                "pro={pro}, ranks={ranks}, graphs={graphs}, rank={rank}");
                            assert_eq!(planned.steady.devices[rank].pool_unit_bytes, admitted.steady.devices[rank].pool_unit_bytes);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn selected_shape_preserves_explicit_pool_and_publishes_exact_units() {
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &mimo_flash_mopd_config(), &mimo_flash_mopd_tensors(), Some(4));
        let input = PlanInput::from_checkpoint(Checkpoint::open(dir.path()).unwrap());
        for rows in [1024, 2048, 2752] {
            let mut options = LayoutOptions { native_mtp_layers: 0, rtx_bytes: vec![95 * GIB], pool_tokens: Some(32768),
                host_memory: Some((128 << 30, 128 << 30)), prefill_rows: rows, prefill_unit_rows: Some(rows), prefill_waves: 3, prefill_lanes: 3,
                concurrency: 4, ..Default::default() };
            let initial = vec![DeviceLayout { kind: DeviceKind::Rtx, index: 0,
                capacity_bytes: 95 * GIB, worker_budget_bytes: 0, items: vec![], kv_tokens: 0 }];
            let mut expected = initial.clone();
            let media = vec![crate::placement::ResidentWeights::default()];
            let (placement, working, ..) = apply_selected(&input, &options, &mut expected,
                &media, 0, 4, rows, 4, 32768, None).unwrap();
            let mut actual = initial;
            let selected = apply_resolved(&input, &mut options, &mut actual,
                &media, 0, 4, 4, 32768, None).unwrap();
            assert_eq!(selected.0, placement);
            assert_eq!(selected.1, working);
            assert_eq!((options.prefill_unit_rows, options.prefill_waves), (Some(rows), 3));
            assert_eq!(actual[0].used_bytes(), expected[0].used_bytes());
            assert_eq!(actual[0].kv_tokens, 32768);
        }
    }

    #[test]
    fn recorded_inventory_pool_matches_runtime_admission() {
        use cuteafd_core::serving_capacity::{CapacityPolicy, DeviceMemory, resolve_capacity_with_startup_peaks};
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &mimo_flash_mopd_config(), &mimo_flash_mopd_tensors(), Some(4));
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let input = PlanInput::from_checkpoint(checkpoint);
        let checkpoint = input.checkpoint();
        for (total, baseline) in [(101_973_491_712u64, 586_416_128u64),
            (101_970_345_984, 586_416_128), (34_144_990_003, 596_901_888)] {
            let baseline = baseline + 35_651_584;
            let opts = LayoutOptions { native_mtp_layers: 0, host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![total], concurrency: 4, ..Default::default() };
            let mut profile = profile(&checkpoint, &opts, 1, 2, 4096, 4, 1 << 20, None).unwrap();
            let cublas = crate::placement::inventory::ArchContext::for_device("sm_120", total).cublas_bytes;
            profile.steady.devices[0].reservations.push(cuteafd_core::serving_capacity::MemoryReservation {
                name: "runtime.cublas_first_gemm".into(), bytes: cublas,
            });
            // The plan keeps the drained 64 MiB intake probe as rank 0's named reserve.
            let reserve = crate::placement::families::mimo_v2::admission::device_headroom(0, 64 << 20);
            assert_eq!(reserve, 64 << 20);
            profile.steady.devices[0].reserve_bytes = reserve;
            let policy = CapacityPolicy { concurrency: 4,
                target_pool_tokens: if total <= 32 * GIB { 1 << 20 } else { 2 << 20 },
                max_context_tokens: Some(1 << 20), ..Default::default() };
            let admitted = resolve_capacity_with_startup_peaks(policy, &profile.steady,
                &[DeviceMemory { device: 0, total_bytes: total, baseline_free_bytes: total - baseline }],
                &[]).unwrap();
            let mut devices = vec![DeviceLayout { kind: DeviceKind::Rtx, index: 0, capacity_bytes: total, worker_budget_bytes: 0,
                items: vec![Item::new(Category::Runtime, "context+modules", "", baseline + cublas, Basis::Exact)], kv_tokens: 0 }];
            let pool = apply(&input, &opts, &mut devices, 2, 4096, 4, 1 << 20, None).unwrap().pool_tokens;
            assert_eq!(pool, admitted.allocated_gpu_kv_tokens, "recorded total {total}");
            assert_eq!(devices[0].capacity_bytes, total - reserve);
        }
    }

    #[test]
    fn spark_residency_uses_the_worker_package_slice_policy() {
        use crate::placement::families::mimo_v2::expert_layer_bytes;
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &mimo_pro_config(), &mimo_pro_tensors(), Some(8));
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let input = PlanInput::from_checkpoint(checkpoint);
        let checkpoint = input.checkpoint();
        let catalog = crate::read_expert_catalog(&checkpoint.snapshot).unwrap();
        let tensors = catalog.fp8().unwrap();
        let options = LayoutOptions { native_mtp_layers: 0, host_memory: Some((128 << 30, 128 << 30)), workspace_manifest: Some(dir.path().join("share/PROGRAMS.json")),
            ..Default::default() };
        assert!(!crate::placement::inventory::image_lib(options.workspace_manifest.as_deref()).exists());
        let recipe = super::super::super::experts::spark_recipe("mimop:fp8", 2048, 6).unwrap();
        assert_eq!(recipe.widths, [352, 352, 352, 352, 320, 320]);
        assert_eq!(recipe.required_layouts, ["tp6-w352", "tp6-w320"]);
        let mut sparks = (0..6).map(|rank| DeviceLayout {
            kind: DeviceKind::Spark, index: rank, capacity_bytes: 108 * GIB, worker_budget_bytes: 108 * GIB,
            items: vec![], kv_tokens: 0,
        }).collect::<Vec<_>>();
        spark_experts(&input, &options, &mut sparks, 0).unwrap();
        let (hidden, experts, intermediate) = (tensors.shape().hidden as u64,
            tensors.shape().experts as u64, tensors.shape().intermediate as u64);
        let mut padding = 0;
        for spark in &sparks {
            let rank = spark.index as usize;
            let logical = recipe.width_for_layer(rank, 1).unwrap() as u64;
            let stored = recipe.stored_width_for_layer(rank, 1).unwrap() as u64;
            let expected = experts * (3 * hidden * stored / 2 + 2 * stored * hidden / 32
                + hidden * stored.div_ceil(128) * 4);
            assert_eq!(spark.used_bytes(), expected, "rank {rank}");
            assert_eq!(spark.used_bytes(), expert_layer_bytes(&tensors.for_layer_metadata(1).unwrap(),
                6, rank, recipe.slicing).unwrap());
            // Gate/up padding and each down-scale row's independent u32 pitch.
            padding += experts * (3 * hidden * (stored - logical) / 2
                + 2 * (stored - logical) * hidden / 32
                + hidden * (stored.div_ceil(128) * 4 - logical / 32));
        }
        assert!(sparks[0].used_bytes() < sparks[4].used_bytes());
        let full = experts * (3 * hidden * intermediate / 2 + 3 * hidden * intermediate / 32);
        assert_eq!(sparks.iter().map(DeviceLayout::used_bytes).sum::<u64>(), full + padding);
    }

    #[test]
    fn spark_free_native_package_admits_all_experts_and_agentic_pool() {
        use crate::placement::ExpertHome;
        for pro in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let config = if pro { mimo_pro_config() } else { mimo_flash_mopd_config() };
            let tensors = if pro { mimo_pro_tensors() } else { mimo_flash_mopd_tensors() };
            write_snapshot(dir.path(), &config, &tensors, Some(if pro { 8 } else { 4 }));
            let checkpoint = Checkpoint::open(dir.path()).unwrap();
            let input = PlanInput::from_checkpoint(checkpoint);
            let checkpoint = input.checkpoint();
            let cfg = MimoV2Config::from_hf(&config).unwrap();
            for ranks in [1, 2] {
                let manifest = dir.path().join(format!("spark-free-tp{ranks}.json"));
                let mut options = LayoutOptions { native_mtp_layers: 0, host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![32 * GIB; ranks],
                    head_split: ranks == 2, concurrency: 4,
                    mimo_expert_manifest: Some(manifest.clone()), ..Default::default() };
                let devices = || (0..ranks).map(|rank| DeviceLayout {
                    kind: DeviceKind::Rtx, index: rank as u32, capacity_bytes: 32 * GIB, worker_budget_bytes: 0,
                    items: vec![], kv_tokens: 0,
                }).collect::<Vec<_>>();
                assert!(apply(&input, &options, &mut devices(), 0, 4096, 4, 256 << 10, None).is_err(),
                    "Spark-free cannot fall back without its mandatory expert package");
                let scratch = match (pro, ranks) {
                    (true, 1) => 805_573_632, (true, _) => 604_247_040,
                    (false, 1) => 671_354_880, (false, _) => 470_028_288,
                };
                let layout = format!("tp{ranks}");
                std::fs::write(&manifest, serde_json::to_vec(&serde_json::json!({"layouts": {layout: {
                    "input": "bf16", "tp": ranks, "hidden": cfg.hidden, "experts": cfg.experts,
                    "top_k": cfg.topk, "intermediate": cfg.moe_intermediate,
                    "slice": cfg.moe_intermediate / ranks, "weights": "mxfp4",
                    "capacities": [{"capacity": 4096, "scratch_bytes": scratch}]
                }}})).unwrap()).unwrap();
                let mut admitted = devices();
                let placement = apply(&input, &options, &mut admitted, 0, 4096, 4, 256 << 10, None).unwrap();
                assert_eq!(placement.onboard_layers, cfg.dense.iter().filter(|&&dense| !dense).count());
                assert!(placement.pool_tokens >= 256 << 10);
                for (layer, assignment) in placement.layers.iter().enumerate().filter(|(layer, _)| !cfg.dense[*layer]) {
                    assert_eq!(assignment.experts, if ranks == 2 { ExpertHome::RtxTp2 }
                        else { ExpertHome::RtxWhole { gpu: 0 } }, "layer {layer}");
                }
                assert!(admitted.iter().all(|gpu| gpu.used_bytes() <= gpu.capacity_bytes));
                options.rtx_bytes = vec![1 * GIB; ranks];
                assert!(apply(&input, &options, &mut devices(), 0, 4096, 4, 256 << 10, None).is_err(),
                    "Mandatory experts or the agentic pool floor must not silently shrink away");
            }
        }
    }

    #[test]
    fn native_mtp_charges_weights_rings_and_load_owners() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = mimo_flash_config();
        let mut tensors = mimo_flash_tensors();
        let mut mtp = tensors.iter().filter(|(name, ..)| name.starts_with("model.layers.0."))
            .cloned().map(|(name, dtype, shape)| (name.replace("model.layers.0.", "model.mtp.layers.0.")
                .replace("post_attention_layernorm", "pre_mlp_layernorm"), dtype, shape)).collect::<Vec<_>>();
        // MTP uses sliding attention's KV head count and sink, with a dense FFN.
        mtp.retain(|(name, ..)| !name.contains(".self_attn.k_proj.") && !name.contains(".self_attn.v_proj."));
        mtp.extend(tensors.iter().filter(|(name, ..)| name.starts_with("model.layers.1.self_attn.k_proj.")
            || name.starts_with("model.layers.1.self_attn.v_proj.") || name.ends_with("attention_sink_bias"))
            .cloned().map(|(name, dtype, shape)| (name.replace("model.layers.1.", "model.mtp.layers.0."), dtype, shape)));
        mtp.push(t("model.mtp.layers.0.eh_proj.weight", "BF16", &[4096, 8192]));
        for norm in ["enorm", "hnorm", "final_layernorm"] {
            mtp.push(t(format!("model.mtp.layers.0.{norm}.weight"), "BF16", &[4096]));
        }
        tensors.extend(mtp);
        write_snapshot(dir.path(), &cfg, &tensors, Some(1));
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let mut options = LayoutOptions { native_mtp_layers: 0, mimo_drafter_disabled: true,
            host_memory: Some((128 << 30, 128 << 30)), ..Default::default() };
        let target = profile(&checkpoint, &options, 1, 4, 2752, 4, 32768, None).unwrap();
        options.native_mtp_layers = 1;
        let native = profile(&checkpoint, &options, 1, 4, 2752, 4, 32768, None).unwrap();
        assert!(native.resident_weights[0].0 > target.resident_weights[0].0);
        assert!(native.loading[0].reservations.iter().any(|owner| owner.name == "loading.mtp_state_before_target_kv"));
        let state = |profile: &MimoCapacityProfiles| profile.steady.devices[0].reservations.iter()
            .filter(|owner| owner.name == "state.active_rings").map(|owner| owner.bytes).sum::<u64>();
        assert!(state(&native) > state(&target));
    }

    #[test]
    fn flash_min_baseline_and_runtime_exclude_observer() {
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &mimo_flash_mopd_config(), &mimo_flash_mopd_tensors(), Some(4));
        let report = crate::plan::plan(dir.path(), &crate::plan::PlanOptions {
            placement: crate::plan::ExpertPlacement::Sparks { ranks: 2 },
            layout: Some(LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), ..Default::default() }),
            ..Default::default() }).unwrap();
        let ledger = report.phase_ledger.unwrap();
        let baseline = &ledger.budgets.iter().find(|b| b.id == "rtx0").unwrap().baseline;
        let crate::placement::planner::Baseline::Forecast { occupied, .. } = baseline else { panic!("forecast") };
        assert_eq!(*occupied, 575 << 20);
        let ready = ledger.phases.iter().find(|row| row.budget == "rtx0"
            && row.phase == crate::placement::planner::Phase::Ready).unwrap();
        assert_eq!(ready.opaque, 570 << 20);
        let measured_baseline = 1_179_189_248u64 - 576_716_800;
        let measured_owned = 97_985_769_216u64;
        let corrected_peak = 99_185_393_664u64;
        let measured_runtime = corrected_peak - measured_owned - measured_baseline;
        assert_eq!(measured_baseline, 602_472_448);
        assert_eq!(measured_runtime, 597_152_000);
        assert_eq!(*occupied, measured_baseline.div_ceil(1 << 20) * (1 << 20));
        assert_eq!(ready.opaque, measured_runtime.div_ceil(1 << 20) * (1 << 20));
        assert_eq!(measured_owned + occupied + ready.opaque - corrected_peak, 995_072);
    }

    #[test]
    fn post_ready_runtime_provenance_preserves_every_rank_allowance() {
        use cuteafd_core::memory_layout::{Basis, DeviceKind};
        for pro in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let config = if pro { mimo_pro_config() } else { mimo_flash_mopd_config() };
            let tensors = if pro { mimo_pro_tensors() } else { mimo_flash_mopd_tensors() };
            write_snapshot(dir.path(), &config, &tensors, Some(if pro { 8 } else { 4 }));
            for (gpus, sparks) in [(1, 2), (1, 4), (2, 2), (2, 4)] {
                let report = crate::plan::plan(dir.path(), &crate::plan::PlanOptions {
                    placement: crate::plan::ExpertPlacement::Sparks { ranks: sparks },
                    layout: Some(LayoutOptions {
                        rtx_bytes: vec![95 * GIB; gpus], head_split: gpus == 2,
                        host_memory: Some((128 << 30, 128 << 30)), ..Default::default()
                    }),
                    ..Default::default()
                }).unwrap();
                let layout = report.memory_layout.unwrap();
                let devices = layout.devices.iter().filter(|device| device.kind == DeviceKind::Rtx)
                    .collect::<Vec<_>>();
                assert_eq!(devices.len(), gpus);
                let measured = !pro && gpus == 1 && sparks == 2;
                for device in devices {
                    let item = device.items.iter().find(|item|
                        item.group.contains("MiMo post-ready runtime growth")).unwrap();
                    assert_eq!(item.bytes, 570 << 20);
                    assert_eq!(item.basis, if measured { Basis::Calibrated } else { Basis::Estimated });
                    assert!(item.group.starts_with(if measured { "measured " } else { "estimated " }));
                }
            }
        }
    }

    #[test]
    fn backend_slots_and_scope_categories_match_allocators() {
        assert_eq!(peer_slots(prefill_shape(6, None).unwrap().waves as usize), 16);
        assert_eq!(peer_slots(prefill_shape(6, None).unwrap().waves as usize), 16);
        assert_eq!(reservation_category("prefill.step_inputs"), Category::Workspace);
        assert_eq!(reservation_category("state.prefill_kv_wide"), Category::Workspace);
        assert_eq!(reservation_category("prefill.scratch"), Category::Workspace);
        assert_eq!(reservation_category("decode.page_table"), Category::Workspace);
        assert_eq!(reservation_category("draft.fp8_scratch"), Category::Workspace);
    }
}
