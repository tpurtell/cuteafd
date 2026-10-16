//! MiMo's planner consumes the same allocation descriptions as runtime admission.
use super::*;
use crate::families::mimo_v2::{admission::*, capacity::*, resident::MimoResidentOptions,
    projection::MimoProjectionRepresentation as R, weight_policy::*, *};
use anyhow::{Context, Result};

pub(crate) fn layout(report: &mut PlanReport, model: &dyn super::super::FamilyModel,
    input: &super::super::PlanInput, options: &LayoutOptions) -> MemoryLayout {
    let shape = Shape::resolve(report, model, options,
        ShapePolicy { context: ContextPolicy::Checkpoint, ..ShapePolicy::manifest_bounded() });
    let (rows, concurrency, context, manifest) = (shape.prefill_rows, shape.concurrency, shape.context_tokens,
        shape.manifest.as_ref());
    let checkpoint = input.checkpoint();
    let ranks = if options.head_split { options.rtx_bytes.len().clamp(1, 2) } else { 1 };
    let spark_ranks = match report.placement { ExpertPlacement::Sparks { ranks } => ranks, _ => 0 };
    let mut notes = vec![format!("resolved context {context} tokens"), "MiMo attention placement heads".into()];
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
        let runtime = match crate::placement::loaded_code(base, "*", ranks == 2, rank as u8) {
            Some(code) => Item::new(Category::Runtime, "context+modules", "",
                crate::placement::ArchContext::coordinator(total, None).context_bytes + code.bytes, Basis::Calibrated),
            None => Item::new(Category::Runtime, "context+modules", "", planned_context(manifest, &families, total)
                + if ranks == 2 { crate::placement::ArchContext::for_device("sm_120", total).peer_context_bytes } else { 0 }, Basis::Formula),
        };
        DeviceLayout { kind: DeviceKind::Rtx, index: rank as u32, capacity_bytes: total,
            items: vec![runtime], kv_tokens: 0 }
    }).collect::<Vec<_>>();
    let mut sparks = (0..spark_ranks).map(|rank| DeviceLayout { kind: DeviceKind::Spark,
        index: rank as u32, capacity_bytes: options.spark_bytes, kv_tokens: 0,
        items: vec![Item::new(Category::Workspace, "expert waves", "", gib(51) * options.spark_capacity_rows / 4096, Basis::Calibrated),
            Item::new(Category::Transport, "rdma rings", "", gib(117), Basis::Calibrated),
            Item::new(Category::Runtime, "context+modules", "", 512 * MIB, Basis::Calibrated)] }).collect::<Vec<_>>();
    // Media sees fixed target/pool demands and mandatory experts, not the
    // optional Auto prefix. Spark-free admission must reserve every expert.
    if report.encoder.is_none() {
        let towers = mimo_towers(report, checkpoint, model.spec().hidden as u64);
        let mut media_options = options.clone();
        if spark_ranks > 0 && options.onboard.unwrap_or_else(|| options.local_expert_layers
            .map(crate::placement::Onboard::Layers).unwrap_or(crate::placement::Onboard::Auto))
            == crate::placement::Onboard::Auto {
            media_options.onboard = Some(crate::placement::Onboard::Layers(0));
        }
        // Fixed zero onboarding must not grow an automatic pool to all free
        // memory: cap it at the target, but let it shrink to fit like the main
        // solve. An explicit POOL_TOKENS stays strict.
        media_options.pool_ceiling_at_target = true;
        let mut admission = devices.clone();
        match apply(input, &media_options, &mut admission, spark_ranks, rows, concurrency, context, manifest) {
            Ok(placement) => {
                if let Err(error) = spark_experts(input, options, &mut sparks, placement.onboard_layers) {
                    report.placement_supported = false; notes.push(format!("MiMo Spark slices: {error:#}"));
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
                report.placement_supported = false;
                notes.push(format!("MiMo media admission: {error:#}"));
                // The refusal text is built from hints, so carry the actual note.
                report.hints.push(super::super::Hint { what: format!("MiMo media admission: {error:#}"),
                    how: "Select a fitting Spark layout or a smaller explicit pool.".into() });
                // A mandatory target failure must not leave media unresolved or
                // advertise its unallocated RTX headroom as available.
                for device in &mut admission {
                    device.items.push(Item::new(Category::Reserved, "unavailable target admission", "",
                        device.capacity_bytes, Basis::Formula));
                }
                let _ = spark_experts(input, options, &mut sparks, 0);
                resolve_encoder(report, &towers, &mut admission, &mut sparks,
                    &vec![0; ranks], options, &mut notes);
                for spark in &mut sparks { spark.items.retain(|item| item.group != "routed_expert"); }
            }
        }
    }
    let mut pool_tokens = 0;
    let mut waste = Vec::new();
    match apply(input, options, &mut devices, spark_ranks, rows, concurrency, context, manifest) {
        Ok(placement) => {
            waste = replicated_weights(&placement);
            report.solver_placement = Some(placement.clone());
            pool_tokens = placement.pool_tokens;
            notes.push(format!("placement: {}", placement.summary()));
            if let Some(tp2) = placement.tp2 {
                notes.push(format!("rtx0/rtx1: {} TP2 expert layer halves ({}..{})", tp2.layers, tp2.first, tp2.first + tp2.layers));
            }
            if spark_ranks > 0 {
                let expert_bytes = spark_experts(input, options, &mut sparks, placement.onboard_layers);
                if let Err(error) = expert_bytes { report.placement_supported = false; notes.push(format!("MiMo Spark slices: {error:#}")); }
            }
        }
        Err(error) => {
            report.placement_supported = false;
            notes.push(format!("MiMo placement: {error:#}"));
            report.hints.push(super::super::Hint { what: format!("MiMo placement: {error:#}"),
                how: "Supply matching generated BF16-input expert and workspace manifests, or select a fitting Spark layout/pool.".into() });
        }
    }
    if options.mimo_prefix_draft && checkpoint.snapshot.join("dflash/config.json").is_file() {
        let slots = draft_representation::mimo_draft_context_slots(options.draft_sequences.max(concurrency),
            options.mimo_rings.max(concurrency), options.draft_context_slots);
        notes.push(format!("MiMo DFlash context slots {slots}"));
    }
    if manifest.is_none() { notes.push("MiMo native scratch needs --workspace-manifest; native scratch is not qualified without it".into()); }
    devices.extend(sparks);
    MemoryLayout { devices, pool_tokens, waste, notes }
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
    let remote = cfg.dense.iter().filter(|&&dense| !dense).count().checked_sub(onboard)
        .context("MiMo onboard exceeds routed layers")?;
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
        let bytes = crate::placement::families::mimo_v2::expert_layer_bytes(tensors,
            ranks, spark.index as usize, slicing)?
            .checked_mul(remote as u64).context("MiMo Spark expert bytes overflow")?;
        let format = match slicing { crate::formats::fp8_experts::Slicing::Padded => "native padded blocks",
            crate::formats::fp8_experts::Slicing::Blocks(_) => "native exact blocks" };
        spark.items.push(Item::new(Category::Experts, "routed_expert", format, bytes, Basis::Exact));
    }
    Ok(())
}

fn use_host_embedding(checkpoint: &super::super::Checkpoint, options: &LayoutOptions) -> bool {
    (options.host_embedding || (!options.force_gpu_embedding && options.rtx_bytes[0] <= 32 * GIB))
        && checkpoint.require_untied_embedding("model.embed_tokens.weight").is_ok()
}

pub(super) fn profile(checkpoint: &super::super::Checkpoint, options: &LayoutOptions,
    ranks: usize, spark_ranks: usize, rows: u64, concurrency: u64, context: u64,
    manifest: Option<&serde_json::Value>) -> Result<MimoCapacityProfiles> {
    let cfg = MimoV2Config::from_hf(&checkpoint.config)?;
    let qualified = default_policy(checkpoint, &cfg) == MimoDefaultPolicy::Fp8;
    let source = |name: &str| checkpoint.tensors.iter().find(|t| t.meta.name == name)
        .map(|t| if t.meta.dtype == cuteafd_core::DType::Bf16 && !qualified { R::Bf16 } else { R::Fp8 });
    let output_formats: std::collections::BTreeMap<_, _> = (0..cfg.layers).filter_map(|layer| {
        let name = format!("model.layers.{layer}.self_attn.o_proj.weight");
        source(&name).map(|r| (name, r))
    }).collect();
    let fp8_output = output_formats.values().any(|&r| r == R::Fp8);
    let lanes = resolve_transport_lanes(spark_ranks > 0, cfg.program_family()?,
        if options.prefill_lanes > 0 { Some(options.prefill_lanes.to_string()) }
        else { std::env::var("CUTEAFD_MIMO_PREFILL_LANES").ok() }.as_deref())?;
    let rings = options.mimo_rings.max(concurrency);
    let cache = crate::serving_capacity::mimo_cache_geometry(&cfg, cfg.layers, ranks, MimoKvCache::Int8, 0)?;
    let dir = checkpoint.snapshot.join("dflash");
    let draft = if dir.join("dflash_draft_model.safetensors").is_file() {
        Some(draft_config::DflashConfig::read(&dir)?)
    } else { None };
    let draft_mark = if options.mimo_prefix_draft {
        if let Some(draft) = &draft { draft_representation::mimo_draft_mark_bytes(draft.layers as u64, draft.kv_width() as u64)? }
        else if dir.join("config.json").is_file() {
            let config = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
            super::mimo_draft_prefix_bytes(&config, 1, rings).map_err(anyhow::Error::msg)?.0
        } else { 0 }
    } else { 0 };
    let mark = cache.ranks.iter().map(|r| r.retained_mark_bytes).sum::<u64>() + draft_mark;
    let marks = options.prefix_slots.unwrap_or_else(|| cuteafd_core::prefix::mark_slots_for(
        concurrency, options.mimo_prefix_entries, mark, options.mimo_prefix_mark_bytes));
    let graphs = decode_graph::MimoDecodeGraphPlan::new(cfg.layers, ranks, 64, options.mimo_decode_graphs)?;
    let output = if options.full_prefill_logits { MimoPrefillOutput::AllRows } else { MimoPrefillOutput::LastRow };
    let mut runtime = Vec::new();
    for rank in 0..ranks {
        let mut additional = Vec::new();
        additional.extend(transport_reservations(&cfg, ranks, rank, rows as usize, lanes, spark_ranks)?);
        if rank == 0 {
            additional.extend(sampling_reservations(cfg.vocab_size)?);
            if let Some(draft) = &draft {
                let sequences = options.draft_sequences.max(concurrency);
                let slots = draft_representation::mimo_draft_context_slots(sequences, rings, options.draft_context_slots);
                let capacity = draft_representation::MimoDraftCapacity::new(slots as usize, sequences as usize, draft.block)?;
                let headers = crate::read_safetensors_metadata(&dir.join("dflash_draft_model.safetensors"))?;
                let costs = draft_reservations_with(draft, &headers,
                    if options.mimo_draft_bf16 { draft_representation::MimoDraftRepresentation::Bf16Only }
                        else { draft_representation::MimoDraftRepresentation::Fp8Only }, capacity,
                    |shape| draft_scratch_bytes(shape, options.physical_sms.unwrap_or_else(||
                        crate::placement::inventory::ArchContext::for_device("sm_120", options.rtx_bytes[0]).sms) as usize))?;
                additional.extend(costs.steady);
                if prefill_lane_taps(lanes, rows as usize, 0, output) {
                    additional.push(cuteafd_core::serving_capacity::MemoryReservation {
                        name: "draft.prefill_first_lane_taps".into(), bytes: draft.prefill_lane_tap_bytes()? as u64 });
                }
            }
            additional.extend(draft_prefix_reservations(options.mimo_prefix_draft, draft_mark, marks, rings)?);
        }
        let workspaces = workspace_options(&cfg, ranks, rank, rows as usize, context as usize,
            lanes, spark_ranks > 0, options.mimo_expert_input, MimoKvCache::Int8, 0,
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
        runtime.push(MimoRankRuntime { device: rank as u32, workspaces,
            loading_additional: Vec::new(), additional });
    }
    let host_embedding = use_host_embedding(checkpoint, options);
    Ok(mimo_capacity_profiles(checkpoint, &cfg, MimoResidentOptions {
        layers: cfg.layers, coordinator_ranks: ranks, checkpoint_tp: checkpoint_tp(&checkpoint.snapshot)?,
        native_mtp_layers: 0, gpu_embedding: !host_embedding, head_format: source("lm_head.weight").unwrap_or(R::Bf16), output_formats,
    }, &MimoCapacityOptions { checkpoint_max_context_tokens:
        crate::serving_capacity::checkpoint_context_limit(&checkpoint.config)?
            .context("MiMo checkpoint lacks max_position_embeddings")?,
        max_context_tokens: context, rings, mark_slots: marks, kv_cache: MimoKvCache::Int8,
        ranks: runtime, host_prefix_bytes: 0 })?)
}

fn expert_costs(input: &super::super::PlanInput, cfg: &MimoV2Config,
    options: &LayoutOptions, ranks: usize, spark_ranks: usize, rows: u64)
    -> Result<(Vec<Option<crate::placement::ExpertCost>>, u64, [u64; 2], crate::placement::Onboard)> {
    use crate::placement::{Bytes2, ExpertCost, Onboard};
    use crate::placement::families::mimo_v2::{expert_layer_bytes, exact_package, package_directory, package_manifest};
    use crate::formats::fp8_experts::Slicing;
    let onboard = options.onboard.unwrap_or_else(|| options.local_expert_layers
        .map(Onboard::Layers).unwrap_or(Onboard::Auto));
    let catalog = input.expert_catalog()?;
    let tensors = catalog.fp8().context("MiMo needs native FP8/MXFP4/NVFP4 expert headers")?;
    let lib = crate::placement::inventory::image_lib(options.workspace_manifest.as_deref());
    let directory = package_directory(&lib, cfg, tensors.format(), ranks)?;
    let explicit = options.mimo_expert_manifest.as_deref();
    let path = explicit.map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| directory.parent().expect("package parent").join("manifest.json"));
    let routed = cfg.dense.iter().filter(|&&dense| !dense).count();
    let metadata = if onboard.layers(routed) == Some(0) { None }
        else { package_manifest(&directory, explicit)? };
    let mut half = [Bytes2::default(); 2];
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
                half[rank] = Bytes2 { resident: expert_layer_bytes(tensors, ranks, rank, slicing)?, staging: 0 };
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
    let whole = Bytes2 { resident: expert_layer_bytes(tensors, 1, 0, Slicing::Padded)?, staging: 0 };
    let costs = cfg.dense.iter().map(|&dense| (!dense).then_some(ExpertCost {
        whole, half, tp2: ranks == 2, spark_ok: spark_ranks > 0,
    })).collect();
    Ok((costs, workspace, tp2_workspace, onboard))
}

pub(super) fn apply(input: &super::super::PlanInput, options: &LayoutOptions,
    devices: &mut [DeviceLayout], spark_ranks: usize, rows: u64, concurrency: u64,
    context: u64, manifest: Option<&serde_json::Value>) -> Result<crate::placement::Placement> {
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
    let (experts, expert_workspace, tp2_workspace, onboard) = expert_costs(input, &cfg, options, ranks, spark_ranks, rows)?;
    let lanes = resolve_transport_lanes(spark_ranks > 0, cfg.program_family()?,
        if options.prefill_lanes > 0 { Some(options.prefill_lanes.to_string()) }
        else { std::env::var("CUTEAFD_MIMO_PREFILL_LANES").ok() }.as_deref())?;
    let inputs = crate::placement::families::mimo_v2::MimoInputs {
        cfg: &cfg, profile: &profiles.steady, layers: cfg.layers, kv: MimoKvCache::Int8, mtp_stages: 0,
        gpus: devices.iter().enumerate().map(|(rank, device)| (options.rtx_bytes[rank],
            crate::placement::Baseline::Planned { context_bytes: 0,
                loaded_bytes: device.used_bytes() })).collect(),
        headroom_floor: options.headroom_bytes, spark_ranks, startup_probe_bytes: probe,
        context, requested_pool: options.pool_tokens,
        pool_ceiling_at_target: options.pool_ceiling_at_target, onboard,
        experts, expert_workspace, tp2_workspace, rows, lanes: lanes as u64,
    };
    let solution = match crate::placement::families::mimo_v2::solve_working_set(&inputs, &no_spark.steady) {
        Ok(solution) => solution,
        Err(error) => {
            // Failed plans still name the full fixed contract, not fictitious
            // free space left by returning before its storage was described.
            for (rank, (device, costs)) in devices.iter_mut().zip(&profiles.steady.devices).enumerate() {
                device.capacity_bytes = options.rtx_bytes[rank].saturating_sub(headroom_bytes(
                    options.rtx_bytes[rank], rank == 0 && probe > 0).max(options.headroom_bytes));
                for cost in &costs.reservations {
                    device.items.push(Item::new(reservation_category(&cost.name),
                        crate::placement::families::mimo_v2::reservation_group(&cost.name), "", cost.bytes, Basis::Formula));
                }
            }
            return Err(error.into());
        }
    };
    let selected = inputs.selected(&no_spark.steady, &solution.working_set);
    let placement = solution.placement;
    for (rank, (device, costs)) in devices.iter_mut().zip(&selected.profile.devices).enumerate() {
        device.capacity_bytes = options.rtx_bytes[rank].saturating_sub(solution.request.inventory.gpus[rank].headroom_bytes);
        for cost in &costs.reservations {
            device.items.push(Item::new(reservation_category(&cost.name),
                crate::placement::families::mimo_v2::reservation_group(&cost.name), "", cost.bytes, Basis::Formula));
        }
    }
    let unit = selected.profile.pool_unit_rows;
    let cache = crate::serving_capacity::mimo_cache_geometry(&cfg, cfg.layers, ranks, MimoKvCache::Int8, 0)?;
    for ((device, rank), storage) in devices.iter_mut().zip(&selected.profile.devices).zip(&cache.ranks) {
        let units = placement.pool_tokens / unit;
        let kv_unit = storage.persistent_unit_bytes + storage.pool_metadata_unit_bytes;
        device.items.push(Item::new(Category::Kv, "records", "", units * kv_unit, Basis::Formula));
        device.items.push(Item::new(Category::Workspace, "pool page tables", "", units * (rank.pool_unit_bytes - kv_unit), Basis::Formula));
        device.kv_tokens = placement.pool_tokens;
    }
    for (rank, device) in devices.iter_mut().enumerate() {
        device.items.extend(crate::placement::families::mimo_v2::expert_items(&selected, &placement, rank)?);
    }
    Ok(placement)
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
            for ranks in [1, 2] {
                for (graphs, expert_input) in [false, true].into_iter().flat_map(|graphs|
                    [MimoExpertInput::Fp8, MimoExpertInput::Bf16, MimoExpertInput::Bf16Decode]
                        .map(|input| (graphs, input))) {
                    let opts = LayoutOptions { rtx_bytes: vec![95 * GIB; ranks], concurrency: 4,
                        pool_tokens: Some(32768), mimo_decode_graphs: graphs,
                        mimo_expert_input: expert_input, ..Default::default() };
                    let planned = profile(&checkpoint, &opts, ranks, 4, 4096, 4, 32768, None).unwrap();
                    let lanes = resolve_transport_lanes(true, cfg.program_family().unwrap(), None).unwrap();
                    let cache = crate::serving_capacity::mimo_cache_geometry(&cfg, cfg.layers, ranks, MimoKvCache::Int8, 0).unwrap();
                    let mark_bytes = cache.ranks.iter().map(|r| r.retained_mark_bytes).sum();
                    let slots = cuteafd_core::prefix::mark_slots_for(4, opts.mimo_prefix_entries, mark_bytes, opts.mimo_prefix_mark_bytes);
                    let graph = decode_graph::MimoDecodeGraphPlan::new(cfg.layers, ranks, 64, graphs).unwrap();
                    let runtime = (0..ranks).map(|rank| {
                        let mut additional = transport_reservations(&cfg, ranks, rank, 4096, lanes, 4).unwrap();
                        if rank == 0 { additional.extend(sampling_reservations(cfg.vocab_size).unwrap()); }
                        additional.extend(graph.reservations(rank, decode_graph::MIMO_GRAPH_EXEC_BOUND_BYTES,
                            decode_graph::MIMO_GRAPH_DRIVER_MARGIN_BYTES).unwrap());
                        MimoRankRuntime { device: rank as u32, loading_additional: vec![], additional,
                            workspaces: workspace_options(&cfg, ranks, rank, 4096, 32768, lanes, true,
                                opts.mimo_expert_input, MimoKvCache::Int8, 0, MimoPrefillOutput::LastRow, pro, |_| Ok(0)).unwrap() }
                    }).collect();
                    let admitted = mimo_capacity_profiles(&checkpoint, &cfg, MimoResidentOptions {
                        layers: cfg.layers, coordinator_ranks: ranks, checkpoint_tp: if pro { 8 } else { 4 },
                        native_mtp_layers: 0, gpu_embedding: true, head_format: if pro { R::Fp8 } else { R::Bf16 },
                        output_formats: (0..cfg.layers).map(|l| (format!("model.layers.{l}.self_attn.o_proj.weight"),
                            if pro { R::Fp8 } else { R::Bf16 })).collect(),
                    }, &MimoCapacityOptions { checkpoint_max_context_tokens: 1 << 20, max_context_tokens: 32768,
                        rings: 16, mark_slots: slots, kv_cache: MimoKvCache::Int8, ranks: runtime, host_prefix_bytes: 0 }).unwrap();
                    let mut devices: Vec<_> = (0..ranks).map(|rank| DeviceLayout {
                        kind: DeviceKind::Rtx, index: rank as u32, capacity_bytes: 95 * GIB,
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
                            + (32768 / admitted.steady.pool_unit_rows) * admitted.steady.devices[rank].pool_unit_bytes;
                        assert_eq!(devices[rank].used_bytes(), total, "mapped planner items equal admission total");
                        assert_eq!(planned.steady.devices[rank].reservations, admitted.steady.devices[rank].reservations,
                            "pro={pro}, ranks={ranks}, graphs={graphs}, rank={rank}");
                        assert_eq!(planned.steady.devices[rank].pool_unit_bytes, admitted.steady.devices[rank].pool_unit_bytes);
                    }
                }
            }
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
            let opts = LayoutOptions { rtx_bytes: vec![total], headroom_bytes: 0, concurrency: 4, ..Default::default() };
            let mut profile = profile(&checkpoint, &opts, 1, 2, 4096, 4, 1 << 20, None).unwrap();
            let cublas = crate::placement::inventory::ArchContext::for_device("sm_120", total).cublas_bytes;
            profile.steady.devices[0].reservations.push(cuteafd_core::serving_capacity::MemoryReservation {
                name: "runtime.cublas_first_gemm".into(), bytes: cublas,
            });
            let policy = CapacityPolicy { concurrency: 4, small_card_headroom: true,
                target_pool_tokens: if total <= 32 * GIB { 1 << 20 } else { 2 << 20 },
                max_context_tokens: Some(1 << 20), ..Default::default() };
            let admitted = resolve_capacity_with_startup_peaks(policy, &profile.steady,
                &[DeviceMemory { device: 0, total_bytes: total, baseline_free_bytes: total - baseline }],
                &[(0, 64 << 20)]).unwrap();
            let mut devices = vec![DeviceLayout { kind: DeviceKind::Rtx, index: 0, capacity_bytes: total,
                items: vec![Item::new(Category::Runtime, "context+modules", "", baseline + cublas, Basis::Exact)], kv_tokens: 0 }];
            let pool = apply(&input, &opts, &mut devices, 2, 4096, 4, 1 << 20, None).unwrap().pool_tokens;
            assert_eq!(pool, admitted.allocated_gpu_kv_tokens, "recorded total {total}");
            assert_eq!(devices[0].capacity_bytes, total - headroom_bytes(total, true));
        }
    }

    fn shipped_mimo(pro: bool) -> (serde_json::Value, Vec<Tensor>) {
        let mut config = if pro { mimo_pro_config() } else { mimo_flash_mopd_config() };
        let layers = if pro { 70 } else { 48 };
        let full: &[usize] = if pro { &[0, 7, 15, 23, 31, 39, 47, 55, 62, 69] }
            else { &[0, 5, 11, 17, 23, 29, 35, 41, 47] };
        config["num_hidden_layers"] = serde_json::json!(layers);
        config["vocab_size"] = serde_json::json!(152576);
        config["hybrid_layer_pattern"] = serde_json::json!((0..layers)
            .map(|layer| usize::from(!full.contains(&layer))).collect::<Vec<_>>());
        config["moe_layer_freq"] = serde_json::json!((0..layers)
            .map(|layer| usize::from(layer > 0)).collect::<Vec<_>>());
        let template = if pro { mimo_pro_tensors() } else { mimo_flash_mopd_tensors() };
        let mut tensors = Vec::new();
        for (name, dtype, shape) in template.iter().filter(|(name, ..)| !name.starts_with("model.layers.")) {
            let mut shape = shape.clone();
            if name == "lm_head.weight" || name == "model.embed_tokens.weight" { shape[0] = 152576; }
            tensors.push((name.clone(), *dtype, shape));
        }
        for layer in 0..layers {
            for (name, dtype, shape) in template.iter().filter(|(name, ..)| name.starts_with("model.layers.")) {
                let source = name.split('.').nth(2).unwrap().parse::<usize>().unwrap();
                let is_mlp = name.contains(".mlp.");
                let selected = if is_mlp { usize::from(layer > 0) } else { usize::from(!full.contains(&layer)) };
                if source == selected {
                    tensors.push((name.replacen(&format!("model.layers.{source}."),
                        &format!("model.layers.{layer}."), 1), *dtype, shape.clone()));
                }
            }
        }
        (config, tensors)
    }

    #[test]
    fn planner_equals_runtime_mimo() {
        use crate::placement::{Baseline, ExpertHome, Onboard};
        use crate::placement::families::mimo_v2::{MimoInputs, solve_working_set, resolved_capacity};
        // Measured Pro min/max post-module inventories from the 2026-10-09
        // MOPD smoke logs. Reused physical inventory for Flash, not a Flash
        // ready measurement. Exported BF16 scratch is SM188/170-identical.
        for pro in [true, false] {
            let dir = tempfile::tempdir().unwrap();
            let (config, tensors) = shipped_mimo(pro);
            write_snapshot(dir.path(), &config, &tensors, Some(if pro { 8 } else { 4 }));
            let checkpoint = Checkpoint::open(dir.path()).unwrap();
            let input = PlanInput::from_checkpoint(checkpoint);
            let checkpoint = input.checkpoint();
            let cfg = MimoV2Config::from_hf(&config).unwrap();
            for ranks in [1, 2] {
                // Native CPU scratch sizing from the pinned SM120 package;
                // TP1 and TP2 retain different intermediate workspaces.
                let scratch = match (pro, ranks) {
                    (true, 1) => 805_573_632, (true, _) => 604_247_040,
                    (false, 1) => 671_354_880, (false, _) => 470_028_288,
                };
                let totals = [101_973_491_712, 101_970_345_984][..ranks].to_vec();
                let baseline = [586_416_128, 596_901_888];
                let package = dir.path().join(format!("tp{ranks}-manifest.json"));
                let layout = format!("tp{ranks}");
                std::fs::write(&package, serde_json::to_vec(&serde_json::json!({"layouts": {layout: {
                    "input": "bf16", "tp": ranks, "hidden": cfg.hidden, "experts": cfg.experts,
                    "top_k": cfg.topk, "intermediate": cfg.moe_intermediate,
                    "slice": cfg.moe_intermediate / ranks, "weights": "mxfp4",
                    "capacities": [{"capacity": 4096, "scratch_bytes": scratch}]
                }}})).unwrap()).unwrap();
                let context = 256 << 10;
                let sparks = if pro { 6 } else if ranks == 1 { 2 } else { 4 };
                let opts = LayoutOptions { rtx_bytes: totals.clone(), head_split: ranks == 2,
                    pool_tokens: Some(2 << 20), concurrency: 16, mimo_expert_manifest: Some(package),
                    ..Default::default() };
                let costs = profile(&checkpoint, &opts, ranks, sparks, 4096, 16, context, None).unwrap();
                let (experts, expert_workspace, tp2_workspace, onboard) =
                    expert_costs(&input, &cfg, &opts, ranks, sparks, 4096).unwrap();
                let lanes = resolve_transport_lanes(true, cfg.program_family().unwrap(), None).unwrap();
                let inputs = MimoInputs { cfg: &cfg, profile: &costs.steady, layers: cfg.layers,
                    kv: MimoKvCache::Int8, mtp_stages: 0, gpus: totals.iter().enumerate().map(|(rank, &total)|
                        (total, Baseline::Measured { free_bytes: total - baseline[rank] })).collect(),
                    headroom_floor: opts.headroom_bytes, spark_ranks: sparks, startup_probe_bytes: 64 << 20,
                    context, requested_pool: opts.pool_tokens, pool_ceiling_at_target: false, onboard, experts, expert_workspace,
                    tp2_workspace, rows: 4096, lanes: lanes as u64 };
                let no_spark = profile(&checkpoint, &opts, ranks, 0, 4096, 16, context, None).unwrap();
                let solution = solve_working_set(&inputs, &no_spark.steady).unwrap();
                let selected = inputs.selected(&no_spark.steady, &solution.working_set);
                let runtime_request = &solution.request;
                let runtime = &solution.placement;
                assert_eq!(solution.working_set.spark_ranks, sparks);
                assert_eq!(solution.working_set.spark_layers,
                    runtime.layers.iter().enumerate().filter_map(|(layer, assignment)|
                        (assignment.experts == ExpertHome::Spark).then_some(layer)).collect::<Vec<_>>());
                if ranks == 1 {
                    let cost = runtime_request.layers.iter().find_map(|layer| layer.experts).unwrap();
                    assert!(cost.whole.resident > 0 && !cost.tp2, "single RTX offers whole TP1 experts");
                    let fixed = costs.steady.devices[0].reservations.iter().map(|r| r.bytes).sum::<u64>();
                    let pool = (runtime.pool_tokens / costs.steady.pool_unit_rows) * costs.steady.devices[0].pool_unit_bytes;
                    let remaining = runtime_request.inventory.gpus[0].available() - fixed - pool;
                    println!("mimo pro={pro} TP1 offered: whole={} workspace={} remaining={remaining}",
                        cost.whole.resident, runtime_request.expert_workspace);
                    if pro {
                        assert!(remaining < cost.whole.resident, "Pro min cannot fit one TP1 layer even with zero scratch");
                    }
                }
                let capacity = resolved_capacity(&selected, runtime_request, runtime, 16, 16).unwrap();
                let mut devices = totals.iter().enumerate().map(|(rank, &total)| DeviceLayout {
                    kind: DeviceKind::Rtx, index: rank as u32, capacity_bytes: total, kv_tokens: 0,
                    items: vec![Item::new(Category::Runtime, "context+modules", "measured post-module",
                        baseline[rank], Basis::Calibrated)],
                }).collect::<Vec<_>>();
                let planned = apply(&input, &opts, &mut devices, sparks, 4096, 16, context, None).unwrap();
                assert_eq!(&planned, runtime, "pro={pro}, ranks={ranks}");
                assert_eq!(planned.pool_tokens, 2 << 20);
                assert_eq!(capacity.active_max_context_sequences, 8);
                for rank in 0..ranks {
                    let admitted = &capacity.devices[rank];
                    assert_eq!(devices[rank].used_bytes(), baseline[rank] + admitted.reserved_bytes + admitted.pool_bytes);
                    let categories = [Category::Embedding, Category::Weights, Category::Drafter, Category::Experts,
                        Category::Workspace, Category::Transport, Category::Prefix, Category::Kv, Category::Runtime];
                    for category in categories {
                        let planned = devices[rank].items.iter().filter(|item| item.category == category)
                            .map(|item| item.bytes).sum::<u64>();
                        let runtime = admitted.reservations.iter().filter(|item|
                            reservation_category(&item.name) == category).map(|item| item.bytes).sum::<u64>();
                        let pool = devices[rank].items.iter().filter(|item| item.category == category
                            && (item.group == "records" || item.group == "pool page tables"))
                            .map(|item| item.bytes).sum::<u64>();
                        assert_eq!(planned, runtime + pool + if category == Category::Runtime { baseline[rank] } else { 0 });
                    }
                    println!("mimo pro={pro} ranks={ranks} rank={rank}: pool={} experts={} reserved={} pool_bytes={} baseline={}",
                        planned.pool_tokens, planned.onboard_layers, admitted.reserved_bytes, admitted.pool_bytes, baseline[rank]);
                }
                if ranks == 2 {
                    assert!(runtime.onboard_layers > 0);
                    assert!(runtime.expert_ranges.iter().all(|range| range.layers == 0));
                    assert!(runtime.layers.iter().all(|layer| !matches!(layer.experts, ExpertHome::RtxWhole { .. })));
                }
                // Auto with no package must retain the automatic pool bound,
                // and cannot charge routes/arena that runtime never allocates.
                let fallback = MimoInputs { requested_pool: None, onboard: Onboard::Auto,
                    expert_workspace: 0, tp2_workspace: [0; 2], ..inputs };
                let fallback_solution = solve_working_set(&fallback, &no_spark.steady).unwrap();
                let fallback_selected = fallback.selected(&no_spark.steady, &fallback_solution.working_set);
                assert_eq!(fallback_solution.request.pool.ceiling, 2 << 20);
                assert_eq!(fallback_solution.placement.pool_tokens, 2 << 20);
                assert_eq!(fallback_solution.placement.onboard_layers, 0);
                assert!(crate::placement::families::mimo_v2::expert_items(&fallback_selected,
                    &fallback_solution.placement, 0).unwrap().is_empty());
            }
        }
    }

    #[test]
    fn spark_residency_uses_the_worker_package_slice_policy() {
        use crate::formats::fp8_experts::Slicing;
        use crate::placement::families::mimo_v2::expert_layer_bytes;
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &mimo_pro_config(), &mimo_pro_tensors(), Some(8));
        let checkpoint = Checkpoint::open(dir.path()).unwrap();
        let input = PlanInput::from_checkpoint(checkpoint);
        let checkpoint = input.checkpoint();
        let catalog = crate::read_expert_catalog(&checkpoint.snapshot).unwrap();
        let tensors = catalog.fp8().unwrap();
        let options = LayoutOptions { workspace_manifest: Some(dir.path().join("share/PROGRAMS.json")),
            ..Default::default() };
        assert!(!crate::placement::inventory::image_lib(options.workspace_manifest.as_deref()).exists());
        let recipe = super::super::super::experts::spark_recipe("mimop:fp8", 2048, 6).unwrap();
        assert_eq!(recipe.widths, [384, 384, 384, 384, 256, 256]);
        assert_eq!(recipe.required_layouts, ["tp6-w384", "tp6-w256"]);
        let mut sparks = (0..6).map(|rank| DeviceLayout {
            kind: DeviceKind::Spark, index: rank, capacity_bytes: 108 * GIB,
            items: vec![], kv_tokens: 0,
        }).collect::<Vec<_>>();
        spark_experts(&input, &options, &mut sparks, 0).unwrap();
        for spark in &sparks {
            assert_eq!(spark.used_bytes(), expert_layer_bytes(tensors, 6, spark.index as usize,
                recipe.slicing).unwrap());
        }
        assert!(sparks[4].used_bytes() < sparks[0].used_bytes());
        assert_eq!(sparks.iter().map(DeviceLayout::used_bytes).sum::<u64>(),
            expert_layer_bytes(tensors, 1, 0, Slicing::Padded).unwrap());
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
                let mut options = LayoutOptions { rtx_bytes: vec![32 * GIB; ranks],
                    head_split: ranks == 2, concurrency: 4,
                    mimo_expert_manifest: Some(manifest.clone()), ..Default::default() };
                let devices = || (0..ranks).map(|rank| DeviceLayout {
                    kind: DeviceKind::Rtx, index: rank as u32, capacity_bytes: 32 * GIB,
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
    fn backend_slots_and_scope_categories_match_allocators() {
        assert_eq!(peer_slots(resolve_transport_lanes(false, "mimop", None).unwrap()), 8);
        assert_eq!(peer_slots(resolve_transport_lanes(true, "mimop", None).unwrap()), 12);
        assert_eq!(reservation_category("prefill.step_inputs"), Category::Workspace);
        assert_eq!(reservation_category("state.prefill_kv_wide"), Category::Workspace);
        assert_eq!(reservation_category("prefill_first_lane.scratch"), Category::Workspace);
        assert_eq!(reservation_category("decode.page_table"), Category::Workspace);
        assert_eq!(reservation_category("draft.fp8_scratch"), Category::Workspace);
    }
}
