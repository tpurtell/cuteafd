//! DeepSeek V4 (Flash/Pro) plan adapter: header-derived resident weights, then
//! the same `placement::families::deepseek_v4` request serve-dsv4 solves, with
//! each GPU's planned context, modules and embedding as the baseline.
use super::*;
use crate::families::deepseek_v4::{DeepseekV4Config, resident::resident_operands};
use crate::placement::{families::deepseek_v4 as v4, Baseline, FfnMode, LayerMode, Onboard};
use crate::placement::planner::Pin;

/// Observed Spark host footprint and non-I/O wave residual (P11 replaces both).
const SPARK: SparkCharges = SparkCharges { host_bytes: Some(13 * GIB), wave_bytes: gib(35), basis: Basis::Estimated };

/// V4's resident inventory shared with the runtime weight loader. All
/// non-embedding weights are explicit admission demands, so the solver sees
/// owner-dependent C4 costs; only the embedding is a baseline item.
fn resident_embedding(checkpoint: &Checkpoint, ranks: usize, dspark: bool) -> Option<Vec<Vec<(String, String, u64)>>> {
    let cfg = DeepseekV4Config::read(&checkpoint.snapshot, 0).ok()?;
    let modes = vec![if ranks == 1 { LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner } }
        else { LayerMode::HeadSplit }; cfg.n_layers];
    let operands = resident_operands(&cfg, &modes, ranks, dspark,
        checkpoint.tensors.iter().map(|t| &t.meta)).ok()?;
    let mut groups = vec![std::collections::BTreeMap::<(String, String), u64>::new(); ranks];
    for operand in operands.into_iter().filter(|operand| operand.group == "embedding") {
        *groups[operand.rank].entry((operand.group, operand.format)).or_default() += operand.bytes;
    }
    Some(groups.into_iter().map(|group| group.into_iter()
        .map(|((group, format), bytes)| (group, format, bytes)).collect()).collect())
}

use crate::plan::Checkpoint;

pub(crate) fn layout(report: &mut PlanReport, model: &dyn super::super::FamilyModel,
    input: &super::super::PlanInput, options: &mut LayoutOptions) -> MemoryLayout {
    let original_report = report.clone();
    let original_options = options.clone();
    let started = std::time::Instant::now();
    let memory = layout_for_experts(report, model, input, options, report.spark_ranks, false);
    tracing::info!(seconds = started.elapsed().as_secs_f64(), "V4 initial geometry admission");
    let initial_timing = report.admission_timing.clone();
    let all_local = report.solver_placement.as_ref().is_some_and(|placement|
        !placement.layers.iter().any(|layer| layer.experts == crate::placement::ExpertHome::Spark));
    if original_report.spark_ranks == 0 || !all_local { return memory; }

    // Workspace, host marks and exchange reserves depend on the executor shape.
    // Re-admit the entire local geometry, never just rewrite the published rows.
    let mut local_report = original_report.clone();
    let mut local_options = original_options.clone();
    let started = std::time::Instant::now();
    let local = layout_for_experts(&mut local_report, model, input, &mut local_options, 0, false);
    tracing::info!(seconds = started.elapsed().as_secs_f64(), "V4 local geometry re-admission");
    add_admission_timing(&mut local_report, initial_timing.as_ref());
    if local_report.placement_supported && local_report.solver_placement.is_some() {
        *report = local_report;
        *options = local_options;
        return local;
    }
    // If the local shape cannot fit (or conflicts with an explicit shape),
    // retain only candidates with real Spark ownership in the remote geometry.
    *report = original_report;
    *options = original_options;
    let remote = layout_for_experts(report, model, input, options, report.spark_ranks, true);
    add_admission_timing(report, local_report.admission_timing.as_ref());
    remote
}

fn add_admission_timing(report: &mut PlanReport, prior: Option<&crate::plan::AdmissionTiming>) {
    if let (Some(current), Some(prior)) = (&mut report.admission_timing, prior) {
        current.solve_us = current.solve_us.saturating_add(prior.solve_us);
        current.admissions = current.admissions.saturating_add(prior.admissions);
    }
}

fn layout_for_experts(report: &mut PlanReport, model: &dyn super::super::FamilyModel,
    input: &super::super::PlanInput, options: &mut LayoutOptions, expert_world: usize,
    require_remote: bool) -> MemoryLayout {
    let checkpoint = input.checkpoint();
    let prefill = match resolve_prefill(input, report, options, expert_world,
        crate::serving_capacity::prefill_waves::PrefillLimits {
            max_waves: if expert_world == 0 { 1 } else { crate::serving_capacity::prefill_waves::MAX_WAVES }, ..crate::plan::Family::prefill_limits(&crate::plan::families::deepseek::DEEPSEEK_V4)
        }) {
        Ok(shape) => shape,
        Err(error) => return unsupported_prefill(report, error),
    };
    let mut shape = Shape::resolve(input, report, model, options,
        ShapePolicy { all_native_stages: true, ..ShapePolicy::manifest_bounded() });
    shape.prefill_rows = prefill.unit_rows;
    report.prefill_span_rows = Some(prefill.span_rows);
    let gpus = options.rtx_bytes.len().clamp(1, 2);
    let split = gpus == 2 && options.head_split;
    let ranks = if split { 2 } else { 1 };
    let mut devices: Vec<DeviceLayout> = options.rtx_bytes.iter().take(gpus).enumerate()
        .map(|(index, &bytes)| DeviceLayout { kind: DeviceKind::Rtx, index: index as u32,
            capacity_bytes: bytes, worker_budget_bytes: 0, items: Vec::new(), kv_tokens: 0 }).collect();
    let mut waste = Vec::new();
    let mut notes = vec![format!("resolved context {} tokens (checkpoint {})", shape.context_tokens, shape.checkpoint_context),
        format!("attention placement {}", report.attention_placement)];
    shape.select_card_plans(report, options, &mut notes);
    // Retain estimated-allowance warnings pending census proof coverage: https://github.com/tpurtell/cuteafd/issues/388.
    notes.push("deepseek_v4: workspace/runtime/Spark allowances are unqualified; validate against the allocation ledger before using this layout for admission".into());
    mapped_table_note(report, &mut notes);
    let dspark = shape.native_layers > 0;
    if let Some(embedding) = resident_embedding(checkpoint, ranks, dspark) {
        push_resident(&mut devices, &embedding, |group| if group == "embedding" { Category::Embedding } else { Category::Weights });
    }
    if options.host_embedding {
        place_host_embedding(report, checkpoint, &mut devices, &mut waste, &mut notes);
    }
    // A GPU embedding is a view of rank 0's target arena, which admission
    // charges with the rest of the retained weights; it is not baseline.
    for device in &mut devices {
        device.items.retain(|item| item.category != Category::Embedding);
    }
    let exl3_workspace = expert_workspace(report, model, input, options.workspace_manifest.as_deref(), shape.prefill_rows);
    if exl3_workspace.is_none() && report.experts.as_ref().is_some_and(|e| e.package.contains("exl3")) {
        notes.push("Local EXL3 workspace allowance is estimated without matching rtx-tp1/m*/v41_exl3.json capacity manifests; images bundle them, or export the exl3 tree alongside PROGRAMS.json".into());
    }
    if options.drafter_bytes > 0 {
        devices[0].items.push(Item::new(Category::Drafter, "drafter", "", options.drafter_bytes, Basis::Estimated));
    }
    let id = if model.spec().hidden == 4096 { "dsv4f" } else { "dsv4p" };
    let workspace = (|| {
        let manifest = shape.manifest.as_deref()?;
        let cfg = DeepseekV4Config::read(&checkpoint.snapshot, shape.native_layers).ok()?;
        let scratch = crate::serving_capacity::deepseek_v4_workspace_scratch(manifest, id, split,
            shape.prefill_rows, shape.decode_rows).ok()?;
        crate::serving_capacity::deepseek_v4_workspace_geometry(&cfg, shape.prefill_rows, shape.decode_rows,
            crate::serving_capacity::compiled_c128_width(manifest, id).ok()?.checked_mul(128)?, ranks, options.prefill_waves, scratch).ok()
    })();
    if workspace.is_none() {
        notes.push("V4 workspace allowance is estimated without a matching PROGRAMS.json; use --workspace-manifest for Flash/Pro allocation geometry".into());
    }
    let mut spark_working = crate::placement::SparkWorkingSet {
        rtx_layers: Vec::new(),
        spark_layers: if matches!(report.placement, ExpertPlacement::Sparks { .. }) {
            let first = input.expert_catalog().map(|catalog| catalog.routed_experts().first_layer).unwrap_or(0);
            (first..model.spec().layers.len()).collect() } else { Vec::new() },
        spark_ranks: match report.placement { ExpertPlacement::Sparks { ranks } => ranks, ExpertPlacement::Local => 0 },
        prefill_lanes: 1,
    };
    let endpoints = spark_endpoints("deepseek_v4", 1);
    let mut sparks = spark_devices(report, model, input, options, SPARK, &spark_working, endpoints,
        &mut waste, &mut notes);
    let mut pool_tokens = 0;
    let mut local_layers = 0;
    let admitted = (|| -> anyhow::Result<_> {
        let (mut inputs, mut startup) = v4_inputs(report, input, options, &devices[..ranks],
            &shape, exl3_workspace, workspace)?;
        inputs.spark_ranks = expert_world;
        let prepare = || {
            let prepared = v4::prepare(&inputs)?;
            if require_remote { prepared.require_remote_experts() } else { Ok(prepared) }
        };
        let prepared = prepare()?;
        let reserves = prepared.reserves().to_vec();
        let request = v4::production_request(&inputs);
        let prefix = prepared.prefix();
        let admitted = prepared.admit(request.clone())?;
        let mut solve_us = admitted.solve_us;
        let mut admissions = 1u32;
        let mut solved = admitted.solved;
        let mut placement = admitted.placement;
        if report.encoder.is_none() {
            // A mandatory encoder displaces an optional expert prefix, not the pool:
            // price it against only compulsory experts, then resolve the tower.
            let mut encoder_request = request.clone();
            if inputs.spark_ranks > 0 && request.onboard == Pin::Auto {
                // Only compulsory owners: auto onboarding is not yet decided, so
                // the tower competes with no expert prefix at all.
                encoder_request.onboard = Pin::Exact(0);
                encoder_request.pool = Pin::Exact(placement.pool_tokens);
            }
            let encoder_admitted = prepare()?.admit(encoder_request)?;
            solve_us = solve_us.saturating_add(encoder_admitted.solve_us);
            admissions += 1;
            let encoder_budget = encoder_admitted.placement;
            let mut encoder_devices = devices[..ranks].to_vec();
            for ((device, reserve), items) in encoder_devices.iter_mut().zip(&reserves).zip(&encoder_budget.items) {
                device.capacity_bytes = device.capacity_bytes.saturating_sub(*reserve);
                device.items.extend(items.iter().cloned());
            }
            resolve_encoder(report, &Towers::checkpoint(report), &mut encoder_devices, &mut sparks,
                &vec![0; ranks], options, &mut notes);
        }
        if let Some(encoder) = &report.encoder {
            if let super::super::encoder::EncoderKind::Rtx { gpu } = encoder.kind {
                anyhow::ensure!(gpu < ranks, "V4 encoder owner is not an admitted coordinator GPU");
                let mut with_tower = inputs.clone();
                with_tower.extra.push(crate::placement::Demand::new(gpu as u8, Category::Weights,
                    "vision tower", encoder.weights, Basis::Formula)
                    .with_resident_weights(crate::placement::ResidentWeights(encoder.weights)));
                with_tower.extra.push(crate::placement::Demand::new(gpu as u8, Category::Workspace,
                    "vision scratch", encoder.scratch, Basis::Formula));
                // Mandatory media displaces optional experts, never the admitted pool.
                let mut tower_request = request.clone();
                tower_request.pool = Pin::Exact(placement.pool_tokens);
                let prepared = v4::prepare(&with_tower)?;
                let prepared = if require_remote { prepared.require_remote_experts()? } else { prepared };
                let admitted = prepared.admit(tower_request)?;
                solve_us = solve_us.saturating_add(admitted.solve_us);
                admissions += 1;
                solved = admitted.solved;
                placement = admitted.placement;
            }
        }
        startup.add_experts(input.expert_catalog()?, shape.native_layers, &placement, options.tp2_dspark_experts)?;
        Ok((placement, reserves, startup.notes()?, solved, prefix, solve_us, admissions))
    })();
    match admitted {
        Ok((placement, reserves, startup_notes, solved, prefix, solve_us, admissions)) => {
            report.admission_timing = Some(crate::plan::AdmissionTiming { solve_us, admissions });
            notes.extend(startup_notes);
            report.phase_ledger = Some(crate::plan::SolvedPhaseLedger::from(&solved));
            report.host_budget_bytes = prefix.owners.quota_bytes;
            notes.extend(prefix.notes(shape.context_tokens, shape.concurrency));
            report.solver_placement = Some(placement.clone());
            for (device, reserve) in devices.iter_mut().zip(&reserves) {
                device.capacity_bytes = options.rtx_bytes[device.index as usize].saturating_sub(*reserve);
            }
            // Admission charges RoPE tables for the full context; with a smaller pool serve then
            // allocates them for the context it serves (`pool_context`: the pool less one unit),
            // which is what the ready ledger holds.
            let context = shape.context_tokens;
            let served = if placement.pool_tokens < context && options.context_tokens == 0 {
                placement.pool_tokens.saturating_sub(256) } else { context };
            for (device, items) in devices.iter_mut().zip(&placement.items) {
                device.items.extend(items.iter().cloned().map(|mut item| {
                    if item.group == "RoPE context tables" && context > 0 {
                        item.bytes = item.bytes / context * served;
                    }
                    item
                }));
                if device.items.iter().any(|i| i.group == "records") { device.kv_tokens = placement.pool_tokens; }
            }
            for (gpu, range) in placement.expert_ranges.iter().enumerate() {
                notes.push(format!("rtx{gpu}: {} local expert layers ({}..{})", range.layers,
                    range.first, range.first + range.layers));
            }
            notes.push(format!("placement: {}", placement.summary()));
            // The solver resolved auto against this build's programs.
            report.attention_placement = placement.attention_placement;
            if let Some(note) = notes.iter_mut().find(|n| n.starts_with("attention placement ")) {
                *note = format!("attention placement {}", placement.attention_placement);
            }
            local_layers = placement.onboard_layers;
            spark_working.spark_layers = placement.layers.iter().enumerate().filter_map(|(layer, home)|
                (home.experts == crate::placement::ExpertHome::Spark).then_some(layer)).collect();
            spark_working.rtx_layers = placement.layers.iter().enumerate().filter_map(|(layer, home)|
                matches!(home.experts, crate::placement::ExpertHome::RtxTp2
                    | crate::placement::ExpertHome::RtxWhole { .. }).then_some(layer)).collect();
            if let Some(t) = placement.tp2 {
                notes.push(format!("rtx0/rtx1: {} TP2 expert layer halves ({}..{})", t.layers, t.first, t.first + t.layers));
            }
            pool_tokens = placement.pool_tokens;
            if pool_tokens < context {
                notes.push(format!("full-context admission shortfall: context {context} tokens, pool {pool_tokens} tokens, shortfall {} tokens", context - pool_tokens));
            }
            if ranks == 2 && placement.attention_placement == crate::placement::AttentionPlacement::Heads {
                waste.push(Waste { device: "rtx1".into(), what: "KV records replicated on both GPUs (MLA latent)".into(),
                    bytes: placement.items[1].iter().filter(|i| i.group == "records").map(|i| i.bytes).sum() });
            }
        }
        Err(error) => {
            let reason = format!("V4 pool-first placement: {error:#}");
            report.refuse_placement(reason.clone(),
                "Supply complete native checkpoint owners within the device and resident-weight budgets.");
            notes.push(reason);
        }
    }
    let towers = Towers::checkpoint(report);
    if report.encoder.is_none() {
        resolve_encoder(report, &towers, &mut devices, &mut sparks, &vec![0; gpus], options, &mut notes);
    }
    if local_layers > 0 { notes.push(format!("{local_layers} routed backbone layers resident on RTX")); }
    if let Ok(catalog) = input.expert_catalog() {
        if catalog.exl3().is_none() && catalog.nvfp4().is_none() {
            let e = catalog.routed_experts();
            if let Some(capacity) = [1, 16, 80, 256, 1024, 4096].into_iter()
                .find(|&c| u64::from(c) >= shape.prefill_rows) {
                if !spark_working.rtx_layers.is_empty() {
                    notes.push(cuteafd_core::mxfp4_layouts::native_prefill_note(e.hidden,
                        [e.intermediate / ranks], 120, capacity, e.experts, e.topk));
                }
                let spark_ranks = spark_working.spark_ranks;
                if spark_ranks > 0 && !spark_working.spark_layers.is_empty() {
                    let widths = cuteafd_core::tp_units::package_width_keys(e.intermediate, 32, spark_ranks, false, true, false);
                    notes.push(cuteafd_core::mxfp4_layouts::native_prefill_note(e.hidden,
                        widths.into_iter().map(|(_, width)| width), 121, capacity, e.experts, e.topk));
                }
            }
        }
    }
    // Reapply after admission: rank-local I/O never belongs to the coordinator KV budget.
    charge_spark_devices(report, model, input, options, SPARK, &spark_working, endpoints, &mut sparks,
        &mut notes, |device| format!("{} I/O admission", device.name()));
    devices.extend(sparks);
    MemoryLayout { devices, pool_tokens, waste, notes }
}

/// V4's planned admission inputs: the same [`v4::V4Inputs`] serve-dsv4 builds.
fn v4_inputs(report: &PlanReport, input: &super::super::PlanInput, options: &LayoutOptions,
    devices: &[DeviceLayout], shape: &Shape, exl3_workspace: Option<u64>,
    workspace: Option<Vec<crate::serving_capacity::V4WorkspaceRank>>)
    -> anyhow::Result<(v4::V4Inputs, crate::families::deepseek_v4::resident::StartupInventory)> {
    let checkpoint = input.checkpoint();
    let cfg = DeepseekV4Config::read(&checkpoint.snapshot, shape.native_layers)?;
    let catalog = input.expert_catalog()?;
    let routed = *catalog.routed_experts();
    anyhow::ensure!(!options.tp2_dspark_experts || (devices.len() == 2 && shape.native_layers > 0),
        "V4 TP2 dSpark requires two coordinator GPUs and retained stages");
    let (experts, draft) = v4::expert_costs(catalog, shape.native_layers)?;
    let spark_ranks = match report.placement { ExpertPlacement::Sparks { ranks } => ranks, ExpertPlacement::Local => 0 };
    let expert_workspace = if catalog.exl3().is_some() { exl3_workspace.unwrap_or(160 * MIB * shape.prefill_rows / 4096) }
        else { crate::serving_capacity::deepseek_v4_native_workspace(routed.hidden as u64, routed.intermediate as u64,
            routed.experts as u64, routed.topk as u64, shape.prefill_rows.max(shape.decode_rows))? };
    let host = options.host_memory.map(Ok).unwrap_or_else(crate::formats::mapped_preload::host_memory_before_process)?;
    let id = if cfg.dim == 4096 { "dsv4f" } else { "dsv4p" };
    // Each coordinator GPU's CUDA context and selected-program code: the
    // baseline's `context_bytes` and the matching `loaded code` remainder,
    // both charged as named Runtime owners by `fixed_demands` (#1614).
    let split = devices.len() == 2;
    let mut gpus = Vec::with_capacity(devices.len());
    let mut pending_code = Vec::with_capacity(devices.len());
    for (rank, (device, &total)) in devices.iter().zip(&options.rtx_bytes).enumerate() {
        let context = crate::placement::ArchContext::coordinator(total, None).context_bytes;
        let runtime = runtime_item(shape.manifest.as_deref(), &program_families(id, split),
            v4::code_experts(catalog), split, rank, total);
        pending_code.push(runtime.bytes.saturating_sub(context));
        gpus.push((total, Baseline::Planned { context_bytes: context, loaded_bytes: device.used_bytes() }));
    }
    let context = shape.manifest.as_deref().and_then(|m| v4::manifest_context(m, id, cfg.n_heads, shape.decode_rows));
    let attention = report.requested_attention.unwrap_or(if devices.len() == 2 && context.is_some() {
        crate::placement::AttentionPlacement::Context
    } else { crate::placement::AttentionPlacement::Heads });
    let host_marks = v4::host_marks_enabled(devices.len(), attention != crate::placement::AttentionPlacement::Heads,
        options.host_cache_bytes != Some(0));
    let mark_slots = if let Some(slots) = options.prefix_slots { slots }
        else { v4::mark_slots(&cfg, devices.len(), shape.prefill_rows,
            shape.native_layers, shape.concurrency, options.prefix_entries("deepseek_v4"), options.mimo_prefix_mark_bytes, host_marks)? };
    let onboard = options.resolve_onboard(routed.routed_layers()).unwrap_or_else(v4::default_onboard);
    let tp2_workspace = if devices.len() == 2 && (options.tp2_dspark_experts || onboard.layers(routed.layers - routed.first_layer) != Some(0)) {
        crate::serving_capacity::deepseek_v4_tp2_workspace(catalog, options.workspace_manifest.as_deref(),
            shape.prefill_rows.max(shape.decode_rows)).ok()
    } else { None };
    if devices.len() == 2 && tp2_workspace.is_none() && (options.tp2_dspark_experts || (onboard != Onboard::Auto && onboard.layers(routed.layers - routed.first_layer) != Some(0))) {
        anyhow::bail!("V4 TP2 expert kernels are missing: export the matching rtx-tp2 package for dSpark or explicit local layers");
    }
    let host_embedding = options.host_embedding && checkpoint.tensors.iter().filter(|t|
        t.meta.name == "embed.weight" || t.meta.name.ends_with("embed_tokens.weight")).count() == 1
        && checkpoint.require_untied_embedding("embed.weight").is_ok();
    let weights = crate::families::deepseek_v4::resident::WeightInventory::from_headers(&cfg,
        devices.len(), shape.native_layers > 0, !host_embedding, checkpoint.tensors.iter().map(|t| &t.meta))?;
    let mut extra = Vec::new();
    if options.drafter_bytes > 0 {
        // An external drafter's estimated device bytes are baseline; only its residency claim is admitted.
        extra.push(crate::placement::Demand::new(0, Category::Weights, "loaded drafter weights", 0, Basis::Exact)
            .with_resident_weights(crate::placement::ResidentWeights(options.drafter_bytes)));
    }
    let inputs = v4::V4Inputs {
        expert_usage: options.expert_usage,
        resident_weight_limit: Some(crate::placement::ResidentWeights(options.weight_budget_bytes)), spark_reply_format: options.resolved_spark_reply_format(false), serving_context: crate::placement::planner::Context::resolve(shape.context_tokens, (options.context_tokens > 0).then_some(options.context_tokens)), cfg: cfg.clone(),
        weights,
        cache_stages: shape.native_layers, gpus,
        spark_ranks, sequences: shape.concurrency, prefill_rows: shape.prefill_rows, prefill_waves: options.prefill_waves, decode_rows: shape.decode_rows,
        max_context: shape.context_tokens, mark_slots, workspace,
        draft_half: if options.tp2_dspark_experts { v4::draft_half_costs(catalog, shape.native_layers)? } else { Vec::new() },
        experts, draft, expert_workspace, first_routed: routed.first_layer,
        experts_half: if tp2_workspace.is_some() { v4::expert_half_costs(catalog)? } else { Vec::new() },
        tp2_workspace: tp2_workspace.map_or([0; 2], |bytes| [bytes; 2]),
        exchange_f32: options.exchange_f32,
        requested_pool: options.pool_tokens.filter(|&n| n > 0), onboard,
        pending_code,
        context,
        attention: report.requested_attention, extra, physical_bytes: Vec::new(),
        host: Some(v4::V4Host { total: host.0, available: host.1, headroom: (8 << 30).max(host.0 / 10),
            cache_bytes: options.host_cache_bytes, prefix_entries: options.prefix_entries("deepseek_v4"),
            span_rows: report.prefill_span_rows.unwrap_or(shape.prefill_rows * options.prefill_waves),
            embedding_bytes: if host_embedding { checkpoint.tensors.iter().find(|t| t.meta.name == "embed.weight")
                .map_or(0, |t| t.meta.byte_length) } else { 0 },
            // Without a manifest, the compiled stride covers the context in 64-row groups.
            c128_width: shape.manifest.as_deref()
                .and_then(|m| crate::serving_capacity::compiled_c128_width(m, id).ok())
                .unwrap_or_else(|| shape.context_tokens.div_ceil(128).next_multiple_of(64).max(64)) }) };
    let startup = inputs.weights.startup.clone();
    Ok((inputs, startup))
}
