//! Header-only inventory for the same pre-allocation Qwen dual request as runtime.
use super::*;
use anyhow::{Context, Result};
use crate::placement::{families::qwen4, Baseline, GpuBudget, Onboard, Placement};
use crate::families::qwen4::{resident::checkpoint_resident_bytes, Qwen4Config};
use crate::serving_capacity::qwen_graphs::QwenAdmissionInputs;

/// Spark ranks at 4096 capacity rows: the observed host footprint and the
/// non-I/O wave residual (P11 replaces both with the worker's measurement).
const SPARK_HOST_BYTES: u64 = 13 * GIB;

/// Qwen plan adapter: a dual 0-Spark layout runs whole-layer TP2 owners
/// (`dual_layout`); every other layout serves on rtx0 (`single_layout`).
pub(crate) fn layout(report: &mut PlanReport, model: &dyn super::super::FamilyModel,
    input: &super::super::PlanInput, options: &mut LayoutOptions) -> MemoryLayout {
    if options.prefill_lanes > 0 { options.prefill_waves = options.prefill_lanes; }
    let shape = Shape::resolve(report, model, options, ShapePolicy::manifest_bounded());
    if options.rtx_bytes.len() == 2 && options.head_split && report.placement == ExpertPlacement::Local {
        return dual_layout(report, model, input, options, shape.prefill_rows, shape.concurrency,
            shape.context_tokens, shape.native_layers > 0, shape.manifest.as_ref());
    }
    options.prefill_unit_rows = Some(shape.prefill_rows);
    report.prefill_transports = Some(1);
    single_layout(report, model, input, options, shape)
}

/// Exact resident weights on rtx0 (`checkpoint_resident_bytes`).
fn resident(checkpoint: &crate::plan::Checkpoint, mtp: bool) -> Option<Vec<Vec<(String, String, u64)>>> {
    use crate::families::qwen4::resident::Qwen4Representation;
    let cfg = Qwen4Config::from_hf(&checkpoint.config).ok()?;
    let mtp = mtp && cfg.mtp_layers > 0;
    let bytes = checkpoint_resident_bytes(checkpoint, &cfg, cfg.layers, mtp,
        Qwen4Representation { fp8_projections: false, fp8_head: true }).ok()?;
    Some(vec![vec![
        ("target incl PLE projections".into(), "bf16+f32".into(), bytes.target_bytes),
        ("embedding".into(), "bf16".into(), bytes.embedding_bytes),
        ("lm_head".into(), "fp8-row128".into(), bytes.head_bytes),
        ("speculator".into(), "bf16+f32".into(), bytes.mtp_bytes),
    ]])
}

fn single_layout(report: &mut PlanReport, model: &dyn super::super::FamilyModel,
    input: &super::super::PlanInput, options: &LayoutOptions, mut shape: Shape) -> MemoryLayout {
    let checkpoint = input.checkpoint();
    let gpus = options.rtx_bytes.len().clamp(1, 2);
    let mut devices: Vec<DeviceLayout> = options.rtx_bytes.iter().take(gpus).enumerate()
        .map(|(index, &bytes)| DeviceLayout { kind: DeviceKind::Rtx, index: index as u32,
            capacity_bytes: bytes.saturating_sub(options.headroom_bytes.max(3 * GIB)), items: Vec::new(), kv_tokens: 0 })
        .collect();
    let mut waste = Vec::new();
    let mut notes = vec![format!("resolved context {} tokens (checkpoint {})", shape.context_tokens, shape.checkpoint_context),
        format!("attention placement {}", report.attention_placement)];
    shape.select_card_plans(report, options, &mut notes);
    let mtp = shape.native_layers > 0;
    let local = report.placement == ExpertPlacement::Local;
    let exl3 = report.experts.as_ref().is_some_and(|e| e.package.contains("exl3"));
    let package = report.experts.as_ref().map(|e| e.package.as_str());
    let reference = gpus == 1 && (94 * GIB..=96 * GIB).contains(&options.rtx_bytes[0])
        && mtp && options.local_expert_layers.is_none() && local && package == Some("qwen4:exl3-k45")
        && model.spec().hidden == 2560 && model.spec().layers.len() == 48
        && shape.prefill_rows == 4096 && shape.concurrency == 8 && matches!(options.prefix_slots, None | Some(18))
        && matches!(shape.context_tokens, 0 | 32768);
    let basis = if reference { Basis::Calibrated } else { Basis::Estimated };
    if reference {
        notes.push("Runtime allowances calibrated at the natural-minimum reference layout after an 8K prefill and C4 decode; other layouts remain estimates".into());
    } else {
        // Retain estimated-allowance warnings pending census proof coverage: https://github.com/tpurtell/cuteafd/issues/388.
        notes.push("qwen4: workspace/runtime/Spark allowances are unqualified; validate against the allocation ledger before using this layout for admission".into());
    }
    if gpus == 2 {
        notes.push("Qwen serves on rtx0; rtx1 is idle and contributes no KV capacity".into());
    }
    mapped_table_note(report, &mut notes);
    match resident(checkpoint, mtp) {
        Some(ranks) => {
            push_resident(&mut devices, &ranks, |group| match group {
                "embedding" => Category::Embedding,
                // The native MTP layer loads with the target (ledger scope `weights`).
                _ => Category::Weights,
            });
            // FP8/NVFP4 local experts load at their checkpoint bytes; EXL3 below as arenas.
            if local && !exl3 {
                for component in report.components.iter().filter(|c| c.component == Component::RoutedExpert
                    && c.status != Status::Unused && c.status != Status::Disabled) {
                    devices[0].items.push(Item::new(Category::Experts, component.component.label(),
                        component.formats.keys().cloned().collect::<Vec<_>>().join("+"), component.bytes, Basis::Exact));
                }
            }
        }
        None => {
            report.placement_supported = false;
            let what = "Qwen resident weights: checkpoint headers do not describe the serve-qwen4 resident layout".to_string();
            report.hints.push(crate::plan::Hint { what: what.clone(),
                how: "Supply the checkpoint's complete coordinator tensor headers.".into() });
            notes.push(what);
        }
    }
    if options.host_embedding {
        place_host_embedding(report, checkpoint, &mut devices, &mut waste, &mut notes);
    }
    if let Some((backbone, draft)) = exl3_arenas(input, mtp) {
        if local {
            devices[0].items.push(Item::new(Category::Experts, "routed expert arenas", "exl3", backbone, Basis::Formula));
        }
        if draft > 0 {
            devices[0].items.push(Item::new(Category::Experts, "native MTP expert arena", "exl3", draft, Basis::Formula));
        }
    } else if mtp {
        let native: u64 = report.components.iter().filter(|c| c.component == Component::SpeculatorExpert && c.status == Status::Unused)
            .map(|c| c.bytes).sum();
        if native > 0 {
            devices[0].items.push(Item::new(Category::Experts, "native MTP experts", "native", native, Basis::Exact));
        }
    }
    if exl3 && (local || mtp) {
        // The EXL3 window retains its shared capacity arenas beside the trellis
        // bytes, from the same capacity manifests the loader reads.
        match expert_workspace(report, model, input, options.workspace_manifest.as_deref(), shape.prefill_rows) {
            Some(bytes) => devices[0].items.push(Item::new(Category::Experts, "local EXL3 workspace", "", bytes, Basis::Formula)),
            // A header-only inventory names no image: its workspace is the layout's question.
            None if shape.manifest.is_none() => notes.push("local EXL3 workspace not charged: no program manifest \
                (--workspace-manifest or an image's PROGRAMS.json) to read its capacity manifests from".into()),
            None => {
                report.placement_supported = false;
                let what = "Qwen local EXL3 workspace: no matching rtx-tp1/m*/v41_exl3.json capacity manifests".to_string();
                report.hints.push(crate::plan::Hint { what: what.clone(), how: "Images bundle them; or export the exl3 tree \
                    alongside PROGRAMS.json (--workspace-manifest).".into() });
                notes.push(what);
            }
        }
    }
    if !local && mtp {
        notes.push("Native Qwen MTP experts stay on rtx0; Spark ranks serve backbone experts only".into());
    }
    if options.drafter_bytes > 0 {
        devices[0].items.push(Item::new(Category::Drafter, "drafter", "", options.drafter_bytes, basis));
    }
    devices[0].items.push(runtime_item(shape.manifest.as_ref(), &program_families("qwen4", false),
        local_package(report), false, 0, options.rtx_bytes[0]));
    let spark_ranks = match report.placement { ExpertPlacement::Sparks { ranks } => ranks, ExpertPlacement::Local => 0 };
    let working = crate::placement::SparkWorkingSet { rtx_layers: Vec::new(),
        spark_layers: if spark_ranks > 0 { (0..model.spec().layers.len()).collect() } else { Vec::new() },
        spark_ranks, prefill_lanes: 1 };
    let charges = SparkCharges { host_bytes: Some(SPARK_HOST_BYTES),
        wave_bytes: crate::placement::families::qwen4::SPARK_WORKSPACE_BYTES, basis };
    let endpoints = spark_endpoints("qwen4", 1);
    let mut sparks = spark_devices(report, model, checkpoint, options, charges, &working, endpoints, &mut waste, &mut notes);
    if report.encoder.is_none() {
        let target = options.pool_tokens.filter(|&tokens| tokens != 0)
            .unwrap_or(options.target_pool_tokens.max(shape.context_tokens));
        let per_token = Qwen4Config::from_hf(&checkpoint.config).ok()
            .and_then(|cfg| crate::serving_capacity::qwen_cache_geometry(&cfg, cfg.layers, mtp, options.qwen_kv).ok())
            .map_or(0, |g| (g.ranks[0].persistent_unit_bytes + g.ranks[0].pool_metadata_unit_bytes)
                .div_ceil(g.logical_unit_rows.max(1)));
        resolve_encoder(report, &qwen_towers(report), &mut devices, &mut sparks, &[per_token * target], options, &mut notes);
    }
    let mut pool_tokens = 0;
    match placement(input, report, options, &mut devices[0], shape.manifest.as_ref(),
        PlanShape { prefill_rows: shape.prefill_rows, concurrency: shape.concurrency, context_tokens: shape.context_tokens, mtp }) {
        Ok(tokens) => pool_tokens = tokens,
        Err(error) => {
            report.placement_supported = false;
            notes.push(format!("Qwen admission: {error}"));
        }
    }
    if pool_tokens < shape.context_tokens {
        notes.push(format!("full-context admission shortfall: context {} tokens, pool {pool_tokens} tokens, shortfall {} tokens",
            shape.context_tokens, shape.context_tokens - pool_tokens));
    }
    charge_spark_devices(report, model, checkpoint, options, charges, &working, endpoints, &mut sparks,
        &mut notes, |device| format!("{} I/O admission", device.name()));
    devices.extend(sparks);
    MemoryLayout { devices, pool_tokens, waste, notes }
}

/// The planner's Qwen shape inputs, beside `LayoutOptions`.
struct PlanShape {
    prefill_rows: u64,
    concurrency: u64,
    context_tokens: u64,
    mtp: bool,
}

/// serve-qwen4's admission on the planned GPU: the shared fixed items
/// (`qwen_admission`), its startup graph set and the largest pool both fit
/// beside, over what the planner already laid out (context, weights, embedding,
/// head, drafter, resident experts and their package scratch, encoder).
fn placement(input: &super::super::PlanInput, report: &mut PlanReport, options: &LayoutOptions,
    device: &mut DeviceLayout, manifest: Option<&serde_json::Value>, shape: PlanShape) -> anyhow::Result<u64> {
    use crate::serving_capacity::qwen_graphs::*;
    let checkpoint = input.checkpoint();
    let cfg = crate::families::qwen4::Qwen4Config::from_hf(&checkpoint.config)?;
    let mtp = shape.mtp && cfg.mtp_layers > 0;
    let geometry = crate::serving_capacity::qwen_cache_geometry(&cfg, cfg.layers, mtp, options.qwen_kv)?;
    let mark = geometry.ranks[0].retained_mark_bytes;
    let marks = options.prefix_slots.unwrap_or_else(|| cuteafd_core::prefix::mark_slots_with_host(shape.concurrency,
        options.mimo_prefix_entries, mark, options.mimo_prefix_mark_bytes, options.host_cache_bytes != Some(0))) * mark;
    let prefill_shape = if options.prefill_waves > 1 {
        anyhow::ensure!(matches!(report.placement, ExpertPlacement::Sparks { .. }),
            "Qwen prefill waves require single-owner Spark experts");
        let unit = if options.prefill_rows > 0 { options.prefill_rows } else { 4096 };
        let units = crate::serving_capacity::PrefillShape { unit_rows: unit, waves: options.prefill_waves }.validate()?;
        anyhow::ensure!(units.waves <= 2, "Qwen mapped PLE gathers admit at most two prefill waves");
        Some(units)
    } else { None };
    let span = prefill_shape.map_or(shape.prefill_rows, |s| s.slot_rows());
    let logits = if options.full_prefill_logits { full_prefill_logits_bytes("qwen4", span,
        cfg.vocab_size as u64) } else { 0 };
    // The PLE n-gram table is mapped (each step's rows uploaded): row width ple_dim / ple_rows, one byte
    // per value in E4M3, two in BF16 (`ple::PleTable::load`).
    let ple = checkpoint.tensors.iter().find(|t| t.meta.name.contains("ngram_embedding.shard_"))
        .map(|t| ((cfg.ple_dim / cfg.ple_rows().max(1)) as u64
            * if t.meta.dtype == cuteafd_core::DType::F8E4M3 { 1 } else { 2 }, t.meta.dtype == cuteafd_core::DType::F8E4M3));
    for (package, bytes) in package_scratch(input, report, options.workspace_manifest.as_deref(),
        shape.prefill_rows, mtp) {
        device.items.push(Item::new(Category::Experts, "package scratch", package, bytes, Basis::Formula));
    }
    let loaded_bytes = device.used_bytes();
    let inputs = crate::placement::families::qwen4::QwenInputs { automatic_context: options.context_tokens == 0,
        admission: QwenAdmissionInputs { prefill_shape, cfg: &cfg, layers: cfg.layers, mtp, kv_format: options.qwen_kv, manifest,
            prefill_rows: span, slots: options.state_slots.unwrap_or(shape.concurrency), mark_bytes: marks,
            full_prefill_logits: logits, ple, future_expert_bytes: 0, headroom: options.headroom_bytes.max(3 * GIB) },
        capacity_bytes: options.rtx_bytes[0],
        baseline: crate::placement::Baseline::Planned { context_bytes: 0, loaded_bytes },
        pending_code_bytes: 0, max_context: shape.context_tokens,
        requested_pool: options.pool_tokens.filter(|&n| n > 0),
        spark_ranks: match report.placement { ExpertPlacement::Sparks { ranks } => ranks, ExpertPlacement::Local => 0 },
        startup_graph_modes: Some((shape.concurrency.min(16) as usize, true)),
    };
    let placement = crate::placement::families::qwen4::placement(&inputs)?;
    report.prefill_span_rows = Some(span);
    device.items.extend(placement.items[0].iter().cloned());
    device.kv_tokens = placement.pool_tokens;
    let pool_tokens = placement.pool_tokens;
    report.solver_placement = Some(placement);
    Ok(pool_tokens)
}

/// Package scratch of Qwen's resident FP8/NVFP4 experts (`Fp8Experts::load`): one scratch per package
/// the backbone and the mixed-format MTP layer load, at the prefill rows' capacity.
fn package_scratch(input: &super::super::PlanInput, report: &PlanReport, workspace_manifest: Option<&std::path::Path>,
    prefill_rows: u64, mtp: bool) -> Vec<(String, u64)> {
    let local = report.placement == ExpertPlacement::Local;
    let Ok(catalog) = input.expert_catalog() else { return Vec::new() };
    let Some(tensors) = catalog.fp8() else { return Vec::new() };
    let lib = crate::placement::inventory::image_lib(workspace_manifest);
    let layers = catalog.routed_experts().layers;
    let mut out = Vec::new();
    if local {
        out.extend(crate::placement::inventory::fp8moe_package_scratch(&lib, "qwen4", tensors.format(), prefill_rows));
    }
    if mtp {
        if let Ok(format) = tensors.layer_format(layers) {
            if !local || format != tensors.format() {
                out.extend(crate::placement::inventory::fp8moe_package_scratch(&lib, "qwen4", format, prefill_rows));
            }
        }
    }
    out
}

fn exl3_arenas(input: &super::super::PlanInput, mtp: bool) -> Option<(u64, u64)> {
    use crate::V41Exl3Layer;
    let catalog = input.expert_catalog().ok()?;
    let shape = catalog.routed_experts();
    let manifest = catalog.exl3()?;
    let bytes = |layer| -> Option<u64> {
        u64::try_from(manifest.residency(layer, 1, 0).ok()?.device_arena_layout().ok()?.1).ok()
    };
    let backbone = (shape.first_layer..shape.layers).try_fold(0u64,
        |total, layer| total.checked_add(bytes(V41Exl3Layer::Backbone(layer))?))?;
    let draft = (0..if mtp { shape.draft_stages.min(1) } else { 0 }).try_fold(0u64,
        |total, stage| total.checked_add(bytes(V41Exl3Layer::Dspark(stage))?))?;
    Some((backbone, draft))
}

fn dual_layout(report: &mut PlanReport, _model: &dyn super::super::FamilyModel,
    input: &super::super::PlanInput, options: &LayoutOptions, rows: u64, concurrency: u64,
    context: u64, mtp: bool, manifest: Option<&serde_json::Value>) -> MemoryLayout {
    let mut devices = options.rtx_bytes.iter().enumerate().map(|(rank, &capacity)| {
        let arch = crate::placement::ArchContext::for_device("sm_120", capacity);
        let programs = manifest.map(|m| crate::placement::ProgramSet::from_manifest(m, &["qwen4"]))
            .unwrap_or_else(|| crate::placement::ProgramSet::from_names(&["qwen4"], []));
        let allowance = crate::placement::inventory::loaded_code_allowance("qwen4", local_package(report),
            true, rank as u8, arch, Some(&programs));
        DeviceLayout { kind: DeviceKind::Rtx, index: rank as u32, capacity_bytes: capacity,
            items: vec![Item::new(Category::Runtime, "context+modules", "SM120",
                arch.context_bytes + allowance.bytes, allowance.basis)], kv_tokens: 0 }
    }).collect::<Vec<_>>();
    let mut notes = vec![format!("resolved context {context} tokens"),
        "Qwen whole-layer attention ranges; all routed experts native TP2; head and MTP on rtx0".into()];
    if report.requested_attention.is_some_and(|mode| mode != crate::placement::AttentionPlacement::Layers) {
        report.placement_supported = false;
        notes.push("Qwen dual supports whole-layer attention only, not head or context shards".into());
        return MemoryLayout { devices, pool_tokens: 0, waste: Vec::new(), notes };
    }
    report.attention_placement = crate::placement::AttentionPlacement::Layers;
    if report.encoder.is_none() {
        let mut media = devices.clone();
        let mut reserved = options.clone();
        reserved.pool_tokens = Some(options.pool_tokens.filter(|&n| n > 0).unwrap_or_else(||
            crate::placement::PoolPolicy::resolve(&options.rtx_bytes, context, None, 256, true).target));
        match apply(input, &reserved, &mut media, rows, concurrency, context, mtp, manifest) {
            Ok(_) => {
                resolve_encoder(report, &qwen_towers(report), &mut media, &mut [], &[0; 2], options, &mut notes);
                for (device, admitted) in devices.iter_mut().zip(media) {
                    device.items.extend(admitted.items.into_iter().filter(|item|
                        item.group.starts_with("vision") || item.group.starts_with("audio")));
                }
            }
            Err(error) => { report.placement_supported = false; notes.push(format!("Qwen media admission: {error:#}")); }
        }
    }
    let pool_tokens = match apply(input, options, &mut devices, rows, concurrency, context, mtp, manifest) {
        Ok(placement) => { report.solver_placement = Some(placement.clone()); notes.push(format!("placement: {}", placement.summary())); placement.pool_tokens },
        Err(error) => {
            report.placement_supported = false;
            report.hints.push(super::super::Hint { what: format!("Qwen dual admission: {error:#}"),
                how: "Supply complete checkpoint headers and exact unequal TP2 plus MTP package manifests.".into() });
            notes.push(format!("Qwen dual admission: {error:#}")); 0
        }
    };
    MemoryLayout { devices, pool_tokens, waste: Vec::new(), notes }
}

#[allow(clippy::too_many_arguments)]
fn apply(input: &super::super::PlanInput, options: &LayoutOptions, devices: &mut [DeviceLayout],
    rows: u64, concurrency: u64, context: u64, mtp: bool, manifest: Option<&serde_json::Value>) -> Result<Placement> {
    let checkpoint = input.checkpoint();
    let cfg = Qwen4Config::from_hf(&checkpoint.config)?;
    let mtp = mtp && cfg.mtp_layers > 0;
    let selected = options.qwen_representation;
    let plan = cuteafd_core::coordinator_programs::AttentionReductionPlan::new(64,
        mtp.then_some(options.native_mtp_layers), options.qwen_attention_plan_rows)?;
    let sms = [0, 1].map(|rank| options.physical_sms.unwrap_or_else(||
        crate::placement::ArchContext::for_device("sm_120", options.rtx_bytes[rank]).sms));
    let selected_manifests = manifest.map(|m| qwen4::dual_program_manifests(m, sms, plan)).transpose()?;
    let owner_manifests = [0, 1].map(|rank| selected_manifests.as_ref().map(|m| &m[rank]));
    let manifest = owner_manifests[0];
    let resident = checkpoint_resident_bytes(checkpoint, &cfg, cfg.layers, mtp, selected)?;
    let catalog = input.expert_catalog()?;
    let costs = qwen4::dual_expert_costs(&catalog, cfg.layers)?;
    let lib = crate::placement::inventory::image_lib(options.workspace_manifest.as_deref()).join("libcuteafd_native.so");
    let packages = qwen4::dual_packages(&catalog, &lib)?;
    let workspace = qwen4::dual_expert_workspace(&catalog, packages.each_ref().map(|p| p.as_path()), rows.max(64))?;
    let host = options.host_embedding;
    let mut fixed = qwen4::dual_owner0_weights(&resident, host);
    fixed.extend(qwen4::dual_projection_load_staging(&cfg, cfg.layers, mtp, selected));
    fixed.extend(qwen4::QwenTp2Rows::new(cfg.hidden as u64, cfg.topk as u64, rows.max(64),
        catalog.exl3().is_some())?.demands()?);
    fixed.extend(qwen4::dual_peer_diagnostics(
        std::env::var("CUTEAFD_PEER_WAIT_DIAGNOSTICS").is_ok_and(|v| v == "1")));
    if mtp { fixed.extend(qwen4::dual_draft_demands(&catalog, &lib, rows)?); }
    let extra = qwen4::dual_ple_demands(&cfg, cfg.layers, &resident)?;
    let geometry = crate::serving_capacity::qwen_cache_geometry(&cfg, cfg.layers, mtp, options.qwen_kv)?;
    let marks = options.prefix_slots.unwrap_or_else(|| cuteafd_core::prefix::mark_slots_with_host(concurrency,
        options.mimo_prefix_entries, geometry.ranks[0].retained_mark_bytes, options.mimo_prefix_mark_bytes, options.host_cache_bytes != Some(0)));
    let ple = checkpoint.tensors.iter().find(|t| t.meta.name.contains("ngram_embedding.shard_"))
        .map(|t| (resident.ple_row_bytes, t.meta.dtype == cuteafd_core::DType::F8E4M3));
    let headroom = options.headroom_bytes.max(3 * GIB);
    let inputs = qwen4::QwenDualInputs { automatic_context: options.context_tokens == 0,
        owner_manifests,
        admission: QwenAdmissionInputs { prefill_shape: None, cfg: &cfg, layers: cfg.layers, mtp, kv_format: options.qwen_kv,
            manifest, prefill_rows: rows, slots: options.state_slots.unwrap_or(concurrency),
            mark_bytes: marks * geometry.ranks[0].retained_mark_bytes,
            full_prefill_logits: if options.full_prefill_logits { full_prefill_logits_bytes("qwen4", rows, cfg.vocab_size as u64) } else { 0 },
            ple, future_expert_bytes: 0, headroom },
        gpus: [0, 1].map(|rank| GpuBudget { capacity_bytes: options.rtx_bytes[rank], headroom_bytes: headroom,
            baseline: Baseline::Planned { context_bytes: 0, loaded_bytes: devices[rank].used_bytes() } }),
        layer_weights: &resident.layer_bytes, layer_experts: &costs, layer_extra: &extra, fixed,
        tp2_workspace: workspace, pending_code_bytes: [0; 2], mark_slots: marks, max_context: context,
        requested_pool: options.pool_tokens.filter(|&n| n > 0), spark_ranks: 0,
        onboard: options.onboard.unwrap_or(Onboard::Auto), startup_graph_modes: Some((concurrency.min(16) as usize, true)),
    };
    let placement = qwen4::dual_placement(&inputs).context("solving Qwen owner ranges")?;
    for (rank, device) in devices.iter_mut().enumerate() {
        device.items.extend(placement.items[rank].iter().cloned());
        device.kv_tokens = placement.pool_tokens;
    }
    Ok(placement)
}
