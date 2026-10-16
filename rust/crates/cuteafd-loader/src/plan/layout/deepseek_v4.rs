//! DeepSeek V4 (Flash/Pro) plan adapter: header-derived resident weights, then
//! the same `placement::families::deepseek_v4` request serve-dsv4 solves, with
//! each GPU's planned context, modules and embedding as the baseline.
use super::*;
use crate::families::deepseek_v4::{DeepseekV4Config, resident::resident_operands};
use crate::placement::{families::deepseek_v4 as v4, Baseline, FfnMode, LayerMode, Onboard};

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
    input: &super::super::PlanInput, options: &LayoutOptions) -> MemoryLayout {
    let checkpoint = input.checkpoint();
    let mut shape = Shape::resolve(report, model, options,
        ShapePolicy { all_native_stages: true, ..ShapePolicy::manifest_bounded() });
    let gpus = options.rtx_bytes.len().clamp(1, 2);
    let split = gpus == 2 && options.head_split;
    let ranks = if split { 2 } else { 1 };
    let mut devices: Vec<DeviceLayout> = options.rtx_bytes.iter().take(gpus).enumerate()
        .map(|(index, &bytes)| DeviceLayout { kind: DeviceKind::Rtx, index: index as u32,
            capacity_bytes: bytes, items: Vec::new(), kv_tokens: 0 }).collect();
    let mut waste = Vec::new();
    let mut notes = vec![format!("resolved context {} tokens (checkpoint {})", shape.context_tokens, shape.checkpoint_context),
        format!("attention placement {}", report.attention_placement)];
    shape.select_card_plans(report, options, &mut notes);
    notes.push("deepseek_v4: workspace/runtime/Spark allowances are unqualified; validate against the allocation ledger before using this layout for admission".into());
    mapped_table_note(report, &mut notes);
    let dspark = shape.native_layers > 0;
    if let Some(embedding) = resident_embedding(checkpoint, ranks, dspark) {
        push_resident(&mut devices, &embedding, |group| if group == "embedding" { Category::Embedding } else { Category::Weights });
    }
    if options.host_embedding {
        place_host_embedding(report, checkpoint, &mut devices, &mut waste, &mut notes);
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
        let manifest = shape.manifest.as_ref()?;
        let cfg = DeepseekV4Config::read(&checkpoint.snapshot, shape.native_layers).ok()?;
        let scratch = crate::serving_capacity::deepseek_v4_workspace_scratch(manifest, id, split,
            shape.prefill_rows, shape.decode_rows).ok()?;
        crate::serving_capacity::deepseek_v4_workspace_geometry(&cfg, shape.prefill_rows, shape.decode_rows,
            crate::serving_capacity::compiled_c128_width(manifest, id).ok()?.checked_mul(128)?, ranks, scratch).ok()
    })();
    if workspace.is_none() {
        notes.push("V4 workspace allowance is estimated without a matching PROGRAMS.json; use --workspace-manifest for Flash/Pro allocation geometry".into());
    }
    // V4's coordinator always holds dSpark (and any RTX layers) in the catalog's package.
    let package = if report.experts.as_ref().is_some_and(|e| e.package.contains("exl3")) { "exl3" } else { "*" };
    for (index, device) in devices.iter_mut().take(ranks).enumerate() {
        device.items.push(runtime_item(shape.manifest.as_ref(), &program_families(id, split), package, split,
            index, options.rtx_bytes[index]));
    }
    let spark_working = crate::placement::SparkWorkingSet {
        rtx_layers: Vec::new(),
        spark_layers: if matches!(report.placement, ExpertPlacement::Sparks { .. }) {
            (0..model.spec().layers.len()).collect() } else { Vec::new() },
        spark_ranks: match report.placement { ExpertPlacement::Sparks { ranks } => ranks, ExpertPlacement::Local => 0 },
        prefill_lanes: v4::PREFILL_LANES,
    };
    let endpoints = spark_endpoints("deepseek_v4", v4::PREFILL_LANES);
    let mut sparks = spark_devices(report, model, checkpoint, options, SPARK, &spark_working, endpoints,
        &mut waste, &mut notes);
    let mut pool_tokens = 0;
    let mut local_layers = 0;
    match solve(report, input, options, &devices[..ranks], &shape, exl3_workspace, workspace) {
        Ok((placement, reserves)) => {
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
            report.placement_supported = false;
            notes.push(format!("V4 pool-first placement: {error}"));
        }
    }
    let towers = Towers::checkpoint(report);
    if report.encoder.is_none() {
        resolve_encoder(report, &towers, &mut devices, &mut sparks, &vec![0; gpus], options, &mut notes);
    }
    if local_layers > 0 { notes.push(format!("{local_layers} routed backbone layers resident on RTX")); }
    // Reapply after admission: rank-local I/O never belongs to the coordinator KV budget.
    charge_spark_devices(report, model, checkpoint, options, SPARK, &spark_working, endpoints, &mut sparks,
        &mut notes, |device| format!("{} I/O admission", device.name()));
    devices.extend(sparks);
    MemoryLayout { devices, pool_tokens, waste, notes }
}

/// V4's planned admission: the same request serve-dsv4 solves. The devices'
/// capacities become `rtx_bytes - reserve` so `free_bytes` matches.
fn solve(report: &PlanReport, input: &super::super::PlanInput, options: &LayoutOptions,
    devices: &[DeviceLayout], shape: &Shape, exl3_workspace: Option<u64>,
    workspace: Option<Vec<crate::serving_capacity::V4WorkspaceRank>>)
    -> anyhow::Result<(crate::placement::Placement, Vec<u64>)> {
    let checkpoint = input.checkpoint();
    let cfg = DeepseekV4Config::read(&checkpoint.snapshot, shape.native_layers)?;
    let catalog = input.expert_catalog()?;
    let routed = *catalog.routed_experts();
    let (experts, draft) = v4::expert_costs(catalog, shape.native_layers)?;
    let spark_ranks = match report.placement { ExpertPlacement::Sparks { ranks } => ranks, ExpertPlacement::Local => 0 };
    let expert_workspace = if catalog.exl3().is_some() { exl3_workspace.unwrap_or(160 * MIB * shape.prefill_rows / 4096) }
        else { crate::serving_capacity::deepseek_v4_native_workspace(routed.hidden as u64, routed.intermediate as u64,
            routed.experts as u64, routed.topk as u64, shape.prefill_rows.max(shape.decode_rows))? };
    let gpus = devices.iter().zip(&options.rtx_bytes).map(|(device, &total)|
        (total, Baseline::Planned { context_bytes: 0, loaded_bytes: device.used_bytes() })).collect();
    let mark_slots = options.prefix_slots.map_or_else(|| v4::mark_slots(&cfg, devices.len(), shape.prefill_rows,
        shape.native_layers, shape.concurrency, options.mimo_prefix_entries, options.mimo_prefix_mark_bytes), Ok)?;
    let onboard = match (options.onboard, options.local_expert_layers) {
        (Some(onboard), _) => onboard,
        (None, Some(n)) => Onboard::Layers(n.saturating_sub(routed.first_layer)),
        (None, None) => v4::default_onboard(),
    };
    let tp2_workspace = if devices.len() == 2 && onboard.layers(routed.layers - routed.first_layer) != Some(0) {
        crate::serving_capacity::deepseek_v4_tp2_workspace(catalog, options.workspace_manifest.as_deref(),
            shape.prefill_rows.max(shape.decode_rows)).ok()
    } else { None };
    if devices.len() == 2 && tp2_workspace.is_none() && onboard != Onboard::Auto && onboard.layers(routed.layers - routed.first_layer) != Some(0) {
        anyhow::bail!("V4 TP2 expert kernels are missing: export the matching rtx-tp2 package for explicit local layers");
    }
    let id = if cfg.dim == 4096 { "dsv4f" } else { "dsv4p" };
    let inputs = v4::V4Inputs { cfg: &cfg,
        weights: crate::families::deepseek_v4::resident::WeightInventory::from_headers(&cfg, devices.len(),
            shape.native_layers > 0, checkpoint.tensors.iter().map(|t| &t.meta))?,
        cache_stages: shape.native_layers, gpus, headroom_floor: options.headroom_bytes,
        spark_ranks, sequences: shape.concurrency, prefill_rows: shape.prefill_rows, decode_rows: shape.decode_rows,
        max_context: shape.context_tokens, reserve_bytes: v4::RESERVE_BYTES, mark_slots, workspace,
        experts, draft, expert_workspace, first_routed: routed.first_layer,
        experts_half: if tp2_workspace.is_some() { v4::expert_half_costs(catalog)? } else { Vec::new() },
        tp2_workspace: tp2_workspace.map_or([0; 2], |bytes| [bytes; 2]),
        exchange_f32: options.exchange_f32,
        requested_pool: options.pool_tokens.filter(|&n| n > 0), onboard,
        full_prefill_logits: 0, code_bytes: v4::code_bytes(cfg.dim, v4::code_experts(catalog), devices.len()),
        context: shape.manifest.as_ref().and_then(|m| v4::manifest_context(m, id, cfg.n_heads, shape.decode_rows)),
        attention: report.requested_attention };
    let request = v4::request(&inputs)?;
    let reserves = request.inventory.gpus.iter().map(|g| g.headroom_bytes).collect();
    Ok((crate::placement::solve(&request)?, reserves))
}
