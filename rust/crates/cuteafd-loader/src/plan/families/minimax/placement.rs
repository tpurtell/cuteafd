//! MiniMax M3 `cuteafd plan --layout` over the shared contract planner (#1856
//! H1). Facts come from the one gatherer serving startup also calls
//! ([`crate::families::minimax_m3::admission::gather`]); the planned edge
//! differs only in its coordinator baseline. This module renders the solve.
use super::Model;
use crate::families::minimax_m3::admission::{gather, Coordinator, FactsRequest};
use crate::placement::families::minimax_m3::LocalPackage;
use crate::placement::planner::{Operation, PrefillShape};
use crate::placement::{AttentionPlacement, Onboard, ResidentWeights};
use crate::plan::{ExpertPlacement, PlanReport};
use cuteafd_core::memory_layout::{DeviceKind, DeviceLayout, MemoryLayout};
use crate::placement::planner::{Device, Domain, Location};

const GIB: f64 = (1u64 << 30) as f64;

fn empty(notes: Vec<String>) -> MemoryLayout {
    MemoryLayout { devices: Vec::new(), pool_tokens: 0, waste: Vec::new(), notes }
}

pub(super) fn layout(model: &Model, input: &crate::plan::PlanInput, report: &mut PlanReport,
    opts: &mut crate::plan::layout::LayoutOptions, requested: ExpertPlacement) -> MemoryLayout {
    let cfg = &model.cfg;
    let gpus = opts.rtx_bytes.len();
    let ranks = requested.spark_ranks();
    report.bytes_by_owner.clear();
    let mut notes = vec![
        "MiniMax shared contract (#1856 H1): one solve over named physical owners; serving stays Planned".to_string(),
    ];
    if opts.host_embedding {
        report.refuse_placement("MiniMax does not support --embedding-placement host yet (#1856 M7)".into(),
            "Use --embedding-placement gpu; the shared host TokenEmbedding lands with stream B.");
        return empty(notes);
    }
    let limits = crate::plan::Family::prefill_limits(&super::MINIMAX_M3);
    let prefill = match crate::plan::layout::resolve_prefill(input, report, opts, ranks, limits) {
        Ok(prefill) => prefill,
        Err(error) => return crate::plan::layout::unsupported_prefill(report, error),
    };
    report.prefill_reserve_rows = Some(prefill.span_rows);
    let shape = PrefillShape { rows: prefill.unit_rows, waves: prefill.waves };
    let attention = report.requested_attention.unwrap_or(if gpus == 2 {
        AttentionPlacement::Layers
    } else { AttentionPlacement::Heads });
    report.attention_placement = attention;
    if attention == AttentionPlacement::Context || (gpus == 1 && attention != AttentionPlacement::Heads) {
        report.refuse_placement(format!("MiniMax {attention} attention is not implemented on {gpus} RTX"),
            "Use heads on one RTX; layers or explicit heads on two RTX.");
        return empty(notes);
    }
    if gpus == 2 && attention == AttentionPlacement::Heads {
        notes.push("Planning only: narrow-head MiniMax kernels not qualified; serving admission refuses heads".into());
    }
    let Some(manifest_path) = opts.workspace_manifest.clone().or_else(|| {
        let path = std::path::PathBuf::from("/opt/cuteafd/share/PROGRAMS.json");
        path.is_file().then_some(path) }) else {
        report.refuse_placement("MiniMax admission needs the installed PROGRAMS.json (native MSA, producer and \
            MXFP8 program capacities are exact owners)".into(),
            "Pass --workspace-manifest PROGRAMS.json from a MiniMax build.");
        return empty(notes);
    };
    let manifest = match input.manifest(&manifest_path) {
        Ok(manifest) => manifest,
        Err(error) => {
            report.refuse_placement(format!("MiniMax program manifest refused: {error:#}"), "Supply a readable PROGRAMS.json.");
            return empty(notes);
        }
    };
    let catalog = match input.expert_catalog() {
        Ok(catalog) => catalog,
        Err(error) => {
            report.refuse_placement(format!("MiniMax expert inventory refused: {error:#}"), "Provide complete NVFP4 expert headers.");
            return empty(notes);
        }
    };
    let host = match opts.host_memory.map(Ok).unwrap_or_else(crate::formats::mapped_preload::host_memory) {
        Ok(memory) => memory,
        Err(error) => {
            report.refuse_placement(format!("MiniMax host inventory refused: {error:#}"), "Provide a host memory sample.");
            return empty(notes);
        }
    };
    let cards = opts.rtx_bytes.iter().map(|&bytes| crate::placement::CoordinatorCard::planned(bytes, opts.physical_sms))
        .collect::<Vec<_>>();
    let spark_bytes = opts.spark_allocation_budget_bytes.unwrap_or(crate::plan::DEFAULT_SPARK_BUDGET_BYTES)
        .min(opts.spark_bytes);
    let sequences = opts.resolved_concurrency("minimax_m3");
    let physical_devices = (0..gpus as i32).collect::<Vec<_>>();
    let request = FactsRequest { checkpoint: input.checkpoint(), cfg, catalog, manifest_path: &manifest_path,
        manifest: manifest.clone(), attention: Some(attention), split_wire: opts.minimax_split_wire, physical_devices: &physical_devices, coordinator: Coordinator::Planned { cards: cards.clone() }, spark_ranks: ranks,
        spark_bytes, prefill: shape, context_tokens: opts.context_tokens, pool_tokens: opts.pool_tokens, sequences,
        host, host_cache_bytes: opts.host_cache_bytes, prefix_entries: opts.prefix_entries("minimax_m3"),
        replies: opts.spark_reply_formats, head: opts.bf16_target_head, expert_usage: opts.expert_usage,
        resident_weight_limit: ResidentWeights(opts.weight_budget_bytes) };
    let inputs = match gather(&request) {
        Ok(inputs) => inputs,
        Err(refused) => { report.refuse_placement(refused.what, refused.how); return empty(notes); }
    };
    if let LocalPackage::Missing(reason) = &inputs.local {
        notes.push(format!("MiniMax RTX onboarding unavailable: {reason}"));
    }
    notes.push("MiniMax bundled dSpark (M8) and CLIP vision tower (M9) are declared recipe domains with their own \
        arenas; this build has no executor for either, so the plan charges neither".into());
    let routed = inputs.routed.len();
    let onboard = opts.resolve_onboard(routed).unwrap_or(Onboard::Auto);
    if matches!(onboard, Onboard::Layers(count) if count > 0) {
        if let LocalPackage::Missing(reason) = &inputs.local {
            report.refuse_placement(format!("MiniMax RTX expert onboarding refused: {reason}"),
                "Install the native fp8-minimax-nvfp4a4 tp1/tp2 exports beside PROGRAMS.json, or use automatic onboarding.");
            return empty(notes);
        }
    }
    let solve_request = crate::placement::families::minimax_m3::request(inputs.context, opts.pool_tokens, onboard,
        shape, if opts.full_prefill_logits { Operation::Scoring } else { Operation::Serving },
        inputs.resident_weight_limit, opts.spark_reply_formats);
    let admitted = match inputs.admit(solve_request) {
        Ok(admitted) => admitted,
        Err(error) => {
            report.refuse_placement(format!("MiniMax joint admission refused: {error}"),
                "Use more Spark ranks, automatic RTX expert onboarding, or a larger --spark-memory-cap.");
            return empty(notes);
        }
    };
    report.phase_ledger = Some(crate::plan::SolvedPhaseLedger::from(&admitted.solved));
    report.host_budget_bytes = inputs.host_prefix.quota_bytes;
    let placed = admitted.placement;
    let onboarded = placed.onboard_layers;
    report.bytes_by_owner.insert("coordinator target arenas".into(),
        inputs.targets.iter().map(|target| target.arena.bytes() as u64).sum());
    report.bytes_by_owner.insert("routed expert weights".into(), model.routed_bytes);
    notes.push(format!("MiniMax attention {}: {}; pool {} tokens; {onboarded} of {routed} routed layers onboarded \
        ({}); unit {} x {} waves; C{sequences}", placed.attention_summary(),
        if inputs.heads() { "TP2 paired query/KV/index groups, replicated index keys, split FFN neurons" }
        else if gpus == 2 { "contiguous whole attention, split FFN neurons and replicated residuals; routed experts TP2 halves" }
        else { "every layer on rtx0" }, placed.pool_tokens, if gpus == 2 { "TP2 halves" } else { "whole layers" },
        shape.rows, shape.waves));
    let mut devices = Vec::new();
    for gpu in 0..gpus {
        let mut device = DeviceLayout { kind: DeviceKind::Rtx, index: gpu as u32,
            capacity_bytes: cards[gpu].physical_bytes.min(cards[gpu].logical_bytes), worker_budget_bytes: 0,
            items: placed.items[gpu].clone(), kv_tokens: placed.pool_tokens };
        device.items.extend(crate::placement::reserve::gpu_item("minimax_m3"));
        notes.push(format!("rtx{gpu}: target arena {} B ({} views); attention layers {:?}",
            inputs.targets[gpu].arena.bytes(), inputs.targets[gpu].arena.views().len(), inputs.targets[gpu].layers));
        devices.push(device);
    }
    let mut widest = 0u64;
    for rank in 0..ranks {
        let charge = inputs.spark[onboarded][rank];
        widest = widest.max(charge.experts);
        let location = Location { device: Device::Spark(rank as u8), domain: Domain::Managed };
        let items = match admitted.solved.items(location) {
            Ok(items) => items,
            Err(error) => {
                report.refuse_placement(format!("MiniMax Spark ledger refused: {error}"),
                    "Supply complete MiniMax admission owner facts.");
                return empty(notes);
            }
        };
        devices.push(DeviceLayout { kind: DeviceKind::Spark, index: rank as u32, capacity_bytes: spark_bytes,
            worker_budget_bytes: spark_bytes, kv_tokens: 0, items });
    }
    report.spark_rank_share = if model.routed_bytes == 0 { 0.0 } else { widest as f64 / model.routed_bytes as f64 };
    if ranks > 0 {
        notes.push(format!("Spark: TP{ranks} native NVFP4 slices of routed layers {}..={}; widest rank {:.1} GiB",
            inputs.routed.get(onboarded).copied().unwrap_or(cfg.layers), cfg.layers - 1, widest as f64 / GIB));
    }
    report.solver_placement = Some(placed);
    MemoryLayout { devices, pool_tokens: report.solver_placement.as_ref().map_or(0, |p| p.pool_tokens),
        waste: Vec::new(), notes }
}
