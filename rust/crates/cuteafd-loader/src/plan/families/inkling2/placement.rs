//! Inkling 2 `cuteafd plan --layout` over the shared contract planner (#1930
//! stage 1). Facts come from the one gatherer serving startup also calls
//! ([`crate::families::inkling2::admission::gather`]); the planned edge
//! differs only in its coordinator baseline. This module renders the solve.
use super::{ExpertStorage, Model};
use crate::families::inkling2::admission::{gather, Coordinator, FactsRequest};
use crate::placement::families::inkling2::LocalPackage;
use crate::placement::planner::{Operation, PrefillShape};
use crate::placement::{AttentionPlacement, Onboard, ResidentWeights};
use crate::plan::{ExpertPlacement, PlanReport};
use cuteafd_core::memory_layout::{Basis, Category, DeviceKind, DeviceLayout, Item, MemoryLayout};

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
        "Inkling 2 shared contract (#1930 stage 1): one solve over named physical owners; serving stays Planned".to_string(),
    ];
    if opts.host_embedding {
        report.refuse_placement("Inkling 2 does not support --embedding-placement host yet".into(),
            "Use --embedding-placement gpu; the shared host TokenEmbedding lands with the serving stage.");
        return empty(notes);
    }
    let limits = crate::plan::Family::prefill_limits(&super::INKLING2);
    let prefill = match crate::plan::layout::resolve_prefill(input, report, opts, ranks, limits) {
        Ok(prefill) => prefill,
        Err(error) => return crate::plan::layout::unsupported_prefill(report, error),
    };
    report.prefill_reserve_rows = Some(prefill.span_rows);
    let shape = PrefillShape { rows: prefill.unit_rows, waves: prefill.waves };
    // The shared mechanism resolves `auto` (heads on a pair: KV, rings and
    // lag rows partition by head with nothing replicated) and refuses modes
    // this executor does not list.
    let attention = match crate::placement::families::inkling2::INKLING2.check_attention(report.requested_attention, gpus, gpus == 2) {
        Ok(attention) => attention,
        Err(error) => {
            report.refuse_placement(format!("Inkling 2 attention placement refused: {error}"),
                "Use ATTENTION_PLACEMENT=auto, heads, context or layers on two RTX; one RTX runs heads.");
            return empty(notes);
        }
    };
    report.attention_placement = attention;
    // The program manifest is optional in stage 1: no Inkling 2 program is
    // installed yet, so the loaded-code allowance uses the shared estimate.
    let manifest_path = opts.workspace_manifest.clone().or_else(|| {
        let path = std::path::PathBuf::from("/opt/cuteafd/share/PROGRAMS.json");
        path.is_file().then_some(path) });
    let manifest = match manifest_path.as_deref().map(|path| input.manifest(path)).transpose() {
        Ok(manifest) => manifest,
        Err(error) => {
            report.refuse_placement(format!("Inkling 2 program manifest refused: {error:#}"), "Supply a readable PROGRAMS.json.");
            return empty(notes);
        }
    };
    let host = match opts.host_memory.map(Ok).unwrap_or_else(crate::formats::mapped_preload::host_memory) {
        Ok(memory) => memory,
        Err(error) => {
            report.refuse_placement(format!("Inkling 2 host inventory refused: {error:#}"), "Provide a host memory sample.");
            return empty(notes);
        }
    };
    let cards = opts.rtx_bytes.iter().map(|&bytes| crate::placement::CoordinatorCard::planned(bytes, opts.physical_sms))
        .collect::<Vec<_>>();
    let spark_bytes = opts.spark_allocation_budget_bytes.unwrap_or(crate::plan::DEFAULT_SPARK_BUDGET_BYTES)
        .min(opts.spark_bytes);
    let sequences = opts.resolved_concurrency("inkling2");
    let request = FactsRequest { checkpoint: input.checkpoint(), cfg, manifest_path: manifest_path.as_deref(),
        manifest, coordinator: Coordinator::Planned { cards: cards.clone() }, attention, spark_ranks: ranks,
        spark_bytes, prefill: shape, context_tokens: opts.context_tokens, pool_tokens: opts.pool_tokens, sequences,
        host, host_cache_bytes: opts.host_cache_bytes, prefix_entries: opts.mimo_prefix_entries,
        replies: opts.spark_reply_formats, head: opts.bf16_target_head, expert_usage: opts.expert_usage,
        resident_weight_limit: ResidentWeights(opts.weight_budget_bytes) };
    let inputs = match gather(&request) {
        Ok(inputs) => inputs,
        Err(refused) => { report.refuse_placement(refused.what, refused.how); return empty(notes); }
    };
    if let LocalPackage::Missing(reason) = &inputs.local {
        notes.push(format!("Inkling 2 RTX onboarding unavailable: {reason}"));
    }
    notes.push("Inkling 2 bundled dSpark drafter and ViT are declared recipe domains with their own arenas; this \
        build has no executor for either, so the plan charges neither".into());
    let routed = inputs.routed.len();
    let onboard = opts.resolve_onboard(routed).unwrap_or(Onboard::Auto);
    let solve_request = crate::placement::families::inkling2::request(inputs.context, opts.pool_tokens, onboard,
        shape, if opts.full_prefill_logits { Operation::Scoring } else { Operation::Serving },
        inputs.resident_weight_limit, opts.spark_reply_formats);
    let admitted = match inputs.admit(solve_request) {
        Ok(admitted) => admitted,
        Err(error) => {
            report.refuse_placement(format!("Inkling 2 joint admission refused: {error}"),
                "Use more Spark ranks or a larger --spark-memory-cap; RTX expert onboarding joins with its packages (stage 3).");
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
    let storages = inputs.experts.iter().filter_map(|layer| layer.storage).collect::<std::collections::BTreeSet<_>>();
    let storage_note = storages.iter().map(|storage| match storage {
        ExpertStorage::Mxfp8 => "MXFP8 W8A8".to_string(),
        ExpertStorage::Nvfp4 => "native NVFP4 W4A4".to_string(),
        ExpertStorage::Exl3 { bits } => format!("EXL3 K{bits}"),
    }).collect::<Vec<_>>().join(" + ");
    let mode_note = match (gpus, attention) {
        (1, _) => "every layer on rtx0",
        (_, AttentionPlacement::Heads) => "24/4 heads per GPU on every layer; KV pages, window rings and conv lags split by head, residual replicated",
        (_, AttentionPlacement::Context) => "global layers page-parity KV with the shared context exchange, window layers head-split",
        (_, AttentionPlacement::Layers) => "contiguous whole-layer owners (24/24, one boundary hop)",
    };
    notes.push(format!("Inkling 2 attention {}: {mode_note}; routed experts {}; pool {} tokens (global layers only; {} \
        sliding-window layers keep per-sequence rings); {onboarded} of {routed} routed layers onboarded; experts {storage_note}; \
        unit {} x {} waves; C{sequences}", placed.attention_summary(), if gpus == 2 { "TP2 halves" } else { "whole layers" },
        placed.pool_tokens, cfg.local_layers(), shape.rows, shape.waves));
    let mut devices = Vec::new();
    for gpu in 0..gpus {
        let mut device = DeviceLayout { kind: DeviceKind::Rtx, index: gpu as u32,
            capacity_bytes: cards[gpu].physical_bytes.min(cards[gpu].logical_bytes), worker_budget_bytes: 0,
            items: placed.items[gpu].clone(), kv_tokens: placed.pool_tokens };
        device.items.extend(crate::placement::reserve::gpu_item("inkling2"));
        notes.push(format!("rtx{gpu}: target arena {} B ({} views); attention layers {}",
            inputs.targets[gpu].arena.bytes(), inputs.targets[gpu].arena.views().len(),
            if inputs.targets[gpu].layers.len() == cfg.layers { "all (split)".to_string() } else { format!("{:?}", inputs.targets[gpu].layers) }));
        devices.push(device);
    }
    let mut widest = 0u64;
    for rank in 0..ranks {
        let charge = inputs.spark[onboarded][rank];
        widest = widest.max(charge.experts);
        devices.push(DeviceLayout { kind: DeviceKind::Spark, index: rank as u32, capacity_bytes: spark_bytes,
            worker_budget_bytes: spark_bytes, kv_tokens: 0, items: vec![
                Item::new(Category::Experts, "routed experts (header-exact intermediate slices)", &storage_note,
                    charge.experts, Basis::Exact),
                Item::new(Category::Workspace, "Spark rank I/O and execution owners", "", charge.owners,
                    Basis::Formula)] });
    }
    report.spark_rank_share = if model.routed_bytes == 0 { 0.0 } else { widest as f64 / model.routed_bytes as f64 };
    if ranks > 0 {
        notes.push(format!("Spark: TP{ranks} slices of routed layers {}..={}; widest rank {:.1} GiB; no Inkling 2 Spark \
            package ships yet, so the world is a planner result, not a package qualification (#1930 stage 3)",
            inputs.routed.get(onboarded).copied().unwrap_or(cfg.layers), cfg.layers - 1, widest as f64 / GIB));
    }
    report.solver_placement = Some(placed);
    MemoryLayout { devices, pool_tokens: report.solver_placement.as_ref().map_or(0, |p| p.pool_tokens),
        waste: Vec::new(), notes }
}
