//! Header lower bound on 2 RTX + 6 Sparks through the shared pool solver: real
//! TP2 residency, Spark runtime/IO charges and optional whole-layer RTX expert
//! onboarding and retained marks. Not runtime admission: coordinator workspaces,
//! graphs and the EXL3 workspace of the final export set come with the engine.
use super::Model;
use crate::{placement::*, plan::{Component, ExpertPlacement, Hint, PlanReport}};
use cuteafd_core::memory_layout::{Basis, Category, DeviceKind, DeviceLayout, Item, MemoryLayout};

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;
const FLOOR: u64 = 262_144;
const PAGE: u64 = 64;
const MODES: &[LayerMode] = &[LayerMode::HeadSplit,
    LayerMode::Whole { gpu: 0, ffn: FfnMode::Split }, LayerMode::Whole { gpu: 1, ffn: FfnMode::Split }];
/// Spark EXL3 workspace across the capacity set, estimated until the Kimi
/// package manifest records it (Gemini review of #273: ~430 MiB at m4096).
const SPARK_EXL3_WORKSPACE: u64 = 512 * MIB;
/// Prefill lanes the Spark transport serves (one QP endpoint each).
const LANES: u64 = 2;

pub(super) fn layout(model: &Model, report: &mut PlanReport, opts: &crate::plan::layout::LayoutOptions,
    requested: ExpertPlacement) -> MemoryLayout {
    let cfg = model.cfg();
    let ranks = requested.spark_ranks();
    let gpus = opts.rtx_bytes.len();
    let mut memory = MemoryLayout { devices: Vec::new(), pool_tokens: 0, waste: Vec::new(), notes: vec![
        "Kimi K3 header lower bound: coordinator workspaces/graphs not charged; NOT serving admission".into(),
        "Spark budgets are offline estimates (including an explicit override), not load/steady census-qualified admission".into(),
    ] };
    let refusal = |report: &mut PlanReport, what: String, how: &str| {
        report.fits = false;
        report.placement_supported = false;
        report.hints.push(Hint { what, how: how.into() });
    };
    report.bytes_by_owner.clear();
    report.attention_placement = report.requested_attention.unwrap_or(AttentionPlacement::Layers);
    if report.requested_attention == Some(AttentionPlacement::Context) {
        refusal(report, "Kimi K3 context attention placement has no owned-page query exchange/LSE-combine ABI".into(),
            "Use layers (owned MLA with split KDA/FFNs) or explicitly select heads; context requires a new native ABI.");
        return memory;
    }
    if !model.complete {
        refusal(report, "Kimi K3 placement cannot infer fit from missing safetensors headers".into(),
            "Place every shard's header locally (nest replicate) or plan from a complete snapshot.");
        return memory;
    }
    let experts = *model.bytes.get(&Component::RoutedExpert).unwrap_or(&0);
    let routed_layers = cfg.dense.iter().filter(|d| !**d).count() as u64;
    let layer_bytes = experts / routed_layers;
    // TP6 width 512: 1/6 of every trellis and of the 3072-wide rotations, whole
    // copies of the 3584-wide input (w1/w3 suh) and output (w2 svh) rotations.
    let replicated_rotations = cfg.experts as u64 * 3 * cfg.latent as u64 * 2;
    let rank_layer = |world: u64| (layer_bytes - replicated_rotations).div_ceil(world) + replicated_rotations;
    if ranks != 6 {
        refusal(report, format!("Kimi K3 needs the max card's 6 Spark ranks (requested {ranks}): TP{ranks} routed experts \
            need {:.1} GiB/rank", (rank_layer(ranks.max(1) as u64) * routed_layers) as f64 / GIB as f64),
            "Serve Kimi K3 on 2 RTX PRO 6000 + 6 Sparks (TP6 width 512); no other card column is targeted.");
        return memory;
    }
    if gpus != 2 || !opts.head_split {
        refusal(report, "Kimi K3 targets two coordinator GPUs with split KDA and FFNs".into(),
            "Use --rtx 2; one RTX cannot hold 105.7 GiB of BF16 text weights.");
        return memory;
    }
    let requested_pool = opts.pool_tokens.filter(|n| *n > 0);
    let edge = |c| *model.bytes.get(&c).unwrap_or(&0);
    let (embedding, head) = (edge(Component::Embedding), edge(Component::LmHead));
    let residency = model.coordinator;
    let mla_weights: u64 = model.mla.iter().map(|r| r.split + r.replicated).sum();
    report.bytes_by_owner.insert("coordinator text weights".into(), embedding + head + residency.split + residency.replicated + mla_weights);
    report.bytes_by_owner.insert("routed expert weights".into(), experts);
    let slots = opts.concurrency.max(1) + 2;
    let state = cfg.kda_layers() as u64 * (cfg.kda_state_bytes() + cfg.kda_conv_bytes()) * slots / 2;
    let mark_bytes = cfg.kda_layers() as u64 * (cfg.kda_state_bytes() + cfg.kda_conv_bytes()) / 2;
    let legacy_marks = opts.prefix_slots.unwrap_or_else(|| cuteafd_core::prefix::mark_slots_for(
        opts.concurrency, opts.mimo_prefix_entries, 2 * mark_bytes, opts.mimo_prefix_mark_bytes));
    let mark_demands = (0..2).map(|gpu| Demand::new(gpu, Category::Prefix,
        "Kimi exact KDA marks", mark_bytes, Basis::Formula)).collect::<Vec<_>>();
    let mut fixed = Vec::new();
    for demand in &mark_demands {
        let mut demand = demand.clone();
        let Some(bytes) = demand.bytes.checked_mul(legacy_marks) else {
            refusal(report, "Kimi mark reservation overflow".into(), "Reduce prefix retention.");
            return memory;
        };
        demand.bytes = bytes;
        fixed.push(demand);
    }
    for gpu in 0..2u8 {
        fixed.push(Demand::new(gpu, Category::Weights, "text weights, head/intermediate split", residency.split / 2, Basis::Exact));
        fixed.push(Demand::new(gpu, Category::Weights, "text weights replicated (KDA low-rank A, routers, latent proj, norms)",
            residency.replicated, Basis::Exact));
        fixed.push(Demand::new(gpu, if gpu == 0 { Category::Embedding } else { Category::Weights },
            if gpu == 0 { "token embedding" } else { "lm head" }, if gpu == 0 { embedding } else { head }, Basis::Exact));
        fixed.push(Demand::new(gpu, Category::Kv, format!("KDA FP32 state + BF16 conv, {slots} slots, 48 heads"), state, Basis::Formula));
    }
    let record = (cfg.mla_record_values() * 2) as u64 * PAGE;
    let cost = ExpertCost { whole: Bytes2 { resident: layer_bytes, staging: 0 },
        half: [Bytes2 { resident: layer_bytes / 2, staging: 0 }; 2], tp2: false, spark_ok: true };
    let layers = (0..cfg.layers).map(|layer| {
        let mla = cfg.attention[layer] == crate::families::kimi_k3::config::KimiAttention::Mla;
        LayerDemand { kind: if mla { AttentionClass::Mla } else { AttentionClass::Kda },
            weights: ModeBytes { whole: model.mla[layer].split + model.mla[layer].replicated,
                split: [model.mla[layer].split / 2 + model.mla[layer].replicated; 2] },
            kv_unit: KvDemand { unit_bytes_whole: if mla { record } else { 0 }, unit_bytes_split: if mla { [record; 2] } else { [0; 2] },
                unit_bytes_context: None },
            fixed_bytes: ModeBytes::default(), extra: Vec::new(), context_indexer: false, colocate: None,
            experts: (!cfg.dense[layer]).then_some(cost), modes: if mla { MODES.to_vec() } else { vec![LayerMode::HeadSplit] } }
    }).collect();
    // Routed experts stay on the Sparks unless the operator onboards layers
    // (--rtx-expert-layers N|auto|max): rtx-tp1, a whole layer on one GPU.
    let onboard = opts.onboard.unwrap_or(Onboard::Layers(0));
    let request = PlacementRequest {
        inventory: Inventory { gpus: opts.rtx_bytes.iter().map(|bytes| GpuBudget { capacity_bytes: *bytes,
            headroom_bytes: opts.headroom_bytes, baseline: Baseline::Planned { context_bytes: 0, loaded_bytes: 0 } }).collect(),
            spark_ranks: ranks, peer_access: true },
        attention_placement: report.requested_attention, context_buffers: ContextBuffers::default(), layers_first_gpu: 0,
        pool: PoolPolicy { compiled_context: opts.context_tokens,
            requested: requested_pool, target: opts.target_pool_tokens.max(FLOOR), floor: FLOOR,
            ceiling: opts.target_pool_tokens.max(requested_pool.unwrap_or(0)).max(FLOOR), unit_rows: PAGE },
        layers, pool_overhead: vec![0; 2], pool_bounded: Vec::new(), fixed, startup: Vec::new(), movables: Vec::new(),
        expert_workspace: 0, tp2_workspace: [0; 2], onboard, expert_gpus: 2, minimum_expert_layers: 0,
        policy: LayerPolicy { default: vec![LayerMode::HeadSplit], ownership: LayerOwnership::Alternating,
            attention: vec![(AttentionClass::Mla, AttentionPlacement::Layers), (AttentionClass::Kda, AttentionPlacement::Heads)],
            ..Default::default() },
        hops: HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: cfg.hidden as u64 * 2, rows: 0, lanes: 0,
            entry_gpu: 0, head_gpu: 1, terminal: TerminalHome::Replicated },
        executor: ExecutorModes { family: "kimi_k3 metadata only", modes: MODES, hops: true, attention: AttentionPlacement::Layers },
    };
    let selected = if legacy_marks == 0 {
        solve(&request).map(|placed| (placed, MarkSelection { legacy_slots: 0, slots: 0, reduction_reason: None }))
    } else { solve_with_marks(&request, legacy_marks, 1, &mark_demands) };
    let onboarded = match selected {
        Ok((placed, selection)) => {
            memory.notes.push(selection.summary());
            memory.pool_tokens = placed.pool_tokens;
            report.attention_placement = placed.attention_placement;
            for gpu in 0..2 {
                let mut device = DeviceLayout { kind: DeviceKind::Rtx, index: gpu as u32, capacity_bytes: opts.rtx_bytes[gpu],
                    items: placed.items[gpu].clone(), kv_tokens: placed.pool_tokens };
                // The solver subtracts this guard from availability, not from its item ledger.
                device.items.push(Item::new(Category::Reserved, "coordinator headroom", "", opts.headroom_bytes, Basis::Formula)
                    .with_lifetime(cuteafd_core::memory_layout::ItemLifetime::PostReadyReserve));
                let owned = placed.layers.iter().enumerate().filter_map(|(layer, assignment)|
                    matches!(assignment.mode, LayerMode::Whole { gpu: owner, .. } if usize::from(owner) == gpu).then_some(layer))
                    .collect::<Vec<_>>();
                let categories = device.by_category();
                memory.notes.push(format!("RTX{gpu}: owned MLA layers {owned:?}; weights {} B, KV/state {} B, total {} B, \
                    reserved {} B, remaining {} B (header lower bound)",
                    categories.get(&Category::Weights).unwrap_or(&0) + categories.get(&Category::Embedding).unwrap_or(&0),
                    categories.get(&Category::Kv).unwrap_or(&0), device.used_bytes(), opts.headroom_bytes, device.free_bytes()));
                memory.devices.push(device);
            }
            memory.notes.push(format!("MLA latent {} B/token over {} layers; {} placement, KDA and FFNs split TP2; \
                pool {} tokens with {} routed layers on RTX ({:.2} GiB each)", cfg.mla_record_values() * 2 * cfg.mla_layers(),
                cfg.mla_layers(), placed.attention_placement, placed.pool_tokens, placed.onboard_layers, layer_bytes as f64 / GIB as f64));
            placed.onboard_layers as u64
        }
        Err(error) => {
            refusal(report, format!("Kimi K3 coordinator lower bound refused: {error}"),
                "Coordinator text weights are 105.7 GiB BF16; free KV room by placement, not a smaller pool floor.");
            0
        }
    };
    // Sparks: resident slices of the routed layers left on them, plus the
    // shared runtime residual, latent-width (3584) I/O and EXL3 workspace.
    let resident = rank_layer(ranks as u64) * (routed_layers - onboarded);
    let working = SparkWorkingSet { rtx_layers: Vec::new(), spark_layers: vec![0], spark_ranks: ranks, prefill_lanes: LANES };
    let rings = crate::plan::layout::spark_ring_geometry(LANES as usize);
    let budget = opts.spark_allocation_budget_bytes.unwrap_or(super::SPARK_BUDGET_BYTES);
    for rank in 0..ranks {
        let mut device = DeviceLayout { kind: DeviceKind::Spark, index: rank as u32, capacity_bytes: opts.spark_bytes,
            kv_tokens: 0, items: vec![Item::new(Category::Experts, "routed experts (TP6 width 512, replicated 3584 rotations)",
                model.expert_formats.join("+"), resident, Basis::Exact)] };
        if let Err(error) = crate::plan::layout::charge_spark_io(&mut device, &working, cuteafd_core::ExpertGeometry::KIMI_K3,
            opts.spark_capacity_rows as u32, false, rings, 0) {
            refusal(report, format!("Kimi K3 Spark rank {rank} I/O: {error}"), "Fix the Spark ring geometry.");
        }
        device.items.push(Item::new(Category::Workspace, "EXL3 expert workspace", "kimik3:exl3-k23", SPARK_EXL3_WORKSPACE, Basis::Estimated));
        memory.devices.push(device);
    }
    if resident > budget {
        refusal(report, format!("Kimi K3 TP{ranks} routed experts need {:.2} GiB/rank, {:.2} GiB over the {:.2} GiB Spark budget",
            resident as f64 / GIB as f64, (resident - budget) as f64 / GIB as f64, budget as f64 / GIB as f64),
            "Kimi K3's family budget is 108 GiB/rank; onboard routed layers to the RTX cards (--rtx-expert-layers N) \
             or raise --spark-budget-gib from a measured census.");
    }
    memory.notes.push(format!("Spark rank: {:.2} GiB routed weights ({} of {} layers) + runtime residual, FP8 K32 latent I/O \
        and EXL3 workspace; each RTX-onboarded layer frees {:.2} GiB per rank", resident as f64 / GIB as f64,
        routed_layers - onboarded, routed_layers, rank_layer(ranks as u64) as f64 / GIB as f64));
    memory
}
