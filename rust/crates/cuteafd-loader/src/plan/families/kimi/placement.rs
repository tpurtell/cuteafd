//! Header-only lower bound on 2 RTX + 6 Sparks through the shared pool solver;
//! not runtime admission (workspaces, graphs, transport and exact state come
//! with the engine).
use super::Model;
use crate::{placement::*, plan::{Component, ExpertPlacement, Hint, PlanReport}};
use cuteafd_core::memory_layout::{Basis, Category, DeviceKind, DeviceLayout, Item, MemoryLayout};

const GIB: u64 = 1 << 30;
const FLOOR: u64 = 262_144;
const PAGE: u64 = 64;
const MODES: &[LayerMode] = &[LayerMode::HeadSplit, LayerMode::ContextSplit];

pub(super) fn layout(model: &Model, report: &mut PlanReport, opts: &crate::plan::layout::LayoutOptions,
    requested: ExpertPlacement) -> MemoryLayout {
    let cfg = model.cfg();
    let ranks = requested.spark_ranks();
    let gpus = opts.rtx_bytes.len();
    let mut memory = MemoryLayout { devices: Vec::new(), pool_tokens: 0, waste: Vec::new(), notes: vec![
        "Kimi K3 header-only lower bound: no workspace/graph/prefix/transport allowance; NOT serving admission".into(),
    ] };
    let refusal = |report: &mut PlanReport, what: String, how: &str| {
        report.fits = false;
        report.placement_supported = false;
        report.hints.push(Hint { what, how: how.into() });
    };
    report.bytes_by_owner.clear();
    if !model.complete {
        refusal(report, "Kimi K3 placement cannot infer fit from missing safetensors headers".into(),
            "Place every shard's header locally (nest replicate) or plan from a complete snapshot.");
        return memory;
    }
    let experts = *model.bytes.get(&Component::RoutedExpert).unwrap_or(&0);
    let coordinator: u64 = model.bytes.iter()
        .filter(|(c, _)| !matches!(c, Component::RoutedExpert | Component::Vision)).map(|(_, b)| *b).sum();
    report.bytes_by_owner.insert("coordinator text weights".into(), coordinator);
    report.bytes_by_owner.insert("routed expert weights".into(), experts);
    // TP6 width 512: 1/6 of every trellis and of the 3072-wide rotations, but
    // whole copies of the 3584-wide input (w1/w3 suh) and output (w2 svh) rotations.
    let replicated = (cfg.layers - cfg.dense.iter().filter(|d| **d).count()) as u64 * cfg.experts as u64 * 3 * cfg.latent as u64 * 2;
    let per_rank = if ranks > 0 { (experts - replicated).div_ceil(ranks as u64) + replicated } else { experts };
    if ranks != 6 {
        refusal(report, format!("Kimi K3 needs the max card's 6 Spark ranks (requested {ranks}): TP{ranks} routed experts \
            need {:.1} GiB/rank", per_rank as f64 / GIB as f64),
            "Serve Kimi K3 on 2 RTX PRO 6000 + 6 Sparks (TP6 width 512); no other card column is targeted.");
    }
    let budget = opts.spark_allocation_budget_bytes.unwrap_or(super::SPARK_BUDGET_BYTES);
    for rank in 0..ranks {
        memory.devices.push(DeviceLayout { kind: DeviceKind::Spark, index: rank as u32, capacity_bytes: opts.spark_bytes,
            kv_tokens: 0, items: vec![Item::new(Category::Experts, "routed experts (TP6 width 512, replicated 3584 rotations)",
                model.expert_formats.join("+"), per_rank, Basis::Exact)] });
    }
    if ranks > 0 && per_rank > budget {
        refusal(report, format!("Kimi K3 TP{ranks} routed experts need {:.2} GiB/rank, {:.2} GiB over the {:.2} GiB Spark budget",
            per_rank as f64 / GIB as f64, (per_rank - budget) as f64 / GIB as f64, budget as f64 / GIB as f64),
            "Kimi K3's family budget is 108 GiB/rank; an explicit --spark-budget-gib below it refuses the max card.");
    }
    if gpus != 2 || !opts.head_split {
        refusal(report, "Kimi K3 targets two coordinator GPUs with a head split".into(),
            "Use --rtx 2; one RTX cannot hold 105.7 GiB of BF16 text weights.");
        return memory;
    }
    let requested_pool = opts.pool_tokens.filter(|n| *n > 0);
    if requested_pool.is_some_and(|n| n < FLOOR) {
        refusal(report, "Kimi K3 pool below the 256K agentic floor is not a card recipe".into(), "Leave the pool automatic.");
        return memory;
    }
    // Embedding on GPU0, head on GPU1; everything else splits evenly by heads
    // (a lower bound: replicated f_a, q_a/kv_a and norms add a few hundred MiB).
    let edge = |c| *model.bytes.get(&c).unwrap_or(&0);
    let (embedding, head) = (edge(Component::Embedding), edge(Component::LmHead));
    let body = coordinator - embedding - head;
    let slots = opts.concurrency.max(1) + 2;
    let kda = cfg.kda_layers() as u64;
    let state = kda * (cfg.kda_state_bytes() + cfg.kda_conv_bytes()) * slots / 2;
    let mut fixed = Vec::new();
    for gpu in 0..2u8 {
        fixed.push(Demand::new(gpu, Category::Weights, "text weights (head split)", body / 2, Basis::Formula));
        fixed.push(Demand::new(gpu, if gpu == 0 { Category::Embedding } else { Category::Weights },
            if gpu == 0 { "token embedding" } else { "lm head" }, if gpu == 0 { embedding } else { head }, Basis::Exact));
        fixed.push(Demand::new(gpu, Category::Kv, format!("KDA FP32 state + BF16 conv, {slots} slots, 48 heads"), state, Basis::Formula));
    }
    let record = (cfg.mla_record_values() * 2) as u64 * PAGE;
    let layers = (0..cfg.layers).map(|layer| {
        let mla = cfg.attention[layer] == crate::families::kimi_k3::config::KimiAttention::Mla;
        LayerDemand { kind: if mla { AttentionClass::Mla } else { AttentionClass::Kda },
            weights: ModeBytes::default(),
            kv_unit: KvDemand { unit_bytes_whole: if mla { record } else { 0 }, unit_bytes_split: if mla { [record; 2] } else { [0; 2] },
                unit_bytes_context: mla.then_some([record / 2; 2]) },
            fixed_bytes: ModeBytes::default(), extra: Vec::new(), context_indexer: false, colocate: None,
            experts: None, modes: if mla { MODES.to_vec() } else { vec![LayerMode::HeadSplit] } }
    }).collect();
    let request = PlacementRequest {
        inventory: Inventory { gpus: opts.rtx_bytes.iter().map(|bytes| GpuBudget { capacity_bytes: *bytes,
            headroom_bytes: opts.headroom_bytes, baseline: Baseline::Planned { context_bytes: 0, loaded_bytes: 0 } }).collect(),
            spark_ranks: ranks, peer_access: true },
        attention_placement: report.requested_attention, context_buffers: ContextBuffers::default(), layers_first_gpu: 0,
        pool: PoolPolicy { requested: requested_pool, target: opts.target_pool_tokens.max(FLOOR), floor: FLOOR,
            ceiling: opts.target_pool_tokens.max(requested_pool.unwrap_or(0)).max(FLOOR), unit_rows: PAGE },
        layers, pool_overhead: vec![0; 2], pool_bounded: Vec::new(), fixed, startup: Vec::new(), movables: Vec::new(),
        expert_workspace: 0, tp2_workspace: [0; 2], onboard: Onboard::Layers(0), expert_gpus: 0, minimum_expert_layers: 0,
        policy: LayerPolicy { default: vec![LayerMode::HeadSplit],
            attention: vec![(AttentionClass::Mla, AttentionPlacement::Context)], ..Default::default() },
        hops: HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: cfg.hidden as u64 * 2, rows: 0, lanes: 0,
            entry_gpu: 0, head_gpu: 1, terminal: TerminalHome::Replicated },
        executor: ExecutorModes { family: "kimi_k3 metadata only", modes: MODES, hops: false, attention: AttentionPlacement::Context },
    };
    match solve(&request) {
        Ok(placed) => {
            memory.pool_tokens = placed.pool_tokens;
            for gpu in 0..2 {
                memory.devices.push(DeviceLayout { kind: DeviceKind::Rtx, index: gpu as u32, capacity_bytes: opts.rtx_bytes[gpu],
                    items: placed.items[gpu].clone(), kv_tokens: placed.pool_tokens });
            }
            memory.notes.push(format!("MLA latent {} B/token over {} layers (context split halves it per GPU); \
                header lower bound pool {} tokens with no routed experts on RTX",
                cfg.mla_record_values() * 2 * cfg.mla_layers(), cfg.mla_layers(), placed.pool_tokens));
        }
        Err(error) => refusal(report, format!("Kimi K3 coordinator lower bound refused: {error}"),
            "Coordinator text weights are 105.7 GiB BF16; free KV room only by placement, not by a smaller pool floor."),
    }
    memory
}
