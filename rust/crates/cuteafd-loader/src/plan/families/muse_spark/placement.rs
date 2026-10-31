//! Muse Spark admission over the shared solver: header-exact coordinator
//! weights (K6 core payload + H128 vectors, BF16/F32 rest), paged full-layer
//! KV plus per-sequence sliding rings, pool first, then RTX expert onboarding
//! of whole routed layers; Spark ranks hold whole experts partitioned by the
//! shared owner table (expert parallel).
//!
//! Header-only: coordinator graphs and step workspaces come from the family's
//! installed PROGRAMS.json once its programs exist; until then this is a
//! planning lower bound (stated in the notes), not serving admission.
use super::Model;
use crate::{placement::*, plan::{Component, ExpertPlacement, Hint, PlanReport}};
use cuteafd_core::memory_layout::{Basis, Category, DeviceKind, DeviceLayout, Item, MemoryLayout};

const GIB: u64 = 1 << 30;
const PAGE: u64 = 64;
/// Spark worlds whole-expert ownership serves: every transport world whose
/// ranks each own at least one expert (the EP program is world-independent).
pub(super) const EP_WORLDS: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
/// Prefill transport endpoints: one shared link.
const LANES: u64 = 1;
const MODES: &[LayerMode] = &[LayerMode::HeadSplit];
const SINGLE: &[LayerMode] = &[LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }];

pub(super) fn layout(model: &Model, input: &crate::plan::PlanInput, report: &mut PlanReport,
    opts: &mut crate::plan::layout::LayoutOptions, requested: ExpertPlacement) -> MemoryLayout {
    let limits = crate::plan::Family::prefill_limits(&super::MUSE_SPARK);
    let prefill = match crate::plan::layout::resolve_prefill(report, opts, requested.spark_ranks(), limits) {
        Ok(prefill) => prefill,
        Err(error) => return crate::plan::layout::unsupported_prefill(report, error),
    };
    let cfg = model.cfg();
    let ranks = requested.spark_ranks();
    let gpus = opts.rtx_bytes.len();
    let mut memory = MemoryLayout { devices: Vec::new(), pool_tokens: 0, waste: Vec::new(), notes: vec![
        "Muse Spark header lower bound: coordinator graphs and step workspaces are charged from the family's \
            PROGRAMS.json once its programs exist; NOT serving admission".into(),
    ] };
    let refusal = |report: &mut PlanReport, what: String, how: &str| {
        report.fits = false;
        report.placement_supported = false;
        report.hints.push(Hint { what, how: how.into() });
    };
    report.bytes_by_owner.clear();
    report.attention_placement = AttentionPlacement::Heads;
    let context = if opts.context_tokens == 0 { cfg.max_position as u64 } else { opts.context_tokens };
    if context > cfg.max_position as u64 {
        refusal(report, format!("Muse Spark context {context} exceeds the checkpoint's {}", cfg.max_position),
            "Lower --context-tokens to at most 1M.");
        return memory;
    }
    if !model.complete {
        refusal(report, "Muse Spark placement cannot infer fit from missing safetensors headers".into(),
            "Place every shard's header locally (nest replicate) or plan from a complete snapshot.");
        return memory;
    }
    if !model.expert_formats.iter().any(|f| f.starts_with("exl3-k")) {
        refusal(report, format!("Muse Spark routed experts are {:?}; only the EXL3 publication has an expert package",
            model.expert_formats), "Serve the EXL3 K4 + Core-K6 snapshot.");
        return memory;
    }
    if ranks == 0 {
        refusal(report, "Muse Spark routed experts (~400 GB at K4) need Spark ranks".into(),
            "Use --spark-ranks 4..8.");
        return memory;
    }
    if !(1..=2).contains(&gpus) || (gpus == 2 && !opts.head_split) {
        refusal(report, "Muse Spark runs on one RTX or a head-split RTX pair".into(), "Use --rtx 1 or 2.");
        return memory;
    }
    // Coordinator text weights by residency: head-split attention halves, the
    // rest whole on each GPU (MoE projections, router, norms, gates).
    let mut split = 0u64;
    let mut replicated = 0u64;
    for layer in 0..cfg.layers {
        let b = format!("{}layers.{layer}", super::TEXT);
        for leaf in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "self_attn.gate_proj"] {
            let whole = model.resident(&format!("{b}.{leaf}"), 1);
            let half = model.resident(&format!("{b}.{leaf}"), 2);
            split += whole;
            replicated += half.saturating_mul(2).saturating_sub(whole);
        }
        for leaf in ["mlp.pre_expert_proj", "mlp.post_expert_proj", "mlp.gate.weight", "mlp.gate.e_score_correction_bias",
            "mlp.pre_expert_norm.weight", "mlp.experts.post_expert_norm.weight", "input_layernorm.weight",
            "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight",
            "post_attention_residual_gate.gate", "post_feedforward_residual_gate.gate"] {
            replicated += model.resident(&format!("{b}.{leaf}"), 1);
        }
    }
    let embedding = *model.bytes.get(&Component::Embedding).unwrap_or(&0);
    let head = model.resident("lm_head", 1);
    let experts = *model.bytes.get(&Component::RoutedExpert).unwrap_or(&0);
    report.bytes_by_owner.insert("coordinator text weights".into(), embedding + head + split + replicated);
    report.bytes_by_owner.insert("routed expert weights".into(), experts);
    memory.notes.push(format!("Muse K6 core: {} header-native projections ({}); payload + H128 vectors charged",
        model.core.projections.len(), if model.core.complete(cfg) { "complete" } else { "partial: BF16 rest" }));

    // KV: full layers page BF16 K+V records; sliding layers keep a per-sequence
    // ring of window + one prefill unit (append-only, exact restores).
    let slots = opts.state_slots.unwrap_or(if opts.concurrency > 0 { opts.concurrency } else { 16 });
    let record = cfg.kv_record_bytes() as u64 * PAGE;
    let ring = cuteafd_core::prefix::AppendRing::new(cfg.window as u64, prefill.unit_rows.max(PAGE), PAGE)
        .map_or(0, |ring| ring.rows) * cfg.kv_record_bytes() as u64;
    let rings = cfg.sliding_layers() as u64 * ring;
    let marks = (cfg.sliding_layers() * cfg.window * cfg.kv_record_bytes()) as u64;
    let split_gpus = gpus as u64;
    let mut fixed = Vec::new();
    for gpu in 0..gpus as u8 {
        fixed.push(Demand::new(gpu, Category::Weights, "text weights, head-split attention",
            split / split_gpus, Basis::Exact));
        fixed.push(Demand::new(gpu, Category::Weights,
            "text weights replicated (MoE projections, router, norms, residual gates, H128 vectors)", replicated, Basis::Exact));
        fixed.push(Demand::new(gpu, Category::Kv, format!("sliding rings, {slots} sequences x {} layers",
            cfg.sliding_layers()), rings * slots / split_gpus, Basis::Formula));
        // One exact retained sliding window per slot (prompt end + turn end marks).
        fixed.push(Demand::new(gpu, Category::Prefix, "Muse exact sliding marks", marks * slots / split_gpus, Basis::Formula));
    }
    fixed.push(Demand::new(0, Category::Embedding, "token embedding", embedding, Basis::Exact));
    fixed.push(Demand::new((gpus - 1) as u8, Category::Weights, "lm head (K6)", head, Basis::Exact));
    // Spark returns land on the coordinator: one prefill span of expert-width partials per rank.
    fixed.push(Demand::new(0, Category::Transport, "Spark intake planes",
        crate::serving_capacity::prefill_return_bytes(prefill.waves, prefill.unit_rows, ranks as u64,
            cfg.expert_hidden as u64), Basis::Formula));

    // Spark: whole experts partitioned by the shared owner table.
    let owners = match cuteafd_core::ExpertOwners::staggered(cfg.layers, cfg.experts, ranks) {
        Ok(owners) => owners,
        Err(error) => { refusal(report, format!("Muse whole-expert ownership: {error}"), "Use 1..=8 Spark ranks."); return memory; }
    };
    let inventory = (|| -> anyhow::Result<_> {
        let catalog = input.expert_catalog()?;
        let working = SparkWorkingSet { rtx_layers: vec![], spark_layers: (0..cfg.layers).collect(),
            spark_ranks: ranks, prefill_lanes: LANES };
        let rings = crate::plan::layout::spark_ring_geometry(LANES as usize);
        let endpoints = [cuteafd_core::SparkEndpoint { ring_depth: None, capacity: prefill.unit_rows as u32,
            ingress: cuteafd_core::SparkIngress::Fp8K32 }];
        let budget = opts.spark_allocation_budget_bytes.unwrap_or(crate::plan::DEFAULT_SPARK_BUDGET_BYTES);
        let reserve = spark_reserve_bytes(prefill.waves, prefill.unit_rows)?;
        // The checkpoint's per-expert output norm makes the residency whole experts.
        anyhow::ensure!(catalog.whole_expert_outputs(), "Muse Spark experts must carry their post-expert norm");
        let spark = SparkBudgetInventory::exl3(catalog, &working, &endpoints,
            cuteafd_core::SparkRings { max_frame_bytes: rings.max_frame_bytes, depth: rings.depth,
                slot_bytes: rings.slot_bytes, alignment: rings.alignment }, budget, reserve)?;
        // An onboarded layer is whole on one GPU: the per-expert norm needs each
        // expert's complete output, so TP2 halves would be inexact.
        let manifest = catalog.exl3().ok_or_else(|| anyhow::anyhow!("Muse Spark requires EXL3 expert headers"))?;
        let mut costs = Vec::with_capacity(cfg.layers);
        for layer in 0..cfg.layers {
            costs.push(manifest.residency(crate::V41Exl3Layer::Backbone(layer), 1, 0)?.device_arena_layout()?.1 as u64);
        }
        Ok((spark, costs))
    })();
    let (spark_budget, costs) = match inventory {
        Ok(value) => value,
        Err(error) => {
            refusal(report, format!("Muse Spark expert inventory refused: {error:#}"), "Provide complete EXL3 expert headers.");
            return memory;
        }
    };
    let modes = if gpus == 2 { MODES } else { SINGLE };
    let layers = (0..cfg.layers).map(|layer| {
        let full = cfg.attention[layer] == crate::families::muse_spark::MuseAttention::Full;
        let whole = costs[layer];
        LayerDemand {
            kind: if full { AttentionClass::Gqa } else { AttentionClass::Swa },
            weights: ModeBytes::default(),
            kv_unit: if full {
                KvDemand { unit_bytes_whole: record, unit_bytes_split: [record / 2; 2], unit_bytes_context: None }
            } else { KvDemand::default() },
            fixed_bytes: ModeBytes::default(), extra: Vec::new(), context_indexer: false, colocate: None,
            experts: Some(ExpertCost { whole: Bytes2 { resident: whole, staging: 0 },
                half: [Bytes2::default(); 2], tp2: false, spark_ok: true }),
            modes: modes.to_vec(),
        }
    }).collect::<Vec<_>>();
    let requested_pool = opts.pool_tokens.filter(|n| *n > 0);
    // An automatic context clamps to the pool the Sparks leave room for.
    let pool = PoolPolicy::resolve(&opts.rtx_bytes, context, requested_pool, PAGE, false)
        .with_automatic_context(opts.context_tokens == 0);
    let mut request = PlacementRequest {
        inventory: Inventory { gpus: opts.rtx_bytes.iter().map(|bytes| GpuBudget { capacity_bytes: *bytes,
            headroom_bytes: opts.headroom_bytes, baseline: Baseline::Planned {
                context_bytes: ArchContext::for_device("sm_120", *bytes).context_bytes, loaded_bytes: 0 } }).collect(),
            spark_ranks: ranks, peer_access: gpus == 2 },
        attention_placement: (gpus == 2).then_some(AttentionPlacement::Heads), context_buffers: ContextBuffers::default(),
        layers_first_gpu: 0, pool, layers, pool_overhead: vec![0; gpus], pool_bounded: Vec::new(), fixed,
        startup: Vec::new(), movables: Vec::new(), expert_workspace: 0, tp2_workspace: [0; 2],
        onboard: opts.onboard.unwrap_or(Onboard::Auto), expert_gpus: gpus, minimum_expert_layers: 0,
        // Whole onboarded layers alternate between the GPUs.
        policy: LayerPolicy { default: modes.to_vec(), ownership: LayerOwnership::Ranges,
            expert_ownership: if gpus == 2 { ExpertOwnership::AlternatingWhole } else { ExpertOwnership::Contiguous },
            ..Default::default() },
        hops: HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: cfg.hidden as u64 * 2, rows: 0, lanes: 0,
            entry_gpu: 0, head_gpu: (gpus - 1) as u8, terminal: if gpus == 2 { TerminalHome::Replicated }
                else { TerminalHome::AttentionOwner } },
        executor: ExecutorModes { family: "muse_spark", modes, hops: false, attention: AttentionPlacement::Heads },
    };
    if let Err(error) = spark_budget.constrain(&mut request) {
        refusal(report, format!("Muse Spark joint admission refused: {error:#}"),
            "Use more Spark ranks, automatic RTX expert selection, or a larger --spark-budget-gib.");
        return memory;
    }
    let placed = match solve(&request) {
        Ok(placed) => placed,
        Err(error) => {
            refusal(report, format!("Muse Spark coordinator admission refused: {error}"),
                "Free KV room by placement or add a second RTX; the pool floor is the compiled context.");
            return memory;
        }
    };
    memory.pool_tokens = placed.pool_tokens;
    memory.notes.push(format!("Muse selected RTX expert layers: {}; pool {} tokens; full-layer KV {} B/token over {} \
        layers; sliding rings {} B/sequence over {} layers", placed.onboard_layers, placed.pool_tokens,
        cfg.kv_record_bytes() * cfg.full_layers(), cfg.full_layers(), rings, cfg.sliding_layers()));
    for gpu in 0..gpus {
        let mut device = DeviceLayout { kind: DeviceKind::Rtx, index: gpu as u32, capacity_bytes: opts.rtx_bytes[gpu],
            items: placed.items[gpu].clone(), kv_tokens: placed.pool_tokens };
        device.items.push(Item::new(Category::Reserved, "coordinator headroom", "", opts.headroom_bytes, Basis::Formula)
            .with_lifetime(cuteafd_core::memory_layout::ItemLifetime::PostReadyReserve));
        memory.devices.push(device);
    }
    let onboarded = placed.onboard_layers;
    report.solver_placement = Some(placed);
    let working = SparkWorkingSet { rtx_layers: Vec::new(), spark_layers: spark_budget.layers[onboarded..].to_vec(),
        spark_ranks: ranks, prefill_lanes: LANES };
    let rings = crate::plan::layout::spark_ring_geometry(LANES as usize);
    let mut widest = 0;
    for rank in 0..ranks {
        let resident = spark_budget.resident[rank][onboarded..].iter().sum::<u64>();
        widest = widest.max(resident);
        let mut device = DeviceLayout { kind: DeviceKind::Spark, index: rank as u32, capacity_bytes: opts.spark_bytes,
            kv_tokens: 0, items: vec![Item::new(Category::Experts, "routed experts (whole experts, expert parallel)",
                model.expert_formats.join("+"), resident, Basis::Exact)] };
        if let Err(error) = crate::plan::layout::charge_spark_io(&mut device, &working, cuteafd_core::ExpertGeometry::MUSE_SPARK,
            prefill.unit_rows as u32, false, rings, 0) {
            refusal(report, format!("Muse Spark rank {rank} I/O: {error}"), "Fix the Spark ring geometry.");
        }
        if let Err(error) = muse_owners(&mut device, &working, prefill.unit_rows, prefill.waves) {
            refusal(report, format!("Muse Spark execution owners: {error:#}"), "Use a supported wave count.");
        }
        memory.devices.push(device);
    }
    report.spark_rank_share = if experts == 0 { 0.0 } else { widest as f64 / experts as f64 };
    memory.notes.push(format!("Spark: whole experts over {ranks} ranks (owner table {}), widest rank {:.1} GiB of {:.1} GiB routed",
        &owners.fingerprint()[..12], widest as f64 / GIB as f64, experts as f64 / GIB as f64));
    memory
}

/// Muse's typed Spark owners on the EXL3 ladder.
pub(crate) fn spark_executions(waves: u64, unit_rows: u64) -> anyhow::Result<cuteafd_core::SparkExecutions> {
    let exports = crate::plan::spark_executions::SparkBackend::Exl3(cuteafd_core::Exl3RowPolicy::Nearest).exports();
    crate::plan::spark_executions::admit_family("muse_spark", &exports, None, unit_rows, waves)
}

/// The pre-placement Spark reserve: the runtime residual plus the typed owners.
fn spark_reserve_bytes(waves: u64, unit_rows: u64) -> anyhow::Result<u64> {
    let owners = spark_executions(waves, unit_rows)?.workspace_bytes(owner_bytes)?
        .and_then(|bytes| bytes.total()).ok_or_else(|| anyhow::anyhow!("Muse Spark owner reserve overflow"))?;
    Ok(cuteafd_core::memory_layout::SPARK_RUNTIME_RESIDUAL_BYTES + owners)
}

fn owner_bytes(extent: cuteafd_core::OwnerExtent) -> anyhow::Result<cuteafd_core::OwnerBytes> {
    crate::plan::layout::spark_slots::exl3_owner_bytes("muse_spark", cuteafd_core::ExpertGeometry::MUSE_SPARK, extent)
}

fn muse_owners(device: &mut DeviceLayout, working: &SparkWorkingSet, unit_rows: u64, waves: u64) -> anyhow::Result<()> {
    if !working.selected_ranks().contains(&(device.index as usize)) { return Ok(()); }
    crate::plan::layout::spark_slots::charge_owners(device, &spark_executions(waves, unit_rows)?, owner_bytes)
}
