//! Muse Spark admission request: one builder for the planner (planned
//! baselines) and serving startup (measured baselines plus the runtime's code,
//! workspace and exchange demands), so plan == runtime by construction.
//!
//! Coordinator: header-exact per-rank layer weights from the shared
//! [`LayerLayout`] the device packs, paged full-layer KV, per-sequence sliding
//! rings and exact sliding marks, pool first, then whole routed layers
//! onboarded on the RTXs (the per-expert output norm needs each expert's
//! complete output, so an onboarded layer is never TP2-halved). Sparks hold
//! whole experts partitioned by the shared owner table (expert parallel).
use crate::families::muse_spark::{core::CoreInventory, weight_layout::{GlobalLayout, LayerLayout}, MuseAttention,
    MuseSparkConfig};
use crate::placement::*;
use crate::serving_capacity::prefill_waves::PrefillPlan;
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{Basis, Category};

pub const PAGE_ROWS: u64 = 64;
/// Prefill transport endpoints: one shared link.
pub const LANES: u64 = 1;
const HEAD_SPLIT: &[LayerMode] = &[LayerMode::HeadSplit];
const WHOLE: &[LayerMode] = &[LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }];
/// Every layer head-split on an RTX pair, or whole on one RTX.
pub const MUSE_SPARK: ExecutorModes = ExecutorModes { family: "muse_spark",
    modes: &[LayerMode::HeadSplit, LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }], hops: false,
    attention: AttentionPlacement::Heads };

/// Program namespaces a coordinator layout binds (`muse_` geometry-free and
/// whole-head programs; `muse2_` the head-split halves).
pub fn program_families(gpus: usize) -> &'static [&'static str] {
    if gpus == 2 { &["muse", "muse2"] } else { &["muse"] }
}

/// Native code a rank holds beyond its CUDA context once ready: offline the
/// full allowance (`pending` = allowance), live what the sample has not yet
/// loaded. One group name on both edges.
pub fn code_demand(rank: usize, pending: u64) -> Result<Demand> {
    Ok(Demand::new(u8::try_from(rank)?, Category::Workspace, "Muse future loaded code", pending, Basis::Formula))
}

/// The executor and the layer modes one coordinator count runs.
pub fn executor(gpus: usize) -> Result<(ExecutorModes, &'static [LayerMode])> {
    match gpus {
        1 => Ok((MUSE_SPARK, WHOLE)),
        2 => Ok((MUSE_SPARK, HEAD_SPLIT)),
        other => anyhow::bail!("Muse Spark runs on one RTX or a head-split RTX pair, not {other}"),
    }
}

pub struct MuseInputs<'a> {
    pub cfg: &'a MuseSparkConfig,
    pub core: &'a CoreInventory,
    pub catalog: &'a crate::OfficialV41Catalog,
    /// Planned (offline) or measured (startup) baselines, one per RTX.
    pub gpus: Vec<GpuBudget>,
    pub spark_ranks: usize,
    pub prefill: PrefillPlan,
    /// Active sequence slots (each owns its sliding rings and one exact mark).
    pub slots: u64,
    /// The compiled context floor; `automatic_context` lets the pool clamp it.
    pub context: u64,
    pub automatic_context: bool,
    pub requested_pool: Option<u64>,
    pub onboard: Onboard,
    pub spark_budget_bytes: u64,
    /// Runtime-only exact demands (loaded code, step workspaces, peer exchange);
    /// the planner passes none until the family's PROGRAMS.json exists.
    pub fixed: Vec<Demand>,
    pub startup: Vec<Demand>,
}

/// The request plus what the caller reports or uploads from it.
pub struct MuseRequest {
    pub request: PlacementRequest,
    pub spark_budget: SparkBudgetInventory,
    /// Per layer, per coordinator rank: the operands the device packs.
    pub layouts: Vec<Vec<LayerLayout>>,
    pub global: GlobalLayout,
    pub owners: cuteafd_core::ExpertOwners,
    /// Per rank: the K6 programs it binds and their shared scratch.
    pub projections: Vec<crate::families::muse_spark::weight_layout::RankProjections>,
}

impl MuseRequest {
    /// Bytes of one rank's layer weights across all layers.
    pub fn layer_bytes(&self, gpu: usize) -> Result<u64> {
        self.layouts.iter().try_fold(0u64, |sum, ranks| {
            let bytes = ranks.get(gpu).context("Muse layout rank")?.bytes()? as u64;
            sum.checked_add(bytes).context("Muse layer bytes overflow")
        })
    }
}

/// Sliding-ring rows: the window plus one prefill unit, page aligned.
pub fn ring_rows(cfg: &MuseSparkConfig, unit_rows: u64) -> Result<u64> {
    Ok(cuteafd_core::prefix::AppendRing::new(cfg.window as u64, unit_rows.max(PAGE_ROWS), PAGE_ROWS)
        .context("Muse sliding ring geometry")?.rows)
}

pub fn request(inputs: MuseInputs<'_>) -> Result<MuseRequest> {
    let MuseInputs { cfg, core, catalog, gpus, spark_ranks, prefill, slots, context, automatic_context,
        requested_pool, onboard, spark_budget_bytes, fixed: runtime_fixed, startup } = inputs;
    let ranks = gpus.len();
    let (executor, modes) = executor(ranks)?;
    ensure!(spark_ranks > 0, "Muse Spark routed experts need Spark ranks");
    ensure!(slots > 0 && context > 0 && context <= cfg.max_position as u64,
        "Muse slots {slots} / context {context} outside 1..={}", cfg.max_position);
    ensure!(catalog.whole_expert_outputs(), "Muse Spark experts must carry their post-expert norm");
    let layouts = (0..cfg.layers).map(|layer| LayerLayout::ranks(cfg, layer, ranks, core))
        .collect::<Result<Vec<_>>>()?;
    let global = GlobalLayout::new(cfg, core)?;
    let owners = cuteafd_core::ExpertOwners::staggered(cfg.layers, cfg.experts, spark_ranks)
        .map_err(|error| anyhow::anyhow!("Muse whole-expert ownership: {error}"))?;

    // KV: full layers page BF16 K+V records; sliding layers keep a per-sequence
    // ring of window + one prefill unit (append-only, exact restores).
    let record = cfg.kv_record_bytes() as u64;
    let rings = cfg.sliding_layers() as u64 * ring_rows(cfg, prefill.unit_rows)? * record;
    let marks = (cfg.sliding_layers() * cfg.window) as u64 * record;
    let mut fixed = Vec::new();
    for gpu in 0..ranks {
        let weights = layouts.iter().try_fold(0u64, |sum, ranks| sum.checked_add(ranks[gpu].bytes()? as u64)
            .context("Muse layer bytes overflow"))?;
        let gpu = u8::try_from(gpu)?;
        fixed.push(Demand::new(gpu, Category::Weights,
            "layer weights (head-split attention; MoE projections, router, norms, residual gates whole)",
            weights, Basis::Exact));
        fixed.push(Demand::new(gpu, Category::Weights, "final norm", global.final_norm as u64, Basis::Exact));
        fixed.push(Demand::new(gpu, Category::Kv, format!("sliding rings, {slots} sequences x {} layers",
            cfg.sliding_layers()), rings * slots / ranks as u64, Basis::Formula));
        // One exact retained sliding window per slot (prompt end + turn end marks).
        fixed.push(Demand::new(gpu, Category::Prefix, "Muse exact sliding marks", marks * slots / ranks as u64,
            Basis::Formula));
    }
    // The lane's activation planes at one prefill unit (the decode width is smaller).
    let lane = crate::families::muse_spark::lane::LanePlanes::new(cfg, ranks, usize::try_from(prefill.unit_rows)?)?;
    for gpu in 0..ranks {
        fixed.push(Demand::new(u8::try_from(gpu)?, Category::Workspace, "Muse lane planes", lane.bytes()? as u64,
            Basis::Exact));
    }
    // One lane-private K6 scratch set per rank (serial projections; the head's on the head rank).
    let projections = crate::families::muse_spark::weight_layout::projections(cfg, core, ranks,
        usize::try_from(prefill.unit_rows)?)?;
    for (gpu, rank) in projections.iter().enumerate() {
        if let Some(scratch) = &rank.scratch {
            fixed.push(Demand::new(u8::try_from(gpu)?, Category::Workspace, "Muse K6 projection scratch",
                scratch.bytes()? as u64, Basis::Exact));
        }
    }
    fixed.push(Demand::new(0, Category::Embedding, "token embedding", global.embedding as u64, Basis::Exact));
    fixed.push(Demand::new(u8::try_from(ranks - 1)?, Category::Weights, "lm head (K6)", global.head_bytes() as u64,
        Basis::Exact));
    // Spark returns land on the coordinator: one prefill span of expert-width partials per rank.
    fixed.push(Demand::new(0, Category::Transport, "Spark intake planes",
        crate::serving_capacity::prefill_return_bytes(prefill.waves, prefill.unit_rows, spark_ranks as u64,
            cfg.expert_hidden as u64), Basis::Formula));
    fixed.extend(runtime_fixed);

    // Spark: whole experts partitioned by the shared owner table.
    let working = SparkWorkingSet { rtx_layers: vec![], spark_layers: (0..cfg.layers).collect(),
        spark_ranks, prefill_lanes: LANES };
    let rings_geometry = crate::plan::layout::spark_ring_geometry(LANES as usize);
    let endpoints = [cuteafd_core::SparkEndpoint { ring_depth: None, capacity: u32::try_from(prefill.unit_rows)?,
        ingress: cuteafd_core::SparkIngress::Fp8K32 }];
    let reserve = spark_reserve_bytes(prefill.waves, prefill.unit_rows)?;
    let spark_budget = SparkBudgetInventory::exl3(catalog, &working, &endpoints,
        cuteafd_core::SparkRings { max_frame_bytes: rings_geometry.max_frame_bytes, depth: rings_geometry.depth,
            slot_bytes: rings_geometry.slot_bytes, alignment: rings_geometry.alignment }, spark_budget_bytes, reserve)?;
    // An onboarded layer is whole on one GPU (exact per-expert norm).
    let manifest = catalog.exl3().context("Muse Spark requires EXL3 expert headers")?;
    let layers = (0..cfg.layers).map(|layer| {
        let whole = manifest.residency(crate::V41Exl3Layer::Backbone(layer), 1, 0)?.device_arena_layout()?.1 as u64;
        let full = cfg.attention[layer] == MuseAttention::Full;
        let unit = record * PAGE_ROWS;
        Ok(LayerDemand {
            kind: if full { AttentionClass::Gqa } else { AttentionClass::Swa },
            weights: ModeBytes::default(),
            kv_unit: if full {
                KvDemand { unit_bytes_whole: unit, unit_bytes_split: [unit / 2; 2], unit_bytes_context: None }
            } else { KvDemand::default() },
            fixed_bytes: ModeBytes::default(), extra: Vec::new(), context_indexer: false, colocate: None,
            experts: Some(ExpertCost { whole: Bytes2 { resident: whole, staging: 0 },
                half: [Bytes2::default(); 2], tp2: false, spark_ok: true }),
            modes: modes.to_vec(),
        })
    }).collect::<Result<Vec<_>>>()?;
    let capacities = gpus.iter().map(|gpu| gpu.capacity_bytes).collect::<Vec<_>>();
    let pool = PoolPolicy::resolve(&capacities, context, requested_pool, PAGE_ROWS, false)
        .with_automatic_context(automatic_context);
    let pair = ranks == 2;
    let mut request = PlacementRequest {
        inventory: Inventory { gpus, spark_ranks, peer_access: pair },
        attention_placement: pair.then_some(AttentionPlacement::Heads), context_buffers: ContextBuffers::default(),
        layers_first_gpu: 0, pool, layers, pool_overhead: vec![0; ranks], pool_bounded: Vec::new(), fixed,
        startup, movables: Vec::new(), expert_workspace: 0, tp2_workspace: [0; 2],
        onboard, expert_gpus: ranks, minimum_expert_layers: 0,
        // Whole onboarded layers alternate between the GPUs.
        policy: LayerPolicy { default: modes.to_vec(), ownership: LayerOwnership::Ranges,
            expert_ownership: if pair { ExpertOwnership::AlternatingWhole } else { ExpertOwnership::Contiguous },
            ..Default::default() },
        hops: HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: cfg.hidden as u64 * 2, rows: 0, lanes: 0,
            entry_gpu: 0, head_gpu: u8::try_from(ranks - 1)?,
            terminal: if pair { TerminalHome::Replicated } else { TerminalHome::AttentionOwner } },
        executor,
    };
    spark_budget.constrain(&mut request)?;
    Ok(MuseRequest { request, spark_budget, layouts, global, owners, projections })
}

/// Admission: the same request solved and checked against the executor.
pub fn placement(inputs: MuseInputs<'_>) -> Result<(MuseRequest, Placement)> {
    let built = request(inputs)?;
    let placed = solve(&built.request)?;
    built.request.executor.check(&placed)?;
    Ok((built, placed))
}

/// Muse's typed Spark owners on the EXL3 ladder.
pub fn spark_executions(waves: u64, unit_rows: u64) -> Result<cuteafd_core::SparkExecutions> {
    let exports = crate::plan::spark_executions::SparkBackend::Exl3(cuteafd_core::Exl3RowPolicy::Nearest).exports();
    crate::plan::spark_executions::admit_family("muse_spark", &exports, None, unit_rows, waves)
}

/// The pre-placement Spark reserve: the runtime residual plus the typed owners.
pub fn spark_reserve_bytes(waves: u64, unit_rows: u64) -> Result<u64> {
    let owners = spark_executions(waves, unit_rows)?.workspace_bytes(owner_bytes)?
        .and_then(|bytes| bytes.total()).context("Muse Spark owner reserve overflow")?;
    Ok(cuteafd_core::memory_layout::SPARK_RUNTIME_RESIDUAL_BYTES + owners)
}

pub fn owner_bytes(extent: cuteafd_core::OwnerExtent) -> Result<cuteafd_core::OwnerBytes> {
    crate::plan::layout::spark_slots::exl3_owner_bytes("muse_spark", cuteafd_core::ExpertGeometry::MUSE_SPARK, extent)
}
