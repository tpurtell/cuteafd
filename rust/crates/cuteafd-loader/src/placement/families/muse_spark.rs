//! Muse Spark admission request: one builder for the planner (planned
//! baselines) and serving startup (measured baselines plus the runtime's code,
//! workspace and exchange demands), so plan == runtime by construction.
//!
//! Coordinator: header-exact per-rank layer weights from the shared
//! [`LayerLayout`] the device packs, paged full-layer KV, per-sequence sliding
//! rings and exact sliding marks, pool first, then routed layers onboarded on
//! the RTXs: TP2 halves on a pair (both GPUs reply BF16 partials over P2P),
//! whole on one RTX (its own partials); the same partial combine norms every
//! holder's partials (ranks 2 or 1). Sparks hold
//! intermediate slices of every expert (TP, as every EXL3 family) and reply
//! per-expert partials (BF16 or NVFP4 by the per-request row cutoff); the
//! coordinator norms, weights and sums them (`replies.rs`).
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
    /// Active sequence slots (each owns its sliding rings).
    pub slots: u64,
    /// Device mark slots from the shared prefix-mark rule (`mark_slots_with_host`):
    /// staging only when the host tier holds retention.
    pub mark_slots: u64,
    /// The compiled context floor; `automatic_context` lets the pool clamp it.
    pub context: u64,
    pub automatic_context: bool,
    pub requested_pool: Option<u64>,
    pub onboard: Onboard,
    pub spark_budget_bytes: u64,
    /// Rows a request may contribute to a step and still reply BF16 partials,
    /// and the format of wider segments.
    pub partial_bf16_max_rows: usize,
    pub partial_wide_format: crate::families::muse_spark::replies::WideFormat,
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

/// Sliding-ring rows: the shared ring geometry (window + the rollback step).
pub fn ring_rows(cfg: &MuseSparkConfig) -> Result<u64> {
    Ok(crate::families::muse_spark::cache::swa_ring(cfg).map_err(|error| anyhow::anyhow!("{error}"))?.rows)
}

/// Bytes of one retained mark on one rank: each sliding layer's visible window.
pub fn mark_bytes(cfg: &MuseSparkConfig, ranks: usize) -> u64 {
    (cfg.sliding_layers() * cfg.window * cfg.kv_record_bytes() / ranks) as u64
}

pub fn request(inputs: MuseInputs<'_>) -> Result<MuseRequest> {
    let MuseInputs { cfg, core, catalog, gpus, spark_ranks, prefill, slots, mark_slots, context, automatic_context,
        requested_pool, onboard, spark_budget_bytes, partial_bf16_max_rows, partial_wide_format,
        fixed: runtime_fixed, startup } = inputs;
    let ranks = gpus.len();
    let (executor, modes) = executor(ranks)?;
    let pair = ranks == 2;
    ensure!(spark_ranks > 0, "Muse Spark routed experts need Spark ranks");
    ensure!(slots > 0 && context > 0 && context <= cfg.max_position as u64,
        "Muse slots {slots} / context {context} outside 1..={}", cfg.max_position);
    ensure!(catalog.per_expert_output_norm(), "Muse Spark experts must carry their post-expert norm");
    let layouts = (0..cfg.layers).map(|layer| LayerLayout::ranks(cfg, layer, ranks, core))
        .collect::<Result<Vec<_>>>()?;
    let global = GlobalLayout::new(cfg, core)?;

    // KV: full layers page BF16 K+V records; sliding layers keep a per-sequence
    // ring of window + one prefill unit (append-only, exact restores).
    let record = cfg.kv_record_bytes() as u64;
    let rings = cfg.sliding_layers() as u64 * ring_rows(cfg)? * record;
    let marks = mark_bytes(cfg, 1);
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
            cfg.sliding_layers()), rings * slots / ranks as u64, Basis::Exact));
        // Device mark slots from the shared prefix rule (the host tier holds retention).
        if mark_slots > 0 {
            fixed.push(Demand::new(gpu, Category::Prefix, format!("Muse exact sliding marks, {mark_slots} slots"),
                marks * mark_slots / ranks as u64, Basis::Exact));
        }
    }
    // The lane's activation planes at one prefill unit (the decode width is smaller).
    let lane = crate::families::muse_spark::lane::LanePlanes::with_waves(cfg, ranks,
        usize::try_from(prefill.unit_rows)?, usize::try_from(prefill.waves)?)?;
    for gpu in 0..ranks {
        fixed.push(Demand::new(u8::try_from(gpu)?, Category::Workspace, "Muse lane planes", lane.bytes()? as u64,
            Basis::Exact));
    }
    // Step metadata: per-row record slots and the page table for the widest
    // decode/verify batch (every slot at the 64-row rollback step) at the context.
    let tables = step_tables(usize::try_from(prefill.unit_rows)?, usize::try_from(slots)?, context,
        usize::try_from(prefill.waves)?)?;
    for gpu in 0..ranks {
        fixed.push(Demand::new(u8::try_from(gpu)?, Category::Workspace, "Muse step metadata", tables.bytes() as u64,
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
    // TP2: the exchange's per-channel receive slots on each GPU (one each, proven by
    // the interleaving model in `families::muse_spark::tp2`). `Partials` (onboarded
    // layers' peer partials) belongs to `tp2_workspace`, opened only on onboarding.
    if pair {
        let unit = usize::try_from(prefill.unit_rows)?;
        let exchange = tp2_exchange_bytes(cfg, unit)?;
        for gpu in 0..ranks {
            fixed.push(Demand::new(u8::try_from(gpu)?, Category::Transport, "Muse TP2 exchange slots",
                exchange as u64, Basis::Exact));
        }
    }
    // Sliding layers' RoPE tables (shared rope_tables_bf16, half width) over the context, on every GPU.
    let rope = rope_tables(cfg, context)?;
    for gpu in 0..ranks {
        fixed.push(Demand::new(u8::try_from(gpu)?, Category::Workspace, "Muse RoPE tables",
            rope.allocation_bytes, Basis::Exact));
    }
    fixed.push(Demand::new(0, Category::Embedding, "token embedding", global.embedding as u64, Basis::Exact));
    // The head rank's logits planes: the K6 head's F32 rows at its stored width
    // and the softcapped logits trimmed to the vocabulary.
    let (stored, trimmed) = crate::families::muse_spark::lane::head_planes(
        global.head.as_ref().map_or(cfg.vocab_size, |head| head.shape[0]), cfg.vocab_size)?;
    fixed.push(Demand::new(u8::try_from(ranks - 1)?, Category::Workspace, "Muse head logits",
        (stored + trimmed) as u64, Basis::Exact));
    fixed.push(Demand::new(u8::try_from(ranks - 1)?, Category::Weights, "lm head (K6)", global.head_bytes() as u64,
        Basis::Exact));
    // Spark partial replies land on the coordinator: per wave slot, every rank's
    // worst-case mix of BF16 (decode/verify segments) and NVFP4 (prefill) partials.
    let cutoff = crate::families::muse_spark::replies::Cutoff::with_wide(partial_bf16_max_rows, 1, partial_wide_format)?;
    // The shared intake allocates rows x row_bytes per rank and slot; the row width
    // is the worst partial-reply mix spread over the unit (replies::intake_row_bytes).
    let (_, planes) = intake(cfg, prefill, cutoff, usize::try_from(slots)?, spark_ranks)?;
    fixed.push(Demand::new(0, Category::Transport, "Muse Spark partial reply planes", planes, Basis::Exact));
    fixed.extend(runtime_fixed);

    // Spark: whole experts partitioned by the shared owner table.
    let working = SparkWorkingSet { rtx_layers: vec![], spark_layers: (0..cfg.layers).collect(),
        spark_ranks, prefill_lanes: LANES };
    let (endpoint, spark_rings) = spark_endpoint(prefill, cutoff, usize::try_from(slots)?)?;
    let reserve = spark_reserve_bytes(prefill.waves, prefill.unit_rows)?;
    let spark_budget = SparkBudgetInventory::exl3(catalog, &working, &[endpoint], spark_rings,
        spark_budget_bytes, reserve)?;
    // Onboarded layers: TP2 halves on a pair (partials combined like the Sparks'), whole on one RTX.
    let manifest = catalog.exl3().context("Muse Spark requires EXL3 expert headers")?;
    let layers = (0..cfg.layers).map(|layer| {
        let backbone = crate::V41Exl3Layer::Backbone(layer);
        let whole = if pair { 0 } else { manifest.residency(backbone, 1, 0)?.device_arena_layout()?.1 as u64 };
        let half = if pair {
            let arena = |rank| -> Result<u64> { Ok(manifest.residency(backbone, 2, rank)?.device_arena_layout()?.1 as u64) };
            [Bytes2 { resident: arena(0)?, staging: 0 }, Bytes2 { resident: arena(1)?, staging: 0 }]
        } else { [Bytes2::default(); 2] };
        let full = cfg.attention[layer] == MuseAttention::Full;
        let unit = record * PAGE_ROWS;
        Ok(LayerDemand {
            kind: if full { AttentionClass::Gqa } else { AttentionClass::Swa },
            weights: ModeBytes::default(),
            kv_unit: if full {
                KvDemand { unit_bytes_whole: unit, unit_bytes_split: [unit / 2; 2], unit_bytes_context: None }
            } else { KvDemand::default() },
            fixed_bytes: ModeBytes::default(), extra: Vec::new(), context_indexer: false, colocate: None,
            experts: Some(ExpertCost { whole: Bytes2 { resident: whole, staging: 0 }, half, tp2: pair, spark_ok: true }),
            modes: modes.to_vec(),
        })
    }).collect::<Result<Vec<_>>>()?;
    let capacities = gpus.iter().map(|gpu| gpu.capacity_bytes).collect::<Vec<_>>();
    let pool = PoolPolicy::resolve(&capacities, context, requested_pool, PAGE_ROWS, false)
        .with_automatic_context(automatic_context);
    let mut request = PlacementRequest {
        inventory: Inventory { gpus, spark_ranks, peer_access: pair },
        attention_placement: pair.then_some(AttentionPlacement::Heads), context_buffers: ContextBuffers::default(),
        layers_first_gpu: 0, pool, layers, pool_overhead: vec![0; ranks], pool_bounded: Vec::new(), fixed,
        // One RTX: an onboarded whole layer replies its own BF16 partials to the same
        // partial combine (ranks 1), charged once the GPU holds a routed layer.
        startup, movables: Vec::new(),
        expert_workspace: if pair { 0 } else { (tp2_partial_bytes(cfg, usize::try_from(prefill.unit_rows)?) / 2) as u64 },
        // TP2 onboarding (opened only when a layer onboards): both GPUs' BF16 partials
        // for one unit (own + peer over P2P) feed the same partial combine (ranks 2).
        tp2_workspace: if pair { [tp2_partial_bytes(cfg, usize::try_from(prefill.unit_rows)?) as u64; 2] } else { [0; 2] },
        onboard, expert_gpus: ranks, minimum_expert_layers: 0,
        policy: LayerPolicy { default: modes.to_vec(), ownership: LayerOwnership::Ranges,
            expert_ownership: ExpertOwnership::Contiguous, ..Default::default() },
        hops: HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: cfg.hidden as u64 * 2, rows: 0, lanes: 0,
            entry_gpu: 0, head_gpu: u8::try_from(ranks - 1)?,
            terminal: if pair { TerminalHome::Replicated } else { TerminalHome::AttentionOwner } },
        executor,
    };
    spark_budget.constrain(&mut request)?;
    Ok(MuseRequest { request, spark_budget, layouts, global, projections })
}

/// The step metadata a rank allocates: a bank per in-flight prefill slot and a
/// decode bank whose page table serves `slots` sequences each stepping up to
/// the 64-row rollback step (decode and verify), capped at the unit's rows.
pub fn step_tables(unit_rows: usize, slots: usize, context: u64, waves: usize)
    -> Result<crate::families::muse_spark::lane::StepTables> {
    let decode = (slots * crate::families::muse_spark::cache::SWA_RING_STEP as usize).min(unit_rows);
    crate::families::muse_spark::lane::StepTables::new(unit_rows, decode, context, waves)
}

/// The sliding layers' rotary tables: the module's default RoPE over the head
/// (span 64, theta from the config), extent = the admitted context.
pub fn rope_tables(cfg: &MuseSparkConfig, context: u64) -> Result<crate::plan::rope::Tables> {
    crate::plan::rope::Tables::new(usize::try_from(context)?,
        [crate::plan::rope::Recipe::new(cfg.head_dim, cfg.rope_theta, None)?])
}

/// Per GPU of a TP2 pair: the receive slots of every channel but `Partials`
/// (attention and post-expert cores, the expert-input and combined halves).
pub fn tp2_exchange_bytes(cfg: &MuseSparkConfig, unit_rows: usize) -> Result<usize> {
    use crate::families::muse_spark::tp2::{receive_slots, slot_bytes, Channel};
    Channel::ALL.iter().filter(|&&c| c != Channel::Partials).try_fold(0usize, |sum, &channel| {
        let bytes = slot_bytes(channel, cfg.hidden, cfg.expert_hidden, cfg.topk, unit_rows)? * receive_slots(channel);
        sum.checked_add(bytes).context("Muse TP2 exchange overflow")
    })
}

/// Per GPU under TP2 onboarding: its own BF16 partials and the `Partials`
/// receive slot holding its peer's for one unit (`rows x topk x E` each).
pub fn tp2_partial_bytes(cfg: &MuseSparkConfig, unit_rows: usize) -> usize {
    2 * unit_rows * cfg.topk * crate::families::muse_spark::replies::PartialFormat::Bf16.row_bytes(cfg.expert_hidden)
}

/// Admission: the same request solved and checked against the executor.
pub fn placement(inputs: MuseInputs<'_>) -> Result<(MuseRequest, Placement)> {
    let built = request(inputs)?;
    let placed = solve(&built.request)?;
    built.request.executor.check(&placed)?;
    Ok((built, placed))
}

/// The coordinator intake of the Spark link: each rank's plane row width
/// (`replies::intake_row_bytes`, the worst reply mix spread over the unit) and
/// the bytes of every rank's planes over the in-flight waves. The runtime link
/// allocates exactly this (`SparkIntake::with_waves(ranks, unit, row, waves)`).
pub fn intake(cfg: &MuseSparkConfig, prefill: PrefillPlan, cutoff: crate::families::muse_spark::replies::Cutoff,
    sequences: usize, spark_ranks: usize) -> Result<(usize, u64)> {
    let unit = usize::try_from(prefill.unit_rows)?;
    let row = crate::families::muse_spark::replies::intake_row_bytes(cutoff, unit, sequences, cfg.topk, cfg.expert_hidden);
    ensure!(row % 2 == 0, "Muse intake rows are whole BF16 elements");
    Ok((row, crate::serving_capacity::prefill_return_bytes(prefill.waves, prefill.unit_rows, spark_ranks as u64,
        row as u64 / 2)))
}

/// NIC largest RC message (ibv `max_msg_sz`, 1 GiB on the ConnectX-7 ports of
/// raptor and every Spark); a partial-reply frame above it is refused at admission.
pub const NIC_MAX_MESSAGE_BYTES: usize = 1 << 30;

/// Muse's one persistent Spark endpoint (decode, verify and prefill share its
/// QP) and its rings, shared by the planner, admission and the transport:
///
/// - the response frame is the worst partial reply one rank sends at the unit
///   (`replies::max_rank_bytes`: BF16 decode/verify segments and the wide
///   prefill rest, over the admitted sequences), one unchunked SEND;
/// - the ring depth is what can be in flight on the QP: the fixed prefill
///   waves plus the decode lane's one step. Slots are uniform, so each holds
///   the worst frame.
pub fn spark_endpoint(prefill: PrefillPlan, cutoff: crate::families::muse_spark::replies::Cutoff,
    sequences: usize) -> Result<(cuteafd_core::SparkEndpoint, cuteafd_core::SparkRings)> {
    let geometry = cuteafd_core::ExpertGeometry::MUSE_SPARK;
    let unit = usize::try_from(prefill.unit_rows)?;
    let reply = crate::families::muse_spark::replies::max_rank_bytes(cutoff, unit, sequences,
        geometry.topk as usize, geometry.hidden as usize);
    let depth = usize::try_from(prefill.waves)?.checked_add(1).context("Muse ring depth")?;
    let base = crate::plan::layout::spark_ring_geometry(LANES as usize);
    let endpoint = cuteafd_core::SparkEndpoint { ring_depth: Some(depth), capacity: u32::try_from(unit)?,
        ingress: cuteafd_core::SparkIngress::Fp8K32, partial_reply_bytes: Some(u64::try_from(reply)?) };
    let rings = cuteafd_core::SparkRings { max_frame_bytes: base.max_frame_bytes, depth: base.depth,
        slot_bytes: base.slot_bytes, alignment: base.alignment };
    let frame = endpoint.frame_bytes(geometry, rings).context("Muse Spark endpoint frame")?;
    ensure!(frame <= NIC_MAX_MESSAGE_BYTES, "Muse Spark partial reply frame of {frame} bytes exceeds the NIC's \
        {NIC_MAX_MESSAGE_BYTES}-byte RC message limit (lower the unit, the sequences or the BF16 cutoff)");
    Ok((endpoint, rings))
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
