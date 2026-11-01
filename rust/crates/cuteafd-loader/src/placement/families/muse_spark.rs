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
    /// The coordinator EXL3 package for onboarded layers (`local_package`).
    pub local: LocalPackage,
    /// The DFlash2 drafter (`drafter_inventory`), charged on the head GPU.
    pub drafter: Option<MuseDrafter>,
    /// Runtime-only exact demands (loaded code, step workspaces, peer exchange);
    /// the planner passes none until the family's PROGRAMS.json exists.
    pub fixed: Vec<Demand>,
    pub startup: Vec<Demand>,
}

/// Header-only bundled tower inventory shared by offline and measured admission.
#[derive(Debug, Clone)]
pub struct MediaOptions {
    pub vision: crate::plan::MediaMode,
    pub audio: crate::plan::MediaMode,
    pub replicas: usize,
    pub vision_bytes: (u64, u64),
    pub audio_bytes: (u64, u64),
}

impl MediaOptions {
    pub fn from_checkpoint(checkpoint: &crate::plan::Checkpoint, vision: crate::plan::MediaMode,
        audio: crate::plan::MediaMode, replicas: usize) -> Result<Self> {
        use crate::{media, plan::MediaMode};
        let vision_bytes = if vision == MediaMode::Off
            || checkpoint.config.get("vision_config").is_none_or(serde_json::Value::is_null) { (0, 0) }
        else {
            (media::muse_vision::resident_weights(&checkpoint.config, checkpoint.tensors.iter().map(|t| &t.meta))
                .map_err(anyhow::Error::msg)?,
                media::muse_vision::scratch_bytes(media::muse_vision::MAX_TOKENS).map_err(anyhow::Error::msg)?)
        };
        ensure!(vision_bytes != (0, 0) || matches!(vision, MediaMode::Auto | MediaMode::Off),
            "checkpoint has no bundled Muse vision tower");
        let audio = crate::plan::resolve_audio(audio, &checkpoint.snapshot)?;
        let audio_bytes = if audio == MediaMode::Off { (0, 0) } else {
            let plan = media::pcm_audio::PcmAudioPlan::from_snapshot(&checkpoint.snapshot)?;
            (plan.weight_bytes(), media::pcm_audio::scratch_bytes(media::audio::MAX_CLIP_SAMPLES)?)
        };
        Ok(Self { vision, audio, replicas, vision_bytes, audio_bytes })
    }
}

/// The request plus what the caller reports or uploads from it.
pub struct MuseRequest {
    pub request: PlacementRequest,
    /// The coordinator package state the request was built with.
    pub local: LocalPackage,
    /// The drafter the request charged on the head GPU.
    pub drafter: Option<MuseDrafter>,
    pub spark_budget: SparkBudgetInventory,
    pub media: Option<MediaAdmission>,
    admitted_placement: Option<Placement>,
    /// Per layer, per coordinator rank: the operands the device packs.
    pub layouts: Vec<Vec<LayerLayout>>,
    pub global: GlobalLayout,
    /// Per rank: the K6 programs it binds and their shared scratch.
    pub projections: Vec<crate::families::muse_spark::weight_layout::RankProjections>,
}

#[derive(Debug, Clone)]
pub struct MediaAdmission {
    pub vision: crate::plan::encoder::EncoderPlacement,
    pub audio: crate::plan::encoder::EncoderPlacement,
    /// Native ledgers only; the Spark guard is reserved separately.
    pub spark_items: Vec<Vec<cuteafd_core::memory_layout::Item>>,
}

impl MuseRequest {
    /// Resolve towers against the shared pool-first placement, then charge them
    /// before the final solve. Planned and measured callers use this verbatim.
    pub fn admit_media(&mut self, options: &MediaOptions) -> Result<()> {
        use crate::plan::{encoder::*, MediaMode};
        use cuteafd_core::memory_layout::Item;
        ensure!(self.media.is_none(), "Muse media already admitted");
        if options.vision_bytes == (0, 0) && options.audio_bytes == (0, 0) {
            let empty = crate::plan::encoder::EncoderHardware { v41: false, gpus: Vec::new(), sparks: Vec::new() };
            self.media = Some(MediaAdmission {
                vision: crate::plan::encoder::encoder_placement(options.vision, &empty, 0, 0, 1),
                audio: crate::plan::encoder::encoder_placement(options.audio, &empty, 0, 0, 1),
                spark_items: vec![Vec::new(); self.spark_budget.resident.len()],
            });
            return Ok(());
        }
        let hardware = EncoderHardware { v41: false,
            gpus: self.request.inventory.gpus.iter().map(|gpu| EncoderGpuBudget {
                free_bytes: gpu.available(), kv_target_bytes: 0 }).collect(),
            sparks: (0..self.spark_budget.resident.len()).map(|rank| EncoderSparkBudget {
                rank, host: format!("spark{rank}"), idle: false,
                expert_bytes: self.spark_budget.resident[rank].iter().sum(),
                free_bytes: self.spark_budget.budget }).collect() };
        let requests = [EncoderRequest { mode: options.vision, weights: options.vision_bytes.0,
            scratch: options.vision_bytes.1, copies: options.replicas },
            EncoderRequest { mode: options.audio, weights: options.audio_bytes.0,
                scratch: options.audio_bytes.1, copies: 1 }];
        let (selected, (request, budget, spark_items, placed)) = admit_encoders(&requests, &hardware, |selected| {
            let mut request = self.request.clone();
            let mut budget = self.spark_budget.clone();
            let mut spark_items = vec![Vec::new(); budget.resident.len()];
            for (name, selected) in ["vision", "audio"].into_iter().zip(selected) {
                let items = [Item::new(Category::Weights, format!("{name} tower"), "BF16 native", selected.weights, Basis::Exact),
                    Item::new(Category::Workspace, format!("{name} scratch"), "resident", selected.scratch, Basis::Exact)];
                if let EncoderKind::Rtx { gpu } = selected.kind {
                    for item in items { request.fixed.push(Demand::new(u8::try_from(gpu)?,
                        item.category, item.group, item.bytes, item.basis)); }
                } else {
                    for rank in selected.spark_ranks() {
                        spark_items[rank].extend(items.iter().cloned());
                        budget.overhead[rank] = budget.overhead[rank].checked_add(selected.admitted_bytes())
                            .context("Muse Spark media overhead")?;
                    }
                }
            }
            for (rank, items) in spark_items.iter_mut().enumerate().filter(|(_, items)| !items.is_empty()) {
                let guard = 1 << 30;
                budget.overhead[rank] = budget.overhead[rank].checked_add(guard).context("Muse Spark media guard")?;
                items.push(Item::new(Category::Reserved, "media admission guard", "not allocated", guard, Basis::Formula)
                    .with_lifetime(cuteafd_core::memory_layout::ItemLifetime::PostReadyReserve));
            }
            budget.constrain(&mut request)?;
            let placed = solve(&request)?;
            if request.pool.requested.is_none() && requests.iter().zip(selected).any(|(request, selected)|
                request.mode == MediaMode::Auto && matches!(selected.kind, EncoderKind::Rtx { .. })) {
                ensure!(placed.pool_tokens >= request.pool.target,
                    "auto native tower would reduce the requested KV target {} to {}",
                    request.pool.target, placed.pool_tokens);
            }
            ensure!(spark_items.iter().all(Vec::is_empty) || placed.onboard_layers < budget.layers.len(),
                "Muse Spark media requires a retained expert worker; select RTX media or retain Spark experts");
            Ok((request, budget, spark_items, placed))
        })?;
        self.admitted_placement = Some(placed);
        self.request = request;
        self.spark_budget = budget;
        self.media = Some(MediaAdmission { vision: selected[0].clone(), audio: selected[1].clone(), spark_items });
        Ok(())
    }

    pub fn admitted_placement(&self) -> Result<Placement> {
        self.admitted_placement.clone().map(Ok).unwrap_or_else(|| solve(&self.request).map_err(Into::into))
    }

    pub fn check_media_workers(&self, placed: &Placement) -> Result<()> {
        ensure!(self.media.as_ref().is_none_or(|media| media.spark_items.iter().all(Vec::is_empty))
            || placed.onboard_layers < self.spark_budget.layers.len(),
            "Muse Spark media requires a retained expert worker; select RTX media or retain Spark experts");
        Ok(())
    }

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
        requested_pool, onboard, spark_budget_bytes, partial_bf16_max_rows, partial_wide_format, local,
        drafter, fixed: runtime_fixed, startup } = inputs;
    let ranks = gpus.len();
    let (executor, modes) = executor(ranks)?;
    // Onboarding always enters the solve: a missing package never turns an
    // automatic placement Spark-only (`check_package` refuses it once memory
    // places a layer); an explicit layer count without it is refused here.
    if let LocalPackage::Missing(package) = &local {
        let explicit = matches!(onboard, Onboard::Layers(n) if n > 0) || matches!(onboard, Onboard::Fraction(f) if f > 0.0);
        ensure!(!explicit, "{}", LocalPackage::refusal(package));
    }
    let local_workspace = local.workspace();
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
    // The Spark path's combine tables on the entry GPU (rank 0 combines every landed
    // wave): one image per return slot, so a slot is never rewritten under a combine.
    fixed.push(Demand::new(0, Category::Workspace, "Muse Spark combine tables",
        combine_tables_bytes(usize::try_from(prefill.unit_rows)?) as u64 * prefill.waves, Basis::Exact));
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
        startup,
        // The drafter lives beside the target head it borrows (the last rank).
        movables: match &drafter {
            None => Vec::new(),
            Some(drafter) => vec![drafter.demand(u8::try_from(ranks - 1)?).movable(ranks)?],
        },
        expert_workspace: if pair { 0 } else {
            (tp2_partial_bytes(cfg, usize::try_from(prefill.unit_rows)?) / 2) as u64 + local_workspace },
        // TP2 onboarding (opened only when a layer onboards): both GPUs' BF16 partials
        // for one unit (own + peer over P2P) feed the same partial combine (ranks 2).
        tp2_workspace: if pair {
            [tp2_partial_bytes(cfg, usize::try_from(prefill.unit_rows)?) as u64 + local_workspace; 2]
        } else { [0; 2] },
        onboard, expert_gpus: ranks, minimum_expert_layers: 0,
        policy: LayerPolicy { default: modes.to_vec(), ownership: LayerOwnership::Ranges,
            expert_ownership: ExpertOwnership::Contiguous, ..Default::default() },
        hops: HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: cfg.hidden as u64 * 2, rows: 0, lanes: 0,
            entry_gpu: 0, head_gpu: u8::try_from(ranks - 1)?,
            terminal: if pair { TerminalHome::Replicated } else { TerminalHome::AttentionOwner } },
        executor,
    };
    spark_budget.constrain(&mut request)?;
    Ok(MuseRequest { request, local, drafter, spark_budget, media: None, admitted_placement: None, layouts, global, projections })
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
    built.local.check(&placed)?;
    Ok((built, placed))
}

/// The coordinator EXL3 package for onboarded layers, as the image carries it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalPackage {
    /// Installed; its per-GPU executor workspace (`local_package`).
    Installed(u64),
    /// The image's manifest is known but the package beside it is not (a
    /// broken image): any onboarded layer the placement chooses is refused.
    Missing(String),
    /// No image to inspect (a plan without a manifest): onboarding is planned,
    /// the package's workspace uncharged.
    Unknown,
}

impl LocalPackage {
    pub fn workspace(&self) -> u64 { if let Self::Installed(bytes) = self { *bytes } else { 0 } }

    fn refusal(package: &str) -> String {
        format!("image lacks {package}; rebuild with the Muse coordinator scope")
    }

    /// Refuses a placement that onboards a layer this image cannot run: the
    /// solve kept onboarding in, so automatic placement lands on 0 RTX layers
    /// only when memory says 0, never because the package is missing.
    pub fn check(&self, placed: &Placement) -> Result<()> {
        if let Self::Missing(package) = self {
            ensure!(placed.onboard_layers == 0, "{} (the placement onboards {} RTX expert layers)",
                Self::refusal(package), placed.onboard_layers);
        }
        Ok(())
    }
}

/// A Muse DFlash2 drafter's coordinator charge: its weights, KV rings and
/// workspaces (the shared DFlash2 inventory), and the head scratch its
/// `TargetHead::Launch` owns to run the target's exact head (K6 head into
/// stored-width F32 rows, trimmed, softcapped) over a draft block batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MuseDrafter {
    pub owned: u64,
    pub scratch: u64,
    pub head_scratch: u64,
    /// The draft batch's input ids and the policy probe's device route ring
    /// (`layers x HEAD_ROWS x topk` ids) on the head GPU.
    pub runtime: u64,
    pub context_slots: usize,
    pub sequences: usize,
    pub block: usize,
    pub head_dim: usize,
}

impl MuseDrafter {
    fn demand(&self, head: u8) -> crate::placement::draft::DrafterDemand {
        crate::placement::draft::DrafterDemand::local(MovableId::Drafter, vec![Bytes2 {
            resident: self.owned + self.scratch + self.head_scratch + self.runtime, staging: 0 }], head)
    }
}

/// The drafter's own config against the target: hidden, vocabulary, taps,
/// and the head's softcap and output multiplier (the drafter has no head of
/// its own and reads the target's, which applies both; a drafter trained
/// against other values is refused by name). `stored` is the K6 head's
/// stored vocabulary width; `sms` the head GPU's SM count.
pub fn drafter_inventory(cfg: &MuseSparkConfig, config: &serde_json::Value, slots: usize, sequences: usize,
    stored: usize, sms: u64, kv_cache: crate::families::qwen4::Qwen4KvCache) -> Result<MuseDrafter> {
    use crate::families::glm5::draft_representation::{draft_geometry, draft_resident_bytes_with_mode, GlmDraftCapacity,
        GlmDraftRepresentation};
    let (geometry, block) = draft_geometry(config).map_err(|error| anyhow::anyhow!("Muse drafter geometry: {error}"))?;
    let d = &config["dflash_config"];
    let taps = d["target_layer_ids"].as_array().context("Muse drafter taps")?.iter()
        .map(|tap| tap.as_u64().map(|t| t as usize).context("Muse drafter tap")).collect::<Result<Vec<_>>>()?;
    ensure!(geometry.hidden as usize == cfg.hidden && geometry.vocab as usize == cfg.vocab_size
        && taps.iter().all(|&tap| tap < cfg.layers), "Muse drafter hidden/vocab/taps do not fit the target");
    let close = |key: &str, target: f64| d[key].as_f64().is_some_and(|value| (value - target).abs() <= 1e-9 * target.abs());
    ensure!(close("final_logit_softcapping", cfg.softcap) && close("output_multiplier", cfg.output_multiplier),
        "Muse drafter head softcap/output_multiplier differ from the target's ({}, {}): its logits would not be \
         the target head's", cfg.softcap, cfg.output_multiplier);
    let block = usize::try_from(block)?;
    let capacity = GlmDraftCapacity::new(slots, sequences, block).map_err(|error| anyhow::anyhow!("{error}"))?;
    let (owned, scratch) = draft_resident_bytes_with_mode(config, capacity.context_slots, capacity.max_batch_sequences,
        sms, GlmDraftRepresentation::Bf16Only, 2, kv_cache).map_err(|error| anyhow::anyhow!("Muse drafter: {error}"))?;
    // Stored-width F32 rows of the K6 head and the trimmed F32 rows, per draft row.
    ensure!(stored >= cfg.vocab_size, "Muse head stored width below the vocabulary");
    let rows = capacity.max_batch_sequences * block;
    let head_scratch = u64::try_from(rows * (stored + cfg.vocab_size) * 4)?;
    let probe = (cfg.layers * crate::families::muse_spark::lane::HEAD_ROWS * cfg.topk * 4) as u64;
    let runtime = u64::try_from(rows * 4)? + probe;
    Ok(MuseDrafter { owned, scratch, head_scratch, runtime, context_slots: capacity.context_slots,
        sequences: capacity.max_batch_sequences, block, head_dim: geometry.head_dim as usize })
}

/// The combine's rank tables the coordinator uploads per wave: a BF16 and a
/// wide `rank_regions` table (`MAX_COMBINE_RANKS` i64 addresses each) and their
/// two row maps (`capacity` u32 each).
pub const MAX_COMBINE_RANKS: usize = 8;
pub fn combine_tables_bytes(capacity: usize) -> usize { 2 * MAX_COMBINE_RANKS * 8 + 2 * capacity * 4 }

/// A missing coordinator EXL3 package (onboarding unavailable in this build).
#[derive(Debug, thiserror::Error)]
#[error("coordinator EXL3 package {0} is not installed beside the program manifest")]
pub struct MissingLocalPackage(pub String);

/// The coordinator's installed EXL3 package for onboarded layers (`rtx-tp2` on
/// a pair, `rtx-tp1` on one GPU) and the device workspace its executor
/// allocates per GPU: every capacity specialization's buffers up to the
/// compiled one (the shared `exl3_workspace_bytes`), plus the executor's F32
/// token output (allocated by the shared executor; Muse reads the partial
/// epilogue's planes instead), the partial epilogue's two row maps and the
/// onboarded combine's tables (two 8-rank address tables, two row maps).
pub fn local_package(catalog: &crate::OfficialV41Catalog, manifest: &std::path::Path, gpus: usize, rows: u64)
    -> Result<(std::path::PathBuf, u64)> {
    let exl3 = catalog.exl3().context("Muse onboarding needs EXL3 expert headers")?;
    let tiers = exl3.decoder_tiers().iter().map(usize::to_string).collect::<String>();
    let stem = format!("exl3-muse-k{tiers}");
    let layout = if gpus == 2 { "rtx-tp2" } else { "rtx-tp1" };
    let parent = manifest.parent().context("Muse program manifest parent")?;
    let package = [parent.join("exl3").join(&stem), parent.join("../lib/exl3").join(&stem)].into_iter()
        .map(|p| p.join(layout)).find(|p| p.is_dir())
        .ok_or_else(|| anyhow::Error::new(MissingLocalPackage(format!("{stem}/{layout}"))))?;
    let capacities = cuteafd_core::EXL3_CAPACITIES.map(u64::from);
    let compiled = capacities.into_iter().find(|&c| c >= rows).context("Muse onboarded rows beyond EXL3 capacities")?;
    let shape = catalog.routed_experts();
    let manifests = capacities.into_iter().filter(|&c| c <= compiled).map(|capacity| {
        let directory = package.join(format!("m{capacity}"));
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(directory.join("v41_exl3.json"))?)?;
        ensure!(value["hidden"].as_u64() == Some(shape.hidden as u64)
            && value["intermediate"].as_u64() == Some(shape.intermediate as u64 / gpus as u64)
            && value["experts"].as_u64() == Some(shape.experts as u64) && value["top_k"].as_u64() == Some(shape.topk as u64)
            && value["capacity"].as_u64() == Some(capacity) && value["output_dtype"].as_str() == Some("fp32")
            && value["bits"] == serde_json::json!(exl3.decoder_tiers()),
            "Muse {layout} EXL3 capacity/geometry/tiers/output mismatch in {}", directory.display());
        Ok(value)
    }).collect::<Result<Vec<_>>>()?;
    let workspace = crate::serving_capacity::exl3_workspace_bytes(&manifests, true)
        .map_err(|error| anyhow::anyhow!("{error}"))? + compiled * shape.hidden as u64 * 4
        + 2 * rows * 4 + combine_tables_bytes(usize::try_from(rows)?) as u64;
    Ok((package, workspace))
}

/// The coordinator package state both the planner and startup build from:
/// installed (its workspace charged), missing beside a known manifest, or
/// unknown without one; any other inventory error refused.
pub fn local_workspace(catalog: &crate::OfficialV41Catalog, manifest: Option<&std::path::Path>, gpus: usize,
    unit_rows: u64) -> Result<LocalPackage> {
    let Some(manifest) = manifest else { return Ok(LocalPackage::Unknown) };
    match local_package(catalog, manifest, gpus, unit_rows) {
        Ok((_, workspace)) => Ok(LocalPackage::Installed(workspace)),
        Err(error) => match error.downcast::<MissingLocalPackage>() {
            Ok(MissingLocalPackage(package)) => Ok(LocalPackage::Missing(package)),
            Err(error) => Err(error),
        },
    }
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

/// The coordinator host's Spark transport for Muse's one endpoint, as the
/// link registers it: every rank's rings at the sealed endpoint (`spark_endpoint`),
/// the request egress (one bootstrap-wide FP8 wire per wave slot), and the
/// intake planes' host-mode upper bound (`intake`; GPU landing uses less).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostTransport { pub rings: u64, pub egress: u64, pub intake: u64 }

pub fn host_transport(cfg: &MuseSparkConfig, prefill: PrefillPlan, cutoff: crate::families::muse_spark::replies::Cutoff,
    sequences: usize, spark_ranks: usize) -> Result<HostTransport> {
    let geometry = cuteafd_core::ExpertGeometry::MUSE_SPARK;
    let (endpoint, rings) = spark_endpoint(prefill, cutoff, sequences)?;
    let per_rank = endpoint.ring_bytes(geometry, rings).context("Muse host ring geometry")?;
    let unit = u32::try_from(prefill.unit_rows)?;
    let egress = geometry.compact_wire_bytes(unit, false).context("Muse egress geometry")?.0 as u64;
    let (_, intake) = intake(cfg, prefill, cutoff, sequences, spark_ranks)?;
    Ok(HostTransport {
        rings: (per_rank as u64).checked_mul(spark_ranks as u64).context("Muse host rings overflow")?,
        egress: egress.checked_mul(prefill.waves).context("Muse host egress overflow")?,
        intake })
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
