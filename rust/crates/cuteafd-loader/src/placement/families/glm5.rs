//! GLM 5.3 coordinator admission shared by the planner and runtime. On two
//! GPUs the executor runs `heads` (replicated MLA/DSA KV, head-split
//! attention) or `context` (K3: the same head split for the projections, KV
//! pages owned by logical-page parity, a q/candidate/partial exchange per
//! layer), or explicit `layers` (whole attention and owner-local KV, with
//! split FFNs). `context` remains the default where its programs exist
//! ([`GlmContext`]); owned attention admits at most two ordered prefill lanes.
use crate::families::glm5::{GlmDsaConfig, GlmIndexer};
use crate::placement::*;
use crate::serving_capacity::{glm_cache_geometry, glm_decode_graph_allowance};
use cuteafd_core::memory_layout::{Basis, Category};

pub const PAGE_ROWS: u64 = 64;
pub const DECODE_ROWS: u64 = 64;
pub const DEFAULT_PREFILL_LANES: u64 = 3;
pub const SKIP_RANKS: usize = 4;

/// Only Spark prefill pipelines execute multiple lanes; local/diagnostic runs are serial.
pub fn prefill_lanes(spark_ranks: usize, configured: u64) -> u64 {
    if spark_ranks == 0 { 1 } else { configured.clamp(1, 4) }
}

const FLOOR: u64 = 256;
const GIB: u64 = 1 << 30;

/// Scratch retained separately by each decode/prefill workspace.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GlmScratch {
    pub programs: u64,
    pub topk: u64,
}

/// The token-split (`context`) programs of a build and their scratch, from
/// PROGRAMS.json or the loaded table (the planner and runtime pass the same).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GlmContext {
    /// `glm_scored_index_topk_decode_m64` scratch.
    pub topk_scratch: u64,
    /// Largest `glm_sparse_mla_partial_h*_s*_m64` scratch.
    pub partial_scratch: u64,
}

/// Programs `context` needs, beside the split partials (any `_s<N>` variant).
pub const CONTEXT_PROGRAMS: [&str; 5] = ["glm_scored_index_topk_decode_m64", "glm_dsa_candidate_merge",
    "glm_lse_combine2", "glm_paged_staging_gather_kv", "glm_paged_staging_gather_index"];

/// The `context` programs beside the split partials, for the fail-closed
/// reason when a build's manifest lacks them (issue #202).
pub fn context_program_list() -> String {
    let mut names: Vec<String> = CONTEXT_PROGRAMS.iter().map(|n| (*n).to_string()).collect();
    names.push("glm_sparse_mla_partial_h0_s*_m64".into());
    names.push("glm_sparse_mla_partial_h32_s*_m64".into());
    names.join(", ")
}

/// The `context` programs and scratch `lookup` (name -> scratch bytes) finds,
/// or None when this build lacks any of them (a `context` request then fails
/// closed; `heads` is the build's only attention executor).
pub fn context_programs<'n>(names: impl IntoIterator<Item = &'n str> + Clone,
    lookup: impl Fn(&str) -> Option<u64>) -> Option<GlmContext> {
    let partial = |begin: u64| names.clone().into_iter()
        .filter(|n| n.starts_with(&format!("glm_sparse_mla_partial_h{begin}_s")) && n.ends_with("_m64"))
        .filter_map(&lookup).max();
    CONTEXT_PROGRAMS.iter().all(|n| lookup(n).is_some()).then_some(())?;
    Some(GlmContext { topk_scratch: lookup(CONTEXT_PROGRAMS[0])?,
        partial_scratch: partial(0)?.max(partial(32)?) })
}

/// Index records per page (64 keys) and their bytes (64 x 128 E4M3, 64 FP32 scales).
pub const INDEX_PAGE_BYTES: u64 = PAGE_ROWS * 132;
/// Latent record pages (64 x 656 B).
pub const RECORD_PAGE_BYTES: u64 = PAGE_ROWS * 656;

/// Most pages one GPU commits per prefill lane of `rows` rows (a range of
/// rows spans one more page than it fills; every second page is this GPU's).
pub fn context_commit_pages(rows: u64) -> u64 {
    (rows.div_ceil(PAGE_ROWS) + 1).div_ceil(2)
}

/// Every allocation the `context` executor adds on one GPU beside the
/// exchange and the staging ([`ContextBuffers`]), in order; `glm5::engine`
/// allocates exactly this list (each with its 256-byte floor) and the solver
/// charges [`context_workspace_bytes`]. Decode reuses the step's own tables:
/// its page table holds the compact half table, its cache lengths the shard
/// lengths, its indices/lengths the merged local selection.
pub fn context_workspace(cfg: &GlmDsaConfig, extent: u64, prefill_rows: u64, lanes: u64, context: GlmContext)
    -> Vec<(&'static str, u64)> {
    let (rows, k, half_heads) = (DECODE_ROWS, cfg.index_topk as u64, cfg.heads as u64 / 2);
    let partial = rows * half_heads * (512 * 2 + 4);
    // History pages of either parity, as compact local-page lists.
    let half = extent.div_ceil(PAGE_ROWS).div_ceil(2);
    // Per lane: source offsets, destination offsets and sizes of each owned
    // page's records, index keys and index scales.
    let commit = 3 * 3 * context_commit_pages(prefill_rows) * 8;
    let mut out = vec![("candidates", rows * k * 8), ("own-head partial", partial), ("peer-head partial", partial),
        ("scored top-k scratch", context.topk_scratch), ("partial scratch", context.partial_scratch),
        ("gather tables", 2 * half * 4)];
    out.extend((0..lanes).map(|_| ("commit blocks", commit)));
    out
}

/// The `context` buffers of a GLM 5.3 engine with `extent` compiled context and
/// `lanes` prefill lanes of `prefill_rows`: the solver's demand and the
/// engine's allocation (`ContextExchange` and staging) both come from this.
pub fn context_buffers(cfg: &GlmDsaConfig, extent: u64, prefill_rows: u64, lanes: u64, context: GlmContext)
    -> ContextBuffers {
    ContextBuffers { staging_unit_bytes: RECORD_PAGE_BYTES + INDEX_PAGE_BYTES, staging_unit_rows: PAGE_ROWS,
        // Own heads' absorbed 576-wide BF16 query; a partial of 32 heads and its LSE.
        query_row_bytes: cfg.heads as u64 / 2 * 576 * 2, partial_row_bytes: cfg.heads as u64 / 2 * (512 * 2 + 4),
        candidate_row_bytes: cfg.index_topk as u64 * 8, compiled_extent: extent, decode_rows: DECODE_ROWS,
        // The gather-route prefill never exchanges; decode runs one lane.
        lanes: 1, workspace_bytes: context_workspace_bytes(cfg, extent, prefill_rows, lanes, context) }
}

/// The `context` programs a manifest carries (PROGRAMS.json), as [`context_programs`].
pub fn manifest_context(manifest: &serde_json::Value, sms: usize,
    plan: &cuteafd_core::coordinator_programs::AttentionReductionPlan) -> Option<GlmContext> {
    let programs = manifest["programs"].as_array()?;
    let lookup = |name: &str| programs.iter().find(|p| p["name"].as_str() == Some(name))
        .map(|p| p["scratch_bytes_at_capacity"]["scratch"].as_u64().unwrap_or(0));
    let names = programs.iter().filter_map(|p| p["name"].as_str());
    let mut context = context_programs(names, lookup)?;
    context.partial_scratch = [0, 32].into_iter().map(|begin| {
        let name = cuteafd_core::coordinator_programs::attention_partial_program(
            manifest, "glm", begin, DECODE_ROWS as usize, sms, plan).ok()?;
        lookup(&name)
    }).collect::<Option<Vec<_>>>()?.into_iter().max()?;
    Some(context)
}

/// [`context_workspace`]'s bytes with the engine's allocation floor.
pub fn context_workspace_bytes(cfg: &GlmDsaConfig, extent: u64, prefill_rows: u64, lanes: u64, context: GlmContext)
    -> u64 {
    context_workspace(cfg, extent, prefill_rows, lanes, context).iter().map(|&(_, b)| b.max(FLOOR)).sum()
}

/// Program scratch selection mirrors `glm5::engine::workspace_here`. Optional
/// BF16/per-tensor programs participate when present, even in a mixed checkpoint.
pub fn step_scratch(cfg: &GlmDsaConfig, split: bool, decode: bool,
    lookup: impl Fn(&str) -> Option<u64>) -> anyhow::Result<GlmScratch> {
    let (cap, mode) = if decode { ("m64", "decode") } else { ("m4096", "prefill") };
    let prefix = if split { "glm2" } else { "glm" };
    let divisor = if split { 2 } else { 1 };
    let (moe, dense) = (cfg.moe_intermediate / divisor, cfg.dense_intermediate / divisor);
    let mut programs = 0;
    for name in [format!("glm_index_producer_{cap}"), format!("{prefix}_producer_{cap}"),
        format!("{prefix}_sparse_mla_{mode}_{cap}"), format!("{prefix}_o_{cap}"),
        format!("{prefix}_ffn_i{moe}_{cap}"), format!("{prefix}_ffn_i{dense}_{cap}")] {
        programs = programs.max(lookup(&name).ok_or_else(|| anyhow::anyhow!("GLM step needs program {name}"))?);
    }
    let optional = [format!("glm_index_producer_bf16_{cap}"), format!("{prefix}_producer_bf16_{cap}"),
        format!("{prefix}_o_bf16_{cap}"), format!("{prefix}_ffn_i{moe}_bf16_{cap}"),
        format!("{prefix}_ffn_i{dense}_pt_{cap}")];
    for name in optional { programs = programs.max(lookup(&name).unwrap_or(0)); }
    let name = format!("glm_index_topk_{mode}_{cap}");
    Ok(GlmScratch { programs, topk: lookup(&name).ok_or_else(|| anyhow::anyhow!("GLM step needs program {name}"))? })
}

/// Device bytes beside a step's pool-dependent page table. Every allocation
/// has the engine's 256-byte floor; pinned router staging is not GPU storage.
pub fn step_bytes(cfg: &GlmDsaConfig, rows: u64, lead: bool, split: bool,
    decode: bool, full_logits: bool, scratch: GlmScratch) -> Result<u64, PlacementError> {
    let h = cfg.hidden as u64;
    let heads = cfg.heads as u64 / if split { 2 } else { 1 };
    let mul = |a: u64, b: u64| a.checked_mul(b).ok_or(PlacementError::Overflow("GLM step buffers"));
    let row = |width: u64| mul(rows, width);
    let lead_only = |bytes| if lead { bytes } else { FLOOR };
    let logits_rows = if decode || full_logits { rows } else { rows.min(DECODE_ROWS) };
    let buffers = [row(h * 2)?, row(h * 2)?, row(heads * 576 * 2)?,
        row(cfg.q_lora_rank as u64 * 2)?, row(cfg.index_heads as u64 * cfg.index_head_dim as u64)?,
        row(cfg.index_heads as u64 * 4)?, row(cfg.index_topk as u64 * 4)?, row(4)?,
        row(heads * 512 * 2)?, row(h * 2)?, row(8)?, row(8)?, row(4)?,
        scratch.programs, scratch.topk,
        lead_only(mul(logits_rows, cfg.vocab_size as u64 * 4)?),
        lead_only(row(cfg.experts as u64 * 4)?), lead_only(row(cfg.topk as u64 * 4)?),
        lead_only(row(cfg.topk as u64 * 4)?), lead_only(row(h + h / 32)?), row(h * 2)?,
        row(4)?, row(8)?, if lead { 4 << 20 } else { FLOOR }];
    buffers.into_iter().try_fold(0u64, |sum, bytes| sum.checked_add(bytes.max(FLOOR))
        .ok_or(PlacementError::Overflow("GLM step buffers")))
}

/// All non-baseline storage is future storage on both sides. Coordinator
/// weights and already-loaded code belong to the caller's baseline, not layers.
#[derive(Debug, Clone)]
pub struct GlmInputs<'a> {
    pub cfg: &'a GlmDsaConfig,
    pub layers: usize,
    pub gpus: Vec<(u64, Baseline)>,
    pub headroom_bytes: u64,
    pub spark_ranks: usize,
    /// Native FP8 K32 ingress, BF16 partials and bounded per-GPU shared execution.
    pub native_experts: bool,
    pub native_workspace: u64,
    /// Coordinator-only diagnostics replace routed FFNs with zero intake planes.
    pub skip_routed_experts: bool,
    pub prefill_rows: u64,
    pub prefill_lanes: u64,
    pub max_context: u64,
    pub automatic_context: bool,
    /// (decode, prefill) scratch from PROGRAMS.json; absent keeps the former
    /// conservative step allowance, explicitly labelled estimated.
    pub scratch: Option<[GlmScratch; 2]>,
    /// Owned drafter weights, arenas and scratch, excluding the borrowed head.
    pub drafter_bytes: u64,
    pub drafter_staging: u64,
    /// Code still to load after the runtime sample (0 if already resident).
    pub pending_code: Vec<u64>,
    /// Exact routed-layer costs; empty until an executable local package exists.
    pub experts: Vec<ExpertCost>,
    pub expert_workspace: u64,
    pub tp2_workspace: [u64; 2],
    pub requested_pool: Option<u64>,
    pub onboard: Onboard,
    pub full_prefill_logits: bool,
    /// The build's `context` programs (None: heads only).
    pub context: Option<GlmContext>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmComponent {
    Always,
    RtxExperts,
    SparkExperts,
}

impl GlmComponent {
    fn selected(self, rtx: bool, sparks: bool) -> bool {
        match self {
            Self::Always => true,
            Self::RtxExperts => rtx,
            Self::SparkExperts => sparks,
        }
    }
}

struct GlmBufferRow {
    component: GlmComponent,
    demand: Demand,
}

impl GlmBufferRow {
    fn new(component: GlmComponent, demand: Demand) -> Self { Self { component, demand } }
}

pub fn request(inputs: &GlmInputs<'_>) -> Result<PlacementRequest, PlacementError> {
    let routed = inputs.layers.saturating_sub(inputs.cfg.first_moe_layer);
    let rtx = !inputs.skip_routed_experts && !inputs.experts.is_empty()
        && inputs.onboard.layers(routed) != Some(0) && routed > 0;
    let gpus = inputs.gpus.len();
    let cfg = inputs.cfg;
    if !(1..=2).contains(&gpus) || !(1..=4).contains(&inputs.prefill_lanes)
        || inputs.prefill_rows == 0 || inputs.prefill_rows > 4096 || inputs.max_context == 0 {
        return Err(PlacementError::Inventory("GLM rank/lane/step/context geometry"));
    }
    if inputs.native_experts && (inputs.prefill_lanes != 1
        || inputs.spark_ranks == 0 || inputs.spark_ranks > 6 || inputs.skip_routed_experts
        || (gpus == 2 && inputs.spark_ranks != 4)
        || !inputs.experts.is_empty() || inputs.drafter_bytes != 0 || inputs.scratch.is_none()
        || inputs.native_workspace == 0) {
        return Err(PlacementError::Inventory("EXLR8 requires serial measured workspaces, real Spark experts and TP4 on two coordinators"));
    }
    let geometry = glm_cache_geometry(cfg, inputs.layers, gpus)
        .map_err(|_| PlacementError::Inventory("GLM cache geometry"))?;
    if inputs.skip_routed_experts && (!inputs.experts.is_empty() || inputs.spark_ranks != 0
        || inputs.prefill_lanes != 1 || inputs.onboard != Onboard::Layers(0)) {
        return Err(PlacementError::Inventory("GLM skipped experts require serial diagnostic admission without local/Spark experts"));
    }
    if !inputs.experts.is_empty() && inputs.experts.len() != routed {
        return Err(PlacementError::Inventory("GLM costs must cover every selected routed layer"));
    }
    if gpus == 2 && inputs.experts.iter().any(|c| !c.tp2) {
        return Err(PlacementError::Inventory("GLM two-GPU local experts require TP2 halves"));
    }
    let mut rows = Vec::new();
    let mut pool_overhead = Vec::new();
    let table_bytes = (DECODE_ROWS + inputs.prefill_lanes) * 4;
    let graphs = glm_decode_graph_allowance(inputs.max_context as usize, inputs.layers)
        .map_err(|_| PlacementError::Overflow("GLM graphs"))?;
    for (rank, cache) in geometry.ranks.iter().enumerate() {
        let gpu = rank as u8;
        rows.push(GlmBufferRow::new(GlmComponent::Always, Demand::new(gpu, Category::Kv, "padded decode scratch page", cache.persistent_unit_bytes, Basis::Formula)));
        rows.push(GlmBufferRow::new(GlmComponent::Always, Demand::new(gpu, Category::Kv, "RoPE context tables",
            cache.context_table_bytes_per_token.checked_mul(inputs.max_context)
                .ok_or(PlacementError::Overflow("GLM RoPE"))?, Basis::Formula)));
        let (steps, basis) = match inputs.scratch {
            Some([decode, prefill]) => (step_bytes(cfg, DECODE_ROWS, rank == 0, gpus == 2, true,
                inputs.full_prefill_logits, decode)?.checked_add(
                    step_bytes(cfg, inputs.prefill_rows, rank == 0, gpus == 2, false,
                        inputs.full_prefill_logits, prefill)?.checked_mul(inputs.prefill_lanes)
                        .ok_or(PlacementError::Overflow("GLM lane workspaces"))?)
                    .ok_or(PlacementError::Overflow("GLM workspaces"))?, Basis::Formula),
            None => {
                let hundredths = if gpus == 1 { 651 } else if rank == 0 { 559 } else { 422 };
                let mut bytes = hundredths * GIB / 100 * inputs.prefill_rows / 4096
                    * inputs.prefill_lanes / DEFAULT_PREFILL_LANES;
                if rank == 0 && inputs.full_prefill_logits {
                    bytes = bytes.checked_add(inputs.prefill_lanes * inputs.prefill_rows.saturating_sub(DECODE_ROWS)
                        * cfg.vocab_size as u64 * 4).ok_or(PlacementError::Overflow("GLM probe logits"))?;
                }
                (bytes, Basis::Estimated)
            }
        };
        let steps = if inputs.native_experts {
            steps.checked_add(inputs.native_workspace)
                .ok_or(PlacementError::Overflow("GLM native workspaces"))?
        } else { steps };
        rows.push(GlmBufferRow::new(GlmComponent::Always, Demand::new(gpu, Category::Workspace, "steps", steps, basis)));
        rows.push(GlmBufferRow::new(GlmComponent::Always, Demand::new(gpu, Category::Runtime, "decode graph allowance", graphs, Basis::Estimated)));
        if let Some(&bytes) = inputs.pending_code.get(rank).filter(|&&bytes| bytes > 0) {
            rows.push(GlmBufferRow::new(GlmComponent::Always, Demand::new(gpu, Category::Runtime, "pending loaded code", bytes, Basis::Formula)));
        }
        if gpus == 2 {
            let slots = 4 * inputs.prefill_lanes;
            let bytes = slots * inputs.prefill_rows.max(DECODE_ROWS) * cfg.hidden as u64 * 2
                + ((slots + 1) * 16).max(FLOOR) + FLOOR;
            rows.push(GlmBufferRow::new(GlmComponent::Always, Demand::new(gpu, Category::Transport, "peer exchange", bytes, Basis::Formula)));
        }
        if rank == 0 {
            if inputs.skip_routed_experts {
                rows.push(GlmBufferRow::new(GlmComponent::Always, Demand::new(gpu, Category::Transport, "diagnostic zero expert planes",
                    SKIP_RANKS as u64 * (inputs.prefill_rows.max(DECODE_ROWS) * cfg.hidden as u64 * 2).max(FLOOR),
                    Basis::Formula)));
            }
            if inputs.spark_ranks > 0 {
                let endpoints = 1 + if inputs.prefill_lanes > 1 { inputs.prefill_lanes } else { 0 };
                rows.push(GlmBufferRow::new(GlmComponent::SparkExperts, Demand::new(gpu, Category::Transport, "Spark intake planes",
                    endpoints * inputs.spark_ranks as u64 * 4096 * cfg.hidden as u64 * 2, Basis::Formula)));
            }
        }
        pool_overhead.push(table_bytes);
    }
    let fixed = rows.into_iter().filter(|row| row.component.selected(rtx, inputs.spark_ranks > 0))
        .map(|row| row.demand).collect();
    // DFlash2 reads GPU0's taps and borrows its head; the shared Movable keeps
    // it there until a measured bridge binding allows the other GPU.
    let movables = if inputs.drafter_bytes == 0 { Vec::new() } else {
        vec![draft::DrafterDemand::itemized(vec![
            Demand::new(0, Category::Drafter, "drafter owned storage", inputs.drafter_bytes, Basis::Formula)], 0)
            .movable(gpus)?]
    };
    let layers = (0..inputs.layers).map(|layer| LayerDemand {
        kind: AttentionClass::Dsa,
        weights: ModeBytes::default(),
        kv_unit: {
            let unit = PAGE_ROWS * 656 + if cfg.indexers[layer] == GlmIndexer::Full { 8448 } else { 0 };
            KvDemand { unit_bytes_whole: unit, unit_bytes_split: [unit; 2],
                unit_bytes_context: Some([unit.div_ceil(2), unit / 2]) }
        },
        fixed_bytes: ModeBytes::default(), extra: Vec::new(),
        context_indexer: cfg.indexers[layer] == GlmIndexer::Full,
        colocate: Some(cfg.index_source(layer) as u16),
        // Preserve routed identity even without local kernels: Spark-free and
        // explicit local requests must fail, not turn these into dense layers.
        experts: if inputs.skip_routed_experts { None } else {
            layer.checked_sub(cfg.first_moe_layer).map(|i| inputs.experts.get(i).copied().unwrap_or(ExpertCost {
                whole: Bytes2::default(), half: [Bytes2::default(); 2], tp2: false, spark_ok: true }))
        },
        modes: match (gpus, inputs.context) {
            (2, Some(_)) => vec![LayerMode::HeadSplit, LayerMode::ContextSplit],
            (2, None) => vec![LayerMode::HeadSplit],
            _ => vec![LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }],
        },
    }).collect();
    let startup = if inputs.drafter_staging > 0 {
        vec![Demand::new(0, Category::Drafter, "drafter packing peak",
            inputs.drafter_bytes.checked_add(inputs.drafter_staging)
                .ok_or(PlacementError::Overflow("GLM drafter packing peak"))?, Basis::Formula)]
    } else { Vec::new() };
    let cards: Vec<_> = inputs.gpus.iter().map(|g| g.0).collect();
    Ok(PlacementRequest {
        pool_bounded: Vec::new(),
        attention_placement: None,
        layers_first_gpu: 0,
        context_buffers: context_buffers(cfg, inputs.max_context, inputs.prefill_rows, inputs.prefill_lanes,
            inputs.context.unwrap_or_default()),
        inventory: Inventory { gpus: inputs.gpus.iter().map(|&(capacity_bytes, baseline)| GpuBudget {
            capacity_bytes, baseline, headroom_bytes: inputs.headroom_bytes }).collect(),
            spark_ranks: inputs.spark_ranks, peer_access: gpus == 2 },
        pool: PoolPolicy { ceiling: 1 << 31, ..PoolPolicy::resolve(&cards, inputs.max_context,
            inputs.requested_pool, PAGE_ROWS, inputs.spark_ranks == 0) },
        layers, pool_overhead, fixed, startup, movables,
        expert_workspace: if GlmComponent::RtxExperts.selected(rtx, inputs.spark_ranks > 0) { inputs.expert_workspace } else { 0 },
        tp2_workspace: if GlmComponent::RtxExperts.selected(rtx, inputs.spark_ranks > 0) { inputs.tp2_workspace } else { [0; 2] },
        minimum_expert_layers: 0,
        onboard: inputs.onboard, expert_gpus: usize::from(!inputs.experts.is_empty()),
        policy: LayerPolicy { default: vec![LayerMode::HeadSplit, LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }], by_kind: Vec::new(), ownership: LayerOwnership::Alternating, ..LayerPolicy::default() },
        hops: HopSpec { storage: crate::placement::HopStorage::ReceiveSlots, row_bytes: cfg.hidden as u64 * 2, rows: inputs.prefill_rows.max(DECODE_ROWS),
            lanes: inputs.prefill_lanes, entry_gpu: 0, head_gpu: 0, terminal: if gpus == 2 { TerminalHome::Replicated } else { TerminalHome::AttentionOwner } },
        // `context` is the recorded default where the build carries its
        // programs. Without them auto keeps heads; an explicit layers request
        // still runs owned attention through whole-width programs, while a
        // chosen `context` fails closed in `solve_working_set` (#202).
        executor: if inputs.context.is_some() { super::GLM5 }
            else { ExecutorModes { attention: AttentionPlacement::Heads, ..super::GLM5 } },
    })
}

/// Configure the same owner policy and full-head workspace charge at plan and load time.
fn configure_attention(req: &mut PlacementRequest, attention: Option<AttentionPlacement>) -> Result<(), PlacementError> {
    req.attention_placement = attention;
    if attention == Some(AttentionPlacement::Layers) {
        // Only explicit layers loads owner-local operands. Keep raw/default
        // requests from admitting these modes after loading split weights.
        for layer in &mut req.layers {
            layer.modes.extend([LayerMode::Whole { gpu: 0, ffn: FfnMode::Split },
                LayerMode::Whole { gpu: 1, ffn: FfnMode::Split }]);
        }
        validate_layer_pipeline_lanes(req.hops.lanes, req.hops.lanes as usize)
            .map_err(|_| PlacementError::Inventory("GLM layers requires one or two prefill lanes"))?;
        req.owner_local_kv_scratch("padded decode scratch page", FLOOR)?;
    }
    Ok(())
}

/// Whole-attention/indexer groups follow the solver's pure alternating rule.
pub fn attention_owners(cfg: &GlmDsaConfig, layers: usize) -> Result<Vec<u8>, PlacementError> {
    if layers == 0 || layers > cfg.layers || layers > cfg.indexers.len() {
        return Err(PlacementError::Inventory("GLM attention owner layer extent"));
    }
    super::super::attention::alternating_owners(
        (0..layers).map(|layer| (true, Some(cfg.index_source(layer) as u16))), 0)
}

/// Storage allocated for the selected Spark suffix, rather than configured peers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlmWorkingSet {
    pub spark: SparkWorkingSet,
    /// Family-local TP2 payload/backend costs for this selected branch.
    pub tp2_workspace: [u64; 2],
}

/// Evaluate each executable local prefix with its own intake, lane and TP2 costs.
/// An empty Spark suffix is a candidate, never a subtraction after admission.
pub fn solve_working_set(inputs: &GlmInputs<'_>, serial_tp2_workspace: [u64; 2],
    attention: Option<AttentionPlacement>) -> Result<(Placement, GlmWorkingSet), PlacementError> {
    request(inputs)?;
    // `context` is the auto default on a two-GPU pair, and the requested mode
    // when asked for. A build whose manifest lacks its context programs cannot
    // run it: fail closed with what to rebuild rather than silently
    // downgrading the executor to heads (issue #202). Explicit heads is
    // unaffected.
    let context_chosen = attention.map_or(inputs.gpus.len() == 2,
        |mode| mode == AttentionPlacement::Context);
    if context_chosen && inputs.context.is_none() {
        return Err(PlacementError::MissingContextPrograms { family: "glm5", programs: context_program_list() });
    }
    let routed = if inputs.skip_routed_experts { 0 }
        else { inputs.layers.saturating_sub(inputs.cfg.first_moe_layer) };
    let run = |branch: &GlmInputs<'_>, cap: Option<u64>, floor: Option<u64>| {
        let mut req = request(branch)?;
        configure_attention(&mut req, attention)?;
        let owned = if attention == Some(AttentionPlacement::Layers) {
            let owners = attention_owners(branch.cfg, branch.layers)?;
            req.fixed.retain(|d| d.group != "decode graph allowance");
            let head_rows = (DECODE_ROWS + branch.prefill_rows * branch.prefill_lanes) * branch.cfg.heads as u64 / 2;
            for gpu in 0..2 {
                req.fixed.push(Demand::new(gpu, Category::Workspace, "whole attention rows",
                    head_rows * (576 + 512) * 2, Basis::Formula));
            }
            Some(owners)
        } else { None };
        if let Some(cap) = cap { req.pool.ceiling = req.pool.ceiling.min(cap); }
        if let Some(floor) = floor { req.pool.floor = floor; }
        let tp2_workspace = req.tp2_workspace;
        let mut placed = solve(&req)?;
        if let Some(owners) = owned {
            placed = super::super::settle_pool((placed.pool_tokens, placed.layers.clone()), |candidate, layers| {
                let context = crate::serving_capacity::pool_context(branch.max_context as usize,
                    branch.automatic_context, candidate as usize, 256)
                    .map_err(|_| PlacementError::Inventory("GLM admitted pool cannot support context"))?;
                let tp2: Vec<_> = layers.iter().map(|layer| layer.experts == ExpertHome::RtxTp2).collect();
                let startup = crate::serving_capacity::glm_graphs::glm_owned_startup_graphs(
                    context, (candidate as usize).div_ceil(PAGE_ROWS as usize), &owners, &tp2)
                    .map_err(|_| PlacementError::Inventory("GLM owned startup graphs"))?;
                req.fixed.retain(|d| d.group != "graphs" && d.group != "graph growth");
                for rank in 0..2 {
                    for item in startup.inventory.items(rank) {
                        req.fixed.push(Demand::new(rank as u8, item.category, item.group, item.bytes, item.basis)
                            .with_lifetime(item.lifetime));
                    }
                }
                req.pool.ceiling = req.pool.ceiling.min(candidate);
                let next = solve(&req)?;
                Ok::<_, PlacementError>((next.pool_tokens, next.layers.clone(), next))
            })?;
        }
        let rtx_layers = placed.layers.iter().enumerate().filter_map(|(layer, home)|
            matches!(home.experts, ExpertHome::RtxWhole { .. } | ExpertHome::RtxTp2).then_some(layer)).collect();
        let spark_layers = placed.layers.iter().enumerate().filter_map(|(layer, home)|
            (home.experts == ExpertHome::Spark).then_some(layer)).collect();
        Ok((placed, GlmWorkingSet {
            spark: SparkWorkingSet { rtx_layers, spark_layers, spark_ranks: branch.spark_ranks,
                prefill_lanes: branch.prefill_lanes },
            tp2_workspace,
        }))
    };
    if inputs.spark_ranks == 0 || routed == 0 {
        let mut branch = inputs.clone();
        branch.spark_ranks = 0;
        branch.prefill_lanes = 1;
        branch.tp2_workspace = serial_tp2_workspace;
        return run(&branch, None, None);
    }
    let counts: Vec<_> = match inputs.onboard.layers(routed) {
        Some(n) => vec![n],
        None if inputs.experts.is_empty() => vec![0],
        None => (0..=routed).collect(),
    };
    let policy = request(inputs)?.pool;
    let cap = inputs.onboard.layers(routed).is_none().then_some(policy.target);
    let requested = policy.requested;
    let mut best: Option<((u64, u64), Placement, GlmWorkingSet)> = None;
    let mut refusal = None;
    for n in counts {
        let mut branch = inputs.clone();
        branch.onboard = Onboard::Layers(n);
        if n == routed {
            branch.spark_ranks = 0;
            branch.prefill_lanes = 1;
            branch.tp2_workspace = serial_tp2_workspace;
        }
        let floor = match inputs.onboard {
            Onboard::ExpertsFirst { pool_floor } => Some(requested.unwrap_or(pool_floor
                .max(request(&branch)?.pool.floor))),
            // Automatic Spark layouts can clamp serving context to the admitted pool.
            Onboard::Auto if requested.is_none() && branch.spark_ranks > 0 => Some(PAGE_ROWS),
            _ => None,
        };
        match run(&branch, cap, floor) {
            Ok((placed, working)) => {
                let expected = (inputs.cfg.first_moe_layer + n..inputs.layers).collect::<Vec<_>>();
                if working.spark.spark_layers != expected || placed.onboard_layers != n {
                    return Err(PlacementError::Inventory("GLM selected Spark set disagrees with solved homes"));
                }
                let score = match inputs.onboard {
                    Onboard::ExpertsFirst { .. } => (n as u64, placed.pool_tokens),
                    _ => (placed.pool_tokens, n as u64),
                };
                if best.as_ref().is_none_or(|(previous, ..)| score > *previous) {
                    best = Some((score, placed, working));
                }
            }
            Err(error) => { if refusal.is_none() { refusal = Some(error); } }
        }
    }
    if let Some((_, placed, working)) = best { return Ok((placed, working)); }
    if let Onboard::ExpertsFirst { pool_floor } = inputs.onboard {
        if requested.is_none() && pool_floor < inputs.max_context {
            let mut branch = inputs.clone();
            branch.onboard = Onboard::Layers(0);
            return run(&branch, cap, Some(pool_floor));
        }
    }
    Err(refusal.unwrap_or(PlacementError::Inventory("GLM has no executable Spark working set")))
}

/// Header-only draft storage shared by plan and serve (mode 0 is GLM's W8A16).
pub fn drafter_bytes(snapshot: &std::path::Path, slots: usize, sequences: usize,
    sms: u64, bf16: bool) -> anyhow::Result<u64> {
    use crate::families::glm5::draft_representation::{draft_resident_bytes_with_mode, GlmDraftRepresentation};
    let config = crate::plan::checkpoint::read_json(&snapshot.join("config.json"))?;
    let (owned, scratch) = draft_resident_bytes_with_mode(&config, slots, sequences, sms,
        if bf16 { GlmDraftRepresentation::Bf16Only } else { GlmDraftRepresentation::Fp8Only }, 0)?;
    owned.checked_add(scratch).ok_or_else(|| anyhow::anyhow!("GLM draft storage overflow"))
}

/// Largest transient BF16 source retained while packing one draft projection.
pub fn drafter_staging(snapshot: &std::path::Path, slots: usize, sequences: usize,
    bf16: bool) -> anyhow::Result<u64> {
    use crate::families::glm5::draft_representation::{draft_geometry, GlmDraftCapacity,
        GlmDraftRepresentation, GlmDraftRuntimeLayout};
    let config = crate::plan::checkpoint::read_json(&snapshot.join("config.json"))?;
    let (geometry, block) = draft_geometry(&config)?;
    let capacity = GlmDraftCapacity::new(slots, sequences, usize::try_from(block)?)?;
    Ok(GlmDraftRuntimeLayout::new(geometry,
        if bf16 { GlmDraftRepresentation::Bf16Only } else { GlmDraftRepresentation::Fp8Only },
        capacity, 2048)?.weights.max_load_staging)
}

pub fn manifest_scratch(cfg: &GlmDsaConfig, split: bool, manifest: &serde_json::Value)
    -> anyhow::Result<[GlmScratch; 2]> {
    let programs = manifest["programs"].as_array().ok_or_else(|| anyhow::anyhow!("program manifest has no programs"))?;
    let lookup = |name: &str| programs.iter().find(|p| p["name"].as_str() == Some(name))
        .map(|p| p["scratch_bytes_at_capacity"]["scratch"].as_u64().unwrap_or(0));
    Ok([step_scratch(cfg, split, true, lookup)?, step_scratch(cfg, split, false, lookup)?])
}

/// Local package arithmetic, including its input and routed-partial representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmLocalFormat {
    Exl3,
    Nvfp4 { w4a4: bool },
}

/// Header-only TP2 inventory. Runtime must attest the native ABI before execution.
#[derive(Debug, Clone)]
pub struct LocalInventory {
    pub package: std::path::PathBuf,
    pub format: GlmLocalFormat,
    pub experts: Vec<ExpertCost>,
    pub backend_workspace: u64,
    pub extra_workspace: [u64; 2],
}

impl LocalInventory {
    pub fn workspace(&self) -> [u64; 2] {
        self.extra_workspace.map(|bytes| bytes + self.backend_workspace)
    }

    pub fn workspace_for(&self, cfg: &GlmDsaConfig, rows: u64, lanes: u64,
        exchange_f32: bool) -> [u64; 2] {
        local_extra_workspace(cfg.hidden as u64, cfg.experts as u64, cfg.topk as u64,
            rows, lanes, exchange_f32).map(|bytes| bytes + self.backend_workspace)
    }
}

fn local_extra_workspace(h: u64, experts: u64, topk: u64, rows: u64, lanes: u64,
    exchange_f32: bool) -> [u64; 2] {
    let dtype = if exchange_f32 { 4 } else { 2 };
    let payload = (DECODE_ROWS * h * dtype).max(FLOOR) + lanes * (rows * h * dtype).max(FLOOR);
    let exchange_rows = rows.max(DECODE_ROWS);
    let routes = 2 * lanes * (exchange_rows * topk * 8).next_multiple_of(16)
        + ((2 * lanes + 1) * 16).max(FLOOR) + FLOOR;
    let wide_exchange = if exchange_f32 { 4 * lanes * exchange_rows * h * 2 } else { 0 };
    // Rank 1 normally owns 256-byte placeholders for logits, ids, weights and wire.
    let peer_routes = [experts * 4, topk * 4, topk * 4, h + h / 32].into_iter().map(|width|
        (DECODE_ROWS * width).max(FLOOR) - FLOOR
            + lanes * ((rows * width).max(FLOOR) - FLOOR)).sum::<u64>();
    // These optional deltas open with the RTX arena; shared step/attention rows stay Always.
    let rows = [("local expert payloads", [payload; 2]), ("canonical route exchange", [routes; 2]),
        ("FP32 exchange widening", [wide_exchange; 2]),
        ("peer local router rows", [0, peer_routes])];
    let mut extra = [0; 2];
    for (name, bytes) in rows {
        for (rank, bytes) in bytes.into_iter().enumerate() {
            let row = GlmBufferRow::new(GlmComponent::RtxExperts,
                Demand::new(rank as u8, Category::Workspace, name, bytes, Basis::Formula));
            extra[rank] += row.demand.bytes;
        }
    }
    extra
}

fn native_layer_resident(tensors: &crate::formats::fp8_experts::Fp8ExpertTensors,
    ranks: usize, rank: usize, slicing: crate::formats::fp8_experts::Slicing) -> anyhow::Result<u64> {
    use crate::formats::fp8_experts::Fp8Projection;
    Fp8Projection::ALL.iter().try_fold(0u64, |sum, &projection| {
        let (weights, _) = tensors.slice_bytes_with(projection, ranks, rank, slicing)?;
        let weights = tensors.shape().experts.checked_mul(weights)
            .ok_or_else(|| anyhow::anyhow!("GLM expert weights overflow"))?.max(FLOOR as usize) as u64;
        let scales = tensors.scale_region_bytes_with(projection, ranks, rank, slicing)?
            .max(FLOOR as usize) as u64;
        sum.checked_add(weights).and_then(|n| n.checked_add(scales))
            .ok_or_else(|| anyhow::anyhow!("GLM expert resident storage overflow"))
    })
}

/// Physical Spark weight arenas for the selected routed suffix. The caller
/// supplies the qualified package's actual slicing, not a proportional tensor
/// share or an assumption based on the configured peer count.
pub fn spark_resident_bytes(catalog: &crate::OfficialV41Catalog, layers: &[usize],
    ranks: usize, slicing: crate::formats::fp8_experts::Slicing) -> anyhow::Result<Vec<u64>> {
    use crate::formats::fp8_experts::{ExpertFormat, Slicing};
    if layers.is_empty() {
        anyhow::ensure!(ranks == 0, "GLM empty Spark suffix must select zero ranks");
        return Ok(Vec::new());
    }
    let shape = catalog.routed_experts();
    anyhow::ensure!(matches!(ranks, 2 | 3 | 4 | 6) && shape.hidden == 6144
        && shape.intermediate == 2048 && shape.experts == 256 && shape.topk == 8
        && shape.layers > shape.first_layer && layers.last() == Some(&(shape.layers - 1))
        && layers.iter().all(|&layer| (shape.first_layer..shape.layers).contains(&layer))
        && layers.windows(2).all(|w| w[0] + 1 == w[1]), "GLM Spark rank/selected-suffix geometry");
    if let Some(exl3) = catalog.exl3() {
        anyhow::ensure!(slicing == Slicing::Blocks(128), "GLM EXL3 needs disjoint whole-128-row Spark slices");
        return (0..ranks).map(|rank| layers.iter().try_fold(0u64, |sum, &layer| {
            let arena = exl3.residency(crate::V41Exl3Layer::Backbone(layer), ranks, rank)?
                .device_arena_layout()?.1 as u64;
            sum.checked_add(arena).ok_or_else(|| anyhow::anyhow!("GLM Spark EXL3 residency overflow"))
        })).collect();
    }
    let tensors = catalog.fp8().filter(|t| matches!(t.format(), ExpertFormat::Fp8Block128 | ExpertFormat::Nvfp4))
        .ok_or_else(|| anyhow::anyhow!("GLM Spark inventory needs EXL3, FP8 or native NVFP4 headers"))?;
    for &layer in layers { tensors.validate_layer(layer)?; }
    (0..ranks).map(|rank| native_layer_resident(tensors, ranks, rank, slicing)?
        .checked_mul(layers.len() as u64).ok_or_else(|| anyhow::anyhow!("GLM Spark residency overflow"))).collect()
}

pub fn local_inventory(catalog: &crate::OfficialV41Catalog, manifest: &std::path::Path,
    selected_layers: usize, rows: u64, lanes: u64, exchange_f32: bool) -> anyhow::Result<LocalInventory> {
    local_inventory_selected(catalog, manifest, selected_layers, rows, lanes, exchange_f32, true)
}

/// Explicit activation selection keeps package admission deterministic in tests.
pub fn local_inventory_selected(catalog: &crate::OfficialV41Catalog, manifest: &std::path::Path,
    selected_layers: usize, rows: u64, lanes: u64, exchange_f32: bool, a4: bool)
    -> anyhow::Result<LocalInventory> {
    let shape = catalog.routed_experts();
    anyhow::ensure!(shape.hidden == 6144 && shape.intermediate == 2048 && shape.experts == 256
        && shape.topk == 8 && (1..=4096).contains(&rows) && (1..=4).contains(&lanes)
        && (shape.first_layer..=shape.layers).contains(&selected_layers), "GLM TP2 geometry/layer extent");
    let Some(exl3) = catalog.exl3() else {
        return nvfp4_inventory(catalog, manifest, selected_layers, rows, lanes, exchange_f32, a4);
    };
    let tiers = exl3.decoder_tiers().iter().map(usize::to_string).collect::<String>();
    let stem = format!("exl3-glm-k{tiers}");
    let parent = manifest.parent().ok_or_else(|| anyhow::anyhow!("GLM program manifest parent"))?;
    let package = [parent.join("exl3").join(&stem), parent.join("../lib/exl3").join(&stem)]
        .into_iter().map(|p| p.join("rtx-tp2")).find(|p| p.is_dir())
        .ok_or_else(|| anyhow::Error::new(MissingLocalPackage(format!("{stem}/rtx-tp2"))))?;
    let capacities = [1u64, 16, 80, 256, 1024, 4096];
    let max_rows = rows.max(DECODE_ROWS);
    let compiled = capacities.into_iter().find(|&c| c >= max_rows).unwrap();
    let manifests = capacities.into_iter().filter(|&c| c <= compiled).map(|capacity| {
        let directory = package.join(format!("m{capacity}"));
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(directory.join("v41_exl3.json"))?)?;
        anyhow::ensure!(value["hidden"].as_u64() == Some(shape.hidden as u64)
            && value["intermediate"].as_u64() == Some(shape.intermediate as u64 / 2)
            && value["experts"].as_u64() == Some(shape.experts as u64)
            && value["top_k"].as_u64() == Some(shape.topk as u64)
            && value["capacity"].as_u64() == Some(capacity)
            && value["output_dtype"].as_str() == Some("fp32")
            && value["bits"] == serde_json::json!(exl3.decoder_tiers()),
            "GLM TP2 EXL3 capacity/geometry/tiers/output mismatch in {}", directory.display());
        let library = directory.join("libcuteafd_exl3.so");
        anyhow::ensure!(library.is_file(), "GLM TP2 missing binary {}", library.display());
        let lut = value["trellis_lut"]["file"].as_str()
            .ok_or_else(|| anyhow::anyhow!("GLM TP2 missing trellis LUT asset name"))?;
        anyhow::ensure!(directory.join(lut).is_file(), "GLM TP2 missing trellis LUT in {}", directory.display());
        if value["direct"].as_bool() != Some(true) {
            let route = value["route_preparation"]["manifest"].as_str()
                .ok_or_else(|| anyhow::anyhow!("GLM TP2 missing packed route export manifest"))?;
            let route_manifest = directory.join(route);
            // Match Exl3Execution: route assets live beside the declared manifest.
            for asset in [route_manifest.clone(), route_manifest.parent().unwrap().join("libv41_exl3_routes.so")] {
                anyhow::ensure!(asset.is_file(), "GLM TP2 missing route asset {}", asset.display());
            }
        }
        // Runtime separately attests ELF geometry and LUT hashes before execution.
        Ok(value)
    }).collect::<anyhow::Result<Vec<_>>>()?;
    let backend_workspace = crate::serving_capacity::exl3_workspace_bytes(&manifests, true)?
        + max_rows * shape.hidden as u64 * 4;
    let mut experts = Vec::new();
    for layer in shape.first_layer..selected_layers {
        let whole = exl3.residency(crate::V41Exl3Layer::Backbone(layer), 1, 0)?.device_arena_layout()?.1 as u64;
        let mut half = [Bytes2::default(); 2];
        for (rank, bytes) in half.iter_mut().enumerate() {
            bytes.resident = exl3.residency(crate::V41Exl3Layer::Backbone(layer), 2, rank)?.device_arena_layout()?.1 as u64;
        }
        for suffix in ["weight", "e_score_correction_bias"] {
            half[1].resident += catalog.tensor(&format!("model.layers.{layer}.mlp.gate.{suffix}"))?
                .metadata.byte_length.max(FLOOR);
        }
        experts.push(ExpertCost { whole: Bytes2 { resident: whole, staging: 0 }, half,
            tp2: true, spark_ok: true });
    }
    Ok(LocalInventory { package, format: GlmLocalFormat::Exl3, experts, backend_workspace,
        extra_workspace: local_extra_workspace(shape.hidden as u64, shape.experts as u64,
            shape.topk as u64, rows, lanes, exchange_f32) })
}

fn nvfp4_inventory(catalog: &crate::OfficialV41Catalog, manifest: &std::path::Path,
    selected_layers: usize, rows: u64, lanes: u64, exchange_f32: bool, a4: bool)
    -> anyhow::Result<LocalInventory> {
    use crate::formats::fp8_experts::{ExpertFormat, Slicing};
    let tensors = catalog.fp8().filter(|t| t.format() == ExpertFormat::Nvfp4)
        .ok_or_else(|| anyhow::anyhow!("GLM RTX TP2 needs EXL3 or native ModelOpt NVFP4 expert tensors"))?;
    anyhow::ensure!(!exchange_f32,
        "GLM NVFP4 TP2 emits BF16 routed partials; use BF16 exchange (no BF16-to-FP32 partial adapter)");
    let parent = manifest.parent().ok_or_else(|| anyhow::anyhow!("GLM program manifest parent"))?;
    let roots = [parent.join("fp8"), parent.join("../lib/fp8")];
    let find = |name: &str| roots.iter().map(|root| root.join(name).join("tp2")).find(|p| p.is_dir());
    anyhow::ensure!(a4, "checkpoint-declared GLM NVFP4 requires W4A4 at every row count");
    let package = find("fp8-glm-nvfp4a4").ok_or_else(|| anyhow::Error::new(
        MissingLocalPackage("calibrated GLM NVFP4 W4A4 tp2".into())))?;
    let w4a4 = true;
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(
        package.parent().unwrap().join("manifest.json"))?)?;
    let shape = catalog.routed_experts();
    let layout = &value["layouts"]["tp2"];
    let geometry = if w4a4 { "glm_nvfp4a4" } else { "glm_nvfp4" };
    anyhow::ensure!(value["schema"].as_str() == Some("cuteafd.fp8moe-package.v1")
        && value["role"].as_str() == Some("coordinator") && value["geometry"].as_str() == Some(geometry)
        && layout["tp"].as_u64() == Some(2) && layout["hidden"].as_u64() == Some(shape.hidden as u64)
        && layout["intermediate"].as_u64() == Some(shape.intermediate as u64)
        && layout["slice"].as_u64() == Some(shape.intermediate as u64 / 2)
        && layout["experts"].as_u64() == Some(shape.experts as u64)
        && layout["top_k"].as_u64() == Some(shape.topk as u64)
        && layout["input"].as_str() == Some("bf16") && layout["weights"].as_str() == Some("nvfp4")
        && layout["swiglu_limit"].as_f64() == Some(0.0),
        "GLM TP2 NVFP4 package geometry/input/format mismatch in {}", package.display());
    // NVFP4 has one program per capacity. Non-default forms need the ABI's
    // largest-form scratch query, which this header-only inventory cannot infer.
    anyhow::ensure!(layout["prefill_forms"].as_object().is_some_and(|forms|
        forms.values().all(|v| v.as_array().is_some_and(Vec::is_empty))),
        "GLM TP2 NVFP4 non-default prefill forms need largest-form scratch metadata");
    let capacities = layout["capacities"].as_array()
        .ok_or_else(|| anyhow::anyhow!("GLM TP2 NVFP4 capacities missing"))?
        .iter().map(|c| -> anyhow::Result<(u64, u64)> {
            Ok((c["capacity"].as_u64().filter(|&n| n > 0)
                .ok_or_else(|| anyhow::anyhow!("GLM TP2 NVFP4 capacity is invalid"))?,
                c["scratch_bytes"].as_u64()
                    .ok_or_else(|| anyhow::anyhow!("GLM TP2 NVFP4 scratch is invalid"))?))
        }).collect::<anyhow::Result<Vec<_>>>()?;
    anyhow::ensure!(!capacities.is_empty() && capacities.windows(2).all(|w| w[0].0 < w[1].0),
        "GLM TP2 NVFP4 capacities must be strictly increasing");
    let max_rows = rows.max(DECODE_ROWS);
    let scratch = capacities.iter().find(|&&(capacity, _)| capacity >= max_rows)
        .map(|&(_, bytes)| bytes.max(FLOOR)).ok_or_else(|| anyhow::anyhow!(
            "GLM TP2 NVFP4 package has no capacity for {max_rows} rows"))?;
    let library = package.join("libcuteafd_fp8moe.so");
    anyhow::ensure!(library.is_file(), "GLM TP2 missing binary {}", library.display());
    let backend_workspace = scratch.checked_add(max_rows * shape.hidden as u64 * 2)
        .ok_or_else(|| anyhow::anyhow!("GLM TP2 NVFP4 workspace overflow"))?;
    let resident = |tp, rank| native_layer_resident(tensors, tp, rank, Slicing::Blocks(128));
    let whole = Bytes2 { resident: resident(1, 0)?, staging: 0 };
    let halves = [resident(2, 0)?, resident(2, 1)?];
    let experts = (shape.first_layer..selected_layers).map(|layer| -> anyhow::Result<ExpertCost> {
        tensors.validate_layer(layer)?;
        let mut half = halves.map(|resident| Bytes2 { resident, staging: 0 });
        for suffix in ["weight", "e_score_correction_bias"] {
            half[1].resident += catalog.tensor(&format!("model.layers.{layer}.mlp.gate.{suffix}"))?
                .metadata.byte_length.max(FLOOR);
        }
        Ok(ExpertCost { whole, half, tp2: true, spark_ok: true })
    }).collect::<anyhow::Result<Vec<_>>>()?;
    Ok(LocalInventory { package, format: GlmLocalFormat::Nvfp4 { w4a4 }, experts, backend_workspace,
        extra_workspace: local_extra_workspace(shape.hidden as u64, shape.experts as u64,
            shape.topk as u64, rows, lanes, false) })
}

#[derive(Debug, thiserror::Error)]
#[error("missing {0} package")]
struct MissingLocalPackage(String);

/// Shared planner/runtime activation policy. Auto remains remote when the build lacks local exports.
#[allow(clippy::too_many_arguments)]
pub fn select_local_inventory(catalog: &crate::OfficialV41Catalog, manifest: &std::path::Path,
    layers: usize, gpus: usize, rows: u64, lanes: u64, exchange_f32: bool,
    onboard: Onboard, skip: bool) -> anyhow::Result<Option<LocalInventory>> {
    if skip || onboard.layers(layers.saturating_sub(catalog.routed_experts().first_layer)) == Some(0)
        || layers <= catalog.routed_experts().first_layer { return Ok(None); }
    if gpus != 2 {
        anyhow::ensure!(matches!(onboard, Onboard::Auto | Onboard::ExpertsFirst { .. }),
            "GLM local experts require a two-GPU TP2 executor");
        return Ok(None);
    }
    let automatic = matches!(onboard, Onboard::Auto | Onboard::ExpertsFirst { .. });
    let supported = catalog.exl3().is_some() || catalog.fp8().is_some_and(|t|
        t.format() == crate::formats::fp8_experts::ExpertFormat::Nvfp4);
    if automatic && !supported { return Ok(None); }
    match local_inventory(catalog, manifest, layers, rows, lanes, exchange_f32) {
        Ok(local) => Ok(Some(local)),
        Err(error) if automatic && error.is::<MissingLocalPackage>() => {
            tracing::info!(reason = %error, "GLM RTX TP2 package unavailable; automatic placement uses Sparks");
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

pub fn local_exchange_f32() -> anyhow::Result<bool> {
    match std::env::var("CUTEAFD_TP2_EXCHANGE").ok().as_deref() {
        None | Some("") | Some("auto") | Some("bf16") => Ok(false),
        Some("f32") => Ok(true),
        Some(value) => anyhow::bail!("CUTEAFD_TP2_EXCHANGE must be auto, bf16 or f32, got {value:?}"),
    }
}

pub fn default_onboard() -> Onboard { Onboard::Auto }

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> GlmDsaConfig {
        let mut indexers = vec![GlmIndexer::Full; 3];
        indexers.extend((3..78).map(|l| if (l - 2) % 4 == 0 { GlmIndexer::Full } else { GlmIndexer::Shared }));
        GlmDsaConfig { vocab_size: 154880, hidden: 6144, layers: 78, heads: 64, q_lora_rank: 2048,
            kv_lora_rank: 512, qk_nope_head_dim: 192, qk_rope_head_dim: 64, v_head_dim: 256,
            index_heads: 32, index_head_dim: 128, index_topk: 2048, indexers, first_moe_layer: 3,
            dense_intermediate: 12288, experts: 256, topk: 8, moe_intermediate: 2048,
            shared_experts: 1, routed_scale: 2.5, rope_theta: 8000000.0, rms_norm_eps: 1e-5,
            mtp_layers: 1, eos_tokens: vec![154820] }
    }

    fn inputs<'a>(cfg: &'a GlmDsaConfig, cards: &[u64]) -> GlmInputs<'a> {
        GlmInputs { cfg, layers: cfg.layers, gpus: cards.iter().enumerate().map(|(rank, &total)| (total,
            Baseline::Planned { context_bytes: GIB, loaded_bytes: if rank == 0 { 14 * GIB } else { 12 * GIB } })).collect(),
            headroom_bytes: 2 * GIB, spark_ranks: 4, native_experts: false, native_workspace: 0, skip_routed_experts: false, prefill_rows: 4096, prefill_lanes: 3,
            max_context: 1 << 20, automatic_context: true, scratch: Some([GlmScratch { programs: 16 << 20, topk: 1 << 20 }; 2]),
            drafter_bytes: 3 * GIB, drafter_staging: 0, pending_code: vec![],
            experts: (cfg.first_moe_layer..cfg.layers).map(|_| ExpertCost {
                whole: Bytes2 { resident: 4 * GIB, staging: 0 }, half: [Bytes2 { resident: 2 * GIB, staging: 0 }; 2],
                tp2: cards.len() == 2, spark_ok: true }).collect(),
            expert_workspace: 128 << 20, tp2_workspace: [128 << 20; 2], requested_pool: None,
            onboard: default_onboard(), full_prefill_logits: false, context: None }
    }

    /// The heads-only fixtures here carry no context programs, so a two-GPU
    /// solve must request `heads` explicitly: auto resolves to `context`,
    /// which fails closed without its programs (issue #202).
    const HEADS: Option<AttentionPlacement> = Some(AttentionPlacement::Heads);

    #[test]
    fn native_split_workspaces_are_exact_and_unsupported_modes_refused() {
        let cfg = config();
        let mut base = inputs(&cfg, &[96 * GIB]);
        base.prefill_lanes = 1;
        base.prefill_rows = 65;
        base.experts.clear();
        base.drafter_bytes = 0;
        base.drafter_staging = 0;
        let mut native = base.clone();
        native.native_experts = true;
        native.native_workspace = 123456;
        let ordinary = request(&base).unwrap();
        let split = request(&native).unwrap();
        let term = |request: &PlacementRequest, name: &str| request.fixed.iter()
            .find(|demand| demand.group == name).unwrap().bytes;
        let h = cfg.hidden as u64;
        assert_eq!(term(&split, "steps") - term(&ordinary, "steps"),
            123456);
        assert_eq!(term(&split, "Spark intake planes"), 4 * 4096 * h * 2);
        let mut invalid = native.clone(); invalid.prefill_lanes = 2;
        assert!(request(&invalid).is_err());
        let mut invalid = native.clone(); invalid.scratch = None;
        assert!(request(&invalid).is_err());
        let mut invalid = native.clone(); invalid.spark_ranks = 0;
        assert!(request(&invalid).is_err());
        let mut pair = native.clone(); pair.gpus.push(pair.gpus[0]);
        assert!(request(&pair).is_ok());
        let mut invalid = pair; invalid.spark_ranks = 6;
        assert!(request(&invalid).is_err());
    }

    #[test]
    fn planner_equals_runtime_glm5() {
        let cfg = config();
        for cards in [vec![96 * GIB], vec![96 * GIB; 2], vec![96 * GIB, 80 * GIB]] {
            for onboard in [Onboard::Auto, Onboard::Layers(2), Onboard::ExpertsFirst { pool_floor: 262144 }] {
                let planned = inputs(&cfg, &cards);
                let planned = GlmInputs { onboard, ..planned };
                let mut runtime = planned.clone();
                for (total, baseline) in &mut runtime.gpus {
                    let Baseline::Planned { context_bytes, loaded_bytes } = *baseline else { unreachable!() };
                    *baseline = Baseline::Measured { free_bytes: *total - context_bytes - loaded_bytes };
                }
                let (p, pw) = solve_working_set(&planned, planned.tp2_workspace, HEADS).unwrap();
                let (r, rw) = solve_working_set(&runtime, runtime.tp2_workspace, HEADS).unwrap();
                assert_eq!(p, r);
                assert_eq!(pw, rw);
                if cards.len() == 2 {
                    assert!(p.expert_ranges.iter().all(|r| r.layers == 0));
                    assert!(p.layers.iter().all(|l| l.mode == LayerMode::HeadSplit));
                    if onboard == Onboard::Auto { assert_eq!(p.onboard_layers, 0); }
                    if onboard == Onboard::Layers(2) { assert_eq!(p.tp2.unwrap().layers, 2); }
                }
                eprintln!("GLM {cards:?} {onboard}: {}", p.summary());
            }
        }
    }

    #[test]
    fn drafter_movable_places_exactly_like_the_former_fixed_demands() {
        let cfg = config();
        for cards in [vec![96 * GIB], vec![96 * GIB; 2], vec![96 * GIB, 80 * GIB]] {
            for onboard in [Onboard::Auto, Onboard::Layers(2)] {
                let mut i = GlmInputs { onboard, ..inputs(&cfg, &cards) };
                i.drafter_staging = 1 << 28;
                let mut req = request(&i).unwrap();
                req.attention_placement = HEADS;
                assert_eq!(req.movables.len(), 1);
                assert_eq!(req.movables[0].allowed, [0]);
                crate::placement::draft::assert_same_as_fixed(&req);
                assert_eq!(crate::placement::draft::home(&solve(&req).unwrap()), Some(0));
            }
        }
    }

    #[test]
    fn skipped_experts_admit_serial_diagnostics_without_a_local_backend() {
        let cfg = config();
        for cards in [vec![96 * GIB], vec![96 * GIB; 2]] {
            let mut i = inputs(&cfg, &cards);
            i.spark_ranks = 0;
            i.prefill_lanes = prefill_lanes(0, 3);
            i.experts.clear();
            i.expert_workspace = 0;
            i.tp2_workspace = [0; 2];
            i.drafter_bytes = 0;
            i.requested_pool = Some(65536);
            i.onboard = Onboard::Layers(0);
            assert!(solve(&request(&i).unwrap()).is_err(), "normal Spark-free execution must fail closed");
            i.skip_routed_experts = true;
            let request = request(&i).unwrap();
            assert!(request.layers.iter().all(|layer| layer.experts.is_none()));
            let planes = request.fixed.iter().filter(|d| d.group == "diagnostic zero expert planes")
                .map(|d| (d.gpu, d.bytes)).collect::<Vec<_>>();
            assert_eq!(planes, vec![(0, 4 * 4096 * 6144 * 2)]);
            let p = solve(&request).unwrap();
            assert_eq!(p.pool_tokens, 65536);
            assert_eq!(p.onboard_layers, 0);
            assert!(p.tp2.is_none());
            i.prefill_lanes = 3;
            assert!(super::request(&i).is_err(), "skip diagnostics must retain serial admission");
        }
        for configured in 1..=4 {
            assert_eq!(prefill_lanes(0, configured), 1);
            assert_eq!(prefill_lanes(4, configured), configured);
        }
    }

    #[test]
    fn selected_spark_working_set_matches_planned_and_measured_admission() {
        let cfg = config();
        let routed = cfg.layers - cfg.first_moe_layer;
        for onboard in [Onboard::Auto, Onboard::Layers(routed), Onboard::Fraction(1.0),
            Onboard::ExpertsFirst { pool_floor: 262144 }] {
            let mut planned = inputs(&cfg, &[96 * GIB; 2]);
            planned.onboard = onboard;
            planned.requested_pool = Some(65536);
            for cost in &mut planned.experts {
                cost.half = [Bytes2 { resident: 1 << 20, staging: 0 }; 2];
            }
            let serial = [64 << 20; 2];
            let (p, working) = solve_working_set(&planned, serial, HEADS).unwrap();
            assert_eq!(p.onboard_layers, routed);
            assert_eq!(working.spark.spark_ranks, 0);
            assert_eq!(working.spark.prefill_lanes, 1);
            assert_eq!(working.tp2_workspace, serial);
            assert!(working.spark.spark_layers.is_empty());
            assert!(working.spark.selected_ranks().is_empty());
            assert!(working.spark.includes(WorkingSetComponent::RtxExperts));
            assert!(!working.spark.includes(WorkingSetComponent::SparkExperts));
            assert!(p.items[0].iter().all(|item| item.group != "Spark intake planes"));
            let mut measured = planned.clone();
            for (capacity, baseline) in &mut measured.gpus {
                let Baseline::Planned { context_bytes, loaded_bytes } = *baseline else { unreachable!() };
                *baseline = Baseline::Measured { free_bytes: *capacity - context_bytes - loaded_bytes };
            }
            assert_eq!(solve_working_set(&measured, serial, HEADS).unwrap(), (p, working));
        }
        let mut input = inputs(&cfg, &[96 * GIB; 2]);
        input.requested_pool = Some(65536);
        for onboard in [Onboard::Layers(0), Onboard::Fraction(0.0), Onboard::Layers(2)] {
            input.onboard = onboard;
            let (p, working) = solve_working_set(&input, [64 << 20; 2], HEADS).unwrap();
            let n = onboard.layers(routed).unwrap();
            assert_eq!(p.onboard_layers, n);
            assert_eq!(working.spark.spark_layers, (cfg.first_moe_layer + n..cfg.layers).collect::<Vec<_>>());
            assert_eq!(working.spark.spark_ranks, 4);
            assert_eq!(working.spark.prefill_lanes, 3);
            assert_eq!(working.spark.rtx_layers, (cfg.first_moe_layer..cfg.first_moe_layer + n).collect::<Vec<_>>());
            assert_eq!(working.spark.selected_ranks(), 0..4);
            assert_eq!(working.spark.includes(WorkingSetComponent::RtxExperts), n > 0);
            assert!(working.spark.includes(WorkingSetComponent::SparkExperts));
            assert_eq!(working.tp2_workspace, if n == 0 { [0; 2] } else { input.tp2_workspace });
            let intake = p.items[0].iter().find(|item| item.group == "Spark intake planes").unwrap();
            assert_eq!(intake.bytes, 4 * 4 * 4096 * 6144 * 2);
        }
    }

    #[test]
    fn component_rows_follow_empty_rtx_and_empty_spark_branches() {
        let cfg = config();
        let mut input = inputs(&cfg, &[96 * GIB; 2]);
        input.requested_pool = Some(65536);
        input.onboard = Onboard::Layers(0);
        let without_rtx = solve_working_set(&input, [64 << 20; 2], HEADS).unwrap();
        assert!(without_rtx.1.spark.rtx_layers.is_empty());
        assert_eq!(without_rtx.1.tp2_workspace, [0; 2]);
        let request = request(&input).unwrap();
        assert_eq!(request.tp2_workspace, [0; 2]);
        assert_eq!(request.expert_workspace, 0);
        assert!(request.fixed.iter().any(|row| row.group == "peer exchange"));
        assert!(request.fixed.iter().any(|row| row.group == "steps" && row.bytes > 0));
        input.tp2_workspace = [u64::MAX; 2];
        input.expert_workspace = u64::MAX;
        assert_eq!(solve_working_set(&input, [u64::MAX; 2], HEADS).unwrap(), without_rtx,
            "unselected RTX rows must never affect admission");

        input.onboard = Onboard::Layers(cfg.layers - cfg.first_moe_layer);
        input.expert_workspace = 0;
        input.tp2_workspace = [128 << 20; 2];
        for cost in &mut input.experts { cost.half = [Bytes2 { resident: 1 << 20, staging: 0 }; 2]; }
        let all_local = solve_working_set(&input, [64 << 20; 2], HEADS).unwrap();
        assert!(all_local.1.spark.spark_layers.is_empty());
        assert_eq!(all_local.1.spark.prefill_lanes, 1);
        assert_eq!(all_local.1.tp2_workspace, [64 << 20; 2]);
        input.spark_ranks = 0;
        input.prefill_lanes = 1;
        assert_eq!(solve_working_set(&input, [64 << 20; 2], HEADS).unwrap(), all_local,
            "configured but unselected Spark peers must not affect admission");
        assert!(all_local.0.items.iter().all(|rank| rank.iter().all(|row| row.group != "Spark intake planes")));
        assert!(all_local.0.items.iter().all(|rank| rank.iter().any(|row| row.group == "peer exchange")));
        for component in [GlmComponent::Always, GlmComponent::RtxExperts, GlmComponent::SparkExperts] {
            assert_eq!(component.selected(false, false), component == GlmComponent::Always);
            assert!(component.selected(true, true));
        }
    }

    #[test]
    fn pool_first_evaluates_serial_all_local_candidate_and_empty_routed_set() {
        let cfg = config();
        let mut input = inputs(&cfg, &[96 * GIB; 2]);
        for cost in &mut input.experts { cost.half = [Bytes2 { resident: 1 << 20, staging: 0 }; 2]; }
        let serial = [64 << 20; 2];
        let (placed, working) = solve_working_set(&input, serial, HEADS).unwrap();
        assert_eq!(placed.onboard_layers, cfg.layers - cfg.first_moe_layer);
        assert!(working.spark.spark_layers.is_empty());
        assert_eq!(working.spark.prefill_lanes, 1);
        let with_peers = GlmInputs { onboard: Onboard::Layers(0), ..input.clone() };
        assert!(placed.pool_tokens > solve_working_set(&with_peers, serial, HEADS).unwrap().0.pool_tokens,
            "pool-first must compare each branch's own costs, not take configured peer costs");
        input.spark_ranks = 0;
        input.prefill_lanes = 1;
        assert_eq!(solve_working_set(&input, serial, HEADS).unwrap(), (placed, working));
        input.layers = cfg.first_moe_layer;
        input.experts.clear();
        input.spark_ranks = 4;
        input.prefill_lanes = 3;
        let (dense, working) = solve_working_set(&input, serial, HEADS).unwrap();
        assert!(working.spark.rtx_layers.is_empty() && working.spark.spark_layers.is_empty());
        assert_eq!(working.spark.spark_ranks, 0);
        assert_eq!(working.spark.prefill_lanes, 1);
        assert_eq!(working.tp2_workspace, [0; 2]);
        assert_eq!(dense.pool_tokens, 2 << 20);
    }

    #[test]
    fn selected_spark_free_max_retains_agentic_floor() {
        let cfg = config();
        let routed = cfg.layers - cfg.first_moe_layer;
        let mut input = inputs(&cfg, &[96 * GIB; 2]);
        input.max_context = 65536;
        input.drafter_bytes = 0;
        input.expert_workspace = 0;
        input.onboard = Onboard::ExpertsFirst { pool_floor: 65536 };
        for cost in &mut input.experts { cost.half = [Bytes2 { resident: 1 << 20, staging: 0 }; 2]; }
        let serial = [64 << 20; 2];
        let local = GlmInputs { spark_ranks: 0, prefill_lanes: 1, tp2_workspace: serial,
            onboard: Onboard::Layers(routed), ..input.clone() };
        let request = request(&local).unwrap();
        for (rank, budget) in input.gpus.iter_mut().enumerate() {
            let unit = request.layers.iter().map(|layer| layer.kv_unit.unit_bytes_split[rank]).sum::<u64>()
                + request.pool_overhead[rank];
            let fixed = request.fixed.iter().filter(|row| usize::from(row.gpu) == rank).map(|row| row.bytes).sum::<u64>();
            let capacity = input.headroom_bytes + fixed + unit * (131072 / PAGE_ROWS)
                + serial[rank] + (routed as u64 + 16) * (1 << 20);
            *budget = (capacity, Baseline::Measured { free_bytes: capacity });
        }
        let (placed, working) = solve_working_set(&input, serial, HEADS).unwrap();
        assert!(placed.onboard_layers < routed, "max must not select an all-local branch below 256K");
        assert_eq!(working.spark.spark_ranks, 4);
        assert!(placed.pool_tokens >= 65536);
        input.spark_ranks = 0;
        input.prefill_lanes = 1;
        input.onboard = Onboard::Layers(routed);
        assert!(matches!(solve_working_set(&input, serial, HEADS), Err(PlacementError::BelowFloor { .. })));
        input.requested_pool = Some(131072);
        assert_eq!(solve_working_set(&input, serial, HEADS).unwrap().0.onboard_layers, routed,
            "an explicit smaller pool remains its own floor");
    }

    #[test]
    fn working_set_preserves_pool_first_max_and_explicit_refusals() {
        let cfg = config();
        let mut input = inputs(&cfg, &[96 * GIB; 2]);
        let serial = [64 << 20; 2];
        let (heads, working) = solve_working_set(&input, serial, HEADS).unwrap();
        assert_eq!(heads.onboard_layers, 0);
        assert_eq!(working.spark.spark_ranks, 4);
        assert_eq!(heads.pool_tokens, solve(&request(&input).unwrap()).unwrap().pool_tokens);
        input.onboard = Onboard::ExpertsFirst { pool_floor: 262144 };
        let (max, _) = solve_working_set(&input, serial, HEADS).unwrap();
        assert_eq!(max.onboard_layers, solve(&request(&input).unwrap()).unwrap().onboard_layers);
        assert!(max.onboard_layers > heads.onboard_layers);
        input.onboard = Onboard::Layers(cfg.layers - cfg.first_moe_layer);
        assert!(solve_working_set(&input, serial, HEADS).is_err());
        input.onboard = Onboard::Auto;
        input.requested_pool = Some(2 << 20);
        assert!(matches!(solve_working_set(&input, serial, HEADS), Err(PlacementError::PoolDoesNotFit { .. })));
        input.experts.clear();
        input.requested_pool = Some(65536);
        assert_eq!(solve_working_set(&input, serial, HEADS).unwrap().0.onboard_layers, 0);
        input.onboard = Onboard::Layers(1);
        assert!(solve_working_set(&input, serial, HEADS).is_err());
    }

    #[test]
    fn replicated_heads_cannot_admit_two_million_tokens() {
        let cfg = config();
        let mut i = inputs(&cfg, &[96 * GIB; 2]);
        i.requested_pool = Some(2 << 20);
        assert!(matches!(solve(&request(&i).unwrap()), Err(PlacementError::PoolDoesNotFit { .. })));
        let geometry = glm_cache_geometry(&cfg, cfg.layers, 2).unwrap();
        assert_eq!(geometry.ranks[0].persistent_unit_bytes / PAGE_ROWS, 53940);
        assert_eq!(geometry.ranks[0], geometry.ranks[1]);
    }

    fn local_fixture() -> (tempfile::TempDir, crate::OfficialV41Catalog, std::path::PathBuf) {
        use crate::plan::testing::{exl3, exl3_compact, glm5_config, glm5_tensors, write_snapshot};
        use serde_json::json;
        let dir = tempfile::tempdir().unwrap();
        let mut config = glm5_config();
        config["quantization_config"] = exl3_compact(4);
        let mut tensors = glm5_tensors(|name, n, k| exl3(name, n, k, 4));
        for expert in 1..256 {
            for (projection, n, k) in [("gate_proj", 2048, 6144), ("up_proj", 2048, 6144),
                ("down_proj", 6144, 2048)] {
                tensors.extend(exl3(&format!("model.layers.1.mlp.experts.{expert}.{projection}"), n, k, 4));
            }
        }
        let snapshot = dir.path().join("snapshot");
        write_snapshot(&snapshot, &config, &tensors, None);
        let catalog = crate::read_expert_catalog(&snapshot).unwrap();
        let manifest = dir.path().join("PROGRAMS.json");
        std::fs::write(&manifest, b"{\"programs\":[]}").unwrap();
        for capacity in [1, 16, 80] {
            let package = dir.path().join(format!("exl3/exl3-glm-k45/rtx-tp2/m{capacity}"));
            std::fs::create_dir_all(&package).unwrap();
            let value = json!({"hidden": 6144, "intermediate": 1024, "experts": 256,
                "top_k": 8, "capacity": capacity, "bits": [4, 5], "output_dtype": "fp32",
                "input_format": "e4m3_k32", "direct": false,
                "route_preparation": {"manifest": "routes/v41_exl3_routes.json"},
                "trellis_lut": {"file": "lut.bin", "bytes": 32}, "buffers": {
                    "scratch": {"allocation": "scratch", "bytes": capacity * 64,
                        "dtype": "f32", "zero_on_create": false},
                    "state": {"allocation": "state", "bytes": 0,
                        "dtype": "i32", "zero_on_create": true}}});
            std::fs::write(package.join("v41_exl3.json"), serde_json::to_vec(&value).unwrap()).unwrap();
            // Header-only fixtures do not attest or execute these stand-in assets.
            std::fs::create_dir_all(package.join("routes")).unwrap();
            for file in ["libcuteafd_exl3.so", "lut.bin", "routes/v41_exl3_routes.json", "routes/libv41_exl3_routes.so"] {
                std::fs::write(package.join(file), b"").unwrap();
            }
        }
        (dir, catalog, manifest)
    }

    #[test]
    fn activation_policy_uses_exports_and_refuses_malformed_packages() {
        let (dir, catalog, manifest) = local_fixture();
        for onboard in [Onboard::Auto, Onboard::ExpertsFirst { pool_floor: 65536 }, Onboard::Layers(1)] {
            assert!(select_local_inventory(&catalog, &manifest, 2, 2, 64, 3, false, onboard, false)
                .unwrap().is_some());
        }
        assert!(select_local_inventory(&catalog, &manifest, 2, 2, 64, 3, false, Onboard::Layers(0), false)
            .unwrap().is_none());
        assert!(select_local_inventory(&catalog, &manifest, 2, 2, 64, 3, false, Onboard::Layers(1), true)
            .unwrap().is_none());
        assert!(select_local_inventory(&catalog, &manifest, 2, 1, 64, 3, false, Onboard::Auto, false)
            .unwrap().is_none());
        assert!(select_local_inventory(&catalog, &manifest, 2, 1, 64, 3, false, Onboard::Layers(1), false).is_err());
        let package = dir.path().join("exl3/exl3-glm-k45/rtx-tp2");
        std::fs::remove_file(package.join("m16/libcuteafd_exl3.so")).unwrap();
        assert!(select_local_inventory(&catalog, &manifest, 2, 2, 64, 3, false, Onboard::Auto, false).is_err(),
            "present but incomplete exports must fail closed");
        std::fs::remove_dir_all(package).unwrap();
        assert!(select_local_inventory(&catalog, &manifest, 2, 2, 64, 3, false, Onboard::Auto, false)
            .unwrap().is_none());
        assert!(select_local_inventory(&catalog, &manifest, 2, 2, 64, 3, false, Onboard::Layers(1), false).is_err());
    }

    #[test]
    fn activated_exl3_inventory_planner_equals_runtime() {
        let (_dir, catalog, manifest) = local_fixture();
        let cfg = GlmDsaConfig::from_hf(&crate::plan::testing::glm5_config()).unwrap();
        for onboard in [Onboard::Auto, Onboard::Layers(1)] {
            let local = select_local_inventory(&catalog, &manifest, 2, 2, 64, 3, false, onboard, false)
                .unwrap().unwrap();
            let planned = GlmInputs { prefill_rows: 64, experts: local.experts.clone(),
                expert_workspace: 0, tp2_workspace: local.workspace(), drafter_bytes: 0,
                requested_pool: Some(65536), onboard, ..inputs(&cfg, &[96 * GIB; 2]) };
            let mut runtime = planned.clone();
            for (total, baseline) in &mut runtime.gpus {
                let Baseline::Planned { context_bytes, loaded_bytes } = *baseline else { unreachable!() };
                *baseline = Baseline::Measured { free_bytes: *total - context_bytes - loaded_bytes };
            }
            let serial = local.workspace_for(&cfg, 64, 1, false);
            let (p, pw) = solve_working_set(&planned, serial, HEADS).unwrap();
            let (r, rw) = solve_working_set(&runtime, serial, HEADS).unwrap();
            assert_eq!((p.clone(), pw.clone()), (r, rw));
            assert_eq!(p.tp2.unwrap().first, 1);
            assert_eq!(pw.spark.rtx_layers, vec![1]);
            assert_eq!(pw.tp2_workspace, serial);
            assert!(pw.spark.spark_layers.is_empty());
        }
    }

    #[test]
    fn local_inventory_charges_all_retained_capacities_and_exact_rank_storage() {
        let (_dir, catalog, manifest) = local_fixture();
        let local = local_inventory(&catalog, &manifest, 2, 64, 3, false).unwrap();
        assert_eq!(local.backend_workspace, 80 * 64 + 3 * (32 + 16) + 64 * 6144 * 4);
        assert_eq!(local.experts.len(), 1);
        let exl3 = catalog.exl3().unwrap();
        let layer = crate::V41Exl3Layer::Backbone(1);
        assert_eq!(local.experts[0].whole.resident, exl3.residency(layer, 1, 0).unwrap()
            .device_arena_layout().unwrap().1 as u64);
        for rank in 0..2 {
            let router = if rank == 1 { 256 * 6144 * 2 + 256 * 4 } else { 0 };
            assert_eq!(local.experts[0].half[rank].resident, exl3.residency(layer, 2, rank).unwrap()
                .device_arena_layout().unwrap().1 as u64 + router);
            assert_eq!(local.experts[0].half[rank].staging, 0);
        }
        let wide = local_inventory(&catalog, &manifest, 2, 64, 3, true).unwrap();
        let widening = (DECODE_ROWS + 3 * 64) * 6144 * 2 + 4 * 3 * 64 * 6144 * 2;
        for rank in 0..2 {
            assert_eq!(wide.workspace()[rank] - local.workspace()[rank], widening);
        }
        assert!(local_inventory(&catalog, &manifest, 0, 64, 3, false).is_err());
        assert!(local_inventory(&catalog, &manifest, 3, 64, 3, false).is_err());
    }

    #[test]
    fn local_inventory_keeps_decode_capacity_with_short_prefill_rows() {
        let (_dir, catalog, manifest) = local_fixture();
        for rows in [1, 16, 63] {
            for lanes in [1, 3, 4] {
                for f32 in [false, true] {
                    let local = local_inventory(&catalog, &manifest, 2, rows, lanes, f32).unwrap();
                    assert_eq!(local.backend_workspace, 80 * 64 + 3 * (32 + 16) + DECODE_ROWS * 6144 * 4);
                    let dtype = if f32 { 4 } else { 2 };
                    let payloads = (DECODE_ROWS * 6144 * dtype).max(FLOOR)
                        + lanes * (rows * 6144 * dtype).max(FLOOR);
                    let route_slots = 2 * lanes * (DECODE_ROWS * 8 * 8)
                        + ((2 * lanes + 1) * 16).max(FLOOR) + FLOOR;
                    let widened_exchange = if f32 { 4 * lanes * DECODE_ROWS * 6144 * 2 } else { 0 };
                    assert_eq!(local.extra_workspace[0], payloads + route_slots + widened_exchange);
                    let peer_routes = [256 * 4, 8 * 4, 8 * 4, 6144 + 6144 / 32].into_iter().map(|width|
                        (DECODE_ROWS * width).max(FLOOR) - FLOOR
                            + lanes * ((rows * width).max(FLOOR) - FLOOR)).sum::<u64>();
                    assert_eq!(local.extra_workspace[1] - local.extra_workspace[0], peer_routes);
                }
            }
        }
    }

    #[test]
    fn local_inventory_refuses_missing_assets_and_mismatched_specializations() {
        use serde_json::json;
        let (dir, catalog, manifest) = local_fixture();
        let package = dir.path().join("exl3/exl3-glm-k45/rtx-tp2/m16");
        let path = package.join("v41_exl3.json");
        let original: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        for (field, value) in [("hidden", json!(5120)), ("intermediate", json!(2048)),
            ("experts", json!(128)), ("top_k", json!(6)), ("capacity", json!(80)),
            ("output_dtype", json!("bf16")), ("bits", json!([4, null, 5]))] {
            let mut changed = original.clone();
            changed[field] = value;
            std::fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
            let error = local_inventory(&catalog, &manifest, 2, 64, 3, false).unwrap_err();
            assert!(error.to_string().contains("capacity/geometry/tiers/output mismatch"), "{field}: {error:#}");
        }
        std::fs::write(&path, serde_json::to_vec(&original).unwrap()).unwrap();
        for asset in ["libcuteafd_exl3.so", "lut.bin", "routes/v41_exl3_routes.json", "routes/libv41_exl3_routes.so"] {
            std::fs::remove_file(package.join(asset)).unwrap();
            let error = local_inventory(&catalog, &manifest, 2, 64, 3, false).unwrap_err();
            assert!(error.to_string().contains("missing"), "{asset}: {error:#}");
            std::fs::write(package.join(asset), b"").unwrap();
        }
        std::fs::remove_dir_all(dir.path().join("exl3")).unwrap();
        let error = local_inventory(&catalog, &manifest, 2, 64, 3, false).unwrap_err();
        assert!(error.to_string().contains("missing exl3-glm-k45/rtx-tp2 package"), "{error:#}");
    }

    #[test]
    fn local_inventory_resolves_declared_route_assets() {
        use serde_json::json;
        let (dir, catalog, manifest) = local_fixture();
        let package = dir.path().join("exl3/exl3-glm-k45/rtx-tp2/m16");
        let path = package.join("v41_exl3.json");
        let mut value: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        std::fs::rename(package.join("routes"), package.join("packed")).unwrap();
        value["route_preparation"]["manifest"] = json!("packed/v41_exl3_routes.json");
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(local_inventory(&catalog, &manifest, 2, 64, 3, false).is_ok());

        value["route_preparation"] = serde_json::Value::Null;
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        let error = local_inventory(&catalog, &manifest, 2, 64, 3, false).unwrap_err();
        assert!(error.to_string().contains("missing packed route export manifest"), "{error:#}");

        value["direct"] = json!(true);
        std::fs::remove_dir_all(package.join("packed")).unwrap();
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(local_inventory(&catalog, &manifest, 2, 64, 3, false).is_ok());
    }

    fn nvfp4_checkpoint_tensors() -> Vec<crate::plan::testing::Tensor> {
        use crate::plan::testing::{glm5_tensors, nvfp4};
        let mut tensors = glm5_tensors(nvfp4);
        for expert in 1..256 {
            for (projection, n, k) in [("gate_proj", 2048, 6144), ("up_proj", 2048, 6144),
                ("down_proj", 6144, 2048)] {
                tensors.extend(nvfp4(&format!("model.layers.1.mlp.experts.{expert}.{projection}"), n, k));
            }
        }
        tensors
    }

    fn nvfp4_fixture() -> (tempfile::TempDir, crate::OfficialV41Catalog, std::path::PathBuf) {
        use crate::plan::testing::{glm5_config, write_snapshot};
        let dir = tempfile::tempdir().unwrap();
        let mut config = glm5_config();
        config["quantization_config"] = serde_json::json!({"quant_method":"modelopt", "quant_algo":"NVFP4",
            "config_groups":{"group_0":{"weights":{"num_bits":4,"type":"float","group_size":16}}}});
        let tensors = nvfp4_checkpoint_tensors();
        let snapshot = dir.path().join("snapshot");
        write_snapshot(&snapshot, &config, &tensors, None);
        let catalog = crate::read_expert_catalog(&snapshot).unwrap();
        let share = dir.path().join("share");
        std::fs::create_dir_all(&share).unwrap();
        let manifest = share.join("PROGRAMS.json");
        std::fs::write(&manifest, b"{\"programs\":[]}").unwrap();
        for (geometry, offset) in [("glm_nvfp4", 1000), ("glm_nvfp4a4", 2000)] {
            let name = geometry.replace('_', "-");
            let package = dir.path().join(format!("lib/fp8/fp8-{name}"));
            std::fs::create_dir_all(package.join("tp2")).unwrap();
            std::fs::write(package.join("manifest.json"), serde_json::json!({
                "schema":"cuteafd.fp8moe-package.v1", "role":"coordinator", "geometry":geometry,
                "layouts":{"tp2":{"tp":2, "hidden":6144, "intermediate":2048, "slice":1024,
                    "experts":256, "top_k":8, "input":"bf16", "weights":"nvfp4", "swiglu_limit":0.0,
                    "prefill_forms":{"w8a16":[], "w8a8":[]},
                    "capacities":([1,16,80,256,1024,4096].map(|capacity|
                        serde_json::json!({"capacity":capacity,"scratch_bytes":offset + capacity * 64})))}}
            }).to_string()).unwrap();
            // Header fixtures do not execute or attest their stand-in native binary.
            std::fs::write(package.join("tp2/libcuteafd_fp8moe.so"), b"").unwrap();
        }
        (dir, catalog, manifest)
    }

    fn extend_routed_fixture(catalog: &crate::OfficialV41Catalog, mut tensors: Vec<crate::plan::testing::Tensor>)
        -> crate::OfficialV41Catalog {
        let mut config = crate::plan::checkpoint::read_json(&catalog.snapshot().join("config.json")).unwrap();
        config["num_hidden_layers"] = serde_json::json!(4);
        config["indexer_types"] = serde_json::json!(["full", "shared", "shared", "shared"]);
        config["mlp_layer_types"] = serde_json::json!(["dense", "sparse", "sparse", "sparse"]);
        let routed: Vec<_> = tensors.iter().filter(|(name, ..)| name.starts_with("model.layers.1."))
            .cloned().collect();
        for layer in 2..4 {
            tensors.extend(routed.iter().cloned().map(|(name, dtype, shape)|
                (name.replacen("model.layers.1.", &format!("model.layers.{layer}."), 1), dtype, shape)));
        }
        crate::plan::testing::write_snapshot(catalog.snapshot(), &config, &tensors, None);
        crate::read_expert_catalog(catalog.snapshot()).unwrap()
    }

    #[test]
    fn spark_nvfp4_residency_charges_selected_suffix_and_replicated_scalars() {
        use crate::formats::fp8_experts::Slicing;
        let (_dir, catalog, _manifest) = nvfp4_fixture();
        let catalog = extend_routed_fixture(&catalog, nvfp4_checkpoint_tensors());
        let scalars = 3 * 256 * 8;
        let bytes = |width| 3 * 256 * 6144 * width * (8 + 1) / 16 + scalars;
        for ranks in [2, 3, 4, 6] {
            for slicing in [Slicing::Blocks(128), Slicing::Padded] {
                let whole_suffix = spark_resident_bytes(&catalog, &[1, 2, 3], ranks, slicing).unwrap();
                let selected = spark_resident_bytes(&catalog, &[2, 3], ranks, slicing).unwrap();
                for rank in 0..ranks {
                    let width = catalog.fp8().unwrap().rank_width(ranks, rank, slicing).unwrap() as u64;
                    assert_eq!(selected[rank], 2 * bytes(width));
                    assert_eq!(whole_suffix[rank], 3 * bytes(width));
                }
            }
        }
        let exact = spark_resident_bytes(&catalog, &[3], 6, Slicing::Blocks(128)).unwrap();
        let padded = spark_resident_bytes(&catalog, &[3], 6, Slicing::Padded).unwrap();
        assert_eq!(exact, [bytes(384), bytes(384), bytes(384), bytes(384), bytes(256), bytes(256)]);
        assert_eq!(padded, vec![bytes(384); 6]);
        // Per-expert alpha/input scales replicate; they must not shrink with the slice.
        assert_ne!(exact[0] * 2, exact[4] * 3);
        assert_eq!(exact.iter().sum::<u64>(), bytes(2048) + 5 * scalars);
        assert!(spark_resident_bytes(&catalog, &[], 0, Slicing::Padded).unwrap().is_empty());
        for (layers, ranks) in [(vec![], 4), (vec![1], 4), (vec![3, 2], 4),
            (vec![1, 3], 4), (vec![2, 3], 0), (vec![2, 3], 5), (vec![4], 4)] {
            assert!(spark_resident_bytes(&catalog, &layers, ranks, Slicing::Padded).is_err());
        }
    }

    #[test]
    fn spark_exl3_residency_uses_selected_layer_arenas_not_checkpoint_shares() {
        use crate::formats::fp8_experts::Slicing;
        use crate::plan::testing::{exl3, glm5_tensors};
        let (_dir, catalog, _manifest) = local_fixture();
        let mut tensors = glm5_tensors(|name, n, k| exl3(name, n, k, 4));
        for expert in 1..256 {
            for (projection, n, k) in [("gate_proj", 2048, 6144), ("up_proj", 2048, 6144),
                ("down_proj", 6144, 2048)] {
                tensors.extend(exl3(&format!("model.layers.1.mlp.experts.{expert}.{projection}"), n, k, 4));
            }
        }
        let catalog = extend_routed_fixture(&catalog, tensors);
        for ranks in [2, 3, 4, 6] {
            let selected = spark_resident_bytes(&catalog, &[2, 3], ranks, Slicing::Blocks(128)).unwrap();
            for (rank, &bytes) in selected.iter().enumerate() {
                let expected: u64 = [2, 3].into_iter().map(|layer|
                    catalog.exl3().unwrap().residency(crate::V41Exl3Layer::Backbone(layer), ranks, rank)
                        .unwrap().device_arena_layout().unwrap().1 as u64).sum();
                assert_eq!(bytes, expected);
                assert_eq!(bytes % (2 << 20), 0);
            }
            assert_eq!(spark_resident_bytes(&catalog, &[3], ranks, Slicing::Blocks(128)).unwrap(),
                selected.iter().map(|bytes| bytes / 2).collect::<Vec<_>>());
        }
        assert!(spark_resident_bytes(&catalog, &[2, 3], 6, Slicing::Padded).is_err());
    }

    #[test]
    fn nvfp4_inventory_retains_native_scales_and_selects_matching_activation_package() {
        let (dir, catalog, manifest) = nvfp4_fixture();
        let whole = 3 * (256 * 6144 * 2048 / 2 + 256 * 6144 * 2048 / 16 + 256 * 8);
        let half = 3 * (256 * 6144 * 1024 / 2 + 256 * 6144 * 1024 / 16 + 256 * 8);
        for rows in [1u64, 16, 63, 64, 81, 4096] {
            assert!(local_inventory_selected(&catalog, &manifest, 2, rows, 3, false, false).is_err());
            for a4 in [true] {
                let local = local_inventory_selected(&catalog, &manifest, 2, rows, 3, false, a4).unwrap();
                assert_eq!(local.format, GlmLocalFormat::Nvfp4 { w4a4: a4 });
                assert_eq!(local.experts.len(), 1);
                assert_eq!(local.experts[0].whole, Bytes2 { resident: whole, staging: 0 });
                assert_eq!(local.experts[0].half[0], Bytes2 { resident: half, staging: 0 });
                assert_eq!(local.experts[0].half[1], Bytes2 {
                    resident: half + 256 * 6144 * 2 + 256 * 4, staging: 0 });
                assert!(local.experts[0].tp2);
                let max_rows = rows.max(DECODE_ROWS);
                let capacity = [1,16,80,256,1024,4096].into_iter().find(|&n| n >= max_rows).unwrap();
                let scratch = if a4 { 2000 } else { 1000 };
                assert_eq!(local.backend_workspace, scratch + capacity * 64 + max_rows * 6144 * 2);
                assert_eq!(local.extra_workspace, local_extra_workspace(6144, 256, 8, rows, 3, false));
                assert_eq!(local.workspace_for(&GlmDsaConfig::from_hf(&crate::plan::testing::glm5_config()).unwrap(), rows, 1, false),
                    local_extra_workspace(6144, 256, 8, rows, 1, false).map(|b| b + local.backend_workspace));
            }
        }
        std::fs::remove_dir_all(dir.path().join("lib/fp8/fp8-glm-nvfp4a4/tp2")).unwrap();
        assert!(local_inventory_selected(&catalog, &manifest, 2, 64, 1, false, true).is_err());
    }

    #[test]
    fn nvfp4_inventory_validates_every_selected_expert_header() {
        use crate::plan::testing::{t, write_snapshot};
        let (_dir, catalog, manifest) = nvfp4_fixture();
        let config = crate::plan::checkpoint::read_json(&catalog.snapshot().join("config.json")).unwrap();
        let snapshot = catalog.snapshot().to_path_buf();
        let input = "model.layers.1.mlp.experts.255.up_proj.input_scale";
        let down = "model.layers.1.mlp.experts.255.down_proj.weight_scale";
        for (name, replacement) in [(input, None), (input, Some(t(input, "BF16", &[]))),
            (down, Some(t(down, "F8_E4M3", &[6144, 1])))] {
            let mut tensors = nvfp4_checkpoint_tensors();
            let index = tensors.iter().position(|(n, ..)| n == name).unwrap();
            if let Some(replacement) = replacement { tensors[index] = replacement; }
            else { tensors.remove(index); }
            write_snapshot(&snapshot, &config, &tensors, None);
            // The first gate still establishes NVFP4; the later expert must not escape admission validation.
            let error = crate::read_expert_catalog(&snapshot)
                .and_then(|catalog| local_inventory_selected(&catalog, &manifest, 2, 64, 3, false, true))
                .unwrap_err();
            assert!(format!("{error:#}").contains(name), "{name}: {error:#}");
        }
    }

    #[test]
    fn native_nvfp4_inventory_planner_equals_runtime_selected_working_set() {
        let (_dir, catalog, manifest) = nvfp4_fixture();
        let cfg = GlmDsaConfig::from_hf(&crate::plan::testing::glm5_config()).unwrap();
        for a4 in [true] {
            for rows in [1, 63, 4096] {
                let local = local_inventory_selected(&catalog, &manifest, 2, rows, 3, false, a4).unwrap();
                let serial = local.workspace_for(&cfg, rows, 1, false);
                for onboard in [Onboard::Auto, Onboard::Layers(0), Onboard::Layers(1), Onboard::Fraction(1.0)] {
                    let planned = GlmInputs { prefill_rows: rows, experts: local.experts.clone(),
                        expert_workspace: 0, tp2_workspace: local.workspace(),
                        drafter_bytes: 0, requested_pool: Some(65536), onboard, ..inputs(&cfg, &[96 * GIB; 2]) };
                    let mut runtime = planned.clone();
                    for (total, baseline) in &mut runtime.gpus {
                        let Baseline::Planned { context_bytes, loaded_bytes } = *baseline else { unreachable!() };
                        *baseline = Baseline::Measured { free_bytes: *total - context_bytes - loaded_bytes };
                    }
                    let (p, pw) = solve_working_set(&planned, serial, HEADS).unwrap();
                    let (r, rw) = solve_working_set(&runtime, serial, HEADS).unwrap();
                    assert_eq!((p.clone(), pw.clone()), (r, rw));
                    assert_eq!(p.pool_tokens, 65536);
                    assert!(p.expert_ranges.iter().all(|range| range.layers == 0));
                    if onboard == Onboard::Layers(0) {
                        assert!(pw.spark.rtx_layers.is_empty());
                        assert_eq!(pw.spark.spark_layers, vec![1]);
                        assert_eq!(pw.tp2_workspace, [0; 2]);
                        assert_eq!((pw.spark.spark_ranks, pw.spark.prefill_lanes), (4, 3));
                    } else {
                        assert_eq!(pw.spark.rtx_layers, vec![1]);
                        assert!(pw.spark.spark_layers.is_empty());
                        assert_eq!(pw.tp2_workspace, serial);
                        assert_eq!((pw.spark.spark_ranks, pw.spark.prefill_lanes), (0, 1));
                        assert_eq!(p.tp2.unwrap().layers, 1);
                    }
                }
            }
        }
    }

    #[test]
    fn nvfp4_inventory_refuses_unimplemented_exchange_and_bad_package_contracts() {
        use serde_json::json;
        let (dir, catalog, manifest) = nvfp4_fixture();
        let error = local_inventory_selected(&catalog, &manifest, 2, 64, 3, true, true).unwrap_err();
        assert!(error.to_string().contains("use BF16 exchange"), "{error:#}");
        let package = dir.path().join("lib/fp8/fp8-glm-nvfp4a4");
        let path = package.join("manifest.json");
        let original: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        for (field, value) in [("tp", json!(1)), ("hidden", json!(5120)), ("slice", json!(2048)),
            ("top_k", json!(6)), ("input", json!("wire")), ("weights", json!("mxfp4")),
            ("swiglu_limit", json!(7.0))] {
            let mut changed = original.clone();
            changed["layouts"]["tp2"][field] = value;
            std::fs::write(&path, changed.to_string()).unwrap();
            let error = local_inventory_selected(&catalog, &manifest, 2, 64, 3, false, true).unwrap_err();
            assert!(error.to_string().contains("geometry/input/format mismatch"), "{field}: {error:#}");
        }
        for (field, value) in [
            ("capacities", json!([{"capacity":80,"scratch_bytes":null}])),
            ("capacities", json!([{"capacity":80,"scratch_bytes":1},{"capacity":16,"scratch_bytes":2}])),
            ("capacities", json!([{"capacity":16,"scratch_bytes":1}])),
            ("prefill_forms", json!({"w8a16":[80],"w8a8":[]})),
        ] {
            let mut changed = original.clone();
            changed["layouts"]["tp2"][field] = value;
            std::fs::write(&path, changed.to_string()).unwrap();
            assert!(local_inventory_selected(&catalog, &manifest, 2, 64, 3, false, true).is_err(), "{field}");
        }
        std::fs::write(&path, original.to_string()).unwrap();
        std::fs::remove_file(package.join("tp2/libcuteafd_fp8moe.so")).unwrap();
        let error = local_inventory_selected(&catalog, &manifest, 2, 64, 3, false, true).unwrap_err();
        assert!(error.to_string().contains("missing binary"), "{error:#}");
        std::fs::remove_dir_all(dir.path().join("lib/fp8")).unwrap();
        let error = local_inventory_selected(&catalog, &manifest, 2, 64, 3, false, true).unwrap_err();
        assert!(error.to_string().contains("missing calibrated GLM NVFP4 W4A4 tp2 package"), "{error:#}");
    }

    #[test]
    fn missing_local_package_fails_closed() {
        let cfg = config();
        let mut i = inputs(&cfg, &[96 * GIB; 2]);
        i.experts.clear();
        assert_eq!(solve(&request(&i).unwrap()).unwrap().onboard_layers, 0);
        i.onboard = Onboard::Layers(1);
        assert!(matches!(solve(&request(&i).unwrap()), Err(PlacementError::ExpertLayers { .. })));
        i.onboard = Onboard::Auto;
        i.spark_ranks = 0;
        assert!(matches!(solve(&request(&i).unwrap()), Err(PlacementError::SparkFree { .. })));
    }

    #[test]
    fn context_costs_and_shared_indexer_groups_are_ready_for_k3() {
        let cfg = config();
        let r = request(&inputs(&cfg, &[96 * GIB; 2])).unwrap();
        assert_eq!(r.layers.iter().filter(|l| l.context_indexer).count(), 21);
        for (layer, demand) in r.layers.iter().enumerate() {
            let halves = demand.kv_unit.unit_bytes_context.unwrap();
            assert_eq!(halves[0] + halves[1], demand.kv_unit.unit_bytes_whole);
            assert_eq!(demand.colocate, Some(cfg.index_source(layer) as u16));
        }
        assert_eq!(r.context_buffers.lanes, 1, "prefill gathers do not use the context exchange");
        assert_eq!(r.context_buffers.decode_rows, DECODE_ROWS);
        assert_eq!(r.hops.lanes, 3, "prefill lane ownership is unchanged");
        // A build without the context programs keeps heads, and refuses context.
        assert_eq!(r.executor.attention_default(), AttentionPlacement::Heads);
        assert!(r.layers.iter().all(|l| l.modes == [LayerMode::HeadSplit]));
        let mut context = r;
        context.attention_placement = Some(AttentionPlacement::Context);
        assert!(matches!(solve(&context), Err(PlacementError::UnsupportedMode { .. })));
    }

    #[test]
    fn whole_attention_colocates_indexers_and_charges_one_logical_kv_copy() {
        let cfg = config();
        let owners = attention_owners(&cfg, cfg.layers).unwrap();
        assert!(owners.contains(&0) && owners.contains(&1));
        let mut i = inputs(&cfg, &[96 * GIB; 2]);
        i.prefill_lanes = 2;
        i.spark_ranks = 6;
        i.requested_pool = Some(2 << 20);
        let (placed, _) = solve_working_set(&i, [0; 2], Some(AttentionPlacement::Layers)).unwrap();
        assert_eq!(placed.pool_tokens, 2 << 20);
        let tp2: Vec<_> = placed.layers.iter().map(|layer| layer.experts == ExpertHome::RtxTp2).collect();
        let startup = crate::serving_capacity::glm_graphs::glm_owned_startup_graphs(
            i.max_context as usize, (i.max_context as usize).div_ceil(PAGE_ROWS as usize), &owners, &tp2).unwrap();
        for rank in 0..2 {
            assert!(!placed.items[rank].iter().any(|item| item.group == "decode graph allowance"));
            for item in startup.inventory.items(rank) {
                assert!(placed.items[rank].contains(&item), "missing {item:?}");
            }
        }
        for (layer, assignment) in placed.layers.iter().enumerate() {
            assert_eq!(assignment.mode, LayerMode::Whole { gpu: owners[layer], ffn: FfnMode::Split });
            assert_eq!(owners[layer], owners[cfg.index_source(layer)]);
        }
        let mut req = request(&i).unwrap();
        configure_attention(&mut req, Some(AttentionPlacement::Layers)).unwrap();
        assert!(!req.fixed.iter().any(|d| d.group == "padded decode scratch page"));
        assert_eq!(req.layers.iter().map(|layer| layer.fixed_bytes.whole).sum::<u64>(),
            glm_cache_geometry(&cfg, cfg.layers, 1).unwrap().ranks[0].persistent_unit_bytes - cfg.layers as u64 * FLOOR);
        assert_eq!(placed.hops.iter().filter(|hop| matches!(hop.at, HopPoint::AfterAttention(_))).count(), cfg.layers);
        i.prefill_lanes = 3;
        assert!(solve_working_set(&i, [0; 2], Some(AttentionPlacement::Layers)).is_err());
        let mut one = inputs(&cfg, &[96 * GIB]);
        one.prefill_lanes = 1;
        assert!(solve_working_set(&one, [0; 2], Some(AttentionPlacement::Layers)).is_err());
    }

    #[test]
    fn owned_graph_census_tracks_the_selected_tp2_prefix() {
        let cfg = config();
        for onboard in [Onboard::Auto, Onboard::Layers(0), Onboard::Layers(2), Onboard::Layers(3)] {
            let mut i = inputs(&cfg, &[96 * GIB; 2]);
            i.spark_ranks = 6;
            i.prefill_lanes = 2;
            i.requested_pool = Some(2 << 20);
            i.onboard = onboard;
            let (placed, _) = solve_working_set(&i, [0; 2], Some(AttentionPlacement::Layers)).unwrap();
            assert_eq!(placed.pool_tokens, 2 << 20);
            let owners: Vec<_> = placed.layers.iter().map(|layer| match layer.mode {
                LayerMode::Whole { gpu, .. } => gpu, _ => unreachable!(),
            }).collect();
            let tp2: Vec<_> = placed.layers.iter().map(|layer| layer.experts == ExpertHome::RtxTp2).collect();
            assert_eq!(tp2.iter().filter(|&&local| local).count(), placed.onboard_layers);
            if onboard == Onboard::Auto { assert!(placed.onboard_layers > 0); }
            let startup = crate::serving_capacity::glm_graphs::glm_owned_startup_graphs(
                i.max_context as usize, (placed.pool_tokens / PAGE_ROWS) as usize, &owners, &tp2).unwrap();
            let expected = (3 * cfg.layers - placed.onboard_layers) as u64 * startup.keys.len() as u64;
            for rank in 0..2 {
                assert_eq!(startup.inventory.ranks[rank].executables, expected);
                for item in startup.inventory.items(rank) {
                    assert!(placed.items[rank].contains(&item), "{onboard:?}: missing {item:?}");
                }
            }
        }
    }

    #[test]
    fn pure_owners_equal_the_admitted_layers_for_each_extent() {
        let cfg = config();
        for layers in [1, 2, 3, 6, 7, cfg.layers] {
            let mut i = inputs(&cfg, &[96 * GIB; 2]);
            i.layers = layers;
            i.skip_routed_experts = true;
            i.spark_ranks = 0;
            i.experts.clear();
            i.onboard = Onboard::Layers(0);
            i.prefill_lanes = 1;
            i.requested_pool = Some(32768);
            let placed = solve_working_set(&i, [0; 2], Some(AttentionPlacement::Layers)).unwrap().0;
            let expected = attention_owners(&cfg, layers).unwrap();
            assert_eq!(placed.layers.iter().map(|l| match l.mode {
                LayerMode::Whole { gpu, .. } => gpu, _ => unreachable!(),
            }).collect::<Vec<_>>(), expected);
        }
        assert!(attention_owners(&cfg, cfg.layers + 1).is_err());
        assert!(attention_owners(&cfg, 0).is_err());
    }

    #[test]
    fn owned_graphs_use_the_runtime_clamped_context_at_small_pools() {
        let cfg = config();
        for (requested_pool, expected_pool, expected_context) in [
            (Some(32768), 32768, 32512), (None, 115072, 114816),
        ] {
            let mut i = inputs(&cfg, &[32 * GIB; 2]);
            i.prefill_lanes = 2;
            i.onboard = if requested_pool.is_some() { Onboard::Layers(0) } else { Onboard::Auto };
            i.requested_pool = requested_pool;
            let (placed, _) = solve_working_set(&i, [0; 2], Some(AttentionPlacement::Layers)).unwrap();
            assert_eq!(placed.pool_tokens, expected_pool);
            let context = crate::serving_capacity::pool_context(i.max_context as usize,
                i.automatic_context, placed.pool_tokens as usize, 256).unwrap();
            assert_eq!(context, expected_context);
            let owners: Vec<_> = placed.layers.iter().map(|layer| match layer.mode {
                LayerMode::Whole { gpu, .. } => gpu, _ => unreachable!(),
            }).collect();
            let tp2: Vec<_> = placed.layers.iter().map(|layer| layer.experts == ExpertHome::RtxTp2).collect();
            let startup = crate::serving_capacity::glm_graphs::glm_owned_startup_graphs(
                context, (placed.pool_tokens as usize).div_ceil(PAGE_ROWS as usize), &owners, &tp2).unwrap();
            for rank in 0..2 {
                for item in startup.inventory.items(rank) {
                    assert!(placed.items[rank].contains(&item), "missing {item:?}");
                }
            }
            i.automatic_context = false;
            assert!(solve_working_set(&i, [0; 2], Some(AttentionPlacement::Layers)).is_err());
        }
    }

    const CONTEXT_SCRATCH: GlmContext = GlmContext { topk_scratch: 3 << 20, partial_scratch: 9 << 20 };

    #[test]
    fn context_is_the_default_where_its_programs_exist_and_charges_what_the_engine_allocates() {
        let cfg = config();
        let mut i = inputs(&cfg, &[96 * GIB; 2]);
        i.context = Some(CONTEXT_SCRATCH);
        i.requested_pool = Some(1 << 20);
        let r = request(&i).unwrap();
        assert_eq!(r.executor.attention_default(), AttentionPlacement::Context);
        let (auto, _) = solve_working_set(&i, [0; 2], None).unwrap();
        assert_eq!(auto.attention_placement, AttentionPlacement::Context);
        assert!(auto.layers.iter().all(|l| l.mode == LayerMode::ContextSplit));
        let (heads, _) = solve_working_set(&i, [0; 2], Some(AttentionPlacement::Heads)).unwrap();
        assert!(heads.layers.iter().all(|l| l.mode == LayerMode::HeadSplit));
        // Per GPU: half the records, plus staging, exchange and workspace.
        let buffers = context_buffers(&cfg, 1 << 20, 4096, 3, CONTEXT_SCRATCH);
        assert_eq!(buffers, r.context_buffers);
        let item = |p: &Placement, gpu: usize, group: &str| p.items[gpu].iter().filter(|i| i.group == group)
            .map(|i| i.bytes).sum::<u64>();
        let demands = buffers.demands().unwrap();
        for gpu in 0..2 {
            for demand in demands.iter().filter(|d| usize::from(d.gpu) == gpu) {
                assert_eq!(item(&auto, gpu, &demand.group), demand.bytes, "{}", demand.group);
                assert_eq!(item(&heads, gpu, &demand.group), 0);
            }
        }
        // 2 x 1M / 64 pages x (41,984 + 8,448) B staging; 64 rows x (2 x 36,864 + 32,896 + 2 x 16,384) exchange.
        assert_eq!(demands[0].bytes, 2 * 16_384 * (41_984 + 8_448));
        assert_eq!(demands[1].bytes, 2 * 64 * (2 * 36_864 + 32_896 + 2 * 16_384) + 2 * 3 * 16);
        let workspace = context_workspace(&cfg, 1 << 20, 4096, 3, CONTEXT_SCRATCH);
        assert_eq!(workspace.len(), 6 + 3, "one commit list per prefill lane");
        assert_eq!(demands[2].bytes, workspace.iter().map(|&(_, b)| b.max(FLOOR)).sum::<u64>());
        let unit = |p: &Placement, gpu: usize| item(p, gpu, "records") / (p.pool_tokens / PAGE_ROWS);
        assert_eq!(unit(&auto, 0) + unit(&auto, 1) - 2 * (DECODE_ROWS + 3) * 4, unit(&heads, 0) - (DECODE_ROWS + 3) * 4);
        // Commit lists: a 4,096-row lane spans 65 pages, 33 of either parity.
        assert_eq!(context_commit_pages(4096), 33);
        assert_eq!(context_commit_pages(64), 1);
        assert_eq!(context_commit_pages(65), 2);
    }

    #[test]
    fn context_admits_two_million_tokens_at_max_where_heads_cannot() {
        let cfg = config();
        let mut i = inputs(&cfg, &[96 * GIB; 2]);
        i.max_context = 1 << 20;
        i.context = Some(CONTEXT_SCRATCH);
        i.requested_pool = Some(2 << 20);
        let (placed, _) = solve_working_set(&i, [0; 2], None).unwrap();
        assert_eq!((placed.pool_tokens, placed.attention_placement), (2 << 20, AttentionPlacement::Context));
        assert!(solve_working_set(&i, [0; 2], Some(AttentionPlacement::Heads)).is_err());
        // Auto without a request grows past 2M; heads stops near 1.3M.
        i.requested_pool = None;
        let (context, _) = solve_working_set(&i, [0; 2], None).unwrap();
        let (heads, _) = solve_working_set(&i, [0; 2], Some(AttentionPlacement::Heads)).unwrap();
        assert!(context.pool_tokens >= 2 << 20 && heads.pool_tokens < 3 << 19, "{} {}", context.pool_tokens, heads.pool_tokens);
        eprintln!("GLM max fixture: context {} heads {}", context.pool_tokens, heads.pool_tokens);
        // One GPU never runs context, even where the build could.
        let mut one = inputs(&cfg, &[96 * GIB]);
        one.context = Some(CONTEXT_SCRATCH);
        one.requested_pool = Some(65536);
        assert_eq!(solve_working_set(&one, [0; 2], None).unwrap().0.attention_placement, AttentionPlacement::Heads);
        assert!(solve_working_set(&one, [0; 2], Some(AttentionPlacement::Context)).is_err());
    }

    #[test]
    fn drafter_packing_peak_is_checked_but_not_retained_beside_pool() {
        let cfg = config();
        let mut i = inputs(&cfg, &[96 * GIB; 2]);
        i.drafter_bytes = 3 * GIB;
        let steady = solve_working_set(&i, [0; 2], Some(AttentionPlacement::Heads)).unwrap().0;
        i.drafter_staging = 432 << 20;
        let request = request(&i).unwrap();
        assert_eq!(request.startup.len(), 1);
        assert_eq!(request.startup[0].bytes, 3 * GIB + (432 << 20));
        assert!(!request.fixed.iter().any(|d| d.category == Category::Drafter));
        assert_eq!(request.movables.len(), 1);
        assert_eq!(request.movables[0].allowed, [0]);
        assert_eq!(request.movables[0].conditional.iter().flat_map(|home| &home.demands)
            .filter(|d| d.category == Category::Drafter).map(|d| d.bytes).sum::<u64>(), 3 * GIB);
        let with_staging = solve_working_set(&i, [0; 2], Some(AttentionPlacement::Heads)).unwrap().0;
        assert_eq!(with_staging.pool_tokens, steady.pool_tokens);
        assert_eq!(draft::home(&with_staging), Some(0));
        assert_eq!(with_staging.items[0].iter().filter(|item| item.category == Category::Drafter)
            .map(|item| item.bytes).sum::<u64>(), 3 * GIB);
        assert!(!with_staging.items.iter().flatten().any(|item| item.group == "drafter packing peak"));
        i.drafter_staging = 96 * GIB;
        assert!(matches!(solve_working_set(&i, [0; 2], Some(AttentionPlacement::Heads)),
            Err(PlacementError::StartupPeak { gpu: 0, .. })));
    }

    #[test]
    fn context_programs_need_every_primitive() {
        let names = ["glm_scored_index_topk_decode_m64", "glm_dsa_candidate_merge", "glm_lse_combine2",
            "glm_paged_staging_gather_kv", "glm_paged_staging_gather_index", "glm_sparse_mla_partial_h0_s2_m64",
            "glm_sparse_mla_partial_h32_s2_m64", "glm_sparse_mla_partial_h32_s4_m64"];
        let scratch = |name: &str| names.contains(&name).then(|| if name.ends_with("s4_m64") { 900 }
            else if name.contains("topk") { 70 } else { 100 });
        assert_eq!(context_programs(names, scratch), Some(GlmContext { topk_scratch: 70, partial_scratch: 900 }));
        for missing in names[..5].iter().chain(&names[5..6]) {
            let kept: Vec<_> = names.iter().copied().filter(|n| n != missing).collect();
            assert_eq!(context_programs(kept.iter().copied(), |n| kept.contains(&n).then_some(1)), None, "{missing}");
        }
    }

    #[test]
    fn context_manifest_scratch_matches_runtime_partial_recipe_for_every_plan() {
        use cuteafd_core::coordinator_programs::{attention_partial_program, AttentionReductionPlan};
        let mut programs: Vec<_> = CONTEXT_PROGRAMS.iter().map(|name|
            serde_json::json!({"name": name, "scratch_bytes_at_capacity": {"scratch": 70}})).collect();
        for begin in [0, 32] {
            for (splits, caps170, caps188, bytes) in [(2, vec![64], vec![64], 100),
                (4, vec![1, 8], vec![1], 900), (8, vec![], vec![8], 1800)] {
                programs.push(serde_json::json!({
                    "name": format!("glm_sparse_mla_partial_h{begin}_s{splits}_m64"),
                    "params": {"device_sm_counts": [170, 188],
                        "row_caps": {"170": caps170, "188": caps188}},
                    "scratch_bytes_at_capacity": {"scratch": bytes}}));
            }
        }
        let manifest = serde_json::json!({"programs": programs});
        for sms in [170, 188] {
            for rows in 1..=64 {
                let plan = AttentionReductionPlan::new(64, None, Some(rows)).unwrap();
                let selected = manifest_context(&manifest, sms, &plan).unwrap();
                let runtime_scratch = [0, 32].into_iter().map(|begin| {
                    let name = attention_partial_program(&manifest, "glm", begin, 64, sms, &plan).unwrap();
                    manifest["programs"].as_array().unwrap().iter().find(|p| p["name"] == name).unwrap()
                        ["scratch_bytes_at_capacity"]["scratch"].as_u64().unwrap()
                }).max().unwrap();
                assert_eq!(selected, GlmContext { topk_scratch: 70, partial_scratch: runtime_scratch });
                assert_eq!(selected.partial_scratch, if rows == 1 || (rows <= 8 && sms == 170) { 900 }
                    else if rows <= 8 { 1800 } else { 100 });
            }
        }
    }

    #[test]
    fn scratch_selection_includes_optional_programs_and_refuses_missing_required() {
        let cfg = config();
        let lookup = |name: &str| Some(if name.contains("bf16") { 500 } else if name.contains("topk") { 80 } else { 100 });
        assert_eq!(step_scratch(&cfg, true, false, lookup).unwrap(), GlmScratch { programs: 500, topk: 80 });
        for split in [false, true] {
            for decode in [false, true] {
                let lookup = |name: &str| Some(if name.contains("_pt_") { 900 } else { 100 });
                assert_eq!(step_scratch(&cfg, split, decode, lookup).unwrap().programs, 900);
            }
        }
        assert!(step_scratch(&cfg, false, true, |_| None).is_err());
        assert_eq!(default_onboard(), Onboard::Auto);
    }
}
