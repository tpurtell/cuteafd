//! GLM Flash admission before weights: planner and runtime differ only in baseline.
use crate::families::glm5_flash::{GlmNextAttention, GlmNextConfig, resident::{GlmfRepresentation, GlmfResidentRank}};
use crate::serving_capacity::{self as cache, glmf_graphs};
use crate::placement::*;
use cuteafd_core::memory_layout::{Basis, Category};

pub mod admission;

/// Two-GPU attention placement of GLM Flash's MLA layers and the `context` programs' scratch.
/// `context` (PLAN.md "v3 attention placement without latent replication", GLM Flash DSA):
/// projections head-split, every 256-token allocation unit of the MLA records and pool keys on
/// one GPU (logical unit `u` on GPU `u % 2`), decode/verify exchanging q / candidates / partials
/// and merging by LSE, prefill gathering the history into a staging view. Explicit `layers`
/// gives complete MLA attention and caches to alternating owners, keeping KDA and FFNs split.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GlmfAttention {
    pub mode: AttentionPlacement,
    pub first_owner: u8,
    pub topk_scratch: u64,
    pub partial_scratch: u64,
}

/// Programs `context` needs beside the split partials (any `_s<N>` variant of both head halves).
pub const CONTEXT_PROGRAMS: [&str; 6] = ["glmf_scored_index_topk_decode_m64", "glmf_dsa_candidate_merge",
    "glmf_lse_combine2", "glmf_paged_staging_gather_kv", "glmf_paged_staging_gather_index",
    "glmf_index_expand_owned"];

/// Every attention mode uses the shared executor's admitted wave policy.
pub fn admitted_prefill_waves(_mode: AttentionPlacement, requested: u64) -> Result<u64, PlacementError> {
    validate_layer_pipeline_lanes(requested, requested as usize)
        .map_err(|_| PlacementError::Inventory("GLM Flash executor needs shared-policy prefill units"))?;
    Ok(requested)
}

/// The `context` programs (beside the split partials), for the fail-closed
/// reason when a build's manifest lacks them (issue #202).
pub fn context_program_list() -> String {
    let mut names: Vec<String> = CONTEXT_PROGRAMS.iter().map(|n| (*n).to_string()).collect();
    names.extend(["glmf2_kda_commit_c", "glmf_index_producer_c_m64", "glmf_index_producer_c_m4096",
        "glmf_sparse_mla_partial_h0_s*_m64", "glmf_sparse_mla_partial_h32_s*_m64"].map(str::to_string));
    names.join(", ")
}

/// The `context` programs of a build and their scratch (`lookup`: name -> scratch bytes), or None
/// when it lacks any of them (a chosen `context` then fails closed).
/// Needs the head split's compact commit too.
pub fn context_programs<'n>(names: impl IntoIterator<Item = &'n str> + Clone,
    lookup: impl Fn(&str) -> Option<u64>) -> Option<GlmfAttention> {
    let partial = |begin: u64| names.clone().into_iter()
        .filter(|n| n.starts_with(&format!("glmf_sparse_mla_partial_h{begin}_s")) && n.ends_with("_m64"))
        .filter_map(&lookup).max();
    CONTEXT_PROGRAMS.iter().chain(&["glmf2_kda_commit_c", "glmf_index_producer_c_m64", "glmf_index_producer_c_m4096"])
        .all(|n| lookup(n).is_some()).then_some(())?;
    Some(GlmfAttention { mode: AttentionPlacement::Context, first_owner: 0, topk_scratch: lookup(CONTEXT_PROGRAMS[0])?,
        partial_scratch: partial(0)?.max(partial(32)?) })
}

/// The `context` programs a manifest carries (PROGRAMS.json), as [`context_programs`].
pub fn manifest_context(manifest: &serde_json::Value) -> Option<GlmfAttention> {
    let programs = manifest["programs"].as_array()?;
    let names = programs.iter().filter_map(|p| p["name"].as_str());
    context_programs(names, |name| programs.iter().find(|p| p["name"].as_str() == Some(name))
        .map(|p| p["scratch_bytes_at_capacity"]["scratch"].as_u64().unwrap_or(0)))
}

/// Context admission selects the same partial program as runtime, not the largest exported variant.
pub fn manifest_context_for_plan(manifest: &serde_json::Value, heads: usize, sms: usize,
    plan: &cuteafd_core::coordinator_programs::AttentionReductionPlan) -> anyhow::Result<Option<GlmfAttention>> {
    let Some(mut context) = manifest_context(manifest) else { return Ok(None) };
    let mut scratch = 0;
    for begin in [0, heads / 2] {
        let Ok(name) = cuteafd_core::coordinator_programs::attention_partial_program(
            manifest, "glmf", begin, 64, sms, plan) else { return Ok(None) };
        let bytes = crate::serving_capacity::glmf_manifest_scratch(manifest)(&name)
            .ok_or_else(|| anyhow::anyhow!("selected Flash context partial missing: {name}"))?;
        scratch = scratch.max(bytes);
    }
    context.partial_scratch = scratch;
    Ok(Some(context))
}

/// Resolve the same per-card numerical recipes for serving and plan admission.
pub fn manifest_context_for_cards(manifest: &serde_json::Value, heads: usize, sms: &[usize],
    plan: cuteafd_core::coordinator_programs::AttentionReductionPlan) -> anyhow::Result<Option<GlmfAttention>> {
    let mut context: Option<GlmfAttention> = None;
    for &sm_count in sms {
        let mut selected = manifest.clone();
        cuteafd_core::coordinator_programs::select_card_plans(&mut selected, sm_count.try_into()?)?;
        cuteafd_core::coordinator_programs::select_attention_plans(&mut selected, plan,
            |name| (name.starts_with("glmf_") && !name.ends_with("_m128")) || name.starts_with("glmf2_"))?;
        let Some(card) = manifest_context_for_plan(&selected, heads, sm_count, &plan)? else { return Ok(None) };
        context = Some(match context {
            Some(previous) => GlmfAttention { topk_scratch: previous.topk_scratch.max(card.topk_scratch),
                partial_scratch: previous.partial_scratch.max(card.partial_scratch), ..card },
            None => card,
        });
    }
    Ok(context)
}

/// The MLA placement a GLM Flash engine on `gpus` GPUs serves: `requested` (None: auto), checked
/// against the build's `context` programs. Auto is `context` on two GPUs, as `ExecutorModes::check_attention`
/// records for this family; without context programs auto uses whole MLA owners (`layers`).
/// Explicit `context` still fails closed naming the programs. One GPU and explicit `heads` stay heads.
pub fn resolve_attention(requested: Option<AttentionPlacement>, gpus: usize, programs: Option<GlmfAttention>)
    -> Result<GlmfAttention, PlacementError> {
    let heads = GlmfAttention { mode: AttentionPlacement::Heads, ..programs.unwrap_or_default() };
    let refuse = |mode, reason| Err(PlacementError::AttentionPlacement { family: "glm5_flash", mode, reason });
    let context = |p: GlmfAttention| GlmfAttention { mode: AttentionPlacement::Context, ..p };
    let missing = || Err(PlacementError::MissingContextPrograms { family: "glm5_flash",
        programs: context_program_list() });
    match requested {
        Some(AttentionPlacement::Heads) => Ok(heads),
        Some(mode) if gpus != 2 => refuse(mode, "requires two coordinator GPUs with peer access"),
        Some(AttentionPlacement::Layers) => Ok(GlmfAttention { mode: AttentionPlacement::Layers, ..heads }),
        Some(AttentionPlacement::Context) => programs.map(context).map_or_else(missing, Ok),
        None if gpus == 2 => Ok(programs.map(context).unwrap_or_else(|| {
            tracing::warn!(family = "glm5_flash", programs = %context_program_list(), fallback = "layers",
                "auto context programs missing; falling back to whole-layer attention without KV replication");
            GlmfAttention { mode: AttentionPlacement::Layers, ..heads }
        })),
        None => Ok(heads),
    }
}

/// Units one GPU commits per prefill lane of `rows` rows: a range spans one more unit than it
/// fills, and every second unit is this GPU's.
pub fn context_commit_units(rows: u64) -> u64 {
    (rows.div_ceil(256) + 1).div_ceil(2)
}

/// Every allocation the `context` executor adds on one GPU beside the exchange and the staging,
/// in order; `glm5_flash::engine::context` allocates exactly this list (each with its 256-byte
/// floor) and the solver charges its sum. Decode reuses the step's own tables: its page table
/// holds the compact unit table, its pool table the compact pool table, its cache lengths the
/// shard's pools; `local` holds the candidate merge's local pool slots and counts.
pub fn context_workspace(cfg: &GlmNextConfig, extent: u64, rows: u64, lanes: u64,
    context: GlmfAttention) -> Vec<(&'static str, u64)> {
    let decode = cache::GLMF_DECODE_ROWS;
    let k = (cfg.index_topk / cfg.index_kpool) as u64;
    let partial = decode * cfg.heads as u64 / 2 * (512 * 2 + 4);
    let units = extent.div_ceil(256);
    // Per owner: a record-page table over every logical page, then its local units.
    let gather = (2 * units * 4 + 2 * units.div_ceil(2)) * 4;
    let mut out = vec![("candidates", decode * k * 8), ("local pools", decode * (k + 1) * 4),
        ("own-head partial", partial), ("peer-head partial", partial),
        ("scored top-k scratch", context.topk_scratch), ("partial scratch", context.partial_scratch),
        ("gather tables", gather)];
    out.extend((0..lanes).map(|_| ("commit blocks", 3 * 3 * context_commit_units(rows) * 8)));
    // The staging view's identity pool-page map (the prefill index expansion reads it).
    out.push(("staging pool map", extent.div_ceil(256) * 4));
    out
}

pub fn context_buffers(cfg: &GlmNextConfig, extent: u64, rows: u64, lanes: u64,
    context: GlmfAttention) -> ContextBuffers {
    ContextBuffers { staging_unit_bytes: 256 * 528 + 64 * 132, staging_unit_rows: 256,
        query_row_bytes: cfg.heads as u64 / 2 * 512 * 2,
        partial_row_bytes: cfg.heads as u64 / 2 * (512 * 2 + 4),
        candidate_row_bytes: (cfg.index_topk / cfg.index_kpool) as u64 * 8,
        compiled_extent: extent, decode_rows: cache::GLMF_DECODE_ROWS, lanes: 1,
        workspace_bytes: context_workspace(cfg, extent, rows, lanes, context).iter()
            .map(|&(_, bytes)| bytes.max(256)).sum() }
}

#[derive(Debug, Clone)]
pub struct GlmfInputs<'a> {
    pub resident_weight_limit: Option<ResidentWeights>,
    pub spark_reply_format: cuteafd_core::SparkReplyFormat,
    /// True only when the operator left context automatic.
    pub automatic_context: bool,
    pub attention: GlmfAttention,
    pub cfg: &'a GlmNextConfig,
    pub layers: usize,
    /// Sample after programs, before any weights; planned baseline charges context/code only.
    pub gpus: Vec<(u64, Baseline)>,
    /// Code not present at the baseline sample: LoadedCode minus already-sampled code, per rank.
    pub pending_code: Vec<u64>,
    /// Measured native EXL3 TP2 program/package path, not the legacy Spark graph path.
    pub local_tp2_exl3: bool,
    pub headroom_bytes: u64,
    pub spark_ranks: usize,
    pub prefill_lanes: u64,
    pub prefill_rows: u64,
    pub decode_rows: u64,
    pub partial_bytes: u64,
    pub max_context: u64,
    pub sequences: u64,
    /// Served graph policy, independent of whether DFlash weights are resident.
    pub speculation: bool,
    pub state_slots: u64,
    pub mark_slots: u64,
    pub pool_marks: bool,
    pub index: cache::GlmfIndexCache,
    pub kda_state_bytes: u64,
    pub shared_replay: bool,
    pub representation: GlmfRepresentation,
    pub resident: Vec<GlmfResidentRank>,
    /// Exact component-owned step rows with Spark lanes and without Sparks.
    pub workspace: Vec<GlmfBufferDemand>,
    pub local_workspace: Vec<GlmfBufferDemand>,
    pub graphs: GraphSet,
    /// Routed layers in backbone order, not including dense layers.
    pub experts: Vec<ExpertCost>,
    pub expert_workspace: u64,
    pub tp2_workspace: [u64; 2],
    /// SM-independent draft weights and owned activations; embedding stays host mapped.
    pub drafter_bytes: u64,
    /// Retained checkpoint weights only, excluding the owned activation/cache payload.
    pub drafter_weights: ResidentWeights,
    /// Linear and attention scratch on each candidate GPU, sized by physical SM count.
    pub drafter_scratch: Vec<u64>,
    /// Present only when the GPU1 DFlash2 executor is requested. dSpark stays local.
    pub drafter_bridge: Option<MovableDemands>,
    pub requested_pool: Option<u64>,
    pub onboard: Onboard,
    pub full_prefill_logits: u64,
}

pub fn default_onboard() -> Onboard { Onboard::Auto }

/// Duplicate context-scaling MLA records on the second GPU, excluding the
/// small replicated compact tails, KDA state and pool metadata.
pub fn replicated_mla_bytes(cfg: &GlmNextConfig, index: cache::GlmfIndexCache,
    placed: &Placement) -> Result<u64, PlacementError> {
    if placed.items.len() != 2 { return Ok(0); }
    if placed.layers.len() > cfg.attention.len() {
        return Err(PlacementError::Inventory("GLM Flash waste layer inventory"));
    }
    let replicas = placed.layers.iter().zip(&cfg.attention).filter(|(layer, kind)|
        **kind == GlmNextAttention::Mla && layer.mode == LayerMode::HeadSplit).count() as u64;
    let unit = 256 * (528 + if index == cache::GlmfIndexCache::Keys { 512 } else { 0 }) + 64 * 132;
    replicas.checked_mul(unit).and_then(|bytes| bytes.checked_mul(placed.pool_tokens.div_ceil(256)))
        .ok_or(PlacementError::Overflow("GLM Flash replicated MLA records"))
}

/// Shared checkpoint-derived inventory; linear/attention scratch follows the chosen GPU.
pub fn drafter_inventory(config: &serde_json::Value, slots: usize, sequences: usize,
    sms: &[u64], representation: crate::families::glm5::draft_representation::GlmDraftRepresentation,
    mode: u8, kv_cache: crate::families::qwen4::Qwen4KvCache) -> Result<(MovablePart, Vec<u64>), crate::families::glm5::draft_representation::GlmDraftStorageError> {
    use crate::families::glm5::draft_representation::{draft_resident_bytes_with_mode, GlmDraftStorageError};
    if ![1, 2].contains(&sms.len()) {
        return Err(GlmDraftStorageError::Unsupported("GLM Flash draft candidate inventory needs one or two GPUs"));
    }
    let mut owned = None;
    let mut scratch = Vec::with_capacity(sms.len());
    for &sms in sms {
        let residency = draft_resident_bytes_with_mode(config, slots, sequences, sms, representation, mode, kv_cache)?;
        let part = MovablePart::new(Bytes2 { resident: residency.owned, staging: 0 }, residency.weights);
        if owned.is_some_and(|previous| previous != part) {
            return Err(GlmDraftStorageError::Unsupported("GLM Flash owned draft bytes depend on SM count"));
        }
        residency.owned.checked_add(residency.scratch).ok_or(GlmDraftStorageError::Overflow)?;
        owned = Some(part);
        scratch.push(residency.scratch);
    }
    Ok((owned.unwrap(), scratch))
}

/// GPU1 DFlash bridge inventory, conditional on installing its executor.
/// `rows` is the drafter's full admitted batch (sequences * block), not a
/// target-head launch chunk. GPU1's existing draft destinations are already
/// in the movable; publisher streams/modules belong to the sampled baseline.
pub fn remote_dflash_bridge_demands(hidden: u64, vocab: u64, taps: u64, rows: u64)
    -> Result<MovableDemands, PlacementError> {
    draft::remote_dflash_bridge_demands(hidden, vocab, taps, rows, 0)
}

/// Opt-in until the remote executor's hardware gates qualify its default.
pub fn remote_dflash_requested() -> bool {
    std::env::var("CUTEAFD_GLMF_REMOTE_DFLASH").as_deref() == Ok("1")
}

/// Derive bridge capacity from the same checkpoint and batch used by runtime.
/// Opaque fixture byte totals and dSpark do not imply a remote executor.
pub fn drafter_bridge_inventory(config: &serde_json::Value, slots: usize, sequences: usize)
    -> anyhow::Result<Option<MovableDemands>> {
    if config["speculators_model_type"] == "dspark" { return Ok(None); }
    use crate::families::glm5::draft_representation::{draft_geometry, GlmDraftCapacity};
    let (g, block) = draft_geometry(config)?;
    let capacity = GlmDraftCapacity::new(slots, sequences, usize::try_from(block)?)?;
    Ok(Some(remote_dflash_bridge_demands(g.hidden, g.vocab, g.taps, capacity.block_rows as u64)?))
}

pub use crate::placement::{ComponentDemand as GlmfBufferDemand, SparkWorkingSet as GlmfWorkingSet,
    WorkingSetComponent as GlmfComponent};

/// Shared attention/head temporaries remain allocated when every expert is remote.
/// Both callers use the exact base plus optional RTX-buffer delta.
pub fn step_workspace_demands(cfg: &GlmNextConfig, lanes: usize, prefill_rows: u64, decode_rows: u64,
    shape: cache::GlmfStepShape, decode: cache::GlmfScratch, prefill: cache::GlmfScratch) -> Vec<GlmfBufferDemand> {
    (0..if shape.split { 2 } else { 1 }).flat_map(|rank| {
        let full = if rank == 0 { shape } else {
            cache::GlmfStepShape { lead: false, local_experts: false, spark: false, ..shape }
        };
        let base = cache::GlmfStepShape { local_experts: false, tp2_experts: false, ..full };
        let bytes = |shape| cache::glmf_step_workspaces(cfg, lanes, prefill_rows, decode_rows,
            &shape, decode, prefill).device_bytes();
        let always = bytes(base);
        [GlmfBufferDemand::new(GlmfComponent::Always, rank as u8, Category::Workspace, "steps", always, Basis::Formula),
            GlmfBufferDemand::new(GlmfComponent::RtxExperts, rank as u8, Category::Workspace, "steps",
                bytes(full) - always, Basis::Formula)]
    }).collect()
}

/// Checkpoint and executor facts, before selection of an expert working set.
pub struct GlmfFacts {
    pub geometry: cache::FamilyCacheGeometry,
    pub layers: Vec<LayerDemand>,
    /// Component-tagged owners before working-set filtering.
    pub fixed: Vec<GlmfBufferDemand>,
    pub pool_overhead: Vec<u64>,
    pub pool_bounded: Vec<BoundedDemand>,
    pub routed: usize,
    pub expert_gpus: usize,
}

pub fn facts(i: &GlmfInputs<'_>) -> Result<GlmfFacts, PlacementError> {
    let ranks = i.gpus.len();
    if !(1..=2).contains(&ranks) || i.layers == 0 || i.layers > i.cfg.layers
        || i.cfg.dense.len() < i.layers || i.cfg.attention.len() < i.layers
        || i.resident.len() != ranks || i.graphs.ranks.len() != ranks
        || i.pending_code.len() != ranks || i.drafter_scratch.len() != ranks
        || ![2, 4].contains(&i.partial_bytes) || i.attention.first_owner as usize >= ranks {
        return Err(PlacementError::Inventory("GLM Flash rank/layer inventory"));
    }
    let prefill_lanes = admitted_prefill_waves(i.attention.mode, i.prefill_lanes)?;
    let geometry = cache::glm_flash_placed_cache_geometry_rows(i.cfg, i.layers, ranks, i.index,
        i.kda_state_bytes, i.decode_rows, i.attention.mode, i.attention.first_owner as usize)
        .map_err(|_| PlacementError::Inventory("GLM Flash cache geometry"))?;
    let routed = i.cfg.dense[..i.layers].iter().filter(|&&d| !d).count();
    if i.experts.len() != routed {
        return Err(PlacementError::Inventory("GLM Flash routed expert inventory"));
    }
    let mul = |a: u64, b: u64| a.checked_mul(b).ok_or(PlacementError::Overflow("GLM Flash fixed demand"));
    let mut fixed = Vec::new();
    let mut expert = i.experts.iter();
    let mut layers = Vec::new();
    for layer in 0..i.layers {
        let mla = i.cfg.attention[layer] == GlmNextAttention::Mla;
        let unit = if mla { 256 * (528 + if i.index == cache::GlmfIndexCache::Keys { 512 } else { 0 }) + 64 * 132 } else { 0 };
        layers.push(LayerDemand { kind: if mla { AttentionClass::Dsa } else { AttentionClass::Kda },
            weights: ModeBytes::default(), kv_unit: KvDemand { unit_bytes_whole: unit, unit_bytes_split: [unit; 2],
                unit_bytes_context: mla.then_some([unit / 2; 2]) },
            fixed_bytes: ModeBytes::default(), context_indexer: mla, colocate: None, extra: vec![],
            experts: if i.cfg.dense[layer] { None } else { expert.next().copied() },
            modes: if ranks == 2 && mla { vec![LayerMode::HeadSplit, LayerMode::ContextSplit,
                LayerMode::Whole { gpu: 0, ffn: FfnMode::Split }, LayerMode::Whole { gpu: 1, ffn: FfnMode::Split }] }
                else if ranks == 2 { vec![LayerMode::HeadSplit] }
                else { vec![LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }] } });
    }
    let mla = i.cfg.attention[..i.layers].iter().filter(|&&a| a == GlmNextAttention::Mla).count() as u64;
    for (rank, g) in geometry.ranks.iter().enumerate() {
        let local_mla = if i.attention.mode == AttentionPlacement::Layers {
            mla / 2 + u64::from(rank == i.attention.first_owner as usize) * (mla % 2)
        } else { mla };
        let tails = if i.index == cache::GlmfIndexCache::Compact { local_mla * cache::GLMF_INDEX_TAIL_BYTES } else { 0 };
        let index_replay = if tails > 0 { local_mla * i.decode_rows * 512 } else { 0 };
        let gpu = rank as u8;
        fixed.push(GlmfBufferDemand::new(GlmfComponent::Always, gpu, Category::Runtime, "pending code", i.pending_code.get(rank).copied().unwrap_or(0), Basis::Calibrated));
        fixed.push(GlmfBufferDemand::new(GlmfComponent::Always, gpu, Category::Embedding, "embedding", i.resident[rank].embedding, Basis::Exact)
            .with_resident_weights(ResidentWeights(i.resident[rank].embedding)));
        fixed.push(GlmfBufferDemand::new(GlmfComponent::Always, gpu, Category::Weights, "target weights", i.resident[rank].weights, Basis::Exact)
            .with_resident_weights(ResidentWeights(i.resident[rank].weights)));
        let replay = if i.shared_replay { 0 } else { g.speculative_replay_bytes - index_replay };
        fixed.push(GlmfBufferDemand::new(GlmfComponent::Always, gpu, Category::Kv, "KDA state and replay",
            mul(g.active_state_per_sequence_bytes - tails, i.state_slots)? + replay, Basis::Formula));
        fixed.push(GlmfBufferDemand::new(GlmfComponent::Always, gpu, Category::Kv, "DSA index tails and replay",
            mul(tails, i.state_slots)? + index_replay, Basis::Formula));
        fixed.push(GlmfBufferDemand::new(GlmfComponent::Always, gpu, Category::Kv, "commit tables", g.fixed_state_bytes, Basis::Formula));
        if !i.pool_marks {
            fixed.push(GlmfBufferDemand::new(GlmfComponent::Always, gpu, Category::Prefix, "marks", mul(g.retained_mark_bytes, i.mark_slots)?, Basis::Formula));
        } else {
            fixed.push(GlmfBufferDemand::new(GlmfComponent::Always, gpu, Category::Prefix, "reserved units",
                (g.persistent_unit_bytes + g.pool_metadata_unit_bytes
                    + g.glmf_pool_mark_backing_unit_bytes(i.pool_marks)) * cache::GLMF_POOL_MARK_RESERVED_UNITS, Basis::Formula));
        }
        fixed.extend(i.graphs.items(rank).into_iter().map(|item| GlmfBufferDemand {
            component: GlmfComponent::Always,
            demand: Demand::new(gpu, item.category, item.group, item.bytes, item.basis)
                .with_lifetime(item.lifetime),
        }));
        if ranks == 2 {
            fixed.push(GlmfBufferDemand::new(GlmfComponent::Always, gpu, Category::Transport, "peer exchange",
                glmf_graphs::peer_exchange_bytes(prefill_lanes, i.prefill_rows.max(i.decode_rows),
                    i.cfg.hidden as u64, i.partial_bytes, i.representation.output_shard), Basis::Formula));
            // new_abortable owns a separate 256-byte word on each exchange end.
            // CUDA publisher streams/function loading are driver costs, not payload storage.
            fixed.push(GlmfBufferDemand::new(GlmfComponent::Always, gpu, Category::Transport,
                "peer abort word", 256, Basis::Formula));
            fixed.push(GlmfBufferDemand::new(GlmfComponent::RtxExperts, gpu, Category::Transport,
                "route abort word", 256, Basis::Formula));
            let slots = 2 * prefill_lanes;
            fixed.push(GlmfBufferDemand::new(GlmfComponent::RtxExperts, gpu, Category::Transport, "route exchange",
                slots * (i.prefill_rows.max(i.decode_rows) * i.cfg.topk as u64 * 8).max(256)
                    + ((slots + 1) * 16).max(256), Basis::Formula));
        }
        if rank == 0 {
            if i.decode_rows > cache::GLMF_DECODE_ROWS {
                fixed.push(GlmfBufferDemand::new(GlmfComponent::Always, gpu, Category::Workspace, "wide decode selector",
                    cache::glmf_selector_bytes(i.decode_rows, i.cfg.vocab_size as u64)
                        - cache::glmf_selector_bytes(cache::GLMF_DECODE_ROWS, i.cfg.vocab_size as u64), Basis::Formula));
            }
            fixed.push(GlmfBufferDemand::new(GlmfComponent::SparkExperts, gpu, Category::Transport, "Spark intake",
                cache::prefill_return_bytes_with_format(prefill_lanes, i.prefill_rows,
                    i.spark_ranks as u64, i.cfg.hidden as u64, i.spark_reply_format)
                    .map_err(|_| PlacementError::Inventory("GLM Flash Spark reply width or allocation overflow"))?, Basis::Formula));
            fixed.push(GlmfBufferDemand::new(GlmfComponent::Always, gpu, Category::Workspace, "probe prefill logits", i.full_prefill_logits, Basis::Formula));
        }
    }

    let pool_overhead: Vec<u64> = geometry.ranks.iter().map(|g| g.pool_metadata_unit_bytes).collect();
    let pool_bounded: Vec<BoundedDemand> = geometry.ranks.iter().enumerate().filter_map(|(rank, g)| {
        let bytes = g.glmf_pool_mark_backing_unit_bytes(i.pool_marks);
        (bytes > 0).then(|| BoundedDemand { gpu: rank as u8, category: Category::Prefix,
            group: format!("KDA-only pool marks ({bytes} bytes/unit)"), bytes_per_unit: bytes,
            offset_units: 0, cap_units: u64::MAX, basis: Basis::Formula })
    }).collect();
    let expert_gpus = usize::from(ranks == 1 || i.experts.iter().all(|cost| cost.tp2));
    Ok(GlmfFacts { geometry, layers, fixed, pool_overhead, pool_bounded, routed, expert_gpus })
}

/// Exact header-only layer residency: the shared RTX onboarding costs.
pub use crate::placement::rtx_onboard::expert_costs;

/// Exact compiled package arenas (`fp8-glmf*` or EXL3); missing packages fail
/// closed, never become zero scratch.
pub fn expert_workspace(catalog: &crate::OfficialV41Catalog, manifest: Option<&std::path::Path>, rows: u64,
    tp: usize) -> anyhow::Result<u64> {
    crate::placement::rtx_onboard::expert_workspace(catalog, manifest, rows, tp, "glmf")
}

/// Family-specific canonical selection and solved-home validation over the shared set.
trait GlmfWorkingSetExt: Sized {
    fn candidate(i: &GlmfInputs<'_>, n: usize) -> Result<Self, PlacementError>;
    fn validate(&self, placed: &Placement) -> Result<(), PlacementError>;
}

impl GlmfWorkingSetExt for GlmfWorkingSet {
    fn candidate(i: &GlmfInputs<'_>, n: usize) -> Result<Self, PlacementError> {
        let dense = i.cfg.dense.get(..i.layers).ok_or(PlacementError::Inventory("GLM Flash layer inventory"))?;
        let routed: Vec<_> = dense.iter().enumerate().filter_map(|(layer, &dense)| (!dense).then_some(layer)).collect();
        if n > routed.len() || (i.spark_ranks == 0 && n != routed.len()) {
            return Err(PlacementError::Inventory("GLM Flash expert working set inventory"));
        }
        let spark_layers = routed[n..].to_vec();
        let spark_ranks = if spark_layers.is_empty() { 0 } else { i.spark_ranks };
        let prefill_lanes = admitted_prefill_waves(i.attention.mode, i.prefill_lanes)?;
        Ok(Self { rtx_layers: routed[..n].to_vec(), spark_layers, spark_ranks, prefill_lanes })
    }

    fn validate(&self, placed: &Placement) -> Result<(), PlacementError> {
        let layers = |spark| placed.layers.iter().enumerate().filter_map(|(layer, home)| {
            let selected = if spark { home.experts == ExpertHome::Spark } else {
                matches!(home.experts, ExpertHome::RtxWhole { .. } | ExpertHome::RtxTp2)
            };
            selected.then_some(layer)
        }).collect::<Vec<_>>();
        if layers(true) != self.spark_layers || layers(false) != self.rtx_layers {
            return Err(PlacementError::Inventory("GLM Flash selected expert set disagrees with solved homes"));
        }
        Ok(())
    }
}

/// The fixed embedding demand is also the executor's immutable table home.
/// Host-mapped tables have no device demand and retain the entry GPU's alias.
pub fn embedding_home(placed: &Placement) -> u8 {
    placed.items.iter().position(|items| items.iter().any(|item|
        item.category == Category::Embedding && item.group == "embedding" && item.bytes > 0))
        .unwrap_or(0) as u8
}

/// The selected first MLA owner is already recorded by the immutable layer plan.
pub fn placed_attention(attention: GlmfAttention, placed: &Placement) -> GlmfAttention {
    if attention.mode != AttentionPlacement::Layers { return attention; }
    let first_owner = placed.layers.iter().find_map(|layer| match layer.mode {
        LayerMode::Whole { gpu, ffn: FfnMode::Split } => Some(gpu),
        _ => None,
    }).unwrap_or(attention.first_owner);
    GlmfAttention { first_owner, ..attention }
}


pub fn startup_graphs(cfg: &GlmNextConfig, layers: usize, context: usize, pool: usize, sequences: usize,
    speculation: bool, gpus: usize, decode_rows: usize, sms: usize) -> GraphSet {
    let counts = glmf_graphs::serving_graph_counts(context, pool, cfg.dense_context(), sequences.min(decode_rows), speculation,
        layers, gpus == 2, &glmf_graphs::DecodeBuckets::new(glmf_graphs::verify_budget(decode_rows, sms)));
    let ranks = counts.iter().enumerate().map(|(rank, &count)| {
        let role = if gpus == 1 { 0 } else { rank + 1 };
        let measured = glmf_graphs::measured_graph_bytes(count, role);
        let margin = (measured * glmf_graphs::GRAPH_MARGIN_PERCENT).div_ceil(100)
            .max(glmf_graphs::GRAPH_RANK_MARGIN_BYTES);
        GraphRank { executables: count, bytes: measured + margin, margin }
    }).collect();
    GraphSet { ranks, shapes: counts[0] / (layers + 1) as u64, lifetime: Lifetime::Startup }
}

/// Estimated maximum-pool bound: only power-of-two stride/width transitions
/// and the dense/long transition can introduce a new graph geometry. This
/// avoids walking every pool unit or materializing executable shape keys.
#[allow(clippy::too_many_arguments)]
pub fn startup_graph_bound(cfg: &GlmNextConfig, layers: usize, context: usize, pool: usize,
    sequences: usize, speculation: bool, gpus: usize, decode_rows: usize, sms: usize) -> GraphSet {
    let pages = pool.div_ceil(glmf_graphs::PAGE_ROWS);
    let pools = pages / glmf_graphs::UNIT_PAGES;
    let limit = context.div_ceil(glmf_graphs::UNIT_ROWS).min(pools);
    let mut boundaries = vec![1, limit];
    let mut power = 1usize;
    while power <= limit {
        boundaries.push(power);
        boundaries.push(power.saturating_add(1).min(limit));
        let Some(next) = power.checked_mul(2) else { break };
        power = next;
    }
    boundaries.push(cfg.dense_context().div_ceil(glmf_graphs::UNIT_ROWS).min(limit));
    boundaries.push(cfg.dense_context().saturating_add(1).div_ceil(glmf_graphs::UNIT_ROWS).min(limit));
    boundaries.sort_unstable();
    boundaries.dedup();
    let (table_pages, table_pools) = cache::glmf_table_pages(context as u64);
    let table = (table_pages as usize, table_pools as usize);
    let mut geometries = Vec::new();
    for units in boundaries.into_iter().filter(|&units| units > 0 && units <= limit) {
        let capacity = (units * glmf_graphs::UNIT_ROWS).min(context);
        let mut width = 1;
        while width / 2 * glmf_graphs::UNIT_ROWS < capacity {
            let low = if width == 1 { 1 } else { width / 2 * glmf_graphs::UNIT_ROWS + 1 };
            let high = (width * glmf_graphs::UNIT_ROWS).min(capacity);
            for long in [false, true] {
                if (!long && low <= high.min(cfg.dense_context()))
                    || (long && low.max(cfg.dense_context() + 1) <= high) {
                    let live = units.min(high.max(131_072).div_ceil(glmf_graphs::UNIT_ROWS).next_power_of_two());
                    let (page_stride, pool_stride) = glmf_graphs::decode_strides(
                        live * glmf_graphs::UNIT_PAGES, live, (pages, pools), table);
                    let key = glmf_graphs::GraphGeometry::keyed(width.min(pool_stride), page_stride, pool_stride, long);
                    if !geometries.contains(&key) { geometries.push(key); }
                }
            }
            width *= 2;
        }
    }
    let buckets = glmf_graphs::DecodeBuckets::new(glmf_graphs::verify_budget(decode_rows, sms));
    let plain = buckets.bucket(sequences.min(decode_rows).clamp(16, 64), false);
    let rows = buckets.plain.iter().filter(|&&rows| rows <= plain).count()
        + if speculation { buckets.spec.len() } else { 0 };
    let shapes = (geometries.len() * rows) as u64;
    let ranks = (0..gpus).map(|rank| {
        let executables = shapes * (layers + usize::from(rank == 0)) as u64;
        let measured = glmf_graphs::measured_graph_bytes(executables, if gpus == 1 { 0 } else { rank + 1 });
        let margin = (measured * glmf_graphs::GRAPH_MARGIN_PERCENT).div_ceil(100)
            .max(glmf_graphs::GRAPH_RANK_MARGIN_BYTES);
        GraphRank { executables, bytes: measured + margin, margin }
    }).collect();
    GraphSet { ranks, shapes, lifetime: Lifetime::Startup }
}

/// P6 staged capture after every expert capacity was primed, not legacy Spark-route graphs.
/// Preserve the separate admission-growth margin; only the ready executable charge changes.
pub fn calibrate_local_tp2_graphs(graphs: &mut GraphSet) {
    if graphs.ranks.len() != 2 || graphs.lifetime != Lifetime::Startup { return; }
    for (rank, graph) in graphs.ranks.iter_mut().enumerate() {
        let (bytes, count) = [(2_074_083_328u64, 12_880u64), (2_004_877_312, 12_600)][rank];
        let measured = (graph.executables as u128 * bytes as u128).div_ceil(count as u128) as u64;
        graph.margin = (measured * glmf_graphs::GRAPH_MARGIN_PERCENT).div_ceil(100)
            .max(glmf_graphs::GRAPH_RANK_MARGIN_BYTES);
        graph.bytes = measured + graph.margin;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mismatched_card_recipes_resolve_auto_to_layers_but_keep_heads_explicit() {
        let plan = cuteafd_core::coordinator_programs::AttentionReductionPlan::new(64, None, None).unwrap();
        let mut programs: Vec<_> = CONTEXT_PROGRAMS.iter().chain(&[
            "glmf2_kda_commit_c", "glmf_index_producer_c_m64", "glmf_index_producer_c_m4096"])
            .map(|name| serde_json::json!({"name": name})).collect();
        for begin in [0, 32] {
            for splits in [2, 4] {
                programs.push(serde_json::json!({"name": format!("glmf_sparse_mla_partial_h{begin}_s{splits}_m64"),
                    "params": {"device_sm_counts": [170], "row_caps": {"170": [64]}},
                    "scratch_bytes_at_capacity": {"scratch": splits * 100}}));
            }
        }
        let manifest = serde_json::json!({"programs": programs});
        assert!(manifest_context(&manifest).is_some());
        assert!(manifest_context_for_cards(&manifest, 64, &[170], plan).unwrap().is_some());
        let context = manifest_context_for_cards(&manifest, 64, &[170, 188], plan).unwrap();
        assert!(context.is_none());
        assert_eq!(resolve_attention(None, 2, context).unwrap().mode, AttentionPlacement::Layers);
        assert!(matches!(resolve_attention(Some(AttentionPlacement::Context), 2, context),
            Err(PlacementError::MissingContextPrograms { .. })));
        assert_eq!(resolve_attention(Some(AttentionPlacement::Heads), 2, context).unwrap().mode, AttentionPlacement::Heads);
    }

    fn workspace_fixture(ranks: usize, bytes: u64) -> Vec<GlmfBufferDemand> {
        (0..ranks).map(|rank| GlmfBufferDemand::new(GlmfComponent::Always,
            rank as u8, Category::Workspace, "steps", bytes, Basis::Formula)).collect()
    }

    fn inputs(cfg: &GlmNextConfig, ranks: usize, capacity: u64) -> GlmfInputs<'_> {
        let context = ArchContext::coordinator(capacity, None).context_bytes;
        GlmfInputs { resident_weight_limit: None, spark_reply_format: cuteafd_core::SparkReplyFormat::Bf16, automatic_context: false,
            attention: Default::default(), cfg, layers: cfg.layers,
            gpus: (0..ranks).map(|rank| (capacity, Baseline::Planned {
                context_bytes: context + loaded_code("glmf", "exl3", ranks == 2, rank as u8).unwrap().bytes,
                loaded_bytes: 0 })).collect(), pending_code: vec![0; ranks], local_tp2_exl3: false, headroom_bytes: 1 << 30,
            spark_ranks: 4, prefill_lanes: 2, prefill_rows: 4096, decode_rows: 64, partial_bytes: 2,
            max_context: 262_144, sequences: 8, speculation: true, state_slots: 8, mark_slots: 16, pool_marks: false,
            index: cache::GlmfIndexCache::Keys, kda_state_bytes: 4, shared_replay: false,
            representation: GlmfRepresentation::default(),
            resident: (0..ranks).map(|rank| GlmfResidentRank {
                embedding: if rank == 0 { 1 << 30 } else { 0 }, weights: (5 << 30) + if rank == 1 { 64 << 20 } else { 0 } }).collect(), workspace: workspace_fixture(ranks, 3 << 30), local_workspace: workspace_fixture(ranks, 2 << 30),
            graphs: startup_graphs(cfg, cfg.layers, 262_144, if capacity <= 34 << 30 { 1 << 20 } else { 2 << 20 },
                8, true, ranks, 64, 188),
            experts: cfg.dense.iter().filter(|&&dense| !dense).map(|_| ExpertCost {
                whole: Bytes2 { resident: 2 << 30, staging: 0 },
                half: [Bytes2 { resident: 1 << 30, staging: 0 }; 2], tp2: ranks == 2, whole_supported: true, spark_ok: true }).collect(),
            expert_workspace: 128 << 20, tp2_workspace: [128 << 20; 2], drafter_bytes: 3 << 30, drafter_weights: ResidentWeights(2 << 30),
            drafter_scratch: vec![0; ranks], drafter_bridge: None, requested_pool: None, onboard: default_onboard(), full_prefill_logits: 0,
        }
    }

    fn contract(input: &GlmfInputs<'_>) -> Result<admission::AdmittedGlmf, planner::Refusal> {
        admission::prepare(input, 188, true, admission::ProcessOwners::new((512 << 30, 512 << 30), 8 << 30))?.admit()
    }

    fn admitted_case(input: &GlmfInputs<'_>, automatic_context: bool, sms: usize,
        rtx_experts: bool, alternate: Option<&[GlmfResidentRank]>)
        -> Result<(Placement, GraphSet, GlmfWorkingSet), planner::Refusal> {
        let mut input = input.clone();
        input.automatic_context = automatic_context;
        let mut process = admission::ProcessOwners::new((512 << 30, 512 << 30), 8 << 30);
        process.alternate_resident = alternate.map(<[GlmfResidentRank]>::to_vec);
        let selected = admission::prepare(&input, sms, rtx_experts, process)?.admit()?;
        Ok((selected.placement, selected.graphs, selected.working))
    }

    fn sampled<'a>(input: &GlmfInputs<'a>) -> GlmfInputs<'a> {
        let mut input = input.clone();
        for (capacity, baseline) in &mut input.gpus {
            let Baseline::Planned { context_bytes, loaded_bytes } = *baseline else { unreachable!() };
            *baseline = Baseline::Measured { free_bytes: *capacity - context_bytes - loaded_bytes };
        }
        input
    }

    #[test]
    fn contract_graph_bound_covers_exact_selected_shapes_without_pool_walk() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(45)).unwrap();
        for context in [2048, 2051, 4096, 65_536, 131_072, 262_144, 1_048_576] {
            for pool in [256, 4096, 32_768, 131_072, 262_144, 1_360_896, 2_097_152] {
                let bound = startup_graph_bound(&cfg, cfg.layers, context, pool, 16, true, 2, 64, 188);
                let exact = startup_graphs(&cfg, cfg.layers, context, pool, 16, true, 2, 64, 188);
                assert_eq!(bound.shapes, exact.shapes, "context={context}, pool={pool}");
                assert_eq!(bound.ranks, exact.ranks);
            }
        }
    }

    #[test]
    fn contract_zero_and_unset_preserve_exact_geometry_and_first_refusal() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        input.graphs = GraphSet::budget(&[64 << 20; 2]);
        let unset = contract(&input).unwrap();
        input.requested_pool = Some(0);
        let zero = contract(&input).unwrap();
        assert_eq!(unset.solved.pool_tokens, zero.solved.pool_tokens);
        assert_eq!(unset.solved.decisions, zero.solved.decisions);
        assert_eq!(unset.solved.facts.allocations, zero.solved.facts.allocations);
        input.requested_pool = Some(1 << 40);
        let first = contract(&input).err().unwrap().to_string();
        assert_eq!(first, contract(&input).err().unwrap().to_string());
        assert!(!first.contains("reserve does not fit"));
    }

    #[test]
    fn contract_graph_exact_substitution_preserves_ready_and_growth_phases() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4)).unwrap();
        for local in [false, true] {
            let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
            input.local_tp2_exl3 = local;
            input.spark_ranks = 0;
            input.requested_pool = Some(262_144);
            let admitted = contract(&input).unwrap();
            assert_eq!(admitted.validate_spark_topology(4), Err(admission::TopologyRefusal::IdleSparks));
            admitted.validate_spark_topology(0).unwrap();
            for gpu in 0..2 {
                let ready = admitted.solved.facts.allocations.iter()
                    .find(|owner| owner.id == format!("glmf.graphs.{gpu}.ready")).unwrap();
                let growth = admitted.solved.facts.allocations.iter()
                    .find(|owner| owner.id == format!("glmf.graphs.{gpu}.growth")).unwrap();
                assert_eq!(ready.bytes + growth.bytes, admitted.graphs.bytes(gpu));
                assert_eq!(growth.bytes, admitted.graphs.ranks[gpu].margin);
                assert_eq!(ready.lifetime.first, planner::Phase::Warmup);
                assert_eq!(growth.lifetime.first, planner::Phase::Growth);
            }
            assert_eq!(admitted.solved.phases,
                planner::evaluate(&admitted.solved.facts, &admitted.solved.budgets).unwrap());
        }
    }

    #[test]
    fn contract_malformed_process_and_family_facts_are_typed_refusals() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        for host in [(0, 0), (100, 101)] {
            assert!(matches!(admission::prepare(&input, 188, true,
                admission::ProcessOwners::new(host, 0)), Err(planner::Refusal::InvalidFact(_))));
        }
        input.drafter_scratch.pop();
        assert!(matches!(contract(&input), Err(planner::Refusal::InvalidFact(_))));
        input.drafter_scratch.push(0);
        input.graphs.ranks[0].margin = input.graphs.ranks[0].bytes + 1;
        assert!(matches!(contract(&input), Err(planner::Refusal::InvalidFact(_))));
    }

    #[test]
    fn contract_encoder_owners_and_host_pinned_capacity_are_joint() {
        use crate::plan::encoder::{EncoderKind, EncoderPlacement};
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        input.graphs = GraphSet::budget(&[64 << 20; 2]);
        input.requested_pool = Some(262_144);
        let encoder = |kind| EncoderPlacement { kind, weights: 123456, scratch: 654321,
            replicas: vec![], reason: "test physical owner".into(), shortfall: 0 };
        for kind in [EncoderKind::Off, EncoderKind::Rtx { gpu: 1 }, EncoderKind::Spark { rank: 2 }] {
            let mut process = admission::ProcessOwners::new((512 << 30, 512 << 30), 8 << 30);
            process.encoders = vec![encoder(kind.clone())];
            process.media_cache_bytes = 7654321;
            let admitted = admission::prepare(&input, 188, true, process).unwrap().admit().unwrap();
            let media = admitted.solved.facts.allocations.iter().find(|owner| owner.id == "glmf.media.cache");
            assert_eq!(media.map(|owner| owner.bytes), (kind != EncoderKind::Off).then_some(7654321));
            if kind == EncoderKind::Off {
                assert!(!admitted.solved.facts.allocations.iter().any(|owner| owner.id.starts_with("glmf.encoder.")));
            } else {
                let owners = admitted.solved.facts.allocations.iter().filter(|owner| owner.id.starts_with("glmf.encoder."))
                    .collect::<Vec<_>>();
                assert_eq!(owners.len(), 2);
                assert_eq!(owners.iter().map(|owner| owner.bytes).sum::<u64>(), 123456 + 654321);
                let expected = if matches!(kind, EncoderKind::Rtx { .. }) { planner::Location::coordinator(1) }
                    else { planner::Location { device: planner::Device::Spark(2), domain: planner::Domain::Managed } };
                assert!(owners.iter().all(|owner| owner.location == expected && owner.lifetime == planner::Lifetime::RESIDENT));
                if matches!(kind, EncoderKind::Spark { .. }) { assert!(!admitted.working.spark_layers.is_empty()); }
            }
        }
        for kind in [EncoderKind::Rtx { gpu: 2 }, EncoderKind::Spark { rank: 4 }] {
            let mut process = admission::ProcessOwners::new((512 << 30, 512 << 30), 8 << 30);
            process.encoders = vec![encoder(kind)];
            assert!(matches!(admission::prepare(&input, 188, true, process), Err(planner::Refusal::InvalidFact(_))));
        }
        input.onboard = Onboard::Layers(input.experts.len());
        let mut process = admission::ProcessOwners::new((512 << 30, 512 << 30), 8 << 30);
        process.encoders = vec![encoder(EncoderKind::Spark { rank: 0 })];
        assert!(admission::prepare(&input, 188, true, process).and_then(|prepared| prepared.admit()).is_err());
        let mut process = admission::ProcessOwners::new((1 << 30, 1 << 30), 0);
        for (id, location) in [("host", planner::Location::HOST), ("pinned", planner::Location::PINNED)] {
            process.allocations.push(planner::Allocation::buffer(id, location, 600 << 20,
                planner::Lifetime::RESIDENT, Category::Staging, id));
        }
        let refusal = admission::prepare(&input, 188, true, process).unwrap().admit().err().unwrap();
        assert!(refusal.to_string().contains("coordinator host RAM"), "{refusal}");
    }

    #[test]
    fn kda_only_pool_backing_is_named_and_limits_automatic_capacity() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        for first in [0, 1] {
            let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
            input.attention = GlmfAttention { mode: AttentionPlacement::Layers, first_owner: first,
                ..Default::default() };
            input.index = cache::GlmfIndexCache::Compact;
            input.pool_marks = true;
            input.onboard = Onboard::Layers(0);
            input.drafter_bytes = 0;
            input.resident.iter_mut().for_each(|rank| rank.embedding = 0);
            input.graphs = GraphSet::budget(&[64 << 20; 2]);
            input.automatic_context = true;
            let owner = (1 - first) as usize;
            let marked = contract(&input).unwrap();
            let pool = marked.solved.facts.allocations.iter()
                .find(|allocation| allocation.id == format!("glmf.pool.{owner}")).unwrap();
            let units = marked.solved.pool_tokens / 256;
            let per_unit = pool.bytes / units;
            let used = marked.solved.phases.iter().find(|phase| phase.phase == planner::Phase::Growth
                && phase.budget == format!("rtx{owner}")).unwrap().owned;
            let runtime = marked.solved.facts.allocations.iter()
                .find(|allocation| allocation.id == format!("glmf.runtime.{owner}")).unwrap().bytes;
            let fixed = used - pool.bytes - runtime;
            input.gpus[owner].1 = Baseline::Measured {
                free_bytes: fixed + 64 * per_unit + input.headroom_bytes };
            let bounded = contract(&input).unwrap();
            assert_eq!(bounded.placement.pool_tokens, 64 * 256);
            input.pool_marks = false;
            assert!(contract(&input).unwrap().placement.pool_tokens > bounded.placement.pool_tokens);
        }
    }

    #[test]
    fn copy_window_startup_graphs_match_serving_inventory() {
        let mut config = crate::plan::testing::glm5_flash_config(45);
        config["text_config"]["layer_types"] = serde_json::json!((0..45).map(|l|
            if l % 4 == 3 { "deepseek_sparse_attention" } else { "linear_attention" }).collect::<Vec<_>>());
        let cfg = GlmNextConfig::from_hf(&config).unwrap();
        // No external drafter still captures speculative graphs for default copy windows.
        let copy = startup_graphs(&cfg, 45, 262_144, 262_144, 8, true, 2, 64, 188);
        assert_eq!(copy.shapes, 280);
        assert_eq!(copy.ranks.iter().map(|r| r.executables).collect::<Vec<_>>(), [12_880, 12_600]);
        assert_eq!([copy.at_ready(0), copy.at_ready(1)], [2_082_618_720, 2_131_554_600]);
        let plain = startup_graphs(&cfg, 45, 262_144, 262_144, 8, false, 2, 64, 188);
        assert_eq!(plain.shapes, 112);
        assert_eq!(plain.ranks.iter().map(|r| r.executables).collect::<Vec<_>>(), [5_152, 5_040]);
    }

    #[test]
    fn local_tp2_runtime_and_graphs_equal_staged_ready() {
        let mut config = crate::plan::testing::glm5_flash_config(45);
        config["text_config"]["layer_types"] = serde_json::json!((0..45).map(|l|
            if l % 4 == 3 { "deepseek_sparse_attention" } else { "linear_attention" }).collect::<Vec<_>>());
        let cfg = GlmNextConfig::from_hf(&config).unwrap();
        let mut graphs = startup_graphs(&cfg, 45, 262_144, 262_144, 8, true, 2, 64, 188);
        calibrate_local_tp2_graphs(&mut graphs);
        for (rank, (baseline, ready)) in [(1_027_511_116u64, 3_101_594_444u64),
            (900_673_356, 2_905_550_668)].into_iter().enumerate() {
            let items = inventory::glmf_local_tp2_runtime_items(rank);
            assert_eq!(items.iter().map(|i| i.bytes).sum::<u64>(), baseline);
            assert_eq!(baseline + graphs.at_ready(rank), ready);
            let code = loaded_code("glmf", "exl3-tp2-local", true, rank as u8).unwrap();
            assert_eq!(inventory::ArchContext::coordinator(inventory::PRO_TOTAL_BYTES, None).context_bytes + code.bytes, baseline);
            let sampled = [598_999_040, 609_484_800][rank];
            assert_eq!(sampled + code.pending(sampled, 586_416_128), baseline);
            assert!(graphs.growth(rank) >= 64 << 20);
        }
    }

    #[test]
    fn planner_equals_runtime_glm5_flash() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(45)).unwrap();
        for (ranks, capacity) in [(1, 32 << 30), (1, inventory::PRO_TOTAL_BYTES), (2, inventory::PRO_TOTAL_BYTES)] {
            let mut planned = inputs(&cfg, ranks, capacity);
            planned.automatic_context = true;
            let p = contract(&planned).unwrap();
            let r = contract(&sampled(&planned)).unwrap();
            assert_eq!(p.solved.pool_tokens, r.solved.pool_tokens);
            assert_eq!(p.solved.decisions, r.solved.decisions);
            assert_eq!(p.solved.facts.allocations, r.solved.facts.allocations);
            assert_eq!(p.graphs, r.graphs);
            assert_eq!(p.working, r.working);
        }
    }

    fn draft_config() -> serde_json::Value {
        serde_json::json!({"hidden_size":4096, "intermediate_size":12288, "num_hidden_layers":5,
            "num_attention_heads":32, "num_key_value_heads":8, "head_dim":128, "vocab_size":154880,
            "dflash_config":{"target_layer_ids":[0,8,16,24,32], "conv_group_size":16,
                "selector_rank":256, "block_size":8}})
    }

    #[test]
    fn selected_spark_working_set_matches_planned_and_sampled_baselines() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4)).unwrap();
        for (onboard, expected) in [(Onboard::Auto, 3), (Onboard::Layers(1), 1)] {
            let mut planned = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
            planned.onboard = onboard;
            planned.requested_pool = Some(2 << 20);
            planned.graphs = GraphSet::budget(&[64 << 20; 2]);
            let target = 2 << 20;
            let (p, pg, working) = admitted_case(&planned, false, 188, true, None).unwrap();
            assert_eq!(p.onboard_layers, expected);
            let spark = expected < planned.experts.len();
            assert_eq!(working.spark_ranks, if spark { 4 } else { 0 });
            assert_eq!(working.prefill_lanes, 2);
            assert_eq!(working.spark_layers, if spark { vec![2, 3] } else { vec![] });
            let mut measured = planned.clone();
            for rank in 0..2 {
                let Baseline::Planned { context_bytes, loaded_bytes } = planned.gpus[rank].1 else { unreachable!() };
                let sampled = context_bytes / 3 + loaded_bytes;
                measured.gpus[rank].1 = Baseline::Measured { free_bytes: measured.gpus[rank].0 - sampled };
                measured.pending_code[rank] += context_bytes - sampled;
            }
            let (r, rg, runtime_working) = admitted_case(&measured, false, 188, true, None).unwrap();
            assert_eq!(pg, rg);
            assert_eq!(working, runtime_working);
            assert_eq!((p.pool_tokens, &p.layers, p.tp2), (r.pool_tokens, &r.layers, r.tp2));
            let group = |placed: &Placement, rank: usize, name: &str| placed.items[rank].iter()
                .filter(|d| d.group == name).map(|d| d.bytes).sum::<u64>();
            assert_eq!(group(&p, 0, "Spark intake"), if spark { 4 * cuteafd_core::serving_capacity::spark_intake_allocation_bytes(2 * 4096 * cfg.hidden as u64 * 2).unwrap() } else { 0 });
            for rank in 0..2 {
                assert_eq!(group(&p, rank, "steps"), if spark { planned.workspace[rank].demand.bytes } else { planned.local_workspace[rank].demand.bytes });
                let total = |placed: &Placement| placed.items[rank].iter().map(|d| d.bytes).sum::<u64>();
                assert_eq!(total(&p), total(&r));
            }
        }
    }

    #[test]
    fn spark_prefill_slots_share_one_padded_backing_per_rank() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4)).unwrap();
        for gpus in [1, 2] {
            let mut input = inputs(&cfg, gpus, inventory::PRO_TOTAL_BYTES);
            input.onboard = Onboard::Layers(1);
            input.prefill_rows = 80;
            for format in [cuteafd_core::SparkReplyFormat::Bf16, cuteafd_core::SparkReplyFormat::Fp8,
                    cuteafd_core::SparkReplyFormat::Nvfp4] {
                input.spark_reply_format = format;
                let selected = contract(&input).unwrap();
                let intake = selected.placement.items.iter().flatten().filter(|d| d.group == "Spark intake")
                    .map(|d| d.bytes).sum::<u64>();
                let rows = input.prefill_lanes * input.prefill_rows;
                assert_eq!(intake, 4 * cuteafd_core::serving_capacity::spark_intake_allocation_bytes(
                    rows * cfg.hidden as u64 * 2).unwrap());
            }
        }
    }

    #[test]
    fn empty_spark_candidate_can_fit_only_because_its_own_costs_are_charged() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        input.graphs = GraphSet::budget(&[64 << 20; 2]);
        input.requested_pool = Some(2 << 20);
        input.workspace = workspace_fixture(2, 100 << 30);
        let (placed, _, working) = admitted_case(&input, false, 188, true, None).unwrap();
        assert_eq!(placed.onboard_layers, input.experts.len());
        assert!(working.spark_layers.is_empty());
        assert_eq!((working.spark_ranks, working.prefill_lanes), (0, 2));
        assert!(placed.items[0].iter().all(|d| d.group != "Spark intake"));
        input.onboard = Onboard::Layers(1);
        assert!(admitted_case(&input, false, 188, true, None).is_err());
    }

    #[test]
    fn zero_peer_auto_pool_stops_at_target_for_planned_and_sampled_baselines() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        input.spark_ranks = 0;
        input.prefill_lanes = 1;
        input.workspace = input.local_workspace.clone();
        input.graphs = GraphSet::budget(&[64 << 20; 2]);
        for pool in [None, Some(2 << 20)] {
            input.requested_pool = pool;
            let (placed, graphs, working) = admitted_case(&input, false, 188, true, None).unwrap();
            assert_eq!(placed.pool_tokens, 2 << 20);
            assert_eq!(placed.onboard_layers, input.experts.len());
            assert!(placed.layers.iter().enumerate().filter(|(layer, _)| !cfg.dense[*layer])
                .all(|(_, layer)| layer.experts == ExpertHome::RtxTp2));
            let mut sampled = input.clone();
            for rank in 0..2 {
                let Baseline::Planned { context_bytes, loaded_bytes } = input.gpus[rank].1 else { unreachable!() };
                sampled.gpus[rank].1 = Baseline::Measured { free_bytes: sampled.gpus[rank].0 - context_bytes - loaded_bytes };
            }
            let (runtime, runtime_graphs, runtime_working) = admitted_case(&sampled, false, 188, true, None).unwrap();
            assert_eq!((placed.pool_tokens, &placed.layers, placed.tp2), (runtime.pool_tokens, &runtime.layers, runtime.tp2));
            assert_eq!(graphs, runtime_graphs);
            assert_eq!(working, runtime_working);
            assert_eq!((working.spark_ranks, working.prefill_lanes), (0, 1));
            assert!(working.spark_layers.is_empty());
        }
    }

    #[test]
    fn native_tp2_zero_spark_dflash_default_concurrency_plan_equals_runtime() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(45)).unwrap();
        let capacity = inventory::PRO_TOTAL_BYTES;
        let options = crate::plan::layout::LayoutOptions { rtx_bytes: vec![capacity; 2], ..Default::default() };
        let mut planned = inputs(&cfg, 2, capacity);
        planned.spark_ranks = 0;
        planned.prefill_lanes = 1;
        planned.prefill_rows = 4096;
        planned.sequences = options.resolved_concurrency("glm5_flash");
        planned.state_slots = options.resolved_state_slots("glm5_flash");
        planned.workspace = planned.local_workspace.clone();
        planned.speculation = true;
        let (slots, batch) = crate::families::glm5::draft_representation::flash_serving_draft_limits(
            planned.sequences as usize, options.glm_draft_sequences() as usize, None);
        let config = draft_config();
        let draft_capacity = crate::families::glm5::draft_representation::flash_draft_capacity(&config, slots, batch).unwrap();
        let (owned, scratch) = drafter_inventory(&config, draft_capacity.context_slots, draft_capacity.max_batch_sequences, &[188; 2],
            crate::families::glm5::draft_representation::GlmDraftRepresentation::Fp8Only,
            2, options.draft_kv_cache).unwrap();
        planned.drafter_bytes = owned.memory.resident;
        planned.drafter_weights = owned.resident_weights;
        planned.drafter_scratch = scratch;
        let mut runtime = planned.clone();
        let boot = cuteafd_core::serving_capacity::default_concurrency(capacity);
        runtime.sequences = boot;
        runtime.state_slots = cuteafd_core::concurrency::state_slots(boot);
        assert_eq!(planned.sequences, 16);
        for rank in 0..2 {
            let Baseline::Planned { context_bytes, loaded_bytes } = planned.gpus[rank].1 else { unreachable!() };
            runtime.gpus[rank].1 = Baseline::Measured { free_bytes: capacity - context_bytes - loaded_bytes };
        }
        let (plan, pg, pw) = admitted_case(&planned, true, 188, true, None).unwrap();
        let (live, lg, lw) = admitted_case(&runtime, true, 188, true, None).unwrap();
        assert_eq!(plan.pool_tokens, live.pool_tokens);
        assert_eq!(plan.items, live.items);
        assert_eq!(plan.layers, live.layers);
        assert_eq!(pg, lg);
        assert_eq!(pw, lw);
    }

    #[test]
    fn local_and_tp2_candidates_preserve_every_admitted_lane_without_spark_costs() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4)).unwrap();
        for ranks in [1, 2] {
            let mut input = inputs(&cfg, ranks, inventory::PRO_TOTAL_BYTES);
            input.spark_ranks = 0;
            input.graphs = GraphSet::budget(&vec![64 << 20; ranks]);
            input.requested_pool = Some(2 << 20);
            for lanes in [1, 2] {
                input.prefill_lanes = lanes;
                let (placed, _, working) = admitted_case(&input, false, 188, true, None).unwrap();
                assert_eq!((working.spark_ranks, working.prefill_lanes), (0, lanes));
                assert_eq!(working.rtx_layers, vec![1, 2, 3]);
                assert!(working.spark_layers.is_empty());
                working.validate(&placed).unwrap();
                input.onboard = Onboard::Layers(working.rtx_layers.len());
                let selected = contract(&input).unwrap();
                assert_eq!(selected.working.prefill_lanes, lanes);
                for rank in 0..ranks {
                    let charged = |name: &str| selected.placement.items[rank].iter().filter(|d| d.group == name)
                        .map(|d| d.bytes).sum::<u64>();
                    assert_eq!(charged("Spark intake"), 0);
                    assert_eq!(charged("steps"), input.local_workspace[rank].demand.bytes);
                    if ranks == 2 {
                        assert_eq!(charged("peer exchange"), glmf_graphs::peer_exchange_bytes(lanes, 4096, 4096, 2,
                            input.representation.kda_fp8 && input.representation.output_shard));
                    }
                }
            }
        }
    }

    #[test]
    fn abort_words_follow_exchange_ownership_on_each_rank() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4)).unwrap();
        for ranks in [1, 2] {
            let mut input = inputs(&cfg, ranks, inventory::PRO_TOTAL_BYTES);
            for onboard in [0, 1, 3] {
                let working = GlmfWorkingSet::candidate(&input, onboard).unwrap();
                input.onboard = Onboard::Layers(working.rtx_layers.len());
                let selected = contract(&input).unwrap();
                for rank in 0..ranks {
                    let bytes = |group: &str| selected.placement.items[rank].iter()
                        .filter(|d| d.group == group).map(|d| d.bytes).sum::<u64>();
                    assert_eq!(bytes("peer abort word"), if ranks == 2 { 256 } else { 0 });
                    assert_eq!(bytes("route abort word"), if ranks == 2 && onboard > 0 { 256 } else { 0 });
                }
            }
        }
    }

    #[test]
    fn component_owned_rows_reject_forged_sets_and_drop_inactive_buffers() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        for profile in [&mut input.workspace, &mut input.local_workspace] {
            for (component, group) in [(GlmfComponent::RtxExperts, "RTX test buffer"),
                (GlmfComponent::SparkExperts, "Spark test buffer")] {
                profile.push(GlmfBufferDemand::new(component, 0, Category::Workspace, group, 987654, Basis::Formula));
            }
        }
        for n in [0, 1, 3] {
            let working: crate::placement::SparkWorkingSet = GlmfWorkingSet::candidate(&input, n).unwrap();
            input.onboard = Onboard::Layers(working.rtx_layers.len());
                let selected = contract(&input).unwrap();
            assert_eq!(working.selected_ranks().count(), if n == input.experts.len() { 0 } else { input.spark_ranks });
            let rows = cache::glmf_expert_rows(input.prefill_rows, input.decode_rows) as u32;
            let pinned = working.coordinator_staging(cuteafd_core::ExpertGeometry::GLM5_FLASH, rows, false).unwrap();
            assert_eq!(pinned, if n == input.experts.len() { 0 } else {
                input.prefill_lanes * u64::from(rows) * (input.cfg.hidden as u64 * 33 / 32 + input.cfg.topk as u64 * 8)
            });
            for (component, group) in [(GlmfComponent::RtxExperts, "RTX test buffer"),
                (GlmfComponent::SparkExperts, "Spark test buffer")] {
                assert_eq!(selected.placement.items.iter().flatten().any(|d| d.group == group), working.includes(component));
            }
            let mut forged = working;
            forged.prefill_lanes += 1;
            assert_ne!(forged, selected.working);
            assert!(forged.validate(&selected.placement).is_ok());
        }
    }

    #[test]
    fn selected_working_set_preserves_exact_counts() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        input.graphs = GraphSet::budget(&[64 << 20; 2]);
        for (onboard, expected) in [(Onboard::Layers(0), 0), (Onboard::Layers(2), 2), (Onboard::Layers(3), 3)] {
            input.onboard = onboard;
            let (placed, _, working) = admitted_case(&input, false, 188, true, None).unwrap();
            assert_eq!(placed.onboard_layers, expected);
            assert_eq!(working.spark_layers.len(), input.experts.len() - expected);
        }
    }

    #[test]
    fn spark_auto_keeps_smaller_pool_and_explicit_pool_stays_strict() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(45)).unwrap();
        let mut input = inputs(&cfg, 1, 32 << 30);
        input.max_context = 4 << 20;
        input.graphs = GraphSet::budget(&[64 << 20]);
        let (placed, _, working) = admitted_case(&input, true, 170, true, None).unwrap();
        assert!(placed.pool_tokens < input.max_context);
        assert_eq!(working.spark_ranks, 4);
        input.requested_pool = Some(4 << 20);
        assert!(admitted_case(&input, true, 170, true, None).is_err());
    }

    #[test]
    fn sparse_routed_candidates_preserve_exact_local_and_spark_inventories() {
        let mut cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4)).unwrap();
        cfg.dense[2] = true;
        for ranks in [1, 2] {
            let mut input = inputs(&cfg, ranks, inventory::PRO_TOTAL_BYTES);
            input.graphs = GraphSet::budget(&vec![64 << 20; ranks]);
            input.requested_pool = Some(2 << 20);
            for n in 0..=2 {
                input.onboard = Onboard::Layers(n);
                let (placed, _, working) = admitted_case(&input, false, 188, true, None).unwrap();
                assert_eq!(placed.onboard_layers, n);
                assert_eq!(working.rtx_layers, [1, 3][..n]);
                assert_eq!(working.spark_layers, [1, 3][n..]);
                assert_eq!(placed.layers[0].experts, ExpertHome::Dense);
                assert_eq!(placed.layers[2].experts, ExpertHome::Dense);
                working.validate(&placed).unwrap();
                let mut missing = working.clone();
                if let Some(layer) = missing.spark_layers.pop() {
                    assert!(missing.validate(&placed).is_err());
                    missing.spark_layers.push(layer);
                }
                missing.spark_layers.push(2);
                assert!(missing.validate(&placed).is_err());
            }
            input.onboard = Onboard::Auto;
            let (placed, _, working) = admitted_case(&input, false, 188, true, None).unwrap();
            assert_eq!(placed.onboard_layers, 2);
            assert_eq!(working.rtx_layers, [1, 3]);
            assert!(working.spark_layers.is_empty());
        }
    }

    #[test]
    fn draft_inventory_keeps_owned_bytes_independent_and_scratch_candidate_local() {
        use crate::families::glm5::draft_representation::{draft_resident_bytes_with_mode, GlmDraftRepresentation};
        use crate::families::qwen4::Qwen4KvCache;
        for mode in 0..=2 {
            for representation in [GlmDraftRepresentation::Fp8Only, GlmDraftRepresentation::Bf16Only] {
                for format in [Qwen4KvCache::Bf16, Qwen4KvCache::Fp8] {
                    let (owned, scratch) = drafter_inventory(&draft_config(), 8, 8, &[188, 170], representation, mode, format).unwrap();
                    for (rank, sms) in [188, 170].into_iter().enumerate() {
                        let expected = draft_resident_bytes_with_mode(&draft_config(), 8, 8,
                            sms, representation, mode, format).unwrap();
                        assert_eq!((owned.memory.resident, owned.resident_weights, scratch[rank]),
                            (expected.owned, expected.weights, expected.scratch));
                    }
                    let attention: Vec<_> = [188, 170].into_iter().map(|sms|
                        cache::draft_attention_workspace_bytes(8, 8, 2, 2056, 128, sms).unwrap()).collect();
                    if representation == GlmDraftRepresentation::Bf16Only { assert_eq!(scratch, attention); }
                    else { assert!(scratch.iter().zip(attention).all(|(&total, arena)| total > arena)); }
                    assert!(scratch[0] > scratch[1]);
                }
            }
        }
        let (owned, scratch) = drafter_inventory(&draft_config(), 8, 8, &[188, 170], GlmDraftRepresentation::Fp8Only, 2, crate::families::qwen4::Qwen4KvCache::Bf16).unwrap();
        let old_attention = 8 * 8 * 17 * 64 * 130 * 4;
        let attention: Vec<_> = [188, 170].into_iter().map(|sms|
            cache::draft_attention_workspace_bytes(8, 8, 2, 2056, 128, sms).unwrap()).collect();
        assert_eq!(owned.memory.resident, 1_835_700_096 - old_attention);
        // The shared fc arena adds 128 rows * (20480 / 256) K tiles * 4 B.
        let row_max_bytes = 128 * 80 * 4;
        assert_eq!(scratch, [30_491_648 + row_max_bytes + attention[0],
            28_394_496 + row_max_bytes + attention[1]]);
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        input.drafter_bytes = owned.memory.resident;
        input.drafter_weights = owned.resident_weights;
        input.drafter_scratch = scratch;
        let actual = contract(&input).unwrap().placement;
        assert_eq!(actual.movables, [(MovableId::Drafter, 0)]);
        assert_eq!(actual.items[0].iter().filter(|d| d.group == "DFlash").map(|d| d.bytes).sum::<u64>(), owned.memory.resident);
        assert_eq!(actual.items[0].iter().find(|d| d.group == "DFlash linear/attention scratch").unwrap().bytes,
            30_491_648 + row_max_bytes + attention[0]);
        assert!(actual.items[1].iter().all(|d| d.group != "DFlash" && d.group != "DFlash linear/attention scratch"));
        input.drafter_bytes += input.drafter_scratch[0];
        input.drafter_scratch.fill(0);
        let lumped = contract(&input).unwrap().placement;
        assert_eq!(actual.pool_tokens, lumped.pool_tokens);
        assert_eq!(actual.layers, lumped.layers);
        for rank in 0..2 {
            assert_eq!(actual.items[rank].iter().map(|d| d.bytes).sum::<u64>(),
                lumped.items[rank].iter().map(|d| d.bytes).sum::<u64>());
        }
    }

    #[test]
    fn draft_inventory_rejects_bad_devices_and_orphan_or_overflow_scratch() {
        use crate::families::glm5::draft_representation::GlmDraftRepresentation;
        for sms in [vec![], vec![188, 170, 188], vec![0], vec![4097], vec![188, 0]] {
            assert!(drafter_inventory(&draft_config(), 8, 8, &sms, GlmDraftRepresentation::Fp8Only, 2, crate::families::qwen4::Qwen4KvCache::Bf16).is_err());
        }
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        input.drafter_scratch = vec![256];
        assert!(matches!(contract(&input), Err(planner::Refusal::InvalidFact(_))));
        input.drafter_scratch = vec![256, 0];
        input.drafter_bytes = 0;
        assert!(matches!(contract(&input), Err(planner::Refusal::InvalidFact(_))));
        input.drafter_bytes = u64::MAX;
        assert!(contract(&input).is_err());
    }

    #[test]
    fn draft_inventory_keeps_dspark_local_and_category_separated() {
        use crate::families::glm5::draft_representation::GlmDraftRepresentation;
        let config = serde_json::json!({"speculators_model_type":"dspark", "block_size":4, "markov_rank":256,
            "aux_hidden_state_layer_ids":[1,9,17,25,33], "markov_head_type":"vanilla",
            "sample_from_anchor":true, "enable_confidence_head":true, "confidence_head_with_markov":true,
            "draft_vocab_size":154880, "mask_token_id":154000, "transformer_layer_config":{
                "hidden_size":4096, "intermediate_size":12288, "num_hidden_layers":3, "num_attention_heads":32,
                "num_key_value_heads":8, "head_dim":64, "vocab_size":154880, "rms_norm_eps":1e-6,
                "rope_parameters":{"rope_type":"default","rope_theta":10000.0}, "hidden_act":"silu",
                "layer_types":["sliding_attention","sliding_attention","sliding_attention"], "sliding_window":2048}});
        let (owned, scratch) = drafter_inventory(&config, 8, 8, &[188, 170], GlmDraftRepresentation::Fp8Only, 2, crate::families::qwen4::Qwen4KvCache::Bf16).unwrap();
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        input.drafter_bytes = owned.memory.resident;
        input.drafter_weights = owned.resident_weights;
        input.drafter_scratch = scratch;
        let selected = contract(&input).unwrap();
        assert_eq!(selected.placement.movables, [(MovableId::Drafter, 0)]);
        assert_eq!(selected.placement.items[0].iter().find(|item| item.group == "DFlash linear/attention scratch").unwrap().category,
            Category::Workspace);
    }

    #[test]
    fn remote_dflash_candidate_combines_scratch_bridge_and_whole_owner() {
        use crate::families::glm5::draft_representation::GlmDraftRepresentation;
        let config = draft_config();
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        let (owned, scratch) = drafter_inventory(&config, 8, 8, &[188, 170], GlmDraftRepresentation::Fp8Only, 2, crate::families::qwen4::Qwen4KvCache::Bf16).unwrap();
        input.drafter_bytes = owned.memory.resident;
        input.drafter_weights = owned.resident_weights;
        input.drafter_scratch = scratch;
        input.drafter_bridge = drafter_bridge_inventory(&config, 8, 8).unwrap();
        for home in [0, 1] {
            let mut process = admission::ProcessOwners::new((512 << 30, 512 << 30), 8 << 30);
            // Domains and component activations must be prepared together, so bias
            // the complete physical inventory rather than rewriting a solved home.
            input.resident[(1 - home) as usize].weights += 16 << 30;
            process.allocations.clear();
            let actual = admission::prepare(&input, 188, true, process).unwrap().admit().unwrap().placement;
            assert_eq!(actual.movables, [(MovableId::Drafter, home)]);
            assert_eq!(actual.items[home as usize].iter().filter(|d| d.group == "DFlash").map(|d| d.bytes).sum::<u64>(), owned.memory.resident);
            for rank in 0..2 {
                let bridge = actual.items[rank].iter().filter(|d| d.group.starts_with("DFlash bridge"))
                    .map(|d| d.bytes).sum::<u64>();
                let expected = if home == 1 { input.drafter_bridge.as_ref().unwrap().demands.iter()
                    .filter(|d| d.gpu as usize == rank).map(|d| d.bytes).sum() } else { 0 };
                assert_eq!(bridge, expected);
            }
            input.resident[(1 - home) as usize].weights -= 16 << 30;
        }
    }

    #[test]
    fn remote_dflash_solver_home_and_charges_equal_planned_and_sampled_baselines() {
        use crate::families::glm5::draft_representation::GlmDraftRepresentation;
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4)).unwrap();
        for expected_home in [0u8, 1] {
            let mut planned = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
            planned.spark_ranks = 0;
            planned.graphs = GraphSet::budget(&[64 << 20; 2]);
            planned.requested_pool = Some(262_144);
            // Make one target rank heavier without forcing the Movable's home.
            planned.resident[1 - expected_home as usize].weights += 6 << 30;
            let (owned, scratch) = drafter_inventory(&draft_config(), 8, 8, &[188, 170],
                GlmDraftRepresentation::Fp8Only, 2, crate::families::qwen4::Qwen4KvCache::Bf16).unwrap();
            planned.drafter_bytes = owned.memory.resident;
            planned.drafter_weights = owned.resident_weights;
            planned.drafter_scratch = scratch;
            planned.drafter_bridge = drafter_bridge_inventory(&draft_config(), 8, 8).unwrap();
            let (p, pg, pw) = admitted_case(&planned, false, 188, true, None).unwrap();
            assert_eq!(p.movables, [(MovableId::Drafter, expected_home)]);
            let mut measured = planned.clone();
            for rank in 0..2 {
                let Baseline::Planned { context_bytes, loaded_bytes } = planned.gpus[rank].1 else { unreachable!() };
                let sampled_code = context_bytes / 3 + loaded_bytes;
                let prepared_controls = 1536;
                let raw_free = measured.gpus[rank].0 - sampled_code - prepared_controls;
                // Runtime credits prepared controls before charging the selected candidate.
                measured.gpus[rank].1 = Baseline::Measured { free_bytes: raw_free + prepared_controls };
                measured.pending_code[rank] = context_bytes - sampled_code;
            }
            let (r, rg, rw) = admitted_case(&measured, false, 188, true, None).unwrap();
            assert_eq!((pg, pw), (rg, rw));
            assert_eq!((&p.movables, p.pool_tokens, &p.layers, p.tp2),
                (&r.movables, r.pool_tokens, &r.layers, r.tp2));
            let bridge = planned.drafter_bridge.as_ref().unwrap();
            for rank in 0..2 {
                let bytes = |placed: &Placement, group: &str| placed.items[rank].iter()
                    .filter(|item| item.group == group).map(|item| item.bytes).sum::<u64>();
                assert_eq!(bytes(&p, "DFlash"), if rank == expected_home as usize { owned.memory.resident } else { 0 });
                for demand in &bridge.demands {
                    if demand.gpu as usize != rank { continue; }
                    let expected = if expected_home == 1 { demand.bytes } else { 0 };
                    assert_eq!(bytes(&p, &demand.group), expected);
                    assert_eq!(bytes(&r, &demand.group), expected);
                }
                assert_eq!(bytes(&p, "DFlash linear/attention scratch"),
                    if rank == expected_home as usize { planned.drafter_scratch[rank] } else { 0 });
                let total = |placed: &Placement| placed.items[rank].iter().map(|item| item.bytes).sum::<u64>();
                assert_eq!(total(&p), total(&r));
            }
        }
    }

    #[test]
    fn remote_dflash_candidate_rejects_absent_owner_and_device() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        for ranks in [1, 2] {
            let mut input = inputs(&cfg, ranks, inventory::PRO_TOTAL_BYTES);
            input.drafter_bridge = drafter_bridge_inventory(&draft_config(), 8, 8).unwrap();
            if ranks == 2 { input.drafter_bytes = 0; }
            assert!(matches!(contract(&input), Err(planner::Refusal::InvalidFact(_))));
        }
        assert!(drafter_bridge_inventory(&draft_config(), 0, 1).is_err());
        assert!(drafter_bridge_inventory(&draft_config(), 8, 9).is_err());
        assert_eq!(drafter_bridge_inventory(&serde_json::json!({"speculators_model_type":"dspark"}), 8, 8).unwrap(), None);
    }

    #[test]
    fn remote_dflash_bridge_counts_full_rows_and_only_new_allocations() {
        let bridge = remote_dflash_bridge_demands(4096, 154_880, 6, 16 * 8).unwrap();
        assert_eq!(bridge.placement_gpu, 1);
        let actual: Vec<_> = bridge.demands.iter().map(|d|
            (d.gpu, d.category, d.group.as_str(), d.bytes, d.basis)).collect();
        assert_eq!(actual, vec![
            (0, Category::Drafter, "DFlash bridge taps", 100_663_296, Basis::Formula),
            (0, Category::Drafter, "DFlash bridge input ids", 512, Basis::Formula),
            (0, Category::Drafter, "DFlash bridge gathered input", 1_048_576, Basis::Formula),
            (0, Category::Drafter, "DFlash bridge normalized hidden", 1_048_576, Basis::Formula),
            (0, Category::Drafter, "DFlash bridge logits", 79_298_560, Basis::Formula),
            (0, Category::Drafter, "DFlash bridge head scratch", 4_194_304, Basis::Formula),
            (0, Category::Transport, "DFlash bridge exchange", 1536, Basis::Formula),
            (1, Category::Transport, "DFlash bridge exchange", 1536, Basis::Formula),
        ]);
        assert_eq!(bridge.demands.iter().filter(|d| d.gpu == 0).map(|d| d.bytes).sum::<u64>(), 186_255_360);
        // Even tiny direct payloads retain their allocator's 256B minimum.
        let tiny = remote_dflash_bridge_demands(1, 1, 1, 1).unwrap();
        for group in ["DFlash bridge input ids", "DFlash bridge gathered input",
            "DFlash bridge normalized hidden", "DFlash bridge logits"] {
            assert_eq!(tiny.demands.iter().find(|d| d.group == group).unwrap().bytes, 256);
        }
    }

    #[test]
    fn remote_dflash_bridge_rejects_geometry_product_and_total_overflow() {
        for (hidden, vocab, taps, rows) in [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1),
            (1, 1, 1, 0), (1, 1, 1, i32::MAX as u64 + 1)] {
            assert!(matches!(remote_dflash_bridge_demands(hidden, vocab, taps, rows),
                Err(PlacementError::Inventory(_))));
        }
        for (hidden, vocab, taps, rows) in [(u64::MAX, 1, 1, 1), (1, u64::MAX, 1, 1),
            (1, 1, u64::MAX, 1), (u64::MAX / 4096, 1, 1, 1)] {
            assert!(matches!(remote_dflash_bridge_demands(hidden, vocab, taps, rows),
                Err(PlacementError::Overflow(_))));
        }
    }

    #[test]
    fn remote_dflash_bridge_stays_dormant_for_gpu0_only_drafter() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        let input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        let actual = contract(&input).unwrap().placement;
        assert_eq!(actual.movables, [(MovableId::Drafter, 0)]);
        assert!(actual.items.iter().flatten().all(|d| !d.group.starts_with("DFlash bridge")));
    }

    #[test]
    fn glm5_flash_header_costs_and_fp8_package_scratch_are_static() {
        use crate::plan::testing::{glm5_flash_config, glm5_flash_tensors, write_snapshot};
        use crate::formats::fp8_experts::{Fp8Projection, Slicing};
        let dir = tempfile::tempdir().unwrap();
        let config = glm5_flash_config(2);
        write_snapshot(dir.path(), &config, &glm5_flash_tensors(&config), None);
        let catalog = crate::read_expert_catalog(dir.path()).unwrap();
        let fp8 = catalog.fp8().unwrap();
        let costs = expert_costs(&catalog, true).unwrap();
        let exact = |tp, rank| Fp8Projection::ALL.iter().map(|&proj| {
            let (bytes, _) = fp8.slice_bytes_with(proj, tp, rank, Slicing::Blocks(128)).unwrap();
            (bytes * fp8.shape().experts).max(256) as u64
                + fp8.scale_region_bytes_with(proj, tp, rank, Slicing::Blocks(128)).unwrap().max(256) as u64
        }).sum::<u64>();
        assert_eq!(costs[0].whole.resident, exact(1, 0));
        assert_eq!(costs[0].half.map(|c| c.resident), [exact(2, 0), exact(2, 1)]);
        let share = dir.path().join("share");
        let package = dir.path().join("lib/fp8/fp8-glmf");
        std::fs::create_dir_all(&share).unwrap();
        std::fs::create_dir_all(&package).unwrap();
        let path = share.join("PROGRAMS.json");
        assert!(expert_workspace(&catalog, Some(&path), 4096, 2).is_err());
        std::fs::write(package.join("manifest.json"), serde_json::json!({"layouts": {
            "tp1": {"capacities":[{"capacity":4096,"scratch_bytes":12345}]},
            "tp2": {"capacities":[{"capacity":4096,"scratch_bytes":67890}]}}}).to_string()).unwrap();
        assert_eq!(expert_workspace(&catalog, Some(&path), 4096, 1).unwrap(), 12345);
        assert_eq!(expert_workspace(&catalog, Some(&path), 4096, 2).unwrap(), 67890 + 4096 * 4096 * 2);
        assert!(expert_workspace(&catalog, Some(&path), 4097, 2).is_err());
        // Measured GLMf TP2 package ABI maxima, plus its owned BF16 output.
        std::fs::write(package.join("manifest.json"), serde_json::json!({"layouts": {
            "tp2": {"capacities":[{"capacity":1024,"scratch_bytes":163779584},
                {"capacity":4096,"scratch_bytes":734346240}]}}}).to_string()).unwrap();
        for (rows, scratch) in [(1024, 163779584), (4096, 734346240)] {
            assert_eq!(expert_workspace(&catalog, Some(&path), rows, 2).unwrap(), scratch + rows * 4096 * 2);
        }
        let cfg = GlmNextConfig::from_hf(&config).unwrap();
        let checkpoint = crate::plan::Checkpoint::open(dir.path()).unwrap();
        assert_eq!(crate::families::glm5_flash::resident::router_replica_bytes(&checkpoint, &cfg, 2).unwrap(),
            288 * 4096 * 2 + 288 * 4);
    }

    #[test]
    fn glm5_flash_nvfp4_requires_native_a4_package_scratch() {
        use crate::plan::testing::{glm5_flash_config, glm5_flash_tensors, nvfp4, write_snapshot};
        let dir = tempfile::tempdir().unwrap();
        let mut config = glm5_flash_config(2);
        config["quantization_config"] = serde_json::json!({"quant_method":"modelopt", "quant_algo":"NVFP4",
            "config_groups":{"group_0":{"weights":{"num_bits":4,"type":"float","group_size":16}}}});
        let mut tensors = glm5_flash_tensors(&config);
        tensors.retain(|(name, ..)| !name.contains(".mlp.experts."));
        for expert in 0..288 {
            for (proj, n, k) in [("gate_proj", 2048, 4096), ("up_proj", 2048, 4096), ("down_proj", 4096, 2048)] {
                tensors.extend(nvfp4(&format!("model.language_model.layers.1.mlp.experts.{expert}.{proj}"), n, k));
            }
        }
        write_snapshot(dir.path(), &config, &tensors, None);
        let catalog = crate::read_expert_catalog(dir.path()).unwrap();
        assert_eq!(catalog.fp8().unwrap().format(), crate::formats::fp8_experts::ExpertFormat::Nvfp4);
        let share = dir.path().join("share");
        std::fs::create_dir_all(&share).unwrap();
        let manifest = share.join("PROGRAMS.json");
        let package = |name: &str, scratch| {
            let root = dir.path().join("lib/fp8").join(name);
            std::fs::create_dir_all(root.join("tp2")).unwrap();
            std::fs::write(root.join("manifest.json"), serde_json::json!({"layouts":{
                "tp2":{"capacities":[{"capacity":4096,"scratch_bytes":scratch}]}}}).to_string()).unwrap();
            root
        };
        package("fp8-glmf-nvfp4", 12345);
        let a4 = package("fp8-glmf-nvfp4a4", 67890);
        // work/p0 at 2b373eeb29 defaulted A4 on and chose this installed
        // directory before its obsolete weight-only fallback.
        let baseline = if a4.join("tp2").is_dir() { a4.clone() }
            else { dir.path().join("lib/fp8/fp8-glmf-nvfp4") };
        let selected = dir.path().join("lib/fp8")
            .join(catalog.fp8().unwrap().format().package_name("glmf"));
        assert_eq!(selected.join("tp2"), baseline.join("tp2"));
        assert_eq!(selected, a4);
        let output = 4096 * 4096 * 2;
        assert_eq!(expert_workspace(&catalog, Some(&manifest), 4096, 2).unwrap(), 67890 + output);
        std::fs::remove_dir(a4.join("tp2")).unwrap();
        assert!(expert_workspace(&catalog, Some(&manifest), 4096, 2).is_err());
    }

    #[test]
    fn glm5_flash_graph_inventory_rebuilds_at_the_smaller_pool() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(45)).unwrap();
        let input = inputs(&cfg, 1, 32 << 30);
        let (placed, graphs) = admitted_case(&input, true, 170, true, None).map(|(placed, graphs, _)| (placed, graphs)).unwrap();
        assert!(placed.pool_tokens < 1 << 20);
        let context = glmf_graphs::admitted_graph_context(input.max_context as usize, placed.pool_tokens as usize, true);
        let expected = startup_graphs(&cfg, cfg.layers, context, placed.pool_tokens as usize,
            input.sequences as usize, true, 1, 64, 170);
        assert_eq!(graphs.ranks, expected.ranks);
    }

    #[test]
    fn glm5_flash_graph_rebuild_uses_served_policy_not_drafter_residency() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        for (drafter, speculation) in [(0, true), (3 << 30, false)] {
            let mut input = inputs(&cfg, 1, inventory::PRO_TOTAL_BYTES);
            input.drafter_bytes = drafter;
            input.speculation = speculation;
            input.sequences = 128;
            let (placed, graphs) = admitted_case(&input, true, 188, true, None).map(|(placed, graphs, _)| (placed, graphs)).unwrap();
            let context = glmf_graphs::admitted_graph_context(input.max_context as usize, placed.pool_tokens as usize, true);
            let expected = startup_graphs(&cfg, cfg.layers, context, placed.pool_tokens as usize,
                64, speculation, 1, 64, 188);
            assert_eq!(graphs.ranks, expected.ranks);
        }
    }

    #[test]
    fn glm5_flash_fixed_onboard_graphs_match_admitted_pool() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(45)).unwrap();
        let mut input = inputs(&cfg, 1, inventory::PRO_TOTAL_BYTES);
        input.onboard = Onboard::Layers(1);
        let (placed, graphs) = admitted_case(&input, true, 188, true, None).map(|(placed, graphs, _)| (placed, graphs)).unwrap();
        assert_eq!(placed.onboard_layers, 1);
        let context = glmf_graphs::admitted_graph_context(input.max_context as usize, placed.pool_tokens as usize, true);
        let expected = startup_graphs(&cfg, cfg.layers, context, placed.pool_tokens as usize,
            input.sequences as usize, true, 1, 64, 188);
        assert_eq!(graphs.ranks, expected.ranks);
    }

    #[test]
    fn unit_admission_keeps_every_shared_policy_wave_without_clamping() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(45)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        input.prefill_lanes = 4;
        for mode in [AttentionPlacement::Heads, AttentionPlacement::Context] {
            input.attention.mode = mode;
            assert_eq!(GlmfWorkingSet::candidate(&input, 0).unwrap().prefill_lanes, 4);
        }
        input.attention.mode = AttentionPlacement::Layers;
        for lanes in 1..=crate::serving_capacity::prefill_waves::MAX_WAVES {
            input.prefill_lanes = lanes;
            for onboard in [0, input.experts.len()] {
                let working = GlmfWorkingSet::candidate(&input, onboard).unwrap();
                assert_eq!(working.prefill_lanes, lanes);
                validate_layer_pipeline_lanes(working.prefill_lanes, working.prefill_lanes as usize).unwrap();
            }
        }
        for lanes in [0, 5] {
            input.prefill_lanes = lanes;
            assert!(GlmfWorkingSet::candidate(&input, 0).is_err());
        }
    }

    #[test]
    fn candidate_mla_modes_have_no_replicated_kv_and_keep_kda_split() {
        let mut config = crate::plan::testing::glm5_flash_config(45);
        config["text_config"]["layer_types"] = serde_json::json!((0..45).map(|l|
            if l % 4 == 3 { "deepseek_sparse_attention" } else { "linear_attention" }).collect::<Vec<_>>());
        let cfg = GlmNextConfig::from_hf(&config).unwrap();
        for sparks in [0, 4] {
            let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
            input.spark_ranks = sparks;
            input.onboard = Onboard::Layers(if sparks == 0 { input.experts.len() } else { 0 });
            input.pool_marks = false;
            input.graphs = GraphSet::budget(&[64 << 20; 2]);
            input.drafter_bytes = 0;
            input.requested_pool = Some(2 << 20);
            // Synthetic experts isolate cache storage while retaining the real
            // working-set, TP2, graph and per-layer hop admission arithmetic.
            for expert in &mut input.experts { expert.whole = Bytes2::default(); expert.half = [Bytes2::default(); 2]; }
            let heads = contract(&input).map(|a| a.placement).unwrap();
            let heads_bytes = replicated_mla_bytes(&cfg, input.index, &heads).unwrap();
            assert_eq!(heads_bytes, 11 * (256 * (528 + 512) + 64 * 132) * (2 << 20) / 256);
            for mode in [AttentionPlacement::Context, AttentionPlacement::Layers] {
                input.index = cache::GlmfIndexCache::Compact;
                input.attention.mode = mode;
                let placed = contract(&input).unwrap().placement;
                assert_eq!(placed.attention_placement, mode);
                assert_eq!(placed.pool_tokens, 2 << 20);
                assert_eq!(replicated_mla_bytes(&cfg, input.index, &placed).unwrap(), 0);
                assert!(placed.layers.iter().zip(&cfg.attention).filter(|(_, kind)| **kind == GlmNextAttention::Kda)
                    .all(|(layer, _)| layer.mode == LayerMode::HeadSplit));
                let mla: Vec<_> = placed.layers.iter().zip(&cfg.attention).filter(|(_, kind)| **kind == GlmNextAttention::Mla)
                    .map(|(layer, _)| layer.mode).collect();
                if mode == AttentionPlacement::Context {
                    assert_eq!(mla, vec![LayerMode::ContextSplit; 11]);
                } else {
                    assert_eq!(mla, (0..11).map(|i| LayerMode::Whole { gpu: (i % 2) as u8,
                        ffn: FfnMode::Split }).collect::<Vec<_>>());
                }
                let geometry = facts(&input).unwrap().geometry;
                let pool_bytes: u64 = placed.items.iter().flatten().filter(|item| item.group == "KV pool")
                    .map(|item| item.bytes).sum();
                assert_eq!(pool_bytes, geometry.ranks.iter().map(|rank|
                    (rank.persistent_unit_bytes + rank.pool_metadata_unit_bytes) * ((2 << 20) / 256))
                    .sum::<u64>());
                let runtime = contract(&sampled(&input)).unwrap().placement;
                assert_eq!(placed.items, runtime.items);
                assert_eq!(placed.layers, runtime.layers);
                assert_eq!(placed.pool_tokens, runtime.pool_tokens);
            }
        }
    }

    #[test]
    fn whole_owner_selection_uses_actual_residency_drafter_and_measured_headroom() {
        use crate::families::glm5_flash::resident::resident_weights_placed;
        use crate::plan::testing::{glm5_flash_config, glm5_flash_tensors, write_snapshot};
        // One MLA means the other rank has KDA state but no MLA cache tails.
        let config = glm5_flash_config(2);
        let cfg = GlmNextConfig::from_hf(&config).unwrap();
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &config, &glm5_flash_tensors(&config), None);
        let checkpoint = crate::plan::Checkpoint::open(dir.path()).unwrap();
        for expected in [1, 0] {
            let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
            input.spark_ranks = 0;
            input.index = cache::GlmfIndexCache::Compact;
            input.requested_pool = Some(2 << 20);
            input.graphs = GraphSet::budget(&[64 << 20; 2]);
            input.attention.mode = AttentionPlacement::Layers;
            input.workspace = workspace_fixture(2, 0);
            input.local_workspace = workspace_fixture(2, 0);
            // The ordinary drafter stays on GPU0. Reversing headroom with a real
            // peer workspace demand must reverse ownership, not add a rank knob.
            if expected == 0 { input.local_workspace[1].demand.bytes = 8 << 30; }
            input.resident = resident_weights_placed(&checkpoint, &cfg, cfg.layers, 2,
                input.representation, AttentionPlacement::Layers, 0).unwrap();
            let mut alternate = input.clone();
            alternate.attention.first_owner = 1;
            alternate.resident = resident_weights_placed(&checkpoint, &cfg, cfg.layers, 2,
                input.representation, AttentionPlacement::Layers, 1).unwrap();
            let selected = admitted_case(&input, false, 188, true, Some(&alternate.resident)).unwrap();
            assert_eq!(placed_attention(input.attention, &selected.0).first_owner, expected);
            assert_eq!(placed_attention(input.attention, &selected.0).first_owner, expected);
            assert_eq!(selected.0.pool_tokens, 2 << 20);
            assert_eq!(selected.0.movables, [(MovableId::Drafter, 0)]);
            let mut measured_input = input.clone();
            for (capacity, baseline) in &mut measured_input.gpus {
                let Baseline::Planned { context_bytes, loaded_bytes } = *baseline else { unreachable!() };
                *baseline = Baseline::Measured { free_bytes: *capacity - context_bytes - loaded_bytes };
            }
            let runtime = admitted_case(&measured_input, false, 188, true, Some(&alternate.resident)).unwrap();
            assert_eq!(selected.0, runtime.0);
            let resident = if expected == 0 { &input.resident } else { &alternate.resident };
            for rank in 0..2 {
                let bytes: u64 = selected.0.items[rank].iter().filter(|i| i.group == "target weights")
                    .map(|i| i.bytes).sum();
                assert_eq!(bytes, resident[rank].weights);
            }
        }
    }

    #[test]
    fn embedding_home_equalizes_complete_inventory_and_matches_measured_admission() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4)).unwrap();
        for requested in [None, Some(262_144)] {
            for expected in [0, 1] {
                let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
                input.spark_ranks = 0;
                input.requested_pool = requested;
                input.graphs = GraphSet::budget(&[64 << 20; 2]);
                // Include a cold tower/foreign reservation in the sampled baseline,
                // not a device-index preference. Reverse it to reverse the home.
                let busy = 1 - expected;
                let Baseline::Planned { loaded_bytes, .. } = &mut input.gpus[busy].1 else { unreachable!() };
                *loaded_bytes += 8 << 30;
                let (planned, _, _) = admitted_case(&input, false, 188, true, None).unwrap();
                assert_eq!(embedding_home(&planned), expected as u8);
                assert_eq!(planned.movables, [(MovableId::Drafter, 0)]);
                let bytes = input.resident[0].embedding;
                for rank in 0..2 {
                    let charged: u64 = planned.items[rank].iter().filter(|item|
                        item.category == Category::Embedding).map(|item| item.bytes).sum();
                    assert_eq!(charged, if rank == expected { bytes } else { 0 });
                }
                let mut sampled = input.clone();
                for (capacity, baseline) in &mut sampled.gpus {
                    let Baseline::Planned { context_bytes, loaded_bytes } = *baseline else { unreachable!() };
                    *baseline = Baseline::Measured { free_bytes: *capacity - context_bytes - loaded_bytes };
                }
                let (runtime, _, _) = admitted_case(&sampled, false, 188, true, None).unwrap();
                assert_eq!(runtime, planned);
                input.resident[0].embedding = 0;
                let host = admitted_case(&input, false, 188, true, None).unwrap().0;
                assert_eq!(embedding_home(&host), 0);
                assert!(host.items.iter().flatten().filter(|item| item.category == Category::Embedding)
                    .all(|item| item.bytes == 0));
            }
        }
    }

    #[test]
    fn whole_owner_score_charges_reserved_units_and_original_ties() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        input.spark_ranks = 0;
        input.attention = GlmfAttention { mode: AttentionPlacement::Layers, first_owner: 1, ..Default::default() };
        input.index = cache::GlmfIndexCache::Compact;
        input.requested_pool = Some(262_144);
        input.graphs = GraphSet::budget(&[64 << 20; 2]);
        let a = admitted_case(&input, false, 188, true, Some(&input.resident)).unwrap();
        let b = admitted_case(&input, false, 188, true, Some(&input.resident)).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.0.pool_tokens, 262_144);
        assert_eq!(a.0.onboard_layers, input.experts.len());
    }

    #[test]
    fn owner_inventory_is_optional_for_heads_and_context_and_invalid_elsewhere() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        for mode in [AttentionPlacement::Heads, AttentionPlacement::Context, AttentionPlacement::Layers] {
            let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
            input.attention.mode = mode;
            input.index = if mode == AttentionPlacement::Heads { cache::GlmfIndexCache::Keys }
                else { cache::GlmfIndexCache::Compact };
            let original = admitted_case(&input, false, 188, true, None).unwrap();
            assert_eq!(placed_attention(input.attention, &original.0), input.attention);
            if mode != AttentionPlacement::Layers {
                assert!(matches!(admitted_case(&input, false, 188, true, Some(&input.resident)),
                    Err(planner::Refusal::InvalidFact(_))));
            }
            for count in [0, 1, 3] {
                let malformed = vec![input.resident[0].clone(); count];
                assert!(matches!(admitted_case(&input, false, 188, true, Some(&malformed)),
                    Err(planner::Refusal::InvalidFact(_))));
            }
        }
    }

    #[test]
    fn whole_two_rtx_planner_and_runtime_charge_every_router_replica_once() {
        use crate::families::glm5_flash::resident::{resident_weights_placed, router_replica_bytes};
        use crate::plan::testing::{glm5_flash_config, glm5_flash_tensors, write_snapshot};
        let config = glm5_flash_config(6);
        let cfg = GlmNextConfig::from_hf(&config).unwrap();
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &config, &glm5_flash_tensors(&config), None);
        let checkpoint = crate::plan::Checkpoint::open(dir.path()).unwrap();
        // Independent upload arithmetic: each routed layer replicates a BF16 gate
        // matrix and an FP32 correction vector, each allocation floored at 256 bytes.
        let routed = cfg.dense.iter().filter(|&&dense| !dense).count() as u64;
        let routers = routed * ((cfg.experts * cfg.hidden * 2) as u64).max(256)
            + routed * ((cfg.experts * 4) as u64).max(256);
        assert_eq!(router_replica_bytes(&checkpoint, &cfg, cfg.layers).unwrap(), routers);
        assert!(routers > 0);
        for first_owner in [0, 1] {
            let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
            input.spark_ranks = 0;
            input.attention = GlmfAttention { mode: AttentionPlacement::Layers, first_owner,
                ..Default::default() };
            input.index = cache::GlmfIndexCache::Compact;
            input.resident = resident_weights_placed(&checkpoint, &cfg, cfg.layers, 2,
                input.representation, AttentionPlacement::Layers, first_owner as usize).unwrap();
            let placed = contract(&input).unwrap().placement;
            for rank in 0..2 {
                let charged: u64 = placed.items[rank].iter()
                    .filter(|item| item.group == "target weights").map(|item| item.bytes).sum();
                assert_eq!(charged, input.resident[rank].weights);
                assert!(charged >= routers);
                assert!(!placed.items[rank].iter().any(|item| item.group == "router replica"),
                    "routers are part of target weights, never an extra demand");
            }
            let admitted = contract(&sampled(&input)).unwrap().placement;
            assert_eq!(placed.items, admitted.items);
            assert_eq!(placed.layers, admitted.layers);
            assert_eq!(placed.pool_tokens, admitted.pool_tokens);
            // Removing only router headers from the inventory changes each rank's
            // charge by precisely the actual replica bytes, independent of owner.
            let mut no_router = crate::plan::Checkpoint::open(dir.path()).unwrap();
            for tensor in &mut no_router.tensors {
                if tensor.meta.name.contains(".mlp.gate.") {
                    tensor.meta.byte_length = 0;
                    tensor.meta.shape = vec![0];
                }
            }
            let floor = routed * 2 * 256;
            let without = resident_weights_placed(&no_router, &cfg, cfg.layers, 2,
                input.representation, AttentionPlacement::Layers, first_owner as usize).unwrap();
            for rank in 0..2 {
                assert_eq!(input.resident[rank].weights - without[rank].weights, routers - floor);
            }
        }
    }

    #[test]
    fn candidate_context_inventory_matches_exchange_and_allocation_floor() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4)).unwrap();
        let context = GlmfAttention { mode: AttentionPlacement::Context, topk_scratch: 123456,
            partial_scratch: 654321, ..Default::default() };
        let buffers = context_buffers(&cfg, 1 << 20, 4096, 3, context);
        assert_eq!((buffers.query_row_bytes, buffers.partial_row_bytes, buffers.candidate_row_bytes),
            (32768, 32896, 4096));
        assert_eq!(buffers.lanes, 1);
        let workspace = context_workspace(&cfg, 1 << 20, 4096, 3, context);
        assert_eq!(workspace.len(), 11);
        assert_eq!(workspace[workspace.len() - 2].1, 3 * 3 * 9 * 8);
        assert_eq!(workspace.last().unwrap().1, (1 << 20) / 256 * 4);
        assert_eq!(buffers.workspace_bytes, workspace.iter().map(|&(_, bytes)| bytes.max(256)).sum::<u64>());
        let demands = buffers.demands().unwrap();
        for gpu in 0..2 {
            let bytes = |group: &str| demands.iter().find(|d| d.gpu == gpu && d.group == group).unwrap().bytes;
            assert_eq!(bytes("context staging"), 2 * (1 << 20) / 256 * (256 * 528 + 64 * 132));
            assert_eq!(bytes("context exchange"), ContextExchangeLayout::new(&buffers).bytes().unwrap());
            assert_eq!(bytes("context workspace"), buffers.workspace_bytes);
        }
    }

    #[test]
    fn glm5_flash_resolves_auto_to_context_or_layers_without_replicating_kv() {
        let programs = Some(GlmfAttention { mode: AttentionPlacement::Context, topk_scratch: 7, partial_scratch: 9,
            ..Default::default() });
        use AttentionPlacement::{Context, Heads, Layers};
        assert_eq!(resolve_attention(None, 2, programs).unwrap().mode, Context);
        assert_eq!(resolve_attention(None, 1, programs).unwrap().mode, Heads);
        assert_eq!(resolve_attention(Some(Heads), 2, programs).unwrap().mode, Heads);
        assert_eq!(resolve_attention(None, 1, None).unwrap().mode, Heads);
        assert_eq!(resolve_attention(Some(Heads), 2, None).unwrap().mode, Heads);
        assert_eq!(resolve_attention(Some(Context), 2, programs).unwrap().topk_scratch, 7);
        assert_eq!(resolve_attention(None, 2, None).unwrap().mode, Layers);
        // Only explicit context fails without its programs.
        for (requested, gpus, programs) in [(Some(Context), 2, None)] {
            let error = resolve_attention(requested, gpus, programs).unwrap_err();
            assert!(matches!(error, PlacementError::MissingContextPrograms { family: "glm5_flash", .. }),
                "{requested:?}: {error}");
            assert!(error.to_string().contains("glmf_dsa_candidate_merge"), "{error}");
        }
        for (mode, gpus, programs) in [(Context, 1, programs), (Layers, 1, programs)] {
            assert!(matches!(resolve_attention(Some(mode), gpus, programs), Err(PlacementError::AttentionPlacement { .. })));
        }
        let cfg_heads = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        let mut reference = None;
        for programs in [programs, None] {
            let mut i = inputs(&cfg_heads, 2, inventory::PRO_TOTAL_BYTES);
            i.attention = resolve_attention(Some(Heads), 2, programs).unwrap();
            let placed = contract(&i).map(|a| a.placement).unwrap();
            assert_eq!(placed.attention_placement, Heads);
            assert!(placed.layers.iter().all(|layer| layer.mode == LayerMode::HeadSplit));
            if let Some(expected) = &reference { assert_eq!(&placed, expected); }
            reference = Some(placed);
            i.attention = resolve_attention(None, 2, programs).unwrap();
            i.index = if i.attention.mode == Context { cache::GlmfIndexCache::Compact }
                else { cache::GlmfIndexCache::Keys };
            let placed = contract(&i).map(|a| a.placement).unwrap();
            assert_eq!(placed.attention_placement, if programs.is_some() { Context } else { Layers });
        }
        // Explicit layers does not need the context programs.
        for programs in [programs, None] {
            assert_eq!(resolve_attention(Some(Layers), 2, programs).unwrap().mode, Layers);
        }
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        let geometry = facts(&input).unwrap();
        assert!(geometry.layers.iter().filter(|l| l.kind == AttentionClass::Kda).all(|l| l.kv_unit.unit_bytes_context.is_none()));
        assert!(geometry.layers.iter().filter(|l| l.kind == AttentionClass::Dsa).all(|l| l.kv_unit.unit_bytes_context.is_some()));
        assert_eq!(contract(&input).unwrap().placement.attention_placement, Heads);
        input.attention.mode = Layers;
        input.index = cache::GlmfIndexCache::Compact;
        let placed = contract(&input).unwrap().placement;
        assert_eq!(placed.attention_placement, Layers);
        assert!(placed.layers.iter().zip(&cfg.attention).all(|(layer, kind)|
            if *kind == GlmNextAttention::Kda { layer.mode == LayerMode::HeadSplit } else {
                matches!(layer.mode, LayerMode::Whole { ffn: FfnMode::Split, .. })
            }));
    }

    #[test]
    fn explicit_layers_keep_the_context_default_and_single_rank_owner_executor() {
        let executor = super::super::GLM5_FLASH;
        assert_eq!(executor.attention_default(), AttentionPlacement::Context);
        assert_eq!(executor.check_attention(Some(AttentionPlacement::Layers), 2, true).unwrap(), AttentionPlacement::Layers);
        assert!(executor.check_attention(Some(AttentionPlacement::Layers), 1, false).is_err());
        for gpu in 0..2 {
            assert!(executor.runs(LayerMode::Whole { gpu, ffn: FfnMode::Split }));
        }
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        let one = inputs(&cfg, 1, inventory::PRO_TOTAL_BYTES);
        let placed = contract(&one).map(|a| a.placement).unwrap();
        assert!(placed.layers.iter().all(|layer| layer.mode == LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }));
        executor.check(&placed).unwrap();
    }

    #[test]
    fn glm5_flash_split_without_tp2_cannot_place_whole_experts() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        for expert in &mut input.experts { expert.tp2 = false; }
        let placed = contract(&input).map(|a| a.placement).unwrap();
        assert_eq!(placed.onboard_layers, 0);
        input.onboard = Onboard::Layers(1);
        assert!(contract(&input).map(|a| a.placement).is_err());
    }
}
