//! GLM Flash admission before weights: planner and runtime differ only in baseline.
use crate::families::glm5_flash::{GlmNextAttention, GlmNextConfig, resident::{GlmfRepresentation, GlmfResidentRank}};
use crate::serving_capacity::{self as cache, glmf_graphs};
use crate::placement::*;
use cuteafd_core::memory_layout::{Basis, Category};

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

/// The MLA placement a GLM Flash engine on `gpus` GPUs serves: `requested` (None: auto), checked
/// against the build's `context` programs. Auto is `context` on two GPUs, as `ExecutorModes::check_attention`
/// records for this family; without the context programs that placement cannot run, so a chosen
/// `context` fails closed naming the programs rather than silently serving heads with replicated
/// KV. One GPU and an explicit `heads` stay heads; `layers` is explicit on two GPUs only.
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
        None if gpus == 2 => programs.map(context).map_or_else(missing, Ok),
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
    mode: u8) -> Result<(u64, Vec<u64>), crate::families::glm5::draft_representation::GlmDraftStorageError> {
    use crate::families::glm5::draft_representation::{draft_resident_bytes_with_mode, GlmDraftStorageError};
    if ![1, 2].contains(&sms.len()) {
        return Err(GlmDraftStorageError::Unsupported("GLM Flash draft candidate inventory needs one or two GPUs"));
    }
    let mut owned = None;
    let mut scratch = Vec::with_capacity(sms.len());
    for &sms in sms {
        let (bytes, workspace) = draft_resident_bytes_with_mode(config, slots, sequences, sms, representation, mode)?;
        if owned.is_some_and(|previous| previous != bytes) {
            return Err(GlmDraftStorageError::Unsupported("GLM Flash owned draft bytes depend on SM count"));
        }
        bytes.checked_add(workspace).ok_or(GlmDraftStorageError::Overflow)?;
        owned = Some(bytes);
        scratch.push(workspace);
    }
    Ok((owned.unwrap(), scratch))
}

/// GPU1 DFlash bridge inventory, conditional on installing its executor.
/// `rows` is the drafter's full admitted batch (sequences * block), not a
/// target-head launch chunk. GPU1's existing draft destinations are already
/// in the movable; publisher streams/modules belong to the sampled baseline.
pub fn remote_dflash_bridge_demands(hidden: u64, vocab: u64, taps: u64, rows: u64)
    -> Result<MovableDemands, PlacementError> {
    if hidden == 0 || vocab == 0 || taps == 0 || rows == 0 || rows > i32::MAX as u64 {
        return Err(PlacementError::Inventory("GLM Flash remote DFlash bridge geometry"));
    }
    let mul = |a: u64, b: u64| a.checked_mul(b)
        .ok_or(PlacementError::Overflow("GLM Flash remote DFlash bridge"));
    let hidden_rows = mul(mul(rows, hidden)?, 2)?.max(256);
    let bytes = [
        ("DFlash bridge taps", mul(mul(mul(2048, taps)?, hidden)?, 2)?.max(256)),
        ("DFlash bridge input ids", mul(rows, 4)?.max(256)),
        ("DFlash bridge gathered input", hidden_rows),
        ("DFlash bridge normalized hidden", hidden_rows),
        ("DFlash bridge logits", mul(mul(rows, vocab)?, 4)?.max(256)),
        ("DFlash bridge head scratch", 4 << 20),
    ];
    let mut demands: Vec<_> = bytes.into_iter().map(|(group, bytes)|
        Demand::new(0, Category::Drafter, group, bytes, Basis::Formula)).collect();
    // Four zero-size receive slots still own 256B each; controls and the
    // abort word own 256B each. No duplicate payload destination is allocated.
    for gpu in 0..2 {
        demands.push(Demand::new(gpu, Category::Transport, "DFlash bridge exchange",
            4 * 256 + 256 + 256, Basis::Formula));
    }
    for gpu in 0..2 {
        demands.iter().filter(|d| d.gpu == gpu).try_fold(0u64, |sum, d|
            sum.checked_add(d.bytes).ok_or(PlacementError::Overflow("GLM Flash remote DFlash bridge")))?;
    }
    Ok(MovableDemands { placement_gpu: 1, demands })
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

/// Explicit-count request (Spark-only validation profile for automatic policies).
/// Automatic callers use `solve_working_set_with_graphs`, not the generic solver directly.
pub fn request(i: &GlmfInputs<'_>) -> Result<PlacementRequest, PlacementError> {
    let routed = i.cfg.dense.iter().take(i.layers).filter(|&&dense| !dense).count();
    let n = if i.spark_ranks == 0 { routed } else { i.onboard.layers(routed).unwrap_or(0) };
    request_for_working_set(i, &GlmfWorkingSet::candidate(i, n)?)
}

pub fn request_for_working_set(i: &GlmfInputs<'_>, working: &GlmfWorkingSet) -> Result<PlacementRequest, PlacementError> {
    let ranks = i.gpus.len();
    if ![1, 2].contains(&ranks) || i.layers == 0 || i.layers > i.cfg.layers
        || i.resident.len() != ranks || i.graphs.ranks.len() != ranks
        || i.pending_code.len() != ranks || ![2, 4].contains(&i.partial_bytes) {
        return Err(PlacementError::Inventory("GLM Flash rank/layer inventory"));
    }
    for profile in [&i.workspace, &i.local_workspace] {
        if profile.iter().any(|row| row.demand.gpu as usize >= ranks) || (0..ranks).any(|rank|
            !profile.iter().any(|row| row.demand.gpu as usize == rank && row.component == GlmfComponent::Always)) {
            return Err(PlacementError::Inventory("GLM Flash step buffer inventory"));
        }
    }
    if *working != GlmfWorkingSet::candidate(i, working.rtx_layers.len())? {
        return Err(PlacementError::Inventory("GLM Flash selected working set inventory"));
    }
    if i.drafter_scratch.len() != ranks || (i.drafter_bytes == 0 && i.drafter_scratch.iter().any(|&bytes| bytes != 0)) {
        return Err(PlacementError::Inventory("GLM Flash draft scratch inventory"));
    }
    for &scratch in &i.drafter_scratch {
        i.drafter_bytes.checked_add(scratch).ok_or(PlacementError::Overflow("GLM Flash draft inventory"))?;
    }
    if i.drafter_bridge.as_ref().is_some_and(|bridge| ranks != 2 || i.drafter_bytes == 0
        || bridge.placement_gpu != 1 || bridge.demands.iter().any(|d| d.gpu > 1)) {
        return Err(PlacementError::Inventory("GLM Flash remote draft candidate"));
    }
    let movables = if i.drafter_bytes == 0 { vec![] } else {
        let conditional = i.drafter_scratch.iter().enumerate().filter_map(|(rank, &bytes)| {
            let mut demands = if rank == 1 { i.drafter_bridge.as_ref().map(|b| b.demands.clone()).unwrap_or_default() }
                else { vec![] };
            if bytes > 0 { demands.push(Demand::new(rank as u8, Category::Workspace,
                "DFlash linear/attention scratch", bytes, Basis::Formula)); }
            (!demands.is_empty()).then_some(MovableDemands { placement_gpu: rank as u8, demands })
        }).collect();
        vec![Movable { id: MovableId::Drafter, parts: vec![Bytes2 { resident: i.drafter_bytes, staging: 0 }],
            allowed: if i.drafter_bridge.is_some() { vec![0, 1] } else { vec![0] },
            expert_arena: false, conditional }]
    };
    if ranks == 2 && i.decode_rows != cache::GLMF_DECODE_ROWS {
        return Err(PlacementError::Inventory("GLM Flash head split needs 64 decode rows"));
    }
    if i.attention.mode == AttentionPlacement::Context && i.pool_marks {
        return Err(PlacementError::Inventory("GLM Flash context prefix marks need an arena"));
    }
    if i.shared_replay && (ranks != 1 || working.spark_ranks == 0) {
        return Err(PlacementError::Inventory("GLM Flash --replay-records shared needs one coordinator GPU and Spark experts (independent of --pool-tokens)"));
    }
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
        fixed.push(GlmfBufferDemand::new(GlmfComponent::Always, gpu, Category::Embedding, "embedding", i.resident[rank].embedding, Basis::Exact));
        fixed.push(GlmfBufferDemand::new(GlmfComponent::Always, gpu, Category::Weights, "target weights", i.resident[rank].weights, Basis::Exact));
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
        fixed.extend(i.graphs.items(rank).into_iter().map(|item|
            GlmfBufferDemand::new(GlmfComponent::Always, gpu, item.category, &item.group, item.bytes, item.basis)));
        if ranks == 2 {
            fixed.push(GlmfBufferDemand::new(GlmfComponent::Always, gpu, Category::Transport, "peer exchange",
                glmf_graphs::peer_exchange_bytes(working.prefill_lanes, i.prefill_rows.max(i.decode_rows),
                    i.cfg.hidden as u64, i.partial_bytes, i.representation.output_shard), Basis::Formula));
            // new_abortable owns a separate 256-byte word on each exchange end.
            // CUDA publisher streams/function loading are driver costs, not payload storage.
            fixed.push(GlmfBufferDemand::new(GlmfComponent::Always, gpu, Category::Transport,
                "peer abort word", 256, Basis::Formula));
            fixed.push(GlmfBufferDemand::new(GlmfComponent::RtxExperts, gpu, Category::Transport,
                "route abort word", 256, Basis::Formula));
            let slots = 2 * working.prefill_lanes;
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
                cache::glmf_spark_intake_bytes(working.prefill_lanes, working.spark_ranks as u64,
                    cache::glmf_expert_rows(i.prefill_rows, i.decode_rows), i.cfg.hidden as u64), Basis::Formula));
            fixed.push(GlmfBufferDemand::new(GlmfComponent::Always, gpu, Category::Workspace, "probe prefill logits", i.full_prefill_logits, Basis::Formula));
        }
    }
    let workspace = if working.spark_ranks == 0 { &i.local_workspace } else { &i.workspace };
    fixed.extend(workspace.iter().cloned());
    let fixed = working.demands(&fixed).cloned().collect();
    Ok(PlacementRequest {
        // The family resolves the mode before admission (its index cache depends on it: compact
        // under context, keys under heads), so two GPUs pin it.
        attention_placement: (ranks == 2).then_some(i.attention.mode),
        context_buffers: context_buffers(i.cfg, i.max_context, i.prefill_rows, working.prefill_lanes, i.attention),
        layers_first_gpu: i.attention.first_owner,
        inventory: Inventory { gpus: i.gpus.iter().map(|&(capacity_bytes, baseline)| GpuBudget {
            capacity_bytes, headroom_bytes: i.headroom_bytes, baseline }).collect(), spark_ranks: working.spark_ranks, peer_access: ranks == 2 },
        pool: PoolPolicy::resolve(&i.gpus.iter().map(|g| g.0).collect::<Vec<_>>(), i.max_context,
            i.requested_pool, geometry.logical_unit_rows, working.spark_ranks == 0),
        pool_bounded: geometry.ranks.iter().enumerate().filter_map(|(rank, g)| {
            let bytes = g.glmf_pool_mark_backing_unit_bytes(i.pool_marks);
            (bytes > 0).then(|| BoundedDemand { gpu: rank as u8, category: Category::Prefix,
                group: format!("KDA-only pool marks ({bytes} bytes/unit)"), bytes_per_unit: bytes,
                offset_units: 0, cap_units: u64::MAX, basis: Basis::Formula })
        }).collect(),
        layers, pool_overhead: geometry.ranks.iter().map(|g| g.pool_metadata_unit_bytes).collect(), fixed,
        movables,
        expert_workspace: if working.includes(GlmfComponent::RtxExperts) { i.expert_workspace } else { 0 },
        tp2_workspace: if working.includes(GlmfComponent::RtxExperts) { i.tp2_workspace } else { [0; 2] },
        onboard: if i.spark_ranks == 0 { i.onboard } else { Onboard::Layers(working.rtx_layers.len()) },
        minimum_expert_layers: 0,
        expert_gpus: usize::from(ranks == 1 || i.experts.iter().all(|cost| cost.tp2)),
        policy: LayerPolicy { default: if ranks == 2 { vec![LayerMode::HeadSplit] } else {
            vec![LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }] }, by_kind: vec![],
            ownership: LayerOwnership::Alternating, ..LayerPolicy::default() },
        hops: HopSpec { storage: crate::placement::HopStorage::ReceiveSlots, row_bytes: 4 * i.cfg.hidden as u64 * 2, rows: i.prefill_rows.max(i.decode_rows),
            lanes: working.prefill_lanes, entry_gpu: 0, head_gpu: 0, terminal: if ranks == 2 { TerminalHome::Replicated } else { TerminalHome::AttentionOwner } }, executor: super::GLM5_FLASH,
    })
}

/// Exact header-only layer residency for the EXL3 and fp8moe loaders.
pub fn expert_costs(catalog: &crate::OfficialV41Catalog, tp2: bool) -> anyhow::Result<Vec<ExpertCost>> {
    use crate::formats::fp8_experts::{Fp8Projection, Slicing};
    let shape = catalog.routed_experts();
    let cost = |layer, tp, rank| -> anyhow::Result<Bytes2> {
        let resident = if let Some(exl3) = catalog.exl3() {
            exl3.residency(crate::V41Exl3Layer::Backbone(layer), tp, rank)?.device_arena_layout()?.1 as u64
        } else if let Some(fp8) = catalog.fp8() {
            Fp8Projection::ALL.iter().try_fold(0u64, |sum, &p| -> anyhow::Result<u64> {
                let (w, _) = fp8.slice_bytes_with(p, tp, rank, Slicing::Blocks(128))?;
                Ok(sum + (shape.experts * w).max(256) as u64 + fp8.scale_region_bytes_with(p, tp, rank, Slicing::Blocks(128))?.max(256) as u64)
            })?
        } else { anyhow::bail!("GLM Flash routed experts need EXL3 or fp8moe") };
        Ok(Bytes2 { resident, staging: 0 })
    };
    (shape.first_layer..shape.layers).map(|layer| Ok(ExpertCost { whole: cost(layer, 1, 0)?,
        half: if tp2 { [cost(layer, 2, 0)?, cost(layer, 2, 1)?] } else { [Bytes2::default(); 2] },
        tp2, spark_ok: true })).collect()
}

/// Match the runtime's activation override: only `a16` disables native A4.
pub fn nvfp4_a4() -> bool {
    std::env::var("CUTEAFD_NVFP4_ACTIVATIONS").as_deref() != Ok("a16")
}

/// Exact compiled package arenas; missing packages fail closed, never become zero scratch.
pub fn expert_workspace(catalog: &crate::OfficialV41Catalog, manifest: Option<&std::path::Path>, rows: u64,
    tp: usize) -> anyhow::Result<u64> {
    expert_workspace_selected(catalog, manifest, rows, tp, nvfp4_a4())
}

/// Explicit activation selection for deterministic admission and package tests.
pub fn expert_workspace_selected(catalog: &crate::OfficialV41Catalog, manifest: Option<&std::path::Path>, rows: u64,
    tp: usize, a4: bool) -> anyhow::Result<u64> {
    anyhow::ensure!([1, 2].contains(&tp), "GLM Flash RTX experts need TP1 or TP2");
    anyhow::ensure!((1..=4096).contains(&rows), "GLM Flash expert capacity must be 1..=4096 rows");
    let shape = catalog.routed_experts();
    let lib = inventory::image_lib(manifest);
    if let Some(exl3) = catalog.exl3() {
        let tiers = exl3.decoder_tiers().iter().map(usize::to_string).collect::<String>();
        let stem = format!("exl3-glmf-k{tiers}");
        let parent = manifest.unwrap_or(std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json")).parent().unwrap();
        let root = [parent.join("exl3").join(&stem), lib.join("exl3").join(&stem)]
            .into_iter().find(|p| p.join(format!("rtx-tp{tp}")).is_dir())
            .ok_or_else(|| anyhow::anyhow!("missing {stem}/rtx-tp{tp} package"))?;
        let capacities = cuteafd_core::Exl3RowPolicy::for_geometry(cuteafd_core::ExpertGeometry::GLM5_FLASH)
            .capacities(usize::try_from(rows)?)?;
        let manifests = capacities.into_iter().map(|capacity| -> anyhow::Result<serde_json::Value> {
            let value: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join(format!("rtx-tp{tp}/m{capacity}/v41_exl3.json")))?)?;
            anyhow::ensure!(value["hidden"].as_u64() == Some(shape.hidden as u64)
                && value["intermediate"].as_u64() == Some((shape.intermediate / tp) as u64)
                && value["experts"].as_u64() == Some(shape.experts as u64), "GLM Flash EXL3 package geometry mismatch");
            Ok(value)
        }).collect::<anyhow::Result<Vec<_>>>()?;
        return Ok(cache::exl3_workspace_bytes(&manifests, true)? + rows.max(1) * shape.hidden as u64 * if tp == 2 { 4 } else { 2 });
    }
    let fp8 = catalog.fp8().ok_or_else(|| anyhow::anyhow!("GLM Flash fp8moe catalog missing"))?;
    let a4_name = "fp8-glmf-nvfp4a4";
    let name = if a4 && fp8.format() == crate::formats::fp8_experts::ExpertFormat::Nvfp4
        && lib.join("fp8").join(a4_name).join(format!("tp{tp}")).is_dir() {
        a4_name.to_string()
    } else { format!("fp8-glmf{}", fp8.format().package_suffix()) };
    let value: serde_json::Value = serde_json::from_slice(&std::fs::read(lib.join("fp8").join(&name).join("manifest.json"))?)?;
    let scratch = inventory::fp8moe_scratch_bytes(&value, &format!("tp{tp}"), rows)
        .ok_or_else(|| anyhow::anyhow!("missing GLM Flash tp{tp} scratch for {rows} rows"))?;
    let bf16 = lib.join("fp8").join(format!("{name}-bf16"));
    let sibling = if bf16.join(format!("tp{tp}")).is_dir() {
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(bf16.join("manifest.json"))?)?;
        inventory::fp8moe_scratch_bytes(&manifest, &format!("tp{tp}"), rows)
            .ok_or_else(|| anyhow::anyhow!("missing BF16-input GLM Flash scratch"))?
    } else { 0 };
    Ok(scratch.max(sibling) + if tp == 2 { rows * shape.hidden as u64 * 2 } else { 0 })
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
        let prefill_lanes = i.prefill_lanes;
        if prefill_lanes == 0 || (i.attention.mode == AttentionPlacement::Layers
            && super::super::validate_layer_pipeline_lanes(prefill_lanes, prefill_lanes as usize).is_err()) {
            return Err(PlacementError::Inventory("GLM Flash layers executor needs one or two admitted prefill lanes"));
        }
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

pub fn solve_with_graphs(i: &GlmfInputs<'_>, target: u64, automatic_context: bool,
    sms: usize, rtx_experts: bool) -> Result<(Placement, GraphSet), PlacementError> {
    solve_working_set_with_graphs(i, target, automatic_context, sms, rtx_experts)
        .map(|(placed, graphs, _)| (placed, graphs))
}

/// Choose Whole MLA ownership with the same admission used for expert homes and
/// movable drafts. The alternate inventory changes only the first MLA owner;
/// ties retain the original owner instead of depending on enumeration order.
pub fn solve_attention_working_set_with_graphs(i: &GlmfInputs<'_>,
    alternate_resident: Option<&[GlmfResidentRank]>, target: u64, automatic_context: bool,
    sms: usize, rtx_experts: bool) -> Result<(Placement, GraphSet, GlmfWorkingSet), PlacementError> {
    let Some(resident) = alternate_resident else {
        return solve_working_set_with_graphs(i, target, automatic_context, sms, rtx_experts);
    };
    if i.attention.mode != AttentionPlacement::Layers || i.gpus.len() != 2
        || i.attention.first_owner > 1 || i.resident.len() != 2 || resident.len() != 2 {
        return Err(PlacementError::Inventory("GLM Flash alternate MLA ownership inventory"));
    }
    let original = solve_working_set_with_graphs(i, target, automatic_context, sms, rtx_experts);
    let mut alternate = i.clone();
    alternate.attention.first_owner ^= 1;
    alternate.resident = resident.to_vec();
    let candidate = solve_working_set_with_graphs(&alternate, target, automatic_context, sms, rtx_experts);
    select_attention_working_set(i.onboard, original, candidate)
}

fn attention_owner_score(onboard: Onboard, placed: &Placement) -> Result<((u64, u64), u64), PlacementError> {
    let policy = match onboard {
        Onboard::ExpertsFirst { .. } => (placed.onboard_layers as u64, placed.pool_tokens),
        _ => (placed.pool_tokens, placed.onboard_layers as u64),
    };
    // GLM Flash admission uses 256-token pool units, including bounded demands'
    // private reserved units. Match BoundedDemand::bytes rather than max_units.
    let units = placed.pool_tokens.div_ceil(256);
    let headroom = placed.pool_limits.0.iter().map(|rank| {
        let used = rank.demands.iter().try_fold(0u64, |used, demand| {
            units.saturating_add(demand.offset_units).min(demand.cap_units)
                .checked_mul(demand.bytes_per_unit).and_then(|bytes| used.checked_add(bytes))
                .ok_or(PlacementError::Overflow("GLM Flash owner pool headroom"))
        })?;
        rank.available_bytes.checked_sub(used)
            .ok_or(PlacementError::Inventory("GLM Flash owner pool demands exceed admission"))
    }).collect::<Result<Vec<_>, PlacementError>>()?.into_iter().min().unwrap_or(0);
    Ok((policy, headroom))
}

fn select_attention_working_set(onboard: Onboard,
    original: Result<(Placement, GraphSet, GlmfWorkingSet), PlacementError>,
    candidate: Result<(Placement, GraphSet, GlmfWorkingSet), PlacementError>)
    -> Result<(Placement, GraphSet, GlmfWorkingSet), PlacementError> {
    match (original, candidate) {
        (Ok(original), Ok(candidate)) => if attention_owner_score(onboard, &candidate.0)?
            > attention_owner_score(onboard, &original.0)? {
            Ok(candidate)
        } else { Ok(original) },
        (Ok(original), Err(_)) => Ok(original),
        (Err(_), Ok(candidate)) => Ok(candidate),
        (Err(error), Err(_)) => Err(error),
    }
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

/// Evaluate every executable RTX prefix/Spark suffix with its own fixed costs.
/// Spark presence never feeds back into a later solve: the empty set is one
/// closed-form candidate. Graph-geometry shrinking stays local to each candidate.
pub fn solve_working_set_with_graphs(i: &GlmfInputs<'_>, target: u64, automatic_context: bool,
    sms: usize, rtx_experts: bool) -> Result<(Placement, GraphSet, GlmfWorkingSet), PlacementError> {
    request(i)?;
    let mut graph_cache = std::collections::BTreeMap::new();
    if i.spark_ranks == 0 {
        let working = GlmfWorkingSet::candidate(i, i.experts.len())?;
        let (placed, graphs) = solve_profile_with_graphs(i, target, automatic_context, sms, rtx_experts,
            None, None, &working, &mut graph_cache)?;
        working.validate(&placed)?;
        return Ok((placed, graphs, working));
    }
    let routed: Vec<_> = (0..i.layers).filter(|&layer| !i.cfg.dense[layer]).collect();
    let counts: Vec<_> = match i.onboard.layers(routed.len()) {
        Some(n) => vec![n],
        None if !rtx_experts => vec![0],
        None => (0..=routed.len()).collect(),
    };
    let cap = i.onboard.layers(routed.len()).is_none().then_some(target.max(i.max_context));
    let mut best: Option<((u64, u64), Placement, GraphSet, GlmfWorkingSet)> = None;
    let mut refusal = None;
    for n in counts {
        let working = GlmfWorkingSet::candidate(i, n)?;
        let mut branch = i.clone();
        branch.onboard = Onboard::Layers(n);
        let floor = match i.onboard {
            Onboard::Auto if working.spark_ranks > 0 && i.requested_pool.is_none() => Some(0),
            Onboard::ExpertsFirst { pool_floor } => Some(i.requested_pool.unwrap_or_else(||
                pool_floor.max(i.max_context).max(if working.spark_ranks == 0 { crate::placement::pool::AGENTIC_FLOOR_TOKENS } else { 0 }))),
            _ => None,
        };
        match solve_profile_with_graphs(&branch, target, automatic_context, sms, rtx_experts,
            cap, floor, &working, &mut graph_cache) {
            Ok((placed, graphs)) => {
                working.validate(&placed)?;
                let score = match i.onboard {
                    Onboard::ExpertsFirst { .. } => (n as u64, placed.pool_tokens),
                    _ => (placed.pool_tokens, n as u64),
                };
                if best.as_ref().is_none_or(|(previous, ..)| score > *previous) {
                    best = Some((score, placed, graphs, working));
                }
            }
            Err(error) => { if refusal.is_none() { refusal = Some(error); } }
        }
    }
    if let Some((_, placed, graphs, working)) = best { return Ok((placed, graphs, working)); }
    // Preserve the legacy experts-first fallback when the compiled context does
    // not fit but its explicitly chosen lower floor does, without onboarded layers.
    if let Onboard::ExpertsFirst { pool_floor } = i.onboard {
        if i.requested_pool.is_none() && pool_floor < i.max_context {
            let mut branch = i.clone();
            branch.onboard = Onboard::Layers(0);
            let working = GlmfWorkingSet::candidate(i, 0)?;
            let (placed, graphs) = solve_profile_with_graphs(&branch, target, automatic_context, sms,
                rtx_experts, cap, Some(pool_floor), &working, &mut graph_cache)?;
            working.validate(&placed)?;
            return Ok((placed, graphs, working));
        }
    }
    Err(refusal.unwrap_or(PlacementError::Inventory("GLM Flash has no executable Spark working set")))
}

/// Rebuild startup graphs at a fixed expert candidate's admitted pool. The
/// optional cap preserves pool-first/experts-first ceilings while exact onboard
/// counts retain their existing fill-to-capacity policy.
fn solve_profile_with_graphs(i: &GlmfInputs<'_>, target: u64, automatic_context: bool,
    sms: usize, rtx_experts: bool, cap: Option<u64>, floor: Option<u64>,
    working: &GlmfWorkingSet, graph_cache: &mut std::collections::BTreeMap<(usize, usize), GraphSet>)
    -> Result<(Placement, GraphSet), PlacementError> {
    let configure = |req: &mut PlacementRequest| {
        if !rtx_experts { req.expert_gpus = 0; }
        if let Some(cap) = cap { req.pool.ceiling = req.pool.ceiling.min(cap); }
        if let Some(floor) = floor { req.pool.floor = floor; }
    };
    let mut candidate = i.requested_pool.unwrap_or(target.max(i.max_context));
    if i.requested_pool.is_none() && (i.onboard.layers(i.experts.len()).is_some() || i.spark_ranks == 0) {
        let mut req = request_for_working_set(i, working)?;
        configure(&mut req);
        candidate = solve(&req)?.pool_tokens;
    }
    loop {
        let mut trial = i.clone();
        if i.graphs.lifetime == Lifetime::Startup {
            let context = glmf_graphs::admitted_graph_context(i.max_context as usize, candidate as usize, automatic_context);
            trial.graphs = graph_cache.entry((context, candidate as usize)).or_insert_with(||
                startup_graphs(i.cfg, i.layers, context, candidate as usize, i.sequences as usize,
                    i.speculation, i.gpus.len(), i.decode_rows as usize, sms)).clone();
            if i.local_tp2_exl3 && working.spark_ranks == 0 {
                calibrate_local_tp2_graphs(&mut trial.graphs);
            }
        }
        let mut req = request_for_working_set(&trial, working)?;
        req.pool.target = req.pool.target.max(target).min(candidate);
        req.pool.ceiling = candidate;
        configure(&mut req);
        let placed = solve(&req)?;
        if i.requested_pool.is_some() || i.graphs.lifetime != Lifetime::Startup || placed.pool_tokens >= candidate {
            return Ok((placed, trial.graphs));
        }
        candidate = placed.pool_tokens;
    }
}

#[allow(clippy::too_many_arguments)]
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

    fn workspace_fixture(ranks: usize, bytes: u64) -> Vec<GlmfBufferDemand> {
        (0..ranks).map(|rank| GlmfBufferDemand::new(GlmfComponent::Always,
            rank as u8, Category::Workspace, "steps", bytes, Basis::Formula)).collect()
    }

    fn inputs(cfg: &GlmNextConfig, ranks: usize, capacity: u64) -> GlmfInputs<'_> {
        let context = ArchContext::coordinator(capacity, None).context_bytes;
        GlmfInputs {
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
                half: [Bytes2 { resident: 1 << 30, staging: 0 }; 2], tp2: ranks == 2, spark_ok: true }).collect(),
            expert_workspace: 128 << 20, tp2_workspace: [128 << 20; 2], drafter_bytes: 3 << 30,
            drafter_scratch: vec![0; ranks], drafter_bridge: None, requested_pool: None, onboard: default_onboard(), full_prefill_logits: 0,
        }
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
            let mut req = request(&input).unwrap();
            let geometry = cache::glm_flash_placed_cache_geometry_rows(&cfg, 2, 2, input.index, 4, 64,
                AttentionPlacement::Layers, first as usize).unwrap();
            let bytes = geometry.ranks[1 - first as usize].retained_mark_bytes;
            assert_eq!(req.pool_bounded.len(), 1);
            let backing = &req.pool_bounded[0];
            assert_eq!((backing.gpu, backing.bytes_per_unit), (1 - first, bytes));
            assert!(backing.group.contains(&format!("{bytes} bytes/unit")));
            assert_eq!(backing.bytes(17).unwrap(), 17 * bytes);
            let reserved = req.fixed.iter().find(|d| d.gpu == 1 - first && d.group == "reserved units").unwrap();
            assert_eq!(reserved.bytes, bytes + 4);
            // Keep the owned cache geometry pinned while probing its exact byte limit.
            req.pool.floor = 256;
            req.pool.target = 2 << 20;
            req.pool.ceiling = 2 << 20;
            // Leave exactly 64 units on the KDA-only owner, far below the
            // 8192-unit ceiling. Its next complete state mark must not fit.
            let probe = solve(&req).unwrap();
            let owner = (1 - first) as usize;
            let rank = &probe.pool_limits.0[owner];
            let unit_bytes = rank.demands.iter().map(|d| d.bytes_per_unit).sum::<u64>();
            let budget = 64 * unit_bytes;
            let free = req.inventory.gpus[owner].available() - (rank.available_bytes - budget);
            req.inventory.gpus[owner].baseline = Baseline::Measured { free_bytes: free + req.inventory.gpus[owner].headroom_bytes };
            let bounded = solve(&req).unwrap();
            let limit = &bounded.pool_limits.0[owner];
            assert_eq!(bounded.pool_tokens, 64 * 256);
            assert_eq!(limit.max_units, 64);
            assert_eq!(limit.available_bytes, budget);
            assert!(65 * unit_bytes > limit.available_bytes);
            assert!(bounded.pool_limits.0[first as usize].max_units > 64);
            assert!(limit.demands.iter().any(|d| d.group.contains("KDA-only pool marks")
                && d.bytes_per_unit == bytes));
            req.pool_bounded.clear();
            let unbounded = solve(&req).unwrap();
            assert!(bounded.pool_tokens < unbounded.pool_tokens);
            assert!(bounded.items[(1 - first) as usize].iter().any(|d|
                d.group.contains("KDA-only pool marks") && d.bytes == bytes * bounded.pool_tokens / 256));
            input.pool_marks = false;
            assert!(request(&input).unwrap().pool_bounded.is_empty());
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
        let mut config = crate::plan::testing::glm5_flash_config(45);
        config["text_config"]["layer_types"] = serde_json::json!((0..45).map(|l|
            if l % 4 == 3 { "deepseek_sparse_attention" } else { "linear_attention" }).collect::<Vec<_>>());
        let cfg = GlmNextConfig::from_hf(&config).unwrap();
        for (ranks, capacity) in [(1, 32 << 30), (1, inventory::PRO_TOTAL_BYTES), (2, inventory::PRO_TOTAL_BYTES)] {
            let planned = inputs(&cfg, ranks, capacity);
            let planned_request = request(&planned).unwrap();
            assert_eq!(planned_request.layers.iter().filter(|l| l.kind == AttentionClass::Dsa).count(), 11);
            let target = planned_request.pool.target;
            let (p, admitted_graphs) = solve_with_graphs(&planned, target, true, 188, true).unwrap();
            let mut measured = planned.clone();
            for rank in 0..ranks {
                let context = ArchContext::coordinator(capacity, None).context_bytes;
                let code = loaded_code("glmf", "exl3", ranks == 2, rank as u8).unwrap();
                let sample = context + code.bytes / 3;
                measured.gpus[rank].1 = Baseline::Measured { free_bytes: capacity - sample };
                measured.pending_code[rank] = code.pending(sample, context);
            }
            let (r, measured_graphs) = solve_with_graphs(&measured, target, true, 188, true).unwrap();
            assert_eq!(admitted_graphs.ranks, measured_graphs.ranks);
            assert_eq!(p.pool_tokens, r.pool_tokens);
            assert_eq!(p.onboard_layers, r.onboard_layers);
            assert_eq!(p.layers, r.layers);
            assert_eq!(p.tp2, r.tp2);
            for rank in 0..ranks {
                let bytes = |placed: &Placement, group: &str| placed.items[rank].iter()
                    .filter(|i| i.group == group).map(|i| i.bytes).sum::<u64>();
                for group in ["graphs", inventory::GRAPH_GROWTH, "records"] {
                    assert_eq!(bytes(&p, group), bytes(&r, group));
                }
                assert_eq!(bytes(&p, "graphs"), admitted_graphs.at_ready(rank));
                assert_eq!(bytes(&p, inventory::GRAPH_GROWTH), admitted_graphs.growth(rank));
                let charged = |placed: &Placement| placed.items[rank].iter().map(|i| i.bytes).sum::<u64>();
                let baseline = |input: &GlmfInputs<'_>| capacity - GpuBudget {
                    capacity_bytes: capacity, headroom_bytes: 0, baseline: input.gpus[rank].1 }.available();
                assert_eq!(baseline(&planned) + charged(&p), baseline(&measured) + charged(&r));
            }
            if ranks == 2 {
                assert!(p.tp2.is_some());
                assert!(!p.layers.iter().any(|l| matches!(l.experts, ExpertHome::RtxWhole { .. })));
            }
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
            let (p, pg, working) = solve_working_set_with_graphs(&planned, target, false, 188, true).unwrap();
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
            let (r, rg, runtime_working) = solve_working_set_with_graphs(&measured, target, false, 188, true).unwrap();
            assert_eq!(pg, rg);
            assert_eq!(working, runtime_working);
            assert_eq!((p.pool_tokens, &p.layers, p.tp2), (r.pool_tokens, &r.layers, r.tp2));
            let group = |placed: &Placement, rank: usize, name: &str| placed.items[rank].iter()
                .filter(|d| d.group == name).map(|d| d.bytes).sum::<u64>();
            assert_eq!(group(&p, 0, "Spark intake"), if spark { 268_435_456 } else { 0 });
            for rank in 0..2 {
                assert_eq!(group(&p, rank, "steps"), if spark { planned.workspace[rank].demand.bytes } else { planned.local_workspace[rank].demand.bytes });
                let total = |input: &GlmfInputs<'_>, placed: &Placement| input.gpus[rank].0 - GpuBudget {
                    capacity_bytes: input.gpus[rank].0, headroom_bytes: 0, baseline: input.gpus[rank].1 }.available()
                    + placed.items[rank].iter().map(|d| d.bytes).sum::<u64>();
                assert!(total(&planned, &p).abs_diff(total(&measured, &r)) <= 64 << 20);
                assert_eq!(total(&planned, &p), total(&measured, &r));
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
        let (placed, _, working) = solve_working_set_with_graphs(&input, 2 << 20, false, 188, true).unwrap();
        assert_eq!(placed.onboard_layers, input.experts.len());
        assert!(working.spark_layers.is_empty());
        assert_eq!((working.spark_ranks, working.prefill_lanes), (0, 2));
        assert!(placed.items[0].iter().all(|d| d.group != "Spark intake"));
        input.onboard = Onboard::Layers(1);
        assert!(solve_working_set_with_graphs(&input, 2 << 20, false, 188, true).is_err());
    }

    #[test]
    fn zero_peer_working_set_preserves_existing_request_and_pool_policy() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        input.spark_ranks = 0;
        input.prefill_lanes = 1;
        input.workspace = input.local_workspace.clone();
        input.graphs = GraphSet::budget(&[64 << 20; 2]);
        for pool in [None, Some(2 << 20)] {
            input.requested_pool = pool;
            let prior = solve_profile_with_graphs(&input, 2 << 20, false, 188, true,
                None, None, &GlmfWorkingSet::candidate(&input, input.experts.len()).unwrap(),
                &mut std::collections::BTreeMap::new()).unwrap();
            let (placed, graphs, working) = solve_working_set_with_graphs(&input, 2 << 20, false, 188, true).unwrap();
            assert_eq!(placed, prior.0);
            assert_eq!(graphs, prior.1);
            assert_eq!((working.spark_ranks, working.prefill_lanes), (0, 1));
            assert!(working.spark_layers.is_empty());
        }
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
                let (placed, _, working) = solve_working_set_with_graphs(&input, 2 << 20, false, 188, true).unwrap();
                assert_eq!((working.spark_ranks, working.prefill_lanes), (0, lanes));
                assert_eq!(working.rtx_layers, vec![1, 2, 3]);
                assert!(working.spark_layers.is_empty());
                working.validate(&placed).unwrap();
                let request = request_for_working_set(&input, &working).unwrap();
                assert_eq!(request.hops.lanes, lanes);
                for rank in 0..ranks {
                    let charged = |name: &str| request.fixed.iter().filter(|d| d.gpu as usize == rank && d.group == name)
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
            let input = inputs(&cfg, ranks, inventory::PRO_TOTAL_BYTES);
            for onboard in [0, 1, 3] {
                let working = GlmfWorkingSet::candidate(&input, onboard).unwrap();
                let request = request_for_working_set(&input, &working).unwrap();
                for rank in 0..ranks {
                    let bytes = |group: &str| request.fixed.iter()
                        .filter(|d| d.gpu as usize == rank && d.group == group)
                        .map(|d| { assert_eq!(d.basis, Basis::Formula); d.bytes }).sum::<u64>();
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
            let request = request_for_working_set(&input, &working).unwrap();
            assert_eq!(working.selected_ranks().count(), if n == input.experts.len() { 0 } else { input.spark_ranks });
            let rows = cache::glmf_expert_rows(input.prefill_rows, input.decode_rows) as u32;
            let pinned = working.coordinator_staging(cuteafd_core::ExpertGeometry::GLM5_FLASH, rows, false).unwrap();
            assert_eq!(pinned, if n == input.experts.len() { 0 } else {
                input.prefill_lanes * u64::from(rows) * (input.cfg.hidden as u64 * 33 / 32 + input.cfg.topk as u64 * 8)
            });
            for (component, group) in [(GlmfComponent::RtxExperts, "RTX test buffer"),
                (GlmfComponent::SparkExperts, "Spark test buffer")] {
                assert_eq!(request.fixed.iter().any(|d| d.group == group), working.includes(component));
            }
            let mut forged = working;
            forged.prefill_lanes += 1;
            assert!(matches!(request_for_working_set(&input, &forged),
                Err(PlacementError::Inventory("GLM Flash selected working set inventory"))));
        }
    }

    #[test]
    fn experts_first_local_branch_keeps_agentic_floor_and_explicit_pool() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        input.graphs = GraphSet::budget(&[64 << 20; 2]);
        input.max_context = 32768;
        input.onboard = Onboard::ExpertsFirst { pool_floor: 16384 };
        let (placed, _, working) = solve_working_set_with_graphs(&input, 2 << 20, false, 188, true).unwrap();
        assert!(working.spark_layers.is_empty());
        assert_eq!(request_for_working_set(&input, &working).unwrap().pool.floor,
            crate::placement::pool::AGENTIC_FLOOR_TOKENS);
        assert!(placed.pool_tokens >= crate::placement::pool::AGENTIC_FLOOR_TOKENS);
        input.requested_pool = Some(32768);
        let (placed, _, working) = solve_working_set_with_graphs(&input, 32768, false, 188, true).unwrap();
        assert!(working.spark_layers.is_empty());
        assert_eq!(placed.pool_tokens, 32768);
    }

    #[test]
    fn selected_working_set_preserves_exact_counts_and_experts_first_priority() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(4)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        input.graphs = GraphSet::budget(&[64 << 20; 2]);
        for (onboard, expected) in [(Onboard::Layers(0), 0), (Onboard::Fraction(0.5), 2),
            (Onboard::ExpertsFirst { pool_floor: 262_144 }, 3)] {
            input.onboard = onboard;
            let (placed, _, working) = solve_working_set_with_graphs(&input, 2 << 20, false, 188, true).unwrap();
            assert_eq!(placed.onboard_layers, expected);
            assert_eq!(working.spark_layers.len(), input.experts.len() - expected);
            if matches!(onboard, Onboard::ExpertsFirst { .. }) { assert!(placed.pool_tokens <= 2 << 20); }
        }
    }

    #[test]
    fn spark_auto_keeps_smaller_pool_and_explicit_pool_stays_strict() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(45)).unwrap();
        let mut input = inputs(&cfg, 1, 32 << 30);
        input.max_context = 4 << 20;
        input.graphs = GraphSet::budget(&[64 << 20]);
        let (placed, _, working) = solve_working_set_with_graphs(&input, 4 << 20, true, 170, true).unwrap();
        assert!(placed.pool_tokens < input.max_context);
        assert_eq!(working.spark_ranks, 4);
        input.requested_pool = Some(4 << 20);
        assert!(solve_working_set_with_graphs(&input, 4 << 20, true, 170, true).is_err());
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
                let (placed, _, working) = solve_working_set_with_graphs(&input, 2 << 20, false, 188, true).unwrap();
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
            let (placed, _, working) = solve_working_set_with_graphs(&input, 2 << 20, false, 188, true).unwrap();
            assert_eq!(placed.onboard_layers, 2);
            assert_eq!(working.rtx_layers, [1, 3]);
            assert!(working.spark_layers.is_empty());
        }
    }

    #[test]
    fn draft_inventory_keeps_owned_bytes_independent_and_scratch_candidate_local() {
        use crate::families::glm5::draft_representation::{draft_resident_bytes_with_mode, GlmDraftRepresentation};
        for mode in 0..=2 {
            for representation in [GlmDraftRepresentation::Fp8Only, GlmDraftRepresentation::Bf16Only] {
                let (owned, scratch) = drafter_inventory(&draft_config(), 8, 8, &[188, 170], representation, mode).unwrap();
                for (rank, sms) in [188, 170].into_iter().enumerate() {
                    assert_eq!((owned, scratch[rank]), draft_resident_bytes_with_mode(&draft_config(), 8, 8,
                        sms, representation, mode).unwrap());
                }
                let attention: Vec<_> = [188, 170].into_iter().map(|sms|
                    cache::draft_attention_workspace_bytes(8, 8, 2, 2056, 128, sms).unwrap()).collect();
                if representation == GlmDraftRepresentation::Bf16Only { assert_eq!(scratch, attention); }
                else { assert!(scratch.iter().zip(attention).all(|(&total, arena)| total > arena)); }
                assert!(scratch[0] > scratch[1]);
            }
        }
        let (owned, scratch) = drafter_inventory(&draft_config(), 8, 8, &[188, 170], GlmDraftRepresentation::Fp8Only, 2).unwrap();
        let old_attention = 8 * 8 * 17 * 64 * 130 * 4;
        let attention: Vec<_> = [188, 170].into_iter().map(|sms|
            cache::draft_attention_workspace_bytes(8, 8, 2, 2056, 128, sms).unwrap()).collect();
        assert_eq!(owned, 1_835_700_096 - old_attention);
        // The shared fc arena adds 128 rows * (20480 / 256) K tiles * 4 B.
        let row_max_bytes = 128 * 80 * 4;
        assert_eq!(scratch, [30_491_648 + row_max_bytes + attention[0],
            28_394_496 + row_max_bytes + attention[1]]);
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        input.drafter_bytes = owned;
        input.drafter_scratch = scratch;
        let req = request(&input).unwrap();
        assert_eq!(req.movables[0].allowed, [0]);
        assert_eq!(req.movables[0].conditional.iter().map(|c| (c.placement_gpu, c.demands[0].gpu,
            c.demands[0].category, c.demands[0].bytes, c.demands[0].basis)).collect::<Vec<_>>(),
            [(0, 0, Category::Workspace, 30_491_648 + row_max_bytes + attention[0], Basis::Formula),
             (1, 1, Category::Workspace, 28_394_496 + row_max_bytes + attention[1], Basis::Formula)]);
        let actual = solve(&req).unwrap();
        assert_eq!(actual.movables, [(MovableId::Drafter, 0)]);
        assert_eq!(actual.items[0].iter().filter(|d| d.category == Category::Drafter).map(|d| d.bytes).sum::<u64>(), owned);
        assert_eq!(actual.items[0].iter().find(|d| d.group == "DFlash linear/attention scratch").unwrap().bytes,
            30_491_648 + row_max_bytes + attention[0]);
        assert!(actual.items[1].iter().all(|d| d.category != Category::Drafter && d.group != "DFlash linear/attention scratch"));
        let mut previous = req.clone();
        for candidate in &mut previous.movables[0].conditional {
            candidate.demands[0].bytes -= row_max_bytes;
        }
        let before = solve(&previous).unwrap();
        assert_eq!(actual.pool_tokens, before.pool_tokens);
        assert_eq!(actual.layers, before.layers);
        let charged = |placed: &Placement, rank: usize| placed.items[rank].iter().map(|d| d.bytes).sum::<u64>();
        assert_eq!(charged(&actual, 0) - charged(&before, 0), row_max_bytes);
        assert_eq!(charged(&actual, 1), charged(&before, 1));
        // At a full GPU0 budget, only one rounded KV allocation unit is lost.
        previous.movables[0].allowed = vec![0];
        previous.inventory.gpus[0].headroom_bytes += previous.inventory.gpus[0].available() - charged(&before, 0);
        let mut constrained = previous.clone();
        constrained.movables[0].conditional[0].demands[0].bytes += row_max_bytes;
        let before = solve(&previous).unwrap();
        let after = solve(&constrained).unwrap();
        assert_eq!(before.pool_tokens - after.pool_tokens, req.pool.unit_rows);
        assert_eq!(before.layers, after.layers);
        // Splitting the ledger categories must not change the old GPU0 total or pool/layer decision.
        input.drafter_bytes += input.drafter_scratch[0];
        input.drafter_scratch.fill(0);
        let lumped = solve(&request(&input).unwrap()).unwrap();
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
            assert!(drafter_inventory(&draft_config(), 8, 8, &sms, GlmDraftRepresentation::Fp8Only, 2).is_err());
        }
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        input.drafter_scratch = vec![256];
        assert!(matches!(request(&input), Err(PlacementError::Inventory("GLM Flash draft scratch inventory"))));
        input.drafter_scratch = vec![256, 0];
        input.drafter_bytes = 0;
        assert!(matches!(request(&input), Err(PlacementError::Inventory("GLM Flash draft scratch inventory"))));
        input.drafter_bytes = u64::MAX;
        assert!(matches!(request(&input), Err(PlacementError::Overflow("GLM Flash draft inventory"))));
    }

    #[test]
    fn draft_inventory_keeps_dspark_local_and_category_separated() {
        use crate::families::glm5::draft_representation::GlmDraftRepresentation;
        let config = serde_json::json!({"speculators_model_type":"dspark", "block_size":4, "markov_rank":256,
            "aux_hidden_state_layer_ids":[0,8,16,24,32], "transformer_layer_config":{
                "hidden_size":4096, "intermediate_size":12288, "num_hidden_layers":3, "num_attention_heads":32,
                "num_key_value_heads":8, "head_dim":64, "vocab_size":154880}});
        let (owned, scratch) = drafter_inventory(&config, 8, 8, &[188, 170], GlmDraftRepresentation::Fp8Only, 2).unwrap();
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        input.drafter_bytes = owned;
        input.drafter_scratch = scratch;
        let req = request(&input).unwrap();
        assert_eq!(req.movables[0].allowed, [0]);
        assert!(req.movables[0].conditional.iter().flat_map(|c| &c.demands).all(|d| d.category == Category::Workspace));
        assert_eq!(solve(&req).unwrap().movables, [(MovableId::Drafter, 0)]);
    }

    #[test]
    fn remote_dflash_candidate_combines_scratch_bridge_and_whole_owner() {
        use crate::families::glm5::draft_representation::GlmDraftRepresentation;
        let config = draft_config();
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        let (owned, scratch) = drafter_inventory(&config, 8, 8, &[188, 170], GlmDraftRepresentation::Fp8Only, 2).unwrap();
        input.drafter_bytes = owned;
        input.drafter_scratch = scratch;
        input.drafter_bridge = drafter_bridge_inventory(&config, 8, 8).unwrap();
        let mut req = request(&input).unwrap();
        assert_eq!(req.movables[0].allowed, [0, 1]);
        assert_eq!(req.movables[0].conditional.iter().map(|c| c.placement_gpu).collect::<Vec<_>>(), [0, 1]);
        let candidate = req.movables[0].conditional[1].clone();
        assert_eq!(candidate.demands.iter().filter(|d| d.group == "DFlash linear/attention scratch")
            .map(|d| (d.gpu, d.bytes)).collect::<Vec<_>>(), [(1, input.drafter_scratch[1])]);
        assert_eq!(candidate.demands.iter().filter(|d| d.group == "DFlash bridge logits")
            .map(|d| (d.gpu, d.bytes)).collect::<Vec<_>>(), [(0, 8 * 8 * 154_880 * 4)]);
        for home in [0, 1] {
            req.movables[0].allowed = vec![home];
            let actual = solve(&req).unwrap();
            assert_eq!(actual.movables, [(MovableId::Drafter, home)]);
            let owner = &actual.items[home as usize];
            assert_eq!(owner.iter().filter(|d| d.group == "Drafter").map(|d| d.bytes).sum::<u64>(), owned);
            for rank in 0..2 {
                let bridge = actual.items[rank].iter().filter(|d| d.group.starts_with("DFlash bridge"))
                    .map(|d| d.bytes).sum::<u64>();
                let expected = if home == 1 { candidate.demands.iter().filter(|d| d.gpu as usize == rank
                    && d.group.starts_with("DFlash bridge")).map(|d| d.bytes).sum() } else { 0 };
                assert_eq!(bridge, expected);
                if rank != home as usize { assert!(actual.items[rank].iter().all(|d| d.group != "Drafter")); }
            }
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
                GlmDraftRepresentation::Fp8Only, 2).unwrap();
            planned.drafter_bytes = owned;
            planned.drafter_scratch = scratch;
            planned.drafter_bridge = drafter_bridge_inventory(&draft_config(), 8, 8).unwrap();
            let (p, pg, pw) = solve_working_set_with_graphs(&planned, 262_144, false, 188, true).unwrap();
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
            let (r, rg, rw) = solve_working_set_with_graphs(&measured, 262_144, false, 188, true).unwrap();
            assert_eq!((pg, pw), (rg, rw));
            assert_eq!((&p.movables, p.pool_tokens, &p.layers, p.tp2),
                (&r.movables, r.pool_tokens, &r.layers, r.tp2));
            let bridge = planned.drafter_bridge.as_ref().unwrap();
            for rank in 0..2 {
                let bytes = |placed: &Placement, group: &str| placed.items[rank].iter()
                    .filter(|item| item.group == group).map(|item| item.bytes).sum::<u64>();
                assert_eq!(bytes(&p, "Drafter"), if rank == expected_home as usize { owned } else { 0 });
                for demand in &bridge.demands {
                    if demand.gpu as usize != rank { continue; }
                    let expected = if expected_home == 1 { demand.bytes } else { 0 };
                    assert_eq!(bytes(&p, &demand.group), expected);
                    assert_eq!(bytes(&r, &demand.group), expected);
                }
                assert_eq!(bytes(&p, "DFlash linear/attention scratch"),
                    if rank == expected_home as usize { planned.drafter_scratch[rank] } else { 0 });
                let total = |input: &GlmfInputs<'_>, placed: &Placement| input.gpus[rank].0 - GpuBudget {
                    capacity_bytes: input.gpus[rank].0, headroom_bytes: 0, baseline: input.gpus[rank].1 }.available()
                    + placed.items[rank].iter().map(|item| item.bytes).sum::<u64>();
                assert_eq!(total(&planned, &p), total(&measured, &r));
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
            assert!(matches!(request(&input), Err(PlacementError::Inventory("GLM Flash remote draft candidate"))));
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
        let mut req = request(&inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES)).unwrap();
        assert_eq!(req.movables[0].allowed, [0]);
        assert!(req.movables[0].conditional.is_empty());
        let expected = solve(&req).unwrap();
        req.movables[0].conditional.push(remote_dflash_bridge_demands(cfg.hidden as u64,
            cfg.vocab_size as u64, 6, 128).unwrap());
        let actual = solve(&req).unwrap();
        assert_eq!(actual.items, expected.items);
        assert_eq!(actual.movables, expected.movables);
        assert_eq!(actual.pool_tokens, expected.pool_tokens);
        assert_eq!(actual.layers, expected.layers);
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
    fn glm5_flash_nvfp4_a4_and_a16_select_matching_package_scratch() {
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
        let output = 4096 * 4096 * 2;
        assert_eq!(expert_workspace_selected(&catalog, Some(&manifest), 4096, 2, true).unwrap(), 67890 + output);
        assert_eq!(expert_workspace_selected(&catalog, Some(&manifest), 4096, 2, false).unwrap(), 12345 + output);
        std::fs::remove_dir(a4.join("tp2")).unwrap();
        assert_eq!(expert_workspace_selected(&catalog, Some(&manifest), 4096, 2, true).unwrap(), 12345 + output);
    }

    #[test]
    fn glm5_flash_graph_inventory_rebuilds_at_the_smaller_pool() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(45)).unwrap();
        let input = inputs(&cfg, 1, 32 << 30);
        let (placed, graphs) = solve_with_graphs(&input, 1 << 20, true, 170, true).unwrap();
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
            let (placed, graphs) = solve_with_graphs(&input, 2 << 20, true, 188, true).unwrap();
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
        let (placed, graphs) = solve_with_graphs(&input, 2 << 20, true, 188, true).unwrap();
        assert_eq!(placed.onboard_layers, 1);
        let context = glmf_graphs::admitted_graph_context(input.max_context as usize, placed.pool_tokens as usize, true);
        let expected = startup_graphs(&cfg, cfg.layers, context, placed.pool_tokens as usize,
            input.sequences as usize, true, 1, 64, 188);
        assert_eq!(graphs.ranks, expected.ranks);
    }

    #[test]
    fn layers_admission_rejects_four_lanes_without_changing_the_spark_only_path() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(45)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        input.prefill_lanes = 4;
        for mode in [AttentionPlacement::Heads, AttentionPlacement::Context] {
            input.attention.mode = mode;
            assert_eq!(GlmfWorkingSet::candidate(&input, 0).unwrap().prefill_lanes, 4);
        }
        input.attention.mode = AttentionPlacement::Layers;
        assert!(GlmfWorkingSet::candidate(&input, 0).is_err());
        for lanes in [1, 2] {
            input.prefill_lanes = lanes;
            assert_eq!(GlmfWorkingSet::candidate(&input, 0).unwrap().prefill_lanes, lanes);
        }
        assert_eq!(GlmfWorkingSet::candidate(&input, input.experts.len()).unwrap().prefill_lanes, 2);
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
            let heads = solve(&request(&input).unwrap()).unwrap();
            let heads_bytes = replicated_mla_bytes(&cfg, input.index, &heads).unwrap();
            assert_eq!(heads_bytes, 11 * (256 * (528 + 512) + 64 * 132) * (2 << 20) / 256);
            for mode in [AttentionPlacement::Context, AttentionPlacement::Layers] {
                input.index = cache::GlmfIndexCache::Compact;
                input.attention.mode = mode;
                let req = request(&input).unwrap();
                assert_eq!(solve(&req).unwrap().attention_placement, mode);
                let placed = solve(&req).unwrap();
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
                let record_bytes: u64 = placed.items.iter().flatten().filter(|i| i.group == "records").map(|i| i.bytes).sum();
                assert_eq!(record_bytes, (11 * (256 * 528 + 64 * 132) + 8) * (2 << 20) / 256);
                let mut measured = req.clone();
                for gpu in &mut measured.inventory.gpus {
                    let Baseline::Planned { context_bytes, loaded_bytes } = gpu.baseline else { unreachable!() };
                    gpu.baseline = Baseline::Measured { free_bytes: gpu.capacity_bytes - context_bytes - loaded_bytes };
                }
                let runtime = solve(&measured).unwrap();
                assert_eq!(placed.items, runtime.items);
                assert_eq!(placed.layers, runtime.layers);
                assert_eq!(placed.pool_tokens, runtime.pool_tokens);
            }
        }
    }

    fn owner_candidate(input: &GlmfInputs<'_>, measured: bool) -> (Placement, GraphSet, GlmfWorkingSet) {
        let mut req = request(input).unwrap();
        if measured {
            for gpu in &mut req.inventory.gpus {
                let Baseline::Planned { context_bytes, loaded_bytes } = gpu.baseline else { unreachable!() };
                gpu.baseline = Baseline::Measured { free_bytes: gpu.capacity_bytes - context_bytes - loaded_bytes };
            }
        }
        (solve(&req).unwrap(), input.graphs.clone(), GlmfWorkingSet::candidate(input, input.experts.len()).unwrap())
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
            let planned = select_attention_working_set(input.onboard, Ok(owner_candidate(&input, false)),
                Ok(owner_candidate(&alternate, false))).unwrap();
            let measured = select_attention_working_set(input.onboard, Ok(owner_candidate(&input, true)),
                Ok(owner_candidate(&alternate, true))).unwrap();
            assert_eq!(placed_attention(input.attention, &planned.0).first_owner, expected);
            assert_eq!(planned.0.items, measured.0.items);
            assert_eq!(planned.0.layers, measured.0.layers);
            assert_eq!(planned.0.pool_tokens, 2 << 20);
            assert_eq!(planned.0.movables, [(MovableId::Drafter, 0)]);
            let selected = solve_attention_working_set_with_graphs(&input, Some(&alternate.resident),
                2 << 20, false, 188, true).unwrap();
            assert_eq!(selected.0, planned.0);
            let mut measured_input = input.clone();
            for (capacity, baseline) in &mut measured_input.gpus {
                let Baseline::Planned { context_bytes, loaded_bytes } = *baseline else { unreachable!() };
                *baseline = Baseline::Measured { free_bytes: *capacity - context_bytes - loaded_bytes };
            }
            let runtime = solve_attention_working_set_with_graphs(&measured_input, Some(&alternate.resident),
                2 << 20, false, 188, true).unwrap();
            assert_eq!(selected.0, runtime.0);
            let resident = if expected == 0 { &input.resident } else { &alternate.resident };
            for rank in 0..2 {
                let bytes: u64 = planned.0.items[rank].iter().filter(|i| i.group == "target weights")
                    .map(|i| i.bytes).sum();
                assert_eq!(bytes, resident[rank].weights);
            }
        }
    }

    #[test]
    fn whole_owner_score_charges_reserved_units_preserves_policy_and_original_ties() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        input.spark_ranks = 0;
        input.attention = GlmfAttention { mode: AttentionPlacement::Layers, first_owner: 1, ..Default::default() };
        let original = owner_candidate(&input, false);
        let mut alternate = original.clone();
        for layer in &mut alternate.0.layers {
            if let LayerMode::Whole { gpu, ffn: FfnMode::Split } = &mut layer.mode { *gpu ^= 1; }
        }
        let selected = select_attention_working_set(input.onboard, Ok(original.clone()), Ok(alternate.clone())).unwrap();
        assert_eq!(placed_attention(input.attention, &selected.0).first_owner, 1);
        assert_eq!(select_attention_working_set(input.onboard, Err(PlacementError::Inventory("original")),
            Ok(alternate.clone())).unwrap().0.layers, alternate.0.layers);
        assert_eq!(select_attention_working_set(input.onboard, Ok(original.clone()),
            Err(PlacementError::Inventory("alternate"))).unwrap().0.layers, original.0.layers);
        assert_eq!(select_attention_working_set(input.onboard, Err(PlacementError::Inventory("original")),
            Err(PlacementError::Inventory("alternate"))).unwrap_err(), PlacementError::Inventory("original"));
        let mut placed = original.0;
        placed.pool_tokens = 4 * 256;
        placed.onboard_layers = 2;
        placed.pool_limits = PoolLimits(vec![PoolRankLimit { gpu: 0, available_bytes: 1000, max_units: 100,
            demands: vec![PoolUnitDemand { group: "bounded".into(), bytes_per_unit: 10,
                offset_units: 3, cap_units: 6 }] }]);
        assert_eq!(attention_owner_score(Onboard::Auto, &placed).unwrap(), ((1024, 2), 940));
        assert_eq!(attention_owner_score(Onboard::ExpertsFirst { pool_floor: 256 }, &placed).unwrap(), ((2, 1024), 940));
        let mut more_experts = placed.clone();
        more_experts.pool_tokens = 3 * 256;
        more_experts.onboard_layers = 3;
        more_experts.pool_limits.0[0].available_bytes = 900;
        assert!(attention_owner_score(Onboard::Auto, &placed).unwrap()
            > attention_owner_score(Onboard::Auto, &more_experts).unwrap());
        assert!(attention_owner_score(Onboard::ExpertsFirst { pool_floor: 256 }, &placed).unwrap()
            < attention_owner_score(Onboard::ExpertsFirst { pool_floor: 256 }, &more_experts).unwrap());
        placed.pool_limits.0[0].available_bytes = 59;
        assert!(attention_owner_score(Onboard::Auto, &placed).is_err());
    }

    #[test]
    fn owner_inventory_is_optional_for_heads_and_context_and_invalid_elsewhere() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        for mode in [AttentionPlacement::Heads, AttentionPlacement::Context] {
            let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
            input.attention.mode = mode;
            input.index = if mode == AttentionPlacement::Heads { cache::GlmfIndexCache::Keys }
                else { cache::GlmfIndexCache::Compact };
            let original = solve_working_set_with_graphs(&input, 2 << 20, false, 188, true).unwrap();
            let placed = solve_attention_working_set_with_graphs(&input, None, 2 << 20, false, 188, true).unwrap();
            assert_eq!(original.0, placed.0);
            assert_eq!(placed_attention(input.attention, &placed.0), input.attention);
            assert!(matches!(solve_attention_working_set_with_graphs(&input, Some(&input.resident),
                2 << 20, false, 188, true), Err(PlacementError::Inventory(_))));
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
            let planner = request(&input).unwrap();
            let placed = solve(&planner).unwrap();
            for rank in 0..2 {
                let charged: u64 = placed.items[rank].iter()
                    .filter(|item| item.group == "target weights").map(|item| item.bytes).sum();
                assert_eq!(charged, input.resident[rank].weights);
                assert!(charged >= routers);
                assert!(!placed.items[rank].iter().any(|item| item.group == "router replica"),
                    "routers are part of target weights, never an extra demand");
            }
            let mut runtime = planner.clone();
            for gpu in &mut runtime.inventory.gpus {
                let Baseline::Planned { context_bytes, loaded_bytes } = gpu.baseline else { unreachable!() };
                gpu.baseline = Baseline::Measured { free_bytes: gpu.capacity_bytes - context_bytes - loaded_bytes };
            }
            let admitted = solve(&runtime).unwrap();
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
    fn glm5_flash_resolves_context_by_default_and_layers_only_when_explicit() {
        let programs = Some(GlmfAttention { mode: AttentionPlacement::Context, topk_scratch: 7, partial_scratch: 9,
            ..Default::default() });
        use AttentionPlacement::{Context, Heads, Layers};
        assert_eq!(resolve_attention(None, 2, programs).unwrap().mode, Context);
        assert_eq!(resolve_attention(None, 1, programs).unwrap().mode, Heads);
        assert_eq!(resolve_attention(Some(Heads), 2, programs).unwrap().mode, Heads);
        assert_eq!(resolve_attention(None, 1, None).unwrap().mode, Heads);
        assert_eq!(resolve_attention(Some(Heads), 2, None).unwrap().mode, Heads);
        assert_eq!(resolve_attention(Some(Context), 2, programs).unwrap().topk_scratch, 7);
        // Auto is context on two GPUs, so a build without the programs fails closed, as an
        // explicit context does (issue #202); heads and one GPU still resolve.
        for (requested, gpus, programs) in [(None, 2, None), (Some(Context), 2, None)] {
            let error = resolve_attention(requested, gpus, programs).unwrap_err();
            assert!(matches!(error, PlacementError::MissingContextPrograms { family: "glm5_flash", .. }),
                "{requested:?}: {error}");
            assert!(error.to_string().contains("glmf_dsa_candidate_merge"), "{error}");
        }
        for (mode, gpus, programs) in [(Context, 1, programs), (Layers, 1, programs)] {
            assert!(matches!(resolve_attention(Some(mode), gpus, programs), Err(PlacementError::AttentionPlacement { .. })));
        }
        // Explicit layers does not need the context programs and never changes auto.
        for programs in [programs, None] {
            assert_eq!(resolve_attention(Some(Layers), 2, programs).unwrap().mode, Layers);
        }
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        let mut req = request(&inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES)).unwrap();
        assert!(req.layers.iter().filter(|l| l.kind == AttentionClass::Kda).all(|l| l.kv_unit.unit_bytes_context.is_none()));
        assert!(req.layers.iter().filter(|l| l.kind == AttentionClass::Dsa).all(|l| l.kv_unit.unit_bytes_context.is_some()));
        assert_eq!(solve(&req).unwrap().attention_placement, Heads);
        req.attention_placement = Some(Layers);
        let placed = solve(&req).unwrap();
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
        let placed = solve(&request(&one).unwrap()).unwrap();
        assert!(placed.layers.iter().all(|layer| layer.mode == LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }));
        executor.check(&placed).unwrap();
    }

    #[test]
    fn glm5_flash_split_without_tp2_cannot_place_whole_experts() {
        let cfg = GlmNextConfig::from_hf(&crate::plan::testing::glm5_flash_config(2)).unwrap();
        let mut input = inputs(&cfg, 2, inventory::PRO_TOTAL_BYTES);
        for expert in &mut input.experts { expert.tp2 = false; }
        let placed = solve(&request(&input).unwrap()).unwrap();
        assert_eq!(placed.onboard_layers, 0);
        input.onboard = Onboard::Layers(1);
        assert!(solve(&request(&input).unwrap()).is_err());
    }
}
