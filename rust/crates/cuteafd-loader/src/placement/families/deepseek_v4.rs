//! DeepSeek V4 (Flash/Pro) onto the shared solver: the one place that turns
//! V4's cache geometry, workspace formulas, exchange slots and expert
//! residency into a [`PlacementRequest`]. `plan --layout` and serve-dsv4
//! both call [`request`]; they differ only in [`V4Inputs::baseline`].
//!
//! Folds V4's former `family_costs`/`layout.rs` branches (P13): the reserve
//! envelope, the 262K legacy expert reserve and the per-device pool sizing
//! now live here, once.
use crate::families::deepseek_v4::DeepseekV4Config;
use crate::placement::*;
use crate::placement::inventory::GraphSet;
use crate::serving_capacity::{deepseek_v4_cache_geometry, deepseek_v4_expert_cost, deepseek_v4_expert_exchange_bytes,
    deepseek_v4_headroom_bytes, deepseek_v4_layer_unit_bytes, deepseek_v4_peer_exchange_bytes, V4ExpertCost,
    V4WorkspaceRank};
use cuteafd_core::memory_layout::{Basis, Category};

/// Prefill lanes of the V4 engine (`engine::PREFILL_LANES`).
pub const PREFILL_LANES: u64 = 2;
/// Lazily captured decode graphs per role (1 GPU, lead, peer): V4 keys its
/// segment graphs by the exact compressed-table width, so the set grows with
/// traffic after ready (rc3: +0.5-0.65 GiB Flash, +1.2-1.3 GiB Pro within a
/// smoke run). Reserved as growth beside the pool, never part of the ready
/// ledger (`inventory::GraphSet::budget`).
pub const GRAPH_BYTES: [u64; 3] = [gib(35), gib(40), gib(25)];
/// Step workspace allowance per role at 4096 prefill rows, used only when no
/// program manifest gives the exact geometry.
pub const WORKSPACE_BYTES: [u64; 3] = [gib(480), gib(414), gib(355)];
/// The largest pool a fixed onboard fills: V4's compressed slots are i32
/// indices of `tokens / 4` C4 groups, so 2^31 tokens stays well inside them.
pub const POOL_CEILING_TOKENS: u64 = 1 << 31;
/// Default `--reserve-gib`.
pub const RESERVE_BYTES: u64 = 10 << 30;

const fn gib(hundredths: u64) -> u64 { hundredths * (1 << 30) / 100 }

/// Everything V4 admission depends on, gathered by the caller from the
/// checkpoint, the program manifest and the CLI.
#[derive(Debug, Clone)]
pub struct V4Inputs<'a> {
    pub cfg: &'a DeepseekV4Config,
    /// dSpark stages that hold caches (all the checkpoint's stages).
    pub cache_stages: usize,
    /// Per-GPU total bytes and baseline; one or two entries. The V4 reserve
    /// becomes each GPU's headroom.
    pub gpus: Vec<(u64, Baseline)>,
    /// Lower bound for that reserve (`plan --headroom-gib`).
    pub headroom_floor: u64,
    pub spark_ranks: usize,
    pub sequences: u64,
    pub prefill_rows: u64,
    pub decode_rows: u64,
    pub max_context: u64,
    /// `--reserve-gib` in bytes.
    pub reserve_bytes: u64,
    /// Prefix mark slots (`MarkArena::slots_for`), 0 without a prefix cache.
    pub mark_slots: u64,
    /// Exact per-rank step workspaces from the program manifest, or `None`
    /// for the calibrated allowance.
    pub workspace: Option<Vec<V4WorkspaceRank>>,
    /// Per-layer whole-layer residency (backbone routed layers in order),
    /// empty when routed experts never stay on RTX.
    pub experts: Vec<V4ExpertCost>,
    /// dSpark stage experts loaded into GPU0's arena (`--dspark`).
    pub draft: Vec<V4ExpertCost>,
    /// Half-layer residency per routed backbone layer and rank.
    pub experts_half: Vec<[V4ExpertCost; 2]>,
    pub tp2_workspace: [u64; 2],
    /// Force FP32 FFN payloads for prefill as well as decode.
    pub exchange_f32: bool,
    /// The local expert arena's workspace.
    pub expert_workspace: u64,
    pub first_routed: usize,
    pub requested_pool: Option<u64>,
    /// `RTX_EXPERT_LAYERS` / `--rtx-expert-layers`.
    pub onboard: Onboard,
    /// Probe-only all-row logits on the lead GPU.
    pub full_prefill_logits: u64,
    /// Per GPU, the measured code it holds at ready (`placement::inventory::loaded_code`, 0 without an
    /// entry). Part of the baseline on both sides; on a PRO card the reserve envelope covers it
    /// (it always did: the code loads after the runtime's sample), so it does not shrink the pool twice.
    pub code_bytes: Vec<u64>,
}

/// Prefix mark slots of the V4 arena (`MarkArena::slots_for` over every
/// rank's mark), 0 without prefix entries.
pub fn mark_slots(cfg: &DeepseekV4Config, gpus: usize, prefill_rows: u64, cache_stages: usize, sequences: u64,
    entries: u64, budget_bytes: u64) -> Result<u64, PlacementError> {
    let geometry = deepseek_v4_cache_geometry(cfg, gpus, prefill_rows, cache_stages)
        .map_err(|_| PlacementError::Inventory("V4 cache geometry"))?;
    let mark: u64 = geometry.ranks.iter().map(|r| r.retained_mark_bytes).sum();
    Ok(cuteafd_core::prefix::mark_slots_for(sequences, entries, mark, budget_bytes))
}

/// Fixed V4 demands and pool overhead per GPU plus each GPU's reserve, the
/// ones both sides previously computed apart.
pub fn fixed_demands(inputs: &V4Inputs<'_>) -> Result<(Vec<Demand>, Vec<u64>, Vec<u64>), PlacementError> {
    let gpus = inputs.gpus.len();
    let geometry = deepseek_v4_cache_geometry(inputs.cfg, gpus, inputs.prefill_rows, inputs.cache_stages)
        .map_err(|_| PlacementError::Inventory("V4 cache geometry"))?;
    let hidden = inputs.cfg.dim as u64;
    let topk = inputs.cfg.n_activated_experts as u64;
    let tables = PREFILL_LANES * inputs.prefill_rows + inputs.decode_rows;
    let table_bytes = tables.checked_mul(4).ok_or(PlacementError::Overflow("V4 lane tables"))?;
    let mut demands = Vec::new();
    let mut overhead = Vec::new();
    let mut reserves = Vec::new();
    for (rank, cache) in geometry.ranks.iter().enumerate() {
        let gpu = rank as u8;
        let role = if gpus == 1 { 0 } else if rank == 0 { 1 } else { 2 };
        let mul = |a: u64, b: u64, what| a.checked_mul(b).ok_or(PlacementError::Overflow(what));
        let state = mul(cache.active_state_per_sequence_bytes, inputs.sequences, "V4 state")?
            + cache.fixed_state_bytes + cache.speculative_replay_bytes + mul(table_bytes, inputs.sequences, "V4 tables")?;
        let context = mul(cache.context_table_bytes_per_token, inputs.max_context, "V4 RoPE tables")?;
        let intake = if rank == 0 && inputs.spark_ranks > 0 {
            PREFILL_LANES * inputs.spark_ranks as u64 * 4096 * hidden * 2
        } else { 0 };
        let exchange = if gpus == 2 {
            deepseek_v4_expert_exchange_bytes(hidden, topk, inputs.prefill_rows, inputs.decode_rows, rank)
                .map_err(|_| PlacementError::Overflow("V4 expert exchange"))?
        } else { 0 };
        let workspace = inputs.workspace.as_ref().and_then(|w| w.get(rank)).map(|w| w.fixed_device_bytes)
            .unwrap_or(WORKSPACE_BYTES[role] * inputs.prefill_rows / 4096) + intake;
        let headroom = if inputs.workspace.is_some() {
            deepseek_v4_headroom_bytes(inputs.gpus[rank].0, inputs.reserve_bytes,
                workspace + exchange + inputs.code_bytes.get(rank).copied().unwrap_or(0), GRAPH_BYTES[role])
        } else {
            inputs.reserve_bytes.saturating_sub(workspace + exchange + GRAPH_BYTES[role]).max(3 << 30)
        }.max(inputs.headroom_floor);
        reserves.push(headroom);
        let basis = if inputs.workspace.is_some() { Basis::Formula } else { Basis::Estimated };
        demands.push(Demand::new(gpu, Category::Kv, "state", state, Basis::Formula));
        demands.push(Demand::new(gpu, Category::Kv, "RoPE context tables", context, Basis::Formula));
        if inputs.mark_slots > 0 && cache.retained_mark_bytes > 0 {
            demands.push(Demand::new(gpu, Category::Prefix, "marks",
                mul(cache.retained_mark_bytes, inputs.mark_slots, "V4 marks")?, Basis::Formula));
        }
        demands.push(Demand::new(gpu, Category::Workspace, "steps", workspace, basis));
        demands.push(GraphSet::budget(&[GRAPH_BYTES[role]]).demand(0).on(gpu));
        if gpus == 2 {
            demands.push(Demand::new(gpu, Category::Transport, "peer exchange",
                deepseek_v4_peer_exchange_bytes(hidden, inputs.prefill_rows, inputs.decode_rows, true, inputs.exchange_f32)
                    .map_err(|_| PlacementError::Overflow("V4 peer exchange"))?, Basis::Formula));
            demands.push(Demand::new(gpu, Category::Transport, "expert peer exchange", exchange, Basis::Formula));
        }
        if rank == 0 && inputs.full_prefill_logits > 0 {
            demands.push(Demand::new(gpu, Category::Workspace, "probe prefill logits", inputs.full_prefill_logits,
                Basis::Formula));
        }
        // Unit metadata and every lane's page table follow the pool, beside the
        // per-layer records.
        overhead.push(cache.pool_metadata_unit_bytes.checked_add(table_bytes)
            .ok_or(PlacementError::Overflow("V4 pool overhead"))?);
    }
    Ok((demands, overhead, reserves))
}

/// The complete V4 request: CSA layers (replicated latent under the head
/// split), TP2 backbone halves on two RTX, dSpark stage experts in
/// GPU0's arena. The pool unit sums each layer's records plus the overhead.
pub fn request(inputs: &V4Inputs<'_>) -> Result<PlacementRequest, PlacementError> {
    let gpus = inputs.gpus.len();
    let (fixed, pool_overhead, reserves) = fixed_demands(inputs)?;
    let budgets = inputs.gpus.iter().zip(&reserves).map(|(&(capacity_bytes, baseline), &headroom_bytes)|
        GpuBudget { capacity_bytes, headroom_bytes, baseline }).collect::<Vec<_>>();
    let first = inputs.first_routed;
    let layers = (0..inputs.cfg.n_layers).map(|layer| {
        let unit = deepseek_v4_layer_unit_bytes(inputs.cfg.compress_ratios[layer]);
        let experts = layer.checked_sub(first).and_then(|i| inputs.experts.get(i)).map(|cost| ExpertCost {
            whole: Bytes2 { resident: cost.resident_bytes, staging: cost.staging_bytes },
            half: inputs.experts_half.get(layer - first).map(|halves| halves.map(|h| Bytes2 { resident: h.resident_bytes, staging: h.staging_bytes })).unwrap_or_default(),
            tp2: gpus == 2 && inputs.experts_half.get(layer - first).is_some(), spark_ok: true });
        LayerDemand {
            kind: AttentionClass::Csa,
            // Coordinator weights load before admission (inside the baseline).
            weights: ModeBytes::default(),
            kv_unit: ModeBytes::replicated(unit),
            experts,
            modes: if gpus == 2 { vec![LayerMode::HeadSplit] } else { vec![LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }] },
        }
    }).collect();
    let movables = if inputs.draft.is_empty() { Vec::new() } else {
        vec![Movable { id: MovableId::DsparkExperts, allowed: vec![0], expert_arena: true,
            parts: inputs.draft.iter().map(|c| Bytes2 { resident: c.resident_bytes, staging: c.staging_bytes }).collect() }]
    };
    let geometry_unit = 256;
    Ok(PlacementRequest {
        inventory: Inventory { gpus: budgets, spark_ranks: inputs.spark_ranks, peer_access: gpus == 2 },
        pool: PoolPolicy { ceiling: POOL_CEILING_TOKENS, ..PoolPolicy::resolve(
            &inputs.gpus.iter().map(|g| g.0).collect::<Vec<_>>(), inputs.max_context, inputs.requested_pool,
            geometry_unit, inputs.spark_ranks == 0) },
        layers,
        pool_overhead,
        fixed,
        movables,
        expert_workspace: inputs.expert_workspace,
        tp2_workspace: inputs.tp2_workspace,
        onboard: inputs.onboard,
        expert_gpus: if gpus == 2 && inputs.experts_half.is_empty() { 0 } else { 1 },
        policy: LayerPolicy { default: vec![LayerMode::HeadSplit, LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }],
            by_kind: Vec::new() },
        // The mHC streams `[T,4,H]` BF16; V4's only hop is the entry
        // broadcast of the embedding rows into GPU1's step input.
        hops: HopSpec { row_bytes: 4 * inputs.cfg.dim as u64 * 2, rows: inputs.prefill_rows.max(inputs.decode_rows),
            lanes: PREFILL_LANES, entry_gpu: 0, head_gpu: 0 },
        executor: super::DEEPSEEK_V4,
    })
}

/// The `LoadedCode::experts` key of a V4 catalog: EXL3 packages load their own modules (the
/// coordinator holds dSpark and any RTX layers in them), native MXFP4 does not.
pub fn code_experts(catalog: &crate::OfficialV41Catalog) -> &'static str {
    if catalog.exl3().is_some() { "exl3" } else { "*" }
}

/// Each coordinator GPU's measured loaded code (`placement::inventory::LOADED_CODE`) for a V4
/// model of hidden size `dim` with `experts` (`code_experts`) on `gpus` GPUs.
pub fn code_bytes(dim: usize, experts: &str, gpus: usize) -> Vec<u64> {
    let family = if dim == 4096 { "dsv4f" } else { "dsv4p" };
    (0..gpus).map(|rank| crate::placement::loaded_code(family, experts, gpus == 2, rank as u8).map_or(0, |c| c.bytes))
        .collect()
}

/// V4 defaults uniformly to pool-first placement. Explicit `max` remains
/// supported; on two RTX every resident backbone expert is a TP2 half.
pub fn default_onboard() -> Onboard {
    Onboard::Auto
}

#[cfg(test)]
mod default_tests {
    use super::*;

    #[test]
    fn default_onboard_is_pool_first_for_every_v4_layout() {
        assert_eq!(default_onboard(), Onboard::Auto);
    }
}

/// Exact TP2 halves in backbone order, CPU only.
pub fn expert_half_costs(catalog: &crate::OfficialV41Catalog) -> anyhow::Result<Vec<[V4ExpertCost; 2]>> {
    let routed = catalog.routed_experts();
    (routed.first_layer..routed.layers).map(|layer| Ok([
        crate::serving_capacity::deepseek_v4_tp2_expert_cost(catalog, layer, 0)?,
        crate::serving_capacity::deepseek_v4_tp2_expert_cost(catalog, layer, 1)?,
    ])).collect()
}

/// Whole-layer residency of every routed backbone layer and `stages` dSpark
/// stages, from the catalog (no CUDA).
pub fn expert_costs(catalog: &crate::OfficialV41Catalog, stages: usize)
    -> anyhow::Result<(Vec<V4ExpertCost>, Vec<V4ExpertCost>)> {
    let routed = catalog.routed_experts();
    let layers = (routed.first_layer..routed.layers).map(|layer| deepseek_v4_expert_cost(catalog, layer, false))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let draft = (0..stages).map(|stage| deepseek_v4_expert_cost(catalog, stage, true))
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok((layers, draft))
}
