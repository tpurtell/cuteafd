//! DeepSeek V4 (Flash/Pro) onto the shared solver: the one place that turns
//! V4's cache geometry, workspace formulas, exchange slots and expert
//! residency into a [`PlacementRequest`]. `plan --layout` and serve-dsv4
//! both call [`request`]; they differ only in [`V4Inputs::baseline`].
//!
//! V4's reserve envelope, the 262K legacy expert reserve and the per-device
//! pool sizing live here, once.
use crate::families::deepseek_v4::DeepseekV4Config;
use crate::placement::*;
use crate::placement::inventory::GraphSet;
use crate::serving_capacity::{deepseek_v4_cache_geometry, deepseek_v4_expert_cost, deepseek_v4_expert_exchange_bytes,
    deepseek_v4_headroom_bytes, deepseek_v4_layer_unit_bytes, deepseek_v4_peer_exchange_bytes, V4ExpertCost,
    V4WorkspaceRank};
use cuteafd_core::memory_layout::{Basis, Category};

/// Outstanding prefill units on the single Spark connection set.
pub const PREFILL_WAVES: u64 = 2;
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
    /// True only when the operator left context automatic.
    pub automatic_context: bool,
    pub cfg: &'a DeepseekV4Config,
    /// Header-derived uploads; no attention weights hidden in the baseline.
    pub weights: crate::families::deepseek_v4::resident::WeightInventory,
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
    pub prefill_waves: u64,
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
    /// Per GPU, the calibrated or formula-based code it holds at ready. Part of the baseline on
    /// both sides; on a PRO card the reserve envelope covers it
    /// (it always did: the code loads after the runtime's sample), so it does not shrink the pool twice.
    pub code_bytes: Vec<u64>,
    /// The build's C4 token-split programs (`manifest_context`): `None` keeps every layer
    /// head-split. Only two GPUs with peer access use it.
    pub context: Option<V4Context>,
    /// `--attention-placement`: `None` is auto (the family's recorded default).
    pub attention: Option<AttentionPlacement>,
}

/// Native C4 context kernel scratch, read from the selected AOT manifest.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct V4Context {
    pub topk_scratch: u64,
    pub partial_scratch: u64,
}

/// Fail closed when either head range or any gather/merge primitive is absent.
/// A build without them keeps every layer head-split.
pub fn manifest_context(manifest: &serde_json::Value, family: &str, heads: usize,
    decode_rows: u64) -> Option<V4Context> {
    let programs = manifest["programs"].as_array()?;
    let lookup = |name: &str| programs.iter().find(|p| p["name"].as_str() == Some(name))
        .map(|p| p["scratch_bytes_at_capacity"]["scratch"].as_u64().unwrap_or(0));
    if heads == 0 || heads % 2 != 0 || decode_rows == 0 { return None; }
    for suffix in ["dsa_candidate_merge", "lse_combine2", "paged_staging_gather_c4", "paged_staging_gather_index"] {
        lookup(&format!("{family}_{suffix}"))?;
    }
    let partial = |begin| programs.iter().filter(|p| p["name"].as_str().is_some_and(|name|
        name.starts_with(&format!("{family}_sparse_mla_partial_h{begin}_s"))
            && name.ends_with(&format!("_m{decode_rows}"))))
        .map(|p| p["scratch_bytes_at_capacity"]["scratch"].as_u64().unwrap_or(0)).max();
    Some(V4Context { topk_scratch: lookup(&format!("{family}_scored_index_topk_decode_m{decode_rows}"))?,
        partial_scratch: partial(0)?.max(partial(heads / 2)?) })
}

/// A prefill lane may finish a group begun in history and span an extra unit.
/// Every second unit is local; four planes cover payload, footer, keys, scales.
pub fn context_commit_units(prefill_rows: u64) -> u64 {
    (prefill_rows.div_ceil(256) + 1).div_ceil(2)
}

/// Non-owner compressor CTAs need distinct output rows, not one aliased slot.
pub fn context_scratch_units(decode_rows: u64) -> u64 {
    decode_rows.div_ceil(64)
}

/// Per-rank buffers beside staging/exchange, with the engine's 256-byte floor.
/// Scratch output pages exist per C4 layer; everything else is shared by layer.
pub fn context_workspace(cfg: &DeepseekV4Config, extent: u64, prefill_rows: u64, decode_rows: u64,
    prefill_lanes: u64, context: V4Context) -> Vec<(&'static str, u64)> {
    let partial = decode_rows * cfg.n_heads as u64 / 2 * (512 * 2 + 4);
    let half = extent.div_ceil(256).div_ceil(2);
    let commit = 4 * 3 * context_commit_units(prefill_rows) * 8;
    let mut out = vec![("candidates", decode_rows * cfg.index_topk as u64 * 8),
        ("own-head partial", partial), ("peer-head partial", partial),
        ("scored top-k scratch", context.topk_scratch), ("partial scratch", context.partial_scratch),
        ("gather tables", prefill_lanes * 2 * half * 4), ("local selected lengths", decode_rows * 4),
        ("empty window lengths", decode_rows * 4)];
    out.extend((0..prefill_lanes).map(|_| ("commit blocks", commit)));
    let scratch_units = context_scratch_units(decode_rows);
    for _ in cfg.compress_ratios.iter().filter(|&&ratio| ratio == 4) {
        out.push(("C4 scratch records", scratch_units * 37_440));
        out.push(("C4 scratch index", scratch_units * 8_448));
    }
    out
}

/// One canonical demand for the runtime's context-only memory. Prefill gathers
/// through two staging parities; decode exchanges on a single lane.
pub fn context_buffers(cfg: &DeepseekV4Config, extent: u64, prefill_rows: u64, decode_rows: u64,
    prefill_lanes: u64, context: V4Context) -> ContextBuffers {
    ContextBuffers { staging_unit_bytes: deepseek_v4_layer_unit_bytes(4), staging_unit_rows: 256,
        query_row_bytes: cfg.n_heads as u64 / 2 * 512 * 2,
        partial_row_bytes: cfg.n_heads as u64 / 2 * (512 * 2 + 4),
        candidate_row_bytes: cfg.index_topk as u64 * 8, compiled_extent: extent, decode_rows, lanes: 1,
        workspace_bytes: context_workspace(cfg, extent, prefill_rows, decode_rows, prefill_lanes, context)
            .iter().map(|&(_, bytes)| bytes.max(256)).sum() }
}

/// Replicated TP2 Heads snapshots cannot use the single-owner host copy path.
/// Keep retention on device until host restore can restore both replicas.
pub fn host_marks_enabled(gpus: usize, partitioned: bool, host: bool) -> bool {
    host && (gpus == 1 || partitioned)
}

/// Prefix mark slots over every rank's mark, 0 without prefix entries.
pub fn mark_slots(cfg: &DeepseekV4Config, gpus: usize, prefill_rows: u64, cache_stages: usize, sequences: u64,
    entries: u64, budget_bytes: u64, host: bool) -> Result<u64, PlacementError> {
    let geometry = deepseek_v4_cache_geometry(cfg, gpus, prefill_rows, cache_stages)
        .map_err(|_| PlacementError::Inventory("V4 cache geometry"))?;
    let mark: u64 = geometry.ranks.iter().map(|r| r.retained_mark_bytes).sum();
    Ok(cuteafd_core::prefix::mark_slots_with_host(sequences, entries, mark, budget_bytes, host))
}

/// Owner-local route ring of the shared draft probe; pinned host bytes have the same extent.
pub fn draft_probe_bytes(cfg: &DeepseekV4Config, decode_rows: u64) -> Result<u64, PlacementError> {
    [cfg.n_layers as u64, decode_rows, cfg.n_activated_experts as u64, 4].into_iter()
        .try_fold(1u64, |bytes, extent| bytes.checked_mul(extent)
            .ok_or(PlacementError::Overflow("V4 draft route ring"))).map(|bytes| bytes.max(256))
}

/// Fixed V4 demands and pool overhead per GPU plus each GPU's reserve, the
/// ones both sides previously computed apart.
pub fn fixed_demands(inputs: &V4Inputs<'_>) -> Result<(Vec<Demand>, Vec<u64>, Vec<u64>), PlacementError> {
    let gpus = inputs.gpus.len();
    let geometry = deepseek_v4_cache_geometry(inputs.cfg, gpus, inputs.prefill_rows, inputs.cache_stages)
        .map_err(|_| PlacementError::Inventory("V4 cache geometry"))?;
    let hidden = inputs.cfg.dim as u64;
    let topk = inputs.cfg.n_activated_experts as u64;
    let tables = inputs.prefill_waves + inputs.decode_rows;
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
            (inputs.prefill_waves * inputs.prefill_rows + inputs.decode_rows)
                * inputs.spark_ranks as u64 * hidden * 2
        } else { 0 };
        let exchange = if gpus == 2 {
            deepseek_v4_expert_exchange_bytes(hidden, topk, inputs.prefill_rows, inputs.decode_rows, rank, inputs.prefill_waves)
                .map_err(|_| PlacementError::Overflow("V4 expert exchange"))?
        } else { 0 };
        let probe = if rank == 0 && !inputs.draft.is_empty() { draft_probe_bytes(inputs.cfg, inputs.decode_rows)? } else { 0 };
        let workspace = inputs.workspace.as_ref().and_then(|w| w.get(rank)).map(|w| w.fixed_device_bytes)
            .unwrap_or(WORKSPACE_BYTES[role] * inputs.prefill_rows / 4096) + intake + probe;
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
        if rank == 0 && inputs.spark_ranks > 0 {
            demands.push(Demand::new(gpu, Category::Transport, "Spark egress",
                (4096 + inputs.decode_rows) * hidden * 2, Basis::Formula));
        }
        demands.push(GraphSet::budget(&[GRAPH_BYTES[role]]).demand(0).on(gpu));
        if gpus == 2 {
            demands.push(Demand::new(gpu, Category::Transport, "peer exchange",
                deepseek_v4_peer_exchange_bytes(hidden, inputs.prefill_rows, inputs.decode_rows, true, inputs.exchange_f32, inputs.prefill_waves)
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
    let (mut fixed, pool_overhead, reserves) = fixed_demands(inputs)?;
    if inputs.weights.c4_attention.len() != inputs.cfg.n_layers {
        return Err(PlacementError::Inventory("V4 attention weight inventory per layer"));
    }
    fixed.extend(inputs.weights.fixed.iter().cloned());
    let budgets = inputs.gpus.iter().zip(&reserves).map(|(&(capacity_bytes, baseline), &headroom_bytes)|
        GpuBudget { capacity_bytes, headroom_bytes, baseline }).collect::<Vec<_>>();
    let first = inputs.first_routed;
    let context = inputs.context.filter(|_| gpus == 2);
    let layers = (0..inputs.cfg.n_layers).map(|layer| {
        let unit = deepseek_v4_layer_unit_bytes(inputs.cfg.compress_ratios[layer]);
        let experts = layer.checked_sub(first).and_then(|i| inputs.experts.get(i)).map(|cost| ExpertCost {
            whole: Bytes2 { resident: cost.resident_bytes, staging: cost.staging_bytes },
            half: inputs.experts_half.get(layer - first).map(|halves| halves.map(|h| Bytes2 { resident: h.resident_bytes, staging: h.staging_bytes })).unwrap_or_default(),
            tp2: gpus == 2 && inputs.experts_half.get(layer - first).is_some(), spark_ok: true });
        LayerDemand {
            kind: AttentionClass::Csa,
            weights: inputs.weights.c4_attention[layer],
            kv_unit: KvDemand { unit_bytes_whole: unit, unit_bytes_split: [unit; 2],
                unit_bytes_context: (inputs.cfg.compress_ratios[layer] == 4).then_some([unit.div_ceil(2), unit / 2]) },
            colocate: None, fixed_bytes: ModeBytes::default(), extra: Vec::new(),
            context_indexer: inputs.cfg.compress_ratios[layer] == 4,
            experts,
            // Only C4 layers token-split; C128 and window-only layers stay head-split.
            modes: match (gpus, context.is_some() && inputs.cfg.compress_ratios[layer] == 4) {
                (2, true) => vec![LayerMode::HeadSplit, LayerMode::ContextSplit],
                (2, false) => vec![LayerMode::HeadSplit],
                _ => vec![LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }],
            },
        }
    }).collect();
    // dSpark reads the head GPU's taps every step, so it stays beside the head (no bridge).
    let movables = if inputs.draft.is_empty() { Vec::new() } else {
        vec![crate::placement::draft::DrafterDemand::local(MovableId::DsparkExperts,
            inputs.draft.iter().map(|c| Bytes2 { resident: c.resident_bytes, staging: c.staging_bytes }).collect(), 0)
            .movable(gpus)?]
    };
    let geometry_unit = 256;
    Ok(PlacementRequest {
        pool_bounded: Vec::new(),
        attention_placement: inputs.attention,
        layers_first_gpu: 0,
        // Exactly what the engine allocates under context (charged only when a layer runs it).
        context_buffers: context_buffers(inputs.cfg, inputs.max_context, inputs.prefill_rows, inputs.decode_rows,
            inputs.prefill_waves, context.unwrap_or_default()),
        inventory: Inventory { gpus: budgets, spark_ranks: inputs.spark_ranks, peer_access: gpus == 2 },
        pool: PoolPolicy { ceiling: POOL_CEILING_TOKENS, ..PoolPolicy::resolve(
            &inputs.gpus.iter().map(|g| g.0).collect::<Vec<_>>(), inputs.max_context, inputs.requested_pool,
            geometry_unit, inputs.spark_ranks == 0).with_automatic_context(inputs.automatic_context) },
        layers,
        pool_overhead,
        fixed,
        startup: Vec::new(),
        movables,
        expert_workspace: inputs.expert_workspace,
        tp2_workspace: inputs.tp2_workspace,
        onboard: inputs.onboard,
        minimum_expert_layers: 0,
        expert_gpus: if gpus == 2 && inputs.experts_half.is_empty() { 0 } else { 1 },
        // `context` is V4's default on two GPUs (#58/#59: no context-scaling KV replicated): the C4
        // layers token-split their records; `--attention-placement heads` stays an explicit option.
        policy: LayerPolicy { default: vec![LayerMode::HeadSplit, LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }],
            by_kind: Vec::new(), attention: if context.is_some() { vec![(AttentionClass::Csa, AttentionPlacement::Context)] }
                else { Vec::new() }, ..LayerPolicy::default() },
        // The mHC streams `[T,4,H]` BF16; V4's only hop is the entry
        // broadcast of the embedding rows into GPU1's step input.
        hops: HopSpec { storage: crate::placement::HopStorage::ReceiveSlots, row_bytes: 4 * inputs.cfg.dim as u64 * 2, rows: inputs.prefill_rows.max(inputs.decode_rows),
            lanes: inputs.prefill_waves, entry_gpu: 0, head_gpu: 0, terminal: TerminalHome::Head },
        // A build without the C4 context programs runs heads only.
        executor: if context.is_some() { super::DEEPSEEK_V4 }
            else { ExecutorModes { modes: &[LayerMode::HeadSplit, LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }],
                attention: AttentionPlacement::Heads, ..super::DEEPSEEK_V4 } },
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
    let arch = crate::placement::ArchContext::for_device("sm_120", crate::placement::inventory::PRO_TOTAL_BYTES);
    (0..gpus).map(|rank| crate::placement::inventory::loaded_code_allowance(
        family, experts, gpus == 2, rank as u8, arch, None).bytes).collect()
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

    fn context_cfg() -> DeepseekV4Config {
        DeepseekV4Config::from_model_args(&serde_json::json!({
            "vocab_size":129280,"dim":4096,"moe_inter_dim":2048,"n_layers":43,"n_hash_layers":3,
            "n_heads":64,"n_routed_experts":256,"n_shared_experts":1,"n_activated_experts":6,
            "score_func":"sqrtsoftplus","route_scale":1.5,"swiglu_limit":10.0,"q_lora_rank":1024,
            "head_dim":512,"rope_head_dim":64,"o_groups":8,"o_lora_rank":1024,"window_size":128,
            "original_seq_len":65536,"rope_theta":10000,"rope_factor":16,"beta_fast":32,"beta_slow":1,
            "index_n_heads":64,"index_head_dim":128,"index_topk":512,"hc_mult":4,"hc_sinkhorn_iters":20,
            "compress_rope_theta":160000,"compress_ratios":(0..43).map(|i|
                if i < 2 { 0 } else if i % 2 == 0 { 4 } else { 128 }).collect::<Vec<_>>()
        }), 0).unwrap()
    }

    #[test]
    fn draft_probe_extent_matches_the_owner_local_ring() {
        let cfg = context_cfg();
        assert_eq!(draft_probe_bytes(&cfg, 64).unwrap(), 43 * 64 * 6 * 4);
        assert_eq!(draft_probe_bytes(&cfg, 1).unwrap(), 43 * 6 * 4);
        assert_eq!(draft_probe_bytes(&cfg, 0).unwrap(), 256);
        assert!(draft_probe_bytes(&cfg, u64::MAX).is_err());
    }

    #[test]
    fn context_workspace_charges_distinct_scratch_pages_and_four_commit_planes() {
        let mut cfg = context_cfg();
        for (heads, topk, c4_layers) in [(64, 512, 21), (128, 1024, 30)] {
            cfg.n_heads = heads;
            cfg.index_topk = topk;
            cfg.compress_ratios = vec![4; c4_layers];
            for rows in [1, 64, 65, 128] {
                let scratch = V4Context { topk_scratch: 12345, partial_scratch: 56789 };
                let buffers = context_buffers(&cfg, 1 << 20, 4096, rows, 2, scratch);
                let workspace = context_workspace(&cfg, 1 << 20, 4096, rows, 2, scratch);
                assert_eq!(buffers.lanes, 1);
                assert_eq!(buffers.staging_unit_bytes, 45_888);
                assert_eq!(buffers.query_row_bytes, heads as u64 / 2 * 1024);
                assert_eq!(buffers.partial_row_bytes, heads as u64 / 2 * 1028);
                assert_eq!(buffers.candidate_row_bytes, topk as u64 * 8);
                assert_eq!(buffers.workspace_bytes, workspace.iter().map(|&(_, n)| n.max(256)).sum::<u64>());
                assert_eq!(workspace.iter().filter(|&&(name, _)| name == "commit blocks").count(), 2);
                assert!(workspace.iter().filter(|&&(name, _)| name == "commit blocks")
                    .all(|&(_, n)| n == 4 * 3 * 9 * 8));
                let scratch_bytes: u64 = workspace.iter().filter(|&&(name, _)| name.starts_with("C4 scratch"))
                    .map(|&(_, n)| n).sum();
                assert_eq!(scratch_bytes, c4_layers as u64 * rows.div_ceil(64) * 45_888);
                let demands = buffers.demands().unwrap();
                assert_eq!(demands.iter().find(|d| d.group == "context staging").unwrap().bytes,
                    2 * 4096 * 45_888);
            }
        }
        assert_eq!(context_commit_units(1), 1);
        assert_eq!(context_commit_units(257), 2);
    }

    #[test]
    fn context_manifest_needs_both_head_ranges_and_every_primitive() {
        let names = ["dsv4f_scored_index_topk_decode_m64", "dsv4f_dsa_candidate_merge", "dsv4f_lse_combine2",
            "dsv4f_paged_staging_gather_c4", "dsv4f_paged_staging_gather_index",
            "dsv4f_sparse_mla_partial_h0_s4_m64", "dsv4f_sparse_mla_partial_h32_s8_m64"];
        let manifest = serde_json::json!({"programs":names.iter().enumerate().map(|(i, name)|
            serde_json::json!({"name":name,"scratch_bytes_at_capacity":{"scratch":i as u64 * 1024}}))
            .collect::<Vec<_>>()});
        assert_eq!(manifest_context(&manifest, "dsv4f", 64, 64),
            Some(V4Context { topk_scratch: 0, partial_scratch: 6144 }));
        for missing in 0..names.len() {
            let mut incomplete = manifest.clone();
            incomplete["programs"].as_array_mut().unwrap().remove(missing);
            assert!(manifest_context(&incomplete, "dsv4f", 64, 64).is_none());
        }
        assert!(manifest_context(&manifest, "dsv4p", 128, 64).is_none());
        assert!(manifest_context(&manifest, "dsv4f", 64, 128).is_none());
        assert!(manifest_context(&manifest, "dsv4f", 0, 64).is_none());
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
