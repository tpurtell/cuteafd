//! DeepSeek V4 (Flash/Pro) onto the shared solver: the one place that turns
//! V4's cache geometry, workspace formulas, exchange slots and expert
//! residency into a [`Contract`]. `plan --layout` and serve-dsv4 both call
//! [`prepare`] and [`production_request`]; they differ only in a
//! [`V4Inputs::gpus`] baseline.
//!
//! V4's reserve envelope, the 262K legacy expert reserve and the per-device
//! pool sizing live here, once.
mod admission;
mod host;
pub use admission::{prepare, production_request, AdmittedV4, PreparedV4};
pub use host::{pool_units, HostPrefix, PinnedTables, V4Host};

use crate::families::deepseek_v4::DeepseekV4Config;
use crate::placement::*;
use crate::placement::inventory::GraphSet;
use crate::serving_capacity::{deepseek_v4_cache_geometry, deepseek_v4_expert_cost, deepseek_v4_expert_exchange_bytes,
    deepseek_v4_layer_unit_bytes, deepseek_v4_peer_exchange_bytes, V4ExpertCost,
    V4WorkspaceRank};
use cuteafd_core::memory_layout::{Basis, Category};

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

const fn gib(hundredths: u64) -> u64 { hundredths * (1 << 30) / 100 }

/// Everything V4 admission depends on, gathered by the caller from the
/// checkpoint, the program manifest and the CLI.
#[derive(Debug, Clone)]
pub struct V4Inputs {
    pub expert_usage: bool,
    pub resident_weight_limit: Option<ResidentWeights>,
    pub spark_reply_format: cuteafd_core::SparkReplyFormat,
    /// Shared model extent and optional operator context pin.
    pub serving_context: crate::placement::planner::Context,
    pub cfg: DeepseekV4Config,
    /// The admitted target/dSpark weight arenas and their Load-only startup owners.
    pub weights: crate::families::deepseek_v4::resident::WeightInventory,
    /// dSpark stages that hold caches (all the checkpoint's stages).
    pub cache_stages: usize,
    /// Per-GPU cap and baseline; one or two entries. Each keeps the memory
    /// policy's named reserve (`placement::reserve::gpu("deepseek_v4")`) free.
    pub gpus: Vec<(u64, Baseline)>,
    pub spark_ranks: usize,
    pub sequences: u64,
    pub prefill_rows: u64,
    pub prefill_waves: u64,
    pub decode_rows: u64,
    pub max_context: u64,
    /// Prefix mark slots (`MarkArena::slots_for`), 0 without a prefix cache.
    pub mark_slots: u64,
    /// Exact per-rank step workspaces from the program manifest, or `None`
    /// for the calibrated allowance.
    pub workspace: Option<Vec<V4WorkspaceRank>>,
    /// Per-layer whole-layer residency (backbone routed layers in order),
    /// empty when routed experts never stay on RTX.
    pub experts: Vec<V4ExpertCost>,
    /// Whole dSpark costs; used on GPU0 unless `draft_half` selects TP2.
    pub draft: Vec<V4ExpertCost>,
    /// dSpark halves in stage order; empty for the single-GPU/explicit-off arena.
    pub draft_half: Vec<[V4ExpertCost; 2]>,
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
    /// The build's C4 token-split programs (`manifest_context`), required for
    /// automatic or explicit context placement on two GPUs.
    pub context: Option<V4Context>,
    /// `--attention-placement`: `None` is auto (the family's recorded default).
    pub attention: Option<AttentionPlacement>,
    /// Owners a caller adds after the family's own facts (an RTX media tower).
    /// They are priced exactly like a fixed demand, so the same admission runs.
    pub extra: Vec<Demand>,
    /// Each GPU's physical capacity, apart from the logical ceiling in `gpus`;
    /// empty when they are the same card size.
    pub physical_bytes: Vec<u64>,
    /// The coordinator host tier; `None` only in unit fixtures.
    pub host: Option<V4Host>,
}

/// Native C4 context kernel scratch, read from the selected AOT manifest.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct V4Context {
    pub topk_scratch: u64,
    pub partial_scratch: u64,
}

/// Return no context capability when either head range or any gather/merge primitive is absent.
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
        ("gather tables", (prefill_lanes + crate::serving_capacity::V4_PACKED_MEMBERS as u64) * 2 * half * 4), ("local selected lengths", decode_rows * 4),
        ("empty window lengths", decode_rows * 4)];
    out.extend((0..prefill_lanes + crate::serving_capacity::V4_PACKED_MEMBERS as u64).map(|_| ("commit blocks", commit)));
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

/// Peak live CUDA streams in one coordinator context at Ready (#1511): the
/// engine's stream and the prefix cache's copy stream on the rank, one intake
/// copy stream per Spark link, and -- when the retained host tier is live --
/// the host copy engine's store/restore pair plus its event probe per device.
/// Load, rotary and push streams retire before Ready. Only the band matters
/// (the driver retains a 2 MiB slab per four live streams): 2 and 4 streams
/// stay in the first band, 6 and 8 in the second.
pub fn startup_peak_live_streams(host_tier: bool, spark_links: bool) -> u64 {
    let serving = 1 + 1 + if spark_links { 2 } else { 0 };
    serving + if host_tier { 4 } else { 0 }
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

/// Sum one owner's exact device requests.
fn requests_bytes(what: &'static str, requests: &[u64]) -> Result<u64, PlacementError> {
    requests.iter().try_fold(0u64, |sum, &bytes| sum.checked_add(bytes))
        .ok_or(PlacementError::Overflow(what))
}

/// The rank's Spark intake allocations: one combined prefill return plane per
/// rank, then one decode intake plane per rank (`SparkIntake::with_waves`).
fn intake_requests(inputs: &V4Inputs, rank: usize) -> Result<Vec<u64>, PlacementError> {
    if rank != 0 || inputs.spark_ranks == 0 { return Ok(Vec::new()); }
    let (hidden, ranks) = (inputs.cfg.dim as u64, inputs.spark_ranks as u64);
    let mut requests = crate::serving_capacity::prefill_return_requests_with_format(
        inputs.prefill_waves, inputs.prefill_rows, ranks, hidden, inputs.spark_reply_format,
    )
    .map_err(|_| PlacementError::Inventory("V4 Spark reply width or allocation overflow"))?;
    requests.extend(crate::serving_capacity::spark_intake_requests(
        1, ranks, inputs.decode_rows, hidden));
    Ok(requests)
}

/// The rank's draft route probe: one contiguous device ring.
fn probe_requests(inputs: &V4Inputs, rank: usize) -> Result<Vec<u64>, PlacementError> {
    if rank == 0 && !inputs.draft.is_empty() {
        Ok(vec![draft_probe_bytes(&inputs.cfg, inputs.decode_rows)?])
    } else {
        Ok(Vec::new())
    }
}

/// The "steps" demand's exact device requests: this rank's workspace
/// builders plus its intake planes and route probe. `None` when the caller
/// supplied no exact workspace geometry for this rank.
pub fn steps_requests(inputs: &V4Inputs, rank: usize) -> Result<Option<Vec<u64>>, PlacementError> {
    let Some(workspace) = inputs.workspace.as_ref().and_then(|ranks| ranks.get(rank)) else {
        return Ok(None);
    };
    let mut requests = workspace.requests.clone();
    requests.extend(intake_requests(inputs, rank)?);
    requests.extend(probe_requests(inputs, rank)?);
    Ok(Some(requests))
}

/// Fixed V4 demands plus each GPU's reserve, the ones both sides previously
/// computed apart. Every pool-owned term (per-layer records, lane page tables)
/// is a separate exact owner in `Recipe::prepare`, so nothing here scales with
/// the pool.
pub fn fixed_demands(inputs: &V4Inputs) -> Result<(Vec<Demand>, Vec<u64>), PlacementError> {
    let gpus = inputs.gpus.len();
    let geometry = deepseek_v4_cache_geometry(&inputs.cfg, gpus, inputs.prefill_rows, inputs.cache_stages)
        .map_err(|_| PlacementError::Inventory("V4 cache geometry"))?;
    let hidden = inputs.cfg.dim as u64;
    let topk = inputs.cfg.n_activated_experts as u64;
    let tables = inputs.prefill_waves.checked_mul(crate::serving_capacity::V4_PACKED_MEMBERS as u64)
        .and_then(|n| n.checked_add(inputs.decode_rows)).ok_or(PlacementError::Overflow("V4 packed lane tables"))?;
    let table_bytes = tables.checked_mul(4).ok_or(PlacementError::Overflow("V4 lane tables"))?;
    let mut demands = Vec::new();
    let mut reserves = Vec::new();
    for (rank, cache) in geometry.ranks.iter().enumerate() {
        let gpu = rank as u8;
        let role = if gpus == 1 { 0 } else if rank == 0 { 1 } else { 2 };
        let mul = |a: u64, b: u64, what| a.checked_mul(b).ok_or(PlacementError::Overflow(what));
        // Window rings, compressor carry and the per-sequence records unit are
        // named per-layer physical owners in `Recipe::prepare`; this keeps the
        // pool-independent replay and lane-table state.
        let state = cache.fixed_state_bytes + cache.speculative_replay_bytes
            + mul(table_bytes, inputs.sequences, "V4 tables")?;
        let context = mul(cache.context_table_bytes_per_token, inputs.max_context, "V4 RoPE tables")?;
        let intake = requests_bytes("V4 Spark intake", &intake_requests(inputs, rank)?)?;
        let exchange = if gpus == 2 {
            deepseek_v4_expert_exchange_bytes(hidden, topk, inputs.prefill_rows, inputs.decode_rows, rank, inputs.prefill_waves)
                .map_err(|_| PlacementError::Overflow("V4 expert exchange"))?
        } else { 0 };
        let probe = requests_bytes("V4 draft route probe", &probe_requests(inputs, rank)?)?;
        let workspace = inputs.workspace.as_ref().and_then(|w| w.get(rank)).map(|w| w.fixed_device_bytes)
            .unwrap_or(WORKSPACE_BYTES[role] * inputs.prefill_rows / 4096) + intake + probe;
        reserves.push(crate::placement::reserve::gpu("deepseek_v4"));
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
                deepseek_v4_peer_exchange_bytes(hidden, inputs.prefill_rows, inputs.decode_rows, true, inputs.exchange_f32, inputs.prefill_waves)
                    .map_err(|_| PlacementError::Overflow("V4 peer exchange"))?, Basis::Formula));
            demands.push(Demand::new(gpu, Category::Transport, "expert peer exchange", exchange, Basis::Formula));
        }
        if rank == 0 && inputs.full_prefill_logits > 0 {
            demands.push(Demand::new(gpu, Category::Workspace, "probe prefill logits", inputs.full_prefill_logits,
                Basis::Formula));
        }
    }
    Ok((demands, reserves))
}

/// The code-inventory package key of a V4 catalog: EXL3 packages load their own modules (the
/// coordinator holds dSpark and any RTX layers in them), native MXFP4 does not.
pub fn code_experts(catalog: &crate::OfficialV41Catalog) -> &'static str {
    if catalog.exl3().is_some() { "exl3" } else { "*" }
}

/// Selected-program module bytes per rank. BLAS handles, modules and driver
/// storage are disjoint named owners charged by the admission contract; this
/// no longer carries a mixed residency sample (#1579).
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
    use crate::placement::planner::Refusal;

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
    fn combined_reply_formats_reuse_bf16_capacity_for_flash_and_pro() {
        for (hidden, heads, topk) in [(4096, 64, 512), (7168, 128, 1024)] {
            let mut cfg = context_cfg();
            cfg.dim = hidden;
            cfg.n_heads = heads;
            cfg.index_topk = topk;
            for gpus in [1, 2] {
                let mut input = V4Inputs {
                    expert_usage: false, resident_weight_limit: None,
                    spark_reply_format: cuteafd_core::SparkReplyFormat::Bf16, serving_context: crate::placement::planner::Context::pinned(262_144),
                    cfg: cfg.clone(), weights: test_weights(gpus),
                    cache_stages: 0, gpus: vec![(96 << 30, Baseline::Planned {
                        context_bytes: 0, loaded_bytes: 0 }); gpus],
                    spark_ranks: 6, sequences: 16, prefill_rows: 2752, prefill_waves: 3,
                    decode_rows: 64, max_context: 262144, mark_slots: 0,
                    workspace: None, experts: Vec::new(), draft: Vec::new(), draft_half: Vec::new(), experts_half: Vec::new(),
                    tp2_workspace: [0; 2], exchange_f32: false, expert_workspace: 0,
                    first_routed: 0, requested_pool: Some(262144), onboard: Onboard::Layers(0),
                    full_prefill_logits: 0, code_bytes: Vec::new(), context: None, attention: None,
                    extra: Vec::new(), physical_bytes: Vec::new(), host: None,
                };
                let baseline = fixed_demands(&input).unwrap().0;
                for format in [cuteafd_core::SparkReplyFormat::Bf16, cuteafd_core::SparkReplyFormat::Fp8,
                    cuteafd_core::SparkReplyFormat::Nvfp4] {
                    input.spark_reply_format = format;
                    let fixed = fixed_demands(&input).unwrap().0;
                    for rank in 0..gpus {
                        let steps = |items: &[Demand]| items.iter().find(|d|
                            d.gpu == rank as u8 && d.group == "steps").unwrap().bytes;
                        assert_eq!(steps(&fixed) - steps(&baseline),
                            0);
                    }
                }
            }
        }
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
                assert_eq!(workspace.iter().filter(|&&(name, _)| name == "commit blocks").count(), 2 + crate::serving_capacity::V4_PACKED_MEMBERS);
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

    /// One minimal admitted target arena per GPU, no drafter.
    fn test_weights(gpus: usize) -> crate::families::deepseek_v4::resident::WeightInventory {
        crate::families::deepseek_v4::resident::WeightInventory {
            target: (0..gpus).map(|rank| crate::serving_capacity::WeightArenaLayout::new(
                format!("v4/target/{rank}"), [("norm.weight".to_string(), 1 << 20)]).unwrap()).collect(),
            draft: None, startup: Default::default() }
    }

    fn plan_inputs(cfg: &DeepseekV4Config) -> V4Inputs {
        V4Inputs {
            expert_usage: false, resident_weight_limit: None, spark_reply_format: cuteafd_core::SparkReplyFormat::Bf16,
            serving_context: crate::placement::planner::Context::pinned(262_144), cfg: cfg.clone(),
            weights: test_weights(2),
            cache_stages: 0, gpus: vec![(96 << 30, Baseline::Planned { context_bytes: 0, loaded_bytes: 0 }); 2],
            spark_ranks: 6, sequences: 1, prefill_rows: 4096, prefill_waves: 3,
            decode_rows: 64, max_context: 262144, mark_slots: 0,
            workspace: None, experts: Vec::new(), draft: Vec::new(), draft_half: Vec::new(), experts_half: Vec::new(),
            tp2_workspace: [0; 2], exchange_f32: false, expert_workspace: 0, first_routed: 0,
            requested_pool: Some(262144), onboard: Onboard::Layers(0), full_prefill_logits: 0,
            code_bytes: Vec::new(), context: None, attention: None, extra: Vec::new(),
            physical_bytes: Vec::new(), host: None,
        }
    }

    fn admitted(input: &V4Inputs) -> Result<Placement, Refusal> {
        let prepared = prepare(input)?;
        let request = production_request(input);
        prepared.admit(request).map(|admitted| admitted.placement)
    }

    #[test]
    fn tp2_draft_is_mandatory_at_zero_onboarding_and_shares_one_workspace() {
        let mut input = plan_inputs(&context_cfg());
        input.attention = Some(AttentionPlacement::Heads);
        input.expert_workspace = 1024;
        input.tp2_workspace = [2048; 2];
        let full = V4ExpertCost { resident_bytes: 8 << 20, staging_bytes: 4096 };
        let half = V4ExpertCost { resident_bytes: 4 << 20, staging_bytes: 2048 };
        input.draft = vec![full; 3];
        input.experts = vec![full; 2];
        input.experts_half = vec![[half; 2]; 2];
        let off = admitted(&input).unwrap();
        assert!(off.tp2.is_none());
        assert!(off.expert_ranges[0].peak_bytes > 0);
        input.draft_half = vec![[half; 2]; 3];
        let payload = crate::serving_capacity::deepseek_v4_tp2_payload_bytes(
            input.cfg.dim as u64, input.prefill_rows, input.decode_rows, input.prefill_waves).unwrap();
        for count in 0..=2 {
            input.onboard = Onboard::Layers(count);
            let prepared = prepare(&input).unwrap();
            let result = prepared.admit(production_request(&input)).unwrap();
            assert!(result.solve_us > 0, "prepare and admission work must be timed");
            let tp2 = result.placement.tp2.unwrap();
            let expected = 2048 + payload + (3 + count as u64) * half.resident_bytes + half.staging_bytes;
            assert_eq!(tp2.peak_bytes, [expected; 2]);
            assert_eq!(tp2.layers, count);
            assert!(result.placement.expert_ranges.iter().all(|r| r.peak_bytes == 0));
            for gpu in 0..2 {
                let actual: u64 = result.placement.items[gpu].iter().filter(|i| i.group == "tp2").map(|i| i.bytes).sum();
                assert_eq!(actual, expected);
            }
            let ids = result.solved.facts.allocations.iter().map(|a| &a.id).collect::<std::collections::BTreeSet<_>>();
            assert_eq!(ids.len(), result.solved.facts.allocations.len());
        }
        input.draft_half.pop();
        assert!(prepare(&input).is_err());
        input.draft_half.push([half; 2]);
        input.gpus.truncate(1);
        assert!(prepare(&input).is_err());
    }

    #[test]
    fn telemetry_geometry_is_shared_with_recipe_and_excludes_dense_prefix() {
        let cfg = context_cfg();
        let mut input = plan_inputs(&cfg);
        input.attention = Some(AttentionPlacement::Heads);
        input.expert_usage = true;
        input.first_routed = 1;
        let prepared = prepare(&input).unwrap();
        let geometry = prepared.expert_usage().unwrap();
        assert_eq!(geometry.routed_layers, (1..cfg.n_layers).collect::<Vec<_>>());
        let contract = planner::Recipe::prepare(&prepared, &production_request(&input)).unwrap();
        let expected = crate::placement::expert_usage::components(geometry, &[0], contract.maximum_pages).unwrap();
        for component in expected {
            let planner::ComponentKind::Allocation { template: expected, .. } = component.kind else { panic!("allocation"); };
            assert!(contract.components.iter().any(|component| matches!(&component.kind,
                planner::ComponentKind::Allocation { template, .. } if template == &expected)));
        }
        input.expert_usage = false;
        assert!(prepare(&input).unwrap().expert_usage().is_none());
    }

    #[test]
    fn auto_requires_matching_context_programs_but_explicit_heads_always_plans() {
        for (family, hidden, heads) in [("dsv4f", 4096, 64), ("dsv4p", 7168, 128)] {
            let mut cfg = context_cfg();
            cfg.dim = hidden;
            cfg.n_heads = heads;
            let names = ["dsa_candidate_merge", "lse_combine2", "paged_staging_gather_c4",
                "paged_staging_gather_index"].map(|suffix| format!("{family}_{suffix}"))
                .into_iter().chain([format!("{family}_scored_index_topk_decode_m64"),
                    format!("{family}_sparse_mla_partial_h0_s4_m64"),
                    format!("{family}_sparse_mla_partial_h{}_s4_m64", heads / 2)]).collect::<Vec<_>>();
            let complete = serde_json::json!({"programs": names.iter().map(|name|
                serde_json::json!({"name": name})).collect::<Vec<_>>()});
            let mut manifests = vec![serde_json::json!({}),
                serde_json::json!({"programs": []}),
                serde_json::from_str(&complete.to_string().replace(family,
                    if family == "dsv4f" { "dsv4p" } else { "dsv4f" })).unwrap(),
                serde_json::from_str(&complete.to_string().replace("_m64", "_m128")).unwrap()];
            for missing in 0..names.len() {
                let mut manifest = complete.clone();
                manifest["programs"].as_array_mut().unwrap().remove(missing);
                manifests.push(manifest);
            }
            for manifest in manifests {
                let mut input = plan_inputs(&cfg);
                input.context = manifest_context(&manifest, family, heads, 64);
                assert!(input.context.is_none());
                for attention in [None, Some(AttentionPlacement::Context)] {
                    input.attention = attention;
                    let Refusal::InvalidFact(message) = admitted(&input).unwrap_err() else {
                        panic!("expected the missing context programs, got {:?}", admitted(&input))
                    };
                    assert!(message.contains(&format!(
                        "{family}_sparse_mla_partial_h{}_s*_m64", heads / 2)), "{message}");
                }
                input.attention = Some(AttentionPlacement::Heads);
                let heads_plan = admitted(&input).unwrap();
                input.context = manifest_context(&complete, family, heads, 64);
                assert_eq!(heads_plan, admitted(&input).unwrap(),
                    "explicit heads replicates and charges the same KV regardless of context programs");
                assert!(heads_plan.layers.iter().all(|layer| layer.mode == LayerMode::HeadSplit));
                input.attention = None;
                let auto = admitted(&input).unwrap();
                assert_eq!(auto.attention_placement, AttentionPlacement::Context);
                for gpu in 0..2 {
                    let records = |plan: &Placement| plan.items[gpu].iter().filter(|item| item.group == "records")
                        .map(|item| item.bytes).sum::<u64>();
                    assert!(records(&heads_plan) > records(&auto));
                }
            }
            let mut one = plan_inputs(&cfg);
            one.gpus.truncate(1);
            one.weights = test_weights(1);
            assert_eq!(admitted(&one).unwrap().attention_placement, AttentionPlacement::Heads);
            assert!(super::super::DEEPSEEK_V4.check_attention(Some(AttentionPlacement::Layers), 2, true).is_err());
        }
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

/// Stage halves use V4's backbone TP2 package and exact same-geometry residency.
pub fn draft_half_costs(catalog: &crate::OfficialV41Catalog, stages: usize) -> anyhow::Result<Vec<[V4ExpertCost; 2]>> {
    (0..stages).map(|stage| {
        let layer = catalog.routed_experts().layers + stage;
        Ok([crate::serving_capacity::deepseek_v4_tp2_expert_cost(catalog, layer, 0)?,
            crate::serving_capacity::deepseek_v4_tp2_expert_cost(catalog, layer, 1)?])
    }).collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn startup_streams_price_each_ready_band() {
        let slab = crate::placement::planner::cuda_stream_slab_bytes;
        // Engine and prefix-copy streams, plus one pair when Sparks carry experts.
        assert_eq!(startup_peak_live_streams(false, false), 2);
        assert_eq!(startup_peak_live_streams(false, true), 4);
        assert_eq!(slab(2).unwrap(), 2 << 20);
        assert_eq!(slab(4).unwrap(), 2 << 20);
        // The host tier's copy engine crosses into the second band.
        assert_eq!(startup_peak_live_streams(true, false), 6);
        assert_eq!(startup_peak_live_streams(true, true), 8);
        assert_eq!(slab(6).unwrap(), 4 << 20);
        assert_eq!(slab(8).unwrap(), 4 << 20);
    }
}
