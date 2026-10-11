//! GLM 5.3 coordinator admission shared by the planner and runtime. The current
//! executor runs replicated MLA/DSA KV with head-split attention on two GPUs;
//! context and layer ownership require their own executor before admission.
use crate::families::glm5::{GlmDsaConfig, GlmIndexer};
use crate::placement::*;
use crate::serving_capacity::{glm_cache_geometry, glm_decode_graph_allowance};
use cuteafd_core::memory_layout::{Basis, Category};

pub const PAGE_ROWS: u64 = 64;
pub const DECODE_ROWS: u64 = 64;
pub const DEFAULT_PREFILL_LANES: u64 = 3;
const FLOOR: u64 = 256;
const GIB: u64 = 1 << 30;

/// Scratch retained separately by each decode/prefill workspace.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GlmScratch {
    pub programs: u64,
    pub topk: u64,
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
    let mut optional = vec![format!("glm_index_producer_bf16_{cap}"), format!("{prefix}_producer_bf16_{cap}"),
        format!("{prefix}_o_bf16_{cap}"), format!("{prefix}_ffn_i{moe}_bf16_{cap}")];
    if !decode { optional.push(format!("{prefix}_ffn_i{dense}_pt_{cap}")); }
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
    pub prefill_rows: u64,
    pub prefill_lanes: u64,
    pub max_context: u64,
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
}

pub fn request(inputs: &GlmInputs<'_>) -> Result<PlacementRequest, PlacementError> {
    let gpus = inputs.gpus.len();
    let cfg = inputs.cfg;
    if !(1..=2).contains(&gpus) || !(1..=4).contains(&inputs.prefill_lanes)
        || inputs.prefill_rows == 0 || inputs.prefill_rows > 4096 || inputs.max_context == 0 {
        return Err(PlacementError::Inventory("GLM rank/lane/step/context geometry"));
    }
    let geometry = glm_cache_geometry(cfg, inputs.layers, gpus)
        .map_err(|_| PlacementError::Inventory("GLM cache geometry"))?;
    let routed = inputs.layers.saturating_sub(cfg.first_moe_layer);
    if !inputs.experts.is_empty() && inputs.experts.len() != routed {
        return Err(PlacementError::Inventory("GLM costs must cover every selected routed layer"));
    }
    if gpus == 2 && inputs.experts.iter().any(|c| !c.tp2) {
        return Err(PlacementError::Inventory("GLM two-GPU local experts require TP2 halves"));
    }
    let mut fixed = Vec::new();
    let mut pool_overhead = Vec::new();
    let table_bytes = (DECODE_ROWS + inputs.prefill_lanes) * 4;
    let graphs = glm_decode_graph_allowance(inputs.max_context as usize, inputs.layers)
        .map_err(|_| PlacementError::Overflow("GLM graphs"))?;
    for (rank, cache) in geometry.ranks.iter().enumerate() {
        let gpu = rank as u8;
        fixed.push(Demand::new(gpu, Category::Kv, "padded decode scratch page", cache.persistent_unit_bytes, Basis::Formula));
        fixed.push(Demand::new(gpu, Category::Kv, "RoPE context tables",
            cache.context_table_bytes_per_token.checked_mul(inputs.max_context)
                .ok_or(PlacementError::Overflow("GLM RoPE"))?, Basis::Formula));
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
        fixed.push(Demand::new(gpu, Category::Workspace, "steps", steps, basis));
        fixed.push(Demand::new(gpu, Category::Runtime, "decode graph allowance", graphs, Basis::Estimated));
        if let Some(&bytes) = inputs.pending_code.get(rank).filter(|&&bytes| bytes > 0) {
            fixed.push(Demand::new(gpu, Category::Runtime, "pending loaded code", bytes, Basis::Calibrated));
        }
        if gpus == 2 {
            let slots = 4 * inputs.prefill_lanes;
            let bytes = slots * inputs.prefill_rows.max(DECODE_ROWS) * cfg.hidden as u64 * 2
                + ((slots + 1) * 16).max(FLOOR);
            fixed.push(Demand::new(gpu, Category::Transport, "peer exchange", bytes, Basis::Formula));
        }
        if rank == 0 {
            if inputs.spark_ranks > 0 {
                let endpoints = 1 + if inputs.prefill_lanes > 1 { inputs.prefill_lanes } else { 0 };
                fixed.push(Demand::new(gpu, Category::Transport, "Spark intake planes",
                    endpoints * inputs.spark_ranks as u64 * 4096 * cfg.hidden as u64 * 2, Basis::Formula));
            }
            if inputs.drafter_bytes > 0 {
                fixed.push(Demand::new(gpu, Category::Drafter, "drafter owned storage",
                    inputs.drafter_bytes, Basis::Formula));
                fixed.push(Demand::new(gpu, Category::Drafter, "drafter load staging",
                    inputs.drafter_staging, Basis::Formula));
            }
        }
        pool_overhead.push(table_bytes);
    }
    let layers = (0..inputs.layers).map(|layer| LayerDemand {
        kind: AttentionClass::Dsa,
        weights: ModeBytes::default(),
        kv_unit: {
            let unit = PAGE_ROWS * 656 + if cfg.indexers[layer] == GlmIndexer::Full { 8448 } else { 0 };
            KvDemand { unit_bytes_whole: unit, unit_bytes_split: [unit; 2],
                unit_bytes_context: Some([unit.div_ceil(2), unit / 2]) }
        },
        fixed_bytes: ModeBytes::default(),
        context_indexer: cfg.indexers[layer] == GlmIndexer::Full,
        colocate: Some(cfg.index_source(layer) as u16),
        // Preserve routed identity even without local kernels: Spark-free and
        // explicit local requests must fail, not turn these into dense layers.
        experts: layer.checked_sub(cfg.first_moe_layer).map(|i| inputs.experts.get(i).copied().unwrap_or(ExpertCost {
            whole: Bytes2::default(), half: [Bytes2::default(); 2], tp2: false, spark_ok: true })),
        modes: if gpus == 2 { vec![LayerMode::HeadSplit] } else { vec![LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }] },
    }).collect();
    let cards: Vec<_> = inputs.gpus.iter().map(|g| g.0).collect();
    Ok(PlacementRequest {
        attention_placement: None,
        layers_first_gpu: 0,
        context_buffers: ContextBuffers { staging_unit_bytes: PAGE_ROWS * 788, staging_unit_rows: PAGE_ROWS,
            query_row_bytes: 36_864, partial_row_bytes: 32_896, candidate_row_bytes: cfg.index_topk as u64 * 8,
            compiled_extent: inputs.max_context, decode_rows: DECODE_ROWS, lanes: 1 },
        inventory: Inventory { gpus: inputs.gpus.iter().map(|&(capacity_bytes, baseline)| GpuBudget {
            capacity_bytes, baseline, headroom_bytes: inputs.headroom_bytes }).collect(),
            spark_ranks: inputs.spark_ranks, peer_access: gpus == 2 },
        pool: PoolPolicy { ceiling: 1 << 31, ..PoolPolicy::resolve(&cards, inputs.max_context,
            inputs.requested_pool, PAGE_ROWS, inputs.spark_ranks == 0) },
        layers, pool_overhead, fixed, movables: Vec::new(),
        expert_workspace: inputs.expert_workspace, tp2_workspace: inputs.tp2_workspace,
        onboard: inputs.onboard, expert_gpus: usize::from(!inputs.experts.is_empty()),
        policy: LayerPolicy { default: vec![LayerMode::HeadSplit, LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }], by_kind: Vec::new() },
        hops: HopSpec { row_bytes: cfg.hidden as u64 * 2, rows: inputs.prefill_rows.max(DECODE_ROWS),
            lanes: inputs.prefill_lanes, entry_gpu: 0, head_gpu: 0 },
        executor: super::GLM5,
    })
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
            headroom_bytes: 2 * GIB, spark_ranks: 4, prefill_rows: 4096, prefill_lanes: 3,
            max_context: 1 << 20, scratch: Some([GlmScratch { programs: 16 << 20, topk: 1 << 20 }; 2]),
            drafter_bytes: 3 * GIB, drafter_staging: 0, pending_code: vec![],
            experts: (cfg.first_moe_layer..cfg.layers).map(|_| ExpertCost {
                whole: Bytes2 { resident: 4 * GIB, staging: 0 }, half: [Bytes2 { resident: 2 * GIB, staging: 0 }; 2],
                tp2: cards.len() == 2, spark_ok: true }).collect(),
            expert_workspace: 128 << 20, tp2_workspace: [128 << 20; 2], requested_pool: None,
            onboard: default_onboard(), full_prefill_logits: false }
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
                let p = solve(&request(&planned).unwrap()).unwrap();
                let r = solve(&request(&runtime).unwrap()).unwrap();
                assert_eq!(p, r);
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
    fn replicated_heads_cannot_admit_two_million_tokens() {
        let cfg = config();
        let mut i = inputs(&cfg, &[96 * GIB; 2]);
        i.requested_pool = Some(2 << 20);
        assert!(matches!(solve(&request(&i).unwrap()), Err(PlacementError::PoolDoesNotFit { .. })));
        let geometry = glm_cache_geometry(&cfg, cfg.layers, 2).unwrap();
        assert_eq!(geometry.ranks[0].persistent_unit_bytes / PAGE_ROWS, 53940);
        assert_eq!(geometry.ranks[0], geometry.ranks[1]);
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
        assert_eq!(r.executor.attention_default(), AttentionPlacement::Heads);
        let mut context = r;
        context.attention_placement = Some(AttentionPlacement::Context);
        assert!(matches!(solve(&context), Err(PlacementError::AttentionPlacement { .. })));
    }

    #[test]
    fn scratch_selection_includes_optional_programs_and_refuses_missing_required() {
        let cfg = config();
        let lookup = |name: &str| Some(if name.contains("bf16") { 500 } else if name.contains("topk") { 80 } else { 100 });
        assert_eq!(step_scratch(&cfg, true, false, lookup).unwrap(), GlmScratch { programs: 500, topk: 80 });
        assert!(step_scratch(&cfg, false, true, |_| None).is_err());
        assert_eq!(default_onboard(), Onboard::Auto);
    }
}
