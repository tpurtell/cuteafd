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
    /// Coordinator-only diagnostics replace routed FFNs with zero intake planes.
    pub skip_routed_experts: bool,
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
    if inputs.skip_routed_experts && (!inputs.experts.is_empty() || inputs.spark_ranks != 0
        || inputs.prefill_lanes != 1 || inputs.onboard != Onboard::Layers(0)) {
        return Err(PlacementError::Inventory("GLM skipped experts require serial diagnostic admission without local/Spark experts"));
    }
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
            if inputs.skip_routed_experts {
                fixed.push(Demand::new(gpu, Category::Transport, "diagnostic zero expert planes",
                    SKIP_RANKS as u64 * (inputs.prefill_rows.max(DECODE_ROWS) * cfg.hidden as u64 * 2).max(FLOOR),
                    Basis::Formula));
            }
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
        experts: if inputs.skip_routed_experts { None } else {
            layer.checked_sub(cfg.first_moe_layer).map(|i| inputs.experts.get(i).copied().unwrap_or(ExpertCost {
                whole: Bytes2::default(), half: [Bytes2::default(); 2], tp2: false, spark_ok: true }))
        },
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

/// Executable EXL3 TP2 inventory; native NVFP4 and TP1 are deliberately not admitted.
#[derive(Debug, Clone)]
pub struct LocalInventory {
    pub package: std::path::PathBuf,
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
    let common = payload + routes + wide_exchange + FLOOR;
    // Rank 1 normally owns 256-byte placeholders for logits, ids, weights and wire.
    let peer_routes = [experts * 4, topk * 4, topk * 4, h + h / 32].into_iter().map(|width|
        (DECODE_ROWS * width).max(FLOOR) - FLOOR
            + lanes * ((rows * width).max(FLOOR) - FLOOR)).sum::<u64>();
    [common, common + peer_routes]
}

pub fn local_inventory(catalog: &crate::OfficialV41Catalog, manifest: &std::path::Path,
    selected_layers: usize, rows: u64, lanes: u64, exchange_f32: bool) -> anyhow::Result<LocalInventory> {
    let exl3 = catalog.exl3().ok_or_else(|| anyhow::anyhow!(
        "GLM RTX TP2 currently requires EXL3; native NVFP4 needs the BF16 routed/shared partial adapter"))?;
    let shape = catalog.routed_experts();
    anyhow::ensure!(shape.hidden == 6144 && shape.intermediate == 2048 && shape.experts == 256
        && shape.topk == 8 && (1..=4096).contains(&rows) && (1..=4).contains(&lanes)
        && (shape.first_layer..=shape.layers).contains(&selected_layers), "GLM TP2 geometry/layer extent");
    let tiers = exl3.decoder_tiers().iter().map(usize::to_string).collect::<String>();
    let stem = format!("exl3-glm-k{tiers}");
    let parent = manifest.parent().ok_or_else(|| anyhow::anyhow!("GLM program manifest parent"))?;
    let package = [parent.join("exl3").join(&stem), parent.join("../lib/exl3").join(&stem)]
        .into_iter().map(|p| p.join("rtx-tp2")).find(|p| p.is_dir())
        .ok_or_else(|| anyhow::anyhow!("missing {stem}/rtx-tp2 package"))?;
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
            for file in ["v41_exl3_routes.json", "libv41_exl3_routes.so"] {
                anyhow::ensure!(directory.join(file).is_file(), "GLM TP2 missing {file} in {}", directory.display());
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
    Ok(LocalInventory { package, experts, backend_workspace,
        extra_workspace: local_extra_workspace(shape.hidden as u64, shape.experts as u64,
            shape.topk as u64, rows, lanes, exchange_f32) })
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
            headroom_bytes: 2 * GIB, spark_ranks: 4, skip_routed_experts: false, prefill_rows: 4096, prefill_lanes: 3,
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
                "trellis_lut": {"file": "lut.bin", "bytes": 32}, "buffers": {
                    "scratch": {"allocation": "scratch", "bytes": capacity * 64,
                        "dtype": "f32", "zero_on_create": false},
                    "state": {"allocation": "state", "bytes": 0,
                        "dtype": "i32", "zero_on_create": true}}});
            std::fs::write(package.join("v41_exl3.json"), serde_json::to_vec(&value).unwrap()).unwrap();
            // Header-only fixtures do not attest or execute these stand-in assets.
            for file in ["libcuteafd_exl3.so", "lut.bin", "v41_exl3_routes.json", "libv41_exl3_routes.so"] {
                std::fs::write(package.join(file), b"").unwrap();
            }
        }
        (dir, catalog, manifest)
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
                    assert_eq!(local.extra_workspace[0], payloads + route_slots + widened_exchange + FLOOR);
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
        for asset in ["libcuteafd_exl3.so", "lut.bin", "v41_exl3_routes.json", "libv41_exl3_routes.so"] {
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
