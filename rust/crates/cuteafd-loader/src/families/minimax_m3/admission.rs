//! One fact gatherer for MiniMax M3 admission (#1856 H1). `cuteafd plan
//! --layout` and serving startup both call [`gather`]; they differ only in
//! [`Coordinator`] (a planned card forecast vs one measured CUDA sample per
//! GPU). The result is the shared recipe
//! [`MiniMaxInputs`](crate::placement::families::minimax_m3::MiniMaxInputs),
//! which `placement::planner::solve` admits. No family code samples memory or
//! sizes a pool here.
use crate::placement::families::minimax_m3::{self as shared, LayerExperts, LocalPackage, MiniMaxInputs, SparkCharge};
use crate::placement::planner::{serving, Budget, Context, PrefillShape};
use crate::placement::{ArchContext, ResidentWeights, SparkWorkingSet};
use anyhow::{ensure, Context as _};
use std::{path::Path, sync::Arc};

use super::{weight_layout::CoordinatorWeights, MiniMaxM3Config};

/// A refusal both edges report the same way: what failed and how to fix it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{what}")]
pub struct Refused {
    pub what: String,
    pub how: String,
}

fn refused(what: impl Into<String>, how: impl Into<String>) -> Refused { Refused { what: what.into(), how: how.into() } }

/// The coordinator edge: planned cards or measured samples.
#[derive(Debug, Clone)]
pub enum Coordinator {
    /// Logical caps per GPU and the planner's physical card class.
    Planned { cards: Vec<crate::placement::CoordinatorCard> },
    /// One CUDA sample per GPU before weights: (physical total, logical cap, free).
    Measured { samples: Vec<(u64, u64, u64)>, loaded_code: Vec<u64> },
}

impl Coordinator {
    pub fn gpus(&self) -> usize {
        match self { Self::Planned { cards } => cards.len(), Self::Measured { samples, .. } => samples.len() }
    }
    fn physical(&self, gpu: usize) -> u64 {
        match self { Self::Planned { cards } => cards[gpu].physical_bytes, Self::Measured { samples, .. } => samples[gpu].0 }
    }
    fn budget(&self, gpu: usize) -> Budget {
        match self {
            Self::Planned { cards } => shared::planned_budget(gpu as u8, cards[gpu].physical_bytes,
                cards[gpu].logical_bytes, ArchContext::for_device("sm_120", cards[gpu].physical_bytes).context_bytes),
            Self::Measured { samples, .. } => shared::measured_budget(gpu as u8, samples[gpu].0, samples[gpu].1,
                samples[gpu].2),
        }
    }
}

/// Every operator input admission reads, resolved once by the caller.
pub struct FactsRequest<'a> {
    pub checkpoint: &'a crate::plan::Checkpoint,
    pub cfg: &'a MiniMaxM3Config,
    pub catalog: &'a crate::OfficialV41Catalog,
    pub manifest_path: &'a Path,
    pub manifest: Arc<serde_json::Value>,
    pub coordinator: Coordinator,
    pub spark_ranks: usize,
    pub spark_bytes: u64,
    pub prefill: PrefillShape,
    /// Explicit context, or 0 for the checkpoint's maximum.
    pub context_tokens: u64,
    pub pool_tokens: Option<u64>,
    pub sequences: u64,
    pub host: (u64, u64),
    pub host_cache_bytes: Option<u64>,
    pub prefix_entries: u64,
    pub replies: cuteafd_core::spark_reply::SparkReplyFormats,
    pub head: serving::BlasHead,
    pub expert_usage: bool,
    pub resident_weight_limit: ResidentWeights,
}

/// Header-exact coordinator residency of one routed layer, whole and as TP2
/// halves, from the checkpoint's own compressed NVFP4 headers.
pub fn layer_experts(catalog: &crate::OfficialV41Catalog) -> anyhow::Result<Vec<LayerExperts>> {
    crate::placement::rtx_onboard::expert_costs(catalog, true)?.into_iter().map(|cost| {
        ensure!(cost.whole_supported && cost.tp2, "MiniMax RTX expert layer lacks a whole or TP2 package layout");
        Ok(LayerExperts { whole: cost.whole.resident, half: [cost.half[0].resident, cost.half[1].resident] })
    }).collect()
}

/// The installed coordinator package's executor workspace at the unit rows.
pub fn local_package(catalog: &crate::OfficialV41Catalog, manifest: &Path, rows: u64, gpus: usize) -> LocalPackage {
    match crate::placement::rtx_onboard::expert_workspace(catalog, Some(manifest), rows, gpus, "minimax") {
        Ok(bytes) => LocalPackage::Installed { workspace: if gpus == 2 { [bytes, bytes] } else { [bytes, 0] } },
        Err(error) => LocalPackage::Missing(format!("{error:#}")),
    }
}

/// Spark charges per onboarding count and rank: header-exact native NVFP4
/// slices, endpoint I/O, typed execution owners (installed TP scratch census)
/// and the shared worker runtime residual.
pub fn spark_charges(checkpoint: &crate::plan::Checkpoint, catalog: &crate::OfficialV41Catalog, routed: &[usize],
    ranks: usize, prefill: PrefillShape) -> anyhow::Result<Vec<Vec<SparkCharge>>> {
    if ranks == 0 { return Ok(vec![Vec::new(); routed.len() + 1]); }
    let recipe = crate::plan::experts::checkpoint_spark_recipe(checkpoint, ranks, routed)?;
    let executions = shared::spark_executions(prefill)?;
    let endpoint = cuteafd_core::SparkEndpoint { ring_depth: None, capacity: u32::try_from(prefill.rows)?,
        ingress: cuteafd_core::SparkIngress::Bf16, partial_reply_bytes: None };
    let rings = crate::plan::layout::spark_ring_geometry(shared::LANES as usize);
    let rings = cuteafd_core::SparkRings { max_frame_bytes: rings.max_frame_bytes, depth: rings.depth,
        slot_bytes: rings.slot_bytes, alignment: rings.alignment };
    let runtime = cuteafd_core::memory_layout::spark_runtime_reserve_bytes(0).context("Spark runtime residual")?;
    let geometry = cuteafd_core::ExpertGeometry::MINIMAX_M3;
    // One installed NVFP4 group per rank: equal native slices, one scratch arena.
    let widths = recipe.stored_widths.clone();
    let mut per_layer = vec![vec![0u64; routed.len()]; ranks];
    for (index, &layer) in routed.iter().enumerate() {
        let working = SparkWorkingSet { rtx_layers: vec![], spark_layers: vec![layer], spark_ranks: ranks,
            prefill_lanes: shared::LANES };
        for rank in 0..ranks {
            // Compressed-tensors NVFP4 is an fp8-family catalog: the worker's
            // own `SparkAdmission::plan_view` selects the same Fp8 layout.
            let layout = crate::placement::SparkResidentLayout::Fp8 { world: ranks, rank, slicing: recipe.slicing };
            per_layer[rank][index] = crate::placement::SparkAssignment::resident_from_catalog(catalog, &working,
                rank, layout)?.resident_bytes;
        }
    }
    (0..=routed.len()).map(|count| {
        let working = SparkWorkingSet { rtx_layers: routed[..count].to_vec(), spark_layers: routed[count..].to_vec(),
            spark_ranks: ranks, prefill_lanes: shared::LANES };
        (0..ranks).map(|rank| {
            if count == routed.len() { return Ok(SparkCharge { experts: 0, owners: 0, runtime: 0 }); }
            let io = working.rank_endpoints_io(rank, geometry, &[endpoint], rings)?;
            let width = u32::try_from(*widths.get(rank).context("Spark recipe stored width")?)?;
            let bytes = |extent| crate::plan::layout::spark_slots::nvfp4_owner_bytes("minimax_m3", geometry, width, extent);
            let owners = executions.workspace_bytes(bytes)?.and_then(|bytes| bytes.total())
                .context("MiniMax Spark owner sum overflow")?;
            let owners = [io.intake, io.host_exchange, io.row_indices, io.mapped_rings, io.endpoint_staging, owners]
                .into_iter().try_fold(0u64, |sum, bytes| sum.checked_add(bytes)).context("MiniMax Spark owners")?;
            Ok(SparkCharge { experts: per_layer[rank][count..].iter().sum(), owners, runtime })
        }).collect()
    }).collect()
}

/// Gather every admitted owner fact into the shared recipe.
pub fn gather(request: &FactsRequest<'_>) -> Result<MiniMaxInputs, Refused> {
    let cfg = request.cfg;
    let gpus = request.coordinator.gpus();
    let ranks = request.spark_ranks;
    if !(1..=2).contains(&gpus) {
        return Err(refused(format!("MiniMax runs on one RTX or an RTX pair, not {gpus}"), "Use --rtx 1 or --rtx 2."));
    }
    let context = if request.context_tokens == 0 { cfg.max_position as u64 } else { request.context_tokens };
    if context > cfg.max_position as u64 {
        return Err(refused(format!("MiniMax context {context} exceeds the checkpoint's {}", cfg.max_position),
            "Lower --max-context-tokens."));
    }
    let shape = request.prefill;
    let routed = (0..cfg.layers).filter(|&layer| cfg.moe[layer]).collect::<Vec<_>>();
    let owners = shared::layer_owners(cfg.layers, gpus);
    let targets = CoordinatorWeights::plan(request.checkpoint, cfg, gpus,
        &owners.iter().map(|&gpu| usize::from(gpu)).collect::<Vec<_>>())
        .map_err(|error| refused(format!("MiniMax target arena inventory refused: {error:#}"),
            "Provide complete native MXFP8 checkpoint headers."))?;
    let mount = targets.iter().map(|target| target.mount_load(request.checkpoint)).collect::<anyhow::Result<Vec<_>>>()
        .map_err(|error| refused(format!("MiniMax mount inventory refused: {error:#}"), "Check checkpoint headers."))?;
    let experts = layer_experts(request.catalog).map_err(|error| refused(
        format!("MiniMax RTX expert residency refused: {error:#}"), "Provide the compressed-tensors NVFP4 expert headers."))?;
    let local = local_package(request.catalog, request.manifest_path, shape.rows, gpus);
    let spark = spark_charges(request.checkpoint, request.catalog, &routed, ranks, shape)
        .map_err(|error| refused(format!("MiniMax Spark inventory refused: {error:#}"),
            format!("Use a Spark world with a minimax:nvfp4 package ({:?}).",
                crate::plan::experts::spark_worlds("minimax:nvfp4a4", cfg.moe_intermediate))))?;
    let links = shared::spark_link((gpus - 1) as u8, ranks, shape, request.replies)
        .map_err(|error| refused(format!("MiniMax Spark transport owners refused: {error}"), "Use a supported unit."))?;
    let programs = crate::placement::ProgramSet::from_manifest(&request.manifest, &["minimax", "exlr8"]);
    let mut code = Vec::new();
    let mut graph_reserve = Vec::new();
    for gpu in 0..gpus {
        code.push(match &request.coordinator {
            Coordinator::Planned { .. } => crate::placement::inventory::loaded_code_allowance("minimax", "nvfp4",
                gpus == 2, gpu as u8, ArchContext::for_device("sm_120", request.coordinator.physical(gpu)),
                Some(&programs)).bytes,
            // The measured sample follows module loading: only what it has not seen.
            Coordinator::Measured { loaded_code, .. } => loaded_code.get(gpu).copied().unwrap_or(0),
        });
        // No traced MiniMax prewarm layout yet: base zero, the shared runtime
        // headroom is the whole canonical reserve (graph-cache stream, H11).
        graph_reserve.push(crate::placement::graph_reserve::current(gpu, false, 0, 0)
            .map_err(|error| refused(format!("MiniMax graph reserve refused: {error}"), "Fix GPU_GRAPH_RESERVE."))?.bytes);
    }
    let mut budgets = (0..gpus).map(|gpu| request.coordinator.budget(gpu)).collect::<Vec<_>>();
    budgets.extend((0..ranks).map(|rank| shared::spark_budget(rank as u8, request.spark_bytes)));
    let (host_total, host_available) = request.host;
    budgets.push(crate::placement::families::kimi::host_budget(host_total, host_available));
    let page_bytes = shared::PAGE_ROWS * (32768 + 32768 + 8448) / shared::PAGE_ROWS * cfg.layers as u64;
    let host_prefix = if request.host_cache_bytes == Some(0) || request.prefix_entries == 0 {
        serving::HostPrefixOwners { quota_bytes: 0, chunk_bytes: 0 }
    } else {
        let geometry = cuteafd_core::prefix::HostCacheGeometry { unit_rows: shared::PAGE_ROWS,
            page_bytes, mark_bytes: 0, draft_bytes: 0 };
        let quota = geometry.resolve(request.host_cache_bytes, context, request.sequences, host_total,
            geometry.chunk_bytes()).unwrap_or(0);
        serving::HostPrefixOwners { quota_bytes: quota, chunk_bytes: geometry.chunk_bytes().min(quota.max(1)) }
    };
    let physical = (0..gpus).map(|gpu| request.coordinator.physical(gpu)).collect::<Vec<_>>();
    let maximum_pool_tokens = crate::placement::PoolPolicy::resolve(&physical, context, request.pool_tokens,
        shared::PAGE_ROWS).maximum_tokens();
    Ok(MiniMaxInputs { cfg: cfg.clone(), gpus, spark_ranks: ranks,
        context: Context::resolve(context, (request.context_tokens > 0).then_some(request.context_tokens)),
        maximum_pool_tokens, prefill: shape, sequences: request.sequences, targets, owners,
        manifest: request.manifest.clone(), routed: routed.clone(), experts, local, spark, links,
        spark_reply_formats: request.replies, drafter: None, vision: None, budgets, code, graph_reserve,
        head: request.head, host_prefix, device_sampling: true,
        expert_usage: request.expert_usage.then(|| cuteafd_core::expert_usage::Geometry { layers: cfg.layers,
            experts: cfg.experts, topk: cfg.topk, routed_layers: routed }),
        resident_weight_limit: Some(request.resident_weight_limit), mount })
}
