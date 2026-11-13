//! One fact gatherer for Inkling 2 admission (#1930 stage 1). `cuteafd plan
//! --layout` and serving startup both call [`gather`]; they differ only in
//! [`Coordinator`] (a planned card forecast vs one measured CUDA sample per
//! GPU). The result is the shared recipe
//! [`InklingInputs`](crate::placement::families::inkling2::InklingInputs),
//! which `placement::planner::solve` admits. No family code samples memory or
//! sizes a pool here.
use crate::placement::families::inkling2::{self as shared, InklingInputs, LocalPackage, SparkCharge};
use crate::placement::planner::{serving, Budget, Context, PrefillShape};
use crate::placement::{ArchContext, ResidentWeights, SparkWorkingSet};
use crate::plan::families::inkling2::ExpertStorage;
use anyhow::Context as _;
use std::{path::Path, sync::Arc};

use super::{experts::{layer_experts, LayerExperts}, weight_layout::CoordinatorWeights, Inkling2Config};

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
    pub cfg: &'a Inkling2Config,
    pub manifest_path: Option<&'a Path>,
    pub manifest: Option<Arc<serde_json::Value>>,
    pub coordinator: Coordinator,
    /// The resolved attention placement (`check_attention` on the plan edge,
    /// the launcher's resolved value at serve).
    pub attention: crate::placement::AttentionPlacement,
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

/// The installed coordinator package's executor workspace at the unit rows.
/// No Inkling 2 RTX expert package ships yet (#1930 stage 3), so this names
/// the missing package; the recipe then refuses onboarding counts above zero
/// and the plan says so.
pub fn local_package(storage: &[ExpertStorage], manifest: Option<&Path>, gpus: usize) -> LocalPackage {
    let formats = storage.iter().map(|s| match s {
        ExpertStorage::Mxfp8 => "fp8-inkling2-mxfp8", ExpertStorage::Nvfp4 => "fp8-inkling2-nvfp4a4",
        ExpertStorage::Exl3 { .. } => "exl3-inkling2-k4" }).collect::<std::collections::BTreeSet<_>>();
    let parent = manifest.and_then(Path::parent).unwrap_or(Path::new("/opt/cuteafd/share"));
    let installed = formats.iter().all(|name| {
        let dir = if name.starts_with("exl3") { parent.join(name) } else { parent.join("fp8").join(name) };
        dir.join("manifest.json").is_file() && (1..=gpus).all(|tp| dir.join(format!("{}tp{tp}", if name.starts_with("exl3") { "rtx-" } else { "" })).is_dir())
    });
    if installed {
        // Workspace sizing from the installed manifest lands with the package (stage 3).
        return LocalPackage::Missing(format!("Inkling 2 RTX expert packages {formats:?} are installed but their executor \
            workspace is not yet admitted; onboarding qualifies in stage 3 (#1930)"));
    }
    LocalPackage::Missing(format!("no installed Inkling 2 RTX expert package {formats:?} beside {} (#1930 stage 3)",
        parent.display()))
}

/// Spark charges per onboarding count and rank: header-exact expert slices
/// (whole intermediate blocks per rank), endpoint I/O, typed execution owners
/// sized by the shared segment formulas at the covering export, and the
/// shared worker runtime residual. The per-package scratch census joins with
/// the package (stage 3); until then the scratch term is the formula bound.
pub fn spark_charges(cfg: &Inkling2Config, experts: &[LayerExperts], routed: &[usize], ranks: usize,
    prefill: PrefillShape, ingress: cuteafd_core::SparkIngress) -> anyhow::Result<Vec<Vec<SparkCharge>>> {
    if ranks == 0 { return Ok(vec![Vec::new(); routed.len() + 1]); }
    anyhow::ensure!((2..=8).contains(&ranks), "Inkling 2 Spark worlds are 2..=8 ranks, not {ranks}");
    let exl3 = experts.iter().any(|layer| matches!(layer.storage, Some(ExpertStorage::Exl3 { .. })));
    let executions = shared::spark_executions(prefill, exl3)?;
    let endpoint = cuteafd_core::SparkEndpoint { ring_depth: None, capacity: u32::try_from(prefill.rows)?,
        ingress, partial_reply_bytes: None };
    let rings = crate::plan::layout::spark_ring_geometry(shared::LANES as usize);
    let rings = cuteafd_core::SparkRings { max_frame_bytes: rings.max_frame_bytes, depth: rings.depth,
        slot_bytes: rings.slot_bytes, alignment: rings.alignment };
    let runtime = cuteafd_core::memory_layout::spark_runtime_reserve_bytes(0).context("Spark runtime residual")?;
    let geometry = cuteafd_core::ExpertGeometry::INKLING2;
    // Equal intermediate slices over whole 128-row blocks: 2304 / 128 = 18
    // blocks, so worlds 2..=6 and 9 split exactly and 7/8 take the ceiling.
    let blocks = cfg.intermediate / 128;
    let widest_blocks = blocks.div_ceil(ranks);
    let per_layer = experts.iter().map(|layer| {
        // The widest rank's share of the layer's checkpoint bytes.
        layer.source.checked_mul(widest_blocks as u64).and_then(|b| b.checked_div(blocks as u64))
            .context("Inkling 2 Spark slice overflow")
    }).collect::<anyhow::Result<Vec<_>>>()?;
    (0..=routed.len()).map(|count| {
        let working = SparkWorkingSet { rtx_layers: routed[..count].to_vec(), spark_layers: routed[count..].to_vec(),
            spark_ranks: ranks, prefill_lanes: shared::LANES };
        (0..ranks).map(|rank| {
            if count == routed.len() { return Ok(SparkCharge { experts: 0, owners: 0, runtime: 0 }); }
            let io = working.rank_endpoints_io(rank, geometry, &[endpoint], rings)?;
            let width = u32::try_from(widest_blocks * 128)?;
            let owners = executions.workspace_bytes(|extent| owner_bytes(geometry, width, extent, ingress))?
                .and_then(|bytes| bytes.total()).context("Inkling 2 Spark owner sum overflow")?;
            let owners = [io.intake, io.host_exchange, io.row_indices, io.mapped_rings, io.endpoint_staging, owners]
                .into_iter().try_fold(0u64, |sum, bytes| sum.checked_add(bytes)).context("Inkling 2 Spark owners")?;
            Ok(SparkCharge { experts: per_layer[count..].iter().sum(), owners, runtime })
        }).collect()
    }).collect()
}

/// One FP8-family owner at the shared segment formulas: streamed intermediate
/// scratch at the compiled export, stream-ordered I/O at the owner's capacity
/// and packed route staging per queued unit. The installed package's measured
/// census replaces the scratch term when the package ships (stage 3).
fn owner_bytes(geometry: cuteafd_core::ExpertGeometry, width: u32, extent: cuteafd_core::OwnerExtent,
    ingress: cuteafd_core::SparkIngress) -> anyhow::Result<cuteafd_core::OwnerBytes> {
    let rows = extent.compiled;
    let scratch = cuteafd_core::spark_executions::segment_scratch_bytes(geometry.hidden, geometry.experts, geometry.topk,
        width.checked_mul(2).context("activation width overflow")?, rows, cuteafd_core::spark_executions::SEGMENT_MAX_ROWS)
        .context("Inkling 2 segment scratch overflow")?;
    let bf16 = !matches!(ingress, cuteafd_core::SparkIngress::Fp8K32);
    let hidden = u64::from(geometry.hidden);
    let input = u64::from(extent.capacity) * if bf16 { hidden * 2 } else { hidden + hidden / 32 };
    let io = input + u64::from(extent.capacity) * (8 * u64::from(geometry.topk) + 2 * hidden)
        + u64::from(rows) * (16 + u64::from(geometry.topk) * 8);
    Ok(cuteafd_core::OwnerBytes { scratch, io, pinned_staging: u64::from(extent.staging_rows) * u64::from(geometry.topk) * 8 })
}

/// Gather every admitted owner fact into the shared recipe.
pub fn gather(request: &FactsRequest<'_>) -> Result<InklingInputs, Refused> {
    let cfg = request.cfg;
    let gpus = request.coordinator.gpus();
    let ranks = request.spark_ranks;
    if !(1..=2).contains(&gpus) {
        return Err(refused(format!("Inkling 2 runs on one RTX or an RTX pair, not {gpus}"), "Use --rtx 1 or --rtx 2."));
    }
    let context = if request.context_tokens == 0 { cfg.max_position as u64 } else { request.context_tokens };
    if context > cfg.max_position as u64 {
        return Err(refused(format!("Inkling 2 context {context} exceeds the checkpoint's {}", cfg.max_position),
            "Lower --max-context-tokens."));
    }
    let shape = request.prefill;
    let routed = (0..cfg.layers).collect::<Vec<_>>();
    let placement = if gpus == 1 { crate::placement::AttentionPlacement::Heads } else { request.attention };
    let modes = shared::layer_modes(cfg, gpus, placement);
    let targets = CoordinatorWeights::plan(request.checkpoint, cfg, gpus, &modes)
        .map_err(|error| refused(format!("Inkling 2 target arena inventory refused: {error:#}"),
            "Provide complete BF16 core headers (every shard of the selected view)."))?;
    let mount = targets.iter().map(|target| target.mount_load(request.checkpoint)).collect::<anyhow::Result<Vec<_>>>()
        .map_err(|error| refused(format!("Inkling 2 mount inventory refused: {error:#}"), "Check checkpoint headers."))?;
    let experts = layer_experts(request.checkpoint, cfg).map_err(|error| refused(
        format!("Inkling 2 routed expert inventory refused: {error:#}"),
        "Provide every routed layer's fused MXFP8/NVFP4 headers or per-expert EXL3 modules."))?;
    let storage = experts.iter().map(|layer| layer.storage.expect("inventory names every layer's storage")).collect::<Vec<_>>();
    let local = local_package(&storage, request.manifest_path, gpus);
    // Mixed native checkpoints (NVFP4 then MXFP8) keep the per-layer native
    // ingress; the link is sized for its widest request row (BF16).
    let ingress = if storage.iter().all(|s| *s == ExpertStorage::Mxfp8) { cuteafd_core::SparkIngress::Fp8K32 }
        else if storage.iter().any(|s| *s == ExpertStorage::Mxfp8) { cuteafd_core::SparkIngress::Native }
        else { cuteafd_core::SparkIngress::Bf16 };
    let spark = spark_charges(cfg, &experts, &routed, ranks, shape, ingress)
        .map_err(|error| refused(format!("Inkling 2 Spark inventory refused: {error:#}"),
            "Use a Spark world of 2..=8 ranks."))?;
    let links = shared::spark_link((gpus - 1) as u8, ranks, shape, request.replies, ingress)
        .map_err(|error| refused(format!("Inkling 2 Spark transport owners refused: {error}"), "Use a supported unit."))?;
    let programs = request.manifest.as_ref().map(|manifest|
        crate::placement::ProgramSet::from_manifest(manifest, &["inkling2", "exlr8"]));
    let mut code = Vec::new();
    let mut graph_reserve = Vec::new();
    for gpu in 0..gpus {
        code.push(match &request.coordinator {
            Coordinator::Planned { .. } => crate::placement::inventory::loaded_code_allowance("inkling2", "experts",
                gpus == 2, gpu as u8, ArchContext::for_device("sm_120", request.coordinator.physical(gpu)),
                programs.as_ref()).bytes,
            // The measured sample follows module loading: only what it has not seen.
            Coordinator::Measured { loaded_code, .. } => loaded_code.get(gpu).copied().unwrap_or(0),
        });
        // No traced Inkling 2 prewarm layout yet: base zero, the shared runtime
        // headroom is the whole canonical reserve (graph-cache stage 4).
        graph_reserve.push(crate::placement::graph_reserve::current(gpu, false, 0, 0)
            .map_err(|error| refused(format!("Inkling 2 graph reserve refused: {error}"), "Fix GPU_GRAPH_RESERVE."))?.bytes);
    }
    let mut budgets = (0..gpus).map(|gpu| request.coordinator.budget(gpu)).collect::<Vec<_>>();
    budgets.extend((0..ranks).map(|rank| shared::spark_budget(rank as u8, request.spark_bytes)));
    let (host_total, host_available) = request.host;
    budgets.push(crate::placement::families::kimi::host_budget(host_total, host_available));
    let page_rows = if modes.iter().any(|mode| *mode == crate::placement::LayerMode::ContextSplit) { 2 * shared::PAGE_ROWS }
        else { shared::PAGE_ROWS };
    // Host prefix pages hold the global layers' K/V; marks hold the rings and
    // conv lags of every layer (byte-identical restores).
    let kv_row = (cfg.kv_heads * cfg.head_dim * 2) as u64;
    let page_bytes = shared::PAGE_ROWS * 2 * kv_row * cfg.global_layers().len() as u64;
    let mark_bytes = (0..cfg.layers).map(|layer| shared::SequenceState::new(cfg, cfg.local[layer])
        .map(|state| state.ring_bytes + state.lag_bytes)).sum::<Result<u64, _>>()
        .map_err(|error| refused(format!("Inkling 2 sequence state refused: {error}"), "Check the window geometry."))?;
    let host_prefix = if request.host_cache_bytes == Some(0) || request.prefix_entries == 0 {
        serving::HostPrefixOwners { quota_bytes: 0, chunk_bytes: 0 }
    } else {
        let geometry = cuteafd_core::prefix::HostCacheGeometry { unit_rows: page_rows,
            page_bytes: page_bytes * page_rows / shared::PAGE_ROWS, mark_bytes, draft_bytes: 0 };
        let quota = geometry.resolve(request.host_cache_bytes, context, request.sequences, host_total,
            geometry.chunk_bytes()).unwrap_or(0);
        serving::HostPrefixOwners { quota_bytes: quota, chunk_bytes: geometry.chunk_bytes().min(quota.max(1)) }
    };
    let physical = (0..gpus).map(|gpu| request.coordinator.physical(gpu)).collect::<Vec<_>>();
    let maximum_pool_tokens = crate::placement::PoolPolicy::resolve(&physical, context, request.pool_tokens,
        page_rows).maximum_tokens();
    Ok(InklingInputs { cfg: cfg.clone(), gpus, spark_ranks: ranks,
        context: Context::resolve(context, (request.context_tokens > 0).then_some(request.context_tokens)),
        maximum_pool_tokens, prefill: shape, sequences: request.sequences, targets, placement, modes,
        routed: routed.clone(), experts, local, spark, links,
        spark_reply_formats: request.replies, drafter: None, vision: None, budgets, code, graph_reserve,
        head: request.head, host_prefix, device_sampling: true,
        expert_usage: request.expert_usage.then(|| cuteafd_core::expert_usage::Geometry { layers: cfg.layers,
            experts: cfg.experts, topk: cfg.topk, routed_layers: routed }),
        resident_weight_limit: Some(request.resident_weight_limit), mount })
}
