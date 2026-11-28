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
    pub vision: crate::plan::MediaMode,
    pub vision_replicas: usize,
    pub media_cache_bytes: Option<u64>,
    pub checkpoint: &'a crate::plan::Checkpoint,
    pub catalog: &'a crate::OfficialV41Catalog,
    pub cfg: &'a Inkling2Config,
    pub kv_format: super::InklingKvCache,
    pub drafter: Option<super::draft::Selection>,
    pub manifest_path: Option<&'a Path>,
    pub manifest: Option<Arc<serde_json::Value>>,
    pub coordinator: Coordinator,
    /// The resolved attention placement (`check_attention` on the plan edge,
    /// the launcher's resolved value at serve).
    pub attention: crate::placement::AttentionPlacement,
    pub spark_ranks: usize,
    pub spark_bytes: u64,
    pub prefill: PrefillShape,
    /// `--decode-graphs` (default on): off charges no graph reserve.
    pub decode_graphs: bool,
    /// Admitted per-lane decode/verify row budget (`DECODE_ROWS`, 1..=128).
    pub decode_rows: u32,
    /// Explicit context, or 0 for the checkpoint's maximum.
    pub context_tokens: u64,
    pub pool_tokens: Option<u64>,
    pub target_pool_tokens: Option<u64>,
    pub sequences: u64,
    pub host: (u64, u64),
    pub host_cache_bytes: Option<u64>,
    pub prefix_entries: u64,
    pub prefix_mark_budget: u64,
    pub replies: cuteafd_core::spark_reply::SparkReplyFormats,
    pub head: serving::BlasHead,
    pub expert_usage: bool,
    pub resident_weight_limit: ResidentWeights,
}

/// Shared installed executor owners, never a family scratch estimate. Each
/// independently retained native backend owns its scratch/output once.
pub fn local_package(catalog: &crate::OfficialV41Catalog, manifest: Option<&Path>,
    gpus: usize, rows: u64) -> LocalPackage {
    let workspace = || -> anyhow::Result<LocalPackage> {
        anyhow::ensure!((1..=2).contains(&gpus), "Inkling local package requires one or two ranks");
        let routed = catalog.routed_experts();
        let tp2_rows = |wire, partial| {
            (gpus == 2).then(|| crate::serving_capacity::OwnerTp2Rows::with_partial(
                routed.hidden as u64, routed.topk as u64, rows, wire, partial)).transpose()
        };
        let layers = (routed.first_layer..routed.layers).collect::<Vec<_>>();
        let exl3_requests = if catalog.exl3().is_some() { Some(crate::placement::rtx_onboard::expert_workspace_requests(
            catalog, manifest, rows, gpus, "inkling2")?) } else { None };
        // Residency constructs descriptor/load tables; compute every physical
        // layer/rank once, not once for every onboarding prefix containing it.
        let exl3_regions = if let Some(exl3) = catalog.exl3() {
            (0..gpus).map(|gpu| layers.iter().map(|&layer| {
                let layout = exl3.residency(crate::V41Exl3Layer::Backbone(layer), gpus, gpu)?;
                let name = if gpus == 1 { format!("BackboneFull {{ layer: {layer} }}.0") }
                    else { format!("BackboneTp2 {{ layer: {layer}, rank: {gpu} }}.0") };
                Ok((name, layout.device_arena_layout()?.1))
            }).collect::<anyhow::Result<Vec<_>>>()).collect::<anyhow::Result<Vec<_>>>()?
        } else { Vec::new() };
        let mut native_requests = std::collections::BTreeMap::<String, (usize, usize)>::new();
        let mut backings = vec![Vec::new()];
        for count in 1..=layers.len() {
            let selected = &layers[..count];
            let mut owners = Vec::new();
            if let Some(tensors) = catalog.fp8() {
                for (view, group_layers) in tensors.homogeneous_layers(selected)? {
                    let key = format!("{:?}", view.format());
                    let (scratch, output) = if let Some(&requests) = native_requests.get(&key) { requests } else {
                        let requests = crate::placement::rtx_onboard::fp8_workspace_requests(&view,
                            manifest, rows, gpus, "inkling2")?;
                        let scratch = *requests.first().context("native package scratch absent")?;
                        let (scratch, output, _) = crate::serving_capacity::fp8moe_tp2_workspace(usize::try_from(scratch)?,
                            routed.hidden, usize::try_from(rows)?)?;
                        native_requests.insert(key, (scratch, output));
                        (scratch, output)
                    };
                    for gpu in 0..gpus {
                        let regions = crate::serving_capacity::fp8_expert_regions(&view.for_coordinator(),
                            &group_layers, gpus, gpu, crate::formats::fp8_experts::Slicing::Blocks(128))?;
                        let id = crate::serving_capacity::expert_weight_arena_id("fp8", gpu as i32, &regions)?;
                        owners.push(crate::placement::families::inkling2::LocalBacking { gpu,
                            layers: group_layers.clone(), arena: crate::serving_capacity::WeightArenaLayout::new(id, regions)?,
                            workspace: vec![("scratch".into(), scratch as u64), ("output".into(), output as u64)],
                            tp2_rows: tp2_rows(view.format() == crate::formats::fp8_experts::ExpertFormat::Mxfp8, 2)? });
                    }
                }
            } else {
                let requests = exl3_requests.as_ref().context("Inkling EXL3 workspace absent")?;
                for gpu in 0..gpus {
                    let regions = exl3_regions[gpu][..count].to_vec();
                    let id = crate::serving_capacity::expert_weight_arena_id("exl3", gpu as i32, &regions)?;
                    let last = requests.len().checked_sub(1).context("EXL3 output owner absent")?;
                    let workspace = requests.iter().enumerate().map(|(index, &bytes)|
                        (if index == last { "output".into() } else { format!("scratch.{index}") }, bytes)).collect();
                    owners.push(crate::placement::families::inkling2::LocalBacking { gpu, layers: selected.to_vec(),
                        arena: crate::serving_capacity::WeightArenaLayout::new(id, regions)?, workspace, tp2_rows: tp2_rows(true, 4)? });
                }
            }
            backings.push(owners);
        }
        Ok(LocalPackage::Installed { backings })
    };
    workspace().unwrap_or_else(|error| LocalPackage::Missing(format!("{error:#}")))
}

/// Spark charges use the shared catalog's physical resident layout, the
/// shared typed-owner census, and endpoint I/O once per rank. Missing census
/// inputs remain explicit errors; no checkpoint-byte fraction sizes scratch.
pub fn spark_charges(catalog: &crate::OfficialV41Catalog, experts: &[LayerExperts], routed: &[usize], ranks: usize,
    prefill: PrefillShape, ingress: cuteafd_core::SparkIngress) -> anyhow::Result<Vec<Vec<SparkCharge>>> {
    spark_charges_for_rows(catalog, experts, routed, ranks, prefill, shared::DECODE_ROWS, ingress)
}

pub fn spark_charges_for_rows(catalog: &crate::OfficialV41Catalog, experts: &[LayerExperts], routed: &[usize], ranks: usize,
    prefill: PrefillShape, decode_rows: u32, ingress: cuteafd_core::SparkIngress) -> anyhow::Result<Vec<Vec<SparkCharge>>> {
    if ranks == 0 { return Ok(vec![Vec::new(); routed.len() + 1]); }
    anyhow::ensure!((2..=8).contains(&ranks), "Inkling 2 Spark worlds are 2..=8 ranks, not {ranks}");
    let exl3 = experts.iter().any(|layer| matches!(layer.storage, Some(ExpertStorage::Exl3 { .. })));
    let executions = shared::spark_executions_for_rows(prefill, exl3, decode_rows)?;
    let endpoint = cuteafd_core::SparkEndpoint { ring_depth: None, capacity: u32::try_from(prefill.rows)?,
        ingress, partial_reply_bytes: None };
    let rings = crate::plan::layout::spark_ring_geometry(shared::LANES as usize);
    let rings = cuteafd_core::SparkRings { max_frame_bytes: rings.max_frame_bytes, depth: rings.depth,
        slot_bytes: rings.slot_bytes, alignment: rings.alignment };
    let runtime = cuteafd_core::memory_layout::spark_runtime_reserve_bytes(0).context("Spark runtime residual")?;
    let geometry = cuteafd_core::ExpertGeometry::INKLING2;
    (0..=routed.len()).map(|count| {
        let working = SparkWorkingSet { rtx_layers: routed[..count].to_vec(), spark_layers: routed[count..].to_vec(),
            spark_ranks: ranks, prefill_lanes: shared::LANES };
        (0..ranks).map(|rank| {
            if count == routed.len() { return Ok(SparkCharge { experts: 0, owners: 0, runtime: 0 }); }
            let io = working.rank_endpoints_io(rank, geometry, &[endpoint], rings)?;
            let owners = executions.workspace_bytes(|extent|
                crate::plan::layout::spark_slots::catalog_owner_bytes(catalog, "inkling2", &working, rank, extent))?
                .and_then(|bytes| bytes.total()).context("Inkling 2 Spark owner sum overflow")?;
            let owners = [io.intake, io.host_exchange, io.row_indices, io.mapped_rings, io.endpoint_staging, owners]
                .into_iter().try_fold(0u64, |sum, bytes| sum.checked_add(bytes)).context("Inkling 2 Spark owners")?;
            let resident = if let Some(tensors) = catalog.fp8() {
                // The shared native worker packs all homogeneous groups in
                // one physical arena, in group/projection order.
                let regions = tensors.homogeneous_layers(&working.spark_layers)?.into_iter()
                    .map(|(view, layers)| -> anyhow::Result<Vec<(String, usize)>> {
                        let format = if view.dynamic_nvfp4() { "nvfp4a4" } else { "mxfp8" };
                        let recipe = crate::plan::experts::spark_recipe(&format!("inkling2:{format}"),
                            catalog.routed_experts().intermediate, ranks)?;
                        crate::serving_capacity::fp8_expert_regions(&view, &layers, ranks, rank, recipe.slicing)
                    }).collect::<anyhow::Result<Vec<_>>>()?.into_iter().flatten();
                crate::serving_capacity::WeightArenaLayout::new("inkling2/spark-residency", regions)?.bytes() as u64
            } else {
                let regions = working.spark_layers.iter().map(|&layer|
                    catalog.exl3_spark_arena_bytes(layer, ranks).map(|bytes| bytes[rank]))
                    .collect::<anyhow::Result<Vec<_>>>()?;
                crate::serving_capacity::PackedWeightRegions::new(regions.into_iter()
                    .map(usize::try_from).collect::<Result<Vec<_>, _>>()?)?.bytes as u64
            };
            Ok(SparkCharge { experts: resident, owners, runtime })
        }).collect()
    }).collect()
}

/// One pool unit's global-layer records on each rank: a wholly owned layer
/// carries the full page, a head- or context-split layer half of it, and a
/// layer owned by the peer none (the shares `rank_owners` charges).
pub(crate) fn pool_memory(cfg: &Inkling2Config, kv_format: super::InklingKvCache,
    modes: &[crate::placement::LayerMode], page_rows: u64, capacities: &[u64])
    -> Vec<crate::placement::PoolMemory> {
    let record_bytes = kv_format.record_bytes(cfg.kv_heads, cfg.head_dim) as u64;
    let rank_unit_bytes = |gpu: usize| -> u64 {
        cfg.local.iter().zip(modes).filter(|(local, _)| !**local).map(|(_, mode)| match mode {
            crate::placement::LayerMode::Whole { gpu: owner, .. } if usize::from(*owner) == gpu =>
                page_rows * record_bytes,
            crate::placement::LayerMode::Whole { .. } => 0,
            crate::placement::LayerMode::HeadSplit | crate::placement::LayerMode::ContextSplit =>
                page_rows * record_bytes / 2,
        }).sum()
    };
    capacities.iter().enumerate().map(|(gpu, &available_bytes)|
        crate::placement::PoolMemory { available_bytes, bytes_per_unit: rank_unit_bytes(gpu) }).collect()
}

/// The one header-derived inventory used both for code preload and admission.
pub struct SelectedWeights {
    modes: Vec<crate::placement::LayerMode>,
    pub targets: Vec<CoordinatorWeights>,
    pub drafter: Option<(crate::formats::block_dspark::recurrent::RecurrentBlockConfig,
        Vec<crate::formats::block_dspark::recurrent::loading::RecurrentWeightLayout>, crate::shared::dense_read::MountLoad)>,
}

impl SelectedWeights {
    pub fn new(checkpoint: &crate::plan::Checkpoint, cfg: &Inkling2Config, gpus: usize,
        attention: crate::placement::AttentionPlacement, draft: bool) -> anyhow::Result<Self> {
        let placement = if gpus == 1 { crate::placement::AttentionPlacement::Heads } else { attention };
        let modes = shared::layer_modes(cfg, gpus, placement);
        let targets = CoordinatorWeights::plan(checkpoint, cfg, gpus, &modes)?;
        let drafter = draft.then(|| super::draft::DraftLayout::select_weights(
            checkpoint.view.as_ref().context("bundled drafter needs selected checkpoint view")?, gpus,
            InklingInputs::head_rank(gpus, &modes))).transpose()?;
        Ok(Self { modes, targets, drafter })
    }

    pub fn compact_programs(&self, manifest: &serde_json::Value, rank: usize, prefill_rows: usize)
        -> anyhow::Result<std::collections::BTreeSet<String>> {
        let target = self.targets.get(rank).context("Inkling target rank missing")?;
        let head_rows = shared::DECODE_ROWS as usize;
        let mut programs = target.trellis_programs(manifest, prefill_rows, head_rows)?;
        programs.extend(target.fp8_programs(manifest, prefill_rows, head_rows)?.into_keys());
        if let Some((config, draft, _)) = &self.drafter {
            let draft = draft.get(rank).context("Inkling drafter rank missing")?;
            programs.extend(draft.trellis_programs(manifest, super::draft::CONTEXT_ROWS)?);
            programs.extend(draft.fp8_programs(manifest, super::draft::CONTEXT_ROWS)?.into_keys());
            if rank == draft.lead {
                if let Some(stored) = target.stored_head() {
                    programs.extend(super::draft::fold_programs(config.transformer.hidden,
                        config.transformer.markov_rank, stored));
                    if let Some(peer_stored) = self.targets.get(1).and_then(CoordinatorWeights::stored_head) {
                        programs.extend(super::draft::fold_programs(config.transformer.hidden,
                            config.transformer.markov_rank, peer_stored));
                    }
                }
            }
        }
        Ok(programs)
    }
}

/// Gather every admitted owner fact into the shared recipe.
pub fn gather(request: &FactsRequest<'_>) -> Result<InklingInputs, Refused> {
    let weights = SelectedWeights::new(request.checkpoint, request.cfg, request.coordinator.gpus(),
        request.attention, request.drafter.is_some()).map_err(|error| refused(
        format!("Inkling 2 target or drafter arena inventory refused: {error:#}"),
        "Provide complete core headers for the selected view."))?;
    gather_with_weights(request, weights)
}

pub fn gather_with_weights(request: &FactsRequest<'_>, weights: SelectedWeights) -> Result<InklingInputs, Refused> {
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
    let decode_rows = request.decode_rows;
    if !(1..=shared::MAX_DECODE_ROWS).contains(&decode_rows) {
        return Err(refused(format!("Inkling 2 decode_rows {decode_rows} exceeds supported 1..={}", shared::MAX_DECODE_ROWS),
            "Set DECODE_ROWS to auto or 1..=128."));
    }
    let routed = (0..cfg.layers).collect::<Vec<_>>();
    let placement = if gpus == 1 { crate::placement::AttentionPlacement::Heads } else { request.attention };
    let body_rows = shape.rows.max(u64::from(decode_rows)) as usize;
    let program_sets = match (&request.coordinator, request.manifest.as_deref()) {
        (Coordinator::Planned { .. }, Some(manifest)) => (0..gpus).map(|rank| {
            let selected = weights.compact_programs(manifest, rank, body_rows)?;
            let mut set = crate::placement::ProgramSet::from_manifest(manifest,
                &["inkling2", "exlr8", "dense_trellis_k6", "native_attention"]);
            let keep = |name: &str| !(name.starts_with("dense_trellis_k6_") || name.starts_with("native_attention_"))
                || selected.contains(name);
            set.names.retain(|name| keep(name));
            set.scratch.retain(|(name, _)| keep(name));
            Ok(set)
        }).collect::<anyhow::Result<Vec<_>>>().map_err(|error| refused(format!("Inkling compact program inventory refused: {error:#}"),
            "Install the Inkling native trellis and block-FP8 projection exports."))?,
        _ => Vec::new(),
    };
    let SelectedWeights { modes, targets, drafter } = weights;
    if targets.len() != gpus || drafter.is_some() != request.drafter.is_some() {
        return Err(refused("Inkling selected weights differ from launch geometry", "Prepare the selected view for the launch ranks."));
    }
    let target_fp8_scratch = targets.iter().map(|target| {
        if target.core.is_empty() && target.fp8.is_empty() { return Ok(0); }
        let manifest = request.manifest.as_deref().context("Inkling compact core requires PROGRAMS.json")?;
        target.trellis_programs(manifest, body_rows, decode_rows.min(64) as usize)?;
        target.fp8_scratch(manifest, body_rows, decode_rows.min(64) as usize)
    }).collect::<anyhow::Result<Vec<_>>>().map_err(|error| refused(format!("Inkling compact program inventory refused: {error:#}"),
        "Install the Inkling native trellis and block-FP8 projection exports."))?;
    let mut mount = targets.iter().map(|target| target.mount_load(request.checkpoint)).collect::<anyhow::Result<Vec<_>>>()
        .map_err(|error| refused(format!("Inkling 2 mount inventory refused: {error:#}"), "Check checkpoint headers."))?;
    let experts = layer_experts(request.checkpoint, cfg).map_err(|error| refused(
        format!("Inkling 2 routed expert inventory refused: {error:#}"),
        "Provide every routed layer's fused MXFP8/NVFP4 headers or per-expert EXL3 modules."))?;
    let storage = experts.iter().map(|layer| layer.storage.expect("inventory names every layer's storage")).collect::<Vec<_>>();
    let catalog = request.catalog;
    let local = local_package(&catalog, request.manifest_path, gpus, shape.rows.max(u64::from(decode_rows)));
    // Mixed native checkpoints (NVFP4 then MXFP8) keep the per-layer native
    // ingress; the link is sized for its widest request row (BF16).
    let first_wire = shared::ingress(storage.first().copied());
    let ingress = if storage.iter().all(|&storage| shared::ingress(Some(storage)) == first_wire) { first_wire }
        else { cuteafd_core::SparkIngress::Native };
    let spark = spark_charges_for_rows(&catalog, &experts, &routed, ranks, shape, decode_rows, ingress)
        .map_err(|error| refused(format!("Inkling 2 Spark inventory refused: {error:#}"),
            "Use a Spark world of 2..=8 ranks."))?;
    let head = InklingInputs::head_rank(gpus, &modes);
    let links = shared::spark_link(head as u8, ranks, shape, request.replies, ingress)
        .map_err(|error| refused(format!("Inkling 2 Spark transport owners refused: {error}"), "Use a supported unit."))?;
    let mut code = Vec::new();
    let mut graph_reserve = Vec::new();
    for gpu in 0..gpus {
        code.push(match &request.coordinator {
            Coordinator::Planned { .. } => crate::placement::inventory::loaded_code_allowance("inkling2", "experts",
                gpus == 2, gpu as u8, ArchContext::for_device("sm_120", request.coordinator.physical(gpu)),
                program_sets.get(gpu)).bytes,
            // The measured sample follows module loading: only what it has not seen.
            Coordinator::Measured { loaded_code, .. } => loaded_code.get(gpu).copied().unwrap_or(0),
        });
        graph_reserve.push(crate::serving_capacity::inkling2_graphs::policy(gpu, gpus, request.decode_graphs)
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
    let page_bytes = shared::PAGE_ROWS * request.kv_format.record_bytes(cfg.kv_heads, cfg.head_dim) as u64
        * cfg.global_layers().len() as u64;
    let mark_bytes = modes.iter().enumerate().map(|(layer, mode)| {
        let parts = if matches!(mode, crate::placement::LayerMode::Whole { .. }) { 1 } else { 2 };
        super::execution::SequenceLayout::new(cfg, cfg.local[layer], parts)
            .map(|layout| (layout.bytes * parts) as u64)
    }).sum::<anyhow::Result<u64>>()
        .map_err(|error| refused(format!("Inkling 2 sequence state refused: {error}"), "Check the window geometry."))?;
    let draft_mark = request.drafter.zip(drafter.as_ref()).map(|(selection, (config, _, _))|
        (0..gpus).map(|rank| config.window_mark_bytes(selection.kv_cache, rank, gpus, head))
            .sum::<anyhow::Result<usize>>()).transpose().map_err(|error|refused(format!("drafter mark inventory refused: {error:#}"),"Check bundled MTP config."))?.unwrap_or(0) as u64;
    let host_prefix = if request.host_cache_bytes == Some(0) || request.prefix_entries == 0 {
        serving::HostPrefixOwners { quota_bytes: 0, chunk_bytes: 0 }
    } else {
        let geometry = cuteafd_core::prefix::HostCacheGeometry { unit_rows: page_rows,
            page_bytes: page_bytes * page_rows / shared::PAGE_ROWS, mark_bytes:mark_bytes+draft_mark,
                draft_bytes:request.drafter.map_or(0,|selection|64*2*selection.kv_cache.plane_record_bytes(cfg.kv_heads,cfg.head_dim) as u64) };
        let quota = geometry.resolve(request.host_cache_bytes, context, request.sequences, host_total,
            geometry.chunk_bytes()).unwrap_or(0);
        serving::HostPrefixOwners { quota_bytes: quota, chunk_bytes: geometry.chunk_bytes().min(quota.max(1)) }
    };
    let push_required = match crate::push_load::namespace(&request.checkpoint.snapshot) {
        Ok(_) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && error.raw_os_error().is_none()
            && request.checkpoint.snapshot.exists() => false,
        Err(error) => return Err(refused(format!("Inkling checkpoint namespace refused: {error}"),
            "Restore SparkNest namespace metadata; required push cannot fall back to mount.")),
    };
    if !push_required {
        if let Some((_, _, draft_mount)) = &drafter {
            // Target and recurrent-body mount readers are serialized.
            for rank in &mut mount { rank.merge(*draft_mount); }
        }
    }
    let capacities = (0..gpus).map(|gpu| request.coordinator.budget(gpu).logical_bytes).collect::<Vec<_>>();
    let mut pool_facts = crate::placement::PoolFacts::new(Context::resolve(context, (request.context_tokens > 0).then_some(request.context_tokens)), request.pool_tokens, page_rows);
    pool_facts.memory = pool_memory(cfg, request.kv_format, &modes, page_rows, &capacities);
    pool_facts.target = request.target_pool_tokens;
    let pool = crate::placement::prepare_pool(pool_facts).map_err(|error|
        refused(format!("invalid KV pool policy: {error}"), "Set KV min <= target <= max."))?;
    let vision = crate::placement::still_encoder::StillEncoder::planned(request.checkpoint,
        request.vision, request.vision_replicas, &budgets, request.media_cache_bytes.unwrap_or(256 << 20))
        .map_err(|error| refused(format!("Inkling vision inventory refused: {error:#}"), "Provide the bundled PE tensors."))?;
    let prefix_mark_slots=super::draft::prefix_mark_slots(request.sequences,request.prefix_entries,
        mark_bytes,draft_mark,request.prefix_mark_budget,host_prefix.quota_bytes>0)
        .map_err(|error|refused(format!("Inkling prefix slots refused: {error:#}"),"Check combined mark extent."))?;
    let drafter=request.drafter.zip(drafter).map(|(selection, (config, weights, _))| {
        super::draft::DraftLayout::from_weights_for_decode_rows(config,weights,usize::try_from(request.sequences)?,
            usize::try_from(request.sequences)?.min(8),usize::try_from(context)?,usize::try_from(pool.maximum_tokens())?,
            usize::try_from(shape.rows)?,usize::try_from(shape.waves)?,decode_rows as usize,selection.sms,selection.kv_cache,request.manifest.as_deref())
            .map(|layout|shared::Drafter{layout})
    }).transpose().map_err(|error|refused(format!("bundled drafter inventory refused: {error:#}"),"Check MTP tensor headers and exports."))?;
    Ok(InklingInputs { cfg: cfg.clone(), kv_format: request.kv_format, gpus, spark_ranks: ranks,
        context: Context::resolve(context, (request.context_tokens > 0).then_some(request.context_tokens)),
        pool, prefill: shape, decode_rows, sequences: request.sequences, prefix_entries: request.prefix_entries,
        prefix_mark_budget: request.prefix_mark_budget, prefix_mark_slots, targets, target_fp8_scratch, placement, modes,
        routed: routed.clone(), experts, local, spark, links,
        spark_reply_formats: request.replies, drafter, vision, budgets, code, graph_reserve,
        head: request.head, host_prefix, device_sampling: true,
        expert_usage: request.expert_usage.then(|| cuteafd_core::expert_usage::Geometry { layers: cfg.layers,
            experts: cfg.experts, topk: cfg.topk, routed_layers: routed }),
        resident_weight_limit: Some(request.resident_weight_limit), mount, push_required })
}
