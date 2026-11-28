//! One fact gatherer for MiniMax M3 admission (#1856 H1). `cuteafd plan
//! --layout` and serving startup both call [`gather`]; they differ only in
//! [`Coordinator`] (a planned card forecast vs one measured CUDA sample per
//! GPU). The result is the shared recipe
//! [`MiniMaxInputs`](crate::placement::families::minimax_m3::MiniMaxInputs),
//! which `placement::planner::solve` admits. No family code samples memory or
//! sizes a pool here.
use crate::placement::families::minimax_m3::{self as shared, LocalPackage, MiniMaxInputs, SparkCharge};
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
            // The sampled occupancy is a named recipe owner; leave no second subtraction.
            Self::Measured { samples, .. } => shared::measured_budget(gpu as u8, samples[gpu].0, samples[gpu].1,
                samples[gpu].0),
        }
    }
}

/// Every operator input admission reads, resolved once by the caller.
pub struct FactsRequest<'a> {
    pub vision: crate::plan::MediaMode,
    pub vision_replicas: usize,
    pub media_cache_bytes: Option<u64>,
    pub checkpoint: &'a crate::plan::Checkpoint,
    pub cfg: &'a MiniMaxM3Config,
    pub catalog: &'a crate::OfficialV41Catalog,
    pub manifest_path: &'a Path,
    pub manifest: Arc<serde_json::Value>,
    pub coordinator: Coordinator,
    /// Explicit planning mode; None keeps the family attention default.
    pub attention: Option<crate::placement::AttentionPlacement>,
    /// Required for two-RTX Layers; qualification chooses the precise layout.
    pub split_wire: Option<super::execution::WirePrecision>,
    /// Logical rank to physical CUDA device, shared with runtime arena IDs.
    pub physical_devices: &'a [i32],
    pub spark_ranks: usize,
    pub spark_bytes: u64,
    pub spark_rank_bytes: Vec<u64>,
    pub prefill: PrefillShape,
    pub decode_rows: u32,
    /// Explicit context, or 0 for the checkpoint's maximum.
    pub context_tokens: u64,
    pub pool_tokens: Option<u64>,
    pub target_pool_tokens: Option<u64>,
    pub sequences: u64,
    pub host: (u64, u64),
    pub host_cache_bytes: Option<u64>,
    pub prefix_entries: u64,
    pub drafter: Option<crate::formats::block_dspark::NativeDsparkLayout>,
    pub replies: cuteafd_core::spark_reply::SparkReplyFormats,
    pub head: serving::BlasHead,
    pub decode_graphs: bool,
    pub expert_usage: bool,
    pub resident_weight_limit: ResidentWeights,
}

/// The installed coordinator package's executor workspace at the unit rows.
pub fn local_package(catalog: &crate::OfficialV41Catalog, manifest: &Path,
    capacity: shared::RequestCapacity, gpus: usize) -> LocalPackage {
    let rows = u64::from(capacity.execution_rows());
    let workspace = || -> anyhow::Result<_> {
        let requests = crate::placement::rtx_onboard::expert_workspace_requests(catalog, Some(manifest), rows, gpus, "minimax")?;
        let package = usize::try_from(*requests.first().context("missing expert package scratch")?)?;
        let (scratch, output, _) = crate::serving_capacity::fp8moe_tp2_workspace(package,
            catalog.routed_experts().hidden, usize::try_from(rows)?)?;
        Ok(LocalPackage::Installed { scratch: scratch as u64, output: output as u64 })
    };
    workspace().unwrap_or_else(|error| LocalPackage::Missing(format!("{error:#}")))
}

pub(crate) fn spark_checkpoint(checkpoint: &crate::plan::Checkpoint, catalog: &crate::OfficialV41Catalog,
    routed: &[usize]) -> anyhow::Result<Option<crate::plan::Checkpoint>> {
    use crate::formats::fp8_experts::Fp8Projection;
    ensure!(checkpoint.snapshot == catalog.snapshot(), "MiniMax Spark catalog and checkpoint snapshots differ");
    let tensors = catalog.fp8().context("MiniMax Spark admission requires native expert headers")?;
    let complete = routed.iter().all(|&layer| (0..catalog.routed_experts().experts).all(|expert|
        [Fp8Projection::Gate, Fp8Projection::Up, Fp8Projection::Down].into_iter().all(|projection|
            tensors.projection_source_names(layer, expert, projection).iter()
                .all(|name| checkpoint.header(name).is_some()))));
    if complete { return Ok(None); }
    // Coordinator role inventories omit routed operands. Recipes validate their
    // scalar headers too, so reopen the same canonical immutable view, not a
    // guessed tensor subset or a converted checkpoint.
    let full = match &checkpoint.view {
        Some(view) => crate::plan::Checkpoint::open_view(view)?,
        None => crate::plan::Checkpoint::open(&checkpoint.snapshot)?,
    };
    ensure!(full.config == checkpoint.config, "MiniMax Spark canonical checkpoint configuration changed");
    Ok(Some(full))
}

/// The Spark expert ingress, decided from the checkpoint's own tensors: the
/// exact NVFP4 wire ([`cuteafd_core::SparkIngress::Nvfp4`]) when every routed
/// layer's experts share one gate/up `input_global_scale` (MiniMax M3.1 Flash:
/// all 1.0) and the installed coordinator manifest carries the wire quantizer;
/// BF16 otherwise. Per-layer raw scales are returned for the quantizer.
#[derive(Debug, Clone, PartialEq)]
pub struct SparkIngressPlan {
    pub ingress: cuteafd_core::SparkIngress,
    /// Raw FP32 shared input scale per decoder layer (`None`: dense layer or BF16 ingress).
    pub scales: Vec<Option<f32>>,
}

pub fn spark_ingress(catalog: &crate::OfficialV41Catalog, cfg: &MiniMaxM3Config, routed: &[usize],
    manifest: &serde_json::Value) -> anyhow::Result<SparkIngressPlan> {
    let bf16 = SparkIngressPlan { ingress: cuteafd_core::SparkIngress::Bf16, scales: vec![None; cfg.layers] };
    let quantizer = manifest["programs"].as_array().is_some_and(|programs| programs.iter()
        .any(|entry| entry["name"] == cuteafd_ffi::minimax_m3::wire::NAME));
    if !quantizer || routed.is_empty() { return Ok(bf16); }
    let tensors = catalog.fp8().context("MiniMax Spark ingress requires native expert headers")?;
    let mut scales = vec![None; cfg.layers];
    for &layer in routed {
        match tensors.for_layer_metadata(layer)?.uniform_input_scale(layer)? {
            Some(scale) => scales[layer] = Some(scale),
            None => return Ok(bf16),
        }
    }
    Ok(SparkIngressPlan { ingress: cuteafd_core::SparkIngress::Nvfp4, scales })
}

/// Spark charges per onboarding count and rank: header-exact native NVFP4
/// slices, endpoint I/O, typed execution owners (installed TP scratch census)
/// and the shared worker runtime residual.
pub fn spark_charges(checkpoint: &crate::plan::Checkpoint, catalog: &crate::OfficialV41Catalog, routed: &[usize],
    ranks: usize, prefill: PrefillShape, ingress: cuteafd_core::SparkIngress) -> anyhow::Result<Vec<Vec<SparkCharge>>> {
    spark_charges_for_rows(checkpoint, catalog, routed, ranks, prefill, 64, ingress)
}

pub fn spark_charges_for_rows(checkpoint:&crate::plan::Checkpoint,catalog:&crate::OfficialV41Catalog,routed:&[usize],
    ranks:usize,prefill:PrefillShape,decode_rows:u32, ingress: cuteafd_core::SparkIngress)
    -> anyhow::Result<Vec<Vec<SparkCharge>>> {
    if ranks == 0 { return Ok(vec![Vec::new(); routed.len() + 1]); }
    let canonical = spark_checkpoint(checkpoint, catalog, routed)?;
    let recipe = crate::plan::experts::checkpoint_spark_recipe(canonical.as_ref().unwrap_or(checkpoint), ranks, routed)?;
    let executions = shared::spark_executions_for_rows(prefill,decode_rows)?;
    let endpoint = cuteafd_core::SparkEndpoint { ring_depth: None, capacity: u32::try_from(prefill.rows)?,
        ingress, partial_reply_bytes: None };
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
            let bytes = |extent| crate::plan::layout::spark_slots::nvfp4_owner_bytes_for("minimax_m3", geometry, width, extent,
                ingress == cuteafd_core::SparkIngress::Nvfp4);
            let owners = executions.workspace_bytes(bytes)?.and_then(|bytes| bytes.total())
                .context("MiniMax Spark owner sum overflow")?;
            let owners = [io.intake, io.host_exchange, io.row_indices, io.mapped_rings, io.endpoint_staging, owners]
                .into_iter().try_fold(0u64, |sum, bytes| sum.checked_add(bytes)).context("MiniMax Spark owners")?;
            Ok(SparkCharge { experts: per_layer[rank][count..].iter().sum(), owners, runtime })
        }).collect()
    }).collect()
}

fn measured_graph_base(gpus: usize, sequences: u64, rank: usize,
    attention: crate::placement::AttentionPlacement, wire: Option<super::execution::WirePrecision>) -> Option<(u64, u64)> {
    use crate::placement::AttentionPlacement;
    use super::execution::WirePrecision;
    match (gpus, sequences, rank, attention, wire) {
        (1, 16, 0, AttentionPlacement::Heads, _) => Some((41_943_040, 2 << 20)),
        // TP2 C16 startup at 5b7dd075a: 531/360 peer-spanning sections,
        // measured Ready graph owners 62/42 MiB; one physical tail each.
        (2, 16, 0, AttentionPlacement::Layers, Some(WirePrecision::F32)) => Some((65_011_712, 2 << 20)),
        (2, 16, 1, AttentionPlacement::Layers, Some(WirePrecision::F32)) => Some((44_040_192, 2 << 20)),
        // Heads C16 rows1/4/16 census at8f993c862:711/540 independent sections,
        // measured82/64MiB; each physical device receives one allocation tail.
        (2, 16, 0, AttentionPlacement::Heads, _) => Some((82 << 20, 2 << 20)),
        (2, 16, 1, AttentionPlacement::Heads, _) => Some((64 << 20, 2 << 20)),
        _ => None,
    }
}

/// The measured C16 base plus the shared extra, else the MiniMax × ranks
/// envelope (other concurrencies, wires and placements), never 0 with graphs on.
/// `decode_graphs = false` yields `graph_reserve::disabled(rank)`.
pub fn graph_policy(gpus: usize, sequences: u64, rank: usize,
    attention: crate::placement::AttentionPlacement, wire: Option<super::execution::WirePrecision>,
    decode_graphs: bool)
    -> Result<crate::placement::graph_reserve::Resolved, crate::placement::graph_reserve::Error> {
    use crate::placement::graph_reserve::{self, GraphFamily};
    if !decode_graphs { return Ok(graph_reserve::disabled(rank)); }
    match measured_graph_base(gpus, sequences, rank, attention, wire) {
        Some((base, tail)) => graph_reserve::current(rank, false, base, tail),
        None => graph_reserve::current_envelope(GraphFamily::MiniMax, gpus, rank),
    }
}

#[cfg(test)]
mod graph_tests {
    use super::*;

    #[test]
    fn measured_c16_whole_graph_base_has_one_standard_tail_and_no_arena_margin() {
        let (base, tail) = measured_graph_base(1, 16, 0, crate::placement::AttentionPlacement::Heads, None).unwrap();
        assert_eq!((base, tail), (41_943_040, 2_097_152));
        let resolved = crate::placement::graph_reserve::resolve(None, 0, false, base, tail).unwrap();
        assert_eq!(resolved.base_bytes, 44_040_192);
        assert_eq!(resolved.bytes, 88_080_384);
        for shape in [(2, 64, 0), (1, 64, 0), (1, 3, 0), (2, 16, 2)] {
            assert_eq!(measured_graph_base(shape.0, shape.1, shape.2,
                crate::placement::AttentionPlacement::Heads, None), None);
        }
    }

    #[test]
    fn measured_c16_split_graphs_charge_each_rank_once_through_shared_extra() {
        for (rank, measured, expected_base) in [(0, 65_011_712, 64 << 20), (1, 44_040_192, 44 << 20)] {
            let (base, tail) = measured_graph_base(2, 16, rank,
                crate::placement::AttentionPlacement::Layers, Some(super::super::execution::WirePrecision::F32)).unwrap();
            assert_eq!((base, tail), (measured, 2 << 20));
            let resolved = crate::placement::graph_reserve::resolve(None, rank, false, base, tail).unwrap();
            assert_eq!(resolved.base_bytes, expected_base);
            assert_eq!(resolved.bytes, expected_base * 2);
        }
    }

    #[test]
    fn measured_heads_graphs_have_independent_rank_bases_and_one_tail() {
        for (rank, measured, total) in [(0, 82 << 20, 168 << 20), (1, 64 << 20, 132 << 20)] {
            let (base, tail) = measured_graph_base(2, 16, rank, crate::placement::AttentionPlacement::Heads, None).unwrap();
            assert_eq!((base, tail), (measured, 2 << 20));
            let policy = crate::placement::graph_reserve::resolve(None, rank, false, base, tail).unwrap();
            assert_eq!(policy.bytes, total);
        }
    }

    #[test]
    fn untraced_split_attention_or_wire_never_borrows_layers_base() {
        use crate::placement::AttentionPlacement;
        use super::super::execution::WirePrecision;
        for rank in 0..2 {
            for attention in [AttentionPlacement::Context] {
                assert_eq!(measured_graph_base(2, 16, rank, attention, Some(WirePrecision::F32)), None);
            }
            for wire in [None, Some(WirePrecision::Bf16)] {
                assert_eq!(measured_graph_base(2, 16, rank, AttentionPlacement::Layers, wire), None);
            }
        }
    }

    #[test]
    fn every_concurrency_and_split_charges_a_nonzero_reserve() {
        use crate::placement::{AttentionPlacement, graph_reserve::{envelope_bytes, GraphFamily}};
        use super::super::execution::WirePrecision;
        cuteafd_core::memory_policy::MemoryPolicy::default().scoped(|| {
            for sequences in 1..=64 {
                for (gpus, attention, wire) in [(1, AttentionPlacement::Heads, None),
                    (2, AttentionPlacement::Heads, None), (2, AttentionPlacement::Layers, Some(WirePrecision::F32)),
                    (2, AttentionPlacement::Layers, Some(WirePrecision::Bf16))] {
                    for rank in 0..gpus {
                        let policy = graph_policy(gpus, sequences, rank, attention, wire, true).unwrap();
                        assert!(policy.bytes > 0, "C{sequences} {gpus}x {attention:?} {wire:?}");
                        if sequences != 16 { assert_eq!(policy.bytes, envelope_bytes(GraphFamily::MiniMax, gpus, rank)); }
                    }
                }
            }
        });
    }
}

/// Gather every admitted owner fact into the shared recipe.
pub fn gather(request: &FactsRequest<'_>) -> Result<MiniMaxInputs, Refused> {
    let cfg = request.cfg;
    let gpus = request.coordinator.gpus();
    let ranks = request.spark_ranks;
    if !(1..=2).contains(&gpus) {
        return Err(refused(format!("MiniMax runs on one RTX or an RTX pair, not {gpus}"), "Use --rtx 1 or --rtx 2."));
    }
    if request.physical_devices.len() != gpus || request.physical_devices.iter().any(|&device| device < 0)
        || request.physical_devices.iter().copied().collect::<std::collections::BTreeSet<_>>().len() != gpus {
        return Err(refused("MiniMax physical device inventory differs from its coordinator ranks",
            "Provide one distinct physical CUDA ID per coordinator rank."));
    }
    let context = if request.context_tokens == 0 { cfg.max_position as u64 } else { request.context_tokens };
    if context > cfg.max_position as u64 {
        return Err(refused(format!("MiniMax context {context} exceeds the checkpoint's {}", cfg.max_position),
            "Lower --max-context-tokens."));
    }
    let shape = request.prefill;
    let capacity = shared::RequestCapacity::new(request.sequences, shape.rows)
        .map_err(|error| refused(format!("MiniMax request capacities refused: {error}"),
            "Use concurrency 1..64 and prefill unit rows 1..4096."))?;
    let routed = (0..cfg.layers).filter(|&layer| cfg.moe[layer]).collect::<Vec<_>>();
    let attention = request.attention.unwrap_or(crate::placement::AttentionPlacement::Heads);
    let heads = gpus == 2 && attention == crate::placement::AttentionPlacement::Heads;
    if attention == crate::placement::AttentionPlacement::Context
        || (gpus == 1 && attention != crate::placement::AttentionPlacement::Heads) {
        return Err(refused("MiniMax attention placement is not implemented", "Use heads on one RTX; layers or heads on two RTX."));
    }
    if gpus == 2 && !heads && request.split_wire.is_none() {
        return Err(refused(shared::ServingRefusal::SplitWireNotQualified.to_string(),
            "Pass an explicitly qualified Bf16 or F32 wire precision to MiniMax admission; no default is selected."));
    }
    let owners = if heads { Vec::new() } else { shared::layer_owners(cfg.layers, gpus) };
    let target_inventory = || -> anyhow::Result<_> {
        if heads {
            let weights = super::weight_layout::HeadsWeights::plan(request.checkpoint, cfg,
                crate::placement::planner::EmbeddingHome::Device(0))?;
            let mount = weights.iter().map(|weight| weight.mount_load(request.checkpoint))
                .collect::<anyhow::Result<Vec<_>>>()?;
            Ok((weights.into_iter().map(|weight| weight.target).collect::<Vec<_>>(), mount))
        } else if gpus == 2 {
            let weights = super::weight_layout::SplitFfnWeights::plan(request.checkpoint, cfg,
                &owners.iter().map(|&gpu| usize::from(gpu)).collect::<Vec<_>>(),
                crate::placement::planner::EmbeddingHome::Device(0))?;
            let mount = weights.iter().map(|weight| weight.mount_load(request.checkpoint))
                .collect::<anyhow::Result<Vec<_>>>()?;
            Ok((weights.into_iter().map(|weight| weight.target).collect::<Vec<_>>(), mount))
        } else {
            let weights = CoordinatorWeights::plan(request.checkpoint, cfg, gpus,
                &owners.iter().map(|&gpu| usize::from(gpu)).collect::<Vec<_>>())?;
            let mount = weights.iter().map(|target| target.mount_load(request.checkpoint))
                .collect::<anyhow::Result<Vec<_>>>()?;
            Ok((weights, mount))
        }
    };
    let (targets, mount) = target_inventory().map_err(|error| refused(
        format!("MiniMax target arena inventory refused: {error:#}"), "Provide complete native MXFP8 checkpoint headers."))?;
    let tensors = request.catalog.fp8().ok_or_else(|| refused("MiniMax native expert catalog missing",
        "Provide compressed-tensors NVFP4 expert headers."))?;
    let expert_backings = shared::ExpertInventory::new(tensors, &routed, gpus)
        .and_then(|inventory| inventory.prefixes(request.physical_devices)).map_err(|error| refused(
            format!("MiniMax expert backing inventory refused: {error:#}"), "Provide complete native expert headers."))?;
    let push_required = match crate::push_load::namespace(&request.checkpoint.snapshot) {
        Ok(_) => true,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound
            && error.raw_os_error().is_none() && request.checkpoint.snapshot.exists() => false,
        Err(error) => return Err(refused(format!("MiniMax checkpoint storage namespace refused: {error}"),
            "Restore readable SparkNest namespace metadata; required push cannot fall back to mount.")),
    };
    let local = local_package(request.catalog, request.manifest_path, capacity, gpus);
    let ingress = spark_ingress(request.catalog, cfg, &routed, &request.manifest)
        .map_err(|error| refused(format!("MiniMax Spark ingress inventory refused: {error:#}"),
            "Provide every routed expert's input_global_scale header and value."))?;
    let decode_cap = if request.decode_rows == 0 { 64 } else { request.decode_rows };
    if decode_cap > shared::MAX_DECODE_ROWS {
        return Err(refused(format!("MiniMax decode rows {decode_cap} exceeds {}", shared::MAX_DECODE_ROWS),
            "Use --decode-rows 1..=128."));
    }
    let decode_rows=match &request.drafter {
        Some(draft) => u32::try_from(draft.verify_rows).map_err(|_| refused("MiniMax verify rows overflow","Use supported concurrency/draft width."))?.min(decode_cap),
        None if decode_cap > 64 => decode_cap,
        None => shared::spark_verify_rows_capped(request.sequences, 0, decode_cap).map_err(|error| refused(error.to_string(),"Use supported concurrency."))?,
    };
    let spark = spark_charges_for_rows(request.checkpoint, request.catalog, &routed, ranks, shape,decode_rows, ingress.ingress)
        .map_err(|error| refused(format!("MiniMax Spark inventory refused: {error:#}"),
            format!("Use a Spark world with a minimax:nvfp4 package ({:?}).",
                crate::plan::experts::spark_worlds("minimax:nvfp4a4", cfg.moe_intermediate))))?;
    let links = shared::spark_link(0, ranks, shape, request.replies, ingress.ingress)
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
        // SM120 C16 rows 1/4/16: whole-rank TP3 at e78eb75b0 and split TP4
        // at 5b7dd075a (#1856). Each rank charges its own measured graph owner.
        // Other rank/concurrency shapes borrow neither these bases nor pins; they
        // charge the family × ranks envelope.
        graph_reserve.push(graph_policy(gpus, request.sequences, gpu, attention, request.split_wire, request.decode_graphs)
            .map_err(|error| refused(format!("MiniMax graph reserve refused: {error}"), "Fix GPU_GRAPH_RESERVE."))?.bytes);
    }
    let measured_context = match &request.coordinator {
        Coordinator::Planned { .. } => vec![0; gpus],
        Coordinator::Measured { samples, .. } => samples.iter().map(|&(total, _, free)|
            total.checked_sub(free).ok_or_else(|| refused("MiniMax sampled free bytes exceed physical capacity",
                "Provide a valid post-module CUDA memory sample."))).collect::<Result<Vec<_>, _>>()?,
    };
    let mut budgets = (0..gpus).map(|gpu| request.coordinator.budget(gpu)).collect::<Vec<_>>();
    budgets.extend((0..ranks).map(|rank| shared::spark_budget(rank as u8, request.spark_rank_bytes.get(rank).copied().unwrap_or(request.spark_bytes))));
    let (host_total, host_available) = request.host;
    budgets.push(crate::placement::families::kimi::host_budget(host_total, host_available));
    // A host prefix snapshot retains both replicated index planes for heads.
    let page_bytes = if heads { 2 * (16384 + 16384 + 8448) } else { 32768 + 32768 + 8448 }
        * cfg.layers as u64;
    let host_prefix = if request.host_cache_bytes == Some(0) || request.prefix_entries == 0 {
        serving::HostPrefixOwners { quota_bytes: 0, chunk_bytes: 0 }
    } else {
        let geometry = cuteafd_core::prefix::HostCacheGeometry { unit_rows: shared::PAGE_ROWS,
            page_bytes, mark_bytes: request.drafter.as_ref().map_or(0,|draft| draft.mark_bytes as u64), draft_bytes: 0 };
        let quota = geometry.resolve(request.host_cache_bytes, context, request.sequences, host_total,
            geometry.chunk_bytes()).unwrap_or(0);
        serving::HostPrefixOwners { quota_bytes: quota, chunk_bytes: geometry.chunk_bytes().min(quota.max(1)) }
    };
    let capacities = (0..gpus).map(|gpu| request.coordinator.budget(gpu).logical_bytes).collect::<Vec<_>>();
    // The planner's MSA page geometry (`plan::families::minimax`): two KV
    // planes plus one index record per 64-token page, owned per rank by the
    // selected head split or balanced whole layers.
    let rank_heads = cfg.kv_heads / if heads { 2 } else { 1 };
    let page = shared::PAGE_ROWS
        * cuteafd_ffi::minimax_m3::kv_plane_record_bytes(rank_heads) as u64 * 2
        + (cfg.msa.index_dim as u64 + 4) * shared::PAGE_ROWS;
    let layer_owner = shared::layer_owners(cfg.layers, gpus);
    let rank_bytes = |gpu: usize| -> u64 { page * if heads { cfg.layers as u64 }
        else { layer_owner.iter().filter(|&&owner| usize::from(owner) == gpu).count() as u64 } };
    let mut pool_facts = crate::placement::PoolFacts::new(Context::resolve(context, (request.context_tokens > 0).then_some(request.context_tokens)), request.pool_tokens, shared::PAGE_ROWS);
    pool_facts.memory = capacities.iter().enumerate().map(|(gpu, &available_bytes)|
        crate::placement::PoolMemory { available_bytes, bytes_per_unit: rank_bytes(gpu) }).collect();
    pool_facts.target = request.target_pool_tokens;
    let pool = crate::placement::prepare_pool(pool_facts).map_err(|error|
        refused(format!("invalid KV pool policy: {error}"), "Set KV min <= target <= max."))?;
    Ok(MiniMaxInputs { cfg: cfg.clone(), gpus, attention, split_wire: request.split_wire, spark_ranks: ranks,
        context: Context::resolve(context, (request.context_tokens > 0).then_some(request.context_tokens)),
        pool, prefill: shape, decode_rows: decode_cap, sequences: request.sequences, targets, owners,
        manifest: request.manifest.clone(), routed: routed.clone(), expert_backings, local, spark, links,
        spark_ingress: ingress,
        spark_reply_formats: request.replies, drafter: request.drafter.as_ref().map(|native| shared::Drafter {
            arena:native.weights.clone(),native:native.clone(),scratch:Vec::new(),state_per_page:0 }),
        vision: crate::placement::still_encoder::StillEncoder::planned(request.checkpoint,
            request.vision, request.vision_replicas, &budgets,
            request.media_cache_bytes.unwrap_or((8 << 30).min(host_total / 20)))
            .map_err(|error| refused(format!("MiniMax bundled vision inventory refused: {error:#}"),
                "Supply the bundled CLIP tower tensors or use --vision off."))?,
        budgets, code, measured_context, graph_reserve,
        head: request.head, host_prefix, device_sampling: true,
        expert_usage: request.expert_usage.then(|| cuteafd_core::expert_usage::Geometry { layers: cfg.layers,
            experts: cfg.experts, topk: cfg.topk, routed_layers: routed }),
        resident_weight_limit: Some(request.resident_weight_limit), push_required, mount })
}
