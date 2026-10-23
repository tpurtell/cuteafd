//! Qwen's shared admission request. Weights, PLE and resident expert arenas
//! are already in the baseline; every allocation after that sample is named
//! here. Startup graphs depend on the admitted pool, so resolve their shape
//! before publishing the final placement, without taking another CUDA sample.
use crate::families::qwen4::Qwen4Attention;
use crate::placement::*;
use crate::serving_capacity::qwen_graphs::{qwen_admission, qwen_startup_graphs, QwenAdmissionInputs, QWEN_UNIT_ROWS};
use cuteafd_core::memory_layout::{Basis, Category, ItemLifetime};

pub const DEFAULT_MARK_SLOTS: u64 = 18;
pub const LAZY_GRAPH_BYTES: u64 = (1 << 30) / 2;
pub const SPARK_WORKSPACE_BYTES: u64 = 56 * (1 << 30) / 100;

/// A dual Qwen executor requires nonempty whole-width ownership on both ends.
/// Do not silently turn the shared solver's legal all-on-one-rank cut into a
/// private cut heuristic or a partly idle dual executor.
pub fn check_dual_layer_owners(owners: &[usize]) -> anyhow::Result<()> {
    anyhow::ensure!(owners.first() == Some(&0) && owners.last() == Some(&1)
        && owners.iter().all(|&owner| owner <= 1)
        && owners.windows(2).filter(|pair| pair[0] != pair[1]).count() == 1,
        "Qwen dual attention needs nonempty owners and one contiguous owner-zero to owner-one cutover");
    Ok(())
}

pub struct QwenInputs<'a> {
    /// True only when the operator left context automatic.
    pub automatic_context: bool,
    pub admission: QwenAdmissionInputs<'a>,
    pub capacity_bytes: u64,
    pub baseline: Baseline,
    pub pending_code_bytes: u64,
    pub max_context: u64,
    pub requested_pool: Option<u64>,
    pub spark_ranks: usize,
    /// None means lazy diagnostic graphs, not the serving startup set.
    pub startup_graph_modes: Option<(usize, bool)>,
}

pub fn request(inputs: &QwenInputs<'_>, graph_bytes: u64) -> anyhow::Result<PlacementRequest> {
    let admission = qwen_admission(&inputs.admission)?;
    let unit_rows = QWEN_UNIT_ROWS as u64;
    let whole = LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner };
    let mut previous = 0;
    let mut layers = Vec::new();
    for (index, &kind) in inputs.admission.cfg.attention[..inputs.admission.layers].iter().enumerate() {
        let geometry = crate::serving_capacity::qwen_cache_geometry(inputs.admission.cfg, index + 1, false, inputs.admission.kv_format)?;
        let bytes = geometry.ranks[0].persistent_unit_bytes;
        layers.push(LayerDemand {
            kind: if kind == Qwen4Attention::Full { AttentionClass::Gqa } else { AttentionClass::Gdn },
            weights: ModeBytes::default(),
            kv_unit: KvDemand { unit_bytes_whole: bytes - previous, ..Default::default() },
            fixed_bytes: ModeBytes::default(), extra: Vec::new(), context_indexer: false, colocate: None,
            experts: None,
            modes: vec![whole],
        });
        previous = bytes;
    }
    let records: u64 = layers.iter().map(|l| l.kv_unit.unit_bytes_whole).sum();
    // Includes optional MTP records, pool metadata, the prefill/decode page
    // tables, and the existing per-token rounding of those table bytes.
    let overhead = admission.per_token.checked_mul(unit_rows).and_then(|n| n.checked_sub(records))
        .ok_or_else(|| anyhow::anyhow!("Qwen pool overhead overflow"))?;
    let mut fixed = admission.items.iter().map(|&(category, group, bytes)|
        Demand::new(0, category, group, bytes, Basis::Formula)).collect::<Vec<_>>();
    fixed.push(Demand::new(0, Category::Runtime, "startup graphs", graph_bytes, Basis::Formula));
    if inputs.spark_ranks > 0 && matches!(inputs.baseline, Baseline::Planned { .. }) {
        // Runtime creates SparkLink before sampling free memory. Only the
        // header-only planner still needs to reserve its BF16 return planes.
        fixed.push(Demand::new(0, Category::Transport, "Spark intake planes",
            inputs.admission.prefill_shape.map_or_else(|| crate::serving_capacity::spark_intake_bytes(1,
                inputs.spark_ranks as u64, inputs.admission.prefill_rows.max(64), inputs.admission.cfg.hidden as u64),
                |shape| crate::serving_capacity::prefill_return_bytes(shape.waves, shape.unit_rows,
                    inputs.spark_ranks as u64, inputs.admission.cfg.hidden as u64)), Basis::Formula));
    }
    if inputs.pending_code_bytes > 0 {
        fixed.push(Demand::new(0, Category::Runtime, "loaded code", inputs.pending_code_bytes, Basis::Formula));
    }
    Ok(PlacementRequest {
        inventory: Inventory { gpus: vec![GpuBudget { capacity_bytes: inputs.capacity_bytes,
            headroom_bytes: admission.headroom, baseline: inputs.baseline }],
            spark_ranks: inputs.spark_ranks, peer_access: false },
        attention_placement: Some(AttentionPlacement::Heads),
        context_buffers: ContextBuffers::default(), layers_first_gpu: 0,
        pool: PoolPolicy::resolve(&[inputs.capacity_bytes], inputs.max_context, inputs.requested_pool,
            unit_rows, inputs.spark_ranks == 0).with_automatic_context(inputs.automatic_context),
        pool_bounded: Vec::new(),
        layers, pool_overhead: vec![overhead], fixed, startup: Vec::new(), movables: Vec::new(),
        minimum_expert_layers: 0,
        expert_workspace: 0, tp2_workspace: [0; 2], onboard: Onboard::Auto, expert_gpus: 0,
        policy: LayerPolicy { default: vec![whole], by_kind: Vec::new(), ..LayerPolicy::default() },
        hops: HopSpec { storage: crate::placement::HopStorage::ReceiveSlots, row_bytes: inputs.admission.cfg.hc_count as u64 * inputs.admission.cfg.hidden as u64 * 2,
            rows: inputs.admission.prefill_rows.max(64), lanes: 1, entry_gpu: 0, head_gpu: 0, terminal: TerminalHome::AttentionOwner },
        executor: super::QWEN4,
    })
}

/// Solve using one baseline; descend through the pool-dependent graph sets
/// exactly as the former graph-aware admission did. An explicit pool stays
/// strict. No state is allocated until this converges.
pub fn placement(inputs: &QwenInputs<'_>) -> anyhow::Result<Placement> {
    let Some((sequences, speculation)) = inputs.startup_graph_modes else {
        let mut result = solve(&request(inputs, LAZY_GRAPH_BYTES)?)?;
        for item in &mut result.items[0] {
            if item.group == "startup graphs" {
                item.group = "graph growth".into();
                item.lifetime = ItemLifetime::PostReadyReserve;
            }
        }
        return Ok(result);
    };
    anyhow::ensure!(sequences <= 16, "Qwen startup graphs support at most 16 concurrent sequences");
    let tokens = solve(&request(inputs, 0)?)?.pool_tokens;
    super::super::settle_pool((tokens, ()), |tokens, _| {
        let graphs = qwen_startup_graphs(usize::try_from(inputs.max_context)?, usize::try_from(tokens)?,
            inputs.admission.cfg.dense_context(), sequences, speculation, inputs.admission.layers)
            .ok_or_else(|| anyhow::anyhow!("Qwen graph count overflow"))?;
        let mut req = request(inputs, graphs.bytes(0))?;
        req.pool.target = req.pool.target.min(tokens);
        let mut result = solve(&req)?;
        result.items[0].retain(|i| i.group != "startup graphs");
        result.items[0].extend(graphs.items(0));
        Ok::<_, anyhow::Error>((result.pool_tokens, (), result))
    })
}

/// Header-only exact expert costs. Qwen TP2 owns complete H128 blocks: at
/// intermediate640 the halves are384/256. Callers must validate packages with
/// those widths before admitting this request; padded320/320 is not this ABI.
pub fn dual_expert_costs(catalog: &crate::OfficialV41Catalog, layers: usize)
    -> anyhow::Result<Vec<Option<ExpertCost>>> {
    let shape = catalog.routed_experts();
    anyhow::ensure!(shape.first_layer == 0 && layers <= shape.layers,
        "Qwen expert costs require a complete routed backbone prefix");
    (0..layers).map(|layer| {
        let bytes = |world, rank| -> anyhow::Result<Bytes2> {
            let resident = if let Some(exl3) = catalog.exl3() {
                exl3.residency(crate::V41Exl3Layer::Backbone(layer), world, rank)?.device_arena_layout()?.1 as u64
            } else {
                use crate::formats::fp8_experts::{Fp8Projection, Slicing};
                let tensors = catalog.fp8().ok_or_else(|| anyhow::anyhow!("Qwen TP2 needs EXL3/FP8/NVFP4 tensors"))?
                    .for_layer(layer)?.for_coordinator();
                tensors.validate_layer(layer)?;
                let slicing = if world == 2 { Slicing::Blocks(128) } else { Slicing::Padded };
                Fp8Projection::ALL.iter().try_fold(0u64, |total, &projection| {
                    let (weights, _) = tensors.slice_bytes_with(projection, world, rank, slicing)?;
                    let scales = tensors.scale_region_bytes_with(projection, world, rank, slicing)?;
                    (weights as u64).checked_mul(shape.experts as u64)
                        .and_then(|n| n.checked_add(scales as u64)).and_then(|n| n.checked_add(total))
                        .ok_or_else(|| anyhow::anyhow!("Qwen TP2 expert residency overflow"))
                })?
            };
            Ok(Bytes2 { resident, staging: 0 })
        };
        Ok(Some(ExpertCost { whole: bytes(1, 0)?, half: [bytes(2, 0)?, bytes(2, 1)?],
            tp2: true, spark_ok: true }))
    }).collect()
}

/// Package selection is checkpoint-native and shared by plan and runtime.
pub fn dual_packages(catalog: &crate::OfficialV41Catalog, native_lib: &std::path::Path)
    -> anyhow::Result<[std::path::PathBuf; 2]> {
    let root = native_lib.parent().unwrap_or(std::path::Path::new("."));
    if let Some(exl3) = catalog.exl3() {
        let tiers = exl3.decoder_tiers().iter().map(usize::to_string).collect::<String>();
        let root = root.join("exl3").join(format!("exl3-qwen4-k{tiers}"));
        Ok([0, 1].map(|rank| root.join(format!("rtx-tp2-rank{rank}"))))
    } else {
        let tensors = catalog.fp8().ok_or_else(|| anyhow::anyhow!("Qwen TP2 needs native expert headers"))?.for_coordinator();
        let root = root.join("fp8").join(tensors.format().package_name("qwen4"));
        let rank = |rank| -> anyhow::Result<_> {
            Ok(root.join(format!("tp2-w{}", tensors.rank_width(2, rank,
                crate::formats::fp8_experts::Slicing::Blocks(128))?)))
        };
        Ok([rank(0)?, rank(1)?])
    }
}

/// The separate, full-width MTP expert arena stays on the head owner. Include
/// every retained TP1 capacity, not the target's half-width workspace.
pub fn dual_draft_demands(catalog: &crate::OfficialV41Catalog, native_lib: &std::path::Path,
    rows: u64) -> anyhow::Result<Vec<Demand>> {
    use crate::formats::fp8_experts::{ExpertFormat, Fp8Projection, Slicing};
    let shape = catalog.routed_experts();
    anyhow::ensure!(rows > 0, "Qwen MTP workspace needs positive rows");
    let root = native_lib.parent().unwrap_or(std::path::Path::new("."));
    let (resident, scratch) = if let Some(exl3) = catalog.exl3() {
        let resident = exl3.residency(crate::V41Exl3Layer::Dspark(0), 1, 0)?.device_arena_layout()?.1 as u64;
        let tiers = exl3.decoder_tiers().iter().map(usize::to_string).collect::<String>();
        let root = root.join("exl3").join(format!("exl3-qwen4-k{tiers}/rtx-tp1"));
        anyhow::ensure!(rows > 0 && rows <= 4096, "Qwen MTP EXL3 capacity exceeds 4096 rows");
        let manifests = crate::placement::inventory::exl3_capacities(rows).into_iter().map(|capacity| {
            let path = root.join(format!("m{capacity}/v41_exl3.json"));
            let m: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).map_err(|error| anyhow::anyhow!("Qwen expert package {}: {error}", path.display()))?)?;
            anyhow::ensure!(m["hidden"].as_u64() == Some(shape.hidden as u64)
                && m["intermediate"].as_u64() == Some(shape.intermediate as u64)
                && m["experts"].as_u64() == Some(shape.experts as u64)
                && m["top_k"].as_u64() == Some(shape.topk as u64)
                && m["capacity"].as_u64() == Some(capacity), "Qwen MTP package geometry: {}", path.display());
            Ok(m)
        }).collect::<anyhow::Result<Vec<_>>>()?;
        (resident, crate::serving_capacity::exl3_workspace_bytes(&manifests, true)? + rows * shape.hidden as u64 * 2)
    } else {
        let tensors = catalog.fp8().ok_or_else(|| anyhow::anyhow!("Qwen MTP format unsupported"))?.for_layer(shape.layers)?;
        tensors.validate_layer(shape.layers)?;
        let resident = Fp8Projection::ALL.iter().try_fold(0u64, |total, &projection| {
            let (weights, _) = tensors.slice_bytes_with(projection, 1, 0, Slicing::Padded)?;
            let scales = tensors.scale_region_bytes_with(projection, 1, 0, Slicing::Padded)?;
            Ok::<_, anyhow::Error>(total + weights as u64 * shape.experts as u64 + scales as u64)
        })?;
        let root = root.join("fp8").join(tensors.format().package_name("qwen4"));
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        let info = &manifest["layouts"]["tp1"];
        let weights = match tensors.format() {
            ExpertFormat::Fp8Block128 => info["weights"].as_str() == Some("fp8"),
            ExpertFormat::Mxfp4 => info["weights"].as_str() == Some("mxfp4"),
            ExpertFormat::Bf16 => info["weights"].as_str() == Some("bf16"),
            ExpertFormat::Nvfp4 => matches!(info["weights"].as_str(), Some("nvfp4" | "nvfp4a4")),
        };
        anyhow::ensure!(info["tp"].as_u64() == Some(1)
            && info["hidden"].as_u64() == Some(shape.hidden as u64)
            && info["intermediate"].as_u64() == Some(shape.intermediate as u64)
            && info["slice"].as_u64() == Some(shape.intermediate as u64)
            && info["experts"].as_u64() == Some(shape.experts as u64)
            && info["top_k"].as_u64() == Some(shape.topk as u64)
            && info["input"].as_str() == Some("bf16") && weights,
            "Qwen MTP package must be native full-width BF16-input TP1: {}", root.display());
        let scratch = crate::placement::inventory::fp8moe_scratch_bytes(&manifest, "tp1", rows)
            .ok_or_else(|| anyhow::anyhow!("Qwen MTP package has no capacity for {rows} rows"))?.max(256);
        (resident, scratch)
    };
    Ok(vec![Demand::new(0, Category::Experts, "MTP expert weights", resident, Basis::Exact),
        Demand::new(0, Category::Experts, "MTP expert workspace", scratch, Basis::Formula)])
}

/// Exact rank-package workspace before CUDA initialization. Packages are
/// explicitly selected by the caller; unequal ranks may never fall back to a
/// padded sibling. The runtime also checks its DSO ABI against these totals.
pub fn dual_expert_workspace(catalog: &crate::OfficialV41Catalog,
    packages: [&std::path::Path; 2], rows: u64) -> anyhow::Result<[u64; 2]> {
    use crate::formats::fp8_experts::{ExpertFormat, Slicing};
    let shape = catalog.routed_experts();
    anyhow::ensure!(rows > 0, "Qwen TP2 workspace needs positive rows");
    let mut bytes = [0; 2];
    for rank in 0..2 {
        let package = packages[rank];
        let width = if let Some(exl3) = catalog.exl3() {
            exl3.residency(crate::V41Exl3Layer::Backbone(0), 2, rank)?.intermediate as u64
        } else {
            catalog.fp8().ok_or_else(|| anyhow::anyhow!("Qwen TP2 expert format is unsupported"))?.for_coordinator()
                .rank_width(2, rank, Slicing::Blocks(128))? as u64
        };
        let scratch = if catalog.exl3().is_some() {
            anyhow::ensure!(rows <= 4096, "Qwen EXL3 TP2 capacity exceeds 4096 rows");
            let manifests = crate::placement::inventory::exl3_capacities(rows).into_iter().map(|capacity| {
                let path = package.join(format!("m{capacity}/v41_exl3.json"));
                let value: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).map_err(|error| anyhow::anyhow!("Qwen expert package {}: {error}", path.display()))?)?;
                anyhow::ensure!(value["hidden"].as_u64() == Some(shape.hidden as u64)
                    && value["intermediate"].as_u64() == Some(width)
                    && value["experts"].as_u64() == Some(shape.experts as u64)
                    && value["top_k"].as_u64() == Some(shape.topk as u64)
                    && value["capacity"].as_u64() == Some(capacity)
                    && value["output_dtype"].as_str() == Some("fp32"),
                    "Qwen EXL3 TP2 rank {rank} package geometry mismatch: {}", path.display());
                Ok(value)
            }).collect::<anyhow::Result<Vec<_>>>()?;
            crate::serving_capacity::exl3_workspace_bytes(&manifests, true)?
        } else {
            let tensors = catalog.fp8().unwrap();
            let parent = package.parent().ok_or_else(|| anyhow::anyhow!("Qwen TP2 package has no parent"))?;
            let name = package.file_name().and_then(|n| n.to_str())
                .ok_or_else(|| anyhow::anyhow!("Qwen TP2 package has no layout name"))?;
            let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(parent.join("manifest.json"))?)?;
            let info = &manifest["layouts"][name];
            let weights = match tensors.format() {
                ExpertFormat::Fp8Block128 => info["weights"].as_str() == Some("fp8"),
                ExpertFormat::Mxfp4 => info["weights"].as_str() == Some("mxfp4"),
            ExpertFormat::Bf16 => info["weights"].as_str() == Some("bf16"),
                ExpertFormat::Nvfp4 => matches!(info["weights"].as_str(), Some("nvfp4" | "nvfp4a4")),
            };
            anyhow::ensure!(info["tp"].as_u64() == Some(2)
                && info["hidden"].as_u64() == Some(shape.hidden as u64)
                && info["intermediate"].as_u64() == Some(shape.intermediate as u64)
                && info["slice"].as_u64() == Some(width)
                && info["experts"].as_u64() == Some(shape.experts as u64)
                && info["top_k"].as_u64() == Some(shape.topk as u64)
                && info["input"].as_str() == Some("bf16") && weights,
                "Qwen FP8/NVFP4 TP2 rank {rank} package geometry mismatch: {}", package.display());
            crate::placement::inventory::fp8moe_scratch_bytes(&manifest, name, rows)
                .ok_or_else(|| anyhow::anyhow!("Qwen TP2 rank {rank} has no capacity for {rows} rows"))?
        };
        let output_element = if catalog.exl3().is_some() { 4 } else { 2 };
        bytes[rank] = rows.checked_mul(shape.hidden as u64).and_then(|n| n.checked_mul(output_element))
            .and_then(|output| output.checked_add(scratch))
            .ok_or_else(|| anyhow::anyhow!("Qwen TP2 workspace overflow"))?;
    }
    Ok(bytes)
}

/// Exact TP2 row ABI shared by admission and the owner executor. EXL3 sends
/// E4M3/K32 wire rows and produces FP32 partials; native packages use BF16.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QwenTp2Rows {
    pub input: u64,
    pub ids: u64,
    pub weights: u64,
    pub bytes: u64,
    pub partial_bytes: u64,
    pub reduction_bytes: u64,
    pub exchange_slot_bytes: u64,
}

impl QwenTp2Rows {
    pub fn new(hidden: u64, topk: u64, rows: u64, wire: bool) -> anyhow::Result<Self> {
        anyhow::ensure!(hidden > 0 && topk > 0 && rows > 0 && (!wire || hidden % 32 == 0),
            "invalid Qwen TP2 row geometry");
        let overflow = || anyhow::anyhow!("Qwen TP2 row extent overflow");
        let mul = |a: u64, b: u64| a.checked_mul(b).ok_or_else(overflow);
        let align = |bytes: u64| bytes.checked_add(15).map(|n| n / 16 * 16).ok_or_else(overflow);
        let stride = if wire { hidden.checked_add(hidden / 32).ok_or_else(overflow)? }
            else { mul(hidden, 2)? };
        let input = mul(rows, stride)?;
        let ids = align(input)?;
        let route_bytes = align(mul(mul(rows, topk)?, 4)?)?;
        let weights = ids.checked_add(route_bytes).ok_or_else(overflow)?;
        let bytes = weights.checked_add(route_bytes).ok_or_else(overflow)?;
        let partial_bytes = mul(mul(rows, hidden)?, if wire { 4 } else { 2 })?;
        Ok(Self { input, ids, weights, bytes, partial_bytes,
            reduction_bytes: mul(mul(rows, hidden)?, 2)?,
            exchange_slot_bytes: align(bytes.max(partial_bytes))? })
    }

    /// Persistent buffers of the abortable two-slot expert exchange. Native
    /// publisher modules and streams belong to the context/module baseline.
    pub fn demands(self) -> anyhow::Result<Vec<Demand>> {
        let receive = self.exchange_slot_bytes.max(256).checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("Qwen TP2 receive extent overflow"))?;
        Ok((0..2).flat_map(|gpu| [
            Demand::new(gpu, Category::Transport, "expert send", self.bytes, Basis::Formula),
            Demand::new(gpu, Category::Transport, "expert receive", receive, Basis::Formula),
            Demand::new(gpu, Category::Transport, "expert reduction", self.reduction_bytes, Basis::Formula),
            Demand::new(gpu, Category::Transport, "expert peer control", 256, Basis::Formula),
            Demand::new(gpu, Category::Transport, "expert abort word", 256, Basis::Formula),
        ]).collect())
    }
}

/// Whole-width non-layer operands on the entry/head/MTP owner. Layer operands
/// are passed to `dual_request` separately; PLE backing follows its layer owner,
/// and draft expert arenas/workspaces must be supplied by their package plan.
pub fn dual_owner0_weights(resident: &crate::families::qwen4::resident::Qwen4CheckpointResident,
    host_embedding: bool) -> Vec<Demand> {
    [
        (Category::Weights, "owner0 head", resident.head_bytes),
        (Category::Weights, "owner0 mixer", resident.entry_bytes),
        (Category::Weights, "owner0 MTP weights", resident.mtp_bytes),
        (Category::Embedding, "owner0 embedding", if host_embedding { 0 } else { resident.embedding_bytes }),
    ].into_iter().filter(|(_, _, bytes)| *bytes > 0)
        .map(|(category, group, bytes)| Demand::new(0, category, group, bytes, Basis::Exact)).collect()
}

/// Conservative load-only conversion peak on both nonempty owners, before the
/// cutover is chosen. Kept as a steady-state reserve until load lifetimes are
/// modeled; the paired group does not bias the owner-boundary objective.
pub fn dual_projection_load_staging(cfg: &crate::families::qwen4::Qwen4Config, layers: usize, mtp: bool,
    selected: crate::families::qwen4::resident::Qwen4Representation) -> Vec<Demand> {
    let bytes = crate::families::qwen4::resident::resident_bytes(cfg, layers, mtp, selected)
        .max_device_staging as u64;
    if bytes == 0 { return Vec::new(); }
    (0..2).map(|gpu| Demand::new(gpu, Category::Staging,
        "FP8 projection load staging", bytes, Basis::Formula)).collect()
}

/// Only the scale follows the owning layer; tables stay mapped and step
/// workspace accounting includes gathered row staging.
pub fn dual_ple_demands(cfg: &crate::families::qwen4::Qwen4Config, layers: usize,
    resident: &crate::families::qwen4::resident::Qwen4CheckpointResident) -> anyhow::Result<Vec<Vec<LayerExtraDemand>>> {
    anyhow::ensure!(layers <= cfg.layers && cfg.ple_layers.len() <= 1,
        "Qwen PLE demands need a backbone prefix and at most one table");
    let mut demands = vec![Vec::new(); layers];
    if let Some(&layer) = cfg.ple_layers.first().filter(|&&layer| layer < layers) {
        anyhow::ensure!(resident.ple_table_bytes > 0, "Qwen PLE layer {layer} has no table backing headers");
        demands[layer].push(LayerExtraDemand::new(Category::Tables, "PLE scale", 256,
            LayerResidency::Owner, Basis::Formula));
    }
    Ok(demands)
}

/// Admission contract for the private whole-owner executor. Unlike the legacy
/// single-owner path, layer weights and routed halves are future allocations;
/// both baselines must be sampled before loading them. The public selector
/// stays single-owner until safety and numerical gates enable this caller.
/// Match each owner's loaded program recipe, including its attention reduction plan.
pub fn dual_program_manifests(manifest: &serde_json::Value, sms: [u32; 2],
    plan: cuteafd_core::coordinator_programs::AttentionReductionPlan) -> anyhow::Result<[serde_json::Value; 2]> {
    let selected = cuteafd_core::coordinator_programs::CoordinatorPrograms { family: "qwen4", split_family: None };
    let mut manifests = [manifest.clone(), manifest.clone()];
    for (owner, manifest) in manifests.iter_mut().enumerate() {
        cuteafd_core::coordinator_programs::select_card_plans(manifest, sms[owner])?;
        for entry in manifest["programs"].as_array_mut().into_iter().flatten() {
            if entry["name"].as_str().is_some_and(|name| selected.contains(name)) {
                if let Some(bytes) = cuteafd_core::coordinator_programs::attention_plan_scratch(&entry["geometry"], plan)? {
                    entry["scratch_bytes_at_capacity"]["scratch"] = bytes.into();
                }
            }
        }
    }
    Ok(manifests)
}

/// PeerExchange has two receive slots plus its control slot; progress is 48
/// bytes per slot, with the allocator's 256-byte minimum on each owner.
pub fn dual_peer_diagnostics(enabled: bool) -> Vec<Demand> {
    if !enabled { return Vec::new(); }
    [0, 1].map(|gpu| Demand::new(gpu, Category::Workspace,
        "expert peer diagnostics", 256, Basis::Exact)).to_vec()
}

pub struct QwenDualInputs<'a> {
    /// True only when the operator left context automatic.
    pub automatic_context: bool,
    pub admission: QwenAdmissionInputs<'a>,
    pub owner_manifests: [Option<&'a serde_json::Value>; 2],
    pub gpus: [GpuBudget; 2],
    pub layer_weights: &'a [u64],
    pub layer_experts: &'a [Option<ExpertCost>],
    /// Owner-local side allocations, such as the PLE scale.
    pub layer_extra: &'a [Vec<LayerExtraDemand>],
    /// Head, MTP weights/experts, embedding and exact transport
    /// buffers. Head/MTP are owner-only groups, not paired rank groups.
    pub fixed: Vec<Demand>,
    pub tp2_workspace: [u64; 2],
    pub pending_code_bytes: [u64; 2],
    pub mark_slots: u64,
    pub max_context: u64,
    pub requested_pool: Option<u64>,
    pub spark_ranks: usize,
    pub onboard: Onboard,
    pub startup_graph_modes: Option<(usize, bool)>,
}

const DUAL_WHOLE: [LayerMode; 2] = [LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner },
    LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner }];

pub fn dual_request(inputs: &QwenDualInputs<'_>, graph_bytes: [u64; 2]) -> anyhow::Result<PlacementRequest> {
    let a = &inputs.admission;
    anyhow::ensure!(a.layers > 1 && a.layers <= a.cfg.layers
        && inputs.layer_weights.len() == a.layers && inputs.layer_experts.len() == a.layers
        && inputs.layer_extra.len() == a.layers,
        "Qwen dual demands must name every backbone layer");
    anyhow::ensure!(a.future_expert_bytes == 0, "Qwen dual experts must be solver-owned TP2 halves");
    anyhow::ensure!(inputs.layer_experts.iter().flatten().all(|cost| cost.tp2),
        "Qwen dual routed experts require TP2 packages");
    let mut layers = Vec::with_capacity(a.layers);
    let mut previous = crate::serving_capacity::RankCacheGeometry::default();
    for index in 0..a.layers {
        let current = crate::serving_capacity::qwen_cache_geometry(a.cfg, index + 1, false, a.kv_format)?.ranks[0];
        let state = (current.active_state_per_sequence_bytes - previous.active_state_per_sequence_bytes)
            .checked_mul(a.slots).and_then(|n| n.checked_add(current.speculative_replay_bytes - previous.speculative_replay_bytes))
            .and_then(|n| (current.retained_mark_bytes - previous.retained_mark_bytes).checked_mul(inputs.mark_slots)
                .and_then(|marks| n.checked_add(marks)))
            .ok_or_else(|| anyhow::anyhow!("Qwen owner state overflow"))?;
        layers.push(LayerDemand {
            kind: if a.cfg.attention[index] == Qwen4Attention::Full { AttentionClass::Gqa } else { AttentionClass::Gdn },
            weights: ModeBytes::replicated(inputs.layer_weights[index]),
            kv_unit: KvDemand { unit_bytes_whole: current.persistent_unit_bytes - previous.persistent_unit_bytes,
                ..Default::default() },
            fixed_bytes: ModeBytes::replicated(state), extra: inputs.layer_extra[index].clone(),
            context_indexer: false, colocate: None,
            experts: inputs.layer_experts[index], modes: DUAL_WHOLE.to_vec(),
        });
        previous = current;
    }
    let complete = crate::serving_capacity::qwen_cache_geometry(a.cfg, a.layers, a.mtp, a.kv_format)?.ranks[0];
    // Both workspaces allocate full-width page tables; owner0 alone owns pool
    // metadata, MTP records/pending rows and the deferred MTP id buffer.
    let tables = (1 + 64) * 5 * 4;
    let mut overhead = [tables, tables];
    overhead[0] += complete.persistent_unit_bytes - previous.persistent_unit_bytes + complete.pool_metadata_unit_bytes;
    // Preserve single-owner per-token rounding; do not hide rounded table bytes.
    for bytes in &mut overhead { *bytes = bytes.div_ceil(QWEN_UNIT_ROWS as u64) * QWEN_UNIT_ROWS as u64; }
    let mut fixed = inputs.fixed.clone();
    for gpu in 0..2 {
        let owner = QwenAdmissionInputs { prefill_shape: None, cfg: a.cfg, layers: a.layers, mtp: a.mtp, kv_format: a.kv_format,
            manifest: inputs.owner_manifests[gpu].or(a.manifest), prefill_rows: a.prefill_rows,
            slots: a.slots, mark_bytes: a.mark_bytes, full_prefill_logits: a.full_prefill_logits,
            ple: a.ple, future_expert_bytes: a.future_expert_bytes, headroom: a.headroom };
        let admitted = qwen_admission(&owner)?;
        fixed.extend(admitted.items.iter().filter(|(_, group, _)| matches!(*group, "decode step" | "prefill step"))
            .map(|&(category, group, bytes)| Demand::new(gpu as u8, category, group, bytes, Basis::Formula)));
        fixed.push(Demand::new(gpu as u8, Category::Kv, "owner commit tables", 3 * 64 * 4, Basis::Formula));
        fixed.push(Demand::new(gpu as u8, Category::Runtime, "startup graphs", graph_bytes[gpu], Basis::Formula));
        fixed.push(Demand::new(gpu as u8, Category::Runtime, "loaded code", inputs.pending_code_bytes[gpu], Basis::Formula));
    }
    let pending = (complete.active_state_per_sequence_bytes - previous.active_state_per_sequence_bytes)
        .checked_mul(a.slots).and_then(|n| n.checked_add(complete.fixed_state_bytes - 3 * 64 * 4))
        .ok_or_else(|| anyhow::anyhow!("Qwen MTP state overflow"))?;
    // The MTP layer reads the head owner's pre-mixer streams every step and
    // shares its head and embedding, so the shared drafter Movable stays there.
    // Its MTP KV record stays in that owner's pool unit.
    let (draft, mut fixed): (Vec<_>, Vec<_>) = fixed.into_iter().partition(|d| d.gpu == 0
        && matches!(d.group.as_str(), "owner0 MTP weights" | "MTP expert weights" | "MTP expert workspace"));
    let mut draft = draft;
    if a.mtp { draft.push(Demand::new(0, Category::Kv, "owner0 MTP pending", pending, Basis::Formula)); }
    else { fixed.push(Demand::new(0, Category::Kv, "owner0 MTP pending", pending, Basis::Formula)); }
    let movables = if draft.is_empty() { Vec::new() } else {
        vec![crate::placement::draft::DrafterDemand::itemized(draft, 0).movable(2)?]
    };
    if a.full_prefill_logits > 0 {
        fixed.push(Demand::new(0, Category::Workspace, "probe prefill logits", a.full_prefill_logits, Basis::Formula));
    }
    let capacities = inputs.gpus.map(|g| g.capacity_bytes);
    let mut pool = PoolPolicy::resolve(&capacities, inputs.max_context, inputs.requested_pool,
        QWEN_UNIT_ROWS as u64, inputs.spark_ranks == 0).with_automatic_context(inputs.automatic_context);
    pool.ceiling = pool.target;
    Ok(PlacementRequest {
        inventory: Inventory { gpus: inputs.gpus.to_vec(), spark_ranks: inputs.spark_ranks, peer_access: true },
        attention_placement: Some(AttentionPlacement::Layers), context_buffers: ContextBuffers::default(), layers_first_gpu: 0,
        pool_bounded: Vec::new(),
        pool, layers, pool_overhead: overhead.to_vec(), fixed, startup: Vec::new(), movables, expert_workspace: 0,
        minimum_expert_layers: 0,
        tp2_workspace: inputs.tp2_workspace, onboard: inputs.onboard, expert_gpus: 2,
        policy: LayerPolicy { default: DUAL_WHOLE.to_vec(), by_kind: Vec::new(),
            ownership: LayerOwnership::NonemptyRanges, ..LayerPolicy::default() },
        hops: HopSpec { storage: crate::placement::HopStorage::ReceiveSlots, row_bytes: a.cfg.hc_width() as u64 * 2, rows: a.prefill_rows.max(64), lanes: 1,
            entry_gpu: 0, head_gpu: 0, terminal: TerminalHome::AttentionOwner },
        executor: ExecutorModes { family: "qwen4", modes: &DUAL_WHOLE, hops: true,
            attention: AttentionPlacement::Heads },
    })
}

/// Converge the owner-partitioned startup graph inventory and pool against the
/// same pre-allocation baselines. Refuse an empty chain or a graph/owner cycle;
/// neither condition is permission to invent a family-private boundary.
pub fn dual_placement(inputs: &QwenDualInputs<'_>) -> anyhow::Result<Placement> {
    use crate::serving_capacity::qwen_graphs::qwen_startup_graphs_placed;
    let owners = |p: &Placement| -> anyhow::Result<Vec<usize>> {
        p.layers.iter().map(|layer| match layer.mode {
            LayerMode::Whole { gpu, ffn: FfnMode::Owner } => Ok(usize::from(gpu)),
            _ => anyhow::bail!("Qwen dual requires whole owner layers"),
        }).collect()
    };
    let Some((sequences, speculation)) = inputs.startup_graph_modes else {
        let mut result = solve(&dual_request(inputs, [LAZY_GRAPH_BYTES; 2])?)?;
        check_dual_layer_owners(&owners(&result)?)?;
        for rank in &mut result.items {
            for item in rank {
                if item.group == "startup graphs" {
                    item.group = "graph growth".into();
                    item.lifetime = ItemLifetime::PostReadyReserve;
                }
            }
        }
        return Ok(result);
    };
    anyhow::ensure!(sequences <= 16, "Qwen startup graphs support at most 16 concurrent sequences");
    let result = solve(&dual_request(inputs, [0; 2])?)?;
    super::super::settle_pool((result.pool_tokens, owners(&result)?), |tokens, layer_owners| {
        check_dual_layer_owners(layer_owners)?;
        let graphs = qwen_startup_graphs_placed(usize::try_from(inputs.max_context)?,
            usize::try_from(tokens)?, inputs.admission.cfg.dense_context(), sequences, speculation, layer_owners)
            .ok_or_else(|| anyhow::anyhow!("Qwen owner graph count overflow"))?;
        let mut req = dual_request(inputs, [graphs.bytes(0), graphs.bytes(1)])?;
        req.pool.target = req.pool.target.min(tokens);
        req.pool.ceiling = req.pool.ceiling.min(tokens);
        let mut next = solve(&req)?;
        let next_owners = owners(&next)?;
        check_dual_layer_owners(&next_owners)?;
        for gpu in 0..2 {
            next.items[gpu].retain(|item| item.group != "startup graphs");
            next.items[gpu].extend(graphs.items(gpu));
        }
        Ok::<_, anyhow::Error>((next.pool_tokens, next_owners, next))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::families::qwen4::{Qwen4Config, Qwen4KvCache};
    use crate::serving_capacity::qwen_graphs::qwen_graph_pool;

    #[test]
    fn qwen_dual_owner0_weights_exclude_layer_and_table_storage() {
        use crate::families::qwen4::resident::Qwen4CheckpointResident;
        let resident = Qwen4CheckpointResident { target_bytes: 700, layer_bytes: vec![200, 400], entry_bytes: 100,
            mtp_bytes: 300, head_bytes: 500, embedding_bytes: 600, ple_table_bytes: 900, ple_row_bytes: 160 };
        for (host, expected) in [(false, 1500), (true, 900)] {
            let demands = dual_owner0_weights(&resident, host);
            assert!(demands.iter().all(|d| d.gpu == 0 && d.basis == Basis::Exact));
            assert_eq!(demands.iter().map(|d| d.bytes).sum::<u64>(), expected);
            assert_eq!(demands.iter().filter(|d| d.category == Category::Embedding).count(), usize::from(!host));
            assert!(!demands.iter().any(|d| d.category == Category::Tables || d.category == Category::Experts));
        }
        assert!(dual_owner0_weights(&Qwen4CheckpointResident::default(), false).is_empty());
    }

    #[test]
    fn qwen_dual_ple_demands_charge_only_scale() {
        use crate::families::qwen4::resident::Qwen4CheckpointResident;
        let mut cfg = Qwen4Config::from_hf(&crate::plan::testing::qwen4_config(4)).unwrap();
        cfg.ple_layers = vec![1];
        let resident = Qwen4CheckpointResident { ple_table_bytes: 123_456, ..Default::default() };
        let demands = dual_ple_demands(&cfg, 4, &resident).unwrap();
        assert!(demands.iter().enumerate().all(|(layer, extra)| layer == 1 || extra.is_empty()));
        assert_eq!(demands[1].iter().map(|d| d.bytes).sum::<u64>(), 256);
        assert!(demands[1].iter().all(|d| d.category == Category::Tables && d.residency == LayerResidency::Owner));
        assert!(dual_ple_demands(&cfg, 1, &resident).unwrap()[0].is_empty());
        assert!(dual_ple_demands(&cfg, 4, &Qwen4CheckpointResident::default()).is_err());
        let resident = Qwen4CheckpointResident { ple_table_bytes: u64::MAX, ..Default::default() };
        assert_eq!(dual_ple_demands(&cfg, 4, &resident).unwrap()[1][0].bytes, 256);
    }

    #[test]
    fn qwen_cutover_moves_ple_scale_to_the_selected_owner() {
        let mut cfg = Qwen4Config::from_hf(&crate::plan::testing::qwen4_config(4)).unwrap();
        cfg.ple_layers = vec![1];
        cfg.ple_dim = 2560;
        let resident = crate::families::qwen4::resident::Qwen4CheckpointResident {
            ple_table_bytes: 1 << 30, ..Default::default() };
        let extra = dual_ple_demands(&cfg, 4, &resident).unwrap();
        let inputs = QwenInputs { automatic_context: false,
            admission: QwenAdmissionInputs { prefill_shape: None, cfg: &cfg, layers: 4, mtp: false, kv_format: Qwen4KvCache::Bf16,
                manifest: None, prefill_rows: 4096, slots: 16, mark_bytes: 0, full_prefill_logits: 0,
                ple: None, future_expert_bytes: 0, headroom: 0 },
            capacity_bytes: 96 << 30, baseline: Baseline::Measured { free_bytes: 96 << 30 },
            pending_code_bytes: 0, max_context: 131072, requested_pool: Some(2 << 20), spark_ranks: 0,
            startup_graph_modes: None,
        };
        let mut req = request(&inputs, 0).unwrap();
        req.inventory.gpus = vec![req.inventory.gpus[0]; 2];
        req.inventory.peer_access = true;
        req.attention_placement = Some(AttentionPlacement::Layers);
        req.executor = ExecutorModes { family: "qwen4", modes: &DUAL_WHOLE, hops: true,
            attention: AttentionPlacement::Heads };
        req.policy.default = DUAL_WHOLE.to_vec();
        req.fixed.clear();
        req.pool_overhead = vec![0; 2];
        for (layer, extra) in req.layers.iter_mut().zip(extra) {
            layer.modes = DUAL_WHOLE.to_vec();
            layer.kv_unit = ModeBytes::replicated((1 << 30) / 4096).into();
            layer.extra = extra;
        }
        // Charge enough entry pressure to move the capacity-maximizing cut,
        // rather than balancing only the bytes of the requested 2M pool.
        for (entry, cut, owner) in [(0, 2, 0), (64 << 30, 1, 1)] {
            req.fixed = vec![Demand::new(0, Category::Weights, "entry", entry, Basis::Exact)];
            let placement = solve(&req).unwrap();
            assert_eq!(placement.pool_tokens, 2 << 20);
            assert_eq!(placement.layers.iter().take_while(|l| l.mode == DUAL_WHOLE[0]).count(), cut);
            assert_eq!(placement.layers[1].mode, DUAL_WHOLE[owner]);
            for gpu in 0..2 {
                assert_eq!(placement.items[gpu].iter().filter(|i| i.category == Category::Tables)
                    .map(|i| i.bytes).sum::<u64>(), if gpu == owner { 256 } else { 0 });
            }
        }
    }

    #[test]
    fn qwen_dual_projection_staging_preserves_bf16_and_host_head_defaults() {
        use crate::families::qwen4::resident::Qwen4Representation;
        let cfg = Qwen4Config::from_hf(&crate::plan::testing::qwen4_config(48)).unwrap();
        for mtp in [false, true] {
            for fp8_head in [false, true] {
                let selected = Qwen4Representation { fp8_projections: false, fp8_head };
                assert!(dual_projection_load_staging(&cfg, 48, mtp, selected).is_empty());
                let demands = dual_projection_load_staging(&cfg, 48, mtp,
                    Qwen4Representation { fp8_projections: true, fp8_head });
                assert_eq!(demands.len(), 2);
                for (gpu, demand) in demands.iter().enumerate() {
                    assert_eq!(demand.gpu, gpu as u8);
                    assert_eq!(demand.bytes, 84_377_600);
                    assert_eq!(demand.category, Category::Staging);
                    assert_eq!(demand.group, "FP8 projection load staging");
                }
            }
        }
    }

    #[test]
    fn qwen_tp2_transport_matches_owner_buffers_and_abi() -> anyhow::Result<()> {
        for wire in [false, true] {
            for rows in [1, 3, 64, 4096] {
                let p = QwenTp2Rows::new(2560, 10, rows, wire)?;
                let align = |bytes: u64| bytes.div_ceil(16) * 16;
                let payload = align(rows * if wire { 2640 } else { 5120 }) + 2 * align(rows * 40);
                assert_eq!(p.bytes, payload);
                assert_eq!(p.partial_bytes, rows * 2560 * if wire { 4 } else { 2 });
                assert_eq!(p.reduction_bytes, rows * 5120);
                assert_eq!(p.exchange_slot_bytes, align(payload.max(p.partial_bytes)));
                let demands = p.demands()?;
                assert_eq!(demands.len(), 10);
                for rank in 0..2 {
                    let rank_bytes = demands.iter().filter(|d| d.gpu == rank).map(|d| d.bytes).sum::<u64>();
                    assert_eq!(rank_bytes, payload + 2 * p.exchange_slot_bytes.max(256) + rows * 5120 + 512);
                }
            }
        }
        let tiny = QwenTp2Rows::new(32, 1, 1, true)?;
        assert_eq!(tiny.demands()?.iter().filter(|d| d.group == "expert receive")
            .map(|d| d.bytes).collect::<Vec<_>>(), vec![512, 512]);
        for (hidden, topk, rows, wire) in [(0, 10, 64, false), (2560, 0, 64, false),
            (2560, 10, 0, true), (2561, 10, 64, true), (u64::MAX, 10, 64, false)] {
            assert!(QwenTp2Rows::new(hidden, topk, rows, wire).is_err());
        }
        Ok(())
    }

    #[test]
    fn qwen_tp2_costs_use_exact_header_storage() {
        use crate::plan::testing::{qwen4_config, qwen4_exl3, nvfp4, write_snapshot};
        let dir = tempfile::tempdir().unwrap();
        let config = qwen4_config(1);
        let (tensors, manifest) = qwen4_exl3(1, 4);
        write_snapshot(dir.path(), &config, &tensors, None);
        crate::plan::testing::write_quantize_config(dir.path(), &manifest);
        let catalog = crate::read_expert_catalog(dir.path()).unwrap();
        let costs = dual_expert_costs(&catalog, 1).unwrap();
        let cost = costs[0].unwrap();
        assert!(cost.half[0].resident > cost.half[1].resident);
        for rank in 0..2 {
            let residency = catalog.exl3().unwrap().residency(crate::V41Exl3Layer::Backbone(0), 2, rank).unwrap();
            assert_eq!(cost.half[rank].resident, residency.device_arena_layout().unwrap().1 as u64);
            assert_eq!(cost.half[rank].staging, 0);
        }
        let packages = [dir.path().join("rtx-tp2-rank0"), dir.path().join("rtx-tp2-rank1")];
        for (rank, width) in [384, 256].into_iter().enumerate() {
            for capacity in [1, 16, 80] {
                let path = packages[rank].join(format!("m{capacity}"));
                std::fs::create_dir_all(&path).unwrap();
                let manifest = serde_json::json!({"hidden": 2560, "intermediate": width, "experts": 512,
                    "top_k": 10, "capacity": capacity, "output_dtype": "fp32", "tile": [64, 128, 64, 128], "trellis_lut": {"bytes": 32},
                    "buffers": {"scratch": {"allocation": "scratch", "bytes": capacity * width,
                        "dtype": "f16", "zero_on_create": false},
                        "state": {"allocation": "state", "bytes": 16, "zero_on_create": true}}});
                std::fs::write(path.join("v41_exl3.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
            }
        }
        assert_eq!(dual_expert_workspace(&catalog, packages.each_ref().map(|p| p.as_path()), 16).unwrap(),
            [80 * 384 + 3 * (32 + 16) + 97 * 2560 * 2 + 16 * 2560 * 4,
             80 * 256 + 3 * (32 + 16) + 97 * 2560 * 2 + 16 * 2560 * 4]);
        assert!(dual_expert_workspace(&catalog, [packages[1].as_path(), packages[0].as_path()], 16).is_err());
        assert_eq!(dual_expert_workspace(&catalog, packages.each_ref().map(|p| p.as_path()), 17).unwrap(),
            [80 * 384 + 3 * (32 + 16) + 97 * 2560 * 2 + 17 * 2560 * 4,
             80 * 256 + 3 * (32 + 16) + 97 * 2560 * 2 + 17 * 2560 * 4]);
        assert!(dual_expert_workspace(&catalog, packages.each_ref().map(|p| p.as_path()), 81).is_err());
        assert!(dual_expert_workspace(&catalog, packages.each_ref().map(|p| p.as_path()), 4097).is_err());
        std::fs::remove_file(packages[1].join("m80/v41_exl3.json")).unwrap();
        assert!(dual_expert_workspace(&catalog, packages.each_ref().map(|p| p.as_path()), 16).is_err());
        let mut config = config;
        config["quantization_config"] = serde_json::json!({"quant_method": "modelopt", "quant_algo": "NVFP4",
            "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "float", "group_size": 16}}}});
        let mut tensors = Vec::new();
        for expert in 0..512 {
            for (projection, n, k) in [("gate_proj", 640, 2560), ("up_proj", 640, 2560), ("down_proj", 2560, 640)] {
                tensors.extend(nvfp4(&format!("model.language_model.layers.0.mlp.experts.{expert}.{projection}"), n, k));
            }
        }
        // New directory avoids the EXL3 side-file cross-check from the first fixture.
        let dir = tempfile::tempdir().unwrap();
        write_snapshot(dir.path(), &config, &tensors, None);
        let catalog = crate::read_expert_catalog(dir.path()).unwrap();
        let cost = dual_expert_costs(&catalog, 1).unwrap()[0].unwrap();
        // Three projection value+scale grids, plus alpha and input scale per expert.
        for (rank, width) in [384_u64, 256].into_iter().enumerate() {
            assert_eq!(cost.half[rank], Bytes2 { resident: 512 * (3 * width * 2560 * 9 / 16 + 3 * 8), staging: 0 });
        }
        assert_eq!(cost.whole.resident, 512 * (3 * 640 * 2560 * 9 / 16 + 3 * 8));
        let package = |width, scratch| serde_json::json!({"tp": 2, "hidden": 2560, "slice": width,
            "intermediate": 640, "experts": 512, "top_k": 10, "input": "bf16", "weights": "nvfp4a4",
            "capacities": [{"capacity": 1, "scratch_bytes": 128}, {"capacity": 16, "scratch_bytes": scratch}]});
        let mut manifest = serde_json::json!({"layouts": {"tp2-w384": package(384, 1000),
            "tp2-w256": package(256, 2000)}});
        let save = |value: &serde_json::Value| std::fs::write(dir.path().join("manifest.json"),
            serde_json::to_vec(value).unwrap()).unwrap();
        save(&manifest);
        let packages = [dir.path().join("tp2-w384"), dir.path().join("tp2-w256")];
        let paths = packages.each_ref().map(|p| p.as_path());
        assert_eq!(dual_expert_workspace(&catalog, paths, 9).unwrap(), [1000 + 9 * 2560 * 2, 2000 + 9 * 2560 * 2]);
        assert_eq!(dual_expert_workspace(&catalog, paths, 1).unwrap(), [256 + 2560 * 2; 2]);
        assert!(dual_expert_workspace(&catalog, [paths[1], paths[0]], 9).is_err());
        assert!(dual_expert_workspace(&catalog, paths, 17).is_err());
        assert!(dual_expert_workspace(&catalog, paths, 0).is_err());
        manifest["layouts"]["tp2-w256"]["input"] = serde_json::json!("wire");
        save(&manifest);
        assert!(dual_expert_workspace(&catalog, paths, 9).is_err());

        config["text_config"]["mtp_num_hidden_layers"] = 1.into();
        for expert in 0..512 {
            for (projection, n, k) in [("gate_proj", 640, 2560), ("up_proj", 640, 2560), ("down_proj", 2560, 640)] {
                tensors.extend(nvfp4(&format!("mtp.layers.0.mlp.experts.{expert}.{projection}"), n, k));
            }
        }
        let draft_dir = tempfile::tempdir().unwrap();
        write_snapshot(draft_dir.path(), &config, &tensors, None);
        let catalog = crate::read_expert_catalog(draft_dir.path()).unwrap();
        let lib = draft_dir.path().join("lib/libcuteafd_native.so");
        let root = lib.parent().unwrap().join("fp8/fp8-qwen4-nvfp4a4");
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("manifest.json");
        let mut draft_manifest = serde_json::json!({"layouts": {"tp1": package(640, 1000)}});
        draft_manifest["layouts"]["tp1"]["tp"] = 1.into();
        let save_draft = |value: &serde_json::Value| std::fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
        save_draft(&draft_manifest);
        let demand = dual_draft_demands(&catalog, &lib, 9).unwrap();
        assert_eq!(demand[0].bytes, cost.whole.resident);
        assert_eq!(demand[1].bytes, 1000);
        assert!(dual_draft_demands(&catalog, &lib, 0).is_err());
        assert!(dual_draft_demands(&catalog, &lib, 17).is_err());
        for (key, invalid) in [("tp", serde_json::json!(2)), ("slice", serde_json::json!(384)),
            ("input", serde_json::json!("wire")), ("weights", serde_json::json!("fp8"))] {
            let mut wrong = draft_manifest.clone();
            wrong["layouts"]["tp1"][key] = invalid;
            save_draft(&wrong);
            assert!(dual_draft_demands(&catalog, &lib, 9).is_err(), "invalid {key}");
        }
    }

    #[test]
    fn dual_card_scratch_selects_each_owner_and_diagnostics_are_opt_in() {
        let manifest = serde_json::json!({"programs": [{"name": "qwen4_sparse_gqa_kv_fp8_m64",
            "card_plans": [
                {"sm_count": 170, "scratch_bytes_at_capacity": {"scratch": 8192}, "geometry": {}},
                {"sm_count": 188, "scratch_bytes_at_capacity": {"scratch": 1024}, "geometry": {}}
            ]}]});
        let plan = cuteafd_core::coordinator_programs::AttentionReductionPlan::new(64, None, None).unwrap();
        for sms in [[170, 188], [188, 170]] {
            let owners = dual_program_manifests(&manifest, sms, plan).unwrap();
            for rank in 0..2 {
                assert_eq!(crate::serving_capacity::qwen_graphs::qwen_step_scratch(
                    &owners[rank], true, false, Qwen4KvCache::Fp8).0,
                    if sms[rank] == 170 { 8192 } else { 1024 });
            }
        }
        assert!(dual_peer_diagnostics(false).is_empty());
        let demands = dual_peer_diagnostics(true);
        assert_eq!(demands.len(), 2);
        for (rank, demand) in demands.iter().enumerate() {
            assert_eq!(demand.gpu as usize, rank);
            assert_eq!(demand.bytes, 256);
        }
    }

    #[test]
    fn dual_mtp_drafter_is_a_movable_placed_like_the_former_fixed_demands() {
        let mut cfg = Qwen4Config::from_hf(&crate::plan::testing::qwen4_config(48)).unwrap();
        cfg.mtp_layers = 1;
        let weights = vec![100 << 20; 48];
        let experts = vec![Some(ExpertCost { whole: Bytes2::default(),
            half: [Bytes2 { resident: 600 << 20, staging: 64 << 20 }; 2], tp2: true, spark_ok: true }); 48];
        let layer_extra = vec![Vec::new(); 48];
        for mtp in [false, true] {
            let mut fixed = vec![Demand::new(0, Category::Weights, "owner0 head", 1 << 30, Basis::Exact)];
            if mtp {
                fixed.extend([Demand::new(0, Category::Weights, "owner0 MTP weights", 900 << 20, Basis::Exact),
                    Demand::new(0, Category::Experts, "MTP expert weights", 1300 << 20, Basis::Exact),
                    Demand::new(0, Category::Experts, "MTP expert workspace", 64 << 20, Basis::Formula)]);
            }
            let inputs = QwenDualInputs { automatic_context: false, owner_manifests: [None, None],
                admission: QwenAdmissionInputs { prefill_shape: None, cfg: &cfg, layers: 48, mtp, kv_format: Qwen4KvCache::Bf16, manifest: None,
                    prefill_rows: 4096, slots: 16, mark_bytes: 0, full_prefill_logits: 0,
                    ple: None, future_expert_bytes: 0, headroom: 3 << 30 },
                gpus: [0, 1].map(|_| GpuBudget { capacity_bytes: 96 << 30, headroom_bytes: 3 << 30,
                    baseline: Baseline::Planned { context_bytes: 512 << 20, loaded_bytes: 0 } }),
                layer_weights: &weights, layer_experts: &experts, layer_extra: &layer_extra, fixed,
                tp2_workspace: [128 << 20; 2], pending_code_bytes: [64 << 20; 2], mark_slots: 0,
                max_context: 131072, requested_pool: None, spark_ranks: 0, onboard: Onboard::Auto,
                startup_graph_modes: None };
            let lazy = dual_placement(&inputs).unwrap();
            for rank in &lazy.items {
                let graph = rank.iter().find(|item| item.group == "graph growth").unwrap();
                assert_eq!(graph.lifetime, ItemLifetime::PostReadyReserve);
                assert_eq!(graph.bytes, LAZY_GRAPH_BYTES);
            }
            let req = dual_request(&inputs, [256 << 20; 2]).unwrap();
            assert_eq!(req.movables.len(), usize::from(mtp));
            if mtp {
                assert_eq!(req.movables[0].allowed, [0]);
                assert_eq!(req.movables[0].conditional[0].demands.iter().map(|d| d.group.as_str()).collect::<Vec<_>>(),
                    ["owner0 MTP weights", "MTP expert weights", "MTP expert workspace", "owner0 MTP pending"]);
                assert!(req.fixed.iter().all(|d| !d.group.contains("MTP")));
                crate::placement::draft::assert_same_as_fixed(&req);
            } else {
                assert!(req.fixed.iter().any(|d| d.group == "owner0 MTP pending"));
            }
        }
    }

    #[test]
    fn planner_equals_runtime_qwen4_dual_contract() {
        use crate::serving_capacity::qwen_graphs::qwen_cache_geometry_placed;
        let mut cfg = Qwen4Config::from_hf(&crate::plan::testing::qwen4_config(48)).unwrap();
        cfg.ple_layers = vec![23];
        cfg.ple_dim = 2560;
        let resident = crate::families::qwen4::resident::Qwen4CheckpointResident {
            ple_table_bytes: 2 << 30, ple_row_bytes: 160, ..Default::default() };
        let weights = vec![100 << 20; 48];
        let experts = vec![Some(ExpertCost { whole: Bytes2::default(),
            half: [Bytes2 { resident: 600 << 20, staging: 64 << 20 }, Bytes2 { resident: 400 << 20, staging: 32 << 20 }],
            tp2: true, spark_ok: true }); 48];
        for (kv_format, wire, fp8_projections) in [Qwen4KvCache::Bf16, Qwen4KvCache::Fp8].into_iter()
            .flat_map(|kv| [false, true].into_iter().flat_map(move |wire|
                [false, true].map(move |fp8| (kv, wire, fp8)))) {
            let layer_extra = dual_ple_demands(&cfg, 48, &resident).unwrap();
            let transport = QwenTp2Rows::new(cfg.hidden as u64, cfg.topk as u64, 4096, wire).unwrap()
                .demands().unwrap();
            let staging = dual_projection_load_staging(&cfg, 48, false,
                crate::families::qwen4::resident::Qwen4Representation { fp8_projections, fp8_head: false });
            let mut fixed = vec![Demand::new(0, Category::Weights, "owner0 head", 1 << 30, Basis::Exact)];
            fixed.extend(transport.iter().chain(&staging).cloned());
            let manifest = serde_json::json!({"programs": [{"name": "qwen4_sparse_gqa_kv_fp8_m64",
                "card_plans": [
                    {"sm_count": 170, "scratch_bytes_at_capacity": {"scratch": 8192}, "geometry": {}},
                    {"sm_count": 188, "scratch_bytes_at_capacity": {"scratch": 1024}, "geometry": {}}
                ]}]});
            let manifests = dual_program_manifests(&manifest, [170, 188],
                cuteafd_core::coordinator_programs::AttentionReductionPlan::new(64, None, None).unwrap()).unwrap();
            let inputs = |measured| QwenDualInputs { automatic_context: false,
                owner_manifests: manifests.each_ref().map(Some),
                admission: QwenAdmissionInputs { prefill_shape: None, cfg: &cfg, layers: 48, mtp: false, kv_format, manifest: None,
                    prefill_rows: 4096, slots: 16, mark_bytes: 0, full_prefill_logits: 0,
                    ple: None, future_expert_bytes: 0, headroom: 3 << 30 },
                gpus: [0, 1].map(|_| GpuBudget { capacity_bytes: 96 << 30, headroom_bytes: 3 << 30,
                    baseline: if measured { Baseline::Measured { free_bytes: (96 << 30) - (512 << 20) } }
                        else { Baseline::Planned { context_bytes: 512 << 20, loaded_bytes: 0 } } }),
                layer_weights: &weights, layer_experts: &experts, layer_extra: &layer_extra,
                fixed: fixed.clone(),
                tp2_workspace: [128 << 20, 96 << 20], pending_code_bytes: [64 << 20; 2], mark_slots: 18,
                max_context: 131072, requested_pool: Some(2 << 20), spark_ranks: 0, onboard: Onboard::Auto,
                startup_graph_modes: Some((16, true)),
            };
            let planned = dual_placement(&inputs(false)).unwrap();
            let measured = dual_placement(&inputs(true)).unwrap();
            assert_eq!(planned, measured);
            assert_eq!(planned.pool_tokens, 2 << 20);
            let decode = |rank: usize| planned.items[rank].iter().find(|item| item.group == "decode step").unwrap().bytes;
            assert_eq!(decode(0) - decode(1), if kv_format == Qwen4KvCache::Fp8 { 8192 - 1024 } else { 0 });
            for enabled in [false, true] {
                let mut diagnostic = inputs(false);
                diagnostic.fixed.extend(dual_peer_diagnostics(enabled));
                let request = dual_request(&diagnostic, [0; 2]).unwrap();
                assert_eq!(request.fixed.iter().filter(|d| d.group == "expert peer diagnostics").count(),
                    if enabled { 2 } else { 0 });
            }
            for demand in transport.iter().chain(&staging) {
                assert!(planned.items[usize::from(demand.gpu)].iter().any(|item|
                    item.group == demand.group && item.category == demand.category && item.bytes == demand.bytes));
            }
            for gpu in 0..2 {
                assert_eq!(planned.items[gpu].iter().filter(|item| item.group == "FP8 projection load staging")
                    .map(|item| item.bytes).sum::<u64>(), if fp8_projections { 84_377_600 } else { 0 });
            }
            assert!(planned.layers.iter().all(|layer| layer.experts == ExpertHome::RtxTp2));
            let owners = planned.layers.iter().map(|layer| match layer.mode {
                LayerMode::Whole { gpu, .. } => usize::from(gpu), _ => unreachable!(),
            }).collect::<Vec<_>>();
            check_dual_layer_owners(&owners).unwrap();
            for gpu in 0..2 {
                let tables = planned.items[gpu].iter().filter(|item| item.category == Category::Tables)
                    .map(|item| item.bytes).sum::<u64>();
                assert_eq!(tables, if gpu == owners[23] { 256 } else { 0 });
            }
            assert_eq!(planned.hops.len(), 2);
            let req = dual_request(&inputs(false), [0; 2]).unwrap();
            let geometry = qwen_cache_geometry_placed(&cfg, &owners, false, kv_format).unwrap();
            for gpu in 0..2 {
                let state = req.layers.iter().zip(&owners).filter(|(_, owner)| **owner == gpu)
                    .map(|(layer, _)| layer.fixed_bytes.whole).sum::<u64>();
                let rank = geometry.ranks[gpu];
                assert_eq!(state, rank.active_state_per_sequence_bytes * 16 + rank.retained_mark_bytes * 18
                    + rank.speculative_replay_bytes);
                let records = req.layers.iter().zip(&owners).filter(|(_, owner)| **owner == gpu)
                    .map(|(layer, _)| layer.kv_unit.unit_bytes_whole).sum::<u64>();
                let tables = 65 * 5 * 4;
                assert_eq!(records + req.pool_overhead[gpu],
                    (rank.persistent_unit_bytes + rank.pool_metadata_unit_bytes + tables).div_ceil(256) * 256);
            }
        }
    }

    #[test]
    fn planner_equals_runtime_qwen4() {
        let mut config = crate::plan::testing::qwen4_config(48);
        config["text_config"]["mtp_num_hidden_layers"] = serde_json::json!(1);
        let cfg = Qwen4Config::from_hf(&config).unwrap();
        for (total, kv_format) in [32_u64 << 30, 101_973_491_712].into_iter().flat_map(|total|
            [Qwen4KvCache::Bf16, Qwen4KvCache::Fp8].map(|format| (total, format))) {
            for (expert_bytes, mtp) in [(45_u64 << 30, false), (50_u64 << 30, true)] {
                for requested in [None, Some(32768), Some(2097152)] {
                    let context = requested.unwrap_or(131072).min(131072);
                    let context_bytes = 586_416_128;
                    let inputs = |baseline| QwenInputs { automatic_context: false,
                        admission: QwenAdmissionInputs { prefill_shape: None, cfg: &cfg, layers: 48, mtp, kv_format, manifest: None,
                            prefill_rows: 4096, slots: 16, mark_bytes: 256 << 20, full_prefill_logits: 0,
                            ple: Some((160, true)), future_expert_bytes: 0, headroom: 3 << 30 },
                        capacity_bytes: total, baseline, pending_code_bytes: 0, max_context: context,
                        requested_pool: requested, spark_ranks: 0, startup_graph_modes: Some((16, true)),
                    };
                    let planned = inputs(Baseline::Planned { context_bytes, loaded_bytes: expert_bytes });
                    let measured = inputs(Baseline::Measured {
                        free_bytes: total.saturating_sub(context_bytes + expert_bytes) });
                    match (placement(&planned), placement(&measured)) {
                        (Ok(p), Ok(r)) => {
                            assert_eq!(p, r);
                            let admission = qwen_admission(&planned.admission).unwrap();
                            let policy = PoolPolicy::resolve(&[total], context, requested, 256, true);
                            let old = qwen_graph_pool(total - context_bytes - expert_bytes, admission.fixed(),
                                admission.per_token, policy.target, requested, |tokens|
                                    qwen_startup_graphs(context as usize, tokens as usize, cfg.dense_context(), 16, true, 48)).unwrap();
                            assert_eq!(p.pool_tokens, old.0);
                            assert_eq!(p.items[0].iter().map(|i| i.bytes).sum::<u64>(),
                                admission.fixed() - admission.headroom + old.1.bytes(0) + old.0 * admission.per_token);
                        }
                        (Err(p), Err(r)) => assert_eq!(p.to_string(), r.to_string()),
                        other => panic!("planner/runtime disagree: {other:?}"),
                    }
                }
            }
        }
    }

    #[test]
    fn spark_intake_is_charged_once_across_planned_and_measured_baselines() {
        let cfg = Qwen4Config::from_hf(&crate::plan::testing::qwen4_config(48)).unwrap();
        let (total, context_bytes, loaded_bytes) = (32_u64 << 30, 1_u64 << 30, 23_u64 << 30);
        for shape in [None, Some(crate::serving_capacity::PrefillShape { unit_rows: 2048, waves: 2 })] {
            let inputs = |baseline| QwenInputs { automatic_context: false,
                admission: QwenAdmissionInputs { prefill_shape: shape, cfg: &cfg, layers: 48, mtp: false,
                    kv_format: Qwen4KvCache::Fp8, manifest: None, prefill_rows: 4096, slots: 16,
                    mark_bytes: 0, full_prefill_logits: 0, ple: Some((160, true)),
                    future_expert_bytes: 0, headroom: 0 },
                capacity_bytes: total, baseline, pending_code_bytes: 0, max_context: 1048576,
                requested_pool: None, spark_ranks: 4, startup_graph_modes: None };
            let planned = request(&inputs(Baseline::Planned { context_bytes, loaded_bytes }), 0).unwrap();
            let intake = planned.fixed.iter().find(|d| d.group == "Spark intake planes").unwrap().clone();
            assert_eq!(intake.bytes, 80 << 20); // The checkpoint's hidden width is 2560, not 2048.
            let measured = inputs(Baseline::Measured {
                free_bytes: total - context_bytes - loaded_bytes - intake.bytes });
            let runtime = request(&measured, 0).unwrap();
            assert!(!runtime.fixed.iter().any(|d| d.group == "Spark intake planes"));
            let (p, r) = (solve(&planned).unwrap(), solve(&runtime).unwrap());
            assert_eq!(p.pool_tokens, r.pool_tokens);
            assert!(r.pool_tokens < runtime.pool.target, "fixture must be memory limited");
            let mut double_charged = runtime;
            double_charged.fixed.push(intake.clone());
            let old = solve(&double_charged).unwrap();
            let delta = r.pool_tokens - old.pool_tokens;
            let per_token = qwen_admission(&measured.admission).unwrap().per_token;
            assert!(delta > 0);
            assert!((delta * per_token).abs_diff(intake.bytes) < 256 * per_token,
                "only one pool allocation unit may be lost to rounding");
            println!("Qwen Spark intake {:?}: {} bytes, pool {} -> {} (+{delta} tokens)",
                shape, intake.bytes, old.pool_tokens, r.pool_tokens);
        }
    }

    #[test]
    fn qwen_pool_demands_match_geometry_and_lazily_loaded_experts() {
        let mut config = crate::plan::testing::qwen4_config(48);
        config["text_config"]["mtp_num_hidden_layers"] = serde_json::json!(1);
        let cfg = Qwen4Config::from_hf(&config).unwrap();
        for (mtp, kv_format) in [false, true].into_iter().flat_map(|mtp|
            [Qwen4KvCache::Bf16, Qwen4KvCache::Fp8].map(|format| (mtp, format))) {
            let inputs = QwenInputs { automatic_context: false,
                admission: QwenAdmissionInputs { prefill_shape: None, cfg: &cfg, layers: 48, mtp, kv_format, manifest: None,
                    prefill_rows: 4096, slots: 16, mark_bytes: 256 << 20, full_prefill_logits: 0,
                    ple: None, future_expert_bytes: 10 << 30, headroom: 3 << 30 },
                capacity_bytes: 96 << 30, baseline: Baseline::Measured { free_bytes: 70 << 30 },
                pending_code_bytes: 780_221_844, max_context: 131072, requested_pool: None,
                spark_ranks: 0, startup_graph_modes: None,
            };
            let lazy = placement(&inputs).unwrap();
            let graph = lazy.items[0].iter().find(|item| item.group == "graph growth").unwrap();
            assert_eq!(graph.lifetime, ItemLifetime::PostReadyReserve);
            assert_eq!(graph.bytes, LAZY_GRAPH_BYTES);
            let req = request(&inputs, LAZY_GRAPH_BYTES).unwrap();
            let admission = qwen_admission(&inputs.admission).unwrap();
            assert_eq!(req.layers.iter().map(|l| l.kv_unit.unit_bytes_whole).sum::<u64>() + req.pool_overhead[0],
                admission.per_token * 256);
            assert!(req.fixed.iter().any(|d| d.group == "lazy EXL3 window" && d.bytes == 10 << 30));
            assert!(req.fixed.iter().any(|d| d.group == "loaded code" && d.bytes == 780_221_844));
        }
    }
}
