//! One fact gatherer for Hy4-flash admission shared by `cuteafd plan --layout`
//! and `serve-hy4f`. They differ only in [`Coordinator`] (a planned card forecast
//! vs one measured pre-weight CUDA sample per GPU). The result is the shared
//! recipe [`Hy4Inputs`], admitted by `placement::planner::solve`.
use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Context as _;

use crate::families::hy4_flash::engram::{mapped_engram_bytes, Hy4EngramShape};
use crate::families::hy4_flash::execution::{CacheLayout, INDEX_GROUP_RECORD_BYTES, PAGE_ROWS};
use crate::families::hy4_flash::weight_layout::{fp8_role, target_arenas};
use crate::families::hy4_flash::{Hy4FlashConfig, Hy4KvCache};
use crate::formats::mapped_preload::MappedPlacement;
use crate::formats::trellis_core::CoreProjection;
use crate::placement::families::hy4_flash::{
    self, Hy4Inputs, LocalBacking, LocalPackage, SparkCharge, LANES,
};
use crate::placement::planner::{Baseline, Budget, Context, Device, Domain, Location, PrefillShape};
use crate::placement::{
    ArchContext, AttentionPlacement, LayerMode, PoolFacts, PoolMemory, ResidentWeights,
    SparkWorkingSet,
};
use crate::serving_capacity::{PackedWeightRegions, WeightArenaLayout};

/// A refusal both plan and serve report the same way: what failed and how to fix it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{what}")]
pub struct Refused {
    pub what: String,
    pub how: String,
}

fn refused(what: impl Into<String>, how: impl Into<String>) -> Refused {
    Refused { what: what.into(), how: how.into() }
}

/// The coordinator edge: planned cards or measured pre-weight samples.
#[derive(Debug, Clone)]
pub enum Coordinator {
    /// Planned card caps in bytes per GPU.
    Planned { rtx_bytes: Vec<u64> },
    /// One CUDA sample per GPU before weights: `(physical_total, logical_cap, free)`
    /// plus pending unloaded code allowance per GPU.
    Measured { samples: Vec<(u64, u64, u64)>, loaded_code: Vec<u64> },
}

impl Coordinator {
    pub fn gpus(&self) -> usize {
        match self {
            Self::Planned { rtx_bytes } => rtx_bytes.len(),
            Self::Measured { samples, .. } => samples.len(),
        }
    }

    pub fn effective_cap(&self, gpu: usize) -> u64 {
        match self {
            Self::Planned { rtx_bytes } => rtx_bytes[gpu],
            Self::Measured { samples, .. } => samples[gpu].0.min(samples[gpu].1),
        }
    }
}

/// Every operator input Hy4-flash admission reads, resolved once by the caller.
pub struct FactsRequest<'a> {
    pub checkpoint: &'a crate::plan::Checkpoint,
    pub catalog: &'a crate::OfficialV41Catalog,
    pub cfg: &'a Hy4FlashConfig,
    pub model_core: &'a BTreeMap<String, CoreProjection>,
    pub kv_format: Hy4KvCache,
    pub dfly2: bool,
    pub tp2_dflash: bool,
    pub manifest_path: Option<&'a Path>,
    pub coordinator: Coordinator,
    pub attention: AttentionPlacement,
    pub spark_ranks: usize,
    pub spark_bytes: Vec<u64>,
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
    pub prefix_mark_budget: u64,
    pub prefix_slots: Option<u64>,
    pub replies: cuteafd_core::spark_reply::SparkReplyFormats,
    pub decode_graphs: bool,
    pub expert_usage: bool,
    pub resident_weight_limit: Option<ResidentWeights>,
    /// Bundled ViT placement policy (`VISION`), replicas and host feature cache.
    pub vision: crate::plan::MediaMode,
    pub vision_replicas: usize,
    pub media_cache_bytes: Option<u64>,
}

/// Shared installed RTX onboarding owners from the real expert catalog and
/// installed `exl3-hy4f-k45-mul1` manifests. Returns `LocalPackage::Missing`
/// honestly whenever the package is absent or incompatible.
pub fn build_local_backings(
    catalog: &crate::OfficialV41Catalog,
    manifest: Option<&Path>,
    gpus: usize,
    rows: u64,
) -> LocalPackage {
    let workspace = || -> anyhow::Result<LocalPackage> {
        anyhow::ensure!((1..=2).contains(&gpus), "Hy4-flash local package requires one or two ranks");
        let exl3 = catalog
            .exl3()
            .context("Hy4-flash RTX onboarding requires an EXL3 MUL1 expert catalog")?;
        let routed = catalog.routed_experts();
        let layers: Vec<usize> = (routed.first_layer..routed.layers).collect();
        let requests = crate::placement::rtx_onboard::expert_workspace_requests(
            catalog, manifest, rows, gpus, "hy4f",
        )?;
        let exl3_regions: Vec<Vec<(String, usize)>> = (0..gpus)
            .map(|gpu| {
                layers
                    .iter()
                    .map(|&layer| {
                        let layout = exl3.residency(crate::V41Exl3Layer::Backbone(layer), gpus, gpu)?;
                        let name = if gpus == 1 {
                            format!("BackboneFull {{ layer: {layer} }}.0")
                        } else {
                            format!("BackboneTp2 {{ layer: {layer}, rank: {gpu} }}.0")
                        };
                        Ok((name, layout.device_arena_layout()?.1))
                    })
                    .collect::<anyhow::Result<Vec<_>>>()
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let last = requests
            .len()
            .checked_sub(1)
            .context("Hy4-flash EXL3 output owner absent")?;
        let mut backings = Vec::with_capacity(layers.len() + 1);
        backings.push(Vec::new()); // count == 0
        for count in 1..=layers.len() {
            let selected = layers[..count].to_vec();
            let mut owners = Vec::with_capacity(gpus);
            for gpu in 0..gpus {
                let regions = exl3_regions[gpu][..count].to_vec();
                let id = crate::serving_capacity::expert_weight_arena_id("exl3", gpu as i32, &regions)?;
                let workspace = requests
                    .iter()
                    .enumerate()
                    .map(|(index, &bytes)| {
                        (
                            if index == last { "output".into() } else { format!("scratch.{index}") },
                            bytes,
                        )
                    })
                    .collect();
                owners.push(LocalBacking {
                    gpu,
                    layers: selected.clone(),
                    arena: WeightArenaLayout::new(id, regions)?,
                    workspace,
                    // Hy4-flash TP2 currently reduces expert partials through SharedScratch::delta_f32
                    // rather than a separate PeerExchange; keep tp2_rows None until PeerExchange lands.
                    tp2_rows: None,
                });
            }
            backings.push(owners);
        }
        Ok(LocalPackage::Installed { backings })
    };
    workspace().unwrap_or_else(|error| LocalPackage::Missing(format!("{error:#}")))
}

/// Measured Spark charges from the shared catalog, shared endpoint I/O census
/// (`SparkWorkingSet::rank_endpoints_io`), and recorded SM121 `exl3-hy4f-k45-mul1`
/// owner census (`spark_slots::catalog_owner_bytes`). Zero hand formulas.
pub fn build_spark_charges(
    catalog: &crate::OfficialV41Catalog,
    routed: &[usize],
    ranks: usize,
    prefill: PrefillShape,
    capacity_rows: u64,
) -> anyhow::Result<Vec<Vec<SparkCharge>>> {
    if ranks == 0 {
        return Ok(vec![Vec::new(); routed.len() + 1]);
    }
    let backend = crate::plan::spark_executions::SparkBackend::for_catalog(catalog)?;
    let executions = crate::plan::spark_executions::admit_family(
        "hy4_flash",
        &backend.exports(),
        None,
        prefill.rows,
        prefill.waves,
    )?;
    let endpoint = cuteafd_core::SparkEndpoint {
        ring_depth: None,
        capacity: u32::try_from(capacity_rows)?,
        ingress: cuteafd_core::SparkIngress::Fp8K32,
        partial_reply_bytes: None,
    };
    let rings = crate::plan::layout::spark_ring_geometry(LANES as usize);
    let rings = cuteafd_core::SparkRings {
        max_frame_bytes: rings.max_frame_bytes,
        depth: rings.depth,
        slot_bytes: rings.slot_bytes,
        alignment: rings.alignment,
    };
    let runtime = cuteafd_core::memory_layout::spark_runtime_reserve_bytes(0)
        .context("Spark runtime residual")?;
    let geometry = cuteafd_core::ExpertGeometry::HY4_FLASH;

    (0..=routed.len())
        .map(|count| {
            let working = SparkWorkingSet {
                rtx_layers: routed[..count].to_vec(),
                spark_layers: routed[count..].to_vec(),
                spark_ranks: ranks,
                prefill_lanes: LANES,
            };
            (0..ranks)
                .map(|rank| {
                    if count == routed.len() {
                        return Ok(SparkCharge { experts: 0, owners: 0, runtime: 0 });
                    }
                    let io = working.rank_endpoints_io(rank, geometry, &[endpoint], rings)?;
                    let owner_ws = executions
                        .workspace_bytes(|extent| {
                            crate::plan::layout::spark_slots::catalog_owner_bytes(
                                catalog, "hy4_flash", &working, rank, extent,
                            )
                        })?
                        .and_then(|bytes| bytes.total())
                        .context("Hy4-flash Spark owner sum overflow")?;
                    let owners = [
                        io.intake,
                        io.host_exchange,
                        io.row_indices,
                        io.mapped_rings,
                        io.endpoint_staging,
                        owner_ws,
                    ]
                    .into_iter()
                    .try_fold(0u64, |sum, bytes| sum.checked_add(bytes))
                    .context("Hy4-flash Spark owners overflow")?;
                    let regions = working
                        .spark_layers
                        .iter()
                        .map(|&layer| catalog.exl3_spark_arena_bytes(layer, ranks).map(|b| b[rank]))
                        .collect::<anyhow::Result<Vec<_>>>()?;
                    let resident = PackedWeightRegions::new(
                        regions
                            .into_iter()
                            .map(usize::try_from)
                            .collect::<Result<Vec<_>, _>>()?,
                    )?
                    .bytes as u64;
                    Ok(SparkCharge { experts: resident, owners, runtime })
                })
                .collect()
        })
        .collect()
}

/// Gather every admitted Hy4-flash owner fact into the shared [`Hy4Inputs`] recipe.
pub fn gather(request: &FactsRequest<'_>) -> Result<Hy4Inputs, Refused> {
    let cfg = request.cfg;
    let gpus = request.coordinator.gpus();
    let ranks = request.spark_ranks;
    if !matches!(gpus, 1 | 2) {
        return Err(refused(
            format!("Hy4-flash needs one or two coordinator GPUs, not {gpus}"),
            "Use --rtx 1 or --rtx 2.",
        ));
    }
    if request.spark_bytes.len() != ranks {
        return Err(refused(
            format!(
                "Hy4-flash Spark memory vector length {} differs from {ranks} ranks",
                request.spark_bytes.len()
            ),
            "Provide one Spark memory cap per configured rank.",
        ));
    }

    // Absorbed MLA keeps `kv_b_proj` unpacked as BF16 W_UK/W_UV in the target arena across all views;
    // validate every FP8-executed core linear projection.
    request
        .checkpoint
        .validate_fp8_roles_where(
            "hy4_flash",
            |name| !name.ends_with(".self_attn.kv_b_proj.weight"),
            fp8_role,
        )
        .map_err(|error| {
            refused(
                format!("Hy4-flash fp8-compact validation refused: {error:#}"),
                "Ensure every fp8-compact core weight is a BF16 matrix with whole 128x128 blocks.",
            )
        })?;

    let shape = request.prefill;
    let modes = hy4_flash::layer_modes(cfg, gpus, request.attention);
    let targets = target_arenas(request.checkpoint, cfg, request.model_core, &modes, gpus)
        .map_err(|error| {
            refused(
                format!("Hy4-flash packed target arena refused: {error:#}"),
                "Check checkpoint core headers and selected MC view.",
            )
        })?;

    let context_tokens = if request.context_tokens == 0 {
        cfg.max_position as u64
    } else {
        request.context_tokens.min(cfg.max_position as u64)
    };
    let context = Context::resolve(
        context_tokens,
        (request.context_tokens > 0).then_some(context_tokens),
    );

    let page_rows = if modes.iter().any(|m| *m == LayerMode::ContextSplit) {
        2 * PAGE_ROWS as u64
    } else {
        PAGE_ROWS as u64
    };
    let per_dsa_layer_page_bytes = PAGE_ROWS as u64 * request.kv_format.record_bytes() as u64
        + (PAGE_ROWS / cfg.indexer_compress_ratio) as u64 * INDEX_GROUP_RECORD_BYTES as u64;
    let dsa_layers_on = |gpu: usize| -> u64 {
        modes
            .iter()
            .enumerate()
            .filter(|&(layer, mode)| {
                cfg.sparse_layer[layer]
                    && match *mode {
                        LayerMode::Whole { gpu: owner, .. } => usize::from(owner) == gpu,
                        LayerMode::HeadSplit | LayerMode::ContextSplit => true,
                    }
            })
            .count() as u64
    };
    let bytes_per_dsa_page_total: u64 =
        (0..gpus).map(|g| dsa_layers_on(g) * per_dsa_layer_page_bytes).sum();
    let mut facts = PoolFacts::new(context, request.pool_tokens.filter(|&n| n > 0), page_rows);
    facts.memory = (0..gpus)
        .map(|gpu| PoolMemory {
            available_bytes: request.coordinator.effective_cap(gpu),
            bytes_per_unit: dsa_layers_on(gpu) * per_dsa_layer_page_bytes,
        })
        .collect();
    facts.target = request.target_pool_tokens;
    let pool = crate::placement::prepare_pool(facts).map_err(|error| {
        refused(
            format!("Hy4-flash pool policy refused: {error}"),
            "Supply consistent KV target, min and max bounds.",
        )
    })?;

    let routed = cfg.routed_layers();
    let capacity_rows = shape.rows.max(u64::from(request.decode_rows));
    let link_shape = PrefillShape { rows: capacity_rows, waves: shape.waves };
    let local = build_local_backings(request.catalog, request.manifest_path, gpus, capacity_rows);
    let spark = build_spark_charges(request.catalog, &routed, ranks, shape, capacity_rows).map_err(|error| {
        refused(
            format!("Hy4-flash Spark charge census refused: {error:#}"),
            "Use a supported Spark world (TP1..=TP6) with an EXL3 MUL1 checkpoint.",
        )
    })?;

    let head_gpu = if modes
        .iter()
        .all(|m| matches!(m, LayerMode::HeadSplit | LayerMode::ContextSplit))
    {
        0u8
    } else {
        (gpus - 1) as u8
    };
    let links = hy4_flash::spark_link(head_gpu, ranks, link_shape, request.replies).map_err(|error| {
        refused(
            format!("Hy4-flash Spark transport refused: {error}"),
            "Use supported Spark rank count and prefill shape.",
        )
    })?;

    let (host_total, host_available) = request.host;
    let unit_rows = usize::try_from(capacity_rows).map_err(|_| {
        refused("Hy4-flash prefill rows overflow", "Use a supported prefill unit size.")
    })?;
    let cache0 = CacheLayout::for_unit(
        cfg,
        request.kv_format,
        request.sequences as usize,
        cfg.sparse_layers().len(),
        cfg.swa_layers().len(),
        unit_rows,
    )
    .map_err(|error| {
        refused(
            format!("Hy4-flash cache layout refused: {error:#}"),
            "Check concurrency and config.",
        )
    })?;
    let prefix_mark_slots = request.prefix_slots.unwrap_or_else(|| {
        if request.prefix_entries == 0 {
            0
        } else {
            cuteafd_core::prefix::mark_slots_with_host(
                request.sequences,
                request.prefix_entries,
                cache0.mark_bytes_per_slot as u64,
                request.prefix_mark_budget,
                request.host_cache_bytes != Some(0),
            )
        }
    }) as usize;

    let host_chunk_bytes = {
        let geom = cuteafd_core::prefix::HostCacheGeometry {
            unit_rows: page_rows,
            page_bytes: bytes_per_dsa_page_total,
            mark_bytes: cache0.mark_bytes_per_slot as u64,
            draft_bytes: 0,
        };
        geom.chunk_bytes()
    };
    let host_quota = if request.host_cache_bytes == Some(0) || request.prefix_entries == 0 {
        0u64
    } else {
        let geom = cuteafd_core::prefix::HostCacheGeometry {
            unit_rows: page_rows,
            page_bytes: bytes_per_dsa_page_total,
            mark_bytes: cache0.mark_bytes_per_slot as u64,
            draft_bytes: 0,
        };
        geom.resolve(
            request.host_cache_bytes,
            context.maximum,
            request.sequences,
            host_total,
            host_chunk_bytes,
        )
        .ok_or_else(|| {
            refused(
                "Hy4-flash host prefix quota overflow",
                "Disable host retention or supply a supported quota.",
            )
        })?
    };
    let host_prefix = crate::placement::planner::serving::HostPrefixOwners {
        quota_bytes: host_quota,
        chunk_bytes: host_chunk_bytes.min(host_quota.max(1)),
    };

    // Real measured host Engram owners (fail-closed, FP8-only, never GPU memory):
    // 1. MappedPlacement for the 4 FP8 Engram tables (`engram-fp8-layer-*-part-*.safetensors`, 61.47 GiB)
    // 2. Per-layer `RowCache` (`Hy4EngramShape::cache_bytes`)
    // 3. Pinned prefill/decode staging pools (`Hy4EngramShape::staging_bytes`)
    // 4. Shared table-read reactor workspace (`table_reactor_workspace_bytes(mapped.mapped)`)
    let (engram_table_bytes, _per_layer) =
        mapped_engram_bytes(request.checkpoint, cfg).map_err(|error| {
            refused(
                format!("Hy4-flash Engram table inventory refused: {error:#}"),
                "Provide FP8 Engram shards (engram-fp8-layer-*-part-*.safetensors) declared in cuteafd-mc.json.",
            )
        })?;
    let engram_shape = Hy4EngramShape::new(
        shape.rows as usize,
        shape.waves as usize,
        request.decode_rows as usize,
        cfg.engram_layers.len(),
    )
    .map_err(|error| {
        refused(
            format!("Hy4-flash Engram staging shape refused: {error:#}"),
            "Use supported prefill unit rows and wave count.",
        )
    })?;
    let engram_row_cache_bytes = engram_shape.cache_bytes() as u64;
    let engram_staging_bytes = engram_shape.staging_bytes() as u64;
    let host_headroom = (8u64 << 30).max(host_total / 10);
    let engram_preload_budget = host_available
        .saturating_sub(host_headroom)
        .saturating_sub(host_quota)
        .saturating_sub(engram_row_cache_bytes)
        .saturating_sub(engram_staging_bytes);
    let engram_mapped = MappedPlacement::resolve(
        cfg.engram_layers.len(),
        0,
        0,
        engram_table_bytes,
        engram_preload_budget,
    )
    .admit_tables(engram_table_bytes);

    let mut budgets: Vec<Budget> = Vec::with_capacity(gpus + ranks + 1);
    let mut code = Vec::with_capacity(gpus);
    let mut graph_reserve = Vec::with_capacity(gpus);
    let graph_layout = crate::serving_capacity::hy4_graphs::Hy4GraphLayout::resolve(gpus, ranks);
    match &request.coordinator {
        Coordinator::Planned { rtx_bytes } => {
            for (gpu, &cap) in rtx_bytes.iter().enumerate() {
                let arch = ArchContext::for_device("sm_120", cap);
                budgets.push(hy4_flash::planned_budget(gpu as u8, cap, cap, arch.context_bytes));
                code.push(
                    crate::placement::inventory::loaded_code_allowance(
                        "hy4_flash",
                        "exl3",
                        gpus == 2,
                        gpu as u8,
                        arch,
                        None,
                    )
                    .bytes,
                );
                let gr = crate::serving_capacity::hy4_graphs::reserve_policy_for(
                    graph_layout,
                    gpus,
                    gpu,
                    request.dfly2 && request.tp2_dflash,
                    request.decode_graphs,
                )
                    .map_err(|error| {
                        refused(
                            format!("Hy4-flash graph reserve refused: {error}"),
                            "Check GPU_GRAPH_RESERVE.",
                        )
                    })?
                    .bytes;
                graph_reserve.push(gr);
            }
        }
        Coordinator::Measured { samples, loaded_code } => {
            if loaded_code.len() != gpus {
                return Err(refused(
                    "Hy4-flash measured loaded_code count differs from GPUs",
                    "Sample one pending code allowance per coordinator GPU.",
                ));
            }
            for (gpu, &(total, logical, free)) in samples.iter().enumerate() {
                budgets.push(hy4_flash::measured_budget(gpu as u8, total, logical, free));
                code.push(loaded_code[gpu]);
                let gr = crate::serving_capacity::hy4_graphs::reserve_policy_for(
                    graph_layout,
                    gpus,
                    gpu,
                    request.dfly2 && request.tp2_dflash,
                    request.decode_graphs,
                )
                    .map_err(|error| {
                        refused(
                            format!("Hy4-flash graph reserve refused: {error}"),
                            "Check GPU_GRAPH_RESERVE.",
                        )
                    })?
                    .bytes;
                graph_reserve.push(gr);
            }
        }
    }

    for (rank, &cap) in request.spark_bytes.iter().enumerate() {
        budgets.push(Budget {
            id: format!("spark{rank}"),
            locations: vec![Location { device: Device::Spark(rank as u8), domain: Domain::Managed }],
            physical_bytes: cap,
            logical_bytes: cap,
            baseline: Baseline::Forecast {
                occupied: 0,
                source: "Spark worker arena owned by the serving process".into(),
                assumed: false,
            },
            headroom: crate::placement::reserve::spark(),
        });
    }
    budgets.push(Budget {
        id: "host".into(),
        locations: vec![Location::HOST, Location::PINNED],
        physical_bytes: host_total,
        logical_bytes: host_total,
        baseline: Baseline::Forecast {
            occupied: host_total.saturating_sub(host_available),
            source: "host memory occupied outside cuteafd".into(),
            assumed: false,
        },
        headroom: host_headroom,
    });

    let target_mount_load = super::weight_layout::target_mount_load(request.checkpoint, gpus)
        .map_err(|error| refused(
            format!("Hy4-flash dense load staging refused: {error:#}"),
            "Check dense tensor header extents.",
        ))?;

    let context_split = modes.iter().any(|m| *m == LayerMode::ContextSplit);
    if request.tp2_dflash && (!request.dfly2 || gpus != 2) {
        return Err(refused(
            "Hy4-flash TP2_DFLASH requires two coordinator GPUs and DFly2",
            "Use two RTX GPUs with SPECULATOR=dfly2, or set TP2_DFLASH=off.",
        ));
    }
    let draft = request.dfly2.then(|| {
        use super::{draft, Hy4DflyConfig};
        let checkpoint = request.checkpoint;
        let config = Hy4DflyConfig::from_hf(&checkpoint.config, cfg)?.context("no bundled DFly2 config")?;
        let source = checkpoint.view.as_ref().map(|view| {
            crate::plan::Checkpoint::open_view_for_role(view, "Hy4 DFly2", |name| name.starts_with(draft::PREFIX))
        }).transpose()?;
        draft::Admission::new_tp_with_context(
            source.as_ref().unwrap_or(checkpoint),
            cfg,
            &config,
            request.model_core,
            request.kv_format,
            request.sequences as usize,
            request.decode_rows as usize,
            if request.tp2_dflash { 2 } else { 1 },
            context_split,
        )
    }).transpose().map_err(|error| refused(
        format!("Hy4 DFly2 admission refused: {error:#}"),
        "Use a supported bundled Hy4 drafter or SPECULATOR=off.",
    ))?;

    // The bundled ViT shares the joint solve: one packed arena + scratch on its
    // chosen owner (an RTX GPU, or Spark replicas when experts leave room).
    let vision = crate::placement::still_encoder::StillEncoder::planned(request.checkpoint,
        request.vision, request.vision_replicas, &budgets,
        request.media_cache_bytes.unwrap_or((8 << 30).min(host_total / 20)))
        .map_err(|error| refused(format!("Hy4-flash bundled vision inventory refused: {error:#}"),
            "Supply the bundled vit.* tower tensors or use VISION=off."))?;

    Ok(Hy4Inputs {
        cfg: cfg.clone(),
        vision,
        kv_format: request.kv_format,
        gpus,
        spark_ranks: ranks,
        context,
        pool,
        prefill: shape,
        decode_rows: request.decode_rows,
        sequences: request.sequences,
        prefix_mark_slots,
        targets,
        target_mount_load,
        draft,
        placement: request.attention,
        modes,
        expert_usage: request.expert_usage.then(|| cuteafd_core::expert_usage::Geometry {
            layers: cfg.layers, experts: cfg.experts, topk: cfg.topk, routed_layers: routed.clone(),
        }),
        routed,
        local,
        spark,
        links,
        budgets,
        code,
        graph_reserve,
        host_prefix,
        engram_mapped,
        engram_row_cache_bytes,
        engram_staging_bytes,
        resident_weight_limit: request.resident_weight_limit,
    })
}
