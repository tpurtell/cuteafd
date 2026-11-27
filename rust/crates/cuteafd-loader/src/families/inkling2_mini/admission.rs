//! Unified fact gatherer for Inkling 2 Mini TP1 admission (`cuteafd plan --layout` and serve).
use crate::families::inkling2::draft::{prefix_mark_slots, DraftLayout, Selection};
use crate::families::inkling2::execution::SequenceLayout;
use crate::families::inkling2_mini::{
    weight_layout::{layer_experts, MiniTargetWeights},
    Inkling2MiniConfig, InklingKvCache,
};
use crate::placement::families::inkling2_mini::{self as shared, LocalBacking, LocalPackage, MiniInputs};
use crate::placement::planner::{serving, Budget, Context, PrefillShape};
use crate::placement::{ArchContext, ResidentWeights};
use anyhow::Context as _;
use std::{path::Path, sync::Arc};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{what}")]
pub struct Refused {
    pub what: String,
    pub how: String,
}

fn refused(what: impl Into<String>, how: impl Into<String>) -> Refused {
    Refused {
        what: what.into(),
        how: how.into(),
    }
}

#[derive(Debug, Clone)]
pub enum Coordinator {
    Planned { card: crate::placement::CoordinatorCard },
    Measured { sample: (u64, u64, u64), loaded_code: u64 },
}

impl Coordinator {
    fn physical(&self) -> u64 {
        match self {
            Self::Planned { card } => card.physical_bytes,
            Self::Measured { sample, .. } => sample.0,
        }
    }

    fn budget(&self) -> Budget {
        match self {
            Self::Planned { card } => shared::planned_budget(
                card.physical_bytes,
                card.logical_bytes,
                ArchContext::for_device("sm_120", card.physical_bytes).context_bytes,
            ),
            Self::Measured { sample, .. } => shared::measured_budget(sample.0, sample.1, sample.2),
        }
    }
}

pub struct FactsRequest<'a> {
    pub vision: crate::plan::MediaMode,
    pub vision_replicas: usize,
    pub media_cache_bytes: Option<u64>,
    pub checkpoint: &'a crate::plan::Checkpoint,
    pub catalog: &'a crate::OfficialV41Catalog,
    pub cfg: &'a Inkling2MiniConfig,
    pub kv_format: InklingKvCache,
    pub drafter: Option<Selection>,
    pub dflash: Option<(&'a Path, Selection)>,
    pub manifest_path: Option<&'a Path>,
    pub manifest: Option<Arc<serde_json::Value>>,
    pub coordinator: Coordinator,
    pub prefill: PrefillShape,
    /// `--decode-graphs` (default on): off charges no graph reserve.
    pub decode_graphs: bool,
    /// Admitted per-lane decode/verify row budget (`DECODE_ROWS`, 1..=128).
    pub decode_rows: u32,
    pub context_tokens: u64,
    pub pool_tokens: Option<u64>,
    pub target_pool_tokens: Option<u64>,
    pub sequences: u64,
    pub host: (u64, u64),
    pub host_cache_bytes: Option<u64>,
    pub prefix_entries: u64,
    pub prefix_mark_budget: u64,
    pub head: serving::BlasHead,
    pub expert_usage: bool,
    pub resident_weight_limit: ResidentWeights,
}

/// Builds homogeneous TP1 local expert backings (NVFP4 layers 0..20, MXFP8 layers 20..24).
/// Fails closed when packages are missing in measured/installed mode; allows a bounded
/// formula estimate only for header-only planning when `allow_header_only_fallback` is true.
pub fn local_package(
    catalog: &crate::OfficialV41Catalog,
    manifest: Option<&Path>,
    rows: u64,
    allow_header_only_fallback: bool,
) -> LocalPackage {
    let build = || -> anyhow::Result<LocalPackage> {
        let routed = catalog.routed_experts();
        let layers: Vec<usize> = (routed.first_layer..routed.layers).collect();
        let tensors = catalog
            .fp8()
            .context("Inkling 2 Mini requires native NVFP4/MXFP8 expert catalog")?;
        let mut backings = Vec::new();
        for (view, group_layers) in tensors.homogeneous_layers(&layers)? {
            let (scratch, output) = match crate::placement::rtx_onboard::fp8_workspace_requests(
                &view,
                manifest,
                rows,
                1,
                "inkling2_mini",
            ) {
                Ok(requests) => {
                    let raw = *requests.first().context("native package scratch absent")?;
                    let (scratch, output, _) = crate::serving_capacity::fp8moe_tp2_workspace(
                        usize::try_from(raw)?,
                        routed.hidden,
                        usize::try_from(rows)?,
                    )?;
                    (scratch, output)
                }
                Err(_) if allow_header_only_fallback && manifest.is_none() => {
                    // Header-only planning without installed packages beside PROGRAMS.json.
                    let (scratch, output, _) = crate::serving_capacity::fp8moe_tp2_workspace(
                        128 << 20,
                        routed.hidden,
                        usize::try_from(rows)?,
                    )?;
                    (scratch, output)
                }
                Err(error) => return Err(error),
            };
            let regions = crate::serving_capacity::fp8_expert_regions(
                &view.for_coordinator(),
                &group_layers,
                1,
                0,
                crate::formats::fp8_experts::Slicing::Blocks(128),
            )?;
            let id = crate::serving_capacity::expert_weight_arena_id("fp8", 0, &regions)?;
            backings.push(LocalBacking {
                gpu: 0,
                layers: group_layers,
                arena: crate::serving_capacity::WeightArenaLayout::new(id, regions)?,
                workspace: vec![("scratch".into(), scratch as u64), ("output".into(), output as u64)],
            });
        }
        Ok(LocalPackage::Installed { backings })
    };
    build().unwrap_or_else(|error| LocalPackage::Missing(format!("{error:#}")))
}

pub struct SelectedWeights {
    pub target: MiniTargetWeights,
    pub drafter: Option<(
        crate::formats::block_dspark::recurrent::RecurrentBlockConfig,
        Vec<crate::formats::block_dspark::recurrent::loading::RecurrentWeightLayout>,
        crate::shared::dense_read::MountLoad,
    )>,
}

impl SelectedWeights {
    pub fn new(checkpoint: &crate::plan::Checkpoint, cfg: &Inkling2MiniConfig, draft: bool) -> anyhow::Result<Self> {
        let target = MiniTargetWeights::plan(checkpoint, cfg)?;
        let drafter = draft
            .then(|| {
                let view = checkpoint
                    .view
                    .as_ref()
                    .context("bundled drafter needs selected checkpoint view")?;
                DraftLayout::select_weights(view, 1, 0)
            })
            .transpose()?;
        Ok(Self { target, drafter })
    }
}

pub fn gather(request: &FactsRequest<'_>) -> Result<MiniInputs, Refused> {
    let weights = SelectedWeights::new(request.checkpoint, request.cfg, request.drafter.is_some()).map_err(|error| {
        refused(
            format!("Inkling 2 Mini target or drafter arena inventory refused: {error:#}"),
            "Provide complete headers for all 24 layers, cross-pass queries, and 144 controllers.",
        )
    })?;
    gather_with_weights(request, weights)
}

pub fn gather_with_weights(request: &FactsRequest<'_>, weights: SelectedWeights) -> Result<MiniInputs, Refused> {
    let cfg = request.cfg;
    let t = &cfg.text;
    let context = if request.context_tokens == 0 {
        t.max_position as u64
    } else {
        request.context_tokens
    };
    if context > t.max_position as u64 {
        return Err(refused(
            format!("Inkling 2 Mini context {context} exceeds checkpoint maximum {}", t.max_position),
            "Lower --max-context-tokens.",
        ));
    }
    let shape = request.prefill;
    let decode_rows = request.decode_rows;
    if !(1..=shared::MAX_DECODE_ROWS).contains(&decode_rows) {
        return Err(refused(
            format!(
                "Inkling 2 Mini decode_rows {decode_rows} exceeds supported 1..={}",
                shared::MAX_DECODE_ROWS
            ),
            "Set DECODE_ROWS to auto or 1..=128.",
        ));
    }
    let routed: Vec<usize> = (0..t.layers).collect();
    let SelectedWeights { target, drafter } = weights;
    let mut mount = target
        .mount_load(request.checkpoint)
        .map_err(|error| refused(format!("Inkling 2 Mini mount inventory refused: {error:#}"), "Check checkpoint headers."))?;
    let experts = layer_experts(request.checkpoint, cfg).map_err(|error| {
        refused(
            format!("Inkling 2 Mini routed expert inventory refused: {error:#}"),
            "Provide complete NVFP4 (layers 0..20) and MXFP8 (layers 20..24) headers.",
        )
    })?;
    let allow_fallback = matches!(request.coordinator, Coordinator::Planned { .. });
    let local = local_package(
        request.catalog,
        request.manifest_path,
        shape.rows.max(u64::from(decode_rows)),
        allow_fallback,
    );
    let program_set = match (&request.coordinator, request.manifest.as_deref()) {
        (Coordinator::Planned { .. }, Some(manifest)) => Some(crate::placement::ProgramSet::from_manifest(
            manifest,
            &["inkling2_mini", "inkling_draft", "draft_bf16"],
        )),
        _ => None,
    };
    let code = match &request.coordinator {
        Coordinator::Planned { .. } => crate::placement::inventory::loaded_code_allowance(
            "inkling2_mini",
            "experts",
            false,
            0,
            ArchContext::for_device("sm_120", request.coordinator.physical()),
            program_set.as_ref(),
        )
        .bytes,
        Coordinator::Measured { loaded_code, .. } => *loaded_code,
    };
    let graph_keys = crate::serving_capacity::inkling2_mini_graphs::KeySpace::for_config(
        request.sequences,
        decode_rows as usize,
        request.drafter.is_some() || request.dflash.is_some(),
    );
    let graph_reserve = crate::serving_capacity::inkling2_mini_graphs::policy(0, graph_keys, request.decode_graphs)
        .map_err(|error| refused(format!("Inkling 2 Mini graph reserve refused: {error}"), "Fix GPU_GRAPH_RESERVE."))?
        .bytes;

    let (host_total, host_available) = request.host;
    let budgets = vec![
        request.coordinator.budget(),
        crate::placement::families::kimi::host_budget(host_total, host_available),
    ];

    let global_slots = cfg.global_cache_slots().len() as u64;
    let record_bytes = request.kv_format.record_bytes(t.kv_heads, t.head_dim) as u64;
    let page_bytes = shared::PAGE_ROWS * record_bytes * global_slots;
    let mark_bytes = (0..cfg.cache_slots())
        .map(|slot| SequenceLayout::new(t, cfg.cache_slot_is_local(slot), 1).map(|l| l.bytes as u64))
        .sum::<anyhow::Result<u64>>()
        .map_err(|error| refused(format!("Inkling 2 Mini 48-slot state refused: {error}"), "Check window geometry."))?;
    let dflash = request.dflash.map(|(snapshot, selection)| {
        super::dflash::Layout::new(snapshot, cfg, usize::try_from(request.sequences)?,
            usize::try_from(shape.rows)?, usize::try_from(shape.waves)?, decode_rows as usize, selection.sms,
            selection.kv_cache, request.manifest.as_deref())
    }).transpose().map_err(|error| refused(format!("Mini DFlash2 inventory refused: {error:#}"),
        "Provide complete DFlash2 checkpoint headers and native BF16 projection exports."))?;
    let draft_mark = request
        .drafter
        .zip(drafter.as_ref())
        .map(|(selection, (config, _, _))| config.window_mark_bytes(selection.kv_cache, 0, 1, 0))
        .transpose()
        .map_err(|error| refused(format!("Mini drafter mark inventory refused: {error:#}"), "Check bundled MTP config."))?
        .unwrap_or(0) as u64 + dflash.as_ref().map_or(0, |d| d.mark_bytes as u64);

    let host_prefix = if request.host_cache_bytes == Some(0) || request.prefix_entries == 0 {
        serving::HostPrefixOwners {
            quota_bytes: 0,
            chunk_bytes: 0,
        }
    } else {
        let geometry = cuteafd_core::prefix::HostCacheGeometry {
            unit_rows: shared::PAGE_ROWS,
            page_bytes,
            mark_bytes: mark_bytes + draft_mark,
            draft_bytes: request.drafter.map_or(0, |selection| {
                64 * 2 * selection.kv_cache.plane_record_bytes(t.kv_heads, t.head_dim) as u64
            }),
        };
        let quota = geometry
            .resolve(
                request.host_cache_bytes,
                context,
                request.sequences,
                host_total,
                geometry.chunk_bytes(),
            )
            .unwrap_or(0);
        serving::HostPrefixOwners {
            quota_bytes: quota,
            chunk_bytes: geometry.chunk_bytes().min(quota.max(1)),
        }
    };

    let push_required = match crate::push_load::namespace(&request.checkpoint.snapshot) {
        Ok(_) => true,
        Err(error)
            if error.kind() == std::io::ErrorKind::NotFound
                && error.raw_os_error().is_none()
                && request.checkpoint.snapshot.exists() =>
        {
            false
        }
        Err(error) => {
            return Err(refused(
                format!("Inkling 2 Mini checkpoint namespace refused: {error}"),
                "Restore SparkNest namespace metadata; required push cannot fall back to mount.",
            ));
        }
    };
    if !push_required {
        if let Some((_, _, draft_mount)) = &drafter {
            mount.merge(*draft_mount);
        }
    }

    let capacity = request.coordinator.budget().logical_bytes;
    let ctx = Context::resolve(context, (request.context_tokens > 0).then_some(request.context_tokens));
    let mut pool_facts = crate::placement::PoolFacts::new(ctx, request.pool_tokens, shared::PAGE_ROWS);
    pool_facts.memory = vec![crate::placement::PoolMemory {
        available_bytes: capacity,
        bytes_per_unit: page_bytes,
    }];
    pool_facts.target = request.target_pool_tokens;
    let pool = crate::placement::prepare_pool(pool_facts)
        .map_err(|error| refused(format!("invalid KV pool policy: {error}"), "Set KV min <= target <= max."))?;

    let vision = crate::placement::still_encoder::StillEncoder::planned(
        request.checkpoint,
        request.vision,
        request.vision_replicas,
        &budgets,
        request.media_cache_bytes.unwrap_or(256 << 20),
    )
    .map_err(|error| refused(format!("Inkling 2 Mini vision inventory refused: {error:#}"), "Provide bundled PE tensors."))?;

    let prefix_mark_slots = prefix_mark_slots(
        request.sequences,
        request.prefix_entries,
        mark_bytes,
        draft_mark,
        request.prefix_mark_budget,
        host_prefix.quota_bytes > 0,
    )
    .map_err(|error| refused(format!("Inkling 2 Mini prefix slots refused: {error:#}"), "Check combined mark extent."))?;

    let drafter = request
        .drafter
        .zip(drafter)
        .map(|(selection, (config, weights, _))| {
            // Scored dSpark plans every eligible active sequence before head-score
            // settling trims to the verify row budget; admit up to the admitted
            // DECODE_ROWS so C48 runs in one compacted drafter batch instead of splitting.
            let draft_sequences = usize::try_from(request.sequences)?.min(decode_rows as usize);
            DraftLayout::from_weights_for_decode_rows(
                config,
                weights,
                usize::try_from(request.sequences)?,
                draft_sequences,
                usize::try_from(context)?,
                usize::try_from(pool.maximum_tokens())?,
                usize::try_from(shape.rows)?,
                usize::try_from(shape.waves)?,
                decode_rows as usize,
                selection.sms,
                selection.kv_cache,
                request.manifest.as_deref(),
            )
            .map(|layout| shared::Drafter { layout })
        })
        .transpose()
        .map_err(|error| refused(format!("Mini bundled drafter inventory refused: {error:#}"), "Check MTP headers and exports."))?;

    Ok(MiniInputs {
        cfg: cfg.clone(),
        kv_format: request.kv_format,
        context: ctx,
        pool,
        prefill: shape,
        decode_rows,
        sequences: request.sequences,
        prefix_entries: request.prefix_entries,
        prefix_mark_budget: request.prefix_mark_budget,
        prefix_mark_slots,
        target,
        routed: routed.clone(),
        experts,
        local,
        drafter,
        dflash,
        vision,
        budgets,
        code,
        graph_reserve,
        head: request.head,
        host_prefix,
        device_sampling: true,
        expert_usage: request.expert_usage.then(|| cuteafd_core::expert_usage::Geometry {
            layers: t.layers,
            experts: t.experts,
            topk: t.topk,
            routed_layers: routed,
        }),
        resident_weight_limit: Some(request.resident_weight_limit),
        mount,
        push_required,
    })
}
