//! Kimi physical-owner admission with checkpoint residency and shared native allocation layouts.
use super::Model;
use anyhow::Context;
use crate::families::kimi_k3::{allocation, core};
use crate::placement::families::kimi::{self, ExpertProfile, FixedOwner, KimiRecipe, SparkOwner, tp2_arenas, whole_arenas};
use crate::{placement::*, plan::{Component, ExpertPlacement, PlanReport}};
use cuteafd_core::memory_layout::{Basis, Category, DeviceKind, DeviceLayout, Item, MemoryLayout};

const GIB: u64 = 1 << 30;
const COMPILED_CAPACITY: u64 = crate::families::kimi_k3::COMPILED_CONTEXT as u64;
const PAGE: u64 = 64;
const WHOLE: LayerMode = LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner };
const MODES: &[LayerMode] = &[LayerMode::HeadSplit, LayerMode::ContextSplit,
    LayerMode::Whole { gpu: 0, ffn: FfnMode::Split }, LayerMode::Whole { gpu: 1, ffn: FfnMode::Split }];
/// Prefill lanes the Spark transport serves (one QP endpoint each).
const LANES: u64 = 1;

fn program_families(dsa: bool, projected: bool) -> &'static [&'static str] {
    match (dsa, projected) {
        (false, false) => &["kimik3"], (false, true) => &["kimik3", "kimi"],
        (true, false) => &["kimik3", "kimik31"], (true, true) => &["kimik3", "kimik31", "kimi"],
    }
}

pub(super) fn layout(model: &Model, input: &crate::plan::PlanInput, report: &mut PlanReport, opts: &mut crate::plan::layout::LayoutOptions,
    requested: ExpertPlacement) -> MemoryLayout {
    if opts.host_embedding {
        report.refuse_placement("Kimi does not support --embedding-placement host".into(),
            "Use --embedding-placement gpu (EMBEDDING=gpu); the serving embedding is GPU-resident.");
        return MemoryLayout { devices: Vec::new(), pool_tokens: 0, waste: Vec::new(), notes: Vec::new() };
    }
    let limits = crate::plan::Family::prefill_limits(&super::KIMI_K3);
    let prefill = match crate::plan::layout::resolve_prefill(input, report, opts, requested.spark_ranks(), limits) {
        Ok(prefill) => prefill,
        Err(error) => return crate::plan::layout::unsupported_prefill(report, error),
    };
    let shape = prefill.units();
    let admitted = shape;
    report.prefill_reserve_rows = Some(prefill.span_rows);
    let rows = prefill.span_rows as usize;
    let unit = prefill.unit_rows as usize;
    let cfg = model.cfg();
    report.tp2_dflash = Some(opts.tp2_dflash);
    let ranks = requested.spark_ranks();
    let gpus = opts.rtx_bytes.len();
    let mut memory = MemoryLayout { devices: Vec::new(), pool_tokens: 0, waste: Vec::new(), notes: vec![
        "Kimi K3 header lower bound: coordinator workspaces/graphs not charged; NOT serving admission".into(),
        "Spark budgets are offline estimates (including an explicit override), not load/steady census-qualified admission".into(),
    ] };
    report.bytes_by_owner.clear();
    // One owner runs whole MLA layers, so its convention is Heads; a pair
    // resolves auto to Context.
    let attention = if gpus == 1 { report.requested_attention.unwrap_or(AttentionPlacement::Heads) }
        else { crate::families::kimi_k3::resolve_attention(report.requested_attention, gpus) };
    report.attention_placement = attention;
    let context = if opts.context_tokens == 0 { cfg.max_position as u64 } else { opts.context_tokens };
    memory.notes.push(format!("Kimi API max context: {context} tokens; compiled max context: {COMPILED_CAPACITY} tokens"));
    if context > COMPILED_CAPACITY {
        report.refuse_placement(format!("Kimi context {context} exceeds compiled capacity {COMPILED_CAPACITY}"),
            "Lower --max-context-tokens to at most 1M; the pool must cover the selected context extent.");
        return memory;
    }
    let context_split = attention == AttentionPlacement::Context;
    if !model.complete {
        report.refuse_placement("Kimi K3 placement cannot infer fit from missing safetensors headers".into(),
            "Place every shard's header locally (nest replicate) or plan from a complete snapshot.");
        return memory;
    }
    let experts = *model.bytes.get(&Component::RoutedExpert).unwrap_or(&0);
    let routed_layers = cfg.dense.iter().filter(|d| !**d).count() as u64;
    let layer_bytes = experts / routed_layers;
    if !matches!(gpus, 1 | 2) || (gpus == 2 && !opts.head_split) {
        report.refuse_placement("Kimi K3 requires one owner or two GPUs with split KDA and FFNs".into(),
            "Use --rtx 1 or --rtx 2 with head split enabled.");
        return memory;
    }
    let head_rank = gpus - 1;
    let modes = if gpus == 1 { &[WHOLE][..] } else { MODES };
    let requested_pool = opts.pool_tokens.filter(|n| *n > 0);
    let edge = |c| *model.bytes.get(&c).unwrap_or(&0);
    let embedding = edge(Component::Embedding);
    let head = if let Some(head) = model.core.fp8_head().expect("validated FP8 head geometry") {
        head.bytes().expect("validated FP8 head extent")
    } else { model.core.projections.get(core::HEAD_STEM)
        .map_or(edge(Component::LmHead), |p| p.resident_bytes(1).expect("validated core dimensions")) };
    let residency = model.coordinator;
    if !model.core.projections.is_empty() {
        memory.notes.push(format!("Kimi K6 core: {} header-native projections; payload + H128 scale vectors charged without BF16 expansion",
            model.core.projections.len()));
    }
    let mla_weights: u64 = model.mla.iter().map(|r| r.split + r.replicated).sum();
    report.bytes_by_owner.insert("coordinator text weights".into(), embedding + head + residency.split + residency.replicated + mla_weights);
    report.bytes_by_owner.insert("routed expert weights".into(), experts);
    let slots = opts.resolved_state_slots("kimi_k3");
    let path = opts.workspace_manifest.as_deref().unwrap_or_else(||
        std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json"));
    // Header-only diagnostics remain available without installed artifacts. An
    // explicit package or a serving image must supply a complete allocation ledger.
    let manifest = if path.is_file() {
        match input.manifest(path) {
            Ok(value) => Some(value),
            Err(error) => { report.refuse_placement(format!("Kimi package inventory refused: {error:#}"),
                "Supply the matching PROGRAMS.json through --workspace-manifest."); return memory; }
        }
    } else if opts.workspace_manifest.is_some() {
        report.refuse_placement("Kimi explicit workspace manifest is missing".into(), "Supply the installed PROGRAMS.json.");
        return memory;
    } else { None };
    let context_layout = if context_split {
        let layout = (|| -> anyhow::Result<_> {
            let manifest = manifest.as_ref().context("Kimi context requires PROGRAMS.json")?;
            let banks = if rows <= 64 { 1 } else { rows.div_ceil(unit) };
            crate::families::kimi_k3::context::ContextLayout::from_manifest_for_config(manifest, opts.mla_kv,
                context as usize, unit, banks, &cfg)
        })();
        match layout {
            Ok(layout) => Some(layout),
            Err(error) => {
                report.refuse_placement(format!("Kimi context program admission refused: {error:#}"),
                    &format!("Install PROGRAMS.json with {}.",
                        crate::families::kimi_k3::context::program_names_for_config(opts.mla_kv, &cfg).join(", ")));
                return memory;
            }
        }
    } else { None };
    let context_buffers = context_layout.as_ref().map_or_else(ContextBuffers::default, |layout| layout.buffers());
    if context_split {
        let staging = |extent: u64| 2 * extent.div_ceil(PAGE) * PAGE * opts.mla_kv.record_bytes() as u64;
        memory.notes.push(format!("Kimi context gathered-prefill staging per GPU: configured {context} tokens = {} B ({:.4} GiB); \
            compiled cap {COMPILED_CAPACITY} tokens = {} B ({:.4} GiB); two full-history planes, independent of logical pool tokens",
            staging(context), staging(context) as f64 / GIB as f64,
            staging(COMPILED_CAPACITY), staging(COMPILED_CAPACITY) as f64 / GIB as f64));
    }
    let state = cfg.kda_layers() as u64 * (cfg.kda_state_bytes() + cfg.kda_conv_bytes()) * slots / gpus as u64;
    let mark_bytes = cfg.kda_layers() as u64 * (cfg.kda_state_bytes() + cfg.kda_conv_bytes()) / gpus as u64;
    let legacy_marks = opts.prefix_slots.unwrap_or_else(|| cuteafd_core::prefix::mark_slots_with_host(
        opts.resolved_concurrency("kimi_k3"), opts.prefix_entries("kimi_k3"), gpus as u64 * mark_bytes, opts.mimo_prefix_mark_bytes,
        opts.host_cache_bytes != Some(0)));
    let mut mark_demands = (0..gpus).map(|gpu| Demand::new(gpu as u8, Category::Prefix,
        "Kimi exact KDA marks", mark_bytes, Basis::Formula)).collect::<Vec<_>>();
    let mut fixed = Vec::new();
    for demand in &mark_demands {
        let mut demand = demand.clone();
        let Some(bytes) = demand.bytes.checked_mul(legacy_marks) else {
            report.refuse_placement("Kimi mark reservation overflow".into(), "Reduce prefix retention.");
            return memory;
        };
        demand.bytes = bytes;
        fixed.push(demand);
    }
    // Weight upload extents come from checkpoint geometry, not executable
    // workspace metadata. Header-only forecasts retain the same physical owners.
    let final_tap = (|| -> anyhow::Result<bool> {
        let Some(snapshot) = &opts.kimi_drafter_snapshot else { return Ok(false) };
        let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
        let stream = crate::families::kimi_k3::draft::DraftHiddenStream::from_config(&raw)?;
        Ok(stream == crate::families::kimi_k3::draft::DraftHiddenStream::AttnRes
            && raw["dflash_config"]["target_layer_ids"].as_array().is_some_and(|taps|
                taps.iter().any(|tap| tap.as_u64() == cfg.layers.checked_sub(1).map(|last| last as u64))))
    })();
    let globals = match final_tap.and_then(|final_tap|
        crate::families::kimi_k3::weight_layout::GlobalLayout::ranks_with_core(cfg, &model.core, gpus)?
            .into_iter().map(|layout| layout.with_final_attn_res(cfg, final_tap)).collect::<anyhow::Result<Vec<_>>>()) {
        Ok(value) => value, Err(error) => {
            report.refuse_placement(format!("Kimi packed global inventory refused: {error:#}"), "Check checkpoint global geometry."); return memory;
        }
    };
    for gpu in 0..gpus as u8 {
        fixed.push(globals[usize::from(gpu)].demand().expect("validated global extent"));
        fixed.push(Demand::new(gpu, Category::Kv, format!("KDA FP32 state + BF16 conv, {slots} slots, {} heads", cfg.kda_heads / gpus), state, Basis::Formula));
    }
    let mut baseline = vec![Baseline::Planned { context_bytes: 0, loaded_bytes: 0 }; gpus];
    let mut tp2_workspace = [0; 2];
    let mut shaped = Vec::new();
    if let Some(manifest) = &manifest {
        let extra = (|| -> anyhow::Result<_> {
            // Speculative verify graphs replay at each exact width up to the drafter's block.
            let verify = opts.kimi_drafter_snapshot.as_ref().map(|snapshot| -> anyhow::Result<usize> {
                let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
                if crate::families::glm5::draft_representation::is_dspark(&raw) {
                    Ok(crate::formats::block_dspark::BlockDsparkConfig::from_json(&raw)?.block + 1)
                } else { Ok(crate::formats::dflash::MlaDflashConfig::from_json(&raw)?.block) }
            }).transpose()?.unwrap_or(0);
            let attention_heads = if matches!(attention, AttentionPlacement::Heads | AttentionPlacement::Context)
                && gpus == 2 { 48 } else { 96 };
            let mut demands = allocation::baseline_demands_for_ranks(cfg, manifest, rows, shape, context as usize,
                usize::try_from(slots)?, verify, &model.core, matches!(attention, AttentionPlacement::Heads | AttentionPlacement::Context),
                ranks, gpus, opts.resolved_spark_reply_format(false), attention_heads)?;
            let mut lane = allocation::NativeLaneLayout::for_ranks(cfg, rows, unit, context as usize,
                allocation::NativeScratch::from_manifest_for_config(manifest, cfg, gpus)?, gpus)?;
            if cfg.dsa.is_some() {
                lane = lane.with_sparse_mla_rows(attention_heads, allocation::sparse_mla_chunk(manifest)?)?.with_dsa(cfg,
                    allocation::DsaScratch::from_manifest(manifest)?)?
                    .with_live_projection_rows(attention_heads, gpus)?;
            }
            demands.retain(|d| d.group != "Kimi lane 0 native operands");
            if let Some(context) = &context_layout {
                // Validate the context key inventory; the graph reserve is the
                // configured arena, charged once by the recipe.
                allocation::graphs_with_context(cfg, 1, unit, usize::try_from(slots)?, verify,
                    gpus, |_, _| true, |layer| cfg.attention[layer] == crate::families::kimi_k3::config::KimiAttention::Mla,
                    Some(context.decode()))?;
            }
            let families = program_families(cfg.dsa.is_some(), !model.core.projections.is_empty());
            let native_manifest = input.programs_manifest(path)?;
            let native = model.core.native_programs(unit, gpus, Some(&native_manifest))?;
            let mut programs = ProgramSet::from_manifest(manifest, families);
            programs.names.extend(native.programs);
            programs.names.sort();
            programs.names.dedup();
            for rank in 0..gpus {
                let arch = ArchContext::for_device("sm_120", opts.rtx_bytes[rank]);
                let code = crate::placement::inventory::loaded_code_allowance("kimik3", "exl3", gpus == 2,
                    rank as u8, arch, Some(&programs));
                baseline[rank] = Baseline::Planned { context_bytes: arch.context_bytes, loaded_bytes: 0 };
                demands.push(allocation::code_demand(rank, code.bytes, gpus)?);
            }
            if model.core.has_split_core() {
                // The baseline already includes the full peer receive allocation.
                let charges = core::CoreCharges::from_package_with_native(cfg, &model.core, rows, unit,
                    manifest, &native_manifest, gpus)?;
                let mut stages = model.core.split_stages(core::package_lane_extents_with_base(manifest, cfg, rows, unit, gpus, &lane)?,
                    unit, gpus, cfg.dense_intermediate);
                stages.extend(model.core.native_stages_from_manifest(&native_manifest, rows, unit, gpus)?);
                lane = lane.with_core_stages(stages)?;
                for rank in 0..gpus {
                    let extra_requests = charges.projection[rank].as_ref().map(|layout|
                        layout.storage().map(|storage| storage.owners.values().copied().collect::<Vec<_>>()))
                        .transpose()?.unwrap_or_default();
                    shaped.push(kimi::ShapedOwner::Lane { rank: rank as u8, lane: 0,
                        layout: lane.clone(), extra_requests });
                }
            } else {
                shaped.extend((0..gpus).map(|rank| kimi::ShapedOwner::Lane { rank: rank as u8,
                    lane: 0, layout: lane.clone(), extra_requests: Vec::new() }));
            }
            if opts.onboard != Some(OnboardSetting::Layers(0)) {
                let package = allocation::expert_package_for_ranks_codebook(path, gpus, model.expert_codebook)?;
                tp2_workspace = [allocation::expert_workspace_for_ranks_with(&package, unit, gpus, |path| input.manifest(path))?; 2];
            }
            Ok(demands)
        })();
        match extra {
            Ok(demands) => {
                fixed.extend(demands);
            },
            Err(error) => { report.refuse_placement(format!("Kimi package admission refused: {error:#}"),
                "Supply complete matching native and EXL3 workspace manifests."); return memory; }
        }
        memory.notes[0] = "Kimi shared runtime allocation ledger; the bounded graph reserve is charged once as a configured per-device arena".into();
    } else if model.core.has_split_core() {
        report.refuse_placement("Kimi compact core requires installed workspace inventory".into(),
            "Supply the matching PROGRAMS.json through --workspace-manifest.");
        return memory;
    }
    let inventory = (|| -> anyhow::Result<_> {
        let catalog = input.expert_catalog()?;
        if ranks == 0 { return Err(kimi::LOCAL_ONLY_REFUSAL.into()); }
        let working = SparkWorkingSet { rtx_layers: vec![], spark_layers: cfg.dense.iter().enumerate()
            .filter_map(|(layer, dense)| (!dense).then_some(layer)).collect(), spark_ranks: ranks, prefill_lanes: LANES };
        let rings = crate::plan::layout::spark_ring_geometry(LANES as usize);
        let endpoints = [cuteafd_core::SparkEndpoint { ring_depth: None, capacity: unit as u32,
            ingress: cuteafd_core::SparkIngress::Fp8K32, partial_reply_bytes: None }];
        let budget = opts.spark_capacities(ranks).into_iter().max().unwrap_or(0);
        let mut spark = SparkBudgetInventory::exl3(&catalog, &working, &endpoints,
            cuteafd_core::SparkRings { max_frame_bytes: rings.max_frame_bytes, depth: rings.depth,
                slot_bytes: rings.slot_bytes, alignment: rings.alignment }, budget,
            crate::families::kimi_k3::spark::reserve_bytes(ranks, admitted.waves, admitted.unit_rows)?)?;
        spark.rank_budgets = opts.spark_capacities(ranks);
        let manifest = catalog.exl3().ok_or_else(|| anyhow::anyhow!("Kimi TP2 requires EXL3 headers"))?;
        let mut costs = vec![[Bytes2::default(); 2]; cfg.layers];
        for &layer in &working.spark_layers {
            for rank in 0..gpus {
                costs[layer][rank].resident = manifest.resident_arena_bytes(crate::V41Exl3Layer::Backbone(layer), gpus, rank, false)?;
            }
        }
        Ok((spark, costs))
    })();
    let (spark_budget, costs) = match inventory {
        Ok(value) => value,
        Err(error) => {
            report.refuse_placement(format!("Kimi header inventory refused: {error:#}"), "Provide complete native EXL3 headers.");
            return memory;
        }
    };
    let draft = opts.kimi_drafter_snapshot.as_ref().map(|snapshot| -> anyhow::Result<_> {
        use crate::{families::glm5::draft_representation::{GlmDraftCapacity, GlmDraftRepresentation},
            formats::dflash::MlaDflashConfig};
        use crate::families::kimi_k3::draft::DraftLayout;
        let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
        let slots = usize::try_from(slots)?;
        let layout = if crate::families::glm5::draft_representation::is_dspark(&raw) {
            anyhow::ensure!(!opts.tp2_dflash,
                "block dSpark uses physical GQA head splits, not TP2_DFLASH");
            let representation=if opts.kimi_draft_fp8 { GlmDraftRepresentation::Fp8Only }
                else { GlmDraftRepresentation::Bf16Only };
            if gpus==2 && opts.draft_context_window==0 && crate::formats::block_dspark::BlockDsparkConfig::from_json(&raw)?.window.is_some() {
                let rank=|rank|crate::formats::block_dspark::BlockDsparkRuntimeLayout::for_rank_representation(snapshot,
                    slots,slots.min(8),opts.physical_sms.unwrap_or(188) as usize,&format!("kimi/draft/{rank}"),
                    opts.draft_kv_cache,rank,2,head_rank,representation);
                DraftLayout::BlockTp2 { ranks:Box::new([rank(0)?,rank(1)?]),lead:head_rank }
            } else {
                DraftLayout::Block(crate::formats::block_dspark::BlockDsparkRuntimeLayout::with_context(snapshot,
                    slots,slots.min(8),opts.physical_sms.unwrap_or(188) as usize,&format!("kimi/draft/{head_rank}"),opts.draft_kv_cache,representation,
                    usize::try_from(context)?,slots.checked_mul(usize::try_from(context)?).context("block draft pool overflow")?,opts.draft_context_window)?)
            }
        } else {
            anyhow::ensure!(opts.draft_context_window==0,"DRAFT_CONTEXT_WINDOW requires a Kimi block DSpark drafter");
            let draft = MlaDflashConfig::from_json(&raw)?;
            anyhow::ensure!(!opts.tp2_dflash || gpus == 2,
                "Kimi TP2 DFlash requires two physical coordinator GPUs");
            let representation = if opts.kimi_draft_fp8 { GlmDraftRepresentation::Fp8Only }
                else { GlmDraftRepresentation::Bf16Only };
            DraftLayout::Mla(draft.serving_runtime_layout(representation, GlmDraftCapacity::new(slots, slots.min(8), draft.block)?,
                usize::try_from(context)?,slots.checked_mul(usize::try_from(context)?).context("draft pool overflow")?,
                opts.physical_sms.unwrap_or(188) as usize,0,draft.cache_format(opts.draft_kv_cache))?.with_context_split(opts.tp2_dflash)?)
        };
        match &layout {
            DraftLayout::Block(local)=>{ local.validate_projection_programs(manifest.as_ref().context("block dSpark requires installed projection inventory")?)?; }
            DraftLayout::BlockTp2 { ranks,.. }=>{ for local in ranks.iter() {
                local.validate_projection_programs(manifest.as_ref().context("block TP2 requires installed projection inventory")?)?;
            } }
            DraftLayout::Mla(_)=>{}
        }
        anyhow::ensure!(layout.target_hidden() == cfg.hidden && layout.target_vocab() == cfg.vocab_size
            && layout.target_taps().iter().all(|&layer| layer < cfg.layers),
            "Kimi drafter target geometry differs from checkpoint");
        fixed.extend(crate::families::kimi_k3::draft::bridge_demands(&layout, rows, gpus)?);
        // A K6 vocabulary head has no BF16 operand to borrow: the drafter owns
        // a head-only K6 scratch on the head rank.
        if let Some(manifest) = &manifest {
            if let Some(mut demand) = crate::families::kimi_k3::core::draft_head_demand(&model.core,
                layout.capacity().block_rows, &manifest["families"]["dense_trellis_k6"])? {
                demand.gpu = head_rank as u8;
                fixed.push(demand);
            }
        }
        if model.core.native.contains_key(core::HEAD_STEM) {
            let programs = input.programs_manifest(path)?;
            if let Some(mut demand) = core::draft_fp8_head_demand(&model.core, &programs)? {
                demand.gpu = head_rank as u8;
                fixed.push(demand);
            }
        }
        for per_mark in layout.window_mark_demands(head_rank as u8)? {
            if legacy_marks>0 {
                let mut total=per_mark.clone();
                total.bytes=total.bytes.checked_mul(legacy_marks).context("Kimi draft mark overflow")?;
                fixed.push(total); mark_demands.push(per_mark);
            }
        }
        Ok(layout)
    }).transpose();
    let draft = match draft {
        Ok(layout) => layout,
        Err(error) => {
            report.refuse_placement(format!("Kimi native MLA draft inventory refused: {error:#}"),
                "Provide a supported MLA DFlash2 checkpoint and matching native representation.");
            return memory;
        }
    };
    let record = opts.mla_kv.record_bytes() as u64 * PAGE;
    let prefill = crate::placement::planner::PrefillShape { rows: shape.unit_rows, waves: shape.waves };
    let routed = cfg.dense.iter().filter(|dense| !**dense).count();
    let fabricated = (|| -> anyhow::Result<_> {
        use crate::families::kimi_k3::config::KimiAttention;
        use crate::families::kimi_k3::weight_layout::LayerLayout;
        let mut modes = Vec::with_capacity(cfg.layers);
        let mut layer_attention = Vec::with_capacity(cfg.layers);
        let mut records = vec![0u64; gpus];
        let mut records_by_layer = Vec::with_capacity(cfg.layers);
        for layer in 0..cfg.layers {
            let mla = cfg.attention[layer] == KimiAttention::Mla;
            let weights = if gpus == 1 {
                ModeBytes { whole: LayerLayout::ranks_with_core(cfg, layer, WHOLE, &model.core, 1)?[0].bytes()? as u64,
                    split: [0; 2] }
            } else {
                let split = LayerLayout::pair_with_core(cfg, layer, LayerMode::HeadSplit, &model.core)?;
                let mut halves = [0; 2];
                for rank in 0..gpus {
                    halves[rank] = split[rank].attention_bytes()? as u64;
                    let fixed_weights = split[rank].bytes()? as u64 - halves[rank];
                    fixed.push(Demand::new(rank as u8, Category::Weights,
                        format!("Kimi layer {layer} TP2 FFN/residual/router/latent"),
                        fixed_weights, Basis::Exact).with_resident_weights(ResidentWeights(fixed_weights)));
                }
                ModeBytes { split: halves, whole: if mla {
                    LayerLayout::pair_with_core(cfg, layer, LayerMode::Whole { gpu: 0, ffn: FfnMode::Split }, &model.core)?[0].attention_bytes()? as u64
                } else { 0 } }
            };
            // Resolved once here; the recipe never flips a layer to fit.
            let mode = if gpus == 1 { WHOLE } else if mla {
                match attention {
                    AttentionPlacement::Context => LayerMode::ContextSplit,
                    AttentionPlacement::Heads => LayerMode::HeadSplit,
                    AttentionPlacement::Layers => LayerMode::Whole { gpu: 0, ffn: FfnMode::Split },
                }
            } else { LayerMode::HeadSplit };
            if mla {
                let mut units = [0u64; 2];
                match mode {
                    LayerMode::Whole { gpu, .. } => {
                        units[usize::from(gpu)] = record;
                        records[usize::from(gpu)] += record;
                    }
                    _ => for rank in 0..gpus {
                        units[rank] = record;
                        records[rank] += record;
                    },
                }
                records_by_layer.push(units);
            } else {
                records_by_layer.push([0; 2]);
            }
            modes.push(mode);
            layer_attention.push(weights);
        }
        Ok((modes, layer_attention, records, records_by_layer))
    })();
    let (modes, layer_attention, mut records_per_page, records_by_layer) = match fabricated {
        Ok(value) => value,
        Err(error) => { report.refuse_placement(format!("Kimi packed layer inventory refused: {error:#}"),
            "Check checkpoint layer geometry."); return memory; }
    };
    if gpus == 2 && attention == AttentionPlacement::Layers {
        // One page of one MLA layer; the Recipe activates it on its owner.
        records_per_page = vec![record; gpus];
    }
    let onboard = opts.resolve_onboard(routed).unwrap_or(Onboard::Auto);
    let counts: Vec<usize> = match onboard {
        Onboard::Auto => (0..=routed).collect(),
        Onboard::Layers(count) if count <= routed => vec![count],
        Onboard::Layers(count) => {
            report.refuse_placement(format!("Kimi --rtx-expert-layers {count} exceeds {routed} routed layers"),
                "Use a count no greater than the routed layer count.");
            return memory;
        }
    };
    let halves = spark_budget.layers.iter().map(|&layer| costs[layer]).collect::<Vec<_>>();
    let arenas = if gpus == 2 {
        tp2_arenas(&counts, tp2_workspace, &halves)
    } else {
        whole_arenas(&counts, tp2_workspace[0], &halves.iter().map(|pair| pair[0]).collect::<Vec<_>>())
    };
    let arenas = match arenas {
        Ok(value) => value,
        Err(error) => { report.refuse_placement(format!("Kimi expert arena inventory refused: {error}"),
            "Check the routed EXL3 arena costs."); return memory; }
    };
    let mut spark = Vec::new();
    let runtime = cuteafd_core::memory_layout::spark_runtime_reserve_bytes(0)
        .expect("zero-arena reserve cannot overflow");
    for rank in 0..ranks {
        let Some(owners) = spark_budget.overhead[rank].checked_sub(runtime) else {
            report.refuse_placement("Spark overhead excludes startup reserve".into(),
                "Keep actual owners separate from the replaceable startup reserve.");
            return memory;
        };
        for &count in &counts {
            let active = count < spark_budget.layers.len();
            spark.push(SparkOwner { count, rank,
                experts: spark_budget.resident[rank][count..].iter().sum(),
                runtime: if active { runtime } else { 0 },
                overhead: if active { owners } else { 0 } });
        }
    }
    let fixed_owners = fixed.iter().map(FixedOwner::ready).collect::<Vec<_>>();
    if let Some(layout) = context_layout {
        shaped.push(kimi::ShapedOwner::Context { layout });
    }
    let peer_row_bytes = if context_split {
        context_buffers.query_row_bytes.saturating_add(context_buffers.partial_row_bytes)
    } else { 0 };
    let page_rows = if modes.iter().any(|mode| *mode == LayerMode::ContextSplit) { PAGE * 2 } else { PAGE };
    let mut facts = crate::placement::PoolFacts::new(crate::placement::planner::Context::resolve(context, (opts.context_tokens > 0).then_some(opts.context_tokens)),
        requested_pool, PAGE);
    // Whole-layer placements may give all MLA to either owner. Sum memory
    // against the checkpoint's total record slope to bound every legal cut.
    facts.memory = vec![crate::placement::PoolMemory {
        available_bytes: opts.rtx_bytes[..gpus].iter().sum(),
        bytes_per_unit: cfg.mla_layers() as u64 * PAGE * opts.mla_kv.record_bytes() as u64,
    }];
    facts.target = opts.target_pool_tokens;
    let pool = match crate::placement::prepare_pool(facts) {
        Ok(pool) => pool,
        Err(error) => { report.refuse_placement(format!("Kimi pool policy: {error}"),
            "Supply consistent KV target, min and max bounds."); return memory; }
    };
    let ceiling = pool.maximum_tokens();
    let maximum_pages = ceiling.div_ceil(page_rows);
    let draft_window_bytes = match draft.as_ref().map(|layout| layout.window_mark_bytes()).transpose() {
        Ok(bytes) => bytes.unwrap_or(0) as u64,
        Err(error) => {
            report.refuse_placement(format!("Kimi draft window inventory refused: {error:#}"),
                "Supply a supported native drafter window layout.");
            return memory;
        }
    };
    let draft_layout = draft.clone();
    let block_tp2 = matches!(draft,Some(crate::families::kimi_k3::draft::DraftLayout::BlockTp2 { .. }));
    let draft = match &draft {
        Some(layout) => match kimi::draft_layout_facts(layout, page_rows, maximum_pages, head_rank as u8) {
            Ok(value) => Some(value),
            Err(error) => { report.refuse_placement(format!("Kimi native MLA draft inventory refused: {error:#}"),
                "Provide a supported MLA DFlash2 checkpoint and matching native representation."); return memory; }
        },
        None => None,
    };
    let first_moe = spark_budget.layers.first().copied().unwrap_or(cfg.layers);
    let mut kinds = Vec::new();
    if cfg.attention.iter().any(|kind| *kind == crate::families::kimi_k3::config::KimiAttention::Mla) {
        kinds.push(AttentionClass::Mla);
    }
    if cfg.attention.iter().any(|kind| *kind != crate::families::kimi_k3::config::KimiAttention::Mla) {
        kinds.push(AttentionClass::Kda);
    }
    let rings = crate::plan::layout::spark_ring_geometry(LANES as usize);
    let links = match kimi::production_links(head_rank as u8, ranks, LANES, prefill,
        cuteafd_core::SparkRings { max_frame_bytes: rings.max_frame_bytes, depth: rings.depth,
            slot_bytes: rings.slot_bytes, alignment: rings.alignment }, opts.spark_reply_formats) {
        Ok(links) => links,
        Err(error) => { report.refuse_placement(format!("Kimi transport owners refused: {error}"),
            "Supply supported Spark endpoint geometry."); return memory; }
    };
    let mut recipe = KimiRecipe {
        pool,
        vision: None,
        expert_usage: kimi::usage_geometry(cfg, opts.expert_usage),
        context, gpus, placement: attention, attention: modes,
        prefill, lanes: LANES, spark_links: ranks > 0,
        hop_spec: HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: cfg.hidden as u64 * 2,
            rows: if manifest.is_some() { unit as u64 } else { 0 },
            lanes: if manifest.is_some() { LANES } else { 0 },
            entry_gpu: 0, head_gpu: head_rank as u8,
            terminal: if gpus == 1 { TerminalHome::AttentionOwner } else { TerminalHome::Replicated } },
        peer_row_bytes, maximum_pages,
        records_per_page, records_by_layer,
        index_bytes_per_page: cfg.dsa.map_or(0, |index| PAGE * (index.head_dim + 4) as u64),
        layer_attention, first_moe,
        kinds,
        experts: Some(ExpertProfile { layers: spark_budget.layers.clone(),
            destination: if gpus == 2 { vec![0, 1] } else { vec![0] },
            counts, rank_sets: if ranks > 0 { vec![(0..ranks as u8).collect()] } else { Vec::new() } }),
        arenas, spark, links, draft, fixed: fixed_owners, shaped, owners: Vec::new(),
    };
    if let Err(error) = recipe.arena_padding(cfg, &model.core) {
        report.refuse_placement(format!("Kimi packed arena inventory refused: {error:#}"),
            "Provide complete selected native weight extents.");
        return memory;
    }
    if let Some(layout) = &draft_layout {
        if let Err(error) = recipe.draft_homes(layout, &model.core) {
            report.refuse_placement(format!("Kimi draft homes refused: {error:#}"), "Supply native full-head geometry.");
            return memory;
        }
    }
    if let Err(error) = recipe.blas_owners(if model.core.get_weight("language_model.lm_head").is_some() {
        crate::placement::planner::serving::BlasHead::Native } else { opts.bf16_target_head }) {
        report.refuse_placement(format!("Kimi BLAS owners refused: {error}"), "Supply supported native head geometry.");
        return memory;
    }
    let occupied = vec![0u64; gpus];
    for (rank, baseline) in baseline.iter().enumerate() {
        if let crate::placement::Baseline::Planned { context_bytes, loaded_bytes } = baseline {
            recipe.fixed.push(FixedOwner { rank: rank as u8, category: Category::Runtime,
                group: allocation::CONTEXT_ESTIMATE.into(), bytes: context_bytes.saturating_add(*loaded_bytes),
                weights: 0, lifetime: crate::placement::planner::Lifetime::RESIDENT, basis: Basis::Estimated });
        }
    }
    let sparks = opts.spark_capacities(ranks);
    let (host_total, host_available) = match crate::plan::layout::planner_host_memory(
        opts.host_memory, crate::plan::layout::HostMemorySample::Raw) {
        Ok(host) => host,
        Err(error) => { report.refuse_placement(format!("Kimi host inventory refused: {error:#}"), "Supply host memory capacity."); return memory; }
    };
    let host_quota = (|| -> anyhow::Result<u64> {
        if opts.host_cache_bytes == Some(0) || opts.mimo_prefix_entries == 0 { return Ok(0); }
        let cache = crate::plan::FamilyModel::cache_geometry(model, crate::serving_capacity::CacheOptions {
            coordinator_ranks: gpus, attention_placement: Some(attention), mla_kv: opts.mla_kv, ..Default::default()
        })?.context("Kimi host cache geometry")?;
        let page_bytes = cache.ranks.iter().map(|rank| rank.persistent_unit_bytes).sum::<u64>();
        let mark_bytes = cache.ranks.iter().map(|rank| rank.retained_mark_bytes).sum::<u64>()
            + draft_window_bytes;
        let geometry = cuteafd_core::prefix::HostCacheGeometry { unit_rows: PAGE,
            page_bytes, mark_bytes, draft_bytes: 0 };
        geometry.resolve(opts.host_cache_bytes, context, opts.resolved_concurrency("kimi_k3"),
            host_total, geometry.chunk_bytes()).context("Kimi retained host prefix quota")
    })();
    let host_quota = match host_quota {
        Ok(bytes) => bytes,
        Err(error) => { report.refuse_placement(format!("Kimi host prefix refused: {error:#}"), "Disable host retention or supply a supported quota."); return memory; }
    };
    recipe.owners.push(kimi::host_prefix_owner(host_quota, maximum_pages));
    let scoring_bytes = match kimi::host_scoring_bytes(rows.min(64), rows, cfg.vocab_size) {
        Ok(bytes) => bytes,
        Err(error) => { report.refuse_placement(error.to_string(), "Use supported scoring dimensions."); return memory; }
    };
    recipe.owners.push(kimi::host_scoring_owner(scoring_bytes, maximum_pages));
    recipe.owners.push(kimi::host_upload_owner(maximum_pages));
    match kimi::stream_slab_owners_with_block(gpus,maximum_pages,
        block_tp2) {
        Ok(owners) => recipe.owners.extend(owners),
        Err(error) => { report.refuse_placement(format!("Kimi stream slab owners refused: {error}"),
            "Use supported coordinator stream counts."); return memory; }
    }
    match kimi::native_load_owners(&model.core, gpus, maximum_pages) {
        Ok(owners) => recipe.owners.extend(owners),
        Err(error) => { report.refuse_placement(format!("Kimi native load owners refused: {error:#}"),
            "Supply supported native attention source geometry."); return memory; }
    }
    let push = kimi::selected_push_host_envelope(cfg, &model.core, gpus).and_then(|envelope|
        kimi::push_host_owners(&envelope, maximum_pages));
    match push {
        Ok(owners) => recipe.owners.extend(owners),
        Err(error) => { report.refuse_placement(format!("Kimi push host owners refused: {error}"),
            "Supply supported coordinator push geometry."); return memory; }
    }
    memory.notes.push(if model.core.largest_selected_dense_bytes.is_some() {
        "Kimi native attention Load-only source, row cache, quant tables and push windows are derived from the selected view; actual records validate before load.".into()
    } else {
        format!("Kimi push Load-only host owners include {} GiB estimated dense staging per coordinator rank; #1329 measured 9,606,907,904 B aggregate with unknown rank split. These do not add to the Ready host ledger.",
            kimi::PUSH_DENSE_BYTES_PER_RANK >> 30)
    });
    let mut budgets = match kimi::budgets(&opts.rtx_bytes.iter().map(|&bytes| (bytes, bytes)).collect::<Vec<_>>(),
        crate::placement::reserve::gpu("kimi_k3"), &occupied, &sparks) {
        Ok(value) => value,
        Err(error) => { report.refuse_placement(format!("Kimi physical budgets refused: {error}"),
            "Check the requested card capacities."); return memory; }
    };
    budgets.push(kimi::host_budget(host_total, host_available));
    recipe.vision = match crate::placement::still_encoder::StillEncoder::planned(input.checkpoint(),
        report.vision, opts.vision_replicas, &budgets,
        opts.media_cache_bytes.unwrap_or((8 << 30).min(host_total / 20))) {
        Ok(vision) => vision,
        Err(error) => { report.refuse_placement(format!("Kimi bundled vision inventory refused: {error:#}"),
            "Supply the bundled MoonViT3d tensors or use --vision off."); return memory; }
    };
    report.host_budget_bytes = host_quota;
    let mut host_items = vec![cuteafd_core::memory_layout::host_table_item("Kimi host scoring logits", scoring_bytes),
        cuteafd_core::memory_layout::host_table_item("Kimi reusable H2D staging (estimated)", kimi::HOST_UPLOAD_BYTES)];
    for (lane, link) in recipe.links.iter().enumerate() {
        for (location, &bytes) in link.envelope() {
            if matches!(*location, crate::placement::planner::Location::HOST | crate::placement::planner::Location::PINNED) {
                host_items.push(cuteafd_core::memory_layout::host_table_item(format!("Kimi lane {lane} transport {location:?}"), bytes));
            }
        }
    }
    report.host_ledger = cuteafd_core::memory_layout::HostLayout::new(host_total, host_available,
        (8 << 30).max(host_total / 10), host_quota, host_items);
    let resolved = recipe.admit(kimi::request(crate::placement::planner::Context::resolve(context, (opts.context_tokens > 0).then_some(opts.context_tokens)), requested_pool, onboard, prefill,
        crate::placement::planner::Operation::Serving, ResidentWeights(opts.weight_budget_bytes), opts.spark_reply_formats), budgets);
    let mut vision_items = vec![Vec::new(); ranks];
    let onboarded = match resolved {
        Ok(selection) => {
            report.encoder = recipe.vision.as_ref().and_then(|vision| vision.admitted(&selection.solved));
            report.phase_ledger = Some(crate::plan::SolvedPhaseLedger::from(&selection.solved));
            for (rank, items) in vision_items.iter_mut().enumerate() {
                *items = selection.solved.items(crate::placement::planner::Location {
                    device: crate::placement::planner::Device::Spark(rank as u8),
                    domain: crate::placement::planner::Domain::Managed }).unwrap_or_default().into_iter()
                    .filter(|item| item.group.starts_with("vision ")).collect();
            }
            let served = selection.solved.request.context.served(selection.solved.pool_tokens);
            memory.notes.retain(|note| !note.starts_with("Kimi API max context:"));
            memory.notes.push(format!("Kimi API max context: {served} tokens; compiled max context: {COMPILED_CAPACITY} tokens"));
            if context_split {
                memory.notes.retain(|note| !note.starts_with("Kimi context gathered-prefill staging"));
                let staging = |extent: u64| 2 * extent.div_ceil(PAGE) * PAGE * opts.mla_kv.record_bytes() as u64;
                memory.notes.push(format!("Kimi context gathered-prefill staging per GPU: configured {served} tokens = {} B ({:.4} GiB); \
                    compiled cap {COMPILED_CAPACITY} tokens = {} B ({:.4} GiB); two full-history planes, bounded by served context and pool",
                    staging(served), staging(served) as f64 / GIB as f64,
                    staging(COMPILED_CAPACITY), staging(COMPILED_CAPACITY) as f64 / GIB as f64));
            }
            let placed = &selection.placement;
            memory.notes.push(format!("Kimi retained prefix marks are a fixed charge: {legacy_marks} slots per rank"));
            memory.notes.push(format!("Kimi selected RTX expert layers: {}; retained marks: {}", placed.onboard_layers, legacy_marks));
            memory.pool_tokens = placed.pool_tokens;
            report.attention_placement = placed.attention_placement;
            for gpu in 0..gpus {
                let mut device = DeviceLayout { kind: DeviceKind::Rtx, index: gpu as u32, capacity_bytes: opts.rtx_bytes[gpu], worker_budget_bytes: 0,
                    items: placed.items[gpu].clone(), kv_tokens: placed.pool_tokens };
                // The solver subtracts the named reserve from availability, not from its item ledger.
                device.items.extend(crate::placement::reserve::gpu_item("kimi_k3"));
                let owned = placed.layers.iter().enumerate().filter_map(|(layer, assignment)|
                    matches!(assignment.mode, LayerMode::Whole { gpu: owner, .. } if usize::from(owner) == gpu).then_some(layer))
                    .collect::<Vec<_>>();
                let categories = device.by_category();
                memory.notes.push(format!("RTX{gpu}: owned MLA layers {owned:?}; weights {} B, KV/state {} B, total {} B, \
                    reserved {} B, remaining {} B",
                    categories.get(&Category::Weights).unwrap_or(&0) + categories.get(&Category::Embedding).unwrap_or(&0),
                    categories.get(&Category::Kv).unwrap_or(&0), device.used_bytes(), crate::placement::reserve::gpu("kimi_k3"), device.free_bytes()));
                memory.devices.push(device);
            }
            memory.notes.push(format!("MLA latent {} B/token over {} layers; {} placement, KDA and FFNs on {gpus} physical owner(s); \
                pool {} tokens with {} routed layers on RTX ({:.2} GiB each)", opts.mla_kv.record_bytes() * cfg.mla_layers(),
                cfg.mla_layers(), placed.attention_placement, placed.pool_tokens, placed.onboard_layers, layer_bytes as f64 / GIB as f64));
            let onboarded = placed.onboard_layers;
            report.solver_placement = Some(selection.placement);
            onboarded
        }
        Err(error) => {
            report.refuse_placement(format!("Kimi K3 coordinator admission refused: {error}"),
                "Coordinator text weights are 105.7 GiB BF16; free KV room by placement, not a smaller pool floor.");
            return memory;
        }
    };
    // Sparks: resident slices of the routed layers left on them, plus the
    // shared runtime residual, latent-width (3584) I/O and EXL3 workspace.
    let working = SparkWorkingSet { rtx_layers: Vec::new(), spark_layers: spark_budget.layers[onboarded..].to_vec(), spark_ranks: ranks, prefill_lanes: LANES };
    let rings = crate::plan::layout::spark_ring_geometry(LANES as usize);
    for rank in 0..ranks {
        let resident = spark_budget.resident[rank][onboarded..].iter().sum::<u64>();
        let mut device = DeviceLayout { kind: DeviceKind::Spark, index: rank as u32, capacity_bytes: opts.spark_capacity(rank), worker_budget_bytes: opts.spark_capacity(rank),
            kv_tokens: 0, items: vec![Item::new(Category::Experts, "routed experts (balanced H128, replicated 3584 rotations)",
                model.expert_formats.join("+"), resident, Basis::Exact)] };
        if let Err(error) = crate::plan::layout::charge_spark_io(&mut device, &working, cuteafd_core::ExpertGeometry::KIMI_K3,
            unit as u32, false, rings, 0) {
            report.refuse_placement(format!("Kimi K3 Spark rank {rank} I/O: {error}"), "Fix the Spark ring geometry.");
        }
        if let Err(error) = kimi_owners(&mut device, &working, admitted.unit_rows, admitted.waves) {
            report.refuse_placement(format!("Kimi Spark execution owners: {error:#}"), "Use a supported wave count.");
        }
        device.items.extend(vision_items[rank].clone());
        memory.devices.push(device);
    }
    memory.notes.push(format!("Spark budget {} B/rank includes rank-specific header arenas, endpoint I/O, runtime residual and typed execution owners (one prefill scratch, {} x {}-row units)",
        spark_budget.budget, admitted.waves, admitted.unit_rows));
    memory
}

#[cfg(test)]
mod program_tests {
    use super::*;

    #[test]
    fn k31_module_census_keeps_shared_and_native_numerics_in_every_core_view() {
        let manifest = serde_json::json!({"programs": [
            {"name":"kimik3_rmsnorm_h7168", "scratch_bytes_at_capacity":{"scratch":16}},
            {"name":"kimik31_index_producer_m64", "scratch_bytes_at_capacity":{"scratch":32}},
            {"name":"kimik31_sparse_mla_h96_m64", "scratch_bytes_at_capacity":{"scratch":64}},
            {"name":"kimi_mla_rank_norm_h96_kv576_m64", "scratch_bytes_at_capacity":{"scratch":128}},
            {"name":"glmf_index_topk_decode_m64", "scratch_bytes_at_capacity":{"scratch":256}}
        ]});
        for dsa in [false, true] { for projected in [false, true] {
            let programs = ProgramSet::from_manifest(&manifest, program_families(dsa, projected));
            assert!(programs.names.iter().any(|name| name == "kimik3_rmsnorm_h7168"));
            for name in ["kimik31_index_producer_m64", "kimik31_sparse_mla_h96_m64"] {
                assert_eq!(programs.names.iter().any(|selected| selected == name), dsa);
            }
            assert_eq!(programs.names.iter().any(|name| name == "kimi_mla_rank_norm_h96_kv576_m64"), projected);
            assert!(!programs.names.iter().any(|name| name.starts_with("glmf_")));
            assert_eq!(programs.names.len(), 1 + usize::from(dsa) * 2 + usize::from(projected));
            assert_eq!(programs.scratch.iter().map(|(_, bytes)| bytes).sum::<u64>(),
                16 + u64::from(dsa) * 96 + u64::from(projected) * 128);
        } }
    }
}

/// Kimi's typed owners, the same sum its pre-placement reserve charges.
fn kimi_owners(device: &mut DeviceLayout, working: &SparkWorkingSet, unit_rows: u64, waves: u64) -> anyhow::Result<()> {
    if !working.selected_ranks().contains(&(device.index as usize)) { return Ok(()); }
    crate::families::kimi_k3::spark::charge(device, working.spark_ranks,
        &crate::families::kimi_k3::spark_executions(waves, unit_rows)?)
}
