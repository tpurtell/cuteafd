//! Kimi physical-owner admission with checkpoint residency and shared native allocation layouts.
use super::Model;
use anyhow::Context;
use crate::families::kimi_k3::allocation;
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

pub(super) fn layout(model: &Model, input: &crate::plan::PlanInput, report: &mut PlanReport, opts: &mut crate::plan::layout::LayoutOptions,
    requested: ExpertPlacement) -> MemoryLayout {
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
    let target_pool = opts.target_pool_tokens.max(context);
    let edge = |c| *model.bytes.get(&c).unwrap_or(&0);
    let (embedding, head) = (edge(Component::Embedding), model.core.projections.get("language_model.lm_head")
        .map_or(edge(Component::LmHead), |p| p.resident_bytes(1).expect("validated core dimensions")));
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
            crate::families::kimi_k3::context::ContextLayout::from_manifest(manifest, opts.mla_kv,
                context as usize, unit, banks)
        })();
        match layout {
            Ok(layout) => Some(layout),
            Err(error) => {
                report.refuse_placement(format!("Kimi context program admission refused: {error:#}"),
                    &format!("Install PROGRAMS.json with {}.",
                        crate::families::kimi_k3::context::program_names(opts.mla_kv).join(", ")));
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
    let globals = match crate::families::kimi_k3::weight_layout::GlobalLayout::ranks_with_core(cfg, &model.core, gpus) {
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
    if let Some(manifest) = &manifest {
        let extra = (|| -> anyhow::Result<_> {
            use crate::families::kimi_k3::{allocation, core};
            // Speculative verify graphs replay at each exact width up to the drafter's block.
            let verify = opts.kimi_drafter_snapshot.as_ref().map(|snapshot|
                crate::formats::dflash::MlaDflashConfig::read(snapshot).map(|draft| draft.block))
                .transpose()?.unwrap_or(0);
            let mut demands = allocation::baseline_demands_for_ranks(cfg, manifest, rows, shape, context as usize,
                usize::try_from(slots)?, verify, &model.core, matches!(attention, AttentionPlacement::Heads | AttentionPlacement::Context),
                ranks, gpus, opts.resolved_spark_reply_format(false))?;
            if let Some(context) = &context_layout {
                demands.retain(|d| d.group != allocation::GRAPH_RESERVE);
                let graphs = allocation::graphs_with_context(cfg, 1, unit, usize::try_from(slots)?, verify,
                    gpus, |_, _| true, |layer| cfg.attention[layer] == crate::families::kimi_k3::config::KimiAttention::Mla,
                    Some(context.decode()))?;
                demands.extend((0..gpus).map(|rank| allocation::graph_demand(&graphs, rank)));
            }
            let families = if model.core.projections.is_empty() { &["kimik3"][..] } else { &["kimik3", "kimi"][..] };
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
            if !model.core.projections.is_empty() {
                // The baseline already includes the full peer receive allocation.
                demands.extend(core::CoreCharges::from_package_with_native(cfg, &model.core, rows, unit, manifest, &native_manifest, gpus)?.demands()?
                    .into_iter().filter(|d| d.group != "Kimi K6 peer receive"));
            }
            if opts.onboard != Some(OnboardSetting::Layers(0)) {
                let package = allocation::expert_package_for_ranks(path, gpus)?;
                tp2_workspace = [allocation::expert_workspace_for_ranks_with(&package, unit, gpus, |path| input.manifest(path))?; 2];
            }
            Ok(demands)
        })();
        match extra {
            Ok(demands) => {
                fixed.extend(demands);
                if opts.full_prefill_logits {
                    let ordinary = crate::families::kimi_k3::allocation::io_extents_for_ranks(
                        cfg.vocab_size, head_rank, rows, false, gpus).expect("validated IO ranks");
                    let scoring = crate::families::kimi_k3::allocation::io_extents_with_logits(
                        cfg.vocab_size, head_rank, rows, rows, false, gpus).expect("validated IO rows");
                    let delta = (scoring[1] - ordinary[1]) as u64;
                    let io = fixed.iter_mut().find(|d| d.gpu as usize == head_rank && d.group == "Kimi lane 0 token IO")
                        .expect("baseline includes terminal token IO");
                    io.bytes += delta;
                }
            },
            Err(error) => { report.refuse_placement(format!("Kimi package admission refused: {error:#}"),
                "Supply complete matching native and EXL3 workspace manifests."); return memory; }
        }
        memory.notes[0] = "Kimi shared runtime allocation ledger; graph executable bytes retain the runtime's explicit startup estimate".into();
    } else if !model.core.projections.is_empty() {
        report.refuse_placement("Kimi native K6 requires installed workspace inventory".into(),
            "Supply the matching K6 PROGRAMS.json through --workspace-manifest.");
        return memory;
    }
    if opts.full_prefill_logits && manifest.is_none() {
        fixed.push(Demand::new(head_rank as u8, Category::Workspace, "Kimi full prefill logits",
            (rows - rows.min(64)) as u64 * cfg.vocab_size as u64 * 4, Basis::Exact));
    }
    let inventory = (|| -> anyhow::Result<_> {
        let catalog = input.expert_catalog()?;
        if ranks == 0 { return Err(kimi::LOCAL_ONLY_REFUSAL.into()); }
        let working = SparkWorkingSet { rtx_layers: vec![], spark_layers: cfg.dense.iter().enumerate()
            .filter_map(|(layer, dense)| (!dense).then_some(layer)).collect(), spark_ranks: ranks, prefill_lanes: LANES };
        let rings = crate::plan::layout::spark_ring_geometry(LANES as usize);
        let endpoints = [cuteafd_core::SparkEndpoint { ring_depth: None, capacity: unit as u32,
            ingress: cuteafd_core::SparkIngress::Fp8K32, partial_reply_bytes: None }];
        let budget = opts.spark_allocation_budget_bytes.unwrap_or(super::SPARK_BUDGET_BYTES);
        let spark = SparkBudgetInventory::exl3(&catalog, &working, &endpoints,
            cuteafd_core::SparkRings { max_frame_bytes: rings.max_frame_bytes, depth: rings.depth,
                slot_bytes: rings.slot_bytes, alignment: rings.alignment }, budget,
            crate::families::kimi_k3::spark::reserve_bytes(ranks, admitted.waves, admitted.unit_rows)?)?;
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
        let draft = MlaDflashConfig::read(snapshot)?;
        anyhow::ensure!(draft.mla.hidden == cfg.hidden && draft.vocab == cfg.vocab_size
            && draft.taps.iter().all(|&layer| layer < cfg.layers), "Kimi drafter target geometry differs from checkpoint");
        anyhow::ensure!(!opts.tp2_dflash || gpus == 2,
            "Kimi TP2 DFlash requires two physical coordinator GPUs");
        let slots = usize::try_from(slots)?;
        let representation = if opts.kimi_draft_fp8 { GlmDraftRepresentation::Fp8Only }
            else { GlmDraftRepresentation::Bf16Only };
        let layout = draft.runtime_layout(representation, GlmDraftCapacity::new(slots, slots.min(8), draft.block)?,
            usize::try_from(context)?, slots.checked_mul(usize::try_from(context)?).context("draft pool overflow")?, opts.physical_sms.unwrap_or(188) as usize, 0, opts.mla_kv)?.with_context_split(opts.tp2_dflash)?;
        fixed.extend(crate::families::kimi_k3::draft_bridge_demands_for_ranks(&layout, rows, gpus)?);
        fixed.push(crate::families::kimi_k3::draft_probe_demand(cfg.layers, cfg.topk, 64)?);
        // A K6 vocabulary head has no BF16 operand to borrow: the drafter owns
        // a head-only K6 scratch on the head rank.
        if let Some(manifest) = &manifest {
            if let Some(mut demand) = crate::families::kimi_k3::core::draft_head_demand(&model.core,
                layout.capacity().block_rows, &manifest["families"]["dense_trellis_k6"])? {
                demand.gpu = head_rank as u8;
                fixed.push(demand);
            }
        }
        let bytes = layout.window_mark_bytes()? as u64;
        if bytes > 0 && legacy_marks > 0 {
            let per_mark = Demand::new(head_rank as u8, Category::Prefix, "Kimi exact draft windows", bytes, Basis::Exact);
            let mut total = per_mark.clone();
            total.bytes = bytes.checked_mul(legacy_marks).context("Kimi draft mark overflow")?;
            fixed.push(total);
            mark_demands.push(per_mark);
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
    let mut fixed_owners = fixed.iter().map(FixedOwner::ready).collect::<Vec<_>>();
    if context_split {
        match context_buffers.demands() {
            Ok(demands) => fixed_owners.extend(demands.iter().map(FixedOwner::ready)),
            Err(error) => { report.refuse_placement(format!("Kimi context buffers refused: {error:#}"),
                "Check the context program geometry."); return memory; }
        }
    }
    let peer_row_bytes = if context_split {
        context_buffers.query_row_bytes.saturating_add(context_buffers.partial_row_bytes)
    } else { 0 };
    let page_rows = if modes.iter().any(|mode| *mode == LayerMode::ContextSplit) { PAGE * 2 } else { PAGE };
    let ceiling = target_pool.max(requested_pool.unwrap_or(0));
    let maximum_pages = ceiling.div_ceil(page_rows);
    let draft_window_bytes = match draft.as_ref().map(|layout| layout.window_mark_bytes()).transpose() {
        Ok(bytes) => bytes.unwrap_or(0) as u64,
        Err(error) => {
            report.refuse_placement(format!("Kimi draft window inventory refused: {error:#}"),
                "Supply a supported native drafter window layout.");
            return memory;
        }
    };
    let draft = match &draft {
        Some(layout) => match kimi::draft_facts(layout, page_rows, maximum_pages, head_rank as u8) {
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
        expert_usage: kimi::usage_geometry(cfg, opts.expert_usage),
        context, gpus, placement: attention, attention: modes,
        prefill, lanes: LANES, spark_links: ranks > 0,
        hop_spec: HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: cfg.hidden as u64 * 2,
            rows: if manifest.is_some() { unit as u64 } else { 0 },
            lanes: if manifest.is_some() { LANES } else { 0 },
            entry_gpu: 0, head_gpu: head_rank as u8,
            terminal: if gpus == 1 { TerminalHome::AttentionOwner } else { TerminalHome::Replicated } },
        peer_row_bytes, maximum_pages,
        records_per_page, records_by_layer, layer_attention, first_moe,
        kinds,
        experts: Some(ExpertProfile { layers: spark_budget.layers.clone(),
            destination: if gpus == 2 { vec![0, 1] } else { vec![0] },
            counts, rank_sets: if ranks > 0 { vec![(0..ranks as u8).collect()] } else { Vec::new() } }),
        arenas, spark, links, draft, fixed: fixed_owners, owners: Vec::new(),
    };
    if let Err(error) = recipe.arena_padding(cfg, &model.core) {
        report.refuse_placement(format!("Kimi packed arena inventory refused: {error:#}"),
            "Provide complete selected native weight extents.");
        return memory;
    }
    if let Err(error) = recipe.blas_owners(if model.core.get_weight("language_model.lm_head").is_some() {
        crate::placement::planner::serving::BlasHead::Native } else { opts.bf16_target_head }) {
        report.refuse_placement(format!("Kimi BLAS owners refused: {error}"), "Supply supported native head geometry.");
        return memory;
    }
    if manifest.is_some() {
        let graph_result = recipe.graph_reserve();
        if let Err(error) = graph_result {
            report.refuse_placement(format!("Kimi graph inventory refused: {error:#}"), "Supply supported graph geometry.");
            return memory;
        }
    }
    let occupied = vec![0u64; gpus];
    for (rank, baseline) in baseline.iter().enumerate() {
        if let crate::placement::Baseline::Planned { context_bytes, loaded_bytes } = baseline {
            recipe.fixed.push(FixedOwner { rank: rank as u8, category: Category::Runtime,
                group: allocation::CONTEXT_ESTIMATE.into(), bytes: context_bytes.saturating_add(*loaded_bytes),
                weights: 0, lifetime: crate::placement::planner::Lifetime::RESIDENT, basis: Basis::Estimated });
        }
    }
    let sparks = vec![spark_budget.budget; ranks];
    let (host_total, host_available) = match opts.host_memory {
        Some(memory) => memory,
        None => match crate::formats::mapped_preload::host_memory() {
            Ok(memory) => memory,
            Err(error) => { report.refuse_placement(format!("Kimi host inventory refused: {error:#}"), "Supply host memory capacity."); return memory; }
        },
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
    let scoring_bytes = match kimi::host_scoring_bytes(if opts.full_prefill_logits { rows } else { rows.min(64) }, rows, cfg.vocab_size) {
        Ok(bytes) => bytes,
        Err(error) => { report.refuse_placement(error.to_string(), "Use supported scoring dimensions."); return memory; }
    };
    recipe.owners.push(kimi::host_scoring_owner(scoring_bytes, maximum_pages));
    recipe.owners.push(kimi::host_upload_owner(maximum_pages));
    match kimi::stream_slab_owners(gpus, maximum_pages) {
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
    let onboarded = match resolved {
        Ok(selection) => {
            let served = selection.solved.request.context.served(selection.solved.pool_tokens);
            memory.notes.retain(|note| !note.starts_with("Kimi API max context:"));
            memory.notes.push(format!("Kimi API max context: {served} tokens; compiled max context: {COMPILED_CAPACITY} tokens"));
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
        let mut device = DeviceLayout { kind: DeviceKind::Spark, index: rank as u32, capacity_bytes: opts.spark_bytes, worker_budget_bytes: opts.spark_bytes,
            kv_tokens: 0, items: vec![Item::new(Category::Experts, "routed experts (balanced H128, replicated 3584 rotations)",
                model.expert_formats.join("+"), resident, Basis::Exact)] };
        if let Err(error) = crate::plan::layout::charge_spark_io(&mut device, &working, cuteafd_core::ExpertGeometry::KIMI_K3,
            unit as u32, false, rings, 0) {
            report.refuse_placement(format!("Kimi K3 Spark rank {rank} I/O: {error}"), "Fix the Spark ring geometry.");
        }
        if let Err(error) = kimi_owners(&mut device, &working, admitted.unit_rows, admitted.waves) {
            report.refuse_placement(format!("Kimi Spark execution owners: {error:#}"), "Use a supported wave count.");
        }
        memory.devices.push(device);
    }
    memory.notes.push(format!("Spark budget {} B/rank includes rank-specific header arenas, endpoint I/O, runtime residual and typed execution owners (one prefill scratch, {} x {}-row units)",
        spark_budget.budget, admitted.waves, admitted.unit_rows));
    memory
}

/// Kimi's typed owners, the same sum its pre-placement reserve charges.
fn kimi_owners(device: &mut DeviceLayout, working: &SparkWorkingSet, unit_rows: u64, waves: u64) -> anyhow::Result<()> {
    if !working.selected_ranks().contains(&(device.index as usize)) { return Ok(()); }
    crate::families::kimi_k3::spark::charge(device, working.spark_ranks,
        &crate::families::kimi_k3::spark_executions(waves, unit_rows)?)
}
