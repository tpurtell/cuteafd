//! Kimi TP2 admission with checkpoint residency and shared native allocation layouts.
use super::Model;
use anyhow::Context;
use crate::{placement::*, plan::{Component, ExpertPlacement, Hint, PlanReport}};
use cuteafd_core::memory_layout::{Basis, Category, DeviceKind, DeviceLayout, Item, MemoryLayout};

const GIB: u64 = 1 << 30;
const COMPILED_CAPACITY: u64 = crate::families::kimi_k3::COMPILED_CONTEXT as u64;
const PAGE: u64 = 64;
const MODES: &[LayerMode] = &[LayerMode::HeadSplit,
    LayerMode::Whole { gpu: 0, ffn: FfnMode::Split }, LayerMode::Whole { gpu: 1, ffn: FfnMode::Split }];
/// Spark EXL3 workspace across the capacity set, estimated until the Kimi
/// package manifest records it (Gemini review of #273: ~430 MiB at m4096).
const SPARK_EXL3_WORKSPACE: u64 = crate::families::kimi_k3::SPARK_ESTIMATED_WORKSPACE_BYTES;
/// Prefill lanes the Spark transport serves (one QP endpoint each).
const LANES: u64 = 1;

pub(super) fn layout(model: &Model, report: &mut PlanReport, opts: &mut crate::plan::layout::LayoutOptions,
    requested: ExpertPlacement) -> MemoryLayout {
    use crate::families::kimi_k3::allocation::{DEFAULT_PREFILL_UNIT_ROWS, DEFAULT_PREFILL_WAVES, LANE_ROWS_MAX};
    let unit_rows = if opts.prefill_rows > 0 { opts.prefill_rows } else { DEFAULT_PREFILL_UNIT_ROWS as u64 };
    let waves = if opts.prefill_lanes > 0 { opts.prefill_lanes } else { DEFAULT_PREFILL_WAVES as u64 };
    let shape = crate::serving_capacity::PrefillShape { unit_rows, waves };
    if let Err(error) = shape.validate().and_then(|shape| {
        anyhow::ensure!((256..=LANE_ROWS_MAX as u64).contains(&shape.slot_rows()),
            "Kimi prefill span must hold 256..={LANE_ROWS_MAX} rows");
        Ok(shape)
    }) {
        report.fits = false;
        report.placement_supported = false;
        report.hints.push(Hint { what: format!("Kimi prefill: {error:#}"),
            how: "Select page-aligned unit rows and waves with a 256..=8192-row span.".into() });
        return MemoryLayout { devices: Vec::new(), pool_tokens: 0, waste: Vec::new(), notes: vec![error.to_string()] };
    }
    let rows = shape.slot_rows() as usize;
    let unit = unit_rows as usize;
    report.prefill_span_rows = Some(shape.slot_rows());
    report.prefill_transports = Some(LANES);
    opts.prefill_unit_rows = Some(unit_rows);
    opts.prefill_waves = waves;
    let cfg = model.cfg();
    let ranks = requested.spark_ranks();
    let gpus = opts.rtx_bytes.len();
    let mut memory = MemoryLayout { devices: Vec::new(), pool_tokens: 0, waste: Vec::new(), notes: vec![
        "Kimi K3 header lower bound: coordinator workspaces/graphs not charged; NOT serving admission".into(),
        "Spark budgets are offline estimates (including an explicit override), not load/steady census-qualified admission".into(),
    ] };
    let refusal = |report: &mut PlanReport, what: String, how: &str| {
        report.fits = false;
        report.placement_supported = false;
        report.hints.push(Hint { what, how: how.into() });
    };
    report.bytes_by_owner.clear();
    report.attention_placement = report.requested_attention.unwrap_or(AttentionPlacement::Layers);
    let context = if opts.context_tokens == 0 { crate::families::kimi_k3::default_context(&model.core) as u64 } else { opts.context_tokens };
    memory.notes.push(format!("Kimi API max context: {context} tokens; compiled max context: {COMPILED_CAPACITY} tokens"));
    if context > COMPILED_CAPACITY {
        refusal(report, format!("Kimi context {context} exceeds compiled capacity {COMPILED_CAPACITY}"),
            "Lower --context-tokens to at most 1M; the pool must cover the selected context extent.");
        return memory;
    }
    if report.requested_attention == Some(AttentionPlacement::Context) {
        refusal(report, "Kimi K3 context attention placement has no owned-page query exchange/LSE-combine ABI".into(),
            "Use layers (owned MLA with split KDA/FFNs) or explicitly select heads; context requires a new native ABI.");
        return memory;
    }
    if !model.complete {
        refusal(report, "Kimi K3 placement cannot infer fit from missing safetensors headers".into(),
            "Place every shard's header locally (nest replicate) or plan from a complete snapshot.");
        return memory;
    }
    let experts = *model.bytes.get(&Component::RoutedExpert).unwrap_or(&0);
    let routed_layers = cfg.dense.iter().filter(|d| !**d).count() as u64;
    let layer_bytes = experts / routed_layers;
    // TP6 width 512: 1/6 of every trellis and of the 3072-wide rotations, whole
    // copies of the 3584-wide input (w1/w3 suh) and output (w2 svh) rotations.
    let replicated_rotations = cfg.experts as u64 * 3 * cfg.latent as u64 * 2;
    let rank_layer = |world: u64| (layer_bytes - replicated_rotations).div_ceil(world) + replicated_rotations;
    if ranks != 6 {
        refusal(report, format!("Kimi K3 missing Spark recipe kimik3:exl3-k23 TP{ranks}: estimated routed experts \
            need {:.1} GiB/rank", (rank_layer(ranks.max(1) as u64) * routed_layers) as f64 / GIB as f64),
            "Export balanced H128 widths in package_exl3_aot.py, register spark_recipe, and parameterize Kimi Spark intake/admission by rank count; TP6 width 512 is currently installed.");
        return memory;
    }
    if gpus != 2 || !opts.head_split {
        refusal(report, "Kimi K3 targets two coordinator GPUs with split KDA and FFNs".into(),
            "Use --rtx 2; one RTX cannot hold 105.7 GiB of BF16 text weights.");
        return memory;
    }
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
    let slots = opts.state_slots.unwrap_or_else(|| if opts.concurrency > 0 { opts.concurrency } else { 16 });
    let path = opts.workspace_manifest.as_deref().unwrap_or_else(||
        std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json"));
    // Header-only diagnostics remain available without installed artifacts. An
    // explicit package or a serving image must supply a complete allocation ledger.
    let manifest = if path.is_file() {
        match std::fs::read(path).map_err(anyhow::Error::from)
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).map_err(Into::into)) {
            Ok(value) => Some(value),
            Err(error) => { refusal(report, format!("Kimi package inventory refused: {error:#}"),
                "Supply the matching PROGRAMS.json through --workspace-manifest."); return memory; }
        }
    } else if opts.workspace_manifest.is_some() {
        refusal(report, "Kimi explicit workspace manifest is missing".into(), "Supply the installed PROGRAMS.json.");
        return memory;
    } else { None };
    let state = cfg.kda_layers() as u64 * (cfg.kda_state_bytes() + cfg.kda_conv_bytes()) * slots / 2;
    let mark_bytes = cfg.kda_layers() as u64 * (cfg.kda_state_bytes() + cfg.kda_conv_bytes()) / 2;
    let legacy_marks = opts.prefix_slots.unwrap_or_else(|| cuteafd_core::prefix::mark_slots_with_host(
        slots, opts.mimo_prefix_entries, 2 * mark_bytes, opts.mimo_prefix_mark_bytes,
        opts.host_cache_bytes != Some(0)));
    let mut mark_demands = (0..2).map(|gpu| Demand::new(gpu, Category::Prefix,
        "Kimi exact KDA marks", mark_bytes, Basis::Formula)).collect::<Vec<_>>();
    let mut fixed = Vec::new();
    for demand in &mark_demands {
        let mut demand = demand.clone();
        let Some(bytes) = demand.bytes.checked_mul(legacy_marks) else {
            refusal(report, "Kimi mark reservation overflow".into(), "Reduce prefix retention.");
            return memory;
        };
        demand.bytes = bytes;
        fixed.push(demand);
    }
    let packed = manifest.as_ref().map(|_| &model.core);
    let globals = packed.map(|core| crate::families::kimi_k3::weight_layout::GlobalLayout::pair_with_core(cfg, core)).transpose();
    let globals = match globals { Ok(value) => value, Err(error) => {
        refusal(report, format!("Kimi packed global inventory refused: {error:#}"), "Check checkpoint global geometry."); return memory;
    }};
    for gpu in 0..2u8 {
        if let Some(globals) = &globals {
            fixed.push(globals[usize::from(gpu)].demand().expect("validated global extent"));
        } else {
        fixed.push(Demand::new(gpu, Category::Weights, "text weights, head/intermediate split", residency.split / 2, Basis::Exact));
        fixed.push(Demand::new(gpu, Category::Weights, "text weights replicated (KDA low-rank A, routers, latent proj, norms)",
            residency.replicated - model.frontends[1 - gpu as usize], Basis::Exact));
        fixed.push(Demand::new(gpu, if gpu == 0 { Category::Embedding } else { Category::Weights },
            if gpu == 0 { "token embedding" } else { "lm head" }, if gpu == 0 { embedding } else { head }, Basis::Exact));
        }
        fixed.push(Demand::new(gpu, Category::Kv, format!("KDA FP32 state + BF16 conv, {slots} slots, 48 heads"), state, Basis::Formula));
    }
    let mut baseline = [Baseline::Planned { context_bytes: 0, loaded_bytes: 0 }; 2];
    let mut tp2_workspace = [0; 2];
    if let Some(manifest) = &manifest {
        let extra = (|| -> anyhow::Result<_> {
            use crate::families::kimi_k3::{allocation, core};
            // Speculative verify graphs replay at each exact width up to the drafter's block.
            let verify = opts.kimi_drafter_snapshot.as_ref().map(|snapshot|
                crate::formats::dflash::MlaDflashConfig::read(snapshot).map(|draft| draft.block))
                .transpose()?.unwrap_or(0);
            let mut demands = allocation::baseline_demands(cfg, manifest, shape, context as usize,
                usize::try_from(slots)?, verify, &model.core, report.requested_attention == Some(AttentionPlacement::Heads))?;
            let families = if model.core.projections.is_empty() { &["kimik3"][..] } else { &["kimik3", "kimi"][..] };
            let programs = ProgramSet::from_manifest(manifest, families);
            for rank in 0..2 {
                let arch = ArchContext::for_device("sm_120", opts.rtx_bytes[rank]);
                let code = crate::placement::inventory::loaded_code_allowance("kimik3", "exl3", true,
                    rank as u8, arch, Some(&programs));
                baseline[rank] = Baseline::Planned { context_bytes: arch.context_bytes, loaded_bytes: 0 };
                demands.push(allocation::code_demand(rank, code.bytes, arch.cublas_bytes, 1)?);
            }
            if !model.core.projections.is_empty() {
                // The baseline already includes the full peer receive allocation.
                demands.extend(core::CoreCharges::from_package(cfg, &model.core, rows, unit, manifest)?.demands()?
                    .into_iter().filter(|d| d.group != "Kimi K6 peer receive"));
            }
            if opts.onboard != Some(Onboard::Layers(0)) {
                let package = allocation::expert_package(path)?;
                tp2_workspace = [allocation::expert_workspace(&package, unit)?; 2];
            }
            Ok(demands)
        })();
        match extra {
            Ok(demands) => fixed.extend(demands),
            Err(error) => { refusal(report, format!("Kimi package admission refused: {error:#}"),
                "Supply complete matching native and EXL3 workspace manifests."); return memory; }
        }
        memory.notes[0] = "Kimi shared runtime allocation ledger; graph executable bytes retain the runtime's explicit startup estimate".into();
    } else if !model.core.projections.is_empty() {
        refusal(report, "Kimi native K6 requires installed workspace inventory".into(),
            "Supply the matching K6 PROGRAMS.json through --workspace-manifest.");
        return memory;
    }
    let inventory = (|| -> anyhow::Result<_> {
        let catalog = crate::read_expert_catalog(&model.snapshot)?;
        let working = SparkWorkingSet { rtx_layers: vec![], spark_layers: cfg.dense.iter().enumerate()
            .filter_map(|(layer, dense)| (!dense).then_some(layer)).collect(), spark_ranks: ranks, prefill_lanes: LANES };
        let rings = crate::plan::layout::spark_ring_geometry(LANES as usize);
        let endpoints = [cuteafd_core::SparkEndpoint { ring_depth: None, capacity: opts.spark_capacity_rows as u32,
            ingress: cuteafd_core::SparkIngress::Fp8K32, execution_lane: 0 }];
        let budget = opts.spark_allocation_budget_bytes.unwrap_or(super::SPARK_BUDGET_BYTES);
        let spark = SparkBudgetInventory::exl3(&catalog, &working, &endpoints,
            cuteafd_core::SparkRings { max_frame_bytes: rings.max_frame_bytes, depth: rings.depth,
                slot_bytes: rings.slot_bytes, alignment: rings.alignment }, budget,
            crate::families::kimi_k3::SPARK_ESTIMATED_RESERVE_BYTES)?;
        let manifest = catalog.exl3().ok_or_else(|| anyhow::anyhow!("Kimi TP2 requires EXL3 headers"))?;
        let mut costs = vec![[Bytes2::default(); 2]; cfg.layers];
        for &layer in &working.spark_layers {
            for rank in 0..2 {
                costs[layer][rank].resident = manifest.residency(crate::V41Exl3Layer::Backbone(layer), 2, rank)?
                    .device_arena_layout()?.1 as u64;
            }
        }
        Ok((spark, costs))
    })();
    let (spark_budget, costs) = match inventory {
        Ok(value) => value,
        Err(error) => {
            refusal(report, format!("Kimi header inventory refused: {error:#}"), "Provide complete native EXL3 headers.");
            return memory;
        }
    };
    let draft = opts.kimi_drafter_snapshot.as_ref().map(|snapshot| -> anyhow::Result<_> {
        use crate::{families::glm5::draft_representation::{GlmDraftCapacity, GlmDraftRepresentation},
            formats::dflash::MlaDflashConfig};
        let draft = MlaDflashConfig::read(snapshot)?;
        anyhow::ensure!(draft.mla.hidden == cfg.hidden && draft.vocab == cfg.vocab_size
            && draft.taps.iter().all(|&layer| layer < cfg.layers), "Kimi drafter target geometry differs from checkpoint");
        let slots = usize::try_from(slots)?;
        let representation = if opts.kimi_draft_fp8 { GlmDraftRepresentation::Fp8Only }
            else { GlmDraftRepresentation::Bf16Only };
        let layout = draft.runtime_layout(representation, GlmDraftCapacity::new(slots, slots.min(8), draft.block)?,
            usize::try_from(context)?, slots.checked_mul(usize::try_from(context)?).context("draft pool overflow")?, opts.physical_sms.unwrap_or(188) as usize, 0, opts.mla_kv)?;
        fixed.extend(crate::families::kimi_k3::draft_bridge_demands(&layout, rows)?);
        fixed.push(crate::families::kimi_k3::draft_probe_demand(cfg.layers, cfg.topk, 64)?);
        // A K6 vocabulary head has no BF16 operand to borrow: the drafter owns
        // a head-only K6 scratch on the head rank.
        if let Some(manifest) = &manifest {
            fixed.extend(crate::families::kimi_k3::core::draft_head_demand(&model.core,
                layout.capacity().block_rows, &manifest["families"]["kimi_k6_dense"])?);
        }
        let bytes = layout.window_mark_bytes()? as u64;
        if bytes > 0 && legacy_marks > 0 {
            let per_mark = Demand::new(1, Category::Prefix, "Kimi exact draft windows", bytes, Basis::Exact);
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
            refusal(report, format!("Kimi native MLA draft inventory refused: {error:#}"),
                "Provide a supported MLA DFlash2 checkpoint and matching native representation.");
            return memory;
        }
    };
    let record = opts.mla_kv.record_bytes() as u64 * PAGE;
    let layers = (|| -> anyhow::Result<Vec<LayerDemand>> { (0..cfg.layers).map(|layer| {
        let mla = cfg.attention[layer] == crate::families::kimi_k3::config::KimiAttention::Mla;
        let mut weights = ModeBytes { whole: model.mla[layer].split + model.mla[layer].replicated,
            split: [model.mla[layer].split / 2 + model.mla[layer].replicated; 2] };
        if let Some(core) = packed {
            use crate::families::kimi_k3::weight_layout::LayerLayout;
            let split = LayerLayout::pair_with_core(cfg, layer, LayerMode::HeadSplit, core)?;
            let mut attention = [0; 2];
            for rank in 0..2 {
                attention[rank] = split[rank].attention_bytes()? as u64;
                fixed.push(Demand::new(rank as u8, Category::Weights,
                    format!("Kimi layer {layer} TP2 FFN/residual/router/latent"),
                    split[rank].bytes()? as u64 - attention[rank], Basis::Exact));
            }
            weights = ModeBytes { split: attention, whole: if mla {
                LayerLayout::pair_with_core(cfg, layer, MODES[1], core)?[0].attention_bytes()? as u64
            } else { 0 } };
        }
        Ok(LayerDemand { kind: if mla { AttentionClass::Mla } else { AttentionClass::Kda },
            weights,
            kv_unit: KvDemand { unit_bytes_whole: if mla { record } else { 0 }, unit_bytes_split: if mla { [record; 2] } else { [0; 2] },
                unit_bytes_context: None },
            fixed_bytes: ModeBytes::default(), extra: Vec::new(), context_indexer: false, colocate: None,
            experts: (!cfg.dense[layer]).then_some(ExpertCost { whole: Bytes2::default(),
                half: costs[layer], tp2: true, spark_ok: true }), modes: if mla { MODES.to_vec() } else { vec![LayerMode::HeadSplit] } })
    }).collect() })();
    let layers = match layers { Ok(value) => value, Err(error) => {
        refusal(report, format!("Kimi packed layer inventory refused: {error:#}"), "Check checkpoint layer geometry."); return memory;
    }};
    // Reuse pool-first TP2 onboarding, constrained by the Spark suffix budget.
    let onboard = opts.onboard.unwrap_or(Onboard::Auto);
    let mut request = PlacementRequest {
        inventory: Inventory { gpus: opts.rtx_bytes.iter().enumerate().map(|(rank, bytes)| GpuBudget { capacity_bytes: *bytes,
            headroom_bytes: opts.headroom_bytes, baseline: baseline[rank] }).collect(),
            spark_ranks: ranks, peer_access: true },
        attention_placement: report.requested_attention, context_buffers: ContextBuffers::default(), layers_first_gpu: 0,
        pool: PoolPolicy { compiled_context: context,
            requested: requested_pool, target: target_pool, floor: context,
            ceiling: target_pool.max(requested_pool.unwrap_or(0)), unit_rows: PAGE },
        layers, pool_overhead: vec![0; 2], pool_bounded: Vec::new(), fixed, startup: Vec::new(), movables: Vec::new(),
        expert_workspace: 0, tp2_workspace, onboard, expert_gpus: 2, minimum_expert_layers: 0,
        policy: LayerPolicy { default: vec![LayerMode::HeadSplit], ownership: LayerOwnership::Ranges, expert_ownership: ExpertOwnership::Contiguous,
            attention: vec![(AttentionClass::Mla, AttentionPlacement::Layers), (AttentionClass::Kda, AttentionPlacement::Heads)],
            ..Default::default() },
        hops: HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: cfg.hidden as u64 * 2, rows: if manifest.is_some() { unit as u64 } else { 0 }, lanes: if manifest.is_some() { LANES } else { 0 },
            entry_gpu: 0, head_gpu: 1, terminal: TerminalHome::Replicated },
        executor: ExecutorModes { family: "kimi_k3", modes: MODES, hops: true, attention: AttentionPlacement::Layers },
    };
    if let Err(error) = spark_budget.constrain(&mut request) {
        refusal(report, format!("Kimi joint admission refused: {error:#}"),
            "Use automatic RTX expert selection or increase the explicit estimated Spark budget.");
        return memory;
    }
    let groups = request.layers.iter().map(|layer|
        (layer.kind == AttentionClass::Mla && request.attention_placement != Some(AttentionPlacement::Heads), layer.colocate))
        .collect::<Vec<_>>();
    let selected = attention::solve_owners(&groups, LayerOwnership::Ranges, request.layers_first_gpu,
        request.onboard, |owners| -> anyhow::Result<_> {
            let mut candidate = request.clone();
            for (layer, (&owner, &(selected, _))) in candidate.layers.iter_mut().zip(owners.iter().zip(&groups)) {
                if selected { layer.modes = vec![LayerMode::Whole { gpu: owner, ffn: FfnMode::Split }]; }
            }
            let admit = |trial: &PlacementRequest| -> anyhow::Result<_> {
                if legacy_marks == 0 {
                    Ok((solve(trial)?, MarkSelection { legacy_slots: 0, slots: 0, reduction_reason: None }))
                } else { Ok(solve_with_marks(trial, legacy_marks, 1, &mark_demands)?) }
            };
            if let Some(layout) = &draft {
                settle_pool((candidate.pool.requested.unwrap_or(candidate.pool.target), ()), |tokens, _| -> anyhow::Result<_> {
                    let mut trial = candidate.clone();
                    let layout = layout.with_pool_tokens(usize::try_from(tokens)?)?;
                    let (demand, peak) = layout.demands(1)?;
                    trial.movables = vec![demand.movable(2)?];
                    trial.startup = vec![peak];
                    trial.pool.target = trial.pool.target.min(tokens);
                    trial.pool.ceiling = trial.pool.ceiling.min(tokens);
                    let (placed, marks) = admit(&trial)?;
                    Ok((placed.pool_tokens, (), (placed, marks)))
                })
            } else { admit(&candidate) }
        });
    let onboarded = match selected {
        Ok((placed, selection)) => {
            memory.notes.push(selection.summary());
            memory.notes.push(format!("Kimi selected RTX expert layers: {}; retained marks: {}", placed.onboard_layers, selection.slots));
            memory.pool_tokens = placed.pool_tokens;
            report.attention_placement = placed.attention_placement;
            for gpu in 0..2 {
                let mut device = DeviceLayout { kind: DeviceKind::Rtx, index: gpu as u32, capacity_bytes: opts.rtx_bytes[gpu],
                    items: placed.items[gpu].clone(), kv_tokens: placed.pool_tokens };
                // The solver subtracts this guard from availability, not from its item ledger.
                device.items.push(Item::new(Category::Reserved, "coordinator headroom", "", opts.headroom_bytes, Basis::Formula)
                    .with_lifetime(cuteafd_core::memory_layout::ItemLifetime::PostReadyReserve));
                let owned = placed.layers.iter().enumerate().filter_map(|(layer, assignment)|
                    matches!(assignment.mode, LayerMode::Whole { gpu: owner, .. } if usize::from(owner) == gpu).then_some(layer))
                    .collect::<Vec<_>>();
                let categories = device.by_category();
                memory.notes.push(format!("RTX{gpu}: owned MLA layers {owned:?}; weights {} B, KV/state {} B, total {} B, \
                    reserved {} B, remaining {} B",
                    categories.get(&Category::Weights).unwrap_or(&0) + categories.get(&Category::Embedding).unwrap_or(&0),
                    categories.get(&Category::Kv).unwrap_or(&0), device.used_bytes(), opts.headroom_bytes, device.free_bytes()));
                memory.devices.push(device);
            }
            memory.notes.push(format!("MLA latent {} B/token over {} layers; {} placement, KDA and FFNs split TP2; \
                pool {} tokens with {} routed layers on RTX ({:.2} GiB each)", opts.mla_kv.record_bytes() * cfg.mla_layers(),
                cfg.mla_layers(), placed.attention_placement, placed.pool_tokens, placed.onboard_layers, layer_bytes as f64 / GIB as f64));
            placed.onboard_layers
        }
        Err(error) => {
            refusal(report, format!("Kimi K3 coordinator admission refused: {error}"),
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
        let mut device = DeviceLayout { kind: DeviceKind::Spark, index: rank as u32, capacity_bytes: opts.spark_bytes,
            kv_tokens: 0, items: vec![Item::new(Category::Experts, "routed experts (TP6 width 512, replicated 3584 rotations)",
                model.expert_formats.join("+"), resident, Basis::Exact)] };
        if let Err(error) = crate::plan::layout::charge_spark_io(&mut device, &working, cuteafd_core::ExpertGeometry::KIMI_K3,
            opts.spark_capacity_rows as u32, false, rings, 0) {
            refusal(report, format!("Kimi K3 Spark rank {rank} I/O: {error}"), "Fix the Spark ring geometry.");
        }
        device.items.push(Item::new(Category::Workspace, "EXL3 expert workspace", "kimik3:exl3-k23", SPARK_EXL3_WORKSPACE, Basis::Estimated));
        memory.devices.push(device);
    }
    memory.notes.push(format!("Spark budget {} B/rank includes exact TP6 header arenas and endpoint I/O plus {} B estimated runtime/workspace reserve; census pending",
        spark_budget.budget, crate::families::kimi_k3::SPARK_ESTIMATED_RESERVE_BYTES));
    memory
}
