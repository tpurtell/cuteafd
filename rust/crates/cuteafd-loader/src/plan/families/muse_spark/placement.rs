//! Muse Spark admission over the shared solver: header-exact coordinator
//! weights (K6 core payload + H128 vectors, BF16/F32 rest), paged full-layer
//! KV plus per-sequence sliding rings, pool first, then RTX expert onboarding
//! of whole routed layers; Spark ranks hold intermediate slices of every expert
//! and reply per-expert partials the coordinator combines.
//!
//! Header-only: coordinator graphs and step workspaces come from the family's
//! installed PROGRAMS.json once its programs exist; until then this is a
//! planning lower bound (stated in the notes), not serving admission.
use super::Model;
use crate::{placement::*, plan::{Component, ExpertPlacement, Hint, PlanReport}};
use cuteafd_core::memory_layout::{Basis, Category, DeviceKind, DeviceLayout, Item, MemoryLayout};

const GIB: u64 = 1 << 30;
/// Spark worlds the EXL3 TP package serves (the shared EXL3 recipe's worlds).
pub(super) const SPARK_WORLDS: [usize; 4] = [2, 3, 4, 6];
use crate::placement::families::muse_spark::LANES;

pub(super) fn layout(model: &Model, input: &crate::plan::PlanInput, report: &mut PlanReport,
    opts: &mut crate::plan::layout::LayoutOptions, requested: ExpertPlacement) -> MemoryLayout {
    let limits = crate::plan::Family::prefill_limits(&super::MUSE_SPARK);
    let prefill = match crate::plan::layout::resolve_prefill(report, opts, requested.spark_ranks(), limits) {
        Ok(prefill) => prefill,
        Err(error) => return crate::plan::layout::unsupported_prefill(report, error),
    };
    let cfg = model.cfg();
    let ranks = requested.spark_ranks();
    let gpus = opts.rtx_bytes.len();
    let mut memory = MemoryLayout { devices: Vec::new(), pool_tokens: 0, waste: Vec::new(), notes: vec![
        "Muse Spark header lower bound: coordinator graphs and step workspaces are charged from the family's \
            PROGRAMS.json once its programs exist; NOT serving admission".into(),
    ] };
    let refusal = |report: &mut PlanReport, what: String, how: &str| {
        report.fits = false;
        report.placement_supported = false;
        report.hints.push(Hint { what, how: how.into() });
    };
    report.bytes_by_owner.clear();
    report.attention_placement = AttentionPlacement::Heads;
    let context = if opts.context_tokens == 0 { cfg.max_position as u64 } else { opts.context_tokens };
    if context > cfg.max_position as u64 {
        refusal(report, format!("Muse Spark context {context} exceeds the checkpoint's {}", cfg.max_position),
            "Lower --context-tokens to at most 1M.");
        return memory;
    }
    if !model.complete {
        refusal(report, "Muse Spark placement cannot infer fit from missing safetensors headers".into(),
            "Place every shard's header locally (nest replicate) or plan from a complete snapshot.");
        return memory;
    }
    if !model.expert_formats.iter().any(|f| f.starts_with("exl3-k")) {
        refusal(report, format!("Muse Spark routed experts are {:?}; only the EXL3 publication has an expert package",
            model.expert_formats), "Serve the EXL3 K4 + Core-K6 snapshot.");
        return memory;
    }
    if ranks == 0 {
        refusal(report, "Muse Spark routed experts (~400 GB at K4) need Spark ranks".into(),
            "Use --spark-ranks 4..8.");
        return memory;
    }
    if !(1..=2).contains(&gpus) || (gpus == 2 && !opts.head_split) {
        refusal(report, "Muse Spark runs on one RTX or a head-split RTX pair".into(), "Use --rtx 1 or 2.");
        return memory;
    }
    // One request builder for planning and serving startup (plan == runtime).
    let slots = opts.state_slots.unwrap_or(if opts.concurrency > 0 { opts.concurrency } else { 16 });
    let catalog = match input.expert_catalog() {
        Ok(catalog) => catalog,
        Err(error) => {
            refusal(report, format!("Muse Spark expert inventory refused: {error:#}"), "Provide complete EXL3 expert headers.");
            return memory;
        }
    };
    // GPU marks follow the shared prefix rule: host-tier retention keeps only staging slots.
    let mark_slots = opts.prefix_slots.unwrap_or_else(|| cuteafd_core::prefix::mark_slots_with_host(slots,
        opts.mimo_prefix_entries, crate::placement::families::muse_spark::mark_bytes(cfg, 1),
        opts.mimo_prefix_mark_bytes, opts.host_cache_bytes != Some(0)));
    let inputs = crate::placement::families::muse_spark::MuseInputs { cfg, core: &model.core, catalog, mark_slots,
        gpus: opts.rtx_bytes.iter().map(|bytes| GpuBudget { capacity_bytes: *bytes, headroom_bytes: opts.headroom_bytes,
            baseline: Baseline::Planned { context_bytes: ArchContext::for_device("sm_120", *bytes).context_bytes,
                loaded_bytes: 0 } }).collect(),
        spark_ranks: ranks, prefill, slots, context, automatic_context: opts.context_tokens == 0,
        requested_pool: opts.pool_tokens.filter(|n| *n > 0), onboard: opts.onboard.unwrap_or(Onboard::Auto),
        spark_budget_bytes: opts.spark_allocation_budget_bytes.unwrap_or(crate::plan::DEFAULT_SPARK_BUDGET_BYTES),
        partial_bf16_max_rows: crate::families::muse_spark::replies::DEFAULT_BF16_MAX_ROWS,
        partial_wide_format: Default::default(),
        fixed: match planned_code(opts, gpus) {
            Ok(fixed) => fixed,
            Err(error) => {
                refusal(report, format!("Muse Spark program manifest refused: {error:#}"), "Supply the image's PROGRAMS.json.");
                return memory;
            }
        }, startup: Vec::new() };
    let built = match crate::placement::families::muse_spark::request(inputs) {
        Ok(built) => built,
        Err(error) => {
            refusal(report, format!("Muse Spark joint admission refused: {error:#}"),
                "Use more Spark ranks, automatic RTX expert selection, or a larger --spark-budget-gib.");
            return memory;
        }
    };
    let embedding = built.global.embedding as u64;
    let head = built.global.head_bytes() as u64;
    let experts = *model.bytes.get(&Component::RoutedExpert).unwrap_or(&0);
    let layer_bytes = (0..gpus).map(|gpu| built.layer_bytes(gpu).unwrap_or(0)).sum::<u64>();
    report.bytes_by_owner.insert("coordinator text weights".into(), embedding + head
        + built.global.final_norm as u64 * gpus as u64 + layer_bytes);
    report.bytes_by_owner.insert("routed expert weights".into(), experts);
    memory.notes.push(format!("Muse K6 core: {} header-native projections ({}); payload + H128 vectors charged",
        model.core.projections.len(), if model.core.complete(cfg) { "complete" } else { "partial: BF16 rest" }));
    let rings = crate::placement::families::muse_spark::ring_rows(cfg).unwrap_or(0)
        * cfg.kv_record_bytes() as u64 * cfg.sliding_layers() as u64;
    let placed = match solve(&built.request) {
        Ok(placed) => placed,
        Err(error) => {
            refusal(report, format!("Muse Spark coordinator admission refused: {error}"),
                "Free KV room by placement or add a second RTX; the pool floor is the compiled context.");
            return memory;
        }
    };
    let spark_budget = built.spark_budget;
    memory.pool_tokens = placed.pool_tokens;
    memory.notes.push(format!("Muse selected RTX expert layers: {}; pool {} tokens; full-layer KV {} B/token over {} \
        layers; sliding rings {} B/sequence over {} layers", placed.onboard_layers, placed.pool_tokens,
        cfg.kv_record_bytes() * cfg.full_layers(), cfg.full_layers(), rings, cfg.sliding_layers()));
    for gpu in 0..gpus {
        let mut device = DeviceLayout { kind: DeviceKind::Rtx, index: gpu as u32, capacity_bytes: opts.rtx_bytes[gpu],
            items: placed.items[gpu].clone(), kv_tokens: placed.pool_tokens };
        device.items.push(Item::new(Category::Reserved, "coordinator headroom", "", opts.headroom_bytes, Basis::Formula)
            .with_lifetime(cuteafd_core::memory_layout::ItemLifetime::PostReadyReserve));
        memory.devices.push(device);
    }
    let onboarded = placed.onboard_layers;
    report.solver_placement = Some(placed);
    let working = SparkWorkingSet { rtx_layers: Vec::new(), spark_layers: spark_budget.layers[onboarded..].to_vec(),
        spark_ranks: ranks, prefill_lanes: LANES };
    let rings = crate::plan::layout::spark_ring_geometry(LANES as usize);
    let mut widest = 0;
    for rank in 0..ranks {
        let resident = spark_budget.resident[rank][onboarded..].iter().sum::<u64>();
        widest = widest.max(resident);
        let mut device = DeviceLayout { kind: DeviceKind::Spark, index: rank as u32, capacity_bytes: opts.spark_bytes,
            kv_tokens: 0, items: vec![Item::new(Category::Experts, "routed experts (intermediate slices)",
                model.expert_formats.join("+"), resident, Basis::Exact)] };
        if let Err(error) = usize::try_from(slots).map_err(anyhow::Error::from)
            .and_then(|slots| muse_spark_io(&mut device, &working, prefill, slots, rings)) {
            refusal(report, format!("Muse Spark rank {rank} I/O: {error:#}"), "Fix the Spark ring geometry.");
        }
        if let Err(error) = muse_owners(&mut device, &working, prefill.unit_rows, prefill.waves) {
            refusal(report, format!("Muse Spark execution owners: {error:#}"), "Use a supported wave count.");
        }
        memory.devices.push(device);
    }
    report.spark_rank_share = if experts == 0 { 0.0 } else { widest as f64 / experts as f64 };
    memory.notes.push(format!("Spark: intermediate slices over {ranks} ranks, widest rank {:.1} GiB of {:.1} GiB routed",
        widest as f64 / GIB as f64, experts as f64 / GIB as f64));
    memory
}

/// Loaded code charged exactly as live startup charges it (the full allowance
/// offline, nothing yet resident), once the image's program manifest is known.
fn planned_code(opts: &crate::plan::layout::LayoutOptions, gpus: usize) -> anyhow::Result<Vec<Demand>> {
    let Some(path) = opts.workspace_manifest.as_deref().filter(|path| path.is_file()) else { return Ok(Vec::new()) };
    let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let programs = ProgramSet::from_manifest(&manifest, crate::placement::families::muse_spark::program_families(gpus));
    (0..gpus).map(|rank| {
        let arch = ArchContext::for_device("sm_120", opts.rtx_bytes[rank]);
        let code = crate::placement::inventory::loaded_code_allowance("muse", "exl3", gpus == 2, rank as u8,
            arch, Some(&programs));
        crate::placement::families::muse_spark::code_demand(rank, code.bytes)
    }).collect()
}

/// The rank's Spark I/O at the shared Muse endpoint (`spark_endpoint`): its
/// partial-reply frame and in-flight depth, exactly as admission seals it.
fn muse_spark_io(device: &mut DeviceLayout, working: &SparkWorkingSet,
    prefill: crate::serving_capacity::prefill_waves::PrefillPlan, slots: usize,
    rings: crate::placement::SparkRingGeometry) -> anyhow::Result<()> {
    let geometry = cuteafd_core::ExpertGeometry::MUSE_SPARK;
    crate::plan::layout::charge_spark_io(device, working, geometry, prefill.unit_rows as u32, false, rings, 0)?;
    // The planner's serving defaults, as the request builder (`MuseInputs`) gets them.
    let cutoff = crate::families::muse_spark::replies::Cutoff::with_wide(
        crate::families::muse_spark::replies::DEFAULT_BF16_MAX_ROWS, 1, Default::default())?;
    let (endpoint, spark_rings) = crate::placement::families::muse_spark::spark_endpoint(prefill, cutoff, slots)?;
    let exact = working.rank_endpoints_io(device.index as usize, geometry, &[endpoint], spark_rings)?;
    for item in &mut device.items {
        match item.group.as_str() {
            "expert intake" => item.bytes = exact.intake,
            "host expert exchange" => item.bytes = exact.host_exchange,
            "expert row indices" => item.bytes = exact.row_indices,
            "rdma rings" => item.bytes = exact.mapped_rings,
            "endpoint response prefixes" => item.bytes = exact.endpoint_staging,
            _ => {},
        }
    }
    Ok(())
}

fn muse_owners(device: &mut DeviceLayout, working: &SparkWorkingSet, unit_rows: u64, waves: u64) -> anyhow::Result<()> {
    if !working.selected_ranks().contains(&(device.index as usize)) { return Ok(()); }
    use crate::placement::families::muse_spark::{owner_bytes, spark_executions};
    crate::plan::layout::spark_slots::charge_owners(device, &spark_executions(waves, unit_rows)?, owner_bytes)
}
