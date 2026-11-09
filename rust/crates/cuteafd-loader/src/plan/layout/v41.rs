//! Native V4.1 allocation owners, shared with serving admission. CED layer
//! ownership is fixed; the shared planner chooses only the KV pool and the
//! local expert prefix from these facts.
pub use crate::placement::families::deepseek_v41::resident::*;
use super::*;
use anyhow::{ensure, Context, Result};
use crate::placement::families::deepseek_v41::admission::{self, Opaque, Owner, ProductionInput};
use crate::placement::{self, families::deepseek_v41 as v41, ArchContext, Demand, Onboard, PoolPolicy,
    ResidentWeights, SparkWorkingSet};
use crate::placement::planner::{Lifetime, Location, Operation, PrefillShape};
use v41::DraftInventory;

const SPARK: SparkCharges = SparkCharges {
    host_bytes: Some(9 * GIB), wave_bytes: 21 * GIB / 10, basis: Basis::Estimated,
};

pub(crate) fn layout(report: &mut PlanReport, model: &dyn crate::plan::FamilyModel,
    input: &crate::plan::PlanInput, options: &mut LayoutOptions) -> Result<MemoryLayout> {
    let checkpoint = input.checkpoint();
    // Units, waves and span come from the shared prefill policy and the
    // descriptor (two units in flight, one target pass per slot).
    let limits = crate::plan::Family::prefill_limits(&crate::plan::families::deepseek::DEEPSEEK_V41);
    // The reserve the units are cut from: explicit, else the standard one.
    // The launcher's serve argv carries it, so a pinned shape reports the
    // standard reserve it replaced. Reported before the shared prefill solve
    // so a refusal keeps it and the count selector never re-solves the shape.
    report.prefill_reserve_rows = Some(options.prefill_reserve_rows
        .unwrap_or(crate::serving_capacity::prefill_waves::RESERVE_ROWS));
    let prefill = super::resolve_prefill(input, report, options, report.placement.spark_ranks(), limits)?;
    let mut shape = Shape::resolve(input, report, model, options, ShapePolicy {
        context: ContextPolicy::Checkpoint, prefill_rows: Some(prefill.unit_rows), all_native_stages: false,
    });
    shape.prefill_rows = prefill.unit_rows;
    report.prefill_span_rows = Some(prefill.span_rows);
    let ranks = options.rtx_bytes.len();
    ensure!((1..=2).contains(&ranks), "V4.1 requires one or two coordinator GPUs");
    cuteafd_core::concurrency::validate("deepseek_v41", shape.concurrency)?;
    let rows = u64::from(cuteafd_core::coordinator_programs::v41_live_rows(
        u32::try_from(shape.prefill_rows)?).context("invalid V4.1 prefill rows")?);
    let marks = options.prefix_slots.unwrap_or_else(|| cuteafd_core::prefix::mark_slots_with_host(0, options.prefix_entries("deepseek_v41"), 1, u64::MAX, options.host_cache_bytes != Some(0)));
    ensure!(marks == 0 || (marks >= 2 && marks % 2 == 0 && marks <= 258),
        "V4.1 prefix slots must be zero or 2 * retained turns + 2 (at most 258)");
    // Immutable COW tails retain the configured conversations, independently of staging marks.
    let turns = if options.host_cache_bytes != Some(0) && marks > 0 { options.prefix_entries("deepseek_v41") }
        else { marks.saturating_sub(2) / 2 };
    let dspark = shape.native_layers > 0;
    ensure!(!options.tp2_dspark_experts || (ranks == 2 && dspark),
        "--tp2-dspark-experts requires two RTX GPUs and dSpark");
    ensure!(!dspark || shape.native_layers == 3, "V4.1 dSpark retains all three stages");
    let manifest = options.workspace_manifest.as_deref()
        .unwrap_or_else(|| std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json"));
    let directory = manifest.parent().context("V4.1 manifest has no parent")?;
    let fp8: serde_json::Value = serde_json::from_slice(&std::fs::read(directory.join("V41_FP8_AOT.json"))
        .context("missing V4.1 native FP8 allocation manifest")?)?;
    let sms = options.physical_sms.unwrap_or_else(|| ArchContext::for_device("sm_120", options.rtx_bytes[0]).sms);
    let programs = v41::owners::NativePrograms::new(&fp8, sms)?;
    let mode = Fp8Head::configured();
    let window_batch = ranks == 1 && std::env::var("CUTEAFD_WINDOW_BATCH").map_or(true, |v| v != "0");
    // The prefill slots, one storage set over their units; two decode lanes
    // at decode width.
    let slots = usize::try_from(prefill.waves)?;
    let span = slots as u64 * rows;
    let target = programs.target(v41::owners::DECODE_LANE_ROWS, span, rows, ranks, slots,
        shape.concurrency, mode, window_batch)?;
    let weights = resident_weights(checkpoint, ranks, false)
        .context("unsupported V4.1 native backbone residency: check non-expert tensor formats and geometry")?;
    let mut devices = options.rtx_bytes.iter().enumerate().map(|(rank, &bytes)| DeviceLayout {
        kind: DeviceKind::Rtx, index: rank as u32, capacity_bytes: bytes, worker_budget_bytes: 0,
        items: weights[rank].clone(), kv_tokens: 0,
    }).collect::<Vec<_>>();
    let mut waste = Vec::new();
    let mut notes = vec![format!("resolved context {} tokens (checkpoint {})", shape.context_tokens, shape.checkpoint_context),
        "Native module residency: selected V41_FP8_AOT projection exports, estimated separately from physical BLAS pools. The existing lazy graph reserve remains; diagnostic startup capture is uncharged and outside this inventory.".into(),
        "Spark host, wave residual and native runtime: Estimated pending qualified worker inventory; explicit rank I/O remains formula-derived.".into(),
        "Lazy graph/runtime growth is a typed post-ready reserve, not part of the ready residency claim.".into()];
    if options.host_embedding || (options.rtx_bytes[0] <= 32 * GIB && !options.force_gpu_embedding) {
        place_host_embedding(report, checkpoint, &mut devices, &mut waste, &mut notes);
    }
    // Retained device weights are one preplanned arena per GPU, from the same
    // header recipe the loader allocates; the embedding joins GPU0's arena
    // only while it stays on the GPU. Planned layouts use the default (unsplit)
    // query/output projections, the only configuration the planner models.
    let gpu_embedding = devices[0].items.iter().any(|item| item.category == Category::Embedding);
    let target_layout = v41::weight_layout::TargetLayout::from_headers(
        checkpoint.tensors.iter().map(|tensor| &tensor.meta),
        (ranks == 2).then(|| std::array::from_fn(|layer| usize::from(layer >= 20))),
        gpu_embedding, mode, false, false)?;
    for device in &mut devices { device.items.retain(|item| !v41::owners::is_arena_weight(item)); }
    let mut fixed = v41::owners::target_arena_demands(&target_layout);
    fixed.extend(devices.iter().flat_map(|d| d.items.iter().map(|i|
        Demand::new(d.index as u8, i.category, i.group.clone(), i.bytes, i.basis)
            .with_resident_weights(ResidentWeights(i.bytes)))));
    let mut startup: Vec<Owner> = Vec::new();
    if ranks == 1 {
        startup.push(Owner::from_demand(&v41::owners::vocabulary_load_phase(&target_layout.ranks[0], mode)?));
    }
    let spark_ranks = report.placement.spark_ranks();
    // The two decode lanes' links, further slots' reduce outputs and the one prefill link.
    let transport = if spark_ranks > 0 { v41::owners::transports_bytes(v41::owners::DECODE_LANE_ROWS, rows, spark_ranks, slots, options.resolved_spark_reply_format(false))? } else { 0 };
    let mut fixed = fixed;
    if options.drafter_bytes > 0 {
        fixed.push(Demand::new(0, Category::Drafter, "external drafter weights",
            options.drafter_bytes, Basis::Exact).with_resident_weights(ResidentWeights(options.drafter_bytes)));
    }
    let mut owners = v41::owners::TargetOwners::new(fixed, target, ranks, transport, marks)?;
    let catalog = input.expert_catalog()?;
    let (experts, workspace) = v41::expert_inventory(catalog, manifest, rows, ranks, sms, slots, span,
        v41::owners::DECODE_LANE_ROWS)?;
    let draft = if dspark {
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(directory.join("V41_EXPERT_AOT.json"))
            .context("missing V4.1 dSpark coordinator allocation manifest")?)?;
        let resident = drafter_weights(checkpoint, ranks).context("unsupported V4.1 dSpark resident format")?;
        let bytes = resident.iter().flatten().try_fold(0u64, |sum, owner|
            sum.checked_add(owner.bytes).context("V4.1 draft weight budget overflow"))?;
        let staging = catalog.expert_staging(crate::V41ExpertSelection::Dspark { stage: 0, expert: 0 })?.staging_bytes() as u64;
        ensure!((1..=cuteafd_core::MAX_DSPARK_PROPOSALS as u8).contains(&options.v41_draft_width), "invalid V4.1 dSpark draft limit");
        let width = if options.v41_draft_width > 5 { 7 } else { 5 };
        let dense = v41::weight_layout::draft_layout_from_headers(checkpoint.tensors.iter().map(|tensor| &tensor.meta))?;
        let mut draft = DraftInventory { resident: bytes, runtime: programs.dspark(&manifest, rows, shape.concurrency, width, ranks,
            mode)?, marks: marks * 3 * 128 * 528,
            staging: [staging, 0], loading_experts: [3 * native_expert_bytes(128, 5120, 2304), 0],
            loader_peak: [bytes, 0], dense: Some(dense), ..Default::default() };
        if options.tp2_dspark_experts {
            let halves = serde_json::from_slice(&std::fs::read(directory.join("V41_EXPERT_WORKSPACES/mxfp4_dspark_tp2.json"))
                .context("missing V4.1 TP2 dSpark allocation manifest")?)?;
            draft.runtime = programs.dspark_tp2(&manifest, &halves, rows, shape.concurrency, width, mode)?;
            let resident = if catalog.mxfp4_scale_codes() {
                native_expert_bytes(128, 5120, 1152)
            } else { v41::resident_bytes(5120, 1152, 128, false)? } * 3;
            let full = if catalog.mxfp4_scale_codes() {
                native_expert_bytes(128, 5120, 2304)
            } else { v41::resident_bytes(5120, 2304, 128, false)? } * 3;
            draft.resident = bytes.checked_sub(full).context("V4.1 draft resident underflow")? + resident;
            draft.peer_half = resident;
            let peer_staging = catalog.expert_staging(crate::V41ExpertSelection::DsparkTp2 {
                stage: 0, expert: 0, rank: 0 })?.staging_bytes() as u64;
            draft.staging = [staging, peer_staging];
            draft.loading_experts = [resident; 2];
            draft.loader_peak = [draft.resident.max(resident + staging), resident + peer_staging];
        }
        Some(draft)
    } else { None };
    let mut pool = PoolPolicy::resolve(&options.rtx_bytes, shape.context_tokens, options.pool_tokens, 512)
        .with_context(crate::placement::planner::Context::resolve(shape.context_tokens, (options.context_tokens > 0).then_some(options.context_tokens)));
    // CLI zero and an omitted pool are the same automatic admission.
    let automatic = pool.requested.is_none();
    if automatic {
        pool.target = pool.target.min(options.target_pool_tokens.max(shape.context_tokens));
    }
    pool.validate_explicit()?;
    let maximum_pool_tokens = pool.maximum_tokens();
    let context = shape.context_tokens;
    let mut working = SparkWorkingSet { rtx_layers: Vec::new(),
        spark_layers: if spark_ranks > 0 { (0..40).collect() } else { Vec::new() },
        spark_ranks, prefill_lanes: slots as u64 };
    let mut sparks = spark_devices(report, model, input, options, SPARK, &working,
        spark_endpoints("deepseek_v41", working.prefill_lanes), &mut waste, &mut notes);
    // Ready-resident owners and the two typed per-GPU reserves. The pool bytes
    // and the routed-expert arenas are separate contract components.
    let mut physical = owners.fixed.iter().map(Owner::from_demand).collect::<Vec<_>>();
    let mut reserves = Vec::new();
    let draft_rows = dspark.then(|| v41::owners::draft_rows(shape.concurrency, options.v41_draft_width));
    let modules = programs.modules(v41::owners::DECODE_LANE_ROWS, rows, ranks, draft_rows)?;
    let mut cards = Vec::with_capacity(ranks);
    for (gpu, &total) in options.rtx_bytes.iter().enumerate() {
        let arch = ArchContext::coordinator(total, None);
        let ceiling = total;
        devices[gpu].capacity_bytes = ceiling;
        devices[gpu].items.clear();
        cards.push((total, ceiling));
        for (bytes, basis, source) in [
            (arch.context_bytes, Basis::Calibrated, "CUDA context"),
            (modules[gpu].module_bytes(), Basis::Estimated, "V4.1 selected native modules"),
        ] {
            reserves.push(Opaque { location: Location::coordinator(gpu as u8), bytes, basis,
                lifetime: Lifetime::RESIDENT, source: source.into() });
        }
        reserves.push(admission::graph_allowance(gpu)?);
    }
    if let Some(draft) = &draft {
        startup.extend(owners.draft_load_phases(draft)?.iter().map(Owner::from_demand));
    }
    let onboard = if spark_ranks == 0 { Onboard::Layers(40) }
        else { options.resolve_onboard(40).unwrap_or(Onboard::Auto) };
    let onboard_minimum = usize::from(ranks == 2 && !matches!(onboard, Onboard::Layers(0)));
    // Encoders are enumerated against the pre-solve hardware; the shared solve
    // keeps the first candidate whose whole placement fits every phase.
    let towers = v41_towers(report);
    let (vision_weights, vision_scratch) = match towers.vision {
        Ok(bytes) => bytes,
        Err(_) if report.vision == crate::plan::MediaMode::Off => (0, 0),
        Err(reason) => anyhow::bail!("bundled vision tower unavailable: {reason}"),
    };
    let mut encoders = encoder_candidates(report, options, &devices[..ranks], &sparks,
        vision_weights, vision_scratch);
    if report.vision == crate::plan::MediaMode::Auto && vision_weights > 0 {
        encoders.push(crate::plan::encoder::encoder_placement(crate::plan::MediaMode::Off,
            &encoder_hardware(true, &devices[..ranks], &sparks, options), 0, 0, 1));
    }
    if encoders.is_empty() { anyhow::bail!("requested bundled encoder has no executable device"); }
    let host = options.host_memory.map(Ok).unwrap_or_else(crate::formats::mapped_preload::host_memory_before_process)?;
    let host_facts = admission::HostFacts::prepare(&checkpoint.config,
        &checkpoint.tensors.iter().map(|tensor| &tensor.meta).collect::<Vec<_>>(),
        PrefillShape { rows, waves: slots as u64 }, ranks, spark_ranks, context, shape.concurrency,
        dspark, options.host_cache_bytes, options.host_cache_bytes != Some(0) && turns > 0,
        256 << 20, host, report.mapped_placement.cap_bytes, Some(report.mapped_placement.host_budget_bytes), options.full_prefill_logits)?;
    report.mapped_placement = host_facts.mapped.clone();
    report.bytes_by_owner.remove("host-preloaded");
    if host_facts.mapped.preloaded_bytes > 0 {
        report.bytes_by_owner.insert("host-preloaded".into(), host_facts.mapped.preloaded_bytes);
    }
    mapped_table_note(report, &mut notes);
    if notes.iter().any(|note| note.starts_with("embedding placement: host")) {
        let bytes = checkpoint.tensors.iter().filter(|tensor| tensor.meta.name == "embed.weight")
            .map(|tensor| tensor.meta.byte_length).sum();
        physical.push(Owner::host(Category::Embedding, "embedding", bytes).at(Location::PINNED));
    }
    physical.extend(host_facts.owners.clone());
    reserves.extend(host_facts.estimates.clone());
    let budgets = admission::production_budgets(&cards, crate::placement::reserve::gpu("deepseek_v41"), Some(host), 0,
        &vec![options.spark_bytes; spark_ranks])?;
    let request = admission::production_request(crate::placement::planner::Context::resolve(context, (options.context_tokens > 0).then_some(options.context_tokens)), options.pool_tokens, onboard,
        PrefillShape { rows, waves: slots as u64 },
        if options.full_prefill_logits { Operation::Scoring } else { Operation::Serving },
        options.weight_budget_bytes, options.spark_reply_formats);
    let media_cache_bytes = options.media_cache_bytes.unwrap_or(0);
    let mut prepared = ProductionInput {
        blas: v41::owners::blas_owners(ranks,
            if mode.target() { crate::placement::planner::serving::BlasHead::Native }
            else { options.bf16_target_head }, dspark)?,
        expert_usage: options.expert_usage,
        config: &checkpoint.config,
        gpus: ranks,
        context,
        maximum_pool_tokens,
        prefill: PrefillShape { rows, waves: slots as u64 },
        concurrency: shape.concurrency,
        retained_turns: turns,
        replicated: false,
        spark_ranks,
        onboard_minimum,
        onboard_counts: onboard.layers(40).map(|layers| vec![layers]),
        experts,
        workspace,
        owners: physical.drain(..).collect(),
        weight_arenas: target_layout.ranks.clone(),
        startup,
        allowances: reserves,
        draft,
        encoders: Vec::new(),
        media_cache_bytes: 0,
        resident_weight_limit: options.weight_budget_bytes,
        spark_reply_formats: options.spark_reply_formats,
        operation: if options.full_prefill_logits { Operation::Scoring } else { Operation::Serving },
        budgets,
    }.prepare()?;
    prepared.recipe.encoders = encoders.clone();
    prepared.recipe.vision_arena = (vision_weights > 0).then(|| v41::weight_layout::vision_layout_from_headers(
        checkpoint.tensors.iter().map(|tensor| &tensor.meta))).transpose()?;
    prepared.recipe.media_cache_bytes = media_cache_bytes;
    let vision_extent = prepared.recipe.vision_arena.as_ref().map_or(0, |layout| layout.bytes() as u64);
    let admitted = prepared.admit(request)?;
    report.host_ledger = Some(admitted.host_ledger(host_facts.prefix_bytes)?);
    let mut encoder = encoders[admitted.solved.decisions.encoder.context("missing admitted encoder")?].clone();
    if report.vision == crate::plan::MediaMode::Auto && vision_weights > 0
        && encoder.kind == crate::plan::encoder::EncoderKind::Off {
        encoder.reason = format!("Auto disabled image input at the pool-first/onboarding solution: {}; free memory or choose a smaller layout",
            admitted.solved.encoder_diagnostics.iter().map(|(reason, _)| reason.as_str()).collect::<Vec<_>>().join("; "));
        encoder.shortfall = admitted.solved.encoder_diagnostics.iter().map(|(_, bytes)| *bytes)
            .filter(|bytes| *bytes > 0).min().unwrap_or(0);
        tracing::warn!(reason = %encoder.reason, shortfall_bytes = encoder.shortfall, "bundled vision disabled by shared admission");
    }
    if encoder.kind == crate::plan::encoder::EncoderKind::Off {
        if let Some(component) = report.components.iter_mut().find(|c| c.component == Component::Vision)
            .filter(|c| c.status != Status::Unused) {
            report.disabled_media_bytes += component.bytes;
            component.bytes = 0;
            component.status = Status::Disabled;
        }
    }
    notes.push(format!("vision {:?}: {}; {} bytes admitted", encoder.kind, encoder.reason, encoder.admitted_bytes()));
    report.encoder = Some(encoder.clone());
    report.phase_ledger = Some(crate::plan::SolvedPhaseLedger::from(&admitted.solved));
    let placement = admitted.placement;
    working = admitted.spark;
    report.solver_placement = Some(placement.clone());
    report.attention_placement = placement.attention_placement;
    notes.push(format!("attention placement {}", placement.attention_placement));
    notes.push(format!("placement: {}", placement.summary()));
    notes.push(format!("{} routed backbone layers resident on RTX", placement.onboard_layers));
    if working.spark_layers.is_empty() { sparks.clear(); }
    for rank in encoder.spark_ranks() {
        ensure_tower_spark(&mut sparks, rank, options);
        let Some(device) = sparks.iter_mut().find(|device| device.index as usize == rank) else { continue };
        let overhead = crate::plan::encoder::V41_SPARK_CUDA_OVERHEAD_BYTES;
        device.items.push(Item::new(Category::Weights, "vision tower", "", vision_extent, Basis::Exact));
        device.items.push(Item::new(Category::Workspace, "vision scratch", "",
            encoder.scratch.saturating_sub(overhead), Basis::Exact));
        device.items.push(Item::new(Category::Runtime, "vision CUDA overhead", "", overhead, Basis::Exact));
    }
    if catalog.exl3().is_none() && catalog.nvfp4().is_none() {
        let e = catalog.routed_experts();
        let capacity = cuteafd_core::V41_TARGET_CAPACITIES.into_iter().find(|&c| c >= rows)
            .context("native expert prefill capacity exceeds 4096")?;
        if !working.rtx_layers.is_empty() {
            notes.push(cuteafd_core::mxfp4_layouts::native_prefill_note(e.hidden,
                [e.intermediate / ranks], 120, capacity as u32, e.experts, e.topk));
        }
        if spark_ranks > 0 && !working.spark_layers.is_empty() {
            let widths = cuteafd_core::tp_units::package_width_keys(e.intermediate, 32, spark_ranks, false, true, false);
            notes.push(cuteafd_core::mxfp4_layouts::native_prefill_note(e.hidden,
                widths.into_iter().map(|(_, width)| width), 121, capacity as u32, e.experts, e.topk));
        }
    }
    update_sparks(catalog, &working, &mut sparks)?;
    charge_spark_devices(report, model, input, options, SPARK, &working,
        spark_endpoints("deepseek_v41", working.prefill_lanes), &mut sparks, &mut notes,
        |d| format!("Spark rank {} I/O admission", d.index));
    for spark in &mut sparks {
        for item in &mut spark.items {
            if item.group == "expert waves" || item.group == "vision CUDA overhead" { item.basis = Basis::Estimated; }
        }
    }
    let pool_tokens = placement.pool_tokens;
    for (device, items) in devices.iter_mut().zip(&placement.items) {
        device.items = items.clone();
        device.kv_tokens = pool_tokens;
    }
    devices.extend(sparks);
    Ok(MemoryLayout { devices, pool_tokens, waste, notes })
}

fn encoder_hardware(v41: bool, devices: &[DeviceLayout], sparks: &[DeviceLayout],
    options: &LayoutOptions) -> crate::plan::encoder::EncoderHardware {
    use crate::plan::encoder::{EncoderGpuBudget, EncoderHardware, EncoderSparkBudget};
    EncoderHardware { v41,
        gpus: devices.iter().map(|device| EncoderGpuBudget {
            free_bytes: device.capacity_bytes, kv_target_bytes: 0 }).collect(),
        sparks: sparks.iter().map(|device| EncoderSparkBudget { rank: device.index as usize,
            host: format!("spark{}", device.index), idle: false,
            expert_bytes: device.items.iter().filter(|item| item.category == Category::Experts)
                .map(|item| item.bytes).sum(),
            free_bytes: spark_encoder_headroom(device, options.spark_allocation_budget_bytes) }).collect() }
}

fn encoder_candidates(report: &PlanReport, options: &LayoutOptions, devices: &[DeviceLayout],
    sparks: &[DeviceLayout], weights: u64, scratch: u64) -> Vec<crate::plan::encoder::EncoderPlacement> {
    crate::plan::encoder::candidates(&crate::plan::encoder::EncoderRequest { mode: report.vision,
        weights, scratch, copies: options.vision_replicas },
        &encoder_hardware(true, devices, sparks, options))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_reserve_stays_separate_from_native_code_and_blas() {
        let owner = admission::graph_allowance(0).unwrap();
        assert_eq!(owner.bytes, 1_048_576_000);
        assert_eq!(owner.lifetime.first, crate::placement::planner::Phase::Ready);
        assert_eq!(owner.source, "graph reserve: v41.graphs.0");
    }

    /// The recipe's KV decomposition is the allocator's own total at every
    /// pool size, and the pool extent stays inside the indexable group domain.
    #[test]
    fn cache_owners_equal_the_allocator_total_at_every_pool_page() -> Result<()> {
        let config: serde_json::Value = serde_json::from_str(include_str!(
            "../../families/deepseek_v41/official-v41-config.json"))?;
        for ranks in [1usize, 2] {
            for tails in [0u64, 3, 20] {
                let maximum = 64;
                let owners = admission::cache_owners(&config, ranks, 1, tails, false, maximum)?;
                for pages in 1..=maximum {
                    let expected = crate::serving_capacity::deepseek_v41_cache_bytes(
                        &config, ranks, 1, pages + tails, false)?;
                    let charged = owners.iter().flatten()
                        .map(|component| component.bytes.at(pages).unwrap()).sum::<u64>();
                    assert_eq!(charged, expected.iter().sum::<u64>(), "ranks {ranks} tails {tails} pages {pages}");
                }
            }
        }
        assert!(admission::cache_owners(&config, 2, 1, 0, true, 64)?.len() == 2);
        Ok(())
    }

    /// The pool and the local prefix come from one solve: the resident-weight
    /// ceiling can only shrink the optional expert prefix, never the pool.
    #[test]
    fn resident_weight_ceiling_limits_the_optional_expert_prefix() -> Result<()> {
        let config: serde_json::Value = serde_json::from_str(include_str!(
            "../../families/deepseek_v41/official-v41-config.json"))?;
        let gib = 1u64 << 30;
        let expert = placement::ExpertCost { whole: placement::Bytes2 { resident: gib, staging: 8 * gib },
            half: [placement::Bytes2 { resident: gib / 2, staging: 4 * gib }; 2],
            tp2: true, whole_supported: true, spark_ok: true };
        let run = |limit: u64|
            -> std::result::Result<(usize, u64, [u64; 2]), crate::placement::planner::Refusal> {
            let cards = vec![(96 * gib, 96 * gib); 2];
            let budgets = admission::production_budgets(&cards, 0, None, 0, &[4 * gib; 4])?;
            let input = ProductionInput {
            blas: Vec::new(),
            expert_usage: false,
                config: &config, gpus: 2, context: 512, maximum_pool_tokens: 512 * 64,
                prefill: PrefillShape { rows: 64, waves: 1 }, concurrency: 1, retained_turns: 0,
                replicated: false, spark_ranks: 4, onboard_minimum: 0, onboard_counts: None,
                experts: vec![expert; 40], workspace: 0,
                owners: vec![Owner::charged(0, Category::Weights, "backbone", 2 * gib),
                    Owner::charged(1, Category::Weights, "backbone", 2 * gib)],
                weight_arenas: Vec::new(), startup: Vec::new(), allowances: Vec::new(), draft: None, encoders: Vec::new(),
                media_cache_bytes: 0, resident_weight_limit: limit,
                spark_reply_formats: Default::default(), operation: Operation::Serving, budgets,
            };
            let request = admission::production_request(512, Some(512 * 64),
                crate::placement::Onboard::Auto, PrefillShape { rows: 64, waves: 1 },
                Operation::Serving, limit, Default::default());
            let admitted = input.prepare()?.admit(request)?;
            let count = admitted.placement.onboard_layers;
            let (peak, _) = admission::arena(count, 2, &vec![expert; 40], 0)?;
            // TP2 names both GPUs but counts each layer once, so the arena peak
            // lives in the plan's TP2 range; a whole-layer home uses the range.
            let admitted_peak = admitted.placement.tp2.map_or_else(|| {
                assert_eq!(admitted.placement.expert_ranges[0].layers, count);
                [admitted.placement.expert_ranges[0].peak_bytes, 0]
            }, |range| {
                assert_eq!(range.layers, count);
                range.peak_bytes
            });
            assert_eq!(admitted_peak, peak, "the ceiling trades layers, never per-layer workspace");
            Ok((count, admitted.placement.pool_tokens, peak))
        };
        let (loose, pool, _) = run(40 * gib)?;
        let (tight, tight_pool, _) = run(10 * gib)?;
        let (none, none_pool, zero) = run(0)?;
        assert!(tight < loose && none < tight, "the weight ceiling shrinks the optional prefix");
        assert_eq!(none, 0);
        assert_eq!(zero, [0, 0]);
        assert_eq!((pool, tight_pool, none_pool), (pool, pool, pool),
            "the pool target does not move with the ceiling");
        // Compulsory resident weights above the ceiling are a typed refusal.
        let cards = vec![(96 * gib, 96 * gib); 2];
        let budgets = admission::production_budgets(&cards, 0, None, 0, &[4 * gib; 4])?;
        let input = ProductionInput {
            blas: Vec::new(),
            expert_usage: false,
            config: &config, gpus: 2, context: 512, maximum_pool_tokens: 512 * 64,
            prefill: PrefillShape { rows: 64, waves: 1 }, concurrency: 1, retained_turns: 0,
            replicated: false, spark_ranks: 4, onboard_minimum: 0, onboard_counts: None,
            experts: vec![expert; 40], workspace: 0,
            owners: vec![Owner::weights(0, Category::Weights, "backbone", 2 * gib)],
            weight_arenas: Vec::new(), startup: Vec::new(), allowances: Vec::new(), draft: None, encoders: Vec::new(),
            media_cache_bytes: 0, resident_weight_limit: 0, spark_reply_formats: Default::default(),
            operation: Operation::Serving, budgets,
        };
        let request = admission::production_request(512, Some(512 * 64),
            crate::placement::Onboard::Auto, PrefillShape { rows: 64, waves: 1 },
            Operation::Serving, 0, Default::default());
        assert!(matches!(input.prepare()?.admit(request),
            Err(crate::placement::planner::Refusal::ResidentWeightLimit { gpu: 0, .. })));
        Ok(())
    }

}

fn update_sparks(catalog: &crate::OfficialV41Catalog, working: &SparkWorkingSet, devices: &mut [DeviceLayout]) -> Result<()> {
    // Native MXFP4 is charged centrally by charge_spark_devices at each placement.
    if catalog.exl3().is_none() && catalog.nvfp4().is_none() { return Ok(()); }
    let shape = catalog.routed_experts();
    for device in devices {
        if !working.selected_ranks().contains(&(device.index as usize)) { continue; }
        let bytes = if let Some(exl3) = catalog.exl3() {
            working.spark_layers.iter().try_fold(0u64, |sum, &layer| -> Result<u64> {
                sum.checked_add(exl3.balanced_residency(crate::V41Exl3Layer::Backbone(layer), working.spark_ranks,
                    device.index as usize)?.device_arena_layout()?.1 as u64).context("V4.1 Spark EXL3 residency overflow")
            })?
        } else {
            let rank = device.index as usize;
            let layout = if catalog.nvfp4().is_some() {
                let width = shape.intermediate.div_ceil(working.spark_ranks).div_ceil(128) * 128;
                placement::SparkResidentLayout::Nvfp4 { world: working.spark_ranks, rank, stored_width: width }
            } else { placement::SparkResidentLayout::Native { world: working.spark_ranks, rank } };
            placement::SparkAssignment::resident_from_catalog(catalog, working, rank, layout)?.resident_bytes
        };
        if let Some(item) = device.items.iter_mut().find(|i| i.group == "routed_expert") { item.bytes = bytes; item.basis = Basis::Formula; }
    }
    Ok(())
}
