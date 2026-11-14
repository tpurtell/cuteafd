//! Muse Spark admission over the shared contract planner: header-exact
//! coordinator weights (K6 core payload + H128 vectors, BF16/F32 rest), paged
//! full-layer KV plus per-sequence sliding rings, pool first, then RTX expert
//! onboarding of whole routed layers; Spark ranks hold intermediate slices of
//! every expert and reply per-expert partials the coordinator combines.
//!
//! Coordinator code and workspaces come from the installed PROGRAMS.json.
use super::Model;
use anyhow::Context as _;
use crate::placement::families::muse_spark as shared;
use crate::placement::planner::{Operation, PrefillShape};
use crate::placement::{ArchContext, AttentionPlacement, Demand, Onboard, ProgramSet, ResidentWeights,
    SparkWorkingSet};
use crate::plan::{Component, ExpertPlacement, PlanReport};
use cuteafd_core::memory_layout::{Basis, Category, DeviceKind, DeviceLayout, Item, MemoryLayout};

const GIB: u64 = 1 << 30;
use crate::placement::families::muse_spark::LANES;

fn disable_media_component(report: &mut PlanReport, component: Component) {
    use crate::plan::Status;
    if let Some(entry) = report.components.iter_mut().find(|entry| entry.component == component)
        .filter(|entry| entry.status != Status::Unused && entry.status != Status::Disabled) {
        report.disabled_media_bytes += entry.bytes;
        entry.bytes = 0;
        entry.status = Status::Disabled;
    }
}

pub(super) fn layout(model: &Model, input: &crate::plan::PlanInput, report: &mut PlanReport,
    opts: &mut crate::plan::layout::LayoutOptions, requested: ExpertPlacement) -> MemoryLayout {
    if opts.host_embedding {
        report.refuse_placement("Muse Spark does not support --embedding-placement host".into(),
            "Use --embedding-placement gpu (EMBEDDING=gpu); the serving embedding is GPU-resident.");
        return MemoryLayout { devices: Vec::new(), pool_tokens: 0, waste: Vec::new(), notes: Vec::new() };
    }
    let limits = crate::plan::Family::prefill_limits(&super::MUSE_SPARK);
    let prefill = match crate::plan::layout::resolve_prefill(input, report, opts, requested.spark_ranks(), limits) {
        Ok(prefill) => prefill,
        Err(error) => return crate::plan::layout::unsupported_prefill(report, error),
    };
    let cfg = model.cfg();
    let ranks = requested.spark_ranks();
    let gpus = opts.rtx_bytes.len();
    let mut memory = MemoryLayout { devices: Vec::new(), pool_tokens: 0, waste: Vec::new(), notes: vec![
        "Muse Spark shared contract: exact physical owners and one admission".into(),
    ] };
    report.bytes_by_owner.clear();
    report.attention_placement = AttentionPlacement::Heads;
    let context = if opts.context_tokens == 0 { cfg.max_position as u64 } else { opts.context_tokens };
    if context > cfg.max_position as u64 {
        report.refuse_placement(format!("Muse Spark context {context} exceeds the checkpoint's {}", cfg.max_position),
            "Lower --max-context-tokens to at most 1M.");
        return memory;
    }
    if !model.complete {
        report.refuse_placement("Muse Spark placement cannot infer fit from missing safetensors headers".into(),
            "Place every shard's header locally (nest replicate) or plan from a complete snapshot.");
        return memory;
    }
    if !model.expert_formats.iter().any(|f| f.starts_with("exl3-k")) {
        report.refuse_placement(format!("Muse Spark routed experts are {:?}; only the EXL3 publication has an expert package",
            model.expert_formats), "Serve the EXL3 K4 + Core-K6 snapshot.");
        return memory;
    }
    if !(1..=2).contains(&gpus) || (gpus == 2 && !opts.head_split) {
        report.refuse_placement("Muse Spark runs on one RTX or a head-split RTX pair".into(), "Use --rtx 1 or 2.");
        return memory;
    }
    // One request builder for planning and serving startup (plan == runtime).
    let slots = opts.resolved_state_slots("muse_spark");
    let catalog = match input.expert_catalog() {
        Ok(catalog) => catalog,
        Err(error) => {
            report.refuse_placement(format!("Muse Spark expert inventory refused: {error:#}"), "Provide complete EXL3 expert headers.");
            return memory;
        }
    };
    // GPU marks follow the shared prefix rule: host-tier retention keeps only staging slots.
    let mark_slots = opts.prefix_slots.unwrap_or_else(|| cuteafd_core::prefix::mark_slots_with_host(slots,
        opts.mimo_prefix_entries, crate::placement::families::muse_spark::mark_bytes(cfg, 1),
        opts.mimo_prefix_mark_bytes, opts.host_cache_bytes != Some(0)));
    if opts.host_cache_bytes == Some(0) {
        memory.notes.push(format!("Muse host prefix tier off: retains {mark_slots} exact prefix marks; \
            {} B/mark across {gpus} GPU(s), device mark budget {} B{}",
            crate::placement::families::muse_spark::mark_bytes(cfg, 1), opts.mimo_prefix_mark_bytes,
            if opts.prefix_slots.is_some() { " (explicit slot override)" } else { "" }));
    }
    let onboard = opts.resolve_onboard(catalog.routed_experts().routed_layers()).unwrap_or(Onboard::Auto);
    let manifest_path = opts.workspace_manifest.as_deref().or_else(|| {
        let path = std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json");
        path.is_file().then_some(path) });
    let cards = opts.rtx_bytes.iter().map(|&bytes| crate::placement::CoordinatorCard::planned(bytes, opts.physical_sms))
        .collect::<Vec<_>>();
    if opts.physical_sms.is_none() { memory.notes.push("Physical GPU class assumed SM120 PRO; logical caps do not identify a card".into()); }
    let local = match shared::local_workspace(catalog, manifest_path, gpus, prefill.unit_rows) {
        Ok(workspace) => workspace,
        Err(error) => {
            report.refuse_placement(format!("Muse coordinator EXL3 package refused: {error:#}"), "Install the matching package.");
            return memory;
        }
    };
    let drafter = match opts.muse_drafter_snapshot.as_deref().map(|snapshot| -> anyhow::Result<_> {
        let config: serde_json::Value = serde_json::from_slice(&std::fs::read(snapshot.join("config.json"))?)?;
        let head = model.core.head_outputs(&cfg);
        let sms = opts.physical_sms.unwrap_or(ArchContext::for_device("sm_120", cards[gpus - 1].physical_bytes).sms);
        let drafter = shared::drafter_inventory(cfg, &config, usize::try_from(slots)?,
            usize::try_from(slots)?, head, u64::from(sms), opts.draft_kv_cache)?;
        if opts.tp2_dflash { drafter.with_tp2(snapshot, &vec![u64::from(sms); gpus], opts.draft_kv_cache) }
        else { Ok(drafter) }
    }).transpose() {
        Ok(drafter) => drafter,
        Err(error) => {
            report.refuse_placement(format!("Muse DFlash2 drafter refused: {error:#}"),
                "Use a drafter trained for this target (nvidia/Muse-Spark-Open-816B-A42B-DFlash2) or drop --draft.");
            return memory;
        }
    };
    let projections = match (|| -> anyhow::Result<_> {
        let manifest = if model.core.native.is_empty() { None } else {
            manifest_path.map(|path| input.programs_manifest(path)).transpose()?
        };
        crate::families::muse_spark::weight_layout::projections_with_manifest(cfg, &model.core, gpus,
            prefill.unit_rows as usize, manifest.as_deref())
    })() {
        Ok(projections) => projections,
        Err(error) => {
            report.refuse_placement(format!("Muse native projection exports refused: {error:#}"),
                "Supply the matching native attention exports in PROGRAMS.json.");
            return memory;
        }
    };
    let fixed = match planned_code(input, opts, &cards, &projections) {
        Ok(fixed) => fixed,
        Err(error) => {
            report.refuse_placement(format!("Muse Spark program manifest refused: {error:#}"), "Supply the image's PROGRAMS.json.");
            return memory;
        }
    };
    let media = match shared::MediaOptions::from_checkpoint(input.checkpoint(), report.vision, report.audio,
        opts.vision_replicas) {
        Ok(media) => media,
        Err(error) => {
            for component in [Component::Vision, Component::Audio] {
                let encoder = crate::plan::encoder::EncoderPlacement {
                    kind: crate::plan::encoder::EncoderKind::Off, weights: 0, scratch: 0, replicas: Vec::new(),
                    reason: format!("Muse bundled media inventory refused: {error:#}"), shortfall: 0,
                };
                if component == Component::Vision { report.encoder = Some(encoder); }
                else { report.audio_encoder = Some(encoder); }
            }
            report.refuse_placement(format!("Muse bundled media inventory refused: {error:#}"),
                "Use the bundled BF16 towers, a fitting encoder placement, or explicitly disable media.");
            return memory;
        }
    };
    // The measured edge differs only in these baselines; everything else the
    // contract charges is the same fact list.
    let spark_budget_bytes = opts.spark_allocation_budget_bytes
        .unwrap_or(crate::plan::DEFAULT_SPARK_BUDGET_BYTES).min(opts.spark_bytes);
    let mut budgets = cards.iter().enumerate().map(|(gpu, card)| shared::planned_budget(gpu as u8,
        card.physical_bytes, card.logical_bytes, crate::placement::reserve::gpu("muse_spark"),
        ArchContext::for_device("sm_120", card.physical_bytes).context_bytes)).collect::<Vec<_>>();
    budgets.extend((0..ranks).map(|rank| shared::spark_budget(rank as u8, spark_budget_bytes)));
    let maximum_pool_tokens = crate::placement::CoordinatorCard::maximum_pool(&cards, context, opts.pool_tokens, shared::PAGE_ROWS);
    let (total, available) = match opts.host_memory.map(Ok).unwrap_or_else(crate::formats::mapped_preload::host_memory) {
        Ok(memory) => memory,
        Err(error) => { report.refuse_placement(format!("Muse host inventory refused: {error:#}"), "Provide a host memory sample."); return memory; }
    };
    let host = shared::HostInput { total, available, reserve: 8 << 30,
        prefix_bytes: opts.host_cache_bytes, prefix_entries: opts.mimo_prefix_entries,
        media_bytes: opts.media_cache_bytes, device_sampling: true };
    // Even a typed placement refusal carries the requested encoder facts.
    for (component, mode, bytes) in [(Component::Vision, report.vision, media.vision_bytes),
        (Component::Audio, report.audio, media.audio_bytes)] {
        let encoder = crate::plan::encoder::EncoderPlacement { kind: crate::plan::encoder::EncoderKind::Off,
            weights: bytes.0, scratch: bytes.1, replicas: vec![], shortfall: 0,
            reason: format!("{mode:?} tower not admitted: joint memory contract refused") };
        if component == Component::Vision { report.encoder = Some(encoder); } else { report.audio_encoder = Some(encoder); }
    }
    let facts = shared::ProductionInput { cfg, core: &model.core, projections, catalog, budgets, host, prefill,
         bf16_target_head: opts.bf16_target_head, slots, mark_slots, context, maximum_pool_tokens,
        partial_bf16_max_rows: opts.muse_partial_bf16_max_rows, partial_reply_formats: opts.spark_reply_formats,
        local, drafter, fixed, media };
    let prepared = match facts.prepare() {
        Ok(prepared) => prepared,
        Err(error) => {
            report.refuse_placement(format!("Muse Spark joint admission refused: {error}"),
                "Use more Spark ranks, automatic RTX expert selection, or a larger --spark-memory-cap.");
            return memory;
        }
    };
    if let Some((vision, audio)) = prepared.recipe.encoders.first() {
        report.encoder = Some(vision.clone());
        report.audio_encoder = Some(audio.clone());
    }
    let request = shared::production_request(crate::placement::planner::Context::resolve(context, (opts.context_tokens > 0).then_some(opts.context_tokens)), opts.pool_tokens, onboard,
        PrefillShape { rows: prefill.unit_rows, waves: prefill.waves },
        Operation::Serving,
        Some(ResidentWeights(opts.weight_budget_bytes)), opts.spark_reply_formats);
    let admitted = match prepared.admit(request) {
        Ok(admitted) => admitted,
        Err(error) => {
            report.refuse_placement(format!("Muse Spark joint admission refused: {error}"),
                "Use more Spark ranks, automatic RTX expert selection, or a larger --spark-memory-cap.");
            return memory;
        }
    };
    report.phase_ledger = Some(crate::plan::SolvedPhaseLedger::from(&admitted.solved));
    report.host_budget_bytes = admitted.host.prefix_bytes;
    let mut host_items = admitted.solved.items(crate::placement::planner::Location::HOST).unwrap_or_default();
    host_items.extend(admitted.solved.items(crate::placement::planner::Location::PINNED).unwrap_or_default());
    host_items.retain(|item| item.category != Category::Prefix);
    for item in &mut host_items { item.lifetime = cuteafd_core::memory_layout::ItemLifetime::PostReadyReserve; }
    report.host_ledger = cuteafd_core::memory_layout::HostLayout::new(total, available,
        host.reserve.max(total / 10), admitted.host.prefix_bytes / admitted.host.prefix_chunk.max(1) * admitted.host.prefix_chunk,
        host_items);
    memory.notes.push(format!("Muse shared solve {} us; host ordinary residual {} B (estimated, #1326)",
        admitted.solve_us, shared::HOST_RUNTIME.bytes));
    report.encoder = Some(admitted.media.vision.clone());
    report.audio_encoder = Some(admitted.media.audio.clone());
    if admitted.media.vision.kind == crate::plan::encoder::EncoderKind::Off {
        disable_media_component(report, Component::Vision);
    }
    if admitted.media.audio.kind == crate::plan::encoder::EncoderKind::Off {
        disable_media_component(report, Component::Audio);
        report.audio = crate::plan::MediaMode::Off;
    }
    let embedding = admitted.global.embedding as u64;
    let head = admitted.global.head_bytes() as u64;
    let experts = *model.bytes.get(&Component::RoutedExpert).unwrap_or(&0);
    let layer_bytes = (0..gpus).map(|gpu| admitted.layer_bytes(gpu).unwrap_or(0)).sum::<u64>();
    report.bytes_by_owner.insert("coordinator text weights".into(), embedding + head
        + admitted.global.final_norm as u64 * gpus as u64 + layer_bytes);
    report.bytes_by_owner.insert("routed expert weights".into(), experts);
    memory.notes.push(format!("Muse K6 core: {} header-native projections ({}); payload + H128 vectors charged",
        model.core.projections.len(), if model.core.complete(cfg) { "complete" } else { "partial: BF16 rest" }));
    let placed = admitted.placement;
    let rings = crate::placement::families::muse_spark::ring_rows(cfg).unwrap_or(0)
        * cfg.kv_record_bytes() as u64 * cfg.sliding_layers() as u64;
    memory.pool_tokens = placed.pool_tokens;
    memory.notes.push(format!("Muse selected RTX expert layers: {}; pool {} tokens; full-layer KV {} B/token over {} \
        layers; sliding rings {} B/sequence over {} layers", placed.onboard_layers, placed.pool_tokens,
        cfg.kv_record_bytes() * cfg.full_layers(), cfg.full_layers(), rings, cfg.sliding_layers()));
    for gpu in 0..gpus {
        let mut device = DeviceLayout { kind: DeviceKind::Rtx, index: gpu as u32, capacity_bytes: cards[gpu].physical_bytes.min(cards[gpu].logical_bytes), worker_budget_bytes: 0,
            items: placed.items[gpu].clone(), kv_tokens: placed.pool_tokens };
        device.items.extend(crate::placement::reserve::gpu_item("muse_spark"));
        memory.devices.push(device);
    }
    let onboarded = placed.onboard_layers;
    report.solver_placement = Some(placed);
    let working = SparkWorkingSet { rtx_layers: Vec::new(),
        spark_layers: admitted.spark_budget.layers[onboarded..].to_vec(), spark_ranks: ranks, prefill_lanes: LANES };
    let rings = crate::plan::layout::spark_ring_geometry(LANES as usize);
    // serve-muse's cutoff: the same overrides, the same widest verify step.
    let cutoff = match crate::families::muse_spark::replies::Cutoff::with_formats(opts.muse_partial_bf16_max_rows,
        crate::families::muse_spark::replies::DEFAULT_BF16_MAX_ROWS, opts.spark_reply_formats) {
        Ok(cutoff) => cutoff,
        Err(error) => {
            report.refuse_placement(format!("Muse partial-reply cutoff refused: {error:#}"), "Use a cutoff of at least 64 rows.");
            return memory;
        }
    };
    let mut widest = 0;
    for rank in 0..ranks {
        let resident = admitted.spark_budget.resident[rank][onboarded..].iter().sum::<u64>();
        widest = widest.max(resident);
        let mut device = DeviceLayout { kind: DeviceKind::Spark, index: rank as u32, capacity_bytes: spark_budget_bytes, worker_budget_bytes: spark_budget_bytes,
            kv_tokens: 0, items: vec![Item::new(Category::Experts, "routed experts (intermediate slices)",
                model.expert_formats.join("+"), resident, Basis::Exact)] };
        if let Err(error) = usize::try_from(slots).map_err(anyhow::Error::from)
            .and_then(|slots| muse_spark_io(&mut device, &working, prefill, slots, rings, cutoff)) {
            report.refuse_placement(format!("Muse Spark rank {rank} I/O: {error:#}"), "Fix the Spark ring geometry.");
        }
        if let Err(error) = muse_owners(&mut device, &working, prefill.unit_rows, prefill.waves) {
            report.refuse_placement(format!("Muse Spark execution owners: {error:#}"), "Use a supported wave count.");
        }
        device.items.extend(admitted.media.spark_items[rank].iter().cloned());
        memory.devices.push(device);
    }
    report.spark_rank_share = if experts == 0 { 0.0 } else { widest as f64 / experts as f64 };
    memory.notes.push(format!("Spark: intermediate slices over {ranks} ranks, widest rank {:.1} GiB of {:.1} GiB routed",
        widest as f64 / GIB as f64, experts as f64 / GIB as f64));
    memory
}

/// Loaded code charged exactly as live startup charges it (the full allowance
/// offline, nothing yet resident), once the image's program manifest is known.
fn planned_code(input: &crate::plan::PlanInput, opts: &crate::plan::layout::LayoutOptions,
    cards: &[crate::placement::CoordinatorCard],
    projections: &[crate::families::muse_spark::weight_layout::RankProjections]) -> anyhow::Result<Vec<Demand>> {
    let gpus = cards.len();
    let path = opts.workspace_manifest.as_deref().or_else(|| {
        let path = std::path::Path::new("/opt/cuteafd/share/PROGRAMS.json");
        path.is_file().then_some(path) });
    let Some(path) = path else { return Ok(Vec::new()) };
    let manifest = input.manifest(path)?;
    let programs = ProgramSet::from_manifest(&manifest, crate::placement::families::muse_spark::program_families(gpus));
    (0..gpus).map(|rank| {
        let mut programs = programs.clone();
        projections[rank].include_native_programs(&mut programs);
        let arch = ArchContext::for_device("sm_120", cards[rank].physical_bytes);
        let code = crate::placement::inventory::loaded_code_allowance("muse", "exl3", gpus == 2, rank as u8,
            arch, Some(&programs));
        crate::placement::families::muse_spark::code_demand(rank, code.bytes)
    }).collect()
}

/// The rank's Spark I/O at the shared Muse endpoint (`spark_endpoint`): its
/// partial-reply frame and in-flight depth, exactly as admission seals it.
fn muse_spark_io(device: &mut DeviceLayout, working: &SparkWorkingSet,
    prefill: crate::serving_capacity::prefill_waves::PrefillPlan, slots: usize,
    rings: crate::placement::SparkRingGeometry, cutoff: crate::families::muse_spark::replies::Cutoff)
    -> anyhow::Result<()> {
    let geometry = cuteafd_core::ExpertGeometry::MUSE_SPARK;
    crate::plan::layout::charge_spark_io(device, working, geometry, prefill.unit_rows as u32, false, rings, 0)?;
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
    crate::plan::layout::spark_slots::charge_owners(device, &spark_executions(waves, unit_rows)?,
        |extent| owner_bytes(working.spark_ranks, extent))
}
