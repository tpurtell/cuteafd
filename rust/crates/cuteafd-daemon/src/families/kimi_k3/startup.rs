//! Shared native startup for serving and diagnostics. Admission precedes all
//! retained operands and ordinary admitted-rank transport connection.
use super::{admission, device::NativePrograms, engine::Engine};
use crate::shared::{inventory::{InventoryPhase, RuntimeInventory}, memory::device::{Device, Stream},
    peer_split::{hop::HopLink, PeerExchange, RankDevice}};
use anyhow::{ensure, Context, Result};
use cuteafd_core::ExpertGeometry;
#[cfg(test)]
use cuteafd_core::memory_layout::Basis;
use cuteafd_loader::{families::kimi_k3::config::KimiK3Config, placement::{Demand, GpuBudget}};
use std::{net::SocketAddr, path::Path, rc::Rc, sync::Arc};
const HIDDEN: usize = 7168;

const MEMORY_PHASE_DIAGNOSTIC: &str = "CUTEAFD_DIAG_KIMI_STARTUP_MEMORY";

fn memory_phase_mode(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("off") => Ok(false),
        Some("pre-capture") => Ok(true),
        _ => anyhow::bail!("{MEMORY_PHASE_DIAGNOSTIC} must be off or pre-capture"),
    }
}

pub(crate) fn memory_phases<'a>(library: &'a cuteafd_ffi::NativeLibrary, devices: &[i32])
    -> Result<crate::shared::inventory::StartupMemoryPhases<'a>> {
    let mode = std::env::var(MEMORY_PHASE_DIAGNOSTIC).ok();
    Ok(crate::shared::inventory::StartupMemoryPhases::new(library, devices, memory_phase_mode(mode.as_deref())?))
}

fn stop_before_capture(enabled: bool) -> Result<()> {
    ensure!(!enabled, "KIMI_STARTUP_MEMORY_PRE_CAPTURE_STOP: diagnostic complete; target graphs and Ready were not reached");
    Ok(())
}

pub(crate) struct Options<'a, const RANKS: usize = 2> {
    pub expert_usage: bool,
    pub expert_usage_reader: cuteafd_core::expert_usage::Reader,
    pub cache_format: cuteafd_core::MlaCacheFormat,
    pub devices: [i32; RANKS],
    pub attention: Option<cuteafd_loader::placement::AttentionPlacement>,
    pub native_lib: &'a Path,
    pub manifest: &'a Path,
    pub peers: Vec<SocketAddr>,
    pub rtx_expert_layers: cuteafd_loader::placement::OnboardSetting,
    /// Lane rows: `waves` prefill units of `rows / waves` each.
    pub rows: usize,
    /// Prefill units in flight (Spark waves outstanding, one return slot each).
    pub waves: usize,
    /// Explicit unit rows; unset takes `rows / waves`.
    pub prefill_rows: Option<usize>,
    pub slots: usize,
    pub marks: usize,
    pub max_context: usize,
    pub context: cuteafd_loader::placement::planner::Context,
    pub pool_tokens: Option<u64>,
    pub graphs: bool,
    pub full_prefill_logits: bool,
    pub ffn_mode: cuteafd_ffi::KimiFfnMode,
    pub hosted_layers: Option<Vec<usize>>,
    pub draft: Option<super::draft::Options>,
    /// Exact speculative verify widths given startup graphs; 0 = the drafter's block.
    pub verify_rows: usize,
    pub prefix: Option<&'a crate::shared::prefix::PrefixArgs>,
}

pub(crate) struct Loaded<'a> {
    pub engine: Engine<'a>,
    pub admission_free: Vec<u64>,
    pub admitted_bytes: Vec<u64>,
    pub ownership: Ownership,
}

#[derive(Debug, serde::Serialize)]
pub(crate) struct Ownership {
    pub rtx_expert_layers: usize,
    pub spark_layers: Vec<usize>,
    pub inactive_seeded_layers: usize,
    pub inactive_loaded_layers: usize,
}

pub(super) fn check_hosted_layers(setting: cuteafd_loader::placement::OnboardSetting,
    selected: &[usize], seeded: &[usize]) -> Result<usize> {
    ensure!(seeded.windows(2).all(|pair| pair[0] < pair[1]),
        "Kimi worker layer subset must be sorted and unique");
    if setting == cuteafd_loader::placement::OnboardSetting::Auto {
        ensure!(selected.iter().all(|layer| seeded.binary_search(layer).is_ok()),
            "Kimi worker layer subset is missing shared placement layers");
    } else {
        ensure!(selected == seeded, "Kimi worker layer subset differs from shared placement");
    }
    Ok(seeded.len() - selected.len())
}

/// Run before loading native programs, whose card selection can query the GPU.
pub(crate) fn preflight(manifest: &Path, native_lib: &Path) -> Result<()> {
    let package = crate::shared::experts::exl3::aot_layout_directory(native_lib, &[2, 3], "");
    cuteafd_loader::families::kimi_k3::validate_expert_inventory(manifest, &package)
}

pub(crate) fn load<'a, const RANKS: usize>(library: &'a Arc<cuteafd_ffi::NativeLibrary>,
    programs: &'a cuteafd_ffi::programs::Programs<'a>,
    checkpoint: &cuteafd_loader::plan::Checkpoint, full: &KimiK3Config,
    cfg: &KimiK3Config, mut options: Options<'_, RANKS>) -> Result<Loaded<'a>> {
    ensure!(matches!(RANKS, 1 | 2), "Kimi startup requires one or two physical devices");
    let mut memory = memory_phases(library, &options.devices)?;
    memory.sample("context-before-family-modules")?;
    if RANKS == 2 { options.attention = Some(cuteafd_loader::families::kimi_k3::resolve_attention(options.attention, RANKS)); }
    check_attention(options.attention, RANKS, options.graphs)?;
    ensure!(options.draft.as_ref().is_none_or(|draft| draft.cache_format == options.cache_format),
        "Kimi target/draft cache records differ");
    let spark_ranks = options.peers.len();
    if spark_ranks == 0 {
        return Err(cuteafd_loader::placement::families::kimi::LOCAL_ONLY_REFUSAL.into());
    }
    ensure!(spark_ranks <= cuteafd_core::MAX_SPARK_WORLD, "Kimi Spark peer inventory exceeds eight");
    let core = cuteafd_loader::plan::families::kimi::core_inventory(checkpoint)?;
    core.validate_runtime_groups(full)?;
    if RANKS == 2 {
        ensure!(crate::shared::peer_split::probed_device(library, options.devices[0], Some(options.devices[1]))?
            == Some(options.devices[1]), "Kimi TP2 requires bidirectional peer access");
    }
    ensure!(options.devices.iter().all(|device| *device >= 0), "Kimi CUDA device ordinal must be nonnegative");
    let devices = options.devices.map(|id| Device { library, id });
    let head = RANKS - 1;
    let rows = options.rows;
    // Lane rows hold `waves` units; every kernel launch, hook-local scratch,
    // peer exchange, hop and Spark wave serves one unit.
    let unit = cuteafd_loader::families::kimi_k3::allocation::prefill_shape_with(rows, options.waves,
        options.prefill_rows)?.unit_rows as usize;
    let native_exports = devices[0].run(|| super::device::resolve_native_programs(programs, &core, unit, RANKS))?;
    let native = devices.iter().map(|&device| NativePrograms::load_resolved(device, full, programs, options.ffn_mode,
        options.cache_format, &core, unit, RANKS, &native_exports).map(Rc::new)).collect::<Result<Vec<_>>>()?;
    let native: [_; RANKS] = native.try_into().map_err(|_| anyhow::anyhow!("Kimi native physical rank inventory"))?;
    memory.sample("modules-target")?;
    ensure!(RANKS == 2 || !options.draft.as_ref().is_some_and(|draft| draft.tp2_dflash),
        "Kimi TP2 DFlash requires two physical coordinator GPUs");
    let draft_layout = options.draft.as_ref().map(|draft| devices[head].run(||
        draft.layout(options.slots, options.max_context, library.sm_count()?))).transpose()?;
    // Resolve native modules before the measured baseline, while retaining their handles.
    let _draft_modules = options.draft.as_ref().map(|_| devices[head].run(|| {
        use cuteafd_ffi::mla_draft::{MlaDraft, MlaDraftWindow};
        Ok([MlaDraft::load(programs, "kimik3", MlaDraftWindow::Full, options.cache_format)?,
            MlaDraft::load(programs, "kimik3", MlaDraftWindow::Sliding4096, options.cache_format)?])
    })).transpose()?;
    memory.sample("modules-draft-mla")?;
    let _draft_context_modules = if draft_layout.as_ref().is_some_and(|layout| layout.context_split()) {
        use cuteafd_ffi::mla_draft::{MlaDraftPartial, MlaDraftCombine, MlaDraftWindow};
        let partials = devices.iter().map(|device| device.run(|| Ok([
            MlaDraftPartial::load(programs, "kimik3", MlaDraftWindow::Full, options.cache_format)?,
            MlaDraftPartial::load(programs, "kimik3", MlaDraftWindow::Sliding4096, options.cache_format)?])))
            .collect::<Result<Vec<_>>>()?;
        Some((partials, devices[head].run(|| Ok(MlaDraftCombine::load(programs, "kimik3")?))?))
    } else { None };
    memory.sample("modules-draft-context")?;
    // The per-Spark cap (`SPARK_MEMORY_CAP`), else Kimi's estimated complete ceiling.
    let spark_budget_bytes = cuteafd_core::memory_policy::MemoryPolicy::current()
        .spark_cap_bytes(cuteafd_loader::families::kimi_k3::SPARK_DEFAULT_BUDGET_BYTES);
    let catalog = cuteafd_loader::read_expert_catalog_view(checkpoint.view.as_ref()
        .context("Kimi startup requires the immutable selected checkpoint view")?)?;
    let routed = cfg.dense.iter().filter(|dense| !**dense).count();
    let onboard = options.rtx_expert_layers.resolve(routed);
    let model = crate::shared::placement::spark_rings(cuteafd_transport::TcpTransportConfig::default().max_frame_bytes)?;
    let working = cuteafd_loader::placement::SparkWorkingSet { rtx_layers: vec![],
        spark_layers: cfg.dense.iter().enumerate().filter_map(|(layer, dense)| (!dense).then_some(layer)).collect(),
        spark_ranks, prefill_lanes: 1 };
    let spark_budget = cuteafd_loader::placement::SparkBudgetInventory::exl3(&catalog, &working,
        &[cuteafd_core::SparkEndpoint { ring_depth: None, capacity: u32::try_from(unit)?, ingress: cuteafd_core::SparkIngress::Fp8K32, partial_reply_bytes: None }],
        model, spark_budget_bytes,
        cuteafd_loader::families::kimi_k3::spark::reserve_bytes(spark_ranks, options.waves as u64, unit as u64)?)?;
    let rtx = if onboard.layers(routed) == Some(0) { None } else {
        let tiers = catalog.exl3().context("Kimi TP2 experts require EXL3 checkpoint")?.decoder_tiers();
        let package = crate::shared::experts::exl3::aot_layout_directory(
            options.native_lib, tiers, &format!("rtx-tp{RANKS}"));
        let mut inventory = admission::plan_rtx_for_ranks(cfg, &catalog, package, unit, routed, RANKS)?;
        inventory.layers = onboard;
        Some(inventory)
    };
    let expert_modules = devices.iter().map(|&device| rtx.as_ref().map(|inventory| {
        use crate::shared::experts::rtx::{exl3::Exl3Tp2, RtxSelectionPolicy};
        if RANKS == 1 { Exl3Tp2::preload_whole_modules(device, &inventory.package, unit,
            ExpertGeometry::KIMI_K3, RtxSelectionPolicy::FirstCapacity) }
        else { Exl3Tp2::preload_tp2_modules(device, &inventory.package, unit,
            ExpertGeometry::KIMI_K3, RtxSelectionPolicy::FirstCapacity) }
    }).transpose()).collect::<Result<Vec<_>>>()?;
    let expert_modules: [_; RANKS] = expert_modules.try_into()
        .map_err(|_| anyhow::anyhow!("Kimi expert module physical rank inventory"))?;
    memory.sample("modules-rtx-experts")?;
    let context_programs = devices.iter().map(|device|
        (options.attention == Some(cuteafd_loader::placement::AttentionPlacement::Context))
            .then(|| device.run(|| super::context::Programs::load(programs, options.cache_format).map(Rc::new)))
            .transpose()).collect::<Result<Vec<_>>>()?;
    let context_programs = context_programs.try_into()
        .map_err(|_| anyhow::anyhow!("Kimi context program rank inventory"))?;
    memory.sample("modules-context-attention")?;
    let diagnostics = std::env::var("CUTEAFD_PEER_WAIT_DIAGNOSTICS").is_ok_and(|v| v == "1");
    let inventory = RuntimeInventory::measure(library, &options.devices, InventoryPhase::PreWeight, |_| {
        if RANKS == 2 {
            library.peer_exchange_initialize()?;
            library.peer_abort_initialize()?;
            if diagnostics { library.peer_diagnostic_initialize()?; }
        }
        library.v41_compact_reducer()?.require_plane_reduction(spark_ranks as u32)?;
        Ok(())
    })?;
    memory.sample("modules-shared-peer-reducer")?;
    let lane_layouts: [_; RANKS] = native.iter().enumerate().map(|(rank, native)|
        native.lane_layout(full, rank, rows, unit, options.max_context)).collect::<Result<Vec<_>>>()?
        .try_into().map_err(|_| anyhow::anyhow!("Kimi lane physical rank inventory"))?;
    let context_layout = (options.attention == Some(cuteafd_loader::placement::AttentionPlacement::Context))
        .then(|| super::context::Programs::layout(programs, options.cache_format,
            options.max_context, unit, lane_layouts[0].metadata_banks())).transpose()?;
    let lane_layouts = vec![lane_layouts];
    let verify_rows = match options.verify_rows {
        0 => options.draft.as_ref().map(super::draft::Options::verify_rows).transpose()?.unwrap_or(0),
        rows => rows,
    };
    let graph_set = options.graphs.then(|| if let Some(layout) = &context_layout {
        super::graphs::inventory_with_context(cfg, 1, unit, options.slots, verify_rows, RANKS,
            |_, _| true, |layer| cfg.attention[layer] == cuteafd_loader::families::kimi_k3::config::KimiAttention::Mla,
            Some(layout.decode()))
    } else if RANKS == 1 || options.attention == Some(cuteafd_loader::placement::AttentionPlacement::Heads) {
        super::graphs::inventory_for_ranks(cfg, 1, unit, options.slots, verify_rows, RANKS, |_, _| true)
    } else { super::graphs::owned_inventory(cfg, 1, unit, options.slots, verify_rows) }).transpose()?;
    // One configured arena per physical device: admission owns it, and the
    // startup capture boundary measures inside it.
    let reserve_demands = if graph_set.is_some() {
        cuteafd_loader::families::kimi_k3::allocation::graph_reserve_demands(RANKS)?
    } else { Vec::new() };
    let reserves: [u64; RANKS] = std::array::from_fn(|rank|
        reserve_demands.get(rank).map_or(0, |demand| demand.bytes));
    let mut fixed = if RANKS == 2 { exchange_demands(unit, diagnostics)? } else { vec![] };
    if core.native_attention_split(RANKS) {
        for rank in 0..RANKS {
            fixed.push(core.native_attention_receive_demand(unit, rank as u8, RANKS)?);
        }
    }
    if let Some(layout) = &draft_layout {
        fixed.extend(cuteafd_loader::families::kimi_k3::draft_bridge_demands_for_ranks(layout, rows, RANKS)?);
        fixed.push(cuteafd_loader::families::kimi_k3::draft_probe_demand(cfg.layers, cfg.topk, rows.min(64))?);
        if core.projections.contains_key("language_model.lm_head") {
            if let Some(mut demand) = cuteafd_loader::families::kimi_k3::core::draft_head_demand(&core,
                layout.capacity().block_rows, programs.family_metadata("dense_trellis_k6")?)? {
                demand.gpu = head as u8;
                fixed.push(demand);
            }
        }
    }
    let mut selected = cuteafd_loader::placement::ProgramSet::from_names(
        if core.projections.is_empty() { &["kimik3"][..] } else { &["kimik3", "kimi"][..] }, programs.names());
    selected.names.extend(native_exports.programs.iter().cloned());
    selected.names.sort();
    selected.names.dedup();
    for (rank, sample) in inventory.gpus.iter().enumerate() {
        let pending = crate::shared::inventory::pending_code(library, sample.device, rank, RANKS == 2,
            "kimik3", "exl3", &selected)?;
        fixed.push(cuteafd_loader::families::kimi_k3::allocation::code_demand(rank, pending, RANKS)?);
    }
    let gpus = inventory.gpus.iter().map(|sample| {
        GpuBudget { capacity_bytes: sample.total_bytes, reserve_bytes: cuteafd_loader::placement::reserve::gpu("kimi_k3"),
            baseline: cuteafd_loader::placement::Baseline::Measured { free_bytes: sample.free_bytes } }
    }).collect::<Vec<_>>().try_into().map_err(|_| anyhow::anyhow!("Kimi GPU sample count"))?;
    let peer_receive = [core.attention_receive_bytes(unit, RANKS)?,
        core.attention_receive_bytes(unit.min(64), RANKS)?];
    let mut admitted = admission::admit(admission::Inputs { expert_usage: options.expert_usage_reader.configure(options.expert_usage, false).map_err(anyhow::Error::msg)?, cfg, core,
        resident_weight_limit: crate::shared::memory::budget::resident_weight_limit(), cache_format: options.cache_format, gpus, devices: options.devices,
        peer_access: RANKS == 2, spark_ranks, attention: options.attention, context: context_layout, serving_context: options.context, requested_pool: options.pool_tokens,
        state_slots: options.slots, prefix_marks: options.marks, rows, lanes: 1, waves: options.waves,
        unit_rows: options.prefill_rows, full_prefill_logits: options.full_prefill_logits, lane_layouts, fixed,
        startup: reserve_demands, pool_bounded: vec![], rtx,
        spark_budget: Some(spark_budget), draft: draft_layout.clone(), verify_rows, prefix: options.prefix })?;
    options.max_context = usize::try_from(options.context.served(admitted.placement.pool_tokens))?;
    {
        use sha2::{Digest, Sha256};
        let hash = |path: &Path| -> Result<String> {
            let mut file = std::fs::File::open(path)?;
            let mut digest = Sha256::new();
            std::io::copy(&mut file, &mut digest)?;
            Ok(format!("{:x}", digest.finalize()))
        };
        cuteafd_ffi::memory_ledger::fault::set_inputs(serde_json::json!({
            "memory_reconciliation_provenance": {
                "schema_version": 1, "family": "kimi_k3",
                "source_commit": env!("CUTEAFD_BUILD_COMMIT"),
                "source_dirty": env!("CUTEAFD_BUILD_DIRTY"), "toolchain": env!("CUTEAFD_BUILD_TOOLCHAIN"),
                "native_sha256": hash(options.native_lib)?, "programs_sha256": hash(options.manifest)?,
                "layout": format!("{RANKS} RTX/{spark_ranks} Sparks; attention/{:?}; context/{}; slots/{}",
                    options.attention, options.max_context, options.slots),
                "devices": inventory.gpus, "spark_world": spark_ranks,
                "onboard_layers": admitted.placement.onboard_layers,
                "graph_basis": "uncalibrated Kimi byte fallback"
            }
        }));
    }
    memory.sample("admission-resolved")?;
    let graph_set = admitted.graph_set.take();
    let spark_layers = admitted.working_set().spark_layers;
    let inactive_seeded_layers = options.hosted_layers.as_ref().map(|expected|
        check_hosted_layers(options.rtx_expert_layers, &spark_layers, expected)).transpose()?.unwrap_or(0);
    // Golden's seed list is only a coverage hint. WorkerSelection loads exactly
    // the admitted working set, never the inactive hinted layers.
    let ownership = Ownership { rtx_expert_layers: admitted.placement.onboard_layers,
        spark_layers, inactive_seeded_layers, inactive_loaded_layers: 0 };
    if let Some(set) = &graph_set {
        super::graphs::check_placement_for_ranks(cfg, &admitted.placement, &set.keys, RANKS)?;
        tracing::info!(executables = set.keys.len(), bytes = ?(0..RANKS)
            .map(|rank| set.inventory.bytes(rank)).collect::<Vec<_>>(),
            reserve = "estimated", "Kimi startup graph inventory");
    }
    tracing::info!(placement = %admitted.placement.summary(), rtx_expert_layers = admitted.placement.onboard_layers,
        inactive_seeded_layers, inactive_loaded_layers = ownership.inactive_loaded_layers,
        prefix_marks = admitted.mark_selection.slots, marks = %admitted.mark_selection.summary(),
        spark_budget_bytes, reserve = "estimated until W9 census",
        "Kimi admitted TP2 expert ownership");
    cuteafd_bench::context::ResolvedSettings::placement(&admitted.placement).publish();
    cuteafd_bench::context::set_resolved("max-context-tokens", &options.max_context.to_string());
    let plan = admitted.execution_plan()?;
    let admission_free = inventory.gpus.iter().map(|sample| sample.free_bytes).collect::<Vec<_>>();
    let admitted_bytes = admitted.placement.items.iter().map(|items| items.iter().map(|item| item.bytes).sum::<u64>())
        .collect::<Vec<_>>();
    let streams: [_; RANKS] = devices.iter().map(|&device| Stream::new(device).map(Rc::new))
        .collect::<Result<Vec<_>>>()?.try_into().map_err(|_| anyhow::anyhow!("Kimi exchange stream count"))?;
    let (peer, hops) = if RANKS == 2 {
        let ranks = [0, 1].map(|rank| RankDevice { device: streams[rank].device.id, stream: streams[rank].raw });
        let peer = devices[0].run(|| PeerExchange::new_abortable(library, ranks, 1,
            peer_receive[0], peer_receive[1]))?;
        let mut hops = devices[0].run(|| HopLink::new(library, ranks, plan.hops(), plan.spec(),
            unit.min(64) * plan.spec().row_bytes as usize))?;
        hops.attach_abort([peer.abort_word(0)?.context("Kimi missing abort owner")?,
            peer.abort_word(1)?.context("Kimi missing abort owner")?])?;
        (Some(Rc::new(peer)), Some(hops))
    } else { (None, None) };
    let lane_streams: [_; RANKS] = devices.iter().map(|&device| Stream::new(device).map(Rc::new))
        .collect::<Result<Vec<_>>>()?.try_into().map_err(|_| anyhow::anyhow!("Kimi lane stream count"))?;
    memory.sample("streams-peer-hop")?;
    let draft_layout = admitted.draft_layout().cloned();
    let recorder = crate::shared::experts::rtx::dense::Recorder::start(
        crate::shared::experts::rtx::push::PushPolicy::for_snapshot(&checkpoint.snapshot)?
            == crate::shared::experts::rtx::push::PushPolicy::Required)?;
    let draft = options.draft.as_ref().zip(draft_layout.as_ref()).map(|(options, layout)|
        super::draft::NativeDraft::load_for_ranks(&mut admitted, programs, options, lane_streams.to_vec(), layout,
            peer.as_ref().map(|peer| ([streams[0].clone(), streams[1].clone()], peer.clone())))).transpose()?;
    let draft_records = recorder.map(|recorder| recorder.finish()).unwrap_or_default();
    memory.sample("draft-allocated-and-recorded")?;
    if memory.enabled() {
        if let Some(draft) = &draft {
            let mut census = crate::shared::memory::receipts::Census::default();
            census.owners(draft.allocation_receipts(
                cuteafd_loader::placement::planner::CoordinatorDevices::new(&options.devices)?)?);
            census.normalize(cuteafd_loader::placement::planner::PhaseOrder::DRAFT_THEN_TARGET)?;
            let roots = census.receipts.iter().filter(|receipt| receipt.allocation.view.is_none()
                && receipt.allocation.group == "weight arena").collect::<Vec<_>>();
            tracing::info!(roots = roots.len(), bytes = roots.iter().map(|root| root.allocation.bytes).sum::<u64>(),
                load_phase = roots.iter().all(|root| root.allocation.lifetime.contains(
                    cuteafd_loader::placement::planner::Phase::Load,
                    cuteafd_loader::placement::planner::PhaseOrder::DRAFT_THEN_TARGET)),
                "Kimi retained draft weight arena census");
        }
    }
    let engine = Engine::load_admitted(library, streams, lane_streams, draft, draft_records, admitted, options.expert_usage_reader.clone(), checkpoint, native, context_programs, peer, hops,
        options.peers.clone(), cuteafd_transport::TcpTransportConfig { timing: cuteafd_transport::protocol_v2_timing_from_env(), ..Default::default() }, expert_modules, &mut memory)?
        .with_served_context(options.max_context)?;
    memory.sample("pre-capture")?;
    stop_before_capture(memory.enabled())?;
    if let Some(set) = graph_set {
        let before = engine.graph_free_bytes()?;
        let capture_started = std::time::Instant::now();
        // Prewarm reachable decode/verify geometry only; row-window prefill
        // stays eager and unselected replayable geometry can capture lazily.
        let prewarm = super::graphs::prewarm_keys(cfg, &set.keys,
            cuteafd_loader::serving_capacity::kimi_prewarm::kimi_layout(RANKS), options.slots, unit);
        let policies = (0..RANKS).map(|rank| {
            let (base, tail) = cuteafd_loader::placement::graph_reserve::kimi_default(RANKS, rank);
            cuteafd_loader::placement::graph_reserve::current(rank, false, base, tail)
        }).collect::<std::result::Result<Vec<_>, _>>()?;
        let bases = policies.iter().map(|policy| policy.base_bytes).collect::<Vec<_>>();
        let bootstraps = (0..RANKS).map(|rank|
            cuteafd_loader::serving_capacity::kimi_prewarm::kimi_bootstrap(RANKS, rank)).collect::<Vec<_>>();
        let prewarm = super::graphs::fit_prewarm(prewarm,
            cuteafd_loader::serving_capacity::kimi_prewarm::kimi_layout(RANKS), &bases, &bootstraps);
        tracing::info!(entries = prewarm.len(), attention_entries = prewarm.iter()
            .filter(|key| matches!(key.op, super::graphs::Op::Attention { .. })).count(),
            "Kimi selected startup prewarm keys");
        let warmup = engine.warm_graphs(&prewarm, &reserves)?;
        let captured = warmup.captured;
        let capture_elapsed_ms = capture_started.elapsed().as_millis() as u64;
        let after = engine.graph_free_bytes()?;
        let census_table = if RANKS == 2 { "GRAPH_CENSUS_TP2" } else { "GRAPH_CENSUS_BYTES" };
        for rank in 0..RANKS {
            let first_touch_bytes = before[rank].saturating_sub(warmup.before_capture[rank]);
            let bytes = warmup.before_capture[rank].saturating_sub(after[rank]);
            let count = prewarm.iter().filter(|key| key.rank == rank).count() as u64;
            let reserved = set.inventory.bytes(rank);
            tracing::info!(rank, first_touch_bytes, owner = "KDA eager first-use storage",
                "Kimi startup first-touch census (excluded from graph arena)");
            let charged = crate::shared::decode_graph::device::snapshots().into_iter()
                .find(|sample| sample.device == options.devices[rank]).map_or(0, |sample| sample.charged);
            let capture_delta = i128::from(bytes) - i128::from(charged);
            tracing::info!(rank, executables = count, captured, measured_bytes = bytes,
                graph_charged_bytes = charged, capture_delta_bytes = %capture_delta, capture_elapsed_ms,
                bytes_per_executable = bytes.div_ceil(count.max(1)),
                census_estimate_bytes = reserved, reserve = reserves[rank], reserve_basis = "checked-in layout row",
                "Kimi startup graph census");
            if bytes > reserves[rank] {
                tracing::warn!(rank, executables = count, measured_bytes = bytes, reserve = reserves[rank],
                    bytes_per_executable = bytes.div_ceil(count.max(1)), census = census_table,
                    "Kimi startup capture exceeds its graph reserve; update \
                     cuteafd_loader::serving_capacity::kimi_prewarm with this capture");
            }
        }
    }
    Ok(Loaded { engine, admission_free, admitted_bytes, ownership })
}

fn check_attention(attention: Option<cuteafd_loader::placement::AttentionPlacement>, ranks: usize, graphs: bool) -> Result<()> {
    use cuteafd_loader::placement::AttentionPlacement;
    if attention == Some(AttentionPlacement::Context) {
        ensure!(ranks == 2 && graphs, "Kimi context requires two ranks and startup decode graphs");
    }
    Ok(())
}

pub(super) fn code_demand(rank: usize, pending: u64) -> Result<Demand> {
    cuteafd_loader::families::kimi_k3::allocation::code_demand(rank, pending, 2)
}

use cuteafd_loader::families::kimi_k3::core::exchange_recv_bytes;

pub(super) fn exchange_demands(rows: usize, diagnostics: bool) -> Result<Vec<Demand>> {
    let progress = if diagnostics { (2 * std::mem::size_of::<cuteafd_ffi::peer_exchange::PeerProgress>()).max(256) } else { 0 };
    (0..2).map(|rank| cuteafd_loader::families::kimi_k3::allocation::exchange_demand(
        rows, rank, progress)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::families::kimi_k3::core::{CoreInventory,CoreProjection,CoreSplit};

    #[test]
    fn memory_diagnostic_defaults_off_and_stops_before_capture() -> Result<()> {
        assert!(!memory_phase_mode(None)?);
        assert!(!memory_phase_mode(Some("off"))?);
        assert!(memory_phase_mode(Some("pre-capture"))?);
        for mode in ["", "1", "on", "ready"] { assert!(memory_phase_mode(Some(mode)).is_err()); }
        stop_before_capture(false)?;
        let error = stop_before_capture(true).unwrap_err().to_string();
        assert!(error.starts_with("KIMI_STARTUP_MEMORY_PRE_CAPTURE_STOP"));
        assert!(error.contains("Ready were not reached"));
        Ok(())
    }

    #[test]
    fn automatic_hosted_hint_may_cover_but_never_omit_selected_layers() -> Result<()> {
        use cuteafd_loader::placement::OnboardSetting::{Auto, All, Layers};
        assert_eq!(check_hosted_layers(Auto, &[3, 4], &[2, 3, 4])?, 1);
        assert!(check_hosted_layers(Auto, &[3, 4], &[2, 3]).is_err());
        for setting in [Layers(1), All] {
            assert!(check_hosted_layers(setting, &[3, 4], &[2, 3, 4]).is_err());
            assert_eq!(check_hosted_layers(setting, &[3, 4], &[3, 4])?, 0);
        }
        assert!(check_hosted_layers(Auto, &[3, 4], &[3, 3, 4]).is_err());
        Ok(())
    }

    #[test]
    fn context_cannot_fall_back_to_single_rank_or_eager_decode() {
        use cuteafd_loader::placement::AttentionPlacement;
        assert!(check_attention(Some(AttentionPlacement::Context), 2, true).is_ok());
        assert!(check_attention(Some(AttentionPlacement::Context), 1, true).is_err());
        assert!(check_attention(Some(AttentionPlacement::Context), 2, false).is_err());
        for mode in [None, Some(AttentionPlacement::Heads), Some(AttentionPlacement::Layers)] {
            assert!(check_attention(mode, 2, false).is_ok());
        }
    }

    #[test]
    fn native_modules_load_inside_the_shared_measured_baseline() {
        assert!(include_str!("../../shared/inventory.rs").contains("library.tp2_native_initialize()"));
        let source = include_str!("startup.rs");
        let measured = source.split("RuntimeInventory::measure(library, &options.devices, InventoryPhase::PreWeight")
            .nth(1).unwrap().split("let lane_layouts").next().unwrap();
        assert!(!measured.contains("tp2_native_initialize"),
            "the shared inventory preloads native modules; a family copy would load them twice");
    }

    fn projection(stem:&str)->CoreProjection {
        CoreProjection {stem:stem.into(),bits:6,codebook:cuteafd_loader::formats::exl3_storage::Exl3Codebook::Mcg,
            shape:[128,128],stored_outputs:128,split:CoreSplit::Whole,source_bytes:0}
    }

    #[test]
    fn header_native_core_checks_mixed_groups_before_device_access()->Result<()> {
        let cfg=KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        let mut core=CoreInventory::default();
        core.validate_runtime_groups(&cfg)?;
        let head="language_model.lm_head";
        core.projections.insert(head.into(),projection(head));
        core.validate_runtime_groups(&cfg)?;
        let root="language_model.model.layers.1.self_attn";
        let stem=format!("{root}.q_proj");
        core.projections.insert(stem.clone(),projection(&stem));
        let error=core.validate_runtime_groups(&cfg).unwrap_err().to_string();
        assert!(error.contains("validated per-projection") && error.contains(root));
        for leaf in ["k_proj","v_proj","g_proj","o_proj"] {
            let stem=format!("{root}.{leaf}");core.projections.insert(stem.clone(),projection(&stem));
        }
        core.validate_runtime_groups(&cfg)?;
        Ok(())
    }

    #[test]
    fn bf16_partial_receivers_and_admission_use_the_same_exact_extent()->Result<()> {
        for rows in [1,8,64,4096] {for diagnostics in [false,true] {
            let native=exchange_demands(rows,diagnostics)?;
            assert_eq!(exchange_recv_bytes(rows)?,rows*HIDDEN*2);
            for rank in 0..2 {
                let progress=if diagnostics {(2*std::mem::size_of::<cuteafd_ffi::peer_exchange::PeerProgress>()).max(256)}else{0};
                assert_eq!(native[rank].bytes as usize,rows*HIDDEN*2+256+256
                    +(((cuteafd_loader::placement::HOP_SLOTS+1)*16)as usize).max(256)+progress);
                assert_eq!(native[rank].basis,Basis::Exact);
            }
        }}
        assert!(exchange_recv_bytes(0).is_err());
        assert!(exchange_recv_bytes(4097).is_err());
        Ok(())
    }
    #[test]
    #[ignore = "requires CUTEAFD_KIMI_RUNTIME_ROOT; validates real lib/share layout in an isolated process"]
    fn mounted_runtime_artifact_preflight() -> anyhow::Result<()> {
        let root = std::env::var_os("CUTEAFD_KIMI_RUNTIME_ROOT")
            .ok_or_else(|| anyhow::anyhow!("Kimi runtime root required"))?;
        let root = std::path::Path::new(&root);
        cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::KIMI_K3)
            .map_err(|_| anyhow::anyhow!("run this mounted preflight test in an isolated process"))?;
        assert!(!root.join("share/exl3").exists());
        super::preflight(&root.join("share/PROGRAMS.json"), &root.join("lib/libcuteafd_native.so"))
    }

    #[test]
    fn missing_inventory_is_refused_without_native_or_device_access() {
        let dir = tempfile::tempdir().unwrap();
        assert!(super::preflight(&dir.path().join("share/absent.json"), &dir.path().join("lib/native.so")).is_err());
    }
}

#[cfg(test)]
mod decode_extent_tests {
    use super::*;

    #[test]
    fn declared_decode_extent_matches_native_projection_and_residual() -> Result<()> {
        for capacity in [32usize, 64, 2048] {
            let rows = capacity.min(64);
            let actual = [rows * HIDDEN * 2, rows * 3584 * 2];
            assert_eq!(exchange_recv_bytes(rows)?, actual.into_iter().max().unwrap());
            for verify in [0, 16, 64] {
                assert!(cuteafd_loader::families::kimi_k3::allocation::graph_rows(verify, capacity)
                    .into_iter().all(|width| width <= rows));
            }
        }
        Ok(())
    }
}
