//! Shared native startup for serving and diagnostics. Admission precedes all
//! retained operands and ordinary six-rank transport connection.
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

pub(crate) struct Options<'a> {
    pub cache_format: cuteafd_core::MlaCacheFormat,
    pub devices: [i32; 2],
    pub native_lib: &'a Path,
    pub peers: Vec<SocketAddr>,
    pub rtx_expert_layers: cuteafd_loader::placement::Onboard,
    pub spark_budget_gib: Option<f64>,
    /// Lane rows: `waves` prefill units of `rows / waves` each.
    pub rows: usize,
    /// Prefill units in flight (Spark waves outstanding, one return slot each).
    pub waves: usize,
    /// Explicit unit rows; unset takes `rows / waves`.
    pub prefill_rows: Option<usize>,
    pub slots: usize,
    pub marks: usize,
    pub max_context: usize,
    pub pool_tokens: Option<u64>,
    pub graphs: bool,
    pub ffn_mode: cuteafd_ffi::KimiFfnMode,
    pub hosted_layers: Option<Vec<usize>>,
    pub draft: Option<super::draft::Options>,
    /// Exact speculative verify widths given startup graphs; 0 = the drafter's block.
    pub verify_rows: usize,
}

pub(crate) struct Loaded<'a> {
    pub engine: Engine<'a>,
    pub admission_free: Vec<u64>,
    pub admitted_bytes: Vec<u64>,
}

/// Run before loading native programs, whose card selection can query the GPU.
pub(crate) fn preflight(manifest: &Path, native_lib: &Path) -> Result<()> {
    let package = crate::shared::experts::exl3::aot_layout_directory(native_lib, &[2, 3], "");
    cuteafd_loader::families::kimi_k3::validate_expert_inventory(manifest, &package)
}

pub(crate) fn load<'a>(library: &'a Arc<cuteafd_ffi::NativeLibrary>,
    programs: &'a cuteafd_ffi::programs::Programs<'a>,
    checkpoint: &cuteafd_loader::plan::Checkpoint, full: &KimiK3Config,
    cfg: &KimiK3Config, options: Options<'_>) -> Result<Loaded<'a>> {
    ensure!(options.draft.as_ref().is_none_or(|draft| draft.cache_format == options.cache_format),
        "Kimi target/draft cache records differ");
    let core = cuteafd_loader::plan::families::kimi::core_inventory(checkpoint)?;
    core.validate_runtime_groups(full)?;
    let f32_partials = needs_f32_partials(cfg, &core);
    ensure!(crate::shared::peer_split::probed_device(library, options.devices[0], Some(options.devices[1]))?
        == Some(options.devices[1]), "Kimi TP2 requires bidirectional peer access");
    let devices = [options.devices[0], options.devices[1]].map(|id| Device { library, id });
    let rows = options.rows;
    // Lane rows hold `waves` units; every kernel launch, hook-local scratch,
    // peer exchange, hop and Spark wave serves one unit.
    let unit = cuteafd_loader::families::kimi_k3::allocation::prefill_shape_with(rows, options.waves,
        options.prefill_rows)?.unit_rows as usize;
    let native = devices.map(|device| NativePrograms::load_with_core(device, full, programs, options.ffn_mode,
        options.cache_format, &core, unit).map(Rc::new));
    let [zero, one] = native;
    let native = [zero?, one?];
    let draft_layout = options.draft.as_ref().map(|draft| devices[1].run(||
        draft.layout(options.slots, options.max_context, library.sm_count()?))).transpose()?;
    // Resolve native modules before the measured baseline, while retaining their handles.
    let _draft_modules = options.draft.as_ref().map(|_| devices[1].run(|| {
        use cuteafd_ffi::mla_draft::{MlaDraft, MlaDraftWindow};
        Ok([MlaDraft::load(programs, "kimik3", MlaDraftWindow::Full, options.cache_format)?,
            MlaDraft::load(programs, "kimik3", MlaDraftWindow::Sliding4096, options.cache_format)?])
    })).transpose()?;
    let spark_budget_gib = options.spark_budget_gib.unwrap_or(
        cuteafd_loader::families::kimi_k3::SPARK_DEFAULT_BUDGET_BYTES as f64 / (1u64 << 30) as f64);
    ensure!(spark_budget_gib.is_finite() && spark_budget_gib > 0.0
        && spark_budget_gib < u64::MAX as f64 / (1u64 << 30) as f64,
        "Kimi --spark-budget-gib must be finite, positive and representable");
    let catalog = cuteafd_loader::read_expert_catalog(&checkpoint.snapshot)?;
    let routed = cfg.dense.iter().filter(|dense| !**dense).count();
    let model = crate::shared::placement::spark_rings(cuteafd_transport::TcpTransportConfig::default().max_frame_bytes)?;
    let working = cuteafd_loader::placement::SparkWorkingSet { rtx_layers: vec![],
        spark_layers: cfg.dense.iter().enumerate().filter_map(|(layer, dense)| (!dense).then_some(layer)).collect(),
        spark_ranks: 6, prefill_lanes: 1 };
    let spark_budget = cuteafd_loader::placement::SparkBudgetInventory::exl3(&catalog, &working,
        &[cuteafd_core::SparkEndpoint { ring_depth: None, capacity: u32::try_from(unit)?, ingress: cuteafd_core::SparkIngress::Fp8K32, execution_lane: 0 }],
        model, (spark_budget_gib * (1u64 << 30) as f64) as u64,
        cuteafd_loader::families::kimi_k3::SPARK_ESTIMATED_RESERVE_BYTES)?;
    let rtx = if options.rtx_expert_layers.layers(routed) == Some(0) { None } else {
        let tiers = catalog.exl3().context("Kimi TP2 experts require EXL3 checkpoint")?.decoder_tiers();
        let package = crate::shared::experts::exl3::aot_layout_directory(
            options.native_lib, tiers, "rtx-tp2");
        let mut inventory = admission::plan_rtx(cfg, &catalog, package, unit, routed)?;
        inventory.layers = options.rtx_expert_layers;
        Some(inventory)
    };
    let expert_modules = if let Some(inventory) = &rtx {
        use crate::shared::experts::rtx::{exl3::Exl3Tp2, RtxSelectionPolicy};
        [Some(Exl3Tp2::preload_tp2_modules(devices[0], &inventory.package, unit,
            ExpertGeometry::KIMI_K3, RtxSelectionPolicy::FirstCapacity)?),
         Some(Exl3Tp2::preload_tp2_modules(devices[1], &inventory.package, unit,
            ExpertGeometry::KIMI_K3, RtxSelectionPolicy::FirstCapacity)?)]
    } else { [None, None] };
    let diagnostics = std::env::var("CUTEAFD_PEER_WAIT_DIAGNOSTICS").is_ok_and(|v| v == "1");
    let inventory = RuntimeInventory::measure(library, &[options.devices[0], options.devices[1]], InventoryPhase::PreWeight, |_| {
        library.tp2_native_initialize()?;
        library.peer_exchange_initialize()?;
        library.peer_abort_initialize()?;
        if diagnostics { library.peer_diagnostic_initialize()?; }
        library.v41_compact_reducer()?.require_plane_reduction(6)?;
        Ok(())
    })?;
    let lane_layouts = vec![[native[0].lane_layout(full, rows, unit, options.max_context)?,
        native[1].lane_layout(full, rows, unit, options.max_context)?]];
    let verify_rows = match options.verify_rows {
        0 => options.draft.as_ref().map(super::draft::Options::verify_rows).transpose()?.unwrap_or(0),
        rows => rows,
    };
    let graph_set = options.graphs.then(|| super::graphs::owned_inventory(cfg, 1, unit, options.slots, verify_rows)).transpose()?;
    let mut fixed = exchange_demands(unit, diagnostics, f32_partials)?;
    if let Some(layout) = &draft_layout {
        fixed.extend(super::draft::bridge_demands(layout, rows)?);
        fixed.push(cuteafd_loader::families::kimi_k3::draft_probe_demand(cfg.layers, cfg.topk, rows.min(64))?);
        if core.projections.contains_key("language_model.lm_head") {
            fixed.extend(cuteafd_loader::families::kimi_k3::core::draft_head_demand(&core,
                layout.capacity().block_rows, programs.family_metadata("kimi_k6_dense")?)?);
        }
    }
    let selected = cuteafd_loader::placement::ProgramSet::from_names(
        if core.projections.is_empty() { &["kimik3"][..] } else { &["kimik3", "kimi"][..] }, programs.names());
    for (rank, sample) in inventory.gpus.iter().enumerate() {
        let pending = crate::shared::inventory::pending_code(library, sample.device, rank, true,
            "kimik3", "exl3", &selected)?;
        let arch = cuteafd_loader::placement::ArchContext::for_device(&sample.arch, sample.total_bytes);
        fixed.push(code_demand(rank, pending, arch.cublas_bytes, 1)?);
    }
    let gpus = inventory.gpus.iter().map(|sample| {
        let capacity = cuteafd_ffi::coordinator_gpu_budget().map_or(sample.total_bytes, |b| b.0.min(sample.total_bytes));
        GpuBudget { capacity_bytes: capacity, headroom_bytes: 2 << 30,
            baseline: cuteafd_loader::placement::Baseline::Measured { free_bytes: sample.free_bytes } }
    }).collect::<Vec<_>>().try_into().map_err(|_| anyhow::anyhow!("Kimi GPU sample count"))?;
    let mut admitted = admission::admit(admission::Inputs { cfg, core, cache_format: options.cache_format, gpus, devices: [options.devices[0], options.devices[1]],
        peer_access: true, spark_ranks: 6, attention: None, requested_pool: options.pool_tokens,
        state_slots: options.slots, prefix_marks: options.marks, rows, lanes: 1, waves: options.waves,
        unit_rows: options.prefill_rows, lane_layouts, fixed,
        startup: graph_set.as_ref().map_or_else(Vec::new, |set|
            (0..2).map(|rank| super::graphs::demand(&set.inventory, rank)).collect()), pool_bounded: vec![], rtx,
        spark_budget: Some(spark_budget), draft: draft_layout.clone(), verify_rows })?;
    let graph_set = admitted.graph_set.take();
    if let Some(expected) = options.hosted_layers.as_ref() {
        ensure!(admitted.working_set().spark_layers == *expected,
            "Kimi worker layer subset differs from shared placement");
    }
    if let Some(set) = &graph_set {
        super::graphs::check_placement(cfg, &admitted.placement, &set.keys)?;
        tracing::info!(executables = set.keys.len(), bytes0 = set.inventory.bytes(0),
            bytes1 = set.inventory.bytes(1), reserve = "estimated", "Kimi startup graph inventory");
    }
    tracing::info!(placement = %admitted.placement.summary(), rtx_expert_layers = admitted.placement.onboard_layers,
        prefix_marks = admitted.mark_selection.slots, marks = %admitted.mark_selection.summary(),
        spark_budget_gib, reserve = "estimated until W9 census",
        "Kimi admitted TP2 expert ownership");
    let plan = admitted.execution_plan()?;
    let admission_free = inventory.gpus.iter().map(|sample| sample.free_bytes).collect::<Vec<_>>();
    let admitted_bytes = admitted.placement.items.iter().map(|items| items.iter().map(|item| item.bytes).sum::<u64>())
        .collect::<Vec<_>>();
    let streams = [Rc::new(Stream::new(devices[0])?), Rc::new(Stream::new(devices[1])?)];
    let ranks = streams.each_ref().map(|s| RankDevice { device: s.device.id, stream: s.raw });
    let peer = devices[0].run(|| PeerExchange::new_abortable(library, ranks, 1, exchange_recv_bytes(unit,f32_partials)?))?;
    let mut hops = devices[0].run(|| HopLink::new(library, ranks, plan.hops(), plan.spec()))?;
    hops.attach_abort([peer.abort_word(0)?.context("Kimi missing abort owner")?,
        peer.abort_word(1)?.context("Kimi missing abort owner")?])?;
    let lane_streams = [Rc::new(Stream::new(devices[0])?), Rc::new(Stream::new(devices[1])?)];
    let draft_layout = admitted.draft_layout().cloned();
    let draft = options.draft.as_ref().zip(draft_layout.as_ref()).map(|(options, layout)|
        super::draft::NativeDraft::load(&mut admitted, programs, options, lane_streams.clone(), layout)).transpose()?;
    let engine = Engine::load_admitted(library, streams, lane_streams, draft, admitted, checkpoint, native, peer, hops,
        options.peers.clone(), cuteafd_transport::TcpTransportConfig::default(), 4096, expert_modules)?;
    if let Some(set) = graph_set {
        let before = RuntimeInventory::measure(library, &options.devices, InventoryPhase::PostLoad, |_| Ok(()))?;
        engine.warm_graphs(&set.keys)?;
        let after = RuntimeInventory::measure(library, &options.devices, InventoryPhase::PostLoad, |_| Ok(()))?;
        for rank in 0..2 {
            let bytes = before.gpus[rank].free_bytes.saturating_sub(after.gpus[rank].free_bytes);
            let count = set.inventory.ranks[rank].executables;
            tracing::info!(rank, executables = count, measured_bytes = bytes,
                bytes_per_executable = bytes.div_ceil(count.max(1)),
                reserved_bytes = set.inventory.bytes(rank), reserve = "estimated",
                "Kimi startup graph census");
            ensure!(bytes <= set.inventory.bytes(rank), "Kimi startup graphs exceed admitted reserve");
        }
    }
    Ok(Loaded { engine, admission_free, admitted_bytes })
}

pub(super) fn code_demand(rank: usize, pending: u64, cublas_bytes: u64, lanes: usize) -> Result<Demand> {
    cuteafd_loader::families::kimi_k3::allocation::code_demand(rank, pending, cublas_bytes, lanes)
}

use cuteafd_loader::families::kimi_k3::core::{needs_f32_partials,exchange_recv_bytes};

pub(super) fn exchange_demands(rows: usize, diagnostics: bool,f32_partials:bool) -> Result<Vec<Demand>> {
    let progress = if diagnostics { (2 * std::mem::size_of::<cuteafd_ffi::peer_exchange::PeerProgress>()).max(256) } else { 0 };
    (0..2).map(|rank| cuteafd_loader::families::kimi_k3::allocation::exchange_demand(
        rows, rank, f32_partials, progress)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cuteafd_loader::families::kimi_k3::core::{CoreInventory,CoreProjection,CoreSplit};

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
        assert!(!needs_f32_partials(&cfg,&core));
        let root="language_model.model.layers.1.self_attn";
        let stem=format!("{root}.q_proj");
        core.projections.insert(stem.clone(),projection(&stem));
        let error=core.validate_runtime_groups(&cfg).unwrap_err().to_string();
        assert!(error.contains(&stem) && error.contains("k_proj") && error.contains("whole-group"));
        for leaf in ["k_proj","v_proj","g_proj","o_proj"] {
            let stem=format!("{root}.{leaf}");core.projections.insert(stem.clone(),projection(&stem));
        }
        core.validate_runtime_groups(&cfg)?;
        assert!(needs_f32_partials(&cfg,&core));
        let mut prefix=cfg.clone();prefix.layers=1;
        assert!(!needs_f32_partials(&prefix,&core));
        Ok(())
    }

    #[test]
    fn f32_partial_receivers_and_admission_use_the_same_exact_extent()->Result<()> {
        for rows in [1,8,64,4096] {for diagnostics in [false,true] {
            let old=exchange_demands(rows,diagnostics,false)?;
            let native=exchange_demands(rows,diagnostics,true)?;
            assert_eq!(exchange_recv_bytes(rows,true)?,rows*HIDDEN*4);
            for rank in 0..2 {
                assert_eq!(native[rank].bytes-old[rank].bytes,(rows*HIDDEN*2) as u64);
                assert_eq!(native[rank].basis,Basis::Exact);
            }
        }}
        assert!(exchange_recv_bytes(0,true).is_err());
        assert!(exchange_recv_bytes(4097,true).is_err());
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
