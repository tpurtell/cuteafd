//! Shared native startup for serving and diagnostics. Admission precedes all
//! retained operands and ordinary six-rank transport connection.
use super::{admission, device::NativePrograms, engine::Engine};
use crate::shared::{inventory::{InventoryPhase, RuntimeInventory}, memory::device::{Device, Stream},
    peer_split::{hop::HopLink, PeerExchange, RankDevice}};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{ExpertGeometry, memory_layout::{Basis, Category}};
use cuteafd_loader::{families::kimi_k3::config::KimiK3Config, placement::{Demand, GpuBudget, HOP_SLOTS}};
use std::{net::SocketAddr, path::Path, rc::Rc, sync::Arc};
const HIDDEN: usize = 7168;

pub(crate) struct Options<'a> {
    pub devices: [i32; 2],
    pub native_lib: &'a Path,
    pub peers: Vec<SocketAddr>,
    pub rtx_expert_layers: usize,
    pub rows: usize,
    pub slots: usize,
    pub marks: usize,
    pub max_context: usize,
    pub pool_tokens: Option<u64>,
    pub graphs: bool,
    pub ffn_mode: cuteafd_ffi::KimiFfnMode,
    pub hosted_layers: Option<Vec<usize>>,
}

pub(crate) struct Loaded<'a> {
    pub engine: Engine<'a>,
    pub admission_free: Vec<u64>,
    pub admitted_bytes: Vec<u64>,
}

pub(crate) fn load<'a>(library: &'a Arc<cuteafd_ffi::NativeLibrary>,
    programs: &'a cuteafd_ffi::programs::Programs<'a>,
    checkpoint: &cuteafd_loader::plan::Checkpoint, full: &KimiK3Config,
    cfg: &KimiK3Config, options: Options<'_>) -> Result<Loaded<'a>> {
    ensure!(crate::shared::peer_split::probed_device(library, options.devices[0], Some(options.devices[1]))?
        == Some(options.devices[1]), "Kimi TP2 requires bidirectional peer access");
    let devices = [options.devices[0], options.devices[1]].map(|id| Device { library, id });
    let native = devices.map(|device| NativePrograms::load(device, full, programs, options.ffn_mode).map(Rc::new));
    let [zero, one] = native;
    let native = [zero?, one?];
    let rows = options.rows;
    let rtx = if options.rtx_expert_layers == 0 { None } else {
        let catalog = cuteafd_loader::read_expert_catalog(&checkpoint.snapshot)?;
        let tiers = catalog.exl3().context("Kimi whole experts require EXL3 checkpoint")?.decoder_tiers();
        let package = crate::shared::experts::exl3::aot_layout_directory(
            options.native_lib, tiers, "rtx-tp1");
        Some(admission::plan_rtx(cfg, &catalog, package, rows, options.rtx_expert_layers)?)
    };
    let expert_modules = if let Some(inventory) = &rtx {
        use crate::shared::experts::rtx::{exl3::Exl3Tp2, RtxSelectionPolicy};
        [Some(Exl3Tp2::preload_whole_modules(devices[0], &inventory.package, rows,
            ExpertGeometry::KIMI_K3, RtxSelectionPolicy::FirstCapacity)?),
         Some(Exl3Tp2::preload_whole_modules(devices[1], &inventory.package, rows,
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
    let lane_layouts = vec![[native[0].lane_layout(full, rows, options.max_context)?,
        native[1].lane_layout(full, rows, options.max_context)?]];
    let graph_set = options.graphs.then(|| super::graphs::owned_inventory(cfg, 1, rows, options.slots)).transpose()?;
    let mut fixed = exchange_demands(rows, diagnostics)?;
    let selected = cuteafd_loader::placement::ProgramSet::from_names(&["kimik3"], programs.names());
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
    let admitted = admission::admit(admission::Inputs { cfg, gpus, devices: [options.devices[0], options.devices[1]],
        peer_access: true, spark_ranks: 6, attention: None, requested_pool: options.pool_tokens,
        state_slots: options.slots, prefix_marks: options.marks, rows, lanes: 1, lane_layouts, fixed,
        startup: graph_set.as_ref().map_or_else(Vec::new, |set|
            (0..2).map(|rank| super::graphs::demand(&set.inventory, rank)).collect()), pool_bounded: vec![], rtx })?;
    if let Some(expected) = options.hosted_layers.as_ref() {
        ensure!(admitted.working_set().spark_layers == *expected,
            "Kimi worker layer subset differs from shared placement");
    }
    if let Some(set) = &graph_set {
        super::graphs::check_placement(cfg, &admitted.placement, &set.keys)?;
        tracing::info!(executables = set.keys.len(), bytes0 = set.inventory.bytes(0),
            bytes1 = set.inventory.bytes(1), reserve = "estimated", "Kimi startup graph inventory");
    }
    tracing::info!(placement = %admitted.placement.summary(), "Kimi admitted whole expert ownership");
    let plan = admitted.execution_plan()?;
    let admission_free = inventory.gpus.iter().map(|sample| sample.free_bytes).collect::<Vec<_>>();
    let admitted_bytes = admitted.placement.items.iter().map(|items| items.iter().map(|item| item.bytes).sum::<u64>())
        .collect::<Vec<_>>();
    let streams = [Rc::new(Stream::new(devices[0])?), Rc::new(Stream::new(devices[1])?)];
    let ranks = streams.each_ref().map(|s| RankDevice { device: s.device.id, stream: s.raw });
    let peer = devices[0].run(|| PeerExchange::new_abortable(library, ranks, 1, rows * HIDDEN * 2))?;
    let mut hops = devices[0].run(|| HopLink::new(library, ranks, plan.hops(), plan.spec()))?;
    hops.attach_abort([peer.abort_word(0)?.context("Kimi missing abort owner")?,
        peer.abort_word(1)?.context("Kimi missing abort owner")?])?;
    let engine = Engine::load_admitted(library, streams, admitted, checkpoint, native, peer, hops,
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
    ensure!(rank < 2 && (1..=2).contains(&lanes), "Kimi code allowance rank/lane geometry");
    // Shared allowance covers one thread-local handle per rank. Each head-rank
    // lane owns an additional private handle beside its explicitly charged workspace.
    let heads = if rank == 1 { lanes as u64 } else { 0 };
    let bytes = heads.checked_mul(cublas_bytes).and_then(|extra| pending.checked_add(extra))
        .context("Kimi future code/handle allowance overflow")?;
    Ok(Demand::new(rank as u8, Category::Workspace, "Kimi future loaded code and private head handles",
        bytes, Basis::Formula))
}

pub(super) fn exchange_demands(rows: usize, diagnostics: bool) -> Result<Vec<Demand>> {
    ensure!((1..=4096).contains(&rows), "Kimi exchange row capacity");
    let recv = (rows * HIDDEN * 2).max(256);
    let progress = if diagnostics { (2 * std::mem::size_of::<cuteafd_ffi::peer_exchange::PeerProgress>()).max(256) } else { 0 };
    // Hop receive planes are charged by the shared solver; only controls are fixed.
    let bytes = recv + 256 + 256 + (((HOP_SLOTS + 1) * 16) as usize).max(256) + progress;
    Ok((0..2).map(|rank| Demand::new(rank, Category::Workspace, "Kimi peer/abort/hop controls",
        bytes as u64, Basis::Exact)).collect())
}
