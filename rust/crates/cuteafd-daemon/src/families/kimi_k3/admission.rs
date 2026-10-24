//! Native coordinator inventory passed to the shared startup solver, before
//! weights, recurrent state or cache owners allocate. This is not the header
//! lower bound exposed by `plan` and does not admit Spark workers by itself.
use super::{device::{IoLayout, TokenIo, Lane, LaneLayout, LocalLane, NativePrograms}, mla::{CacheBank, CacheLayout}, state::{KdaBank, KdaLayout}, weights::{GlobalLayout, GlobalWeights, LayerLayout, LayerWeights}};
use crate::shared::memory::device::{Device, DeviceOwner, Stream};
use crate::shared::speculation::dflash::DflashDrafter;
use cuteafd_loader::formats::dflash::MlaDraftRuntimeLayout;
use std::rc::{Rc, Weak};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{Basis, Category};
use cuteafd_ffi::{MLA_PAGE_ROWS, MLA_RECORD_BYTES};
use cuteafd_loader::{families::kimi_k3::config::{KimiAttention, KimiK3Config}, placement::*};

const EXECUTOR: ExecutorModes = families::KIMI_K3;
const MODES: &[LayerMode] = EXECUTOR.modes;
const DRAFT_MARKS: &str = "Kimi exact draft windows";

/// Diagnostic inventories retain checkpoint numerics and select only a prefix.
pub(super) fn prefix(cfg: &KimiK3Config, layers: usize) -> Result<KimiK3Config> {
    ensure!((1..=cfg.layers).contains(&layers), "Kimi prefix layer count outside checkpoint");
    let mut selected = cfg.clone();
    selected.layers = layers;
    selected.attention.truncate(layers);
    selected.dense.truncate(layers);
    Ok(selected)
}

pub(super) fn check_prefix(full: &KimiK3Config, selected: &KimiK3Config) -> Result<()> {
    ensure!(prefix(full, selected.layers)? == *selected,
        "Kimi startup inventory differs from exact checkpoint prefix");
    Ok(())
}

/// All demands are still future to the two post-module CUDA samples. The
/// engine supplies its native programs' lane extents, exchange, graphs and
/// page tables; global weights, token IO and state banks add their own demands.
pub(crate) struct Inputs<'a> {
    pub cfg: &'a KimiK3Config,
    pub cache_format: cuteafd_core::MlaCacheFormat,
    pub core: cuteafd_loader::families::kimi_k3::core::CoreInventory,
    pub gpus: [GpuBudget; 2],
    pub devices: [i32; 2],
    pub peer_access: bool,
    pub spark_ranks: usize,
    pub attention: Option<AttentionPlacement>,
    pub requested_pool: Option<u64>,
    pub state_slots: usize,
    pub prefix_marks: usize,
    pub rows: usize,
    pub lanes: usize,
    /// Prefill units in flight on the lane: `rows` holds `waves` units, and
    /// the Spark transport admits that many waves (one return slot each).
    pub waves: usize,
    /// Explicit unit rows (else `rows / waves`).
    pub unit_rows: Option<usize>,
    pub lane_layouts: Vec<[LaneLayout; 2]>,
    pub fixed: Vec<Demand>,
    pub startup: Vec<Demand>,
    pub pool_bounded: Vec<BoundedDemand>,
    pub rtx: Option<RtxInventory>,
    pub spark_budget: Option<SparkBudgetInventory>,
    pub draft: Option<MlaDraftRuntimeLayout>,
    /// Exact speculative verify widths with startup graphs: the drafter's block, else 0.
    pub verify_rows: usize,
}

/// CPU-planned native residency; no backend owner allocates before solve.
pub(crate) struct RtxInventory {
    pub package: std::path::PathBuf,
    pub costs: Vec<[Bytes2; 2]>,
    pub workspace: usize,
    pub layers: Onboard,
}

pub(crate) fn plan_rtx(cfg: &KimiK3Config, catalog: &cuteafd_loader::OfficialV41Catalog,
    package: std::path::PathBuf, rows: usize, layers: usize) -> Result<RtxInventory> {
    use crate::shared::experts::{exl3::Exl3Weights, layer::ExpertLayer};
    ensure!(layers > 0 && layers <= cfg.dense.iter().filter(|dense| !**dense).count(),
        "Kimi TP2 expert count outside routed inventory");
    let workspace = usize::try_from(cuteafd_loader::families::kimi_k3::allocation::expert_workspace(&package, rows)?)?;
    let costs = cfg.dense.iter().enumerate().map(|(layer, dense)| {
        if *dense { return Ok([Bytes2::default(); 2]); }
        let mut costs = [Bytes2::default(); 2];
        for (rank, cost) in costs.iter_mut().enumerate() {
            let plan = Exl3Weights::plan(catalog, ExpertLayer::BackboneTp2 { layer, rank })?;
            *cost = Bytes2 { resident: plan.resident_bytes as u64, staging: plan.device_staging_bytes as u64 };
        }
        Ok(costs)
    }).collect::<Result<Vec<_>>>()?;
    Ok(RtxInventory { package, costs, workspace, layers: Onboard::Layers(layers) })
}

pub(crate) struct Admission<'a> {
    pub placement: Placement,
    pub mark_selection: MarkSelection,
    pub graph_set: Option<cuteafd_loader::placement::inventory::StartupGraphs<super::graphs::Key>>,
    cfg: KimiK3Config,
    core: cuteafd_loader::families::kimi_k3::core::CoreInventory,
    rtx: Option<RtxInventory>,
    rtx_loaded: bool,
    draft: Option<MlaDraftRuntimeLayout>,
    draft_allocated: bool,
    draft_marks_allocated: bool,
    layouts: Vec<[LayerLayout; 2]>,
    layer_owners: Vec<[Weak<LayerWeights<'a>>; 2]>,
    layer_uploaded: Vec<[bool; 2]>,
    local_bound: Vec<[bool; 2]>,
    devices: [i32; 2],
    lanes: u64,
    waves: usize,
    unit_rows: usize,
    lane_layouts: Vec<[LaneLayout; 2]>,
    lane_allocated: Vec<[bool; 2]>,
    lane_readers: Vec<[Weak<Stream<'a>>; 2]>,
    kda_layout: KdaLayout,
    kda_allocated: [bool; 2],
    kda_owners: [Weak<KdaBank<'a>>; 2],
    mla_layouts: [CacheLayout; 2],
    mla_allocated: [bool; 2],
    mla_owners: [Weak<CacheBank<'a>>; 2],
    global_layouts: [GlobalLayout; 2],
    global_uploaded: [bool; 2],
    global_owners: [Weak<GlobalWeights<'a>>; 2],
    io_layouts: [IoLayout; 2],
    io_allocated: Vec<[bool; 2]>,
    spark_connected: Vec<bool>,
}

/// Startup retains these until persistent local lanes acquire the same owners.
pub(crate) struct LoadedWeights<'a> {
    pub global: [Rc<GlobalWeights<'a>>; 2],
    pub layers: Vec<[Rc<LayerWeights<'a>>; 2]>,
}

/// The normal Spark lane consumes these selected layer IDs; selection still
/// goes through shared `SparkAdmission` before any transport connects.
impl<'a> Admission<'a> {
    /// Construct the final shared owner once, inside its admitted device scope.
    /// Failed construction frees partial storage and never retries allocation.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn allocate_draft(&mut self, device: Device<'a>,
        programs: &'a cuteafd_ffi::programs::Programs<'a>, snapshot: &std::path::Path,
        file: Vec<u8>, stream: *mut std::ffi::c_void,
        representation: cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation,
        scales: crate::shared::fp8_linear::Fp8Scales, rows: crate::shared::fp8_linear::Fp8Rows)
        -> Result<DeviceOwner<'a, DflashDrafter<'a>>> {
        self.allocate_draft_with(device.id, |layout| device.own(||
            DflashDrafter::load_mla(device.library, programs, "kimik3", snapshot, file, stream, layout,
                representation, scales, rows)))
    }

    fn check_draft(&self, device: i32) -> Result<&MlaDraftRuntimeLayout> {
        ensure!(device == self.devices[1] && !self.draft_allocated,
            "Kimi draft allocation differs from admitted rank or was already attempted");
        ensure!(!self.rtx_loaded && self.global_uploaded == [false; 2]
            && self.layer_uploaded.iter().all(|pair| *pair == [false; 2])
            && self.lane_allocated.iter().all(|pair| *pair == [false; 2])
            && self.io_allocated.iter().all(|pair| *pair == [false; 2])
            && self.kda_allocated == [false; 2] && self.mla_allocated == [false; 2],
            "Kimi drafter startup peak requires allocation before target residency");
        let layout = self.draft.as_ref().context("Kimi drafter was not admitted")?;
        ensure!(draft::home(&self.placement) == Some(1), "Kimi drafter must stay on target rank 1");
        let (demand, _) = layout.demands(1)?;
        for demand in demand.owned {
            let charges = self.placement.items[1].iter().filter(|item| item.group == demand.group).collect::<Vec<_>>();
            ensure!(charges.len() == 1 && charges[0].category == demand.category
                && charges[0].basis == demand.basis && charges[0].bytes == demand.bytes,
                "Kimi draft allocation differs from accepted {} charge", demand.group);
            ensure!(!self.placement.items[0].iter().any(|item| item.group == demand.group),
                "Kimi draft allocation has an unexecutable remote charge");
        }
        Ok(layout)
    }

    fn allocate_draft_with<T>(&mut self, device: i32,
        allocate: impl FnOnce(MlaDraftRuntimeLayout) -> Result<T>) -> Result<T> {
        let layout = self.check_draft(device)?.clone();
        // A partial native allocation must never become an uncharged retry.
        self.draft_allocated = true;
        allocate(layout)
    }
    pub(crate) fn allocate_draft_marks(&mut self, device: Device<'a>) -> Result<Rc<crate::shared::memory::device::Allocation<'a>>> {
        ensure!(device.id == self.devices[1] && !self.draft_marks_allocated, "Kimi draft mark allocation owner");
        let layout = self.draft.as_ref().context("Kimi draft marks without admitted draft")?;
        let bytes = layout.window_mark_bytes()?.checked_mul(usize::try_from(self.mark_selection.slots)?)
            .context("Kimi draft mark arena overflow")?;
        ensure!(self.placement.items[1].iter().filter(|item| item.group == DRAFT_MARKS)
            .map(|item| item.bytes).collect::<Vec<_>>() == vec![bytes as u64], "Kimi draft mark charge differs from arena");
        self.draft_marks_allocated = true;
        Ok(Rc::new(crate::shared::memory::device::Allocation::new(device, bytes.max(256))?))
    }
    pub(crate) fn draft_layout(&self) -> Option<&MlaDraftRuntimeLayout> { self.draft.as_ref() }

    /// Hops, the peer exchange and Spark waves serve one prefill unit.
    pub(crate) fn execution_plan(&self) -> Result<LayerExecutionPlan> {
        Ok(LayerExecutionPlan::new(&self.placement, hop_spec(self.cfg.hidden,
            self.unit_rows, self.lanes as usize))?)
    }

    pub(crate) fn working_set(&self) -> SparkWorkingSet {
        SparkWorkingSet { rtx_layers: self.placement.layers.iter().enumerate()
            .filter_map(|(layer, assignment)| (assignment.experts == ExpertHome::RtxTp2).then_some(layer)).collect(), spark_layers: self.placement.layers.iter().enumerate()
            .filter_map(|(layer, assignment)| (assignment.experts == ExpertHome::Spark).then_some(layer)).collect(),
            spark_ranks: 6, prefill_lanes: self.lanes }
    }

    /// The caller completes shared SparkAdmission reconciliation first. Keep
    /// one transport/intake owner per admitted lane; expired owners cannot be
    /// replaced by an uncharged second allocation.
    pub(crate) fn waves(&self) -> usize { self.waves }
    pub(crate) fn unit_rows(&self) -> usize { self.unit_rows }
    /// Rows of one prefill call: the lane's row planes (`waves` units).
    pub(crate) fn lane_rows(&self) -> usize { self.lane_layouts[0][0].rows() }

    pub(crate) fn connect_spark(&mut self, library: &'a std::sync::Arc<cuteafd_ffi::NativeLibrary>,
        local: &LocalLane<'a>,
        world: cuteafd_transport::expert::SparkWorld, peers: Vec<std::net::SocketAddr>,
        config: cuteafd_transport::TcpTransportConfig) -> Result<super::spark::RoutedLane<'a>> {
        let lane = local.lane_index();
        let expected = self.check_spark(lane)?;
        ensure!(Rc::ptr_eq(&expected, local.stream()), "Kimi Spark lane differs from admitted GPU1 intake owner");
        let routed = super::spark::RoutedLane::connect(library, local, &self.placement, world, peers, config,
            self.waves, self.unit_rows)?;
        self.spark_connected[lane] = true;
        Ok(routed)
    }

    fn check_spark(&self, lane: usize) -> Result<Rc<Stream<'a>>> {
        ensure!(lane < self.spark_connected.len() && !self.spark_connected[lane],
            "Kimi Spark lane already connected or outside admission");
        self.lane_readers[lane][1].upgrade().context("Kimi Spark requires admitted GPU1 intake stream")
    }

    /// Starts the Spark admission and loads every coordinator weight inside it:
    /// the dense operands (shared records) and the admitted TP2 expert halves
    /// join the Spark push group as two coordinator ranks when the admission
    /// pushes, planned from headers before membership (a planning error offers
    /// no coordinator ranks: the Sparks still push, the coordinator mounts).
    /// Returns the pending admission with the loaded owners.
    pub(crate) fn start_and_load(&mut self, spark: crate::shared::placement::SparkAdmission,
        peers: &[std::net::SocketAddr], devices: [Device<'a>; 2], checkpoint: &cuteafd_loader::plan::Checkpoint,
        catalog: &cuteafd_loader::OfficialV41Catalog, streams: [*mut std::ffi::c_void; 2])
        -> Result<(crate::shared::placement::PendingSparkAdmission,
            [Option<crate::shared::experts::rtx::exl3::Exl3Tp2Weights<'a>>; 2], LoadedWeights<'a>)> {
        use crate::shared::experts::rtx::{dense::{self, DenseLoad}, push::{CoordinatorExperts, CoordinatorPush}};
        check_prefix(&KimiK3Config::from_hf(&checkpoint.config)?, &self.cfg)?;
        for rank in 0..2 {
            ensure!(devices[rank].id == self.devices[rank], "Kimi load device differs from admission");
            self.check_global(rank, devices[rank].id)?;
            for layer in 0..self.cfg.layers { self.check_layer(rank, layer, devices[rank].id)?; }
        }
        let streams = [(devices[0].id, streams[0]), (devices[1].id, streams[1])];
        let rtx = self.rtx_plan(devices)?;
        let load = DenseLoad::push(devices[0].library);
        // Dense owners and records, planned before membership.
        let planned = if spark.coordinator_pushes() { (|| -> Result<_> {
            let owners = self.record(devices, checkpoint, &load, streams)?;
            let records = load.sink.as_ref().context("push sink")?.take_all();
            let (landed, operands, staged, steps, largest, finishes) = records.summary();
            tracing::info!(landed, operands, staged, steps, largest_step_bytes = largest, finishes,
                "Kimi coordinator dense records planned");
            let dense = dense::plan_records(&checkpoint.snapshot, records.clone(), &devices, peers.len())?;
            let experts = rtx.as_ref().map(|(layers, budgets, _)| CoordinatorExperts::Tp2 { layers: layers.clone(),
                budgets: budgets.to_vec() });
            let prepared = CoordinatorPush::prepare(&checkpoint.snapshot, Some(catalog), 2, experts)?;
            Ok((prepared, owners, records, dense))
        })().map_err(|error| tracing::info!(%error, "Kimi coordinator push unavailable; mount")).ok() } else { None };
        let (planned, ranks) = match planned {
            Some(((push, streams, ranks), owners, records, dense)) => (Some((push, owners, records, dense)), Some((streams, ranks))),
            None => (None, None),
        };
        let pending = spark.start_with(peers, ranks)?;
        let started = std::time::Instant::now();
        let (weights, pushed) = match planned {
            Some((push, owners, records, dense)) => {
                let (pushed, ()) = push.receive_with(devices.to_vec(), Some(catalog), peers.len(), Some(Ok(dense)), || ());
                let pushed = pushed?;
                if pushed.is_none() {
                    dense::fill_records_by_mount(devices[0].library, &records, |device| streams.iter()
                        .find(|(id, _)| *id == device).map_or(std::ptr::null_mut(), |(_, s)| *s))?;
                }
                tracing::info!(pushed = pushed.is_some(), operands = records.operands.len(), steps = records.steps.len(),
                    elapsed_ms = started.elapsed().as_millis() as u64, "Kimi coordinator dense operands landed");
                (owners, pushed)
            }
            None => {
                let mount = DenseLoad::mount(devices[0].library);
                let owners = self.record(devices, checkpoint, &mount, streams)?;
                mount.release();
                tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "Kimi coordinator dense operands mounted");
                (owners, None)
            }
        };
        let experts = self.finish_rtx(devices, catalog, rtx, pushed)?;
        Ok((pending, experts, self.register(devices, weights)?))
    }

    /// The admitted TP2 expert layers and per-rank weight budgets (workspace
    /// excluded), or none.
    #[allow(clippy::type_complexity)]
    fn rtx_plan(&self, devices: [Device<'a>; 2]) -> Result<Option<(Vec<usize>, [usize; 2], std::path::PathBuf)>> {
        let Some(inventory) = &self.rtx else { return Ok(None) };
        let layers = self.working_set().rtx_layers;
        if layers.is_empty() { return Ok(None); }
        let mut budgets = [0; 2];
        for rank in 0..2 {
            ensure!(devices[rank].id == self.devices[rank], "Kimi TP2 expert device differs from admission");
            let charged = usize::try_from(self.placement.tp2.as_ref()
                .context("Kimi missing admitted TP2 arena")?.peak_bytes[rank])?;
            let other_lanes = inventory.workspace.checked_mul(self.lanes as usize - 1)
                .context("Kimi TP2 lane workspace overflow")?;
            budgets[rank] = charged.checked_sub(other_lanes).context("Kimi TP2 workspace exceeds charge")?;
        }
        Ok(Some((layers, budgets, inventory.package.clone())))
    }

    /// Publishes the pushed TP2 halves, or loads them through the mount.
    #[allow(clippy::type_complexity)]
    fn finish_rtx(&mut self, devices: [Device<'a>; 2], catalog: &cuteafd_loader::OfficialV41Catalog,
        rtx: Option<(Vec<usize>, [usize; 2], std::path::PathBuf)>,
        pushed: Option<Vec<crate::shared::experts::rtx::push::PushedExperts<'a>>>)
        -> Result<[Option<crate::shared::experts::rtx::exl3::Exl3Tp2Weights<'a>>; 2]> {
        use crate::shared::experts::rtx::{exl3::Exl3Tp2, RtxSelectionPolicy};
        ensure!(!self.rtx_loaded, "Kimi TP2 experts already loaded");
        self.rtx_loaded = true;
        let Some((layers, budgets, package)) = rtx else { return Ok([None, None]) };
        // A TP2 FFN completes inside one unit's hook.
        let rows = self.unit_rows;
        let geometry = cuteafd_core::ExpertGeometry::KIMI_K3;
        let pair = match pushed {
            Some(ranks) => {
                let ranks = ranks.into_iter().map(|r| r.exl3()).collect::<Result<Vec<_>>>()?;
                let pushed: [_; 2] = ranks.try_into().map_err(|_| anyhow::anyhow!("two coordinator GPUs"))?;
                Exl3Tp2::pushed_pair_weights(devices, catalog, geometry, [&package, &package], layers, rows, budgets,
                    RtxSelectionPolicy::FirstCapacity, pushed)?
            }
            None => Exl3Tp2::load_pair_weights(devices, catalog, geometry, [&package, &package], layers, rows, budgets,
                RtxSelectionPolicy::FirstCapacity)?,
        };
        Ok(pair.map(Some))
    }

    /// Every global and layer owner through the shared records on `load`.
    #[allow(clippy::type_complexity)]
    fn record(&self, devices: [Device<'a>; 2], checkpoint: &cuteafd_loader::plan::Checkpoint,
        load: &crate::shared::experts::rtx::dense::DenseLoad<'a>, streams: [(i32, *mut std::ffi::c_void); 2])
        -> Result<([GlobalWeights<'a>; 2], Vec<[LayerWeights<'a>; 2]>)> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("weights");
        let records = super::weights::records::KimiRecords { checkpoint, cfg: &self.cfg, core: &self.core, dense: load, streams };
        let global = [records.global(devices[0], &self.global_layouts[0])?, records.global(devices[1], &self.global_layouts[1])?];
        let layers = (0..self.cfg.layers).map(|layer| records.layer(devices, layer, &self.layouts[layer]))
            .collect::<Result<Vec<_>>>()?;
        Ok((global, layers))
    }

    /// Registers loaded owners once, as the host-pack uploads do.
    fn register(&mut self, devices: [Device<'a>; 2], (global, layers): ([GlobalWeights<'a>; 2], Vec<[LayerWeights<'a>; 2]>))
        -> Result<LoadedWeights<'a>> {
        let mut out = Vec::new();
        for (rank, weights) in global.into_iter().enumerate() {
            self.check_global(rank, devices[rank].id)?;
            let weights = Rc::new(weights);
            self.global_owners[rank] = Rc::downgrade(&weights);
            self.global_uploaded[rank] = true;
            out.push(weights);
        }
        let mut owned = Vec::new();
        for (layer, pair) in layers.into_iter().enumerate() {
            let pair = pair.map(Rc::new);
            for rank in 0..2 {
                self.check_layer(rank, layer, devices[rank].id)?;
                self.layer_owners[layer][rank] = Rc::downgrade(&pair[rank]);
                self.layer_uploaded[layer][rank] = true;
            }
            owned.push(pair);
        }
        Ok(LoadedWeights { global: out.try_into().map_err(|_| anyhow::anyhow!("Kimi global rank count"))?, layers: owned })
    }

    fn check_layer(&self, rank: usize, layer: usize, device: i32) -> Result<&LayerLayout> {
        ensure!(rank < 2 && device == self.devices[rank], "Kimi upload device differs from admitted rank");
        let layout = &self.layouts.get(layer).context("Kimi upload layer outside admission")?[rank];
        ensure!(!self.layer_uploaded[layer][rank], "Kimi layer weights already uploaded");
        Ok(layout)
    }

    /// Bind once for the engine lifetime. No layer/step method drains this lane.
    pub(crate) fn bind_local(&mut self, rank: usize, lane: usize, allocated: Lane<'a>,
        programs: Rc<NativePrograms<'a>>) -> Result<LocalLane<'a>> {
        ensure!(rank < 2 && lane < self.lanes as usize && !self.local_bound[lane][rank],
            "Kimi local lane already bound or outside admission");
        programs.check_inventory(&self.cfg)?;
        let expected = self.lane_readers[lane][rank].upgrade().context("Kimi local binding requires admitted lane")?;
        ensure!(Rc::ptr_eq(&expected, &allocated.stream) && allocated.stream.device.id == self.devices[rank],
            "Kimi local binding differs from allocated lane owner");
        let layers = self.layer_owners.iter().map(|pair| pair[rank].upgrade()
            .context("Kimi local binding requires every admitted layer weight owner")).collect::<Result<Vec<_>>>()?;
        let global = self.global_owners[rank].upgrade().context("Kimi local binding requires admitted global weights")?;
        let kda = self.kda_owners[rank].upgrade().context("Kimi local binding requires admitted KDA bank")?;
        let mla = self.mla_owners[rank].upgrade().context("Kimi local binding requires admitted MLA bank")?;
        let binding = LocalLane::new(allocated, programs, self.cfg.clone(), layers, global, lane, rank, kda, mla)?;
        self.local_bound[lane][rank] = true;
        Ok(binding)
    }

    pub(crate) fn allocate_io(&mut self, device: Device<'a>, rank: usize, lane: usize) -> Result<TokenIo<'a>> {
        let layout = self.check_io(rank, lane, device.id)?;
        let weights = self.global_owners[rank].upgrade().context("Kimi IO requires live admitted global weights")?;
        let stream = self.lane_readers[lane][rank].upgrade().context("Kimi IO requires a live admitted lane stream")?;
        let io = layout.allocate(device, stream, weights)?;
        self.io_allocated[lane][rank] = true;
        Ok(io)
    }

    fn check_io(&self, rank: usize, lane: usize, device: i32) -> Result<&IoLayout> {
        ensure!(rank < 2 && device == self.devices[rank] && lane < self.lanes as usize,
            "Kimi IO allocation differs from admitted rank/device/lane");
        ensure!(!self.io_allocated[lane][rank], "Kimi token IO already allocated");
        let layout = &self.io_layouts[rank];
        let demand = layout.demand(lane)?;
        let charges = self.placement.items[rank].iter().filter(|item| item.group == demand.group).collect::<Vec<_>>();
        ensure!(charges.len() == 1 && charges[0].category == demand.category
            && charges[0].basis == Basis::Exact && charges[0].bytes == demand.bytes,
            "Kimi token IO allocation differs from accepted native charge");
        Ok(layout)
    }

    pub(crate) fn global_layout(&self, rank: usize) -> Result<&GlobalLayout> {
        self.global_layouts.get(rank).context("Kimi global rank outside TP2")
    }

    fn check_global(&self, rank: usize, device: i32) -> Result<&GlobalLayout> {
        ensure!(rank < 2 && device == self.devices[rank], "Kimi global upload differs from admitted device");
        ensure!(!self.global_uploaded[rank], "Kimi global weights already uploaded");
        let layout = &self.global_layouts[rank];
        let demand = layout.demand()?;
        let charges = self.placement.items[rank].iter().filter(|item| item.group == demand.group).collect::<Vec<_>>();
        ensure!(charges.len() == 1 && charges[0].category == demand.category
            && charges[0].basis == Basis::Exact && charges[0].bytes == demand.bytes,
            "Kimi global allocation differs from accepted native weight charge");
        Ok(layout)
    }

    pub(crate) fn allocate_lane(&mut self, device: Device<'a>, rank: usize, lane: usize) -> Result<Lane<'a>> {
        let layout = self.check_lane(rank, lane, device.id)?;
        let allocated = layout.allocate(device)?;
        self.lane_readers[lane][rank] = Rc::downgrade(&allocated.stream);
        self.lane_allocated[lane][rank] = true;
        Ok(allocated)
    }

    pub(crate) fn allocate_lane_on(&mut self, device: Device<'a>, rank: usize, lane: usize,
        stream: Rc<Stream<'a>>) -> Result<Lane<'a>> {
        let layout = self.check_lane(rank, lane, device.id)?;
        let allocated = layout.allocate_on(device, stream)?;
        self.lane_readers[lane][rank] = Rc::downgrade(&allocated.stream);
        self.lane_allocated[lane][rank] = true;
        Ok(allocated)
    }

    pub(crate) fn allocate_kda(&mut self, device: Device<'a>, rank: usize) -> Result<Rc<KdaBank<'a>>> {
        self.check_kda(rank, device.id)?;
        let readers = self.lane_readers.iter().map(|pair| pair[rank].upgrade()
            .context("Kimi KDA requires live admitted lane streams")).collect::<Result<Vec<_>>>()?;
        let bank = Rc::new(self.kda_layout.allocate(device, readers)?);
        self.kda_owners[rank] = Rc::downgrade(&bank);
        self.kda_allocated[rank] = true;
        Ok(bank)
    }

    pub(crate) fn allocate_mla(&mut self, device: Device<'a>, rank: usize) -> Result<Rc<CacheBank<'a>>> {
        let layout = self.check_mla(rank, device.id)?;
        let readers = self.lane_readers.iter().map(|pair| pair[rank].upgrade()
            .context("Kimi MLA requires live admitted lane streams")).collect::<Result<Vec<_>>>()?;
        let bank = Rc::new(layout.allocate(device, readers)?);
        self.mla_owners[rank] = Rc::downgrade(&bank);
        self.mla_allocated[rank] = true;
        Ok(bank)
    }

    fn check_mla(&self, rank: usize, device: i32) -> Result<&CacheLayout> {
        ensure!(rank < 2 && device == self.devices[rank], "Kimi MLA allocation differs from admitted device");
        ensure!(!self.mla_allocated[rank], "Kimi MLA bank already allocated");
        let layout = &self.mla_layouts[rank];
        let charges = self.placement.items[rank].iter().filter(|item| item.group == "records").collect::<Vec<_>>();
        ensure!(if layout.bytes() == 0 { charges.is_empty() } else {
            charges.len() == 1 && charges[0].category == Category::Kv
                && charges[0].basis == Basis::Formula && charges[0].bytes == layout.bytes() as u64
        }, "Kimi MLA allocation differs from accepted record charge");
        Ok(layout)
    }

    fn check_kda(&self, rank: usize, device: i32) -> Result<()> {
        ensure!(rank < 2 && device == self.devices[rank], "Kimi KDA allocation differs from admitted device");
        ensure!(!self.kda_allocated[rank], "Kimi KDA bank already allocated");
        for demand in self.kda_layout.demands(rank as u8)? {
            let charges = self.placement.items[rank].iter().filter(|item| item.group == demand.group).collect::<Vec<_>>();
            ensure!(charges.len() == 1 && charges[0].category == demand.category
                && charges[0].basis == Basis::Exact && charges[0].bytes == demand.bytes,
                "Kimi KDA allocation differs from accepted native pool charge");
        }
        Ok(())
    }

    fn check_lane(&self, rank: usize, lane: usize, device: i32) -> Result<&LaneLayout> {
        ensure!(rank < 2 && device == self.devices[rank] && lane < self.lanes as usize,
            "Kimi lane allocation differs from admitted rank/device/index");
        ensure!(!self.lane_allocated[lane][rank], "Kimi admitted lane already allocated");
        let layout = &self.lane_layouts[lane][rank];
        let demand = layout.demand(rank as u8, lane)?;
        let charges = self.placement.items[rank].iter().filter(|item| item.group == demand.group).collect::<Vec<_>>();
        ensure!(charges.len() == 1 && charges[0].category == demand.category
            && charges[0].basis == Basis::Exact && charges[0].bytes == demand.bytes,
            "Kimi lane allocation differs from accepted operand charge");
        Ok(layout)
    }
}

pub(crate) fn admit<'a>(mut inputs: Inputs<'_>) -> Result<Admission<'a>> {
    let cfg = inputs.cfg;
    let cache_format = inputs.cache_format;
    inputs.core.validate_runtime_groups(cfg)?;
    let core = inputs.core.clone();
    let devices = inputs.devices;
    let lanes = inputs.lanes as u64;
    let inputs_waves = inputs.waves;
    let unit_rows = cuteafd_loader::families::kimi_k3::allocation::prefill_shape_with(inputs.rows, inputs.waves,
        inputs.unit_rows)?.unit_rows as usize;
    let lane_layouts = inputs.lane_layouts.clone();
    let state_slots = inputs.state_slots;
    let legacy_marks = inputs.prefix_marks;
    let one_mark = KdaLayout::new(cfg, state_slots, 1, inputs.lanes)?;
    let mut mark_demands = (0..2).map(|rank| one_mark.demands(rank).map(|demands| demands[2].clone()))
        .collect::<Result<Vec<_>>>()?;
    let global_layouts = GlobalLayout::pair_with_core(cfg,&core)?;
    let io_layouts = [IoLayout::with_core(cfg, 0, inputs.rows, &core)?, IoLayout::with_core(cfg, 1, inputs.rows, &core)?];
    let rtx = inputs.rtx.take();
    let spark_budget = inputs.spark_budget.take();
    let draft = inputs.draft.clone();
    let graph_reserve = inputs.startup.iter().any(|d| d.group == super::graphs::RESERVE);
    let graph_rows = unit_rows;
    let graph_slots = inputs.state_slots;
    let graph_verify = inputs.verify_rows;
    let mut request = request(inputs)?;
    if let Some(layout) = &draft {
        let demand = Demand::new(1, Category::Prefix, DRAFT_MARKS, layout.window_mark_bytes()? as u64, Basis::Exact);
        if demand.bytes > 0 {
            let mut total = demand.clone();
            total.bytes = total.bytes.checked_mul(legacy_marks as u64).context("Kimi draft window marks overflow")?;
            request.fixed.push(total);
            mark_demands.push(demand);
        }
    }
    if let Some(inventory) = &rtx {
        ensure!(inventory.costs.len() == cfg.layers, "Kimi TP2 expert inventory extent");
        for (layer, cost) in request.layers.iter_mut().zip(&inventory.costs) {
            if let Some(experts) = &mut layer.experts { experts.half = *cost; experts.tp2 = true; }
        }
        let workspace = (inventory.workspace as u64).checked_mul(lanes).context("Kimi TP2 workspace overflow")?;
        request.tp2_workspace = [workspace; 2];
        request.onboard = inventory.layers;
        request.expert_gpus = 2;
        request.policy.expert_ownership = ExpertOwnership::Contiguous;
    }
    if let Some(budget) = &spark_budget { budget.constrain(&mut request)?; }
    let groups = request.layers.iter().map(|layer|
        (layer.kind == AttentionClass::Mla && request.attention_placement != Some(AttentionPlacement::Heads), layer.colocate))
        .collect::<Vec<_>>();
    let (placement, (mark_selection, graph_set, draft)) = attention::solve_owners(&groups,
        LayerOwnership::Ranges, request.layers_first_gpu, request.onboard, |owners| -> Result<_> {
            let mut candidate = request.clone();
            for (layer, (&owner, &(selected, _))) in candidate.layers.iter_mut().zip(owners.iter().zip(&groups)) {
                if selected { layer.modes = vec![LayerMode::Whole { gpu: owner, ffn: FfnMode::Split }]; }
            }
            let graphs = graph_reserve.then(|| super::graphs::inventory(cfg, lanes as usize, graph_rows,
                graph_slots, graph_verify, |rank, layer| !groups[layer].0 || usize::from(owners[layer]) == rank)).transpose()?;
            candidate.fixed.retain(|d| d.group != super::graphs::RESERVE);
            if let Some(graphs) = &graphs {
                candidate.fixed.extend((0..2).map(|rank| super::graphs::demand(&graphs.inventory, rank)));
            }
            let (placed, marks, layout) = if let Some(layout) = &draft {
                let initial = candidate.pool.requested.unwrap_or(candidate.pool.target);
                settle_pool((initial, ()), |tokens, _| -> Result<_> {
                    let mut trial = candidate.clone();
                    let layout = layout.with_pool_tokens(usize::try_from(tokens)?)?;
                    let (demand, peak) = layout.demands(1)?;
                    trial.movables = vec![demand.movable(2)?];
                    trial.startup.retain(|d| d.group != peak.group);
                    trial.startup.push(peak);
                    trial.pool.target = trial.pool.target.min(tokens);
                    trial.pool.ceiling = trial.pool.ceiling.min(tokens);
                    let (placed, marks) = solve_with_marks(&trial, legacy_marks as u64, 1, &mark_demands)?;
                    Ok((placed.pool_tokens, (), (placed, marks, Some(layout))))
                })?
            } else {
                let (placed, marks) = solve_with_marks(&candidate, legacy_marks as u64, 1, &mark_demands)?;
                (placed, marks, None)
            };
            if let Some(graphs) = &graphs { super::graphs::check_placement(cfg, &placed, &graphs.keys)?; }
            Ok((placed, (marks, graphs, layout)))
        })?;
    if let Some(budget) = &spark_budget {
        for rank in 0..budget.resident.len() {
            let bytes = budget.rank_bytes(rank, placement.onboard_layers)?;
            ensure!(bytes <= budget.budget, "Kimi selected Spark working set exceeds estimated budget");
            tracing::info!(rank, retained_bytes = bytes, budget_bytes = budget.budget,
                reserve = "estimated; pending load/steady census", "Kimi Spark budget selection");
        }
    }
    let kda_layout = KdaLayout::new(cfg, state_slots, usize::try_from(mark_selection.slots)?, lanes as usize)?;
    tracing::info!(selection = %mark_selection.summary(), "Kimi retained mark admission");
    EXECUTOR.check(&placement)?;
    let layouts = placement.layers.iter().enumerate().map(|(layer, assignment)|
        LayerLayout::pair_with_core(cfg, layer, assignment.mode, &core)).collect::<Result<Vec<_>>>()?;
    for rank in 0..2 {
        let native = layouts.iter().try_fold(0u64, |sum, pair| sum.checked_add(pair[rank].bytes()? as u64)
            .context("Kimi admitted native weight sum overflow"))?;
        let charged = placement.items[rank].iter().filter(|item|
            item.category == Category::Weights && (item.group == "layers" || item.group.starts_with("Kimi layer ")))
            .try_fold(0u64, |sum, item| sum.checked_add(item.bytes).context("Kimi weight charge overflow"))?;
        ensure!(native == charged, "Kimi rank {rank} native weights differ from shared admission: {native} != {charged}");
    }
    let layer_owners = (0..layouts.len()).map(|_| std::array::from_fn(|_| Weak::new())).collect();
    let layer_uploaded = vec![[false; 2]; layouts.len()];
    let local_bound = vec![[false; 2]; lane_layouts.len()];
    let lane_allocated = vec![[false; 2]; lane_layouts.len()];
    let io_allocated = vec![[false; 2]; lane_layouts.len()];
    let lane_readers = (0..lane_layouts.len()).map(|_| std::array::from_fn(|_| Weak::new())).collect();
    let mla_layouts = [CacheLayout::new(cfg, &placement, 0, cache_format)?, CacheLayout::new(cfg, &placement, 1, cache_format)?];
    let admission = Admission { placement, mark_selection, graph_set, rtx, rtx_loaded: false, draft, draft_allocated: false, draft_marks_allocated: false, cfg: cfg.clone(), core, layouts, layer_owners, layer_uploaded, local_bound,
        devices, lanes, waves: inputs_waves, unit_rows, lane_layouts, lane_allocated, lane_readers,
        kda_layout, kda_allocated: [false; 2], kda_owners: std::array::from_fn(|_| Weak::new()),
        mla_layouts, mla_allocated: [false; 2], mla_owners: std::array::from_fn(|_| Weak::new()),
        global_layouts, global_uploaded: [false; 2], global_owners: std::array::from_fn(|_| Weak::new()),
        io_layouts, io_allocated, spark_connected: vec![false; lanes as usize] };
    for rank in 0..2 {
        admission.check_kda(rank, devices[rank])?;
        admission.check_mla(rank, devices[rank])?;
        admission.check_global(rank, devices[rank])?;
        for lane in 0..lanes as usize { admission.check_io(rank, lane, devices[rank])?; }
    }
    if admission.draft.is_some() { admission.check_draft(devices[1])?; }
    Ok(admission)
}

fn hop_spec(hidden: usize, rows: usize, lanes: usize) -> HopSpec {
    HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: (hidden * 2) as u64,
        rows: rows as u64, lanes: lanes as u64, entry_gpu: 0, head_gpu: 1,
        terminal: TerminalHome::Replicated }
}

fn request(i: Inputs<'_>) -> Result<PlacementRequest> {
    let cfg = i.cfg;
    ensure!(cfg.layers > 0 && cfg.layers == cfg.attention.len() && cfg.layers == cfg.dense.len(),
        "Kimi admission layer inventory differs from checkpoint");
    ensure!(i.spark_ranks == 6 && i.peer_access, "Kimi native admission requires TP2 peer access and six Spark ranks");
    ensure!(i.devices.iter().all(|device| *device >= 0) && i.devices[0] != i.devices[1],
        "Kimi admitted devices must be distinct CUDA ordinals");
    ensure!(i.attention != Some(AttentionPlacement::Context), "Kimi context attention has no query/LSE exchange ABI");
    ensure!((1..=64).contains(&i.state_slots) && i.prefix_marks > 0
        && (1..=cuteafd_loader::families::kimi_k3::allocation::LANE_ROWS_MAX).contains(&i.rows)
        && (1..=2).contains(&i.lanes), "Kimi native admission slot/prefix/row/lane geometry");
    let unit = cuteafd_loader::families::kimi_k3::allocation::prefill_shape_with(i.rows, i.waves, i.unit_rows)?
        .unit_rows as usize;
    ensure!(cfg.kda_heads == 96 && cfg.kda_head_dim == 128 && cfg.kda_conv == 4
        && cfg.mla_record_values() * 2 == MLA_RECORD_BYTES, "Kimi native state/cache geometry");
    ensure!(i.gpus.iter().all(|gpu| matches!(gpu.baseline, Baseline::Measured { .. })),
        "Kimi runtime admission needs post-module CUDA samples");
    // Only startup graphs carry a temporary estimate pending the first serving
    // census. No weight, workspace, fixed or pool charge inherits this exception.
    ensure!(i.fixed.iter().all(|d| d.gpu < 2 && d.basis != Basis::Estimated)
        && i.startup.iter().all(|d| d.gpu < 2 && (d.basis != Basis::Estimated
            || (d.group == super::graphs::RESERVE && d.category == Category::Runtime
                && d.lifetime == cuteafd_core::memory_layout::ItemLifetime::ReadyResident)))
        && i.pool_bounded.iter().all(|d| d.gpu < 2 && d.basis != Basis::Estimated),
        "Kimi runtime admission allows estimates only for its named startup graph reserve");
    ensure!(i.lane_layouts.len() == i.lanes && i.lane_layouts.iter().flatten()
        .all(|layout| layout.rows() == i.rows && layout.unit == unit),
        "Kimi lane operand layouts differ from executor geometry");
    let context = i.lane_layouts[0][0].max_context();
    ensure!(i.lane_layouts.iter().flatten().all(|layout| layout.max_context() == context),
        "Kimi lanes must share the compiled context extent");
    if let Some(layout) = &i.draft {
        ensure!(layout.cache_format() == i.cache_format
            && layout.capacity().context_slots == i.state_slots
            && layout.capacity().max_batch_sequences <= i.rows.min(64)
            && layout.program().context == context
            && layout.cache().layers.iter().all(|layer| layer.slots() == i.state_slots),
            "Kimi drafter capacity differs from admitted KDA slots or lane context/rows");
        ensure!(layout.target_hidden() == cfg.hidden && layout.target_vocab() == cfg.vocab_size
            && layout.target_taps().iter().all(|&layer| layer < cfg.layers),
            "Kimi drafter taps, embedding or vocabulary differ from target inventory");
    }
    let mut fixed = i.fixed;
    // Solver startup phases are transient peaks. Captured executables survive
    // readiness, so retain their validated reserve beside the live KV pool.
    let (graphs, mut startup): (Vec<_>, Vec<_>) = i.startup.iter().cloned()
        .partition(|d| d.group == super::graphs::RESERVE);
    for graph in graphs {
        ensure!(!fixed.iter().any(|d| d.gpu == graph.gpu && d.group == graph.group),
            "Kimi startup graph reserve supplied twice");
        fixed.push(graph);
    }
    for layout in GlobalLayout::pair_with_core(cfg,&i.core)? {
        let demand = layout.demand()?;
        ensure!(!fixed.iter().chain(&i.startup).any(|item| item.gpu == demand.gpu && item.group == demand.group),
            "Kimi global weight charge supplied twice");
        fixed.push(demand);
    }
    for (lane, pair) in i.lane_layouts.iter().enumerate() {
        for (rank, layout) in pair.iter().enumerate() {
            let demand = layout.demand(rank as u8, lane)?;
            ensure!(!fixed.iter().chain(&i.startup).any(|item| item.gpu == demand.gpu && item.group == demand.group),
                "Kimi lane operand charge supplied twice");
            fixed.push(demand);
        }
    }
    for lane in 0..i.lanes {
        let demand = super::spark::demand_with(i.rows, i.waves, i.unit_rows, lane)?;
        ensure!(!fixed.iter().chain(&i.startup).any(|item| item.gpu == demand.gpu && item.group == demand.group),
            "Kimi Spark intake charge supplied twice");
        fixed.push(demand);
    }
    for rank in 0..2 {
        let layout = IoLayout::with_core(cfg, rank, i.rows, &i.core)?;
        for lane in 0..i.lanes {
            let demand = layout.demand(lane)?;
            ensure!(!fixed.iter().chain(&i.startup).any(|item| item.gpu == demand.gpu && item.group == demand.group),
                "Kimi token IO charge supplied twice");
            fixed.push(demand);
        }
    }
    let kda = KdaLayout::new(cfg, i.state_slots, i.prefix_marks, i.lanes)?;
    for rank in 0..2 {
        for demand in kda.demands(rank)? {
            ensure!(!fixed.iter().chain(&i.startup).any(|item| item.gpu == demand.gpu && item.group == demand.group),
                "Kimi KDA pool charge supplied twice");
            fixed.push(demand);
        }
    }
    let mut movables = Vec::new();
    if let Some(layout) = &i.draft {
        let (demand, peak) = layout.demands(1)?;
        ensure!(demand.owned.iter().chain(std::iter::once(&peak)).all(|draft|
            !fixed.iter().chain(&startup).any(|item| item.group == draft.group)),
            "Kimi drafter charge supplied twice");
        movables.push(demand.movable(2)?);
        startup.push(peak);
    }
    let mut layers = Vec::with_capacity(cfg.layers);
    for layer in 0..cfg.layers {
        let split = LayerLayout::pair_with_core(cfg, layer, LayerMode::HeadSplit, &i.core)?;
        let mut split_attention = [0; 2];
        // Whole describes attention residency, not the split FFN's residency.
        // Keep these always-resident native operands on both ranks, as GLM's
        // existing fixed target-weight inventory does for the same mode.
        for rank in 0..2 {
            let attention = split[rank].attention_bytes()?;
            split_attention[rank] = attention as u64;
            fixed.push(Demand::new(rank as u8, Category::Weights, format!("Kimi layer {layer} TP2 FFN/residual/router/latent"),
                (split[rank].bytes()? - attention) as u64, Basis::Exact));
        }
        let mla = cfg.attention[layer] == KimiAttention::Mla;
        let whole = if mla { LayerLayout::pair_with_core(cfg, layer, MODES[1], &i.core)?[0].attention_bytes()? as u64 } else { 0 };
        let unit = if mla { (MLA_PAGE_ROWS * i.cache_format.record_bytes()) as u64 } else { 0 };
        layers.push(LayerDemand { kind: if mla { AttentionClass::Mla } else { AttentionClass::Kda },
            weights: ModeBytes { whole, split: split_attention },
            kv_unit: KvDemand { unit_bytes_whole: unit, unit_bytes_split: [unit; 2], unit_bytes_context: None },
            fixed_bytes: ModeBytes::default(), extra: vec![], context_indexer: false, colocate: None,
            // admit() installs the checkpoint's TP2 rank-slice costs before solve.
            experts: (!cfg.dense[layer]).then_some(ExpertCost { whole: Bytes2::default(),
                half: [Bytes2::default(); 2], tp2: false, spark_ok: true }),
            modes: if mla { MODES.to_vec() } else { vec![LayerMode::HeadSplit] } });
    }
    let mut pool = PoolPolicy::resolve(&i.gpus.map(|gpu| gpu.capacity_bytes), context as u64,
        i.requested_pool, MLA_PAGE_ROWS as u64, true);
    pool.ceiling = pool.target.max(pool.requested.unwrap_or(0));
    Ok(PlacementRequest { inventory: Inventory { gpus: i.gpus.to_vec(), spark_ranks: i.spark_ranks, peer_access: i.peer_access },
        attention_placement: i.attention, context_buffers: ContextBuffers::default(), layers_first_gpu: 0,
        pool, layers, pool_overhead: vec![0; 2], pool_bounded: i.pool_bounded, fixed, startup, movables,
        expert_workspace: 0, tp2_workspace: [0; 2], onboard: Onboard::Layers(0), expert_gpus: 0, minimum_expert_layers: 0,
        policy: LayerPolicy { default: vec![LayerMode::HeadSplit], ownership: LayerOwnership::Ranges,
            attention: [(KimiAttention::Mla, AttentionClass::Mla, AttentionPlacement::Layers),
                (KimiAttention::Kda, AttentionClass::Kda, AttentionPlacement::Heads)].into_iter()
                .filter(|(kind, _, _)| cfg.attention.contains(kind))
                .map(|(_, class, placement)| (class, placement)).collect(),
            ..Default::default() },
        hops: hop_spec(cfg.hidden, unit, i.lanes), executor: EXECUTOR })
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// The unit admission derives for these test inputs.
    pub(in crate::families::kimi_k3) fn test_unit(rows: usize, waves: usize, unit: Option<usize>) -> usize {
        cuteafd_loader::families::kimi_k3::allocation::prefill_shape_with(rows, waves, unit).unwrap().unit_rows as usize
    }

    #[test]
    #[ignore = "CPU-only receipt replay; requires CUTEAFD_KIMI_MODEL, CUTEAFD_KIMI_DRAFT and CUTEAFD_KIMI_EXL3_PACKAGE"]
    fn recorded_runtime_owner_capacity() -> Result<()> {
        use super::super::{device::NativeScratch, startup, graphs};
        use std::path::PathBuf;
        cuteafd_core::set_expert_geometry(cuteafd_core::ExpertGeometry::KIMI_K3)
            .map_err(|_| anyhow::anyhow!("run receipt replay in an isolated process"))?;
        let path = |key| std::env::var_os(key).map(PathBuf::from).context(key);
        let cfg = KimiK3Config::read(&path("CUTEAFD_KIMI_MODEL")?)?;
        let catalog = cuteafd_loader::read_expert_catalog(&path("CUTEAFD_KIMI_MODEL")?)?;
        let package = path("CUTEAFD_KIMI_EXL3_PACKAGE")?;
        let working = SparkWorkingSet { rtx_layers: vec![], spark_layers: cfg.dense.iter().enumerate()
            .filter_map(|(layer, dense)| (!dense).then_some(layer)).collect(), spark_ranks: 6, prefill_lanes: 1 };
        let spark = SparkBudgetInventory::exl3(&catalog, &working,
            &[cuteafd_core::SparkEndpoint { ring_depth: None, capacity: 4096, ingress: cuteafd_core::SparkIngress::Fp8K32, execution_lane: 0 }],
            crate::shared::placement::spark_rings(cuteafd_transport::TcpTransportConfig::default().max_frame_bytes)?,
            cuteafd_loader::families::kimi_k3::SPARK_DEFAULT_BUDGET_BYTES,
            cuteafd_loader::families::kimi_k3::SPARK_ESTIMATED_RESERVE_BYTES)?;
        println!("spark minimum_onboard={} max_four_expert_bytes={}", spark.minimum_rtx_layers()?,
            (0..6).map(|rank| spark.rank_bytes(rank, 4)).collect::<Result<Vec<_>>>()?.into_iter().max().unwrap());
        let context = std::env::var("CUTEAFD_KIMI_REPLAY_CONTEXT").ok().map(|v| v.parse::<usize>()).transpose()?.unwrap_or(1 << 20);
        let draft = super::super::draft::Options { snapshot: path("CUTEAFD_KIMI_DRAFT")?, fp8: true, cache_format: cuteafd_core::MlaCacheFormat::Bf16 }.layout(8, context, 188)?;
        // Sealed program manifests and the failed card's post-module CUDA samples.
        // These are CPU arithmetic inputs, not a new hardware measurement.
        let scratch = NativeScratch { kda: 629_464_064, mla: [168_296_448, 101_056_512, 201_326_592] };
        let free = [101_298_995_200, 101_287_460_864];
        let total = [101_973_491_712, 101_970_345_984];
        for enabled in [false, true] {
            let make_inputs = || -> Result<Inputs<'_>> {
                let mut i = inputs(&cfg);
                i.gpus = std::array::from_fn(|rank| GpuBudget { capacity_bytes: total[rank],
                    headroom_bytes: 2 << 30, baseline: Baseline::Measured { free_bytes: free[rank] } });
                i.state_slots = 8; i.prefix_marks = 1; i.lanes = 1; i.requested_pool = Some(1 << 20);
                i.lane_layouts = vec![std::array::from_fn(|_| LaneLayout::new(&cfg, 4096, 4096, context, scratch).unwrap())];
                i.fixed = startup::exchange_demands(4096, false, false)?;
                for rank in 0..2 { i.fixed.push(startup::code_demand(rank, 71_303_168, 71_303_168, 1)?); }
                if enabled {
                    i.fixed.extend(super::super::draft::bridge_demands(&draft, i.rows)?);
                    i.fixed.push(cuteafd_loader::families::kimi_k3::draft_probe_demand(cfg.layers, cfg.topk, 64)?);
                    i.draft = Some(draft.with_pool_tokens(1 << 20)?);
                }
                Ok(i)
            };
            let mut automatic = make_inputs()?;
            automatic.requested_pool = None; automatic.prefix_marks = 2;
            automatic.spark_budget = Some(spark.clone());
            automatic.rtx = Some(plan_rtx(&cfg, &catalog, package.clone(), 4096, 5)?);
            let graphs = graphs::owned_inventory(&cfg, 1, 4096, 8, 0)?;
            automatic.startup = (0..2).map(|rank| graphs::demand(&graphs.inventory, rank)).collect();
            match admit(automatic) {
                Ok(admitted) => println!("automatic context={context} draft={enabled} pool={} owners={}/{} marks={}",
                    admitted.placement.pool_tokens,
                    admitted.placement.layers.iter().filter(|l| matches!(l.mode, LayerMode::Whole { gpu: 0, .. })).count(),
                    admitted.placement.layers.iter().filter(|l| matches!(l.mode, LayerMode::Whole { gpu: 1, .. })).count(),
                    admitted.mark_selection.slots),
                Err(error) => println!("automatic context={context} draft={enabled} refusal={error}"),
            }
            let rtx = plan_rtx(&cfg, &catalog, package.clone(), 4096, 5)?;
            let mut req = request(make_inputs()?)?;
            if enabled { req.fixed.push(Demand::new(1, Category::Prefix, DRAFT_MARKS,
                draft.window_mark_bytes()? as u64, Basis::Exact)); }
            for (layer, cost) in req.layers.iter_mut().zip(&rtx.costs) {
                if let Some(experts) = &mut layer.experts { experts.half = *cost; experts.tp2 = true; }
            }
            req.tp2_workspace = [rtx.workspace as u64; 2]; req.onboard = rtx.layers; req.expert_gpus = 2;
            req.policy.expert_ownership = ExpertOwnership::Contiguous;
            let groups = req.layers.iter().map(|l| (l.kind == AttentionClass::Mla, l.colocate)).collect::<Vec<_>>();
            for (scenario, onboard, pool_cap, drop_windows, drop_workspace) in if enabled {
                vec![("normal", 5, None, false, false), ("four_experts", 4, None, false, false),
                    ("half_draft_pool", 5, Some(524_288), false, false),
                    ("minimal_draft_pool", 5, Some(64), false, false),
                    ("no_window_marks", 5, None, true, false),
                    ("no_draft_workspace", 5, None, false, true)]
            } else { vec![("normal", 5, None, false, false)] } {
                for boundary in attention::owner_boundaries(&groups, false) {
                    let mut candidate = req.clone();
                    candidate.onboard = Onboard::Layers(onboard);
                    if drop_windows { candidate.fixed.retain(|d| d.group != DRAFT_MARKS); }
                    for (index, layer) in candidate.layers.iter_mut().enumerate() {
                        if groups[index].0 { layer.modes = vec![LayerMode::Whole { gpu: u8::from(index >= boundary), ffn: FfnMode::Split }]; }
                    }
                    let set = graphs::inventory(&cfg, 1, 4096, 8, 0,
                        |rank, layer| !groups[layer].0 || usize::from(layer >= boundary) == rank)?;
                    candidate.fixed.extend((0..2).map(|rank| graphs::demand(&set.inventory, rank)));
                    let rank0 = groups[..boundary].iter().filter(|g| g.0).count();
                    let strict = solve(&candidate);
                    if enabled && rank0 == 0 && scenario == "normal" && context == 1 << 20 {
                        let Err(PlacementError::PoolDoesNotFit { limits, .. }) = &strict else { panic!("{strict:?}") };
                        // The old early refusal was pre-expert. Restore its exact bytes
                        // to verify the recorded 937,792-token hardware receipt.
                        let peak = rtx.workspace as u64 + rtx.costs.iter().filter(|pair| pair[1].resident > 0)
                            .take(5).map(|pair| pair[1].resident).sum::<u64>()
                            + rtx.costs.iter().filter(|pair| pair[1].resident > 0).take(5)
                                .map(|pair| pair[1].staging).max().unwrap_or(0);
                        assert_eq!(limits.0[1].available_bytes + peak, 25_928_157_276);
                        assert_eq!((limits.0[1].available_bytes + peak) / 1_769_472 * 64, 937_792);
                    }
                    println!("strict scenario={scenario} draft={enabled} owners={rank0}/{} result={}", 24-rank0,
                        strict.as_ref().map(|p| p.pool_tokens.to_string()).unwrap_or_else(|e| e.to_string()));
                    // Find physical capacity with the selected compiled lane extents kept.
                    // Relax only the diagnostic pool floor, never production admission.
                    let mut low = 0; let mut high = (2 << 20) / 64;
                    let mut best = None;
                    while low < high {
                        let units = (low + high + 1) / 2;
                        let tokens = units * 64;
                        let mut trial = candidate.clone();
                        trial.pool.compiled_context = 0;
                        trial.pool.requested = Some(tokens); trial.pool.floor = tokens;
                        trial.pool.target = tokens; trial.pool.ceiling = tokens;
                        if enabled {
                            let layout = draft.with_pool_tokens(pool_cap.map_or(tokens as usize, |cap| cap.min(tokens as usize)))?;
                            let (mut demand, peak) = layout.demands(1)?;
                            if drop_workspace { demand.owned.retain(|d| d.group == "MLA DFlash owned weights"
                                || d.group.starts_with("MLA DFlash layer.")); }
                            trial.movables = vec![demand.movable(2)?]; trial.startup = vec![peak];
                        }
                        match solve(&trial) {
                            Ok(placed) => { low = units; best = Some(placed); }
                            Err(_) => high = units - 1,
                        }
                    }
                    println!("capacity scenario={scenario} draft={enabled} owners={rank0}/{} tokens={}", 24-rank0, low*64);
                    if let Some(placed) = best {
                        println!("items scenario={scenario} draft={enabled} owners={rank0}/{} {}", 24-rank0, serde_json::to_string(&placed.items)?);
                    }
                }
            }
        }
        Ok(())
    }

    fn cfg() -> KimiK3Config {
        let mut cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config()).unwrap();
        cfg.layers = 5;
        cfg.attention = vec![KimiAttention::Kda, KimiAttention::Mla, KimiAttention::Kda, KimiAttention::Mla, KimiAttention::Kda];
        cfg.dense.truncate(5);
        cfg
    }

    pub(in crate::families::kimi_k3) fn inputs(cfg: &KimiK3Config) -> Inputs<'_> {
        Inputs { cfg, core: Default::default(), cache_format: cuteafd_core::MlaCacheFormat::Bf16, gpus: [GpuBudget { capacity_bytes: 96 << 30, headroom_bytes: 2 << 30,
            baseline: Baseline::Measured { free_bytes: 96 << 30 } }; 2], devices: [0, 1], peer_access: true, spark_ranks: 6,
            attention: None, requested_pool: None, state_slots: 2, prefix_marks: 2, rows: 4096, lanes: 2, waves: 2, unit_rows: None,
            lane_layouts: vec![std::array::from_fn(|_| LaneLayout::new(
                &KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config()).unwrap(), 4096, 2048, 1_048_576,
                super::super::device::NativeScratch { kda: 1024, mla: [512, 0, 1024] }).unwrap()); 2],
            fixed: (0..2).map(|rank| Demand::new(rank, Category::Workspace, "native engine buffers", 1 << 20, Basis::Exact)).collect(),
            startup: vec![], pool_bounded: vec![], rtx: None, spark_budget: None, draft: None, verify_rows: 0 }
    }

    fn draft_layout(slots: usize, sequences: usize, context: usize) -> MlaDraftRuntimeLayout {
        use cuteafd_loader::{formats::dflash::MlaDflashConfig,
            families::glm5::draft_representation::{GlmDraftCapacity, GlmDraftRepresentation}};
        let config = serde_json::json!({"architectures":["DFlash2DraftModel"], "hidden_size":7168,
            "intermediate_size":14336, "num_hidden_layers":5, "num_attention_heads":64,
            "num_key_value_heads":64, "q_lora_rank":1536, "kv_lora_rank":512,
            "qk_nope_head_dim":128, "qk_rope_head_dim":64, "v_head_dim":128,
            "rms_norm_eps":1e-5, "vocab_size":163840, "draft_vocab_size":163840,
            "max_position_embeddings":1048576, "num_target_layers":93,
            "layer_types":["sliding_attention","sliding_attention","sliding_attention",
                "sliding_attention","full_attention"], "sliding_window":4096,
            "use_sliding_window":true, "is_causal":false, "mla_use_output_gate":false,
            "attention_bias":false, "hidden_act":"silu", "fc_norm":false,
            "rope_parameters":{"rope_type":"yarn", "rope_theta":50000.0, "factor":32.0,
                "original_max_position_embeddings":32768, "beta_fast":32, "beta_slow":1,
                "mscale":1.0, "mscale_all_dim":1.0},
            "dflash_config":{"attention_mode":"mla", "block_size":8,
                "conv_kernel_size":2, "conv_group_size":16, "selector_rank":256,
                "selector_top_k":16, "target_layer_ids":[0,1], "mask_token_id":163592}});
        MlaDflashConfig::from_json(&config).unwrap().runtime_layout(GlmDraftRepresentation::Fp8Only,
            GlmDraftCapacity::new(slots, sequences, 8).unwrap(), context, slots * context, 188, 2, cuteafd_core::MlaCacheFormat::Bf16).unwrap()
    }

    #[test]
    fn mla_draft_admission_pins_exact_inventory_and_allocates_once_before_target() -> Result<()> {
        let cfg = cfg();
        let full = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        for context in [262144, 1048576] {
            let mut i = inputs(&cfg);
            i.lane_layouts = vec![std::array::from_fn(|_| LaneLayout::new(&full, i.rows, test_unit(i.rows, i.waves, i.unit_rows), context,
                super::super::device::NativeScratch { kda: 1024, mla: [512, 0, 1024] }).unwrap()); i.lanes];
            let layout = draft_layout(i.state_slots, i.state_slots, context);
            let (demand, peak) = layout.demands(1)?;
            i.draft = Some(layout.clone());
            let request = request(i)?;
            assert_eq!(request.movables, [demand.movable(2)?]);
            assert!(request.startup.contains(&peak));
            assert!(!request.fixed.iter().any(|d| d.group == peak.group));
            let mut i = inputs(&cfg);
            i.lane_layouts = vec![std::array::from_fn(|_| LaneLayout::new(&full, i.rows, test_unit(i.rows, i.waves, i.unit_rows), context,
                super::super::device::NativeScratch { kda: 1024, mla: [512, 0, 1024] }).unwrap()); i.lanes];
            i.draft = Some(layout.clone());
            let mut admitted = admit(i)?;
            assert!(admitted.check_draft(0).is_err());
            let layout = layout.with_pool_tokens(usize::try_from(admitted.placement.pool_tokens)?)?;
            assert_eq!(admitted.check_draft(1)?, &layout);
            let allocated = admitted.allocate_draft_with(1, |owned| {
                assert_eq!(owned, layout);
                owned.allocate_with(|buffer| Ok((buffer.name.clone(), buffer.bytes)))
            })?;
            assert_eq!(allocated.iter().map(|(_, bytes)| *bytes as u64).sum::<u64>(), layout.buffer_bytes()?);
            assert!(admitted.check_draft(1).is_err());
        }
        let mut i = inputs(&cfg);
        i.draft = Some(draft_layout(i.state_slots, i.state_slots, 1048576));
        let mut admitted = admit(i)?;
        assert!(admitted.allocate_draft_with::<()>(1, |_| anyhow::bail!("injected partial allocation")).is_err());
        assert!(admitted.check_draft(1).is_err());
        Ok(())
    }

    #[test]
    fn owned_candidates_keep_matching_graphs_and_shared_draft_pool() -> Result<()> {
        let cfg = cfg();
        let mut i = inputs(&cfg);
        let seed = super::super::graphs::owned_inventory(&cfg, i.lanes, i.rows, i.state_slots, i.verify_rows)?;
        i.startup = (0..2).map(|rank| super::super::graphs::demand(&seed.inventory, rank)).collect();
        i.draft = Some(draft_layout(i.state_slots, i.state_slots, 1048576));
        let admitted = admit(i)?;
        let layout = admitted.draft_layout().unwrap();
        assert_eq!(layout.paged_cache().target_tokens as u64, admitted.placement.pool_tokens);
        assert_eq!(layout.paged_cache().full_bytes,
            admitted.placement.pool_tokens.div_ceil(64) * 64 * layout.paged_cache().record_bytes as u64);
        let set = admitted.graph_set.as_ref().context("winning owner graphs missing")?;
        super::super::graphs::check_placement(&cfg, &admitted.placement, &set.keys)?;
        for rank in 0..2 {
            let charge = admitted.placement.items[rank].iter().find(|item| item.group == super::super::graphs::RESERVE).unwrap();
            assert_eq!(charge.bytes, set.inventory.bytes(rank));
        }
        let marks = admitted.placement.items[1].iter().find(|item| item.group == DRAFT_MARKS).unwrap();
        assert_eq!(marks.bytes, layout.window_mark_bytes()? as u64 * admitted.mark_selection.slots);
        let owners = admitted.placement.layers.iter().filter_map(|layer| match layer.mode {
            LayerMode::Whole { gpu, .. } => Some(gpu), _ => None,
        }).collect::<Vec<_>>();
        assert!(owners.windows(2).filter(|pair| pair[0] != pair[1]).count() <= 1);
        Ok(())
    }

    #[test]
    fn mla_draft_rejects_wrong_inventory_duplicate_charge_and_late_allocation() -> Result<()> {
        let cfg = cfg();
        for layout in [draft_layout(1, 1, 1048576), draft_layout(2, 2, 262144)] {
            let mut i = inputs(&cfg); i.draft = Some(layout);
            assert!(request(i).is_err());
        }
        let mut mixed = inputs(&cfg);
        mixed.cache_format = cuteafd_core::MlaCacheFormat::Fp8;
        mixed.draft = Some(draft_layout(2, 2, 1048576));
        assert!(request(mixed).is_err(), "target/draft record mismatch must fail before admission");
        let mut i = inputs(&cfg);
        i.draft = Some(draft_layout(2, 2, 1048576));
        i.fixed.push(i.draft.as_ref().unwrap().demands(1)?.0.owned[0].clone());
        assert!(request(i).is_err());
        let mut wrong = cfg.clone(); wrong.vocab_size -= 1;
        let mut i = inputs(&wrong); i.draft = Some(draft_layout(2, 2, 1048576));
        assert!(request(i).is_err());
        let mut wrong = cfg.clone(); wrong.layers = 1; wrong.attention.truncate(1); wrong.dense.truncate(1);
        let mut i = inputs(&wrong); i.draft = Some(draft_layout(2, 2, 1048576));
        assert!(request(i).is_err());
        let mut i = inputs(&cfg); i.draft = Some(draft_layout(2, 2, 1048576));
        let mut admitted = admit(i)?;
        admitted.global_uploaded[0] = true;
        assert!(admitted.check_draft(1).is_err());
        admitted.global_uploaded[0] = false;
        admitted.placement.movables[0].1 = 0;
        assert!(admitted.check_draft(1).is_err());
        admitted.placement.movables[0].1 = 1;
        let charge = admitted.placement.items[1].iter_mut().find(|item| item.group == "MLA DFlash owned weights").unwrap();
        charge.bytes -= 1;
        assert!(admitted.check_draft(1).is_err());
        assert!(admit(inputs(&cfg))?.check_draft(1).is_err());
        Ok(())
    }

    #[test]
    fn tp2_rank_slices_are_admitted_on_both_gpus_and_removed_from_spark_selection() -> Result<()> {
        let cfg = cfg();
        let mut i = inputs(&cfg);
        i.rtx = Some(RtxInventory { package: "not-loaded".into(), costs: vec![[Bytes2 {
            resident: 1024 * 1024, staging: 4096 }; 2]; cfg.layers], workspace: 8192, layers: Onboard::Layers(3) });
        let admitted = admit(i)?;
        let local = admitted.working_set();
        assert_eq!(local.rtx_layers, [1, 2, 3]);
        assert_eq!(local.spark_layers, [4]);
        for layer in 1..=3 { assert_eq!(admitted.placement.layers[layer].experts, ExpertHome::RtxTp2); }
        for rank in 0..2 {
            assert_eq!(admitted.placement.tp2.as_ref().unwrap().peak_bytes[rank], 3 * 1024 * 1024 + 2 * 8192 + 4096);
        }
        Ok(())
    }

    #[test]
    fn automatic_tp2_admission_uses_complete_spark_budget_and_numeric_override_is_strict() -> Result<()> {
        let cfg = cfg();
        let make = |selection| {
            let mut i = inputs(&cfg);
            i.rtx = Some(RtxInventory { package: "not-loaded".into(), costs: vec![[Bytes2 {
                resident: 1024 * 1024, staging: 4096 }; 2]; cfg.layers], workspace: 8192, layers: selection });
            i.spark_budget = Some(SparkBudgetInventory { layers: vec![1, 2, 3, 4],
                resident: vec![vec![100; 4]; 6], overhead: vec![50; 6], budget: 250 });
            i
        };
        let admitted = admit(make(Onboard::Auto))?;
        assert!(admitted.placement.onboard_layers >= 2);
        assert!(admitted.placement.pool_tokens >= 1 << 20);
        assert!(admitted.mark_selection.slots > 0);
        let error = admit(make(Onboard::Layers(1))).err().context("strict override must refuse")?.to_string();
        assert!(error.contains("explicit --rtx-expert-layers 1") && error.contains("--spark-budget-gib"));
        Ok(())
    }

    #[test]
    fn automatic_and_explicit_pools_preserve_compiled_context() -> Result<()> {
        let cfg = cfg();
        for requested in [None, Some(0), Some(262_144), Some(2 << 20)] {
            let mut i = inputs(&cfg);
            i.requested_pool = requested;
            let request = request(i)?;
            assert_eq!(request.pool.target, 2 << 20);
            assert_eq!(request.pool.floor, requested.filter(|&n| n > 0).unwrap_or(1 << 20).max(1 << 20));
            if requested.is_some_and(|n| n > 0 && n < 1 << 20) {
                assert_eq!(solve(&request), Err(PlacementError::RequestedBelowContext {
                    requested: requested.unwrap(), floor: 1 << 20,
                }));
            }
            assert_eq!(request.pool.requested, requested.filter(|&n| n > 0));
        }
        let mut i = inputs(&cfg);
        let full = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        i.lane_layouts[1][1] = LaneLayout::new(&full, i.rows, test_unit(i.rows, i.waves, i.unit_rows), 524_288,
            super::super::device::NativeScratch { kda: 1024, mla: [512, 0, 1024] })?;
        assert!(request(i).is_err());
        Ok(())
    }

    #[test]
    fn installed_max_planner_and_runtime_remaining_kv_match_per_gpu() -> Result<()> {
        use cuteafd_loader::{families::kimi_k3::allocation, plan::{self, layout::LayoutOptions,
            testing, ExpertPlacement, PlanOptions}};
        use cuteafd_core::memory_layout::DeviceKind;
        let snapshot = tempfile::tempdir()?;
        testing::write_kimi_snapshot(snapshot.path(), true);
        let package_root = tempfile::tempdir()?;
        let manifest_path = testing::write_kimi_installed_inventory(package_root.path(), false);
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
        let cfg = KimiK3Config::from_hf(&testing::kimi_k3_config())?;
        let rows = 4096;
        let context = 1_048_576;
        let capacity = cuteafd_loader::placement::inventory::PRO_TOTAL_BYTES;
        let spark_bytes = 115_964_116_992;
        use cuteafd_core::MlaCacheFormat;
        for (slots, prefix_slots, host_cache_bytes, entries, format) in [
            (16, Some(1), Some(0), 20, MlaCacheFormat::Fp8),
            (8, None, None, 20, MlaCacheFormat::Fp8),
            (16, None, Some(8 << 30), 20, MlaCacheFormat::Fp8),
            (8, None, None, 20, MlaCacheFormat::Bf16),
        ] {
            let rule = crate::shared::prefix::MarkRule { lanes: slots as usize, entries,
                budget_bytes: 512 << 20, host: host_cache_bytes != Some(0) };
            let mark_bytes = cfg.kda_layers() as u64 * (cfg.kda_state_bytes() + cfg.kda_conv_bytes());
            let marks = prefix_slots.unwrap_or_else(|| rule.slots(usize::try_from(mark_bytes).unwrap()) as u64);
            assert_eq!(marks, if entries == 0 { 0 } else if rule.host { 2 } else { 1 });
            let options = PlanOptions { placement: ExpertPlacement::Sparks { ranks: 6 },
                layout: Some(LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![capacity; 2], prefill_rows: rows as u64 / 2, prefill_lanes: 2,
                    concurrency: slots, state_slots: Some(slots), prefix_slots,
                    mimo_prefix_entries: entries as u64, mimo_prefix_mark_bytes: rule.budget_bytes,
                    context_tokens: context as u64, host_cache_bytes, mla_kv: format,
                    workspace_manifest: Some(manifest_path.clone()), onboard: Some(Onboard::Auto),
                    spark_allocation_budget_bytes: Some(spark_bytes), ..Default::default() }), ..Default::default() };
            let report = plan::plan(snapshot.path(), &options)?;
            ensure!(report.fits && report.placement_supported, "{:?}", report.hints);
            let memory = report.memory_layout.context("installed Kimi planner ledger")?;
            let catalog = cuteafd_loader::read_expert_catalog(snapshot.path())?;
            let package = allocation::expert_package(&manifest_path)?;
            // Startup charges hook-local owners (TP2 workspace, Spark endpoint,
            // peer exchange, graphs) at the unit; row planes at the lane.
            let unit = test_unit(rows, 2, None);
            let mut rtx = plan_rtx(&cfg, &catalog, package, unit, 92)?;
            rtx.layers = Onboard::Auto;
            let working = SparkWorkingSet { rtx_layers: vec![], spark_layers: (1..93).collect(),
                spark_ranks: 6, prefill_lanes: 1 };
            let rings = cuteafd_loader::plan::layout::spark_ring_geometry(1);
            let spark = SparkBudgetInventory::exl3(&catalog, &working,
                &[cuteafd_core::SparkEndpoint { ring_depth: None, capacity: unit as u32, ingress: cuteafd_core::SparkIngress::Fp8K32, execution_lane: 0 }],
                cuteafd_core::SparkRings { max_frame_bytes: rings.max_frame_bytes, depth: rings.depth,
                    slot_bytes: rings.slot_bytes, alignment: rings.alignment }, spark_bytes,
                cuteafd_loader::families::kimi_k3::SPARK_ESTIMATED_RESERVE_BYTES)?;
            let programs = ProgramSet::from_manifest(&manifest, &["kimik3"]);
            let arch = ArchContext::for_device("sm_120", capacity);
            let mut i = inputs(&cfg);
            i.cache_format = format;
            i.rows = rows; i.lanes = 1; i.state_slots = slots as usize; i.prefix_marks = marks as usize;
            i.lane_layouts = vec![std::array::from_fn(|_| LaneLayout::new(&cfg, rows, test_unit(rows, i.waves, i.unit_rows), context,
                allocation::NativeScratch::from_manifest(&manifest).unwrap()).unwrap())];
            i.fixed = super::super::startup::exchange_demands(unit, false, false)?;
            i.startup = (0..2).map(|rank| super::super::graphs::demand(
                &super::super::graphs::owned_inventory(&cfg, 1, unit, slots as usize, 0).unwrap().inventory, rank)).collect();
            for rank in 0..2 {
                let code = cuteafd_loader::placement::inventory::loaded_code_allowance(
                    "kimik3", "exl3", true, rank as u8, arch, Some(&programs));
                // Model an identical post-module sample: context + resident code are
                // already allocated, while BLAS and private head handles are future.
                let resident = code.bytes - arch.cublas_bytes;
                i.gpus[rank] = GpuBudget { capacity_bytes: capacity, headroom_bytes: 2 << 30,
                    baseline: Baseline::Measured { free_bytes: capacity - arch.context_bytes - resident } };
                i.fixed.push(super::super::startup::code_demand(rank,
                    code.pending(arch.context_bytes + resident, arch.context_bytes), arch.cublas_bytes, 1)?);
            }
            i.rtx = Some(rtx); i.spark_budget = Some(spark);
            let admitted = admit(i)?;
            assert_eq!(memory.pool_tokens, admitted.placement.pool_tokens);
            assert_eq!(admitted.mark_selection.slots, marks);
            for rank in 0..2 {
                let planned = memory.devices.iter().find(|d| d.kind == DeviceKind::Rtx && d.index == rank as u32)
                    .context("planned RTX rank")?;
                let planned_marks = planned.items.iter().filter(|item| item.category == Category::Prefix)
                    .map(|item| item.bytes).sum::<u64>();
                assert_eq!(planned_marks, mark_bytes as u64 / 2 * marks,
                    "rank {rank}, concurrency {slots}, {format}: planner must charge runtime staging marks");
                let non_pool = planned.items.iter().filter(|item| item.group != "records").map(|item| item.bytes).sum::<u64>();
                let remaining = capacity - arch.context_bytes - non_pool;
                assert_eq!(remaining, admitted.placement.pool_limits.0[rank].available_bytes,
                    "rank {rank}, {format}: installed planner and runtime must leave identical KV room");
            }
            assert!(admitted.placement.pool_tokens >= context as u64);
        }
        Ok(())
    }

    #[test]
    fn planner_equals_runtime_context_extent_at_256k_512k_and_1m() -> Result<()> {
        use clap::Parser;
        use cuteafd_loader::plan::{self, layout::LayoutOptions, testing, ExpertPlacement, PlanOptions};
        let snapshot = tempfile::tempdir()?;
        testing::write_kimi_snapshot(snapshot.path(), true);
        let cfg = KimiK3Config::from_hf(&testing::kimi_k3_config())?;
        for format in [cuteafd_core::MlaCacheFormat::Bf16, cuteafd_core::MlaCacheFormat::Fp8] {
            let mut lane_bytes = Vec::new();
            let mut cache_bytes = Vec::new();
            for context in [262_144, 524_288, 1_048_576] {
                let text = context.to_string();
                let cli = crate::cli::Cli::try_parse_from(["cuteafd", "serve-kimi", "--snapshot", "/fixture",
                    "--native-lib", "/fixture.so", "--max-context", &text, "--pool-tokens", &text,
                    "--kv-cache", format.label()])?;
                let crate::cli::Commands::ServeKimi(args) = cli.command else { panic!("Kimi command") };
                let mut i = inputs(&cfg);
                i.cache_format = args.kv_cache;
                i.requested_pool = args.pool_tokens;
                i.lane_layouts = vec![std::array::from_fn(|_| LaneLayout::new(&cfg, i.rows, test_unit(i.rows, i.waves, i.unit_rows), args.max_context.unwrap(),
                    super::super::device::NativeScratch { kda: 1024, mla: [512, 0, 1024] }).unwrap()); i.lanes];
                lane_bytes.push(i.lane_layouts[0][0].bytes()?);
                let mut under = request(i)?;
                assert_eq!(under.pool.compiled_context, context);
                assert_eq!(under.pool.floor, context);
                under.pool.requested = Some(context - 64);
                assert_eq!(solve(&under), Err(PlacementError::RequestedBelowContext {
                    requested: context - 64, floor: context,
                }));
                let mut i = inputs(&cfg);
                i.cache_format = format;
                i.requested_pool = Some(context);
                for layouts in &mut i.lane_layouts {
                    for layout in layouts {
                        *layout = LaneLayout::new(&cfg, i.rows, test_unit(i.rows, i.waves, i.unit_rows), context as usize,
                            super::super::device::NativeScratch { kda: 1024, mla: [512, 0, 1024] })?;
                    }
                }
                let admitted = admit(i)?;
                let mut options = PlanOptions { placement: ExpertPlacement::Sparks { ranks: 6 },
                    layout: Some(LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![96 << 30; 2], concurrency: 2,
                        mla_kv: format, context_tokens: context, pool_tokens: Some(context), prefix_slots: Some(2),
                        host_cache_bytes: Some(0), onboard: Some(Onboard::Layers(5)), ..Default::default() }),
                    ..Default::default() };
                let report = plan::plan(snapshot.path(), &options)?;
                ensure!(report.fits && report.placement_supported, "{:?}", report.hints);
                let memory = report.memory_layout.context("Kimi planned layout")?;
                assert_eq!(memory.pool_tokens, admitted.placement.pool_tokens);
                assert_eq!(memory.pool_tokens, context);
                let mut total = 0;
                for rank in 0..2 {
                    let mla = super::super::mla::CacheLayout::new(&cfg, &admitted.placement, rank, format)?;
                    total += mla.bytes();
                    let planned = memory.devices.iter().find(|d| d.kind == cuteafd_core::memory_layout::DeviceKind::Rtx
                        && d.index == rank as u32).context("planned RTX")?;
                    let kv = |items: &[cuteafd_core::memory_layout::Item]| items.iter()
                        .filter(|item| item.category == Category::Kv).map(|item| item.bytes).sum::<u64>();
                    assert_eq!(kv(&planned.items), kv(&admitted.placement.items[rank]), "rank {rank}, context {context}, format {format}");
                }
                cache_bytes.push(total);
                options.layout.as_mut().unwrap().pool_tokens = Some(context - 64);
                let refused = plan::plan(snapshot.path(), &options)?;
                assert!(!refused.fits, "planner must not shrink the explicit context floor");
            }
            assert_eq!(lane_bytes[1] - lane_bytes[0], 1 << 20);
            assert_eq!(lane_bytes[2] - lane_bytes[0], 3 << 20);
            assert_eq!(cache_bytes[1] - cache_bytes[0], 24 * (524_288 - 262_144) * format.record_bytes());
            assert_eq!(cache_bytes[2] - cache_bytes[0], 24 * (1_048_576 - 262_144) * format.record_bytes());
        }
        Ok(())
    }

    #[test]
    fn automatic_pool_grows_beyond_default_context_floor() -> Result<()> {
        use cuteafd_loader::families::kimi_k3::DEFAULT_CONTEXT;
        use cuteafd_loader::plan::{self, layout::LayoutOptions, testing, ExpertPlacement, PlanOptions};
        let snapshot = tempfile::tempdir()?;
        testing::write_kimi_snapshot(snapshot.path(), true);
        let cfg = KimiK3Config::from_hf(&testing::kimi_k3_config())?;
        for format in [cuteafd_core::MlaCacheFormat::Bf16, cuteafd_core::MlaCacheFormat::Fp8] {
            let mut i = inputs(&cfg);
            i.cache_format = format;
            i.lane_layouts = vec![std::array::from_fn(|_| LaneLayout::new(&cfg, i.rows, test_unit(i.rows, i.waves, i.unit_rows), DEFAULT_CONTEXT,
                super::super::device::NativeScratch { kda: 1024, mla: [512, 0, 1024] }).unwrap()); i.lanes];
            let request = request(i)?;
            assert_eq!(request.pool.floor, DEFAULT_CONTEXT as u64);
            assert_eq!(request.pool.compiled_context, DEFAULT_CONTEXT as u64);
            assert!(request.pool.target > request.pool.floor);
            assert!(solve(&request)?.pool_tokens > request.pool.floor);
            let report = plan::plan(snapshot.path(), &PlanOptions {
                placement: ExpertPlacement::Sparks { ranks: 6 },
                layout: Some(LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![96 << 30; 2], concurrency: 2,
                    mla_kv: format, prefix_slots: Some(2), host_cache_bytes: Some(0),
                    onboard: Some(Onboard::Layers(5)), ..Default::default() }),
                ..Default::default()
            })?;
            ensure!(report.fits && report.placement_supported, "{:?}", report.hints);
            let memory = report.memory_layout.context("Kimi default planned layout")?;
            assert!(memory.pool_tokens > DEFAULT_CONTEXT as u64);
            assert!(memory.notes.iter().any(|note| note.contains("API max context: 524288 tokens; compiled max context: 1048576 tokens")));
        }
        Ok(())
    }

    #[test]
    fn diagnostic_prefix_admits_only_selected_layers_and_uses_shared_plan() -> Result<()> {
        let full = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        for layers in [1, 2, 4, 5, 12, 92, 93] {
            let selected = prefix(&full, layers)?;
            check_prefix(&full, &selected)?;
            let admitted = admit(inputs(&selected))?;
            assert_eq!(admitted.execution_plan()?.modes().len(), layers);
            assert_eq!(admitted.layouts.len(), layers);
            assert_eq!(admitted.cfg, selected);
            assert!(admitted.working_set().spark_layers.iter().all(|&layer| layer < layers));
        }
        for layers in [0, 94] { assert!(prefix(&full, layers).is_err()); }
        let mut changed = prefix(&full, 4)?;
        changed.rms_norm_eps *= 2.0;
        assert!(check_prefix(&full, &changed).is_err());
        let mut changed = prefix(&full, 4)?;
        changed.attention[0] = KimiAttention::Mla;
        assert!(check_prefix(&full, &changed).is_err());
        Ok(())
    }

    #[test]
    fn owned_mla_charges_nonowner_ffn_and_exact_native_weights() -> Result<()> {
        let cfg = cfg();
        let admitted = admit(inputs(&cfg))?;
        assert_eq!(admitted.placement.pool_tokens, 2 << 20);
        for (layer, owner) in [(1, 0), (3, 1)] {
            assert_eq!(admitted.placement.layers[layer].mode, LayerMode::Whole { gpu: owner, ffn: FfnMode::Split });
            let nonowner = usize::from(1 - owner);
            assert_eq!(admitted.layouts[layer][nonowner].attention_bytes()?, 0);
            let bytes = admitted.layouts[layer][nonowner].bytes()? as u64;
            assert!(bytes > 0);
            let group = format!("Kimi layer {layer} TP2 FFN/residual/router/latent");
            assert_eq!(admitted.placement.items[nonowner].iter().find(|item| item.group == group).unwrap().bytes, bytes);
        }
        let working = admitted.working_set();
        assert!(working.rtx_layers.is_empty());
        assert_eq!(working.spark_layers, cfg.dense.iter().enumerate().filter_map(|(layer, dense)| (!dense).then_some(layer)).collect::<Vec<_>>());
        assert_eq!(working.spark_ranks, 6);
        Ok(())
    }

    #[test]
    fn mixed_k6_weights_match_shared_admission_in_owned_and_split_modes() -> Result<()> {
        use cuteafd_loader::{families::kimi_k3::core::{CoreInventory,CoreProjection,split},
            formats::exl3_storage::Exl3Codebook};
        let cfg=cfg();
        let mut core=CoreInventory::default();
        for (layer,leaf,shape,stored_outputs) in [
            (0,"self_attn.q_proj",[12288,7168],12288),
            (0,"self_attn.k_proj",[12288,7168],12288),
            (0,"self_attn.v_proj",[12288,7168],12288),
            (0,"self_attn.g_proj",[12288,7168],12288),
            (0,"self_attn.o_proj",[7168,12288],7168),
            (1,"self_attn.q_a_proj",[1536,7168],1536),
            (1,"self_attn.q_b_proj",[18432,1536],18432),
            (1,"self_attn.g_proj",[12288,7168],12288),
            (1,"self_attn.kv_a_proj_with_mqa",[576,7168],640),
            (1,"self_attn.o_proj",[7168,12288],7168),
            (2,"block_sparse_moe.shared_experts.gate_proj",[6144,7168],6144),
            (2,"block_sparse_moe.shared_experts.up_proj",[6144,7168],6144),
            (2,"block_sparse_moe.shared_experts.down_proj",[7168,6144],7168),
            (1,"block_sparse_moe.routed_expert_down_proj",[3584,7168],3584),
            (1,"block_sparse_moe.routed_expert_up_proj",[7168,3584],7168),
            (2,"block_sparse_moe.routed_expert_down_proj",[3584,7168],3584),
            (2,"block_sparse_moe.routed_expert_up_proj",[7168,3584],7168)] {
            let stem=format!("language_model.model.layers.{layer}.{leaf}");
            core.projections.insert(stem.clone(),CoreProjection{split:split(&stem).unwrap(),stem,
                bits:6,codebook:Exl3Codebook::Mul1,shape,stored_outputs,source_bytes:0});
        }
        let stem="language_model.lm_head".to_owned();
        core.projections.insert(stem.clone(),CoreProjection{split:split(&stem).unwrap(),stem,bits:6,
            codebook:Exl3Codebook::Mul1,shape:[cfg.vocab_size,cfg.hidden],stored_outputs:cfg.vocab_size,source_bytes:0});
        for attention in [None,Some(AttentionPlacement::Heads)] {
            let mut i=inputs(&cfg);i.core=core.clone();i.attention=attention;
            let mixed=admit(i)?;
            let mut i=inputs(&cfg);i.attention=attention;
            let original=admit(i)?;
            let expected_global=GlobalLayout::pair_with_core(&cfg,&core)?;
            for rank in 0..2 {
                assert_eq!(mixed.global_layouts[rank],expected_global[rank]);
                assert_eq!(mixed.placement.items[rank].iter().find(|item|item.group=="Kimi global weights").unwrap().bytes,
                    expected_global[rank].bytes()? as u64);
                assert_eq!(mixed.io_layouts[rank],IoLayout::with_core(&cfg,rank,mixed.lane_layouts[0][rank].rows(),&core)?);
            }
            assert!(mixed.global_layouts[1].bytes()?<original.global_layouts[1].bytes()?);
            let mut freed=[0;2];
            for (layer,assignment) in mixed.placement.layers.iter().enumerate() {
                let expected=LayerLayout::pair_with_core(&cfg,layer,assignment.mode,&core)?;
                let baseline=LayerLayout::pair(&cfg,layer,assignment.mode)?;
                for rank in 0..2 {
                    assert_eq!(mixed.layouts[layer][rank],expected[rank]);
                    freed[rank]+=baseline[rank].bytes()? - expected[rank].bytes()?;
                }
            }
            for rank in 0..2 {
                let charge=|a:&Admission<'_>|a.placement.items[rank].iter().filter(|item|
                    item.category==Category::Weights && (item.group=="layers" || item.group.starts_with("Kimi layer ")))
                    .map(|item|item.bytes).sum::<u64>();
                let baseline_same_placement=mixed.placement.layers.iter().enumerate().map(|(layer,assignment)|
                    LayerLayout::pair(&cfg,layer,assignment.mode).unwrap()[rank].bytes().unwrap() as u64).sum::<u64>();
                assert_eq!(baseline_same_placement-charge(&mixed),freed[rank] as u64);
                assert!(freed[rank]>0);
                if attention.is_some() {assert_eq!(charge(&original)-charge(&mixed),freed[rank] as u64);}
            }
            // BF16 KDA fallback is selected by the inventory, with no mode flag.
            for rank in 0..2 {
                assert_eq!(mixed.layouts[2][rank].attention_bytes()?,
                    LayerLayout::pair(&cfg,2,LayerMode::HeadSplit)?[rank].attention_bytes()?);
            }
        }
        Ok(())
    }

    #[test]
    fn explicit_heads_preserves_ffns_and_replicates_mla_cache() -> Result<()> {
        let cfg = cfg();
        let owned = admit(inputs(&cfg))?;
        let mut i = inputs(&cfg);
        i.attention = Some(AttentionPlacement::Heads);
        let heads = admit(i)?;
        assert!(heads.placement.layers.iter().all(|layer| layer.mode == LayerMode::HeadSplit));
        let records = |admitted: &Admission| admitted.placement.items.iter().flatten()
            .filter(|item| item.group == "records").map(|item| item.bytes).sum::<u64>();
        assert_eq!(records(&heads), 2 * records(&owned));
        for rank in 0..2 {
            for category in [Category::Kv, Category::Workspace, Category::Prefix] {
                let state = |a: &Admission| a.placement.items[rank].iter().filter(|item|
                    item.category == category && item.group.starts_with("Kimi")).map(|item| item.bytes).sum::<u64>();
                assert_eq!(state(&heads), state(&owned));
            }
        }
        Ok(())
    }

    #[test]
    fn compact_cache_admission_and_owner_ranges_use_the_same_record() -> Result<()> {
        let cfg = cfg();
        for attention in [None, Some(AttentionPlacement::Heads)] {
            for format in [cuteafd_core::MlaCacheFormat::Bf16, cuteafd_core::MlaCacheFormat::Fp8] {
                let mut input = inputs(&cfg);
                input.cache_format = format;
                input.attention = attention;
                let admitted = admit(input)?;
                for rank in 0..2 {
                    let layout = admitted.check_mla(rank, rank as i32)?;
                    let layers = if attention.is_some() { 2 } else { 1 };
                    assert_eq!(layout.bytes(), layers * admitted.placement.pool_tokens as usize * format.record_bytes());
                    for layer in [1, 3] {
                        if let Ok((offset, bytes)) = layout.page_range(layer, 1, 17) {
                            assert_eq!(offset, 64 * format.record_bytes());
                            assert_eq!(bytes, 17 * format.record_bytes());
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn mla_cache_residency_and_pages_follow_accepted_attention_modes() -> Result<()> {
        let cfg = cfg();
        let owned = admit(inputs(&cfg))?;
        let mut i = inputs(&cfg);
        i.attention = Some(AttentionPlacement::Heads);
        let heads = admit(i)?;
        let tokens = owned.placement.pool_tokens as usize;
        let pages = tokens / MLA_PAGE_ROWS;
        let layer_bytes = tokens * MLA_RECORD_BYTES;
        for rank in 0..2 {
            let layout = owned.check_mla(rank, rank as i32)?;
            assert_eq!(layout.bytes(), layer_bytes);
            assert_eq!(heads.check_mla(rank, rank as i32)?.bytes(), 2 * layer_bytes);
            for layer in 0..cfg.layers {
                let resident = layer == 1 + 2 * rank;
                assert_eq!(layout.page_range(layer, 0, 64).is_ok(), resident);
                assert_eq!(heads.mla_layouts[rank].page_range(layer, 0, 64).is_ok(),
                    cfg.attention[layer] == KimiAttention::Mla);
                if !resident { continue; }
                let mut next = 0;
                for page in 0..pages {
                    let (offset, bytes) = layout.page_range(layer, page, 64)?;
                    assert_eq!(offset, next);
                    next += bytes;
                }
                assert_eq!(next, layer_bytes);
                assert_eq!(layout.page_range(layer, pages - 1, 1)?,
                    (layer_bytes - MLA_PAGE_ROWS * MLA_RECORD_BYTES, MLA_RECORD_BYTES));
                for (page, rows) in [(pages, 1), (0, 0), (0, 65)] {
                    assert!(layout.page_range(layer, page, rows).is_err());
                }
            }
            assert!(layout.page_range(cfg.layers, 0, 1).is_err());
        }
        let mut placement = owned.placement.clone();
        for invalid in [LayerMode::ContextSplit, LayerMode::Whole { gpu: 2, ffn: FfnMode::Split }] {
            placement.layers[1].mode = invalid;
            assert!(CacheLayout::new(&cfg, &placement, 0, cuteafd_core::MlaCacheFormat::Bf16).is_err());
        }
        placement = owned.placement.clone();
        for pool in [0, 65, u64::MAX] {
            placement.pool_tokens = pool;
            assert!(CacheLayout::new(&cfg, &placement, 0, cuteafd_core::MlaCacheFormat::Bf16).is_err());
        }
        assert!(CacheLayout::new(&cfg, &owned.placement, 2, cuteafd_core::MlaCacheFormat::Bf16).is_err());
        Ok(())
    }

    #[test]
    fn mla_allocation_rejects_duplicate_device_and_tampered_record_charges() -> Result<()> {
        let cfg = cfg();
        let mut admitted = admit(inputs(&cfg))?;
        assert!(admitted.check_mla(2, 0).is_err());
        assert!(admitted.check_mla(0, 1).is_err());
        admitted.mla_allocated[0] = true;
        assert!(admitted.check_mla(0, 0).is_err());
        admitted.mla_allocated[0] = false;
        for case in 0..5 {
            let mut admitted = admit(inputs(&cfg))?;
            let record = admitted.placement.items[0].iter_mut().find(|item| item.group == "records").unwrap();
            match case {
                0 => record.bytes -= 1,
                1 => record.category = Category::Workspace,
                2 => record.basis = Basis::Exact,
                3 => { let duplicate = record.clone(); admitted.placement.items[0].push(duplicate); }
                _ => admitted.placement.items[0].retain(|item| item.group != "records"),
            }
            assert!(admitted.check_mla(0, 0).is_err(), "case {case}");
        }
        Ok(())
    }

    #[test]
    fn spark_connections_cannot_replace_expired_admitted_owners() -> Result<()> {
        let cfg = cfg();
        let mut admitted = admit(inputs(&cfg))?;
        assert!(admitted.check_spark(admitted.spark_connected.len()).is_err());
        for lane in 0..admitted.spark_connected.len() {
            assert!(admitted.lane_readers[lane][1].upgrade().is_none());
            let missing = admitted.check_spark(lane).err().unwrap().to_string();
            assert!(missing.contains("requires admitted GPU1 intake stream"));
            admitted.spark_connected[lane] = true;
            let duplicate = admitted.check_spark(lane).err().unwrap().to_string();
            assert!(duplicate.contains("already connected"));
        }
        Ok(())
    }

    #[test]
    fn native_layer_uploads_are_single_owner_per_rank_even_after_owner_release() -> Result<()> {
        let cfg = cfg();
        let mut admitted = admit(inputs(&cfg))?;
        for layer in 0..cfg.layers {
            for rank in 0..2 { admitted.check_layer(rank, layer, rank as i32)?; }
        }
        for (rank, layer, device) in [(2, 0, 0), (0, cfg.layers, 0), (0, 0, 1)] {
            assert!(admitted.check_layer(rank, layer, device).is_err());
        }
        admitted.layer_uploaded[0][1] = true;
        assert!(admitted.layer_owners[0][1].upgrade().is_none());
        assert!(admitted.check_layer(1, 0, 1).is_err());
        let source = include_str!("admission.rs").split("#[cfg(test)]").next().unwrap();
        let bind = source.split("pub(crate) fn bind_local").nth(1).unwrap().split("pub(crate) fn allocate_io").next().unwrap();
        assert!(bind.contains("programs.check_inventory(&self.cfg)?"));
        assert!(bind.contains("Rc::ptr_eq(&expected, &allocated.stream)"));
        assert!(bind.contains("self.layer_owners.iter()"));
        assert!(bind.contains("self.global_owners[rank].upgrade()"));
        assert!(bind.contains("self.kda_owners[rank].upgrade()"));
        assert!(bind.contains("self.mla_owners[rank].upgrade()"));
        assert!(bind.contains("!self.local_bound[lane][rank]"));
        for rank in 0..2 {
            assert!(admitted.kda_owners[rank].upgrade().is_none());
            assert!(admitted.mla_owners[rank].upgrade().is_none());
            admitted.kda_allocated[rank] = true;
            admitted.mla_allocated[rank] = true;
            assert!(admitted.check_kda(rank, rank as i32).is_err());
            assert!(admitted.check_mla(rank, rank as i32).is_err());
        }
        Ok(())
    }

    #[test]
    fn global_weight_admission_rejects_duplicate_and_altered_inventory() -> Result<()> {
        let cfg = cfg();
        let mut admitted = admit(inputs(&cfg))?;
        assert_eq!(admitted.global_layout(0)?.bytes()?, cfg.vocab_size * cfg.hidden * 2);
        assert_eq!(admitted.global_layout(1)?.bytes()?, cfg.vocab_size * cfg.hidden * 2 + cfg.hidden * 6);
        admitted.check_global(0, 0)?;
        admitted.check_global(1, 1)?;
        assert!(admitted.check_global(2, 0).is_err());
        assert!(admitted.check_global(0, 1).is_err());
        admitted.global_uploaded[0] = true;
        assert!(admitted.check_global(0, 0).is_err());
        for case in 0..5 {
            let mut admitted = admit(inputs(&cfg))?;
            let charge = admitted.placement.items[1].iter_mut().find(|item| item.group == "Kimi global weights").unwrap();
            match case {
                0 => charge.bytes -= 1,
                1 => charge.category = Category::Workspace,
                2 => charge.basis = Basis::Formula,
                3 => { let duplicate = charge.clone(); admitted.placement.items[1].push(duplicate); }
                _ => admitted.placement.items[1].retain(|item| item.group != "Kimi global weights"),
            }
            assert!(admitted.check_global(1, 1).is_err(), "case {case}");
        }
        let mut i = inputs(&cfg);
        i.fixed.push(GlobalLayout::pair(&cfg)?[0].demand()?);
        assert!(admit(i).is_err());
        Ok(())
    }

    #[test]
    fn token_io_admission_rejects_duplicate_device_and_charge_mismatch() -> Result<()> {
        let cfg = cfg();
        let mut admitted = admit(inputs(&cfg))?;
        for rank in 0..2 {
            for lane in 0..2 { admitted.check_io(rank, lane, rank as i32)?; }
        }
        for (rank, lane, device) in [(2, 0, 0), (0, 2, 0), (0, 0, 1)] {
            assert!(admitted.check_io(rank, lane, device).is_err());
        }
        admitted.io_allocated[0][1] = true;
        assert!(admitted.check_io(1, 0, 1).is_err());
        admitted.io_allocated[0][1] = false;
        let group = admitted.io_layouts[1].demand(0)?.group;
        admitted.placement.items[1].iter_mut().find(|item| item.group == group).unwrap().bytes -= 1;
        assert!(admitted.check_io(1, 0, 1).is_err());
        let mut i = inputs(&cfg);
        i.fixed.push(IoLayout::new(&cfg, 1, i.rows)?.demand(0)?);
        assert!(admit(i).is_err());
        Ok(())
    }

    #[test]
    fn lane_allocations_require_immutable_accepted_rank_extents() -> Result<()> {
        let cfg = cfg();
        let mut admitted = admit(inputs(&cfg))?;
        let expected = admitted.lane_layouts[0][0].bytes()?;
        assert_eq!(admitted.check_lane(0, 0, 0)?.bytes()?, expected);
        for (rank, lane, device) in [(2, 0, 0), (0, 2, 0), (0, 0, 1), (1, 0, -1)] {
            assert!(admitted.check_lane(rank, lane, device).is_err());
        }
        admitted.lane_allocated[0][0] = true;
        assert!(admitted.check_lane(0, 0, 0).is_err());
        admitted.lane_allocated[0][0] = false;
        let group = admitted.lane_layouts[0][0].demand(0, 0)?.group;
        admitted.placement.items[0].iter_mut().find(|item| item.group == group).unwrap().bytes -= 1;
        assert!(admitted.check_lane(0, 0, 0).is_err());
        let mut i = inputs(&cfg);
        i.fixed.push(i.lane_layouts[0][0].demand(0, 0)?);
        assert!(admit(i).is_err());
        let mut i = inputs(&cfg);
        i.lane_layouts.pop();
        assert!(admit(i).is_err());
        Ok(())
    }

    #[test]
    fn spark_intake_planes_are_charged_once_per_lane_before_allocation() -> Result<()> {
        let cfg = cfg();
        let i = inputs(&cfg);
        let rows = i.rows;
        let lanes = i.lanes;
        let admitted = admit(i)?;
        for lane in 0..lanes {
            let demand = super::super::spark::demand(rows, 2, lane)?;
            let items = admitted.placement.items[1].iter().filter(|item| item.group == demand.group).collect::<Vec<_>>();
            assert_eq!(items.len(), 1);
            assert_eq!(items[0].bytes, (6 * rows * 3584 * 2) as u64);
            assert_eq!(items[0].basis, Basis::Exact);
            assert_eq!(items[0].category, Category::Workspace);
            assert!(!admitted.placement.items[0].iter().any(|item| item.group == demand.group));
        }
        let mut duplicate = inputs(&cfg);
        duplicate.fixed.push(super::super::spark::demand(rows, 2, 0)?);
        assert!(admit(duplicate).is_err());
        Ok(())
    }

    #[test]
    fn kda_pool_allocation_requires_exact_accepted_lane_isolated_charges() -> Result<()> {
        let cfg = cfg();
        let mut admitted = admit(inputs(&cfg))?;
        admitted.check_kda(0, 0)?;
        admitted.check_kda(1, 1)?;
        assert!(admitted.check_kda(2, 0).is_err());
        assert!(admitted.check_kda(0, 1).is_err());
        admitted.kda_allocated[0] = true;
        assert!(admitted.check_kda(0, 0).is_err());
        admitted.kda_allocated[0] = false;
        let replay = admitted.placement.items[0].iter_mut().find(|item| item.group == "Kimi KDA replay").unwrap();
        assert_eq!(replay.bytes, 3 * 2 * cuteafd_ffi::KimiKda::replay_bytes() as u64);
        replay.bytes /= 2;
        assert!(admitted.check_kda(0, 0).is_err());
        let mut i = inputs(&cfg);
        i.fixed.push(KdaLayout::new(&cfg, i.state_slots, i.prefix_marks, i.lanes)?.demands(0)?[0].clone());
        assert!(admit(i).is_err());
        Ok(())
    }

    #[test]
    fn only_named_startup_graph_reserve_may_be_estimated() -> Result<()> {
        let cfg = cfg();
        let inventory = super::super::graphs::owned_inventory(&cfg, 1, 64, 4, 0)?;
        let reserve = super::super::graphs::demand(&inventory.inventory, 0);
        let mut i = inputs(&cfg);
        i.startup.push(reserve.clone());
        let admitted = admit(i)?;
        assert!(admitted.placement.items[0].iter().any(|item|
            item.group == super::super::graphs::RESERVE && item.basis == Basis::Estimated));
        for case in 0..7 {
            let mut i = inputs(&cfg);
            let mut changed = reserve.clone();
            match case {
                0 => changed.group = "unqualified startup allocation".into(),
                1 => changed.category = Category::Workspace,
                2 => changed.category = Category::Weights,
                3 => changed.gpu = 2,
                4 => changed.lifetime = cuteafd_core::memory_layout::ItemLifetime::PostReadyReserve,
                5 => { i.fixed.push(changed); assert!(admit(i).is_err()); continue; }
                _ => {
                    i.pool_bounded.push(BoundedDemand { gpu: 0, category: Category::Runtime,
                        group: super::super::graphs::RESERVE.into(), bytes_per_unit: 1,
                        offset_units: 0, cap_units: 1, basis: Basis::Estimated });
                    assert!(admit(i).is_err()); continue;
                }
            }
            i.startup.push(changed);
            assert!(admit(i).is_err(), "case {case}");
        }
        Ok(())
    }

    #[test]
    fn refuses_unsupported_or_unmeasured_runtime_inventory() {
        let cfg = cfg();
        for case in 0..10 {
            let mut i = inputs(&cfg);
            match case {
                0 => i.spark_ranks = 4,
                1 => i.peer_access = false,
                2 => i.attention = Some(AttentionPlacement::Context),
                3 => i.requested_pool = Some(262_143),
                4 => i.prefix_marks = 0,
                5 => i.gpus[0].baseline = Baseline::Planned { context_bytes: 0, loaded_bytes: 0 },
                6 => i.fixed[0].basis = Basis::Estimated,
                7 => i.fixed[0].gpu = 2,
                8 => i.devices = [1, 1],
                _ => i.devices[0] = -1,
            }
            assert!(admit(i).is_err(), "case {case}");
        }
        let mut i = inputs(&cfg);
        i.requested_pool = Some(2 << 20);
        i.gpus[1].baseline = Baseline::Measured { free_bytes: 2 << 30 };
        assert!(admit(i).is_err());
    }
}
