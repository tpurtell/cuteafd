//! Native coordinator inventory passed to the shared startup solver, before
//! weights, recurrent state or cache owners allocate. This is not the header
//! lower bound exposed by `plan` and does not admit Spark workers by itself.
use super::{device::{IoLayout, TokenIo, Lane, LaneLayout, LocalLane, NativePrograms}, mla::{CacheBank, CacheLayout}, state::{KdaBank, KdaLayout}, weights::{GlobalLayout, GlobalWeights, PackedGlobal, LayerLayout, LayerWeights, PackedLayer}};
use crate::shared::memory::device::{Device, Stream};
use std::rc::{Rc, Weak};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{Basis, Category};
use cuteafd_ffi::{MLA_PAGE_ROWS, MLA_RECORD_BYTES};
use cuteafd_loader::{families::kimi_k3::config::{KimiAttention, KimiK3Config}, placement::*};

const MODES: &[LayerMode] = &[LayerMode::HeadSplit,
    LayerMode::Whole { gpu: 0, ffn: FfnMode::Split }, LayerMode::Whole { gpu: 1, ffn: FfnMode::Split }];
const EXECUTOR: ExecutorModes = ExecutorModes { family: "kimi_k3", modes: MODES,
    hops: true, attention: AttentionPlacement::Layers };
const FLOOR: u64 = 262_144;

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
    pub lane_layouts: Vec<[LaneLayout; 2]>,
    pub fixed: Vec<Demand>,
    pub startup: Vec<Demand>,
    pub pool_bounded: Vec<BoundedDemand>,
    pub rtx: Option<RtxInventory>,
}

/// CPU-planned native residency; no backend owner allocates before solve.
pub(crate) struct RtxInventory {
    pub package: std::path::PathBuf,
    pub costs: Vec<Bytes2>,
    pub workspace: usize,
    pub layers: usize,
}

pub(crate) fn plan_rtx(cfg: &KimiK3Config, catalog: &cuteafd_loader::OfficialV41Catalog,
    package: std::path::PathBuf, rows: usize, layers: usize) -> Result<RtxInventory> {
    use crate::shared::experts::{exl3::Exl3Weights, layer::ExpertLayer,
        rtx::{exl3::Exl3Tp2, RtxSelectionPolicy}};
    ensure!(layers > 0 && layers <= cfg.dense.iter().filter(|dense| !**dense).count(),
        "Kimi whole expert count outside routed inventory");
    let workspace = Exl3Tp2::whole_workspace_bytes(&package, rows,
        cuteafd_core::ExpertGeometry::KIMI_K3, RtxSelectionPolicy::FirstCapacity)?;
    let costs = cfg.dense.iter().enumerate().map(|(layer, dense)| {
        if *dense { return Ok(Bytes2::default()); }
        let plan = Exl3Weights::plan(catalog, ExpertLayer::BackboneFull { layer })?;
        Ok(Bytes2 { resident: plan.resident_bytes as u64, staging: plan.device_staging_bytes as u64 })
    }).collect::<Result<Vec<_>>>()?;
    Ok(RtxInventory { package, costs, workspace, layers })
}

pub(crate) struct Admission<'a> {
    pub placement: Placement,
    cfg: KimiK3Config,
    rtx: Option<RtxInventory>,
    rtx_loaded: bool,
    layouts: Vec<[LayerLayout; 2]>,
    layer_owners: Vec<[Weak<LayerWeights<'a>>; 2]>,
    layer_uploaded: Vec<[bool; 2]>,
    local_bound: Vec<[bool; 2]>,
    devices: [i32; 2],
    lanes: u64,
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
    pub(crate) fn execution_plan(&self) -> Result<LayerExecutionPlan> {
        Ok(LayerExecutionPlan::new(&self.placement, hop_spec(self.cfg.hidden,
            self.lane_layouts[0][0].rows(), self.lanes as usize))?)
    }

    pub(crate) fn working_set(&self) -> SparkWorkingSet {
        SparkWorkingSet { rtx_layers: self.placement.layers.iter().enumerate()
            .filter_map(|(layer, assignment)| matches!(assignment.experts, ExpertHome::RtxWhole { .. }).then_some(layer)).collect(), spark_layers: self.placement.layers.iter().enumerate()
            .filter_map(|(layer, assignment)| (assignment.experts == ExpertHome::Spark).then_some(layer)).collect(),
            spark_ranks: 6, prefill_lanes: self.lanes }
    }

    pub(crate) fn load_rtx(&mut self, devices: [Device<'a>; 2],
        catalog: &cuteafd_loader::OfficialV41Catalog)
        -> Result<[Option<crate::shared::experts::rtx::exl3::Exl3Tp2Weights<'a>>; 2]> {
        use crate::shared::experts::rtx::{exl3::Exl3Tp2, RtxSelectionPolicy};
        ensure!(!self.rtx_loaded, "Kimi whole experts already loaded");
        self.rtx_loaded = true;
        let mut owners = [None, None];
        let Some(inventory) = &self.rtx else { return Ok(owners); };
        for rank in 0..2 {
            let layers = self.placement.layers.iter().enumerate().filter_map(|(layer, assignment)|
                (assignment.experts == ExpertHome::RtxWhole { gpu: rank as u8 }).then_some(layer)).collect::<Vec<_>>();
            if layers.is_empty() { continue; }
            ensure!(devices[rank].id == self.devices[rank], "Kimi whole expert device differs from admission");
            let charged = usize::try_from(self.placement.expert_ranges[rank].peak_bytes)?;
            let other_lanes = inventory.workspace.checked_mul(self.lanes as usize - 1)
                .context("Kimi whole lane workspace overflow")?;
            owners[rank] = Some(Exl3Tp2::load_whole_weights(devices[rank], catalog,
                &inventory.package, layers, self.lane_layouts[0][rank].rows(),
                charged.checked_sub(other_lanes).context("Kimi whole workspace exceeds charge")?,
                RtxSelectionPolicy::FirstCapacity)?);
        }
        Ok(owners)
    }

    /// The caller completes shared SparkAdmission reconciliation first. Keep
    /// one transport/intake owner per admitted lane; expired owners cannot be
    /// replaced by an uncharged second allocation.
    pub(crate) fn connect_spark(&mut self, library: &'a std::sync::Arc<cuteafd_ffi::NativeLibrary>,
        local: &LocalLane<'a>,
        world: cuteafd_transport::expert::SparkWorld, peers: Vec<std::net::SocketAddr>,
        config: cuteafd_transport::TcpTransportConfig) -> Result<super::spark::RoutedLane<'a>> {
        let lane = local.lane_index();
        let expected = self.check_spark(lane)?;
        ensure!(Rc::ptr_eq(&expected, local.stream()), "Kimi Spark lane differs from admitted entry owner");
        let routed = super::spark::RoutedLane::connect(library, local, &self.placement, world, peers, config)?;
        self.spark_connected[lane] = true;
        Ok(routed)
    }

    fn check_spark(&self, lane: usize) -> Result<Rc<Stream<'a>>> {
        ensure!(lane < self.spark_connected.len() && !self.spark_connected[lane],
            "Kimi Spark lane already connected or outside admission");
        self.lane_readers[lane][0].upgrade().context("Kimi Spark requires admitted entry stream")
    }

    /// Stream one packed layer pair at a time after shared admission. Each
    /// source tensor is read once for both ranks; no routed experts are loaded.
    pub(crate) fn load_weights(&mut self, devices: [Device<'a>; 2],
        checkpoint: &cuteafd_loader::plan::Checkpoint) -> Result<LoadedWeights<'a>> {
        check_prefix(&KimiK3Config::from_hf(&checkpoint.config)?, &self.cfg)?;
        for rank in 0..2 {
            self.check_global(rank, devices[rank].id)?;
            for layer in 0..self.cfg.layers { self.check_layer(rank, layer, devices[rank].id)?; }
        }
        let mut global = Vec::new();
        for rank in 0..2 {
            let packed = self.global_layouts[rank].pack(&self.cfg,
                |name| super::weights::read_tensor(checkpoint, name))?;
            global.push(self.upload_global(devices[rank], rank, packed)?);
        }
        let mut layers = Vec::new();
        for layer in 0..self.cfg.layers {
            let packed = PackedLayer::pair(&self.cfg, layer, self.placement.layers[layer].mode,
                |name| super::weights::read_tensor(checkpoint, name))?;
            let [zero, one] = packed;
            layers.push([self.upload(devices[0], 0, layer, zero)?, self.upload(devices[1], 1, layer, one)?]);
        }
        Ok(LoadedWeights { global: global.try_into().map_err(|_| anyhow::anyhow!("Kimi global rank count"))?, layers })
    }

    /// No caller-supplied byte count: upload uses the immutable native layout
    /// reconciled against the shared solver's accepted weight ledger.
    pub(crate) fn upload(&mut self, device: Device<'a>, rank: usize, layer: usize,
        packed: PackedLayer) -> Result<Rc<LayerWeights<'a>>> {
        let layout = self.check_layer(rank, layer, device.id)?;
        let weights = Rc::new(LayerWeights::upload(device, packed, layout, layout.bytes()?)?);
        self.layer_owners[layer][rank] = Rc::downgrade(&weights);
        self.layer_uploaded[layer][rank] = true;
        Ok(weights)
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
        let binding = LocalLane::new(allocated, programs, self.cfg.clone(), layers, global, lane, kda, mla)?;
        self.local_bound[lane][rank] = true;
        Ok(binding)
    }

    pub(crate) fn upload_global(&mut self, device: Device<'a>, rank: usize, packed: PackedGlobal) -> Result<Rc<GlobalWeights<'a>>> {
        let layout = self.check_global(rank, device.id)?;
        let weights = Rc::new(layout.upload(device, packed)?);
        self.global_owners[rank] = Rc::downgrade(&weights);
        self.global_uploaded[rank] = true;
        Ok(weights)
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
    let devices = inputs.devices;
    let lanes = inputs.lanes as u64;
    let lane_layouts = inputs.lane_layouts.clone();
    let kda_layout = KdaLayout::new(cfg, inputs.state_slots, inputs.prefix_marks, inputs.lanes)?;
    let global_layouts = GlobalLayout::pair(cfg)?;
    let io_layouts = [IoLayout::new(cfg, 0, inputs.rows)?, IoLayout::new(cfg, 1, inputs.rows)?];
    let rtx = inputs.rtx.take();
    let mut request = request(inputs)?;
    if let Some(inventory) = &rtx {
        ensure!(inventory.costs.len() == cfg.layers && inventory.layers > 0, "Kimi whole expert inventory extent");
        for (layer, cost) in request.layers.iter_mut().zip(&inventory.costs) {
            if let Some(experts) = &mut layer.experts { experts.whole = *cost; }
        }
        request.expert_workspace = (inventory.workspace as u64).checked_mul(lanes).context("Kimi whole workspace overflow")?;
        request.onboard = Onboard::Layers(inventory.layers);
        request.expert_gpus = 2;
        request.policy.expert_ownership = ExpertOwnership::AlternatingWhole;
    }
    let placement = solve(&request)?;
    EXECUTOR.check(&placement)?;
    let layouts = placement.layers.iter().enumerate().map(|(layer, assignment)|
        LayerLayout::pair(cfg, layer, assignment.mode)).collect::<Result<Vec<_>>>()?;
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
    let mla_layouts = [CacheLayout::new(cfg, &placement, 0)?, CacheLayout::new(cfg, &placement, 1)?];
    let admission = Admission { placement, rtx, rtx_loaded: false, cfg: cfg.clone(), layouts, layer_owners, layer_uploaded, local_bound,
        devices, lanes, lane_layouts, lane_allocated, lane_readers,
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
    ensure!((1..=64).contains(&i.state_slots) && i.prefix_marks > 0 && (1..=4096).contains(&i.rows)
        && (1..=2).contains(&i.lanes), "Kimi native admission slot/prefix/row/lane geometry");
    ensure!(cfg.kda_heads == 96 && cfg.kda_head_dim == 128 && cfg.kda_conv == 4
        && cfg.mla_record_values() * 2 == MLA_RECORD_BYTES, "Kimi native state/cache geometry");
    ensure!(i.requested_pool.filter(|n| *n > 0).is_none_or(|n| n >= FLOOR),
        "Kimi pool cannot be below the 256K agentic floor");
    ensure!(i.gpus.iter().all(|gpu| matches!(gpu.baseline, Baseline::Measured { .. })),
        "Kimi runtime admission needs post-module CUDA samples");
    ensure!(i.fixed.iter().chain(&i.startup).all(|d| d.gpu < 2 && d.basis != Basis::Estimated)
        && i.pool_bounded.iter().all(|d| d.gpu < 2 && d.basis != Basis::Estimated),
        "Kimi runtime admission cannot use estimated or out-of-rank demands");
    ensure!(i.lane_layouts.len() == i.lanes && i.lane_layouts.iter().flatten().all(|layout| layout.rows() == i.rows),
        "Kimi lane operand layouts differ from executor geometry");
    let mut fixed = i.fixed;
    for layout in GlobalLayout::pair(cfg)? {
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
        let demand = super::spark::demand(i.rows, lane)?;
        ensure!(!fixed.iter().chain(&i.startup).any(|item| item.gpu == demand.gpu && item.group == demand.group),
            "Kimi Spark intake charge supplied twice");
        fixed.push(demand);
    }
    for rank in 0..2 {
        let layout = IoLayout::new(cfg, rank, i.rows)?;
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
    let mut layers = Vec::with_capacity(cfg.layers);
    for layer in 0..cfg.layers {
        let split = LayerLayout::pair(cfg, layer, LayerMode::HeadSplit)?;
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
        let whole = if mla { LayerLayout::pair(cfg, layer, MODES[1])?[0].attention_bytes()? as u64 } else { 0 };
        let unit = if mla { (MLA_PAGE_ROWS * MLA_RECORD_BYTES) as u64 } else { 0 };
        layers.push(LayerDemand { kind: if mla { AttentionClass::Mla } else { AttentionClass::Kda },
            weights: ModeBytes { whole, split: split_attention },
            kv_unit: KvDemand { unit_bytes_whole: unit, unit_bytes_split: [unit; 2], unit_bytes_context: None },
            fixed_bytes: ModeBytes::default(), extra: vec![], context_indexer: false, colocate: None,
            // RTX routed-expert onboarding is deliberately not executable yet.
            experts: (!cfg.dense[layer]).then_some(ExpertCost { whole: Bytes2::default(),
                half: [Bytes2::default(); 2], tp2: false, spark_ok: true }),
            modes: if mla { MODES.to_vec() } else { vec![LayerMode::HeadSplit] } });
    }
    let mut pool = PoolPolicy::resolve(&i.gpus.map(|gpu| gpu.capacity_bytes), FLOOR, i.requested_pool, MLA_PAGE_ROWS as u64, true);
    pool.ceiling = pool.target.max(pool.requested.unwrap_or(0));
    Ok(PlacementRequest { inventory: Inventory { gpus: i.gpus.to_vec(), spark_ranks: i.spark_ranks, peer_access: i.peer_access },
        attention_placement: i.attention, context_buffers: ContextBuffers::default(), layers_first_gpu: 0,
        pool, layers, pool_overhead: vec![0; 2], pool_bounded: i.pool_bounded, fixed, startup: i.startup, movables: vec![],
        expert_workspace: 0, tp2_workspace: [0; 2], onboard: Onboard::Layers(0), expert_gpus: 0, minimum_expert_layers: 0,
        policy: LayerPolicy { default: vec![LayerMode::HeadSplit], ownership: LayerOwnership::Alternating,
            attention: [(KimiAttention::Mla, AttentionClass::Mla, AttentionPlacement::Layers),
                (KimiAttention::Kda, AttentionClass::Kda, AttentionPlacement::Heads)].into_iter()
                .filter(|(kind, _, _)| cfg.attention.contains(kind))
                .map(|(_, class, placement)| (class, placement)).collect(),
            ..Default::default() },
        hops: hop_spec(cfg.hidden, i.rows, i.lanes), executor: EXECUTOR })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> KimiK3Config {
        let mut cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config()).unwrap();
        cfg.layers = 5;
        cfg.attention = vec![KimiAttention::Kda, KimiAttention::Mla, KimiAttention::Kda, KimiAttention::Mla, KimiAttention::Kda];
        cfg.dense.truncate(5);
        cfg
    }

    fn inputs(cfg: &KimiK3Config) -> Inputs<'_> {
        Inputs { cfg, gpus: [GpuBudget { capacity_bytes: 96 << 30, headroom_bytes: 2 << 30,
            baseline: Baseline::Measured { free_bytes: 96 << 30 } }; 2], devices: [0, 1], peer_access: true, spark_ranks: 6,
            attention: None, requested_pool: None, state_slots: 2, prefix_marks: 2, rows: 4096, lanes: 2,
            lane_layouts: vec![std::array::from_fn(|_| LaneLayout::new(
                &KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config()).unwrap(), 4096, 1_048_576,
                super::super::device::NativeScratch { kda: 1024, mla: [512, 0, 1024] }).unwrap()); 2],
            fixed: (0..2).map(|rank| Demand::new(rank, Category::Workspace, "native engine buffers", 1 << 20, Basis::Exact)).collect(),
            startup: vec![], pool_bounded: vec![], rtx: None }
    }

    #[test]
    fn whole_experts_are_admitted_alternately_and_removed_from_spark_selection() -> Result<()> {
        let cfg = cfg();
        let mut i = inputs(&cfg);
        i.rtx = Some(RtxInventory { package: "not-loaded".into(), costs: vec![Bytes2 {
            resident: 1024 * 1024, staging: 4096 }; cfg.layers], workspace: 8192, layers: 3 });
        let admitted = admit(i)?;
        let local = admitted.working_set();
        assert_eq!(local.rtx_layers, [1, 2, 3]);
        assert_eq!(local.spark_layers, [4]);
        assert_eq!(admitted.placement.layers[1].experts, ExpertHome::RtxWhole { gpu: 0 });
        assert_eq!(admitted.placement.layers[2].experts, ExpertHome::RtxWhole { gpu: 1 });
        assert_eq!(admitted.placement.layers[3].experts, ExpertHome::RtxWhole { gpu: 0 });
        assert_eq!(admitted.placement.expert_ranges[0].peak_bytes, 2 * 1024 * 1024 + 2 * 8192 + 4096);
        assert_eq!(admitted.placement.expert_ranges[1].peak_bytes, 1024 * 1024 + 2 * 8192 + 4096);
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
            assert!(CacheLayout::new(&cfg, &placement, 0).is_err());
        }
        placement = owned.placement.clone();
        for pool in [0, 65, u64::MAX] {
            placement.pool_tokens = pool;
            assert!(CacheLayout::new(&cfg, &placement, 0).is_err());
        }
        assert!(CacheLayout::new(&cfg, &owned.placement, 2).is_err());
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
            assert!(admitted.lane_readers[lane][0].upgrade().is_none());
            let missing = admitted.check_spark(lane).err().unwrap().to_string();
            assert!(missing.contains("requires admitted entry stream"));
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
        let bind = source.split("pub(crate) fn bind_local").nth(1).unwrap().split("pub(crate) fn upload_global").next().unwrap();
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
            let demand = super::super::spark::demand(rows, lane)?;
            let items = admitted.placement.items[0].iter().filter(|item| item.group == demand.group).collect::<Vec<_>>();
            assert_eq!(items.len(), 1);
            assert_eq!(items[0].bytes, (6 * rows * 3584 * 2) as u64);
            assert_eq!(items[0].basis, Basis::Exact);
            assert_eq!(items[0].category, Category::Workspace);
            assert!(!admitted.placement.items[1].iter().any(|item| item.group == demand.group));
        }
        let mut duplicate = inputs(&cfg);
        duplicate.fixed.push(super::super::spark::demand(rows, 0)?);
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
    fn refuses_unsupported_or_unmeasured_runtime_inventory() {
        let cfg = cfg();
        for case in 0..10 {
            let mut i = inputs(&cfg);
            match case {
                0 => i.spark_ranks = 4,
                1 => i.peer_access = false,
                2 => i.attention = Some(AttentionPlacement::Context),
                3 => i.requested_pool = Some(FLOOR - 1),
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
