//! Native coordinator inventory passed to the shared startup solver, before
//! weights, recurrent state or cache owners allocate. This is not the header
//! lower bound exposed by `plan` and does not admit Spark workers by itself.
use super::{device::{IoLayout, TokenIo, Lane, LaneLayout, LocalLane, NativePrograms}, mla::{CacheBank, CacheLayout}, state::{KdaBank, KdaLayout}, weights::{GlobalLayout, GlobalWeights, LayerLayout, LayerWeights}};
use crate::shared::memory::device::{Device, DeviceOwner, Stream};
use crate::shared::speculation::dflash::DflashDrafter;
use cuteafd_loader::formats::dflash::MlaDraftRuntimeLayout;
use std::rc::{Rc, Weak};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{Basis, Behaviour, Category, MemoryKind};
use cuteafd_ffi::{MLA_PAGE_ROWS, MLA_RECORD_BYTES};
use cuteafd_loader::{families::kimi_k3::config::{KimiAttention, KimiK3Config}, placement::*};
use cuteafd_core::memory_layout::ArenaTag;

const EXECUTOR: ExecutorModes = families::KIMI_K3;
const MODES: &[LayerMode] = EXECUTOR.modes;
const WHOLE: LayerMode = LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner };
const SINGLE: ExecutorModes = ExecutorModes { family: "kimi_k3", modes: &[WHOLE],
    hops: false, attention: AttentionPlacement::Layers };
const DRAFT_MARKS: &str = "Kimi exact draft windows";
fn reconcile_arena(rank: usize, layout: &cuteafd_loader::serving_capacity::WeightArenaLayout,
    kind: MemoryKind, facts: &planner::Facts) -> Result<()> {
    use planner::{GranularityBacking, Lifetime, Location};
    ensure!(kind.behaviour() == Behaviour::Immutable, "Kimi arena kind must be immutable");
    let roots = facts.allocations.iter().filter(|owner| owner.id == layout.id()).collect::<Vec<_>>();
    ensure!(roots.len() == 1, "Kimi rank {rank} arena root count differs");
    let root = roots[0];
    ensure!(root.location == Location::coordinator(rank as u8) && root.view.is_none()
        && root.bytes == layout.bytes() as u64 && root.category == kind.category()
        && root.group == "weight arena" && root.lifetime == Lifetime::RESIDENT,
        "Kimi rank {rank} arena differs from admitted backing");
    let weights = facts.resident_weights.iter().filter(|owner| owner.id == layout.id()).collect::<Vec<_>>();
    ensure!(weights.len() == 1 && weights[0].location == root.location && weights[0].weights.0 == root.bytes,
        "Kimi rank {rank} arena resident-weight claim differs");
    let tail_id = format!("{}.tail", layout.id());
    let tails = facts.allocation_granularity.iter().filter(|owner| owner.id == tail_id).collect::<Vec<_>>();
    ensure!(tails.len() == 1 && tails[0].location == root.location && tails[0].lifetime == root.lifetime
        && tails[0].backings == [GranularityBacking { backing: root.id.clone(), requests: vec![root.bytes] }],
        "Kimi rank {rank} arena tail differs from its single request");
    Ok(())
}

fn push_resident_budgets(peaks: [usize; 2], workspace: usize) -> Result<[usize; 2]> {
    Ok([peaks[0].checked_sub(workspace).context("Kimi rank 0 workspace exceeds expert peak")?,
        peaks[1].checked_sub(workspace).context("Kimi rank 1 workspace exceeds expert peak")?])
}

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
pub(crate) struct Inputs<'a, const RANKS: usize = 2> {
    pub expert_usage: bool,
    pub cfg: &'a KimiK3Config,
    pub resident_weight_limit: ResidentWeights,
    pub cache_format: cuteafd_core::MlaCacheFormat,
    pub core: cuteafd_loader::families::kimi_k3::core::CoreInventory,
    pub gpus: [GpuBudget; RANKS],
    pub devices: [i32; RANKS],
    pub peer_access: bool,
    pub spark_ranks: usize,
    pub attention: Option<AttentionPlacement>,
    pub context: Option<cuteafd_loader::families::kimi_k3::context::ContextLayout>,
    pub serving_context: cuteafd_loader::placement::planner::Context,
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
    pub full_prefill_logits: bool,
    pub lane_layouts: Vec<[LaneLayout; RANKS]>,
    pub fixed: Vec<Demand>,
    pub startup: Vec<Demand>,
    pub pool_bounded: Vec<BoundedDemand>,
    pub rtx: Option<RtxInventory>,
    pub spark_budget: Option<SparkBudgetInventory>,
    pub draft: Option<MlaDraftRuntimeLayout>,
    /// Exact speculative verify widths with startup graphs: the drafter's block, else 0.
    pub verify_rows: usize,
    pub prefix: Option<&'a crate::shared::prefix::PrefixArgs>,
}

/// CPU-planned native residency; no backend owner allocates before solve.
pub(crate) struct RtxInventory {
    pub package: std::path::PathBuf,
    pub costs: Vec<Vec<Bytes2>>,
    pub workspace: usize,
    pub layers: Onboard,
}

pub(crate) fn plan_rtx(cfg: &KimiK3Config, catalog: &cuteafd_loader::OfficialV41Catalog,
    package: std::path::PathBuf, rows: usize, layers: usize) -> Result<RtxInventory> {
    plan_rtx_for_ranks(cfg, catalog, package, rows, layers, 2)
}

pub(crate) fn plan_rtx_for_ranks(cfg: &KimiK3Config, catalog: &cuteafd_loader::OfficialV41Catalog,
    package: std::path::PathBuf, rows: usize, layers: usize, ranks: usize) -> Result<RtxInventory> {
    use crate::shared::experts::{exl3::Exl3Weights, layer::ExpertLayer};
    ensure!(matches!(ranks, 1 | 2) && layers > 0 && layers <= cfg.dense.iter().filter(|dense| !**dense).count(),
        "Kimi expert count or physical owners outside routed inventory");
    let workspace = usize::try_from(cuteafd_loader::families::kimi_k3::allocation::expert_workspace_for_ranks(
        &package, rows, ranks)?)?;
    let costs = cfg.dense.iter().enumerate().map(|(layer, dense)| {
        (0..ranks).map(|rank| {
            if *dense { return Ok(Bytes2::default()); }
            let selection = if ranks == 1 { ExpertLayer::BackboneFull { layer } }
                else { ExpertLayer::BackboneTp2 { layer, rank } };
            let plan = Exl3Weights::plan(catalog, selection)?;
            Ok(Bytes2 { resident: plan.resident_bytes as u64, staging: plan.device_staging_bytes as u64 })
        }).collect::<Result<Vec<_>>>()
    }).collect::<Result<Vec<_>>>()?;
    Ok(RtxInventory { package, costs, workspace, layers: Onboard::Layers(layers) })
}

pub(crate) struct Admission<'a> {
    pub expert_usage: Option<cuteafd_core::expert_usage::Geometry>,
    pub placement: Placement,
    pub host_prefix: Option<cuteafd_hostcache::config::Config>,
    pub mark_selection: MarkSelection,
    pub graph_set: Option<cuteafd_loader::placement::inventory::StartupGraphs<super::graphs::Key>>,
    receipt: cuteafd_loader::placement::planner::ResolvedPlan,
    cfg: KimiK3Config,
    core: cuteafd_loader::families::kimi_k3::core::CoreInventory,
    rtx: Option<RtxInventory>,
    rtx_loaded: bool,
    draft: Option<MlaDraftRuntimeLayout>,
    draft_allocated: bool,
    draft_marks_allocated: bool,
    context: Option<cuteafd_loader::families::kimi_k3::context::ContextLayout>,
    context_allocated: bool,
    layouts: Vec<Vec<LayerLayout>>,
    layer_owners: Vec<Vec<Weak<LayerWeights<'a>>>>,
    layer_uploaded: Vec<Vec<bool>>,
    local_bound: Vec<Vec<bool>>,
    devices: Vec<i32>,
    lanes: u64,
    spark_ranks: usize,
    waves: usize,
    unit_rows: usize,
    lane_layouts: Vec<Vec<LaneLayout>>,
    lane_allocated: Vec<Vec<bool>>,
    lane_readers: Vec<Vec<Weak<Stream<'a>>>>,
    kda_layout: KdaLayout,
    kda_allocated: Vec<bool>,
    kda_owners: Vec<Weak<KdaBank<'a>>>,
    mla_layouts: Vec<CacheLayout>,
    mla_allocated: Vec<bool>,
    mla_owners: Vec<Weak<CacheBank<'a>>>,
    global_layouts: Vec<GlobalLayout>,
    global_uploaded: Vec<bool>,
    global_owners: Vec<Weak<GlobalWeights<'a>>>,
    io_layouts: Vec<IoLayout>,
    io_allocated: Vec<Vec<bool>>,
    spark_connected: Vec<bool>,
    spark_links: Vec<cuteafd_loader::placement::PreparedSparkLink>,
    push_host: cuteafd_loader::push_load::PushHostEnvelope,
}

/// Startup retains these until persistent local lanes acquire the same owners.
pub(crate) struct LoadedWeights<'a> {
    pub global: Vec<Rc<GlobalWeights<'a>>>,
    pub layers: Vec<Vec<Rc<LayerWeights<'a>>>>,
}

/// The normal Spark lane consumes these selected layer IDs; selection still
/// goes through shared `SparkAdmission` before any transport connects.
impl<'a> Admission<'a> {
    pub(crate) fn context_layout(&self) -> Option<&cuteafd_loader::families::kimi_k3::context::ContextLayout> {
        self.context.as_ref()
    }

    pub(crate) fn allocate_context(&mut self, library: &'a cuteafd_ffi::NativeLibrary,
        streams: [Rc<Stream<'a>>; 2], abort: [*const u32; 2]) -> Result<Rc<super::context::ContextOwner<'a>>> {
        let layout = self.context.as_ref().context("Kimi context storage was not admitted")?;
        ensure!(!self.context_allocated && self.devices.len() == 2 && self.lanes == 1,
            "Kimi context allocation differs from admitted ranks/lanes or was already attempted");
        for rank in 0..2 {
            let reader = self.lane_readers[0][rank].upgrade().context("Kimi context lane stream expired")?;
            ensure!(Rc::ptr_eq(&reader, &streams[rank]) && self.devices[rank] == streams[rank].device.id,
                "Kimi context allocation differs from admitted compute stream");
        }
        for demand in layout.buffers().demands()? {
            let charges = self.placement.items[demand.gpu as usize].iter()
                .filter(|item| item.group == demand.group).collect::<Vec<_>>();
            ensure!(charges.len() == 1 && charges[0].bytes == demand.bytes
                && charges[0].category == demand.category && charges[0].basis == Basis::Exact,
                "Kimi context allocation differs from accepted {} charge", demand.group);
        }
        let caches = [0, 1].map(|rank| self.mla_owners[rank].upgrade()
            .context("Kimi context requires both admitted cache owners"))
            .into_iter().collect::<Result<Vec<_>>>()?;
        self.context_allocated = true;
        let mut owner = super::context::ContextOwner::new(library, layout, streams, Some(abort))?;
        owner.retain_caches([caches[0].clone(), caches[1].clone()])?;
        Ok(Rc::new(owner))
    }

    /// Construct the final shared owner once, inside its admitted device scope.
    /// Failed construction frees partial storage and never retries allocation.
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(crate) fn allocate_draft(&mut self, device: Device<'a>,
        programs: &'a cuteafd_ffi::programs::Programs<'a>, snapshot: &std::path::Path,
        file: Vec<u8>, stream: *mut std::ffi::c_void,
        representation: cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation,
        scales: crate::shared::fp8_linear::Fp8Scales, rows: crate::shared::fp8_linear::Fp8Rows,
        exchange: Option<([Rc<Stream<'a>>; 2], Rc<crate::shared::peer_split::PeerExchange<'a>>)>)
        -> Result<DeviceOwner<'a, DflashDrafter<'a>>> {
        let rank = self.devices.len() - 1;
        let target_projection = crate::shared::speculation::dflash::TargetProjection::from_bf16(
            !self.core.projections.contains_key("language_model.lm_head"));
        self.allocate_draft_with(device.id, |layout| device.own(|| {
            let split = if layout.context_split() {
                let (streams, peer) = exchange.context("Kimi Context draft requires target exchange")?;
                Some(crate::shared::speculation::dflash::context_split::ContextOwner::allocate(
                    programs, &layout, streams, peer, stream)?)
            } else { None };
            let requests = DflashDrafter::mla_arena_requests(snapshot, representation, device.id)?;
            let arena_layout = cuteafd_loader::serving_capacity::WeightArenaLayout::new(
                format!("kimi/draft/{}", device.id), requests[&device.id].clone())?;
            let admitted = cuteafd_loader::serving_capacity::WeightArenaLayout::new(
                format!("kimi/draft/{rank}"), layout.weight_arena_requests())?;
            ensure!(arena_layout.bytes() == admitted.bytes() && arena_layout.views() == admitted.views(),
                "Kimi draft binder differs from admitted immutable arena");
            let arena = crate::shared::memory::weight_arena::DeviceWeightArena::new(device, arena_layout, ArenaTag::drafter_weights("kimi/draft"))?;
            DflashDrafter::load_mla_with(device.library, programs,
                target_projection, "kimik3", snapshot, file, stream, layout,
                representation, scales, rows, split, |physical, name, bytes| {
                    ensure!(physical == device.id, "Kimi draft arena physical owner changed");
                    arena.view(name, bytes)
                })
        }))
    }

    fn check_draft(&self, device: i32) -> Result<&MlaDraftRuntimeLayout> {
        let head = self.devices.len() - 1;
        ensure!(device == self.devices[head] && !self.draft_allocated,
            "Kimi draft allocation differs from admitted rank or was already attempted");
        ensure!(!self.rtx_loaded && !self.global_uploaded.iter().any(|uploaded| *uploaded)
            && !self.layer_uploaded.iter().flatten().any(|uploaded| *uploaded)
            && !self.lane_allocated.iter().flatten().any(|allocated| *allocated)
            && !self.io_allocated.iter().flatten().any(|allocated| *allocated)
            && !self.kda_allocated.iter().any(|allocated| *allocated)
            && !self.mla_allocated.iter().any(|allocated| *allocated),
            "Kimi drafter startup peak requires allocation before target residency");
        let layout = self.draft.as_ref().context("Kimi drafter was not admitted")?;
        ensure!(draft::home(&self.placement) == Some(head as u8), "Kimi drafter must stay on target head owner");
        let arena = cuteafd_loader::serving_capacity::WeightArenaLayout::new(
            format!("kimi/draft/{head}"), layout.weight_arena_requests())?;
        reconcile_arena(head, &arena, MemoryKind::DrafterWeights, &self.receipt.facts)?;
        let (demand, _) = layout.demands(head as u8)?;
        for demand in demand.owned.into_iter().filter(|owner| owner.group != "MLA DFlash owned weights") {
            let rank = usize::from(demand.gpu);
            ensure!(rank < self.devices.len(), "Kimi draft physical owner absent");
            let charges = self.placement.items[rank].iter().filter(|item| item.group == demand.group).collect::<Vec<_>>();
            ensure!(charges.len() == 1 && charges[0].category == demand.category
                && charges[0].basis == Basis::Exact && charges[0].bytes == demand.bytes,
                "Kimi draft allocation differs from accepted {} charge", demand.group);
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
    fn draft_mark_bytes(&self, device: i32) -> Result<usize> {
        let head = self.devices.len() - 1;
        ensure!(device == self.devices[head] && !self.draft_marks_allocated, "Kimi draft mark allocation owner");
        let layout = self.draft.as_ref().context("Kimi draft marks without admitted draft")?;
        let bytes = layout.window_mark_bytes()?.checked_mul(usize::try_from(self.mark_selection.slots)?)
            .context("Kimi draft mark arena overflow")?;
        ensure!(self.placement.items[head].iter().filter(|item| item.group == DRAFT_MARKS)
            .map(|item| item.bytes).collect::<Vec<_>>() == if bytes == 0 { vec![] } else { vec![bytes as u64] },
            "Kimi draft mark charge differs from arena");
        Ok(bytes)
    }

    pub(crate) fn allocate_draft_marks(&mut self, device: Device<'a>) -> Result<Option<Rc<crate::shared::memory::device::Allocation<'a>>>> {
        let bytes = self.draft_mark_bytes(device.id)?;
        self.draft_marks_allocated = true;
        (bytes > 0).then(|| crate::shared::memory::device::Allocation::new(device, bytes,
            crate::shared::memory::Tag::prefix("kimi/draft-marks")).map(Rc::new)).transpose()
    }
    pub(crate) fn draft_layout(&self) -> Option<&MlaDraftRuntimeLayout> { self.draft.as_ref() }

    /// Hops, the peer exchange and Spark waves serve one prefill unit.
    pub(crate) fn execution_plan(&self) -> Result<LayerExecutionPlan> {
        Ok(LayerExecutionPlan::new(&self.placement, hop_spec_for_ranks(self.cfg.hidden,
            self.unit_rows, self.lanes as usize, self.devices.len()))?)
    }

    pub(crate) fn working_set(&self) -> SparkWorkingSet {
        SparkWorkingSet { rtx_layers: self.placement.layers.iter().enumerate()
            .filter_map(|(layer, assignment)| matches!(assignment.experts, ExpertHome::RtxTp2 | ExpertHome::RtxWhole { .. }).then_some(layer)).collect(), spark_layers: self.placement.layers.iter().enumerate()
            .filter_map(|(layer, assignment)| (assignment.experts == ExpertHome::Spark).then_some(layer)).collect(),
            spark_ranks: self.spark_ranks, prefill_lanes: self.lanes }
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
            self.waves, self.unit_rows, self.spark_links.get(lane).context("Kimi missing admitted Spark link")?.clone())?;
        self.spark_connected[lane] = true;
        Ok(routed)
    }

    fn check_spark(&self, lane: usize) -> Result<Rc<Stream<'a>>> {
        ensure!(lane < self.spark_connected.len() && !self.spark_connected[lane],
            "Kimi Spark lane already connected or outside admission");
        self.lane_readers[lane][self.devices.len() - 1].upgrade().context("Kimi Spark requires admitted GPU1 intake stream")
    }

    /// Starts the Spark admission and loads every coordinator weight inside it:
    /// the dense operands, drafter and admitted expert slices join the same
    /// Spark push group, planned from headers before membership. Required
    /// push planning/delivery errors never fall back to the mount.
    /// Returns the pending admission with the loaded owners.
    pub(crate) fn start_and_load<const RANKS: usize>(&mut self, spark: crate::shared::placement::SparkAdmission,
        peers: &[std::net::SocketAddr], devices: [Device<'a>; RANKS], checkpoint: &cuteafd_loader::plan::Checkpoint,
        catalog: &cuteafd_loader::OfficialV41Catalog, streams: [*mut std::ffi::c_void; RANKS],
        draft_records: crate::shared::experts::rtx::dense::Records,
        memory: &mut crate::shared::inventory::StartupMemoryPhases<'_>)
        -> Result<(crate::shared::placement::PendingSparkAdmission,
            Vec<Option<crate::shared::experts::rtx::exl3::Exl3Tp2Weights<'a>>>, LoadedWeights<'a>)> {
        use crate::shared::experts::rtx::{dense::{self, DenseLoad}, push::{CoordinatorExperts, CoordinatorPush}};
        ensure!(RANKS == self.devices.len(), "Kimi load physical rank inventory differs");
        check_prefix(&KimiK3Config::from_hf(&checkpoint.config)?, &self.cfg)?;
        for rank in 0..RANKS {
            ensure!(devices[rank].id == self.devices[rank], "Kimi load device differs from admission");
            self.check_global(rank, devices[rank].id)?;
            for layer in 0..self.cfg.layers { self.check_layer(rank, layer, devices[rank].id)?; }
        }
        let streams = devices.iter().zip(streams).map(|(device, stream)| (device.id, stream)).collect::<Vec<_>>();
        let rtx = self.rtx_plan(&devices)?;
        let load = DenseLoad::push(devices[0].library);
        // Dense owners and records, planned before membership.
        let policy = crate::shared::experts::rtx::push::PushPolicy::for_snapshot(&checkpoint.snapshot)?;
        let planned = policy.plan(|| -> Result<_> {
            ensure!(spark.coordinator_pushes(), "Kimi Spark admission did not select required push");
            let owners = self.record(&devices, checkpoint, &load, &streams, memory)?;
            let mut records = draft_records;
            records.extend(load.sink.as_ref().context("push sink")?.take_all());
            let (landed, operands, staged, steps, largest, finishes) = records.summary();
            tracing::info!(landed, operands, staged, steps, largest_step_bytes = largest, finishes,
                "Kimi coordinator dense records planned");
            let dense = dense::plan_records(&checkpoint.snapshot, records.clone(), &devices, peers.len())?;
            let experts = rtx.as_ref().map(|(layers, budgets, _)| -> Result<_> {
                let workspace = self.rtx.as_ref().context("Kimi missing TP2 inventory")?.workspace;
                let budgets = budgets.iter().map(|budget| budget.checked_sub(workspace)
                    .context("Kimi pushed expert payload cannot spend workspace")).collect::<Result<Vec<_>>>()?;
                Ok(if RANKS == 1 {
                    CoordinatorExperts::Whole { owners: vec![crate::shared::experts::rtx::push::WholeOwner {
                        rank: 0, selections: layers.iter().map(|&layer|
                            crate::shared::experts::layer::ExpertLayer::BackboneFull { layer }).collect(),
                        budget: budgets[0] }] }
                } else { CoordinatorExperts::Tp2 { layers: layers.clone(), budgets } })
            }).transpose()?;
            let prepared = CoordinatorPush::prepare_view_admitted(checkpoint.view.as_ref()
                .context("Kimi push requires the immutable selected checkpoint view")?, Some(catalog), RANKS,
                experts, self.push_host.clone())?;
            Ok((prepared, owners, records, dense))
        })?;
        let (planned, ranks) = match planned {
            Some(((push, streams, ranks), owners, records, dense)) => (Some((push, owners, records, dense)), Some((streams, ranks))),
            None => (None, None),
        };
        memory.sample("coordinator-push-planned")?;
        let pending = spark.start_with(peers, ranks)?;
        let started = std::time::Instant::now();
        let (weights, pushed) = match planned {
            Some((push, owners, records, dense)) => {
                let (pushed, ()) = push.receive_with(devices.to_vec(), Some(catalog), peers.len(), Some(Ok(dense)), || ());
                let pushed = policy.committed(pushed)?;
                tracing::info!(pushed = pushed.is_some(), operands = records.operands.len(), steps = records.steps.len(),
                    elapsed_ms = started.elapsed().as_millis() as u64, "Kimi coordinator dense operands landed");
                (owners, pushed)
            }
            None => {
                let mount = DenseLoad::mount(devices[0].library);
                let owners = self.record(&devices, checkpoint, &mount, &streams, memory)?;
                mount.release();
                tracing::info!(elapsed_ms = started.elapsed().as_millis() as u64, "Kimi coordinator dense operands mounted");
                (owners, None)
            }
        };
        memory.sample("dense-and-pushed-experts-delivered")?;
        let experts = self.finish_rtx(&devices, catalog, rtx, pushed)?;
        memory.sample("rtx-onboarding-finished")?;
        Ok((pending, experts, self.register(&devices, weights)?))
    }

    /// Binder peaks retain one lane workspace; other lanes allocate only execution scratch.
    #[allow(clippy::type_complexity)]
    fn rtx_plan(&self, devices: &[Device<'a>]) -> Result<Option<(Vec<usize>, Vec<usize>, std::path::PathBuf)>> {
        let Some(inventory) = &self.rtx else { return Ok(None) };
        let layers = self.working_set().rtx_layers;
        if layers.is_empty() { return Ok(None); }
        ensure!(devices.len() == self.devices.len(), "Kimi expert physical inventory differs");
        let mut budgets = Vec::new();
        for (rank, device) in devices.iter().enumerate() {
            ensure!(device.id == self.devices[rank], "Kimi expert device differs from admission");
            let charged = if devices.len() == 1 {
                self.placement.expert_ranges[rank].peak_bytes
            } else { self.placement.tp2.as_ref().context("Kimi missing admitted TP2 arena")?.peak_bytes[rank] };
            let other_lanes = inventory.workspace.checked_mul(self.lanes as usize - 1)
                .context("Kimi expert lane workspace overflow")?;
            budgets.push(usize::try_from(charged)?.checked_sub(other_lanes)
                .context("Kimi expert workspace exceeds charge")?);
        }
        Ok(Some((layers, budgets, inventory.package.clone())))
    }

    /// Publish committed payloads or mount the identical actual-owner slices.
    #[allow(clippy::type_complexity)]
    fn finish_rtx(&mut self, devices: &[Device<'a>], catalog: &cuteafd_loader::OfficialV41Catalog,
        rtx: Option<(Vec<usize>, Vec<usize>, std::path::PathBuf)>,
        pushed: Option<Vec<crate::shared::experts::rtx::push::PushedExperts<'a>>>)
        -> Result<Vec<Option<crate::shared::experts::rtx::exl3::Exl3Tp2Weights<'a>>>> {
        use crate::shared::experts::rtx::{exl3::Exl3Tp2, RtxSelectionPolicy};
        ensure!(!self.rtx_loaded && devices.len() == self.devices.len(), "Kimi experts already loaded or wrong owners");
        self.rtx_loaded = true;
        let Some((layers, budgets, package)) = rtx else { return Ok(devices.iter().map(|_| None).collect()) };
        let rows = self.unit_rows;
        let selection = RtxSelectionPolicy::FirstCapacity;
        let geometry = cuteafd_core::ExpertGeometry::KIMI_K3;
        let pushed = pushed.map(|ranks| ranks.into_iter().map(|r| r.exl3()).collect::<Result<Vec<_>>>()).transpose()?;
        if devices.len() == 1 {
            let weights = match pushed {
                Some(mut ranks) => {
                    ensure!(ranks.len() == 1, "Kimi whole pushed physical rank count");
                    Exl3Tp2::pushed_whole_weights(devices[0], catalog, &package, layers, rows, budgets[0], selection,
                        ranks.pop().unwrap())?
                }
                None => Exl3Tp2::load_whole_weights(devices[0], catalog, &package, layers, rows, budgets[0], selection)?,
            };
            return Ok(vec![Some(weights)]);
        }
        let pair = [devices[0], devices[1]];
        let budgets = [budgets[0], budgets[1]];
        let weights = match pushed {
            Some(ranks) => {
                let pushed = ranks.try_into().map_err(|_| anyhow::anyhow!("Kimi pushed pair rank count"))?;
                Exl3Tp2::pushed_pair_weights(pair, catalog, geometry, [&package, &package], layers, rows, budgets, selection, pushed)?
            }
            None => Exl3Tp2::load_pair_weights(pair, catalog, geometry, [&package, &package], layers, rows, budgets, selection)?,
        };
        Ok(weights.into_iter().map(Some).collect())
    }

    /// Every global and layer owner through the shared records on `load`.
    #[allow(clippy::type_complexity)]
    fn record(&mut self, devices: &[Device<'a>], checkpoint: &cuteafd_loader::plan::Checkpoint,
        load: &crate::shared::experts::rtx::dense::DenseLoad<'a>, streams: &[(i32, *mut std::ffi::c_void)],
        memory: &mut crate::shared::inventory::StartupMemoryPhases<'_>)
        -> Result<(Vec<GlobalWeights<'a>>, Vec<Vec<LayerWeights<'a>>>)> {
        ensure!(devices.len() == self.devices.len() && streams.len() == devices.len(),
            "Kimi dense records physical inventory differs from admission");
        let mut records = super::weights::records::KimiRecords::new(checkpoint, &self.cfg, &self.core,
            load, streams.to_vec());
        // Reconcile every rank before the first physical allocation.
        let layouts = devices.iter().enumerate().map(|(rank, _)| {
            let global = self.global_layouts[rank].shared_layout();
            let layers = self.layouts.iter().map(|layouts| layouts[rank].shared_layout()).collect::<Vec<_>>();
            let layout = cuteafd_loader::families::kimi_k3::weight_layout::coordinator_arena(
                rank, &global, &layers.iter().collect::<Vec<_>>())?;
            reconcile_arena(rank, &layout, MemoryKind::Weights, &self.receipt.facts)?;
            Ok(layout)
        }).collect::<Result<Vec<_>>>()?;
        records.arenas = Rc::new(devices.iter().zip(layouts).map(|(&device, layout)|
            Ok((device.id, crate::shared::memory::weight_arena::DeviceWeightArena::new(device, layout, ArenaTag::weights("kimi/coordinator"))?)))
            .collect::<Result<std::collections::BTreeMap<_, _>>>()?);
        memory.sample("weights-coordinator-arenas-allocated")?;
        let global = devices.iter().zip(&self.global_layouts).map(|(&device, layout)| records.global(device, layout))
            .collect::<Result<Vec<_>>>()?;
        memory.sample("weights-global-upload-staging-recorded")?;
        let mut layers = Vec::with_capacity(self.cfg.layers);
        for layer in 0..self.cfg.layers {
            layers.push(records.layer_ranks(devices, layer, &self.layouts[layer])?);
            memory.sample_layer(layer, self.cfg.layers)?;
        }
        memory.sample("weights-core-layers-upload-staging-recorded")?;
        Ok((global, layers))
    }

    /// Validate every owner before publishing any weak registry entry.
    #[allow(clippy::type_complexity)]
    fn register(&mut self, devices: &[Device<'a>], (global, layers): (Vec<GlobalWeights<'a>>, Vec<Vec<LayerWeights<'a>>>))
        -> Result<LoadedWeights<'a>> {
        let ranks = self.devices.len();
        ensure!(devices.len() == ranks && global.len() == ranks && layers.len() == self.cfg.layers
            && layers.iter().all(|owners| owners.len() == ranks), "Kimi registered physical owner inventory differs");
        for rank in 0..ranks {
            self.check_global(rank, devices[rank].id)?;
            global[rank].check_device(devices[rank])?;
            for (layer, owners) in layers.iter().enumerate() {
                self.check_layer(rank, layer, devices[rank].id)?;
                owners[rank].check_device(devices[rank])?;
            }
        }
        let global = global.into_iter().map(Rc::new).collect::<Vec<_>>();
        let layers = layers.into_iter().map(|owners| owners.into_iter().map(Rc::new).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        for rank in 0..ranks {
            self.global_owners[rank] = Rc::downgrade(&global[rank]);
            self.global_uploaded[rank] = true;
            for (layer, owners) in layers.iter().enumerate() {
                self.layer_owners[layer][rank] = Rc::downgrade(&owners[rank]);
                self.layer_uploaded[layer][rank] = true;
            }
        }
        Ok(LoadedWeights { global, layers })
    }

    fn check_layer(&self, rank: usize, layer: usize, device: i32) -> Result<&LayerLayout> {
        ensure!(rank < self.devices.len() && device == self.devices[rank], "Kimi upload device differs from admitted rank");
        let layout = &self.layouts.get(layer).context("Kimi upload layer outside admission")?[rank];
        ensure!(!self.layer_uploaded[layer][rank], "Kimi layer weights already uploaded");
        Ok(layout)
    }

    /// Bind once for the engine lifetime. No layer/step method drains this lane.
    pub(crate) fn bind_local(&mut self, rank: usize, lane: usize, allocated: Lane<'a>,
        programs: Rc<NativePrograms<'a>>) -> Result<LocalLane<'a>> {
        ensure!(rank < self.devices.len() && lane < self.lanes as usize && !self.local_bound[lane][rank],
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
        ensure!(rank < self.devices.len() && device == self.devices[rank] && lane < self.lanes as usize,
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
        ensure!(rank < self.devices.len() && device == self.devices[rank], "Kimi global upload differs from admitted device");
        ensure!(!self.global_uploaded[rank], "Kimi global weights already uploaded");
        let layout = &self.global_layouts[rank];
        let global = layout.shared_layout();
        let layers = self.layouts.iter().map(|pair| pair[rank].shared_layout()).collect::<Vec<_>>();
        let arena = cuteafd_loader::families::kimi_k3::weight_layout::coordinator_arena(
            rank, &global, &layers.iter().collect::<Vec<_>>())?;
        reconcile_arena(rank, &arena, MemoryKind::Weights, &self.receipt.facts)?;
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
        ensure!(rank < self.devices.len() && device == self.devices[rank], "Kimi MLA allocation differs from admitted device");
        ensure!(!self.mla_allocated[rank], "Kimi MLA bank already allocated");
        let layout = &self.mla_layouts[rank];
        let charges = self.placement.items[rank].iter().filter(|item| item.group == "records").collect::<Vec<_>>();
        ensure!(if layout.bytes() == 0 { charges.is_empty() } else {
            charges.len() == 1 && charges[0].category == Category::Kv
                && charges[0].basis == Basis::Exact && charges[0].bytes == layout.reserved_bytes()
        }, "Kimi MLA allocation differs from accepted record charge");
        Ok(layout)
    }

    fn check_kda(&self, rank: usize, device: i32) -> Result<()> {
        ensure!(rank < self.devices.len() && device == self.devices[rank], "Kimi KDA allocation differs from admitted device");
        ensure!(!self.kda_allocated[rank], "Kimi KDA bank already allocated");
        for demand in self.kda_layout.demands(rank as u8)? {
            let charges = self.placement.items[rank].iter().filter(|item| item.group == demand.group).collect::<Vec<_>>();
            ensure!(if demand.bytes == 0 { charges.is_empty() } else {
                charges.len() == 1 && charges[0].category == demand.category
                    && charges[0].basis == Basis::Exact && charges[0].bytes == demand.bytes
            }, "Kimi KDA allocation differs from accepted native pool charge");
        }
        Ok(())
    }

    fn check_lane(&self, rank: usize, lane: usize, device: i32) -> Result<&LaneLayout> {
        ensure!(rank < self.devices.len() && device == self.devices[rank] && lane < self.lanes as usize,
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

pub(crate) fn admit<'a, const RANKS: usize>(mut inputs: Inputs<'_, RANKS>) -> Result<Admission<'a>> {
    ensure!(matches!(RANKS, 1 | 2), "Kimi admission physical rank count");
    let cfg = inputs.cfg;
    let cache_format = inputs.cache_format;
    inputs.core.validate_runtime_groups(cfg)?;
    let core = inputs.core.clone();
    let devices = inputs.devices;
    let lanes = inputs.lanes as u64;
    let inputs_waves = inputs.waves;
    let spark_ranks = inputs.spark_ranks;
    let unit_rows = cuteafd_loader::families::kimi_k3::allocation::prefill_shape_with(inputs.rows, inputs.waves,
        inputs.unit_rows)?.unit_rows as usize;
    let lane_layouts = inputs.lane_layouts.iter().map(|pair| pair.to_vec()).collect::<Vec<_>>();
    let state_slots = inputs.state_slots;
    let legacy_marks = inputs.prefix_marks;
    let global_layouts = GlobalLayout::ranks_with_core(cfg, &core, devices.len())?;
    let full_prefill_logits = inputs.full_prefill_logits;
    let io_layouts = (0..devices.len()).map(|rank| IoLayout::for_ranks(cfg, rank, inputs.rows, &core, devices.len())
        .map(|layout| layout.with_prefill_logits(full_prefill_logits)))
        .collect::<Result<Vec<_>>>()?;
    let rtx = inputs.rtx.take();
    let spark_budget = inputs.spark_budget.take();
    let draft = inputs.draft.clone();
    let graph_reserve = inputs.startup.iter().any(|d| d.group == super::graphs::RESERVE);
    let graph_rows = unit_rows;
    let graph_slots = inputs.state_slots;
    let graph_verify = inputs.verify_rows;
    let context = inputs.context.clone();
    let prefix_args = inputs.prefix;
    let expert_usage = inputs.expert_usage;
    let mut request = request(inputs)?;
    if let Some(layout) = &draft {
        let demand = Demand::new((RANKS - 1) as u8, Category::Prefix, DRAFT_MARKS, layout.window_mark_bytes()? as u64, Basis::Exact);
        if demand.bytes > 0 {
            let mut total = demand.clone();
            total.bytes = total.bytes.checked_mul(legacy_marks as u64).context("Kimi draft window marks overflow")?;
            request.fixed.push(total);
        }
    }
    if let Some(inventory) = &rtx {
        ensure!(inventory.costs.len() == cfg.layers && inventory.costs.iter().all(|cost| cost.len() == RANKS),
            "Kimi expert inventory physical extent");
        for (layer, cost) in request.layers.iter_mut().zip(&inventory.costs) {
            if let Some(experts) = &mut layer.experts {
                if RANKS == 1 { experts.whole = cost[0]; }
                else { experts.half = [cost[0], cost[1]]; experts.tp2 = true; }
            }
        }
        let workspace = (inventory.workspace as u64).checked_mul(lanes).context("Kimi TP2 workspace overflow")?;
        if RANKS == 1 { request.expert_workspace = workspace; }
        else { request.tp2_workspace = [workspace; 2]; }
        request.onboard = inventory.layers;
        request.expert_gpus = RANKS;
        request.policy.expert_ownership = ExpertOwnership::Contiguous;
    }
    if let Some(budget) = &spark_budget { budget.constrain(&mut request)?; }
    use cuteafd_loader::placement::{families::kimi::{self, ExpertProfile, FixedOwner,
        KimiRecipe, SparkOwner}, planner as contract};
    let placement_mode = request.attention_placement.unwrap_or(AttentionPlacement::Heads);
    let attention = request.layers.iter().map(|layer| {
        if RANKS == 1 { WHOLE }
        else if layer.kind != AttentionClass::Mla { LayerMode::HeadSplit }
        else { match placement_mode {
            AttentionPlacement::Heads => LayerMode::HeadSplit,
            AttentionPlacement::Context => LayerMode::ContextSplit,
            AttentionPlacement::Layers => LayerMode::Whole { gpu: 0, ffn: FfnMode::Split },
        } }
    }).collect::<Vec<_>>();
    let page_rows = if placement_mode == AttentionPlacement::Context { 128 } else { 64 };
    let maximum_pages = request.pool.ceiling.div_ceil(page_rows);
    let routed = request.layers.iter().enumerate().filter_map(|(layer, demand)| demand.experts.is_some().then_some(layer)).collect::<Vec<_>>();
    let counts = match request.onboard {
        Onboard::Auto => (0..=routed.len()).collect::<Vec<_>>(),
        Onboard::Layers(count) => vec![count],
    };
    let arenas = if RANKS == 2 {
        let halves = routed.iter().map(|&layer| request.layers[layer].experts.as_ref().unwrap().half).collect::<Vec<_>>();
        kimi::tp2_arenas(&counts, request.tp2_workspace, &halves)?
    } else {
        let parts = routed.iter().map(|&layer| request.layers[layer].experts.as_ref().unwrap().whole).collect::<Vec<_>>();
        kimi::whole_arenas(&counts, request.expert_workspace, &parts)?
    };
    let mut records = vec![0; RANKS];
    let mut records_by_layer = vec![[0u64; 2]; cfg.layers];
    if RANKS == 2 && placement_mode == AttentionPlacement::Layers && cfg.mla_layers() > 0 {
        records.fill((MLA_PAGE_ROWS * cache_format.record_bytes()) as u64);
    } else {
        for (index, (layer, mode)) in request.layers.iter().zip(&attention).enumerate() {
            for rank in 0..RANKS {
                let unit = match mode {
                    LayerMode::Whole { gpu, .. } => if usize::from(*gpu) == rank { layer.kv_unit.unit_bytes_whole } else { 0 },
                    LayerMode::HeadSplit | LayerMode::ContextSplit => layer.kv_unit.unit_bytes_split[rank],
                };
                records_by_layer[index][rank] = unit;
                records[rank] += unit;
            }
        }
    }
    let mut spark = Vec::new();
    let runtime = cuteafd_core::memory_layout::spark_runtime_reserve_bytes(0)
        .context("Spark reserve overflow")?;
    if let Some(budget) = &spark_budget {
        for rank in 0..budget.resident.len() {
            let owners = budget.overhead[rank].checked_sub(runtime)
                .context("Spark overhead excludes startup reserve")?;
            for &count in &counts {
                let active = count < budget.layers.len();
                spark.push(SparkOwner { count, rank,
                    experts: budget.resident[rank][count..].iter().sum(),
                    runtime: if active { runtime } else { 0 },
                    overhead: if active { owners } else { 0 } });
            }
        }
    }
    let mut fixed = request.fixed.iter().map(FixedOwner::ready).collect::<Vec<_>>();
    fixed.extend(request.context_buffers.demands()?.iter().filter(|demand| demand.bytes > 0).map(FixedOwner::ready));
    // The post-module sample is declared once as an existing physical owner;
    // budgets are full capacities, not a second free-memory subtraction.
    for rank in 0..RANKS {
        let Baseline::Measured { free_bytes } = request.inventory.gpus[rank].baseline else { unreachable!() };
        fixed.push(FixedOwner { rank: rank as u8, category: Category::Runtime,
            group: cuteafd_loader::families::kimi_k3::allocation::CONTEXT_MEASURED.into(),
            bytes: request.inventory.gpus[rank].capacity_bytes.saturating_sub(free_bytes), weights: 0,
            lifetime: contract::Lifetime::RESIDENT, basis: Basis::Exact });
    }
    let mut recipe = KimiRecipe { expert_usage: kimi::usage_geometry(cfg, expert_usage), context: request.pool.compiled_context, gpus: RANKS,
        placement: placement_mode, attention,
        prefill: contract::PrefillShape { rows: unit_rows as u64, waves: inputs_waves as u64 },
        lanes, spark_links: spark_ranks > 0, hop_spec: request.hops.clone(),
        peer_row_bytes: request.context_buffers.query_row_bytes + request.context_buffers.partial_row_bytes,
        maximum_pages, records_per_page: records, records_by_layer,
        layer_attention: request.layers.iter().map(|layer| layer.weights).collect(),
        first_moe: routed.first().copied().unwrap_or(cfg.layers),
        kinds: request.layers.iter().map(|layer| layer.kind).fold(Vec::new(), |mut kinds, kind| {
            if !kinds.contains(&kind) { kinds.push(kind); } kinds
        }),
        experts: (!routed.is_empty()).then_some(ExpertProfile { layers: routed, destination: (0..RANKS as u8).collect(), counts,
            rank_sets: vec![(0..spark_ranks as u8).collect()] }),
        arenas, spark,
        links: kimi::production_links((RANKS - 1) as u8, spark_ranks, lanes,
            contract::PrefillShape { rows: unit_rows as u64, waves: inputs_waves as u64 },
            crate::shared::placement::spark_rings(cuteafd_transport::TcpTransportConfig::default().max_frame_bytes)?,
            cuteafd_core::spark_reply::requested())?,
        draft: draft.as_ref().map(|layout| kimi::draft_facts(layout, page_rows, maximum_pages, (RANKS - 1) as u8)).transpose()?,
        fixed, owners: Vec::new(),
    };
    recipe.arena_padding(cfg, &core)?;
    recipe.blas_owners(crate::shared::target_head::TargetHead::configured().blas_head(
        core.get_weight("language_model.lm_head").is_none()))?;
    let modes = recipe.resolved_attention();
    let prepared_graph_set = (graph_reserve && (RANKS != 2 || placement_mode != AttentionPlacement::Layers)).then(|| super::graphs::inventory_with_context(cfg, lanes as usize, graph_rows,
        graph_slots, graph_verify, RANKS,
        |rank, layer| !matches!(modes[layer], LayerMode::Whole { gpu, .. } if usize::from(gpu) != rank),
        |layer| modes[layer] == LayerMode::ContextSplit,
        context.as_ref().map(|layout| layout.decode()))).transpose()?;
    ensure!(request.pool_bounded.is_empty(), "Kimi has no pool-bounded native owners");
    let cards = request.inventory.gpus.iter().map(|gpu| (gpu.capacity_bytes,
        cuteafd_ffi::coordinator_gpu_budget().map_or(gpu.capacity_bytes, |limit| limit.0.min(gpu.capacity_bytes))))
        .collect::<Vec<_>>();
    let sparks = spark_budget.as_ref().map_or_else(Vec::new, |budget| vec![budget.budget; budget.resident.len()]);
    let mut budgets = kimi::budgets(&cards, request.inventory.gpus[0].reserve_bytes, &vec![0; RANKS], &sparks)?;
    let (host_total, host_available) = cuteafd_loader::formats::mapped_preload::host_memory()?;
    let resident = modes.iter().enumerate().fold(vec![0usize; RANKS], |mut resident, (layer, mode)| {
        if cfg.attention[layer] == KimiAttention::Mla {
            for rank in 0..RANKS {
                if !matches!(mode, LayerMode::Whole { gpu, .. } if usize::from(*gpu) != rank) { resident[rank] += 1; }
            }
        }
        resident
    });
    let geometry = cuteafd_loader::families::kimi_k3::kda::cache_geometry_for_ranks(cfg, &resident, cache_format)?;
    let page_bytes = geometry.ranks.iter().map(|rank| rank.persistent_unit_bytes).sum::<u64>()
        / if placement_mode == AttentionPlacement::Context { 2 } else { 1 };
    let mark_bytes = geometry.ranks.iter().map(|rank| rank.retained_mark_bytes).sum::<u64>()
        + draft.as_ref().map(|layout| layout.window_mark_bytes()).transpose()?.unwrap_or(0) as u64;
    let host_prefix = prefix_args.map(|args| args.resolve_host_config(cuteafd_engine::prefix::FamilyLayout {
        page_rows: MLA_PAGE_ROWS, pages: 0, page_bytes: usize::try_from(page_bytes)?,
        mark_bytes: usize::try_from(mark_bytes)?, draft_bytes: 0, rule: super::mla::PREFIX_REUSE,
        mark_store: cuteafd_engine::prefix::MarkStore::Arena,
        page_owners: if placement_mode == AttentionPlacement::Context { cuteafd_engine::prefix::PageOwners::Parity }
            else { Default::default() },
    }, usize::try_from(request.pool.compiled_context)?, graph_slots)).transpose()?.flatten();
    recipe.owners.push(kimi::host_prefix_owner(host_prefix.as_ref().map_or(0, |config| config.bytes), maximum_pages));
    let rows = lane_layouts[0][0].rows();
    let scoring_bytes = kimi::host_scoring_bytes(if full_prefill_logits { rows } else { rows.min(64) }, rows, cfg.vocab_size)?;
    recipe.owners.push(kimi::host_scoring_owner(scoring_bytes, maximum_pages));
    recipe.owners.push(kimi::host_upload_owner(maximum_pages));
    recipe.owners.extend(kimi::stream_slab_owners(RANKS, maximum_pages)?);
    let push_host = kimi::selected_push_host_envelope(cfg, &core, RANKS)?;
    recipe.owners.extend(kimi::push_host_owners(&push_host, maximum_pages)?);
    recipe.owners.extend(kimi::native_load_owners(&core, RANKS, maximum_pages)?);
    let mut host_budget = kimi::host_budget(host_total, host_available);
    host_budget.headroom = host_budget.headroom.max(prefix_args.map_or(0, |args| args.host_cache_headroom_bytes));
    budgets.push(host_budget);
    let admitted = recipe.admit(kimi::request(request.pool.context, request.pool.requested, request.onboard,
        recipe.prefill, contract::Operation::Serving, request.resident_weight_limit.unwrap_or(ResidentWeights::DEFAULT_LIMIT),
        cuteafd_core::spark_reply::requested()), budgets)?;
    tracing::info!(phases = ?admitted.solved.phases, "Kimi admitted per-phase physical ledger");
    cuteafd_loader::placement::reserve::log_plan(&admitted.solved, "kimi_k3");
    let placement = admitted.placement;
    cuteafd_bench::context::ResolvedSettings::placement(&placement).publish();
    let mark_selection = MarkSelection { legacy_slots: legacy_marks as u64, slots: legacy_marks as u64, reduction_reason: None };
    let pool_tokens = usize::try_from(placement.pool_tokens)?;
    let draft = draft.map(|layout| layout.with_pool_tokens(pool_tokens)).transpose()?;
    let graph_set = if graph_reserve && RANKS == 2 && placement_mode == AttentionPlacement::Layers {
        Some(super::graphs::inventory_with_context(cfg, lanes as usize, graph_rows, graph_slots,
            graph_verify, RANKS,
            |rank, layer| !matches!(placement.layers[layer].mode, LayerMode::Whole { gpu, .. } if usize::from(gpu) != rank),
            |_| false, None)?)
    } else { prepared_graph_set };
    if let Some(graphs) = &graph_set { super::graphs::check_placement_for_ranks(cfg, &placement, &graphs.keys, RANKS)?; }
    if let Some(budget) = &spark_budget {
        for rank in 0..budget.resident.len() {
            let bytes = budget.rank_bytes(rank, placement.onboard_layers)?;
            ensure!(bytes <= budget.budget, "Kimi selected Spark working set exceeds estimated budget");
            tracing::info!(rank, retained_bytes = bytes, budget_bytes = budget.budget,
                reserve = "estimated; pending load/steady census", "Kimi Spark budget selection");
        }
    }
    let kda_layout = KdaLayout::for_ranks(cfg, state_slots, usize::try_from(mark_selection.slots)?, lanes as usize, RANKS)?;
    tracing::info!(selection = %mark_selection.summary(), "Kimi retained mark admission");
    if RANKS == 1 { SINGLE.check(&placement)?; } else { EXECUTOR.check(&placement)?; }
    let layouts = placement.layers.iter().enumerate().map(|(layer, assignment)|
        LayerLayout::ranks_with_core(cfg, layer, assignment.mode, &core, devices.len())).collect::<Result<Vec<_>>>()?;
    for rank in 0..RANKS {
        let global = global_layouts[rank].shared_layout();
        let rank_layers = layouts.iter().map(|pair| pair[rank].shared_layout()).collect::<Vec<_>>();
        let arena = cuteafd_loader::families::kimi_k3::weight_layout::coordinator_arena(
            rank, &global, &rank_layers.iter().collect::<Vec<_>>())?;
        reconcile_arena(rank, &arena, MemoryKind::Weights, &admitted.solved.facts)?;
    }
    let layer_owners = vec![vec![Weak::new(); devices.len()]; layouts.len()];
    let layer_uploaded = vec![vec![false; devices.len()]; layouts.len()];
    let local_bound = vec![vec![false; devices.len()]; lane_layouts.len()];
    let lane_allocated = vec![vec![false; devices.len()]; lane_layouts.len()];
    let io_allocated = vec![vec![false; devices.len()]; lane_layouts.len()];
    let lane_readers = vec![vec![Weak::new(); devices.len()]; lane_layouts.len()];
    let mla_layouts = (0..devices.len()).map(|rank| CacheLayout::for_ranks(cfg, &placement, rank, cache_format, devices.len()))
        .collect::<Result<Vec<_>>>()?;
    let admission = Admission { receipt: admitted.solved, expert_usage: recipe.expert_usage.clone(), placement, host_prefix, mark_selection, graph_set, rtx, rtx_loaded: false, draft, draft_allocated: false, draft_marks_allocated: false, context, context_allocated: false, cfg: cfg.clone(), core, layouts, layer_owners, layer_uploaded, local_bound,
        devices: devices.to_vec(), lanes, spark_ranks, waves: inputs_waves, unit_rows, lane_layouts, lane_allocated, lane_readers,
        kda_layout, kda_allocated: vec![false; devices.len()], kda_owners: vec![Weak::new(); devices.len()],
        mla_layouts, mla_allocated: vec![false; devices.len()], mla_owners: vec![Weak::new(); devices.len()],
        global_layouts, global_uploaded: vec![false; devices.len()], global_owners: vec![Weak::new(); devices.len()],
        io_layouts, io_allocated, spark_connected: vec![false; lanes as usize], spark_links: recipe.links.clone(), push_host };
    for rank in 0..RANKS {
        admission.check_kda(rank, devices[rank])?;
        admission.check_mla(rank, devices[rank])?;
        admission.check_global(rank, devices[rank])?;
        for lane in 0..lanes as usize { admission.check_io(rank, lane, devices[rank])?; }
    }
    if admission.draft.is_some() { admission.check_draft(devices[RANKS - 1])?; }
    Ok(admission)
}

fn hop_spec(hidden: usize, rows: usize, lanes: usize) -> HopSpec {
    hop_spec_for_ranks(hidden, rows, lanes, 2)
}

fn hop_spec_for_ranks(hidden: usize, rows: usize, lanes: usize, ranks: usize) -> HopSpec {
    HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: (hidden * 2) as u64,
        rows: rows as u64, lanes: lanes as u64, entry_gpu: 0, head_gpu: (ranks - 1) as u8,
        terminal: if ranks == 1 { TerminalHome::AttentionOwner } else { TerminalHome::Replicated } }
}

fn request<const RANKS: usize>(mut i: Inputs<'_, RANKS>) -> Result<PlacementRequest> {
    ensure!(matches!(RANKS, 1 | 2), "Kimi request physical rank count");
    if RANKS == 2 { i.attention = Some(cuteafd_loader::families::kimi_k3::resolve_attention(i.attention, RANKS)); }
    let cfg = i.cfg;
    ensure!(cfg.layers > 0 && cfg.layers == cfg.attention.len() && cfg.layers == cfg.dense.len(),
        "Kimi admission layer inventory differs from checkpoint");
    ensure!((1..=8).contains(&i.spark_ranks) && (RANKS == 1 || i.peer_access),
        "Kimi native admission requires 1..8 Spark ranks and peer access for a pair");
    ensure!(i.devices.iter().all(|device| *device >= 0) && (RANKS == 1 || i.devices[0] != i.devices[1]),
        "Kimi admitted devices must be distinct CUDA ordinals");
    ensure!(i.context.is_some() == (i.attention == Some(AttentionPlacement::Context)),
        "Kimi context requires its validated native layout, and other placements must not allocate it");
    ensure!((1..=cuteafd_core::concurrency::state_slots(cuteafd_core::concurrency::MAX_CONCURRENCY) as usize).contains(&i.state_slots)
        && (1..=cuteafd_loader::families::kimi_k3::allocation::LANE_ROWS_MAX).contains(&i.rows)
        && (1..=2).contains(&i.lanes), "Kimi native admission slot/prefix/row/lane geometry");
    let unit = cuteafd_loader::families::kimi_k3::allocation::prefill_shape_with(i.rows, i.waves, i.unit_rows)?
        .unit_rows as usize;
    ensure!(cfg.kda_heads == 96 && cfg.kda_head_dim == 128 && cfg.kda_conv == 4
        && cfg.mla_record_values() * 2 == MLA_RECORD_BYTES, "Kimi native state/cache geometry");
    ensure!(i.gpus.iter().all(|gpu| matches!(gpu.baseline, Baseline::Measured { .. })),
        "Kimi runtime admission needs post-module CUDA samples");
    // Only selected-program code and startup graphs may be explicitly Estimated.
    // Neither exception permits guessed weight/workspace payloads.
    ensure!(i.fixed.iter().all(|d| (d.gpu as usize) < RANKS && (d.basis != Basis::Estimated
            || (d.group == cuteafd_loader::families::kimi_k3::allocation::CODE_OWNER
                && d.category == Category::Runtime
                && d.lifetime == cuteafd_core::memory_layout::ItemLifetime::ReadyResident)))
        && i.startup.iter().all(|d| (d.gpu as usize) < RANKS && (d.basis != Basis::Estimated
            || (d.group == super::graphs::RESERVE && d.category == Category::Runtime
                && d.lifetime == cuteafd_core::memory_layout::ItemLifetime::ReadyResident)))
        && i.pool_bounded.iter().all(|d| (d.gpu as usize) < RANKS && d.basis != Basis::Estimated),
        "Kimi runtime admission allows estimates only for selected-program code and startup graphs");
    ensure!(i.lane_layouts.len() == i.lanes && i.lane_layouts.iter().flatten()
        .all(|layout| layout.rows() == i.rows && layout.unit == unit),
        "Kimi lane operand layouts differ from executor geometry");
    let context = i.lane_layouts[0][0].max_context();
    if let Some(layout) = &i.context {
        ensure!(RANKS == 2 && i.lanes == 1 && cfg.mla_layers() > 0,
            "Kimi context requires two physical ranks, one retained lane and MLA layers");
        let expected = layout.buffers();
        ensure!(expected.compiled_extent == context as u64
            && expected.staging_unit_bytes == (MLA_PAGE_ROWS * i.cache_format.record_bytes()) as u64
            && expected.lanes == i.lanes as u64,
            "Kimi context layout differs from admitted cache extent/format/lanes");
        for name in ["context slots", "gather tables", "commit blocks", "active splits"] {
            let (_, bytes) = layout.metadata_extent(name, i.lane_layouts[0][0].metadata_banks() - 1)?;
            ensure!(bytes > 0, "Kimi context metadata bank is empty");
        }
    }
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
    for layout in GlobalLayout::ranks_with_core(cfg, &i.core, RANKS)? {
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
        let demand = super::spark::admitted_demand_on(i.waves, unit, lane, i.spark_ranks, RANKS - 1,
            cuteafd_core::SparkReplyFormat::resolve(cuteafd_core::spark_reply::requested().prefill, false))?;
        ensure!(!fixed.iter().chain(&i.startup).any(|item| item.gpu == demand.gpu && item.group == demand.group),
            "Kimi Spark intake charge supplied twice");
        fixed.push(demand);
    }
    for rank in 0..RANKS {
        let layout = IoLayout::for_ranks(cfg, rank, i.rows, &i.core, RANKS)?
            .with_prefill_logits(i.full_prefill_logits);
        for lane in 0..i.lanes {
            let demand = layout.demand(lane)?;
            ensure!(!fixed.iter().chain(&i.startup).any(|item| item.gpu == demand.gpu && item.group == demand.group),
                "Kimi token IO charge supplied twice");
            fixed.push(demand);
        }
    }
    let kda = KdaLayout::for_ranks(cfg, i.state_slots, i.prefix_marks, i.lanes, RANKS)?;
    for rank in 0..RANKS {
        for demand in kda.demands(rank as u8)? {
            ensure!(!fixed.iter().chain(&i.startup).any(|item| item.gpu == demand.gpu && item.group == demand.group),
                "Kimi KDA pool charge supplied twice");
            fixed.push(demand);
        }
    }
    let mut movables = Vec::new();
    if let Some(layout) = &i.draft {
        let (demand, peak) = layout.demands((RANKS - 1) as u8)?;
        ensure!(demand.owned.iter().chain(std::iter::once(&peak)).all(|draft|
            !fixed.iter().chain(&startup).any(|item| item.group == draft.group)),
            "Kimi drafter charge supplied twice");
        movables.push(demand.movable(RANKS)?);
        startup.push(peak);
    }
    let mut layers = Vec::with_capacity(cfg.layers);
    for layer in 0..cfg.layers {
        let mla = cfg.attention[layer] == KimiAttention::Mla;
        let (whole, split_attention) = if RANKS == 1 {
            let layouts = LayerLayout::ranks_with_core(cfg, layer, WHOLE, &i.core, 1)?;
            (layouts[0].bytes()? as u64, [0; 2])
        } else {
            let split = LayerLayout::pair_with_core(cfg, layer, LayerMode::HeadSplit, &i.core)?;
            let mut attention = [0; 2];
            for rank in 0..2 {
                attention[rank] = split[rank].attention_bytes()? as u64;
                let weights = split[rank].bytes()? as u64 - attention[rank];
                fixed.push(Demand::new(rank as u8, Category::Weights,
                    format!("Kimi layer {layer} TP2 FFN/residual/router/latent"),
                    weights, Basis::Exact).with_resident_weights(ResidentWeights(weights)));
            }
            let whole = if mla {
                LayerLayout::pair_with_core(cfg, layer,
                    LayerMode::Whole { gpu: 0, ffn: FfnMode::Split }, &i.core)?[0].attention_bytes()? as u64
            } else { 0 };
            (whole, attention)
        };
        let unit = if mla { (MLA_PAGE_ROWS * i.cache_format.record_bytes()) as u64 } else { 0 };
        layers.push(LayerDemand { kind: if mla { AttentionClass::Mla } else { AttentionClass::Kda },
            weights: ModeBytes { whole, split: split_attention },
            kv_unit: KvDemand { unit_bytes_whole: unit, unit_bytes_split: [unit; 2],
                unit_bytes_context: (mla && i.context.is_some()).then_some([unit / 2; 2]) },
            fixed_bytes: ModeBytes::default(), extra: vec![], context_indexer: false, colocate: None,
            // admit() installs the checkpoint's TP2 rank-slice costs before solve.
            experts: (!cfg.dense[layer]).then_some(ExpertCost { whole: Bytes2::default(),
                half: [Bytes2::default(); 2], tp2: false, whole_supported: true, spark_ok: true }),
            modes: if RANKS == 1 { vec![WHOLE] } else if mla { MODES.to_vec() } else { vec![LayerMode::HeadSplit] } });
    }
    let mut pool = PoolPolicy::resolve(&i.gpus.map(|gpu| gpu.capacity_bytes), context as u64,
        i.requested_pool, MLA_PAGE_ROWS as u64).with_context(i.serving_context);
    pool.ceiling = pool.target.max(pool.requested.unwrap_or(0));
    Ok(PlacementRequest {
        resident_weight_limit: Some(i.resident_weight_limit), inventory: Inventory { gpus: i.gpus.to_vec(), spark_ranks: i.spark_ranks, peer_access: i.peer_access },
        attention_placement: i.attention, context_buffers: i.context.as_ref().map_or_else(ContextBuffers::default, |layout| layout.buffers()), layers_first_gpu: 0,
        pool, layers, pool_overhead: vec![0; RANKS], pool_bounded: i.pool_bounded, fixed, startup, movables,
        expert_workspace: 0, tp2_workspace: [0; 2], onboard: Onboard::Layers(0), expert_gpus: 0, minimum_expert_layers: 0,
        policy: LayerPolicy { default: if RANKS == 1 { vec![WHOLE] } else { vec![LayerMode::HeadSplit] }, ownership: LayerOwnership::Ranges,
            attention: [(KimiAttention::Mla, AttentionClass::Mla, AttentionPlacement::Layers),
                (KimiAttention::Kda, AttentionClass::Kda, if RANKS == 1 { AttentionPlacement::Layers } else { AttentionPlacement::Heads })].into_iter()
                .filter(|(kind, _, _)| cfg.attention.contains(kind))
                .map(|(_, class, placement)| (class, placement)).collect(),
            ..Default::default() },
        hops: hop_spec_for_ranks(cfg.hidden, unit, i.lanes, RANKS), executor: if RANKS == 1 { SINGLE } else { EXECUTOR } })
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    fn arena_facts(layout: &cuteafd_loader::serving_capacity::WeightArenaLayout, rank: u8) -> Result<planner::Facts> {
        use planner::{Contract, Facts, Lifetime, Location, PhaseOrder};
        let facts = Facts { phase_order: PhaseOrder::DRAFT_THEN_TARGET, resident_weights: vec![],
            allocations: vec![], deferred: vec![], sequential: vec![], reservations: vec![], allowances: vec![],
            driver_page_tables: vec![], allocation_granularity: vec![] };
        let mut contract = Contract::fixed(64, 1, usize::from(rank) + 1, facts);
        contract.components = planner::weight_arena_components(layout, Location::coordinator(rank), Lifetime::RESIDENT, 1, cuteafd_core::memory_layout::MemoryKind::Weights)?;
        contract.variables = vec![planner::Variable::Prefill(vec![planner::PrefillShape { rows: 64, waves: 1 }]),
            planner::Variable::Lanes(vec![1])];
        contract.facts(&[0, 0], 1).map_err(Into::into)
    }

    #[test]
    fn arena_backing_and_tail_refuse_when_only_payload_fits() -> Result<()> {
        use planner::{Baseline, Budget};
        let layout = cuteafd_loader::serving_capacity::WeightArenaLayout::new("kimi/coordinator/0",
            [("a_log".into(), 384), ("next".into(), 256)])?;
        assert_eq!((layout.payload_bytes(), layout.padding_bytes(), layout.bytes()), (640, 128, 768));
        let facts = arena_facts(&layout, 0)?;
        let budget = |bytes| Budget { id: "gpu".into(), locations: vec![planner::Location::coordinator(0)],
            physical_bytes: bytes, logical_bytes: bytes, baseline: Baseline::Forecast {
                occupied: 0, source: "exact test capacity".into(), assumed: false }, headroom: 0 };
        assert!(matches!(planner::evaluate(&facts, &[budget(768)]), Err(planner::Refusal::Capacity { .. })));
        let phases = planner::evaluate(&facts, &[budget(2 << 20)])?;
        assert!(phases.iter().all(|phase| phase.owned == 768 && phase.opaque == (2 << 20) - 768));
        Ok(())
    }

    #[test]
    fn arena_reconciliation_requires_one_exact_backing_and_tail() -> Result<()> {
        let layout = cuteafd_loader::serving_capacity::WeightArenaLayout::new("kimi/coordinator/0",
            [("a_log".into(), 384), ("next".into(), 256)])?;
        let correct = arena_facts(&layout, 0)?;
        reconcile_arena(0, &layout, MemoryKind::Weights, &correct)?;
        for case in 0..8 {
            let mut stale = correct.clone();
            match case {
                0 => stale.allocations.clear(),
                1 => stale.allocations[0].bytes -= 1,
                2 => stale.allocations.push(stale.allocations[0].clone()),
                3 => stale.allocations[0].location = planner::Location::coordinator(1),
                4 => stale.resident_weights[0].weights.0 -= 1,
                5 => stale.allocation_granularity.clear(),
                6 => stale.allocation_granularity[0].backings[0].requests = vec![384, 384],
                _ => stale.allocation_granularity.push(stale.allocation_granularity[0].clone()),
            }
            assert!(reconcile_arena(0, &layout, MemoryKind::Weights, &stale).is_err(), "case {case}");
        }
        Ok(())
    }

    #[test]
    fn draft_arena_reconciliation_keeps_kind_and_physical_remapping_exact() -> Result<()> {
        use planner::{Contract, CoordinatorDevices, Lifetime, Location, Variable};
        let layout = cuteafd_loader::serving_capacity::WeightArenaLayout::new("kimi/draft/0",
            [("draft.weight".into(), 384), ("draft.norm".into(), 256)])?;
        let mut contract = Contract::fixed(64, 1, 1, arena_facts(&layout, 0)?);
        contract.components = planner::weight_arena_components(&layout, Location::coordinator(0),
            Lifetime::RESIDENT, 1, MemoryKind::DrafterWeights)?;
        contract.variables = vec![Variable::Prefill(vec![planner::PrefillShape { rows: 64, waves: 1 }]),
            Variable::Lanes(vec![1])];
        let facts = contract.facts(&[0, 0], 1)?;
        reconcile_arena(0, &layout, MemoryKind::DrafterWeights, &facts)?;
        assert!(reconcile_arena(0, &layout, MemoryKind::Weights, &facts).is_err());
        assert!(reconcile_arena(0, &layout, MemoryKind::DrafterKv, &facts).is_err());
        assert!(reconcile_arena(0, &layout, MemoryKind::DrafterWorkspace, &facts).is_err());
        let mapped = CoordinatorDevices::new(&[1])?.remap_weight_arenas(&facts.allocations)?;
        assert_eq!(mapped.len(), 1);
        assert_eq!(mapped[0].id, "kimi/draft/1");
        assert_eq!(mapped[0].category, MemoryKind::DrafterWeights.category());
        assert_eq!(mapped[0].bytes, layout.bytes() as u64);
        assert_eq!(mapped[0].location, Location::coordinator(0));
        Ok(())
    }

    #[test]
    fn arena_components_activate_only_the_selected_attention_profile() -> Result<()> {
        use planner::{Activation, Contract, Lifetime, Location, Variable};
        let mut contract = Contract::fixed(64, 1, 2, arena_facts(
            &cuteafd_loader::serving_capacity::WeightArenaLayout::new("unused", [("x".into(), 256)])?, 0)?);
        contract.components.clear();
        contract.variables = vec![Variable::AttentionProfiles { layers: vec![0], values: vec![vec![WHOLE], vec![LayerMode::HeadSplit]] },
            Variable::Prefill(vec![planner::PrefillShape { rows: 64, waves: 1 }]), Variable::Lanes(vec![1])];
        for (option, bytes) in [384, 320].into_iter().enumerate() {
            for rank in 0..2 {
                let layout = cuteafd_loader::serving_capacity::WeightArenaLayout::new(format!("kimi/coordinator/{rank}"),
                    [("a_log".into(), bytes), ("next".into(), 256)])?;
                let mut components = planner::weight_arena_components(&layout, Location::coordinator(rank), Lifetime::RESIDENT, 1, cuteafd_core::memory_layout::MemoryKind::Weights)?;
                for component in &mut components { component.activation = Activation::choice(0, option); }
                contract.components.extend(components);
            }
        }
        for option in 0..2 {
            let facts = contract.facts(&[option, 0, 0], 1)?;
            assert_eq!(facts.allocations.len(), 2);
            assert_eq!(facts.allocation_granularity.len(), 2);
            for rank in 0..2 {
                let layout = cuteafd_loader::serving_capacity::WeightArenaLayout::new(format!("kimi/coordinator/{rank}"),
                    [("a_log".into(), [384, 320][option]), ("next".into(), 256)])?;
                reconcile_arena(rank, &layout, MemoryKind::Weights, &facts)?;
            }
        }
        Ok(())
    }

    #[test]
    fn push_payload_cannot_spend_the_binders_execution_workspace() -> Result<()> {
        let peaks = [4096, 8192];
        let resident = push_resident_budgets(peaks, 1024)?;
        assert_eq!(resident, [3072, 7168]);
        for rank in 0..2 { assert_eq!(resident[rank] + 1024, peaks[rank]); }
        assert_eq!(push_resident_budgets([1024; 2], 1024)?, [0; 2]);
        assert!(push_resident_budgets([1023, 8192], 1024).is_err());
        assert!(push_resident_budgets([4096, 1023], 1024).is_err());
        Ok(())
    }

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
            &[cuteafd_core::SparkEndpoint { ring_depth: None, capacity: 4096, ingress: cuteafd_core::SparkIngress::Fp8K32, partial_reply_bytes: None }],
            crate::shared::placement::spark_rings(cuteafd_transport::TcpTransportConfig::default().max_frame_bytes)?,
            cuteafd_loader::families::kimi_k3::SPARK_DEFAULT_BUDGET_BYTES,
            cuteafd_loader::families::kimi_k3::spark::reserve_bytes(6, 2, 4096)?)?;
        println!("spark minimum_onboard={} max_four_expert_bytes={}", spark.minimum_rtx_layers()?,
            (0..6).map(|rank| spark.rank_bytes(rank, 4)).collect::<Result<Vec<_>>>()?.into_iter().max().unwrap());
        let context = std::env::var("CUTEAFD_KIMI_REPLAY_CONTEXT").ok().map(|v| v.parse::<usize>()).transpose()?.unwrap_or(1 << 20);
        let draft = super::super::draft::Options { snapshot: path("CUTEAFD_KIMI_DRAFT")?, fp8: true, tp2_dflash: false, cache_format: cuteafd_core::MlaCacheFormat::Bf16 }.layout(8, context, 188)?;
        // Sealed program manifests and the failed card's post-module CUDA samples.
        // These are CPU arithmetic inputs, not a new hardware measurement.
        let scratch = NativeScratch { kda: 629_464_064, mla: [168_296_448, 101_056_512, 201_326_592] };
        let free = [101_298_995_200, 101_287_460_864];
        let total = [101_973_491_712, 101_970_345_984];
        for enabled in [false, true] {
            let make_inputs = || -> Result<Inputs<'_>> {
                let mut i = inputs(&cfg);
                i.gpus = std::array::from_fn(|rank| GpuBudget { capacity_bytes: total[rank],
                    reserve_bytes: 2 << 30, baseline: Baseline::Measured { free_bytes: free[rank] } });
                i.state_slots = 8; i.prefix_marks = 1; i.lanes = 1; i.requested_pool = Some(1 << 20);
                i.lane_layouts = vec![std::array::from_fn(|_| LaneLayout::new(&cfg, 4096, 4096, context, scratch).unwrap())];
                i.fixed = startup::exchange_demands(4096, false)?;
                for rank in 0..2 { i.fixed.push(startup::code_demand(rank, 71_303_168)?); }
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
            automatic.startup = cuteafd_loader::families::kimi_k3::allocation::graph_reserve_demands(2)?;
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
                if let Some(experts) = &mut layer.experts {
                    experts.half = [cost[0], cost[1]];
                    experts.tp2 = true;
                }
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
                        trial.pool.target = tokens; trial.pool.ceiling = tokens; trial.pool.max = tokens;
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
        inputs_for_ranks::<2>(cfg)
    }

    fn inputs_for_ranks<const RANKS: usize>(cfg: &KimiK3Config) -> Inputs<'_, RANKS> {
        Inputs { expert_usage: false, cfg, resident_weight_limit: ResidentWeights::DEFAULT_LIMIT, core: Default::default(), cache_format: cuteafd_core::MlaCacheFormat::Bf16, gpus: [GpuBudget { capacity_bytes: 96 << 30, reserve_bytes: 2 << 30,
            baseline: Baseline::Measured { free_bytes: 96 << 30 } }; RANKS], devices: std::array::from_fn(|rank| rank as i32), peer_access: RANKS == 2, spark_ranks: 6,
            attention: (RANKS == 2).then_some(AttentionPlacement::Layers), context: None, serving_context: cuteafd_loader::placement::planner::Context::pinned(1 << 20), requested_pool: None, state_slots: 2, prefix_marks: 2, rows: 4096, lanes: 2, waves: 2, unit_rows: None, full_prefill_logits: false,
            lane_layouts: vec![std::array::from_fn(|_| LaneLayout::for_ranks(
                &KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config()).unwrap(), 4096, 2048, 1_048_576,
                super::super::device::NativeScratch { kda: 1024, mla: [512, 0, 1024] }, RANKS).unwrap()); 2],
            fixed: (0..RANKS).map(|rank| Demand::new(rank as u8, Category::Workspace, "native engine buffers", 1 << 20, Basis::Exact)).collect(),
            startup: vec![], pool_bounded: vec![], rtx: None,
            spark_budget: Some(SparkBudgetInventory {
                layers: cfg.dense.iter().enumerate().filter_map(|(layer, dense)| (!dense).then_some(layer)).collect(),
                resident: vec![vec![0; cfg.dense.iter().filter(|dense| !**dense).count()]; 6],
                overhead: vec![cuteafd_core::memory_layout::spark_runtime_reserve_bytes(0).unwrap(); 6],
                budget: 112 << 30,
            }), draft: None, verify_rows: 0, prefix: None }
    }

    #[test]
    fn explicit_host_headroom_changes_admitted_host_budget() -> Result<()> {
        use clap::Parser;
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            serve: super::super::serve::ServeArgs,
        }
        let cfg = cfg();
        let mut args = Cli::try_parse_from(["serve", "--snapshot", "/fixture", "--native-lib", "/native",
            "--peers", "127.0.0.1:19000,127.0.0.1:19001,127.0.0.1:19002,127.0.0.1:19003,127.0.0.1:19004,127.0.0.1:19005",
            "--prefill-rows", "4096", "--prefill-waves", "2", "--host-cache-bytes", "0"])?.serve;
        crate::commands::bench::assert_resolved_family_seam("kimi_k3", &["max-context-tokens"], || {
            args.prefix.resolve_entries(args.max_sequences as usize);
            let shape = args.prefill()?;
            let mut i = inputs(&cfg);
            assert_eq!((shape.unit_rows as usize, shape.waves as usize), (i.rows, i.waves));
            i.prefix = Some(&args.prefix);
            admit(i)
        })?;
        let mut prefix = args.prefix;
        prefix.host_cache_headroom_bytes = u64::MAX;
        let mut i = inputs(&cfg);
        i.prefix = Some(&prefix);
        let error = admit(i).err().context("explicit host reserve must refuse")?;
        assert!(matches!(error.downcast_ref::<cuteafd_loader::placement::planner::Refusal>(),
            Some(cuteafd_loader::placement::planner::Refusal::Capacity { budget, .. }) if budget == "host"), "{error:#}");
        Ok(())
    }

    fn context_layout(format: cuteafd_core::MlaCacheFormat) -> Result<cuteafd_loader::families::kimi_k3::context::ContextLayout> {
        let scratch = (64usize * 48 * 16 * 512 * 2).next_multiple_of(1024) + 64 * 48 * 16 * 4;
        let programs = cuteafd_loader::families::kimi_k3::context::program_names(format).into_iter()
            .map(|name| {
                let tile = if name.contains("_t8_") { 8 } else { 16 };
                serde_json::json!({"name": name, "scratch_bytes_at_capacity": {"scratch": scratch},
                    "geometry": {"num_splits":16,"runtime_splits":true,"device_splits":true,
                        "fixed_split_grid":16,"lse_base":2,"heads_per_tile":tile,
                        "context_heads_per_tile": (1..=64).map(|rows| if rows < 4 { 8 } else { 16 }).collect::<Vec<_>>()}})
            })
            .collect::<Vec<_>>();
        cuteafd_loader::families::kimi_k3::context::ContextLayout::from_manifest(
            &serde_json::json!({"programs": programs}), format, 1_048_576, 2048, 2)
    }

    #[test]
    fn explicit_context_charges_one_parity_pool_and_whole_step_graphs() -> Result<()> {
        let cfg = cfg();
        for format in [cuteafd_core::MlaCacheFormat::Bf16, cuteafd_core::MlaCacheFormat::Fp8] {
            let layout = context_layout(format)?;
            let mut i = inputs(&cfg);
            i.lanes = 1;
            i.lane_layouts.truncate(1);
            i.cache_format = format;
            i.attention = Some(AttentionPlacement::Context);
            i.context = Some(layout.clone());
            i.requested_pool = Some(1_048_576);
            let initial = super::super::graphs::owned_inventory(&cfg, 1, 2048, i.state_slots, 0)?;
            i.startup = (0..2).map(|rank| super::super::graphs::demand(&initial.inventory, rank)).collect();
            let admitted = admit(i)?;
            assert_eq!(admitted.context_layout(), Some(&layout));
            assert!(!admitted.context_allocated);
            assert_eq!(admitted.placement.pool_tokens, 1_048_576);
            for (layer, assignment) in admitted.placement.layers.iter().enumerate() {
                assert_eq!(assignment.mode, if cfg.attention[layer] == KimiAttention::Mla {
                    LayerMode::ContextSplit
                } else { LayerMode::HeadSplit });
            }
            let expected_kv = cfg.mla_layers() as u64 * admitted.placement.pool_tokens / 2 * format.record_bytes() as u64;
            for rank in 0..2 {
                assert_eq!(admitted.mla_layouts[rank].bytes() as u64, expected_kv);
                assert_eq!(admitted.placement.items[rank].iter().filter(|item| item.group == "records")
                    .map(|item| item.bytes).sum::<u64>(), expected_kv);
                let kda_kv = admitted.kda_layout.demands(rank as u8)?.into_iter()
                    .filter(|demand| demand.category == Category::Kv).map(|demand| demand.bytes).sum::<u64>();
                assert_eq!(admitted.placement.items[rank].iter().filter(|item| item.category == Category::Kv)
                    .map(|item| item.bytes).sum::<u64>(), expected_kv + kda_kv);
            }
            for demand in layout.buffers().demands()? {
                assert_eq!(admitted.placement.items[demand.gpu as usize].iter().filter(|item| item.group == demand.group)
                    .map(|item| (item.bytes, item.basis)).collect::<Vec<_>>(), [(demand.bytes, Basis::Exact)]);
            }
            let set = admitted.graph_set.as_ref().context("Kimi context graphs missing")?;
            super::super::graphs::check_placement(&cfg, &admitted.placement, &set.keys)?;
            assert_eq!(set.keys.iter().filter(|key| matches!(key.op, super::super::graphs::Op::Context { .. })).count(),
                cfg.mla_layers() * 2 * layout.decode().plain().len());
        }
        Ok(())
    }

    #[test]
    fn auto_pair_charges_exactly_explicit_context_and_single_stays_layers() -> Result<()> {
        let cfg = cfg();
        let pair = |attention| -> Result<_> {
            let mut i = inputs(&cfg);
            i.attention = attention;
            i.context = Some(context_layout(i.cache_format)?);
            i.lanes = 1;
            i.lane_layouts.truncate(1);
            admit(i)
        };
        let auto = pair(None)?;
        let context = pair(Some(AttentionPlacement::Context))?;
        assert_eq!(auto.placement.attention_placement, AttentionPlacement::Context);
        assert_eq!(auto.placement.layers, context.placement.layers);
        assert_eq!(auto.placement.items, context.placement.items);
        assert_eq!(auto.placement.pool_tokens, context.placement.pool_tokens);
        assert_eq!(auto.context_layout(), context.context_layout());
        for (layer, assignment) in auto.placement.layers.iter().enumerate() {
            assert_eq!(assignment.mode, if cfg.attention[layer] == KimiAttention::Mla {
                LayerMode::ContextSplit
            } else { LayerMode::HeadSplit });
        }
        let mut i = inputs_for_ranks::<1>(&cfg);
        i.attention = None;
        let single = admit(i)?;
        assert_eq!(cuteafd_loader::families::kimi_k3::resolve_attention(None, 1), AttentionPlacement::Layers);
        assert_eq!(single.placement.attention_placement, AttentionPlacement::Heads);
        assert!(single.context_layout().is_none());
        assert!(single.placement.layers.iter().all(|layer| layer.mode == WHOLE));
        Ok(())
    }

    #[test]
    fn explicit_layers_and_heads_do_not_charge_context_storage() -> Result<()> {
        let cfg = cfg();
        for attention in [Some(AttentionPlacement::Layers), Some(AttentionPlacement::Heads)] {
            let mut i = inputs(&cfg);
            i.attention = attention;
            let admitted = admit(i)?;
            assert!(admitted.context_layout().is_none());
            assert!(admitted.placement.items.iter().flatten().all(|item| !item.group.starts_with("context ")));
            for (layer, assignment) in admitted.placement.layers.iter().enumerate() {
                if cfg.attention[layer] == KimiAttention::Kda || attention == Some(AttentionPlacement::Heads) {
                    assert_eq!(assignment.mode, LayerMode::HeadSplit);
                } else {
                    assert_eq!(assignment.mode, LayerMode::Whole {
                        gpu: u8::from(layer == 3), ffn: FfnMode::Split });
                }
            }
            for rank in 0..2 {
                let owners = if attention == Some(AttentionPlacement::Heads) { 2 } else { 1 };
                assert_eq!(admitted.mla_layouts[rank].bytes(), owners
                    * admitted.placement.pool_tokens as usize * MLA_RECORD_BYTES);
            }
        }
        Ok(())
    }

    #[test]
    fn context_layout_is_rejected_for_other_modes_or_mismatched_storage() -> Result<()> {
        let cfg = cfg();
        for case in 0..4 {
            let mut i = inputs(&cfg);
            i.context = Some(context_layout(cuteafd_core::MlaCacheFormat::Fp8)?);
            match case {
                0 => i.attention = None,
                1 => i.attention = Some(AttentionPlacement::Heads),
                2 => i.attention = Some(AttentionPlacement::Context), // unsupported second lane
                _ => {
                    i.attention = Some(AttentionPlacement::Context);
                    i.lanes = 1;
                    i.lane_layouts.truncate(1); // BF16 cache versus FP8 staging
                }
            }
            assert!(admit(i).is_err(), "case {case}");
        }
        Ok(())
    }

    #[test]
    fn single_owner_admits_full_state_whole_experts_and_no_peer_hops() -> Result<()> {
        let cfg = cfg();
        let mut i = inputs_for_ranks::<1>(&cfg);
        i.rtx = Some(RtxInventory { package: Default::default(),
            costs: cfg.dense.iter().map(|dense| vec![if *dense { Bytes2::default() }
                else { Bytes2 { resident: 1 << 20, staging: 4096 } }]).collect(),
            workspace: 8192, layers: Onboard::Layers(1) });
        let admitted = admit(i)?;
        assert_eq!(admitted.devices, [0]);
        assert_eq!(admitted.global_layouts.len(), 1);
        assert_eq!(admitted.mla_layouts.len(), 1);
        assert!(admitted.layouts.iter().all(|owners| owners.len() == 1));
        assert!(admitted.placement.layers.iter().all(|layer| layer.mode == WHOLE));
        assert_eq!(admitted.placement.onboard_layers, 1);
        let working = admitted.working_set();
        assert_eq!(working.rtx_layers.len(), 1);
        assert_eq!(working.spark_layers.len(), cfg.layers - 2);
        assert!(admitted.placement.layers.iter().filter(|layer| layer.experts != ExpertHome::Dense)
            .all(|layer| matches!(layer.experts, ExpertHome::RtxWhole { gpu: 0 } | ExpertHome::Spark)));
        let full_state = KdaLayout::for_ranks(&cfg, 2, admitted.mark_selection.slots as usize, 2, 1)?;
        assert_eq!(admitted.kda_layout, full_state);
        let plan = admitted.execution_plan()?;
        assert_eq!((plan.spec().entry_gpu, plan.spec().head_gpu), (0, 0));
        assert!(plan.hops().is_empty());
        assert!(admitted.placement.items[0].iter().all(|item|
            !item.group.contains("peer exchange") && !item.group.contains("hop link")));
        assert!(admitted.check_kda(1, 1).is_err());
        assert!(admitted.check_global(1, 1).is_err());
        Ok(())
    }

    #[test]
    fn single_owner_cap_counts_both_frontends_and_whole_layer_weights() -> Result<()> {
        let cfg = cfg();
        let core = Default::default();
        let expected = (0..cfg.layers).try_fold(
            GlobalLayout::ranks_with_core(&cfg, &core, 1)?[0].bytes()? as u64,
            |sum, layer| -> Result<_> {
                sum.checked_add(LayerLayout::ranks_with_core(&cfg, layer, WHOLE, &core, 1)?[0].bytes()? as u64)
                    .context("single-owner native resident weights overflow")
            })?;
        let mut i = inputs_for_ranks::<1>(&cfg);
        i.resident_weight_limit = ResidentWeights(expected);
        let mut request = request(i)?;
        assert_eq!(request.fixed.iter().map(|d| d.resident_weights.0).sum::<u64>()
            + request.layers.iter().map(|d| d.weights.whole).sum::<u64>(), expected);
        assert!(solve(&request).is_ok());
        request.resident_weight_limit = Some(ResidentWeights(expected - 1));
        let error = solve(&request).unwrap_err();
        let mut typed = &error;
        while let PlacementError::OwnerCandidates { source, .. }
            | PlacementError::PrefixMarks { source, .. } = typed { typed = source; }
        assert_eq!(*typed, PlacementError::ResidentWeightLimit { gpu: 0, required: expected, limit: expected - 1 });
        Ok(())
    }

    #[test]
    fn single_owner_draft_is_head_local_and_allocation_is_once_only() -> Result<()> {
        let cfg = cfg();
        let mut i = inputs_for_ranks::<1>(&cfg);
        i.draft = Some(draft_layout(i.state_slots, i.state_slots, 1_048_576));
        let mut admitted = admit(i)?;
        assert!(admitted.check_draft(1).is_err());
        let expected = admitted.check_draft(0)?.clone();
        assert_eq!(draft::home(&admitted.placement), Some(0));
        assert!(admitted.placement.items[0].iter().any(|item| item.group == DRAFT_MARKS));
        admitted.allocate_draft_with(0, |layout| { assert_eq!(layout, expected); Ok(()) })?;
        assert!(admitted.allocate_draft_with(0, |_| Ok(())).is_err());
        let mut bad = inputs_for_ranks::<1>(&cfg);
        bad.fixed.push(Demand::new(1, Category::Workspace, "foreign owner", 1, Basis::Exact));
        assert!(request(bad).is_err());
        let mut bad = inputs_for_ranks::<1>(&cfg);
        bad.unit_rows = Some(4096);
        assert!(request(bad).is_err());
        let mut bad = inputs_for_ranks::<1>(&cfg);
        bad.devices[0] = -1;
        assert!(request(bad).is_err());
        Ok(())
    }

    fn draft_layout(slots: usize, sequences: usize, context: usize) -> MlaDraftRuntimeLayout {
        draft_layout_with_cache(slots, sequences, context, cuteafd_core::MlaCacheFormat::Bf16)
    }

    fn draft_layout_with_cache(slots: usize, sequences: usize, context: usize,
        cache: cuteafd_core::MlaCacheFormat) -> MlaDraftRuntimeLayout {
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
            GlmDraftCapacity::new(slots, sequences, 8).unwrap(), context, slots * context, 188, 2, cache).unwrap()
    }

    #[test]
    fn tp1_four_onboard_draft_marks_match_omitted_zero_and_exact_positive_charges() -> Result<()> {
        let fixture = tempfile::tempdir()?;
        cuteafd_loader::plan::testing::write_kimi_core_snapshot(fixture.path(), false);
        let checkpoint = cuteafd_loader::plan::Checkpoint::open(fixture.path())?;
        let cfg = KimiK3Config::from_hf(&checkpoint.config)?;
        let core = cuteafd_loader::plan::families::kimi::core_inventory(&checkpoint)?;
        for marks in [0, 1, 2] {
            let mut i = inputs_for_ranks::<1>(&cfg);
            i.core = core.clone();
            i.cache_format = cuteafd_core::MlaCacheFormat::Fp8;
            i.rows = 2048; i.waves = 2; i.unit_rows = Some(1024); i.lanes = 1;
            i.state_slots = 2; i.prefix_marks = marks; i.requested_pool = Some(16384);
            i.serving_context = cuteafd_loader::placement::planner::Context::pinned(8192);
            i.lane_layouts = vec![[LaneLayout::for_ranks(&cfg, 2048, 1024, 8192,
                super::super::device::NativeScratch { kda: 1024, mla: [512, 0, 1024] }, 1)?]];
            i.draft = Some(draft_layout_with_cache(2, 2, 8192, i.cache_format));
            // Only checkpoint expert payload sizes are synthetic; the live shape,
            // pool, slots, drafter and exact allocation check match the failed run.
            i.rtx = Some(RtxInventory { package: Default::default(),
                costs: cfg.dense.iter().map(|dense| vec![if *dense { Bytes2::default() }
                    else { Bytes2 { resident: 1 << 20, staging: 4096 } }]).collect(),
                workspace: 8192, layers: Onboard::Layers(4) });
            let mut admitted = admit(i)?;
            assert_eq!(admitted.placement.onboard_layers, 4);
            assert_eq!(admitted.mark_selection.slots, marks as u64);
            let bytes = admitted.draft_layout().unwrap().window_mark_bytes()? * marks;
            let charges = admitted.placement.items[0].iter().filter(|item| item.group == DRAFT_MARKS)
                .map(|item| item.bytes).collect::<Vec<_>>();
            assert_eq!(charges, if marks == 0 { vec![] } else { vec![bytes as u64] });
            assert_eq!(admitted.draft_mark_bytes(0)?, bytes);
            assert!(admitted.draft_mark_bytes(1).is_err());
            let original = admitted.placement.items[0].clone();
            if marks == 0 {
                let mut invalid = original[0].clone();
                invalid.group = DRAFT_MARKS.into(); invalid.bytes = 0;
                admitted.placement.items[0].push(invalid);
                assert!(admitted.draft_mark_bytes(0).is_err());
            } else {
                let charge = admitted.placement.items[0].iter_mut().find(|item| item.group == DRAFT_MARKS).unwrap();
                charge.bytes += 1;
                assert!(admitted.draft_mark_bytes(0).is_err());
                admitted.placement.items[0].retain(|item| item.group != DRAFT_MARKS);
                assert!(admitted.draft_mark_bytes(0).is_err());
                admitted.placement.items[0] = original.clone();
                let duplicate = original.iter().find(|item| item.group == DRAFT_MARKS).unwrap().clone();
                admitted.placement.items[0].push(duplicate);
                assert!(admitted.draft_mark_bytes(0).is_err());
            }
            admitted.placement.items[0] = original;
            admitted.draft_marks_allocated = true;
            assert!(admitted.draft_mark_bytes(0).is_err());
        }
        Ok(())
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
    fn context_draft_admission_preserves_both_physical_owners_and_page_pairs() -> Result<()> {
        let cfg = cfg();
        let mut i = inputs(&cfg);
        i.draft = Some(draft_layout(i.state_slots, i.state_slots, 1048576).with_context_split(true)?);
        let admitted = admit(i)?;
        let layout = admitted.check_draft(1)?;
        assert!(layout.context_split());
        assert_eq!(admitted.placement.pool_tokens % 128, 0);
        let (demand, _) = layout.demands(1)?;
        for rank in 0..2 {
            assert!(demand.owned.iter().any(|owner| usize::from(owner.gpu) == rank));
            assert!(admitted.placement.items[rank].iter().any(|owner|
                owner.group.starts_with("MLA DFlash Context layer.")));
        }
        let mut i = inputs_for_ranks::<1>(&cfg);
        i.draft = Some(draft_layout(i.state_slots, i.state_slots, 1048576).with_context_split(true)?);
        assert!(admit(i).is_err());
        Ok(())
    }

    #[test]
    fn owned_candidates_keep_matching_graphs_and_shared_draft_pool() -> Result<()> {
        let cfg = cfg();
        let mut i = inputs(&cfg);
        // The configured arena is independent of the winning key inventory.
        let reserve = cuteafd_loader::families::kimi_k3::allocation::graph_reserve_demands(2)?;
        i.startup = reserve.clone();
        i.draft = Some(draft_layout(i.state_slots, i.state_slots, 1048576));
        let admitted = admit(i)?;
        let layout = admitted.draft_layout().unwrap();
        assert_eq!(layout.paged_cache().target_tokens as u64, admitted.placement.pool_tokens);
        assert_eq!(layout.paged_cache().full_bytes,
            admitted.placement.pool_tokens.div_ceil(64) * 64 * layout.paged_cache().record_bytes as u64);
        let set = admitted.graph_set.as_ref().context("winning owner graphs missing")?;
        super::super::graphs::check_placement(&cfg, &admitted.placement, &set.keys)?;
        for rank in 0..2 {
            let charges = admitted.placement.items[rank].iter().filter(|item|
                item.group == super::super::graphs::RESERVE).collect::<Vec<_>>();
            assert_eq!(charges.len(), 1, "one configured graph arena per device");
            assert_eq!(charges[0].bytes, reserve[rank].bytes);
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
        let charge = admitted.receipt.facts.allocations.iter_mut().find(|owner| owner.id == "kimi/draft/1").unwrap();
        charge.bytes -= 1;
        assert!(admitted.check_draft(1).is_err());
        assert!(admit(inputs(&cfg))?.check_draft(1).is_err());
        Ok(())
    }

    #[test]
    fn tp2_rank_slices_are_admitted_on_both_gpus_and_removed_from_spark_selection() -> Result<()> {
        let cfg = cfg();
        let mut i = inputs(&cfg);
        i.rtx = Some(RtxInventory { package: "not-loaded".into(), costs: vec![vec![Bytes2 {
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
            i.rtx = Some(RtxInventory { package: "not-loaded".into(), costs: vec![vec![Bytes2 {
                resident: 1024 * 1024, staging: 4096 }; 2]; cfg.layers], workspace: 8192, layers: selection });
            let runtime = cuteafd_core::memory_layout::spark_runtime_reserve_bytes(0).unwrap();
            i.spark_budget = Some(SparkBudgetInventory { layers: vec![1, 2, 3, 4],
                resident: vec![vec![100; 4]; 6], overhead: vec![runtime + 50; 6], budget: runtime + 253 });
            i
        };
        let admitted = admit(make(Onboard::Auto))?;
        assert!(admitted.placement.onboard_layers >= 2);
        assert!(admitted.placement.onboard_layers < 4);
        assert!(!admitted.working_set().spark_layers.is_empty());
        assert!(admitted.placement.pool_tokens >= 1 << 20);
        assert!(admitted.mark_selection.slots > 0);
        let error = admit(make(Onboard::Layers(1))).err().context("strict override must refuse")?.to_string();
        assert!(error.contains("explicit --rtx-expert-layers 1") && error.contains("--spark-memory-cap"));
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
    fn installed_single_owner_planner_and_runtime_remaining_kv_match() -> Result<()> {
        use cuteafd_loader::{families::kimi_k3::allocation, plan::{layout::LayoutOptions,
            testing, ExpertPlacement, PlanOptions}};
        use cuteafd_core::memory_layout::DeviceKind;
        let snapshot = tempfile::tempdir()?;
        testing::write_kimi_snapshot(snapshot.path(), true);
        let plan_input = testing::BudgetedPlanInput::open(snapshot.path())?;
        let package_root = tempfile::tempdir()?;
        let path = testing::write_kimi_installed_inventory(package_root.path(), false);
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
        let cfg = KimiK3Config::from_hf(&testing::kimi_k3_config())?;
        // BF16 headers isolate owner accounting; the real 96-GiB card uses K6.
        let capacity = 192 << 30;
        let context = 1_048_576;
        let (unit, waves, slots, marks) = (1024, 3, 2, 2);
        let rows = unit * waves;
        let spark_bytes = 110 << 30;
        let options = PlanOptions {
            placement: ExpertPlacement::Sparks { ranks: 6 }, coordinator_budget_bytes: capacity,
            layout: Some(LayoutOptions { host_memory: Some((128 << 30, 128 << 30)),
                rtx_bytes: vec![capacity], head_split: false, concurrency: slots as u64,
                state_slots: Some(slots as u64), prefix_slots: Some(marks as u64),
                host_cache_bytes: Some(0), context_tokens: context as u64,
                prefill_rows: unit as u64, prefill_lanes: waves as u64,
                workspace_manifest: Some(path.clone()), onboard: Some(OnboardSetting::Auto),
                spark_allocation_budget_bytes: Some(spark_bytes), ..Default::default() }),
            ..Default::default()
        };
        let report = plan_input.plan(&options)?;
        ensure!(report.fits && report.placement_supported && report.executable(), "{:?}", report.hints);
        let catalog = plan_input.expert_catalog()?;
        let mut rtx = plan_rtx_for_ranks(&cfg, &catalog,
            allocation::expert_package_for_ranks(&path, 1)?, unit, 92, 1)?;
        rtx.layers = Onboard::Auto;
        let working = SparkWorkingSet { rtx_layers: vec![], spark_layers: (1..93).collect(),
            spark_ranks: 6, prefill_lanes: 1 };
        let rings = cuteafd_loader::plan::layout::spark_ring_geometry(1);
        let spark = SparkBudgetInventory::exl3(&catalog, &working,
            &[cuteafd_core::SparkEndpoint { ring_depth: None, capacity: unit as u32,
                ingress: cuteafd_core::SparkIngress::Fp8K32, partial_reply_bytes: None }],
            cuteafd_core::SparkRings { max_frame_bytes: rings.max_frame_bytes, depth: rings.depth,
                slot_bytes: rings.slot_bytes, alignment: rings.alignment }, spark_bytes,
            cuteafd_loader::families::kimi_k3::spark::reserve_bytes(6, waves as u64, unit as u64)?)?;
        let arch = ArchContext::for_device("sm_120", capacity);
        let programs = ProgramSet::from_manifest(&manifest, &["kimik3"]);
        let code = cuteafd_loader::placement::inventory::loaded_code_allowance(
            "kimik3", "exl3", false, 0, arch, Some(&programs));
        let resident = code.bytes;
        let runtime_inputs = || -> Result<_> {
            let mut i = inputs_for_ranks::<1>(&cfg);
            i.resident_weight_limit = ResidentWeights(options.coordinator_budget_bytes);
            i.rows = rows; i.waves = waves; i.unit_rows = Some(unit);
            i.lanes = 1; i.state_slots = slots; i.prefix_marks = marks;
            i.cache_format = cuteafd_core::MlaCacheFormat::Fp8;
            i.lane_layouts = vec![[LaneLayout::for_ranks(&cfg, rows, unit, context,
                allocation::NativeScratch::from_manifest_for_ranks(&manifest, 1)?, 1)?]];
            i.fixed = vec![allocation::code_demand(0,
                code.pending(arch.context_bytes + resident, arch.context_bytes), 1)?];
            i.startup = cuteafd_loader::families::kimi_k3::allocation::graph_reserve_demands(1)?;
            i.gpus[0] = GpuBudget { capacity_bytes: capacity, reserve_bytes: cuteafd_loader::placement::reserve::gpu("kimi_k3"),
                baseline: Baseline::Measured { free_bytes: capacity - arch.context_bytes - resident } };
            Ok(i)
        };
        assert_eq!(request(runtime_inputs()?)?.resident_weight_limit,
            Some(ResidentWeights(options.coordinator_budget_bytes)));
        let mut i = runtime_inputs()?;
        i.rtx = Some(rtx); i.spark_budget = Some(spark);
        let admitted = admit(i)?;
        let memory = report.memory_layout.context("single-owner installed ledger")?;
        assert_eq!(memory.pool_tokens, admitted.placement.pool_tokens);
        assert_eq!(admitted.devices, [0]);
        assert!(admitted.placement.layers.iter().all(|layer| layer.mode == WHOLE));
        let devices = memory.devices.iter().filter(|d| d.kind == DeviceKind::Rtx).collect::<Vec<_>>();
        assert_eq!(devices.len(), 1);
        let planned = devices[0];
        assert_eq!(planned.index, 0);
        use cuteafd_loader::families::kimi_k3::weight_layout;
        let global = weight_layout::GlobalLayout::ranks_with_core(&cfg, &Default::default(), 1)?;
        let layers = (0..cfg.layers).map(|layer|
            weight_layout::LayerLayout::ranks_with_core(&cfg, layer, WHOLE, &Default::default(), 1))
            .collect::<Result<Vec<_>>>()?;
        let arena = weight_layout::coordinator_arena(0,
            &global[0], &layers.iter().map(|pair| &pair[0]).collect::<Vec<_>>())?;
        // Each of the 69 whole KDA A_log views is 384 bytes, then aligns to 512.
        assert_eq!(arena.padding_bytes(), cfg.kda_layers() * 128);
        assert_eq!(arena.padding_bytes(), 8832);
        reconcile_arena(0, &arena, MemoryKind::Weights, &admitted.receipt.facts)?;
        for items in [&planned.items, &admitted.placement.items[0]] {
            let roots = items.iter().filter(|item| item.group == "weight arena").collect::<Vec<_>>();
            assert_eq!(roots.len(), 1);
            assert_eq!(roots[0].bytes, arena.bytes() as u64);
            assert!(!items.iter().any(|item| item.group == "kimi arena padding" || item.group == "Kimi global weights"
                || item.group == "layers" || item.group.starts_with("Kimi layer ")));
        }
        let runtime_bytes = admitted.placement.items[0].iter().map(|item| item.bytes).sum::<u64>();
        // The plan's named-reserve item (absent at the default 0) is the runtime's kept-free bytes.
        assert_eq!(planned.used_bytes(), runtime_bytes + cuteafd_loader::placement::reserve::gpu("kimi_k3"));
        let mark_bytes = cfg.kda_layers() as u64 * (cfg.kda_state_bytes() + cfg.kda_conv_bytes());
        assert_eq!(planned.items.iter().filter(|item| item.category == Category::Prefix)
            .map(|item| item.bytes).sum::<u64>(), mark_bytes * marks as u64);
        assert!(!planned.items.iter().any(|item| item.group == "Kimi peer/abort/hop controls"));
        Ok(())
    }

    #[test]
    fn plan_command_and_live_admission_enforce_the_same_resident_weight_cap() -> Result<()> {
        use clap::Parser;
        use cuteafd_loader::plan::testing;
        let snapshot = tempfile::tempdir()?;
        testing::write_kimi_snapshot(snapshot.path(), true);
        let package = tempfile::tempdir()?;
        let manifest_path = testing::write_kimi_installed_inventory(package.path(), false);
        let manifest = serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
        let config = serde_json::from_slice(&std::fs::read(snapshot.path().join("config.json"))?)?;
        let cfg = KimiK3Config::from_hf(&config)?;
        let globals = cuteafd_loader::families::kimi_k3::weight_layout::GlobalLayout::pair(&cfg)?;
        let layers = (0..cfg.layers).map(|layer|
            cuteafd_loader::families::kimi_k3::weight_layout::LayerLayout::pair(&cfg, layer, LayerMode::HeadSplit))
            .collect::<Result<Vec<_>>>()?;
        let arena = cuteafd_loader::families::kimi_k3::weight_layout::coordinator_arena(0,
            &globals[0], &layers.iter().map(|pair| &pair[0]).collect::<Vec<_>>())?;
        let expected = arena.bytes() as u64;
        for (gib, accepted) in [(80u64, true), (1, false)] {
            let cli = crate::cli::Cli::try_parse_from([
                "cuteafd".into(), "plan".into(), snapshot.path().as_os_str().to_owned(),
                "--layout".into(), "--rtx".into(), "2".into(), "--spark-ranks".into(), "6".into(),
                "--spark-memory-cap".into(), "256GiB".into(),
                "--workspace-manifest".into(), manifest_path.as_os_str().to_owned(),
                "--coordinator-weight-budget-gib".into(), gib.to_string().into(),
                "--attention-placement".into(), "heads".into(), "--rtx-expert-layers".into(), "0".into(),
                "--max-context-tokens".into(), "262144".into(), "--pool-tokens".into(), "262144".into(),
                "--concurrency".into(), "2".into(), "--prefix-slots".into(), "0".into(),
                "--host-cache-bytes".into(), "0".into(), "--prefill-rows".into(), "2048".into(),
                "--prefill-waves".into(), "2".into(), "--kv-cache".into(), "bf16".into(),
                "--speculator".into(), "off".into(), "--vision".into(), "off".into(),
                "--audio".into(), "off".into(), "--require-ready".into(),
            ])?;
            let crate::cli::Commands::Plan(mut args) = cli.command else { panic!("plan command") };
            args.coordinator_weight_budget_gib = cli.coordinator_weight_budget_gib.context("global cap")?;
            args.memory = cli.memory;
            let planned = crate::commands::plan::run_plan(args);
            let mut i = inputs(&cfg);
            i.resident_weight_limit = ResidentWeights(gib << 30);
            i.attention = Some(AttentionPlacement::Heads);
            i.requested_pool = Some(262144);
            i.serving_context = cuteafd_loader::placement::planner::Context::pinned(262144);
            i.prefix_marks = 0;
            i.lanes = 1;
            i.lane_layouts = vec![std::array::from_fn(|_| LaneLayout::new(&cfg, i.rows,
                test_unit(i.rows, i.waves, i.unit_rows), 262144,
                super::super::device::NativeScratch::from_manifest(&manifest).unwrap()).unwrap())];
            let admitted = admit(i);
            if accepted {
                planned?;
                assert_eq!(admitted?.placement.pool_tokens, 262144);
            } else {
                let live_error = admitted.err().context("live admission must refuse the cap")?;
                let typed = live_error.downcast_ref::<cuteafd_loader::placement::planner::Refusal>()
                    .context("typed live refusal")?;
                assert_eq!(*typed, cuteafd_loader::placement::planner::Refusal::ResidentWeightLimit {
                    gpu: 0, required: expected, limit: gib << 30 });
                let plan_error = planned.expect_err("plan command must refuse the same cap");
                assert!(plan_error.to_string().contains(&typed.to_string()), "{plan_error:#}; {live_error:#}");
            }
        }
        Ok(())
    }

    #[test]
    fn installed_max_planner_and_runtime_remaining_kv_match_per_gpu() -> Result<()> {
        use cuteafd_loader::{families::kimi_k3::allocation, plan::{self, layout::LayoutOptions,
            testing, ExpertPlacement, PlanOptions}};
        use cuteafd_core::memory_layout::DeviceKind;
        use clap::Parser;
        #[derive(clap::Parser)]
        struct PrefixCli { #[command(flatten)] serve: super::super::serve::ServeArgs }
        let snapshot = tempfile::tempdir()?;
        testing::write_kimi_snapshot(snapshot.path(), true);
        let plan_input = cuteafd_loader::plan::testing::BudgetedPlanInput::open(snapshot.path())?;
        let package_root = tempfile::tempdir()?;
        let manifest_path = testing::write_kimi_installed_inventory(package_root.path(), false);
        let manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
        let cfg = KimiK3Config::from_hf(&testing::kimi_k3_config())?;
        let context = 1_048_576;
        let capacity = cuteafd_loader::placement::inventory::PRO_TOTAL_BYTES;
        let spark_bytes = 115_964_116_992;
        use cuteafd_core::MlaCacheFormat;
        let automatic = plan_input.plan_spark_count(&PlanOptions {
            attention_placement: Some(AttentionPlacement::Layers),
            placement: ExpertPlacement::Sparks { ranks: 6 },
            layout: Some(LayoutOptions { host_memory: Some((128 << 30, 128 << 30)),
                rtx_bytes: vec![capacity; 2], concurrency: 16,
                workspace_manifest: Some(manifest_path.clone()),
                spark_allocation_budget_bytes: Some(spark_bytes), ..Default::default() }),
            ..Default::default()
        }, 6, Some(6))?;
        ensure!(automatic.fits && automatic.placement_supported && automatic.executable(), "Auto shape/pool: {:?}", automatic.hints);
        assert!(!automatic.hints.iter().any(|hint| hint.what.contains("prefill reserve does not fit")));
        assert_eq!((automatic.prefill_unit_rows, automatic.prefill_waves), (Some(2752), Some(3)));
        let unit = automatic.prefill_unit_rows.context("planned Kimi unit rows")? as usize;
        let receive = cuteafd_loader::families::kimi_k3::core::exchange_recv_bytes(unit)?;
        crate::shared::peer_split::validate_slot_sizes([0, 1], &[receive])?;
        let demands = super::super::startup::exchange_demands(unit, false)?;
        assert!(demands.iter().all(|demand| demand.bytes >= receive as u64));
        // The max config's remote DFlash bridge has control-only slots. Its
        // planner charge and runtime extents must agree before touching CUDA.
        use cuteafd_loader::placement::draft::{BRIDGE_SLOT_SIZES, BRIDGE_EXCHANGE_BYTES};
        crate::shared::peer_split::validate_slot_sizes([0, 1], &BRIDGE_SLOT_SIZES)?;
        assert_eq!(BRIDGE_SLOT_SIZES.iter().sum::<usize>() as u64 + 256 + 256, BRIDGE_EXCHANGE_BYTES);
        let draft = draft_layout(16, 16, context);
        let demands = super::super::draft::bridge_demands(&draft,
            automatic.prefill_span_rows.context("planned Kimi span rows")? as usize)?;
        for gpu in 0..2 {
            let exchange = demands.iter().find(|demand| demand.gpu == gpu && demand.group == "DFlash bridge exchange")
                .context("planned Kimi remote draft exchange")?;
            assert_eq!(exchange.bytes, BRIDGE_EXCHANGE_BYTES);
        }
        assert!(automatic.memory_layout.context("automatic Kimi ledger")?.pool_tokens
            >= cuteafd_core::memory_policy::AGENTIC_FLOOR_TOKENS);
        let mut refused = 0;
        for ((slots, prefix_slots, host_cache_bytes, entries, format, requested_pool), waves) in [
            (16, Some(1), Some(0), 20, MlaCacheFormat::Fp8, None),
            (8, None, Some(32 << 30), 20, MlaCacheFormat::Fp8, None),
            (16, None, Some(8 << 30), 20, MlaCacheFormat::Fp8, None),
            (8, None, Some(0), 20, MlaCacheFormat::Bf16, None),
        ].into_iter().flat_map(|case| [2usize, 3, 4].map(move |waves| (case, waves)))
            // Reduced Spark scratch lets all automatic pools fit; retain an
            // explicitly infeasible pool to exercise the typed refusal path.
            .chain([((8, None, Some(0), 20, MlaCacheFormat::Bf16, Some(1 << 30)), 3)]) {
            let shape = cuteafd_loader::serving_capacity::prefill_waves::prefill_shape(6,
                Default::default(), cuteafd_loader::serving_capacity::prefill_waves::PrefillOverrides {
                    waves: Some(waves as u64), ..Default::default()
                })?;
            let rows = shape.span_rows as usize;
            let unit = shape.unit_rows as usize;
            let rule = crate::shared::prefix::MarkRule { lanes: slots as usize, entries,
                budget_bytes: 512 << 20, host: host_cache_bytes != Some(0) };
            let mark_bytes = cfg.kda_layers() as u64 * (cfg.kda_state_bytes() + cfg.kda_conv_bytes());
            let marks = prefix_slots.unwrap_or_else(|| rule.slots(usize::try_from(mark_bytes).unwrap()) as u64);
            assert_eq!(marks, if entries == 0 { 0 } else if rule.host { 2 } else { 1 });
            let options = PlanOptions { attention_placement: Some(AttentionPlacement::Layers), placement: ExpertPlacement::Sparks { ranks: 6 },
                layout: Some(LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![capacity; 2], prefill_rows: unit as u64, prefill_lanes: waves as u64,
                    concurrency: slots, state_slots: Some(slots), prefix_slots,
                    mimo_prefix_entries: entries as u64, mimo_prefix_mark_bytes: rule.budget_bytes,
                    context_tokens: context as u64, pool_tokens: requested_pool, host_cache_bytes, mla_kv: format,
                    workspace_manifest: Some(manifest_path.clone()), onboard: Some(OnboardSetting::Auto),
                    spark_allocation_budget_bytes: Some(spark_bytes), ..Default::default() }), ..Default::default() };
            let report = plan_input.plan(&options)?;
            assert_eq!((report.prefill_unit_rows, report.prefill_waves, report.prefill_span_rows),
                (Some(unit as u64), Some(waves as u64), Some(rows as u64)));
            let catalog = plan_input.expert_catalog()?;
            let package = allocation::expert_package(&manifest_path)?;
            // Startup charges hook-local owners (TP2 workspace, Spark endpoint,
            // peer exchange, graphs) at the unit; row planes at the lane.
            let mut rtx = plan_rtx(&cfg, &catalog, package, unit, 92)?;
            rtx.layers = Onboard::Auto;
            let working = SparkWorkingSet { rtx_layers: vec![], spark_layers: (1..93).collect(),
                spark_ranks: 6, prefill_lanes: 1 };
            let rings = cuteafd_loader::plan::layout::spark_ring_geometry(1);
            let spark = SparkBudgetInventory::exl3(&catalog, &working,
                &[cuteafd_core::SparkEndpoint { ring_depth: None, capacity: unit as u32, ingress: cuteafd_core::SparkIngress::Fp8K32, partial_reply_bytes: None }],
                cuteafd_core::SparkRings { max_frame_bytes: rings.max_frame_bytes, depth: rings.depth,
                    slot_bytes: rings.slot_bytes, alignment: rings.alignment }, spark_bytes,
                cuteafd_loader::families::kimi_k3::spark::reserve_bytes(6, waves as u64, unit as u64)?)?;
            let programs = ProgramSet::from_manifest(&manifest, &["kimik3"]);
            let arch = ArchContext::for_device("sm_120", capacity);
            // The installed plan resolves the same pinned host retention this
            // case asks for; the runtime must be given that quota too, or its
            // coordinator ledger omits the retained prefix and its mapping.
            let prefix_cli = PrefixCli::try_parse_from(["serve", "--snapshot", "/fixture",
                "--native-lib", "/native",
                "--peers", "127.0.0.1:19000,127.0.0.1:19001,127.0.0.1:19002,127.0.0.1:19003,127.0.0.1:19004,127.0.0.1:19005",
                "--host-cache-bytes", &host_cache_bytes.unwrap_or(0).to_string()])?;
            let prefix_args = prefix_cli.serve.prefix;
            let mut i = inputs(&cfg);
            i.prefix = Some(&prefix_args);
            i.cache_format = format;
            i.requested_pool = requested_pool;
            i.rows = rows; i.waves = waves; i.unit_rows = Some(unit);
            i.lanes = 1; i.state_slots = slots as usize; i.prefix_marks = marks as usize;
            i.lane_layouts = vec![std::array::from_fn(|_| LaneLayout::new(&cfg, rows, test_unit(rows, i.waves, i.unit_rows), context,
                allocation::NativeScratch::from_manifest(&manifest).unwrap()).unwrap())];
            i.fixed = super::super::startup::exchange_demands(unit, false)?;
            i.startup = cuteafd_loader::families::kimi_k3::allocation::graph_reserve_demands(2)?;
            for rank in 0..2 {
                let code = cuteafd_loader::placement::inventory::loaded_code_allowance(
                    "kimik3", "exl3", true, rank as u8, arch, Some(&programs));
                // Model an identical post-module sample: context + resident code are
                // already allocated; actual native head handles are separate future owners.
                let resident = code.bytes;
                i.gpus[rank] = GpuBudget { capacity_bytes: capacity, reserve_bytes: cuteafd_loader::placement::reserve::gpu("kimi_k3"),
                    baseline: Baseline::Measured { free_bytes: capacity - arch.context_bytes - resident } };
                i.fixed.push(super::super::startup::code_demand(rank,
                    code.pending(arch.context_bytes + resident, arch.context_bytes))?);
            }
            i.rtx = Some(rtx); i.spark_budget = Some(spark);
            let admitted = match admit(i) {
                Ok(admitted) => {
                    ensure!(report.fits && report.placement_supported,
                        "concurrency={slots} waves={waves} format={format}: {:?}", report.hints);
                    admitted
                }
                Err(error) => {
                    assert!(!report.fits && !report.placement_supported,
                        "planner admitted a shape refused by runtime: {error:#}");
                    use cuteafd_loader::placement::planner::Refusal;
                    let typed = error.downcast_ref::<Refusal>()
                        .context("explicit Kimi capacity refusal must remain typed")?;
                    match typed {
                        Refusal::FullContextShortfall { context: floor, tokens, .. } =>
                            assert!(*floor == context as u64 && *tokens < *floor, "{typed}"),
                        Refusal::Capacity { required, available, .. } =>
                            assert!(required > available, "{typed}"),
                        Refusal::PoolBelowContext { tokens, context: floor } =>
                            assert!(*tokens < *floor, "{typed}"),
                        _ => panic!("unexpected Kimi capacity refusal: {typed}"),
                    }
                    assert!(report.hints.iter().any(|hint| hint.what.contains(&typed.to_string())),
                        "planner/runtime capacity refusal differs: {:?}; {error:#}", report.hints);
                    assert_eq!(report.memory_layout.context("refused Kimi ledger")?.pool_tokens, 0);
                    refused += 1;
                    continue;
                }
            };
            let memory = report.memory_layout.context("installed Kimi planner ledger")?;
            assert_eq!(memory.pool_tokens, admitted.placement.pool_tokens);
            assert_eq!(admitted.mark_selection.slots, marks);
            for rank in 0..2 {
                let planned = memory.devices.iter().find(|d| d.kind == DeviceKind::Rtx && d.index == rank as u32)
                    .context("planned RTX rank")?;
                let planned_marks = planned.items.iter().filter(|item| item.category == Category::Prefix)
                    .map(|item| item.bytes).sum::<u64>();
                assert_eq!(planned_marks, mark_bytes as u64 / 2 * marks,
                    "rank {rank}, concurrency {slots}, {format}: planner must charge runtime staging marks");
                let runtime_bytes = admitted.placement.items[rank].iter().map(|item| item.bytes).sum::<u64>();
                assert_eq!(planned.used_bytes(), runtime_bytes + cuteafd_loader::placement::reserve::gpu("kimi_k3"),
                    "rank {rank}, {format}: installed planner and runtime must charge identical owners");
            }
            assert!(admitted.placement.pool_tokens >= context as u64);
        }
        assert!(refused > 0, "the explicit-shape matrix must exercise a real capacity refusal");
        Ok(())
    }

    #[test]
    fn layers_graphs_and_native_owners_follow_the_admitted_uneven_split() -> Result<()> {
        let cfg = KimiK3Config::from_hf(&cuteafd_loader::plan::testing::kimi_k3_config())?;
        let mut i = inputs(&cfg);
        i.attention = Some(AttentionPlacement::Layers);
        let reserve = cuteafd_loader::families::kimi_k3::allocation::graph_reserve_demands(2)?;
        i.startup = reserve.clone();
        // A fixed asymmetric owner makes equal layer counts the wrong balance.
        let whole = LayerLayout::pair(&cfg, 3,
            LayerMode::Whole { gpu: 0, ffn: FfnMode::Split })?[0].attention_bytes()? as u64;
        i.fixed.push(Demand::new(1, Category::Runtime, "asymmetric runtime owner",
            3 * (whole + (2_097_152 * i.cache_format.record_bytes()) as u64), Basis::Exact));
        let admitted = admit(i)?;
        let owners = [0, 1].map(|rank| admitted.placement.layers.iter().filter(|layer|
            matches!(layer.mode, LayerMode::Whole { gpu, .. } if usize::from(gpu) == rank)).count());
        assert!(owners[0] > owners[1], "{owners:?}");
        let graphs = admitted.graph_set.as_ref().context("admitted graph set")?;
        super::super::graphs::check_placement(&cfg, &admitted.placement, &graphs.keys)?;
        for rank in 0..2 {
            let bytes = admitted.placement.items[rank].iter().filter(|item| item.group == super::super::graphs::RESERVE)
                .map(|item| item.bytes).sum::<u64>();
            assert_eq!(bytes, reserve[rank].bytes);
            for (layer, assignment) in admitted.placement.layers.iter().enumerate() {
                let layouts = LayerLayout::pair(&cfg, layer, assignment.mode)?;
                assert_eq!(admitted.layouts[layer][rank].bytes()?, layouts[rank].bytes()?);
            }
        }
        Ok(())
    }

    #[test]
    fn planner_equals_runtime_context_extent_at_256k_512k_and_1m() -> Result<()> {
        use clap::Parser;
        use cuteafd_loader::plan::{self, layout::LayoutOptions, testing, ExpertPlacement, PlanOptions};
        let snapshot = tempfile::tempdir()?;
        testing::write_kimi_snapshot(snapshot.path(), true);
        let plan_input = cuteafd_loader::plan::testing::BudgetedPlanInput::open(snapshot.path())?;
        let cfg = KimiK3Config::from_hf(&testing::kimi_k3_config())?;
        for format in [cuteafd_core::MlaCacheFormat::Bf16, cuteafd_core::MlaCacheFormat::Fp8] {
            let mut lane_bytes = Vec::new();
            let mut cache_bytes = Vec::new();
            for context in [262_144, 524_288, 1_048_576] {
                let text = context.to_string();
                let cli = crate::cli::Cli::try_parse_from(["cuteafd", "serve-kimi", "--snapshot", "/fixture",
                    "--native-lib", "/fixture.so", "--max-context-tokens", &text, "--pool-tokens", &text,
                    "--kv-cache", format.label()])?;
                let crate::cli::Commands::ServeKimi(args) = cli.command else { panic!("Kimi command") };
                let mut i = inputs(&cfg);
                i.cache_format = args.kv_cache;
                i.serving_context = cuteafd_loader::placement::planner::Context::pinned(context);
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
                i.serving_context = cuteafd_loader::placement::planner::Context::pinned(context);
                i.requested_pool = Some(context);
                for layouts in &mut i.lane_layouts {
                    for layout in layouts {
                        *layout = LaneLayout::new(&cfg, i.rows, test_unit(i.rows, i.waves, i.unit_rows), context as usize,
                            super::super::device::NativeScratch { kda: 1024, mla: [512, 0, 1024] })?;
                    }
                }
                let admitted = admit(i)?;
                let mut options = PlanOptions { attention_placement: Some(AttentionPlacement::Layers), placement: ExpertPlacement::Sparks { ranks: 6 },
                    layout: Some(LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![96 << 30; 2], concurrency: 2,
                        mla_kv: format, context_tokens: context, pool_tokens: Some(context), prefix_slots: Some(2),
                        host_cache_bytes: Some(0), onboard: Some(OnboardSetting::Auto), ..Default::default() }),
                    ..Default::default() };
                let report = plan_input.plan(&options)?;
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
                let refused = plan_input.plan(&options)?;
                assert!(!refused.fits, "planner must not shrink the explicit context floor");
            }
            let i = inputs(&cfg);
            let banks = i.lane_layouts[0][0].metadata_banks();
            assert_eq!(lane_bytes[1] - lane_bytes[0], banks << 20);
            assert_eq!(lane_bytes[2] - lane_bytes[0], 3 * banks << 20);
            assert_eq!(cache_bytes[1] - cache_bytes[0], 24 * (524_288 - 262_144) * format.record_bytes());
            assert_eq!(cache_bytes[2] - cache_bytes[0], 24 * (1_048_576 - 262_144) * format.record_bytes());
        }
        Ok(())
    }

    #[test]
    fn automatic_pool_serves_model_maximum_clamped_to_admission() -> Result<()> {
        use cuteafd_core::memory_policy::AGENTIC_FLOOR_TOKENS;
        use cuteafd_loader::plan::{self, layout::LayoutOptions, testing, ExpertPlacement, PlanOptions};
        let snapshot = tempfile::tempdir()?;
        testing::write_kimi_snapshot(snapshot.path(), true);
        let plan_input = cuteafd_loader::plan::testing::BudgetedPlanInput::open(snapshot.path())?;
        let cfg = KimiK3Config::from_hf(&testing::kimi_k3_config())?;
        for format in [cuteafd_core::MlaCacheFormat::Bf16, cuteafd_core::MlaCacheFormat::Fp8] {
            let mut i = inputs(&cfg);
            i.cache_format = format;
            i.serving_context = cuteafd_loader::placement::planner::Context::automatic(cfg.max_position as u64);
            i.lane_layouts = vec![std::array::from_fn(|_| LaneLayout::new(&cfg, i.rows, test_unit(i.rows, i.waves, i.unit_rows), cfg.max_position,
                super::super::device::NativeScratch { kda: 1024, mla: [512, 0, 1024] }).unwrap()); i.lanes];
            let request = request(i)?;
            assert_eq!(request.pool.floor, AGENTIC_FLOOR_TOKENS);
            assert_eq!(request.pool.compiled_context, cfg.max_position as u64);
            assert!(request.pool.target > request.pool.floor);
            assert!(solve(&request)?.pool_tokens > request.pool.floor);
            let report = plan_input.plan(&PlanOptions {
                attention_placement: Some(AttentionPlacement::Layers),
                placement: ExpertPlacement::Sparks { ranks: 6 },
                layout: Some(LayoutOptions { host_memory: Some((128 << 30, 128 << 30)), rtx_bytes: vec![96 << 30; 2], concurrency: 2,
                    mla_kv: format, prefix_slots: Some(2), host_cache_bytes: Some(0),
                    onboard: Some(OnboardSetting::Auto), ..Default::default() }),
                ..Default::default()
            })?;
            ensure!(report.fits && report.placement_supported, "{:?}", report.hints);
            let memory = report.memory_layout.context("Kimi default planned layout")?;
            assert!(memory.pool_tokens > AGENTIC_FLOOR_TOKENS);
            let served = (cfg.max_position as u64).min(memory.pool_tokens);
            assert!(memory.notes.iter().any(|note| note.contains(&format!("API max context: {served} tokens; compiled max context: 1048576 tokens"))));
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
            admitted.check_global(nonowner, nonowner as i32)?;
            assert!(!admitted.placement.items[nonowner].iter().any(|item| item.group.starts_with("Kimi layer ")));
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
        for attention in [Some(AttentionPlacement::Layers),Some(AttentionPlacement::Heads)] {
            let mut i=inputs(&cfg);i.core=core.clone();i.attention=attention;
            let mixed=admit(i)?;
            let mut i=inputs(&cfg);i.attention=attention;
            let original=admit(i)?;
            let expected_global=GlobalLayout::pair_with_core(&cfg,&core)?;
            for rank in 0..2 {
                assert_eq!(mixed.global_layouts[rank],expected_global[rank]);
                mixed.check_global(rank, rank as i32)?;
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
                let charge=|a:&Admission<'_>|a.layouts.iter().map(|pair| pair[rank].bytes().unwrap() as u64).sum::<u64>();
                let baseline_same_placement=mixed.placement.layers.iter().enumerate().map(|(layer,assignment)|
                    LayerLayout::pair(&cfg,layer,assignment.mode).unwrap()[rank].bytes().unwrap() as u64).sum::<u64>();
                assert_eq!(baseline_same_placement-charge(&mixed),freed[rank] as u64);
                assert!(freed[rank]>0);
                if attention == Some(AttentionPlacement::Heads) {assert_eq!(charge(&original)-charge(&mixed),freed[rank] as u64);}
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
        for attention in [Some(AttentionPlacement::Layers), Some(AttentionPlacement::Heads)] {
            for format in [cuteafd_core::MlaCacheFormat::Bf16, cuteafd_core::MlaCacheFormat::Fp8] {
                let mut input = inputs(&cfg);
                input.cache_format = format;
                input.attention = attention;
                let admitted = admit(input)?;
                for rank in 0..2 {
                    let layout = admitted.check_mla(rank, rank as i32)?;
                    let layers = if attention == Some(AttentionPlacement::Heads) { 2 } else { 1 };
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
                2 => record.basis = Basis::Formula,
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
            let charge = admitted.receipt.facts.allocations.iter_mut().find(|owner| owner.id == "kimi/coordinator/1").unwrap();
            match case {
                0 => charge.bytes -= 1,
                1 => charge.category = Category::Workspace,
                2 => charge.lifetime = planner::Lifetime::TARGET,
                3 => { let duplicate = charge.clone(); admitted.receipt.facts.allocations.push(duplicate); }
                _ => admitted.receipt.facts.allocations.retain(|owner| owner.id != "kimi/coordinator/1"),
            }
            assert!(admitted.check_global(1, 1).is_err(), "case {case}");
        }
        let mut i = inputs(&cfg);
        i.fixed.push(GlobalLayout::pair(&cfg)?[0].demand()?);
        assert!(admit(i).is_err());
        Ok(())
    }

    #[test]
    fn full_prefill_logits_charge_is_opt_in_and_precedes_allocation() -> Result<()> {
        let cfg = cfg();
        let ordinary = admit(inputs(&cfg))?;
        let mut scoring = inputs(&cfg);
        scoring.full_prefill_logits = true;
        let scoring = admit(scoring)?;
        for rank in 0..2 {
            for lane in 0..2 {
                let a = ordinary.io_layouts[rank].demand(lane)?;
                let b = scoring.io_layouts[rank].demand(lane)?;
                assert_eq!(b.bytes - a.bytes, if rank == 1 { (4096 - 64) * cfg.vocab_size as u64 * 4 } else { 0 });
                let charged = scoring.placement.items[rank].iter().find(|item| item.group == b.group).unwrap();
                assert_eq!(charged.bytes, b.bytes);
                scoring.check_io(rank, lane, rank as i32)?;
            }
        }
        let mut explicit_off = inputs(&cfg);
        explicit_off.full_prefill_logits = false;
        let explicit_off = admit(explicit_off)?;
        assert_eq!(serde_json::to_value(&ordinary.placement)?, serde_json::to_value(&explicit_off.placement)?);
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
        let ranks = i.spark_ranks;
        let lanes = i.lanes;
        let admitted = admit(i)?;
        let waves = admitted.waves;
        let unit = admitted.unit_rows;
        for lane in 0..lanes {
            let demand = super::super::spark::admitted_demand(waves, unit, lane, ranks, cuteafd_core::SparkReplyFormat::Bf16)?;
            let items = admitted.placement.items[1].iter().filter(|item| item.group == demand.group).collect::<Vec<_>>();
            assert_eq!(items.len(), 1);
            assert_eq!(items[0].bytes, ranks as u64
                * cuteafd_core::serving_capacity::spark_intake_allocation_bytes(
                    (waves * unit * 3584 * 2) as u64)?);
            assert_eq!(items[0].basis, Basis::Exact);
            assert_eq!(items[0].category, Category::Workspace);
            assert!(!admitted.placement.items[0].iter().any(|item| item.group == demand.group));
        }
        let mut duplicate = inputs(&cfg);
        duplicate.fixed.push(super::super::spark::admitted_demand(waves, unit, 0, ranks, cuteafd_core::SparkReplyFormat::Bf16)?);
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
            item.group == super::super::graphs::RESERVE && item.basis == Basis::Exact));
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
                0 => i.spark_ranks = 9,
                1 => i.peer_access = false,
                2 => i.attention = Some(AttentionPlacement::Context),
                3 => i.requested_pool = Some(262_143),
                4 => i.state_slots = cuteafd_core::concurrency::MAX_CONCURRENCY as usize + 1,
                5 => i.gpus[0].baseline = Baseline::Planned { context_bytes: 0, loaded_bytes: 0 },
                6 => i.fixed[0].basis = Basis::Estimated,
                7 => i.fixed[0].gpu = 2,
                8 => i.devices = [1, 1],
                _ => i.devices[0] = -1,
            }
            assert!(admit(i).is_err(), "case {case}");
        }
        let mut no_marks = inputs(&cfg);
        no_marks.prefix_marks = 0;
        assert_eq!(admit(no_marks).unwrap().mark_selection.slots, 0);
        let mut i = inputs(&cfg);
        i.requested_pool = Some(2 << 20);
        i.gpus[1].baseline = Baseline::Measured { free_bytes: 2 << 30 };
        assert!(admit(i).is_err());
    }
}
