//! Retained coordinator submission owner. Startup supplies admitted operands and
//! ordinary Spark links; this uses the shared executor and terminal sequence.
use super::{device::{LocalLane, SequenceRows, TokenIo}, layers::Execution, spark::RoutedLane};
use crate::shared::{experts::rtx::pair::{submit_scoped, teardown, Teardown},
    peer_split::{hop::HopLink, PeerExchange, TerminalState}};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::placement::LayerExecutionPlan;
use std::{cell::{Cell, RefCell}, rc::Rc};
use crate::shared::memory::device::Stream;

struct Lanes<'a> {
    local: Vec<Vec<LocalLane<'a>>>,
    routed: Vec<RoutedLane<'a>>,
    io: Vec<Vec<TokenIo<'a>>>,
    experts: Vec<Vec<Option<crate::shared::experts::rtx::exl3::Exl3Tp2<'a>>>>,
}

pub(crate) struct Engine<'a> {
    pub max_pool_tokens_no_onboard: Option<u64>,
    pub host_prefix: Option<cuteafd_hostcache::config::Config>,
    target_view: cuteafd_loader::plan::checkpoint::CheckpointView,
    draft: Option<Rc<super::draft::NativeDraft<'a>>>,
    draft_probe: RefCell<Option<crate::shared::memory::device::DeviceOwner<'a,
        crate::shared::draft::probe::RoundProbe<'a>>>>,
    execution: Execution<'a>,
    lanes: RefCell<Lanes<'a>>,
    prefix: RefCell<Option<super::prefix::KimiPrefix<'a>>>,
    provisional: RefCell<Option<super::state::Provisional>>,
    hops: Option<HopLink<'a>>,
    peer: Option<Rc<PeerExchange<'a>>>,
    terminal: Cell<TerminalState>,
    // PeerExchange and HopLink borrow raw handles; keep their exact streams
    // after both links and their terminal abort storage have been released.
    exchange_streams: Vec<Rc<Stream<'a>>>,
    /// Prefill units: the unit and the waves in flight.
    prefill: PrefillUnits,
    packed_row_cap: Cell<Option<usize>>,
    coordinator_packing_cap: usize,
    /// Admission's API/scheduler limit; physical lane and prefix owners stay compiled-wide.
    served_context: usize,
    /// The same shape as the shared prefill depth.
    depth: crate::shared::prefill_depth::PrefillDepth,
}

/// The lane holds `waves` units of `unit_rows`; a step of `rows` is cut into
/// equal units of at most `unit_rows` (`prefill_units`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PrefillUnits { pub unit_rows: usize, pub waves: usize }

fn packed_row_cap(mode: crate::shared::prefill_packing::PackingMode,
    coordinator: usize, expert: usize) -> Option<usize> {
    (mode == crate::shared::prefill_packing::PackingMode::Exact).then_some(coordinator.min(expert))
}

impl Drop for Engine<'_> {
    fn drop(&mut self) {
        crate::shared::decode_graph::fatal_drain(self.shutdown(), "Kimi engine teardown");
    }
}

fn check_exchange_streams(peer: &PeerExchange<'_>, streams: &[Rc<Stream<'_>>; 2]) -> Result<()> {
    ensure!(std::ptr::eq(streams[0].device.library, streams[1].device.library),
        "Kimi exchange stream native libraries differ");
    check_exchange_handles(std::array::from_fn(|rank| (peer.device(rank), peer.stream(rank))),
        streams.each_ref().map(|stream| (stream.device.id, stream.raw)))
}

fn check_exchange_handles(expected: [(i32, *mut std::ffi::c_void); 2],
    actual: [(i32, *mut std::ffi::c_void); 2]) -> Result<()> {
    ensure!(expected[0].0 != expected[1].0 && expected == actual
        && actual.iter().all(|(_, stream)| !stream.is_null()),
        "Kimi exchange stream owners differ from exact peer/hop handles");
    Ok(())
}

fn check_expert_owners<const RANKS: usize>(layers: &[cuteafd_loader::placement::LayerAssignment],
    owners: [&[usize]; RANKS]) -> Result<()> {
    use cuteafd_loader::placement::ExpertHome;
    ensure!(matches!(RANKS, 1 | 2), "Kimi expert physical owner count");
    ensure!(layers.iter().all(|layer| match layer.experts {
        ExpertHome::RtxTp2 => RANKS == 2,
        ExpertHome::RtxWhole { gpu } => RANKS == 1 && gpu == 0,
        _ => true,
    }), "Kimi expert placement differs from physical owner geometry");
    for (rank, owned) in owners.iter().enumerate() {
        let expected = layers.iter().enumerate().filter_map(|(layer, assignment)|
            (assignment.experts == ExpertHome::RtxTp2
                || assignment.experts == ExpertHome::RtxWhole { gpu: rank as u8 }).then_some(layer))
            .collect::<Vec<_>>();
        ensure!(*owned == expected, "Kimi expert owners differ from accepted per-layer placement");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_packing_runs_unknown_exl3_and_k6_arithmetic_routes_alone() {
        use crate::shared::prefill_packing::{self, PackingMode, Limits};
        let chunks = [(0, 17), (4, 31), (0, 65)];
        let groups = |mode| {
            let limits = Limits { rows: packed_row_cap(mode, 64, 0).unwrap_or(2752), chunk_rows: 2752,
                max_context: 1024, taps: Some(2048) };
            prefill_packing::groups(chunks.len(), |range| prefill_packing::plan(&chunks[range], &limits).is_ok())
        };
        assert_eq!(groups(PackingMode::Exact), [0..1, 1..2, 2..3]);
        assert_eq!(groups(PackingMode::Unrestricted), [0..3]);
    }

    #[test]
    fn exact_packing_intersects_coordinator_and_expert_routes() {
        use crate::shared::prefill_packing::{self, PackingMode, Limits};
        assert_eq!(packed_row_cap(PackingMode::Exact, 64, 128), Some(64));
        assert_eq!(packed_row_cap(PackingMode::Exact, 128, 32), Some(32));
        assert_eq!(packed_row_cap(PackingMode::Exact, 0, 128), Some(0));
        assert_eq!(packed_row_cap(PackingMode::Unrestricted, 0, 0), None);
        let chunks = [(0, 29), (0, 31), (0, 80), (0, 17)];
        let limits = Limits { rows: packed_row_cap(PackingMode::Exact, 64, 128).unwrap(),
            chunk_rows: 256, max_context: 1024, taps: Some(2048) };
        assert_eq!(prefill_packing::groups(chunks.len(), |range|
            prefill_packing::plan(&chunks[range], &limits).is_ok()), [0..2, 2..3, 3..4]);
        assert_eq!(chunks[2].1, 80, "oversized singleton is not rechunked");
    }

    #[test]
    fn whole_expert_bindings_require_exact_sparse_owner_sets() -> Result<()> {
        use cuteafd_loader::placement::{LayerAssignment, LayerMode, ExpertHome};
        let layers = [ExpertHome::Dense, ExpertHome::RtxTp2,
            ExpertHome::RtxTp2, ExpertHome::RtxTp2, ExpertHome::Spark]
            .map(|experts| LayerAssignment { mode: LayerMode::HeadSplit, experts });
        check_expert_owners(&layers, [&[1, 2, 3], &[1, 2, 3]])?;
        for wrong in [[&[1, 2, 3][..], &[2][..]], [&[1, 3][..], &[][..]],
            [&[2][..], &[1, 3][..]]] {
            assert!(check_expert_owners(&layers, wrong).is_err());
        }
        Ok(())
    }

    #[test]
    fn exchange_owners_require_exact_rank_device_and_stream() {
        let handles = [(0, 1usize as *mut _), (1, 2usize as *mut _)];
        assert!(check_exchange_handles(handles, handles).is_ok());
        assert!(check_exchange_handles(handles, [handles[1], handles[0]]).is_err());
        assert!(check_exchange_handles(handles, [(0, 3usize as *mut _), handles[1]]).is_err());
        assert!(check_exchange_handles(handles, [(2, handles[0].1), handles[1]]).is_err());
        let null = [(0, std::ptr::null_mut()), handles[1]];
        assert!(check_exchange_handles(null, null).is_err());
        let same_device = [handles[0], (0, handles[1].1)];
        assert!(check_exchange_handles(same_device, same_device).is_err());
    }
}

impl<'a> Engine<'a> {
    /// Complete startup from the measured shared admission and already charged
    /// exchange owners. Worker reconciliation precedes ordinary lane connection.
    pub(crate) fn load_admitted<const RANKS: usize>(library: &'a std::sync::Arc<cuteafd_ffi::NativeLibrary>,
        exchange_streams: [Rc<Stream<'a>>; RANKS],
        lane_streams: [Rc<Stream<'a>>; RANKS],
        mut draft: Option<super::draft::NativeDraft<'a>>,
        draft_records: crate::shared::experts::rtx::dense::Records,
        mut admission: super::admission::Admission<'a>,
        usage_reader: cuteafd_core::expert_usage::Reader,
        checkpoint: &cuteafd_loader::plan::Checkpoint,
        programs: [std::rc::Rc<super::device::NativePrograms<'a>>; RANKS],
        context_programs: [Option<Rc<super::context::Programs<'a>>>; RANKS],
        peer: Option<Rc<PeerExchange<'a>>>, hops: Option<HopLink<'a>>, peers: Vec<std::net::SocketAddr>,
        config: cuteafd_transport::TcpTransportConfig,
        mut expert_modules: [Option<Vec<crate::shared::experts::exl3::execution::Exl3Modules<'a>>>; RANKS],
        memory: &mut crate::shared::inventory::StartupMemoryPhases<'_>) -> Result<Self> {
        use crate::shared::placement::{spark_rings, SparkAdmission};
        use cuteafd_core::{ExpertGeometry, SparkEndpoint, SparkIngress};
        use cuteafd_transport::{expert::SparkWorld, worker_selection::ModelAdmission};
        ensure!(cuteafd_core::expert_geometry() == ExpertGeometry::KIMI_K3 && (1..=8).contains(&peers.len()),
            "Kimi startup needs process latent geometry and 1..8 real expert ranks");
        let coordinator_packing_cap = programs.iter().map(|programs| programs.packing_cap).min().unwrap_or(0);
        let plan = admission.execution_plan()?;
        let expert_usage = admission.expert_usage.take().map(|geometry| {
            let devices = cuteafd_loader::placement::families::kimi::usage_devices(&geometry, RANKS)
                .into_iter().map(|logical| cuteafd_ffi::expert_usage::UsageDevice {
                    logical, cuda: programs[usize::from(logical)].device.id }).collect::<Vec<_>>();
            cuteafd_ffi::expert_usage::ExpertUsage::new(library, geometry, &devices, usage_reader)
                .map(Rc::new)
        }).transpose()?;
        memory.sample("expert-usage")?;
        cuteafd_bench::context::set_rtx_expert_layers(admission.placement.onboard_layers);
        ensure!(matches!(RANKS, 1 | 2), "Kimi startup physical rank count");
        let devices = programs.each_ref().map(|programs| programs.device);
        ensure!(devices.iter().all(|device| std::ptr::eq(device.library, library.as_ref()))
            && exchange_streams.iter().zip(devices).all(|(stream, device)|
                stream.device.id == device.id && std::ptr::eq(stream.device.library, device.library)),
            "Kimi startup programs differ from exchange devices/library");
        match (&peer, &hops) {
            (Some(peer), Some(hops)) if RANKS == 2 => {
                hops.validate_binding(&plan, peer)?;
                check_exchange_streams(peer, &[exchange_streams[0].clone(), exchange_streams[1].clone()])?;
                ensure!((0..RANKS).all(|rank| devices[rank].id == peer.device(rank)),
                    "Kimi startup program devices differ from peer ranks");
            }
            (None, None) if RANKS == 1 => ensure!(plan.hops().is_empty(), "Kimi single owner cannot have hops"),
            _ => anyhow::bail!("Kimi exchange inventory differs from physical rank count"),
        }
        let rows = u32::try_from(admission.unit_rows())?;
        let endpoints = vec![SparkEndpoint { ring_depth: None, capacity: rows, ingress: SparkIngress::Fp8K32, partial_reply_bytes: None };
            plan.spec().lanes as usize];
        let catalog = cuteafd_loader::read_expert_catalog_view(checkpoint.view.as_ref()
            .context("Kimi engine requires the immutable selected checkpoint view")?)?;
        let executions = cuteafd_loader::plan::spark_executions::admit_family("kimi_k3",
            &cuteafd_loader::plan::spark_executions::SparkBackend::for_catalog(&catalog)?.exports(), None,
            u64::from(rows), admission.waves() as u64)?;
        let model = ModelAdmission::new(endpoints, spark_rings(config.max_frame_bytes)?, executions)?;
        model.validate(ExpertGeometry::KIMI_K3)?;
        let spark = SparkAdmission::plan_view(checkpoint.view.as_ref()
            .context("Kimi bootstrap requires the immutable selected checkpoint view")?, ExpertGeometry::KIMI_K3,
            Some(&catalog), &admission.working_set(), peers.len(), model, None)?;
        // Dense operands and TP2 halves land inside the Spark load (pushed as
        // coordinator ranks when the group pushes, else through the mount).
        let (pending, expert_weights, weights) = admission.start_and_load(spark, &peers, devices, checkpoint, &catalog,
            exchange_streams.each_ref().map(|stream| stream.raw), draft_records, memory)?;
        let lanes = usize::try_from(plan.spec().lanes)?;
        let allocated = (0..lanes).map(|lane| (0..RANKS).map(|rank|
            admission.allocate_lane_on(devices[rank], rank, lane, lane_streams[rank].clone()))
            .collect::<Result<Vec<_>>>()).collect::<Result<Vec<_>>>()?;
        memory.sample("lane-workspaces")?;
        let kda = (0..RANKS).map(|rank| admission.allocate_kda(devices[rank], rank)).collect::<Result<Vec<_>>>()?;
        memory.sample("kda-state")?;
        let mla = (0..RANKS).map(|rank| admission.allocate_mla(devices[rank], rank)).collect::<Result<Vec<_>>>()?;
        memory.sample("mla-kv")?;
        ensure!(context_programs.iter().all(Option::is_some) == admission.context_layout().is_some()
            && (admission.context_layout().is_some() || context_programs.iter().all(Option::is_none)),
            "Kimi context programs differ from admitted storage");
        let context = if admission.context_layout().is_some() {
            let peer = peer.as_deref().context("Kimi context missing ordinary peer abort owner")?;
            Some(admission.allocate_context(library,
                [lane_streams[0].clone(), lane_streams[1].clone()],
                [peer.abort_word(0)?.context("Kimi context missing rank 0 abort word")?,
                    peer.abort_word(1)?.context("Kimi context missing rank 1 abort word")?])?)
        } else { None };
        memory.sample("context-workspace")?;
        let mut experts = Vec::with_capacity(lanes);
        let mut local = Vec::with_capacity(lanes);
        let mut io = Vec::with_capacity(lanes);
        for (lane, allocated) in allocated.into_iter().enumerate() {
            experts.push((0..RANKS).map(|rank| expert_weights[rank].as_ref()
                .map(|owner| owner.execution_with_modules(expert_modules[rank].take())).transpose())
                .collect::<Result<Vec<_>>>()?);
            local.push(allocated.into_iter().enumerate().map(|(rank, allocated)| {
                let mut local = admission.bind_local(rank, lane, allocated, programs[rank].clone())?;
                if let Some(owner) = &context {
                    local.bind_context(owner.clone(), context_programs[rank].as_ref()
                        .context("Kimi context missing rank programs")?.clone())?;
                }
                Ok(local)
            }).collect::<Result<Vec<_>>>()?);
            io.push((0..RANKS).map(|rank| admission.allocate_io(devices[rank], rank, lane))
                .collect::<Result<Vec<_>>>()?);
        }
        memory.sample("expert-execution-and-target-io")?;
        for owners in &experts {
            use crate::shared::experts::rtx::RtxExpertLayer;
            check_expert_owners(&admission.placement.layers,
                std::array::from_fn::<_, RANKS, _>(|rank| owners[rank].as_ref().map_or(&[][..], |e| e.layers())))?;
        }
        // Admission charged this arena before allocation; retain its exact device owner.
        let draft_probe = draft.as_ref().map(|_| devices[0].own(||
            crate::shared::draft::probe::RoundProbe::new(devices[0].library,
                local[0][0].config().dense.clone(), local[0][0].config().topk, 64))).transpose()?;
        // Weak admission registries must not outlive these strong startup owners
        // until every persistent local binding has acquired the same operands.
        if let Some(draft) = draft.as_mut() { draft.bind_target_ranks(weights.global.clone(), programs[RANKS - 1].clone())?; }
        memory.sample("draft-probe-and-target-bind")?;
        drop((weights, kda, mla));
        pending.wait()?;
        memory.sample("workers-loaded")?;
        let world = SparkWorld::canonical(peers.len())?;
        let routed = local.iter().map(|owners| admission.connect_spark(library, &owners[RANKS - 1],
            world, peers.clone(), config.clone())).collect::<Result<Vec<_>>>()?;
        memory.sample("registered-transport-egress-staging")?;
        let prefill = PrefillUnits { unit_rows: admission.unit_rows(), waves: admission.waves() };
        let target_view = checkpoint.view.as_ref()
            .context("Kimi engine requires the immutable selected checkpoint view")?.clone();
        let mut engine = Self::bind(exchange_streams.into(), plan, peer, hops,
            local, routed, io, experts, prefill, coordinator_packing_cap, target_view)?;
        engine.execution.expert_usage = expert_usage;
        engine.max_pool_tokens_no_onboard = admission.placement.max_pool_tokens_no_onboard;
        engine.host_prefix = admission.host_prefix.take();
        engine.draft = draft.map(Rc::new);
        *engine.draft_probe.borrow_mut() = draft_probe;
        Ok(engine)
    }

    /// Bind only exact owners already allocated through Admission. No device
    /// allocations or program loading occur after this constructor returns.
    pub(crate) fn bind(exchange_streams: Vec<Rc<Stream<'a>>>,
        plan: LayerExecutionPlan, peer: Option<Rc<PeerExchange<'a>>>, hops: Option<HopLink<'a>>,
        local: Vec<Vec<LocalLane<'a>>>, routed: Vec<RoutedLane<'a>>,
        io: Vec<Vec<TokenIo<'a>>>,
        experts: Vec<Vec<Option<crate::shared::experts::rtx::exl3::Exl3Tp2<'a>>>>,
        prefill: PrefillUnits, coordinator_packing_cap: usize,
        target_view: cuteafd_loader::plan::checkpoint::CheckpointView) -> Result<Self> {
        ensure!(matches!(exchange_streams.len(), 1 | 2) && !local.is_empty()
            && local.len() == routed.len() && local.len() == io.len() && local.len() == experts.len(),
            "Kimi engine physical ranks or local/Spark/IO lane counts differ");
        ensure!(prefill.waves > 0 && local.iter().all(|ranks| ranks.len() == exchange_streams.len()
            && ranks.iter().all(|local| local.rows() >= prefill.unit_rows * prefill.waves || prefill.waves == 1)),
            "Kimi lane holds fewer ranks or rows than admitted");
        for (lane, ranks) in local.iter().enumerate() {
            ensure!(io[lane].len() == ranks.len() && experts[lane].len() == ranks.len(),
                "Kimi lane IO/expert physical rank count differs");
            for rank in 0..ranks.len() {
                io[lane][rank].check_lane(&ranks[rank], rank)?;
                ensure!(std::ptr::eq(ranks[rank].stream().device.library,
                    exchange_streams[rank].device.library), "Kimi local and exchange native libraries differ");
            }
        }
        let streams = local.iter().map(|ranks| ranks.iter().map(|local| local.stream().clone()).collect()).collect();
        let execution = Execution::new(plan, peer.as_deref(), hops.as_ref(), exchange_streams.clone(), streams)?;
        let depth = crate::shared::prefill_depth::PrefillDepth::new(prefill.waves, prefill.unit_rows)?;
        let served_context = local[0][0].max_context();
        Ok(Self { host_prefix: None, max_pool_tokens_no_onboard: None, target_view, depth, draft: None, draft_probe: RefCell::new(None), execution, lanes: RefCell::new(Lanes { local, routed, io, experts }),
            prefix: RefCell::new(None), provisional: RefCell::new(None),
            hops, peer, terminal: Cell::new(TerminalState::Active), exchange_streams, prefill, packed_row_cap: Cell::new(None), coordinator_packing_cap, served_context })
    }

    pub(crate) fn with_served_context(mut self, context: usize) -> Result<Self> {
        ensure!(context > 0 && context <= self.served_context,
            "Kimi served context exceeds compiled lane extent");
        self.served_context = context;
        Ok(self)
    }

    pub(crate) fn target_view(&self) -> &cuteafd_loader::plan::checkpoint::CheckpointView {
        &self.target_view
    }

    pub(crate) fn prefix_family(&self) -> Result<super::prefix::KimiPrefix<'a>> {
        let mut prefix = self.prefix.try_borrow_mut()?;
        if let Some(prefix) = prefix.as_ref() { return Ok(prefix.clone()); }
        let lanes = self.lanes.try_borrow()?;
        let (kda, mla) = lanes.local[0].iter().map(|lane| lane.prefix_banks()).unzip();
        let family = super::prefix::KimiPrefix::from_ranks(lanes.local[0][0].config(), lanes.local[0][0].max_context(),
            kda, mla)?.with_draft(self.draft.clone())?;
        *prefix = Some(family.clone());
        Ok(family)
    }

    /// Units of one call at the engine's prefill shape.
    fn prefill_units(&self, rows: usize) -> Result<Vec<crate::shared::prefill_pipeline::Unit>> {
        let (layers, depth) = (self.execution.plan.modes().len(), self.depth);
        let units = crate::shared::prefill_pipeline::prefill_units(rows, depth.unit_rows, depth.waves, 256)?;
        // Every unit's rows lie in the lane; a slot is the unit's row range.
        Ok(units.into_iter().enumerate().map(|(index, rows)| crate::shared::prefill_pipeline::Unit {
            index, sequence: 0, segments: Vec::new(), rows, layers: 0..layers, slot: index % depth.waves, barrier: false }).collect())
    }

    pub(crate) fn geometry(&self) -> Result<crate::shared::qwen::scheduler::Geometry> {
        let lanes = self.lanes.try_borrow()?;
        let local = &lanes.local[0][0];
        let (state, _) = local.prefix_banks();
        Ok(crate::shared::qwen::scheduler::Geometry { family: "kimi_k3",
            vocab_size: local.config().vocab_size, max_context: self.served_context,
            // A call is the slots' units.
            slots: state.slots(), prefill_rows: local.rows().min(self.depth.call_rows()), decode_rows: local.rows().min(64),
            full_prefill_logits: lanes.io[0][self.head()].full_prefill_logits() })
    }

    pub(crate) fn configure_packing(&self, mode: crate::shared::prefill_packing::PackingMode) -> Result<()> {
        use crate::shared::prefill_packing::log_mode;
        // Coordinator metadata and the expert route both constrain exact packs.
        // EXL3 arithmetic bounds remain unknown, so that route still runs alone.
        let cap = packed_row_cap(mode, self.coordinator_packing_cap, 0);
        self.packed_row_cap.set(cap);
        log_mode(mode, cap.unwrap_or(self.prefill.unit_rows));
        Ok(())
    }

    pub(crate) fn packed_limits(&self) -> Result<crate::shared::prefill_packing::Limits> {
        Ok(crate::shared::prefill_packing::Limits {
            rows: self.packed_row_cap.get().unwrap_or(self.prefill.unit_rows),
            chunk_rows: self.prefill.unit_rows, max_context: self.geometry()?.max_context,
            taps: self.draft.as_ref().map(|_| crate::shared::speculation::dflash::TAP_ROWS),
        })
    }

    /// One queued unit; state remains member-local and shared head scratch is
    /// consumed before projecting the next member. Callback failures are local.
    pub(crate) fn prefill_packed(&self, sequences: &mut [(&mut super::prefix::Placement, &[u32])],
        pool: &cuteafd_engine::prefix::RefPagePool,
        on_logits: &mut dyn FnMut(usize, crate::shared::token_io::DeviceLogits)
            -> crate::shared::prefill_packing::MemberOutcome)
        -> Result<Vec<Result<()>>> {
        self.prefill_packed_heads::<false>(sequences, pool, &mut |i, _, logits| on_logits(i, logits))
    }

    #[cfg(test)]
    pub(crate) fn prefill_packed_all_rows(&self, sequences: &mut [(&mut super::prefix::Placement, &[u32])],
        pool: &cuteafd_engine::prefix::RefPagePool,
        on_logits: &mut dyn FnMut(usize, usize, crate::shared::token_io::DeviceLogits)
            -> crate::shared::prefill_packing::MemberOutcome) -> Result<Vec<Result<()>>> {
        self.prefill_packed_heads::<true>(sequences, pool, on_logits)
    }

    fn prefill_packed_heads<const ALL_ROWS: bool>(&self,
        sequences: &mut [(&mut super::prefix::Placement, &[u32])],
        pool: &cuteafd_engine::prefix::RefPagePool,
        on_logits: &mut dyn FnMut(usize, usize, crate::shared::token_io::DeviceLogits)
            -> crate::shared::prefill_packing::MemberOutcome) -> Result<Vec<Result<()>>> {
        self.require_live()?;
        ensure!(self.provisional.borrow().is_none(), "Kimi packed prefill cannot overwrite pending verify");
        let segments = crate::shared::prefill_packing::plan(&sequences.iter()
            .map(|(p, tokens)| (p.len, tokens.len())).collect::<Vec<_>>(), &self.packed_limits()?)?;
        let family = self.prefix_family()?;
        for (placement, tokens) in sequences.iter() { family.check_append(placement, tokens.len())?; }
        if let Some(draft) = &self.draft { draft.prepare_target()?; }
        submit_scoped(|| self.require_live(), || self.shutdown(), |scope| {
            let mut lanes = self.lanes.try_borrow_mut()?;
            for local in lanes.local.iter_mut().flatten() {
                local.graph_cycle(crate::shared::decode_graph::device::Cycle::Prefill);
            }
            let metadata = sequences.iter().map(|(p, tokens)| SequenceRows {
                slot: p.slot, start: p.len, rows: tokens.len(), pages: &p.units,
            }).collect::<Vec<_>>();
            let tokens = sequences.iter().flat_map(|(_, tokens)| tokens.iter().copied()).collect::<Vec<_>>();
            lanes.io[0][0].embed(&tokens, lanes.local[0][0].buffer("delta")?)?;
            let Lanes { local, routed, io, experts } = &mut *lanes;
            let normalized = self.execution.run_packed(scope, self.peer.as_deref(), self.hops.as_ref(),
                &mut local[0], &mut routed[0], &mut experts[0], &metadata, pool, self.draft.as_deref())?;
            drop(metadata);
            for (placement, tokens) in sequences.iter_mut() {
                family.commit(placement, placement.len, tokens.len())?;
            }
            crate::shared::prefill_packing::finish_members(sequences.len(), |i| Ok(i), &mut |i, _| {
                let segment = segments[i];
                let first = if ALL_ROWS { 0 } else { segment.rows - 1 };
                for offset in (first..segment.rows).step_by(64) {
                    let count = if ALL_ROWS { (segment.rows - offset).min(64) } else { 1 };
                    let input = crate::shared::prefix::view(normalized,
                        (segment.first + offset) * 7168 * 2, count * 7168 * 2)?;
                    let logits = io[0][self.head()].project(input, count, &local[0][self.head()])?;
                    let row = crate::shared::token_io::DeviceLogits { ptr: logits.ptr.cast_const(), rows: count,
                        vocab: local[0][self.head()].config().vocab_size, stride: local[0][self.head()].config().vocab_size,
                        stream: local[0][self.head()].stream().raw, greedy: None };
                    // Consume every diagnostic tile before reusing shared head scratch.
                    if let Err(error) = on_logits(i, offset, row)? { return Ok(Err(error)); }
                }
                Ok(Ok(()))
            })
        })
    }

    pub(crate) fn arm_draft_probe(&self) -> Result<()> {
        self.require_live()?;
        ensure!(self.draft_probe.borrow().is_some(), "Kimi draft policy probe was not admitted");
        Ok(())
    }
    pub(crate) fn probe_begin(&self, rows: usize) {
        if let Some(probe) = self.draft_probe.borrow_mut().as_mut() { probe.begin(rows); }
    }
    pub(crate) fn probe_end(&self) {
        if let Some(probe) = self.draft_probe.borrow_mut().as_mut() { probe.end(); }
    }
    pub(crate) fn probe_finish<T>(&self, observe: impl FnOnce(&crate::shared::draft::routes::RoundRoutes,
        &[Option<f64>]) -> T) -> Option<T> {
        let mut probe = self.draft_probe.borrow_mut();
        let probe = probe.as_mut()?;
        let device = probe.device;
        device.run(|| Ok(probe.finish().map(|(routes, times)| observe(routes, &times))))
            .unwrap_or_else(|error| { tracing::debug!(%error, "Kimi draft probe read failed"); None })
    }

    pub(crate) fn native_draft(&self) -> Option<&super::draft::NativeDraft<'a>> { self.draft.as_deref() }

    pub(crate) fn draft_geometry(&self, snapshot: &std::path::Path, width: usize)
        -> Result<cuteafd_core::PolicyGeometry> {
        use crate::shared::{draft::resource::{self, LayerHome}, experts::rtx::RtxExpertLayer};
        ensure!(snapshot == self.target_view.snapshot(), "Kimi draft target snapshot differs from selected view");
        let catalog = cuteafd_loader::read_expert_catalog_view(&self.target_view)?;
        let manifest = catalog.exl3().ok_or_else(|| anyhow::anyhow!("Kimi draft needs EXL3 headers"))?;
        let lanes = self.lanes.try_borrow()?;
        let cfg = lanes.local[0][0].config();
        let reports = lanes.routed[0].installed_grouping()?;
        let local = &lanes.experts[0];
        let homes = cfg.dense.iter().enumerate().map(|(layer, dense)| {
            if *dense { return Ok(LayerHome::Dense); }
            let resident = local[0].as_ref().is_some_and(|rank| rank.layers().contains(&layer));
            ensure!(local.iter().all(|rank| resident == rank.as_ref().is_some_and(|rank| rank.layers().contains(&layer))),
                "Kimi TP2 draft policy layer homes disagree");
            let bytes = resource::exl3_slice_bytes(manifest, layer, if resident { local.len() } else { lanes.routed[0].world_size()? })?;
            Ok(if resident { LayerHome::Local { slice_bytes: bytes } }
                else { LayerHome::Remote { slice_bytes: bytes } })
        }).collect::<Result<Vec<_>>>()?;
        resource::geometry_with_grouping(&homes, cfg.experts, cfg.topk, 64, width, 0, |layer, rows| {
            if matches!(homes[layer], LayerHome::Local { .. }) {
                let group = local[0].as_ref().unwrap().group_rows(rows)?;
                ensure!(local.iter().map(|rank| rank.as_ref().unwrap().group_rows(rows)).collect::<Result<Vec<_>>>()?.iter().all(|&rows| rows == group),
                    "Kimi TP2 draft policy grouping disagrees");
                Ok(group)
            } else { resource::worker_group_rows_for_input(&reports, layer, rows, false) }
        })
    }

    pub(crate) fn library(&self) -> &'a cuteafd_ffi::NativeLibrary {
        self.exchange_streams[0].device.library
    }

    fn head(&self) -> usize { self.execution.plan.spec().head_gpu as usize }

    pub(crate) fn devices(&self) -> Vec<i32> {
        self.exchange_streams.iter().map(|stream| stream.device.id).collect()
    }

    pub(crate) fn require_live(&self) -> Result<()> {
        ensure!(self.terminal.get() == TerminalState::Active, "Kimi engine is terminal ({:?})", self.terminal.get());
        if let Some(peer) = &self.peer { peer.require_live()?; }
        if let Some(draft) = &self.draft { draft.require_live()?; }
        Ok(())
    }

    pub(crate) fn attach_capture(&self, capture: Rc<super::golden::Capture>) -> Result<()> {
        self.require_live()?;
        let mut lanes = self.lanes.try_borrow_mut()?;
        ensure!(lanes.local.len() == 1 && self.provisional.borrow().is_none(),
            "Kimi capture requires one idle diagnostic lane after startup");
        for local in &mut lanes.local[0] { local.capture = Some(capture.clone()); }
        Ok(())
    }

    pub(crate) fn graph_replays(&self) -> Result<u64> {
        Ok(self.lanes.try_borrow()?.local.iter().flatten().map(LocalLane::graph_replays).sum())
    }

    pub(crate) fn set_grouped_kda(&self, enabled: bool) -> Result<()> {
        for local in self.lanes.try_borrow()?.local.iter().flatten() { local.set_grouped_kda(enabled)?; }
        Ok(())
    }

    pub(crate) fn usage_phase(&self, phase: cuteafd_ffi::expert_usage::UsagePhase) {
        self.execution.usage_phase.set(phase);
    }

    /// Prewarm the pinned geometry on every lane and measure the startup arena
    /// once. `reserves` is one bounded graph reserve per physical rank.
    pub(crate) fn warm_graphs(&self, keys: &[super::graphs::Key], reserves: &[u64]) -> Result<usize> {
        ensure!(reserves.len() == self.exchange_streams.len(), "Kimi graph reserve per physical rank");
        // No route for this family has calibrated node/executable coefficients,
        // so the whole device selects capture from measured bytes.
        tracing::warn!("Kimi graph cache uses per-site byte estimates: no calibrated node rates for this route");
        let mut lanes = self.lanes.try_borrow_mut()?;
        let mut consumed = 0;
        for (lane, pair) in lanes.local.iter_mut().enumerate() {
            for (rank, local) in pair.iter_mut().enumerate() {
                let binding = keys.iter().copied().filter(|key| key.lane == lane && key.rank == rank).collect::<Vec<_>>();
                ensure!(!binding.is_empty(), "Kimi graph inventory omits a binding");
                local.prewarm_kda_graphs(&binding, rank)?;
                // KDA has no peer waits. Serialize the startup bindings that
                // share slot0 before another lane can write that state bank.
                local.stream().drain()?;
                consumed += binding.len();
            }
        }
        ensure!(consumed == keys.len(), "Kimi startup graph inventory has unbound keys");
        // Submit every binding first. No zero may race another lane's warm
        // writes to the shared bank; restore drains all retained readers.
        for local in lanes.local.first().into_iter().flatten() { local.restore_startup_kda()?; }
        let mut captured = 0;
        for (lane, pair) in lanes.local.iter_mut().enumerate() {
            for (rank, local) in pair.iter_mut().enumerate() {
                let binding = keys.iter().copied().filter(|key| key.lane == lane && key.rank == rank).collect::<Vec<_>>();
                captured += local.warm_graphs(&binding, rank, reserves[rank])?;
            }
        }
        drop(lanes);
        // Startup activity is attributed to Other and charged once, here,
        // before the first serving step can see the arenas as its own.
        self.graph_settle()?;
        Ok(captured)
    }

    /// Completed-step graph boundary. Every rank's step is queued and no lane is
    /// mid-segment, so each physical device is sampled once and growth charged.
    fn graph_settle(&self) -> Result<()> {
        let mut lanes = self.lanes.try_borrow_mut()?;
        let boundary = lanes.local.iter().flatten().any(|lane| lane.graph_needs_boundary());
        if !boundary {
            for lane in lanes.local.iter_mut().flatten() { lane.graph_tick(); }
            return Ok(());
        }
        let library = self.library();
        // SAFETY: Kimi owners pin `()`; a capture references only the lane's
        // persistent buffers and admitted weights, never a per-graph allocation,
        // and those outlive every queued launch.
        unsafe { crate::shared::decode_graph::device::observe_completed(library, |visit| {
            for lane in lanes.local.iter_mut().flatten() {
                let spans = lane.graph_spans_devices();
                let Some(device) = lane.graph_device() else { continue };
                let Some(owner) = lane.graph_owner() else { continue };
                visit(device, "kimi_k3", spans, owner)?;
            }
            Ok(())
        }) }
    }

    /// Scheduler submission keeps full logits on the terminal device. Page
    /// metadata validates against the cache's real refcounts after its COW.
    pub(crate) fn step_device(&self,
        sequences: &mut [(&mut super::prefix::Placement, &[u32])],
        pool: &cuteafd_engine::prefix::RefPagePool, logit_rows: usize, graphed: bool)
        -> Result<crate::shared::token_io::DeviceLogits> {
        self.step_device_mode(sequences, pool, logit_rows, graphed, false)
    }

    pub(crate) fn verify_device(&self, sequences: &mut [(&mut super::prefix::Placement, &[u32])],
        pool: &cuteafd_engine::prefix::RefPagePool, graphed: bool)
        -> Result<crate::shared::token_io::DeviceLogits> {
        let rows = sequences.iter().try_fold(0usize, |rows, (_, tokens)|
            rows.checked_add(tokens.len()).ok_or_else(|| anyhow::anyhow!("Kimi verify rows overflow")))?;
        self.step_device_mode(sequences, pool, rows, graphed, true)
    }

    fn step_device_mode(&self, sequences: &mut [(&mut super::prefix::Placement, &[u32])],
        pool: &cuteafd_engine::prefix::RefPagePool, logit_rows: usize, graphed: bool, speculative: bool)
        -> Result<crate::shared::token_io::DeviceLogits> {
        self.require_live()?;
        ensure!(self.provisional.borrow().is_none(), "Kimi pending verify must be accepted or discarded before reuse");
        let family = self.prefix_family()?;
        let provisional = if speculative { Some(super::state::Provisional::new(sequences, self.geometry()?.slots)?) }
            else { None };
        let rows = sequences.iter().try_fold(0usize, |rows, (_, tokens)|
            rows.checked_add(tokens.len()).ok_or_else(|| anyhow::anyhow!("Kimi submission rows overflow")))?;
        let geometry = self.geometry()?;
        let logit_capacity = if geometry.full_prefill_logits { geometry.prefill_rows } else { geometry.decode_rows };
        ensure!(!sequences.is_empty() && logit_rows > 0 && logit_rows <= rows && logit_rows <= logit_capacity,
            "Kimi device submission logit extent");
        // Validate every frontier before enqueue: a failed batch cannot publish
        // only some requests' state. Non-speculative execution commits all rows.
        for (placement, tokens) in sequences.iter() {
            family.check_append(placement, tokens.len())?;
        }
        if let Some(draft) = &self.draft { draft.prepare_target()?; }
        let output = submit_scoped(|| self.require_live(), || self.shutdown(), |scope| {
            let mut lanes = self.lanes.try_borrow_mut()?;
            lanes.local[0][self.head()].config().release_geometry()?;
            let cycle = if rows > 64 { crate::shared::decode_graph::device::Cycle::Prefill }
                else if speculative { crate::shared::decode_graph::device::Cycle::Verify }
                else { crate::shared::decode_graph::device::Cycle::Decode };
            for local in &mut lanes.local[0] { local.graph_cycle(cycle); }
            let metadata = sequences.iter().map(|(placement, tokens)| SequenceRows {
                slot: placement.slot, start: placement.len, rows: tokens.len(), pages: &placement.units,
            }).collect::<Vec<_>>();
            // Unit calls stage all metadata banks before the pipeline; a single
            // pass stages its whole step (at most one unit) here.
            let unit_call = !speculative && sequences.len() == 1 && rows > 64;
            if !unit_call { for local in &mut lanes.local[0] { local.stage_attention(&metadata, pool)?; } }
            let tokens = sequences.iter().flat_map(|(_, tokens)| tokens.iter().copied()).collect::<Vec<_>>();
            lanes.io[0][0].embed(&tokens, lanes.local[0][0].buffer("delta")?)?;
            let Lanes { local, routed, io, experts } = &mut *lanes;
            for binding in &mut local[0] { binding.enable_replay(graphed); }
            let result = if unit_call {
                let units = self.prefill_units(rows)?;
                self.execution.run_units(scope, self.peer.as_deref(), self.hops.as_ref(), &mut local[0],
                    &mut routed[0], &mut experts[0], &units, self.depth.waves, &metadata[0], pool,
                    self.draft.as_deref()).map(|output| vec![output])
            } else {
                ensure!(rows <= self.prefill.unit_rows, "Kimi {rows}-row multi-sequence step exceeds one prefill unit");
                self.execution.run(scope, self.peer.as_deref(), self.hops.as_ref(),
                    &mut local[..1], &mut routed[..1], &mut experts[..1], &[rows], speculative,
                    super::layers::Output::Normalized, self.draft.as_deref(), Some(&self.draft_probe))
            };
            for binding in &mut local[0] { binding.enable_replay(true); }
            let normalized = result?[0];
            let input = crate::shared::prefix::view(normalized,
                (rows - logit_rows) * 7168 * 2, logit_rows * 7168 * 2)?;
            let logits = io[0][self.head()].project(input, logit_rows, &local[0][self.head()])?;
            let stream = local[0][self.head()].stream();
            Ok(crate::shared::token_io::DeviceLogits { ptr: logits.ptr.cast_const(), rows: logit_rows,
                vocab: local[0][self.head()].config().vocab_size, stride: local[0][self.head()].config().vocab_size,
                stream: stream.raw, greedy: None })
        })?;
        if let Err(error) = self.graph_settle() { self.shutdown()?; return Err(error); }
        // A nested native operator capturing here is a contract violation too.
        // Abort/drain before propagating it, never release live graph operands.
        if let Err(error) = self.require_live() { self.shutdown()?; return Err(error); }
        if speculative {
            *self.provisional.borrow_mut() = provisional;
        } else {
            for (placement, tokens) in sequences {
                family.commit(placement, placement.len, tokens.len())?;
            }
        }
        Ok(output)
    }

    pub(crate) fn accept_provisional<D, H>(&self,
        verified: &mut [crate::shared::qwen::scheduler::Verified<'_, super::prefix::Placement, D, H>]) -> Result<()> {
        self.require_live()?;
        let result = (|| {
            let pending = self.provisional.borrow();
            let pending = pending.as_ref().ok_or_else(|| anyhow::anyhow!("Kimi accept has no provisional verify"))?;
            let accepted = verified.iter().map(|sequence| super::state::Accepted {
                placement: sequence.placement, start: sequence.start, first_row: sequence.first_row,
                rows: sequence.rows, kept: sequence.kept.map(|(rows, _)| rows),
            }).collect::<Vec<_>>();
            let tables = pending.tables(&accepted)?;
            let family = self.prefix_family()?;
            for sequence in verified.iter() {
                family.check_provisional(sequence.placement, sequence.start, sequence.rows.len())?;
            }
            submit_scoped(|| self.require_live(), || self.shutdown(), |_| {
                let lanes = self.lanes.try_borrow()?;
                self.execution.commit_kda(&lanes.local[0], &tables)
            })?;
            // Host publication follows the all-rank enqueue; future attention
            // joins the exact accepted-state events, without a compute drain.
            for sequence in verified {
                if let Some((keep, _)) = sequence.kept {
                    family.commit(sequence.placement, sequence.start, keep)?;
                } else {
                    use cuteafd_engine::prefix::PrefixFamily;
                    family.discard(sequence.placement).map_err(|error| anyhow::anyhow!("{error}"))?;
                }
            }
            Ok(())
        })();
        if result.is_err() { self.shutdown()?; }
        self.provisional.borrow_mut().take();
        result
    }

    pub(crate) fn cancel_provisional(&self, placement: &super::prefix::Placement) -> Result<()> {
        let mut pending = self.provisional.try_borrow_mut()?;
        if let Some(replay) = pending.as_mut() {
            replay.cancel(placement)?;
            if replay.sequences.is_empty() { pending.take(); }
        }
        Ok(())
    }

    fn shutdown(&self) -> Result<()> {
        if self.terminal.get() == TerminalState::Drained { return Ok(()); }
        let lanes = self.lanes.try_borrow_mut()?;
        let lanes = RefCell::new(lanes);
        if let Some(draft) = &self.draft { draft.abort()?; }
        let result = teardown(&mut Teardown {
            abort: || {
                self.terminal.set(TerminalState::Aborting);
                self.peer.as_deref().map_or(Ok(()), PeerExchange::publish_abort)
            },
            ranks: || {
                if let Some(peer) = &self.peer { peer.drain_compute()?; }
                for stream in &self.exchange_streams {
                    stream.device.run(|| {
                        // SAFETY: exchange storage and streams remain owned through this terminal drain.
                        unsafe { stream.device.library.cuda_stream_synchronize(stream.raw) }
                    })?;
                }
                for pair in &lanes.borrow().local {
                    for local in pair {
                        let stream = local.stream();
                        stream.device.run(|| {
                            // SAFETY: peer aborts are visible; every lane operand
                            // and recorded event remains owned through drainage.
                            unsafe { stream.device.library.cuda_stream_synchronize(stream.raw) }
                        })?;
                    }
                }
                Ok(())
            },
            links: || {
                // #511: checked ordinary-lane QP quiescence; failure retains
                // intake and library owners without entering copy/release phases.
                for routed in &mut lanes.borrow_mut().routed { routed.quiesce_links()?; }
                Ok(())
            },
            copies: || {
                for routed in &lanes.borrow().routed { routed.drain_copies()?; }
                if let Some(draft) = &self.draft { draft.drain()?; }
                Ok(())
            },
            graphs: || {
                for routed in &mut lanes.borrow_mut().routed { routed.release_payloads()?; }
                Ok(())
            },
            finish: |drained| {
                self.terminal.set(if drained { TerminalState::Drained } else { TerminalState::Retained });
                if let Some(peer) = &self.peer { peer.finish_terminal(drained); }
                if let Some(draft) = &self.draft { draft.finish_terminal(drained); }
            },
        });
        crate::shared::decode_graph::fatal_drain(result, "Kimi submission shutdown");
        Ok(())
    }

    /// Cold diagnostic prefix: the admitted inventory is already truncated.
    /// Never applies output mixing, normalization or vocabulary projection.
    pub(crate) fn prefix_residual(&self, tokens: &[u32], pages: &[u32],
        pool: &cuteafd_engine::prefix::RefPagePool) -> Result<Vec<f32>> {
        ensure!(self.provisional.borrow().is_none(), "Kimi diagnostic cannot overwrite pending verify");
        submit_scoped(|| self.require_live(), || self.shutdown(), |scope| {
            let mut lanes = self.lanes.try_borrow_mut()?;
            ensure!(!tokens.is_empty() && tokens.len() <= self.prefill.unit_rows,
                "Kimi diagnostic prefix has no tokens or exceeds one prefill unit");
            let sequences = [SequenceRows { slot: 0, start: 0, rows: tokens.len(), pages }];
            for local in &mut lanes.local[0] {
                local.graph_cycle(crate::shared::decode_graph::device::Cycle::Other);
                local.stage_attention(&sequences, pool)?;
            }
            lanes.io[0][0].embed(tokens, lanes.local[0][0].buffer("delta")?)?;
            let Lanes { local, routed, experts, .. } = &mut *lanes;
            let outputs = self.execution.run(scope, self.peer.as_deref(), self.hops.as_ref(),
                &mut local[..1], &mut routed[..1], &mut experts[..1], &[tokens.len()], false,
                super::layers::Output::Residual, None, None)?;
            let stream = local[0][self.head()].stream();
            stream.device.run(|| {
                // SAFETY: this uncaptured diagnostic retains every operand until
                // the terminal lane has completed, then downloads its BF16 rows.
                unsafe { stream.device.library.cuda_stream_synchronize(stream.raw)?; }
                let mut bytes = vec![0; outputs[0].bytes];
                stream.device.library.copy_d2h(&mut bytes, outputs[0])?;
                Ok(bytes.chunks_exact(2).map(|bytes| f32::from_bits(
                    u32::from(u16::from_le_bytes(bytes.try_into().unwrap())) << 16)).collect())
            })
        })
    }

    /// Sequential cold diagnostics reuse a single recurrent slot. Each rank's
    /// reset follows its own completed lane work; no binding/storage teardown.
    pub(crate) fn reset_cold_state(&self) -> Result<()> {
        ensure!(self.provisional.borrow().is_none(), "Kimi reset cannot overwrite pending verify");
        let lanes = self.lanes.try_borrow()?;
        ensure!(lanes.local.len() == 1, "Kimi cold diagnostic reset needs one lane");
        for local in &lanes.local[0] {
            local.stream().device.run(|| {
                // SAFETY: retained lane stream and operands outlive synchronization.
                unsafe { local.stream().device.library.cuda_stream_synchronize(local.stream().raw) }
            })?;
            local.reset_cold_state()?;
        }
        Ok(())
    }

    /// Diagnostic embedding-to-logits path. No graph/replay or speculative
    /// transaction qualification is implied by this uncaptured forward.
    pub(crate) fn forward(&self, tokens: &[Vec<u32>], sequences: &[Vec<SequenceRows<'_>>],
        pool: &cuteafd_engine::prefix::RefPagePool,
        mut consume: impl FnMut(usize, usize, usize, &[f32]) -> Result<()>) -> Result<()> {
        // A truncated inventory has no meaningful model output mix or logits.
        self.lanes.try_borrow()?.local[0][self.head()].config().release_geometry()?;
        ensure!(self.provisional.borrow().is_none(), "Kimi diagnostic cannot overwrite pending verify");
        submit_scoped(|| self.require_live(), || self.shutdown(), |scope| {
            let mut lanes = self.lanes.try_borrow_mut()?;
            ensure!(!tokens.is_empty() && tokens.len() == sequences.len() && tokens.len() <= lanes.local.len(),
                "Kimi forward token/sequence lane count");
            let rows = tokens.iter().map(Vec::len).collect::<Vec<_>>();
            let unit_call = tokens.len() == 1 && sequences[0].len() == 1 && rows[0] > 64;
            for (lane, ids) in tokens.iter().enumerate() {
                ensure!(!ids.is_empty(), "Kimi forward has an empty lane");
                for local in &mut lanes.local[lane] {
                    local.graph_cycle(crate::shared::decode_graph::device::Cycle::Other);
                    ensure!(sequences[lane].iter().map(|sequence| sequence.rows).sum::<usize>() == ids.len(),
                        "Kimi forward sequence rows differ from tokens");
                    // Unit calls stage all metadata banks before the pipeline.
                    if !unit_call { local.stage_attention(&sequences[lane], pool)?; }
                }
                lanes.io[lane][0].embed(ids, lanes.local[lane][0].buffer("delta")?)?;
            }
            let Lanes { local, routed, io, experts } = &mut *lanes;
            let normalized = if unit_call {
                let units = self.prefill_units(rows[0])?;
                vec![self.execution.run_units(scope, self.peer.as_deref(), self.hops.as_ref(), &mut local[0],
                    &mut routed[0], &mut experts[0], &units, self.depth.waves, &sequences[0][0], pool, None)?]
            } else {
                ensure!(rows.iter().sum::<usize>() <= self.prefill.unit_rows,
                    "Kimi multi-sequence diagnostic step exceeds one prefill unit");
                self.execution.run(scope, self.peer.as_deref(), self.hops.as_ref(),
                    &mut local[..tokens.len()], &mut routed[..tokens.len()], &mut experts[..tokens.len()], &rows, false,
                    super::layers::Output::Normalized, None, None)?
            };
            for (lane, normalized) in normalized.into_iter().enumerate() {
                let stream = local[lane][self.head()].stream();
                for first in (0..rows[lane]).step_by(64) {
                    let count = (rows[lane] - first).min(64);
                    let input = crate::shared::prefix::view(normalized, first * 7168 * 2, count * 7168 * 2)?;
                    let logits = io[lane][self.head()].project(input, count, &local[lane][self.head()])?;
                    let values = stream.device.run(|| {
                        // SAFETY: diagnostic download is outside capture; the
                        // exact lane's head finishes before its logits are reused.
                        unsafe { stream.device.library.cuda_stream_synchronize(stream.raw)?; }
                        let mut bytes = vec![0; logits.bytes];
                        stream.device.library.copy_d2h(&mut bytes, logits)?;
                        Ok(bytes.chunks_exact(4).map(|bytes|
                            f32::from_le_bytes(bytes.try_into().unwrap())).collect::<Vec<_>>())
                    })?;
                    consume(lane, first, count, &values)?;
                }
            }
            Ok(())
        })
    }
}
