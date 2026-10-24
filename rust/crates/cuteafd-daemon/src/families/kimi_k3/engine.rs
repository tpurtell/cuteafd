//! Retained coordinator submission owner. Startup supplies admitted operands and
//! ordinary Spark links; this uses the shared executor and terminal sequence.
use super::{device::{LocalLane, SequenceRows, TokenIo}, layers::Execution, spark::RoutedLane};
use crate::shared::{experts::rtx::pair::{submit_scoped, teardown, Teardown},
    peer_split::{hop::HopLink, PeerExchange, TerminalState}};
use anyhow::{ensure, Result};
use cuteafd_loader::placement::LayerExecutionPlan;
use std::{cell::{Cell, RefCell}, rc::Rc};
use crate::shared::memory::device::Stream;

struct Lanes<'a> {
    local: Vec<[LocalLane<'a>; 2]>,
    routed: Vec<RoutedLane<'a>>,
    io: Vec<[TokenIo<'a>; 2]>,
    experts: Vec<[Option<crate::shared::experts::rtx::exl3::Exl3Tp2<'a>>; 2]>,
}

pub(crate) struct Engine<'a> {
    draft: Option<Rc<super::draft::NativeDraft<'a>>>,
    draft_probe: RefCell<Option<crate::shared::memory::device::DeviceOwner<'a,
        crate::shared::draft::probe::RoundProbe<'a>>>>,
    execution: Execution<'a>,
    lanes: RefCell<Lanes<'a>>,
    prefix: RefCell<Option<super::prefix::KimiPrefix<'a>>>,
    startup_captures: Cell<Option<u64>>,
    provisional: RefCell<Option<super::state::Provisional>>,
    hops: HopLink<'a>,
    peer: PeerExchange<'a>,
    // PeerExchange and HopLink borrow raw handles; keep their exact streams
    // after both links and their terminal abort storage have been released.
    exchange_streams: [Rc<Stream<'a>>; 2],
    /// Prefill units: their rows and how many are in flight on the lane.
    prefill: PrefillUnits,
}

/// The lane holds `waves` units of `unit_rows`; a step of `rows` is cut into
/// equal units of at most `unit_rows` (`prefill_units`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PrefillUnits { pub unit_rows: usize, pub waves: usize }

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

fn check_expert_owners(layers: &[cuteafd_loader::placement::LayerAssignment], owners: [&[usize]; 2]) -> Result<()> {
    for rank in 0..2 {
        let expected = layers.iter().enumerate().filter_map(|(layer, assignment)|
            (assignment.experts == cuteafd_loader::placement::ExpertHome::RtxTp2).then_some(layer))
            .collect::<Vec<_>>();
        ensure!(owners[rank] == expected, "Kimi TP2 expert owners differ from accepted per-layer placement");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
    pub(crate) fn load_admitted(library: &'a std::sync::Arc<cuteafd_ffi::NativeLibrary>,
        exchange_streams: [Rc<Stream<'a>>; 2],
        lane_streams: [Rc<Stream<'a>>; 2],
        mut draft: Option<super::draft::NativeDraft<'a>>,
        mut admission: super::admission::Admission<'a>,
        checkpoint: &cuteafd_loader::plan::Checkpoint,
        programs: [std::rc::Rc<super::device::NativePrograms<'a>>; 2],
        peer: PeerExchange<'a>, hops: HopLink<'a>, peers: Vec<std::net::SocketAddr>,
        config: cuteafd_transport::TcpTransportConfig, compiled_rows: u32,
        mut expert_modules: [Option<Vec<crate::shared::experts::exl3::execution::Exl3Modules<'a>>>; 2]) -> Result<Self> {
        use crate::shared::placement::{spark_rings, SparkAdmission};
        use cuteafd_core::{ExpertGeometry, SparkEndpoint, SparkIngress};
        use cuteafd_transport::{expert::SparkWorld, worker_selection::ModelAdmission};
        ensure!(cuteafd_core::expert_geometry() == ExpertGeometry::KIMI_K3 && peers.len() == 6,
            "Kimi startup needs process latent geometry and six real expert ranks");
        let plan = admission.execution_plan()?;
        hops.validate_binding(&plan, &peer)?;
        check_exchange_streams(&peer, &exchange_streams)?;
        let devices = programs.each_ref().map(|programs| programs.device);
        ensure!((0..2).all(|rank| devices[rank].id == peer.device(rank))
            && std::ptr::eq(devices[0].library, devices[1].library)
            && std::ptr::eq(devices[0].library, library.as_ref()),
            "Kimi startup programs differ from exchange devices/library");
        let rows = u32::try_from(plan.spec().rows)?;
        let endpoints = vec![SparkEndpoint { ring_depth: None, capacity: rows, ingress: SparkIngress::Fp8K32 };
            plan.spec().lanes as usize];
        let model = ModelAdmission { capacity: rows, compiled_capacity: compiled_rows,
            endpoints, rings: spark_rings(config.max_frame_bytes)?,
            // Each routed lane's wave is in flight to a rank at once.
            execution_slots: cuteafd_loader::plan::layout::spark_execution_slots("kimi_k3", plan.spec().lanes as u64, 1) };
        model.validate(ExpertGeometry::KIMI_K3)?;
        let catalog = cuteafd_loader::read_expert_catalog(&checkpoint.snapshot)?;
        let spark = SparkAdmission::plan(&checkpoint.snapshot, ExpertGeometry::KIMI_K3,
            Some(&catalog), &admission.working_set(), 6, None, model)?;
        // Dense operands and TP2 halves land inside the Spark load (pushed as
        // coordinator ranks when the group pushes, else through the mount).
        let (pending, expert_weights, weights) = admission.start_and_load(spark, &peers, devices, checkpoint, &catalog,
            [exchange_streams[0].raw, exchange_streams[1].raw])?;
        let lanes = usize::try_from(plan.spec().lanes)?;
        let mut allocated = (0..lanes).map(|lane| Ok([
            admission.allocate_lane_on(devices[0], 0, lane, lane_streams[0].clone())?,
            admission.allocate_lane_on(devices[1], 1, lane, lane_streams[1].clone())?,
        ])).collect::<Result<Vec<_>>>()?;
        let kda = [admission.allocate_kda(devices[0], 0)?, admission.allocate_kda(devices[1], 1)?];
        let mla = [admission.allocate_mla(devices[0], 0)?, admission.allocate_mla(devices[1], 1)?];
        let mut experts = Vec::with_capacity(lanes);
        let mut local = Vec::with_capacity(lanes);
        let mut io = Vec::with_capacity(lanes);
        for (lane, [zero, one]) in allocated.drain(..).enumerate() {
            experts.push([expert_weights[0].as_ref().map(|owner| owner.execution_with_modules(expert_modules[0].take())).transpose()?,
                expert_weights[1].as_ref().map(|owner| owner.execution_with_modules(expert_modules[1].take())).transpose()?]);
            local.push([admission.bind_local(0, lane, zero, programs[0].clone())?,
                admission.bind_local(1, lane, one, programs[1].clone())?]);
            io.push([admission.allocate_io(devices[0], 0, lane)?,
                admission.allocate_io(devices[1], 1, lane)?]);
        }
        for pair in &experts {
            use crate::shared::experts::rtx::RtxExpertLayer;
            check_expert_owners(&admission.placement.layers,
                std::array::from_fn(|rank| pair[rank].as_ref().map_or(&[][..], |e| e.layers())))?;
        }
        // Admission charged this arena before allocation; retain its exact device owner.
        let draft_probe = draft.as_ref().map(|_| devices[0].own(||
            crate::shared::draft::probe::RoundProbe::new(devices[0].library,
                local[0][0].config().dense.clone(), local[0][0].config().topk, 64))).transpose()?;
        // Weak admission registries must not outlive these strong startup owners
        // until every persistent local binding has acquired the same operands.
        if let Some(draft) = draft.as_mut() { draft.bind_target(weights.global.clone(), programs[1].clone())?; }
        drop((weights, kda, mla));
        pending.wait()?;
        let world = SparkWorld::canonical(6)?;
        let routed = local.iter().map(|pair| admission.connect_spark(library, &pair[1],
            world, peers.clone(), config.clone())).collect::<Result<Vec<_>>>()?;
        let prefill = PrefillUnits { unit_rows: admission.unit_rows(), waves: admission.waves() };
        let mut engine = Self::bind(exchange_streams, plan, peer, hops, local, routed, io, experts, prefill)?;
        engine.draft = draft.map(Rc::new);
        *engine.draft_probe.borrow_mut() = draft_probe;
        Ok(engine)
    }

    /// Bind only exact owners already allocated through Admission. No device
    /// allocations or program loading occur after this constructor returns.
    pub(crate) fn bind(exchange_streams: [Rc<Stream<'a>>; 2],
        plan: LayerExecutionPlan, peer: PeerExchange<'a>, hops: HopLink<'a>,
        local: Vec<[LocalLane<'a>; 2]>, routed: Vec<RoutedLane<'a>>,
        io: Vec<[TokenIo<'a>; 2]>,
        experts: Vec<[Option<crate::shared::experts::rtx::exl3::Exl3Tp2<'a>>; 2]>,
        prefill: PrefillUnits) -> Result<Self> {
        ensure!(prefill.waves > 0 && local.iter().all(|pair| pair[0].rows() >= prefill.unit_rows * prefill.waves
            || prefill.waves == 1), "Kimi lane rows hold fewer than the admitted prefill units in flight");
        ensure!(!local.is_empty() && local.len() == routed.len() && local.len() == io.len() && local.len() == experts.len(),
            "Kimi engine local, Spark and IO lane counts differ");
        check_exchange_streams(&peer, &exchange_streams)?;
        for (lane, pair) in local.iter().enumerate() {
            for rank in 0..2 {
                io[lane][rank].check_lane(&pair[rank], rank)?;
                ensure!(std::ptr::eq(pair[rank].stream().device.library,
                    exchange_streams[rank].device.library), "Kimi local and exchange native libraries differ");
            }
        }
        let streams = local.iter().map(|pair| pair.each_ref().map(|local| local.stream().clone())).collect();
        let execution = Execution::new(plan, &peer, &hops, exchange_streams.clone(), streams)?;
        Ok(Self { draft: None, draft_probe: RefCell::new(None), execution, lanes: RefCell::new(Lanes { local, routed, io, experts }),
            prefix: RefCell::new(None), startup_captures: Cell::new(None), provisional: RefCell::new(None),
            hops, peer, exchange_streams, prefill })
    }

    pub(crate) fn prefix_family(&self) -> Result<super::prefix::KimiPrefix<'a>> {
        let mut prefix = self.prefix.try_borrow_mut()?;
        if let Some(prefix) = prefix.as_ref() { return Ok(prefix.clone()); }
        let lanes = self.lanes.try_borrow()?;
        let [(kda0, mla0), (kda1, mla1)] = lanes.local[0].each_ref().map(|lane| lane.prefix_banks());
        let family = super::prefix::KimiPrefix::new(lanes.local[0][0].config(), lanes.local[0][0].max_context(),
            [kda0, kda1], [mla0, mla1])?.with_draft(self.draft.clone())?;
        *prefix = Some(family.clone());
        Ok(family)
    }

    fn prefill_units(&self, rows: usize) -> Result<Vec<crate::shared::prefill_pipeline::Unit>> {
        let layers = self.execution.plan.modes().len();
        let units = crate::shared::prefill_pipeline::prefill_units(rows, self.prefill.unit_rows, self.prefill.waves, 256)?;
        // Every unit's rows lie in the lane; a slot is the unit's row range.
        Ok(units.into_iter().enumerate().map(|(index, rows)| crate::shared::prefill_pipeline::Unit {
            index, sequence: 0, rows, layers: 0..layers, slot: index % self.prefill.waves, barrier: false }).collect())
    }

    pub(crate) fn geometry(&self) -> Result<crate::shared::qwen::scheduler::Geometry> {
        let lanes = self.lanes.try_borrow()?;
        let local = &lanes.local[0][0];
        let (state, _) = local.prefix_banks();
        Ok(crate::shared::qwen::scheduler::Geometry { family: "kimi_k3",
            vocab_size: local.config().vocab_size, max_context: local.max_context(),
            slots: state.slots(), prefill_rows: local.rows(), decode_rows: local.rows().min(64),
            full_prefill_logits: false })
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
        let catalog = cuteafd_loader::read_expert_catalog(snapshot)?;
        let manifest = catalog.exl3().ok_or_else(|| anyhow::anyhow!("Kimi draft needs EXL3 headers"))?;
        let lanes = self.lanes.try_borrow()?;
        let cfg = lanes.local[0][0].config();
        let reports = lanes.routed[0].installed_grouping()?;
        let local = &lanes.experts[0];
        let homes = cfg.dense.iter().enumerate().map(|(layer, dense)| {
            if *dense { return Ok(LayerHome::Dense); }
            let resident = local[0].as_ref().is_some_and(|rank| rank.layers().contains(&layer));
            ensure!(resident == local[1].as_ref().is_some_and(|rank| rank.layers().contains(&layer)),
                "Kimi TP2 draft policy layer homes disagree");
            let bytes = resource::exl3_slice_bytes(manifest, layer, if resident { 2 } else { 6 })?;
            Ok(if resident { LayerHome::Local { slice_bytes: bytes } }
                else { LayerHome::Remote { slice_bytes: bytes } })
        }).collect::<Result<Vec<_>>>()?;
        resource::geometry_with_grouping(&homes, cfg.experts, cfg.topk, 64, width, 0, |layer, rows| {
            if matches!(homes[layer], LayerHome::Local { .. }) {
                let group = local[0].as_ref().unwrap().group_rows(rows)?;
                ensure!(local[1].as_ref().unwrap().group_rows(rows)? == group,
                    "Kimi TP2 draft policy grouping disagrees");
                Ok(group)
            } else { resource::worker_group_rows_for_input(&reports, layer, rows, false) }
        })
    }

    pub(crate) fn library(&self) -> &'a cuteafd_ffi::NativeLibrary {
        self.exchange_streams[0].device.library
    }

    pub(crate) fn devices(&self) -> [i32; 2] {
        self.exchange_streams.each_ref().map(|stream| stream.device.id)
    }

    pub(crate) fn require_live(&self) -> Result<()> {
        self.peer.require_live()?;
        if let Some(draft) = &self.draft { draft.require_live()?; }
        if let Some(startup) = self.startup_captures.get() {
            super::graphs::check_captures(startup, cuteafd_ffi::graph_captures())?;
        }
        Ok(())
    }

    pub(crate) fn graph_replays(&self) -> Result<u64> {
        Ok(self.lanes.try_borrow()?.local.iter().flatten().map(LocalLane::graph_replays).sum())
    }

    pub(crate) fn set_grouped_kda(&self, enabled: bool) -> Result<()> {
        for local in self.lanes.try_borrow()?.local.iter().flatten() { local.set_grouped_kda(enabled)?; }
        Ok(())
    }

    pub(crate) fn warm_graphs(&self, keys: &[super::graphs::Key]) -> Result<()> {
        ensure!(self.startup_captures.get().is_none(), "Kimi startup graphs already sealed");
        let mut lanes = self.lanes.try_borrow_mut()?;
        let mut consumed = 0;
        for (lane, pair) in lanes.local.iter_mut().enumerate() {
            for (rank, local) in pair.iter_mut().enumerate() {
                let binding = keys.iter().copied().filter(|key| key.lane == lane && key.rank == rank).collect::<Vec<_>>();
                ensure!(!binding.is_empty(), "Kimi graph inventory omits a binding");
                local.warm_graphs(&binding, rank)?;
                consumed += binding.len();
            }
        }
        ensure!(consumed == keys.len(), "Kimi startup graph inventory has unbound keys");
        self.startup_captures.set(Some(cuteafd_ffi::graph_captures()));
        Ok(())
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
        ensure!(!sequences.is_empty() && logit_rows > 0 && logit_rows <= rows && logit_rows <= 64,
            "Kimi device submission logit extent");
        // Validate every frontier before enqueue: a failed batch cannot publish
        // only some requests' state. Non-speculative execution commits all rows.
        for (placement, tokens) in sequences.iter() {
            family.check_append(placement, tokens.len())?;
        }
        if let Some(draft) = &self.draft { draft.prepare_target()?; }
        let output = submit_scoped(|| self.require_live(), || self.shutdown(), |scope| {
            let mut lanes = self.lanes.try_borrow_mut()?;
            lanes.local[0][1].config().release_geometry()?;
            let metadata = sequences.iter().map(|(placement, tokens)| SequenceRows {
                slot: placement.slot, start: placement.len, rows: tokens.len(), pages: &placement.units,
            }).collect::<Vec<_>>();
            // Unit calls stage each unit's metadata inside its hooks; a single
            // pass stages its whole step (at most one unit) here.
            let unit_call = !speculative && sequences.len() == 1 && rows > 64;
            if !unit_call { for local in &mut lanes.local[0] { local.stage_attention(&metadata, pool)?; } }
            let tokens = sequences.iter().flat_map(|(_, tokens)| tokens.iter().copied()).collect::<Vec<_>>();
            lanes.io[0][0].embed(&tokens, lanes.local[0][0].buffer("delta")?)?;
            let Lanes { local, routed, io, experts } = &mut *lanes;
            for binding in &mut local[0] { binding.enable_replay(graphed); }
            let units = self.prefill_units(rows)?;
            let result = if unit_call {
                self.execution.run_units(scope, &self.peer, &self.hops, &mut local[0],
                    &mut routed[0], &mut experts[0], &units, self.prefill.waves, &metadata[0], pool,
                    self.draft.as_deref()).map(|output| vec![output])
            } else {
                ensure!(rows <= self.prefill.unit_rows, "Kimi {rows}-row multi-sequence step exceeds one prefill unit");
                self.execution.run(scope, &self.peer, &self.hops,
                    &mut local[..1], &mut routed[..1], &mut experts[..1], &[rows], speculative,
                    super::layers::Output::Normalized, self.draft.as_deref(), Some(&self.draft_probe))
            };
            for binding in &mut local[0] { binding.enable_replay(true); }
            let normalized = result?[0];
            let input = crate::shared::prefix::view(normalized,
                (rows - logit_rows) * 7168 * 2, logit_rows * 7168 * 2)?;
            let logits = io[0][1].project(input, logit_rows, &local[0][1])?;
            let stream = local[0][1].stream();
            Ok(crate::shared::token_io::DeviceLogits { ptr: logits.ptr.cast_const(), rows: logit_rows,
                vocab: local[0][1].config().vocab_size, stride: local[0][1].config().vocab_size,
                stream: stream.raw, greedy: None })
        })?;
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
        if self.peer.terminal_state() == TerminalState::Drained { return Ok(()); }
        let lanes = self.lanes.try_borrow_mut()?;
        let lanes = RefCell::new(lanes);
        if let Some(draft) = &self.draft { draft.abort()?; }
        let result = teardown(&mut Teardown {
            abort: || self.peer.publish_abort(),
            ranks: || {
                self.peer.drain_compute()?;
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
                self.peer.finish_terminal(drained);
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
        submit_scoped(|| self.peer.require_live(), || self.shutdown(), |scope| {
            let mut lanes = self.lanes.try_borrow_mut()?;
            ensure!(!tokens.is_empty() && tokens.len() <= self.prefill.unit_rows,
                "Kimi diagnostic prefix has no tokens or exceeds one prefill unit");
            let sequences = [SequenceRows { slot: 0, start: 0, rows: tokens.len(), pages }];
            for local in &mut lanes.local[0] { local.stage_attention(&sequences, pool)?; }
            lanes.io[0][0].embed(tokens, lanes.local[0][0].buffer("delta")?)?;
            let Lanes { local, routed, experts, .. } = &mut *lanes;
            let outputs = self.execution.run(scope, &self.peer, &self.hops,
                &mut local[..1], &mut routed[..1], &mut experts[..1], &[tokens.len()], false,
                super::layers::Output::Residual, None, None)?;
            let stream = local[0][1].stream();
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
        self.lanes.try_borrow()?.local[0][1].config().release_geometry()?;
        ensure!(self.provisional.borrow().is_none(), "Kimi diagnostic cannot overwrite pending verify");
        submit_scoped(|| self.peer.require_live(), || self.shutdown(), |scope| {
            let mut lanes = self.lanes.try_borrow_mut()?;
            ensure!(!tokens.is_empty() && tokens.len() == sequences.len() && tokens.len() <= lanes.local.len(),
                "Kimi forward token/sequence lane count");
            let rows = tokens.iter().map(Vec::len).collect::<Vec<_>>();
            let unit_call = tokens.len() == 1 && sequences[0].len() == 1 && rows[0] > 64;
            for (lane, ids) in tokens.iter().enumerate() {
                ensure!(!ids.is_empty(), "Kimi forward has an empty lane");
                for local in &mut lanes.local[lane] {
                    ensure!(sequences[lane].iter().map(|sequence| sequence.rows).sum::<usize>() == ids.len(),
                        "Kimi forward sequence rows differ from tokens");
                    // Unit calls stage each unit's metadata inside its hooks.
                    if !unit_call { local.stage_attention(&sequences[lane], pool)?; }
                }
                lanes.io[lane][0].embed(ids, lanes.local[lane][0].buffer("delta")?)?;
            }
            let Lanes { local, routed, io, experts } = &mut *lanes;
            let normalized = if unit_call {
                let units = self.prefill_units(rows[0])?;
                vec![self.execution.run_units(scope, &self.peer, &self.hops, &mut local[0],
                    &mut routed[0], &mut experts[0], &units, self.prefill.waves, &sequences[0][0], pool, None)?]
            } else {
                ensure!(rows.iter().sum::<usize>() <= self.prefill.unit_rows,
                    "Kimi multi-sequence diagnostic step exceeds one prefill unit");
                self.execution.run(scope, &self.peer, &self.hops,
                    &mut local[..tokens.len()], &mut routed[..tokens.len()], &mut experts[..tokens.len()], &rows, false,
                    super::layers::Output::Normalized, None, None)?
            };
            for (lane, normalized) in normalized.into_iter().enumerate() {
                let stream = local[lane][1].stream();
                for first in (0..rows[lane]).step_by(64) {
                    let count = (rows[lane] - first).min(64);
                    let input = crate::shared::prefix::view(normalized, first * 7168 * 2, count * 7168 * 2)?;
                    let logits = io[lane][1].project(input, count, &local[lane][1])?;
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
