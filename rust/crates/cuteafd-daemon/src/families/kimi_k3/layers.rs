//! Per-lane stream bindings for the shared layer executor. The shared peer and
//! hop protocols own publication order; events join the exact lane's producers
//! and consumers to their admitted exchange streams.
use crate::shared::{
    memory::device::{Event, Stream},
    peer_split::{attention::LayerProtocol, hop::{HopLink, HopTicket}, PeerExchange},
};
use anyhow::{ensure, Context, Result};
use cuteafd_loader::placement::{FfnMode, Hop, HopPoint, LayerExecutionPlan, LayerMode};
use cuteafd_ffi::CuteafdDeviceBuffer;
use crate::shared::peer_split::attention::{AttentionStateOwners, LayerKernels, PipelineKernels};
use super::{block::MixSite, device::LocalLane, spark::RoutedLane};
use crate::shared::experts::rtx::{exl3::Exl3Tp2, ExpertInput, Routes, RtxExpertLayer, RtxShard, combine::TokenOwnerReduce, pair::SubmissionScope};
use std::{cell::{Cell, RefCell}, ffi::c_void, rc::Rc};

struct LaneJoins<'a> {
    streams: Vec<Rc<Stream<'a>>>,
    produced: RefCell<Vec<Event<'a>>>,
    landed: RefCell<Vec<Event<'a>>>,
    reducer: Option<RefCell<TokenOwnerReduce<'a>>>,
    narrow: cuteafd_ffi::F32Bf16Narrow<'a>,
}

#[derive(Clone, Copy)]
pub(crate) enum Output { Normalized, Residual }

/// Own these bindings for the engine lifetime, not in a temporary submission:
/// recorded event nodes, streams and all exchanged operands must survive replay.
/// The enclosing submission owner aborts all links before draining these streams.
type DraftProbe<'a> = RefCell<Option<crate::shared::memory::device::DeviceOwner<'a,
    crate::shared::draft::probe::RoundProbe<'a>>>>;

pub(crate) struct Execution<'a> {
    pub skip_routed_experts: bool,
    pub expert_usage: Option<Rc<cuteafd_ffi::expert_usage::ExpertUsage<'a>>>,
    pub usage_phase: Cell<cuteafd_ffi::expert_usage::UsagePhase>,
    pub plan: LayerExecutionPlan,
    lanes: Vec<LaneJoins<'a>>,
    entry_lane: Cell<Option<usize>>,
    state_ready: Vec<RefCell<Vec<Event<'a>>>>,
    state_pending: Vec<Vec<Cell<bool>>>,
}

impl Drop for Execution<'_> {
    fn drop(&mut self) {
        for lane in &self.lanes {
            for stream in &lane.streams { stream.drain_or_abort("Kimi layer event owners"); }
        }
    }
}

/// A ticket retains the lane whose consumer owes the event join. In particular,
/// landing a lane-1 hop must not silently join lane 0's compute stream.
pub(crate) struct Ticket {
    hop: HopTicket,
    lane: usize,
    to: usize,
}

impl<'a> Execution<'a> {
    pub(crate) fn new(plan: LayerExecutionPlan, peer: Option<&PeerExchange<'a>>, hops: Option<&HopLink<'a>>,
        exchange_streams: Vec<Rc<Stream<'a>>>, streams: Vec<Vec<Rc<Stream<'a>>>>) -> Result<Self> {
        ensure!(streams.len() as u64 == plan.spec().lanes && !streams.is_empty(),
            "Kimi layer streams differ from admitted lanes");
        let devices = exchange_streams.iter().map(|stream| stream.device.id).collect::<Vec<_>>();
        validate_rank_plan(&plan, devices.len())?;
        let handles = streams.iter().map(|ranks| ranks.iter().map(|stream|
            (stream.device.id, stream.raw)).collect::<Vec<_>>()).collect::<Vec<_>>();
        validate_rank_streams(&handles, &devices)?;
        match (peer, hops, devices.len()) {
            (Some(peer), Some(hops), 2) => {
                hops.validate_binding(&plan, peer)?;
                ensure!(exchange_streams.iter().enumerate().all(|(rank, stream)|
                    stream.device.id == peer.device(rank) && stream.raw == peer.stream(rank)),
                    "Kimi execution exchange streams differ from peer owners");
            }
            (None, None, 1) => {}
            _ => anyhow::bail!("Kimi exchange owners differ from physical rank count"),
        }
        ensure!(streams.iter().flatten().chain(&exchange_streams).all(|stream| std::ptr::eq(
            stream.device.library, streams[0][0].device.library)), "Kimi lane native libraries differ");
        let mut state_ready = Vec::new();
        let mut state_pending = Vec::new();
        for _ in plan.modes() {
            state_ready.push(RefCell::new(streams[0].iter().map(|stream| Event::new(stream.device))
                .collect::<Result<Vec<_>>>()?));
            state_pending.push(devices.iter().map(|_| Cell::new(false)).collect());
        }
        let lanes = streams.into_iter().enumerate().map(|(lane, streams)| {
            let events = || streams.iter().map(|stream| Event::new(stream.device)).collect::<Result<Vec<_>>>();
            let produced = RefCell::new(events()?);
            let landed = RefCell::new(events()?);
            let reducer = peer.map(|peer| TokenOwnerReduce::new(peer,
                [exchange_streams[0].clone(), exchange_streams[1].clone()],
                [streams[0].clone(), streams[1].clone()], lane, plan.spec().rows as usize, 3584))
                .transpose()?.map(RefCell::new);
            let narrow = streams[0].device.run(|| streams[0].device.library.f32_bf16_narrow())?;
            Ok(LaneJoins { streams, produced, landed, reducer, narrow })
        }).collect::<Result<_>>()?;
        Ok(Self { skip_routed_experts: false, expert_usage: None, usage_phase: Cell::new(cuteafd_ffi::expert_usage::UsagePhase::Prefill), plan, lanes, entry_lane: Cell::new(None), state_ready, state_pending })
    }

    /// The family has no second scheduler: use the shared layer-major pipeline
    /// for chunks sharing recurrent history, then materialize the selected terminal view.
    pub(crate) fn run<'e>(&'e self, scope: &'e SubmissionScope<'_>,
        peer: Option<&'e PeerExchange<'a>>, hops: Option<&'e HopLink<'a>>,
        locals: &'e mut [Vec<LocalLane<'a>>], routed: &'e mut [RoutedLane<'a>],
        experts: &'e mut [Vec<Option<Exl3Tp2<'a>>>], rows: &[usize], speculative: bool, output: Output, draft: Option<&'e super::draft::NativeDraft<'a>>, probe: Option<&'e DraftProbe<'a>>) -> Result<Vec<CuteafdDeviceBuffer>> {
        ensure!(!rows.is_empty() && rows.len() == locals.len() && rows.len() == routed.len() && rows.len() == experts.len(),
            "Kimi shared pipeline needs matching local, Spark and row lanes");
        let protocol = self.protocol(peer, hops)?;
        let mut bindings = locals.iter_mut().zip(routed).zip(experts).zip(rows).enumerate().map(
            |(lane, (((local, routed), experts), &rows))| Bindings::new(local, routed, experts, self, peer, hops, scope, lane, rows, speculative)
        ).collect::<Result<Vec<_>>>()?;
        for binding in &mut bindings { binding.draft = draft; binding.probe = probe; }
        crate::shared::peer_split::attention::layer_pipeline_with(scope, &self.plan, &protocol,
            crate::shared::peer_split::attention::LaneOrder::OrderedChunks, rows, &mut bindings)?;
        bindings.iter_mut().map(|binding| binding.output(output)).collect()
    }

    pub(crate) fn run_packed<'e>(&'e self, scope: &'e SubmissionScope<'_>,
        peer: Option<&'e PeerExchange<'a>>, hops: Option<&'e HopLink<'a>>,
        local: &'e mut [LocalLane<'a>], routed: &'e mut RoutedLane<'a>,
        experts: &'e mut [Option<Exl3Tp2<'a>>],
        sequences: &[super::device::SequenceRows<'_>], pool: &cuteafd_engine::prefix::RefPagePool,
        draft: Option<&'e super::draft::NativeDraft<'a>>) -> Result<CuteafdDeviceBuffer> {
        let rows = sequences.iter().map(|sequence| sequence.rows).sum();
        let protocol = self.protocol(peer, hops)?;
        let mut binding = Bindings::new(local, routed, experts, self, peer, hops, scope, 0, rows, false)?;
        binding.draft = draft;
        // Validate all slots/pages before any member can update recurrent state.
        binding.packed = Some(binding.local[0].packed_attention(sequences, pool)?);
        // Selecting a row view also disables replay: captured singleton attention
        // cannot stand in for a pack's member-specific state and positions.
        for local in binding.local.iter_mut() { local.select_subwave(Some(0..rows))?; }
        crate::shared::prefill_pipeline::with_cleanup(&mut binding, |binding| {
            crate::shared::peer_split::attention::layer_pipeline_with(scope, &self.plan, &protocol,
                crate::shared::peer_split::attention::LaneOrder::OrderedChunks, &[rows],
                std::slice::from_mut(binding))?;
            binding.output(Output::Normalized)
        }, |binding| {
            for local in binding.local.iter_mut() { local.clear_subwave(); }
            Ok(())
        })
    }

    /// One sequence's prefill as row units on the single admitted lane and
    /// Spark link (PLAN "Prefill contract"): up to `waves` units' expert waves
    /// in flight, each unit owning its rows of every lane operand.
    pub(crate) fn run_units(&self, scope: &SubmissionScope<'_>, peer: Option<&PeerExchange<'a>>,
        hops: Option<&HopLink<'a>>, local: &mut [LocalLane<'a>], routed: &mut RoutedLane<'a>,
        experts: &mut [Option<Exl3Tp2<'a>>], units: &[crate::shared::prefill_pipeline::Unit], waves: usize,
        sequence: &super::device::SequenceRows<'_>, pool: &cuteafd_engine::prefix::RefPagePool,
        draft: Option<&super::draft::NativeDraft<'a>>) -> Result<CuteafdDeviceBuffer> {
        let protocol = self.protocol(peer, hops)?;
        ensure!(!units.is_empty(), "Kimi prefill has no units");
        for lane in local.iter_mut() { lane.stage_attention_units(units, sequence, pool)?; }
        tracing::info!(start = sequence.start, rows = sequence.rows, units = units.len(), waves,
            sizes = ?units.iter().map(|unit| unit.rows.len()).collect::<Vec<_>>(),
            "Kimi prefill units");
        // Spark waves build, post and land on the routed link's exchange
        // worker while this thread queues GPU work.
        let mut binding = Bindings::new(local, routed, experts, self, peer, hops, scope, 0,
            units[0].rows.len(), false)?;
        binding.draft = draft;
        binding.defer_publication = true;
        let mut hooks = UnitHooks { binding, protocol: &protocol,
            state: units.iter().map(|_| UnitState::default()).collect(), consumed:
                vec![vec![0; self.plan.hops().len()]; units.len()] };
        crate::shared::prefill_pipeline::with_cleanup(&mut hooks, |hooks| {
            crate::shared::prefill_pipeline::run_prefill_units(hooks, units, waves)?;
            ensure!(hooks.consumed.iter().flatten().all(|&count| count == 1), "Kimi unit hop inventory");
            Ok(())
        }, |hooks| {
            for local in hooks.binding.local.iter_mut() { local.clear_subwave(); }
            Ok(())
        })?;
        hooks.binding.local[self.plan.spec().head_gpu as usize].buffer("normalized")
    }

    /// Accepted replay writes every KDA layer on this lane. Publish those exact
    /// streams into the same dependency chain consumed by the next attention.
    pub(crate) fn commit_kda(&self, locals: &[LocalLane<'a>], tables: &[i32]) -> Result<()> {
        if tables.is_empty() { return Ok(()); }
        let lane = locals[0].lane_index();
        let bound = self.lanes.get(lane).context("Kimi commit lane outside admission")?;
        ensure!(locals.len() == bound.streams.len() && locals.iter().all(|local| local.lane_index() == lane) && (0..locals.len()).all(|rank|
            Rc::ptr_eq(locals[rank].stream(), &bound.streams[rank])),
            "Kimi commit streams differ from replay lane");
        for local in locals { local.commit_kda(tables)?; }
        for (layer, kind) in locals[0].config().attention.iter().enumerate() {
            if *kind != cuteafd_loader::families::kimi_k3::config::KimiAttention::Kda { continue; }
            let mut ready = self.state_ready[layer].borrow_mut();
            for rank in 0..locals.len() {
                ready[rank].record(locals[rank].stream())?;
                self.state_pending[layer][rank].set(true);
            }
        }
        Ok(())
    }

    /// Validate every borrowed link before entering the shared executor. No
    /// device guard spans the submission or an asynchronous yield.
    pub(crate) fn protocol<'e>(&'e self, peer: Option<&'e PeerExchange<'a>>, hops: Option<&'e HopLink<'a>>)
        -> Result<Protocol<'e, 'a>> {
        match (peer, hops, self.lanes[0].streams.len()) {
            (Some(peer), Some(hops), 2) => hops.validate_binding(&self.plan, peer)?,
            (None, None, 1) => {},
            _ => anyhow::bail!("Kimi protocol exchange owners differ from physical ranks"),
        }
        ensure!(self.entry_lane.get().is_none(), "Kimi entry publication was not consumed");
        Ok(Protocol { execution: self, peer, hops })
    }
}

pub(crate) struct Protocol<'e, 'a> {
    execution: &'e Execution<'a>,
    peer: Option<&'e PeerExchange<'a>>,
    hops: Option<&'e HopLink<'a>>,
}

impl<'e, 'a> Protocol<'e, 'a> {
    fn peer(&self) -> Result<&'e PeerExchange<'a>> { self.peer.context("single-device Kimi has no peer exchange") }
    fn hops(&self) -> Result<&'e HopLink<'a>> { self.hops.context("single-device Kimi has no hop link") }

    /// A source exchange stream joins both exact lane streams. Joining the
    /// destination's previous consumer also prevents receive-slot overwrite
    /// when a later hop reuses it. Cross-device event waits are intentional.
    fn producers(&self, lane: usize, exchange_rank: usize) -> Result<()> {
        let joins = self.execution.lanes.get(lane).context("Kimi hop lane outside admission")?;
        let mut events = joins.produced.borrow_mut();
        let raw = events.iter().map(|event| event.raw).collect::<Vec<_>>();
        let producers = producer_streams(self.execution.lanes.iter().map(|lane|
            [lane.streams[0].raw, lane.streams[1].raw]), lane)?;
        join_producers(producers, |rank, producer| {
            ensure!(producer == joins.streams[rank].raw, "Kimi hop producer stream changed");
            events[rank].record(&joins.streams[rank])
        }, |rank| self.peer()?.on(exchange_rank, || {
            // SAFETY: retained events were recorded on this exact lane's
            // producers; the exchange stream belongs to its validated rank.
            unsafe { joins.streams[exchange_rank].device.library.cuda_stream_wait_event(
                self.peer()?.stream(exchange_rank), raw[rank]) }
        }))
    }

    fn consumer(&self, lane: usize, rank: usize) -> Result<()> {
        let joins = self.execution.lanes.get(lane).context("Kimi hop lane outside admission")?;
        let event = &mut joins.landed.borrow_mut()[rank];
        self.peer()?.on(rank, || {
            // SAFETY: both the exchange stream and retained event belong to rank.
            unsafe { joins.streams[rank].device.library.cuda_event_record(
                event.raw, self.peer()?.stream(rank)) }
        })?;
        joins.streams[rank].device.run(|| {
            // SAFETY: this exact lane's consumer follows the matching hop wait.
            unsafe { joins.streams[rank].device.library.cuda_stream_wait_event(
                joins.streams[rank].raw, event.raw) }
        })
    }
}

impl LayerProtocol for Protocol<'_, '_> {
    type Ticket = Ticket;

    fn require_live(&self) -> Result<()> { self.peer.map_or(Ok(()), PeerExchange::require_live) }

    fn push_entry(&self, from: usize, source: *const c_void, destination: *mut c_void, bytes: usize) -> Result<()> {
        let lane = self.execution.entry_lane.get().context("Kimi entry source lane was not selected")?;
        self.producers(lane, from)?;
        self.peer()?.push_to(from, crate::shared::peer_split::DIRECT, source, destination, bytes)?;
        self.consumer(lane, from)
    }

    fn wait_entry(&self, to: usize) -> Result<()> {
        let lane = self.execution.entry_lane.get().context("Kimi entry publication has no lane")?;
        self.peer()?.wait(to, crate::shared::peer_split::DIRECT)?;
        self.consumer(lane, to)?;
        self.execution.entry_lane.set(None);
        Ok(())
    }

    fn send(&self, index: usize, lane: usize, source: *const c_void, rows: usize) -> Result<Ticket> {
        let hop = self.execution.plan.hops().get(index).context("Kimi hop outside plan")?;
        self.producers(lane, usize::from(hop.from))?;
        let ticket = Ticket { hop: self.hops()?.send(index, lane, source, rows)?, lane, to: usize::from(hop.to) };
        // The source delta may be overwritten by the FFN only after its copy.
        self.consumer(lane, usize::from(hop.from))?;
        Ok(ticket)
    }

    fn land(&self, ticket: Ticket) -> Result<*mut c_void> {
        let residual = self.hops()?.land(ticket.hop)?;
        self.consumer(ticket.lane, ticket.to)?;
        Ok(residual)
    }
}

impl Execution<'_> {
    /// Called by the family's entry pack binding: the shared protocol's entry
    /// ABI carries endpoints, not a lane ordinal. Refuse overlapping entries.
    pub(crate) fn entry(&self, lane: usize) -> Result<()> {
        ensure!(lane < self.lanes.len() && self.entry_lane.get().is_none(),
            "Kimi entry lane outside admission or publication still pending");
        self.entry_lane.set(Some(lane));
        Ok(())
    }
}

fn join_producers(producers: [*mut c_void; 2], mut record: impl FnMut(usize, *mut c_void) -> Result<()>,
    mut wait: impl FnMut(usize) -> Result<()>) -> Result<()> {
    for (rank, producer) in producers.into_iter().enumerate() { record(rank, producer)?; wait(rank)?; }
    Ok(())
}

trait ContextRankKernels {
    fn wait_prior_commit(&mut self, reader: usize, owner: usize) -> Result<()>;
    fn attention_commit(&mut self, rank: usize) -> Result<()>;
    fn record_commit(&mut self, rank: usize) -> Result<()>;
}

fn context_pair_with(kernels: &mut impl ContextRankKernels) -> Result<()> {
    // Snapshot both ranks' previous event generations before either event is
    // re-recorded. Either next-unit gather reads both persistent owner pools.
    for reader in 0..2 { for owner in 0..2 { kernels.wait_prior_commit(reader, owner)?; }}
    for rank in 0..2 {
        kernels.attention_commit(rank)?;
        kernels.record_commit(rank)?;
    }
    Ok(())
}

fn producer_streams(streams: impl Iterator<Item = [*mut c_void; 2]>, lane: usize)
    -> Result<[*mut c_void; 2]> {
    streams.into_iter().nth(lane).context("Kimi producer lane outside admission")
}

fn validate_rank_plan(plan: &LayerExecutionPlan, ranks: usize) -> Result<()> {
    validate_rank_geometry(plan.modes(), plan.spec(), ranks)?;
    ensure!(ranks == 2 || plan.hops().is_empty(), "single-device Kimi cannot retain peer hops");
    Ok(())
}

fn validate_rank_geometry(modes: &[LayerMode], spec: cuteafd_loader::placement::HopSpec, ranks: usize) -> Result<()> {
    use cuteafd_loader::placement::TerminalHome;
    ensure!(spec.entry_gpu == 0 && !modes.is_empty(), "Kimi entry or layer geometry");
    match ranks {
        1 => ensure!(spec.head_gpu == 0 && spec.terminal == TerminalHome::AttentionOwner
            && modes.iter().all(|mode| *mode == LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }),
            "single-device Kimi requires one whole attention/FFN owner and a local head"),
        2 => ensure!(spec.head_gpu == 1 && spec.terminal == TerminalHome::Replicated
            && modes.iter().all(|mode| matches!(mode, LayerMode::HeadSplit | LayerMode::ContextSplit
                | LayerMode::Whole { ffn: FfnMode::Split, .. })), "Kimi TP2 requires split FFNs"),
        _ => anyhow::bail!("Kimi needs one or two physical ranks"),
    }
    Ok(())
}

fn validate_rank_streams(streams: &[Vec<(i32, *mut c_void)>], devices: &[i32]) -> Result<()> {
    ensure!(matches!(devices.len(), 1 | 2) && devices.iter().all(|device| *device >= 0)
        && (devices.len() == 1 || devices[0] != devices[1]), "Kimi needs distinct physical rank devices");
    for (lane, ranks) in streams.iter().enumerate() {
        ensure!(ranks.len() == devices.len(), "Kimi lane physical rank count differs");
        for (rank, &(device, stream)) in ranks.iter().enumerate() {
            ensure!(device == devices[rank] && !stream.is_null()
                && streams[..lane].iter().all(|prior| prior[rank] != (device, stream)),
                "Kimi lane {lane} rank {rank} stream differs or aliases another lane");
        }
    }
    Ok(())
}

#[cfg(test)]
fn validate_streams(streams: &[[(i32, *mut c_void); 2]], devices: [i32; 2]) -> Result<()> {
    ensure!(devices[0] != devices[1], "Kimi exchange needs distinct devices");
    for (lane, pair) in streams.iter().enumerate() {
        for rank in 0..2 {
            ensure!(pair[rank].0 == devices[rank] && !pair[rank].1.is_null(),
                "Kimi lane {lane} rank {rank} stream differs from exchange device");
            ensure!(streams[..lane].iter().all(|prior| prior[rank] != pair[rank]),
                "Kimi lane {lane} aliases another lane's rank {rank} stream");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gathered_unit_history_requires_both_prior_owner_commits() {
        #[derive(Clone, Copy)]
        enum Step { Wait { unit: usize, owner: usize }, Gather(usize), Commit(usize) }

        // Explore every interleaving, not one favorable rank order: host enqueue
        // order does not order the two GPU streams' remote history reads.
        fn check(streams: &[Vec<Step>; 2]) -> std::result::Result<(), (usize, usize, usize)> {
            let mut pending = vec![[0; 2]];
            let mut visited = std::collections::HashSet::new();
            while let Some(at) = pending.pop() {
                if !visited.insert(at) { continue; }
                let committed = |owner: usize, unit: usize| streams[owner][..at[owner]].iter()
                    .any(|step| matches!(step, Step::Commit(done) if *done == unit));
                let mut progressed = false;
                for rank in 0..2 {
                    let Some(step) = streams[rank].get(at[rank]) else { continue; };
                    match *step {
                        Step::Wait { unit, owner } if !committed(owner, unit) => continue,
                        Step::Gather(unit) if unit > 0 => {
                            for owner in 0..2 {
                                if !committed(owner, unit - 1) { return Err((rank, unit, owner)); }
                            }
                        }
                        _ => {}
                    }
                    progressed = true;
                    let mut next = at;
                    next[rank] += 1;
                    pending.push(next);
                }
                assert!(progressed || at == streams.each_ref().map(Vec::len),
                    "gathered owner-commit waits deadlock at {at:?}");
            }
            Ok(())
        }

        struct Recording {
            streams: [Vec<Step>; 2], unit: usize, calls: Vec<(&'static str, usize, usize)>,
        }
        impl ContextRankKernels for Recording {
            fn wait_prior_commit(&mut self, reader: usize, owner: usize) -> Result<()> {
                self.calls.push(("wait", reader, owner));
                if self.unit > 0 {
                    self.streams[reader].push(Step::Wait { unit: self.unit - 1, owner });
                }
                Ok(())
            }
            fn attention_commit(&mut self, rank: usize) -> Result<()> {
                use crate::shared::peer_split::context::{gathered_prefill_with, GatheredKernels};
                self.calls.push(("attention", rank, rank));
                struct Gathered<'a> { stream: &'a mut Vec<Step>, unit: usize }
                impl GatheredKernels for Gathered<'_> {
                    fn gather(&mut self, _: usize) -> Result<()> {
                        self.stream.push(Step::Gather(self.unit)); Ok(())
                    }
                    fn produce(&mut self) -> Result<()> { Ok(()) }
                    fn attention(&mut self) -> Result<()> { Ok(()) }
                    fn commit(&mut self) -> Result<()> {
                        self.stream.push(Step::Commit(self.unit)); Ok(())
                    }
                    fn output(&mut self) -> Result<()> { Ok(()) }
                }
                gathered_prefill_with(&mut Gathered { stream: &mut self.streams[rank], unit: self.unit })
            }
            fn record_commit(&mut self, rank: usize) -> Result<()> {
                self.calls.push(("record", rank, rank)); Ok(())
            }
        }
        let mut recording = Recording { streams: std::array::from_fn(|_| Vec::new()),
            unit: 0, calls: Vec::new() };
        for unit in 0..4 {
            recording.unit = unit;
            context_pair_with(&mut recording).unwrap();
        }
        for calls in recording.calls.chunks_exact(8) {
            assert!(calls[..4].iter().all(|call| call.0 == "wait"),
                "both previous event generations must be consumed before either re-record");
        }
        let streams = recording.streams;
        assert_eq!(check(&streams), Ok(()));
        for reader in 0..2 { for unit in 1..4 {
            let peer = 1 - reader;
            let mut missing_peer_wait = streams.clone();
            missing_peer_wait[reader].retain(|step| !matches!(*step,
                Step::Wait { unit: prior, owner } if prior == unit - 1 && owner == peer));
            // Removing only the peer edge must expose a read of stale owner
            // pages even though the reader still waits for its own commit.
            assert_eq!(check(&missing_peer_wait), Err((reader, unit, peer)));
        }}
    }

    #[test]
    fn batched_route_transfer_keeps_exact_extents_and_rejects_invalid_planes() -> Result<()> {
        for rows in [1, 16, 64, 2752] {
            let lengths = [rows * 16 * 4, rows * 16 * 4, rows * cuteafd_ffi::KimiExpertInput::ROW_BYTES];
            for bytes in lengths {
                let source = CuteafdDeviceBuffer { ptr: 0x1000usize as *mut _, bytes, device_id: 3, flags: 0 };
                let destination = CuteafdDeviceBuffer { ptr: 0x2000usize as *mut _, bytes, device_id: 7, flags: 0 };
                let plane = transfer_plane(source, destination, bytes)?;
                assert_eq!((plane.rows, plane.row_bytes, plane.source_pitch, plane.destination_pitch),
                    (1, bytes as u64, bytes as u64, bytes as u64));
                assert!(transfer_plane(source, destination, bytes + 16).is_err());
                assert!(transfer_plane(source, destination, bytes - 1).is_err());
                assert!(transfer_plane(source, source, bytes).is_err());
                assert!(transfer_plane(source, CuteafdDeviceBuffer { ptr: std::ptr::null_mut(), ..destination }, bytes).is_err());
            }
        }
        let code = include_str!("layers.rs").split("#[cfg(test)]").next().unwrap();
        assert_eq!(code.matches("local.stage_attention(&metadata").count(), 0);
        Ok(())
    }

    #[test]
    fn lane_one_cannot_bind_only_lane_zero_streams() -> Result<()> {
        let lanes = [[(3, 0x100usize as *mut c_void), (7, 0x200usize as *mut c_void)],
            [(3, 0x300usize as *mut c_void), (7, 0x400usize as *mut c_void)]];
        validate_streams(&lanes, [3, 7])?;
        let streams = lanes.map(|pair| pair.map(|(_, stream)| stream));
        assert_eq!(producer_streams(streams.into_iter(), 1)?, streams[1]);
        assert_ne!(producer_streams(streams.into_iter(), 1)?, streams[0]);
        assert!(producer_streams(streams.into_iter(), 2).is_err());
        let calls = RefCell::new(Vec::new());
        join_producers(producer_streams(streams.into_iter(), 1)?, |rank, stream| {
            calls.borrow_mut().push((rank, stream)); Ok(())
        }, |rank| { calls.borrow_mut().push((rank, std::ptr::null_mut())); Ok(()) })?;
        assert_eq!(*calls.borrow(), [(0, streams[1][0]), (0, std::ptr::null_mut()),
            (1, streams[1][1]), (1, std::ptr::null_mut())]);
        for rank in 0..2 {
            let mut wrong = lanes;
            wrong[1][rank] = lanes[0][rank];
            assert!(validate_streams(&wrong, [3, 7]).is_err());
            wrong[1][rank] = (lanes[1][rank].0, std::ptr::null_mut());
            assert!(validate_streams(&wrong, [3, 7]).is_err());
            wrong[1][rank] = (lanes[1][1-rank].0, lanes[1][rank].1);
            assert!(validate_streams(&wrong, [3, 7]).is_err());
        }
        Ok(())
    }

    #[test]
    fn single_device_binds_one_real_owner_without_aliasing_a_peer() -> Result<()> {
        use cuteafd_loader::placement::{HopSpec, HopStorage, TerminalHome};
        let spec = HopSpec { storage: HopStorage::ReceiveSlots, row_bytes: 14336, rows: 64,
            lanes: 2, entry_gpu: 0, head_gpu: 0, terminal: TerminalHome::AttentionOwner };
        let modes = [LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }; 4];
        validate_rank_geometry(&modes, spec, 1)?;
        for ranks in [0, 2, 3] { assert!(validate_rank_geometry(&modes, spec, ranks).is_err()); }
        assert!(validate_rank_geometry(&[LayerMode::HeadSplit], spec, 1).is_err());
        assert!(validate_rank_geometry(&[LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner }], spec, 1).is_err());
        assert!(validate_rank_geometry(&modes, HopSpec { head_gpu: 1, ..spec }, 1).is_err());
        let streams = vec![vec![(3, 0x100usize as *mut c_void)], vec![(3, 0x200usize as *mut c_void)]];
        validate_rank_streams(&streams, &[3])?;
        assert!(validate_rank_streams(&streams, &[3, 3]).is_err());
        assert!(validate_rank_streams(&streams, &[7]).is_err());
        assert!(validate_rank_streams(&[streams[0].clone(), streams[0].clone()], &[3]).is_err());
        assert!(validate_rank_streams(&[vec![(3, std::ptr::null_mut())]], &[3]).is_err());
        Ok(())
    }

    #[test]
    fn producer_join_failure_never_queues_an_unrecorded_wait() {
        let calls = RefCell::new(Vec::new());
        let streams = [1usize as *mut c_void, 2usize as *mut c_void];
        let result = join_producers(streams, |rank, _| {
            calls.borrow_mut().push((rank, "record"));
            ensure!(rank != 1, "injected record failure"); Ok(())
        }, |rank| { calls.borrow_mut().push((rank, "wait")); Ok(()) });
        assert!(result.is_err());
        assert_eq!(*calls.borrow(), [(0, "record"), (0, "wait"), (1, "record")]);
    }
}

/// The shared pipeline owns layer/lane ordering; this binding supplies only
/// arithmetic and the exact payload at each planned boundary.
pub(crate) struct Bindings<'e, 'a, 's> {
    execution: &'e Execution<'a>,
    scope: &'e SubmissionScope<'s>,
    local: &'e mut [LocalLane<'a>],
    routed: &'e mut RoutedLane<'a>,
    experts: &'e mut [Option<Exl3Tp2<'a>>],
    protocol: Protocol<'e, 'a>,
    lane: usize,
    rows: usize,
    speculative: bool,
    owners: Vec<(Vec<u8>, Vec<u8>, Option<Vec<u8>>)>,
    completed: usize,
    pending: Option<FfnTicket>,
    draft: Option<&'e super::draft::NativeDraft<'a>>,
    probe: Option<&'e DraftProbe<'a>>,
    tap_first: usize,
    defer_publication: bool,
    packed: Option<Vec<super::device::AttentionRows>>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct FfnTicket { layer: usize, spark: Option<Spark>, rtx: Option<usize> }

/// The Spark wave of one FFN, on the routed link's exchange.
type Spark = super::spark::Ticket;

fn transfer_plane(source: CuteafdDeviceBuffer, destination: CuteafdDeviceBuffer, bytes: usize)
    -> Result<cuteafd_ffi::peer_exchange::PeerPlane> {
    ensure!(bytes > 0 && bytes <= source.bytes && bytes <= destination.bytes
        && bytes % 16 == 0 && !source.ptr.is_null() && !destination.ptr.is_null()
        && (source.ptr as usize | destination.ptr as usize) % 16 == 0
        && source.device_id != destination.device_id, "Kimi canonical transfer extent/device/alignment");
    Ok(cuteafd_ffi::peer_exchange::PeerPlane { source: source.ptr, destination: destination.ptr,
        rows: 1, row_bytes: bytes as u64, source_pitch: bytes as u64, destination_pitch: bytes as u64 })
}

impl<'e, 'a, 's> Bindings<'e, 'a, 's> {
    pub(crate) fn new(local: &'e mut [LocalLane<'a>], routed: &'e mut RoutedLane<'a>,
        experts: &'e mut [Option<Exl3Tp2<'a>>], execution: &'e Execution<'a>, peer: Option<&'e PeerExchange<'a>>, hops: Option<&'e HopLink<'a>>,
        scope: &'e SubmissionScope<'s>, lane: usize, rows: usize, speculative: bool) -> Result<Self> {
        let protocol = execution.protocol(peer, hops)?;
        let joins = execution.lanes.get(lane).context("Kimi binding lane outside admission")?;
        validate_rank_plan(&execution.plan, local.len())?;
        ensure!((1..=execution.plan.spec().rows as usize).contains(&rows)
            && local.len() == joins.streams.len() && experts.len() == local.len()
            && execution.plan.modes().len() == local[0].config().layers,
            "Kimi binding row shape or physical rank geometry");
        for rank in 0..local.len() {
            ensure!(Rc::ptr_eq(local[rank].stream(), &joins.streams[rank])
                && local[rank].lane_index() == lane && rows <= local[rank].rows(),
                "Kimi layer binding differs from admitted lane owner");
        }
        let mut owners = Vec::new();
        for layer in 0..execution.plan.modes().len() {
            let mut resident = (Vec::new(), Vec::new(), None::<Vec<u8>>);
            for rank in 0..local.len() {
                let (weights, state, indexer) = local[rank].attention_resident(layer)?;
                if weights { resident.0.push(rank as u8); }
                if state { resident.1.push(rank as u8); }
                if let Some(present) = indexer {
                    let list = resident.2.get_or_insert_with(Vec::new);
                    if present { list.push(rank as u8); }
                }
            }
            execution.plan.validate_state_owners(layer, &resident.0)?;
            execution.plan.validate_state_owners(layer, &resident.1)?;
            if let Some(indexer) = &resident.2 {
                execution.plan.validate_state_owners(layer, indexer)?;
            }
            owners.push(resident);
        }
        if let Some(peer) = peer {
            for rank in 0..local.len() { peer.recv(rank, lane)?; }
            ensure!(peer.slot_bytes() >= rows * 7168 * 2, "Kimi peer slot shorter than hidden rows");
        }
        Ok(Self { execution, scope, local, routed, experts, protocol, lane, rows, speculative, owners, completed: 0, pending: None, draft: None, probe: None, tap_first: 0, defer_publication: false, packed: None })
    }

    fn begin_local_ffn(&mut self, layer: usize, owner: Option<usize>) -> Result<FfnTicket> {
        ensure!(owner == Some(0) && self.pending.is_none(), "Kimi single-owner FFN binding");
        let dense = self.local[0].config().dense[layer];
        if !dense { self.local[0].residual(MixSite::Ffn(layer), self.rows)?; }
        let rtx = !dense && self.experts[0].as_ref().is_some_and(|expert| expert.layers().contains(&layer));
        let spark = if dense { None } else {
            let tally = self.execution.expert_usage.as_ref().map(|usage|
                usage.routes(self.local[0].stream().device.id, layer, self.execution.usage_phase.get())).transpose()?;
            let (routes, wire) = self.local[0].route(layer, self.rows, tally)?;
            self.with_probe(|probe| {
                if probe.wants_ring(false) {
                    let stream = self.local[0].stream();
                    // SAFETY: this lane owns the route plane until its ordered readers finish.
                    unsafe { probe.ring(stream.device.library, layer, routes.ids, self.rows, stream.raw)?; }
                }
                Ok(())
            })?;
            if self.execution.skip_routed_experts { None } else if rtx {
                let stream = self.local[0].stream();
                let expert = self.experts[0].as_mut().context("Kimi whole expert absent")?;
                ensure!(expert.shard() == RtxShard::Whole && expert.device() == stream.device.id,
                    "Kimi single-device expert is not a whole physical owner");
                // SAFETY: all operands and whole-expert workspace are retained through lane drainage.
                unsafe { expert.enqueue(layer, self.rows, ExpertInput::Fp8K32(wire.ptr),
                    Routes { ids: routes.ids.ptr, weights: routes.weights.ptr }, stream.raw, &mut || {})?; }
                ensure!(expert.partial() == crate::shared::experts::rtx::PartialDtype::F32
                    && expert.partial_layout(self.rows)? == crate::shared::experts::rtx::PartialLayout::Tokens,
                    "Kimi whole expert must emit final FP32 token sums");
                let partial = CuteafdDeviceBuffer { ptr: expert.output(), bytes: self.rows * 3584 * 4,
                    device_id: expert.device(), flags: 0 };
                let output = self.local[0].buffer("expert combined")?;
                stream.device.run(|| crate::shared::dense_trellis::retain_partial(
                    &self.protocol.execution.lanes[self.lane].narrow, partial, output, self.rows * 3584, stream.raw))?;
                None
            } else {
                let kind = if self.speculative { cuteafd_transport::ExpertV2SourceKind::MtpVerify }
                    else if self.rows > 64 { cuteafd_transport::ExpertV2SourceKind::Prefill }
                    else { cuteafd_transport::ExpertV2SourceKind::Decode };
                Some(self.routed.begin(&mut self.local[0], layer, self.rows, kind, routes, wire)?)
            }
        };
        self.local[0].ffn(layer, self.rows)?;
        let ticket = FfnTicket { layer, spark, rtx: rtx.then_some(0) };
        self.pending = Some(ticket);
        Ok(ticket)
    }

    fn finish_local_ffn(&mut self, layer: usize, ticket: FfnTicket) -> Result<()> {
        let shared = self.local[0].ffn_result(layer, self.rows)?;
        if ticket.spark.is_some() || ticket.rtx.is_some() {
            self.local[0].adopt("shared output", shared, self.rows, 7168)?;
            if let Some(spark) = ticket.spark { self.routed.finish(&self.local[0], spark)?; }
            self.local[0].latent_up(layer, self.rows)?;
            let combined = self.local[0].combine_moe(self.rows)?;
            self.local[0].adopt("delta", combined, self.rows, 7168)?;
        } else {
            self.local[0].adopt("delta", shared, self.rows, 7168)?;
        }
        self.local[0].capture_post_ffn(layer, self.rows)?;
        if let Some(draft) = self.draft { draft.tap_at(layer, &mut self.local[0], self.rows, self.tap_first)?; }
        self.with_probe(|probe| {
            // SAFETY: this mark follows all FFN consumers on the only physical lane stream.
            unsafe { probe.mark(layer, self.local[0].stream().raw) }
        })?;
        self.pending = None;
        self.completed += 1;
        Ok(())
    }

    fn with_probe(&self, work: impl FnOnce(&mut crate::shared::draft::probe::RoundProbe<'a>) -> Result<()>) -> Result<()> {
        if let Some(owner) = self.probe {
            if let Some(probe) = owner.borrow_mut().as_mut() {
                let device = probe.device;
                device.run(|| work(probe))?;
            }
        }
        Ok(())
    }

    fn before_attention(&mut self, layer: usize) -> Result<()> {
        ensure!(layer == self.completed && self.pending.is_none(), "Kimi layer order differs from shared pipeline");
        for local in self.local.iter_mut() {
            if layer == 0 { local.seed(self.rows)?; }
            // Packed members and context buckets stage metadata between the
            // live-row residual and attention; ordinary attention folds it in.
            if self.packed.is_some() || self.execution.plan.modes()[layer] == LayerMode::ContextSplit
                || local.windowed() || !local.attention_resident(layer)?.0 {
                local.residual(MixSite::Attention(layer), self.rows)?;
            }
        }
        Ok(())
    }

    fn attention_rank(&mut self, layer: usize, rank: usize) -> Result<()> {
        let stream = self.local[rank].stream().clone();
        let execution = self.protocol.execution;
        let mut ready = execution.state_ready[layer].borrow_mut();
        if execution.state_pending[layer][rank].get() {
            stream.device.run(|| {
                // SAFETY: ordered chunks share this layer's state. The retained
                // event names the previous exact lane, not a rank-global stream.
                unsafe { stream.device.library.cuda_stream_wait_event(stream.raw, ready[rank].raw) }
            })?;
        }
        if let Some(members) = &self.packed {
            self.local[rank].attention_packed(layer, members)?;
        } else {
            self.local[rank].attention(layer, self.rows, self.speculative)?;
        }
        ready[rank].record(&stream)?;
        execution.state_pending[layer][rank].set(true);
        Ok(())
    }

    fn publish(&self, source: &str) -> Result<()> {
        // Both publications precede either wait. Every event names this lane,
        // never an assumed rank-global compute stream.
        for rank in 0..2 {
            self.protocol.producers(self.lane, rank)?;
            self.protocol.peer()?.push(rank, self.lane, self.local[rank].buffer(source)?.ptr,
                self.rows * 7168 * 2)?;
        }
        Ok(())
    }

    fn meet(&mut self, source: &str, destination: &str) -> Result<()> {
        for rank in 0..2 {
            self.protocol.peer()?.wait(rank, self.lane)?;
            self.protocol.peer()?.add(rank, self.local[rank].buffer(source)?.ptr,
                self.protocol.peer()?.recv(rank, self.lane)?, self.local[rank].buffer(destination)?.ptr,
                self.rows * 7168)?;
            self.protocol.consumer(self.lane, rank)?;
        }
        Ok(())
    }

    fn publish_core(&self,layer:usize,ffn:bool)->Result<()> {
        let partials=[self.local[0].core_partial(layer,ffn,self.rows)?,self.local[1].core_partial(layer,ffn,self.rows)?];
        ensure!(std::mem::discriminant(&partials[0]) == std::mem::discriminant(&partials[1]),
            "Kimi peer partial precision/basis differs between ranks");
        for rank in 0..2 {
            let buffer=partials[rank].buffer();
            ensure!(self.protocol.peer()?.slot_bytes()>=buffer.bytes,"Kimi peer slot shorter than typed partial");
            self.protocol.producers(self.lane,rank)?;
            self.protocol.peer()?.push(rank,self.lane,buffer.ptr,buffer.bytes)?;
        }
        Ok(())
    }
    /// Native K6 and fused FFN partials both cross the peer link as BF16.
    fn publish_ffn(&self, layer: usize) -> Result<()> {
        if self.local[0].native_ffn(layer)? {
            ensure!(self.local[1].native_ffn(layer)?, "Kimi K6 FFN rank formats differ");
            self.publish_core(layer,true)
        } else {
            self.publish("ffn output")
        }
    }
    fn meet_core(&mut self,layer:usize,ffn:bool,destination:&str)->Result<()> {
        for rank in 0..2 {
            self.protocol.peer()?.wait(rank,self.lane)?;
            let partial=self.local[rank].core_partial(layer,ffn,self.rows)?;
            let peer=partial.received(CuteafdDeviceBuffer{ptr:self.protocol.peer()?.recv(rank,self.lane)?,
                bytes:partial.buffer().bytes,device_id:self.local[rank].stream().device.id,flags:0})?;
            self.protocol.peer()?.on(rank,||self.local[rank].reduce_core(layer,ffn,self.rows,peer,destination,
                self.protocol.peer()?.stream(rank)))?;
            self.protocol.consumer(self.lane,rank)?;
        }
        Ok(())
    }

    /// Copies into retained lane operands, not a second exchange ring. Both
    /// producer joins include previous readers before any destination overwrite.
    fn transfer(&self, from: usize, name: &str, bytes: usize) -> Result<()> {
        ensure!(from < 2, "Kimi transfer source rank");
        let to = 1 - from;
        let plane = transfer_plane(self.local[from].buffer(name)?, self.local[to].buffer(name)?, bytes)?;
        self.protocol.producers(self.lane, from)?;
        self.protocol.peer()?.push_to(from, crate::shared::peer_split::DIRECT,
            plane.source, plane.destination, bytes)?;
        self.protocol.consumer(self.lane, from)?;
        self.protocol.peer()?.wait(to, crate::shared::peer_split::DIRECT)?;
        self.protocol.consumer(self.lane, to)
    }

    fn transfer_planes(&self, from: usize, operands: &[(&str, usize)]) -> Result<()> {
        ensure!(from < 2, "Kimi transfer source rank");
        let to = 1 - from;
        let planes = operands.iter().map(|&(name, bytes)| {
            transfer_plane(self.local[from].buffer(name)?, self.local[to].buffer(name)?, bytes)
        }).collect::<Result<Vec<_>>>()?;
        self.protocol.producers(self.lane, from)?;
        self.protocol.peer()?.push_planes(from, crate::shared::peer_split::DIRECT, &planes)?;
        self.protocol.consumer(self.lane, from)?;
        self.protocol.peer()?.wait(to, crate::shared::peer_split::DIRECT)?;
        self.protocol.consumer(self.lane, to)
    }

    /// Called after the shared pipeline, only on the head rank that owns the
    /// output score/norm weights. The other rank's stored history remains intact.
    pub(crate) fn output(&mut self, output: Output) -> Result<CuteafdDeviceBuffer> {
        ensure!(self.pending.is_none() && self.completed == self.protocol.execution.plan.modes().len(),
            "Kimi output requested before admitted layer completion");
        let head = self.protocol.execution.plan.spec().head_gpu as usize;
        let mut probed = false;
        self.with_probe(|probe| { probed = probe.active(); Ok(()) })?;
        if probed && self.local.len() == 2 {
            // The final GPU0 ring download must precede GPU1's existing selector sync.
            // Join the exact lane producers into the retained head exchange/consumer.
            self.protocol.producers(self.lane, 1)?;
            self.protocol.consumer(self.lane, 1)?;
        }
        match output {
            Output::Normalized => self.local[head].residual(MixSite::Output, self.rows),
            Output::Residual => self.local[head].terminal_residual(self.rows),
        }
    }
}

impl LayerKernels for Bindings<'_, '_, '_> {
    fn pack(&mut self, hop: Hop) -> Result<*const c_void> {
        ensure!(matches!(hop.at, HopPoint::Entry | HopPoint::AfterAttention(_)),
            "Kimi split-FFN plan cannot move a residual history arena");
        if hop.at == HopPoint::Entry { self.protocol.execution.entry(self.lane)?; }
        // Owned MLA publishes only its rounded attended delta. Both ranks have
        // already advanced identical residual histories before this hop.
        Ok(self.local[usize::from(hop.from)].buffer("delta")?.ptr)
    }
    fn entry_destination(&mut self, rank: usize) -> Result<*mut c_void> {
        Ok(self.local.get(rank).context("Kimi entry rank")?.buffer("delta")?.ptr)
    }
    fn adopt(&mut self, hop: Hop, residual: *mut c_void) -> Result<()> {
        let rank = usize::from(hop.to);
        if hop.at == HopPoint::Entry {
            ensure!(residual == self.local[rank].buffer("delta")?.ptr, "Kimi entry destination changed");
            return Ok(());
        }
        self.local[rank].adopt("delta", CuteafdDeviceBuffer { ptr: residual,
            bytes: self.rows * 7168 * 2, device_id: self.local[rank].stream().device.id, flags: 0 }, self.rows, 7168)
    }
    fn heads(&mut self, layer: usize) -> Result<()> {
        self.before_attention(layer)?;
        for rank in 0..2 { self.attention_rank(layer, rank)?; }
        if self.local[0].native_attention(layer)? {
            ensure!(self.local[1].native_attention(layer)?,"Kimi K6 attention rank formats differ");
            self.publish_core(layer,false)?;self.meet_core(layer,false,"delta")
        } else {
            self.publish("attention output")?;
            self.meet("attention output", "delta")
        }
    }
    fn context(&mut self, layer: usize, _: usize) -> Result<()> {
        self.before_attention(layer)?;
        struct Pair<'b, 'e, 'a, 's> { binding: &'b mut Bindings<'e, 'a, 's>, layer: usize }
        impl ContextRankKernels for Pair<'_, '_, '_, '_> {
            fn wait_prior_commit(&mut self, reader: usize, owner: usize) -> Result<()> {
                let execution = self.binding.protocol.execution;
                if execution.state_pending[self.layer][owner].get() {
                    let ready = execution.state_ready[self.layer].borrow();
                    let stream = self.binding.local[reader].stream();
                    stream.device.run(|| {
                        // SAFETY: the event and both exact compute streams are
                        // retained by Execution; cross-device waits are allowed.
                        unsafe { stream.device.library.cuda_stream_wait_event(stream.raw, ready[owner].raw) }
                    })?;
                }
                Ok(())
            }
            fn attention_commit(&mut self, rank: usize) -> Result<()> {
                if let Some(members) = &self.binding.packed {
                    self.binding.local[rank].attention_packed(self.layer, members)
                } else {
                    self.binding.local[rank].context_attention(self.layer, self.binding.rows, self.binding.speculative)
                }
            }
            fn record_commit(&mut self, rank: usize) -> Result<()> {
                let execution = self.binding.protocol.execution;
                execution.state_ready[self.layer].borrow_mut()[rank]
                    .record(self.binding.local[rank].stream())?;
                execution.state_pending[self.layer][rank].set(true);
                Ok(())
            }
        }
        context_pair_with(&mut Pair { binding: self, layer })?;
        // Diagnostic only: both peers must have issued their waits/pushes before
        // an opted-in readback can drain either producer stream.
        self.local[0].context_dsa_dump(layer, self.rows)?;
        // Both whole-step replays must be queued before ordinary producer
        // joins, otherwise rank 0's device wait can block rank 1's publication.
        if self.local[0].native_attention(layer)? {
            ensure!(self.local[1].native_attention(layer)?, "Kimi K6 attention rank formats differ");
            self.publish_core(layer,false)?; self.meet_core(layer, false, "delta")
        } else {
            self.publish("attention output")?;
            self.meet("attention output", "delta")
        }
    }
    fn owned(&mut self, layer: usize, gpu: usize) -> Result<()> {
        ensure!(gpu < self.local.len(), "Kimi attention owner rank");
        self.before_attention(layer)?;
        self.attention_rank(layer, gpu)?;
        let delta = self.local[gpu].buffer("attention output")?;
        self.local[gpu].adopt("delta", delta, self.rows, 7168)
    }
    fn ffn(&mut self, layer: usize, owner: Option<usize>) -> Result<()> {
        let ticket = self.begin_ffn(layer, owner)?;
        self.finish_ffn(layer, ticket)
    }
}

impl PipelineKernels for Bindings<'_, '_, '_> {
    type FfnTicket = FfnTicket;
    fn state_owners(&self, layer: usize) -> Result<AttentionStateOwners<'_>> {
        let owners = self.owners.get(layer).context("Kimi layer census outside inventory")?;
        Ok(AttentionStateOwners { weights: &owners.0, kv: &owners.1, indexer: owners.2.as_deref(), compressor: None })
    }
    fn begin_ffn(&mut self, layer: usize, owner: Option<usize>) -> Result<FfnTicket> {
        if self.local.len() == 1 { return self.begin_local_ffn(layer, owner); }
        ensure!(owner.is_none() && self.pending.is_none(), "Kimi FFN owner or unfinished wave");
        let dense = self.local[0].config().dense[layer];
        if !dense {
            for local in self.local.iter_mut() { local.residual(MixSite::Ffn(layer), self.rows)?; }
        }
        let owner = if dense { 0 } else { super::weights::routed_owner(self.local[0].config(), layer)? };
        let rtx = !dense && self.experts.iter().any(|expert|
            expert.as_ref().is_some_and(|e| e.layers().contains(&layer)));
        let spark = if dense || self.execution.skip_routed_experts {
            if !dense { self.local[owner].route(layer, self.rows, None)?; }
            for local in self.local.iter_mut() { local.ffn(layer, self.rows)?; }
            None
        } else {
            let tally = self.execution.expert_usage.as_ref().map(|usage|
                usage.routes(self.local[owner].stream().device.id, layer, self.execution.usage_phase.get())).transpose()?;
            let (canonical, wire) = self.local[owner].route(layer, self.rows, tally)?;
            if rtx || owner == 0 {
                self.transfer_planes(owner, &[("route ids", canonical.ids.bytes),
                    ("route weights", canonical.weights.bytes), ("expert wire", wire.bytes)])?;
            }
            // The optional shared ring check also preserves remote rank1 routes on GPU0.
            let mut check_remote = false;
            // Every routed layer rings its ids on rank 0 (Spark layers included:
            // their staged ids are read by the exchange worker, not the host).
            self.with_probe(|probe| { check_remote = probe.wants_ring(false); Ok(()) })?;
            if check_remote && !rtx && owner == 1 { self.transfer(1, "route ids", canonical.ids.bytes)?; }
            // Local IDs have joined rank0 before the reusable router outputs change.
            self.with_probe(|probe| {
                if probe.wants_ring(false) {
                    let stream = self.local[0].stream();
                    // SAFETY: transfer joined these exact rank0 operands and stream.
                    unsafe { probe.ring(stream.device.library, layer,
                        self.local[0].buffer("route ids")?, self.rows, stream.raw)?; }
                }
                Ok(())
            })?;
            if rtx {
                for rank in 0..2 {
                    let stream = self.local[rank].stream();
                    let expert = self.experts[rank].as_mut().context("Kimi missing TP2 expert rank")?;
                    ensure!(expert.shard() == (RtxShard::Tp2 { rank: rank as u8 })
                        && expert.device() == stream.device.id && expert.layers().contains(&layer),
                        "Kimi TP2 expert shard/device/layers differ from admission");
                    let peer = self.protocol.peer()?;
                    // SAFETY: canonical input publication joins this exact lane;
                    // weights and scratch survive replay and terminal teardown.
                    unsafe { expert.enqueue(layer, self.rows,
                        ExpertInput::Fp8K32(self.local[rank].buffer("expert wire")?.ptr),
                        Routes { ids: self.local[rank].buffer("route ids")?.ptr,
                            weights: self.local[rank].buffer("route weights")?.ptr }, stream.raw,
                        &mut || crate::shared::decode_graph::fatal_drain(peer.publish_abort(),
                            "Kimi TP2 expert abort"))?; }
                }
                let experts = [self.experts[0].as_ref().unwrap(), self.experts[1].as_ref().unwrap()];
                ensure!(experts[0].partial() == experts[1].partial()
                    && experts[0].partial_layout(self.rows)? == experts[1].partial_layout(self.rows)?,
                    "Kimi TP2 partial contracts differ");
                let partials = std::array::from_fn(|rank| CuteafdDeviceBuffer {
                    ptr: experts[rank].output(), bytes: self.rows * 3584 * 4,
                    device_id: experts[rank].device(), flags: 0 });
                // SAFETY: both final FP32 token partials are on the retained
                // producer streams; reduction completes before hidden-slot reuse.
                unsafe { self.protocol.execution.lanes[self.lane].reducer.as_ref().context("Kimi TP2 reducer absent")?.borrow_mut().reduce(
                    self.scope, self.protocol.peer()?, owner, self.rows, partials,
                    self.local[owner].buffer("expert combined")?, experts[0].partial(),
                    experts[0].partial_layout(self.rows)?)?; }
                for local in self.local.iter_mut() { local.ffn(layer, self.rows)?; }
                None
            } else {
                let kind = if self.speculative { cuteafd_transport::ExpertV2SourceKind::MtpVerify }
                    else if self.rows > 64 { cuteafd_transport::ExpertV2SourceKind::Prefill }
                    else { cuteafd_transport::ExpertV2SourceKind::Decode };
                let routes = cuteafd_ffi::KimiRoutes {
                    ids: self.local[1].buffer("route ids")?, weights: self.local[1].buffer("route weights")? };
                let wire = self.local[1].buffer("expert wire")?;
                let spark = self.routed.begin(&mut self.local[1], layer, self.rows, kind, routes, wire)?;
                self.local[0].ffn(layer, self.rows)?;
                Some(spark)
            }
        };
        if !self.defer_publication { self.publish_ffn(layer)?; }
        let ticket = FfnTicket { layer, spark, rtx: rtx.then_some(owner) };
        self.pending = Some(ticket);
        Ok(ticket)
    }
    fn finish_ffn(&mut self, layer: usize, ticket: FfnTicket) -> Result<()> {
        ensure!(self.pending == Some(ticket) && ticket.layer == layer, "Kimi FFN ticket differs from pending layer");
        if self.local.len() == 1 { return self.finish_local_ffn(layer, ticket); }
        if self.defer_publication { self.publish_ffn(layer)?; }
        if ticket.spark.is_some() || ticket.rtx.is_some() {
            if self.local[0].native_ffn(layer)? {self.meet_core(layer,true,"shared output")?;}
            else {self.meet("ffn output", "shared output")?;}
            let rank = super::weights::routed_owner(self.local[0].config(), layer)?;
            if let Some(spark) = ticket.spark {
                self.routed.finish(&self.local[1], spark)?;
                if rank == 0 { self.transfer(1, "expert combined", self.rows * 3584 * 2)?; }
            }
            let output = self.local[rank].latent_up(layer, self.rows)?;
            let other = 1 - rank;
            self.protocol.producers(self.lane, rank)?;
            self.protocol.peer()?.push_to(rank, crate::shared::peer_split::DIRECT, output.ptr,
                self.local[other].buffer("routed output")?.ptr, self.rows * 7168 * 2)?;
            self.protocol.consumer(self.lane, rank)?;
            self.protocol.peer()?.wait(other, crate::shared::peer_split::DIRECT)?;
            self.protocol.consumer(self.lane, other)?;
            for local in self.local.iter_mut() {
                let combined = local.combine_moe(self.rows)?;
                local.adopt("delta", combined, self.rows, 7168)?;
            }
        } else {
            if self.local[0].native_ffn(layer)? {self.meet_core(layer,true,"delta")?;}
            else {self.meet("ffn output", "delta")?;}
        }
        self.local[0].capture_post_ffn(layer, self.rows)?;
        // Checkpoint taps are the completed decoder prefix plus final FFN
        // delta, before the following residual mixer or final output norm.
        if let Some(draft) = self.draft { draft.tap_at(layer, &mut self.local[0], self.rows, self.tap_first)?; }
        self.with_probe(|probe| {
            // SAFETY: the FFN join above orders both ranks before this rank0 mark.
            unsafe { probe.mark(layer, self.local[0].stream().raw) }
        })?;
        self.pending = None;
        self.completed += 1;
        Ok(())
    }
}

#[derive(Default, Clone, Copy)]
struct UnitState { completed: usize }

/// Each unit owns its rows of the lane's operands: selecting a unit points
/// every row plane (residual arena, delta, outputs, router/wire, K6 partials)
/// at its rows, so another unit's hooks never touch them. Attention metadata
/// are immutable per-unit banks; the phase arena is rewritten inside each hook.
struct UnitHooks<'h, 'e, 'a, 's> {
    binding: Bindings<'e, 'a, 's>,
    protocol: &'h Protocol<'e, 'a>,
    state: Vec<UnitState>,
    consumed: Vec<Vec<usize>>,
}

impl UnitHooks<'_, '_, '_, '_> {
    fn select(&mut self, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
        for local in self.binding.local.iter_mut() {
            local.select_subwave(Some(unit.rows.clone()))?;
            local.select_core_view_unit(unit.index, unit.rows.len())?;
        }
        self.binding.rows = unit.rows.len();
        self.binding.completed = self.state[unit.index].completed;
        self.binding.tap_first = unit.rows.start;
        Ok(())
    }
}

impl crate::shared::prefill_pipeline::PrefillHooks for UnitHooks<'_, '_, '_, '_> {
    type Ticket = FfnTicket;
    fn route(&self, layer: usize) -> crate::shared::prefill_pipeline::FfnRoute {
        // Dense and RTX-resident layers complete inside one hook; the Spark
        // ones hold a wave across hooks.
        let cfg = self.binding.local[0].config();
        let rtx = self.binding.experts.iter().any(|expert| expert.as_ref().is_some_and(|e| e.layers().contains(&layer)));
        if cfg.dense[layer] || rtx || self.binding.execution.skip_routed_experts { crate::shared::prefill_pipeline::FfnRoute::Local }
        else { crate::shared::prefill_pipeline::FfnRoute::Spark }
    }
    fn begin_unit(&mut self, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
        ensure!(unit.layers.start == 0 && unit.sequence == 0, "Kimi units run every layer of one sequence");
        Ok(())
    }
    fn attention(&mut self, layer: usize, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
        self.select(unit)?;
        self.binding.pending = None;
        crate::shared::peer_split::attention::pipeline_attention(self.protocol, &self.binding.protocol.execution.plan,
            layer, 0, unit.rows.len(), &mut self.binding, &mut self.consumed[unit.index])?;
        Ok(())
    }
    fn local_ffn(&mut self, layer: usize, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
        self.select(unit)?;
        self.binding.pending = None;
        let ticket = self.binding.begin_ffn(layer, (self.binding.local.len() == 1).then_some(0))?;
        self.binding.finish_ffn(layer, ticket)?;
        self.state[unit.index].completed += 1;
        Ok(())
    }
    fn send_experts(&mut self, layer: usize, unit: &crate::shared::prefill_pipeline::Unit) -> Result<FfnTicket> {
        self.select(unit)?;
        self.binding.pending = None;
        let ticket = self.binding.begin_ffn(layer, (self.binding.local.len() == 1).then_some(0))?;
        self.binding.pending = None;
        Ok(ticket)
    }
    fn land_experts(&mut self, layer: usize, unit: &crate::shared::prefill_pipeline::Unit, ticket: FfnTicket)
        -> Result<()> {
        self.select(unit)?;
        self.binding.pending = Some(ticket);
        self.binding.finish_ffn(layer, ticket)?;
        self.state[unit.index].completed += 1;
        Ok(())
    }
    fn end_unit(&mut self, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
        self.select(unit)?;
        crate::shared::peer_split::attention::pipeline_hops(self.protocol, &self.binding.protocol.execution.plan,
            HopPoint::Exit, 0, unit.rows.len(), &mut self.binding, &mut self.consumed[unit.index])?;
        self.binding.output(Output::Normalized)?;
        Ok(())
    }
    fn commit_unit(&mut self, _: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
        // KDA state and MLA records were written in attention; the caller
        // commits the frontier once for the whole call.
        Ok(())
    }
}
