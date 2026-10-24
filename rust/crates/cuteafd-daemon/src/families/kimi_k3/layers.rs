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
    streams: [Rc<Stream<'a>>; 2],
    produced: RefCell<[Event<'a>; 2]>,
    landed: RefCell<[Event<'a>; 2]>,
    reducer: RefCell<TokenOwnerReduce<'a>>,
}

#[derive(Clone, Copy)]
pub(crate) enum Output { Normalized, Residual }

/// Own these bindings for the engine lifetime, not in a temporary submission:
/// recorded event nodes, streams and all exchanged operands must survive replay.
/// The enclosing submission owner aborts all links before draining these streams.
type DraftProbe<'a> = RefCell<Option<crate::shared::memory::device::DeviceOwner<'a,
    crate::shared::draft::probe::RoundProbe<'a>>>>;

pub(crate) struct Execution<'a> {
    pub plan: LayerExecutionPlan,
    lanes: Vec<LaneJoins<'a>>,
    entry_lane: Cell<Option<usize>>,
    state_ready: Vec<RefCell<[Event<'a>; 2]>>,
    state_pending: Vec<[Cell<bool>; 2]>,
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
    pub(crate) fn new(plan: LayerExecutionPlan, peer: &PeerExchange<'a>, hops: &HopLink<'a>,
        exchange_streams: [Rc<Stream<'a>>; 2], streams: Vec<[Rc<Stream<'a>>; 2]>) -> Result<Self> {
        hops.validate_binding(&plan, peer)?;
        ensure!(streams.len() as u64 == plan.spec().lanes && !streams.is_empty(),
            "Kimi layer streams differ from admitted lanes");
        let handles = streams.iter().map(|pair| pair.each_ref().map(|stream|
            (stream.device.id, stream.raw))).collect::<Vec<_>>();
        validate_streams(&handles, [peer.device(0), peer.device(1)])?;
        ensure!(streams.iter().flatten().all(|stream| std::ptr::eq(
            stream.device.library, streams[0][0].device.library)), "Kimi lane native libraries differ");
        let mut state_ready = Vec::new();
        let mut state_pending = Vec::new();
        for _ in plan.modes() {
            state_ready.push(RefCell::new([Event::new(streams[0][0].device)?, Event::new(streams[0][1].device)?]));
            state_pending.push([Cell::new(false), Cell::new(false)]);
        }
        let lanes = streams.into_iter().enumerate().map(|(lane, streams)| {
            let produced = [Event::new(streams[0].device)?, Event::new(streams[1].device)?];
            let landed = [Event::new(streams[0].device)?, Event::new(streams[1].device)?];
            let reducer = RefCell::new(TokenOwnerReduce::new(peer, exchange_streams.clone(),
                streams.clone(), lane, plan.spec().rows as usize, 3584)?);
            Ok(LaneJoins { streams, produced: RefCell::new(produced), landed: RefCell::new(landed), reducer })
        }).collect::<Result<_>>()?;
        Ok(Self { plan, lanes, entry_lane: Cell::new(None), state_ready, state_pending })
    }

    /// The family has no second scheduler: use the shared layer-major pipeline
    /// for chunks sharing recurrent history, then materialize the selected terminal view.
    pub(crate) fn run<'e>(&'e self, scope: &'e SubmissionScope<'_>,
        peer: &'e PeerExchange<'a>, hops: &'e HopLink<'a>,
        locals: &'e mut [[LocalLane<'a>; 2]], routed: &'e mut [RoutedLane<'a>],
        experts: &'e mut [[Option<Exl3Tp2<'a>>; 2]], rows: &[usize], speculative: bool, output: Output, draft: Option<&'e super::draft::NativeDraft<'a>>, probe: Option<&'e DraftProbe<'a>>) -> Result<Vec<CuteafdDeviceBuffer>> {
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

    /// One sequence's prefill as row units on the single admitted lane and
    /// Spark link (PLAN "Prefill contract"): up to `waves` units' expert waves
    /// in flight, each unit owning its rows of every lane operand.
    pub(crate) fn run_units(&self, scope: &SubmissionScope<'_>, peer: &PeerExchange<'a>,
        hops: &HopLink<'a>, local: &mut [LocalLane<'a>; 2], routed: &mut RoutedLane<'a>,
        experts: &mut [Option<Exl3Tp2<'a>>; 2], units: &[crate::shared::prefill_pipeline::Unit], waves: usize,
        sequence: &super::device::SequenceRows<'_>, pool: &cuteafd_engine::prefix::RefPagePool,
        draft: Option<&super::draft::NativeDraft<'a>>) -> Result<CuteafdDeviceBuffer> {
        let protocol = self.protocol(peer, hops)?;
        tracing::info!(start = sequence.start, rows = sequence.rows, units = units.len(), waves,
            sizes = ?units.iter().map(|unit| unit.rows.len()).collect::<Vec<_>>(),
            "Kimi prefill units");
        // Spark waves build, post and land on the routed link's transport
        // worker while this thread queues GPU work.
        let spark = routed.has_link();
        let mut run = |routed: &mut RoutedLane<'a>, waves_stage: Option<&mut crate::shared::spark_waves::Waves<'_, 'a>>|
            -> Result<CuteafdDeviceBuffer> {
            let mut binding = Bindings::new(local, routed, experts, self, peer, hops, scope, 0,
                units[0].rows.len(), false)?;
            binding.draft = draft;
            binding.defer_publication = true;
            binding.waves = waves_stage;
            let mut hooks = UnitHooks { binding, protocol: &protocol, sequence, pool,
                state: units.iter().map(|_| UnitState::default()).collect(), consumed:
                    vec![vec![0; self.plan.hops().len()]; units.len()] };
            crate::shared::prefill_pipeline::run_prefill_units(&mut hooks, units, waves)?;
            let UnitHooks { binding, consumed, .. } = hooks;
            ensure!(consumed.iter().flatten().all(|&count| count == 1), "Kimi unit hop inventory");
            ensure!(binding.worker_tickets.is_empty(), "Kimi prefill ended with worker waves outstanding");
            for local in binding.local.iter_mut() { local.select_subwave(None)?; }
            binding.local[1].buffer("normalized")
        };
        if spark {
            routed.with_waves_mut(|routed, stage| run(routed, Some(stage)))
        } else {
            run(routed, None)
        }
    }

    /// Accepted replay writes every KDA layer on this lane. Publish those exact
    /// streams into the same dependency chain consumed by the next attention.
    pub(crate) fn commit_kda(&self, locals: &[LocalLane<'a>; 2], tables: &[i32]) -> Result<()> {
        if tables.is_empty() { return Ok(()); }
        let lane = locals[0].lane_index();
        let bound = self.lanes.get(lane).context("Kimi commit lane outside admission")?;
        ensure!(locals[1].lane_index() == lane && (0..2).all(|rank|
            Rc::ptr_eq(locals[rank].stream(), &bound.streams[rank])),
            "Kimi commit streams differ from replay lane");
        for local in locals { local.commit_kda(tables)?; }
        for (layer, kind) in locals[0].config().attention.iter().enumerate() {
            if *kind != cuteafd_loader::families::kimi_k3::config::KimiAttention::Kda { continue; }
            let mut ready = self.state_ready[layer].borrow_mut();
            for rank in 0..2 {
                ready[rank].record(locals[rank].stream())?;
                self.state_pending[layer][rank].set(true);
            }
        }
        Ok(())
    }

    /// Validate every borrowed link before entering the shared executor. No
    /// device guard spans the submission or an asynchronous yield.
    pub(crate) fn protocol<'e>(&'e self, peer: &'e PeerExchange<'a>, hops: &'e HopLink<'a>)
        -> Result<Protocol<'e, 'a>> {
        hops.validate_binding(&self.plan, peer)?;
        let handles = self.lanes.iter().map(|lane| lane.streams.each_ref().map(|stream|
            (stream.device.id, stream.raw))).collect::<Vec<_>>();
        validate_streams(&handles, [peer.device(0), peer.device(1)])?;
        ensure!(self.entry_lane.get().is_none(), "Kimi entry publication was not consumed");
        Ok(Protocol { execution: self, peer, hops })
    }
}

pub(crate) struct Protocol<'e, 'a> {
    execution: &'e Execution<'a>,
    peer: &'e PeerExchange<'a>,
    hops: &'e HopLink<'a>,
}

impl Protocol<'_, '_> {
    /// A source exchange stream joins both exact lane streams. Joining the
    /// destination's previous consumer also prevents receive-slot overwrite
    /// when a later hop reuses it. Cross-device event waits are intentional.
    fn producers(&self, lane: usize, exchange_rank: usize) -> Result<()> {
        let joins = self.execution.lanes.get(lane).context("Kimi hop lane outside admission")?;
        let mut events = joins.produced.borrow_mut();
        let raw = events.each_ref().map(|event| event.raw);
        let producers = producer_streams(self.execution.lanes.iter().map(|lane|
            lane.streams.each_ref().map(|stream| stream.raw)), lane)?;
        join_producers(producers, |rank, producer| {
            ensure!(producer == joins.streams[rank].raw, "Kimi hop producer stream changed");
            events[rank].record(&joins.streams[rank])
        }, |rank| self.peer.on(exchange_rank, || {
            // SAFETY: retained events were recorded on this exact lane's
            // producers; the exchange stream belongs to its validated rank.
            unsafe { joins.streams[exchange_rank].device.library.cuda_stream_wait_event(
                self.peer.stream(exchange_rank), raw[rank]) }
        }))
    }

    fn consumer(&self, lane: usize, rank: usize) -> Result<()> {
        let joins = self.execution.lanes.get(lane).context("Kimi hop lane outside admission")?;
        let event = &mut joins.landed.borrow_mut()[rank];
        self.peer.on(rank, || {
            // SAFETY: both the exchange stream and retained event belong to rank.
            unsafe { joins.streams[rank].device.library.cuda_event_record(
                event.raw, self.peer.stream(rank)) }
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

    fn require_live(&self) -> Result<()> { self.peer.require_live() }

    fn push_entry(&self, from: usize, source: *const c_void, destination: *mut c_void, bytes: usize) -> Result<()> {
        let lane = self.execution.entry_lane.get().context("Kimi entry source lane was not selected")?;
        self.producers(lane, from)?;
        self.peer.push_to(from, crate::shared::peer_split::DIRECT, source, destination, bytes)?;
        self.consumer(lane, from)
    }

    fn wait_entry(&self, to: usize) -> Result<()> {
        let lane = self.execution.entry_lane.get().context("Kimi entry publication has no lane")?;
        self.peer.wait(to, crate::shared::peer_split::DIRECT)?;
        self.consumer(lane, to)?;
        self.execution.entry_lane.set(None);
        Ok(())
    }

    fn send(&self, index: usize, lane: usize, source: *const c_void, rows: usize) -> Result<Ticket> {
        let hop = self.execution.plan.hops().get(index).context("Kimi hop outside plan")?;
        self.producers(lane, usize::from(hop.from))?;
        let ticket = Ticket { hop: self.hops.send(index, lane, source, rows)?, lane, to: usize::from(hop.to) };
        // The source delta may be overwritten by the FFN only after its copy.
        self.consumer(lane, usize::from(hop.from))?;
        Ok(ticket)
    }

    fn land(&self, ticket: Ticket) -> Result<*mut c_void> {
        let residual = self.hops.land(ticket.hop)?;
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

fn producer_streams(streams: impl Iterator<Item = [*mut c_void; 2]>, lane: usize)
    -> Result<[*mut c_void; 2]> {
    streams.into_iter().nth(lane).context("Kimi producer lane outside admission")
}

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
pub(crate) struct Bindings<'e, 'a, 's, 'w, 'x, 'l> {
    scope: &'e SubmissionScope<'s>,
    local: &'e mut [LocalLane<'a>; 2],
    routed: &'e mut RoutedLane<'a>,
    /// A prefill call's transport worker on the routed lane's link: Spark
    /// waves send and land through it instead of inline.
    waves: Option<&'w mut crate::shared::spark_waves::Waves<'x, 'l>>,
    experts: &'e mut [Option<Exl3Tp2<'a>>; 2],
    protocol: Protocol<'e, 'a>,
    lane: usize,
    rows: usize,
    speculative: bool,
    owners: Vec<(Vec<u8>, Vec<u8>)>,
    completed: usize,
    pending: Option<FfnTicket>,
    draft: Option<&'e super::draft::NativeDraft<'a>>,
    probe: Option<&'e DraftProbe<'a>>,
    tap_first: usize,
    defer_publication: bool,
    /// Worker tickets in dispatch order (each FFN ticket names its sequence).
    worker_tickets: std::collections::VecDeque<(u64, crate::shared::spark_waves::WaveTicket)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct FfnTicket { layer: usize, spark: Option<Spark>, rtx: Option<usize> }

/// A Spark wave of one FFN: inline (decode, verify) or on the prefill worker.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Spark { Inline(super::spark::Ticket), Worker { sequence: u64 } }

impl<'e, 'a, 's, 'w, 'x, 'l> Bindings<'e, 'a, 's, 'w, 'x, 'l> {
    pub(crate) fn new(local: &'e mut [LocalLane<'a>; 2], routed: &'e mut RoutedLane<'a>,
        experts: &'e mut [Option<Exl3Tp2<'a>>; 2], execution: &'e Execution<'a>, peer: &'e PeerExchange<'a>, hops: &'e HopLink<'a>,
        scope: &'e SubmissionScope<'s>, lane: usize, rows: usize, speculative: bool) -> Result<Self> {
        let protocol = execution.protocol(peer, hops)?;
        let joins = execution.lanes.get(lane).context("Kimi binding lane outside admission")?;
        ensure!((1..=execution.plan.spec().rows as usize).contains(&rows)
            && execution.plan.spec().entry_gpu == 0 && execution.plan.spec().head_gpu == 1
            && execution.plan.spec().terminal == cuteafd_loader::placement::TerminalHome::Replicated,
            "Kimi binding row shape or terminal geometry");
        ensure!(execution.plan.modes().len() == local[0].config().layers
            && execution.plan.modes().iter().all(|mode| matches!(mode,
                LayerMode::HeadSplit | LayerMode::Whole { ffn: FfnMode::Split, .. })),
            "Kimi layer binding requires head-split or owned attention with split FFN");
        for rank in 0..2 {
            ensure!(Rc::ptr_eq(local[rank].stream(), &joins.streams[rank])
                && local[rank].lane_index() == lane && rows <= local[rank].rows(),
                "Kimi layer binding differs from admitted lane owner");
        }
        let mut owners = Vec::new();
        for layer in 0..execution.plan.modes().len() {
            let mut resident = (Vec::new(), Vec::new());
            for rank in 0..2 {
                let (weights, state) = local[rank].attention_resident(layer)?;
                if weights { resident.0.push(rank as u8); }
                if state { resident.1.push(rank as u8); }
            }
            execution.plan.validate_state_owners(layer, &resident.0)?;
            execution.plan.validate_state_owners(layer, &resident.1)?;
            owners.push(resident);
        }
        for rank in 0..2 { peer.recv(rank, lane)?; }
        ensure!(peer.slot_bytes() >= rows * 7168 * 2, "Kimi peer slot shorter than hidden rows");
        Ok(Self { scope, local, routed, waves: None, experts, protocol, lane, rows, speculative, owners, completed: 0, pending: None, draft: None, probe: None, tap_first: 0, defer_publication: false, worker_tickets: std::collections::VecDeque::new() })
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
            local.residual(MixSite::Attention(layer), self.rows)?;
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
        self.local[rank].attention(layer, self.rows, self.speculative)?;
        ready[rank].record(&stream)?;
        execution.state_pending[layer][rank].set(true);
        Ok(())
    }

    fn publish(&self, source: &str) -> Result<()> {
        // Both publications precede either wait. Every event names this lane,
        // never an assumed rank-global compute stream.
        for rank in 0..2 {
            self.protocol.producers(self.lane, rank)?;
            self.protocol.peer.push(rank, self.lane, self.local[rank].buffer(source)?.ptr,
                self.rows * 7168 * 2)?;
        }
        Ok(())
    }

    fn meet(&mut self, source: &str, destination: &str) -> Result<()> {
        for rank in 0..2 {
            self.protocol.peer.wait(rank, self.lane)?;
            self.protocol.peer.add(rank, self.local[rank].buffer(source)?.ptr,
                self.protocol.peer.recv(rank, self.lane)?, self.local[rank].buffer(destination)?.ptr,
                self.rows * 7168)?;
            self.protocol.consumer(self.lane, rank)?;
        }
        Ok(())
    }

    fn publish_core(&self,ffn:bool)->Result<()> {
        let name=if ffn {"K6 FFN partial"}else{"K6 attention partial"};
        ensure!(self.protocol.peer.slot_bytes()>=self.rows*7168*4,"Kimi K6 peer slot shorter than F32 hidden rows");
        for rank in 0..2 {
            self.protocol.producers(self.lane,rank)?;
            self.protocol.peer.push(rank,self.lane,self.local[rank].buffer(name)?.ptr,self.rows*7168*4)?;
        }
        Ok(())
    }
    /// Native K6 FFN partials cross the peer link as F32; BF16 otherwise.
    fn publish_ffn(&self, layer: usize) -> Result<()> {
        if self.local[0].native_ffn(layer)? {
            ensure!(self.local[1].native_ffn(layer)?, "Kimi K6 FFN rank formats differ");
            self.publish_core(true)
        } else {
            self.publish("ffn output")
        }
    }
    fn meet_core(&mut self,layer:usize,ffn:bool,destination:&str)->Result<()> {
        for rank in 0..2 {
            self.protocol.peer.wait(rank,self.lane)?;
            let peer=CuteafdDeviceBuffer{ptr:self.protocol.peer.recv(rank,self.lane)?,
                bytes:self.rows*7168*4,device_id:self.local[rank].stream().device.id,flags:0};
            self.protocol.peer.on(rank,||self.local[rank].reduce_core(layer,ffn,self.rows,peer,destination,
                self.protocol.peer.stream(rank)))?;
            self.protocol.consumer(self.lane,rank)?;
        }
        Ok(())
    }

    /// Copies into retained lane operands, not a second exchange ring. Both
    /// producer joins include previous readers before any destination overwrite.
    fn transfer(&self, from: usize, name: &str, bytes: usize) -> Result<()> {
        let to = 1 - from;
        let source = self.local[from].buffer(name)?;
        let destination = self.local[to].buffer(name)?;
        ensure!(bytes <= source.bytes && bytes <= destination.bytes, "Kimi canonical transfer extent");
        self.protocol.producers(self.lane, from)?;
        self.protocol.peer.push_to(from, crate::shared::peer_split::DIRECT,
            source.ptr, destination.ptr, bytes)?;
        self.protocol.consumer(self.lane, from)?;
        self.protocol.peer.wait(to, crate::shared::peer_split::DIRECT)?;
        self.protocol.consumer(self.lane, to)
    }

    /// Called after the shared pipeline, only on the head rank that owns the
    /// output score/norm weights. The other rank's stored history remains intact.
    pub(crate) fn output(&mut self, output: Output) -> Result<CuteafdDeviceBuffer> {
        ensure!(self.pending.is_none() && self.completed == self.protocol.execution.plan.modes().len(),
            "Kimi output requested before admitted layer completion");
        let mut probed = false;
        self.with_probe(|probe| { probed = probe.active(); Ok(()) })?;
        if probed {
            // The final GPU0 ring download must precede GPU1's existing selector sync.
            // Join the exact lane producers into the retained head exchange/consumer.
            self.protocol.producers(self.lane, 1)?;
            self.protocol.consumer(self.lane, 1)?;
        }
        match output {
            Output::Normalized => self.local[1].residual(MixSite::Output, self.rows),
            Output::Residual => self.local[1].terminal_residual(self.rows),
        }
    }
}

impl LayerKernels for Bindings<'_, '_, '_, '_, '_, '_> {
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
            self.publish_core(false)?;self.meet_core(layer,false,"delta")
        } else {
            self.publish("attention output")?;
            self.meet("attention output", "delta")
        }
    }
    fn context(&mut self, _: usize, _: usize) -> Result<()> {
        anyhow::bail!("Kimi context split has no query/LSE exchange ABI")
    }
    fn owned(&mut self, layer: usize, gpu: usize) -> Result<()> {
        ensure!(gpu < 2, "Kimi attention owner rank");
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

impl PipelineKernels for Bindings<'_, '_, '_, '_, '_, '_> {
    type FfnTicket = FfnTicket;
    fn state_owners(&self, layer: usize) -> Result<AttentionStateOwners<'_>> {
        let owners = self.owners.get(layer).context("Kimi layer census outside inventory")?;
        Ok(AttentionStateOwners { weights: &owners.0, kv: &owners.1, indexer: None, compressor: None })
    }
    fn begin_ffn(&mut self, layer: usize, owner: Option<usize>) -> Result<FfnTicket> {
        ensure!(owner.is_none() && self.pending.is_none(), "Kimi FFN owner or unfinished wave");
        for local in self.local.iter_mut() { local.residual(MixSite::Ffn(layer), self.rows)?; }
        let dense = self.local[0].config().dense[layer];
        let owner = if dense { 0 } else { super::weights::routed_owner(self.local[0].config(), layer)? };
        let rtx = !dense && self.experts.iter().any(|expert|
            expert.as_ref().is_some_and(|e| e.layers().contains(&layer)));
        let spark = if dense {
            for local in self.local.iter_mut() { local.ffn(layer, self.rows)?; }
            None
        } else {
            let (canonical, wire) = self.local[owner].route(layer, self.rows)?;
            if rtx || owner == 0 {
                for (name, bytes) in [("route ids", canonical.ids.bytes),
                    ("route weights", canonical.weights.bytes), ("expert wire", wire.bytes)] {
                    self.transfer(owner, name, bytes)?;
                }
            }
            // The optional shared ring check also preserves remote rank1 routes on GPU0.
            let mut check_remote = false;
            self.with_probe(|probe| { check_remote = probe.wants_ring(true); Ok(()) })?;
            if check_remote && !rtx && owner == 1 { self.transfer(1, "route ids", canonical.ids.bytes)?; }
            // Local IDs have joined rank0 before the reusable router outputs change.
            self.with_probe(|probe| {
                if probe.wants_ring(!rtx) {
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
                    // SAFETY: canonical input publication joins this exact lane;
                    // weights and scratch survive replay and terminal teardown.
                    unsafe { expert.enqueue(layer, self.rows,
                        ExpertInput::Fp8K32(self.local[rank].buffer("expert wire")?.ptr),
                        Routes { ids: self.local[rank].buffer("route ids")?.ptr,
                            weights: self.local[rank].buffer("route weights")?.ptr }, stream.raw,
                        &mut || crate::shared::decode_graph::fatal_drain(self.protocol.peer.publish_abort(),
                            "Kimi TP2 expert abort"))?; }
                }
                let experts = self.experts.each_ref().map(|expert| expert.as_ref().unwrap());
                ensure!(experts[0].partial() == experts[1].partial()
                    && experts[0].partial_layout(self.rows)? == experts[1].partial_layout(self.rows)?,
                    "Kimi TP2 partial contracts differ");
                let partials = std::array::from_fn(|rank| CuteafdDeviceBuffer {
                    ptr: experts[rank].output(), bytes: self.rows * 3584 * 4,
                    device_id: experts[rank].device(), flags: 0 });
                // SAFETY: both final FP32 token partials are on the retained
                // producer streams; reduction completes before hidden-slot reuse.
                unsafe { self.protocol.execution.lanes[self.lane].reducer.borrow_mut().reduce(
                    self.scope, self.protocol.peer, owner, self.rows, partials,
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
                let spark = match self.waves.as_deref_mut() {
                    Some(waves) => {
                        let ticket = self.routed.begin_wave(waves, &mut self.local[1], layer, self.rows, kind, routes, wire)?;
                        let sequence = self.worker_tickets.back().map_or(0, |(s, _)| s + 1);
                        self.worker_tickets.push_back((sequence, ticket));
                        Spark::Worker { sequence }
                    }
                    None => {
                        let mut probe = self.probe.map(|probe| probe.borrow_mut());
                        Spark::Inline(self.routed.begin(&mut self.local[1], layer, self.rows, kind, routes, wire,
                            probe.as_mut().and_then(|p| p.as_mut()).map(|p| p.get_mut()))?)
                    }
                };
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
        if self.defer_publication { self.publish_ffn(layer)?; }
        if ticket.spark.is_some() || ticket.rtx.is_some() {
            if self.local[0].native_ffn(layer)? {self.meet_core(layer,true,"shared output")?;}
            else {self.meet("ffn output", "shared output")?;}
            let rank = super::weights::routed_owner(self.local[0].config(), layer)?;
            if let Some(spark) = ticket.spark {
                match spark {
                    Spark::Inline(spark) => self.routed.finish(&self.local[1], spark)?,
                    Spark::Worker { sequence } => {
                        let (front, wave) = self.worker_tickets.pop_front().context("Kimi worker wave missing")?;
                        ensure!(front == sequence, "Kimi worker waves land out of dispatch order");
                        let waves = self.waves.as_deref_mut().context("Kimi worker wave without its transport")?;
                        let slot = wave.slot();
                        self.routed.finish_wave(waves, &self.local[1], wave)?;
                        let mut probe = self.probe.map(|probe| probe.borrow_mut());
                        if let Some(probe) = probe.as_mut().and_then(|p| p.as_mut()).map(|p| p.get_mut()) {
                            probe.staged(layer, waves.staged_ids(slot)?);
                        }
                    }
                }
                if rank == 0 { self.transfer(1, "expert combined", self.rows * 3584 * 2)?; }
            }
            let output = self.local[rank].latent_up(layer, self.rows)?;
            let other = 1 - rank;
            self.protocol.producers(self.lane, rank)?;
            self.protocol.peer.push_to(rank, crate::shared::peer_split::DIRECT, output.ptr,
                self.local[other].buffer("routed output")?.ptr, self.rows * 7168 * 2)?;
            self.protocol.consumer(self.lane, rank)?;
            self.protocol.peer.wait(other, crate::shared::peer_split::DIRECT)?;
            self.protocol.consumer(self.lane, other)?;
            for local in self.local.iter_mut() {
                let combined = local.combine_moe(self.rows)?;
                local.adopt("delta", combined, self.rows, 7168)?;
            }
        } else {
            if self.local[0].native_ffn(layer)? {self.meet_core(layer,true,"delta")?;}
            else {self.meet("ffn output", "delta")?;}
        }
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
/// and the phase arena are restaged or rewritten inside each hook.
struct UnitHooks<'h, 'e, 'a, 's, 'p, 'w, 'x, 'l> {
    binding: Bindings<'e, 'a, 's, 'w, 'x, 'l>,
    protocol: &'h Protocol<'e, 'a>,
    sequence: &'h super::device::SequenceRows<'p>,
    pool: &'h cuteafd_engine::prefix::RefPagePool,
    state: Vec<UnitState>,
    consumed: Vec<Vec<usize>>,
}

impl UnitHooks<'_, '_, '_, '_, '_, '_, '_, '_> {
    fn select(&mut self, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
        for local in self.binding.local.iter_mut() { local.select_subwave(Some(unit.rows.clone()))?; }
        self.binding.rows = unit.rows.len();
        self.binding.completed = self.state[unit.index].completed;
        self.binding.tap_first = unit.rows.start;
        Ok(())
    }
}

impl crate::shared::prefill_pipeline::PrefillHooks for UnitHooks<'_, '_, '_, '_, '_, '_, '_, '_> {
    type Ticket = FfnTicket;
    fn route(&self, layer: usize) -> crate::shared::prefill_pipeline::FfnRoute {
        // Dense and RTX-resident layers complete inside one hook; the Spark
        // ones hold a wave across hooks.
        let cfg = self.binding.local[0].config();
        let rtx = self.binding.experts.iter().any(|expert| expert.as_ref().is_some_and(|e| e.layers().contains(&layer)));
        if cfg.dense[layer] || rtx { crate::shared::prefill_pipeline::FfnRoute::Local }
        else { crate::shared::prefill_pipeline::FfnRoute::Spark }
    }
    fn begin_unit(&mut self, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
        ensure!(unit.layers.start == 0 && unit.sequence == 0, "Kimi units run every layer of one sequence");
        Ok(())
    }
    fn attention(&mut self, layer: usize, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
        self.select(unit)?;
        self.binding.pending = None;
        let metadata = [super::device::SequenceRows { slot: self.sequence.slot,
            start: self.sequence.start + unit.rows.start, rows: unit.rows.len(), pages: self.sequence.pages }];
        for local in self.binding.local.iter_mut() { local.stage_attention(&metadata, self.pool)?; }
        crate::shared::peer_split::attention::pipeline_attention(self.protocol, &self.binding.protocol.execution.plan,
            layer, 0, unit.rows.len(), &mut self.binding, &mut self.consumed[unit.index])?;
        Ok(())
    }
    fn local_ffn(&mut self, layer: usize, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
        self.select(unit)?;
        self.binding.pending = None;
        let ticket = self.binding.begin_ffn(layer, None)?;
        self.binding.finish_ffn(layer, ticket)?;
        self.state[unit.index].completed += 1;
        Ok(())
    }
    fn send_experts(&mut self, layer: usize, unit: &crate::shared::prefill_pipeline::Unit) -> Result<FfnTicket> {
        self.select(unit)?;
        self.binding.pending = None;
        let ticket = self.binding.begin_ffn(layer, None)?;
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
