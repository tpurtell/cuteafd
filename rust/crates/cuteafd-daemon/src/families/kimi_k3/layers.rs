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
use std::{cell::{Cell, RefCell}, ffi::c_void, rc::Rc};

struct LaneJoins<'a> {
    streams: [Rc<Stream<'a>>; 2],
    produced: RefCell<[Event<'a>; 2]>,
    landed: RefCell<[Event<'a>; 2]>,
}

#[derive(Clone, Copy)]
pub(crate) enum Output { Normalized, Residual }

/// Own these bindings for the engine lifetime, not in a temporary submission:
/// recorded event nodes, streams and all exchanged operands must survive replay.
/// The enclosing submission owner aborts all links before draining these streams.
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
        streams: Vec<[Rc<Stream<'a>>; 2]>) -> Result<Self> {
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
        let lanes = streams.into_iter().map(|streams| {
            let produced = [Event::new(streams[0].device)?, Event::new(streams[1].device)?];
            let landed = [Event::new(streams[0].device)?, Event::new(streams[1].device)?];
            Ok(LaneJoins { streams, produced: RefCell::new(produced), landed: RefCell::new(landed) })
        }).collect::<Result<_>>()?;
        Ok(Self { plan, lanes, entry_lane: Cell::new(None), state_ready, state_pending })
    }

    /// The family has no second scheduler: use the shared layer-major pipeline
    /// for chunks sharing recurrent history, then materialize the selected terminal view.
    pub(crate) fn run<'e>(&'e self, scope: &crate::shared::experts::rtx::pair::SubmissionScope<'_>,
        peer: &'e PeerExchange<'a>, hops: &'e HopLink<'a>,
        locals: &'e mut [[LocalLane<'a>; 2]], routed: &'e mut [RoutedLane<'a>],
        rows: &[usize], speculative: bool, output: Output) -> Result<Vec<CuteafdDeviceBuffer>> {
        ensure!(!rows.is_empty() && rows.len() == locals.len() && rows.len() == routed.len(),
            "Kimi shared pipeline needs matching local, Spark and row lanes");
        let protocol = self.protocol(peer, hops)?;
        let mut bindings = locals.iter_mut().zip(routed).zip(rows).enumerate().map(
            |(lane, ((local, routed), &rows))| Bindings::new(local, routed, self, peer, hops, lane, rows, speculative)
        ).collect::<Result<Vec<_>>>()?;
        crate::shared::peer_split::attention::layer_pipeline_with(scope, &self.plan, &protocol,
            crate::shared::peer_split::attention::LaneOrder::OrderedChunks, rows, &mut bindings)?;
        bindings.iter_mut().map(|binding| binding.output(output)).collect()
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
pub(crate) struct Bindings<'e, 'a> {
    local: &'e mut [LocalLane<'a>; 2],
    routed: &'e mut RoutedLane<'a>,
    protocol: Protocol<'e, 'a>,
    lane: usize,
    rows: usize,
    speculative: bool,
    owners: Vec<(Vec<u8>, Vec<u8>)>,
    completed: usize,
    pending: Option<FfnTicket>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct FfnTicket { layer: usize, spark: Option<super::spark::Ticket> }

impl<'e, 'a> Bindings<'e, 'a> {
    pub(crate) fn new(local: &'e mut [LocalLane<'a>; 2], routed: &'e mut RoutedLane<'a>,
        execution: &'e Execution<'a>, peer: &'e PeerExchange<'a>, hops: &'e HopLink<'a>,
        lane: usize, rows: usize, speculative: bool) -> Result<Self> {
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
        Ok(Self { local, routed, protocol, lane, rows, speculative, owners, completed: 0, pending: None })
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

    /// Called after the shared pipeline, only on the head rank that owns the
    /// output score/norm weights. The other rank's stored history remains intact.
    pub(crate) fn output(&mut self, output: Output) -> Result<CuteafdDeviceBuffer> {
        ensure!(self.pending.is_none() && self.completed == self.protocol.execution.plan.modes().len(),
            "Kimi output requested before admitted layer completion");
        match output {
            Output::Normalized => self.local[1].residual(MixSite::Output, self.rows),
            Output::Residual => self.local[1].terminal_residual(self.rows),
        }
    }
}

impl LayerKernels for Bindings<'_, '_> {
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
        self.publish("attention output")?;
        self.meet("attention output", "delta")
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

impl PipelineKernels for Bindings<'_, '_> {
    type FfnTicket = FfnTicket;
    fn state_owners(&self, layer: usize) -> Result<AttentionStateOwners<'_>> {
        let owners = self.owners.get(layer).context("Kimi layer census outside inventory")?;
        Ok(AttentionStateOwners { weights: &owners.0, kv: &owners.1, indexer: None, compressor: None })
    }
    fn begin_ffn(&mut self, layer: usize, owner: Option<usize>) -> Result<FfnTicket> {
        ensure!(owner.is_none() && self.pending.is_none(), "Kimi FFN owner or unfinished wave");
        for local in self.local.iter_mut() { local.residual(MixSite::Ffn(layer), self.rows)?; }
        let spark = if self.local[0].config().dense[layer] {
            for local in self.local.iter_mut() { local.ffn(layer, self.rows)?; }
            None
        } else {
            let kind = if self.speculative { cuteafd_transport::ExpertV2SourceKind::MtpVerify }
                else if self.rows > 64 { cuteafd_transport::ExpertV2SourceKind::Prefill }
                else { cuteafd_transport::ExpertV2SourceKind::Decode };
            let ticket = self.routed.begin(&mut self.local[0], layer, self.rows, kind)?;
            self.local[1].ffn(layer, self.rows)?;
            Some(ticket)
        };
        self.publish("ffn output")?;
        let ticket = FfnTicket { layer, spark };
        self.pending = Some(ticket);
        Ok(ticket)
    }
    fn finish_ffn(&mut self, layer: usize, ticket: FfnTicket) -> Result<()> {
        ensure!(self.pending == Some(ticket) && ticket.layer == layer, "Kimi FFN ticket differs from pending layer");
        if let Some(spark) = ticket.spark {
            self.meet("ffn output", "shared output")?;
            self.routed.finish(&self.local[0], spark)?;
            let output = self.local[0].latent_up(layer, self.rows)?;
            self.protocol.producers(self.lane, 0)?;
            self.protocol.peer.push_to(0, crate::shared::peer_split::DIRECT, output.ptr,
                self.local[1].buffer("routed output")?.ptr, self.rows * 7168 * 2)?;
            self.protocol.consumer(self.lane, 0)?;
            self.protocol.peer.wait(1, crate::shared::peer_split::DIRECT)?;
            self.protocol.consumer(self.lane, 1)?;
            for local in self.local.iter_mut() {
                let combined = local.combine_moe(self.rows)?;
                local.adopt("delta", combined, self.rows, 7168)?;
            }
        } else {
            self.meet("ffn output", "delta")?;
        }
        self.pending = None;
        self.completed += 1;
        Ok(())
    }
}
