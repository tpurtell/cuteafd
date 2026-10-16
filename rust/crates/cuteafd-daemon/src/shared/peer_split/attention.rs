//! Shared per-layer attention orchestration. Families provide kernel bindings,
//! not their own ordering of peer pushes, waits or LSE operands.
use super::context::{combine_order, slot, ContextExchange};
use super::{hop::HopLink, PeerExchange, DIRECT};
use anyhow::{ensure, Result};
use cuteafd_loader::placement::{FfnMode, Hop, HopPoint, LayerExecutionPlan, LayerMode, Placement};
use std::ffi::c_void;

/// Replicate the producer's already-rounded residual into existing peer storage.
/// Call outside graph capture, after all producers needed by the receiving stream
/// have been queued. Stream order puts the copy after mHC; the landing is device-only.
pub(crate) fn replicate_terminal(exchange: &PeerExchange<'_>, source: cuteafd_ffi::CuteafdDeviceBuffer,
    destination: cuteafd_ffi::CuteafdDeviceBuffer, bytes: usize) -> Result<()> {
    replicate_terminal_with(exchange, source, destination, bytes)
}

trait TerminalProtocol {
    fn device(&self, rank: usize) -> i32;
    fn publish(&self, source: *const c_void, destination: *mut c_void, bytes: usize) -> Result<()>;
    fn land(&self) -> Result<()>;
}

impl TerminalProtocol for PeerExchange<'_> {
    fn device(&self, rank: usize) -> i32 { self.device(rank) }
    fn publish(&self, source: *const c_void, destination: *mut c_void, bytes: usize) -> Result<()> {
        self.push_to(0, DIRECT, source, destination, bytes)
    }
    fn land(&self) -> Result<()> { self.wait(1, DIRECT) }
}

fn replicate_terminal_with(exchange: &impl TerminalProtocol, source: cuteafd_ffi::CuteafdDeviceBuffer,
    destination: cuteafd_ffi::CuteafdDeviceBuffer, bytes: usize) -> Result<()> {
    ensure!(bytes > 0 && bytes <= source.bytes && bytes <= destination.bytes,
        "terminal residual copy exceeds live source or destination storage");
    ensure!(!source.ptr.is_null() && !destination.ptr.is_null()
        && source.device_id == exchange.device(0) && destination.device_id == exchange.device(1)
        && source.device_id != destination.device_id, "terminal residual copy has invalid rank storage");
    exchange.publish(source.ptr, destination.ptr, bytes)?;
    exchange.land()
}

#[cfg(test)]
mod terminal_tests {
    use super::*;
    use std::cell::RefCell;
    use cuteafd_ffi::CuteafdDeviceBuffer;

    struct CopyProtocol {
        source: Vec<u8>,
        destination: RefCell<Vec<u8>>,
        calls: RefCell<Vec<&'static str>>,
        fail_publish: bool,
    }

    impl TerminalProtocol for CopyProtocol {
        fn device(&self, rank: usize) -> i32 { [3, 7][rank] }
        fn publish(&self, source: *const c_void, destination: *mut c_void, bytes: usize) -> Result<()> {
            assert_eq!((source as usize, destination as usize), (1, 2));
            self.calls.borrow_mut().push("publish");
            ensure!(!self.fail_publish, "injected terminal publication failure");
            self.destination.borrow_mut()[..bytes].copy_from_slice(&self.source[..bytes]);
            Ok(())
        }
        fn land(&self) -> Result<()> {
            self.calls.borrow_mut().push("land");
            Ok(())
        }
    }

    fn buffers(bytes: usize) -> [CuteafdDeviceBuffer; 2] {
        [CuteafdDeviceBuffer { ptr: 1usize as *mut c_void, bytes, device_id: 3, flags: 0 },
            CuteafdDeviceBuffer { ptr: 2usize as *mut c_void, bytes, device_id: 7, flags: 0 }]
    }

    #[test]
    fn terminal_replication_preserves_every_bf16_bit_pattern_without_rerounding() {
        // Signed zeros, infinities, NaN payloads and all finite encodings survive.
        let source: Vec<u8> = (0..=u16::MAX).flat_map(u16::to_le_bytes).collect();
        let bytes = source.len();
        let exchange = CopyProtocol { source, destination: RefCell::new(vec![0xa5; bytes + 16]),
            calls: RefCell::new(Vec::new()), fail_publish: false };
        let [source, destination] = buffers(bytes);
        replicate_terminal_with(&exchange, source, destination, bytes).unwrap();
        assert_eq!(&exchange.destination.borrow()[..bytes], exchange.source.as_slice());
        assert_eq!(&exchange.destination.borrow()[bytes..], &[0xa5; 16]);
        assert_eq!(*exchange.calls.borrow(), ["publish", "land"]);
    }

    #[test]
    fn invalid_terminal_storage_and_failed_publish_never_queue_a_wait() {
        let mut exchange = CopyProtocol { source: vec![1; 8], destination: RefCell::new(vec![0; 8]),
            calls: RefCell::new(Vec::new()), fail_publish: false };
        let [source, destination] = buffers(8);
        for bytes in [0, 9] {
            assert!(replicate_terminal_with(&exchange, source, destination, bytes).is_err());
        }
        for invalid in [CuteafdDeviceBuffer { bytes: 7, ..destination },
            CuteafdDeviceBuffer { device_id: 3, ..destination },
            CuteafdDeviceBuffer { ptr: std::ptr::null_mut(), ..destination }] {
            assert!(replicate_terminal_with(&exchange, source, invalid, 8).is_err());
        }
        assert!(exchange.calls.borrow().is_empty());
        exchange.fail_publish = true;
        assert!(replicate_terminal_with(&exchange, source, destination, 8).is_err());
        assert_eq!(*exchange.calls.borrow(), ["publish"]);
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Query {
    pub values: *const c_void,
    pub row_pitch: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct Candidates {
    pub scores: *const c_void,
    pub indices: *const c_void,
}

#[derive(Clone, Copy)]
pub(crate) struct Partial {
    pub values: *mut c_void,
    pub lse: *mut c_void,
}

/// Bindings run on the selected rank's stream without synchronizing the host.
/// `produce` writes only owned page rows (or scratch) and this rank's queries;
/// `score` returns None on layers reusing an indexer's local selection. Merge
/// keeps this rank's picks; partials are normalized with base-2 LSE, no sink.
/// Empty selections must produce zero values and negative-infinite LSE.
pub(crate) trait ContextKernels {
    fn produce(&mut self) -> Result<Query>;
    fn score(&mut self) -> Result<Option<Candidates>>;
    fn merge(&mut self, scores: *mut c_void, indices: *mut c_void) -> Result<()>;
    fn partial(&mut self, query: *mut c_void, heads_rank: usize) -> Result<Partial>;
    /// Operands are always GPU0 then GPU1; add any sink exactly once here.
    fn combine(&mut self, partials: [Partial; 2]) -> Result<()>;
}

/// The exchange protocol, separated from CUDA so CPU tests exercise precisely
/// the same ordering as serving. Implementations enqueue work; none host-poll.
trait ContextProtocol {
    fn push_query(&self, rank: usize, slot: usize, query: Query, rows: usize) -> Result<()>;
    fn push_candidates(&self, rank: usize, slot: usize, candidates: Candidates, rows: usize) -> Result<()>;
    fn wait_candidates(&self, rank: usize, slot: usize) -> Result<(*mut c_void, *mut c_void)>;
    fn wait_query(&self, rank: usize, slot: usize) -> Result<*mut c_void>;
    fn push_partial(&self, rank: usize, slot: usize, partial: Partial, rows: usize) -> Result<()>;
    fn wait_partial(&self, rank: usize, slot: usize) -> Result<Partial>;
}

impl ContextProtocol for ContextExchange<'_> {
    fn push_query(&self, rank: usize, slot: usize, query: Query, rows: usize) -> Result<()> {
        self.push_query(rank, slot, query.values, query.row_pitch, rows)
    }
    fn push_candidates(&self, rank: usize, slot: usize, candidates: Candidates, rows: usize) -> Result<()> {
        self.push_candidates(rank, slot, candidates.scores, candidates.indices, rows)
    }
    fn wait_candidates(&self, rank: usize, slot: usize) -> Result<(*mut c_void, *mut c_void)> {
        self.wait_candidates(rank, slot)
    }
    fn wait_query(&self, rank: usize, slot: usize) -> Result<*mut c_void> {
        self.wait_query(rank, slot)
    }
    fn push_partial(&self, rank: usize, slot: usize, partial: Partial, rows: usize) -> Result<()> {
        self.push_partial(rank, slot, partial.values, partial.lse, rows)
    }
    fn wait_partial(&self, rank: usize, slot: usize) -> Result<Partial> {
        let (values, lse) = self.wait_partial(rank, slot)?;
        Ok(Partial { values, lse })
    }
}

/// Decode and speculative verify use this same sequence; prefill gathers and
/// commits owned rows through the shared paging tables, without an exchange.
/// `context_index` is the ordinal among context layers, not the physical layer
/// index: interleaved KDA/SWA layers must not collapse the two exchange slots.
pub(crate) fn context_decode(exchange: &ContextExchange<'_>, rank: usize, context_index: usize,
    lane: usize, rows: usize, kernels: &mut impl ContextKernels) -> Result<()> {
    context_decode_with(exchange, rank, context_index, lane, rows, kernels)
}

fn context_decode_with(exchange: &impl ContextProtocol, rank: usize, context_index: usize,
    lane: usize, rows: usize, kernels: &mut impl ContextKernels) -> Result<()> {
    ensure!(rank < 2 && rows > 0, "context attention needs rank 0/1 and positive rows");
    let slot = slot(context_index, lane);
    let query = kernels.produce()?;
    exchange.push_query(rank, slot, query, rows)?;
    if let Some(candidates) = kernels.score()? {
        exchange.push_candidates(rank, slot, candidates, rows)?;
        let (scores, indices) = exchange.wait_candidates(rank, slot)?;
        kernels.merge(scores, indices)?;
    }
    let query = exchange.wait_query(rank, slot)?;
    let peer = kernels.partial(query, 1 - rank)?;
    exchange.push_partial(rank, slot, peer, rows)?;
    let own = kernels.partial(query, rank)?;
    let received = exchange.wait_partial(rank, slot)?;
    kernels.combine(combine_order(rank, own, received))
}

/// Family buffers and kernels. Packing is queued on the source stream;
/// adoption is queued on the destination after the shared wait. Both buffers
/// must remain live until their streams drain (including graph replay).
pub(crate) trait LayerKernels {
    fn pack(&mut self, hop: Hop) -> Result<*const c_void>;
    fn entry_destination(&mut self, rank: usize) -> Result<*mut c_void>;
    fn adopt(&mut self, hop: Hop, residual: *mut c_void) -> Result<()>;
    fn heads(&mut self, layer: usize) -> Result<()>;
    fn context(&mut self, layer: usize, context_index: usize) -> Result<()>;
    fn owned(&mut self, layer: usize, gpu: usize) -> Result<()>;
    fn ffn(&mut self, layer: usize, owner: Option<usize>) -> Result<()>;
}

pub(crate) trait LayerProtocol {
    type Ticket;
    fn require_live(&self) -> Result<()>;
    fn push_entry(&self, from: usize, source: *const c_void, destination: *mut c_void, bytes: usize) -> Result<()>;
    fn wait_entry(&self, to: usize) -> Result<()>;
    fn send(&self, index: usize, lane: usize, source: *const c_void, rows: usize) -> Result<Self::Ticket>;
    fn land(&self, ticket: Self::Ticket) -> Result<*mut c_void>;
}

/// Deferred FFN work is retained in the engine, never in a temporary ticket:
/// dropping this non-owning handle on error cannot free an in-flight wave.
/// The enclosing whole-submit boundary aborts and drains the engine's owners.
pub(crate) struct AttentionStateOwners<'a> {
    pub weights: &'a [u8],
    pub kv: &'a [u8],
    /// None declares that this layer has no indexer/compressor state.
    pub indexer: Option<&'a [u8]>,
    pub compressor: Option<&'a [u8]>,
}

pub(crate) trait PipelineKernels: LayerKernels {
    type FfnTicket: Copy;
    /// Describe actual allocated state, including prefix-cache bindings, not
    /// intended owners. Checked for every layer/lane before any enqueue.
    fn state_owners(&self, layer: usize) -> Result<AttentionStateOwners<'_>>;
    /// Queue both ranks' publications in the same lane order. Defer any wait
    /// for another unit's publication to finish_ffn, never to this method.
    fn begin_ffn(&mut self, layer: usize, owner: Option<usize>) -> Result<Self::FfnTicket>;
    fn finish_ffn(&mut self, layer: usize, ticket: Self::FfnTicket) -> Result<()>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LaneOrder {
    /// Consecutive prefill chunks share mutable compressor/KV history.
    OrderedChunks,
    /// Decode/verify lanes contain disjoint request state and can be staggered.
    Independent,
}

/// A graph segment cannot alias another rank, lane, mode or payload shape.
/// Families append their exact pointer/table/workspace binding key; GraphBank
/// retains GraphOwner pins and releases retired segments only after drainage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct LayerSegmentKey<K> {
    pub layer: usize,
    pub lane: usize,
    pub rank: u8,
    pub mode: LayerMode,
    pub rows: usize,
    pub binding: K,
}

impl<K> LayerSegmentKey<K> {
    pub fn new(plan: &LayerExecutionPlan, layer: usize, lane: usize, rank: u8, rows: usize, binding: K) -> Result<Self> {
        ensure!(rank < 2 && (lane as u64) < plan.spec().lanes && rows > 0 && rows as u64 <= plan.spec().rows,
            "graph segment outside admitted rank/lane/rows");
        Ok(Self { layer, lane, rank, mode: plan.mode(layer)?, rows, binding })
    }
}

struct LayerExchange<'a, 'b> {
    peer: &'a PeerExchange<'b>,
    hops: &'a HopLink<'b>,
}

impl LayerProtocol for LayerExchange<'_, '_> {
    type Ticket = super::hop::HopTicket;
    fn require_live(&self) -> Result<()> { self.peer.require_live() }
    fn push_entry(&self, from: usize, source: *const c_void, destination: *mut c_void, bytes: usize) -> Result<()> {
        self.peer.push_to(from, DIRECT, source, destination, bytes)
    }
    fn wait_entry(&self, to: usize) -> Result<()> { self.peer.wait(to, DIRECT) }
    fn send(&self, index: usize, lane: usize, source: *const c_void, rows: usize) -> Result<Self::Ticket> {
        self.hops.send(index, lane, source, rows)
    }
    fn land(&self, ticket: Self::Ticket) -> Result<*mut c_void> { self.hops.land(ticket) }
}

fn transfer(exchange: &impl LayerProtocol, index: usize, hop: Hop, lane: usize, rows: usize,
    kernels: &mut dyn LayerKernels) -> Result<()> {
    exchange.require_live()?;
    let source = kernels.pack(hop)?;
    let residual = if hop.at == HopPoint::Entry {
        let destination = kernels.entry_destination(usize::from(hop.to))?;
        let bytes = rows.checked_mul(usize::try_from(hop.row_bytes)?)
            .ok_or_else(|| anyhow::anyhow!("entry hop bytes overflow"))?;
        exchange.push_entry(usize::from(hop.from), source, destination, bytes)?;
        exchange.wait_entry(usize::from(hop.to))?;
        destination
    } else {
        let ticket = exchange.send(index, lane, source, rows)?;
        exchange.land(ticket)?
    };
    kernels.adopt(hop, residual)
}

/// Runs the planner's exact per-layer modes and hop points. An owned attention
/// layer with split FFN broadcasts after attention, not after the FFN, and its
/// neighbouring KDA/GQA layers may remain head-split. Families must advertise
/// these capabilities only after all their kernel bindings exist.
pub(crate) fn layer_stack(placement: &Placement, peer: &PeerExchange<'_>, hops: &HopLink<'_>,
    lane: usize, rows: usize, kernels: &mut impl LayerKernels) -> Result<()> {
    layer_stack_with(placement, &LayerExchange { peer, hops }, lane, rows, kernels)
}

fn layer_stack_with(placement: &Placement, exchange: &impl LayerProtocol,
    lane: usize, rows: usize, kernels: &mut impl LayerKernels) -> Result<()> {
    ensure!(rows > 0, "layer stack needs positive rows");
    validate_stack(placement)?;
    let mut hop_index = 0;
    let mut hops_at = |point, kernels: &mut dyn LayerKernels| -> Result<()> {
        while let Some(&hop) = placement.hops.get(hop_index).filter(|h| h.at == point) {
            transfer(exchange, hop_index, hop, lane, rows, kernels)?;
            hop_index += 1;
        }
        Ok(())
    };
    hops_at(HopPoint::Entry, kernels)?;
    let mut context_index = 0;
    for (layer, assignment) in placement.layers.iter().enumerate() {
        hops_at(HopPoint::BeforeLayer(layer), kernels)?;
        let owner = match assignment.mode {
            LayerMode::HeadSplit => { kernels.heads(layer)?; None }
            LayerMode::ContextSplit => {
                kernels.context(layer, context_index)?;
                context_index += 1;
                None
            }
            LayerMode::Whole { gpu, ffn } => {
                ensure!(gpu < 2, "owned attention GPU {gpu}");
                kernels.owned(layer, usize::from(gpu))?;
                (ffn == FfnMode::Owner).then_some(usize::from(gpu))
            }
        };
        hops_at(HopPoint::AfterAttention(layer), kernels)?;
        kernels.ffn(layer, owner)?;
    }
    hops_at(HopPoint::Exit, kernels)?;
    ensure!(hop_index == placement.hops.len(), "attention placement has out-of-order hop points");
    Ok(())
}

/// Shared two-lane serving executor. Binding happens before any capture or
/// enqueue, and requires hop/FFN waits to share one terminal owner. A family
/// supplies only attention/FFN kernels and residual packing, not peer order.
pub(crate) fn layer_pipeline<K: PipelineKernels>(scope: &crate::shared::experts::rtx::pair::SubmissionScope<'_>,
    plan: &LayerExecutionPlan, peer: &PeerExchange<'_>, hops: &HopLink<'_>,
    order: LaneOrder, rows: &[usize], kernels: &mut [K]) -> Result<()> {
    hops.validate_binding(plan, peer)?;
    layer_pipeline_with(scope, plan, &LayerExchange { peer, hops }, order, rows, kernels)
}

fn pipeline_hops(exchange: &impl LayerProtocol, plan: &LayerExecutionPlan, point: HopPoint,
    lane: usize, rows: usize, kernels: &mut impl LayerKernels, consumed: &mut [usize]) -> Result<()> {
    for (index, &hop) in plan.hops().iter().enumerate().filter(|(_, hop)| hop.at == point) {
        ensure!(consumed[index] == 0, "duplicate planned hop {index} in lane {lane}");
        transfer(exchange, index, hop, lane, rows, kernels)?;
        consumed[index] += 1;
    }
    Ok(())
}

fn pipeline_attention(exchange: &impl LayerProtocol, plan: &LayerExecutionPlan, layer: usize,
    lane: usize, rows: usize, kernels: &mut impl LayerKernels, consumed: &mut [usize]) -> Result<Option<usize>> {
    exchange.require_live()?;
    if layer == 0 { pipeline_hops(exchange, plan, HopPoint::Entry, lane, rows, kernels, consumed)?; }
    pipeline_hops(exchange, plan, HopPoint::BeforeLayer(layer), lane, rows, kernels, consumed)?;
    let owner = match plan.mode(layer)? {
        LayerMode::HeadSplit => { kernels.heads(layer)?; None }
        LayerMode::ContextSplit => {
            let ordinal = plan.modes()[..layer].iter().filter(|&&mode| mode == LayerMode::ContextSplit).count();
            kernels.context(layer, ordinal)?;
            None
        }
        LayerMode::Whole { gpu, ffn } => {
            kernels.owned(layer, usize::from(gpu))?;
            (ffn == FfnMode::Owner).then_some(usize::from(gpu))
        }
    };
    pipeline_hops(exchange, plan, HopPoint::AfterAttention(layer), lane, rows, kernels, consumed)?;
    Ok(owner)
}

pub(crate) fn layer_pipeline_with<K: PipelineKernels>(_scope: &crate::shared::experts::rtx::pair::SubmissionScope<'_>,
    plan: &LayerExecutionPlan, exchange: &impl LayerProtocol,
    order: LaneOrder, rows: &[usize], kernels: &mut [K]) -> Result<()> {
    cuteafd_loader::placement::validate_layer_pipeline_lanes(plan.spec().lanes,
        usize::try_from(plan.spec().lanes)?)?;
    ensure!(!kernels.is_empty() && kernels.len() == rows.len()
        && kernels.len() as u64 <= plan.spec().lanes, "layer pipeline needs one or two admitted lanes");
    ensure!(rows.iter().all(|&rows| rows > 0 && rows as u64 <= plan.spec().rows),
        "layer pipeline rows outside admission");
    exchange.require_live()?;
    let count = plan.modes().len();
    for kernel in kernels.iter() {
        for layer in 0..count {
            let state = kernel.state_owners(layer)?;
            for (category, owners) in [("weights", Some(state.weights)), ("KV", Some(state.kv)),
                ("indexer", state.indexer), ("compressor", state.compressor)] {
                if let Some(owners) = owners {
                    plan.validate_state_owners(layer, owners)
                        .map_err(|error| anyhow::anyhow!("{category}: {error}"))?;
                }
            }
        }
    }
    let mut consumed = vec![vec![0; plan.hops().len()]; kernels.len()];
    match order {
        LaneOrder::OrderedChunks => {
            for layer in 0..count {
                for lane in 0..kernels.len() {
                    let owner = pipeline_attention(exchange, plan, layer, lane, rows[lane], &mut kernels[lane], &mut consumed[lane])?;
                    let ticket = kernels[lane].begin_ffn(layer, owner)?;
                    exchange.require_live()?;
                    kernels[lane].finish_ffn(layer, ticket)?;
                }
            }
        }
        LaneOrder::Independent => {
            // Lane 1 trails lane 0 by one layer. All publications in a wave
            // precede its completion waits, with identical FFN lane order on
            // both ranks. A wave closes before reusing its parity receive slots.
            for wave in 0..count + kernels.len() - 1 {
                let mut pending = Vec::with_capacity(kernels.len());
                for lane in 0..kernels.len() {
                    let Some(layer) = wave.checked_sub(lane).filter(|&layer| layer < count) else { continue };
                    let owner = pipeline_attention(exchange, plan, layer, lane, rows[lane], &mut kernels[lane], &mut consumed[lane])?;
                    pending.push((lane, layer, kernels[lane].begin_ffn(layer, owner)?));
                }
                for (lane, layer, ticket) in pending {
                    exchange.require_live()?;
                    kernels[lane].finish_ffn(layer, ticket)?;
                }
            }
        }
    }
    for lane in 0..kernels.len() {
        pipeline_hops(exchange, plan, HopPoint::Exit, lane, rows[lane], &mut kernels[lane], &mut consumed[lane])?;
    }
    validate_hop_inventory(plan, &consumed)
}

fn validate_hop_inventory(plan: &LayerExecutionPlan, consumed: &[Vec<usize>]) -> Result<()> {
    ensure!(consumed.iter().all(|lane| lane.len() == plan.hops().len()
        && lane.iter().all(|&count| count == 1)), "planned hop missing or duplicated");
    Ok(())
}

fn validate_stack(placement: &Placement) -> Result<()> {
    ensure!(placement.layers.iter().all(|assignment| match assignment.mode {
        LayerMode::Whole { gpu, .. } => gpu < 2,
        _ => true,
    }), "owned attention GPU must be 0/1");
    let expected_broadcasts = placement.layers.iter().filter(|assignment|
        matches!(assignment.mode, LayerMode::Whole { ffn: FfnMode::Split, .. })).count();
    ensure!(placement.hops.iter().filter(|hop| matches!(hop.at, HopPoint::AfterAttention(_))).count()
        == expected_broadcasts, "owned attention missing its split FFN broadcast");
    let mut previous = None;
    for hop in &placement.hops {
        ensure!(hop.from < 2 && hop.to < 2 && hop.from != hop.to && hop.row_bytes > 0,
            "invalid residual hop");
        let order = match hop.at {
            HopPoint::Entry => 0,
            HopPoint::BeforeLayer(layer) => {
                ensure!(layer < placement.layers.len(), "hop layer past stack");
                1 + 2 * layer
            }
            HopPoint::AfterAttention(layer) => {
                ensure!(layer < placement.layers.len(), "hop layer past stack");
                ensure!(placement.layers[layer].mode == LayerMode::Whole { gpu: hop.from, ffn: FfnMode::Split },
                    "after-attention hop requires the split FFN's owner");
                2 + 2 * layer
            }
            HopPoint::Exit => 1 + 2 * placement.layers.len(),
        };
        ensure!(previous.is_none_or(|previous| previous < order), "attention placement has out-of-order hop points");
        previous = Some(order);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    struct Recorder {
        calls: RefCell<Vec<String>>,
        indexer: bool,
    }
    impl Recorder {
        fn note(&self, s: impl Into<String>) { self.calls.borrow_mut().push(s.into()); }
        fn ptr(n: usize) -> *mut c_void { n as *mut c_void }
    }
    impl ContextKernels for &Recorder {
        fn produce(&mut self) -> Result<Query> {
            self.note("produce");
            Ok(Query { values: std::ptr::null(), row_pitch: 32 })
        }
        fn score(&mut self) -> Result<Option<Candidates>> {
            self.note("score");
            Ok(self.indexer.then_some(Candidates { scores: std::ptr::null(), indices: std::ptr::null() }))
        }
        fn merge(&mut self, _: *mut c_void, _: *mut c_void) -> Result<()> { self.note("merge"); Ok(()) }
        fn partial(&mut self, _: *mut c_void, heads_rank: usize) -> Result<Partial> {
            self.note(format!("partial{heads_rank}"));
            Ok(Partial { values: Recorder::ptr(heads_rank + 1), lse: std::ptr::null_mut() })
        }
        fn combine(&mut self, partials: [Partial; 2]) -> Result<()> {
            self.note(format!("combine{},{}", partials[0].values as usize, partials[1].values as usize));
            Ok(())
        }
    }
    impl ContextProtocol for Recorder {
        fn push_query(&self, _: usize, _: usize, _: Query, _: usize) -> Result<()> { self.note("push-q"); Ok(()) }
        fn push_candidates(&self, _: usize, _: usize, _: Candidates, _: usize) -> Result<()> { self.note("push-candidates"); Ok(()) }
        fn wait_candidates(&self, _: usize, _: usize) -> Result<(*mut c_void, *mut c_void)> {
            self.note("wait-candidates"); Ok((std::ptr::null_mut(), std::ptr::null_mut()))
        }
        fn wait_query(&self, _: usize, _: usize) -> Result<*mut c_void> { self.note("wait-q"); Ok(std::ptr::null_mut()) }
        fn push_partial(&self, _: usize, _: usize, _: Partial, _: usize) -> Result<()> { self.note("push-partial"); Ok(()) }
        fn wait_partial(&self, rank: usize, _: usize) -> Result<Partial> {
            self.note("wait-partial");
            Ok(Partial { values: Self::ptr(4 - rank), lse: std::ptr::null_mut() })
        }
    }

    impl LayerKernels for &Recorder {
        fn pack(&mut self, hop: Hop) -> Result<*const c_void> {
            self.note(format!("pack{}", hop.from)); Ok(Recorder::ptr(10 + usize::from(hop.from)))
        }
        fn entry_destination(&mut self, rank: usize) -> Result<*mut c_void> {
            self.note(format!("entry{rank}")); Ok(Recorder::ptr(20 + rank))
        }
        fn adopt(&mut self, hop: Hop, residual: *mut c_void) -> Result<()> {
            self.note(format!("adopt{}:{}", hop.to, residual as usize)); Ok(())
        }
        fn heads(&mut self, layer: usize) -> Result<()> { self.note(format!("heads{layer}")); Ok(()) }
        fn context(&mut self, layer: usize, context_index: usize) -> Result<()> {
            self.note(format!("context{layer}:{context_index}")); Ok(())
        }
        fn owned(&mut self, layer: usize, gpu: usize) -> Result<()> { self.note(format!("owned{layer}:{gpu}")); Ok(()) }
        fn ffn(&mut self, layer: usize, owner: Option<usize>) -> Result<()> { self.note(format!("ffn{layer}:{owner:?}")); Ok(()) }
    }
    impl LayerProtocol for Recorder {
        type Ticket = usize;
        fn require_live(&self) -> Result<()> { self.note("live"); Ok(()) }
        fn push_entry(&self, from: usize, source: *const c_void, destination: *mut c_void, bytes: usize) -> Result<()> {
            assert_eq!(source as usize, 10 + from);
            assert_eq!(destination as usize, 20 + 1 - from);
            assert_eq!(bytes, 6 * 32);
            self.note(format!("push-entry{from}")); Ok(())
        }
        fn wait_entry(&self, to: usize) -> Result<()> { self.note(format!("wait-entry{to}")); Ok(()) }
        fn send(&self, index: usize, lane: usize, _: *const c_void, rows: usize) -> Result<usize> {
            assert_eq!((lane, rows), (1, 6)); self.note(format!("send{index}")); Ok(index)
        }
        fn land(&self, ticket: usize) -> Result<*mut c_void> {
            self.note(format!("land{ticket}")); Ok(Self::ptr(30 + ticket))
        }
    }

    fn placement(modes: &[LayerMode]) -> Placement {
        use cuteafd_loader::placement::{plan_hops, ExpertHome, HopSpec, LayerAssignment};
        Placement {
            pool_tokens: 0, pool_limits: cuteafd_loader::placement::PoolLimits(vec![]), attention_placement: cuteafd_loader::placement::AttentionPlacement::Layers,
            attention_by_kind: vec![], peer_row_bytes: 0, onboard_layers: 0,
            layers: modes.iter().map(|&mode| LayerAssignment { mode, experts: ExpertHome::Dense }).collect(),
            movables: vec![], expert_ranges: vec![], residual: vec![], tp2: None, items: vec![],
            hops: plan_hops(modes, &HopSpec { storage: cuteafd_loader::placement::HopStorage::ReceiveSlots, row_bytes: 32, rows: 6, lanes: 2, entry_gpu: 0, head_gpu: 0, terminal: match modes.last() { Some(LayerMode::Whole { ffn: FfnMode::Owner, .. }) => cuteafd_loader::placement::TerminalHome::AttentionOwner, _ => cuteafd_loader::placement::TerminalHome::Replicated } }).unwrap(),
        }
    }

    #[test]
    fn mixed_owned_stack_transfers_residual_before_the_split_ffn() {
        let r = Recorder { calls: RefCell::new(vec![]), indexer: false };
        let p = placement(&[LayerMode::HeadSplit, LayerMode::Whole { gpu: 1, ffn: FfnMode::Split },
            LayerMode::ContextSplit, LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner },
            LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner }]);
        layer_stack_with(&p, &r, 1, 6, &mut &r).unwrap();
        assert_eq!(*r.calls.borrow(), ["live", "pack0", "entry1", "push-entry0", "wait-entry1", "adopt1:21",
            "heads0", "ffn0:None", "owned1:1", "live", "pack1", "send1", "land1", "adopt0:31", "ffn1:None",
            "context2:0", "ffn2:None", "owned3:0", "ffn3:Some(0)", "live", "pack0", "send2", "land2", "adopt1:32",
            "owned4:1", "ffn4:Some(1)", "live", "pack1", "send3", "land3", "adopt0:33"]);
    }

    #[test]
    fn interleaved_context_layers_alternate_exchange_ordinals() {
        let r = Recorder { calls: RefCell::new(vec![]), indexer: false };
        let p = placement(&[LayerMode::HeadSplit, LayerMode::ContextSplit, LayerMode::HeadSplit,
            LayerMode::ContextSplit, LayerMode::HeadSplit, LayerMode::ContextSplit]);
        layer_stack_with(&p, &r, 1, 6, &mut &r).unwrap();
        assert_eq!(r.calls.borrow().iter().filter(|call| call.starts_with("context")).cloned().collect::<Vec<_>>(),
            ["context1:0", "context3:1", "context5:2"]);
    }

    #[test]
    fn malformed_stack_is_rejected_before_enqueuing_anything() {
        let r = Recorder { calls: RefCell::new(vec![]), indexer: false };
        let mut p = placement(&[LayerMode::HeadSplit, LayerMode::Whole { gpu: 1, ffn: FfnMode::Split }]);
        p.hops.swap(0, 1);
        assert!(layer_stack_with(&p, &r, 1, 6, &mut &r).is_err());
        p.hops.clear();
        assert!(layer_stack_with(&p, &r, 1, 6, &mut &r).is_err(), "missing FFN broadcast");
        p.layers[1].mode = LayerMode::Whole { gpu: 2, ffn: FfnMode::Owner };
        assert!(layer_stack_with(&p, &r, 1, 6, &mut &r).is_err());
        assert!(layer_stack_with(&p, &r, 1, 0, &mut &r).is_err());
        assert!(r.calls.borrow().is_empty());
    }

    fn execution_plan(modes: &[LayerMode], entry: u8, head: u8) -> LayerExecutionPlan {
        use cuteafd_loader::placement::{plan_hops, HopSpec, ResidualHome};
        let spec = HopSpec { storage: cuteafd_loader::placement::HopStorage::ReceiveSlots, row_bytes: 32, rows: 6, lanes: 2, entry_gpu: entry, head_gpu: head, terminal: match modes.last() { Some(LayerMode::Whole { ffn: FfnMode::Owner, .. }) => cuteafd_loader::placement::TerminalHome::AttentionOwner, _ => cuteafd_loader::placement::TerminalHome::Replicated } };
        let mut p = placement(modes);
        p.hops = plan_hops(modes, &spec).unwrap();
        p.residual = vec![ResidualHome::Owned(entry)];
        for &mode in modes {
            p.residual.push(p.residual.last().unwrap().transition(mode).home);
        }
        LayerExecutionPlan::new(&p, spec).unwrap()
    }

    #[test]
    fn execution_plan_rejects_stale_hops_residuals_and_prefix_owners() {
        use cuteafd_loader::placement::LayerPlanError;
        let modes = [LayerMode::Whole { gpu: 1, ffn: FfnMode::Split }, LayerMode::HeadSplit];
        let plan = execution_plan(&modes, 0, 1);
        plan.validate_state_owners(0, &[1]).unwrap();
        plan.validate_state_owners(1, &[1, 0]).unwrap();
        for owners in [&[][..], &[0][..], &[0, 1][..], &[1, 1][..]] {
            assert!(plan.validate_state_owners(0, owners).is_err());
        }
        let mut p = placement(&modes);
        p.residual = plan.homes().to_vec();
        p.hops = plan.hops().to_vec();
        for index in 0..p.hops.len() {
            let mut missing = p.clone();
            missing.hops.remove(index);
            assert_eq!(LayerExecutionPlan::new(&missing, plan.spec()), Err(LayerPlanError::Hops));
            let mut wrong = p.clone();
            wrong.hops[index].from = 1 - wrong.hops[index].from;
            assert_eq!(LayerExecutionPlan::new(&wrong, plan.spec()), Err(LayerPlanError::Hops));
        }
        p.residual[1] = cuteafd_loader::placement::ResidualHome::Owned(1);
        assert_eq!(LayerExecutionPlan::new(&p, plan.spec()), Err(LayerPlanError::Residual));
    }

    #[test]
    fn graph_segments_distinguish_owner_lane_shape_and_binding() {
        let plan = execution_plan(&[LayerMode::Whole { gpu: 1, ffn: FfnMode::Split }], 0, 0);
        let key = LayerSegmentKey::new(&plan, 0, 0, 1, 4, 17usize).unwrap();
        for (lane, rank, rows, binding) in [(1, 1, 4, 17), (0, 0, 4, 17), (0, 1, 6, 17), (0, 1, 4, 18)] {
            assert_ne!(key, LayerSegmentKey::new(&plan, 0, lane, rank, rows, binding).unwrap());
        }
        for (layer, lane, rank, rows) in [(1, 0, 0, 1), (0, 2, 0, 1), (0, 0, 2, 1), (0, 0, 0, 0), (0, 0, 0, 7)] {
            assert!(LayerSegmentKey::new(&plan, layer, lane, rank, rows, 17).is_err());
        }
    }

    struct PipelineRecorder<'a> {
        shared: &'a PipelineTrace,
        lane: usize,
    }
    struct PipelineTrace {
        hops: Vec<Hop>,
        schedule: RefCell<super::super::order::Schedule>,
        calls: RefCell<Vec<(usize, usize, &'static str)>>,
        owners: RefCell<crate::shared::experts::rtx::pair::TestSubmission<'static>>,
        callbacks: RefCell<Vec<&'static str>>,
        fault: Option<(usize, bool)>,
        active_lane: std::cell::Cell<usize>,
        transfers: RefCell<Vec<(usize, usize, bool)>>,
        state: Vec<Vec<u8>>,
        stale: Option<(usize, usize, usize)>,
    }
    impl PipelineTrace {
        fn new(plan: &LayerExecutionPlan) -> Self {
            Self { hops: plan.hops().to_vec(), schedule: RefCell::new(Default::default()),
                calls: RefCell::new(vec![]), owners: RefCell::new(crate::shared::experts::rtx::pair::TestSubmission::new(None)),
                callbacks: RefCell::new(vec![]), fault: None, active_lane: std::cell::Cell::new(0),
                transfers: RefCell::new(vec![]), state: (0..plan.modes().len()).map(|layer|
                    plan.attention_owners(layer).unwrap()).collect(), stale: None }
        }
        fn callback(&self, name: &'static str) -> Result<()> {
            self.owners.borrow().require_live()?;
            let at = self.callbacks.borrow().len();
            self.callbacks.borrow_mut().push(name);
            if let Some((failure, panic)) = self.fault {
                if at == failure {
                    if panic { panic!("injected pipeline callback {at}: {name}"); }
                    anyhow::bail!("injected pipeline callback {at}: {name}");
                }
            }
            Ok(())
        }
        fn inventory(&self, plan: &LayerExecutionPlan, lanes: usize) -> Result<()> {
            let actual = self.transfers.borrow();
            for lane in 0..lanes {
                for index in 0..plan.hops().len() {
                    for pushed in [true, false] {
                        ensure!(actual.iter().filter(|&&event| event == (lane, index, pushed)).count() == 1,
                            "missing/duplicate planned hop {index} lane {lane}");
                    }
                }
            }
            ensure!(actual.len() == lanes * plan.hops().len() * 2, "extra hop");
            Ok(())
        }
        fn run(&self, plan: &LayerExecutionPlan, order: LaneOrder, lanes: usize) -> Result<()> {
            use crate::shared::experts::rtx::pair::{submit_scoped, teardown};
            let mut kernels: Vec<_> = (0..lanes).map(|lane| PipelineRecorder { shared: self, lane }).collect();
            submit_scoped(|| self.owners.borrow().require_live(),
                || teardown(&mut &*self),
                |scope| layer_pipeline_with(scope, plan, self, order, &vec![6; lanes], &mut kernels))
        }
        fn pair(&self, exchange: &'static str, slot: usize) {
            let mut s = self.schedule.borrow_mut();
            for rank in 0..2 { s.push(rank, exchange, slot, format!("{exchange} push {slot}")); }
            for rank in 0..2 { s.wait(rank, exchange, slot, format!("{exchange} wait {slot}")); }
        }
    }
    impl LayerProtocol for PipelineTrace {
        type Ticket = (usize, usize, usize, usize);
        fn require_live(&self) -> Result<()> { self.callback("live") }
        fn push_entry(&self, from: usize, _: *const c_void, _: *mut c_void, _: usize) -> Result<()> {
            self.callback("push_entry")?;
            let index = self.hops.iter().position(|hop| hop.at == HopPoint::Entry).unwrap();
            self.transfers.borrow_mut().push((self.active_lane.get(), index, true));
            self.schedule.borrow_mut().push(from, "entry", DIRECT, "entry push"); Ok(())
        }
        fn wait_entry(&self, to: usize) -> Result<()> {
            self.callback("wait_entry")?;
            let index = self.hops.iter().position(|hop| hop.at == HopPoint::Entry).unwrap();
            self.transfers.borrow_mut().push((self.active_lane.get(), index, false));
            self.schedule.borrow_mut().wait(to, "entry", DIRECT, "entry wait"); Ok(())
        }
        fn send(&self, index: usize, lane: usize, source: *const c_void, _: usize) -> Result<Self::Ticket> {
            self.callback("send")?;
            let from = source as usize - 1;
            let to = self.hops[index].to;
            let ordinal = self.hops[..index].iter().filter(|hop| hop.charged() && hop.to == to).count();
            let slot = 2 * lane + ordinal % 2;
            self.schedule.borrow_mut().push(from, "hop", slot, "hop push");
            self.transfers.borrow_mut().push((lane, index, true));
            Ok((1 - from, slot, index, lane))
        }
        fn land(&self, (to, slot, index, lane): Self::Ticket) -> Result<*mut c_void> {
            self.callback("land")?;
            self.transfers.borrow_mut().push((lane, index, false));
            self.schedule.borrow_mut().wait(to, "hop", slot, "hop wait");
            Ok(std::ptr::null_mut())
        }
    }
    impl LayerKernels for PipelineRecorder<'_> {
        fn pack(&mut self, hop: Hop) -> Result<*const c_void> {
            self.shared.callback("pack")?;
            self.shared.active_lane.set(self.lane);
            Ok((usize::from(hop.from) + 1) as *const c_void)
        }
        fn entry_destination(&mut self, _: usize) -> Result<*mut c_void> { self.shared.callback("entry_destination")?; Ok(std::ptr::null_mut()) }
        fn adopt(&mut self, _: Hop, _: *mut c_void) -> Result<()> { self.shared.callback("adopt") }
        fn heads(&mut self, layer: usize) -> Result<()> {
            self.shared.callback("heads")?;
            self.shared.calls.borrow_mut().push((self.lane, layer, "heads"));
            self.shared.pair("heads", 2 * self.lane + layer % 2); Ok(())
        }
        fn context(&mut self, layer: usize, ordinal: usize) -> Result<()> {
            self.shared.callback("context")?;
            self.shared.calls.borrow_mut().push((self.lane, layer, "context"));
            for exchange in ["query", "candidate", "partial"] {
                self.shared.pair(exchange, 2 * self.lane + ordinal % 2);
            }
            Ok(())
        }
        fn owned(&mut self, layer: usize, _: usize) -> Result<()> {
            self.shared.callback("owned")?;
            self.shared.calls.borrow_mut().push((self.lane, layer, "owned")); Ok(())
        }
        fn ffn(&mut self, _: usize, _: Option<usize>) -> Result<()> { panic!("pipeline uses deferred FFN"); }
    }
    impl PipelineKernels for PipelineRecorder<'_> {
        type FfnTicket = Option<usize>;
        fn state_owners(&self, layer: usize) -> Result<AttentionStateOwners<'_>> {
            self.shared.callback("state_owners")?;
            let owners = &self.shared.state[layer][..];
            let category = self.shared.stale.filter(|&(lane, l, _)| lane == self.lane && l == layer).map(|(_, _, c)| c);
            let bound = |c| if category == Some(c) { &[0][..] } else { owners };
            Ok(AttentionStateOwners { weights: bound(0), kv: bound(1),
                indexer: Some(bound(2)), compressor: Some(bound(3)) })
        }
        fn begin_ffn(&mut self, layer: usize, owner: Option<usize>) -> Result<Self::FfnTicket> {
            self.shared.callback("begin")?;
            self.shared.calls.borrow_mut().push((self.lane, layer, "begin"));
            if owner.is_none() {
                let slot = 2 * self.lane + layer % 2;
                let mut s = self.shared.schedule.borrow_mut();
                s.push(0, "routes", slot, "routes");
                s.wait(1, "routes", slot, "routes wait");
                s.push(1, "results", slot, "results");
                for rank in 0..2 { s.push(rank, "ffn", slot, format!("ffn push lane {} L{layer}", self.lane)); }
                Ok(Some(slot))
            } else { Ok(None) }
        }
        fn finish_ffn(&mut self, layer: usize, ticket: Self::FfnTicket) -> Result<()> {
            self.shared.callback("finish")?;
            self.shared.calls.borrow_mut().push((self.lane, layer, "finish"));
            if let Some(slot) = ticket {
                let mut s = self.shared.schedule.borrow_mut();
                s.wait(0, "results", slot, "result wait");
                for rank in 0..2 { s.wait(rank, "ffn", slot, format!("ffn wait lane {} L{layer}", self.lane)); }
            }
            Ok(())
        }
    }

    impl crate::shared::experts::rtx::pair::SubmissionOwners for &PipelineTrace {
        fn abort_all(&mut self) -> Result<()> { self.owners.borrow_mut().abort_all() }
        fn drain_ranks(&mut self) -> Result<()> {
            self.owners.borrow_mut().drain_ranks()?;
            super::super::order::check_terminal(&self.schedule.borrow(), !self.owners.borrow().live)
                .map_err(|error| anyhow::anyhow!(error.to_string()))
        }
        fn quiesce_links(&mut self) -> Result<()> { self.owners.borrow_mut().quiesce_links() }
        fn drain_copies(&mut self) -> Result<()> { self.owners.borrow_mut().drain_copies() }
        fn retire_graphs(&mut self) -> Result<()> { self.owners.borrow_mut().retire_graphs() }
        fn finish(&mut self, drained: bool) { self.owners.borrow_mut().finish(drained); }
    }

    #[test]
    fn pipeline_callback_errors_and_panics_use_real_terminal_teardown() {
        use crate::shared::experts::rtx::pair::TerminalStepError;
        let modes = [LayerMode::HeadSplit, LayerMode::ContextSplit,
            LayerMode::Whole { gpu: 1, ffn: FfnMode::Split },
            LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner },
            LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner }];
        for order in [LaneOrder::OrderedChunks, LaneOrder::Independent] {
            for lanes in 1..=2 {
                for entry in 0..2 {
                    let plan = execution_plan(&modes, entry, 0);
                    let healthy = PipelineTrace::new(&plan);
                    healthy.run(&plan, order, lanes).unwrap();
                    for at in 0..healthy.callbacks.borrow().len() {
                        for panic in [false, true] {
                            let mut trace = PipelineTrace::new(&plan);
                            trace.fault = Some((at, panic));
                            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| trace.run(&plan, order, lanes)));
                            if panic { assert!(result.is_err()); }
                            else { assert!(result.unwrap().unwrap_err().downcast_ref::<TerminalStepError>().is_some()); }
                            trace.owners.borrow().verify_retired().unwrap();
                            assert_eq!(trace.callbacks.borrow().len(), at + 1);
                            let before = trace.schedule.borrow().streams.clone();
                            assert!(trace.run(&plan, order, lanes).is_err());
                            assert_eq!(before, trace.schedule.borrow().streams);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn pipeline_rejects_stale_state_in_every_category_before_enqueue() {
        let plan = execution_plan(&[LayerMode::Whole { gpu: 1, ffn: FfnMode::Split }], 0, 0);
        for lane in 0..2 {
            for category in 0..4 {
                let mut trace = PipelineTrace::new(&plan);
                trace.stale = Some((lane, 0, category));
                let error = trace.run(&plan, LaneOrder::Independent, 2).unwrap_err();
                assert!(error.chain().any(|cause| cause.to_string().contains("attention state is bound")));
                assert!(trace.schedule.borrow().streams.iter().all(Vec::is_empty));
                assert!(trace.calls.borrow().is_empty());
                trace.owners.borrow().verify_retired().unwrap();
            }
        }
    }

    #[test]
    fn balanced_deleted_hop_fails_expected_inventory() {
        use super::super::order::{check, Op};
        let plan = execution_plan(&[LayerMode::Whole { gpu: 1, ffn: FfnMode::Split }], 0, 0);
        let trace = PipelineTrace::new(&plan);
        trace.run(&plan, LaneOrder::Independent, 1).unwrap();
        trace.inventory(&plan, 1).unwrap();
        // Delete an entire entry transfer (push AND wait): balance alone passes.
        for stream in &mut trace.schedule.borrow_mut().streams {
            stream.retain(|(op, _)| match op { Op::Push(flag) | Op::Wait(flag) => flag.exchange != "entry" });
        }
        check(&trace.schedule.borrow()).unwrap();
        trace.transfers.borrow_mut().retain(|&(_, index, _)| index != 0);
        assert!(trace.inventory(&plan, 1).is_err());
        let mut consumed = vec![vec![1; plan.hops().len()]];
        consumed[0][0] = 0;
        assert!(validate_hop_inventory(&plan, &consumed).is_err());
        consumed[0][0] = 2;
        assert!(validate_hop_inventory(&plan, &consumed).is_err());
    }

    #[test]
    fn real_pipeline_schedule_drains_every_mixed_owner_interleaving_and_fault_prefix() {
        use super::super::order::check;
        let modes = [LayerMode::HeadSplit, LayerMode::ContextSplit,
            LayerMode::Whole { gpu: 0, ffn: FfnMode::Split }, LayerMode::Whole { gpu: 1, ffn: FfnMode::Split },
            LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }, LayerMode::Whole { gpu: 1, ffn: FfnMode::Owner }];
        for order in [LaneOrder::OrderedChunks, LaneOrder::Independent] {
            for lanes in 1..=2 {
                for entry in 0..2 {
                    for head in 0..2 {
                        let plan = execution_plan(&modes, entry, head);
                        let trace = PipelineTrace::new(&plan);
                        trace.run(&plan, order, lanes).unwrap();
                        trace.inventory(&plan, lanes).unwrap();
                        trace.schedule.borrow_mut().end_request();
                        check(&trace.schedule.borrow()).unwrap();

                        let calls = trace.calls.borrow();
                        assert_eq!(calls.len(), modes.len() * lanes * 3);
                        if order == LaneOrder::OrderedChunks && lanes == 2 {
                            for layer in 0..modes.len() {
                                let done = calls.iter().position(|&call| call == (0, layer, "finish")).unwrap();
                                let next = calls.iter().position(|&(lane, l, kind)| lane == 1 && l == layer && kind != "begin" && kind != "finish").unwrap();
                                assert!(done < next, "shared prefill history must commit before its next chunk");
                            }
                        }
                        trace.owners.borrow_mut().live = false;
                        let before = trace.schedule.borrow().streams.clone();
                        assert!(trace.run(&plan, order, lanes).is_err());
                        assert_eq!(before, trace.schedule.borrow().streams);
                    }
                }
            }
        }
    }

    #[test]
    fn split_protocol_pushes_before_waits_and_combines_in_gpu_order() {
        for rank in 0..2 {
            for indexer in [false, true] {
                let r = Recorder { calls: RefCell::new(Vec::new()), indexer };
                context_decode_with(&r, rank, 7, 1, 6, &mut &r).unwrap();
                let mut expected = vec!["produce".into(), "push-q".into(), "score".into()];
                if indexer { expected.extend(["push-candidates", "wait-candidates", "merge"].map(str::to_owned)); }
                expected.extend(["wait-q".into(), format!("partial{}", 1 - rank), "push-partial".into(),
                    format!("partial{rank}"), "wait-partial".into(), if rank == 0 { "combine1,4" } else { "combine3,2" }.into()]);
                assert_eq!(*r.calls.borrow(), expected);
            }
        }
    }
}
