//! Shared per-layer attention orchestration. Families provide kernel bindings,
//! not their own ordering of peer pushes, waits or LSE operands.
use super::context::{combine_order, slot, ContextExchange};
use super::{hop::HopLink, PeerExchange, DIRECT};
use anyhow::{ensure, Result};
use cuteafd_loader::placement::{FfnMode, Hop, HopPoint, LayerMode, Placement};
use std::ffi::c_void;

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

trait LayerProtocol {
    type Ticket;
    fn require_live(&self) -> Result<()>;
    fn push_entry(&self, from: usize, source: *const c_void, destination: *mut c_void, bytes: usize) -> Result<()>;
    fn wait_entry(&self, to: usize) -> Result<()>;
    fn send(&self, index: usize, lane: usize, source: *const c_void, rows: usize) -> Result<Self::Ticket>;
    fn land(&self, ticket: Self::Ticket) -> Result<*mut c_void>;
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
            pool_tokens: 0, attention_placement: cuteafd_loader::placement::AttentionPlacement::Layers,
            attention_by_kind: vec![], peer_row_bytes: 0, onboard_layers: 0,
            layers: modes.iter().map(|&mode| LayerAssignment { mode, experts: ExpertHome::Dense }).collect(),
            movables: vec![], expert_ranges: vec![], residual: vec![], tp2: None, items: vec![],
            hops: plan_hops(modes, &HopSpec { row_bytes: 32, rows: 6, lanes: 2, entry_gpu: 0, head_gpu: 0 }),
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
