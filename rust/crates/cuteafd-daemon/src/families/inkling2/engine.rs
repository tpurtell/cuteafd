//! Single-rank and head-parallel arithmetic on the shared request scheduler.
use super::{cache::Placement, experts::Dispatcher, head::Head, layers::Kernels,
    owners::RequestOwners, prefill::{HeadRows, Stateful, TokenIo}, weights::TargetWeights};
use crate::shared::{prefill_pipeline::Unit, spark_intake::SparkLink, token_io::DeviceLogits};
use anyhow::{ensure, Result};
use cuteafd_loader::families::inkling2::Inkling2Config;

use super::graphs::{DecodeGraphs, Mode};
use crate::shared::decode_graph::device::Cycle;

pub(crate) struct Peer<'a> {
    pub(super) graphs: Option<DecodeGraphs<'a>>,
    pub exchange: crate::shared::peer_split::PeerExchange<'a>,
    pub requests: RequestOwners<'a>,
    pub kernels: Kernels<'a>,
    pub target: TargetWeights<'a>,
}
impl<'a> Peer<'a> {
    fn units(&mut self, decode: bool) -> (super::head::PeerHead<'_, 'a>, super::prefill::PeerUnits<'_, 'a>) {
        let (owners, uploads) = if decode {
            (std::slice::from_ref(&self.requests.decode), std::slice::from_mut(&mut self.requests.decode_uploads))
        } else {
            (self.requests.prefill.as_slice(), self.requests.prefill_uploads.as_mut_slice())
        };
        (super::head::PeerHead { kernels: &self.kernels, target: &self.target },
        super::prefill::PeerUnits { exchange: &self.exchange, cache: &self.requests.cache,
            target: &self.target, kernels: &self.kernels, owners, uploads,
            graphs: if decode { self.graphs.as_mut() } else { None } })
    }
}

pub(crate) struct Execution<'a> {
    pub(super) graphs: Option<DecodeGraphs<'a>>,
    graph_cycles: Option<crate::shared::decode_graph::device::EventCycles<'a>>,
    pub link: SparkLink<'a>,
    pub requests: RequestOwners<'a>,
    pub peer: Option<Peer<'a>>,
    pub head: Head<'a>,
    pub target: TargetWeights<'a>,
    pub kernels: Kernels<'a>,
    pub local: Vec<Box<dyn Dispatcher + 'a>>,
    pub cfg: Inkling2Config,
    pub draft:Option<std::rc::Rc<super::draft::Binding<'a>>>,
    pub width_trace:Option<std::cell::RefCell<super::width_invariance::Trace>>,
    ingress: Vec<cuteafd_core::SparkIngress>,
    next_identity: u64,
    terminal: bool,
}
impl<'a> Execution<'a> {
    pub fn request_ranks(&self) -> impl Iterator<Item=&RequestOwners<'a>> {
        std::iter::once(&self.requests).chain(self.peer.as_ref().map(|peer|&peer.requests))
    }
    pub fn restore_verify(&self,mark:&super::verify::TargetMark<'_>)->Result<()> {
        let draft=self.draft.as_ref().ok_or_else(||anyhow::anyhow!("Inkling drafter missing"))?;
        ensure!(mark.conv.len()==draft.verify.len(),"Inkling verify mark rank count differs");
        for ((verify,mark),requests) in draft.verify.iter().zip(&mark.conv).zip(self.request_ranks()) {
            requests.decode.stream.device.run(||verify.restore(mark,requests.cache.conv_backing()))?;
        }
        Ok(())
    }
    pub fn require_live(&self) -> Result<()> {
        ensure!(!self.terminal, "Inkling execution is terminal");
        if let Some(peer) = &self.peer { peer.exchange.require_live()?; }
        Ok(())
    }
    /// Finish a published step or quiesce both ranks for graph upkeep. A
    /// timed-out peer wait can skip itself, so check liveness after draining.
    fn completed(&self) -> Result<()> {
        self.requests.decode.stream.drain()?;
        if let Some(peer) = &self.peer {
            peer.requests.decode.stream.drain()?;
            peer.exchange.require_live()?;
        }
        crate::shared::spark_waves::healthy()
    }
    pub fn new(link: SparkLink<'a>, requests: RequestOwners<'a>, head: Head<'a>,
        target: TargetWeights<'a>, kernels: Kernels<'a>, local: Vec<Box<dyn Dispatcher + 'a>>,
        cfg: Inkling2Config, ingress: Vec<cuteafd_core::SparkIngress>) -> Result<Self> {
        ensure!(target.rank == 0 && target.layers.iter().copied().eq(0..cfg.layers)
            && !requests.prefill.is_empty() && local.len() <= cfg.layers && ingress.len() == cfg.layers,
            "Inkling whole execution differs from admitted owners");
        Ok(Self { graphs: None, graph_cycles: None, link, requests, peer: None, head, target, kernels, local, cfg, ingress, next_identity: 1, terminal: false, draft:None, width_trace:None })
    }

    /// Tokens are submitted suffixes; placement frontiers define their origins
    /// while unit members retain absolute request-local causal spans. Every scratch head is consumed before another overwrites it.
    pub fn prefill(&mut self, placements: &mut [Placement], tokens: &[&[u32]], units: &[Unit],
        version: u64, head_rows: HeadRows, on_logits: &mut dyn FnMut(usize, usize, DeviceLogits) -> Result<()>) -> Result<()> {
        ensure!(!self.terminal && placements.len() == tokens.len(), "Inkling execution terminal or request count differs");
        let slots = placements.iter().map(|placement| placement.slot).collect::<Vec<_>>();
        let starts = placements.iter().map(|placement| placement.len).collect::<Vec<_>>();
        let media = placements.iter().map(|placement| placement.media.clone()).collect::<Vec<_>>();
        let result = (|| {
            let waves = self.link.exchange()?;
            let (peer_head, peer_units) = self.peer.as_mut().map(|peer| peer.units(false)).unzip();
            let mut io = TokenIo { kernels: &self.kernels, head: &self.head, peer_head, cache: &self.requests.cache,
                target: &self.target, cfg: &self.cfg, tokens, media: &media, starts: &starts, slots: &slots, retained_rows: false, head_rows, draft:self.draft.as_deref(),publish_draft:true,trace:None,on_logits };
            Stateful { graphs: None, peer: peer_units, kernels: &self.kernels, owners: &self.requests.prefill, cache: &self.requests.cache,
                target: &self.target, cfg: &self.cfg, ingress: &self.ingress, placements, uploads: &mut self.requests.prefill_uploads,
                waves, onboarded: self.local.len(), local: &self.local, next_identity: &mut self.next_identity,
                placement_version: version, kind: cuteafd_transport::ExpertV2SourceKind::Prefill, io: &mut io }.run(units)?;
            self.completed()
        })();
        if result.is_err() { self.terminate(); }
        result
    }
    /// Decode and verify use the admitted decode owner and shared unit executor. Each member's output row remains in the one bounded head.
    pub fn decode(&mut self, placements: &mut [Placement], tokens: &[&[u32]], unit: &Unit,
        version: u64) -> Result<DeviceLogits> {
        self.decode_inner(placements, tokens, unit, version, false)
    }

    /// A verify proposal stays private until its accepted-prefix commit.
    /// The caller consumes logits on their producing stream before committing.
    pub fn verify(&mut self, placements: &mut [Placement], tokens: &[&[u32]], unit: &Unit,
        version: u64) -> Result<DeviceLogits> {
        self.decode_inner(placements, tokens, unit, version, true)
    }

    fn decode_inner(&mut self, placements: &mut [Placement], tokens: &[&[u32]], unit: &Unit,
        version: u64, staged: bool) -> Result<DeviceLogits> {
        self.require_live()?;
        ensure!(!placements.is_empty() && placements.len() == tokens.len()
            && unit.rows.len() <= self.requests.decode.layout.rows && unit.slot == 0, "Inkling decode exceeds admitted head/owner");
        let mode = if self.requests.decode.replaying() { Mode::Verify } else { Mode::Decode };
        let cycle = if mode == Mode::Verify { Cycle::Verify } else { Cycle::Decode };
        self.for_each_graph(|graphs| graphs.cycle(cycle));
        let slots = placements.iter().map(|placement| placement.slot).collect::<Vec<_>>();
        let starts = placements.iter().map(|placement| placement.len).collect::<Vec<_>>();
        let media = placements.iter().map(|placement| placement.media.clone()).collect::<Vec<_>>();
        let result = (|| {
            self.begin_graph_cycle()?;
            let waves = self.link.exchange()?;
            let (peer_head, peer_units) = self.peer.as_mut().map(|peer| peer.units(true)).unzip();
            let mut io = TokenIo { kernels: &self.kernels, head: &self.head, peer_head, cache: &self.requests.cache,
                target: &self.target, cfg: &self.cfg, tokens, media: &media, starts: &starts, slots: &slots, retained_rows: true, head_rows: HeadRows::All, draft:self.draft.as_deref(),publish_draft:false,trace:self.width_trace.as_ref(),
                on_logits: &mut |_, _, _| Ok(()) };
            Stateful { graphs: self.graphs.as_mut(), peer: peer_units, kernels: &self.kernels, owners: std::slice::from_ref(&self.requests.decode),
                cache: &self.requests.cache, target: &self.target, cfg: &self.cfg, ingress: &self.ingress, placements,
                uploads: std::slice::from_mut(&mut self.requests.decode_uploads), waves,
                onboarded: self.local.len(), local: &self.local, next_identity: &mut self.next_identity,
                placement_version: version, kind: cuteafd_transport::ExpertV2SourceKind::Decode, io: &mut io }.run(std::slice::from_ref(unit))?;
            let first_touch = self.graphs.as_ref().is_some_and(DecodeGraphs::first_touch_pending)
                || self.peer.as_ref().and_then(|p| p.graphs.as_ref()).is_some_and(DecodeGraphs::first_touch_pending);
            self.for_each_graph(DecodeGraphs::tick);
            if let Some(cycles) = &mut self.graph_cycles {
                // SAFETY: both graph owner streams have queued the whole decode.
                unsafe { cycles.complete()?; }
            }
            if !staged || first_touch {
                // Plain decode publishes state; first touch initializes lazy native storage.
                let wait = std::time::Instant::now();
                self.completed()?;
                if let Some(cycles) = &mut self.graph_cycles {
                    if staged && first_touch { cycles.record_drain(wait.elapsed()); }
                    cycles.poll()?;
                }
                self.for_each_graph(DecodeGraphs::observed);
            }
            Ok(DeviceLogits { ptr: self.head.logits.buffer.ptr, rows: unit.rows.len(),
                vocab: self.cfg.unpadded_vocab_size, stride: self.cfg.unpadded_vocab_size,
                stream: self.requests.decode.stream.raw, greedy: None })
        })();
        if result.is_err() { self.terminate(); }
        result
    }
    fn for_each_graph(&mut self, mut f: impl FnMut(&mut DecodeGraphs<'a>)) {
        if let Some(graphs) = &mut self.graphs { f(graphs); }
        if let Some(graphs) = self.peer.as_mut().and_then(|peer| peer.graphs.as_mut()) { f(graphs); }
    }
    pub(super) fn enable_graph_cycles(&mut self) -> Result<()> {
        if self.graphs.is_some() {
            let streams = self.request_ranks().map(|r| (r.decode.stream.device.id, r.decode.stream.raw)).collect::<Vec<_>>();
            self.graph_cycles = Some(crate::shared::decode_graph::device::EventCycles::new(self.requests.decode.stream.device.library, &streams)?);
        }
        Ok(())
    }
    fn begin_graph_cycle(&mut self) -> Result<()> {
        let Some(cycles) = &mut self.graph_cycles else { return Ok(()) };
        // SAFETY: graph owners borrow persistent request storage.
        unsafe { cycles.begin_persistent(|visit| {
            if let Some(graphs) = &mut self.graphs { graphs.visit(visit)?; }
            if let Some(graphs) = self.peer.as_mut().and_then(|p| p.graphs.as_mut()) { graphs.visit(visit)?; }
            Ok(())
        }) }
    }
    pub fn terminate(&mut self) {
        if self.terminal { return; }
        self.terminal = true;
        // Every owner that queues peer waits on either rank's streams: the head
        // split, each TP2 expert owner and the TP2 drafter (#2543).
        use crate::shared::peer_split::TerminalOwner;
        let mut owners: Vec<&dyn TerminalOwner> = Vec::new();
        if let Some(peer) = &self.peer { owners.push(&peer.exchange); }
        owners.extend(self.local.iter().filter_map(|local| local.terminal()));
        if let Some(draft) = &self.draft { owners.push(&*draft.drafter); }
        let link = std::cell::RefCell::new(&mut self.link);
        let drained = crate::shared::peer_split::terminal_sequence(&owners, || link.borrow_mut().terminal_quiesce(), || {
            if let Some(peer) = &self.peer { peer.requests.cache.drain()?; }
            self.requests.decode.stream.drain()?;
            for owner in &self.requests.prefill { owner.stream.drain()?; }
            self.requests.cache.drain()?;
            let mut link = link.borrow_mut();
            link.terminal_drain_copies()?;
            link.terminal_release()
        });
        crate::shared::decode_graph::fatal_drain(drained, "Inkling request executor retirement");
        self.for_each_graph(DecodeGraphs::clear_drained);
    }
}
impl Drop for Execution<'_> {
    fn drop(&mut self) { self.terminate(); }
}

#[cfg(test)]
mod tests {
    #[test]
    fn request_executor_reuses_shared_causal_units_and_terminal_transport_order() {
        let source = include_str!("engine.rs").split("#[cfg(test)]").next().unwrap();
        assert!(source.contains("Stateful {"));
        assert!(source.contains(".run(units)"));
        assert!(source.contains("self.link.exchange()?"));
        assert!(source.contains("cycles.begin_persistent("));
        assert!(!source.split("pub fn decode(").nth(1).unwrap().contains("begin_capture("));
        let terminal = source.split("pub fn terminate(").nth(1).unwrap();
        // #2543: the head split, every TP2 expert owner and the drafter are aborted together.
        for owner in ["owners.push(&peer.exchange)", "self.local.iter().filter_map(|local| local.terminal())",
            "owners.push(&*draft.drafter)", "terminal_sequence(&owners"] {
            assert!(terminal.contains(owner), "{owner}");
        }
        assert!(terminal.find("terminal_quiesce").unwrap() < terminal.find("decode.stream.drain").unwrap());
        assert!(terminal.find("cache.drain").unwrap() < terminal.find("terminal_release").unwrap());
        assert!(!source.split("pub fn decode(").nth(1).unwrap().contains("for layer in"));
    }
}
