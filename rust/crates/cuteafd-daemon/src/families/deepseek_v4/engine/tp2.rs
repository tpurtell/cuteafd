//! Head-split RTX routed halves. Scratch is shared on each rank's single
//! stream; only the per-lane fused payload survives until the FFN close.
use super::*;
use crate::shared::experts::rtx::{Combine, ExchangeDtype, ExpertInput, FusedCombine,
    PartialDtype, RouteCheck, RouteIdentity, RouteSource, Routes, RtxExpertLayer, RtxShard};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::ops::Range;
use crate::shared::peer_split::order::{self, Schedule};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExchangePolicy { Auto, Bf16, F32 }

impl ExchangePolicy {
    pub(crate) fn from_env() -> Result<Self> {
        Self::parse(std::env::var("CUTEAFD_TP2_EXCHANGE").ok().as_deref().unwrap_or("auto"))
    }

    fn parse(value: &str) -> Result<Self> {
        Ok(match value {
            "" | "auto" => Self::Auto,
            "bf16" => Self::Bf16,
            "f32" => Self::F32,
            other => anyhow::bail!("CUTEAFD_TP2_EXCHANGE must be auto, bf16 or f32, got {other:?}"),
        })
    }

    pub(super) fn is_f32(self, rows: usize, decode_rows: usize) -> bool {
        self.dtype(rows, decode_rows) == ExchangeDtype::F32
    }

    fn dtype(self, _rows: usize, _decode_rows: usize) -> ExchangeDtype {
        match self {
            Self::F32 => ExchangeDtype::F32,
            // Flash: BF16 stays near the atomic self-KL floor; F32 adds KL without a decode win.
            Self::Auto | Self::Bf16 => ExchangeDtype::Bf16,
        }
    }

    pub(super) fn slot_bytes(self, prefill: usize, decode: usize, hidden: usize) -> usize {
        match self {
            Self::F32 => prefill.max(decode) * hidden * 4,
            // Reserve decode F32 even under the BF16 override: identical admission.
            _ => (prefill * hidden * 2).max(decode * hidden * 4),
        }
    }
}

struct PendingCheck<'a> {
    rows: usize,
    ranks: [Option<HostAllocation<'a>>; 2],
}

#[derive(Debug, Clone, Copy)]
struct PackedRoutes {
    section_bytes: usize,
    weights_at: usize,
    push_bytes: usize,
}

impl PackedRoutes {
    fn new(rows: usize, topk: usize) -> Self {
        let section_bytes = rows * topk * 4;
        let weights_at = section_bytes.next_multiple_of(16);
        let push_bytes = (weights_at + section_bytes).next_multiple_of(16);
        Self { section_bytes, weights_at, push_bytes }
    }
}

pub(super) struct Tp2State<'a> {
    pub(super) layers: Vec<usize>,
    ranks: [RefCell<Box<dyn RtxExpertLayer + 'a>>; 2],
    combine: [FusedCombine<'a>; 2],
    routes: PeerExchange<'a>,
    source: Cell<RouteSource>,
    pub(super) checking: Cell<bool>,
    check_units: RefCell<Vec<usize>>,
    checks: RefCell<BTreeMap<(usize, usize), PendingCheck<'a>>>,
    schedule: RefCell<Option<Schedule>>,
    identity: RefCell<RouteIdentity>,
    interval: usize,
    steps: Cell<usize>,
    fallbacks: Cell<u64>,
    streams: [*mut c_void; 2],
    output_pending: [Cell<bool>; 2],
}

impl Tp2State<'_> {
    pub(super) fn broadcast(&self) -> bool { self.source.get() == RouteSource::Broadcast }

    pub(super) fn group_rows(&self, rows: usize) -> Result<usize> {
        let group = self.ranks[0].borrow().group_rows(rows)?;
        ensure!(group == self.ranks[1].borrow().group_rows(rows)?, "TP2 resource grouping differs by rank");
        Ok(group)
    }
}

impl<'a> Engine<'a> {
    pub fn install_tp2(&mut self, ranks: [Box<dyn RtxExpertLayer + 'a>; 2]) -> Result<()> {
        ensure!(self.tp2.is_none(), "TP2 experts already installed");
        let peer = self.peer()?;
        let layers = ranks[0].layers().to_vec();
        ensure!(layers == ranks[1].layers() && !layers.is_empty() && layers.iter().all(|&id| id < self.cfg.n_layers),
            "TP2 ranks must hold the same nonempty backbone prefix");
        ensure!(layers.iter().copied().eq(0..layers.len()), "V4 TP2 currently requires a prefix starting at layer 0");
        for (rank, experts) in ranks.iter().enumerate() {
            ensure!(experts.shard() == RtxShard::Tp2 { rank: rank as u8 }
                && experts.partial() == PartialDtype::F32
                && experts.device() == if rank == 0 { self.device } else { peer.device },
                "TP2 rank {rank} shard, dtype or device differs from head split");
        }
        ensure!(self.local.borrow().as_ref().is_none_or(|l| l.layers().is_empty()),
            "TP1 backbone experts cannot coexist with TP2");
        let rows = self.prefill_rows.max(self.decode_rows);
        let routes = PeerExchange::new_abortable(self.library,
            [RankDevice { device: self.device, stream: self.stream },
             RankDevice { device: peer.device, stream: peer.stream }],
            2 * self.prefill_waves, rows * self.cfg.n_activated_experts * 8 + 16)?;
        let interval = std::env::var("CUTEAFD_ROUTE_CHECK").ok()
            .map(|v| v.parse::<usize>().context("CUTEAFD_ROUTE_CHECK must be a nonnegative step interval"))
            .transpose()?.unwrap_or(0);
        let combine = [FusedCombine::new(self.library, ExchangeDtype::Bf16)?,
            FusedCombine::new(self.library, ExchangeDtype::F32)?];
        self.tp2 = Some(Tp2State { layers: layers.clone(), ranks: ranks.map(RefCell::new), combine, routes,
            source: Cell::new(RouteSource::Replicated), checking: Cell::new(false), check_units: RefCell::new(Vec::new()), checks: RefCell::new(BTreeMap::new()),
            schedule: RefCell::new(None), identity: RefCell::new(RouteIdentity::default()),
            interval, steps: Cell::new(0), fallbacks: Cell::new(0),
            streams: [self.stream, peer.stream], output_pending: [Cell::new(false), Cell::new(false)] });
        tracing::info!(layers = ?layers, "V4 RTX TP2 expert halves resident on both GPUs");
        cuteafd_bench::context::set_resolved("rtx-tp2-layers", &format!("{layers:?}"));
        Ok(())
    }

    fn publish_tp2_abort(&self) -> Result<()> {
        let mut failures = Vec::new();
        for exchange in self.exchange.iter().chain(self.tp2.iter().map(|tp2| &tp2.routes)) {
            match exchange.terminal_state() {
                crate::shared::peer_split::TerminalState::Aborting
                | crate::shared::peer_split::TerminalState::Drained => continue,
                _ => {}
            }
            if let Err(error) = exchange.publish_abort() { failures.push(format!("{error:#}")); }
        }
        ensure!(failures.is_empty(), "V4 TP2 abort publication: {}", failures.join("; "));
        Ok(())
    }

    fn abort_tp2_or_terminate(&self) {
        // enqueue may fail inside rank 1's device scope; legacy exchange.on(0)
        // assumes the home device is current before publishing either word.
        let home = crate::shared::memory::device::Device { library: self.library, id: self.device };
        crate::shared::decode_graph::fatal_drain(home.run(|| self.publish_tp2_abort()), "V4 RTX enqueue abort");
    }

    pub(crate) fn require_live(&self) -> Result<()> {
        use crate::shared::peer_split::{TerminalPeerError, TerminalState};
        let state = self.terminal.get();
        if state != TerminalState::Active { return Err(TerminalPeerError::Closed(state).into()); }
        if let Some(exchange) = &self.exchange { exchange.require_live()?; }
        if let Some(tp2) = &self.tp2 { tp2.routes.require_live()?; }
        Ok(())
    }

    pub(crate) fn shutdown_submission(&self, transports: &mut [SparkLink<'_>]) -> Result<()> {
        use crate::shared::{experts::rtx::pair::{teardown, Teardown}, peer_split::TerminalState};
        if self.terminal.get() == TerminalState::Drained { return Ok(()); }
        self.terminal.set(TerminalState::Aborting);
        let transports = RefCell::new(transports);
        let result = teardown(&mut Teardown {
            abort: || {
                let home = crate::shared::memory::device::Device { library: self.library, id: self.device };
                home.run(|| self.publish_tp2_abort())?;
                // The proxy cannot abort its completion waits yet. Refuse before
                // rank drainage rather than blocking on an unprovable consumer.
                ensure!(self.device_link.is_none(), "device Spark proxy requires terminal owner retention");
                Ok(())
            },
            ranks: || {
                let mut failures = Vec::new();
                for rank in 0..(1 + usize::from(self.peer.is_some())) {
                    // SAFETY: queued operands remain owned throughout shared teardown.
                    if let Err(error) = self.on(rank, || unsafe { self.library.cuda_stream_synchronize(self.stream_of(rank)) }) {
                        failures.push(format!("rank {rank}: {error:#}"));
                    }
                }
                ensure!(failures.is_empty(), "{}", failures.join("; ")); Ok(())
            },
            links: || {
                let mut failures = Vec::new();
                for link in transports.borrow_mut().iter_mut() {
                    if let Err(error) = link.terminal_quiesce() { failures.push(format!("{error:#}")); }
                }
                ensure!(failures.is_empty(), "{}", failures.join("; ")); Ok(())
            },
            copies: || {
                let mut failures = Vec::new();
                for link in transports.borrow().iter() {
                    if let Err(error) = link.terminal_drain_copies() { failures.push(format!("{error:#}")); }
                }
                ensure!(failures.is_empty(), "{}", failures.join("; ")); Ok(())
            },
            graphs: || {
                for link in transports.borrow_mut().iter_mut() { link.terminal_release()?; }
                for rank in 0..(1 + usize::from(self.peer.is_some())) {
                    self.on(rank, || {
                        if rank == 0 { self.graphs.borrow_mut().clear(); }
                        else { self.peer()?.graphs.borrow_mut().clear(); }
                        Ok(())
                    })?;
                }
                Ok(())
            },
            finish: |drained| {
                self.terminal.set(if drained { TerminalState::Drained } else { TerminalState::Retained });
                for exchange in self.exchange.iter().chain(self.tp2.iter().map(|tp2| &tp2.routes)) {
                    exchange.finish_terminal(drained);
                }
            },
        });
        crate::shared::decode_graph::fatal_drain(result, "V4 submission shutdown");
        Ok(())
    }

    pub(super) fn tp2_layer(&self, layer: usize) -> bool {
        !self.skip_routed && self.tp2.as_ref().is_some_and(|t| t.layers.contains(&layer))
    }

    pub(super) fn tp2_layer_weights(&self, weights: &LayerWeights<'_>) -> bool {
        self.peer.as_ref().is_some_and(|peer| peer.layers.iter().position(|w| std::ptr::eq(w, weights))
            .is_some_and(|layer| self.tp2_layer(layer)))
    }

    fn combine(&self, rows: usize) -> Result<&FusedCombine<'a>> {
        let dtype = self.exchange_policy.dtype(rows, self.decode_rows);
        let Combine::FusedAllReduce { exchange } = (Combine::FusedAllReduce { exchange: dtype }) else { unreachable!() };
        Ok(&self.tp2.as_ref().context("TP2 experts")?.combine[usize::from(exchange == ExchangeDtype::F32)])
    }

    pub(super) fn tp2_key(&self, layer: usize, rows: usize, index: usize) -> GraphKey {
        let mut key = Self::expert_key(layer, rows, index);
        key.attention = if self.tp2.as_ref().is_some_and(|t| t.source.get() == RouteSource::Broadcast) {
            "tp2-broadcast"
        } else { "tp2-replicated" };
        key
    }

    /// The indivisible sequence enqueue -> partial -> push protects the single
    /// expert output from the next lane. All three use the executor's one stream.
    pub(super) fn tp2_experts(&self, rank: usize, layer: usize, index: usize, w: &Workspace<'_>, rows: usize)
        -> Result<()> {
        let tp2 = self.tp2.as_ref().context("TP2 experts")?;
        let lane = &w.slots[index];
        let payload = lane.payload.as_ref().context("TP2 lane payload")?;
        let stream = self.stream_of(rank);
        ensure!(stream == tp2.streams[rank], "TP2 executor moved streams; per-lane scratch ownership is required");
        debug_assert!(!tp2.output_pending[rank].get(), "previous routed output was not copied to its lane payload");
        self.collect_routes(rank, layer, index, rows, w)?;
        self.on(rank, || {
            let mut routes = Routes { ids: w.route_ids.buffer.ptr, weights: w.route_weights.buffer.ptr };
            if tp2.source.get() == RouteSource::Broadcast {
                let slot = 2 * index + layer % 2;
                let packed = PackedRoutes::new(rows, self.cfg.n_activated_experts);
                if rank == 0 {
                    // The scores are consumed: their larger buffer doubles as route packing scratch.
                    ensure!(packed.push_bytes <= w.logits.buffer.bytes, "router scratch cannot hold canonical routes");
                    // SAFETY: both sections fit the rank's logits allocation; zero padding and copies precede the push.
                    unsafe {
                        self.library.cuda_zero_bytes_async(w.logits.buffer, packed.push_bytes, stream)?;
                        self.library.copy_d2d_async(w.logits.buffer, w.route_ids.buffer, packed.section_bytes, stream)?;
                        self.library.copy_d2d_async(cuteafd_ffi::CuteafdDeviceBuffer {
                            ptr: w.logits.buffer.ptr.cast::<u8>().add(packed.weights_at).cast(),
                            bytes: w.logits.buffer.bytes - packed.weights_at, ..w.logits.buffer
                        }, w.route_weights.buffer, packed.section_bytes, stream)?;
                    }
                    self.record_peer(0, "routes", slot, true, layer, index);
                    tp2.routes.push(0, slot, w.logits.buffer.ptr, packed.push_bytes)?;
                } else {
                    self.record_peer(1, "routes", slot, false, layer, index);
                    tp2.routes.wait(1, slot)?;
                    let received = tp2.routes.recv(1, slot)?;
                    // SAFETY: the received slot contains ids then aligned weights, published by rank 0.
                    routes = Routes { ids: received, weights: unsafe { received.cast::<u8>().add(packed.weights_at).cast() } };
                }
            }
            let combine = self.combine(rows)?;
            ensure!(combine.payload_bytes(rows, self.cfg.dim) <= payload.buffer.bytes, "TP2 payload exceeds admission");
            let mut experts = tp2.ranks[rank].borrow_mut();
            // SAFETY: routes and locally quantized wire rows are complete on this rank's stream.
            unsafe { experts.enqueue(layer, rows, ExpertInput::Fp8K32(w.wire.buffer.ptr), routes, stream,
                &mut || self.abort_tp2_or_terminate())?; }
            tp2.output_pending[rank].set(true);
            // SAFETY: owned FP32 output and shared BF16 rows remain live through this stream-ordered copy.
            let copied = unsafe { combine.partial(experts.output().cast(), lane.shared.buffer.ptr.cast(),
                payload.buffer.ptr, rows, self.cfg.dim, stream) };
            copied?;
            tp2.output_pending[rank].set(false);
            // Publish before either stream queues its matching wait; no wait in the local expert path.
            if rank == 1 || layer + 1 < self.cfg.n_layers {
                self.record_peer(rank, "ffn", slot(layer, true, index), true, layer, index);
                self.exchange()?.push(rank, slot(layer, true, index), payload.buffer.ptr,
                    combine.payload_bytes(rows, self.cfg.dim))?;
            }
            Ok(())
        })
    }

    pub(super) fn tp2_post(&self, rank: usize, layer: usize, index: usize, w: &Workspace<'_>, lane: &UnitSlot<'_>, rows: usize)
        -> Result<()> {
        let exchange = self.exchange()?;
        let at = slot(layer, true, index);
        self.record_peer(rank, "ffn", at, false, layer, index);
        exchange.wait(rank, at)?;
        let received = exchange.recv(rank, at)?;
        let payload = lane.payload.as_ref().context("TP2 lane payload")?;
        let (rank0, rank1) = if rank == 0 { (payload.buffer.ptr, received) } else { (received, payload.buffer.ptr) };
        self.on(rank, || {
            // SAFETY: both payloads are complete, same dtype, summed rank0 first on both GPUs.
            unsafe { self.combine(rows)?.sum(rank0, rank1, w.sum.buffer.ptr.cast(), rows, self.cfg.dim, self.stream_of(rank)) }
        })?;
        self.run_on(rank, false, "mhc_post", &[
            ("x", w.sum.buffer.ptr), ("residual", lane.stream_b.buffer.ptr), ("prev_post", lane.post.buffer.ptr),
            ("prev_comb", lane.comb.buffer.ptr), ("out", lane.stream_a.buffer.ptr),
        ], &[Scalar::I32(rows as i32)])
    }

    pub(super) fn prepare_route_checks(&self, rows: &[usize]) -> Result<()> {
        let Some(tp2) = &self.tp2 else { return Ok(()) };
        if !tp2.checking.get() { return Ok(()); }
        *tp2.check_units.borrow_mut() = (0..rows.len()).collect();
        // cudaHostAlloc may synchronize. Allocate before queuing any peer waits,
        // never from a layer whose matching push has not been enqueued yet.
        let mut checks = tp2.checks.borrow_mut();
        ensure!(checks.is_empty(), "previous route diagnostics did not drain");
        for &layer in &tp2.layers {
            for (lane, &rows) in rows.iter().enumerate() {
                let bytes = rows * self.cfg.n_activated_experts * 8;
                let ranks = [Some(self.on(0, || HostAllocation::new(self.library, bytes))?),
                    Some(self.on(1, || HostAllocation::new(self.library, bytes))?)];
                checks.insert((layer, lane), PendingCheck { rows, ranks });
            }
        }
        Ok(())
    }

    pub(super) fn route_check_unit(&self, slot: usize, unit: usize) {
        if let Some(tp2) = self.tp2.as_ref().filter(|t| t.checking.get()) {
            tp2.check_units.borrow_mut()[slot] = unit;
        }
    }

    fn collect_routes(&self, rank: usize, layer: usize, lane: usize, rows: usize, w: &Workspace<'_>) -> Result<()> {
        let tp2 = self.tp2.as_ref().context("TP2 experts")?;
        if !tp2.checking.get() || self.capture_only.get() { return Ok(()); }
        self.on(rank, || {
            let bytes = rows * self.cfg.n_activated_experts * 4;
            let checks = tp2.checks.borrow();
            let check = checks.get(&(layer, tp2.check_units.borrow()[lane])).context("route diagnostic storage not prepared")?;
            ensure!(check.rows == rows, "route-check rows changed");
            let host = check.ranks[rank].as_ref().context("route diagnostic rank buffer")?;
            // SAFETY: pinned host storage is retained until both streams drain in finish_route_step.
            unsafe {
                self.library.copy_d2h_host_buffer_async(host.buffer, w.route_ids.buffer, bytes, self.stream_of(rank))?;
                self.library.copy_d2h_host_buffer_async(cuteafd_ffi::CuteafdHostBuffer {
                    ptr: host.buffer.ptr.cast::<u8>().add(bytes).cast(), bytes, ..host.buffer
                }, w.route_weights.buffer, bytes, self.stream_of(rank))?;
            }
            Ok(())
        })
    }

    pub(super) fn record_peer(&self, rank: usize, exchange: &'static str, slot: usize,
        push: bool, layer: usize, lane: usize) {
        let Some(tp2) = &self.tp2 else { return };
        let mut schedule = tp2.schedule.borrow_mut();
        let Some(schedule) = schedule.as_mut() else { return };
        let label = format!("gpu{rank} {exchange} {} L{layer} lane{lane}", if push { "push" } else { "wait" });
        if exchange == "unit-commit" {
            // Completion has no payload; distinct slots may publish before either joins.
            if push { schedule.push_to(rank, exchange, slot, lane, label); }
            else { schedule.wait_for(rank, exchange, slot, lane, label); }
        } else if push { schedule.push(rank, exchange, slot, label); }
        else { schedule.wait(rank, exchange, slot, label); }
    }

    pub(super) fn count_route_fallback(&self, layer: usize) {
        if let Some(tp2) = self.tp2.as_ref().filter(|t| t.layers.contains(&layer)
            && t.source.get() == RouteSource::Broadcast && !self.capture_only.get()) {
            // Count host-enqueued units, including graph replays (not graph construction).
            tp2.fallbacks.set(tp2.fallbacks.get() + 1);
        }
    }

    pub(super) fn begin_route_step(&self) {
        if let Some(tp2) = &self.tp2 {
            let step = tp2.steps.get() + 1;
            tp2.steps.set(step);
            if tp2.interval > 0 && step % tp2.interval == 0 { tp2.checking.set(true); }
        }
    }

    pub(super) fn finish_route_step(&self) -> Result<()> {
        let Some(tp2) = &self.tp2 else { return Ok(()) };
        self.publish_route_source();
        if !tp2.checking.replace(false) { return Ok(()); }
        for rank in 0..2 {
            self.on(rank, || {
                // SAFETY: every wait of the completed step has its paired push enqueued.
                unsafe { self.library.cuda_stream_synchronize(self.stream_of(rank)) }
            })?;
        }
        let checks = std::mem::take(&mut *tp2.checks.borrow_mut()).into_iter().map(|((layer, _), check)| {
            let [rank0, rank1] = check.ranks;
            Ok(RouteCheck { layer, rows: check.rows, topk: self.cfg.n_activated_experts,
                ranks: [rank0.context("rank0 route check missing")?.bytes().to_vec(),
                    rank1.context("rank1 route check missing")?.bytes().to_vec()] })
        }).collect::<Result<Vec<_>>>()?;
        let identity = RouteIdentity::check(&checks);
        if identity.source() == RouteSource::Broadcast { tp2.source.set(RouteSource::Broadcast); }
        tracing::info!(layers_checked = identity.layers_checked, rows_checked = identity.rows_checked,
            first_mismatch = ?identity.first_mismatch, source = ?tp2.source.get(), fallbacks = tp2.fallbacks.get(),
            "V4 TP2 route identity");
        let mut aggregate = tp2.identity.borrow_mut();
        aggregate.layers_checked += identity.layers_checked;
        aggregate.rows_checked += identity.rows_checked;
        if aggregate.first_mismatch.is_none() { aggregate.first_mismatch = identity.first_mismatch; }
        cuteafd_bench::context::set_resolved("route-identity", &format!("layers={} rows={} mismatch={:?}",
            aggregate.layers_checked, aggregate.rows_checked, aggregate.first_mismatch));
        drop(aggregate);
        self.publish_route_source();
        if let Some(schedule) = tp2.schedule.borrow_mut().take() {
            order::check(&schedule).map_err(|e| anyhow::anyhow!("V4 TP2 recorded schedule: {e}"))?;
            tracing::info!(ops = schedule.streams.iter().map(Vec::len).sum::<usize>(),
                "V4 TP2 recorded push/wait schedule drains");
        }
        Ok(())
    }

    fn publish_route_source(&self) {
        cuteafd_bench::context::set_resolved("v4-graph-captures", &self.graph_captures.get().to_string());
        cuteafd_bench::context::set_resolved("tp2-expert-graph-captures", &self.expert_graph_captures.get().to_string());
        if let Some(tp2) = &self.tp2 {
            cuteafd_bench::context::set_resolved("route-source",
                if tp2.source.get() == RouteSource::Replicated { "replicated" } else { "broadcast" });
            cuteafd_bench::context::set_resolved("route-fallbacks", &tp2.fallbacks.get().to_string());
        }
    }

    pub fn check_routes(&self, transports: &mut [SparkLink<'_>], runtime: &tokio::runtime::Runtime) -> Result<()> {
        let Some(tp2) = &self.tp2 else { return Ok(()) };
        let mut allocator = super::super::pool::PoolAllocator::for_engine(self)?;
        // Both 256-row lanes are checked independently before their route buffers are reused.
        let tokens: Vec<u32> = (0..512).map(|i| (i % self.cfg.vocab_size) as u32).collect();
        let mut placement = allocator.admit(tokens.len() + 1)?;
        tp2.checking.set(true);
        *tp2.schedule.borrow_mut() = Some(Schedule::default());
        self.prefill(&mut placement, &tokens, transports, runtime, 0, None)?;
        tp2.checking.set(true);
        *tp2.schedule.borrow_mut() = Some(Schedule::default());
        self.decode(&mut [(&mut placement, 1)], transports.first_mut(), runtime)?;
        allocator.release(placement);
        // Cache/state writes were to scratch placement; released slots are overwritten on admission.
        self.warm_broadcast_graphs()?;
        self.publish_route_source();
        Ok(())
    }

    pub(crate) fn graph_capture_counts(&self) -> (usize, usize) {
        (self.graph_captures.get(), self.expert_graph_captures.get())
    }

    pub(super) fn warm_tp2_routers(&self) -> Result<()> {
        let Some(tp2) = &self.tp2 else { return Ok(()) };
        for rank in 0..2 {
            let workspace = if rank == 0 { self.decode_workspace.borrow() }
                else { self.peer()?.decode_workspace.borrow() };
            let w = workspace.as_ref().context("router warm-up workspace")?;
            let lane = &w.slots[0];
            let layers = if rank == 0 { &self.weights.layers } else { &self.peer()?.layers };
            self.on(rank, || {
                // CUDA runtime kernels load lazily, which may synchronize the device.
                // Load both router branches and combine dtypes before any unmatched wait.
                for allocation in [&w.y, &lane.tokens, &lane.shared] {
                    self.library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?;
                }
                for hash in [true, false] {
                    let Some(weights) = layers.iter().find(|w| w.hash == hash) else { continue };
                    self.run_on(rank, false, "router_scores", &[
                        ("x", w.y.buffer.ptr), ("w", weights.ptr("gate")?), ("logits", w.logits.buffer.ptr),
                    ], &[Scalar::I32(1)])?;
                    let (bias, table) = if hash { (std::ptr::null_mut(), weights.ptr("gate.tid2eid")?) }
                        else { (weights.ptr("gate.bias")?, std::ptr::null_mut()) };
                    // SAFETY: initialized persistent rows and rank-local route buffers, no peer work queued.
                    unsafe { self.library.dsv4_router_select(w.logits.buffer.ptr, bias, table, lane.tokens.buffer.ptr,
                        w.route_ids.buffer.ptr, w.route_weights.buffer.ptr, 1, self.cfg.n_routed_experts,
                        self.cfg.n_activated_experts, self.cfg.route_scale as f32, self.stream_of(rank))?; }
                }
                self.quantize_input(rank, w, 1)?;
                let payload = lane.payload.as_ref().context("combine warm-up payload")?;
                for combine in &tp2.combine {
                    // SAFETY: zero shared rows, persistent payload and disjoint output, all on this rank.
                    unsafe {
                        combine.partial(std::ptr::null(), lane.shared.buffer.ptr.cast(), payload.buffer.ptr,
                            1, self.cfg.dim, self.stream_of(rank))?;
                        combine.sum(payload.buffer.ptr, payload.buffer.ptr, w.sum.buffer.ptr.cast(),
                            1, self.cfg.dim, self.stream_of(rank))?;
                    }
                }
                // SAFETY: startup-only independent work; neither stream has a peer wait yet.
                unsafe { self.library.cuda_stream_synchronize(self.stream_of(rank)) }
            })?;
        }
        Ok(())
    }

    fn warm_broadcast_graphs(&self) -> Result<()> {
        let Some(tp2) = &self.tp2 else { return Ok(()) };
        let source = tp2.source.replace(RouteSource::Broadcast);
        let result = self.warm_local_graphs();
        tp2.source.set(source);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schedule(layers: usize, local: usize, lanes: usize, decode: bool, broadcast: bool) -> Schedule {
        let mut s = Schedule::default();
        let attention = |s: &mut Schedule, rank, layer, lane| {
            s.push(rank, "attention", slot(layer, false, lane), "attention partial");
            s.wait(rank, "attention", slot(layer, false, lane), "attention sum");
        };
        let experts = |s: &mut Schedule, rank, layer, lane| {
            if broadcast && layer < local {
                let at = 2 * lane + layer % 2;
                if rank == 0 { s.push(0, "routes", at, "canonical routes"); }
                else { s.wait(1, "routes", at, "canonical routes"); }
            }
            if rank == 1 || layer + 1 < layers {
                s.push(rank, "ffn", slot(layer, true, lane), "fused payload");
            }
        };
        let front = |s: &mut Schedule, layer, lane| {
            if layer == 0 { s.wait_for(1, "entry", DIRECT, lane, "peer residual"); }
            attention(s, 1, layer, lane);
            if !decode || layer >= local { experts(s, 1, layer, lane); }
        };
        for lane in 0..lanes { s.push_to(0, "entry", DIRECT, lane, "residual"); }
        if !decode { for lane in 0..lanes { front(&mut s, 0, lane); } }
        for layer in 0..layers {
            for lane in 0..lanes {
                attention(&mut s, 0, layer, lane);
                if decode {
                    if layer == 0 { front(&mut s, 0, lane); }
                    if layer < local { experts(&mut s, 1, layer, lane); }
                    if layer + 1 < layers {
                        s.wait(1, "ffn", slot(layer, true, lane), "peer FFN sum");
                        front(&mut s, layer + 1, lane);
                    }
                }
                experts(&mut s, 0, layer, lane);
                if !decode && layer + 1 < layers {
                    s.wait(1, "ffn", slot(layer, true, lane), "peer FFN sum");
                    front(&mut s, layer + 1, lane);
                }
                s.wait(0, "ffn", slot(layer, true, lane), "FFN sum");
            }
        }
        s
    }

    fn unit_schedule(layers: usize, local: usize, waves: usize, broadcast: bool) -> Result<Schedule> {
        use crate::shared::prefill_pipeline::{self, FfnRoute, Step};
        let routes: Vec<_> = (0..layers).map(|layer| if layer < local {
            FfnRoute::Local
        } else { FfnRoute::Spark }).collect();
        let units = prefill_pipeline::sequence_units(6 * 4096, 4096, waves, layers)?;
        let mut schedule = Schedule::default();
        for step in prefill_pipeline::prefill_schedule(&routes, &units, waves)? {
            match step {
                Step::Begin { unit } => schedule.push(0, "entry", slot(0, false, units[unit].slot), "unit residual"),
                Step::Attention { layer, unit } => {
                    let lane = units[unit].slot;
                    // UnitHooks queues rank 0, then rank 1 of this same layer.
                    schedule.push(0, "attention", slot(layer, false, lane), "attention partial");
                    schedule.wait(0, "attention", slot(layer, false, lane), "attention sum");
                    if layer == 0 { schedule.wait(1, "entry", slot(0, false, lane), "unit residual"); }
                    schedule.push(1, "attention", slot(layer, false, lane), "peer attention partial");
                    schedule.wait(1, "attention", slot(layer, false, lane), "peer attention sum");
                    if broadcast && layer < local {
                        schedule.wait(1, "routes", 2 * lane + layer % 2, "canonical routes");
                    }
                    schedule.push(1, "ffn", slot(layer, true, lane), "peer FFN payload");
                }
                Step::Local { layer, unit } | Step::Land { layer, unit } => {
                    let lane = units[unit].slot;
                    if broadcast && layer < local {
                        schedule.push(0, "routes", 2 * lane + layer % 2, "canonical routes");
                    }
                    if layer + 1 < layers {
                        schedule.push(0, "ffn", slot(layer, true, lane), "FFN payload");
                    }
                    schedule.wait(0, "ffn", slot(layer, true, lane), "FFN sum");
                    if layer + 1 < layers {
                        schedule.wait(1, "ffn", slot(layer, true, lane), "peer FFN sum");
                    }
                }
                Step::Commit { unit } => {
                    let lane = units[unit].slot;
                    schedule.push_to(1, "unit-commit", DIRECT, lane, "peer final reader complete");
                    schedule.wait_for(0, "unit-commit", DIRECT, lane, "join peer before slot reuse");
                }
                Step::Send { .. } | Step::End { .. } => {}
            }
        }
        Ok(schedule)
    }

    #[test]
    fn tp2_continuous_unit_and_decode_push_wait_schedules_drain() -> Result<()> {
        // Startup also records actual engine calls; the CPU mirror uses the executor's order.
        for layers in [1, 2, 6, 43, 61] {
            for local in [0, 1, layers / 2, layers] {
                for broadcast in [false, true] {
                    for waves in [1, 2, 3] {
                        assert_eq!(order::check_submission_faults(&unit_schedule(layers, local, waves, broadcast)?), Ok(()));
                    }
                    assert_eq!(order::check_submission_faults(&schedule(layers, local, 1, true, broadcast)), Ok(()));
                }
            }
        }
        Ok(())
    }

    #[test]
    fn packed_routes_align_every_row_shape_within_admission() {
        for topk in [6, 8] {
            for rows in 1..=4096 {
                let packed = PackedRoutes::new(rows, topk);
                assert_eq!(packed.weights_at % 16, 0);
                assert_eq!(packed.push_bytes % 16, 0);
                assert!(packed.weights_at >= packed.section_bytes);
                assert!(packed.weights_at + packed.section_bytes <= packed.push_bytes);
                assert!(packed.push_bytes <= rows * topk * 8 + 16);
            }
        }
        assert_eq!(PackedRoutes::new(1, 6).push_bytes, 64);
    }

    #[test]
    fn startup_route_identity_prefill_then_decode_schedule_drains() {
        // Prewarm and capture-only graphs precede the first peer wait and queue
        // no peer operations. Hardware also checks the actual identity steps.
        for broadcast in [false, true] {
            let mut startup = unit_schedule(43, 8, 2, broadcast).unwrap();
            let decode = schedule(43, 8, 1, true, broadcast);
            for (prefill, decode) in startup.streams.iter_mut().zip(decode.streams) {
                prefill.extend(decode);
            }
            assert_eq!(order::check(&startup), Ok(()));
        }
    }

    #[test]
    fn startup_route_check_reserves_prefill_and_decode_units() -> Result<()> {
        use super::super::super::pool::{PoolAllocator, PoolShape};
        let shape = PoolShape::new(1, 4096, 4);
        let mut allocator = PoolAllocator::new(shape);
        let placement = allocator.admit(512 + 1)?;
        assert!(placement.group_slot(4, 127).is_ok());
        assert!(placement.group_slot(4, 128).is_ok());
        allocator.release(placement);
        Ok(())
    }

    #[test]
    fn exchange_policy_matches_admission_and_auto_uses_qualified_bf16() {
        assert_eq!(ExchangePolicy::Auto.slot_bytes(4096, 80, 4096), 4096 * 4096 * 2);
        assert_eq!(ExchangePolicy::F32.slot_bytes(4096, 80, 4096), 4096 * 4096 * 4);
        assert_eq!(ExchangePolicy::Auto.slot_bytes(16, 80, 4096), 80 * 4096 * 4);
        for rows in [1, 16, 64, 80, 81, 256, 4096] {
            assert_eq!(ExchangePolicy::Auto.dtype(rows, 80), ExchangeDtype::Bf16);
            assert_eq!(ExchangePolicy::Bf16.dtype(rows, 80), ExchangeDtype::Bf16);
            assert_eq!(ExchangePolicy::F32.dtype(rows, 80), ExchangeDtype::F32);
        }
        assert!(ExchangePolicy::parse("fp8").is_err());
    }
}
