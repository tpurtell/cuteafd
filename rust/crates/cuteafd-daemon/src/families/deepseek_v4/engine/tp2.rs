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

    fn dtype(self, rows: usize, decode_rows: usize) -> ExchangeDtype {
        match self {
            Self::F32 => ExchangeDtype::F32,
            Self::Bf16 => ExchangeDtype::Bf16,
            Self::Auto if rows <= decode_rows => ExchangeDtype::F32,
            Self::Auto => ExchangeDtype::Bf16,
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

pub(super) struct Tp2State<'a> {
    pub(super) layers: Range<usize>,
    ranks: [RefCell<Box<dyn RtxExpertLayer + 'a>>; 2],
    combine: [FusedCombine<'a>; 2],
    routes: PeerExchange<'a>,
    source: Cell<RouteSource>,
    pub(super) checking: Cell<bool>,
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
}

impl<'a> Engine<'a> {
    pub fn install_tp2(&mut self, ranks: [Box<dyn RtxExpertLayer + 'a>; 2]) -> Result<()> {
        ensure!(self.tp2.is_none(), "TP2 experts already installed");
        let peer = self.peer()?;
        let layers = ranks[0].layers();
        ensure!(layers == ranks[1].layers() && !layers.is_empty() && layers.end <= self.cfg.n_layers,
            "TP2 ranks must hold the same nonempty backbone prefix");
        ensure!(layers.start == 0, "V4 TP2 currently requires a prefix starting at layer 0");
        for (rank, experts) in ranks.iter().enumerate() {
            ensure!(experts.shard() == RtxShard::Tp2 { rank: rank as u8 }
                && experts.partial() == PartialDtype::F32
                && experts.device() == if rank == 0 { self.device } else { peer.device },
                "TP2 rank {rank} shard, dtype or device differs from head split");
        }
        ensure!(self.local.borrow().as_ref().is_none_or(|l| l.layers() == 0),
            "TP1 backbone experts cannot coexist with TP2");
        let rows = self.prefill_rows.max(self.decode_rows);
        let routes = PeerExchange::new(self.library,
            [RankDevice { device: self.device, stream: self.stream },
             RankDevice { device: peer.device, stream: peer.stream }],
            2 * PREFILL_LANES, rows * self.cfg.n_activated_experts * 8 + 16)?;
        let interval = std::env::var("CUTEAFD_ROUTE_CHECK").ok()
            .map(|v| v.parse::<usize>().context("CUTEAFD_ROUTE_CHECK must be a nonnegative step interval"))
            .transpose()?.unwrap_or(0);
        let combine = [FusedCombine::new(self.library, ExchangeDtype::Bf16)?,
            FusedCombine::new(self.library, ExchangeDtype::F32)?];
        self.tp2 = Some(Tp2State { layers: layers.clone(), ranks: ranks.map(RefCell::new), combine, routes,
            source: Cell::new(RouteSource::Replicated), checking: Cell::new(false), checks: RefCell::new(BTreeMap::new()),
            schedule: RefCell::new(None), identity: RefCell::new(RouteIdentity::default()),
            interval, steps: Cell::new(0), fallbacks: Cell::new(0),
            streams: [self.stream, peer.stream], output_pending: [Cell::new(false), Cell::new(false)] });
        tracing::info!(layers = ?layers, "V4 RTX TP2 expert halves resident on both GPUs");
        cuteafd_bench::context::set_resolved("rtx-tp2-layers", &format!("{}..{}", layers.start, layers.end));
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
        let lane = &w.lanes[index];
        let payload = lane.payload.as_ref().context("TP2 lane payload")?;
        let stream = self.stream_of(rank);
        ensure!(stream == tp2.streams[rank], "TP2 executor moved streams; per-lane scratch ownership is required");
        debug_assert!(!tp2.output_pending[rank].get(), "previous routed output was not copied to its lane payload");
        self.collect_routes(rank, layer, index, rows, w)?;
        self.on(rank, || {
            let mut routes = Routes { ids: w.route_ids.buffer.ptr, weights: w.route_weights.buffer.ptr };
            if tp2.source.get() == RouteSource::Broadcast {
                let slot = 2 * index + layer % 2;
                let bytes = rows * self.cfg.n_activated_experts * 4;
                let weights_at = bytes.next_multiple_of(16);
                if rank == 0 {
                    // The scores are consumed: their larger buffer doubles as route packing scratch.
                    ensure!(weights_at + bytes <= w.logits.buffer.bytes, "router scratch cannot hold canonical routes");
                    // SAFETY: both sections fit the rank's logits allocation; copies precede the push.
                    unsafe {
                        self.library.copy_d2d_async(w.logits.buffer, w.route_ids.buffer, bytes, stream)?;
                        self.library.copy_d2d_async(cuteafd_ffi::CuteafdDeviceBuffer {
                            ptr: w.logits.buffer.ptr.cast::<u8>().add(weights_at).cast(), ..w.logits.buffer
                        }, w.route_weights.buffer, bytes, stream)?;
                    }
                    self.record_peer(0, "routes", slot, true, layer, index);
                    tp2.routes.push(0, slot, w.logits.buffer.ptr, weights_at + bytes)?;
                } else {
                    self.record_peer(1, "routes", slot, false, layer, index);
                    tp2.routes.wait(1, slot)?;
                    let packed = tp2.routes.recv(1, slot)?;
                    // SAFETY: the received slot contains ids then aligned weights, published by rank 0.
                    routes = Routes { ids: packed, weights: unsafe { packed.cast::<u8>().add(weights_at).cast() } };
                }
            }
            let combine = self.combine(rows)?;
            ensure!(combine.payload_bytes(rows, self.cfg.dim) <= payload.buffer.bytes, "TP2 payload exceeds admission");
            let mut experts = tp2.ranks[rank].borrow_mut();
            // SAFETY: routes and locally quantized wire rows are complete on this rank's stream.
            unsafe { experts.enqueue(layer, rows, ExpertInput::Fp8K32(w.wire.buffer.ptr), routes, stream)?; }
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

    pub(super) fn tp2_post(&self, rank: usize, layer: usize, index: usize, w: &Workspace<'_>, lane: &Lane<'_>, rows: usize)
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
        // cudaHostAlloc may synchronize. Allocate before queuing any peer waits,
        // never from a layer whose matching push has not been enqueued yet.
        let mut checks = tp2.checks.borrow_mut();
        ensure!(checks.is_empty(), "previous route diagnostics did not drain");
        for layer in tp2.layers.clone() {
            for (lane, &rows) in rows.iter().enumerate() {
                let bytes = rows * self.cfg.n_activated_experts * 8;
                let ranks = [Some(self.on(0, || HostAllocation::new(self.library, bytes))?),
                    Some(self.on(1, || HostAllocation::new(self.library, bytes))?)];
                checks.insert((layer, lane), PendingCheck { rows, ranks });
            }
        }
        Ok(())
    }

    fn collect_routes(&self, rank: usize, layer: usize, lane: usize, rows: usize, w: &Workspace<'_>) -> Result<()> {
        let tp2 = self.tp2.as_ref().context("TP2 experts")?;
        if !tp2.checking.get() || self.capture_only.get() { return Ok(()); }
        self.on(rank, || {
            let bytes = rows * self.cfg.n_activated_experts * 4;
            let checks = tp2.checks.borrow();
            let check = checks.get(&(layer, lane)).context("route diagnostic storage not prepared")?;
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
        if push { schedule.push(rank, exchange, slot, label); }
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
        let mut allocator = super::super::pool::PoolAllocator::new(self.shape);
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
            let lane = &w.lanes[0];
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
            if layer == 0 { s.wait(1, "entry", DIRECT, "peer residual"); }
            attention(s, 1, layer, lane);
            if !decode || layer >= local { experts(s, 1, layer, lane); }
        };
        for lane in 0..lanes { s.push(0, "entry", DIRECT, "residual"); }
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

    #[test]
    fn tp2_two_lane_and_decode_push_wait_schedules_drain() {
        // Runtime startup also records the engine's actual calls and checks them,
        // so this CPU mirror cannot silently qualify a different enqueue order.
        for layers in [1, 2, 6] {
            for local in 1..=layers {
                for broadcast in [false, true] {
                    for lanes in [1, 2] {
                        assert_eq!(order::check(&schedule(layers, local, lanes, false, broadcast)), Ok(()));
                    }
                    assert_eq!(order::check(&schedule(layers, local, 1, true, broadcast)), Ok(()));
                }
            }
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
    fn exchange_policy_matches_admission_and_auto_switches_by_rows() {
        assert_eq!(ExchangePolicy::Auto.slot_bytes(4096, 80, 4096), 4096 * 4096 * 2);
        assert_eq!(ExchangePolicy::F32.slot_bytes(4096, 80, 4096), 4096 * 4096 * 4);
        assert_eq!(ExchangePolicy::Auto.slot_bytes(16, 80, 4096), 80 * 4096 * 4);
        assert_eq!(ExchangePolicy::Auto.dtype(80, 80), ExchangeDtype::F32);
        assert_eq!(ExchangePolicy::Auto.dtype(81, 80), ExchangeDtype::Bf16);
        assert_eq!(ExchangePolicy::Bf16.dtype(1, 80), ExchangeDtype::Bf16);
        assert!(ExchangePolicy::parse("fp8").is_err());
    }
}
