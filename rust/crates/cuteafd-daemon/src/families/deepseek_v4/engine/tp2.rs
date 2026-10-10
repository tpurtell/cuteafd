//! Head-split RTX routed halves. Scratch is shared on each rank's single
//! stream; only the per-lane fused payload survives until the FFN close.
use super::*;
use crate::shared::experts::rtx::{Combine, ExchangeDtype, ExpertInput, FusedCombine,
    PartialDtype, RouteCheck, RouteIdentity, RouteSource, Routes, RtxExpertLayer, RtxShard};
use std::cell::Cell;
use std::collections::BTreeMap;
use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ExchangePolicy { Auto, Bf16, F32 }

impl ExchangePolicy {
    pub(super) fn from_env() -> Result<Self> {
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
    interval: usize,
    steps: Cell<usize>,
    fallbacks: Cell<u64>,
    streams: [*mut c_void; 2],
    output_pending: [Cell<bool>; 2],
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
                    tp2.routes.push(0, slot, w.logits.buffer.ptr, weights_at + bytes)?;
                    if !self.capture_only.get() { tp2.fallbacks.set(tp2.fallbacks.get() + 1); }
                } else {
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

    fn collect_routes(&self, rank: usize, layer: usize, lane: usize, rows: usize, w: &Workspace<'_>) -> Result<()> {
        let tp2 = self.tp2.as_ref().context("TP2 experts")?;
        if !tp2.checking.get() || self.capture_only.get() { return Ok(()); }
        self.on(rank, || {
            let bytes = rows * self.cfg.n_activated_experts * 4;
            let host = HostAllocation::new(self.library, 2 * bytes)?;
            // SAFETY: pinned host storage is retained until both streams drain in finish_route_step.
            unsafe {
                self.library.copy_d2h_host_buffer_async(host.buffer, w.route_ids.buffer, bytes, self.stream_of(rank))?;
                self.library.copy_d2h_host_buffer_async(cuteafd_ffi::CuteafdHostBuffer {
                    ptr: host.buffer.ptr.cast::<u8>().add(bytes).cast(), bytes, ..host.buffer
                }, w.route_weights.buffer, bytes, self.stream_of(rank))?;
            }
            let mut checks = tp2.checks.borrow_mut();
            let check = checks.entry((layer, lane)).or_insert_with(|| PendingCheck { rows, ranks: [None, None] });
            ensure!(check.rows == rows && check.ranks[rank].is_none(), "duplicate or inconsistent route-check unit");
            check.ranks[rank] = Some(host);
            Ok(())
        })
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
        cuteafd_bench::context::set_resolved("route-identity", &format!("layers={} rows={} mismatch={:?}",
            identity.layers_checked, identity.rows_checked, identity.first_mismatch));
        cuteafd_bench::context::set_resolved("route-source", if tp2.source.get() == RouteSource::Replicated { "replicated" } else { "broadcast" });
        cuteafd_bench::context::set_resolved("route-fallbacks", &tp2.fallbacks.get().to_string());
        Ok(())
    }

    pub fn check_routes(&self, transports: &mut [SparkLink<'_>], runtime: &tokio::runtime::Runtime) -> Result<()> {
        let Some(tp2) = &self.tp2 else { return Ok(()) };
        let mut allocator = super::super::pool::PoolAllocator::new(self.shape);
        let mut placement = allocator.admit(1)?;
        // One lane is intentional: check all 512 rows of each layer, not the final lane only.
        let tokens: Vec<u32> = (0..512).map(|i| (i % self.cfg.vocab_size) as u32).collect();
        tp2.checking.set(true);
        self.prefill(&mut placement, &tokens, transports, runtime, 0, None)?;
        tp2.checking.set(true);
        self.decode(&mut [(&mut placement, 1)], transports.first_mut(), runtime)?;
        allocator.release(placement);
        // Cache/state writes were to scratch placement; released slots are overwritten on admission.
        self.warm_broadcast_graphs()?;
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
