//! Routed RTX halves share one serialized package workspace per rank. Copy
//! each result to its lane before the next package launch reuses that output.
use super::*;

impl<'a> MimoEngine<'a> {
    pub(in crate::families::mimo_v2) fn install_tp2(&mut self, ranks: [Fp8MoeTp2<'a>; 2]) -> Result<()> {
        ensure!(self.tp2.is_none() && self.peer.is_some(), "MiMo TP2 requires a head-split peer");
        let layers = ranks[0].layers();
        ensure!(layers == ranks[1].layers() && !layers.is_empty(), "MiMo TP2 halves disagree");
        let rows = self.prefill_rows.max(DECODE_ROWS);
        let route_bytes = (rows * self.cfg.topk * 4).next_multiple_of(16) * 2;
        let routes = PeerExchange::new_abortable(self.library,
            [RankDevice { device: self.device, stream: self.stream },
             RankDevice { device: self.peer.as_ref().context("MiMo peer")?.device, stream: self.stream_of(1) }],
            2 * self.transport_lanes(), route_bytes)?;
        let combine = FusedCombine::new(self.library, ExchangeDtype::Bf16)?;
        let interval = std::env::var("CUTEAFD_ROUTE_CHECK").ok()
            .map(|value| value.parse::<usize>().context("CUTEAFD_ROUTE_CHECK must be a nonnegative step interval"))
            .transpose()?.unwrap_or(0);
        self.tp2 = Some(Tp2 { layers, ranks: ranks.map(RefCell::new), combine, routes,
            source: Cell::new(RouteSource::Broadcast), checking: Cell::new(false), interval,
            steps: Cell::new(0), checks: RefCell::new(HashMap::new()), identity: RefCell::new(RouteIdentity::default()) });
        if self.prefill_rows >= DECODE_ROWS { self.prepare_scoring_prefill()?; }
        else {
            *self.decode_workspace.borrow_mut() = Some(self.workspace(0, DECODE_ROWS, true)?);
            *self.peer.as_ref().context("MiMo peer")?.decode_workspace.borrow_mut() = Some(self.workspace(1, DECODE_ROWS, true)?);
        }
        // Prime with initialized, valid routes before any peer wait is queued.
        for rank in 0..2 {
            let owned = if self.prefill_rows >= DECODE_ROWS {
                if rank == 0 { self.workspace.borrow() }
                else { self.peer.as_ref().context("MiMo peer")?.workspace.borrow() }
            } else if rank == 0 { self.decode_workspace.borrow() }
                else { self.peer.as_ref().context("MiMo peer")?.decode_workspace.borrow() };
            let w = owned.as_ref().context("MiMo prime workspace")?;
            self.on(rank, || {
                self.library.cuda_zero_bytes(w.delta.buffer, w.delta.buffer.bytes)?;
                // SAFETY: startup buffers are disjoint and initialized; force
                // the native combine's first use before any unmatched wait.
                unsafe {
                    self.tp2.as_ref().context("MiMo TP2")?.combine.sum(w.delta.buffer.ptr, w.delta.buffer.ptr,
                        w.x.buffer.ptr.cast(), rows, self.cfg.hidden, self.stream_of(rank))?;
                    self.library.cuda_stream_synchronize(self.stream_of(rank))?;
                }
                self.put(&w.route_ids, &(0..rows * self.cfg.topk)
                    .map(|i| (i % self.cfg.experts) as u32).collect::<Vec<_>>())?;
                self.library.cuda_zero_bytes(w.route_weights.buffer, w.route_weights.buffer.bytes)?;
                // SAFETY: initialized admitted max-row input/routes are live on
                // this rank until the package priming synchronizes every launch.
                unsafe { self.tp2.as_ref().context("MiMo TP2")?.ranks[rank].borrow_mut().prime(
                    RtxInput::Bf16(w.x.buffer.ptr), Routes { ids: w.route_ids.buffer.ptr,
                        weights: w.route_weights.buffer.ptr }, self.stream_of(rank)) }
            })?;
        }
        Ok(())
    }

    pub(super) fn tp2_layer(&self, index: usize) -> bool {
        self.tp2.as_ref().is_some_and(|tp2| tp2.layers.contains(&index)) && !self.weights.layers[index].dense
    }

    pub(super) fn tp2_experts(&self, rank: usize, index: usize, lane: usize,
        w: &Workspace<'_>, rows: usize) -> Result<()> {
        let tp2 = self.tp2.as_ref().context("MiMo TP2")?;
        let stream = self.stream_of(rank);
        let bytes = rows * self.cfg.hidden * 2;
        self.collect_tp2_routes(rank, index, lane, rows, w)?;
        self.on(rank, || {
            let mut routes = Routes { ids: w.route_ids.buffer.ptr, weights: w.route_weights.buffer.ptr };
            if tp2.source.get() == RouteSource::Broadcast {
                let section = rows * self.cfg.topk * 4;
                let aligned = section.next_multiple_of(16);
                let total = (aligned + section).next_multiple_of(16);
                let slot = 2 * lane + index % 2;
                if rank == 0 {
                    ensure!(total <= w.router_logits.buffer.bytes, "MiMo route packing exceeds admitted logits");
                    // SAFETY: the consumed router logits hold both padded route
                    // sections; copies complete before their peer publication.
                    unsafe {
                        self.library.cuda_zero_bytes_async(w.router_logits.buffer, total, stream)?;
                        self.library.copy_d2d_async(w.router_logits.buffer, w.route_ids.buffer, section, stream)?;
                        self.library.copy_d2d_async(Self::region(&w.router_logits, aligned, section),
                            w.route_weights.buffer, section, stream)?;
                    }
                    tp2.routes.push(0, slot, w.router_logits.buffer.ptr, total)?;
                } else {
                    tp2.routes.wait(1, slot)?;
                    let received = tp2.routes.recv(1, slot)?;
                    // SAFETY: rank 0 published the aligned ids/weights sections.
                    routes = Routes { ids: received, weights: unsafe { received.cast::<u8>().add(aligned).cast() } };
                }
            }
            let mut experts = tp2.ranks[rank].borrow_mut();
            let mut abort = || {
                crate::shared::decode_graph::fatal_drain(tp2.routes.publish_abort(),
                    "MiMo TP2 RTX route abort");
                if let Some(exchange) = &self.exchange {
                    crate::shared::decode_graph::fatal_drain(exchange.publish_abort(),
                        "MiMo TP2 RTX FFN abort");
                }
            };
            // SAFETY: router, BF16 input and package output are rank-local,
            // stream-ordered, and no next launch may precede this output copy.
            unsafe {
                experts.enqueue(index, rows, RtxInput::Bf16(w.x.buffer.ptr), routes, stream, &mut abort)?;
                self.library.copy_d2d_async(w.delta.buffer, cuteafd_ffi::CuteafdDeviceBuffer {
                    ptr: experts.output(), bytes, ..w.delta.buffer }, bytes, stream)?;
            }
            self.push(rank, lane_slot(index, true, lane), w.delta.buffer.ptr, bytes)
        })
    }

    pub(super) fn tp2_post(&self, rank: usize, index: usize, lane: usize,
        w: &Workspace<'_>, rows: usize) -> Result<()> {
        let slot = lane_slot(index, true, lane);
        self.wait(rank, slot)?;
        let received = self.recv(rank, slot);
        let (rank0, rank1) = if rank == 0 { (w.delta.buffer.ptr, received) } else { (received, w.delta.buffer.ptr) };
        self.on(rank, || {
            // SAFETY: BF16 partials are complete and summed in rank order on
            // both GPUs. The consumed normalized input is a disjoint output
            // staging buffer; delta remains the epilogue operand.
            unsafe {
                self.tp2.as_ref().context("MiMo TP2")?.combine.sum(rank0, rank1,
                    w.x.buffer.ptr.cast(), rows, self.cfg.hidden, self.stream_of(rank))?;
                self.library.copy_d2d_async(w.delta.buffer, w.x.buffer, rows * self.cfg.hidden * 2,
                    self.stream_of(rank))
            }
        })
    }

    pub(super) fn prepare_tp2_route_step(&self, rows: &[usize]) -> Result<()> {
        let Some(tp2) = &self.tp2 else { return Ok(()) };
        let step = tp2.steps.get() + 1;
        tp2.steps.set(step);
        if tp2.interval > 0 && step % tp2.interval == 0 { tp2.checking.set(true); }
        if !tp2.checking.get() { return Ok(()); }
        let mut checks = tp2.checks.borrow_mut();
        ensure!(checks.is_empty(), "previous MiMo route diagnostics did not drain");
        // Pinned allocations can synchronize; reserve them before peer waits.
        for layer in tp2.layers.clone().filter(|&layer| self.tp2_layer(layer)) {
            for (lane, &rows) in rows.iter().enumerate() {
                let bytes = rows * self.cfg.topk * 8;
                checks.insert((layer, lane), PendingRouteCheck { rows, ranks: [
                    self.on(0, || HostAllocation::new(self.library, bytes))?,
                    self.on(1, || HostAllocation::new(self.library, bytes))?] });
            }
        }
        Ok(())
    }

    fn collect_tp2_routes(&self, rank: usize, layer: usize, lane: usize, rows: usize,
        w: &Workspace<'_>) -> Result<()> {
        let tp2 = self.tp2.as_ref().context("MiMo TP2")?;
        if !tp2.checking.get() { return Ok(()); }
        let checks = tp2.checks.borrow();
        let check = checks.get(&(layer, lane)).context("MiMo route diagnostics were not admitted")?;
        ensure!(rows == check.rows, "MiMo route diagnostic shape changed");
        let host = &check.ranks[rank];
        let bytes = rows * self.cfg.topk * 4;
        self.on(rank, || {
            // SAFETY: pinned owners survive both streams' completion; each
            // router output is copied before its workspace is reused.
            unsafe {
                self.library.copy_d2h_host_buffer_async(host.buffer, w.route_ids.buffer, bytes, self.stream_of(rank))?;
                self.library.copy_d2h_host_buffer_async(cuteafd_ffi::CuteafdHostBuffer {
                    ptr: host.buffer.ptr.cast::<u8>().add(bytes).cast(), bytes, ..host.buffer
                }, w.route_weights.buffer, bytes, self.stream_of(rank))?;
            }
            Ok(())
        })
    }

    pub(super) fn finish_tp2_route_step(&self) -> Result<()> {
        let Some(tp2) = &self.tp2 else { return Ok(()) };
        if !tp2.checking.replace(false) { return Ok(()); }
        for rank in 0..2 {
            // SAFETY: the completed step enqueued every matching peer push.
            self.on(rank, || unsafe { self.library.cuda_stream_synchronize(self.stream_of(rank)) })?;
        }
        let checks = std::mem::take(&mut *tp2.checks.borrow_mut()).into_iter()
            .map(|((layer, _), check)| RouteCheck { layer, rows: check.rows, topk: self.cfg.topk,
                ranks: check.ranks.map(|host| host.bytes().to_vec()) }).collect::<Vec<_>>();
        let identity = RouteIdentity::check(&checks);
        if identity.source() == RouteSource::Broadcast { tp2.source.set(RouteSource::Broadcast); }
        tracing::info!(?identity, source = ?tp2.source.get(), "MiMo TP2 actual-state route identity");
        let mut aggregate = tp2.identity.borrow_mut();
        aggregate.layers_checked = aggregate.layers_checked.saturating_add(identity.layers_checked);
        aggregate.rows_checked = aggregate.rows_checked.saturating_add(identity.rows_checked);
        if aggregate.first_mismatch.is_none() { aggregate.first_mismatch = identity.first_mismatch; }
        Ok(())
    }

    /// Qualify real prefill/decode residuals after capture warm-up. Only the
    /// package launches consult the route source, never a captured graph node.
    pub(in crate::families::mimo_v2) fn check_tp2_routers(&self) -> Result<()> {
        let Some(tp2) = &self.tp2 else { return Ok(()) };
        let rows = self.prefill_rows.min(256).min(self.max_context.saturating_sub(1));
        ensure!(rows > 0, "MiMo route qualification needs prefill plus decode");
        let mut allocator = Allocator::new(self.pages, self.rings);
        let mut placement = allocator.admit(rows + 1)?;
        let tokens = (0..rows).map(|i| (i % self.cfg.vocab_size) as u32).collect::<Vec<_>>();
        *tp2.identity.borrow_mut() = RouteIdentity::default();
        tp2.checking.set(true);
        self.prefill_device(&mut placement, &tokens, false, None, None)?;
        let prefill = tp2.identity.borrow().clone();
        ensure!(prefill.layers_checked > 0 && prefill.rows_checked > 0,
            "MiMo startup router qualification checked no prefill routes");
        tp2.checking.set(true);
        self.verify_device(&mut [(&mut placement, 1)], &[1], None)?;
        allocator.release(placement);
        let identity = tp2.identity.borrow().clone();
        ensure!(identity.layers_checked > prefill.layers_checked && identity.rows_checked > prefill.rows_checked,
            "MiMo startup router qualification checked no decode routes");
        tp2.source.set(identity.source());
        tracing::info!(?identity, source = ?tp2.source.get(), "MiMo TP2 startup router qualification");
        *tp2.identity.borrow_mut() = RouteIdentity::default();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::peer_split::order::{check, Schedule};

    #[derive(Clone, Copy, PartialEq)]
    enum Home { Dense, Tp2, Spark }

    fn attention(s: &mut Schedule, rank: usize, layer: usize, lane: usize) {
        let slot = lane_slot(layer, false, lane);
        s.push(rank, "main", slot, format!("attention push {rank}/{layer}/{lane}"));
        s.wait(rank, "main", slot, format!("attention wait {rank}/{layer}/{lane}"));
    }

    fn experts(s: &mut Schedule, rank: usize, layer: usize, lane: usize, broadcast: bool) {
        if broadcast {
            let slot = 2 * lane + layer % 2;
            if rank == 0 { s.push(0, "routes", slot, "canonical routes"); }
            else { s.wait(1, "routes", slot, "canonical routes"); }
        }
        s.push(rank, "main", lane_slot(layer, true, lane), "TP2 partial");
    }

    fn post(s: &mut Schedule, rank: usize, layer: usize, lane: usize) {
        s.wait(rank, "main", lane_slot(layer, true, lane), "FFN combine");
    }

    // Mirror eager peer_layer lookahead and the lane producer/post split.
    fn prefill(s: &mut Schedule, homes: &[Home], lanes: usize, broadcast: bool, ahead: bool) {
        for lane in 0..lanes { s.push_to(0, "main", DIRECT, lane, "embedded residual"); }
        for lane in 0..lanes { s.wait_for(1, "main", DIRECT, lane, format!("embedded lane {lane}")); }
        let peer = |s: &mut Schedule, layer: usize, lane| {
            attention(s, 1, layer, lane);
            match homes[layer] {
                Home::Dense => s.push(1, "main", lane_slot(layer, true, lane), "dense partial"),
                Home::Tp2 => experts(s, 1, layer, lane, broadcast),
                Home::Spark => {},
            }
            if ahead && (homes[layer] != Home::Spark || layer + 1 < homes.len()) { post(s, 1, layer, lane); }
        };
        if ahead { peer(s, 0, 0); }
        for (layer, home) in homes.iter().enumerate() {
            for lane in 0..lanes {
                if !ahead { peer(s, layer, lane); }
                attention(s, 0, layer, lane);
                if ahead && layer + 1 < homes.len() { peer(s, layer + 1, lane); }
                match home {
                    Home::Dense => s.push(0, "main", lane_slot(layer, true, lane), "dense partial"),
                    Home::Tp2 => experts(s, 0, layer, lane, broadcast),
                    Home::Spark if layer + 1 < homes.len() =>
                        s.push(0, "main", lane_slot(layer, true, lane), "landed Spark sum"),
                    Home::Spark => {},
                }
                if !ahead && (*home != Home::Spark || layer + 1 < homes.len()) { post(s, 1, layer, lane); }
                if *home != Home::Spark { post(s, 0, layer, lane); }
            }
        }
    }

    // Decode consumes the previous FFN in the next segment. Rank 1 has no
    // tail segment, so the last TP2 post is explicitly queued by decode_layers.
    fn decode(s: &mut Schedule, homes: &[Home], broadcast: bool) {
        for layer in 0..=homes.len() {
            if layer == 0 { s.push(0, "main", DIRECT, "decode embedded rows"); }
            else {
                match homes[layer - 1] {
                    Home::Dense | Home::Tp2 => post(s, 0, layer - 1, 0),
                    Home::Spark if layer < homes.len() => s.push(0, "main", slot(layer - 1, true), "Spark sum"),
                    Home::Spark => {},
                }
            }
            if layer == homes.len() { break; }
            attention(s, 0, layer, 0);
            if homes[layer] == Home::Dense && layer + 1 < homes.len() {
                s.push(0, "main", slot(layer, true), "dense partial");
            }
            if layer == 0 { s.wait(1, "main", DIRECT, "decode embedded rows"); }
            else { post(s, 1, layer - 1, 0); }
            attention(s, 1, layer, 0);
            match homes[layer] {
                Home::Dense => s.push(1, "main", slot(layer, true), "dense partial"),
                Home::Tp2 => {
                    experts(s, 0, layer, 0, broadcast);
                    experts(s, 1, layer, 0, broadcast);
                    if layer + 1 == homes.len() { post(s, 1, layer, 0); }
                },
                Home::Spark => {},
            }
        }
    }

    #[test]
    fn mimo_tp2_native_first_use_must_precede_eager_peer_lookahead() {
        use crate::shared::peer_split::order::host::{check_host, HostSchedule};
        for preloaded in [false, true] {
            let mut schedule = HostSchedule::default();
            if preloaded {
                for rank in 0..2 { schedule.sync(rank, "native helpers loaded at startup"); }
            }
            schedule.push(0, "main", DIRECT, "embedded residual");
            schedule.wait(1, "main", DIRECT, "embedded residual");
            for rank in [1, 0] {
                schedule.push(rank, "main", slot(0, false), "layer 0 attention partial");
                schedule.wait(rank, "main", slot(0, false), "layer 0 attention sum");
            }
            schedule.push(1, "main", slot(0, true), "layer 0 peer dense partial");
            schedule.wait(1, "main", slot(0, true), "layer 0 lead dense partial");
            // peer_layer(1) is queued before rank 0's layer 0 FFN push.
            schedule.push(1, "main", slot(1, false), "layer 1 peer attention partial");
            schedule.wait(1, "main", slot(1, false), "layer 1 lead attention partial");
            if !preloaded { schedule.sync(1, "lazy native router first use"); }
            schedule.push(0, "main", slot(0, true), "layer 0 lead dense partial");
            schedule.wait(0, "main", slot(0, true), "layer 0 peer dense partial");
            schedule.push(0, "main", slot(1, false), "layer 1 lead attention partial");
            schedule.wait(0, "main", slot(1, false), "layer 1 peer attention partial");
            assert_eq!(check(&schedule.streams()), Ok(()));
            if preloaded { assert_eq!(check_host(&schedule), Ok(())); }
            else {
                let blocked = check_host(&schedule).unwrap_err();
                assert_eq!(blocked.sync, "lazy native router first use");
                assert_eq!(blocked.blocked[1].as_deref(), Some("layer 0 lead dense partial"));
            }
        }
    }

    #[test]
    fn mimo_tp2_cross_exchange_order_drains_eager_lanes_decode_and_repeated_steps() {
        for homes in [vec![Home::Dense, Home::Tp2, Home::Tp2, Home::Spark, Home::Spark],
            vec![Home::Dense, Home::Tp2, Home::Tp2], vec![Home::Dense, Home::Spark]] {
            for broadcast in [false, true] {
                for lanes in [1, 2, 3] {
                    let mut s = Schedule::default();
                    for _ in 0..3 { prefill(&mut s, &homes, lanes, broadcast, false); s.end_request(); }
                    assert_eq!(crate::shared::peer_split::order::check_submission_faults(&s), Ok(()), "lanes={lanes} broadcast={broadcast}");
                }
                let mut eager = Schedule::default();
                let mut graph = Schedule::default();
                for _ in 0..3 {
                    prefill(&mut eager, &homes, 1, broadcast, true);
                    decode(&mut graph, &homes, broadcast);
                    eager.end_request();
                    graph.end_request();
                }
                assert_eq!(crate::shared::peer_split::order::check_submission_faults(&eager), Ok(()), "eager broadcast={broadcast}");
                assert_eq!(crate::shared::peer_split::order::check_submission_faults(&graph), Ok(()), "decode broadcast={broadcast}");
            }
        }
    }
}
