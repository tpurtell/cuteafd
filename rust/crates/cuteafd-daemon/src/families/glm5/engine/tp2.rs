//! GLM's admitted routed prefix uses shared paired backends and the existing FFN slot.
use super::*;

impl<'a> GlmEngine<'a> {
    pub(crate) fn install_whole(&mut self, backend: Box<dyn RtxExpertLayer + 'a>) -> Result<()> {
        ensure!(self.peer.is_none() && self.tp2.is_none() && self.whole.is_none()
            && self.decode_workspace.borrow().is_none() && self.prefill_slots.borrow().is_empty(),
            "install whole experts before graph/workspace allocation");
        ensure!(backend.layers().iter().enumerate().all(|(slot, &id)|
            id.checked_sub(self.cfg.first_moe_layer) == Some(slot)
                && self.weights.layers.get(id).is_some_and(|layer| !layer.dense)),
            "invalid GLM whole routed prefix");
        let whole = crate::shared::experts::rtx::pair::Whole::new(self.library, self.device,
            backend, self.cfg.hidden, self.prefill_rows.max(DECODE_ROWS))?;
        cuteafd_bench::context::set_resolved("rtx-whole-layers", &format!("{:?}", whole.layers));
        self.whole = Some(whole);
        Ok(())
    }

    pub(crate) fn local_layer(&self, layer: usize) -> bool {
        self.tp2_layer(layer) || self.whole.as_ref().is_some_and(|owner| owner.layers.contains(&layer))
    }

    /// Rank-0 local shared + routed Whole FFN (capture-safe on 1x RTX).
    pub(super) fn whole_ffn_local(&self, layer: usize, w: &Workspace<'_>, rows: usize, cap: &str) -> Result<()> {
        let whole = self.whole.as_ref().context("whole routed owner")?;
        self.shared_on_local(0, layer, w, &self.weights.layers[layer], Scalar::I32(rows as i32), cap)?;
        let input = if whole.uses_fp8_input() { ExpertInput::Fp8K32(w.wire.buffer.ptr) }
            else { ExpertInput::Bf16(w.x.buffer.ptr) };
        // SAFETY: one home-device stream owns the input, routes, shared and delta;
        // the shared owner combines before another unit reuses backend scratch.
        unsafe { whole.enqueue(layer, rows, input, Routes { ids: w.route_ids.buffer.ptr,
            weights: w.route_weights.buffer.ptr }, w.shared.buffer, w.delta.buffer, self.stream,
            &mut || crate::shared::decode_graph::fatal_drain(self.abort_tp2(), "GLM whole enqueue abort")) }
    }

    /// Shared local FFN boundary for eager decode and prefill units.
    pub(super) fn local_ffn(&self, layer: usize, lane: usize, w: &Workspace<'_>,
        peer: Option<&Workspace<'_>>, rows: usize, cap: &str) -> Result<()> {
        if self.tp2_layer(layer) {
            return self.tp2_pair(layer, lane, w, peer.context("TP2 peer workspace")?, rows, cap);
        }
        self.whole_ffn_local(layer, w, rows, cap)
    }

    /// Whether TP2 experts run strictly rank-local (replicated routes, no route checks active).
    pub(super) fn tp2_routes_replicated(&self) -> bool {
        self.tp2.as_ref().is_some_and(|pair| pair.routes_replicated())
    }

    pub(crate) fn set_tp2_exchange(&mut self, dtype: ExchangeDtype) {
        self.tp2_exchange = dtype;
    }

    pub(crate) fn install_tp2(&mut self, ranks: [Box<dyn RtxExpertLayer + 'a>; 2]) -> Result<()> {
        let peer = self.peer()?;
        let layers = ranks[0].layers().to_vec();
        ensure!(
            !layers.is_empty()
                && layers.iter().enumerate().all(|(slot, &id)| id.checked_sub(self.cfg.first_moe_layer) == Some(slot))
                && layers.iter().all(|&id| self.weights.layers.get(id).is_some_and(|l| !l.dense)),
            "invalid GLM TP2 routed prefix"
        );
        ensure!(
            self.tp2.is_none()
                && self.decode_workspace.borrow().is_none()
                && self.prefill_slots.borrow().is_empty()
                && peer.decode_workspace.borrow().is_none()
                && peer.prefill_slots.borrow().is_empty(),
            "install TP2 before graph/workspace allocation"
        );
        let devices = [
            RankDevice {
                device: self.device,
                stream: self.stream,
            },
            RankDevice {
                device: peer.device,
                stream: peer.stream,
            },
        ];
        for (rank, producer) in devices.into_iter().enumerate() {
            self.exchange()?
                .require_cooperative_producer_stream(rank, producer)?;
        }
        self.tp2 = Some(Pair::new(
            self.library,
            devices,
            ranks,
            self.cfg.hidden,
            self.cfg.topk,
            self.prefill_rows.max(DECODE_ROWS),
            DECODE_ROWS,
            self.prefill_lanes,
            self.tp2_exchange,
        )?);
        cuteafd_bench::context::set_resolved(
            "rtx-tp2-layers",
            &format!("{layers:?}"),
        );
        Ok(())
    }

    pub(crate) fn has_local(&self) -> bool {
        self.tp2.is_some() || self.whole.is_some()
    }
    pub(crate) fn local_layers(&self) -> Option<&[usize]> {
        self.tp2.as_ref().map(|pair| pair.layers.as_slice())
            .or_else(|| self.whole.as_ref().map(|owner| owner.layers.as_slice()))
    }

    pub(super) fn tp2_layer(&self, layer: usize) -> bool {
        self.tp2
            .as_ref()
            .is_some_and(|pair| pair.layers.contains(&layer))
    }

    pub(crate) fn require_live(&self) -> Result<()> {
        let state = self.terminal.get();
        if state != crate::shared::peer_split::TerminalState::Active {
            return Err(crate::shared::peer_split::TerminalPeerError::Closed(state).into());
        }
        if let Some(exchange) = &self.exchange { exchange.require_live()?; }
        if let Some(pair) = &self.tp2 { pair.routes.require_live()?; }
        if let Some(drafter) = &self.drafter { drafter.require_live()?; }
        Ok(())
    }

    fn abort_tp2(&self) -> Result<()> {
        // A backend failure may occur with rank1 current; publish on the home device.
        crate::shared::memory::device::Device { library: self.library, id: self.device }.run(|| {
            let mut failures = Vec::new();
            for exchange in self.exchange.as_deref().into_iter().chain(self.tp2.iter().map(|pair| &pair.routes)) {
                if let Err(error) = exchange.publish_abort() { failures.push(format!("{error:#}")); }
            }
            if let Some(drafter) = &self.drafter {
                if let Err(error) = drafter.publish_abort() { failures.push(format!("draft bridge: {error:#}")); }
            }
            ensure!(failures.is_empty(), "GLM abort publication: {}", failures.join("; "));
            Ok(())
        })
    }

    pub(crate) fn retire_tp2(&self) -> Result<()> {
        use crate::shared::peer_split::{TerminalState, TerminalPeerError};
        match self.terminal.get() {
            TerminalState::Drained => return Ok(()),
            TerminalState::Active => self.terminal.set(TerminalState::Aborting),
            state => return Err(TerminalPeerError::Closed(state).into()),
        }
        let result = crate::shared::experts::rtx::pair::teardown(&mut crate::shared::experts::rtx::pair::Teardown {
            abort: || self.abort_tp2(),
            ranks: || {
                if let Some(exchange) = &self.exchange { exchange.drain_compute()?; }
                else {
                    // SAFETY: one-GPU stream and all its queued storage remain owned.
                    unsafe { self.library.cuda_stream_synchronize(self.stream)?; }
                }
                if let Some(pair) = &self.tp2 { pair.routes.drain_compute()?; }
                if let Some(drafter) = &self.drafter { drafter.synchronize()?; }
                Ok(())
            },
            // The engine owns no terminal Spark links: prefill waves run on the
            // caller's prefill link through its transport worker, which drains
            // its uploads before every call returns.
            links: || Ok(()),
            copies: || Ok(()),
            graphs: || {
                for rank in 0..self.ranks() {
                    self.on(rank, || {
                        if rank == 0 { self.graphs.borrow_mut().cache_clear_drained(); }
                        else { self.peer()?.graphs.borrow_mut().cache_clear_drained(); }
                        Ok(())
                    })?;
                }
                Ok(())
            },
            finish: |drained| {
                self.terminal.set(if drained { TerminalState::Drained } else { TerminalState::Retained });
                for exchange in self.exchange.as_deref().into_iter().chain(self.tp2.iter().map(|pair| &pair.routes)) {
                    exchange.finish_terminal(drained);
                }
                if let Some(drafter) = &self.drafter { drafter.finish_terminal(drained); }
            },
        });
        result
    }

    pub(crate) fn prime_tp2(&self) -> Result<()> {
        if let Some(whole) = &self.whole {
            if self.decode_workspace.borrow().is_none() {
                *self.decode_workspace.borrow_mut() = Some(self.workspace(DECODE_ROWS, true)?);
            }
            self.ensure_prefill_slots()?;
            let owned = self.prefill_slots.borrow();
            let decode = self.decode_workspace.borrow();
            let w = if self.prefill_rows >= DECODE_ROWS {
                &owned[0]
            } else {
                decode.as_ref().context("decode workspace")?
            };
            let cap = if self.prefill_rows >= DECODE_ROWS { "m4096" } else { "m64" };
            let layer_id = whole.layers[0];
            let layer = &self.weights.layers[layer_id];
            // SAFETY: startup buffers are initialized and exclusively owned on GPU 0's stream.
            unsafe {
                self.library.cuda_zero_bytes_async(w.x.buffer, w.rows * self.cfg.hidden * 2, self.stream)?;
            }
            self.moe_front_on(0, layer_id, w, layer, w.rows)?;
            self.shared_on_local(0, layer_id, w, layer, Scalar::I32(w.rows as i32), cap)?;
            let input = if whole.uses_fp8_input() { ExpertInput::Fp8K32(w.wire.buffer.ptr) }
                else { ExpertInput::Bf16(w.x.buffer.ptr) };
            // SAFETY: maximum-capacity initialized startup operands on GPU 0's stream.
            unsafe {
                whole.prime_whole(input, Routes { ids: w.route_ids.buffer.ptr, weights: w.route_weights.buffer.ptr },
                    w.rows, w.shared.buffer, w.delta.buffer, self.stream, self.library)?;
            }
        }
        let Some(pair) = &self.tp2 else {
            return Ok(());
        };
        if self.decode_workspace.borrow().is_none() {
            *self.decode_workspace.borrow_mut() = Some(self.workspace(DECODE_ROWS, true)?);
        }
        self.ensure_prefill_slots()?;
        drop(self.peer_workspaces(true, 1)?);
        for rank in 0..2 {
            // Slot 0: one unit's rows, the widest a backend sees.
            let owned = if rank == 0 {
                self.prefill_slots.borrow()
            } else {
                self.peer()?.prefill_slots.borrow()
            };
            let owned = &owned[..];
            let decode = if rank == 0 {
                self.decode_workspace.borrow()
            } else {
                self.peer()?.decode_workspace.borrow()
            };
            let w = if self.prefill_rows >= DECODE_ROWS {
                &owned[0]
            } else {
                decode.as_ref().context("decode workspace")?
            };
            self.on(rank, || {
                // SAFETY: startup buffers have no outstanding consumers or peer waits.
                unsafe {
                    self.library.cuda_zero_bytes_async(
                        w.x.buffer,
                        w.rows * self.cfg.hidden * 2,
                        self.stream_of(rank),
                    )?;
                }
                let layer = if rank == 0 {
                    &self.weights.layers[pair.layers[0]]
                } else {
                    &self.peer()?.layers[pair.layers[0]]
                };
                self.moe_front_on(rank, pair.layers[0], w, layer, w.rows)?;
                let cap = if self.prefill_rows >= DECODE_ROWS {
                    "m4096"
                } else {
                    "m64"
                };
                if Self::core_split(layer, "down") {
                    self.ffn_on(rank, w, layer, self.cfg.moe_intermediate, cap,
                        w.shared.buffer.ptr, Scalar::I32(w.rows as i32))?;
                    // SAFETY: prime uses zero x, hence zero SwiGLU/down output.
                    // Do not queue a peer wait before the other rank is primed.
                    unsafe { self.library.cuda_zero_bytes_async(w.shared.buffer,
                        w.rows * self.cfg.hidden * 2, self.stream_of(rank))?; }
                } else {
                    self.shared_on(rank, pair.layers[0], w, layer, Scalar::I32(w.rows as i32), cap)?;
                }
                let input = if pair.uses_fp8_input() {
                    ExpertInput::Fp8K32(w.wire.buffer.ptr)
                } else {
                    ExpertInput::Bf16(w.x.buffer.ptr)
                };
                // SAFETY: maximum-capacity initialized startup input, routes, shared and disjoint payload.
                unsafe {
                    pair.prime(
                        rank,
                        input,
                        Routes {
                            ids: w.route_ids.buffer.ptr,
                            weights: w.route_weights.buffer.ptr,
                        },
                        w.rows,
                        w.shared.buffer.ptr.cast(),
                        w.payload.as_ref().context("TP2 payload")?.buffer.ptr,
                    )
                }
            })?;
        }
        Ok(())
    }

    /// Local expert outputs are consumed before another lane can reuse backend scratch.
    pub(super) fn tp2_pair(
        &self,
        layer: usize,
        lane: usize,
        w: &Workspace<'_>,
        w1: &Workspace<'_>,
        rows: usize,
        cap: &str,
    ) -> Result<()> {
        let pair = self.tp2.as_ref().context("TP2 experts")?;
        self.shared_on(0, layer, w, &self.weights.layers[layer], Scalar::I32(rows as i32), cap)?;
        self.moe_front_on(1, layer, w1, &self.peer()?.layers[layer], rows)?;
        self.tp2_rank(pair, 0, layer, lane, w, rows)?;
        self.tp2_rank(pair, 1, layer, lane, w1, rows)?;
        // Rank1 has no final norm/head, but must consume every publication
        // so the next request cannot observe this step's stale FFN sequence.
        if layer + 1 == self.weights.layers.len() {
            self.exchange()?.wait(1, slot(layer, true, lane))?;
        }
        Ok(())
    }

    /// Rank-local TP2 routed expert enqueue + partial combine into `w.payload` (capture-safe
    /// when `tp2_routes_replicated()` is true).
    pub(super) fn tp2_rank_local(
        &self,
        pair: &Pair<'_>,
        rank: usize,
        layer: usize,
        lane: usize,
        w: &Workspace<'_>,
        rows: usize,
    ) -> Result<()> {
        let input = if pair.uses_fp8_input() {
            ExpertInput::Fp8K32(w.wire.buffer.ptr)
        } else {
            ExpertInput::Bf16(w.x.buffer.ptr)
        };
        let out = w.payload.as_ref().context("TP2 payload")?.buffer.ptr;
        // SAFETY: each workspace is rank-local; payload consumes backend output in stream order.
        unsafe {
            pair.enqueue(
                rank,
                layer,
                lane,
                rows,
                input,
                Routes {
                    ids: w.route_ids.buffer.ptr,
                    weights: w.route_weights.buffer.ptr,
                },
                w.router_logits.buffer,
                w.shared.buffer.ptr.cast(),
                out,
                &mut || {
                    crate::shared::decode_graph::fatal_drain(
                        self.abort_tp2(),
                        "GLM TP2 enqueue abort",
                    )
                },
            )
        }
    }

    /// Publish `rank`'s TP2 combined payload on `ffn_slot` (outside graph capture).
    pub(super) fn tp2_publish(
        &self,
        pair: &Pair<'_>,
        rank: usize,
        layer: usize,
        lane: usize,
        w: &Workspace<'_>,
        rows: usize,
    ) -> Result<()> {
        let out = w.payload.as_ref().context("TP2 payload")?.buffer.ptr;
        self.exchange()?.push(
            rank,
            slot(layer, true, lane),
            out,
            pair.combine.payload_bytes(rows, self.cfg.hidden),
        )
    }

    pub(super) fn tp2_rank(
        &self,
        pair: &Pair<'_>,
        rank: usize,
        layer: usize,
        lane: usize,
        w: &Workspace<'_>,
        rows: usize,
    ) -> Result<()> {
        self.tp2_rank_local(pair, rank, layer, lane, w, rows)?;
        self.tp2_publish(pair, rank, layer, lane, w, rows)
    }

    pub(super) fn tp2_norm(
        &self,
        rank: usize,
        w: &Workspace<'_>,
        weight: *mut c_void,
        layer: usize,
        lane: usize,
        rows: Scalar,
    ) -> Result<()> {
        let Scalar::I32(rows_count) = rows else {
            anyhow::bail!("GLM TP2 norm rows must be i32");
        };
        let pair = self.tp2.as_ref().context("TP2 pair")?;
        let local = w.payload.as_ref().context("TP2 payload")?.buffer.ptr;
        let received = self.recv(rank, slot(layer, true, lane))?;
        let (rank0, rank1) = if rank == 0 {
            (local, received)
        } else {
            (received, local)
        };
        self.on(rank, || {
            // SAFETY: FFN wait precedes the rank-ordered sum into disjoint delta storage.
            unsafe {
                pair.combine.sum(
                    rank0,
                    rank1,
                    w.delta.buffer.ptr.cast(),
                    rows_count as usize,
                    self.cfg.hidden,
                    self.stream_of(rank),
                )
            }
        })?;
        self.norm_full(
            rank,
            w,
            weight,
            1,
            rows,
            w.delta.buffer.ptr,
            w.delta.buffer.ptr,
        )
    }

    pub(crate) fn check_tp2_routes(
        &self,
        experts: Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>,
    ) -> Result<()> {
        if self.tp2.is_none() {
            return Ok(());
        }
        let rows = route_check_rows(self.prefill_rows, self.max_context);
        ensure!(rows > 0, "TP2 route diagnostics need two context rows");
        let mut allocator = PageAllocator::new(self);
        let mut placement = allocator.admit(rows + 1)?;
        let tokens: Vec<_> = (0..rows)
            .map(|i| (i % self.cfg.vocab_size) as u32)
            .collect();
        self.force_route_check.set(true);
        let mut experts = experts;
        let result: Result<()> = (|| {
            self.prefill_device(
                &mut placement,
                &tokens,
                experts.as_mut().map(|(l, r)| (&mut **l, *r)),
                1,
            )?;
            self.verify_device(
                &mut [(&mut placement, 1)],
                &[1],
                experts.as_mut().map(|(l, r)| (&mut **l, *r)),
            )?;
            Ok(())
        })();
        self.force_route_check.set(false);
        result?;
        if let Some(pair) = &self.tp2 { pair.publish_route_source(); }
        Ok(())
    }
}

fn route_check_rows(prefill_rows: usize, max_context: usize) -> usize {
    512.min(prefill_rows).min(max_context.saturating_sub(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::peer_split::order::{check, Schedule};

    #[test]
    fn diagnostic_prompt_fits_short_and_serial_all_local_lanes() {
        assert_eq!(route_check_rows(64, 4096), 64);
        assert_eq!(route_check_rows(256, 4096), 256);
        assert_eq!(route_check_rows(4096, 129), 128);
        assert_eq!(route_check_rows(4096, 1), 0);
    }

    fn schedule(
        lanes: usize,
        local: std::ops::Range<usize>,
        broadcast: bool,
        repeats: usize,
    ) -> Schedule {
        let mut s = Schedule::default();
        for _ in 0..repeats {
            for lane in 0..lanes {
                s.push_to(0, "main", DIRECT, lane, "embedding");
            }
            let mut pending = std::collections::VecDeque::new();
            let post = |s: &mut Schedule, layer, lane, remote| {
                let ffn = slot(layer, true, lane);
                if remote && layer < 7 {
                    s.push(0, "main", ffn, "Spark sum");
                }
                if layer < 7 || local.contains(&layer) {
                    s.wait(1, "main", ffn, "rank1 FFN");
                }
                s.wait(0, "main", ffn, "rank0 FFN");
            };
            for layer in 0..8 {
                for lane in 0..lanes {
                    while pending.iter().any(|&(_, k)| k == lane) {
                        let (l, k) = pending.pop_front().unwrap();
                        post(&mut s, l, k, true);
                    }
                    if layer == 0 {
                        s.wait_for(1, "main", DIRECT, lane, "embedding received");
                    }
                    for rank in 0..2 {
                        s.push(rank, "main", slot(layer, false, lane), "attention");
                        s.wait(rank, "main", slot(layer, false, lane), "attention received");
                    }
                    if local.contains(&layer) {
                        if broadcast {
                            let at = 2 * lane + layer % 2;
                            s.push(0, "routes", at, "canonical routes");
                            s.wait(0, "routes", at, "canonical consumed");
                            s.wait(1, "routes", at, "canonical received");
                            s.push(1, "routes", at, "canonical backend finished");
                        }
                        for rank in 0..2 {
                            s.push(rank, "main", slot(layer, true, lane), "paired payload");
                        }
                        post(&mut s, layer, lane, false);
                    } else {
                        s.push(1, "main", slot(layer, true, lane), "shared or dense");
                        if layer < 3 {
                            s.push(0, "main", slot(layer, true, lane), "dense");
                            post(&mut s, layer, lane, false);
                        } else {
                            pending.push_back((layer, lane));
                        }
                    }
                }
            }
            while let Some((l, k)) = pending.pop_front() {
                post(&mut s, l, k, true);
            }
            s.end_request();
        }
        s
    }

    #[test]
    fn missing_final_rank1_consumption_is_rejected() {
        let mut s = schedule(1, 3..8, true, 1);
        let at = s.streams[1].iter().rposition(|(_, label)| label == "rank1 FFN").unwrap();
        s.streams[1].remove(at);
        assert!(crate::shared::peer_split::order::check_terminal(&s, false).is_err());
    }

    #[test]
    fn abort_and_cancellation_drain_every_enqueue_prefix() {
        use crate::shared::peer_split::order::check_terminal;
        for lanes in 1..=4 {
            for end in 4..=8 {
                for broadcast in [false, true] {
                    let full = schedule(lanes, 3..end, broadcast, 1);
                    let mut ops = Vec::new();
                    for at in 0..full.streams.iter().map(Vec::len).max().unwrap() {
                        for rank in 0..2 {
                            if let Some(op) = full.streams[rank].get(at) {
                                ops.push((rank, op.clone()));
                            }
                        }
                    }
                    for failure in 0..ops.len() {
                        for cancel in [false, true] {
                            let queued = RefCell::new(Schedule::default());
                            let retired = Cell::new(false);
                            let aborted = Cell::new(false);
                            let drained = Cell::new(false);
                            let result: Result<()> = crate::shared::experts::rtx::pair::submit(
                                || { ensure!(!retired.get(), "closed"); Ok(()) },
                                || {
                                    aborted.set(true);
                                    check_terminal(&queued.borrow(), aborted.get()).map_err(anyhow::Error::msg)?;
                                    drained.set(true);
                                    retired.set(true);
                                    Ok(())
                                },
                                || {
                                    for (at, (rank, op)) in ops.iter().enumerate() {
                                        ensure!(at != failure, "{} at enqueue {at}", if cancel { "cancel" } else { "fault" });
                                        queued.borrow_mut().streams[*rank].push(op.clone());
                                    }
                                    Ok(())
                                },
                            );
                            assert!(result.is_err());
                            assert!(aborted.get() && drained.get() && retired.get());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn local_prefix_and_spark_suffix_drain_all_admitted_lanes_and_reused_flags() {
        for lanes in 1..=4 {
            for end in 4..=8 {
                for broadcast in [false, true] {
                    assert_eq!(
                        check(&schedule(lanes, 3..end, broadcast, 3)),
                        Ok(()),
                        "lanes={lanes} local=3..{end} broadcast={broadcast}"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod whole_tests {
    #[test]
    fn eager_captured_and_wave_prefill_use_the_same_local_boundary() {
        let engine = include_str!("../engine.rs");
        let local = include_str!("local_graph.rs");
        let eager = engine.split("fn step_inner(").nth(1).unwrap().split("fn ").next().unwrap();
        let prefill = engine.split("fn local_ffn(&mut self").nth(1).unwrap().split("fn send_experts(").next().unwrap();
        assert!(eager.contains("self.local_ffn(index, 0, w, w1, t, cap)?"));
        assert!(local.contains("self.whole_ffn_local(index, w, t, \"m64\")"));
        assert!(local.contains("self.tp2_rank_local(pair, 1, index, 0, w1, t)"));
        assert!(local.contains("self.tp2_rank_local(pair, 0, index, 0, w, t)"));
        assert!(prefill.contains("self.engine.local_ffn(layer, slot"));
    }
}
