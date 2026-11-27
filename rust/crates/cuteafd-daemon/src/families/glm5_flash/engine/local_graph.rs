//! Rank-local decode/verify graph executor (`glm5_flash.decode_local`) with
//! one-layer-ahead Rank-1 pipelining (#2656).
//!
//! Every `PeerExchange` and `ContextExchange` push/wait runs on the host outside
//! `replay_on`, and whenever `replay_on` runs every queued wait on either GPU
//! already has its matching push queued on the peer. Each captured CUDA graph
//! is strictly single-device (`Scope::Local`, `spans_devices = false`), so
//! runtime lazy capture and cold-donor `cudaGraphExecUpdate` stay safe after
//! readiness (#1878, #2460) and match eager arithmetic bit for bit.

use super::*;

impl<'a> GlmfEngine<'a> {
    fn local_rank_layer(&self, rank: usize, index: usize) -> Result<&GlmfLayer<'a>> {
        if rank == 0 {
            Ok(&self.weights.layers[index])
        } else {
            Ok(&self.peer()?.layers[index])
        }
    }

    fn local_tap(&self, previous: usize, w0: &Workspace<'_>, t: usize) -> Result<()> {
        if let Some(drafter) = &self.drafter {
            drafter.tap_streams(previous, w0.streams[0].buffer.ptr, HC, 0, t)?;
        }
        Ok(())
    }

    fn tp2_routes_replicated(&self) -> bool {
        self.tp2.as_ref().is_some_and(|tp2| tp2.source.get() == RouteSource::Replicated)
    }

    /// Push `rank`'s attention partial outside capture when this layer uses either
    /// native-core (`core_publish_attention`) or standard BF16/FP32 `PeerExchange`.
    /// Returns `false` only for `output_shard_attention` (two-round KDA head/row join).
    fn local_push_attention(&self, rank: usize, w: &Workspace<'_>, index: usize, t: usize)
        -> Result<bool> {
        let layer = self.local_rank_layer(rank, index)?;
        let att_slot = slot(index, false, 0);
        if self.core_publish_attention(rank, w, layer, att_slot, t)? {
            return Ok(true);
        }
        if self.output_shard_attention(layer) {
            return Ok(false);
        }
        let bytes = t * self.cfg.hidden * if self.precise_attention(layer) { 4 } else { 2 };
        self.exchange_push(self.exchange()?, "main", rank, att_slot, w.delta.buffer.ptr, bytes)?;
        Ok(true)
    }

    /// Complete two-round `output_shard_attention` outside capture.
    fn local_exchange_output_shard(&self, index: usize, w0: &Workspace<'_>, w1: &Workspace<'_>,
        t: usize, cap: &str) -> Result<[*mut c_void; 2]> {
        let layer0 = self.local_rank_layer(0, index)?;
        let layer1 = self.local_rank_layer(1, index)?;
        let att_slot = slot(index, false, 0);
        let out1 = self.complete_output_shard(1, w1, layer1, att_slot, t, cap)?;
        let out0 = self.complete_output_shard(0, w0, layer0, att_slot, t, cap)?;
        Ok([out0, out1])
    }

    /// Stage Rank 1's Phase-0 attention and outgoing push for layer `index` one layer ahead
    /// of Rank 0 (#2656), matching `decode_graphed`'s `peer_segment(index + 1)` overlap while
    /// keeping every exchange push/wait outside capture and zero unmatched waits during `replay_on`.
    ///
    /// Precondition: for `index == 0`, Rank 0 has already queued `exchange_direct`; for
    /// `index > 0`, both Rank 0 and Rank 1 have already queued `ffn[index - 1]` pushes.
    fn local_stage_rank1_attention(&self, index: usize, w1: &Workspace<'_>, tables1: &StepTables,
        t: usize, rows: Scalar, cap: &str) -> Result<()> {
        let layer1 = self.local_rank_layer(1, index)?;
        let is_context_mla = self.context.is_some() && layer1.attention == GlmNextAttention::Mla;
        let exchange = self.exchange()?;

        if index == 0 {
            self.record_direct(1, w1.streams[0].buffer.ptr, false);
            exchange.wait(1, DIRECT)?;
        } else {
            self.exchange_wait(exchange, "main", 1, slot(index - 1, true, 0))?;
        }

        let key1 = GraphKey::new_sub(index, 0, t, tables1).normalized(Some(layer1.attention));
        self.replay_on(1, key1, || {
            if index == 0 {
                self.pre_on(1, w1, &w1.streams[0], layer1, rows)?;
            } else {
                let prev_out = self.local_add_prev_ffn(1, w1, index - 1, t)?;
                self.post_pre_on(1, w1, prev_out, 1, layer1, "attn", "input_norm", rows, cap)?;
            }
            if is_context_mla {
                self.context_local_produce_score(1, w1, index, layer1, rows, cap, tables1)
            } else {
                self.attention(1, w1, index, layer1, rows, cap, tables1, None)
            }
        })?;

        if is_context_mla {
            let slot1 = self.context_local_slot(1, index)?;
            self.context_local_push_query(1, w1, slot1, t)?;
            if tables1.long {
                self.context_local_push_candidates(1, slot1, t)?;
            }
        } else {
            self.local_push_attention(1, w1, index, t)?;
        }
        Ok(())
    }

    /// Rank-local attention partial reduction in identical operand order (rank 0 then rank 1).
    /// Handles both native-core (`core_reduce_attention`) and standard BF16/FP32 exchanges.
    fn local_add_attention(&self, rank: usize, w: &Workspace<'_>, index: usize, t: usize) -> Result<*mut c_void> {
        let layer = self.local_rank_layer(rank, index)?;
        let att_slot = slot(index, false, 0);
        if core::stage_attention(layer) {
            return self.core_reduce_attention(rank, w, layer, att_slot, t);
        }
        let exchange = self.exchange()?;
        let precise = self.precise_attention(layer);
        let recv = exchange.recv(rank, att_slot)?;
        let (a, b) = if rank == 0 { (w.delta.buffer.ptr, recv) } else { (recv, w.delta.buffer.ptr) };
        if precise {
            self.run_on(rank, true, "add_fp32",
                &[("a", a), ("b", b), ("out", w.sum_ptr()?)], &[Scalar::I32(t as i32)])?;
        } else {
            exchange.add(rank, a, b, w.sum_ptr()?, t * self.cfg.hidden)?;
        }
        w.sum_ptr()
    }

    /// Rank-local FFN partial reduction in rank-0-then-rank-1 operand order.
    fn local_add_prev_ffn(&self, rank: usize, w: &Workspace<'_>, previous: usize, t: usize) -> Result<*mut c_void> {
        let Some(exchange) = &self.exchange else { return Ok(w.delta.buffer.ptr) };
        let ffn_slot = slot(previous, true, 0);
        let recv = exchange.recv(rank, ffn_slot)?;
        let (a, b) = if rank == 0 {
            (w.delta.buffer.ptr, recv)
        } else {
            let layer1 = self.local_rank_layer(1, previous)?;
            (recv, self.peer_ffn_out(previous, w, layer1)?)
        };
        exchange.add(rank, a, b, w.sum_ptr()?, t * self.cfg.hidden)?;
        w.sum_ptr()
    }

    /// Rank-local dense FFN execution on rank `rank` (no peer exchange; split native-core
    /// FFNs publish and reduce outside capture via `core_publish_ffn` / `core_finish_ffn`).
    fn local_dense_ffn(&self, rank: usize, w: &Workspace<'_>, index: usize, t: usize, cap: &str) -> Result<()> {
        let layer = self.local_rank_layer(rank, index)?;
        let rows = Scalar::I32(t as i32);
        let out = if rank == 0 { w.delta.buffer.ptr } else { self.peer_ffn_out(index, w, layer)? };
        if core::native_ffn(layer) {
            return self.core_ffn_local(rank, w, layer, cap, out, rows);
        }
        if rank == 0 {
            self.ffn(w, layer, self.cfg.dense_intermediate, cap, out, rows)
        } else if layer.has("w_gate_up_fp8") || native::dense_ffn(layer) {
            self.ffn_on(1, w, layer, self.cfg.dense_intermediate, cap, out, rows)
        } else {
            Ok(())
        }
    }

    /// Run rank-local decode/verify step under `glm5_flash.decode_local` (`Scope::Local`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_graphed_local(&self, w: &Workspace<'_>, w1: Option<&Workspace<'_>>,
        tables: &StepTables, tables1: &StepTables, t: usize, rows: Scalar, logit_rows: usize)
        -> Result<Option<DeviceLogits>> {
        let layers = &self.weights.layers;
        let head = layers.len() == self.cfg.layers && logit_rows == t;
        let gather = self.embedding.device_gather();
        let cap = decode_cap(t);

        // Single-GPU path: already 100% rank-local; capture one segment per layer + head.
        let Some(w1) = w1 else {
            for index in 0..=layers.len() {
                let key = GraphKey::new_sub(index, 0, t, tables)
                    .normalized(layers.get(index).map(|layer| layer.attention));
                let segment = || -> Result<()> {
                    let Some(layer) = layers.get(index) else {
                        self.run("mhc_post", &[("x", w.delta.buffer.ptr), ("residual", w.streams[1].buffer.ptr),
                            ("prev_post", w.post.buffer.ptr), ("prev_comb", w.comb.buffer.ptr),
                            ("out", w.streams[0].buffer.ptr)], &[rows])?;
                        if let Some(prev) = index.checked_sub(1) { self.local_tap(prev, w, t)?; }
                        if head {
                            self.run("head", &[("streams", w.streams[0].buffer.ptr),
                                ("weight", self.weights.norm.buffer.ptr), ("out", w.x.buffer.ptr)], &[rows])?;
                            self.logits(w, t, t, true)?;
                            self.select_greedy(w, t)?;
                        }
                        return Ok(());
                    };
                    if index == 0 {
                        if gather { self.gather_streams(w, t)?; }
                        self.pre(w, &w.streams[0], layer, rows)?;
                    } else {
                        self.post_pre_on(0, w, w.delta.buffer.ptr, 1, layer, "attn", "input_norm", rows, cap)?;
                        self.local_tap(index - 1, w, t)?;
                    }
                    self.attention(0, w, index, layer, rows, cap, tables, None)?;
                    self.post_pre_on(0, w, w.delta.buffer.ptr, 0, layer, "ffn", "post_norm", rows, cap)?;
                    if layer.dense {
                        self.ffn(w, layer, self.cfg.dense_intermediate, cap, w.delta.buffer.ptr, rows)
                    } else {
                        Ok(())
                    }
                };
                if index == layers.len() && !head { segment()?; } else { self.replay(key, segment)?; }
                if layers.get(index).is_some_and(|layer| !layer.dense) {
                    self.moe(w, index, &layers[index], t, rows, cap, true,
                        &[Span { first: 0, rows: t }], None)?;
                }
                if index < layers.len() { crate::shared::console::layer_mark(index); }
            }
            if layers.len() < self.cfg.layers { self.synchronize()?; return Ok(None); }
            if !head {
                let timer = std::time::Instant::now();
                self.run("head", &[("streams", w.streams[0].buffer.ptr), ("weight", self.weights.norm.buffer.ptr),
                    ("out", w.x.buffer.ptr)], &[rows])?;
                self.logits(w, t, logit_rows, tables.decode)?;
                self.profile.borrow_mut()[2] += timer.elapsed().as_secs_f64();
            }
            return Ok(Some(self.device_logits(w, logit_rows, head)));
        };

        // Two-GPU rank-local pipelined path (#2656):
        // All PeerExchange & ContextExchange push/wait run outside capture, with Rank 1
        // staged one layer ahead so its attention and outgoing push overlap Rank 0's compute
        // while keeping zero unmatched waits during every `replay_on`.
        if !layers.is_empty() {
            if gather { self.gather_streams(w, t)?; }
            self.exchange_direct(w.streams[0].buffer.ptr, w1.streams[0].buffer.ptr,
                t * HC * self.cfg.hidden * 2)?;
            self.local_stage_rank1_attention(0, w1, tables1, t, rows, cap)?;
        }

        let exchange = self.exchange()?;
        let ffn_bytes = t * self.cfg.hidden * 2;
        for index in 0..layers.len() {
            let layer0 = &layers[index];
            let layer1 = self.local_rank_layer(1, index)?;
            let is_context_mla = self.context.is_some() && layer0.attention == GlmNextAttention::Mla;
            let last = index + 1 == layers.len();

            // 1. Rank 0 Phase 0 (sub=0): wait for prev FFN (Rank 1 already pushed it) and run attention.
            if let Some(previous) = index.checked_sub(1) {
                self.exchange_wait(exchange, "main", 0, slot(previous, true, 0))?;
            }
            let key0 = GraphKey::new_sub(index, 0, t, tables).normalized(Some(layer0.attention));
            self.replay_on(0, key0, || {
                if index == 0 {
                    self.pre_on(0, w, &w.streams[0], layer0, rows)?;
                } else {
                    let prev_out = self.local_add_prev_ffn(0, w, index - 1, t)?;
                    self.post_pre_on(0, w, prev_out, 1, layer0, "attn", "input_norm", rows, cap)?;
                    self.local_tap(index - 1, w, t)?;
                }
                if is_context_mla {
                    self.context_local_produce_score(0, w, index, layer0, rows, cap, tables)
                } else {
                    self.attention(0, w, index, layer0, rows, cap, tables, None)
                }
            })?;

            // 2. Finish attention exchange (Rank 1 already pushed query/candidates or att[index]).
            let output_shard_ptrs = if is_context_mla {
                let slot0 = self.context_local_slot(0, index)?;
                let slot1 = self.context_local_slot(1, index)?;
                self.context_local_push_query(0, w, slot0, t)?;
                if tables.long {
                    self.context_local_push_candidates(0, slot0, t)?;
                }

                // Both ranks' queries (+ candidates on long) are now pushed; run Rank 1 first.
                if tables.long {
                    let (ps1, pi1) = self.context_local_wait_candidates(1, slot1)?;
                    let aq1 = self.context_local_wait_query(1, slot1)?;
                    let key1 = GraphKey::new_sub(index, 1, t, tables1).normalized(Some(layer1.attention));
                    self.replay_on(1, key1, || {
                        self.context_local_merge_and_partials(1, w1, index, rows, tables1, ps1, pi1, aq1)
                    })?;
                    self.context_local_push_partial(1, slot1, t)?;

                    let (ps0, pi0) = self.context_local_wait_candidates(0, slot0)?;
                    let aq0 = self.context_local_wait_query(0, slot0)?;
                    let key0 = GraphKey::new_sub(index, 1, t, tables).normalized(Some(layer0.attention));
                    self.replay_on(0, key0, || {
                        self.context_local_merge_and_partials(0, w, index, rows, tables, ps0, pi0, aq0)
                    })?;
                    self.context_local_push_partial(0, slot0, t)?;
                } else {
                    // Short context: peer partial -> push_partial -> own partial per rank,
                    // batched by rank (Rank 1 first) to overlap partial P2P and avoid device ping-pong.
                    let aq1 = self.context_local_wait_query(1, slot1)?;
                    self.context_local_partial(1, w1, index, rows, t, aq1, 0)?;
                    self.context_local_push_partial(1, slot1, t)?;
                    self.context_local_partial(1, w1, index, rows, t, aq1, 1)?;

                    let aq0 = self.context_local_wait_query(0, slot0)?;
                    self.context_local_partial(0, w, index, rows, t, aq0, 1)?;
                    self.context_local_push_partial(0, slot0, t)?;
                    self.context_local_partial(0, w, index, rows, t, aq0, 0)?;
                }

                // Both ranks' partials are pushed: wait -> combine_out -> push attention (Rank 1 first).
                let recv1 = self.context_local_wait_partial(1, slot1)?;
                self.context_local_combine_out(1, w1, layer1, rows, cap, t, recv1)?;
                let pushed = self.local_push_attention(1, w1, index, t)?;

                let recv0 = self.context_local_wait_partial(0, slot0)?;
                self.context_local_combine_out(0, w, layer0, rows, cap, t, recv0)?;
                if pushed {
                    self.local_push_attention(0, w, index, t)?;
                    None
                } else {
                    Some(self.local_exchange_output_shard(index, w, w1, t, cap)?)
                }
            } else if self.local_push_attention(0, w, index, t)? {
                None
            } else {
                Some(self.local_exchange_output_shard(index, w, w1, t, cap)?)
            };

            // 3. Post-attention + FFN + FFN pushes (Rank 1 first so GPU 1 stays ahead of GPU 0).
            let att_slot = slot(index, false, 0);
            let ffn_slot = slot(index, true, 0);
            let split_core_ffn = core::split_trellis_ffn(layer0);
            if layer0.dense {
                // Rank 1 dense FFN (strictly rank-local capture) -> push!
                if output_shard_ptrs.is_none() {
                    self.exchange_wait(exchange, "main", 1, att_slot)?;
                }
                let key1 = GraphKey::new_sub(index, 1, t, tables1).normalized(Some(layer1.attention));
                let pre1 = output_shard_ptrs.map(|p| p[1]);
                self.replay_on(1, key1, || {
                    let attended = match pre1 {
                        Some(ptr) => ptr,
                        None => self.local_add_attention(1, w1, index, t)?,
                    };
                    self.post_pre_on(1, w1, attended, 0, layer1, "ffn", "post_norm", rows, cap)?;
                    self.local_dense_ffn(1, w1, index, t, cap)
                })?;
                let out1 = self.peer_ffn_out(index, w1, layer1)?;
                if split_core_ffn {
                    // One-way 1->0 core partial: Rank 1 zeros its local output and publishes immediately
                    // without waiting for Rank 0.
                    self.core_publish_ffn(1, w1, index, t)?;
                    self.core_zero_ffn(1, w1, out1, t)?;
                    self.exchange_push(exchange, "main", 1, ffn_slot, out1, ffn_bytes)?;
                } else {
                    self.exchange_push(exchange, "main", 1, ffn_slot, out1, ffn_bytes)?;
                }

                // Rank 0 dense FFN (strictly rank-local capture) -> reduce (if split_core) -> push!
                if output_shard_ptrs.is_none() {
                    self.exchange_wait(exchange, "main", 0, att_slot)?;
                }
                let key0 = GraphKey::new_sub(index, 1, t, tables).normalized(Some(layer0.attention));
                let pre0 = output_shard_ptrs.map(|p| p[0]);
                self.replay_on(0, key0, || {
                    let attended = match pre0 {
                        Some(ptr) => ptr,
                        None => self.local_add_attention(0, w, index, t)?,
                    };
                    self.post_pre_on(0, w, attended, 0, layer0, "ffn", "post_norm", rows, cap)?;
                    self.local_dense_ffn(0, w, index, t, cap)
                })?;
                if split_core_ffn {
                    self.core_finish_ffn(w, layer0, index, w.delta.buffer.ptr, t)?;
                }
                if !last {
                    self.exchange_push(exchange, "main", 0, ffn_slot, w.delta.buffer.ptr, ffn_bytes)?;
                }
            } else if self.tp2_layer(index) && self.tp2_routes_replicated() {
                // Production TP2 MoE (replicated routers): run Rank 1's entire post-attention,
                // shared FFN, router, EXL3 routed experts and publish without stalling on Rank 0.
                if output_shard_ptrs.is_none() {
                    self.exchange_wait(exchange, "main", 1, att_slot)?;
                }
                let att1 = match output_shard_ptrs {
                    Some(ptrs) => ptrs[1],
                    None => self.local_add_attention(1, w1, index, t)?,
                };
                self.post_pre_on(1, w1, att1, 0, layer1, "ffn", "post_norm", rows, cap)?;
                if split_core_ffn {
                    self.core_ffn_local(1, w1, layer1, cap, w1.shared.buffer.ptr, rows)?;
                    self.core_publish_ffn(1, w1, index, t)?;
                    self.core_zero_ffn(1, w1, w1.shared.buffer.ptr, t)?;
                    self.router_on(1, w1, layer1, t, &[Span { first: 0, rows: t }])?;
                    self.tp2_rank(1, index, 0, w1, t, w1.delta.buffer.ptr)?;
                    self.tp2_publish(index, 0, t, Some(w1))?;

                    if output_shard_ptrs.is_none() {
                        self.exchange_wait(exchange, "main", 0, att_slot)?;
                    }
                    let att0 = match output_shard_ptrs {
                        Some(ptrs) => ptrs[0],
                        None => self.local_add_attention(0, w, index, t)?,
                    };
                    self.post_pre_on(0, w, att0, 0, layer0, "ffn", "post_norm", rows, cap)?;
                    self.moe_front(w, index, layer0, t, rows, cap, &[Span { first: 0, rows: t }])?;
                    self.core_ffn_local(0, w, layer0, cap, w.shared.buffer.ptr, rows)?;
                    self.core_finish_ffn(w, layer0, index, w.shared.buffer.ptr, t)?;
                    self.tp2_rank(0, index, 0, w, t, w.delta.buffer.ptr)?;
                    if !last {
                        self.exchange_push(exchange, "main", 0, ffn_slot, w.delta.buffer.ptr, ffn_bytes)?;
                    }
                } else {
                    self.ffn_on(1, w1, layer1, self.cfg.moe_intermediate, cap, w1.shared.buffer.ptr, rows)?;
                    self.router_on(1, w1, layer1, t, &[Span { first: 0, rows: t }])?;
                    self.tp2_rank(1, index, 0, w1, t, w1.delta.buffer.ptr)?;
                    self.tp2_publish(index, 0, t, Some(w1))?;

                    if output_shard_ptrs.is_none() {
                        self.exchange_wait(exchange, "main", 0, att_slot)?;
                    }
                    let att0 = match output_shard_ptrs {
                        Some(ptrs) => ptrs[0],
                        None => self.local_add_attention(0, w, index, t)?,
                    };
                    self.post_pre_on(0, w, att0, 0, layer0, "ffn", "post_norm", rows, cap)?;
                    self.moe_front(w, index, layer0, t, rows, cap, &[Span { first: 0, rows: t }])?;
                    self.ffn(w, layer0, self.cfg.moe_intermediate, cap, w.shared.buffer.ptr, rows)?;
                    self.tp2_rank(0, index, 0, w, t, w.delta.buffer.ptr)?;
                    if !last {
                        self.exchange_push(exchange, "main", 0, ffn_slot, w.delta.buffer.ptr, ffn_bytes)?;
                    }
                }
            } else if self.tp2_layer(index) {
                // Broadcast-route TP2 fallback: preserve `tp2_pair`'s rank-0-first route broadcast order.
                if output_shard_ptrs.is_none() {
                    self.exchange_wait(exchange, "main", 0, att_slot)?;
                    self.exchange_wait(exchange, "main", 1, att_slot)?;
                }
                let att0 = match output_shard_ptrs {
                    Some(ptrs) => ptrs[0],
                    None => self.local_add_attention(0, w, index, t)?,
                };
                self.post_pre_on(0, w, att0, 0, layer0, "ffn", "post_norm", rows, cap)?;
                let att1 = match output_shard_ptrs {
                    Some(ptrs) => ptrs[1],
                    None => self.local_add_attention(1, w1, index, t)?,
                };
                self.post_pre_on(1, w1, att1, 0, layer1, "ffn", "post_norm", rows, cap)?;
                self.moe(w, index, layer0, t, rows, cap, true,
                    &[Span { first: 0, rows: t }], Some(w1))?;
                self.tp2_publish(index, 0, t, Some(w1))?;
                if !last {
                    self.exchange_push(exchange, "main", 0, ffn_slot, w.delta.buffer.ptr, ffn_bytes)?;
                }
            } else {
                // Non-TP2 (Spark or local-on-rank0) MoE: finish Rank 1's shared FFN, push ffn[index]
                // and issue Rank 1 L2 weight prefetch before Rank 0 enters Spark/local MoE.
                if output_shard_ptrs.is_none() {
                    self.exchange_wait(exchange, "main", 1, att_slot)?;
                }
                let att1 = match output_shard_ptrs {
                    Some(ptrs) => ptrs[1],
                    None => self.local_add_attention(1, w1, index, t)?,
                };
                self.post_pre_on(1, w1, att1, 0, layer1, "ffn", "post_norm", rows, cap)?;
                let out1 = self.peer_ffn_out(index, w1, layer1)?;
                self.ffn_on(1, w1, layer1, self.cfg.moe_intermediate, cap, out1, rows)?;
                self.exchange_push(exchange, "main", 1, ffn_slot, out1, ffn_bytes)?;
                if !last {
                    if let Some(l2) = self.peer()?.l2.as_ref() {
                        self.on(1, || l2.issue(self.library, index, self.stream_of(1)))?;
                    }
                }

                if output_shard_ptrs.is_none() {
                    self.exchange_wait(exchange, "main", 0, att_slot)?;
                }
                let att0 = match output_shard_ptrs {
                    Some(ptrs) => ptrs[0],
                    None => self.local_add_attention(0, w, index, t)?,
                };
                self.post_pre_on(0, w, att0, 0, layer0, "ffn", "post_norm", rows, cap)?;
                self.moe(w, index, layer0, t, rows, cap, true,
                    &[Span { first: 0, rows: t }], Some(w1))?;
                if !last {
                    self.exchange_push(exchange, "main", 0, ffn_slot, w.delta.buffer.ptr, ffn_bytes)?;
                }
            }

            // 4. Pipeline Rank 1 one layer ahead: both ffn[index] pushes are queued, so staging
            // layer `index + 1` on Rank 1 leaves zero unmatched waits for the next `replay_on`.
            if !last {
                self.local_stage_rank1_attention(index + 1, w1, tables1, t, rows, cap)?;
            }
            crate::shared::console::layer_mark(index);
        }

        // 5. Final layer FFN wait on Rank 0 (Rank 1 already pushed ffn[last]) -> local Tail.
        if let Some(last) = layers.len().checked_sub(1) {
            self.exchange_wait(exchange, "main", 0, slot(last, true, 0))?;
        }
        let tail_key = GraphKey::new_sub(layers.len(), 0, t, tables).normalized(None);
        let tail_segment = || -> Result<()> {
            let out = match layers.len().checked_sub(1) {
                Some(last) => self.local_add_prev_ffn(0, w, last, t)?,
                None => w.delta.buffer.ptr,
            };
            self.run("mhc_post", &[("x", out), ("residual", w.streams[1].buffer.ptr),
                ("prev_post", w.post.buffer.ptr), ("prev_comb", w.comb.buffer.ptr),
                ("out", w.streams[0].buffer.ptr)], &[rows])?;
            if let Some(last) = layers.len().checked_sub(1) {
                self.local_tap(last, w, t)?;
            }
            if head {
                self.run("head", &[("streams", w.streams[0].buffer.ptr),
                    ("weight", self.weights.norm.buffer.ptr), ("out", w.x.buffer.ptr)], &[rows])?;
                self.logits(w, t, t, true)?;
                self.select_greedy(w, t)?;
            }
            Ok(())
        };
        if !head { tail_segment()?; } else { self.replay(tail_key, tail_segment)?; }
        self.replicate_terminal(w, w1, 0, t)?;
        if layers.len() < self.cfg.layers {
            self.synchronize()?;
            return Ok(None);
        }
        if !head {
            let timer = std::time::Instant::now();
            self.run("head", &[("streams", w.streams[0].buffer.ptr), ("weight", self.weights.norm.buffer.ptr),
                ("out", w.x.buffer.ptr)], &[rows])?;
            self.logits(w, t, logit_rows, tables.decode)?;
            self.profile.borrow_mut()[2] += timer.elapsed().as_secs_f64();
        }
        Ok(Some(self.device_logits(w, logit_rows, head)))
    }
}

#[cfg(test)]
mod tests {
    use crate::shared::peer_split::order::{self, host::{check_host, HostSchedule}};

    #[test]
    fn local_decode_captures_contain_no_peer_or_context_exchanges() {
        let source = include_str!("local_graph.rs");
        let body = source.split_once("#[cfg(test)]").unwrap().0;
        // Every replay_on / replay closure in local_graph.rs must be strictly rank-local.
        let mut count = 0usize;
        for part in body.split("self.replay_on(").skip(1) {
            count += 1;
            let closure = part.split_once("|| {").unwrap().1.split_once("})?;").unwrap().0;
            for forbidden in ["exchange_push", "exchange_wait", "exchange_direct", "push_query",
                "wait_query", "push_candidates", "wait_candidates", "push_partial", "wait_partial",
                "replicate_terminal", "tp2_publish", "core_attention_meet", "core_publish",
                "core_finish_ffn", "core_ffn(", "complete_output_shard"] {
                assert!(!closure.contains(forbidden),
                    "rank-local capture closure unexpectedly contains {forbidden}");
            }
        }
        assert!(count >= 6, "expected rank-local replay_on sites in local_graph.rs, found {count}");
        // `context_local_partial` requires the assembled [rows, H, D] query returned by
        // `wait_query`, never the half-stride producer buffer `w.query.buffer.ptr`.
        let context_src = include_str!("context.rs");
        assert!(!context_src.contains("context_local_partial(rank, w, index, rows, t, w.query.buffer.ptr"),
            "context_local_partial must never receive half-stride w.query.buffer.ptr");
    }

    #[test]
    fn pipelined_local_decode_schedule_is_deadlock_free_at_every_capture_point() {
        use super::slot;
        use crate::shared::peer_split::DIRECT;

        #[derive(Clone, Copy)]
        enum Mode { Dense, Tp2Replicated, Tp2Broadcast, Spark }

        let layers = [
            (Mode::Dense, false),
            (Mode::Dense, true),
            (Mode::Tp2Replicated, false),
            (Mode::Tp2Replicated, true),
            (Mode::Tp2Broadcast, false),
            (Mode::Spark, true),
            (Mode::Tp2Replicated, false),
        ];

        for long in [false, true] {
            for split_core in [false, true] {
                let mut s = HostSchedule::default();
                let capture_point = |s: &mut HostSchedule, label: &str| {
                    // Worst-case `cudaGraphInstantiate` / `cudaGraphExecUpdate` may synchronize
                    // either GPU: every queued wait on both streams must already be matched.
                    s.sync(0, format!("capture {label} sync gpu0"));
                    s.sync(1, format!("capture {label} sync gpu1"));
                };
                let stage_r1 = |s: &mut HostSchedule, index: usize, is_ctx: bool| {
                    if index == 0 {
                        s.wait_for(1, "main", DIRECT, 1, "L0 direct");
                    } else {
                        s.wait(1, "main", slot(index - 1, true, 0), format!("L{index} r1 wait prev ffn"));
                    }
                    capture_point(s, &format!("L{index} r1 sub0"));
                    if is_ctx {
                        let cs = index % 2;
                        s.push(1, "ctx_q", cs, format!("L{index} r1 push q"));
                        if long { s.push(1, "ctx_c", cs, format!("L{index} r1 push cand")); }
                    } else {
                        s.push(1, "main", slot(index, false, 0), format!("L{index} r1 push att"));
                    }
                };

                s.push_to(0, "main", DIRECT, 1, "L0 direct");
                stage_r1(&mut s, 0, layers[0].1);

                for (index, &(mode, is_ctx)) in layers.iter().enumerate() {
                    let last = index + 1 == layers.len();
                    let att = slot(index, false, 0);
                    let ffn = slot(index, true, 0);

                    if index > 0 {
                        s.wait(0, "main", slot(index - 1, true, 0), format!("L{index} r0 wait prev ffn"));
                    }
                    capture_point(&mut s, &format!("L{index} r0 sub0"));

                    if is_ctx {
                        let cs = index % 2;
                        s.push(0, "ctx_q", cs, format!("L{index} r0 push q"));
                        if long { s.push(0, "ctx_c", cs, format!("L{index} r0 push cand")); }
                        if long {
                            s.wait(1, "ctx_c", cs, format!("L{index} r1 wait cand"));
                            s.wait(1, "ctx_q", cs, format!("L{index} r1 wait q"));
                            capture_point(&mut s, &format!("L{index} r1 ctx sub1"));
                            s.push(1, "ctx_p", cs, format!("L{index} r1 push partial"));

                            s.wait(0, "ctx_c", cs, format!("L{index} r0 wait cand"));
                            s.wait(0, "ctx_q", cs, format!("L{index} r0 wait q"));
                            capture_point(&mut s, &format!("L{index} r0 ctx sub1"));
                            s.push(0, "ctx_p", cs, format!("L{index} r0 push partial"));
                        } else {
                            s.wait(1, "ctx_q", cs, format!("L{index} r1 wait q"));
                            s.push(1, "ctx_p", cs, format!("L{index} r1 push partial"));
                            s.wait(0, "ctx_q", cs, format!("L{index} r0 wait q"));
                            s.push(0, "ctx_p", cs, format!("L{index} r0 push partial"));
                        }
                        s.wait(1, "ctx_p", cs, format!("L{index} r1 wait partial"));
                        s.push(1, "main", att, format!("L{index} r1 push att"));
                        s.wait(0, "ctx_p", cs, format!("L{index} r0 wait partial"));
                        s.push(0, "main", att, format!("L{index} r0 push att"));
                    } else {
                        s.push(0, "main", att, format!("L{index} r0 push att"));
                    }

                    match mode {
                        Mode::Dense => {
                            s.wait(1, "main", att, format!("L{index} r1 wait att"));
                            capture_point(&mut s, &format!("L{index} r1 dense sub1"));
                            if split_core {
                                s.push_to(1, "main", ffn, 1, format!("L{index} r1 push core_ffn"));
                                s.push(1, "main", ffn, format!("L{index} r1 push ffn"));
                            } else {
                                s.push(1, "main", ffn, format!("L{index} r1 push ffn"));
                            }
                            s.wait(0, "main", att, format!("L{index} r0 wait att"));
                            capture_point(&mut s, &format!("L{index} r0 dense sub1"));
                            if split_core {
                                s.wait_for(0, "main", ffn, 1, format!("L{index} r0 wait core_ffn"));
                            }
                            if !last {
                                s.push(0, "main", ffn, format!("L{index} r0 push ffn"));
                            }
                        }
                        Mode::Tp2Replicated => {
                            s.wait(1, "main", att, format!("L{index} r1 wait att"));
                            if split_core {
                                s.push_to(1, "main", ffn, 1, format!("L{index} r1 push core_ffn"));
                                s.push(1, "main", ffn, format!("L{index} r1 push ffn"));
                                s.wait(0, "main", att, format!("L{index} r0 wait att"));
                                s.wait_for(0, "main", ffn, 1, format!("L{index} r0 wait core_ffn"));
                                if !last { s.push(0, "main", ffn, format!("L{index} r0 push ffn")); }
                            } else {
                                s.push(1, "main", ffn, format!("L{index} r1 push ffn"));
                                s.wait(0, "main", att, format!("L{index} r0 wait att"));
                                if !last { s.push(0, "main", ffn, format!("L{index} r0 push ffn")); }
                            }
                        }
                        Mode::Tp2Broadcast => {
                            s.wait(0, "main", att, format!("L{index} r0 wait att"));
                            s.wait(1, "main", att, format!("L{index} r1 wait att"));
                            s.push(0, "routes", index % 2, format!("L{index} r0 push routes"));
                            s.wait(1, "routes", index % 2, format!("L{index} r1 wait routes"));
                            s.push(1, "main", ffn, format!("L{index} r1 push ffn"));
                            if !last { s.push(0, "main", ffn, format!("L{index} r0 push ffn")); }
                        }
                        Mode::Spark => {
                            s.wait(1, "main", att, format!("L{index} r1 wait att"));
                            s.push(1, "main", ffn, format!("L{index} r1 push ffn"));
                            s.wait(0, "main", att, format!("L{index} r0 wait att"));
                            s.sync(0, format!("L{index} spark host wait"));
                            if !last { s.push(0, "main", ffn, format!("L{index} r0 push ffn")); }
                        }
                    }

                    if !last {
                        stage_r1(&mut s, index + 1, layers[index + 1].1);
                    }
                }

                s.wait(0, "main", slot(layers.len() - 1, true, 0), "tail wait last ffn");
                capture_point(&mut s, "tail");
                s.push_to(0, "main", DIRECT, 2, "terminal");
                s.wait_for(1, "main", DIRECT, 2, "terminal");

                check_host(&s).expect("pipelined local decode schedule must never deadlock at any capture or sync");
                order::check_submission_faults(&s.streams()).expect("submission streams must drain cleanly");
            }
        }
    }
}
