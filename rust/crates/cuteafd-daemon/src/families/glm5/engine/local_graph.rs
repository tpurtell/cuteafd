//! Rank-local exact-row decode/verify graph executor (`glm5.decode_local`) with
//! graphed local/TP2 MoE FFN and one-layer-ahead Rank-1 pipelining (#2609).
//!
//! Every `PeerExchange` and `ContextExchange` push/wait runs on the host outside
//! `replay_on`, and whenever `replay_on` runs every queued wait on either GPU
//! already has its matching push queued on the peer. Each captured CUDA graph
//! is strictly single-device (`Scope::Local`, `spans_devices = false`), so
//! runtime lazy capture and cold-donor `cudaGraphExecUpdate` stay safe after
//! readiness (#1878, #2460) and match eager arithmetic bit for bit.

use super::*;

impl<'a> GlmEngine<'a> {
    fn local_rank_layer(&self, rank: usize, index: usize) -> Result<&GlmLayer<'a>> {
        if rank == 0 {
            Ok(&self.weights.layers[index])
        } else {
            Ok(&self.peer()?.layers[index])
        }
    }

    fn local_key(layer: usize, sub: u8, rows: usize, tables: &StepTables,
        reads_page_table: bool, previous: Previous, head: bool) -> GraphKey {
        GraphKey {
            layer,
            sub,
            rows,
            table_width: tables.table_width,
            table_stride: tables.table_stride,
            previous,
            head,
        }
        .normalized(reads_page_table)
    }

    /// Stage Rank 1's Phase-0 (`sub = 0`) attention and outgoing push for layer `index`
    /// one layer ahead of Rank 0, keeping every exchange push/wait outside capture and
    /// leaving zero unmatched waits during every `replay_on`.
    ///
    /// Precondition: for `index == 0`, Rank 0 has already queued `DIRECT`; for
    /// `index > 0`, Rank 0 has already queued `ffn[index - 1]` push.
    #[allow(clippy::too_many_arguments)]
    fn local_stage_rank1_attention(&self, index: usize, previous: Previous, w1: &Workspace<'_>,
        ranked: &[StepTables; 2], t: usize, rows: Scalar) -> Result<()> {
        let tables1 = &ranked[1];
        let layer1 = self.local_rank_layer(1, index)?;
        let exchange = self.exchange()?;
        let is_context = self.context_kv();

        if index == 0 {
            exchange.wait(1, DIRECT)?;
        } else {
            exchange.wait(1, slot(index - 1, true, 0))?;
        }

        let key1 = Self::local_key(index, 0, t, tables1, layer1.full_indexer, previous, false);
        self.replay_on(1, key1, || {
            if index == 0 {
                self.norm_full(1, w1, layer1.ptr("input_norm")?, 0, rows,
                    w1.delta.buffer.ptr, w1.delta.buffer.ptr)?;
            } else {
                let before = index - 1;
                let prev_layer = self.local_rank_layer(1, before)?;
                if self.tp2_layer(before) {
                    self.tp2_norm(1, w1, layer1.ptr("input_norm")?, before, 0, rows)?;
                } else {
                    let ffn = slot(before, true, 0);
                    let received = self.recv(1, ffn)?;
                    let (first, second) = if prev_layer.dense {
                        (w1.delta.buffer.ptr, received)
                    } else {
                        (received, w1.shared.buffer.ptr)
                    };
                    self.ffn_norm(1, before, 0, w1, prev_layer, layer1.ptr("input_norm")?,
                        rows, first, second)?;
                }
            }
            if is_context {
                self.context_local_produce_score(1, w1, index, layer1, rows, t, tables1)
            } else {
                self.attention_unit(1, w1, index, layer1, rows, "m64", ranked, 0)
            }
        })?;

        if is_context {
            let cslot = self.context_local_slot(index);
            self.context_local_push_query(1, w1, index, cslot, t)?;
            if layer1.full_indexer {
                self.context_local_push_candidates(1, cslot, t)?;
            }
        } else {
            self.publish_projection(1, index, 0, w1, layer1, "o", t)?;
        }
        Ok(())
    }

    /// Rank-local decode/verify layers (`glm5.decode_local`, `Scope::Local`).
    ///
    /// On 1x RTX, onboarded `Whole` MoE layers run `moe_front + whole_ffn_local` inside the
    /// layer's captured graph segment (zero eager launches). On 2x RTX (`heads` / `context`),
    /// every `PeerExchange` and `ContextExchange` push/wait runs outside capture while TP2 MoE,
    /// dense FFN, Spark MoE front, context attention, and terminal head all run inside
    /// strictly rank-local CUDA graphs with 1-layer-ahead Rank-1 pipelining.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_layers_local(&self, w: &Workspace<'_>, w1: Option<&Workspace<'_>>,
        ranked: &[StepTables; 2], t: usize,
        experts: &mut Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>,
        gather: bool, head: bool) -> Result<()> {
        let tables = &ranked[0];
        let rows = Scalar::I32(t as i32);
        let layers = &self.weights.layers;
        let bytes = t * self.cfg.hidden * 2;

        // Single-GPU path: 100% rank-0 local; onboarded Whole MoE runs inside capture.
        let Some(w1) = w1 else {
            let reduction = match (experts.as_mut(), &self.skip) {
                (_, Some(_)) => None,
                (Some((transport, _)), None) => {
                    let waves = transport.exchange()?;
                    ensure!(waves.waves() == 1, "GLM decode graphs bake one return slot's planes");
                    Some(waves.reduction(0))
                }
                (None, None) => None,
            };
            let mut previous = Previous::First;
            for index in 0..=layers.len() {
                let layer = layers.get(index);
                let segment = || -> Result<()> {
                    if index == 0 && gather {
                        // SAFETY: the step's ids are in `ids` before replay; `h` holds its rows.
                        unsafe {
                            self.embedding.gather(w.ids.buffer.ptr, std::ptr::null(), t, 1,
                                std::ptr::null(), w.h.buffer.ptr, None,
                                RankDevice { device: self.device, stream: self.stream })?;
                        }
                    }
                    if let Previous::Planes(ranks) = previous {
                        if let Some(reduction) = &reduction {
                            // SAFETY: landed decode slot and workspace outlive graph replay.
                            unsafe {
                                reduction.launch(self.library, w.shared.buffer.ptr.cast(),
                                    w.delta.buffer.ptr.cast(), t, self.stream)?;
                            }
                        } else {
                            self.reduce(self.skip.as_ref().context("skip intake")?.pointers(), w, ranks, t)?;
                        }
                    }
                    let weight = match layer {
                        Some(layer) => layer.ptr("input_norm")?,
                        None => self.weights.norm.buffer.ptr,
                    };
                    let deltas = if matches!(previous, Previous::First) { 0 } else { 1 };
                    self.run(&self.family_program("norm"), &[("residual", w.h.buffer.ptr),
                        ("delta0", w.delta.buffer.ptr), ("delta1", w.delta.buffer.ptr),
                        ("weight", weight), ("out", w.x.buffer.ptr)],
                        &[rows, Scalar::I32(deltas)])?;
                    if let (Some(drafter), Some(prev)) = (&self.drafter, index.checked_sub(1)) {
                        drafter.tap(prev, w.h.buffer.ptr, 0, t)?;
                    }
                    let Some(layer) = layer else {
                        if head {
                            self.launch_head(w, t, t)?;
                            self.select_greedy(w, t)?;
                        }
                        return Ok(());
                    };
                    self.attention_unit(0, w, index, layer, rows, "m64", ranked, 0)?;
                    self.run(&self.family_program("norm"), &[("residual", w.h.buffer.ptr),
                        ("delta0", w.delta.buffer.ptr), ("delta1", w.delta.buffer.ptr),
                        ("weight", layer.ptr("post_norm")?), ("out", w.x.buffer.ptr)],
                        &[rows, Scalar::I32(1)])?;
                    if layer.dense {
                        self.ffn(w, layer, self.cfg.dense_intermediate, "m64", w.delta.buffer.ptr, rows)
                    } else if self.local_layer(index) {
                        self.moe_front(index, w, layer, t)?;
                        self.whole_ffn_local(index, w, t, "m64")
                    } else {
                        self.moe_front(index, w, layer, t)
                    }
                };
                let key = Self::local_key(index, 0, t, tables,
                    layer.is_some_and(|l| l.full_indexer), previous, head && layer.is_none());
                {
                    let _range = crate::shared::probe::verify_timing::range(Some(self.library), || match layer {
                        Some(layer) if layer.dense => format!("L{index} attention+dense"),
                        Some(_) if self.local_layer(index) => format!("L{index} attention+local_moe"),
                        Some(_) => format!("L{index} attention+router"),
                        None => "head".into(),
                    });
                    if layer.is_none() && !head {
                        segment()?;
                    } else {
                        self.replay_on(0, key, segment)?;
                    }
                }
                if let (Previous::Planes(_), None, Some((transport, _))) = (previous, &self.skip, experts.as_mut()) {
                    transport.exchange()?.consumed(0, self.stream)?;
                }
                previous = match layer {
                    None => break,
                    Some(layer) if layer.dense || self.local_layer(index) => Previous::Delta,
                    Some(layer) if self.skip.is_some() => Previous::Planes(
                        self.moe_skip(w, index, layer, t, "m64")?),
                    Some(layer) => {
                        let _range = crate::shared::probe::verify_timing::range(Some(self.library),
                            || format!("L{index} experts exchange {t}/{t}"));
                        let (transport, _) = experts.as_mut()
                            .with_context(|| format!("layer {index} is an MoE layer; pass Spark peers for its routed experts"))?;
                        let waves = transport.exchange()?;
                        self.moe_exchange(w, index, layer, t, "m64", true, waves)?;
                        Previous::Planes(waves.ranks())
                    }
                };
                crate::shared::console::layer_mark(index);
            }
            return Ok(());
        };

        // Two-GPU rank-local pipelined path (`heads` / `context`).
        let exchange = self.exchange()?;
        let is_context = self.context_kv();
        if !layers.is_empty() {
            if gather {
                // SAFETY: `ids` holds `t` token ids; `h` holds `t` rows on GPU 0's stream.
                unsafe {
                    self.embedding.gather(w.ids.buffer.ptr, std::ptr::null(), t, 1,
                        std::ptr::null(), w.h.buffer.ptr, None,
                        RankDevice { device: self.device, stream: self.stream })?;
                }
            }
            exchange.push_to(0, DIRECT, w.h.buffer.ptr, w1.h.buffer.ptr, bytes)?;
            self.local_stage_rank1_attention(0, Previous::First, w1, ranked, t, rows)?;
        }

        let mut prev_r0 = Previous::First;
        for index in 0..layers.len() {
            let layer0 = &layers[index];
            let layer1 = self.local_rank_layer(1, index)?;
            let last = index + 1 == layers.len();
            let att_slot = slot(index, false, 0);
            let ffn_slot = slot(index, true, 0);

            // 1. Rank 0 Phase 0 (`sub = 0`): wait for prev FFN (Rank 1 already pushed it) and run attention.
            if let Some(before) = index.checked_sub(1) {
                exchange.wait(0, slot(before, true, 0))?;
            }
            let key0 = Self::local_key(index, 0, t, tables, layer0.full_indexer, prev_r0, false);
            self.replay_on(0, key0, || {
                if index == 0 {
                    self.norm_full(0, w, layer0.ptr("input_norm")?, 0, rows,
                        w.delta.buffer.ptr, w.delta.buffer.ptr)?;
                } else {
                    let before = index - 1;
                    if self.tp2_layer(before) {
                        self.tp2_norm(0, w, layer0.ptr("input_norm")?, before, 0, rows)?;
                    } else {
                        let prev_ffn = slot(before, true, 0);
                        self.ffn_norm(0, before, 0, w, &layers[before], layer0.ptr("input_norm")?,
                            rows, w.delta.buffer.ptr, self.recv(0, prev_ffn)?)?;
                    }
                    if let Some(drafter) = &self.drafter {
                        drafter.tap(before, w.h.buffer.ptr, 0, t)?;
                    }
                }
                if is_context {
                    self.context_local_produce_score(0, w, index, layer0, rows, t, tables)
                } else {
                    self.attention_unit(0, w, index, layer0, rows, "m64", ranked, 0)
                }
            })?;

            // 2. Finish attention exchange (Rank 1 already pushed query/candidates or `o`).
            if is_context {
                let cslot = self.context_local_slot(index);
                self.context_local_push_query(0, w, index, cslot, t)?;
                if layer0.full_indexer {
                    self.context_local_push_candidates(0, cslot, t)?;

                    let (ps1, pi1) = self.context_local_wait_candidates(1, cslot)?;
                    let aq1 = self.context_local_wait_query(1, cslot)?;
                    let key1 = Self::local_key(index, 1, t, &ranked[1], true, Previous::First, false);
                    self.replay_on(1, key1, || {
                        self.context_local_merge_and_partials(1, w1, index, rows, t, &ranked[1], ps1, pi1, aq1)
                    })?;
                    self.context_local_push_partial(1, cslot, t)?;

                    let (ps0, pi0) = self.context_local_wait_candidates(0, cslot)?;
                    let aq0 = self.context_local_wait_query(0, cslot)?;
                    let key0 = Self::local_key(index, 1, t, tables, true, Previous::First, false);
                    self.replay_on(0, key0, || {
                        self.context_local_merge_and_partials(0, w, index, rows, t, tables, ps0, pi0, aq0)
                    })?;
                    self.context_local_push_partial(0, cslot, t)?;
                } else {
                    // Shared-indexer layer: overlap peer partial P2P with own partial compute
                    // (two kernels total, avoids <=2-node GRAPH_TOO_SMALL).
                    let aq1 = self.context_local_wait_query(1, cslot)?;
                    self.context_local_partial(1, w1, index, rows, t, aq1, 0)?;
                    self.context_local_push_partial(1, cslot, t)?;
                    self.context_local_partial(1, w1, index, rows, t, aq1, 1)?;

                    let aq0 = self.context_local_wait_query(0, cslot)?;
                    self.context_local_partial(0, w, index, rows, t, aq0, 1)?;
                    self.context_local_push_partial(0, cslot, t)?;
                    self.context_local_partial(0, w, index, rows, t, aq0, 0)?;
                }

                // Multi-kernel o_proj (K6/K8/native) exceeds 2 nodes and captures cleanly.
                let graph_combine = layer0.core.contains_key("o")
                    || layer0.attention_native.contains_key("o")
                    || !layer0.native_projections.is_empty();
                let recv1 = self.context_local_wait_partial(1, cslot)?;
                if graph_combine {
                    let key1 = Self::local_key(index, 2, t, &ranked[1], false, Previous::First, false);
                    self.replay_on(1, key1, || {
                        self.context_local_combine_out(1, w1, layer1, rows, t, recv1)
                    })?;
                } else {
                    self.context_local_combine_out(1, w1, layer1, rows, t, recv1)?;
                }
                self.publish_projection(1, index, 0, w1, layer1, "o", t)?;

                let recv0 = self.context_local_wait_partial(0, cslot)?;
                if graph_combine {
                    let key0 = Self::local_key(index, 2, t, tables, false, Previous::First, false);
                    self.replay_on(0, key0, || {
                        self.context_local_combine_out(0, w, layer0, rows, t, recv0)
                    })?;
                } else {
                    self.context_local_combine_out(0, w, layer0, rows, t, recv0)?;
                }
                self.publish_projection(0, index, 0, w, layer0, "o", t)?;
            } else {
                self.publish_projection(0, index, 0, w, layer0, "o", t)?;
            }

            // 3. Post-attention + FFN + FFN pushes (Rank 1 first so GPU 1 stays ahead of GPU 0).
            if layer0.dense {
                exchange.wait(1, att_slot)?;
                let key1 = Self::local_key(index, 3, t, &ranked[1], false, Previous::First, false);
                self.replay_on(1, key1, || {
                    self.projection_norm(1, index, 0, w1, layer1, "o", layer1.ptr("post_norm")?,
                        rows, w1.delta.buffer.ptr, self.recv(1, att_slot)?)?;
                    self.ffn_on(1, w1, layer1, self.cfg.dense_intermediate, "m64", w1.delta.buffer.ptr, rows)
                })?;
                self.publish_projection(1, index, 0, w1, layer1, "down", t)?;

                exchange.wait(0, att_slot)?;
                let key0 = Self::local_key(index, 3, t, tables, false, Previous::First, false);
                self.replay_on(0, key0, || {
                    self.projection_norm(0, index, 0, w, layer0, "o", layer0.ptr("post_norm")?,
                        rows, w.delta.buffer.ptr, self.recv(0, att_slot)?)?;
                    self.ffn(w, layer0, self.cfg.dense_intermediate, "m64", w.delta.buffer.ptr, rows)
                })?;
                if !last {
                    self.publish_projection(0, index, 0, w, layer0, "down", t)?;
                }
            } else if self.tp2_layer(index) && self.tp2_routes_replicated() {
                let pair = self.tp2.as_ref().context("TP2 experts")?;
                exchange.wait(1, att_slot)?;
                let key1 = Self::local_key(index, 3, t, &ranked[1], false, Previous::First, false);
                self.replay_on(1, key1, || {
                    self.projection_norm(1, index, 0, w1, layer1, "o", layer1.ptr("post_norm")?,
                        rows, w1.delta.buffer.ptr, self.recv(1, att_slot)?)?;
                    self.shared_on_local(1, index, w1, layer1, rows, "m64")?;
                    self.moe_front_on(1, index, w1, layer1, t)?;
                    self.tp2_rank_local(pair, 1, index, 0, w1, t)
                })?;
                self.tp2_publish(pair, 1, index, 0, w1, t)?;

                exchange.wait(0, att_slot)?;
                let key0 = Self::local_key(index, 3, t, tables, false, Previous::First, false);
                self.replay_on(0, key0, || {
                    self.projection_norm(0, index, 0, w, layer0, "o", layer0.ptr("post_norm")?,
                        rows, w.delta.buffer.ptr, self.recv(0, att_slot)?)?;
                    self.moe_front_on(0, index, w, layer0, t)?;
                    self.shared_on_local(0, index, w, layer0, rows, "m64")?;
                    self.tp2_rank_local(pair, 0, index, 0, w, t)
                })?;
                if !last {
                    self.tp2_publish(pair, 0, index, 0, w, t)?;
                }
            } else if self.tp2_layer(index) {
                // Broadcast-route or route-checking TP2 fallback (eager).
                exchange.wait(0, att_slot)?;
                exchange.wait(1, att_slot)?;
                self.projection_norm(0, index, 0, w, layer0, "o", layer0.ptr("post_norm")?,
                    rows, w.delta.buffer.ptr, self.recv(0, att_slot)?)?;
                self.projection_norm(1, index, 0, w1, layer1, "o", layer1.ptr("post_norm")?,
                    rows, w1.delta.buffer.ptr, self.recv(1, att_slot)?)?;
                self.moe_front(index, w, layer0, t)?;
                self.shared_on(1, index, w1, layer1, rows, "m64")?;
                self.tp2_pair(index, 0, w, w1, t, "m64")?;
            } else {
                // Spark or skip MoE: Rank 1 finishes shared FFN & pushes immediately,
                // while Rank 0 runs router + Spark exchange + reduction.
                exchange.wait(1, att_slot)?;
                let key1 = Self::local_key(index, 3, t, &ranked[1], false, Previous::First, false);
                self.replay_on(1, key1, || {
                    self.projection_norm(1, index, 0, w1, layer1, "o", layer1.ptr("post_norm")?,
                        rows, w1.delta.buffer.ptr, self.recv(1, att_slot)?)?;
                    self.shared_on_local(1, index, w1, layer1, rows, "m64")
                })?;
                exchange.push(1, ffn_slot, w1.shared.buffer.ptr, bytes)?;
                if !last {
                    if let Some(l2) = self.peer()?.l2.as_ref() {
                        self.on(1, || l2.issue(self.library, index, self.stream_of(1)))?;
                    }
                }

                exchange.wait(0, att_slot)?;
                let key0 = Self::local_key(index, 3, t, tables, false, Previous::First, false);
                self.replay_on(0, key0, || {
                    self.projection_norm(0, index, 0, w, layer0, "o", layer0.ptr("post_norm")?,
                        rows, w.delta.buffer.ptr, self.recv(0, att_slot)?)?;
                    self.moe_front(index, w, layer0, t)
                })?;
                if self.skip.is_some() {
                    let ranks = self.moe_skip(w, index, layer0, t, "m64")?;
                    self.reduce(self.skip.as_ref().context("skip intake")?.pointers(), w, ranks, t)?;
                } else {
                    let _range = crate::shared::probe::verify_timing::range(Some(self.library),
                        || format!("L{index} experts exchange {t}/{t}"));
                    let (transport, _) = experts.as_mut()
                        .with_context(|| format!("layer {index} is an MoE layer; pass Spark peers for its routed experts"))?;
                    let waves = transport.exchange()?;
                    let wave_slot = self.moe_exchange(w, index, layer0, t, "m64", true, waves)?;
                    // SAFETY: the wave landed on GPU 0's stream; shared and delta are live views.
                    unsafe {
                        waves.reduce(wave_slot, w.shared.buffer.ptr.cast(),
                            w.delta.buffer.ptr.cast(), t, self.stream)?;
                    }
                }
                if !last {
                    exchange.push(0, ffn_slot, w.delta.buffer.ptr, bytes)?;
                }
            }

            // 4. Pipeline Rank 1 one layer ahead: Rank 0 has already queued `ffn[index]`.
            let next_prev = if layer0.dense || self.tp2_layer(index) {
                Previous::Delta
            } else {
                Previous::Planes(0)
            };
            if !last {
                self.local_stage_rank1_attention(index + 1, next_prev, w1, ranked, t, rows)?;
            }
            prev_r0 = next_prev;
            crate::shared::console::layer_mark(index);
        }

        // 5. Terminal tail on Rank 0: wait for Rank 1's last FFN push, run final norm + tap + head.
        if let Some(last) = layers.len().checked_sub(1) {
            exchange.wait(0, slot(last, true, 0))?;
        }
        let tail_key = Self::local_key(layers.len(), 0, t, tables, false, prev_r0, head);
        let tail_segment = || -> Result<()> {
            if let Some(last) = layers.len().checked_sub(1) {
                if self.tp2_layer(last) {
                    self.tp2_norm(0, w, self.weights.norm.buffer.ptr, last, 0, rows)?;
                } else {
                    let last_ffn = slot(last, true, 0);
                    self.ffn_norm(0, last, 0, w, &layers[last], self.weights.norm.buffer.ptr,
                        rows, w.delta.buffer.ptr, self.recv(0, last_ffn)?)?;
                }
                if let Some(drafter) = &self.drafter {
                    drafter.tap(last, w.h.buffer.ptr, 0, t)?;
                }
            } else {
                self.norm_full(0, w, self.weights.norm.buffer.ptr, 0, rows,
                    w.delta.buffer.ptr, w.delta.buffer.ptr)?;
            }
            if head {
                self.launch_head(w, t, t)?;
                self.select_greedy(w, t)?;
            }
            Ok(())
        };
        if !head {
            tail_segment()?;
        } else {
            self.replay_on(0, tail_key, tail_segment)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::shared::peer_split::order::{self, host::{check_host, HostSchedule}};

    #[test]
    fn local_decode_captures_contain_no_peer_or_context_exchanges() {
        let source = include_str!("local_graph.rs");
        let body = source.split_once("#[cfg(test)]").unwrap().0;
        let mut count = 0usize;
        for part in body.split("self.replay_on(").skip(1) {
            let Some((_, after_arrow)) = part.split_once("|| {") else { continue };
            count += 1;
            let closure = after_arrow.split_once("})?;").unwrap().0;
            for forbidden in [
                "exchange.push", "exchange.wait", "exchange.push_to", "publish_projection",
                "tp2_publish", "tp2_pair", "push_query", "wait_query", "push_candidates",
                "wait_candidates", "push_partial", "wait_partial", "meet_shared_core",
                "core_zero_shared", "shared_on(", "moe_exchange", "moe_skip",
            ] {
                assert!(!closure.contains(forbidden),
                    "rank-local capture closure #{count} unexpectedly contains {forbidden}");
            }
        }
        assert!(count >= 10, "expected rank-local inline replay_on sites in local_graph.rs, found {count}");
        assert!(!body.contains("split_core_down"),
            "MoE local decode graph must keep each rank's shared partial in w.shared without split_core_down");
    }

    #[test]
    fn pipelined_local_decode_schedule_is_deadlock_free_at_every_capture_point() {
        use super::slot;
        use crate::shared::peer_split::DIRECT;

        #[derive(Clone, Copy)]
        enum Mode { Dense, Tp2Replicated, Spark }

        let layers = [
            (Mode::Dense, true),
            (Mode::Dense, false),
            (Mode::Tp2Replicated, true),
            (Mode::Tp2Replicated, false),
            (Mode::Spark, true),
            (Mode::Spark, false),
            (Mode::Tp2Replicated, true),
        ];

        for is_context in [false, true] {
            let mut s = HostSchedule::default();
            let capture_point = |s: &mut HostSchedule, label: &str| {
                // Worst-case cudaGraphInstantiate / cudaGraphExecUpdate may sync either GPU:
                // every queued wait on both streams must already be matched.
                s.sync(0, format!("capture {label} sync gpu0"));
                s.sync(1, format!("capture {label} sync gpu1"));
            };
            let stage_r1 = |s: &mut HostSchedule, index: usize, full_indexer: bool| {
                if index == 0 {
                    s.wait(1, "main", DIRECT, "L0 direct");
                } else {
                    s.wait(1, "main", slot(index - 1, true, 0), format!("L{index} r1 wait prev ffn"));
                }
                capture_point(s, &format!("L{index} r1 sub0"));
                if is_context {
                    let cs = index % 2;
                    s.push(1, "ctx_q", cs, format!("L{index} r1 push q"));
                    if full_indexer { s.push(1, "ctx_c", cs, format!("L{index} r1 push cand")); }
                } else {
                    s.push(1, "main", slot(index, false, 0), format!("L{index} r1 push o"));
                }
            };

            // Run two consecutive steps to prove cross-step parity slot safety as well.
            for step in 0..2 {
                s.push(0, "main", DIRECT, format!("step{step} direct"));
                stage_r1(&mut s, 0, layers[0].1);

                for (index, &(mode, full_indexer)) in layers.iter().enumerate() {
                    let last = index + 1 == layers.len();
                    let att = slot(index, false, 0);
                    let ffn = slot(index, true, 0);
                    if index > 0 {
                        s.wait(0, "main", slot(index - 1, true, 0), format!("step{step} L{index} r0 wait prev ffn"));
                    }
                    capture_point(&mut s, &format!("step{step} L{index} r0 sub0"));

                    if is_context {
                        let cs = index % 2;
                        s.push(0, "ctx_q", cs, format!("step{step} L{index} r0 push q"));
                        if full_indexer {
                            s.push(0, "ctx_c", cs, format!("step{step} L{index} r0 push cand"));
                            s.wait(1, "ctx_c", cs, format!("step{step} L{index} r1 wait cand"));
                            s.wait(1, "ctx_q", cs, format!("step{step} L{index} r1 wait q"));
                            capture_point(&mut s, &format!("step{step} L{index} r1 sub1"));
                            s.push(1, "ctx_p", cs, format!("step{step} L{index} r1 push partial"));

                            s.wait(0, "ctx_c", cs, format!("step{step} L{index} r0 wait cand"));
                            s.wait(0, "ctx_q", cs, format!("step{step} L{index} r0 wait q"));
                            capture_point(&mut s, &format!("step{step} L{index} r0 sub1"));
                            s.push(0, "ctx_p", cs, format!("step{step} L{index} r0 push partial"));
                        } else {
                            s.wait(1, "ctx_q", cs, format!("step{step} L{index} r1 wait q"));
                            s.push(1, "ctx_p", cs, format!("step{step} L{index} r1 push partial"));
                            s.wait(0, "ctx_q", cs, format!("step{step} L{index} r0 wait q"));
                            s.push(0, "ctx_p", cs, format!("step{step} L{index} r0 push partial"));
                        }
                        s.wait(1, "ctx_p", cs, format!("step{step} L{index} r1 wait partial"));
                        capture_point(&mut s, &format!("step{step} L{index} r1 sub2"));
                        s.push(1, "main", att, format!("step{step} L{index} r1 push o"));

                        s.wait(0, "ctx_p", cs, format!("step{step} L{index} r0 wait partial"));
                        capture_point(&mut s, &format!("step{step} L{index} r0 sub2"));
                        s.push(0, "main", att, format!("step{step} L{index} r0 push o"));
                    } else {
                        s.push(0, "main", att, format!("step{step} L{index} r0 push o"));
                    }

                    match mode {
                        Mode::Dense => {
                            s.wait(1, "main", att, format!("step{step} L{index} r1 wait o"));
                            capture_point(&mut s, &format!("step{step} L{index} r1 dense"));
                            s.push(1, "main", ffn, format!("step{step} L{index} r1 push dense"));

                            s.wait(0, "main", att, format!("step{step} L{index} r0 wait o"));
                            capture_point(&mut s, &format!("step{step} L{index} r0 dense"));
                            if !last {
                                s.push(0, "main", ffn, format!("step{step} L{index} r0 push dense"));
                            }
                        }
                        Mode::Tp2Replicated => {
                            s.wait(1, "main", att, format!("step{step} L{index} r1 wait o"));
                            capture_point(&mut s, &format!("step{step} L{index} r1 tp2"));
                            s.push(1, "main", ffn, format!("step{step} L{index} r1 push tp2"));

                            s.wait(0, "main", att, format!("step{step} L{index} r0 wait o"));
                            capture_point(&mut s, &format!("step{step} L{index} r0 tp2"));
                            if !last {
                                s.push(0, "main", ffn, format!("step{step} L{index} r0 push tp2"));
                            }
                        }
                        Mode::Spark => {
                            s.wait(1, "main", att, format!("step{step} L{index} r1 wait o"));
                            capture_point(&mut s, &format!("step{step} L{index} r1 spark shared"));
                            s.push(1, "main", ffn, format!("step{step} L{index} r1 push shared"));
                            s.wait(0, "main", att, format!("step{step} L{index} r0 wait o"));
                            capture_point(&mut s, &format!("step{step} L{index} r0 spark front"));
                            if !last {
                                s.push(0, "main", ffn, format!("step{step} L{index} r0 push spark"));
                            }
                        }
                    }

                    if !last {
                        stage_r1(&mut s, index + 1, layers[index + 1].1);
                    }
                }

                let last = layers.len() - 1;
                s.wait(0, "main", slot(last, true, 0), format!("step{step} tail r0 wait last ffn"));
                capture_point(&mut s, &format!("step{step} tail"));
            }

            assert_eq!(check_host(&s), Ok(()), "deadlock for context={is_context}");
            assert_eq!(order::check_submission_faults(&s.streams()), Ok(()),
                "submission fault for context={is_context}");
        }
    }
}
