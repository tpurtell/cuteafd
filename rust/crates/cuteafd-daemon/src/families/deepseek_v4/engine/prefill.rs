//! Continuous V4 row units on the shared executor and one FIFO Spark link.
use super::*;
use crate::shared::prefill_pipeline::{self, FfnRoute, PrefillHooks, Unit};

impl<'a> Engine<'a> {
    pub(super) fn prefill_units(
        &self, placement: &Placement, tokens: &[u32], transports: &mut [SparkLink<'_>],
        runtime: &tokio::runtime::Runtime, logit_rows: usize,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>, download: bool,
    ) -> Result<Logits> {
        let transports = RefCell::new(transports);
        crate::shared::experts::rtx::pair::submit(
            || self.require_live(), || self.shutdown_submission(&mut transports.borrow_mut()), || {
                self.begin_route_step();
                ensure!(logit_rows <= tokens.len(), "prefill logits exceed the span");
                let waves = if on_layer.is_some() { 1 } else { self.prefill_waves };
                let units = prefill_pipeline::sequence_units(tokens.len(), self.prefill_rows, waves, self.cfg.n_layers)?;
                let tables = units.iter().map(|unit| metadata::prefill_step(placement, &self.shape,
                    placement.len + unit.rows.start, unit.rows.len(), self.cfg.index_topk, self.c128_width))
                    .collect::<Result<Vec<_>>>()?;
                if self.prefill_workspace.borrow().is_none() {
                    *self.prefill_workspace.borrow_mut() = Some(self.workspace(self.prefill_rows, self.prefill_waves, true)?);
                }
                if let Some(peer) = &self.peer {
                    if peer.prefill_workspace.borrow().is_none() {
                        *peer.prefill_workspace.borrow_mut() = Some(self.workspace_on(1, self.prefill_rows, self.prefill_waves, true)?);
                    }
                }
                let workspace = self.prefill_workspace.borrow();
                let w = workspace.as_ref().context("prefill workspace")?;
                let peer_workspace = self.peer.as_ref().map(|p| p.prefill_workspace.borrow());
                let w1 = peer_workspace.as_ref().and_then(|w| w.as_ref());
                if self.context.is_some() {
                    *self.context_units.borrow_mut() = vec![Default::default(); self.prefill_waves];
                }
                // Diagnostics own rows by slot too; a slot is reused only after its unit commits.
                self.prepare_route_checks(&units.iter().map(|u| u.rows.len()).collect::<Vec<_>>())?;
                let mut links = transports.borrow_mut();
                let mut hooks = UnitHooks { engine: self, placement, tokens, tables: &tables, w, w1,
                    transport: links.get_mut(1), runtime, logit_rows, download, on_layer,
                    logits: Vec::new(), landed: 0,
                    context_plans: vec![Default::default(); self.prefill_waves], staging_owner: [None; 2] };
                prefill_pipeline::run_prefill_units(&mut hooks, &units, waves)?;
                self.finish_route_step()?;
                // No unit cursor survives into decode/verify: all geometry is selected by hook.
                self.context_units.borrow_mut().clear();
                if logit_rows == 0 {
                    self.sync()?;
                    Ok(if download { Logits::Host(Vec::new()) } else { Logits::Device(None) })
                } else if download {
                    Ok(Logits::Host(hooks.logits))
                } else {
                    Ok(Logits::Device(Some(self.device_logits(w, hooks.landed))))
                }
            })
    }
}

struct UnitHooks<'h, 'a, 't, 'l, 'c> {
    engine: &'h Engine<'a>,
    placement: &'h Placement,
    tokens: &'h [u32],
    tables: &'h [StepTables],
    w: &'h Workspace<'a>,
    w1: Option<&'h Workspace<'a>>,
    transport: Option<&'t mut SparkLink<'l>>,
    runtime: &'h tokio::runtime::Runtime,
    logit_rows: usize,
    download: bool,
    on_layer: Option<&'c mut dyn FnMut(usize, &[u8]) -> Result<()>>,
    logits: Vec<f32>,
    landed: usize,
    context_plans: Vec<[context::ContextUnit; 2]>,
    staging_owner: [Option<(usize, usize)>; 2],
}

/// Only an immediate same-layer predecessor leaves this parity's full history live.
fn reuse_context(owner: &mut [Option<(usize, usize)>; 2], ordinal: usize, layer: usize, unit: usize) -> bool {
    let view = &mut owner[ordinal % 2];
    let reuse = unit.checked_sub(1).is_some_and(|previous| *view == Some((layer, previous)));
    *view = Some((layer, unit));
    reuse
}

impl UnitHooks<'_, '_, '_, '_, '_> {
    fn post(&mut self, layer: usize, unit: &Unit, ranks: usize) -> Result<()> {
        let e = self.engine;
        let rows = Scalar::I32(unit.rows.len() as i32);
        let planes = self.transport.as_ref().map(|t| t.intake.pointers())
            .unwrap_or([std::ptr::null(); crate::shared::spark_intake::MAX_INTAKE_RANKS]);
        e.post_layer(self.w, &self.w.slots[unit.slot], unit.slot, ranks, planes, rows, layer)?;
        if let Some(w1) = self.w1 {
            e.peer_post(layer, &w1.slots[unit.slot], unit.slot, w1, rows)?;
        }
        e.tap(self.w, &self.w.slots[unit.slot], layer, unit.rows.len())?;
        if ranks != LOCAL_EXPERTS && ranks != SKIPPED_EXPERTS {
            self.transport.as_ref().context("Spark link")?.intake.consumed(e.stream)?;
        }
        if let Some(callback) = self.on_layer.as_mut() {
            callback(layer, &e.download(&self.w.slots[unit.slot].stream_a,
                unit.rows.len() * 4 * e.cfg.dim * 2)?)?;
        }
        Ok(())
    }
}

impl PrefillHooks for UnitHooks<'_, '_, '_, '_, '_> {
    type Ticket = SparkExpertWave;

    fn route(&self, layer: usize) -> FfnRoute {
        if self.engine.skip_routed || layer < self.engine.local_layers() { FfnRoute::Local }
        else { FfnRoute::Spark }
    }

    fn begin_unit(&mut self, unit: &Unit) -> Result<()> {
        let e = self.engine;
        let lane = &self.w.slots[unit.slot];
        let tables = &self.tables[unit.index];
        let tokens = &self.tokens[unit.rows.clone()];
        e.route_check_unit(unit.slot, unit.index);
        e.embedding.check(tokens)?;
        e.fill(&lane.tables, tables)?;
        e.upload(lane.tokens.buffer, bytes_of(tokens))?;
        if e.embedding.device_gather() {
            e.gather_streams(lane, tokens.len())?;
        } else {
            let embedding = e.embedding.host_rows_repeated(tokens, 4)?;
            e.upload(lane.stream_a.buffer, &embedding)?;
        }
        if let Some(w1) = self.w1 {
            let peer = &w1.slots[unit.slot];
            e.on(1, || {
                e.fill(&peer.tables, tables)?;
                e.upload(peer.tokens.buffer, bytes_of(tokens))
            })?;
            if e.context.is_some() {
                let contexts = [e.put_context_tables(0, &lane.tables, tables, unit.slot, true, &self.placement.units)?,
                    e.put_context_tables(1, &peer.tables, tables, unit.slot, true, &self.placement.units)?];
                self.context_plans[unit.slot] = contexts;
            }
            // The entry copy lands in this slot's residual, never attention scratch.
            let entry = slot(0, false, unit.slot);
            e.record_peer(0, "entry", entry, true, 0, unit.slot);
            e.exchange()?.push_to(0, entry, lane.stream_a.buffer.ptr, peer.stream_a.buffer.ptr,
                tokens.len() * 4 * e.cfg.dim * 2)?;
        }
        Ok(())
    }

    fn attention(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let e = self.engine;
        let tables = &self.tables[unit.index];
        let rows = Scalar::I32(unit.rows.len() as i32);
        if let Some(ordinal) = e.context.as_ref().and_then(|split| split.ordinal(layer)) {
            let mut plans = self.context_plans[unit.slot].clone();
            if reuse_context(&mut self.staging_owner, ordinal, layer, unit.index) {
                for plan in &mut plans { plan.shards = [0; 2]; }
            }
            e.context_units.borrow_mut()[unit.slot] = plans;
        }
        if let Some(w1) = self.w1 {
            e.attention_split(layer, &e.weights.layers[layer], tables, &self.w.slots[unit.slot],
                unit.slot, self.w, rows, e.prefill_rows)?;
            // Never pre-run the next layer on rank 1: its same-layer predecessor must publish first.
            e.peer_front(layer, tables, &w1.slots[unit.slot], unit.slot, w1, rows, e.prefill_rows)
        } else {
            e.attention(layer, &e.weights.layers[layer], tables, &self.w.slots[unit.slot], self.w, rows, e.prefill_rows)
        }
    }

    fn local_ffn(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let e = self.engine;
        let slot = &self.w.slots[unit.slot];
        let ranks = if e.skip_routed {
            e.shared_ffn(layer, self.w, slot, Scalar::I32(unit.rows.len() as i32),
                e.prefill_rows, &e.weights.layers[layer])?;
            SKIPPED_EXPERTS
        } else {
            e.local_experts(layer, unit.rows.len(), self.w, slot, e.prefill_rows, unit.slot)?;
            LOCAL_EXPERTS
        };
        // The local output is hook-local and consumed before the next unit runs.
        self.post(layer, unit, ranks)
    }

    fn send_experts(&mut self, layer: usize, unit: &Unit) -> Result<SparkExpertWave> {
        let e = self.engine;
        let transport = self.transport.as_mut().context("no prefill Spark link")?;
        let mut request = e.stage_request(layer, unit.rows.len(), self.w, ExpertV2SourceKind::Prefill, transport)?;
        for row in &mut request.rows {
            row.token_position += (self.placement.len + unit.rows.start) as u64;
        }
        let wave = transport.dispatch(&request)?;
        e.shared_ffn(layer, self.w, &self.w.slots[unit.slot], Scalar::I32(unit.rows.len() as i32),
            e.prefill_rows, &e.weights.layers[layer])?;
        Ok(wave)
    }

    fn land_experts(&mut self, layer: usize, unit: &Unit, ticket: SparkExpertWave) -> Result<()> {
        let transport = self.transport.as_mut().context("no prefill Spark link")?;
        let ranks = self.runtime.block_on(self.engine.land(transport, ticket, unit.rows.len(), self.w))?;
        self.post(layer, unit, ranks)
    }

    fn end_unit(&mut self, unit: &Unit) -> Result<()> {
        let e = self.engine;
        let rows = unit.rows.len();
        let wanted = unit.rows.end.saturating_sub((self.tokens.len() - self.logit_rows).max(unit.rows.start));
        if self.download {
            let mut done = 0;
            while done < wanted {
                let chunk = (wanted - done).min(LOGIT_ROWS);
                e.head_launch(&self.w.slots[unit.slot].stream_a, rows - wanted + done + chunk,
                    chunk, 0, self.w)?;
                let bytes = e.download(&self.w.vocab_logits, chunk * e.cfg.vocab_size * 4)?;
                self.logits.extend(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())));
                done += chunk;
            }
        } else if wanted > 0 {
            let capacity = self.w.vocab_logits.buffer.bytes / (e.cfg.vocab_size * 4);
            ensure!(self.landed + wanted <= capacity, "prefill device logits exceed {capacity} rows");
            e.head_launch(&self.w.slots[unit.slot].stream_a, rows, wanted, self.landed, self.w)?;
            self.landed += wanted;
        }
        Ok(())
    }

    fn commit_unit(&mut self, unit: &Unit) -> Result<()> {
        let e = self.engine;
        e.write_draft_kv(self.w, &self.w.slots[unit.slot], unit.rows.len(), e.prefill_rows)?;
        if self.w1.is_some() {
            // Rank 0 must join the peer's last reader before reusing this slot's entry residual.
            e.record_peer(1, "unit-commit", DIRECT, true, e.cfg.n_layers, unit.slot);
            e.exchange()?.signal(1, DIRECT)?;
            e.record_peer(0, "unit-commit", DIRECT, false, e.cfg.n_layers, unit.slot);
            e.exchange()?.wait(0, DIRECT)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c4_staging_reuse_requires_the_same_layers_immediate_predecessor() {
        let mut owner = [None; 2];
        assert!(!reuse_context(&mut owner, 0, 2, 0));
        assert!(reuse_context(&mut owner, 0, 2, 1));
        assert!(!reuse_context(&mut owner, 2, 6, 1));
        assert!(!reuse_context(&mut owner, 0, 2, 2));
        assert!(reuse_context(&mut owner, 0, 2, 3));
        assert!(!reuse_context(&mut owner, 1, 4, 3));
        assert!(reuse_context(&mut owner, 0, 2, 4));
        assert!(!reuse_context(&mut owner, 0, 2, 6));
        assert!(!reuse_context(&mut owner, 0, 2, 0));
    }

    #[test]
    fn c4_staging_matches_cold_history_across_ramps_and_slot_reuse() -> Result<()> {
        use prefill_pipeline::Step;
        for layers in [43, 61] {
            for local in [0, 2, 8, layers] {
                let routes: Vec<_> = (0..layers).map(|layer| if layer < local {
                    FfnRoute::Local
                } else { FfnRoute::Spark }).collect();
                for waves in [1, 2, 3] {
                    let units = prefill_pipeline::sequence_units(32769, 4096, waves, layers)?;
                    for start in [0, 257, 4097] {
                        let groups = (start + 32769) / 4;
                        let value = |layer, group| (layer + 1) * (groups + 1) + group;
                        let mut pools: Vec<Vec<_>> = (0..layers).map(|layer|
                            (0..start / 4).map(|group| value(layer, group)).collect()).collect();
                        let mut staging = [Vec::new(), Vec::new()];
                        let mut owner = [None; 2];
                        let mut reused = 0;
                        let mut gathered = 0;
                        for step in prefill_pipeline::prefill_schedule(&routes, &units, waves)? {
                            let Step::Attention { layer, unit } = step else { continue };
                            // C4 ordinals, not physical layer parity: C128 layers never touch staging.
                            if layer < 2 || layer % 2 != 0 { continue; }
                            let ordinal = (layer - 2) / 2;
                            let view = &mut staging[ordinal % 2];
                            let first = (start + units[unit].rows.start) / 4;
                            assert_eq!(pools[layer].len(), first);
                            if reuse_context(&mut owner, ordinal, layer, unit) {
                                reused += 1;
                            } else {
                                *view = pools[layer].clone();
                                gathered += 1;
                            }
                            assert_eq!(&view[..first], pools[layer].as_slice(),
                                "{layers} layers/{local} local/{waves} waves/start {start}/unit {unit}/layer {layer}");
                            let end = (start + units[unit].rows.end) / 4;
                            view.resize(end, 0);
                            for group in first..end { view[group] = value(layer, group); }
                            pools[layer].extend_from_slice(&view[first..end]);
                            assert_eq!(*view, pools[layer]);
                        }
                        assert!(gathered > 0);
                        if waves > 1 && local == 0 { assert!(reused > 0); }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn v4_two_waves_bound_spark_idle_across_slot_reuse() -> Result<()> {
        use prefill_pipeline::Step;
        for layers in [43, 61] {
            for local in [0usize, 2, 8] {
                let routes: Vec<_> = (0..layers).map(|l| if l < local { FfnRoute::Local } else { FfnRoute::Spark }).collect();
                let units = prefill_pipeline::sequence_units(32768, 4096, 2, layers)?;
                let schedule = prefill_pipeline::prefill_schedule(&routes, &units, 2)?;
                let (mut gpu, mut spark, mut idle) = (0.0f64, 0.0f64, 0.0f64);
                let mut finishes = std::collections::VecDeque::new();
                let mut sent = 0;
                for step in schedule {
                    match step {
                        Step::Attention { .. } | Step::Local { .. } => gpu += 4.0,
                        Step::Send { .. } => {
                            gpu += 1.0;
                            let start = gpu.max(spark);
                            if sent > 0 { idle += start - spark; }
                            spark = start + 10.0;
                            finishes.push_back(spark);
                            sent += 1;
                        }
                        Step::Land { .. } => gpu = gpu.max(finishes.pop_front().unwrap()) + 1.0,
                        _ => {}
                    }
                }
                let lead = 4.0 * (2 * local + 1) as f64 + 2.0;
                let bound = (units.len() - 1) as f64 * (lead - 10.0).max(0.0);
                assert!(idle <= bound + 1e-9, "{layers} layers/{local} local: idle {idle} > {bound}");
                assert_eq!(sent, units.len() * (layers - local));
                assert!(finishes.is_empty());
            }
        }
        Ok(())
    }

    #[test]
    fn v4_routes_cover_local_prefix_and_fifo_spark_suffix() -> Result<()> {
        for layers in [43, 61] {
            for local in [0, 2, layers] {
                let routes: Vec<_> = (0..layers).map(|l| if l < local { FfnRoute::Local } else { FfnRoute::Spark }).collect();
                for waves in [1, 2, 3] {
                    let units = prefill_pipeline::sequence_units(24576, 4096, waves, layers)?;
                    let schedule = prefill_pipeline::prefill_schedule(&routes, &units, waves)?;
                    assert_eq!(schedule.iter().filter(|s| matches!(s, prefill_pipeline::Step::Commit { .. })).count(), units.len());
                    assert!(units.len() > waves);
                }
            }
        }
        Ok(())
    }
}
