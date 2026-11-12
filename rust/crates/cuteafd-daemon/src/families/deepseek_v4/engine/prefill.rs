//! Continuous V4 row units on the shared executor and one FIFO Spark link.
use super::*;
use crate::shared::prefill_pipeline::{self, FfnRoute, PrefillHooks, Unit};

impl<'a> Engine<'a> {
    pub(super) fn prefill_units(
        &self, placement: &Placement, tokens: &[u32], transports: &mut [SparkLink<'_>], logit_rows: usize,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>, download: bool,
    ) -> Result<Logits> {
        self.prefill_units_with(placement, tokens, transports, logit_rows, on_layer, download, None)
    }

    pub(super) fn prefill_units_with(
        &self, placement: &Placement, tokens: &[u32], transports: &mut [SparkLink<'_>], logit_rows: usize,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>, download: bool,
        scoring: Option<crate::shared::probe::ScoringRows<'_>>,
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
                // Waves build, post and land on the prefill link's exchange
                // worker while this thread queues GPU work.
                let transport = match links.get_mut(1) {
                    Some(link) => {
                        let exchange = link.exchange()?;
                        ensure!(exchange.fits(self.wave_staging_bytes()?), "prefill link staging is narrower than a unit");
                        Some(exchange)
                    }
                    None => None,
                };
                let (logits, landed) = self.run_units(placement, tokens, &tables, w, w1, transport, logit_rows,
                    download, on_layer, &units, waves, scoring)?;
                self.finish_route_step()?;
                // No unit cursor survives into decode/verify: all geometry is selected by hook.
                self.context_units.borrow_mut().clear();
                if logit_rows == 0 {
                    self.sync()?;
                    Ok(if download { Logits::Host(Vec::new()) } else { Logits::Device(None) })
                } else if download {
                    Ok(Logits::Host(logits))
                } else {
                    Ok(Logits::Device(Some(self.device_logits(w, landed))))
                }
            })
    }
}

impl<'a> Engine<'a> {
    #[allow(clippy::too_many_arguments)]
    fn run_units(&self, placement: &Placement, tokens: &[u32], tables: &[StepTables], w: &Workspace<'a>,
        w1: Option<&Workspace<'a>>, transport: Option<&mut crate::shared::spark_waves::Waves<'_>>,
        logit_rows: usize, download: bool, on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        units: &[Unit], waves: usize, scoring: Option<crate::shared::probe::ScoringRows<'_>>)
        -> Result<(Vec<f32>, usize)> {
        let mut hooks = UnitHooks { engine: self, placement, tokens, tables, w, w1, transport, logit_rows, download,
            on_layer, scoring, logits: Vec::new(), landed: 0,
            packed: None, context_plans: vec![Default::default(); self.prefill_waves], staging_owner: [None; 2] };
        prefill_pipeline::run_prefill_units(&mut hooks, units, waves)?;
        Ok((hooks.logits, hooks.landed))
    }
}

pub(super) struct UnitHooks<'h, 'a, 't, 'l, 'c> {
    pub(super) engine: &'h Engine<'a>,
    pub(super) placement: &'h Placement,
    pub(super) tokens: &'h [u32],
    pub(super) tables: &'h [StepTables],
    pub(super) w: &'h Workspace<'a>,
    pub(super) w1: Option<&'h Workspace<'a>>,
    pub(super) transport: Option<&'t mut crate::shared::spark_waves::Waves<'l>>,
    pub(super) logit_rows: usize,
    pub(super) download: bool,
    pub(super) scoring: Option<crate::shared::probe::ScoringRows<'h>>,
    pub(super) on_layer: Option<&'c mut dyn FnMut(usize, &[u8]) -> Result<()>>,
    pub(super) logits: Vec<f32>,
    pub(super) landed: usize,
    pub(super) context_plans: Vec<[context::ContextUnit; 2]>,
    pub(super) staging_owner: [Option<(usize, usize)>; 2],
    pub(super) packed: Option<&'h [super::packing::PackedMember<'a>]>,
}

/// Only an immediate same-layer predecessor leaves this parity's full history live.
fn reuse_context(owner: &mut [Option<(usize, usize)>; 2], ordinal: usize, layer: usize, unit: usize) -> bool {
    let view = &mut owner[ordinal % 2];
    let reuse = unit.checked_sub(1).is_some_and(|previous| *view == Some((layer, previous)));
    *view = Some((layer, unit));
    reuse
}

fn retains_draft_rows(rows: &std::ops::Range<usize>, span: usize) -> bool {
    rows.end > span.saturating_sub(TAP_ROWS)
}

impl UnitHooks<'_, '_, '_, '_, '_> {
    fn post(&mut self, layer: usize, unit: &Unit, ranks: usize, slot: Option<usize>) -> Result<()> {
        let e = self.engine;
        let rows = Scalar::I32(unit.rows.len() as i32);
        if let Some(slot) = slot {
            let lane = &self.w.slots[unit.slot];
            // SAFETY: the wave landed on the engine stream; the unit owns these workspace views.
            unsafe { self.transport.as_mut().context("Spark link")?.reduce(slot, lane.shared.buffer.ptr.cast(),
                self.w.delta.buffer.ptr.cast(), unit.rows.len(), e.stream)?; }
        }
        e.post_layer(self.w, &self.w.slots[unit.slot], unit.slot, ranks, None, rows, layer)?;
        if let Some(w1) = self.w1 {
            e.peer_post(layer, &w1.slots[unit.slot], unit.slot, w1, rows)?;
        }
        if let Some(members) = self.packed {
            for member in members { e.tap(self.w, &member.views[0], layer, member.segment.rows)?; }
        } else if retains_draft_rows(&unit.rows, self.tokens.len()) {
            e.tap(self.w, &self.w.slots[unit.slot], layer, unit.rows.len())?;
        }
        if let Some(callback) = self.on_layer.as_mut() {
            callback(layer, &e.download(&self.w.slots[unit.slot].stream_a,
                unit.rows.len() * 4 * e.cfg.dim * 2)?)?;
        }
        Ok(())
    }
}

impl PrefillHooks for UnitHooks<'_, '_, '_, '_, '_> {
    type Ticket = crate::shared::spark_waves::WaveTicket;

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
        if let Some(members) = self.packed { e.stage_packed_tables(0, &lane.tables, members)?; }
        else { e.fill(&lane.tables, tables)?; }
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
                if let Some(members) = self.packed { e.stage_packed_tables(1, &peer.tables, members)?; }
                else { e.fill(&peer.tables, tables)?; }
                e.upload(peer.tokens.buffer, bytes_of(tokens))
            })?;
            if e.context.is_some() && self.packed.is_none() {
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
        if let Some(members) = self.packed {
            return e.packed_attention(layer, members, self.w, self.w1);
        }
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
        self.post(layer, unit, ranks, None)
    }

    fn send_experts(&mut self, layer: usize, unit: &Unit) -> Result<Self::Ticket> {
        let e = self.engine;
        let transport = self.transport.as_mut().context("no prefill Spark link")?;
        let members = self.packed.map_or_else(Vec::new, |members| members.iter().map(|m|
            (m.segment.first, m.segment.rows, m.segment.start,
                m.tables.main_slots[0] as usize / (e.shape.ring_pages * metadata::SOURCE_PAGE_TOKENS))).collect());
        let wave = e.wave_send(layer, unit.rows.len(), self.w, (self.placement.len + unit.rows.start) as u64, members, transport)?;
        e.shared_ffn(layer, self.w, &self.w.slots[unit.slot], Scalar::I32(unit.rows.len() as i32),
            e.prefill_rows, &e.weights.layers[layer])?;
        Ok(wave)
    }

    fn land_experts(&mut self, layer: usize, unit: &Unit, ticket: Self::Ticket) -> Result<()> {
        let transport = self.transport.as_mut().context("no prefill Spark link")?;
        let slot = ticket.slot();
        let ranks = self.engine.wave_land(transport, ticket)?;
        self.post(layer, unit, ranks, Some(slot))
    }

    fn end_unit(&mut self, unit: &Unit) -> Result<()> {
        let e = self.engine;
        let rows = unit.rows.len();
        if self.packed.is_some() { return Ok(()); }
        let wanted = unit.rows.end.saturating_sub((self.tokens.len() - self.logit_rows).max(unit.rows.start));
        if let Some(scoring) = self.scoring {
            let capacity = self.w.vocab_logits.buffer.bytes / (e.cfg.vocab_size * 4);
            for span in crate::shared::probe::head_spans(rows, capacity.min(LOGIT_ROWS))? {
                e.head_launch(&self.w.slots[unit.slot].stream_a, span.end, span.len(), 0, self.w)?;
                scoring.record(e.library, unit.rows.start + span.start, &e.device_logits(self.w, span.len()))?;
            }
        } else if self.download {
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
        if let Some(members) = self.packed {
            for member in members {
                e.write_draft_kv(self.w, &member.views[0], member.segment.rows, e.prefill_rows)?;
            }
        } else if retains_draft_rows(&unit.rows, self.tokens.len()) {
            e.write_draft_kv(self.w, &self.w.slots[unit.slot], unit.rows.len(), e.prefill_rows)?;
        }
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
    fn scoring_consumes_bounded_heads_before_unit_commit() -> Result<()> {
        for capacity in [1, 48, LOGIT_ROWS] {
            let units = prefill_pipeline::sequence_units(8257, 2752, 3, 43)?;
            let mut positions = Vec::new();
            for unit in units {
                for span in crate::shared::probe::head_spans(unit.rows.len(), capacity)? {
                    positions.extend(span.map(|row| unit.rows.start + row));
                }
            }
            assert_eq!(positions, (0..8257).collect::<Vec<_>>());
        }
        let source = include_str!("prefill.rs");
        let units = source.split_once("    pub(super) fn prefill_units_with(").unwrap().1
            .split_once("\n    }").unwrap().0;
        assert!(units.contains("crate::shared::experts::rtx::pair::submit("));
        assert!(units.contains("sequence_units(tokens.len(), self.prefill_rows, waves,"));
        let end = source.split_once("    fn end_unit(").unwrap().1.split_once("\n    }").unwrap().0;
        let scoring = end.split_once("if let Some(scoring) = self.scoring").unwrap().1
            .split_once("} else if self.download").unwrap().0;
        assert!(scoring.contains("head_spans(rows, capacity.min(LOGIT_ROWS))"));
        assert!(scoring.find("e.head_launch(").unwrap() < scoring.find("scoring.record(").unwrap());
        assert!(!scoring.contains("self.logits.extend"));
        Ok(())
    }

    #[test]
    fn draft_ring_keeps_previous_unit_when_final_tail_is_short() -> Result<()> {
        for span in [1, 127, 128, 129, 2752, 2753, 5504, 5505, 5631, 5632, 8192, 8257] {
            let units = prefill_pipeline::sequence_units(span, 2752, 3, 43)?;
            for start in [0, 1, 127, 8192] {
                let mut reference = [usize::MAX; TAP_ROWS];
                let mut pruned = reference;
                let mut final_only = reference;
                for unit in &units {
                    for row in unit.rows.end - unit.rows.len().min(TAP_ROWS)..unit.rows.end {
                        let position = start + row;
                        reference[position % TAP_ROWS] = position;
                        if retains_draft_rows(&unit.rows, span) {
                            pruned[position % TAP_ROWS] = position;
                        }
                        if unit.rows.end == span { final_only[position % TAP_ROWS] = position; }
                    }
                }
                assert_eq!(pruned, reference, "span {span}, start {start}");
                if units.len() > 1 && units.last().unwrap().rows.len() < TAP_ROWS {
                    assert_ne!(final_only, reference, "negative control: final unit alone loses previous tail");
                    assert!(retains_draft_rows(&units[units.len() - 2].rows, span));
                }
            }
        }
        assert!(!retains_draft_rows(&(0..2752), 8192));
        assert!(!retains_draft_rows(&(2752..5504), 8192));
        assert!(retains_draft_rows(&(5504..8192), 8192));
        Ok(())
    }

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
    fn v4_fixed_three_waves_bound_spark_idle_across_slot_reuse() -> Result<()> {
        use prefill_pipeline::Step;
        for layers in [43, 61] {
            for local in [0usize, 2, 8] {
                let routes: Vec<_> = (0..layers).map(|l| if l < local { FfnRoute::Local } else { FfnRoute::Spark }).collect();
                let units = prefill_pipeline::sequence_units(32768, 2752, 3, layers)?;
                let schedule = prefill_pipeline::prefill_schedule(&routes, &units, 3)?;
                assert!(units.len() > 3 && units.iter().any(|unit| unit.slot == 2));
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
