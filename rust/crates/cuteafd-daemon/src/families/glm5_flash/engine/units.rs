//! Continuous prefill units on the shared executor. Decode retains its graph path.
use super::*;
use crate::shared::prefill_pipeline::{FfnRoute, PrefillHooks, Unit, run_prefill_units};

pub(super) fn logits_capacity(bytes: usize, vocabulary: usize) -> Result<usize> {
    let row = vocabulary.checked_mul(std::mem::size_of::<f32>()).filter(|row| *row > 0)
        .context("invalid logits row size")?;
    ensure!(bytes >= row && bytes % row == 0, "invalid admitted logits storage");
    Ok(bytes / row)
}

/// Normalized terminal rows, consumed in order through the admitted logits buffer.
fn head_spans(total: usize, wanted: usize, capacity: usize) -> impl Iterator<Item = std::ops::Range<usize>> {
    (total - wanted..total).step_by(capacity).map(move |start| start..(start + capacity).min(total))
}

enum Ticket { Spark(crate::shared::spark_waves::WaveTicket), Mixed(layers::Ticket) }

/// The MLA ordinal, not the absolute layer, selects Flash's two staging views.
fn staging_turn(owners: &mut [Option<(usize, usize)>; 2], ordinal: usize, layer: usize, unit: usize) -> bool {
    let owner = &mut owners[ordinal % 2];
    let reuse = unit > 0 && *owner == Some((layer, unit - 1));
    *owner = Some((layer, unit));
    reuse
}

struct UnitHooks<'e, 'a, 'w, 'd, 'f, 't, 'n> {
    engine: &'e GlmfEngine<'a>,
    /// The Spark link's transport worker (every Spark layer's waves).
    transport: Option<&'t mut crate::shared::spark_waves::Waves<'n>>,
    steps: &'w [(StepTables, &'w [u32])],
    ranked: Vec<[StepTables; 2]>,
    work: &'w [Workspace<'a>],
    peer: Option<&'w PeerWorkspaces<'w, 'a>>,
    released: Vec<Option<crate::shared::memory::device::Event<'a>>>,
    consumed: Vec<Vec<usize>>,
    waves: usize,
    total: usize,
    wanted: usize,
    device: bool,
    logits: Vec<f32>,
    result: Option<DeviceLogits>,
    media: Option<&'w cuteafd_engine::media::RequestMedia>,
    on_layer: Option<&'d mut dyn FnMut(usize, &[u8]) -> Result<()>>,
    forced: Option<&'f dyn Fn(usize) -> Option<Vec<u8>>>,
}

impl<'a> UnitHooks<'_, 'a, '_, '_, '_, '_, '_> {
    fn rows(&self, unit: &Unit) -> usize { unit.rows.len() }
    fn peer(&self, unit: &Unit) -> Result<Option<&Workspace<'a>>> {
        self.peer.map(|p| p.get(unit.slot)).transpose()
    }
    fn post(&mut self, layer: usize, unit: &Unit) -> Result<()> {
        let e = self.engine;
        let w = &self.work[unit.slot];
        let t = self.rows(unit);
        let tables = &self.steps[unit.index].0;
        let spans = tables.spans();
        // Publish rank 1's retained FFN partial only when this unit lands.
        if let Some(w1) = self.peer(unit)? {
            if e.layer_execution.is_none() && !e.tp2_layer(layer) {
                e.exchange_push(e.exchange()?, "main", 1, slot(layer, true, unit.slot),
                    e.peer_ffn_out(layer, w1, &e.peer()?.layers[layer])?, t * e.cfg.hidden * 2)?;
            }
        }
        let out = e.meet_ffn(w, layer, unit.slot, e.weights.layers.len(), t)?;
        if let Some(w1) = self.peer(unit)? { e.peer_post(layer, unit.slot, w1, t, PREFILL_CAP)?; }
        for span in &spans {
            match e.weights.layers.get(layer + 1) {
                Some(next) => e.post_pre_at(0, w, out, 1, next, "attn", "input_norm",
                    span.first, span.rows(), PREFILL_CAP)?,
                None => e.mhc_post_at(w, out, 1, span)?,
            }
        }
        if let Some(drafter) = &e.drafter {
            if !tables.segments.is_empty() {
                for s in tables.segments.iter().filter(|s| s.tap_rows > 0) {
                    drafter.tap_streams_at(layer, w.streams[0].buffer.ptr, HC,
                        s.end_row() - s.tap_rows, s.tap_rows, s.tap_offset)?;
                }
            } else {
                let window = self.total.saturating_sub(crate::shared::speculation::dflash::TAP_ROWS);
                let from = unit.rows.start.max(window);
                if from < unit.rows.end {
                    drafter.tap_streams_at(layer, w.streams[0].buffer.ptr, HC,
                        from - unit.rows.start, unit.rows.end - from, from - window)?;
                }
            }
        }
        if let Some(on_layer) = self.on_layer.as_mut() {
            on_layer(layer, &e.download(&w.streams[0], t * HC * e.cfg.hidden * 2)?)?;
        }
        if let Some(forced) = self.forced.and_then(|f| f(layer)) {
            ensure!(forced.len() == t * HC * e.cfg.hidden * 2, "teacher-forced streams of the wrong size");
            e.library.copy_h2d(cuteafd_ffi::CuteafdDeviceBuffer { bytes: forced.len(),
                ..w.streams[0].buffer }, &forced)?;
            if let Some(next) = e.weights.layers.get(layer + 1) {
                e.pre(w, &w.streams[0], next, Scalar::I32(t as i32))?;
            }
        }
        crate::shared::console::layer_mark(layer);
        e.probe_mark(layer)?;
        Ok(())
    }
}

impl PrefillHooks for UnitHooks<'_, '_, '_, '_, '_, '_, '_> {
    type Ticket = Ticket;
    fn route(&self, layer: usize) -> FfnRoute {
        if !self.engine.weights.layers[layer].dense
            && matches!(self.engine.routed_experts(layer), Some(Experts::Spark { .. })) {
            FfnRoute::Spark
        } else { FfnRoute::Local }
    }
    fn begin_unit(&mut self, unit: &Unit) -> Result<()> {
        let e = self.engine;
        let w = &self.work[unit.slot];
        e.begin_unit_route_checks(unit)?;
        if unit.index >= self.waves {
            if let Some(event) = &self.released[unit.slot] {
                // SAFETY: both streams and the per-slot event outlive this call.
                e.on(1, || unsafe { e.library.cuda_event_record(event.raw, e.stream_of(1)) })?;
                unsafe { e.library.cuda_stream_wait_event(e.stream, event.raw)?; }
            }
        }
        let (tables, tokens) = &self.steps[unit.index];
        let ranked = &self.ranked[unit.index];
        e.put_tables(w, &ranked[0])?;
        if let Some(w1) = self.peer(unit)? { e.peer_tables(w1, &ranked[1])?; }
        if e.context.is_some() {
            let start = tables.positions[0] as usize;
            let ids = placement_units(tables);
            for rank in 0..2 {
                let plan = e.context_prefill_tables(rank, unit.slot, start, start, unit.rows.len(), &ids)?;
                e.context_units.borrow_mut()[unit.index][rank] = plan;
            }
        }
        e.load_streams(w, tokens, false)?;
        e.inject_media(w, tables, self.media)?;
        if self.media.is_some_and(|m| !m.spans().is_empty()) { e.put(&w.ids, tokens)?; }
        Ok(())
    }
    fn attention(&mut self, index: usize, unit: &Unit) -> Result<()> {
        let e = self.engine;
        let w = &self.work[unit.slot];
        let layer = &e.weights.layers[index];
        let rows = Scalar::I32(unit.rows.len() as i32);
        e.context_unit.set(unit.index);
        if e.context.is_some() && layer.attention == GlmNextAttention::Mla {
            let ordinal = e.caches.mla_ordinal[index].context("MLA ordinal")?;
            e.context_reuse.set(staging_turn(&mut e.staging_owner.borrow_mut(), ordinal, index, unit.index));
        }
        if e.layer_execution.is_some() {
            let w1 = self.peer.context("owned peer slots")?.get(unit.slot)?;
            return e.unit_attention([w, w1],
                &self.ranked[unit.index], unit, index, self.total, &mut self.consumed[unit.index]);
        }
        if let Some(w1) = self.peer(unit)? {
            // Begin admits several slots before attention runs. Publish the single
            // DIRECT mailbox only here, next to its wait; mixed Entry owns its copy.
            if index == 0 {
                e.exchange_direct(w.streams[0].buffer.ptr, w1.streams[0].buffer.ptr,
                    unit.rows.len() * HC * e.cfg.hidden * 2)?;
            }
            e.peer_attention(index, unit.slot, w1, unit.rows.len(), PREFILL_CAP, &self.ranked[unit.index][1])?;
        }
        let spans = self.steps[unit.index].0.spans();
        if index == 0 {
            for span in &spans { e.pre_at(0, w, &w.streams[0], layer, span.first, span.rows())?; }
        }
        e.attention(0, w, index, layer, rows, PREFILL_CAP, &self.ranked[unit.index][0], None)?;
        let attended = e.meet_attention(w, slot(index, false, unit.slot), unit.rows.len(), layer, PREFILL_CAP)?;
        for span in &spans {
            e.post_pre_at(0, w, attended, 0, layer, "ffn", "post_norm", span.first, span.rows(), PREFILL_CAP)?;
        }
        Ok(())
    }
    fn local_ffn(&mut self, index: usize, unit: &Unit) -> Result<()> {
        let e = self.engine;
        let w = &self.work[unit.slot];
        let layer = &e.weights.layers[index];
        let rows = Scalar::I32(unit.rows.len() as i32);
        if e.layer_execution.is_some() {
            let work = [w, self.peer(unit)?.context("owned peer slot")?];
            let ticket = e.unit_ffn_begin(work, &self.ranked[unit.index], unit, index, self.total)?;
            e.unit_ffn_finish(work, &self.ranked[unit.index], unit, index, self.total, ticket)?;
            return self.post(index, unit);
        }
        if layer.dense {
            e.ffn(w, layer, e.cfg.dense_intermediate, PREFILL_CAP, w.delta.buffer.ptr, rows)?;
        } else {
            e.moe_front(w, index, layer, unit.rows.len(), rows, PREFILL_CAP,
                &self.steps[unit.index].0.spans())?;
            let w1 = self.peer(unit)?;
            if e.tp2_layer(index) {
                e.tp2_pair(index, unit.slot, w, w1, unit.rows.len(), rows, PREFILL_CAP)?;
                e.tp2_publish(index, unit.slot, unit.rows.len(), w1)?;
            } else {
                e.moe_experts(w, index, layer, unit.rows.len(), rows, PREFILL_CAP, false, w1)?;
            }
        }
        self.post(index, unit)
    }
    fn send_experts(&mut self, index: usize, unit: &Unit) -> Result<Self::Ticket> {
        let e = self.engine;
        let w = &self.work[unit.slot];
        let layer = &e.weights.layers[index];
        let rows = Scalar::I32(unit.rows.len() as i32);
        if e.layer_execution.is_some() {
            return Ok(Ticket::Mixed(e.unit_ffn_begin([w, self.peer(unit)?.context("owned peer slot")?],
                &self.ranked[unit.index], unit, index, self.total)?));
        }
        e.moe_front(w, index, layer, unit.rows.len(), rows, PREFILL_CAP,
            &self.steps[unit.index].0.spans())?;
        let transport = self.transport.as_mut().context("Spark unit without its transport worker")?;
        // The shared expert runs while the worker sends and the Sparks compute.
        let wave = e.spark_send(w, index, unit.rows.len(), false, transport,
            || e.ffn(w, layer, e.cfg.moe_intermediate, PREFILL_CAP, w.shared.buffer.ptr, rows))?;
        Ok(Ticket::Spark(wave))
    }
    fn land_experts(&mut self, layer: usize, unit: &Unit, wave: Self::Ticket) -> Result<()> {
        match wave {
            Ticket::Spark(wave) => {
                let transport = self.transport.as_mut().context("Spark unit without its transport worker")?;
                self.engine.spark_land(&self.work[unit.slot], unit.rows.len(), transport, wave)?
            }
            Ticket::Mixed(ticket) => self.engine.unit_ffn_finish(
                [&self.work[unit.slot], self.peer(unit)?.context("owned peer slot")?],
                &self.ranked[unit.index], unit, layer, self.total, ticket)?,
        }
        self.post(layer, unit)
    }
    fn end_unit(&mut self, unit: &Unit) -> Result<()> {
        let e = self.engine;
        let w = &self.work[unit.slot];
        if e.layer_execution.is_some() {
            let w1 = self.peer.context("owned peer slots")?.get(unit.slot)?;
            e.unit_exit([w, w1], &self.ranked[unit.index], unit, self.total, &mut self.consumed[unit.index])?;
        }
        if let Some(w1) = self.peer(unit)? { e.replicate_terminal(w, w1, 0, unit.rows.len())?; }
        if e.weights.layers.len() < e.cfg.layers { return Ok(()); }
        if !self.steps[unit.index].0.segments.is_empty() {
            // Packed consumers project each sequence's last normalized row later.
            e.run("head", &[("streams", w.streams[0].buffer.ptr), ("weight", e.weights.norm.buffer.ptr),
                ("out", w.x.buffer.ptr)], &[Scalar::I32(unit.rows.len() as i32)])?;
            return Ok(());
        }
        let wanted = unit.rows.end.saturating_sub((self.total - self.wanted).max(unit.rows.start));
        if wanted > 0 {
            e.normalize_head(w, unit.rows.len())?;
            if self.device {
                e.logits(w, unit.rows.len(), wanted, false)?;
                ensure!(wanted == self.wanted, "device logits must lie in the last unit");
                self.result = Some(e.device_logits(w, wanted, false));
            } else {
                // Admission determines whether the head uses bounded tiles or its full M geometry.
                let capacity = logits_capacity(w.logits.buffer.bytes, e.cfg.vocab_size)?;
                for rows in head_spans(unit.rows.len(), wanted, capacity) {
                    e.logits(w, rows.end, rows.len(), false)?;
                    self.logits.extend(e.download(&w.logits, rows.len() * e.cfg.vocab_size * 4)?
                        .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())));
                }
            }
        }
        Ok(())
    }
    fn commit_unit(&mut self, unit: &Unit) -> Result<()> { self.engine.finish_unit_route_checks(unit) }
}

impl<'a> GlmfEngine<'a> {
    pub(super) fn step_units_inner(&self, _scope: &crate::shared::experts::rtx::pair::SubmissionScope<'_>,
        steps: &[(StepTables, &[u32])], logit_rows: usize, device: bool,
        media: Option<&cuteafd_engine::media::RequestMedia>, max_waves: usize,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>,
        forced: Option<&dyn Fn(usize) -> Option<Vec<u8>>>) -> Result<Option<StepLogits>> {
        let waves = max_waves.min(self.prefill_waves).min(steps.len()).max(1);
        ensure!(on_layer.is_none() && forced.is_none() || waves == 1,
            "prefill diagnostics require executor N=1");
        ensure!(forced.is_none() || self.peer.is_none(), "teacher forcing runs without a head split");
        let work = self.prefill_slots_of(0, waves)?;
        let peer = self.peer_workspaces(false, Some(waves))?;
        for rank in 0..self.ranks() {
            // SAFETY: one call-boundary drain, never a per-dispatch stream wait.
            self.on(rank, || unsafe { self.library.cuda_stream_synchronize(self.stream_of(rank)) })?;
        }
        let total: usize = steps.iter().map(|(t, _)| t.positions.len()).sum();
        let capacity = logits_capacity(work[0].logits.buffer.bytes, self.cfg.vocab_size)?;
        ensure!(!device || logit_rows <= capacity, "device prefill logits exceed admitted rows");
        ensure!(logit_rows <= total && (self.full_prefill_logits || logit_rows <= DECODE_ROWS),
            "prefill logits exceed admitted rows");
        let mut ranked = Vec::new();
        let mut units = Vec::new();
        let mut first = 0;
        for (index, (tables, tokens)) in steps.iter().enumerate() {
            ensure!(!tokens.is_empty() && tokens.len() == tables.positions.len() && tokens.len() <= self.prefill_rows,
                "invalid prefill unit rows");
            let mut pair = if self.context.is_some() {
                [self.rank_tables(tables, 0)?, self.rank_tables(tables, 1)?]
            } else { [tables.clone(), tables.clone()] };
            for t in &mut pair { t.lane = index % waves; }
            ranked.push(pair);
            let segments = tables.segments.iter().enumerate().map(|(sequence, s)|
                crate::shared::prefill_pipeline::UnitSegment { sequence, rows: s.first..s.end_row(),
                    tokens: s.start..s.start + s.rows, prompt_end: false, snapshot: false, head: true }).collect();
            units.push(Unit { index, sequence: 0, segments, rows: first..first + tokens.len(),
                layers: 0..self.weights.layers.len(), slot: index % waves, barrier: false });
            first += tokens.len();
        }
        *self.context_units.borrow_mut() = vec![[context::ContextUnit::default(); 2]; steps.len()];
        *self.staging_owner.borrow_mut() = [None; 2];
        let released = (0..waves).map(|_| self.peer.as_ref().map(|p|
            crate::shared::memory::device::Event::new(crate::shared::memory::device::Device {
                library: self.library, id: p.device })).transpose()).collect::<Result<_>>()?;
        self.prepare_unit_route_checks(waves)?;
        let consumed = (0..steps.len()).map(|_| vec![0; self.layer_execution.as_ref()
            .map_or(0, |execution| execution.plan.hops().len())]).collect();
        self.unit_prefill.set(true);
        let run = |transport: Option<&mut crate::shared::spark_waves::Waves<'_>>| {
            let mut hooks = UnitHooks { engine: self, transport, steps, ranked, work: &work, peer: peer.as_ref(),
                released, consumed, waves, total, wanted: logit_rows, device, logits: Vec::new(), result: None,
                media, on_layer, forced };
            run_prefill_units(&mut hooks, &units, waves).map(|()| (hooks.result, hooks.logits))
        };
        // Spark waves build, post and land on the link's exchange worker
        // while this thread queues GPU work; owned-attention units exchange
        // inside their own FFN hooks.
        let outcome = match (&self.experts, &self.layer_execution) {
            (Some(Experts::Spark { transport, .. }), None) => {
                let mut link = transport.borrow_mut();
                let exchange = link.exchange()?;
                ensure!(exchange.fits(self.wave_staging_bytes()?), "Spark exchange staging is narrower than a prefill unit");
                run(Some(exchange))
            }
            _ => run(None),
        };
        self.unit_prefill.set(false);
        self.context_unit.set(0);
        self.context_reuse.set(false);
        let complete = self.finish_unit_route_span();
        let (result, logits) = match outcome {
            Ok(outcome) => outcome,
            Err(error) => return match complete {
                Ok(()) => Err(error),
                Err(unfinished) => Err(error.context(format!("{unfinished:#}"))),
            },
        };
        complete?;
        // Partial layer sets are diagnostics read on the host. A chunk without logits
        // stays queued: the next call's boundary drain (above) or the next reader
        // (selection, prefix capture) waits for it.
        if self.weights.layers.len() < self.cfg.layers { self.synchronize()?; return Ok(None); }
        if logit_rows == 0 { return Ok(None); }
        Ok(Some(if device { StepLogits::Device(result.context("last-unit logits")?) }
            else { StepLogits::Host(logits) }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::prefill_pipeline::{prefill_schedule, sequence_units, Step};
    use std::collections::VecDeque;

    #[test]
    fn head_tiles_cover_only_wanted_rows_in_order_with_bounded_tails() {
        for total in [1, 63, 64, 65, 129, 512, 4096] {
            for wanted in [0, 1, total / 2, total] {
                let spans = head_spans(total, wanted, DECODE_ROWS).collect::<Vec<_>>();
                assert!(spans.iter().all(|s| !s.is_empty() && s.len() <= DECODE_ROWS));
                assert_eq!(spans.into_iter().flatten().collect::<Vec<_>>(),
                    (total - wanted..total).collect::<Vec<_>>());
            }
        }
        assert_eq!(head_spans(4096, 129, 64).collect::<Vec<_>>(),
            vec![3967..4031, 4031..4095, 4095..4096]);
    }

    #[test]
    fn scoring_tiles_follow_admitted_logits_storage() -> Result<()> {
        use cuteafd_loader::families::glm5_flash::GlmNextAttention;
        use cuteafd_loader::serving_capacity::GlmfScratch;
        let cfg = GlmNextConfig { vocab_size: 154880, hidden: 4096, layers: 2,
            attention: vec![GlmNextAttention::Kda, GlmNextAttention::Mla], dense: vec![false; 2],
            dense_intermediate: 12288, experts: 288, topk: 8, moe_intermediate: 2048,
            routed_scale: 2.5, swiglu_limit: 10.0, rms_norm_eps: 1e-5, hc_mult: 4,
            kda_heads: 64, kda_head_dim: 128, heads: 64, q_lora_rank: 1536, kv_lora_rank: 512,
            qk_nope_dim: 256, v_head_dim: 256, index_topk: 2048, index_kpool: 4, eos: vec![] };
        for rows in [1, 63, 64, 65, 129, 4096] {
            for full in [false, true] {
                let shape = GlmfStepShape { lead: true, split: false, tp2_experts: false,
                    local_experts: true, spark: false, partial_bytes: 2, output_shard: false,
                    full_prefill_logits: full, table_pages: 32768, table_pool_pages: 8192 };
                let bytes = glmf_temporary_bytes(&cfg, rows as u64, false, &shape,
                    GlmfScratch { programs: 0, topk: 0 }).logits as usize;
                let capacity = logits_capacity(bytes, cfg.vocab_size)?;
                assert_eq!(capacity, if full { rows } else { rows.min(DECODE_ROWS) });
                let spans = head_spans(rows, rows, capacity).collect::<Vec<_>>();
                assert_eq!(spans.len() > 1, bytes < rows * cfg.vocab_size * 4);
                assert!(spans.iter().all(|span| span.len() * cfg.vocab_size * 4 <= bytes));
                assert_eq!(spans.into_iter().flatten().collect::<Vec<_>>(), (0..rows).collect::<Vec<_>>());
            }
        }
        assert!(logits_capacity(0, cfg.vocab_size).is_err());
        assert!(logits_capacity(cfg.vocab_size * 4 + 1, cfg.vocab_size).is_err());
        assert!(logits_capacity(256, 0).is_err());
        assert!(logits_capacity(256, usize::MAX).is_err());
        Ok(())
    }

    #[test]
    fn mla_staging_requires_same_layer_immediate_predecessor() {
        let mut owners = [None; 2];
        assert!(!staging_turn(&mut owners, 0, 3, 0));
        assert!(staging_turn(&mut owners, 0, 3, 1));
        // All Flash MLA layers have odd absolute parity, but adjacent MLA
        // ordinals use different views. KDA layers do not touch either view.
        assert!(!staging_turn(&mut owners, 1, 7, 0));
        assert!(staging_turn(&mut owners, 0, 3, 2));
        assert!(!staging_turn(&mut owners, 2, 11, 0));
        assert!(!staging_turn(&mut owners, 0, 3, 3));
        assert!(!staging_turn(&mut owners, 0, 3, 5));
    }

    fn routes(onboard: usize) -> Vec<FfnRoute> {
        (0..45).map(|layer| if layer < 3 + onboard { FfnRoute::Local } else { FfnRoute::Spark }).collect()
    }

    #[test]
    fn flash_units_keep_kda_and_mla_causal_across_slot_reuse() -> Result<()> {
        for waves in 1..=4 {
            let units = sequence_units(65536, 4096, waves, 45)?;
            for onboard in [0, 7, 42] {
                let routes = routes(onboard);
                let steps = prefill_schedule(&routes, &units, waves)?;
                let mut slots = vec![None; waves];
                let mut attention = vec![0; 45];
                let mut pending = VecDeque::new();
                let mut committed = 0;
                for step in steps {
                    match step {
                        Step::Begin { unit } => {
                            assert!(slots[units[unit].slot].replace(unit).is_none());
                        }
                        Step::Attention { layer, unit } => {
                            assert_eq!(slots[units[unit].slot], Some(unit));
                            assert_eq!(attention[layer], unit);
                            attention[layer] += 1;
                        }
                        Step::Send { layer, unit } => {
                            assert_eq!(routes[layer], FfnRoute::Spark);
                            pending.push_back((layer, unit));
                            assert!(pending.len() <= waves);
                        }
                        Step::Land { layer, unit } => assert_eq!(pending.pop_front(), Some((layer, unit))),
                        Step::Commit { unit } => {
                            assert_eq!(unit, committed);
                            assert_eq!(slots[units[unit].slot].take(), Some(unit));
                            committed += 1;
                        }
                        _ => {}
                    }
                }
                assert_eq!(committed, units.len());
                assert!(pending.is_empty());
                assert_eq!(attention, vec![units.len(); 45]);
            }
        }
        Ok(())
    }

    fn occupancy(steps: &[Step]) -> (f64, f64) {
        let (mut gpu, mut free, mut idle) = (0.0_f64, 0.0_f64, 0.0_f64);
        let mut pending = VecDeque::new();
        let mut sent = false;
        for step in steps {
            match step {
                Step::Attention { .. } | Step::Local { .. } => gpu += 4.0,
                Step::Send { .. } => {
                    gpu += 1.0;
                    let start = gpu.max(free);
                    if sent { idle += start - free; }
                    sent = true;
                    free = start + 10.0;
                    pending.push_back(free);
                }
                Step::Land { .. } => gpu = gpu.max(pending.pop_front().unwrap()) + 1.0,
                _ => {}
            }
        }
        (gpu, idle)
    }

    #[test]
    fn flash_two_wave_occupancy_only_pays_the_new_units_local_lead() -> Result<()> {
        let units = sequence_units(65536, 4096, 2, 45)?;
        for onboard in [0, 3, 7] {
            let routes = routes(onboard);
            let two = prefill_schedule(&routes, &units, 2)?;
            let one = prefill_schedule(&routes, &sequence_units(65536, 4096, 1, 45)?, 1)?;
            let (time, idle) = occupancy(&two);
            let (serial_time, serial_idle) = occupancy(&one);
            let lead = 4.0 * (2 * (3 + onboard) + 1) as f64 + 2.0;
            assert!(idle <= (units.len() - 1) as f64 * (lead - 10.0).max(0.0) + 1e-9);
            assert!(time < serial_time && idle < serial_idle);
        }
        Ok(())
    }
}
