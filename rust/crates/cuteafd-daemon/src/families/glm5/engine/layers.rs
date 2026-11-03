//! Whole-attention GLM bindings for the shared, owner-aware layer pipeline.
use super::*;
use crate::shared::experts::rtx::pair::SubmissionScope;
use crate::shared::peer_split::{attention::{self, AttentionStateOwners, LaneOrder, LayerKernels, PipelineKernels}, hop::HopLink};
use cuteafd_loader::serving_capacity::glm_graphs::{GlmOwnedPhase, GlmOwnedSegment, glm_owned_segments};
use cuteafd_loader::placement::{FfnMode, Hop, HopPoint, LayerExecutionPlan, LayerMode, Placement};

pub(super) struct Execution<'a> {
    plan: LayerExecutionPlan,
    hops: HopLink<'a>,
    // Keep dispatched receivers owned across submission errors until engine drainage.
    waves: RefCell<Vec<Option<PendingWave>>>,
    pub(super) segments: Vec<GlmOwnedSegment>,
}

enum Transport<'t, 'l> {
    Serial(Option<(&'t mut SparkLink<'l>, &'t tokio::runtime::Runtime)>),
    Continuous(&'t mut crate::shared::spark_waves::Waves<'l>),
    Local,
}

struct Bindings<'e, 'a, 't, 'l, 'c> {
    engine: &'e GlmEngine<'a>,
    work: [&'e Workspace<'a>; 2],
    tables: &'e [StepTables; 2],
    lane: usize,
    transport: Transport<'t, 'l>,
    owners: Vec<Vec<u8>>,
    on_layer: Option<&'c mut dyn FnMut(usize, &[u8]) -> Result<()>>,
    tap: (usize, usize, usize),
}

enum PendingWave {
    Landed(usize),
    Dispatched(crate::shared::spark_waves::WaveTicket),
}

struct UnitExchange<'e, 'a> {
    engine: &'e GlmEngine<'a>,
    execution: &'e Execution<'a>,
}

impl attention::LayerProtocol for UnitExchange<'_, '_> {
    type Ticket = crate::shared::peer_split::hop::HopTicket;
    fn require_live(&self) -> Result<()> { self.engine.exchange()?.require_live() }
    fn push_entry(&self, from: usize, source: *const c_void, destination: *mut c_void, bytes: usize) -> Result<()> {
        self.engine.exchange()?.push_to(from, DIRECT, source, destination, bytes)
    }
    fn wait_entry(&self, to: usize) -> Result<()> { self.engine.exchange()?.wait(to, DIRECT) }
    fn send(&self, index: usize, lane: usize, source: *const c_void, rows: usize) -> Result<Self::Ticket> {
        self.execution.hops.send(index, lane, source, rows)
    }
    fn land(&self, ticket: Self::Ticket) -> Result<*mut c_void> { self.execution.hops.land(ticket) }
}

#[derive(Clone, Copy)]
pub(super) enum Ticket {
    Complete,
    Skip(usize),
    Spark,
    Lane,
}

fn validate_owned_kv(cfg: &GlmDsaConfig, index: usize, pages: usize, owned: bool, bytes: usize) -> Result<()> {
    ensure!(bytes == if owned { (pages + 1) * PAGE_ROWS * cfg.record_bytes(index) } else { 256 },
        "GLM layer {index} KV ownership differs from placement");
    Ok(())
}

impl<'a> GlmEngine<'a> {
    pub(crate) fn layers_kv(&self) -> bool {
        self.weights.layers.first().is_some_and(|layer| layer.attention_owner.is_some())
    }

    pub(crate) fn bind_layer_execution(&mut self, placement: &Placement) -> Result<()> {
        ensure!(self.layers_kv() && !self.context_kv() && self.layer_execution.is_none(), "whole attention binding");
        let spec = cuteafd_loader::placement::HopSpec { storage: cuteafd_loader::placement::HopStorage::ReceiveSlots, row_bytes: self.cfg.hidden as u64 * 2,
            rows: self.prefill_rows.max(DECODE_ROWS) as u64, lanes: self.prefill_lanes as u64,
            entry_gpu: 0, head_gpu: 0, terminal: cuteafd_loader::placement::TerminalHome::Replicated };
        cuteafd_loader::placement::validate_layer_pipeline_lanes(spec.lanes, self.prefill_lanes)?;
        let plan = LayerExecutionPlan::new(placement, spec)?;
        for (index, &mode) in plan.modes().iter().enumerate() {
            let LayerMode::Whole { gpu, ffn: FfnMode::Split } = mode else { anyhow::bail!("GLM layers needs whole attention and split FFNs"); };
            for rank in 0..2 {
                let layer = if rank == 0 { &self.weights.layers[index] } else { &self.peer()?.layers[index] };
                ensure!(layer.split && !layer.attention_split && layer.attention_owner == Some(gpu)
                    && layer.holds_attention() == (gpu as usize == rank), "GLM layer {index} owner differs from loaded weights");
                let (kv, index_cache, _) = self.rank_caches_of(rank, index);
                let owned = rank == gpu as usize && self.cfg.owns_kv(index);
                validate_owned_kv(&self.cfg, index, self.pages, owned, kv.buffer.bytes)?;
                ensure!(index_cache.is_some() == (rank == gpu as usize && layer.full_indexer), "GLM index ownership");
            }
            if self.cfg.index_source(index) < plan.modes().len() {
                ensure!(plan.mode(self.cfg.index_source(index))? == mode, "GLM shared indexer owner mismatch");
            }
        }
        let exchange = self.exchange()?;
        let mut hops = HopLink::new(self.library, [0, 1].map(|rank| RankDevice {
            device: exchange.device(rank), stream: exchange.stream(rank) }), plan.hops(), spec, DECODE_ROWS * self.cfg.hidden * 2)?;
        hops.attach_abort([exchange.abort_word(0)?.context("rank 0 abort")?, exchange.abort_word(1)?.context("rank 1 abort")?])?;
        hops.validate_binding(&plan, exchange)?;
        let owners: Vec<_> = self.weights.layers.iter().map(|layer| layer.attention_owner.unwrap()).collect();
        let tp2: Vec<_> = placement.layers.iter().map(|layer|
            layer.experts == cuteafd_loader::placement::ExpertHome::RtxTp2).collect();
        let segments = glm_owned_segments(&owners, &tp2).map_err(anyhow::Error::msg)?;
        self.layer_execution = Some(Execution { plan, hops, segments,
            waves: RefCell::new((0..self.prefill_lanes).map(|_| None).collect()) });
        Ok(())
    }

    fn loaded_attention_owners(&self) -> Vec<Vec<u8>> {
        (0..self.weights.layers.len()).map(|index| (0..2).filter(|&rank| {
            let layer = if rank == 0 { &self.weights.layers[index] }
                else { &self.peer.as_ref().unwrap().layers[index] };
            layer.holds_attention() && self.rank_caches(rank, index).0.buffer.bytes > 256
        }).map(|rank| rank as u8).collect()).collect()
    }

    fn run_layer_pipeline(&self, scope: &SubmissionScope<'_>, order: LaneOrder,
        bindings: &mut [Bindings<'_, 'a, '_, '_, '_>]) -> Result<()> {
        let execution = self.layer_execution.as_ref().context("GLM layer execution")?;
        let rows = bindings.iter().map(|binding| binding.tables[0].positions.len()).collect::<Vec<_>>();
        attention::layer_pipeline(scope, &execution.plan, self.exchange()?, &execution.hops, order, &rows, bindings)
    }

    pub(super) fn layer_step(&self, scope: &SubmissionScope<'_>, w: &Workspace<'a>, w1: &Workspace<'a>,
        tables: &[StepTables; 2], rows: usize, wanted: usize,
        experts: &mut Option<(&mut SparkLink<'_>, &tokio::runtime::Runtime)>,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>) -> Result<Option<DeviceLogits>> {
        let owners = self.loaded_attention_owners();
        let mut bindings = [Bindings { engine: self, work: [w, w1], tables, lane: 0,
            transport: Transport::Serial(experts.as_mut().map(|(link, runtime)| (&mut **link, *runtime))),
            owners, on_layer, tap: (rows.saturating_sub(crate::shared::speculation::dflash::TAP_ROWS), rows.min(crate::shared::speculation::dflash::TAP_ROWS), 0) }];
        self.run_layer_pipeline(scope, LaneOrder::Independent, &mut bindings)?;
        if let Some(pair) = &self.tp2 { pair.finish_checks()?; }
        if self.weights.layers.len() < self.cfg.layers { return Ok(None); }
        if wanted == 0 { return Ok(None); }
        self.launch_head(w, rows, wanted)?;
        if tables[0].decode && rows == wanted { self.select_greedy(w, rows)?; }
        Ok(Some(self.device_logits(w, wanted, tables[0].decode && rows == wanted)))
    }

    pub(super) fn validate_owned_units(&self) -> Result<usize> {
        let execution = self.layer_execution.as_ref().context("GLM layer execution")?;
        execution.hops.validate_binding(&execution.plan, self.exchange()?)?;
        for (layer, owners) in self.loaded_attention_owners().iter().enumerate() {
            execution.plan.validate_state_owners(layer, owners)?;
        }
        Ok(execution.plan.hops().len())
    }

    fn unit_binding<'s, 't, 'l>(&'s self, work: [&'s Workspace<'a>; 2], tables: &'s [StepTables; 2],
        unit: &crate::shared::prefill_pipeline::Unit, total: usize, transport: Transport<'t, 'l>)
        -> Bindings<'s, 'a, 't, 'l, 's> {
        let window = total.saturating_sub(crate::shared::speculation::dflash::TAP_ROWS);
        let from = unit.rows.start.max(window).min(unit.rows.end);
        Bindings { engine: self, work, tables, lane: unit.slot, transport,
            owners: self.loaded_attention_owners(), on_layer: None,
            tap: (from - unit.rows.start, unit.rows.end - from, from.saturating_sub(window)) }
    }

    pub(super) fn owned_unit_attention(&self, work: [&Workspace<'a>; 2], tables: &[StepTables; 2],
        unit: &crate::shared::prefill_pipeline::Unit, layer: usize, total: usize, consumed: &mut [usize]) -> Result<()> {
        let execution = self.layer_execution.as_ref().context("GLM layer execution")?;
        let exchange = UnitExchange { engine: self, execution };
        let mut binding = self.unit_binding(work, tables, unit, total, Transport::Local);
        unit_attention(&exchange, &execution.plan, unit, layer, &mut binding, consumed)
    }

    pub(super) fn owned_unit_begin(&self, work: [&Workspace<'a>; 2], tables: &[StepTables; 2],
        unit: &crate::shared::prefill_pipeline::Unit, layer: usize, total: usize,
        waves: &mut crate::shared::spark_waves::Waves<'_>) -> Result<Ticket> {
        self.unit_binding(work, tables, unit, total, Transport::Continuous(waves)).begin_ffn(layer, None)
    }

    pub(super) fn owned_unit_finish(&self, work: [&Workspace<'a>; 2], tables: &[StepTables; 2],
        unit: &crate::shared::prefill_pipeline::Unit, layer: usize, total: usize,
        waves: &mut crate::shared::spark_waves::Waves<'_>, ticket: Ticket) -> Result<()> {
        self.unit_binding(work, tables, unit, total, Transport::Continuous(waves)).finish_ffn(layer, ticket)
    }

    pub(super) fn owned_unit_exit(&self, work: [&Workspace<'a>; 2], tables: &[StepTables; 2],
        unit: &crate::shared::prefill_pipeline::Unit, total: usize, consumed: &mut [usize]) -> Result<()> {
        let execution = self.layer_execution.as_ref().context("GLM layer execution")?;
        let exchange = UnitExchange { engine: self, execution };
        let mut binding = self.unit_binding(work, tables, unit, total, Transport::Local);
        unit_exit(&exchange, &execution.plan, unit, &mut binding, consumed)
    }
}

fn unit_attention(exchange: &impl attention::LayerProtocol, plan: &LayerExecutionPlan,
    unit: &crate::shared::prefill_pipeline::Unit, layer: usize, kernels: &mut impl LayerKernels,
    consumed: &mut [usize]) -> Result<()> {
    attention::pipeline_attention(exchange, plan, layer, unit.slot, unit.rows.len(), kernels, consumed)?;
    Ok(())
}

fn unit_exit(exchange: &impl attention::LayerProtocol, plan: &LayerExecutionPlan,
    unit: &crate::shared::prefill_pipeline::Unit, kernels: &mut impl LayerKernels,
    consumed: &mut [usize]) -> Result<()> {
    attention::pipeline_hops(exchange, plan, HopPoint::Exit, unit.slot, unit.rows.len(), kernels, consumed)?;
    ensure!(consumed.len() == plan.hops().len() && consumed.iter().all(|&count| count == 1),
        "planned unit hop missing or duplicated");
    Ok(())
}

impl Bindings<'_, '_, '_, '_, '_> {
    fn rows(&self) -> usize { self.tables[0].positions.len() }
    fn cap(&self) -> &'static str { if self.tables[0].decode { "m64" } else { "m4096" } }
    fn layer(&self, rank: usize, layer: usize) -> &GlmLayer<'_> {
        if rank == 0 { &self.engine.weights.layers[layer] } else { &self.engine.peer.as_ref().unwrap().layers[layer] }
    }
    fn segment(&self, rank: usize, layer: usize, phase: GlmOwnedPhase, body: impl Fn() -> Result<()>) -> Result<()> {
        if self.tables[0].decode && self.on_layer.is_none() {
            if self.engine.graphs_warming.get() {
                ensure!(self.engine.layer_execution.as_ref().context("GLM layer execution")?.segments
                    .contains(&GlmOwnedSegment { rank, layer, phase }), "capture outside GLM GraphSet");
            }
            let table = &self.tables[rank];
            self.engine.replay_on(rank, GraphKey { layer: phase.key(layer), rows: self.rows(),
                table_width: table.table_width, table_stride: table.table_stride, previous: Previous::First, head: false }, body)
        } else { body() }
    }
    fn join(&self, index: usize) -> Result<()> {
        let e = self.engine;
        for rank in 0..2 {
            let w = self.work[rank];
            self.segment(rank, index, GlmOwnedPhase::FfnJoin, || {
                e.exchange()?.wait(rank, slot(index, true, self.lane))?;
                let weight = if let Some(next) = e.weights.layers.get(index + 1) { next.ptr("input_norm")? }
                    else { e.weights.norm.buffer.ptr };
                // The terminal norm has only rank-0 weights; the residual itself
                // is replicated by the same rank-ordered BF16 addition on both GPUs.
                let weight = if rank == 1 { if let Some(next) = e.peer()?.layers.get(index + 1) { next.ptr("input_norm")? }
                    else { self.layer(rank, index).ptr("input_norm")? } } else { weight };
                if e.tp2_layer(index) { e.tp2_norm(rank, w, weight, index, self.lane, Scalar::I32(self.rows() as i32)) }
                else {
                    let local = if rank == 0 || self.layer(rank, index).dense { w.delta.buffer.ptr } else { w.shared.buffer.ptr };
                    let peer = e.recv(rank, slot(index, true, self.lane))?;
                    let (first, second) = if rank == 0 { (local, peer) } else { (peer, local) };
                    e.ffn_norm(rank, index, self.lane, w, self.layer(rank, index), weight,
                        Scalar::I32(self.rows() as i32), first, second)
                }
            })?;
        }
        if let Some(drafter) = &e.drafter {
            if self.tables[0].segments.is_empty() {
                let (first, rows, offset) = self.tap;
                if rows > 0 { drafter.tap_at(index, self.work[0].h.buffer.ptr, first, rows, offset)?; }
            } else {
                for (s, _) in &self.tables[0].segments {
                    drafter.tap_at(index, self.work[0].h.buffer.ptr, s.end_row() - s.tap_rows, s.tap_rows, s.tap_offset)?;
                }
            }
        }
        Ok(())
    }
}

impl LayerKernels for Bindings<'_, '_, '_, '_, '_> {
    fn pack(&mut self, hop: Hop) -> Result<*const c_void> { Ok(self.work[hop.from as usize].h.buffer.ptr) }
    fn entry_destination(&mut self, rank: usize) -> Result<*mut c_void> { Ok(self.work[rank].h.buffer.ptr) }
    fn adopt(&mut self, hop: Hop, residual: *mut c_void) -> Result<()> {
        let rank = hop.to as usize;
        let w = self.work[rank];
        let e = self.engine;
        if hop.at == HopPoint::Entry { return Ok(()); }
        let layer = match hop.at { HopPoint::AfterAttention(layer) => Some(layer), HopPoint::BeforeLayer(0) => None,
            _ => anyhow::bail!("unexpected whole-attention hop") };
        let bytes = self.rows() * e.cfg.hidden * 2;
        e.on(rank, || unsafe {
            // SAFETY: HopLink landed these admitted rows before this stream-ordered copy.
            e.library.copy_d2d_async(w.h.buffer, cuteafd_ffi::CuteafdDeviceBuffer { ptr: residual, bytes,
                device_id: e.exchange()?.device(rank), flags: 0 }, bytes, e.stream_of(rank))
        })?;
        if let Some(layer) = layer {
            self.segment(rank, layer, GlmOwnedPhase::RemoteNorm, || e.norm_full(rank, w, self.layer(rank, layer).ptr("post_norm")?,
                0, Scalar::I32(self.rows() as i32), w.delta.buffer.ptr, w.delta.buffer.ptr))?;
        }
        Ok(())
    }
    fn heads(&mut self, _layer: usize) -> Result<()> { anyhow::bail!("whole attention binding cannot execute heads") }
    fn context(&mut self, _layer: usize, _ordinal: usize) -> Result<()> { anyhow::bail!("whole attention binding cannot execute context") }
    fn owned(&mut self, index: usize, rank: usize) -> Result<()> {
        let e = self.engine;
        let w = self.work[rank];
        self.segment(rank, index, GlmOwnedPhase::Attention, || {
            let rows = Scalar::I32(self.rows() as i32);
            if index == 0 { e.norm_full(rank, w, self.layer(rank, index).ptr("input_norm")?, 0, rows,
                w.delta.buffer.ptr, w.delta.buffer.ptr)?; }
            e.attention_unit(rank, w, index, self.layer(rank, index), rows, self.cap(), self.tables, self.lane)?;
            e.norm_full(rank, w, self.layer(rank, index).ptr("post_norm")?, 1, rows,
                w.delta.buffer.ptr, w.delta.buffer.ptr)
        })
    }
    fn ffn(&mut self, index: usize, owner: Option<usize>) -> Result<()> {
        let ticket = self.begin_ffn(index, owner)?;
        self.finish_ffn(index, ticket)
    }
}

impl PipelineKernels for Bindings<'_, '_, '_, '_, '_> {
    type FfnTicket = Ticket;
    fn state_owners(&self, index: usize) -> Result<AttentionStateOwners<'_>> {
        let owners = &self.owners[index];
        Ok(AttentionStateOwners { weights: owners, kv: owners,
            indexer: self.engine.weights.layers[index].full_indexer.then_some(owners.as_slice()), compressor: None })
    }
    fn begin_ffn(&mut self, index: usize, owner: Option<usize>) -> Result<Ticket> {
        ensure!(owner.is_none(), "GLM FFNs are split");
        let e = self.engine;
        let [w, w1] = self.work;
        let layer = &e.weights.layers[index];
        let rows = Scalar::I32(self.rows() as i32);
        if !layer.dense && e.tp2_layer(index) {
            e.moe_front(w, layer, self.rows())?;
            e.ring_local_routes(index, w, self.rows())?;
            e.shared_on(0, index, w, layer, rows, self.cap())?;
            e.moe_front_on(1, w1, self.layer(1, index), self.rows())?;
            e.shared_on(1, index, w1, self.layer(1, index), rows, self.cap())?;
            let pair = e.tp2.as_ref().context("TP2")?;
            e.tp2_rank(pair, 0, index, self.lane, w, self.rows())?;
            e.tp2_rank(pair, 1, index, self.lane, w1, self.rows())?;
            return Ok(Ticket::Complete);
        }
        self.segment(1, index, GlmOwnedPhase::FfnFront, || {
            e.ffn_on(1, w1, self.layer(1, index),
                if layer.dense { e.cfg.dense_intermediate } else { e.cfg.moe_intermediate }, self.cap(),
                if layer.dense { w1.delta.buffer.ptr } else { w1.shared.buffer.ptr }, rows)?;
            if !layer.dense { e.meet_shared_core(1, index, w1, self.layer(1, index), self.rows())?; }
            Ok(())
        })?;
        if self.tables[0].decode && !layer.dense {
            if let Some(l2) = &e.peer()?.l2 {
                e.on(1, || l2.issue(e.library, index, e.stream_of(1)))?;
            }
        }
        if layer.dense {
            self.segment(0, index, GlmOwnedPhase::FfnFront, || e.ffn(w, layer, e.cfg.dense_intermediate, self.cap(), w.delta.buffer.ptr, rows))?;
            return Ok(Ticket::Complete);
        }
        self.segment(0, index, GlmOwnedPhase::FfnFront, || e.moe_front(w, layer, self.rows()))?;
        let real = self.tables[0].exchange_rows;
        if e.skip.is_some() { return Ok(Ticket::Skip(e.moe_skip(w, index, layer, real, self.cap())?)); }
        let decode = self.tables[0].decode;
        let execution = e.layer_execution.as_ref().context("layer execution")?;
        ensure!(execution.waves.borrow()[self.lane].is_none(), "previous Spark wave is still live");
        // Routes and wire rows go to the link's exchange; the shared expert
        // and the prefetch queue behind them; the planes' wait is queued in
        // `finish_ffn`. The host never waits for the GPU or the Sparks.
        let cap = self.cap();
        let (pending, ticket) = match &mut self.transport {
            Transport::Serial(Some((link, _))) => (PendingWave::Landed(e.moe_exchange(w, index, layer, real, cap, decode,
                link.exchange()?)?), Ticket::Spark),
            Transport::Continuous(waves) => (PendingWave::Dispatched(e.moe_wave(w, index, layer, real, cap, waves)?), Ticket::Lane),
            _ => anyhow::bail!("GLM MoE layer {index} needs Spark peers"),
        };
        execution.waves.borrow_mut()[self.lane] = Some(pending);
        Ok(ticket)
    }
    fn finish_ffn(&mut self, index: usize, ticket: Ticket) -> Result<()> {
        let e = self.engine;
        let w = self.work[0];
        let real = self.tables[0].exchange_rows;
        let planes = match ticket {
            Ticket::Skip(ranks) => Some((e.skip.as_ref().context("skip")?.pointers(), ranks)),
            Ticket::Spark | Ticket::Lane => {
                let pending = e.layer_execution.as_ref().context("layer execution")?.waves.borrow_mut()[self.lane]
                    .take().context("missing dispatched Spark wave")?;
                let mut reduce = |waves: &mut crate::shared::spark_waves::Waves<'_>, slot| -> Result<()> {
                    e.reduce(waves.planes(slot), w, waves.ranks(), real)?;
                    waves.consumed(slot, e.stream)
                };
                match (&mut self.transport, ticket, pending) {
                    (Transport::Serial(Some((link, _))), Ticket::Spark, PendingWave::Landed(slot)) => reduce(link.exchange()?, slot)?,
                    (Transport::Continuous(waves), Ticket::Lane, PendingWave::Dispatched(wave)) => {
                        let slot = wave.slot();
                        waves.land(wave, e.stream)?;
                        reduce(waves, slot)?;
                    }
                    _ => anyhow::bail!("Spark completion transport changed"),
                }
                None
            }
            Ticket::Complete => None,
        };
        if let Some((planes, ranks)) = planes { e.reduce(planes, w, ranks, real)?; }
        if !self.layer(0, index).dense { e.tap_width(w, index, real)?; }
        if !e.tp2_layer(index) {
            let bytes = self.rows() * e.cfg.hidden * 2;
            for rank in 0..2 {
                let w = self.work[rank];
                let layer = self.layer(rank, index);
                if layer.dense { e.publish_projection(rank, index, self.lane, w, layer, "down", self.rows())?; }
                else {
                    let out = if rank == 0 { w.delta.buffer.ptr } else { w.shared.buffer.ptr };
                    e.exchange()?.push(rank, slot(index, true, self.lane), out, bytes)?;
                }
            }
        }
        self.join(index)?;
        if !self.layer(0, index).dense { e.finish_width_tap(w, index, real)?; }
        let bytes = self.rows() * e.cfg.hidden * 2;
        if let Some(callback) = self.on_layer.as_mut() { callback(index, &e.download(&w.h, bytes)?)?; }
        if let Some(probe) = e.draft_probe.borrow_mut().as_mut() {
            // SAFETY: this submission retains the rank-0 stream and probe events.
            unsafe { probe.mark(index, e.stream)?; }
        }
        crate::shared::console::layer_mark(index);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{experts::rtx::pair, peer_split::{attention::LayerProtocol,
        order::{self, host::{check_host, HostSchedule}}}};
    use cuteafd_loader::placement::{plan_hops, residual_homes, AttentionPlacement, ExpertHome,
        HopSpec, LayerAssignment, TerminalHome};

    #[test]
    fn ml4_binding_validates_local_kv_while_execution_reads_the_latent_source() -> Result<()> {
        let cfg = GlmDsaConfig::from_hf(&cuteafd_loader::plan::testing::mistral_large4_config())?;
        let borrower = (0..cfg.layers).find(|&layer| !cfg.owns_kv(layer)).context("ML4 latent borrower")?;
        let source = cfg.kv_source(borrower);
        let pages = 3;
        let mut buffers = vec![256usize; cfg.layers];
        buffers[source] = (pages + 1) * PAGE_ROWS * cfg.record_bytes(source);
        assert!(cfg.owns_kv(source));
        validate_owned_kv(&cfg, source, pages, true, buffers[source])?;
        validate_owned_kv(&cfg, borrower, pages, false, buffers[borrower])?;
        let effective = latent_cache(&cfg, borrower, &buffers);
        assert!(std::ptr::eq(effective, &buffers[source]));
        assert!(!std::ptr::eq(effective, &buffers[borrower]));
        assert!(validate_owned_kv(&cfg, borrower, pages, false, *effective).is_err());
        assert!(validate_owned_kv(&cfg, source, pages, true, 256).is_err());
        Ok(())
    }

    struct Trace {
        plan: LayerExecutionPlan,
        schedule: RefCell<HostSchedule>,
        owners: Vec<Vec<u8>>,
        lane: Cell<usize>,
        fail: Option<usize>,
        calls: Cell<usize>,
        k6: bool,
    }
    struct Lane<'a> { trace: &'a Trace, lane: usize }

    impl Trace {
        fn new(first: u8, lanes: usize) -> Self {
            // Each four-layer group reuses its full indexer's owner.
            let modes: Vec<_> = (0..12).map(|layer| LayerMode::Whole {
                gpu: first ^ ((layer / 4) % 2) as u8, ffn: FfnMode::Split }).collect();
            let spec = HopSpec { storage: cuteafd_loader::placement::HopStorage::ReceiveSlots, row_bytes: 32, rows: 8, lanes: lanes as u64,
                entry_gpu: 0, head_gpu: 0, terminal: TerminalHome::Replicated };
            let placement = Placement { pool_tokens: 0, pool_limits: cuteafd_loader::placement::PoolLimits(vec![]),
                attention_placement: AttentionPlacement::Layers, attention_by_kind: vec![],
                peer_row_bytes: 0, onboard_layers: 0, max_pool_tokens_no_onboard: 0,
                layers: modes.iter().map(|&mode| LayerAssignment { mode, experts: ExpertHome::Dense }).collect(),
                movables: vec![], expert_ranges: vec![], residual: residual_homes(&modes, &spec).unwrap(),
                tp2: None, items: vec![], hops: plan_hops(&modes, &spec).unwrap() };
            let plan = LayerExecutionPlan::new(&placement, spec).unwrap();
            let owners = (0..modes.len()).map(|layer| plan.attention_owners(layer).unwrap()).collect();
            Self { plan, owners, schedule: RefCell::new(Default::default()), lane: Cell::new(0),
                fail: None, calls: Cell::new(0), k6: false }
        }
        fn callback(&self) -> Result<()> {
            let call = self.calls.replace(self.calls.get() + 1);
            ensure!(self.fail != Some(call), "injected GLM layer callback failure");
            Ok(())
        }
        fn run(&self, lanes: usize, order: LaneOrder) -> Result<()> {
            let mut kernels: Vec<_> = (0..lanes).map(|lane| Lane { trace: self, lane }).collect();
            pair::submit_scoped(|| Ok(()), || Ok(()), |scope|
                attention::layer_pipeline_with(scope, &self.plan, self, order, &vec![8; lanes], &mut kernels))
        }
    }
    impl LayerProtocol for Trace {
        type Ticket = (usize, usize, usize);
        fn require_live(&self) -> Result<()> { self.callback() }
        fn push_entry(&self, from: usize, _: *const c_void, destination: *mut c_void, _: usize) -> Result<()> {
            self.schedule.borrow_mut().push_to(from, "main", DIRECT, destination as usize, "entry");
            Ok(())
        }
        fn wait_entry(&self, to: usize) -> Result<()> {
            self.schedule.borrow_mut().wait_for(to, "main", DIRECT, self.lane.get() + 100, "entry");
            Ok(())
        }
        fn send(&self, index: usize, lane: usize, _: *const c_void, _: usize) -> Result<Self::Ticket> {
            self.callback()?;
            let hop = self.plan.hops()[index];
            let ordinal = self.plan.hops()[..index].iter().filter(|hop2| hop2.charged() && hop2.to == hop.to).count();
            let slot = 2 * lane + ordinal % 2;
            self.schedule.borrow_mut().push_to(hop.from as usize, "hop", slot, hop.to as usize, "attention residual");
            Ok((hop.to as usize, slot, hop.to as usize))
        }
        fn land(&self, (to, slot, target): Self::Ticket) -> Result<*mut c_void> {
            self.schedule.borrow_mut().wait_for(to, "hop", slot, target, "attention residual");
            Ok(std::ptr::null_mut())
        }
    }
    impl LayerKernels for Lane<'_> {
        fn pack(&mut self, _: Hop) -> Result<*const c_void> {
            self.trace.lane.set(self.lane);
            Ok(std::ptr::null())
        }
        fn entry_destination(&mut self, _: usize) -> Result<*mut c_void> { Ok((100 + self.lane) as *mut c_void) }
        fn adopt(&mut self, _: Hop, _: *mut c_void) -> Result<()> { self.trace.callback() }
        fn owned(&mut self, _: usize, _: usize) -> Result<()> { self.trace.callback() }
        fn heads(&mut self, _: usize) -> Result<()> { unreachable!() }
        fn context(&mut self, _: usize, _: usize) -> Result<()> { unreachable!() }
        fn ffn(&mut self, _: usize, _: Option<usize>) -> Result<()> { unreachable!() }
    }
    impl PipelineKernels for Lane<'_> {
        type FfnTicket = usize;
        fn state_owners(&self, layer: usize) -> Result<AttentionStateOwners<'_>> {
            let owners = &self.trace.owners[layer];
            Ok(AttentionStateOwners { weights: owners, kv: owners, indexer: Some(owners), compressor: None })
        }
        fn begin_ffn(&mut self, layer: usize, owner: Option<usize>) -> Result<usize> {
            self.trace.callback()?;
            assert!(owner.is_none());
            let mut schedule = self.trace.schedule.borrow_mut();
            match layer % 3 {
                1 => { // Paired RTX route publication precedes either FFN completion.
                    let at = 2 * self.lane + layer % 2;
                    schedule.push(0, "routes", at, "TP2 routes");
                    schedule.wait(1, "routes", at, "TP2 routes");
                }
                2 => {
                    if self.trace.k6 {
                        let at = slot(layer, true, self.lane);
                        schedule.push_to(1, "main", at, 1, "K6 shared core");
                        schedule.wait_for(1, "main", at, 1, "K6 shared meet");
                    }
                    schedule.sync(0, "Spark route staging");
                }
                _ => (),
            }
            if self.trace.k6 && layer % 3 != 0 {
                let at = slot(layer, true, self.lane);
                for rank in 0..if layer % 3 == 1 { 2 } else { 1 } {
                    schedule.push_to(rank, "main", at, 1, "K6 shared core");
                    schedule.wait_for(rank, "main", at, 1, "K6 shared meet");
                }
            }
            if layer % 3 == 1 {
                for rank in 0..2 { schedule.push(rank, "main", slot(layer, true, self.lane), "FFN publication"); }
            }
            Ok(layer)
        }
        fn finish_ffn(&mut self, layer: usize, ticket: usize) -> Result<()> {
            self.trace.callback()?;
            assert_eq!(layer, ticket);
            let mut schedule = self.trace.schedule.borrow_mut();
            let at = slot(layer, true, self.lane);
            let dense_core = self.trace.k6 && layer % 3 == 0;
            if layer % 3 != 1 {
                for rank in 0..2 {
                    schedule.push_to(rank, "main", at, usize::from(dense_core), "FFN publication");
                }
            }
            for rank in 0..2 {
                schedule.wait_for(rank, "main", at, usize::from(dense_core), "FFN join");
            }
            schedule.sync(0, "layer diagnostic download");
            Ok(())
        }
    }

    struct Units<'a> {
        trace: &'a Trace,
        live: [Option<usize>; 3],
        pending: std::collections::VecDeque<(usize, usize)>,
        consumed: Vec<Vec<usize>>,
        committed: usize,
        peak: usize,
        retired: bool,
    }

    impl crate::shared::prefill_pipeline::PrefillHooks for Units<'_> {
        type Ticket = usize;
        fn route(&self, layer: usize) -> crate::shared::prefill_pipeline::FfnRoute {
            if layer % 3 == 2 { crate::shared::prefill_pipeline::FfnRoute::Spark }
            else { crate::shared::prefill_pipeline::FfnRoute::Local }
        }
        fn begin_unit(&mut self, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
            self.trace.callback()?;
            ensure!(self.live[unit.slot].replace(unit.index).is_none(), "reused an in-flight GLM unit slot");
            Ok(())
        }
        fn attention(&mut self, layer: usize, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
            assert_eq!(self.live[unit.slot], Some(unit.index));
            unit_attention(self.trace, &self.trace.plan, unit, layer,
                &mut Lane { trace: self.trace, lane: unit.slot }, &mut self.consumed[unit.index])
        }
        fn local_ffn(&mut self, layer: usize, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
            let mut lane = Lane { trace: self.trace, lane: unit.slot };
            let ticket = lane.begin_ffn(layer, None)?;
            lane.finish_ffn(layer, ticket)
        }
        fn send_experts(&mut self, layer: usize, unit: &crate::shared::prefill_pipeline::Unit) -> Result<usize> {
            let ticket = Lane { trace: self.trace, lane: unit.slot }.begin_ffn(layer, None)?;
            self.pending.push_back((unit.index, layer));
            self.peak = self.peak.max(self.pending.len());
            Ok(ticket)
        }
        fn land_experts(&mut self, layer: usize, unit: &crate::shared::prefill_pipeline::Unit, ticket: usize) -> Result<()> {
            ensure!(self.pending.pop_front() == Some((unit.index, layer)), "GLM Spark completion is not FIFO");
            Lane { trace: self.trace, lane: unit.slot }.finish_ffn(layer, ticket)
        }
        fn end_unit(&mut self, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
            unit_exit(self.trace, &self.trace.plan, unit, &mut Lane { trace: self.trace, lane: unit.slot },
                &mut self.consumed[unit.index])
        }
        fn commit_unit(&mut self, unit: &crate::shared::prefill_pipeline::Unit) -> Result<()> {
            assert_eq!(self.committed, unit.index);
            assert_eq!(self.live[unit.slot].take(), Some(unit.index));
            assert!(!self.pending.iter().any(|&(index, _)| index == unit.index));
            self.committed += 1;
            Ok(())
        }
    }

    #[test]
    fn owned_continuous_depth_three_recycles_five_units_and_drains_every_failure() {
        use crate::shared::prefill_pipeline::{run_prefill_units, Unit};
        for first in [0, 1] {
            let mut calls = 0;
            for failure in std::iter::once(None).chain((0..4096).map(Some)) {
                if failure.is_some_and(|cut| cut >= calls) { break; }
                let mut trace = Trace::new(first, 3);
                trace.k6 = true;
                trace.fail = failure;
                let units: Vec<_> = (0..5).map(|index| Unit { index, sequence: 0, segments: vec![],
                    rows: index * 8..(index + 1) * 8, layers: 0..12, slot: index % 3, barrier: false }).collect();
                let state = RefCell::new(Units { trace: &trace, live: [None; 3], pending: Default::default(),
                    consumed: vec![vec![0; trace.plan.hops().len()]; 5], committed: 0, peak: 0, retired: false });
                let result = pair::submit_scoped(|| Ok(()), || {
                    let mut state = state.borrow_mut();
                    state.pending.clear();
                    state.live.fill(None);
                    state.retired = true;
                    Ok(())
                }, |_| run_prefill_units(&mut *state.borrow_mut(), &units, 3));
                let state = state.borrow();
                assert!(state.pending.is_empty() && state.live.iter().all(Option::is_none));
                if failure.is_none() {
                    result.unwrap();
                    assert!(!state.retired);
                    assert_eq!(state.committed, 5);
                    assert_eq!(state.peak, 3, "depth three collapsed to serialized GLM waves");
                    assert!(state.consumed.iter().flatten().all(|&n| n == 1));
                    check_host(&trace.schedule.borrow()).unwrap();
                    order::check_submission_faults(&trace.schedule.borrow().streams()).unwrap();
                    calls = trace.calls.get();
                    assert!(calls < 4096);
                } else {
                    assert!(result.is_err());
                    assert!(state.retired, "submission failure did not retire GLM units");
                    order::check_terminal(&trace.schedule.borrow().streams(), true).unwrap();
                }
            }
        }
    }

    #[test]
    fn packed_decode_and_ordered_prefill_drain_for_both_indexer_group_owners() {
        for first in [0, 1] {
            for (lanes, order) in [(1, LaneOrder::Independent), (1, LaneOrder::OrderedChunks), (2, LaneOrder::OrderedChunks)] {
                let trace = Trace::new(first, 2);
                trace.run(lanes, order).unwrap();
                let schedule = trace.schedule.borrow();
                check_host(&schedule).unwrap();
                order::check_submission_faults(&schedule.streams()).unwrap();
            }
        }
    }

    #[test]
    fn k6_meet_precedes_routed_publication_on_each_existing_ffn_slot() {
        use crate::shared::peer_split::order::Op;
        for first in [0, 1] {
            for (lanes, lane_order) in [(1, LaneOrder::Independent),
                (1, LaneOrder::OrderedChunks), (2, LaneOrder::OrderedChunks)] {
                let mut trace = Trace::new(first, 2);
                trace.k6 = true;
                for _ in 0..3 { trace.run(lanes, lane_order).unwrap(); }
                let host = trace.schedule.borrow();
                check_host(&host).unwrap();
                let schedule = host.streams();
                order::check(&schedule).unwrap();
                for rank in 0..2 {
                    let stream = &schedule.streams[rank];
                    let cores = stream.iter().filter(|(_, label)| label == "K6 shared core").count();
                    assert_eq!(cores, 3 * lanes * 8, "all MoE layers meet, including the terminal layer");
                    for at in (1..4 * lanes).step_by(2) {
                        let publications = stream.iter().filter_map(|(op, _)| match op {
                            Op::Push(flag) if flag.exchange == "main" && flag.slot == at => Some(flag.target),
                            _ => None,
                        }).collect::<Vec<_>>();
                        assert!(!publications.is_empty());
                        assert!(publications.windows(2).any(|pair| pair == [1, 0]),
                            "the native F32 suffix precedes the ordinary routed prefix");
                    }
                }
                // A consumer cannot skip the first sequence just because the
                // following routed publication uses a different destination.
                let mut crossed = schedule.clone();
                let (op, _) = crossed.streams[1].iter_mut()
                    .find(|(_, label)| label == "K6 shared meet").unwrap();
                let Op::Wait(flag) = op else { unreachable!() };
                flag.target = 0;
                assert!(order::check(&crossed).unwrap_err().to_string().contains("crossed destination"));
            }
        }
    }

    #[test]
    fn every_callback_failure_stops_before_a_later_layer_is_submitted() {
        let trace = Trace::new(1, 2);
        trace.run(2, LaneOrder::OrderedChunks).unwrap();
        for fail in 0..trace.calls.get() {
            let mut fault = Trace::new(1, 2);
            fault.fail = Some(fail);
            assert!(fault.run(2, LaneOrder::OrderedChunks).is_err());
            assert_eq!(fault.calls.get(), fail + 1);
            order::check_terminal(&fault.schedule.borrow().streams(), true).unwrap();
        }
    }
}
