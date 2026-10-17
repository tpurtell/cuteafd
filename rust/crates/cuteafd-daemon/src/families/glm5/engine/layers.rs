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
    waves: RefCell<Vec<Option<SparkExpertWave>>>,
    pub(super) segments: Vec<GlmOwnedSegment>,
}

enum Transport<'t, 'l> {
    Serial(Option<(&'t mut SparkLink<'l>, &'t tokio::runtime::Runtime)>),
    Lane(&'t mut SparkLane<'l>),
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

#[derive(Clone, Copy)]
enum Ticket {
    Complete,
    Skip(usize),
    Spark,
    Lane,
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
                    && layer.range("q_a_norm").is_some() == (gpu as usize == rank), "GLM layer {index} owner differs from loaded weights");
                let (kv, index_cache, _) = self.rank_caches(rank, index);
                ensure!(kv.buffer.bytes == if rank == gpu as usize { (self.pages + 1) * RECORD_PAGE_BYTES } else { 256 },
                    "GLM layer {index} KV ownership differs from placement");
                ensure!(index_cache.is_some() == (rank == gpu as usize && layer.full_indexer), "GLM index ownership");
            }
            if self.cfg.index_source(index) < plan.modes().len() {
                ensure!(plan.mode(self.cfg.index_source(index))? == mode, "GLM shared indexer owner mismatch");
            }
        }
        let exchange = self.exchange()?;
        let mut hops = HopLink::new(self.library, [0, 1].map(|rank| RankDevice {
            device: exchange.device(rank), stream: exchange.stream(rank) }), plan.hops(), spec)?;
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
            layer.range("q_a_norm").is_some() && self.rank_caches(rank, index).0.buffer.bytes > 256
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
            owners, on_layer, tap: (rows.saturating_sub(super::super::dflash::TAP_ROWS), rows.min(super::super::dflash::TAP_ROWS), 0) }];
        self.run_layer_pipeline(scope, LaneOrder::Independent, &mut bindings)?;
        if let Some(pair) = &self.tp2 { pair.finish_checks()?; }
        if self.weights.layers.len() < self.cfg.layers { return Ok(None); }
        self.launch_head(w, rows, wanted)?;
        if tables[0].decode && rows == wanted { self.select_greedy(w, rows)?; }
        Ok(Some(self.device_logits(w, wanted, tables[0].decode && rows == wanted)))
    }

    pub(super) fn layer_prefill(&self, scope: &SubmissionScope<'_>, work: &[&Workspace<'a>],
        peers: &PeerWorkspaces<'_, 'a>, tables: &[[StepTables; 2]], wanted: usize) -> Result<Option<StepLogits>> {
        let mut transports = self.lanes.borrow_mut();
        ensure!(transports.len() >= tables.len(), "GLM per-layer lane transport inventory");
        let total = tables.iter().map(|t| t[0].positions.len()).sum::<usize>();
        let mut first = 0;
        let mut bindings = Vec::new();
        for (lane, ((table, w), transport)) in tables.iter().zip(work).zip(transports.iter_mut()).enumerate() {
            let rows = table[0].positions.len();
            let start = total.saturating_sub(super::super::dflash::TAP_ROWS).max(first).min(first + rows);
            let owners = self.loaded_attention_owners();
            bindings.push(Bindings { engine: self, work: [w, peers.get(lane)?], tables: table, lane,
                transport: Transport::Lane(transport), owners, on_layer: None,
                tap: (start - first, first + rows - start, start.saturating_sub(total.saturating_sub(super::super::dflash::TAP_ROWS))) });
            first += rows;
        }
        self.run_layer_pipeline(scope, LaneOrder::OrderedChunks, &mut bindings)?;
        if let Some(pair) = &self.tp2 { pair.finish_checks()?; }
        let counts: Vec<_> = tables.iter().map(|table| table[0].positions.len()).collect();
        self.prefill_logits(work, &counts, wanted)
    }
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
                    e.norm_full(rank, w, weight, 2, Scalar::I32(self.rows() as i32), first, second)
                }
            })?;
        }
        if let Some(drafter) = &e.drafter {
            let (first, rows, offset) = self.tap;
            if rows > 0 { drafter.tap_at(index, self.work[0].h.buffer.ptr, first, rows, offset)?; }
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
            e.attention_on(rank, w, index, self.layer(rank, index), rows, self.cap(), &self.tables[rank])?;
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
        let layer = self.layer(0, index);
        let rows = Scalar::I32(self.rows() as i32);
        if !layer.dense && e.tp2_layer(index) {
            e.moe_front(w, layer, self.rows())?;
            e.ring_local_routes(index, w, self.rows())?;
            e.ffn(w, layer, e.cfg.moe_intermediate, self.cap(), w.shared.buffer.ptr, rows)?;
            e.moe_front_on(1, w1, self.layer(1, index), self.rows())?;
            e.ffn_on(1, w1, self.layer(1, index), e.cfg.moe_intermediate, self.cap(), w1.shared.buffer.ptr, rows)?;
            let pair = e.tp2.as_ref().context("TP2")?;
            e.tp2_rank(pair, 0, index, self.lane, w, self.rows())?;
            e.tp2_rank(pair, 1, index, self.lane, w1, self.rows())?;
            return Ok(Ticket::Complete);
        }
        self.segment(1, index, GlmOwnedPhase::FfnFront, || e.ffn_on(1, w1, self.layer(1, index),
            if layer.dense { e.cfg.dense_intermediate } else { e.cfg.moe_intermediate }, self.cap(),
            if layer.dense { w1.delta.buffer.ptr } else { w1.shared.buffer.ptr }, rows))?;
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
        e.moe_stage(w, layer, real, self.cap())?;
        let decode = self.tables[0].decode;
        if decode { e.prefetch(index)?; }
        match &mut self.transport {
            Transport::Serial(Some((link, _))) => {
                let execution = e.layer_execution.as_ref().context("layer execution")?;
                ensure!(execution.waves.borrow()[self.lane].is_none(), "previous Spark wave is still live");
                execution.waves.borrow_mut()[self.lane] = Some(e.moe_send(w, index, real, decode, link)?);
                Ok(Ticket::Spark)
            },
            Transport::Lane(lane) => { e.lane_send(w, index, real, lane)?; Ok(Ticket::Lane) }
            _ => anyhow::bail!("GLM MoE layer {index} needs Spark peers"),
        }
    }
    fn finish_ffn(&mut self, index: usize, ticket: Ticket) -> Result<()> {
        let e = self.engine;
        let w = self.work[0];
        let real = self.tables[0].exchange_rows;
        let planes = match ticket {
            Ticket::Skip(ranks) => Some((e.skip.as_ref().context("skip")?.pointers(), ranks)),
            Ticket::Spark => match &mut self.transport {
                Transport::Serial(Some((link, runtime))) => {
                    let wave = e.layer_execution.as_ref().context("layer execution")?.waves.borrow_mut()[self.lane]
                        .take().context("missing dispatched Spark wave")?;
                    Some((link.intake.pointers(), runtime.block_on(e.moe_land(real, link, wave))?))
                },
                _ => anyhow::bail!("Spark completion transport changed"),
            },
            Ticket::Lane => match &mut self.transport {
                Transport::Lane(lane) => Some((lane.intake.pointers(), e.lane_land(real, lane)?)),
                _ => anyhow::bail!("lane completion transport changed"),
            },
            Ticket::Complete => None,
        };
        if let Some((planes, ranks)) = planes { e.reduce(planes, w, ranks, real)?; }
        if !e.tp2_layer(index) {
            let bytes = self.rows() * e.cfg.hidden * 2;
            for rank in 0..2 {
                let w = self.work[rank];
                let out = if rank == 0 || self.layer(rank, index).dense { w.delta.buffer.ptr } else { w.shared.buffer.ptr };
                e.exchange()?.push(rank, slot(index, true, self.lane), out, bytes)?;
            }
        }
        self.join(index)?;
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

    struct Trace {
        plan: LayerExecutionPlan,
        schedule: RefCell<HostSchedule>,
        owners: Vec<Vec<u8>>,
        lane: Cell<usize>,
        fail: Option<usize>,
        calls: Cell<usize>,
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
                peer_row_bytes: 0, onboard_layers: 0,
                layers: modes.iter().map(|&mode| LayerAssignment { mode, experts: ExpertHome::Dense }).collect(),
                movables: vec![], expert_ranges: vec![], residual: residual_homes(&modes, &spec).unwrap(),
                tp2: None, items: vec![], hops: plan_hops(&modes, &spec).unwrap() };
            let plan = LayerExecutionPlan::new(&placement, spec).unwrap();
            let owners = (0..modes.len()).map(|layer| plan.attention_owners(layer).unwrap()).collect();
            Self { plan, owners, schedule: RefCell::new(Default::default()), lane: Cell::new(0),
                fail: None, calls: Cell::new(0) }
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
                2 => schedule.sync(0, "Spark route staging"),
                _ => (),
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
            if layer % 3 != 1 {
                for rank in 0..2 { schedule.push(rank, "main", slot(layer, true, self.lane), "FFN publication"); }
            }
            for rank in 0..2 { schedule.wait(rank, "main", slot(layer, true, self.lane), "FFN join"); }
            schedule.sync(0, "layer diagnostic download");
            Ok(())
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
