//! GLM bindings for explicit per-layer execution; Context remains the default.
use super::*;
use crate::shared::experts::rtx::pair::SubmissionScope;
use crate::shared::peer_split::{attention::{self, AttentionStateOwners, LaneOrder, LayerKernels, PipelineKernels},
    hop::HopLink, RankDevice};
use cuteafd_loader::placement::{FfnMode, Hop, HopPoint, LayerExecutionPlan, LayerMode};

pub(super) struct Execution<'a> {
    pub(super) plan: LayerExecutionPlan,
    hops: HopLink<'a>,
}

struct RecordedExchange<'e, 'a> {
    engine: &'e GlmfEngine<'a>,
    execution: &'e Execution<'a>,
    entry: Cell<Option<*mut c_void>>,
}

impl attention::LayerProtocol for RecordedExchange<'_, '_> {
    type Ticket = (crate::shared::peer_split::hop::HopTicket, usize, usize);

    fn require_live(&self) -> Result<()> { self.engine.exchange()?.require_live() }

    fn push_entry(&self, from: usize, source: *const c_void, destination: *mut c_void, bytes: usize) -> Result<()> {
        ensure!(self.entry.get().is_none(), "unconsumed layer entry publication");
        self.engine.record_direct(from, destination, true);
        self.engine.exchange()?.push_to(from, crate::shared::peer_split::DIRECT, source, destination, bytes)?;
        self.entry.set(Some(destination));
        Ok(())
    }

    fn wait_entry(&self, to: usize) -> Result<()> {
        let destination = self.entry.take().context("layer entry publication")?;
        self.engine.record_direct(to, destination, false);
        self.engine.exchange()?.wait(to, crate::shared::peer_split::DIRECT)
    }

    fn send(&self, index: usize, lane: usize, source: *const c_void, rows: usize) -> Result<Self::Ticket> {
        let hop = self.execution.plan.hops()[index];
        let per_lane = self.execution.plan.hops().iter().filter(|h| h.charged() && h.to == hop.to)
            .count().min(cuteafd_loader::placement::HOP_SLOTS as usize);
        let ordinal = self.execution.plan.hops()[..index].iter().filter(|h| h.charged() && h.to == hop.to).count();
        let slot = lane * cuteafd_loader::placement::HOP_SLOTS as usize + ordinal % per_lane;
        self.engine.record_exchange(hop.from as usize, "hop", slot, true);
        Ok((self.execution.hops.send(index, lane, source, rows)?, hop.to as usize, slot))
    }

    fn land(&self, (ticket, to, slot): Self::Ticket) -> Result<*mut c_void> {
        self.engine.record_exchange(to, "hop", slot, false);
        self.execution.hops.land(ticket)
    }
}

struct Owners {
    weights: Vec<u8>,
    kv: Vec<u8>,
    indexer: Option<Vec<u8>>,
    compressor: Option<Vec<u8>>,
}

struct Bindings<'e, 'a, 'c> {
    engine: &'e GlmfEngine<'a>,
    work: [&'e Workspace<'a>; 2],
    tables: [&'e StepTables; 2],
    owners: Vec<Owners>,
    lane: usize,
    rows: usize,
    graphed: bool,
    tap: (usize, usize, usize),
    on_layer: Option<&'c mut dyn FnMut(usize, &[u8]) -> Result<()>>,
    prepared: Cell<Option<usize>>,
}

#[derive(Clone, Copy)]
struct Ticket {
    layer: usize,
    lane: usize,
    spark: bool,
}

impl<'a> GlmfEngine<'a> {
    pub(crate) fn bind_layer_execution(&mut self, plan: LayerExecutionPlan) -> Result<()> {
        ensure!(self.layer_execution.is_none() && self.graphs.borrow().len() == 0
            && self.peer()?.graphs.borrow().len() == 0, "layer execution must bind once before capture");
        self.validate_attention_homes(&plan)?;
        ensure!(plan.spec().entry_gpu == 0 && plan.spec().head_gpu == 0
            && plan.spec().terminal == cuteafd_loader::placement::TerminalHome::Replicated,
            "GLM layer executor requires rank-0 embedding/head and replicated terminal storage");
        // Context tables currently describe one global unit partition, not per-layer geometry.
        ensure!(self.context.is_none() && !plan.modes().contains(&LayerMode::ContextSplit),
            "GLM mixed context execution requires per-layer paging tables");
        cuteafd_loader::placement::validate_layer_pipeline_lanes(plan.spec().lanes, self.prefill_lane_count)?;
        ensure!(plan.modes().iter().all(|mode| !matches!(mode,
            LayerMode::Whole { ffn: FfnMode::Owner, .. })), "GLM layer executor requires split FFNs");
        let exchange = self.exchange()?;
        let mut hops = HopLink::new(self.library,
            [0, 1].map(|rank| RankDevice { device: exchange.device(rank), stream: exchange.stream(rank) }),
            plan.hops(), plan.spec())?;
        hops.attach_abort([exchange.abort_word(0)?.context("rank 0 terminal owner")?,
            exchange.abort_word(1)?.context("rank 1 terminal owner")?])?;
        hops.validate_binding(&plan, exchange)?;
        self.layer_waves = RefCell::new((0..self.prefill_lane_count).map(|_| None).collect());
        self.layer_execution = Some(Execution { plan, hops });
        Ok(())
    }

    fn layer_owners(&self) -> Vec<Owners> {
        (0..self.weights.layers.len()).map(|index| {
            let mut owners = Owners { weights: Vec::new(), kv: Vec::new(),
                indexer: (self.weights.layers[index].attention == GlmNextAttention::Mla).then(Vec::new),
                compressor: (self.weights.layers[index].attention == GlmNextAttention::Mla
                    && self.index_cache == IndexCache::Compact).then(Vec::new) };
            for rank in 0..self.ranks() {
                let layer = if rank == 0 { &self.weights.layers[index] } else { &self.peer.as_ref().unwrap().layers[index] };
                let cache = self.caches_of(rank);
                // Operand presence is checked against the complete required set at startup.
                let operand = if layer.attention == GlmNextAttention::Mla { "q_a_norm" } else { "conv_w" };
                if layer.has(operand) { owners.weights.push(rank as u8); }
                if layer.attention == GlmNextAttention::Mla {
                    if cache.kv[index].is_some() { owners.kv.push(rank as u8); }
                    if cache.index[index].is_some() { owners.indexer.as_mut().unwrap().push(rank as u8); }
                    if cache.index_tails.is_some() && cache.mla_ordinal[index].is_some() {
                        if let Some(owners) = &mut owners.compressor { owners.push(rank as u8); }
                    }
                } else if cache.kda_heads > 0 && cache.kda_state.buffer.bytes > 0 && cache.kda_conv.buffer.bytes > 0 {
                    owners.kv.push(rank as u8);
                }
            }
            owners
        }).collect()
    }

    fn layer_pipeline(&self, scope: &SubmissionScope<'_>, bindings: &mut [Bindings<'_, 'a, '_>]) -> Result<()> {
        let execution = self.layer_execution.as_ref().context("per-layer execution binding")?;
        let rows = bindings.iter().map(|binding| binding.rows).collect::<Vec<_>>();
        execution.hops.validate_binding(&execution.plan, self.exchange()?)?;
        let exchange = RecordedExchange { engine: self, execution, entry: Cell::new(None) };
        attention::layer_pipeline_with(scope, &execution.plan, &exchange,
            LaneOrder::OrderedChunks, &rows, bindings)?;
        for binding in bindings.iter_mut() { binding.finish()?; }
        self.finish_route_checks()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn layer_step(&self, scope: &SubmissionScope<'_>, w: &Workspace<'a>, w1: &Workspace<'a>,
        tables: &StepTables, tables1: &StepTables, rows: usize, logit_rows: usize, graphed: bool,
        on_layer: Option<&mut dyn FnMut(usize, &[u8]) -> Result<()>>)
        -> Result<Option<DeviceLogits>> {
        ensure!(!graphed || on_layer.is_none(), "layer callbacks must run outside capture");
        if graphed {
            self.release_retired_graphs()?;
            if self.embedding.device_gather() { self.gather_streams(w, rows)?; }
        }
        let mut bindings = [Bindings { engine: self, work: [w, w1], tables: [tables, tables1],
            owners: self.layer_owners(), lane: 0, rows, graphed, on_layer, prepared: Cell::new(None),
            tap: (rows.saturating_sub(crate::families::glm5::dflash::TAP_ROWS),
                rows.min(crate::families::glm5::dflash::TAP_ROWS), 0) }];
        self.layer_pipeline(scope, &mut bindings)?;
        if self.weights.layers.len() < self.cfg.layers { self.synchronize()?; return Ok(None); }
        self.run("head", &[("streams", w.streams[0].buffer.ptr), ("weight", self.weights.norm.buffer.ptr),
            ("out", w.x.buffer.ptr)], &[Scalar::I32(rows as i32)])?;
        self.logits(w, rows, logit_rows, tables.decode)?;
        Ok(Some(self.device_logits(w, logit_rows, false)))
    }

    pub(super) fn layer_steps(&self, scope: &SubmissionScope<'_>, lanes: &[(StepTables, &[u32])],
        logit_rows: usize, device: bool, media: Option<&cuteafd_engine::media::RequestMedia>)
        -> Result<Option<StepLogits>> {
        let workspaces = self.prefill_lanes_of(0, lanes.len())?;
        let peers = self.peer_workspaces(false, Some(lanes.len()))?.context("per-layer peer lanes")?;
        let total: usize = lanes.iter().map(|(tables, _)| tables.kv_slots.len()).sum();
        ensure!(!lanes.is_empty() && logit_rows <= total && (self.full_prefill_logits || logit_rows <= DECODE_ROWS),
            "per-layer prefill logit rows exceed admission");
        let units = lanes.iter().zip(workspaces.iter()).enumerate().map(|(lane, ((tables, _), w))|
            Ok((w, Some(peers.get(lane)?), tables))).collect::<Result<Vec<_>>>()?;
        self.on(1, || unsafe {
            // SAFETY: drain this engine's peer stream before replacing its persistent tables.
            self.library.cuda_stream_synchronize(self.stream_of(1))
        })?;
        let ranked = self.step_tables(&units, &lanes.iter().map(|(t, _)| placement_units(t)).collect::<Vec<_>>())?;
        let mut bindings = Vec::new();
        let mut first = 0;
        for (lane, ((tables, tokens), w)) in lanes.iter().zip(workspaces.iter()).enumerate() {
            let rows = tables.kv_slots.len();
            ensure!(rows <= w.rows && tokens.len() == rows && tables.segments.is_empty(),
                "per-layer lane needs unpacked rows within its workspace");
            self.load_streams(w, tokens, false)?;
            self.inject_media(w, tables, media)?;
            let tap_start = (total - total.min(crate::families::glm5::dflash::TAP_ROWS)).max(first).min(first + rows);
            let n = first + rows - tap_start;
            bindings.push(Bindings { engine: self, work: [w, peers.get(lane)?],
                tables: [&ranked[lane][0], &ranked[lane][1]], owners: self.layer_owners(),
                lane, rows, graphed: false, on_layer: None, prepared: Cell::new(None), tap: (tap_start - first, n,
                    tap_start.saturating_sub(total - total.min(crate::families::glm5::dflash::TAP_ROWS))) });
            first += rows;
        }
        self.prepare_route_checks(&bindings.iter().map(|binding| binding.rows).collect::<Vec<_>>())?;
        self.layer_pipeline(scope, &mut bindings)?;
        if logit_rows == 0 { self.synchronize()?; return Ok(None); }
        let last = lanes.len() - 1;
        if device {
            let w = &workspaces[last];
            let rows = lanes[last].0.kv_slots.len();
            ensure!(logit_rows <= rows, "device logits span per-layer lanes");
            self.layer_head(w, rows, logit_rows)?;
            return Ok(Some(StepLogits::Device(self.device_logits(w, logit_rows, false))));
        }
        let mut logits = Vec::with_capacity(logit_rows * self.cfg.vocab_size);
        let mut first = 0;
        for ((tables, _), w) in lanes.iter().zip(workspaces.iter()) {
            let rows = tables.kv_slots.len();
            let wanted = (first + rows).saturating_sub((total - logit_rows).max(first));
            if wanted > 0 {
                self.layer_head(w, rows, wanted)?;
                logits.extend(self.download(&w.logits, wanted * self.cfg.vocab_size * 4)?.chunks_exact(4)
                    .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())));
            }
            first += rows;
        }
        Ok(Some(StepLogits::Host(logits)))
    }

    fn layer_head(&self, w: &Workspace<'_>, rows: usize, wanted: usize) -> Result<()> {
        self.run("head", &[("streams", w.streams[0].buffer.ptr), ("weight", self.weights.norm.buffer.ptr),
            ("out", w.x.buffer.ptr)], &[Scalar::I32(rows as i32)])?;
        self.logits(w, rows, wanted, false)
    }
}

fn segments(owner: Option<usize>, mut segment: impl FnMut(usize, bool) -> Result<()>) -> Result<()> {
    for rank in 0..2 { segment(rank, owner.is_none_or(|owner| owner == rank))?; }
    Ok(())
}

impl Bindings<'_, '_, '_> {
    fn cap(&self) -> &'static str { if self.tables[0].decode { decode_cap(self.rows) } else { PREFILL_CAP } }

    fn layer(&self, rank: usize, layer: usize) -> &GlmfLayer<'_> {
        if rank == 0 { &self.engine.weights.layers[layer] } else { &self.engine.peer.as_ref().unwrap().layers[layer] }
    }

    fn tap(&self, layer: usize) -> Result<()> {
        if let Some(drafter) = &self.engine.drafter {
            let (first, rows, offset) = self.tap;
            if rows > 0 { drafter.tap_streams_at(layer, self.work[0].streams[0].buffer.ptr, HC, first, rows, offset)?; }
        }
        Ok(())
    }

    fn segment(&self, rank: usize, index: usize, active: bool) -> Result<()> {
        let e = self.engine;
        let w = self.work[rank];
        let layer = self.layer(rank, index);
        let rows = Scalar::I32(self.rows as i32);
        let body = || {
            if let Some(previous) = index.checked_sub(1).filter(|_| self.prepared.get() != Some(index)) {
                if rank == 0 {
                    let out = e.meet_ffn(w, previous, self.lane, e.weights.layers.len(), self.rows)?;
                    e.post_pre_on(0, w, out, 1, layer, "attn", "input_norm", rows, self.cap())?;
                    self.tap(previous)?;
                } else { e.peer_post(previous, self.lane, w, self.rows, self.cap())?; }
            } else if index == 0 && active { e.pre_on(rank, w, &w.streams[0], layer, rows)?; }
            if !active { return Ok(()); }
            e.attention(rank, w, index, layer, rows, self.cap(), self.tables[rank], None)?;
            let attended = if layer.attention_share.split() { self.meet(rank, index)? } else { w.delta.buffer.ptr };
            e.post_pre_on(rank, w, attended, 0, layer, "ffn", "post_norm", rows, self.cap())
        };
        if self.graphed { e.replay_on(rank, GraphKey::new(index, self.rows, self.tables[rank]), body) } else { body() }
    }

    fn meet(&self, rank: usize, layer: usize) -> Result<*mut c_void> {
        let e = self.engine;
        let w = self.work[rank];
        let weights = self.layer(rank, layer);
        let slot = slot(layer, false, self.lane);
        if rank == 0 { return e.meet_attention(w, slot, self.rows, weights, self.cap()); }
        if e.output_shard_attention(weights) { return e.complete_output_shard(rank, w, weights, slot, self.rows, self.cap()); }
        let exchange = e.exchange()?;
        let precise = e.precise_attention(weights);
        e.exchange_push(exchange, "main", rank, slot, w.delta.buffer.ptr, self.rows * e.cfg.hidden * if precise { 4 } else { 2 })?;
        e.exchange_wait(exchange, "main", rank, slot)?;
        if precise {
            e.run_on(rank, true, "add_fp32", &[("a", exchange.recv(rank, slot)?), ("b", w.delta.buffer.ptr),
                ("out", w.sum_ptr()?)], &[Scalar::I32(self.rows as i32)])?;
        } else {
            exchange.add(rank, exchange.recv(rank, slot)?, w.delta.buffer.ptr, w.sum_ptr()?, self.rows * e.cfg.hidden)?;
        }
        w.sum_ptr()
    }

    fn completed_layer(&mut self, index: usize) -> Result<()> {
        let Some(on_layer) = self.on_layer.as_mut() else { return Ok(()) };
        let e = self.engine;
        let rank = match e.layer_execution.as_ref().context("layer execution")?.plan.mode(index)? {
            LayerMode::Whole { gpu, .. } => usize::from(gpu),
            _ => 0,
        };
        let buffer = self.work[rank].streams[0].buffer;
        let bytes = self.rows * HC * e.cfg.hidden * 2;
        ensure!(bytes <= buffer.bytes, "completed layer rows exceed residual storage");
        e.on(rank, || {
            // SAFETY: both FFN joins (or the terminal copy) precede this owner
            // stream wait. The callback borrows only downloaded residual bytes.
            unsafe { e.library.cuda_stream_synchronize(e.stream_of(rank))?; }
            let mut residual = vec![0; bytes];
            e.library.copy_d2h(&mut residual, cuteafd_ffi::CuteafdDeviceBuffer { bytes, ..buffer })?;
            on_layer(index, &residual)
        })
    }

    fn finish(&mut self) -> Result<()> {
        let e = self.engine;
        let w = self.work[0];
        let last = e.weights.layers.len() - 1;
        let body = || {
            let out = e.meet_ffn(w, last, self.lane, last + 1, self.rows)?;
            e.mhc_post_at(w, out, 1, &Span { first: 0, rows: self.rows })?;
            self.tap(last)
        };
        if self.graphed { e.replay(GraphKey::new(last + 1, self.rows, self.tables[0]), body)?; } else { body()?; }
        e.replicate_terminal(w, self.work[1], 0, self.rows)?;
        self.completed_layer(last)
    }
}

impl LayerKernels for Bindings<'_, '_, '_> {
    fn pack(&mut self, hop: Hop) -> Result<*const c_void> {
        let stream = if matches!(hop.at, HopPoint::AfterAttention(_)) { 1 } else { 0 };
        Ok(self.work[usize::from(hop.from)].streams[stream].buffer.ptr)
    }
    fn entry_destination(&mut self, rank: usize) -> Result<*mut c_void> { Ok(self.work[rank].streams[0].buffer.ptr) }
    fn adopt(&mut self, hop: Hop, residual: *mut c_void) -> Result<()> {
        let rank = usize::from(hop.to);
        let w = self.work[rank];
        if let HopPoint::AfterAttention(layer) = hop.at {
            let bytes = self.rows * HC * self.engine.cfg.hidden * 2;
            self.engine.on(rank, || unsafe {
                // SAFETY: the landed buffer and persistent streams hold exactly these admitted rows.
                self.engine.library.copy_d2d_async(w.streams[1].buffer,
                    cuteafd_ffi::CuteafdDeviceBuffer { ptr: residual, bytes,
                        device_id: self.engine.exchange()?.device(rank), flags: 0 }, bytes, self.engine.stream_of(rank))
            })?;
            let weights = self.layer(rank, layer);
            self.engine.run_on(rank, false, "mhc_pre", &[("residual", w.streams[1].buffer.ptr),
                ("fn", weights.ptr("ffn.fn")?), ("scale", weights.ptr("ffn.scale")?), ("base", weights.ptr("ffn.base")?),
                ("norm", weights.ptr("post_norm")?), ("post", w.post.buffer.ptr), ("comb", w.comb.buffer.ptr),
                ("y", w.x.buffer.ptr), ("scratch", w.scratch.buffer.ptr)], &[Scalar::I32(self.rows as i32)])?;
        } else if hop.at == HopPoint::BeforeLayer(0) {
            let bytes = self.rows * HC * self.engine.cfg.hidden * 2;
            self.engine.on(rank, || unsafe {
                // SAFETY: the initial ownership hop landed these rows before the first collapse.
                self.engine.library.copy_d2d_async(w.streams[0].buffer,
                    cuteafd_ffi::CuteafdDeviceBuffer { ptr: residual, bytes,
                        device_id: self.engine.exchange()?.device(rank), flags: 0 }, bytes, self.engine.stream_of(rank))
            })?;
        } else {
            ensure!(hop.at == HopPoint::Entry && residual == w.streams[0].buffer.ptr,
                "split-FFN GLM executor has an unexpected boundary hop");
        }
        Ok(())
    }
    fn heads(&mut self, layer: usize) -> Result<()> {
        segments(None, |rank, active| self.segment(rank, layer, active))
    }
    fn context(&mut self, layer: usize, _ordinal: usize) -> Result<()> { self.heads(layer) }
    fn owned(&mut self, layer: usize, gpu: usize) -> Result<()> {
        segments(Some(gpu), |rank, active| self.segment(rank, layer, active))
    }
    fn ffn(&mut self, layer: usize, owner: Option<usize>) -> Result<()> {
        let ticket = self.begin_ffn(layer, owner)?;
        self.finish_ffn(layer, ticket)
    }
}

impl PipelineKernels for Bindings<'_, '_, '_> {
    type FfnTicket = Ticket;
    fn state_owners(&self, layer: usize) -> Result<AttentionStateOwners<'_>> {
        let owners = self.owners.get(layer).context("allocated layer state")?;
        Ok(AttentionStateOwners { weights: &owners.weights, kv: &owners.kv,
            indexer: owners.indexer.as_deref(), compressor: owners.compressor.as_deref() })
    }
    fn begin_ffn(&mut self, index: usize, owner: Option<usize>) -> Result<Ticket> {
        ensure!(owner.is_none(), "GLM per-layer FFNs must be split");
        let e = self.engine;
        let [w, w1] = self.work;
        let layer = self.layer(0, index);
        let peer = self.layer(1, index);
        let real = if self.tables[0].decode && e.startup_graphs { self.tables[0].real_rows } else { self.rows };
        let rows = Scalar::I32((if layer.dense { self.rows } else { real }) as i32);
        let masked = self.graphed && e.warming_graphs.get() && !layer.dense;
        ensure!(masked || layer.dense || real > 0, "per-layer MoE needs positive real rows");
        let mut spark = false;
        if !masked && !e.tp2_layer(index) {
            let out = GlmfEngine::peer_ffn_out(w1, peer)?;
            if !layer.dense || peer.has("w_gate_up_fp8") {
                e.ffn_on(1, w1, peer, if layer.dense { e.cfg.dense_intermediate } else { e.cfg.moe_intermediate },
                    self.cap(), out, rows)?;
            }
        }
        if masked {
            for (rank, buffer) in [(0, w.delta.buffer), (1, w1.shared.buffer)] {
                e.on(rank, || unsafe {
                    // SAFETY: masked startup owns both persistent FFN outputs of the bucket width.
                    e.library.cuda_zero_bytes_async(buffer, self.rows * e.cfg.hidden * 2, e.stream_of(rank))
                })?;
            }
        } else if layer.dense {
            e.ffn(w, layer, e.cfg.dense_intermediate, self.cap(), w.delta.buffer.ptr, rows)?;
        } else {
            e.moe_front(w, index, layer, real, rows, self.cap(), &[Span { first: 0, rows: real }])?;
            e.probe_ring(index, w, real, matches!(e.routed_experts(index), Some(Experts::Spark { .. })))?;
            if e.tp2_layer(index) {
                e.tp2_pair(index, self.lane, w, Some(w1), real, rows, self.cap())?;
            } else if let Some(Experts::Spark { transports, .. }) = e.routed_experts(index) {
                ensure!(e.layer_waves.borrow()[self.lane].is_none(), "unconsumed per-layer Spark wave");
                let mut transports = transports.borrow_mut();
                let transport = transports.get_mut(self.lane).context("per-layer Spark lane transport")?;
                let wave = e.spark_dispatch(w, index, real, self.tables[0].decode, transport, || {
                    e.ffn(w, layer, e.cfg.moe_intermediate, self.cap(), w.shared.buffer.ptr, rows)
                })?;
                e.layer_waves.borrow_mut()[self.lane] = Some(wave);
                spark = true;
            } else {
                e.moe_experts(w, index, layer, real, rows, self.cap(), self.tables[0].decode, Some(w1))?;
            }
        }
        Ok(Ticket { layer: index, lane: self.lane, spark })
    }
    fn finish_ffn(&mut self, index: usize, ticket: Ticket) -> Result<()> {
        ensure!(ticket.layer == index && ticket.lane == self.lane, "per-layer FFN ticket identity");
        let e = self.engine;
        let real = if self.tables[0].decode && e.startup_graphs { self.tables[0].real_rows } else { self.rows };
        if ticket.spark {
            let Some(Experts::Spark { transports, runtime }) = e.routed_experts(index) else { anyhow::bail!("Spark ticket backend changed"); };
            let wave = e.layer_waves.borrow_mut()[self.lane].take().context("retained Spark wave")?;
            // Only protocol assembly state moves into receive. All QPs, receive registrations,
            // intake planes and pinned upload owners remain in the engine through teardown.
            let mut transports = transports.borrow_mut();
            runtime.block_on(e.spark_land(self.work[0], real,
                transports.get_mut(self.lane).context("Spark completion transport")?, wave))?;
        }
        if !self.layer(0, index).dense && real < self.rows {
            let offset = real * e.cfg.hidden * 2;
            let bytes = (self.rows - real) * e.cfg.hidden * 2;
            let peer = self.layer(1, index);
            let peer_buffer = if peer.dense && peer.has("w_gate_up_fp8") { self.work[1].delta.buffer }
                else if peer.dense { self.work[1].zero.as_ref().context("peer zero rows")?.buffer }
                else { self.work[1].shared.buffer };
            for (rank, buffer) in [(0, self.work[0].delta.buffer), (1, peer_buffer)] {
                e.on(rank, || unsafe {
                    // SAFETY: the padded tail is within the live bucket output, after real-row FFNs.
                    e.library.cuda_zero_bytes_async(cuteafd_ffi::CuteafdDeviceBuffer {
                        ptr: buffer.ptr.cast::<u8>().add(offset).cast(), bytes, ..buffer }, bytes, e.stream_of(rank))
                })?;
            }
        }
        e.exchange_push(e.exchange()?, "main", 1, slot(index, true, self.lane),
            GlmfEngine::peer_ffn_out(self.work[1], self.layer(1, index))?, self.rows * e.cfg.hidden * 2)?;
        crate::shared::console::layer_mark(index);
        e.probe_mark(index)?;
        if self.on_layer.is_some() && index + 1 < e.weights.layers.len() {
            // Materialize the normal fused boundary early for diagnostics.
            // Both rank joins must precede any callback host synchronization.
            let out = e.meet_ffn(self.work[0], index, self.lane, e.weights.layers.len(), self.rows)?;
            e.post_pre_on(0, self.work[0], out, 1, self.layer(0, index + 1),
                "attn", "input_norm", Scalar::I32(self.rows as i32), self.cap())?;
            e.peer_post(index, self.lane, self.work[1], self.rows, self.cap())?;
            self.tap(index)?;
            self.prepared.set(Some(index + 1));
            self.completed_layer(index)?;
        }
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use crate::shared::{experts::rtx::pair::{self, SubmissionOwners, TestSubmission},
        peer_split::{attention::LayerProtocol, order::{self, host::{check_host, HostOp, HostSchedule}}, DIRECT}};
    use cuteafd_loader::placement::{plan_hops, residual_homes, AttentionPlacement, ExpertHome,
        HopSpec, LayerAssignment, Placement, TerminalHome};

    #[derive(Clone, Copy, Debug)]
    enum Backend { Dense, Local, Tp2, Spark }

    struct Trace {
        plan: LayerExecutionPlan,
        schedule: RefCell<HostSchedule>,
        owners: RefCell<TestSubmission<'static>>,
        state: Vec<Vec<u8>>,
        backends: Vec<Backend>,
        lane: Cell<usize>,
        broadcast: bool,
        output_shard: bool,
        fail: Option<(usize, bool)>,
        calls: Cell<usize>,
        pending: Cell<usize>,
        diagnostics: bool,
        completed: RefCell<Vec<(usize, usize, usize)>>,
    }
    struct Lane<'a> { trace: &'a Trace, lane: usize }

    fn plan(modes: &[LayerMode], lanes: usize) -> LayerExecutionPlan {
        let spec = HopSpec { storage: cuteafd_loader::placement::HopStorage::ReceiveSlots,
            row_bytes: 32, rows: 8, lanes: lanes as u64, entry_gpu: 0,
            head_gpu: 0, terminal: TerminalHome::Replicated };
        let placement = Placement { pool_tokens: 0, pool_limits: cuteafd_loader::placement::PoolLimits(vec![]), attention_placement: AttentionPlacement::Layers,
            attention_by_kind: vec![], peer_row_bytes: 0, onboard_layers: 0,
            layers: modes.iter().map(|&mode| LayerAssignment { mode, experts: ExpertHome::Dense }).collect(),
            movables: vec![], expert_ranges: vec![], residual: residual_homes(modes, &spec).unwrap(),
            tp2: None, items: vec![], hops: plan_hops(modes, &spec).unwrap() };
        LayerExecutionPlan::new(&placement, spec).unwrap()
    }
    #[test]
    fn peer_attention_shares_must_match_the_immutable_plan_before_caches_allocate() {
        use crate::shared::placement::layers::AttentionShare;
        for first in [0, 1] {
            let modes = [LayerMode::HeadSplit, LayerMode::Whole { gpu: first, ffn: FfnMode::Split },
                LayerMode::Whole { gpu: 1 - first, ffn: FfnMode::Split }];
            let execution = plan(&modes, 2);
            for rank in 0..2 {
                let mut shares: Vec<_> = modes.iter().map(|&mode| AttentionShare::bind(mode, 2).unwrap()[rank]).collect();
                validate_attention_shares(&execution, shares.iter().copied(), rank, 2).unwrap();
                shares[1] = AttentionShare::Split;
                assert!(validate_attention_shares(&execution, shares.iter().copied(), rank, 2).is_err());
                assert!(validate_attention_shares(&execution, shares[..2].iter().copied(), rank, 2).is_err());
            }
        }
    }

    #[test]
    fn unsupported_whole_ffn_and_mixed_context_refuse_before_cache_allocation() {
        use crate::shared::placement::layers::AttentionShare;
        for modes in [
            vec![LayerMode::Whole { gpu: 0, ffn: FfnMode::Owner }, LayerMode::HeadSplit],
            vec![LayerMode::ContextSplit, LayerMode::Whole { gpu: 1, ffn: FfnMode::Split }],
        ] {
            let execution = plan(&modes, 2);
            for rank in 0..2 {
                let shares = modes.iter().map(|&mode| AttentionShare::bind(mode, 2).unwrap()[rank]);
                assert!(validate_attention_shares(&execution, shares, rank, 2).is_err());
            }
        }
    }

    impl Trace {
        fn new(modes: &[LayerMode], lanes: usize, broadcast: bool, output_shard: bool) -> Self {
            let plan = plan(modes, lanes);
            Self { state: (0..modes.len()).map(|layer| plan.attention_owners(layer).unwrap()).collect(),
                plan, schedule: RefCell::new(Default::default()), owners: RefCell::new(TestSubmission::new(None)),
                backends: (0..modes.len()).map(|i| [Backend::Dense, Backend::Local, Backend::Tp2, Backend::Spark][i % 4]).collect(),
                lane: Cell::new(0), broadcast, output_shard, fail: None, calls: Cell::new(0), pending: Cell::new(0),
                diagnostics: false, completed: RefCell::new(Vec::new()) }
        }
        fn callback(&self) -> Result<()> {
            self.owners.borrow().require_live()?;
            let at = self.calls.replace(self.calls.get() + 1);
            if let Some((failure, panic)) = self.fail {
                if at == failure {
                    if panic { panic!("injected GLM adapter callback {at}"); }
                    anyhow::bail!("injected GLM adapter callback {at}");
                }
            }
            Ok(())
        }
        fn queue(&self, op: impl FnOnce(&mut HostSchedule)) -> Result<()> {
            self.callback()?;
            op(&mut self.schedule.borrow_mut());
            Ok(())
        }
        fn final_lane(&self, lane: usize) -> Result<()> {
            let last = self.plan.modes().len() - 1;
            self.queue(|s| s.wait(0, "main", slot(last, true, lane), "final FFN"))?;
            let target = 100 + lane;
            self.queue(|s| s.push_to(0, "main", DIRECT, target, "terminal residual"))?;
            self.queue(|s| s.wait_for(1, "main", DIRECT, target, "terminal residual"))?;
            if self.diagnostics { self.completed_layer(last, lane)?; }
            Ok(())
        }
        fn join(&self, layer: usize, lane: usize) -> Result<()> {
            let at = slot(layer, true, lane);
            self.queue(|s| s.push(0, "main", at, "previous FFN"))?;
            for rank in 0..2 { self.queue(|s| s.wait(rank, "main", at, "previous FFN"))?; }
            Ok(())
        }
        fn completed_layer(&self, layer: usize, lane: usize) -> Result<()> {
            let owner = match self.plan.mode(layer)? {
                LayerMode::Whole { gpu, .. } => usize::from(gpu),
                _ => 0,
            };
            self.queue(|s| s.sync(owner, "completed-layer callback"))?;
            self.callback()?;
            self.completed.borrow_mut().push((layer, lane, owner));
            Ok(())
        }
        fn run(&self, lanes: usize) -> Result<()> {
            let mut kernels: Vec<_> = (0..lanes).map(|lane| Lane { trace: self, lane }).collect();
            pair::submit_scoped(|| self.owners.borrow().require_live(), || pair::teardown(&mut &*self), |scope| {
                attention::layer_pipeline_with(scope, &self.plan, self, LaneOrder::OrderedChunks,
                    &vec![8; lanes], &mut kernels)?;
                // Every last-layer peer FFN producer must precede the first terminal landing.
                for lane in 0..lanes { self.final_lane(lane)?; }
                Ok(())
            })
        }
    }
    impl SubmissionOwners for &Trace {
        fn abort_all(&mut self) -> Result<()> { self.owners.borrow_mut().abort_all() }
        fn drain_ranks(&mut self) -> Result<()> {
            self.owners.borrow_mut().drain_ranks()?;
            order::check_terminal(&self.schedule.borrow().streams(), true).map_err(|error| anyhow::anyhow!("{error}"))
        }
        fn quiesce_links(&mut self) -> Result<()> { self.owners.borrow_mut().quiesce_links() }
        fn drain_copies(&mut self) -> Result<()> { self.owners.borrow_mut().drain_copies() }
        fn retire_graphs(&mut self) -> Result<()> {
            self.owners.borrow_mut().retire_graphs()?;
            self.pending.set(0);
            Ok(())
        }
        fn finish(&mut self, drained: bool) { self.owners.borrow_mut().finish(drained); }
    }
    impl LayerProtocol for Trace {
        type Ticket = (usize, usize, usize);
        fn require_live(&self) -> Result<()> { self.callback() }
        fn push_entry(&self, from: usize, _: *const c_void, destination: *mut c_void, _: usize) -> Result<()> {
            self.queue(|s| s.push_to(from, "main", DIRECT, destination as usize, "entry"))
        }
        fn wait_entry(&self, to: usize) -> Result<()> {
            let target = 100 + self.lane.get();
            self.queue(|s| s.wait_for(to, "main", DIRECT, target, "entry"))
        }
        fn send(&self, index: usize, lane: usize, _: *const c_void, _: usize) -> Result<Self::Ticket> {
            let hop = self.plan.hops()[index];
            let ordinal = self.plan.hops()[..index].iter().filter(|h| h.charged() && h.to == hop.to).count();
            let slot = 2 * lane + ordinal % 2;
            self.queue(|s| s.push_to(hop.from as usize, "hop", slot, hop.to as usize, "hop"))?;
            Ok((hop.to as usize, slot, hop.to as usize))
        }
        fn land(&self, (to, slot, target): Self::Ticket) -> Result<*mut c_void> {
            self.queue(|s| s.wait_for(to, "hop", slot, target, "hop"))?;
            Ok(std::ptr::null_mut())
        }
    }
    impl LayerKernels for Lane<'_> {
        fn pack(&mut self, _: Hop) -> Result<*const c_void> {
            self.trace.callback()?;
            self.trace.lane.set(self.lane);
            Ok(std::ptr::null())
        }
        fn entry_destination(&mut self, _: usize) -> Result<*mut c_void> { Ok((100 + self.lane) as *mut c_void) }
        fn adopt(&mut self, _: Hop, _: *mut c_void) -> Result<()> { self.trace.callback() }
        fn heads(&mut self, layer: usize) -> Result<()> { self.segment(layer, None) }
        fn context(&mut self, _: usize, _: usize) -> Result<()> { anyhow::bail!("mixed context unbound") }
        fn owned(&mut self, layer: usize, gpu: usize) -> Result<()> { self.segment(layer, Some(gpu)) }
        fn ffn(&mut self, _: usize, _: Option<usize>) -> Result<()> { unreachable!() }
    }
    impl Lane<'_> {
        fn segment(&self, layer: usize, owner: Option<usize>) -> Result<()> {
            segments(owner, |rank, active| {
                if layer > 0 && !self.trace.diagnostics {
                    let previous = slot(layer - 1, true, self.lane);
                    if rank == 0 { self.trace.queue(|s| s.push(0, "main", previous, "previous FFN"))?; }
                    self.trace.queue(|s| s.wait(rank, "main", previous, "previous FFN"))?;
                }
                if active && owner.is_none() {
                    let output = slot(layer, false, self.lane);
                    let slots = if self.trace.output_shard { vec![norm_slot(self.trace.plan.spec().lanes as usize, output), output] }
                        else { vec![output] };
                    for at in slots {
                        self.trace.queue(|s| s.push(rank, "main", at, "attention"))?;
                        self.trace.queue(|s| s.wait(rank, "main", at, "attention"))?;
                    }
                }
                Ok(())
            })
        }
    }
    impl PipelineKernels for Lane<'_> {
        type FfnTicket = Ticket;
        fn state_owners(&self, layer: usize) -> Result<AttentionStateOwners<'_>> {
            self.trace.callback()?;
            let owners = &self.trace.state[layer];
            Ok(AttentionStateOwners { weights: owners, kv: owners, indexer: Some(owners), compressor: None })
        }
        fn begin_ffn(&mut self, layer: usize, owner: Option<usize>) -> Result<Ticket> {
            assert!(owner.is_none());
            self.trace.callback()?;
            if matches!(self.trace.backends[layer], Backend::Tp2) && self.trace.broadcast {
                let at = 2 * self.lane + layer % 2;
                self.trace.queue(|s| s.push(0, "routes", at, "routes"))?;
                self.trace.queue(|s| s.wait(1, "routes", at, "routes"))?;
            }
            let spark = matches!(self.trace.backends[layer], Backend::Spark);
            if spark {
                self.trace.pending.set(self.trace.pending.get() | (1 << self.lane));
                self.trace.queue(|s| s.sync(0, "Spark route copies ready"))?;
            }
            Ok(Ticket { layer, lane: self.lane, spark })
        }
        fn finish_ffn(&mut self, layer: usize, ticket: Ticket) -> Result<()> {
            self.trace.callback()?;
            assert_eq!((ticket.layer, ticket.lane), (layer, self.lane));
            if ticket.spark { self.trace.pending.set(self.trace.pending.get() & !(1 << self.lane)); }
            self.trace.queue(|s| s.push(1, "main", slot(layer, true, self.lane), "peer FFN"))?;
            if self.trace.diagnostics && layer + 1 < self.trace.plan.modes().len() {
                self.trace.join(layer, self.lane)?;
                self.trace.completed_layer(layer, self.lane)?;
            }
            Ok(())
        }
    }

    fn modes(first: u8) -> Vec<LayerMode> {
        (0..8).map(|layer| match layer % 3 {
            0 => LayerMode::Whole { gpu: first, ffn: FfnMode::Split },
            1 => LayerMode::HeadSplit,
            _ => LayerMode::Whole { gpu: 1 - first, ffn: FfnMode::Split },
        }).collect()
    }
    #[test]
    fn mixed_owner_adapter_schedule_drains_device_and_spark_host_waits() {
        for lanes in 1..=2 { for first in 0..2 { for broadcast in [false, true] { for shard in [false, true] {
            let trace = Trace::new(&modes(first), lanes, broadcast, shard);
            trace.run(lanes).unwrap();
            let schedule = trace.schedule.borrow();
            check_host(&schedule).unwrap();
            order::check_submission_faults(&schedule.streams()).unwrap();
            assert_eq!(trace.pending.get(), 0);
            let first_terminal = schedule.ops.iter().position(|op| matches!(op,
                HostOp::Queue { label, .. } if label == "terminal residual")).unwrap();
            assert_eq!(schedule.ops[..first_terminal].iter().filter(|op| matches!(op,
                HostOp::Queue { label, .. } if label == "peer FFN")).count(), lanes * trace.plan.modes().len());
        }}}}
    }
    #[test]
    fn completed_layer_callbacks_drain_at_each_actual_owner() {
        for first in 0..2 {
            let mut trace = Trace::new(&modes(first), 1, true, true);
            trace.diagnostics = true;
            trace.run(1).unwrap();
            check_host(&trace.schedule.borrow()).unwrap();
            let expected: Vec<_> = (0..trace.plan.modes().len()).map(|layer| {
                let owner = match trace.plan.mode(layer).unwrap() {
                    LayerMode::Whole { gpu, .. } => usize::from(gpu),
                    _ => 0,
                };
                (layer, 0, owner)
            }).collect();
            assert_eq!(*trace.completed.borrow(), expected);
            for at in 0..trace.calls.get() { for panic in [false, true] {
                let mut fault = Trace::new(&modes(first), 1, true, true);
                fault.diagnostics = true;
                fault.fail = Some((at, panic));
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| fault.run(1)));
                assert!(result.is_err() || result.unwrap().is_err());
                fault.owners.borrow().verify_retired().unwrap();
                assert!(fault.run(1).is_err());
            }}
        }
    }

    #[test]
    fn adapter_faults_retire_retained_waves_and_reject_replay() {
        for first in 0..2 {
            let healthy = Trace::new(&modes(first), 2, true, true);
            healthy.run(2).unwrap();
            for at in 0..healthy.calls.get() { for panic in [false, true] {
                let mut trace = Trace::new(&modes(first), 2, true, true);
                trace.fail = Some((at, panic));
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| trace.run(2)));
                assert!(result.is_err() || result.unwrap().is_err());
                trace.owners.borrow().verify_retired().unwrap();
                assert_eq!(trace.pending.get(), 0);
                assert!(trace.run(2).is_err());
            }}
        }
    }
}
