//! Native projections share the neutral K6 owners; only Flash stage wiring lives here.
use super::*;
use crate::shared::{dense_trellis::{CoreHead, CoreWeights, Projection, ProjectionLayout, ProjectionScratch, codebook},
    memory::device::{Allocation, Device}};
use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer, dense_trellis::TrellisDType, F32Add};
use cuteafd_loader::families::glm5_flash::core_admission::{CoreCharges, CoreRankLayout};
use std::collections::BTreeMap;

type Key = (usize, usize, usize, &'static str);
fn key(w: &CoreWeights<'_>) -> Key {
    (w.shape[1], w.shape[0], if w.side == cuteafd_loader::formats::dense_trellis::TrellisSide::Input { 2 } else { 1 }, codebook(w))
}
fn rows(value: Scalar) -> Result<usize> {
    match value { Scalar::I32(n) if n > 0 => Ok(n as usize), _ => anyhow::bail!("Flash K6 positive row count") }
}
fn ffn_key(layer: &GlmfLayer<'_>, projection: &str) -> String {
    format!("mlp.{}{projection}_proj", if layer.dense { "" } else { "shared_experts." })
}
pub(super) fn native_ffn(layer: &GlmfLayer<'_>) -> bool { layer.core.contains_key(&ffn_key(layer, "down")) }
fn native_attention(layer: &GlmfLayer<'_>) -> bool { layer.core.contains_key("self_attn.o_proj") }

struct Scratch<'a> { projection: ProjectionScratch<'a>, stages: BTreeMap<&'static str, Allocation<'a>> }
impl Scratch<'_> {
    fn stage(&self, name: &str) -> Result<Buffer> { Ok(self.stages.get(name).context("Flash K6 stage")?.buffer) }
}
pub(super) struct Rank<'a> {
    projections: BTreeMap<Key, Projection<'a>>,
    scratch: [Option<Rc<Scratch<'a>>>; 2],
    add: F32Add<'a>,
}
pub(super) struct Slot<'a> {
    scratch: Rc<Scratch<'a>>,
    retained: [Allocation<'a>; 3],
    lane: usize,
}
impl<'a> Rank<'a> {
    fn new(engine: &GlmfEngine<'a>, rank: usize, admission: &CoreRankLayout, layers: &[GlmfLayer<'_>]) -> Result<Self> {
        let device = Device { library: engine.library, id: if rank == 0 { engine.device } else { engine.peer()?.device } };
        let mut projections = BTreeMap::new();
        for weights in layers.iter().flat_map(|l| l.core.values()) {
            let k = key(weights);
            if projections.contains_key(&k) { continue; }
            let p = Projection::load_geometry(engine.programs, k.0, k.1, k.2, k.3,
                engine.prefill_rows.max(engine.decode_rows), TrellisDType::Bf16)?;
            ProjectionLayout::new(&p.program, engine.prefill_rows.max(engine.decode_rows))?;
            projections.insert(k, p);
        }
        let allocate = |index: usize| -> Result<Option<Rc<Scratch<'a>>>> {
            let expected = if index == 0 { admission.decode.as_ref() } else { admission.prefill.as_ref() };
            let Some(expected) = expected else { ensure!(projections.is_empty(), "Flash K6 body not admitted"); return Ok(None); };
            let mut actual: Option<cuteafd_loader::formats::trellis_core::ProjectionScratchLayout> = None;
            for weights in layers.iter().flat_map(|l| l.core.values()) {
                let next = cuteafd_loader::formats::trellis_core::ProjectionScratchLayout::new(weights.shape[1], weights.shape[0], expected.rows)?;
                if let Some(actual) = &mut actual { actual.merge(&next)?; } else { actual = Some(next); }
            }
            ensure!(actual.as_ref() == Some(expected), "Flash K6 loaded/admitted projection scratch mismatch");
            let projection = ProjectionLayout { rows: expected.rows, extents: expected.extents.clone() }.allocate(device)?;
            let stages = admission.stages.as_ref().context("Flash K6 stages not admitted")?[index].extents.iter()
                .map(|(&n, &b)| Ok((n, Allocation::new(device, b)?))).collect::<Result<_>>()?;
            Ok(Some(Rc::new(Scratch { projection, stages })))
        };
        Ok(Self { scratch: [allocate(0)?, allocate(1)?], projections, add: engine.library.f32_add()? })
    }
    fn slot(&self, engine: &GlmfEngine<'a>, rank: usize, rows: usize, decode: bool, lane: usize) -> Result<Option<Slot<'a>>> {
        let Some(scratch) = &self.scratch[usize::from(!decode)] else { return Ok(None); };
        let device = Device { library: engine.library, id: if rank == 0 { engine.device } else { engine.peer()?.device } };
        let allocate = || Allocation::new(device, rows * engine.cfg.hidden * 4);
        Ok(Some(Slot { scratch: scratch.clone(), retained: [allocate()?, allocate()?, allocate()?], lane }))
    }
}

impl<'a> GlmfEngine<'a> {
    pub(crate) fn bind_core(&mut self, admission: Option<CoreCharges>) -> Result<()> {
        ensure!(self.core.is_empty() && self.core_admission.is_none(), "Flash K6 bound twice");
        let loaded = self.weights.layers.iter().any(|l| !l.core.is_empty()) || matches!(self.weights.head, GlmfHead::K6(_));
        if !loaded { ensure!(admission.is_none(), "Flash K6 admission without weights"); return Ok(()); }
        let admitted = admission.as_ref().context("Flash K6 requires exact scratch admission")?;
        ensure!(admitted.ranks.len() == self.ranks(), "Flash K6 admitted rank count");
        for name in &admitted.programs { self.programs.spec(name)?; }
        for rank in 0..self.ranks() {
            self.on(rank, || self.programs.load_matching(|name| admitted.programs.iter().any(|n| n == name)))?;
            let layers = if rank == 0 { &self.weights.layers } else { &self.peer()?.layers };
            let layout = &admitted.ranks[rank];
            ensure!(layout.retained_rows == if layout.body.is_empty() { 0 } else {
                self.decode_rows + self.prefill_rows * self.prefill_waves }, "Flash K6 retained slots differ from admission");
            self.core.push(Some(self.on(rank, || Rank::new(self, rank, layout, layers))?));
        }
        if let GlmfHead::K6(weights) = &self.weights.head {
            let layout = admitted.ranks[0].target_head.as_ref().context("Flash K6 target head not admitted")?;
            self.core_head = Some(CoreHead::new(self.programs, weights, layout.rows,
                Device { library: self.library, id: self.device }, self.stream, layout)?);
        }
        self.core_admission = admission;
        Ok(())
    }
    pub(crate) fn bind_draft_core_head(&mut self) -> Result<()> {
        if let (Some(drafter), GlmfHead::K6(weights)) = (&self.drafter, &self.weights.head) {
            let rows = drafter.block().checked_mul(drafter.max_batch_sequences()).context("Flash K6 draft rows overflow")?;
            let layout = self.core_admission.as_ref().and_then(|a| a.ranks[0].draft_head.as_ref())
                .context("Flash K6 draft head scratch not admitted")?;
            self.draft_core_head = Some(CoreHead::new(self.programs, weights, rows,
                Device { library: self.library, id: self.device }, self.stream, layout)?);
        }
        Ok(())
    }
    pub(super) fn core_slot(&self, rank: usize, w: &mut Workspace<'a>, decode: bool, lane: usize) -> Result<()> {
        if let Some(Some(core)) = self.core.get(rank) { w.core = core.slot(self, rank, w.rows, decode, lane)?; }
        Ok(())
    }
    fn core_rank(&self, rank: usize) -> Result<&Rank<'a>> { self.core.get(rank).and_then(Option::as_ref).context("Flash K6 programs not bound") }
    fn core_buffer(ptr: *mut c_void, bytes: usize, device: i32) -> Buffer { Buffer { ptr, bytes, device_id: device, flags: 0 } }
    fn core_received(&self, rank: usize, slot: usize, rows: usize) -> Result<Buffer> {
        let exchange = self.exchange()?;
        let prefix = self.prefill_rows.max(self.decode_rows) * self.cfg.hidden * self.partial_bytes();
        let bytes = rows * self.cfg.hidden * 4;
        ensure!(prefix + bytes <= exchange.slot_bytes(), "Flash K6 receive suffix not admitted");
        Ok(Self::core_buffer(exchange.recv(rank, slot)?.wrapping_byte_add(prefix), bytes, exchange.device(rank)))
    }
    fn core_linear(&self, rank: usize, w: &Workspace<'_>, layer: &GlmfLayer<'_>, name: &str,
        input: Buffer, output: Buffer, rows: usize, plane: usize) -> Result<()> {
        let weights = layer.core.get(name).with_context(|| format!("Flash K6 missing {name}"))?;
        let core = self.core_rank(rank)?;
        let projection = core.projections.get(&key(weights)).context("Flash K6 unbound geometry")?;
        let slot = w.core.as_ref().context("Flash K6 row slot not admitted")?;
        self.on(rank, || {
            if weights.side == cuteafd_loader::formats::dense_trellis::TrellisSide::Input {
                let result = projection.core(rows, input, weights, &slot.scratch.projection, self.stream_of(rank))?;
                // SAFETY: retain this producer's F32 core before another projection reuses the accumulator.
                unsafe { self.library.copy_d2d_async(slot.retained[plane].buffer, result,
                    rows * self.cfg.hidden * 4, self.stream_of(rank)) }
            } else { projection.local(rows, input, weights, &slot.scratch.projection, output, self.stream_of(rank)) }
        })
    }
    fn core_stage(&self, rank: usize, split: bool, name: &str, pointers: &[(&str, *mut c_void)], scalars: &[Scalar]) -> Result<()> {
        let prefix = if split { "glmf2_stage" } else { "glmf_stage" };
        let program = self.programs.program(&format!("{prefix}_{name}"), &pointers.iter().map(|p| p.0).collect::<Vec<_>>())?;
        // SAFETY: each caller binds the preserved resident operands and admitted rank-local stage buffers.
        self.on(rank, || unsafe { program.launch(&pointers.iter().map(|p| p.1).collect::<Vec<_>>(), scalars, self.stream_of(rank)) })
    }
    fn core_publish(&self, rank: usize, w: &Workspace<'_>, at: usize, rows: usize, plane: usize) -> Result<()> {
        let slot = w.core.as_ref().context("Flash K6 retained slot")?;
        self.record_exchange(rank, "main", at, true);
        self.exchange()?.push_to(rank, at, slot.retained[plane].buffer.ptr,
            self.core_received(1 - rank, at, rows)?.ptr, rows * self.cfg.hidden * 4)
    }
    fn core_reduce(&self, rank: usize, w: &Workspace<'_>, layer: &GlmfLayer<'_>, name: &str,
        at: usize, rows: usize, plane: usize, output: Buffer) -> Result<()> {
        let weights = &layer.core[name];
        let slot = w.core.as_ref().context("Flash K6 retained slot")?;
        let local = slot.retained[plane].buffer;
        let peer = self.core_received(rank, at, rows)?;
        let (first, second) = if rank == 0 { (local, peer) } else { (peer, local) };
        let core = self.core_rank(rank)?;
        self.on(rank, || core.projections[&key(weights)].reduced(&core.add, rows, first, second,
            slot.retained[2].buffer, weights, output, self.stream_of(rank)))
    }
    pub(super) fn core_attention_meet(&self, rank: usize, w: &Workspace<'_>, layer: &GlmfLayer<'_>, at: usize,
        rows: usize) -> Result<Option<*mut c_void>> {
        if !native_attention(layer) { return Ok(None); }
        self.core_publish(rank, w, at, rows, 0)?;
        self.exchange_wait(self.exchange()?, "main", rank, at)?;
        let output = w.sum.as_ref().context("Flash K6 split sum")?.buffer;
        self.core_reduce(rank, w, layer, "self_attn.o_proj", at, rows, 0, output)?;
        Ok(Some(output.ptr))
    }
    pub(super) fn core_ffn(&self, rank: usize, w: &Workspace<'_>, layer: &GlmfLayer<'_>, cap: &str,
        output: *mut c_void, count: Scalar) -> Result<()> {
        let rows = rows(count)?;
        let slot = w.core.as_ref().context("Flash K6 FFN slot")?;
        let s = &slot.scratch;
        let (gate, up, hidden, scratch) = (s.stage("gate")?, s.stage("up")?, s.stage("hidden")?, s.stage("scratch")?);
        self.core_linear(rank, w, layer, &ffn_key(layer, "gate"), w.x.buffer, gate, rows, 1)?;
        self.core_linear(rank, w, layer, &ffn_key(layer, "up"), w.x.buffer, up, rows, 1)?;
        let inter = layer.core[&ffn_key(layer, "gate")].shape[0];
        self.core_stage(rank, layer.split, &format!("swiglu_i{inter}_{cap}"), &[("gate", gate.ptr), ("up", up.ptr),
            ("hidden", hidden.ptr), ("scratch", scratch.ptr)], &[count])?;
        let output = Self::core_buffer(output, rows * self.cfg.hidden * 2, w.x.buffer.device_id);
        let key = ffn_key(layer, "down");
        self.core_linear(rank, w, layer, &key, hidden, output, rows, 1)?;
        if layer.split {
            // The suffix is disjoint from the following BF16 routed/dense publication.
            let index = self.weights.layers.iter().position(|l| std::ptr::eq(l, layer))
                .or_else(|| self.peer.as_ref().and_then(|p| p.layers.iter().position(|l| std::ptr::eq(l, layer))))
                .context("Flash K6 FFN layer owner")?;
            let at = super::slot(index, true, slot.lane);
            self.core_publish(rank, w, at, rows, 1)?;
            self.exchange_wait(self.exchange()?, "main", rank, at)?;
            if rank == 0 { self.core_reduce(rank, w, layer, &key, at, rows, 1, output)?; }
            else {
                // SAFETY: only rank 0 contributes the fully transformed shared/dense result.
                self.on(rank, || unsafe { self.library.cuda_zero_bytes_async(output, output.bytes, self.stream_of(rank)) })?;
            }
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn core_kda(&self, rank: usize, w: &Workspace<'_>, layer: &GlmfLayer<'_>, count: Scalar,
        first: usize, cap: &str, spec: bool, [conv, state, replay]: [*mut c_void; 3]) -> Result<()> {
        let rows = rows(count)?;
        let s = &w.core.as_ref().context("Flash K6 KDA slot")?.scratch;
        let (q,k,v,out,scratch) = (s.stage("q")?,s.stage("k")?,s.stage("v")?,s.stage("values")?,s.stage("scratch")?);
        let input = Self::core_buffer(row_at(&w.x, first, self.cfg.hidden * 2), rows*self.cfg.hidden*2, w.x.buffer.device_id);
        for (name, output) in [("q",q),("k",k),("v",v)] {
            self.core_linear(rank,w,layer,&format!("self_attn.{name}_proj"),input,output,rows,0)?;
        }
        self.core_stage(rank,layer.attention_share.split(),&format!("kda_core_{cap}"),&[
            ("x",input.ptr),("q",q.ptr),("k",k.ptr),("v",v.ptr),("gate",q.ptr),("w_fab",layer.ptr("w_aux")?),
            ("w_f",layer.ptr("w_fg")?),("conv_w",layer.ptr("conv_w")?),("a_log",layer.ptr("a_log")?),
            ("dt_bias",layer.ptr("dt_bias")?),("o_norm",layer.ptr("o_norm")?),("conv_state",conv),("state",state),
            ("slots",row_at(&w.kda_slots,first,4)),("seq_first",row_at(&w.seq_first,first,4)),("out",out.ptr),
            ("replay",replay),("scratch",scratch.ptr)],&[count,Scalar::I32(i32::from(spec))])?;
        let destination = Self::core_buffer(w.delta.buffer.ptr.wrapping_byte_add(first*self.cfg.hidden*2),
            rows*self.cfg.hidden*2,w.x.buffer.device_id);
        self.core_linear(rank,w,layer,"self_attn.o_proj",out,destination,rows,0)
    }
    pub(super) fn core_mla_producer(&self, rank: usize, w: &Workspace<'_>, layer: &GlmfLayer<'_>, count: Scalar,
        cap: &str, cache: *mut c_void) -> Result<()> {
        let rows = rows(count)?;
        let s = &w.core.as_ref().context("Flash K6 MLA slot")?.scratch;
        let (q,k,scratch) = (s.stage("q")?,s.stage("k")?,s.stage("scratch")?);
        self.core_linear(rank,w,layer,"self_attn.q_a_proj",w.x.buffer,q,rows,0)?;
        self.core_linear(rank,w,layer,"self_attn.kv_a_proj_with_mqa",w.x.buffer,k,rows,0)?;
        self.core_stage(rank,layer.attention_share.split(),&format!("rank_norm_{cap}"),&[("q_a",q.ptr),("kv_a",k.ptr),
            ("q_a_norm",layer.ptr("q_a_norm")?),("kv_a_norm",layer.ptr("kv_a_norm")?),
            ("kv_slots",w.kv_slots.buffer.ptr),("q_resid",w.q_resid.buffer.ptr),("kv_cache",cache),("scratch",scratch.ptr)],&[count])?;
        self.core_linear(rank,w,layer,"self_attn.q_b_proj",w.q_resid.buffer,q,rows,0)?;
        self.core_stage(rank,layer.attention_share.split(),&format!("absorb_{cap}"),&[("a",q.ptr),
            ("w",layer.ptr("w_uk")?),("out",w.query.buffer.ptr)],&[count])
    }
    pub(super) fn core_mla_output(&self, rank: usize, w: &Workspace<'_>, layer: &GlmfLayer<'_>, count: Scalar, cap: &str) -> Result<()> {
        let rows = rows(count)?;
        let values = w.core.as_ref().context("Flash K6 MLA slot")?.scratch.stage("values")?;
        self.core_stage(rank,layer.attention_share.split(),&format!("values_{cap}"),&[("a",w.latent.buffer.ptr),
            ("w",layer.ptr("w_uv")?),("out",values.ptr)],&[count])?;
        self.core_linear(rank,w,layer,"self_attn.o_proj",values,w.delta.buffer,rows,0)
    }
}
