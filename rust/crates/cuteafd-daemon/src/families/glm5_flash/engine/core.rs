//! Native projections share the neutral K6 owners; only Flash stage wiring lives here.
use super::*;
use crate::shared::{dense_trellis::{CoreHead, CoreWeights, Projection, ProjectionLayout, ProjectionScratch, codebook},
    memory::device::{Allocation, Device}};
use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer, dense_trellis::TrellisDType, F32Add};
use cuteafd_loader::families::glm5_flash::core_admission::{CoreCharges, CoreRankLayout, CoreStageLayout, route_stage_names};
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
fn native_attention(layer: &GlmfLayer<'_>) -> bool {
    layer.core.get("self_attn.o_proj").is_some_and(|w|
        w.side == cuteafd_loader::formats::dense_trellis::TrellisSide::Input)
}
fn rank_order<T>(rank: usize, local: T, peer: T) -> (T, T) {
    if rank == 0 { (local, peer) } else { (peer, local) }
}
fn retained_bytes(rows: usize, hidden: usize) -> Result<usize> {
    rows.checked_mul(hidden).and_then(|n| n.checked_mul(4)).context("Flash K6 retained extent overflow")
}

fn diagnostic_partial_views(narrow: Buffer, rounded: Buffer, count: usize) -> Result<(Buffer, [Buffer; 2])> {
    let f32_bytes = count.checked_mul(4).filter(|&n| n > 0).context("BF16 partial extent overflow")?;
    ensure!(narrow.device_id == rounded.device_id, "BF16 partial scratch device mismatch");
    let view = crate::shared::prefix::view;
    Ok((view(narrow, 0, f32_bytes / 2)?,
        [view(rounded, 0, f32_bytes)?, view(rounded, f32_bytes, f32_bytes)?]))
}

struct Scratch<'a> { projection: ProjectionScratch<'a>, stages: BTreeMap<&'static str, Allocation<'a>> }
impl Scratch<'_> {
    fn stage(&self, name: &str) -> Result<Buffer> { Ok(self.stages.get(name).context("Flash K6 stage")?.buffer) }
}
pub(super) struct Rank<'a> {
    projections: BTreeMap<Key, Projection<'a>>,
    scratch: [Option<Rc<Scratch<'a>>>; 2],
    add: F32Add<'a>,
    bf16_partials: bool,
}
pub(super) struct Slot<'a> {
    scratch: Rc<Scratch<'a>>,
    retained: [Allocation<'a>; 3],
    lane: usize,
}
impl Slot<'_> {
    pub(crate) fn stage(&self, name: &str) -> Result<Buffer> { self.scratch.stage(name) }
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
            let whole = engine.weights.layers.iter().chain(engine.peer.iter().flat_map(|p| p.layers.iter()))
                .any(|l| l.attention == GlmNextAttention::Mla && !l.attention_share.split());
            let owned = cuteafd_loader::families::glm5_flash::core_admission::owned_mla(
                &engine.cfg, layers.len(), engine.ranks(), whole);
            let mut stage_bytes = 0;
            for name in route_stage_names(&engine.cfg, engine.ranks(), owned, index == 0, engine.decode_rows) {
                stage_bytes = stage_bytes.max(engine.programs.spec(&name)?.scratch.get("scratch").copied().unwrap_or(0) as usize);
            }
            let actual_stage = CoreStageLayout::new(&engine.cfg, expected.rows, engine.ranks(), owned, stage_bytes)?;
            ensure!(admission.stages.as_ref().context("Flash K6 stages not admitted")?[index] == actual_stage,
                "Flash K6 loaded/admitted stage extents mismatch");
            let stages = actual_stage.extents.iter()
                .map(|(&n, &b)| Ok((n, Allocation::new(device, b)?))).collect::<Result<_>>()?;
            Ok(Some(Rc::new(Scratch { projection, stages })))
        };
        let bf16_partials = std::env::var("CUTEAFD_DIAG_K6_BF16_PARTIALS").is_ok_and(|v| v == "1");
        tracing::info!(rank, bf16_partials, "K6 partial reduction diagnostic (measurement only)");
        Ok(Self { scratch: [allocate(0)?, allocate(1)?], projections, add: engine.library.f32_add()?, bf16_partials })
    }
    fn slot(&self, engine: &GlmfEngine<'a>, rank: usize, rows: usize, decode: bool, lane: usize) -> Result<Option<Slot<'a>>> {
        let Some(scratch) = &self.scratch[usize::from(!decode)] else { return Ok(None); };
        let device = Device { library: engine.library, id: if rank == 0 { engine.device } else { engine.peer()?.device } };
        ensure!(rows == scratch.projection.rows, "Flash K6 retained slot capacity differs from scratch");
        let bytes = retained_bytes(rows, engine.cfg.hidden)?;
        let allocate = || Allocation::new(device, bytes);
        Ok(Some(Slot { scratch: scratch.clone(), retained: [allocate()?, allocate()?, allocate()?], lane }))
    }
}

impl<'a> GlmfEngine<'a> {
    pub(crate) fn bind_core(&mut self, admission: Option<CoreCharges>) -> Result<()> {
        ensure!(self.core.is_empty() && self.core_admission.is_none(), "Flash K6 bound twice");
        let loaded = self.weights.layers.iter().chain(self.peer.iter().flat_map(|p| p.layers.iter()))
            .any(|l| !l.core.is_empty()) || matches!(self.weights.head, GlmfHead::K6(_));
        if !loaded { ensure!(admission.is_none(), "Flash K6 admission without weights"); return Ok(()); }
        if self.weights.layers.iter().any(|l| l.core.contains_key("self_attn.q_proj")) {
            ensure!(self.kda_state == KdaState::F32 && !self.kda_output_shard,
                "native Flash K6 requires F32 recurrent stages and core-reduced output projection");
        }
        let admitted = admission.as_ref().context("Flash K6 requires exact scratch admission")?;
        ensure!(admitted.ranks.len() == self.ranks(), "Flash K6 admitted rank count");
        for name in &admitted.programs { self.programs.spec(name)?; }
        for rank in 0..self.ranks() {
            self.on(rank, || self.programs.load_matching(|name| admitted.programs.iter().any(|n| n == name)))?;
            let layers = if rank == 0 { &self.weights.layers } else { &self.peer()?.layers };
            let layout = &admitted.ranks[rank];
            ensure!(layout.retained_rows == if layout.body.is_empty() { 0 } else {
                self.decode_rows + self.prefill_rows * self.prefill_waves }, "Flash K6 retained slots differ from admission");
            if self.ranks() == 2 {
                let prefix = self.prefill_rows.max(self.decode_rows) * self.cfg.hidden * self.partial_bytes();
                let suffix = self.exchange()?.slot_bytes().checked_sub(prefix).context("Flash K6 exchange prefix")?;
                let rows = self.prefill_rows.max(self.decode_rows);
                let expected_suffix = if admitted.ranks.iter().flat_map(|r| &r.body)
                    .any(|w| w.side == cuteafd_loader::formats::dense_trellis::TrellisSide::Input) {
                    rows * self.cfg.hidden * 4
                } else { 0 };
                ensure!(suffix == expected_suffix, "Flash K6 exchange slot suffix differs from admission");
                let allocated = if suffix == 0 { 0 } else {
                    cuteafd_loader::families::glm5_flash::core_admission::receive_suffix_bytes(
                        rows, self.cfg.hidden, self.prefill_waves, self.kda_output_shard)?
                };
                ensure!(layout.receive_suffix == allocated, "Flash K6 receive suffix differs from admission");
            } else { ensure!(layout.receive_suffix == 0, "single-rank Flash K6 has no receive suffix"); }
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
                ensure!(plane < 2 && weights.shape[0] == self.cfg.hidden, "Flash K6 retained projection extent");
                let bytes = retained_bytes(rows, self.cfg.hidden)?;
                ensure!(bytes <= slot.retained[plane].buffer.bytes, "Flash K6 retained rows exceed slot");
                let result = projection.core(rows, input, weights, &slot.scratch.projection, self.stream_of(rank))?;
                // SAFETY: retain this producer's F32 core before another projection reuses the accumulator.
                unsafe { self.library.copy_d2d_async(slot.retained[plane].buffer, result,
                    bytes, self.stream_of(rank)) }
            } else { projection.local(rows, input, weights, &slot.scratch.projection, output, self.stream_of(rank)) }
        })
    }
    fn core_stage(&self, rank: usize, split: bool, name: &str, pointers: &[(&str, *mut c_void)], scalars: &[Scalar]) -> Result<()> {
        let prefix = if split { "glmf2_stage" } else { "glmf_stage" };
        self.on(rank, || {
            let program = self.programs.program(&format!("{prefix}_{name}"), &pointers.iter().map(|p| p.0).collect::<Vec<_>>())?;
            // SAFETY: each caller binds preserved operands and admitted buffers on the program's current rank.
            unsafe { program.launch(&pointers.iter().map(|p| p.1).collect::<Vec<_>>(), scalars, self.stream_of(rank)) }
        })
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
        let (first, second) = rank_order(rank, local, peer);
        let core = self.core_rank(rank)?;
        self.on(rank, || {
            let stream = self.stream_of(rank);
            let (first, second) = if core.bf16_partials {
                let count = rows.checked_mul(self.cfg.hidden).context("BF16 partial count overflow")?;
                let scratch = &slot.scratch.projection.buffers;
                let (narrow, rounded) = diagnostic_partial_views(scratch["rotated"], scratch["accumulator"], count)?;
                // Private serialized projection scratch is dead after the retained core copy.
                // Never modify the published rank partial while its peer may still read it.
                for (source, destination) in [(first, rounded[0]), (second, rounded[1])] {
                    // SAFETY: checked disjoint views on this rank's stream, kept alive
                    // through graph replay; FIFO orders narrowing before exact widening.
                    unsafe {
                        self.library.load_narrow_f32_bf16(source.ptr, narrow.ptr, count, stream)?;
                        self.library.load_widen_bf16_f32(narrow.ptr, destination.ptr, count, stream)?;
                    }
                }
                (rounded[0], rounded[1])
            } else { (first, second) };
            core.projections[&key(weights)].reduced(&core.add, rows, first, second,
                slot.retained[2].buffer, weights, output, stream)
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_rounds_both_partials_before_the_single_existing_reduce() -> Result<()> {
        let narrow = |x: f32| {
            let bits = x.to_bits();
            f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
        };
        let (a, b) = (1.00390625_f32, -1.0_f32);
        assert_eq!(narrow(a) + narrow(b), 0.0);
        assert_ne!(narrow(a) + narrow(b), narrow(a + b));
        assert_eq!(narrow(f32::from_bits(0x3f818000)), f32::from_bits(0x3f820000));
        let buffer = |ptr, bytes| Buffer { ptr: std::ptr::without_provenance_mut(ptr), bytes, device_id: 0, flags: 0 };
        let (n, r) = diagnostic_partial_views(buffer(0x1000, 32), buffer(0x2000, 128), 16)?;
        assert_eq!(n.bytes, 32);
        assert_eq!(r[0].bytes, 64);
        assert_eq!(r[1].ptr as usize, 0x2040);
        assert!(diagnostic_partial_views(n, buffer(0x2000, 127), 16).is_err());
        assert!(diagnostic_partial_views(n, r[0], usize::MAX).is_err());
        let source = include_str!("core.rs");
        let reduce = source.split("fn core_reduce(").nth(1).unwrap().split("fn core_attention_meet(").next().unwrap();
        assert!(reduce.contains("[(first, rounded[0]), (second, rounded[1])]"));
        assert!(reduce.find("load_narrow_f32_bf16").unwrap() < reduce.find("load_widen_bf16_f32").unwrap());
        assert!(reduce.find("load_widen_bf16_f32").unwrap() < reduce.find(".reduced(").unwrap());
        assert_eq!(reduce.matches(".reduced(").count(), 1);
        Ok(())
    }

    #[test]
    fn native_tp2_cores_reduce_in_rank_order_before_one_egress() -> Result<()> {
        let narrow = |x: f32| {
            let bits = x.to_bits();
            f32::from_bits(bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff0000)
        };
        let transform = |mut x: Vec<f32>| {
            for width in [1, 2, 4, 8, 16, 32, 64] {
                for block in (0..128).step_by(width * 2) {
                    for offset in 0..width {
                        let (a, b) = (x[block + offset], x[block + offset + width]);
                        x[block + offset] = a + b;
                        x[block + offset + width] = a - b;
                    }
                }
            }
            x.into_iter().map(|x| narrow(x / 16.0)).collect::<Vec<_>>()
        };
        let first = (0..128).map(|n| ((n * 37 % 257) as f32 - 128.0) / 29.0).collect::<Vec<_>>();
        let second = (0..128).map(|n| ((n * 61 % 251) as f32 - 125.0) / 31.0).collect::<Vec<_>>();
        let reference = transform(first.iter().zip(&second).map(|(a,b)| a + b).collect());
        for rank in 0..2 {
            let (a,b) = if rank == 0 { (&first,&second) } else { (&second,&first) };
            let (a,b) = rank_order(rank,a,b);
            assert!(std::ptr::eq(a,&first) && std::ptr::eq(b,&second));
            assert_eq!(transform(a.iter().zip(b).map(|(a,b)| a + b).collect()), reference);
        }
        let lossy = transform(first).iter().zip(transform(second)).map(|(a,b)| narrow(a+b)).collect::<Vec<_>>();
        assert_ne!(lossy, reference, "fixture must detect premature BF16 egress");
        assert_eq!(retained_bytes(4096,4096)?, 4096*4096*4);
        assert!(retained_bytes(usize::MAX,4096).is_err());
        Ok(())
    }

    #[test]
    fn native_slots_retain_both_producers_and_one_shared_contribution() {
        let source = include_str!("core.rs");
        let allocate = source.split("fn slot(&self").nth(1).unwrap().split("impl<'a> GlmfEngine").next().unwrap();
        assert!(allocate.contains("retained: [allocate()?, allocate()?, allocate()?]"));
        let linear = source.split("fn core_linear(").nth(1).unwrap().split("fn core_stage(").next().unwrap();
        assert!(linear.find("projection.core(").unwrap() < linear.find("copy_d2d_async").unwrap());
        assert!(linear.contains("slot.retained[plane].buffer"));
        let reduce = source.split("fn core_reduce(").nth(1).unwrap().split("fn core_attention_meet(").next().unwrap();
        assert!(reduce.contains("rank_order(rank, local, peer)") && reduce.contains("slot.retained[2].buffer"));
        let ffn = source.split("fn core_ffn(").nth(1).unwrap().split("fn core_kda(").next().unwrap();
        assert!(ffn.contains("if rank == 0 { self.core_reduce") && ffn.contains("cuda_zero_bytes_async"));
        assert!(ffn.find("self.core_publish(").unwrap() < ffn.find("self.exchange_wait(").unwrap());
    }

    #[test]
    fn priming_enqueues_both_native_peers_before_synchronizing() {
        let source = include_str!("../engine.rs");
        let prime = source.split("fn prime_tp2(").nth(1).unwrap().split("fn check_tp2_routes(").next().unwrap();
        let loops = prime.split("for rank in 0..2 {").collect::<Vec<_>>();
        assert_eq!(loops.len(), 4);
        assert!(loops[2].contains("self.ffn_on("));
        assert!(!loops[2].contains("cuda_stream_synchronize"));
        assert!(loops[3].contains("cuda_stream_synchronize"));
    }

    #[test]
    fn target_and_drafter_bind_independent_scratch_owners() -> Result<()> {
        let stream = std::ptr::without_provenance_mut::<c_void>(1);
        let foreign = std::ptr::without_provenance_mut::<c_void>(2);
        let mut calls = 0;
        for rows in [1,8,64] {
            crate::shared::dense_trellis::head_launch(64,stream,rows,stream,|| { calls += 1; Ok(()) })?;
        }
        for (rows, actual) in [(0,stream),(65,stream),(1,foreign)] {
            assert!(crate::shared::dense_trellis::head_launch(64,stream,rows,actual,|| -> Result<()> {
                panic!("invalid head call reached reusable scratch")
            }).is_err());
        }
        assert_eq!(calls,3);
        let source = include_str!("core.rs");
        let bind = source.split("fn bind_draft_core_head(").nth(1).unwrap().split("fn core_slot(").next().unwrap();
        assert!(bind.contains("self.draft_core_head = Some(CoreHead::new("));
        assert!(bind.contains("draft_head.as_ref()"));
        Ok(())
    }
}
