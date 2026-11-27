//! Native projections share the neutral K6 owners; only Flash stage wiring lives here.
use super::*;
use crate::shared::{dense_trellis::{CoreHead, CoreWeights, Projection, ProjectionLayout, ProjectionScratch, codebook, retain_partial},
    memory::{device::{Allocation, Device}, Tag}};
use cuteafd_ffi::{CuteafdDeviceBuffer as Buffer, dense_trellis::TrellisDType, Bf16F32Add, F32Bf16Narrow};
use cuteafd_loader::families::glm5_flash::core_admission::{CoreCharges, CoreRankLayout, CoreStageLayout, route_stage_names, fp8_projection_name, fp8_projection_prefix};
use std::collections::BTreeMap;

type Key = (usize, usize, usize, &'static str, usize);
fn key(w: &CoreWeights<'_>) -> Key {
    (w.shape[1], w.shape[0], if w.side == cuteafd_loader::formats::dense_trellis::TrellisSide::Input { 2 } else { 1 }, codebook(w), w.bits)
}
fn rows(value: Scalar) -> Result<usize> {
    match value { Scalar::I32(n) if n > 0 => Ok(n as usize), _ => anyhow::bail!("Flash K6 positive row count") }
}
fn ffn_key(layer: &GlmfLayer<'_>, projection: &str) -> String {
    format!("mlp.{}{projection}_proj", if layer.dense { "" } else { "shared_experts." })
}
pub(super) fn native_ffn(layer: &GlmfLayer<'_>) -> bool { layer.core.contains_key(&ffn_key(layer, "down")) || layer.attention_native.contains_key(&ffn_key(layer, "down")) }
/// Split trellis FFN down-projections carry a post-reduction `svh` Hadamard
/// egress and must reduce rank 0 + rank 1 before egress. Linear native
/// BF16/FP8 FFN down-projections (`attention_native`) have no egress transform,
/// so each rank writes its BF16 partial directly into `w.delta` / `w.shared`
/// and lets the regular FFN exchange (`meet_ffn` / `tp2_publish`) sum once.
pub(super) fn split_trellis_ffn(layer: &GlmfLayer<'_>) -> bool { layer.split && layer.core.contains_key(&ffn_key(layer, "down")) }
pub(super) fn stage_attention(layer: &GlmfLayer<'_>) -> bool {
    layer.attention_native.keys().any(|name| name.starts_with("self_attn.")) || layer.core.contains_key(match layer.attention {
        GlmNextAttention::Kda => "self_attn.q_proj", GlmNextAttention::Mla => "self_attn.q_a_proj",
    })
}
fn native_attention(layer: &GlmfLayer<'_>) -> bool {
    layer.core.get("self_attn.o_proj").is_some_and(|w|
        w.side == cuteafd_loader::formats::dense_trellis::TrellisSide::Input)
        || layer.attention_share.split() && layer.attention_native.contains_key("self_attn.o_proj")
}
fn rank_order<T>(rank: usize, local: T, peer: T) -> (T, T) {
    if rank == 0 { (local, peer) } else { (peer, local) }
}
fn retained_bytes(rows: usize, hidden: usize) -> Result<usize> {
    rows.checked_mul(hidden).and_then(|n| n.checked_mul(2)).context("Flash K6 retained extent overflow")
}

struct Scratch<'a> { projection: Option<ProjectionScratch<'a>>, rows: usize, stages: BTreeMap<&'static str, Allocation<'a>> }
impl Scratch<'_> {
    fn stage(&self, name: &str) -> Result<Buffer> { Ok(self.stages.get(name).context("Flash K6 stage")?.buffer) }
}
pub(super) struct Rank<'a> {
    projections: BTreeMap<Key, Projection<'a>>,
    bf16: cuteafd_ffi::draft_projection::DraftProjections<'a>,
    scratch: [Option<Rc<Scratch<'a>>>; 2],
    add: Bf16F32Add<'a>,
    bf16_add: cuteafd_ffi::Bf16Add<'a>,
    narrow: F32Bf16Narrow<'a>,
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
            let p = Projection::load_geometry_bits(engine.programs, k.0, k.1, k.2, k.4, k.3,
                engine.prefill_rows.max(engine.decode_rows), TrellisDType::Bf16)?;
            ProjectionLayout::new(&p.program, engine.prefill_rows.max(engine.decode_rows))?;
            projections.insert(k, p);
        }
        let allocate = |index: usize| -> Result<Option<Rc<Scratch<'a>>>> {
            let Some(stages) = &admission.stages else {
                ensure!(projections.is_empty(), "Flash native body not admitted"); return Ok(None);
            };
            let rows = if index == 0 { engine.decode_rows } else { engine.prefill_rows };
            let expected = if index == 0 { admission.decode.as_ref() } else { admission.prefill.as_ref() };
            let mut actual: Option<cuteafd_loader::formats::trellis_core::ProjectionScratchLayout> = None;
            for weights in layers.iter().flat_map(|l| l.core.values()) {
                let fused = projections[&key(weights)].program.has_fused();
                let next = cuteafd_loader::formats::trellis_core::ProjectionScratchLayout::new_with_fused(
                    weights.shape[1], weights.shape[0], rows, fused)?;
                if let Some(actual) = &mut actual { actual.merge(&next)?; } else { actual = Some(next); }
            }
            ensure!(actual.as_ref() == expected, "Flash loaded/admitted projection scratch mismatch");
            let projection = expected.map(|layout| ProjectionLayout { rows: layout.rows, extents: layout.extents.clone() }
                .allocate(device, crate::shared::memory::Tag::workspace("glmf/native-projection"))).transpose()?;
            let whole = engine.weights.layers.iter().chain(engine.peer.iter().flat_map(|p| p.layers.iter()))
                .any(|l| l.attention == GlmNextAttention::Mla && !l.attention_share.split());
            let owned = cuteafd_loader::families::glm5_flash::core_admission::owned_mla(
                &engine.cfg, layers.len(), engine.ranks(), whole);
            let mut stage_bytes = 0;
            for name in route_stage_names(&engine.cfg, engine.ranks(), owned, index == 0, engine.decode_rows) {
                stage_bytes = stage_bytes.max(engine.programs.spec(&name)?.scratch.get("scratch").copied().unwrap_or(0) as usize);
            }
            for layer in layers {
                for weight in layer.attention_native.values().filter(|w| w.layout.fp8) {
                    let prefix = fp8_projection_prefix(&weight.layout);
                    let caps = if index == 0 { vec![64, engine.decode_rows] } else { vec![4096] };
                    for cap in caps {
                        let name = fp8_projection_name(prefix, weight.layout.shape, cap);
                        stage_bytes = stage_bytes.max(engine.programs.spec(&name)?.scratch.get("scratch").copied().unwrap_or(0) as usize);
                    }
                }
            }
            let actual_stage = CoreStageLayout::new(&engine.cfg, rows, engine.ranks(), owned, stage_bytes)?;
            ensure!(stages[index] == actual_stage,
                "Flash K6 loaded/admitted stage extents mismatch");
            let stages = actual_stage.extents.iter()
                .map(|(&n, &b)| Ok((n, Allocation::new(device, b, Tag::workspace("glmf/core-stages"))?)))
                .collect::<Result<_>>()?;
            Ok(Some(Rc::new(Scratch { projection, stages, rows })))
        };
        let shapes = layers.iter().flat_map(|l| l.attention_native.values())
            .filter(|w| !w.layout.fp8).map(|w| (w.layout.shape[1], w.layout.shape[0])).collect::<Vec<_>>();
        let bf16 = cuteafd_ffi::draft_projection::DraftProjections::load(engine.programs, &shapes,
            engine.prefill_rows.max(engine.decode_rows))?;
        Ok(Self { scratch: [allocate(0)?, allocate(1)?], projections, bf16,
            add: engine.library.bf16_f32_add()?, bf16_add: engine.library.bf16_add()?, narrow: engine.library.f32_bf16_narrow()? })
    }
    fn slot(&self, engine: &GlmfEngine<'a>, rank: usize, rows: usize, decode: bool, lane: usize) -> Result<Option<Slot<'a>>> {
        let Some(scratch) = &self.scratch[usize::from(!decode)] else { return Ok(None); };
        let device = Device { library: engine.library, id: if rank == 0 { engine.device } else { engine.peer()?.device } };
        ensure!(rows == scratch.rows, "Flash K6 retained slot capacity differs from scratch");
        let bytes = retained_bytes(rows, engine.cfg.hidden)?;
        let allocate = |width| Allocation::new(device, bytes * width, Tag::workspace("glmf/core-retained"));
        Ok(Some(Slot { scratch: scratch.clone(), retained: [allocate(1)?, allocate(1)?, allocate(2)?], lane }))
    }
}

impl<'a> GlmfEngine<'a> {
    pub(crate) fn bind_core(&mut self, admission: Option<CoreCharges>) -> Result<()> {
        ensure!(self.core.is_empty() && self.core_admission.is_none(), "Flash K6 bound twice");
        let loaded = self.weights.layers.iter().chain(self.peer.iter().flat_map(|p| p.layers.iter()))
            .any(|l| !l.core.is_empty() || !l.attention_native.is_empty())
            || matches!(self.weights.head, GlmfHead::K6(_) | GlmfHead::BlockFp8 { .. });
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
            ensure!(layout.retained_rows == if layout.stages.is_none() { 0 } else {
                self.decode_rows + self.prefill_rows * self.prefill_waves }, "Flash K6 retained slots differ from admission");
            if self.ranks() == 2 {
                let prefix = self.prefill_rows.max(self.decode_rows) * self.cfg.hidden * self.partial_bytes();
                let suffix = self.exchange()?.slot_bytes().checked_sub(prefix).context("Flash K6 exchange prefix")?;
                let rows = self.prefill_rows.max(self.decode_rows);
                let expected_suffix = if admitted.ranks.iter().flat_map(|r| &r.body)
                    .any(|w| w.side == cuteafd_loader::formats::dense_trellis::TrellisSide::Input)
                    || self.weights.layers.iter().chain(self.peer.iter().flat_map(|p| p.layers.iter()))
                        .any(|l| l.attention_native.values().any(|w| w.layout.ranks > 1 && w.layout.split == cuteafd_loader::formats::trellis_core::CoreSplit::Input)) {
                    rows * self.cfg.hidden * 2
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
                Device { library: self.library, id: self.device }, self.stream, layout,
                crate::shared::memory::Tag::workspace("glmf/target-head-projection"))?);
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
                Device { library: self.library, id: self.device }, self.stream, layout,
                crate::shared::memory::Tag::drafter_workspace("glmf/draft-head-projection"))?);
        }
        // A packed head's drafter reads the coarse plane in scratch of its own (admitted beside the
        // drafter's scratch); the drafter launches it on its own stream.
        if let (Some(drafter), GlmfHead::Packed { .. }) = (&self.drafter, &self.weights.head) {
            let rows = drafter.block().checked_mul(drafter.max_batch_sequences()).context("packed draft rows overflow")?;
            let library = self.library;
            Device { library, id: self.device }.run(|| self.weights.head.bind_draft(library, rows))?;
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
        let bytes = rows * self.cfg.hidden * 2;
        ensure!(prefix + bytes <= exchange.slot_bytes(), "Flash K6 receive suffix not admitted");
        Ok(Self::core_buffer(exchange.recv(rank, slot)?.wrapping_byte_add(prefix), bytes, exchange.device(rank)))
    }
    fn core_linear(&self, rank: usize, w: &Workspace<'_>, layer: &GlmfLayer<'_>, name: &str,
        input: Buffer, output: Buffer, rows: usize, plane: usize, cap: &str) -> Result<()> {
        if let Some(weights) = layer.attention_native.get(name) {
            let [n,k] = weights.layout.shape;
            ensure!(input.bytes >= rows*k*2 && output.bytes >= rows*n*2, "Flash native projection operand extent");
            let output = if plane == 0 && weights.layout.ranks > 1 && weights.layout.split == cuteafd_loader::formats::trellis_core::CoreSplit::Input {
                ensure!(n == self.cfg.hidden, "Flash native retained projection extent");
                let retained = w.core.as_ref().context("Flash native retained slot")?.retained[plane].buffer;
                ensure!(retained.bytes >= retained_bytes(rows, self.cfg.hidden)?, "Flash native retained rows exceed slot");
                retained
            } else { output };
            return self.on(rank, || {
                if weights.layout.fp8 {
                    let prefix = fp8_projection_prefix(&weights.layout);
                    let capacity = if is_decode(cap) { if cap == "m128" { 128 } else { 64 } } else { 4096 };
                    let a8 = if name.starts_with("mlp.") { self.fp8_prefill.ffn }
                        else if layer.attention == GlmNextAttention::Kda {
                            self.fp8_prefill.kda_bits & if name.ends_with("o_proj") { 2 } else { 1 } != 0
                        } else { self.fp8_prefill.mla };
                    let name = fp8_projection_name(prefix, weights.layout.shape, capacity);
                    let program = self.programs.program(&name, &["x", "weight", "scales", "out", "scratch"])?;
                    let scratch = w.core.as_ref().context("Flash FP8 slot")?.stage("scratch")?;
                    let scales = weights.scales.as_ref().context("Flash FP8 scales")?.buffer;
                    // SAFETY: admitted native rank slices, output and stage workspace
                    // remain live on this stream throughout graph replay.
                    return unsafe { program.launch(&[input.ptr, weights.values.buffer.ptr, scales.ptr,
                        output.ptr, scratch.ptr], &self.fp8_scalars(Scalar::I32(i32::try_from(rows)?),
                            is_decode(cap), is_decode(cap) || a8), self.stream_of(rank)) };
                }
                // SAFETY: selected rank slice and retained stage buffers cover the
                // checked BF16 geometry and remain live throughout graph replay.
                unsafe { self.core_rank(rank)?.bf16.launch(input.ptr, weights.values.buffer.ptr, output.ptr,
                    rows, k, n, self.stream_of(rank))?; }
                Ok(())
            });
        }
        let weights = layer.core.get(name).with_context(|| format!("Flash K6 missing {name}"))?;
        let core = self.core_rank(rank)?;
        let projection = core.projections.get(&key(weights)).context("Flash K6 unbound geometry")?;
        let slot = w.core.as_ref().context("Flash K6 row slot not admitted")?;
        self.on(rank, || {
            if weights.side == cuteafd_loader::formats::dense_trellis::TrellisSide::Input {
                ensure!(plane < 2 && weights.shape[0] == self.cfg.hidden, "Flash K6 retained projection extent");
                let bytes = retained_bytes(rows, self.cfg.hidden)?;
                ensure!(bytes <= slot.retained[plane].buffer.bytes, "Flash K6 retained rows exceed slot");
                let result = projection.core(rows, input, weights, slot.scratch.projection.as_ref().context("Flash trellis scratch absent")?, self.stream_of(rank))?;
                retain_partial(&self.core_rank(rank)?.narrow, result, slot.retained[plane].buffer,
                    rows * self.cfg.hidden, self.stream_of(rank))
            } else { projection.local(rows, input, weights, slot.scratch.projection.as_ref().context("Flash trellis scratch absent")?, output, self.stream_of(rank)) }
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
            self.core_received(1 - rank, at, rows)?.ptr, rows * self.cfg.hidden * 2)
    }
    fn core_reduce(&self, rank: usize, w: &Workspace<'_>, layer: &GlmfLayer<'_>, name: &str,
        at: usize, rows: usize, plane: usize, output: Buffer) -> Result<()> {
        let slot = w.core.as_ref().context("Flash K6 retained slot")?;
        let local = slot.retained[plane].buffer;
        let peer = self.core_received(rank, at, rows)?;
        let (first, second) = rank_order(rank, local, peer);
        let core = self.core_rank(rank)?;
        if layer.attention_native.contains_key(name) {
            // SAFETY: both rank-ordered BF16 partials and the output are live on
            // this rank's stream after the exchange wait, through graph replay.
            return self.on(rank, || unsafe { core.bf16_add.launch(first, second, output,
                rows * self.cfg.hidden, self.stream_of(rank)) });
        }
        let weights = &layer.core[name];
        self.on(rank, || core.projections[&key(weights)].reduced(&core.add, rows, first, second,
            slot.retained[2].buffer, weights, output, self.stream_of(rank)))
    }
    pub(super) fn core_publish_attention(&self, rank: usize, w: &Workspace<'_>, layer: &GlmfLayer<'_>, at: usize,
        rows: usize) -> Result<bool> {
        if !native_attention(layer) { return Ok(false); }
        self.core_publish(rank, w, at, rows, 0)?;
        Ok(true)
    }
    pub(super) fn core_reduce_attention(&self, rank: usize, w: &Workspace<'_>, layer: &GlmfLayer<'_>, at: usize,
        rows: usize) -> Result<*mut c_void> {
        let output = w.sum.as_ref().context("Flash K6 split sum")?.buffer;
        self.core_reduce(rank, w, layer, "self_attn.o_proj", at, rows, 0, output)?;
        Ok(output.ptr)
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
    /// Rank-local portion of [`Self::core_ffn`] (`gate`, `up`, `swiglu`, `down`) with no peer exchange.
    pub(super) fn core_ffn_local(&self, rank: usize, w: &Workspace<'_>, layer: &GlmfLayer<'_>, cap: &str,
        output: *mut c_void, count: Scalar) -> Result<()> {
        let rows = rows(count)?;
        let slot = w.core.as_ref().context("Flash K6 FFN slot")?;
        let s = &slot.scratch;
        let (gate, up, hidden, scratch) = (s.stage("gate")?, s.stage("up")?, s.stage("hidden")?, s.stage("scratch")?);
        self.core_linear(rank, w, layer, &ffn_key(layer, "gate"), w.x.buffer, gate, rows, 1, cap)?;
        self.core_linear(rank, w, layer, &ffn_key(layer, "up"), w.x.buffer, up, rows, 1, cap)?;
        let gate_key = ffn_key(layer, "gate");
        let inter = layer.core.get(&gate_key).map(|w| w.shape[0])
            .or_else(|| layer.attention_native.get(&gate_key).map(|w| w.layout.shape[0]))
            .context("Flash FFN gate extent")?;
        self.core_stage(rank, layer.split, &format!("swiglu_i{inter}_{cap}"), &[("gate", gate.ptr), ("up", up.ptr),
            ("hidden", hidden.ptr), ("scratch", scratch.ptr)], &[count])?;
        let output = Self::core_buffer(output, rows * self.cfg.hidden * 2, w.x.buffer.device_id);
        let key = ffn_key(layer, "down");
        self.core_linear(rank, w, layer, &key, hidden, output, rows, 1, cap)
    }
    pub(super) fn core_publish_ffn(&self, rank: usize, w: &Workspace<'_>, index: usize, rows: usize) -> Result<()> {
        let slot = w.core.as_ref().context("Flash K6 FFN slot")?;
        let at = super::slot(index, true, slot.lane);
        // `core_received` is disjoint from `exchange.recv` (target 0); tag target 1 so
        // pipelined 1->0 `core_publish_ffn` + `tp2_publish`/`exchange_push` on `at` stay distinct.
        self.record_exchange_target(rank, "main", at, 1, true);
        self.exchange()?.push_to(rank, at, slot.retained[1].buffer.ptr,
            self.core_received(1 - rank, at, rows)?.ptr, rows * self.cfg.hidden * 2)
    }
    /// Zero Rank 1's shared/dense FFN output buffer without waiting on Rank 0:
    /// only Rank 0 contributes the reduced split-core FFN partial.
    pub(super) fn core_zero_ffn(&self, rank: usize, w: &Workspace<'_>, output: *mut c_void, rows: usize) -> Result<()> {
        ensure!(rank == 1, "core_zero_ffn is rank-1 only");
        let output = Self::core_buffer(output, rows * self.cfg.hidden * 2, w.x.buffer.device_id);
        // SAFETY: only rank 0 contributes the fully transformed shared/dense result.
        self.on(rank, || unsafe { self.library.cuda_zero_bytes_async(output, output.bytes, self.stream_of(rank)) })
    }
    /// Rank 0 consumes Rank 1's `core_publish_ffn` partial and reduces `rank 0 + rank 1` into `output`.
    pub(super) fn core_finish_ffn(&self, w: &Workspace<'_>, layer: &GlmfLayer<'_>, index: usize,
        output: *mut c_void, rows: usize) -> Result<()> {
        let slot = w.core.as_ref().context("Flash K6 FFN slot")?;
        let at = super::slot(index, true, slot.lane);
        self.exchange_wait_target(self.exchange()?, "main", 0, at, 1)?;
        let output = Self::core_buffer(output, rows * self.cfg.hidden * 2, w.x.buffer.device_id);
        let key = ffn_key(layer, "down");
        self.core_reduce(0, w, layer, &key, at, rows, 1, output)
    }
    pub(super) fn core_ffn(&self, rank: usize, w: &Workspace<'_>, layer: &GlmfLayer<'_>, cap: &str,
        output: *mut c_void, count: Scalar) -> Result<()> {
        let rows = rows(count)?;
        let slot = w.core.as_ref().context("Flash K6 FFN slot")?;
        let s = &slot.scratch;
        let (gate, up, hidden, scratch) = (s.stage("gate")?, s.stage("up")?, s.stage("hidden")?, s.stage("scratch")?);
        self.core_linear(rank, w, layer, &ffn_key(layer, "gate"), w.x.buffer, gate, rows, 1, cap)?;
        self.core_linear(rank, w, layer, &ffn_key(layer, "up"), w.x.buffer, up, rows, 1, cap)?;
        let gate_key = ffn_key(layer, "gate");
        let inter = layer.core.get(&gate_key).map(|w| w.shape[0])
            .or_else(|| layer.attention_native.get(&gate_key).map(|w| w.layout.shape[0]))
            .context("Flash FFN gate extent")?;
        self.core_stage(rank, layer.split, &format!("swiglu_i{inter}_{cap}"), &[("gate", gate.ptr), ("up", up.ptr),
            ("hidden", hidden.ptr), ("scratch", scratch.ptr)], &[count])?;
        let output = Self::core_buffer(output, rows * self.cfg.hidden * 2, w.x.buffer.device_id);
        let key = ffn_key(layer, "down");
        self.core_linear(rank, w, layer, &key, hidden, output, rows, 1, cap)?;
        if split_trellis_ffn(layer) {
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
            self.core_linear(rank,w,layer,&format!("self_attn.{name}_proj"),input,output,rows,0,cap)?;
        }
        self.core_stage(rank,layer.attention_share.split(),&format!("kda_core_{cap}"),&[
            ("x",input.ptr),("q",q.ptr),("k",k.ptr),("v",v.ptr),("gate",q.ptr),("w_fab",layer.ptr("w_aux")?),
            ("w_f",layer.ptr("w_fg")?),("conv_w",layer.ptr("conv_w")?),("a_log",layer.ptr("a_log")?),
            ("dt_bias",layer.ptr("dt_bias")?),("o_norm",layer.ptr("o_norm")?),("conv_state",conv),("state",state),
            ("slots",row_at(&w.kda_slots,first,4)),("seq_first",row_at(&w.seq_first,first,4)),("out",out.ptr),
            ("replay",replay),("scratch",scratch.ptr)],&[count,Scalar::I32(i32::from(spec))])?;
        let destination = Self::core_buffer(w.delta.buffer.ptr.wrapping_byte_add(first*self.cfg.hidden*2),
            rows*self.cfg.hidden*2,w.x.buffer.device_id);
        self.core_linear(rank,w,layer,"self_attn.o_proj",out,destination,rows,0,cap)
    }
    pub(super) fn core_mla_producer(&self, rank: usize, w: &Workspace<'_>, layer: &GlmfLayer<'_>, count: Scalar,
        cap: &str, cache: *mut c_void) -> Result<()> {
        let rows = rows(count)?;
        let s = &w.core.as_ref().context("Flash K6 MLA slot")?.scratch;
        let (q,k,scratch) = (s.stage("q")?,s.stage("k")?,s.stage("scratch")?);
        self.core_linear(rank,w,layer,"self_attn.q_a_proj",w.x.buffer,q,rows,0,cap)?;
        self.core_linear(rank,w,layer,"self_attn.kv_a_proj_with_mqa",w.x.buffer,k,rows,0,cap)?;
        self.core_stage(rank,layer.attention_share.split(),&format!("rank_norm_{cap}"),&[("q_a",q.ptr),("kv_a",k.ptr),
            ("q_a_norm",layer.ptr("q_a_norm")?),("kv_a_norm",layer.ptr("kv_a_norm")?),
            ("kv_slots",w.kv_slots.buffer.ptr),("q_resid",w.q_resid.buffer.ptr),("kv_cache",cache),("scratch",scratch.ptr)],&[count])?;
        self.core_linear(rank,w,layer,"self_attn.q_b_proj",w.q_resid.buffer,q,rows,0,cap)?;
        self.core_stage(rank,layer.attention_share.split(),&format!("absorb_{cap}"),&[("a",q.ptr),
            ("w",layer.ptr("w_uk")?),("out",w.query.buffer.ptr)],&[count])
    }
    pub(super) fn core_mla_output(&self, rank: usize, w: &Workspace<'_>, layer: &GlmfLayer<'_>, count: Scalar, cap: &str) -> Result<()> {
        let rows = rows(count)?;
        let values = w.core.as_ref().context("Flash K6 MLA slot")?.scratch.stage("values")?;
        self.core_stage(rank,layer.attention_share.split(),&format!("values_{cap}"),&[("a",w.latent.buffer.ptr),
            ("w",layer.ptr("w_uv")?),("out",values.ptr)],&[count])?;
        self.core_linear(rank,w,layer,"self_attn.o_proj",values,w.delta.buffer,rows,0,cap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let reference = transform(first.iter().zip(&second).map(|(a,b)| narrow(*a) + narrow(*b)).collect());
        for rank in 0..2 {
            let (a,b) = if rank == 0 { (&first,&second) } else { (&second,&first) };
            let (a,b) = rank_order(rank,a,b);
            assert!(std::ptr::eq(a,&first) && std::ptr::eq(b,&second));
            assert_eq!(transform(a.iter().zip(b).map(|(a,b)| narrow(*a) + narrow(*b)).collect()), reference);
        }
        let lossy = transform(first).iter().zip(transform(second)).map(|(a,b)| narrow(a+b)).collect::<Vec<_>>();
        assert_ne!(lossy, reference, "fixture must detect premature BF16 egress");
        assert_eq!(retained_bytes(4096,4096)?, 4096*4096*2);
        assert!(retained_bytes(usize::MAX,4096).is_err());
        Ok(())
    }

    #[test]
    fn native_slots_retain_both_producers_and_one_shared_contribution() {
        let source = include_str!("core.rs");
        let allocate = source.split("fn slot(&self").nth(1).unwrap().split("impl<'a> GlmfEngine").next().unwrap();
        assert!(allocate.contains("retained: [allocate(1)?, allocate(1)?, allocate(2)?]"));
        let linear = source.split("fn core_linear(").nth(1).unwrap().split("fn core_stage(").next().unwrap();
        assert!(linear.find("projection.core(").unwrap() < linear.find("retain_partial(").unwrap());
        assert!(linear.contains("slot.retained[plane].buffer"));
        let reduce = source.split("fn core_reduce(").nth(1).unwrap().split("fn core_attention_meet(").next().unwrap();
        assert!(reduce.contains("rank_order(rank, local, peer)") && reduce.contains("slot.retained[2].buffer"));
        let ffn = source.split("fn core_ffn(").nth(1).unwrap().split("fn core_kda(").next().unwrap();
        assert!(ffn.contains("if rank == 0 { self.core_reduce") && ffn.contains("cuda_zero_bytes_async"));
        assert!(ffn.find("self.core_publish(").unwrap() < ffn.find("self.exchange_wait(").unwrap());
    }

    #[test]
    fn linear_native_ffn_skips_pre_egress_core_exchange() {
        let source = include_str!("core.rs");
        let linear = source.split("fn core_linear(").nth(1).unwrap().split("fn core_stage(").next().unwrap();
        assert!(linear.contains("if plane == 0 && weights.layout.ranks > 1"),
            "attention_native FFN down_proj must write directly to output instead of retained[1]");
        let ffn = source.split("fn core_ffn(").nth(1).unwrap().split("fn core_kda(").next().unwrap();
        assert!(ffn.contains("if split_trellis_ffn(layer)"),
            "only trellis core FFN down_proj needs pre-egress core exchange");
        let local_graph = include_str!("local_graph.rs");
        assert!(local_graph.contains("let split_core_ffn = core::split_trellis_ffn(layer0);"),
            "local decode graph must skip core_publish_ffn for linear attention_native FFNs");
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

    #[test]
    fn runtime_dense_core_selects_mul1_and_mcg_programs_by_weight_codebook() -> Result<()> {
        use cuteafd_loader::formats::{dense_trellis::TrellisSide, exl3_storage::Exl3Codebook};
        use serde_json::json;

        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.configure_memory(&library, 0, 1 << 30)?;
        fixture.immediate_copies(&library)?;
        let device = Device { library: &library, id: 0 };

        let (k, n, side, rows) = (128usize, 256usize, 1usize, 64usize);
        let mut entries = Vec::new();
        let mut policies = Vec::new();
        for bits in [6usize, 8] {
            for book in ["mcg", "mul1"] {
                let shape = format!("glmf_test_k{bits}_{book}");
                let routes = [format!("input_{shape}"), format!("core_{shape}"), format!("output_{shape}")];
                policies.push(json!({"shape": shape, "input_features": k, "output_features": n,
                    "sharded_side": side, "bits": bits, "codebook": book,
                    "native_input": routes[0], "native_program": routes[1], "output_program": routes[2]}));
                let geometries = [
                    json!({"features": k, "input_dtype": "bf16", "max_rows": 4096}),
                    json!({"abi_version": 1, "input_features": k, "output_features": n, "bits": bits,
                        "codebook": book, "sharded_side": side, "rotation_count": 1, "max_rows": 4096,
                        "split_k_blocks": 8, "splits": 1, "ctas_per_sm": 4}),
                    json!({"features": n, "output_dtype": "bf16", "max_rows": 4096}),
                ];
                let pointers = [
                    vec!["input", "suh", "rotated", "input_block_scales"],
                    vec!["rotated_inputs", "payload", "input_block_scales", "partial", "output"],
                    vec!["input", "svh", "output"],
                ];
                for i in 0..3 {
                    entries.push(json!({"name": format!("dense_trellis_k6_{}", routes[i]), "capacity_rows": 4096,
                        "geometry": geometries[i],
                        "pointers": pointers[i].iter().map(|p| json!({"name": p})).collect::<Vec<_>>(),
                        "scalars": vec![json!({"type": "int32"}); if i == 1 { 3 } else { 2 }],
                        "scratch_bytes_at_capacity": if i == 1 { json!({"partial": 4096 * n * 4}) } else { json!({}) }}));
                }
            }
        }
        fixture.register_programs(&library, &entries)?;
        let dir = tempfile::tempdir()?;
        let write_manifest = |name: &str, keep: Option<&str>| -> Result<std::path::PathBuf> {
            let path = dir.path().join(name);
            let filtered: Vec<_> = policies.iter()
                .filter(|p| keep.is_none_or(|b| p["codebook"].as_str() == Some(b)))
                .cloned().collect();
            std::fs::write(&path, serde_json::to_vec(&json!({
                "families": {"dense_trellis_k6": {"abi_version": 1, "row_policies": filtered}},
                "programs": entries,
            }))?)?;
            Ok(path)
        };
        let dual_programs = library.programs()?.with_manifest(&write_manifest("dual.json", None)?)?;
        let mcg_only = library.programs()?.with_manifest(&write_manifest("mcg.json", Some("mcg"))?)?;

        let make_weights = |bits: usize, book: Exl3Codebook| -> Result<CoreWeights<'_>> {
            Ok(CoreWeights {
                payload: Allocation::new(device, n * k * bits / 8, crate::shared::memory::TEST)?,
                suh: Allocation::new(device, k * 2, crate::shared::memory::TEST)?,
                svh: Allocation::new(device, n * 2, crate::shared::memory::TEST)?,
                shape: [n, k], side: TrellisSide::Output, bits, codebook: book,
            })
        };

        for bits in [6usize, 8] {
            let mul1_w = make_weights(bits, Exl3Codebook::Mul1)?;
            let mcg_w = make_weights(bits, Exl3Codebook::Mcg)?;
            let mul1_key = key(&mul1_w);
            let mcg_key = key(&mcg_w);
            assert_eq!(mul1_key.3, "mul1");
            assert_eq!(mcg_key.3, "mcg");
            assert_ne!(mul1_key, mcg_key);

            let mul1_proj = Projection::load_geometry_bits(&dual_programs,
                mul1_key.0, mul1_key.1, mul1_key.2, mul1_key.4, mul1_key.3, rows, TrellisDType::Bf16)?;
            let mcg_proj = Projection::load_geometry_bits(&dual_programs,
                mcg_key.0, mcg_key.1, mcg_key.2, mcg_key.4, mcg_key.3, rows, TrellisDType::Bf16)?;

            assert_eq!(mul1_proj.program.policy().codebook, "mul1");
            assert_eq!(mul1_proj.program.policy().native, format!("dense_trellis_k6_core_glmf_test_k{bits}_mul1"));
            assert_eq!(mcg_proj.program.policy().codebook, "mcg");
            assert_eq!(mcg_proj.program.policy().native, format!("dense_trellis_k6_core_glmf_test_k{bits}_mcg"));

            // Mismatched weight codebook must be rejected at launch time on both core() and core_into().
            let scratch = ProjectionLayout::new(&mul1_proj.program, rows)?
                .allocate(device, crate::shared::memory::TEST)?;
            let input = Allocation::new(device, rows * k * 2, crate::shared::memory::TEST)?;
            let accum = Allocation::new(device, rows * n * 4, crate::shared::memory::TEST)?;
            assert!(mul1_proj.core(1, input.buffer, &mcg_w, &scratch, std::ptr::null_mut()).is_err());
            assert!(mcg_proj.core(1, input.buffer, &mul1_w, &scratch, std::ptr::null_mut()).is_err());
            assert!(mul1_proj.core_into(1, input.buffer, &mcg_w, &scratch, accum.buffer, std::ptr::null_mut()).is_err());
            assert!(mcg_proj.core_into(1, input.buffer, &mul1_w, &scratch, accum.buffer, std::ptr::null_mut()).is_err());

            // Missing mul1 policy in PROGRAMS.json must fail closed, never fall back to mcg.
            assert!(Projection::load_geometry_bits(&mcg_only,
                mul1_key.0, mul1_key.1, mul1_key.2, mul1_key.4, mul1_key.3, rows, TrellisDType::Bf16).is_err());
        }
        Ok(())
    }
}
