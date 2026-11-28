//! Family arithmetic over caller-owned native producers and expert dispatch.
use super::{experts::Dispatcher, owners::UnitBuffers, weights::{ProjectionWeight, TargetWeights}};
use crate::shared::{dense_trellis::{Projection, ProjectionLayout, ProjectionScratch},
    memory::device::{Allocation, Stream}};
use cuteafd_core::memory_layout::Tag;
use cuteafd_ffi::{dense_trellis::{DenseTrellisDispatch, DenseTrellisPolicy, TrellisDType},
    native_attention::Fp8Projections};
use cuteafd_loader::families::inkling2::weight_layout::{CoordinatorWeights, HEAD};
use std::rc::Rc;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{draft_projection::{DraftProjections, ProjectionOutput}, inkling2::{Attention, Kind, Route},
    inkling2::producers::{Op, Precision, Producer}, programs::Programs, CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::families::inkling2::{execution::ResidualSite, Inkling2Config};
use std::collections::BTreeMap;

pub(crate) struct Kernels<'a> {
    pub projections: DraftProjections<'a>,
    pub local: Attention<'a>,
    pub global: Attention<'a>,
    local_decode: Attention<'a>,
    global_decode: Attention<'a>,
    kv_format: cuteafd_loader::families::inkling2::InklingKvCache,
    head_parts: usize,
    producers: BTreeMap<String, Producer<'a>>,
    strided: BTreeMap<(usize, bool), cuteafd_ffi::fir_strided::Convolution<'a>>,
    replay: BTreeMap<(usize, bool), cuteafd_ffi::fir_replay::Convolution<'a>>,
    pub add: cuteafd_ffi::F32Add<'a>,
    pub narrow: cuteafd_ffi::F32Bf16Narrow<'a>,
    copy: cuteafd_ffi::V41PeerCopy<'a>,
    widen_add: cuteafd_ffi::Bf16F32Add<'a>,
    quant: cuteafd_ffi::programs::Program<'a>,
    sms: usize,
    pub(super) draft_head: Option<DraftProjections<'a>>,
    compact: Option<CompactProjections<'a>>,
}
type TrellisGeometry = (usize, usize, usize, usize, &'static str);

struct CompactProjections<'a> {
    stream: Rc<Stream<'a>>,
    trellis: BTreeMap<TrellisGeometry, Projection<'a>>,
    trellis_scratch: Option<ProjectionScratch<'a>>,
    fp8_decode: Option<Fp8Projections<'a>>,
    fp8_prefill: Option<Fp8Projections<'a>>,
    fp8_scratch: Option<Allocation<'a>>,
}

fn trellis_geometry(shape: [usize; 2], side: cuteafd_loader::formats::dense_trellis::TrellisSide,
    bits: usize, codebook: cuteafd_loader::formats::exl3_storage::Exl3Codebook) -> TrellisGeometry {
    use cuteafd_loader::formats::{dense_trellis::TrellisSide, exl3_storage::Exl3Codebook};
    (shape[1], shape[0], match side { TrellisSide::Output => 1, TrellisSide::Input => 2 }, bits,
        match codebook { Exl3Codebook::Mcg => "mcg", Exl3Codebook::Mul1 => "mul1" })
}

impl<'a> CompactProjections<'a> {
    fn bind(programs: &'a Programs<'a>, stream: Rc<Stream<'a>>, target: &CoordinatorWeights,
        rows: usize, head_rows: usize, fp8_bytes: usize) -> Result<Self> {
        let device = stream.device;
        let mut trellis = BTreeMap::new();
        let mut prefill = Vec::new();
        let mut decode = Vec::new();
        for l in target.core.values() {
            let key @ (k, n, side, bits, book) = trellis_geometry(l.shape, l.side, l.bits, l.codebook);
            if trellis.contains_key(&key) { continue; }
            let live = if l.stem == HEAD { head_rows.min(64) } else { rows };
            let projection = device.run(|| {
                let mut policy = DenseTrellisPolicy::from_package_geometry_bits(programs, k, n, side, bits, book)?;
                // The recurrent drafter shares the geometry but requests BF16 egress.
                policy.output_program = format!("dense_trellis_k6_trellis_output_n{n}_f32");
                Ok(Projection { program: DenseTrellisDispatch::load(programs, policy, live, TrellisDType::F32)? })
            }).with_context(|| format!("Inkling trellis program for {}", l.stem))?;
            trellis.insert(key, projection);
        }
        for l in target.fp8.values() {
            let [n, k] = l.shape;
            decode.push((k, n));
            if l.stem != HEAD { prefill.push((k, n)); }
        }
        let bind_fp8 = |shapes: &[(usize, usize)], live| -> Result<Option<Fp8Projections<'a>>> {
            if shapes.is_empty() { return Ok(None); }
            let programs = device.run(|| Ok(Fp8Projections::load(programs, shapes, live, ProjectionOutput::F32)?))?;
            for &(k, n) in shapes {
                ensure!(programs.scratch_bytes(k, n)? <= fp8_bytes, "Inkling FP8 scratch differs from admission");
            }
            Ok(Some(programs))
        };
        let fp8_decode = bind_fp8(&decode, 64)?;
        let fp8_prefill = if rows > 64 { bind_fp8(&prefill, rows)? } else { None };
        let fp8_scratch = (fp8_bytes > 0).then(|| Allocation::new(device, fp8_bytes,
            Tag::workspace("inkling2/target/fp8"))).transpose()?;
        let trellis_scratch = target.trellis_scratch(rows, head_rows.min(64))?.map(|l|
            ProjectionLayout { extents: l.extents, rows: l.rows }.allocate_packed(device,
                Tag::workspace("inkling2/target/trellis"))).transpose()?;
        Ok(Self { stream, trellis, trellis_scratch, fp8_decode, fp8_prefill, fp8_scratch })
    }
}

impl<'a> Kernels<'a> {
    pub fn bind(programs: &'a Programs<'a>, kv_format: cuteafd_loader::families::inkling2::InklingKvCache) -> Result<Self> {
        Self::bind_rank(programs, kv_format, 1)
    }
    pub fn bind_rank(programs: &'a Programs<'a>, kv_format: cuteafd_loader::families::inkling2::InklingKvCache,
        head_parts: usize) -> Result<Self> {
        Self::bind_rank_rows(programs, kv_format, head_parts, 64)
    }
    pub fn bind_rank_rows(programs: &'a Programs<'a>, kv_format: cuteafd_loader::families::inkling2::InklingKvCache,
        head_parts: usize, decode_rows: usize) -> Result<Self> {
        ensure!(matches!(head_parts, 1 | 2), "Inkling kernel head partition differs");
        let decode_cap = cuteafd_loader::families::inkling2::execution::replay_capacity(decode_rows);
        let kv = 1024 / head_parts;
        let fir = [(4608, false), (kv, false), (kv, true)];
        let mut ops = Vec::new();
        for precision in [Precision::Bf16, Precision::F32] {
            for (width, dynamic) in fir {
                ops.push(Op::Convolution { width, dynamic, precision });
            }
            for (relative, key, global) in [(false, false, false), (false, true, false),
                (true, false, false), (false, false, true), (true, false, true)] {
                ops.push(Op::Clip { relative, key, global, precision });
            }
            for situ in [false, true] { ops.push(Op::Gate { situ, precision }); }
            for value in [false, true] { ops.push(Op::Norm { value, precision }); }
            ops.push(Op::KvPack { precision, fp8: false });
            if kv_format == cuteafd_loader::families::inkling2::InklingKvCache::Fp8 {
                ops.push(Op::KvPack { precision, fp8: true });
            }
        }
        for queries in [1, 3] {
            for addend in [false, true] { ops.push(Op::Residual { queries, addend }); }
        }
        ops.push(Op::Router);
        let mut producers = BTreeMap::new();
        for op in ops { producers.insert(op.rank_name(head_parts)?, Producer::load_rank(programs, op, head_parts)?); }
        let quant = programs.program("inkling2_expert_input_quant", &["source_ptr", "values_ptr", "scale_rows_ptr", "scale_mma_ptr"])?;
        let spec = quant.spec();
        ensure!(spec.capacity_rows == 4096 && spec.scalar_kinds == b"ii" && spec.scratch.is_empty()
            && spec.geometry["hidden"] == 4608 && spec.geometry["row_bytes"] == 4752
            && spec.geometry["amax_floor"] == 0.0 && spec.geometry["expected_m"] == 80,
            "Inkling wire quantizer ABI/geometry/floor differs");
        let strided = fir.into_iter().map(|geometry|
            Ok((geometry, cuteafd_ffi::fir_strided::Convolution::load(programs, geometry.0, geometry.1)?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let replay = fir.into_iter().map(|geometry|
            Ok((geometry, cuteafd_ffi::fir_replay::Convolution::load_rows(programs, geometry.0, geometry.1, decode_cap)?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self { projections: DraftProjections::load_with_output(programs, &[
            (4608, 6144 / head_parts), (4608, 768 / head_parts), (4608, kv),
            (4608, 48 / head_parts), (4608, 32 / head_parts), (4608, 384), (6144 / head_parts, 4608), (64, 1024)], 4096, ProjectionOutput::F32)?,
            local: Attention::load_rank(programs, Kind::Local, Route::Prefill, false, head_parts)?,
            global: Attention::load_rank(programs, Kind::Global, Route::Prefill, kv_format == cuteafd_loader::families::inkling2::InklingKvCache::Fp8, head_parts)?,
            local_decode: Attention::load_rank_rows(programs, Kind::Local, Route::Decode, false, head_parts, decode_cap)?,
            global_decode: Attention::load_rank_rows(programs, Kind::Global, Route::Decode, kv_format == cuteafd_loader::families::inkling2::InklingKvCache::Fp8, head_parts, decode_cap)?, kv_format, head_parts,
            add: programs.library().f32_add()?, narrow: programs.library().f32_bf16_narrow()?,
            copy: programs.library().v41_peer_copy()?, widen_add: programs.library().bf16_f32_add()?,
            quant, compact: None, draft_head: None,
            sms: programs.library().sm_count()? as usize, producers, strided, replay })
    }
    fn producer(&self, op: Op) -> Result<&Producer<'a>> {
        self.producers.get(&op.rank_name(self.head_parts)?).context("Inkling producer not bound at startup")
    }
    fn widen_zero(&self, owner: &impl UnitBuffers<'a>, src: Buffer, dst: Buffer, count: usize) -> Result<()> {
        let stream = owner.stream();
        // SAFETY: both routes use disjoint retained planes and the same +0 sum.
        unsafe {
            if owner.decode_attention() {
                self.widen_add.launch(owner.buffer("decode_zero_bf16")?, src, dst, count, stream.raw)
            } else {
                stream.device.library.cuda_zero_bytes_async(dst, count * 4, stream.raw)?;
                stream.device.library.cuda_accumulate_bf16_to_f32_async(src, dst, count, stream.raw)
            }
        }
    }

    fn copy_residual(&self, owner: &impl UnitBuffers<'a>, dst: Buffer, src: Buffer, bytes: usize) -> Result<()> {
        let stream = owner.stream();
        // SAFETY: retained F32 arena slots and their source planes are disjoint.
        unsafe {
            if owner.decode_attention() {
                self.copy.launch_rows(dst, src, bytes, 1, bytes, bytes, stream.raw)
            } else {
                stream.device.library.copy_d2d_async(dst, src, bytes, stream.raw)
            }
        }
    }

    pub fn norm(&self, owner: &impl UnitBuffers<'a>, input: Buffer, weight: Buffer, output: Buffer,
        value: bool, precision: Precision, rows: usize) -> Result<()> {
        // SAFETY: the family supplies disjoint admitted typed input/output views;
        // retained unit stream orders the producer and all consuming projections.
        unsafe { self.producer(Op::Norm { value, precision })?.launch(&[input, weight, output],
            rows, None, owner.stream().raw) }
    }
    pub fn convolution(&self, owner: &impl UnitBuffers<'a>, input: Buffer, weight: Buffer, dynamic: Buffer,
        output: Buffer, history: &str, width: usize, global: bool, rows: usize) -> Result<()> {
        if let Some((slot_stride, lag_stride, slots)) = owner.convolution_strides() {
            if let Some((replay, replay_stride)) = owner.convolution_replay(history)? {
                // SAFETY: input persistence is fused into the producer; the
                // retained verify owner covers every packed replay row.
                return unsafe { self.replay.get(&(width,global)).context("replay convolution not bound")?.launch(
                    [input,weight,dynamic,owner.buffer(history)?,owner.buffer("first")?,owner.buffer("conv_slots")?,output,replay],
                    rows,slots,slot_stride,lag_stride,replay_stride,owner.stream().raw) };
            }
            // SAFETY: request-local metadata is validated by the unit, fields
            // are strided views of the one admitted contiguous FIR backing.
            return unsafe { self.strided.get(&(width, global)).context("strided convolution not bound")?.launch(
                [input, weight, dynamic, owner.buffer(history)?, owner.buffer("first")?,
                    owner.buffer("conv_slots")?, output], rows, slots, slot_stride, lag_stride, owner.stream().raw) };
        }
        let history_geometry = None;
        // SAFETY: native F32 planes and BF16 taps are retained by unit/weights;
        // metadata addresses the unit's exclusive history and valid packed rows.
        unsafe { self.producer(Op::Convolution { width, dynamic: global, precision: Precision::F32 })?
            .launch(&[input, weight, dynamic, owner.buffer(history)?, owner.buffer("first")?,
                owner.buffer("conv_slots")?, output], rows, history_geometry, owner.stream().raw) }
    }
    pub fn residual(&self, owner: &impl UnitBuffers<'a>, query: Buffer, addend: Buffer, output: Buffer,
        rows: usize, queries: usize, candidates: usize, add: bool) -> Result<()> {
        for query_index in 0..queries {
            let query = slice(query, query_index*4608*4, 4608*4)?;
            let output = slice(output, query_index*owner.layout().rows*4608*4, rows*4608*4)?;
            // SAFETY: one query writes one segment at the original plane pitch;
            // the source arena retains that pitch across packed members.
            unsafe { self.producer(Op::Residual { queries: 1, addend: add })?.launch(&[
                owner.buffer("residual_arena")?, addend, query, output], rows,
                Some((owner.layout().rows, candidates)), owner.stream().raw)?; }
        }
        Ok(())
    }
    pub fn route(&self, owner: &impl UnitBuffers<'a>, logits: Buffer, bias: Buffer, rows: usize) -> Result<()> {
        // SAFETY: native F32 logits/bias and output routes have admitted extents.
        unsafe { self.producer(Op::Router)?.launch(&[logits, bias, owner.buffer("route_ids")?,
            owner.buffer("route_weights")?], rows, None, owner.stream().raw) }
    }
    pub fn compact_inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>, prefix: &str) -> Result<()> {
        if let Some(c) = &self.compact {
            if let Some(trellis) = &c.trellis_scratch {
                let backing = &trellis._owners["projection arena"];
                backing.inventory(inventory, &format!("{prefix}.trellis"))?;
            }
            if let Some(fp8) = &c.fp8_scratch { fp8.inventory(inventory, &format!("{prefix}.fp8"))?; }
        }
        Ok(())
    }

    pub fn bind_compact(&mut self, programs: &'a Programs<'a>, stream: Rc<Stream<'a>>, target: &CoordinatorWeights,
        rows: usize, head_rows: usize, fp8_bytes: usize) -> Result<()> {
        if !target.core.is_empty() || !target.fp8.is_empty() {
            self.compact = Some(CompactProjections::bind(programs, stream, target, rows, head_rows, fp8_bytes)?);
        }
        Ok(())
    }

    pub fn project_target(&self, owner: &impl UnitBuffers<'a>, input: Buffer, weight: ProjectionWeight<'_, '_>,
        output: Buffer, rows: usize, k: usize, n: usize) -> Result<()> {
        if let ProjectionWeight::Bf16(weight) = weight { return self.project(owner, input, weight.buffer, output, rows, k, n); }
        self.project_compact(owner.stream(), input, weight, output, rows, k, n)
    }

    /// Compact head and attention share one ordered projection owner. The
    /// recurrent drafter can use it without borrowing a target row unit.
    pub fn project_compact(&self, stream: &Stream<'a>, input: Buffer, weight: ProjectionWeight<'_, '_>,
        output: Buffer, rows: usize, k: usize, n: usize) -> Result<()> {
        let c = self.compact.as_ref().context("Inkling compact target runtime was not admitted")?;
        ensure!(std::ptr::eq(c.stream.as_ref(), stream) && input.bytes >= rows*k*2 && output.bytes >= rows*n*4,
            "Inkling compact projection stream or extents differ from admission");
        match weight {
            ProjectionWeight::Bf16(_) => anyhow::bail!("Inkling compact projection requires compact weights"),
            ProjectionWeight::Trellis(w) => {
                let key = trellis_geometry(w.shape, w.side, w.bits, w.codebook);
                ensure!(w.shape == [n, k], "Inkling trellis projection geometry differs from forward");
                c.trellis.get(&key).context("Inkling trellis program missing")?.local(rows, input, w,
                    c.trellis_scratch.as_ref().context("Inkling trellis scratch missing")?, output, stream.raw)
            }
            ProjectionWeight::Fp8(w) => {
                ensure!(w.shape == [n, k], "Inkling FP8 projection geometry differs from forward");
                let programs = if rows <= 64 { &c.fp8_decode } else { &c.fp8_prefill };
                let scratch = c.fp8_scratch.as_ref().context("Inkling FP8 scratch missing")?.buffer;
                // SAFETY: this ordered stream exclusively reuses the admitted scratch;
                // resident packed weights and checked input/output extents outlive the launch.
                unsafe { programs.as_ref().context("Inkling FP8 projection program missing")?.launch(
                    input.ptr, w.values.buffer.ptr, w.scales.buffer.ptr, output, scratch, rows, k, n, stream.raw)?; }
                Ok(())
            }
        }
    }

    fn trellis_parts(&self, stream: &Stream<'a>, weight: &crate::shared::dense_trellis::CoreWeights<'_>)
        -> Result<(&Projection<'a>, &ProjectionScratch<'a>)> {
        let c = self.compact.as_ref().context("Inkling trellis runtime was not admitted")?;
        ensure!(std::ptr::eq(c.stream.as_ref(), stream), "Inkling trellis scratch is bound to another stream");
        let key = trellis_geometry(weight.shape, weight.side, weight.bits, weight.codebook);
        Ok((c.trellis.get(&key).context("Inkling trellis program missing")?,
            c.trellis_scratch.as_ref().context("Inkling trellis scratch missing")?))
    }

    pub fn reduce_heads(&self, owner: &impl UnitBuffers<'a>, weight: ProjectionWeight<'_, '_>,
        remote: Buffer, rows: usize, hidden: usize) -> Result<Buffer> {
        let local = owner.buffer("projection_input")?;
        let sum = owner.buffer("mlp")?;
        let stream = owner.stream().raw;
        if let ProjectionWeight::Trellis(w) = weight {
            let (projection, _) = self.trellis_parts(owner.stream(), w)?;
            // The incoming attention mix is dead before either W_O partial is published.
            let output = owner.buffer("mixed")?;
            projection.reduced(&self.widen_add, rows, local, remote, sum, w, output, stream)?;
            Ok(output)
        } else {
            // SAFETY: a matching peer wait publishes the retained remote BF16 partial;
            // local BF16 partial and F32 sum are disjoint on this rank's stream.
            unsafe { self.widen_add.launch(local, remote, sum, rows * hidden, stream)?; }
            Ok(sum)
        }
    }

    pub fn project(&self, owner: &impl UnitBuffers<'a>, input: Buffer, weight: Buffer, output: Buffer,
        rows: usize, k: usize, n: usize) -> Result<()> {
        ensure!(input.bytes >= rows*k*2 && weight.bytes >= n*k*2 && output.bytes >= rows*n*4,
            "Inkling projection exceeds admitted views");
        // SAFETY: checked BF16 input/weight extents and F32 destination; arena
        // and unit retain every pointer on the preloaded shared projection stream.
        unsafe { self.projections.launch_checked(input.ptr, weight.ptr, output, rows, k, n, owner.stream().raw)?; }
        Ok(())
    }
}

/// Route/MLP stage has no knowledge of checkpoint expert format or transport.
pub(crate) fn routed_ffn<'a>(kernels: &Kernels<'a>, owner: &impl UnitBuffers<'a>, target: &TargetWeights<'_>,
    experts: &dyn Dispatcher, cfg: &Inkling2Config, layer: usize, rows: usize) -> Result<()> {
    prepare_routes(kernels, owner, target, cfg, layer, rows)?;
    dispatch_experts(owner, experts, rows)
}

pub(crate) fn dispatch_experts<'a>(owner: &impl UnitBuffers<'a>, experts: &dyn Dispatcher, rows: usize) -> Result<()> {
    let input = owner.buffer(if experts.wire_input() { "expert_wire" } else { "projection_input" })?;
    experts.enqueue(input, owner.buffer("route_ids")?, owner.buffer("route_weights")?,
        owner.buffer("expert_output")?, rows, owner.stream())
}

pub(crate) fn prepare_routes<'a>(kernels: &Kernels<'a>, owner: &impl UnitBuffers<'a>, target: &TargetWeights<'_>,
    cfg: &Inkling2Config, layer: usize, rows: usize) -> Result<()> {
    ensure!(layer < cfg.layers && rows > 0 && rows <= owner.layout().rows, "Inkling FFN layer/rows differ");
    let prefix = format!("model.llm.layers.{layer}");
    kernels.project(owner, owner.buffer("projection_input")?, target.operand(&format!("{prefix}.mlp.gate.weight"))?.buffer,
        owner.buffer("route_logits")?, rows, cfg.hidden, cfg.experts)?;
    kernels.route(owner, owner.buffer("route_logits")?, target.operand(&format!("{prefix}.mlp.gate.bias"))?.buffer, rows)?;
    let wire = owner.buffer("expert_wire")?;
    let grid = (rows * (cfg.hidden/32).div_ceil(8)).div_ceil(8).min(kernels.sms*4).max(1);
    // SAFETY: wire holds payload and per-row UE8M0 bytes, source BF16 projection
    // input is retained; unused MMA pointer is valid and never dereferenced.
    unsafe { kernels.quant.launch(&[owner.buffer("projection_input")?.ptr, wire.ptr,
        wire.ptr.cast::<u8>().wrapping_add(cfg.hidden).cast(), wire.ptr],
        &[cuteafd_ffi::programs::Scalar::I32(rows as i32), cuteafd_ffi::programs::Scalar::I32(grid as i32)], owner.stream().raw)?; }
    Ok(())
}

pub(crate) fn residual_site(cfg: &Inkling2Config, layer: usize) -> Result<ResidualSite> {
    ResidualSite::layer(cfg, layer)
}

fn slice(buffer: Buffer, offset: usize, bytes: usize) -> Result<Buffer> {
    ensure!(offset.checked_add(bytes).is_some_and(|end| end <= buffer.bytes), "Inkling view exceeds owner");
    Ok(Buffer { ptr: buffer.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..buffer })
}

/// Cold one-sequence attention: every cache row is produced before the native
/// ring/paged attention program, without host synchronization on the layer path.
pub(crate) fn attend_heads<'a>(k: &Kernels<'a>, owner: &impl UnitBuffers<'a>, target: &TargetWeights<'_>,
    cfg: &Inkling2Config, layer: usize, rows: usize, input: Buffer) -> Result<()> {
    ensure!(k.head_parts == owner.layout().head_parts, "Inkling attention owner head partition differs");
    let heads = cfg.heads / k.head_parts;
    let q_width = heads * cfg.head_dim;
    let kv_width = cfg.kv_heads / k.head_parts * cfg.head_dim;
    let rel_width = heads * cfg.d_rel;
    let local = cfg.local[layer];
    let prefix = format!("model.llm.layers.{layer}");
    let weight = |name: &str| -> Result<Buffer> { Ok(target.operand(&format!("{prefix}.{name}"))?.buffer) };
    let b = |name: &str| owner.buffer(name);
    let norm = if local { "attn_norm.weight" } else { "q_input_norm.weight" };
    k.norm(owner, input, weight(norm)?, b("projection_input")?, false, Precision::Bf16, rows)?;
    for (name, out, n) in [("wq_du", "q", q_width), ("wr_du", "r", rel_width),
        ("gate_attn", "gate", q_width), ("attn_sink_proj", "sinks", heads)] {
        k.project_target(owner, b("projection_input")?, target.projection(&format!("{prefix}.attn.{name}.weight"))?, b(out)?, rows, 4608, n)?;
    }
    for (branch, proj, out, convolved, history) in [("k", "wk_dv", "k", "conv_k", "k_conv_history"),
        ("v", "wv_dv", "v", "conv_v", "v_conv_history")] {
        if !local {
            let query = if branch == "k" { 1 } else { 2 };
            let mixed = slice(owner.buffer("global_mixes")?, query*owner.layout().rows*4608*4, rows*4608*4)?;
            k.norm(owner, mixed, weight(&format!("{branch}_input_norm.weight"))?, b("projection_input")?,
                false, Precision::Bf16, rows)?;
        }
        k.project_target(owner, b("projection_input")?, target.projection(&format!("{prefix}.attn.{proj}.weight"))?, b(out)?, rows, 4608, kv_width)?;
        let dynamic = if local { b(out)? } else {
            let name = format!("{branch}_dynamic");
            k.project(owner, b("projection_input")?, weight(&format!("attn.{branch}_sconv.weight_proj"))?,
                b(&name)?, rows, 4608, cfg.kv_heads / k.head_parts * cfg.sconv_kernel)?;
            b(&name)?
        };
        k.convolution(owner, b(out)?, weight(&format!("attn.{branch}_sconv.{}", if local { "weight" } else { "static_w" }))?,
            dynamic, b(convolved)?, history, kv_width, !local, rows)?;
    }
    for (name, source, output, relative, key) in [("q", "q", "q_clipped", false, false),
        ("r", "r", "r_clipped", true, false), ("k", "conv_k", "k_clipped", false, true)] {
        let scale = if key { b(source)? } else { weight(&format!("attn.{}_l2_clip.scale", if relative { "rel" } else { name }))? };
        // SAFETY: checked native producer geometry, admitted typed planes and
        // immutable scales are retained and stream-ordered until unit drain.
        unsafe { k.producer(Op::Clip { relative, key, global: !local && !key, precision: Precision::F32 })?
            .launch(&[b(source)?, scale, b("positions")?, b(output)?], rows, None, owner.stream().raw)?; }
    }
    k.norm(owner, b("conv_v")?, b("conv_v")?, b("v_norm")?, true, Precision::F32, rows)?;
    let l = owner.stream().device.library;
    // SAFETY: retained unit owns disjoint source/destination planes. The padded
    // relative row starts zero and only its logical16 columns are overwritten.
    unsafe {
        k.narrow.launch(b("q_clipped")?.ptr, b("q_bf16")?.ptr, rows*q_width, owner.stream().raw)?;
        k.narrow.launch(b("r_clipped")?.ptr, b("relative_source")?.ptr, rows*rel_width, owner.stream().raw)?;
        if owner.decode_attention() {
            // Decode owns this plane for life; allocation zeroes the tail and nothing writes it.
            k.copy.launch_rows(b("relative_input")?, b("relative_source")?, 16*2, rows*heads, 64*2, 16*2, owner.stream().raw)?;
        } else {
            l.cuda_zero_bytes_async(b("relative_input")?, rows*heads*64*2, owner.stream().raw)?;
            l.copy_d2d_2d_async(b("relative_input")?, 64*2, b("relative_source")?, 16*2, 16*2, rows*heads, owner.stream().raw)?;
        }
    }
    // A token contributes one independent relative row per local head. Bound each projection call
    // by its4096-row export without changing the ordered-K16 reduction.
    let mut first = 0;
    while first < rows*heads {
        let count = (rows*heads-first).min(4096);
        k.project(owner, slice(b("relative_input")?, first*64*2, count*64*2)?,
            weight("attn.rel_logits_proj.padded")?, slice(b("relative_f32")?, first*1024*4, count*1024*4)?,
            count, 64, 1024)?;
        first += count;
    }
    // SAFETY: retained F32 relative biases narrow into the whole-unit BF16 plane.
    unsafe { k.narrow.launch(b("relative_f32")?.ptr, b("relative")?.ptr, rows*heads*1024, owner.stream().raw)?; }
    attention_cache(k, owner, cfg, layer, rows)?;
    Ok(())
}

/// Fixed-shape private epilogue: no cache, metadata, or peer operands.
pub(crate) fn project_heads<'a>(k: &Kernels<'a>, owner: &impl UnitBuffers<'a>, target: &TargetWeights<'_>,
    cfg: &Inkling2Config, layer: usize, rows: usize) -> Result<()> {
    let q_width = cfg.heads / k.head_parts * cfg.head_dim;
    let b = |name: &str| owner.buffer(name);
    k.widen_zero(owner, b("attended")?, b("q")?, rows*q_width)?;
    // SAFETY: q holds the local heads and is no longer read by attention;
    // it is safely reused as widened attention scratch on the same stream.
    unsafe {
        k.producer(Op::Gate { situ: false, precision: Precision::F32 })?.launch(&[b("q")?, b("gate")?, b("gate")?,
            b("q_clipped")?], rows, None, owner.stream().raw)?;
        k.narrow.launch(b("q_clipped")?.ptr, b("gated")?.ptr, rows*q_width, owner.stream().raw)?;
    }
    let weight = output_weight(target, layer)?;
    if let ProjectionWeight::Trellis(w) = weight {
        if k.head_parts == 2 {
            ensure!(w.side == cuteafd_loader::formats::dense_trellis::TrellisSide::Input,
                "Inkling TP2 W_O requires an input-sharded trellis core");
            ensure!(w.shape == [cfg.hidden, q_width], "Inkling TP2 W_O geometry differs from forward");
            let (projection, scratch) = k.trellis_parts(owner.stream(), w)?;
            return projection.core_into(rows, b("gated")?, w, scratch, b("branch")?, owner.stream().raw);
        }
    }
    k.project_target(owner, b("gated")?, weight, b("branch")?, rows, q_width, cfg.hidden)
}

pub(crate) fn output_weight<'s, 'a>(target: &'s TargetWeights<'a>, layer: usize) -> Result<ProjectionWeight<'s, 'a>> {
    target.projection(&format!("model.llm.layers.{layer}.attn.wo_ud.weight"))
}

pub(crate) fn finish_attention<'a>(k: &Kernels<'a>, owner: &impl UnitBuffers<'a>, target: &TargetWeights<'_>,
    cfg: &Inkling2Config, layer: usize, rows: usize, projected: Buffer) -> Result<()> {
    let weight = target.operand(&format!("model.llm.layers.{layer}.attn_sconv.weight"))?.buffer;
    k.convolution(owner, projected, weight, projected, owner.buffer("attention")?,
        "attn_conv_history", cfg.hidden, false, rows)
}

pub(crate) fn attention<'a>(k: &Kernels<'a>, owner: &impl UnitBuffers<'a>, target: &TargetWeights<'_>,
    cfg: &Inkling2Config, layer: usize, rows: usize, input: Buffer) -> Result<()> {
    ensure!(k.head_parts == 1, "Inkling split attention needs F32 reduction before FIR");
    attend_heads(k, owner, target, cfg, layer, rows, input)?;
    project_heads(k, owner, target, cfg, layer, rows)?;
    finish_attention(k, owner, target, cfg, layer, rows, owner.buffer("branch")?)
}

fn attention_cache<'a>(k: &Kernels<'a>, owner: &impl UnitBuffers<'a>, cfg: &Inkling2Config,
    layer: usize, rows: usize) -> Result<()> {
    use cuteafd_ffi::inkling2::{Batch, Global, Local};
    let local = cfg.local[layer];
    let b = |name: &str| owner.buffer(name);
    // SAFETY: the per-sequence view retains private cache slots and writes only
    // its own scratch rows before attention consumes them on the same stream.
    unsafe { k.producer(Op::KvPack { precision: Precision::F32, fp8: !local && k.kv_format == cuteafd_loader::families::inkling2::InklingKvCache::Fp8 })?.launch(&[b("k_clipped")?, b("v_norm")?,
        b("cache_slots")?, if local { b("kv_step")? } else { b("global_cache")? }], rows, None, owner.stream().raw)?; }
    let decode=owner.decode_attention();
    let tile_rows=if decode {1}else{10};
    let (tiles, tables, table_width, ring_stride_records) = if let Some((unit, cache, _, members)) = owner.packed() {
        (members.iter().map(|member| member.rows.len().div_ceil(tile_rows)).sum(),
            members.len(), unit.layout.buffers["page_table"]/(cache.layout.sequences*4),
            cache.layout.ring_stride_records())
    } else { (rows.div_ceil(10), 1, owner.layout().buffers["page_table"]/4, 1088) };
    let batch = Batch { rows, tiles, tile_rows: b("tile_rows")?, q: b("q_bf16")?, kv: if local { b("kv_step")? } else { b("global_cache")? },
        positions: b("positions")?, relative: b("relative")?, sinks: b("sinks")?, output: b("attended")?, scratch: b("attention_scratch")? };
    if local {
        let attention=if decode{&k.local_decode}else{&k.local};
        attention.local(batch, Local { ring: b("ring")?, ring_stride_records, ring_slots: b("ring_slots")?, first: b("first")? }, owner.stream().raw)
    } else {
        let attention=if decode{&k.global_decode}else{&k.global};
        attention.global(batch, Global {
            pages: b("global_cache")?.bytes/(64*k.kv_format.record_bytes(cfg.kv_heads / k.head_parts, cfg.head_dim)), page_table: b("page_table")?,
            table_width, tables, table_indices: b("table_indices")?, first: b("first")? }, owner.stream().raw)
    }
}

/// Complete layer over a cold, contiguous one-sequence unit. Embedding and
/// completed six-layer block values remain F32 throughout residual mixing.
pub(crate) fn mix_attention_input<'a>(k: &Kernels<'a>, owner: &impl UnitBuffers<'a>, target: &TargetWeights<'_>,
    cfg: &Inkling2Config, index: usize, rows: usize) -> Result<Buffer> {
    let partial = owner.buffer("partial")?;
    if index == 0 { return Ok(partial); }
    let site = residual_site(cfg, index)?;
    let query = target.operand(&format!("model.llm.attn_res.attn_queries.{}", index-1))?.buffer;
    let output = owner.buffer(if site.queries == 3 { "global_mixes" } else { "mixed" })?;
    k.residual(owner, query, partial, output, rows, site.queries, site.attention_candidates, false)?;
    slice(output, 0, rows*cfg.hidden*4)
}

pub(crate) fn begin_layer<'a>(k: &Kernels<'a>, owner: &impl UnitBuffers<'a>, target: &TargetWeights<'_>,
    cfg: &Inkling2Config, index: usize, rows: usize) -> Result<()> {
    let input = mix_attention_input(k, owner, target, cfg, index, rows)?;
    attention(k, owner, target, cfg, index, rows, input)?;
    prepare_ffn_input(k, owner, target, cfg, index, rows)
}

pub(crate) fn prepare_ffn_input<'a>(k: &Kernels<'a>, owner: &impl UnitBuffers<'a>, target: &TargetWeights<'_>,
    cfg: &Inkling2Config, index: usize, rows: usize) -> Result<()> {
    let site = residual_site(cfg, index)?;
    let b = |name: &str| owner.buffer(name);
    let partial = b("partial")?;
    let stride = owner.layout().rows*cfg.hidden*4;
    let arena = b("residual_arena")?;
    if site.starts_block {
        // First layer of a block: freeze its incoming partial as a completed
        // value, use attention alone as the next partial for the MLP query.
        k.copy_residual(owner, slice(arena, (site.attention_candidates-1)*stride, rows*cfg.hidden*4)?,
            partial, rows*cfg.hidden*4)?;
        k.copy_residual(owner, slice(arena, (site.mlp_candidates-1)*stride, rows*cfg.hidden*4)?,
            b("attention")?, rows*cfg.hidden*4)?;
    } else {
        // SAFETY: add writes an independent branch; the following copy publishes
        // the new partial candidate after its complete F32 sum.
        unsafe {
            k.add.launch(partial, b("attention")?, b("branch")?, rows*cfg.hidden, owner.stream().raw)?;
        }
        k.copy_residual(owner, slice(arena, (site.mlp_candidates-1)*stride, rows*cfg.hidden*4)?,
            b("branch")?, rows*cfg.hidden*4)?;
    }
    k.residual(owner, target.operand(&format!("model.llm.attn_res.mlp_queries.{index}"))?.buffer,
        partial, b("mixed")?, rows, 1, site.mlp_candidates, false)?;
    k.norm(owner, b("mixed")?, target.operand(&format!("model.llm.layers.{index}.mlp_norm.weight"))?.buffer,
        b("projection_input")?, false, Precision::Bf16, rows)
}

pub(crate) fn finish_layer<'a>(k: &Kernels<'a>, owner: &impl UnitBuffers<'a>, target: &TargetWeights<'_>,
    experts: &dyn Dispatcher, cfg: &Inkling2Config, index: usize, rows: usize) -> Result<()> {
    routed_ffn(k, owner, target, experts, cfg, index, rows)?;
    join_experts(k, owner, target, cfg, index, rows)
}

pub(crate) fn join_experts<'a>(k: &Kernels<'a>, owner: &impl UnitBuffers<'a>, target: &TargetWeights<'_>,
    cfg: &Inkling2Config, index: usize, rows: usize) -> Result<()> {
    let site = residual_site(cfg, index)?;
    let b = |name: &str| owner.buffer(name);
    let arena = b("residual_arena")?;
    let stride = owner.layout().rows*cfg.hidden*4;
    let partial = b("partial")?;
    k.widen_zero(owner, b("expert_output")?, b("mlp")?, rows*cfg.hidden)?;
    k.convolution(owner, b("mlp")?, target.operand(&format!("model.llm.layers.{index}.mlp_sconv.weight"))?.buffer,
        b("mlp")?, b("mixed")?, "mlp_conv_history", cfg.hidden, false, rows)?;
    let base = if site.starts_block { b("attention")? } else { b("branch")? };
    // SAFETY: result is retained F32 partial; completed immutable candidates
    // are unchanged, and only the final mutable arena slot is refreshed.
    unsafe {
        k.add.launch(base, b("mixed")?, partial, rows*cfg.hidden, owner.stream().raw)?;
    }
    k.copy_residual(owner, slice(arena, (site.mlp_candidates-1)*stride, rows*cfg.hidden*4)?,
        partial, rows*cfg.hidden*4)
}
