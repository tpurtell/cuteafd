//! Family arithmetic over caller-owned native producers and expert dispatch.
use super::{experts::Dispatcher, owners::UnitOwner, weights::TargetWeights};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{draft_projection::{DraftProjections, ProjectionOutput}, inkling2::{Attention, Kind, Route},
    inkling2::producers::{Op, Precision, Producer}, programs::Programs, CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::families::inkling2::{execution::ResidualSite, Inkling2Config};
use std::collections::BTreeMap;

pub(crate) struct Kernels<'a> {
    pub projections: DraftProjections<'a>,
    pub local: Attention<'a>,
    pub global: Attention<'a>,
    producers: BTreeMap<String, Producer<'a>>,
    pub add: cuteafd_ffi::F32Add<'a>,
    pub narrow: cuteafd_ffi::F32Bf16Narrow<'a>,
    quant: cuteafd_ffi::programs::Program<'a>,
    sms: usize,
}
impl<'a> Kernels<'a> {
    pub fn bind(programs: &'a Programs<'a>) -> Result<Self> {
        let mut ops = Vec::new();
        for precision in [Precision::Bf16, Precision::F32] {
            for (width, dynamic) in [(4608, false), (1024, false), (1024, true)] {
                ops.push(Op::Convolution { width, dynamic, precision });
            }
            for (relative, key, global) in [(false, false, false), (false, true, false),
                (true, false, false), (false, false, true), (true, false, true)] {
                ops.push(Op::Clip { relative, key, global, precision });
            }
            for situ in [false, true] { ops.push(Op::Gate { situ, precision }); }
            for value in [false, true] { ops.push(Op::Norm { value, precision }); }
            ops.push(Op::KvPack { precision });
        }
        for queries in [1, 3] {
            for addend in [false, true] { ops.push(Op::Residual { queries, addend }); }
        }
        ops.push(Op::Router);
        let mut producers = BTreeMap::new();
        for op in ops { producers.insert(op.name()?, Producer::load(programs, op)?); }
        Ok(Self { projections: DraftProjections::load_with_output(programs, &[
            (4608, 6144), (4608, 768), (4608, 1024), (4608, 48), (4608, 32),
            (4608, 384), (6144, 4608), (64, 1024)], 4096, ProjectionOutput::F32)?,
            local: Attention::load(programs, Kind::Local, Route::Prefill)?,
            global: Attention::load(programs, Kind::Global, Route::Prefill)?,
            add: programs.library().f32_add()?, narrow: programs.library().f32_bf16_narrow()?,
            quant: programs.program("inkling2_expert_input_quant", &["source_ptr", "values_ptr", "scale_rows_ptr", "scale_mma_ptr"])?,
            sms: programs.library().sm_count()? as usize, producers })
    }
    fn producer(&self, op: Op) -> Result<&Producer<'a>> {
        self.producers.get(&op.name()?).context("Inkling producer not bound at startup")
    }
    pub fn norm(&self, owner: &UnitOwner<'_>, input: Buffer, weight: Buffer, output: Buffer,
        value: bool, precision: Precision, rows: usize) -> Result<()> {
        // SAFETY: the family supplies disjoint admitted typed input/output views;
        // retained unit stream orders the producer and all consuming projections.
        unsafe { self.producer(Op::Norm { value, precision })?.launch(&[input, weight, output],
            rows, None, owner.stream.raw) }
    }
    pub fn convolution(&self, owner: &UnitOwner<'_>, input: Buffer, weight: Buffer, dynamic: Buffer,
        output: Buffer, history: &str, width: usize, global: bool, rows: usize) -> Result<()> {
        // SAFETY: native F32 planes and BF16 taps are retained by unit/weights;
        // metadata addresses the unit's exclusive history and valid packed rows.
        unsafe { self.producer(Op::Convolution { width, dynamic: global, precision: Precision::F32 })?
            .launch(&[input, weight, dynamic, owner.buffer(history)?, owner.buffer("first")?,
                owner.buffer("conv_slots")?, output], rows, None, owner.stream.raw) }
    }
    pub fn residual(&self, owner: &UnitOwner<'_>, query: Buffer, addend: Buffer, output: Buffer,
        rows: usize, queries: usize, candidates: usize, add: bool) -> Result<()> {
        // SAFETY: caller initialized each candidate in the F32 arena; the query
        // and optional addend are retained through the mix on the unit stream.
        unsafe { self.producer(Op::Residual { queries, addend: add })?.launch(&[
            owner.buffer("residual_arena")?, addend, query, output], rows,
            Some((owner.layout.rows, candidates)), owner.stream.raw) }
    }
    pub fn route(&self, owner: &UnitOwner<'_>, logits: Buffer, bias: Buffer, rows: usize) -> Result<()> {
        // SAFETY: native F32 logits/bias and output routes have admitted extents.
        unsafe { self.producer(Op::Router)?.launch(&[logits, bias, owner.buffer("route_ids")?,
            owner.buffer("route_weights")?], rows, None, owner.stream.raw) }
    }
    pub fn project(&self, owner: &UnitOwner<'_>, input: Buffer, weight: Buffer, output: Buffer,
        rows: usize, k: usize, n: usize) -> Result<()> {
        ensure!(input.bytes >= rows*k*2 && weight.bytes >= n*k*2 && output.bytes >= rows*n*4,
            "Inkling projection exceeds admitted views");
        // SAFETY: checked BF16 input/weight extents and F32 destination; arena
        // and unit retain every pointer on the preloaded shared projection stream.
        unsafe { self.projections.launch_checked(input.ptr, weight.ptr, output, rows, k, n, owner.stream.raw)?; }
        Ok(())
    }
}

/// Route/MLP stage has no knowledge of checkpoint expert format or transport.
pub(crate) fn routed_ffn(kernels: &Kernels<'_>, owner: &UnitOwner<'_>, target: &TargetWeights<'_>,
    experts: &dyn Dispatcher, cfg: &Inkling2Config, layer: usize, rows: usize) -> Result<()> {
    ensure!(layer < cfg.layers && rows > 0 && rows <= owner.layout.rows, "Inkling FFN layer/rows differ");
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
        &[cuteafd_ffi::programs::Scalar::I32(rows as i32), cuteafd_ffi::programs::Scalar::I32(grid as i32)], owner.stream.raw)?; }
    experts.enqueue(owner.buffer("expert_wire")?, owner.buffer("route_ids")?, owner.buffer("route_weights")?,
        owner.buffer("expert_output")?, rows, &owner.stream)
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
pub(crate) fn attention(k: &Kernels<'_>, owner: &UnitOwner<'_>, target: &TargetWeights<'_>,
    cfg: &Inkling2Config, layer: usize, rows: usize, input: Buffer) -> Result<()> {
    use cuteafd_ffi::inkling2::{Batch, Global, Local};
    let local = cfg.local[layer];
    let prefix = format!("model.llm.layers.{layer}");
    let weight = |name: &str| -> Result<Buffer> { Ok(target.operand(&format!("{prefix}.{name}"))?.buffer) };
    let b = |name: &str| owner.buffer(name);
    let norm = if local { "attn_norm.weight" } else { "q_input_norm.weight" };
    k.norm(owner, input, weight(norm)?, b("projection_input")?, false, Precision::Bf16, rows)?;
    for (name, out, n) in [("wq_du", "q", 6144), ("wr_du", "r", 768),
        ("gate_attn", "gate", 6144), ("attn_sink_proj", "sinks", 48)] {
        k.project(owner, b("projection_input")?, weight(&format!("attn.{name}.weight"))?, b(out)?, rows, 4608, n)?;
    }
    for (branch, proj, out, convolved, history) in [("k", "wk_dv", "k", "conv_k", "k_conv_history"),
        ("v", "wv_dv", "v", "conv_v", "v_conv_history")] {
        if !local {
            let query = if branch == "k" { 1 } else { 2 };
            let mixed = slice(owner.buffer("global_mixes")?, query*rows*4608*4, rows*4608*4)?;
            k.norm(owner, mixed, weight(&format!("{branch}_input_norm.weight"))?, b("projection_input")?,
                false, Precision::Bf16, rows)?;
        }
        k.project(owner, b("projection_input")?, weight(&format!("attn.{proj}.weight"))?, b(out)?, rows, 4608, 1024)?;
        let dynamic = if local { b(out)? } else {
            let name = format!("{branch}_dynamic");
            k.project(owner, b("projection_input")?, weight(&format!("attn.{branch}_sconv.weight_proj"))?,
                b(&name)?, rows, 4608, 32)?;
            b(&name)?
        };
        k.convolution(owner, b(out)?, weight(&format!("attn.{branch}_sconv.{}", if local { "weight" } else { "static_w" }))?,
            dynamic, b(convolved)?, history, 1024, !local, rows)?;
    }
    for (name, source, output, relative, key) in [("q", "q", "q_clipped", false, false),
        ("r", "r", "r_clipped", true, false), ("k", "conv_k", "k_clipped", false, true)] {
        let scale = if key { b(source)? } else { weight(&format!("attn.{}_l2_clip.scale", if relative { "rel" } else { name }))? };
        // SAFETY: checked native producer geometry, admitted typed planes and
        // immutable scales are retained and stream-ordered until unit drain.
        unsafe { k.producer(Op::Clip { relative, key, global: !local && !key, precision: Precision::F32 })?
            .launch(&[b(source)?, scale, b("positions")?, b(output)?], rows, None, owner.stream.raw)?; }
    }
    k.norm(owner, b("conv_v")?, b("conv_v")?, b("v_norm")?, true, Precision::F32, rows)?;
    let l = owner.stream.device.library;
    // SAFETY: retained unit owns disjoint source/destination planes. The padded
    // relative row starts zero and only its logical16 columns are overwritten.
    unsafe {
        k.narrow.launch(b("q_clipped")?.ptr, b("q_bf16")?.ptr, rows*6144, owner.stream.raw)?;
        k.narrow.launch(b("r_clipped")?.ptr, b("relative_source")?.ptr, rows*768, owner.stream.raw)?;
        l.cuda_zero_bytes_async(b("relative_input")?, rows*48*64*2, owner.stream.raw)?;
        // Disjoint source and destination; columns16..63 stay exact zero.
        l.copy_d2d_2d_async(b("relative_input")?, 64*2, b("relative_source")?, 16*2, 16*2, rows*48, owner.stream.raw)?;
    }
    // A token contributes48 independent head rows. Bound each projection call
    // by its4096-row export without changing the ordered-K16 reduction.
    let mut first = 0;
    while first < rows*48 {
        let count = (rows*48-first).min(4096);
        k.project(owner, slice(b("relative_input")?, first*64*2, count*64*2)?,
            weight("attn.rel_logits_proj.padded")?, slice(b("relative_f32")?, first*1024*4, count*1024*4)?,
            count, 64, 1024)?;
        first += count;
    }
    // SAFETY: relative F32 bias narrows into admitted BF16 bias; pack writes
    // unique causal cache slots after both K and V producers on this stream.
    unsafe {
        k.narrow.launch(b("relative_f32")?.ptr, b("relative")?.ptr, rows*48*1024, owner.stream.raw)?;
        k.producer(Op::KvPack { precision: Precision::F32 })?.launch(&[b("k_clipped")?, b("v_norm")?,
            b("cache_slots")?, if local { b("kv_step")? } else { b("global_cache")? }], rows, None, owner.stream.raw)?;
    }
    let batch = Batch { rows, q: b("q_bf16")?, kv: if local { b("kv_step")? } else { b("global_cache")? },
        positions: b("positions")?, relative: b("relative")?, sinks: b("sinks")?, output: b("attended")?, scratch: b("attention_scratch")? };
    if local {
        k.local.local(batch, Local { ring: b("ring")?, ring_count: 1, ring_slots: b("ring_slots")?, first: b("first")? }, owner.stream.raw)?;
    } else {
        k.global.global(batch, Global { pages: b("global_cache")?.bytes/(64*4096), page_table: b("page_table")?,
            table_width: owner.layout.buffers["page_table"]/4, splits: 1 }, owner.stream.raw)?;
    }
    // F32 gate producer consumes F32 attended values. Widen BF16 through a
    // zeroed F32 destination before gating; no host data path.
    // SAFETY: q has6144 F32 values per row and is no longer read by attention;
    // it is safely reused as widened attention scratch on the same stream.
    unsafe {
        l.cuda_zero_bytes_async(b("q")?, rows*6144*4, owner.stream.raw)?;
        l.cuda_accumulate_bf16_to_f32_async(b("attended")?, b("q")?, rows*6144, owner.stream.raw)?;
        k.producer(Op::Gate { situ: false, precision: Precision::F32 })?.launch(&[b("q")?, b("gate")?, b("gate")?,
            b("q_clipped")?], rows, None, owner.stream.raw)?;
        k.narrow.launch(b("q_clipped")?.ptr, b("gated")?.ptr, rows*6144, owner.stream.raw)?;
    }
    k.project(owner, b("gated")?, weight("attn.wo_ud.weight")?, b("branch")?, rows, 6144, 4608)?;
    k.convolution(owner, b("branch")?, weight("attn_sconv.weight")?, b("branch")?, b("attention")?,
        "attn_conv_history", 4608, false, rows)
}

/// Complete layer over a cold, contiguous one-sequence unit. Embedding and
/// completed six-layer block values remain F32 throughout residual mixing.
pub(crate) fn layer(k: &Kernels<'_>, owner: &UnitOwner<'_>, target: &TargetWeights<'_>,
    experts: &dyn Dispatcher, cfg: &Inkling2Config, index: usize, rows: usize) -> Result<()> {
    let site = residual_site(cfg, index)?;
    let b = |name: &str| owner.buffer(name);
    let partial = b("partial")?;
    let input = if index == 0 { partial } else {
        let query = target.operand(&format!("model.llm.attn_res.attn_queries.{}", index-1))?.buffer;
        let output = if site.queries == 3 { b("global_mixes")? } else { b("mixed")? };
        k.residual(owner, query, partial, output, rows, site.queries, site.attention_candidates, false)?;
        slice(output, 0, rows*cfg.hidden*4)?
    };
    attention(k, owner, target, cfg, index, rows, input)?;
    let library = owner.stream.device.library;
    let stride = owner.layout.rows*cfg.hidden*4;
    let arena = b("residual_arena")?;
    if site.starts_block {
        // First layer of a block: freeze its incoming partial as a completed
        // value, use attention alone as the next partial for the MLP query.
        // SAFETY: arena slots are disjoint retained views on the same stream.
        unsafe {
            library.copy_d2d_async(slice(arena, (site.attention_candidates-1)*stride, rows*cfg.hidden*4)?,
                partial, rows*cfg.hidden*4, owner.stream.raw)?;
            library.copy_d2d_async(slice(arena, (site.mlp_candidates-1)*stride, rows*cfg.hidden*4)?,
                b("attention")?, rows*cfg.hidden*4, owner.stream.raw)?;
        }
    } else {
        // SAFETY: add writes an independent branch; the following copy publishes
        // the new partial candidate after its complete F32 sum.
        unsafe {
            k.add.launch(partial, b("attention")?, b("branch")?, rows*cfg.hidden, owner.stream.raw)?;
            library.copy_d2d_async(slice(arena, (site.mlp_candidates-1)*stride, rows*cfg.hidden*4)?,
                b("branch")?, rows*cfg.hidden*4, owner.stream.raw)?;
        }
    }
    k.residual(owner, target.operand(&format!("model.llm.attn_res.mlp_queries.{index}"))?.buffer,
        partial, b("mixed")?, rows, 1, site.mlp_candidates, false)?;
    k.norm(owner, b("mixed")?, target.operand(&format!("model.llm.layers.{index}.mlp_norm.weight"))?.buffer,
        b("projection_input")?, false, Precision::Bf16, rows)?;
    routed_ffn(k, owner, target, experts, cfg, index, rows)?;
    // SAFETY: expert BF16 output widens into an independently zeroed F32
    // plane; no extra weight representation or host synchronization is used.
    unsafe {
        library.cuda_zero_bytes_async(b("mlp")?, rows*cfg.hidden*4, owner.stream.raw)?;
        library.cuda_accumulate_bf16_to_f32_async(b("expert_output")?, b("mlp")?, rows*cfg.hidden, owner.stream.raw)?;
    }
    k.convolution(owner, b("mlp")?, target.operand(&format!("model.llm.layers.{index}.mlp_sconv.weight"))?.buffer,
        b("mlp")?, b("mixed")?, "mlp_conv_history", cfg.hidden, false, rows)?;
    let base = if site.starts_block { b("attention")? } else { b("branch")? };
    // SAFETY: result is retained F32 partial; completed immutable candidates
    // are unchanged, and only the final mutable arena slot is refreshed.
    unsafe {
        k.add.launch(base, b("mixed")?, partial, rows*cfg.hidden, owner.stream.raw)?;
        library.copy_d2d_async(slice(arena, (site.mlp_candidates-1)*stride, rows*cfg.hidden*4)?,
            partial, rows*cfg.hidden*4, owner.stream.raw)?;
    }
    Ok(())
}
