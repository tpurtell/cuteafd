//! Inkling 2 Mini TP1 looped forward arithmetic matching authoritative precision `f1940f40eb`.
//!
//! Precision contract (`f1940f40eb`):
//! - BF16 compute: embedding & layer state, dense projections (`Q/K/V/R/gate/wo/rel_proj`),
//!   QK & PV GEMMs, controller `down_proj` and its `rank` output.
//! - F32 compute: `attn_res` queries & mixing, cross-pass queries & retrieval, norms/L2/log-scale
//!   internals, controller `forget`/`gain` GEMMs + `tanh` + blend, short-conv internals, router
//!   logits/scores/gamma, combined main+sink softmax, output gate sigmoid & multiply, vocabulary head.
use super::{experts::Dispatcher, owners::UnitBuffers, weights::MiniTargetWeights};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{
    draft_projection::{DraftProjections, ProjectionOutput},
    inkling2::producers::Precision,
    inkling2_mini::{
        producers::{Op, Producer},
        Attention, Batch, Global, Kind, Local, Route,
    },
    programs::Programs,
    CuteafdDeviceBuffer as Buffer,
};
use cuteafd_loader::families::inkling2_mini::{
    execution::{residual_site, DECODE_TILE_ROWS, PREFILL_TILE_ROWS},
    Inkling2MiniConfig, InklingKvCache, LoopVisit,
};
use std::collections::BTreeMap;

const HIDDEN: usize = 2304;
const HEADS: usize = 24;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const Q_WIDTH: usize = HEADS * HEAD_DIM; // 3072
const KV_WIDTH: usize = KV_HEADS * HEAD_DIM; // 1024
const REL_WIDTH: usize = HEADS * 16; // 384
const GATE_RANK: usize = 64;

pub(crate) struct MiniKernels<'a> {
    pub projections_bf16: DraftProjections<'a>,
    pub projections_f32: DraftProjections<'a>,
    pub local: Attention<'a>,
    pub global: Attention<'a>,
    local_decode: Attention<'a>,
    local_decode_128: Option<Attention<'a>>,
    global_decode: Attention<'a>,
    global_decode_128: Option<Attention<'a>>,
    kv_format: InklingKvCache,
    producers: BTreeMap<String, Producer<'a>>,
    strided: BTreeMap<(usize, bool), cuteafd_ffi::fir_strided::Convolution<'a>>,
    replay: BTreeMap<(usize, bool), cuteafd_ffi::fir_replay::Convolution<'a>>,
    replay_128: Option<BTreeMap<(usize, bool), cuteafd_ffi::fir_replay::Convolution<'a>>>,
    pub add: cuteafd_ffi::F32Add<'a>,
    pub narrow: cuteafd_ffi::F32Bf16Narrow<'a>,
    copy: cuteafd_ffi::DeviceRowCopy<'a>,
    widen_add: cuteafd_ffi::Bf16F32Add<'a>,
    quant: cuteafd_ffi::programs::Program<'a>,
    sms: usize,
}

/// Deterministic export capacity for runtime decode/verify kernels (`<= 64 -> m64`, `> 64 -> m128`).
pub(crate) fn select_decode_capacity(rows: usize) -> usize {
    cuteafd_loader::families::inkling2_mini::execution::replay_capacity(rows)
}

impl<'a> MiniKernels<'a> {
    pub fn bind(programs: &'a Programs<'a>, kv_format: InklingKvCache) -> Result<Self> {
        Self::bind_rows(programs, kv_format, 64)
    }

    pub fn bind_rows(programs: &'a Programs<'a>, kv_format: InklingKvCache, decode_rows: usize) -> Result<Self> {
        let load_128 = select_decode_capacity(decode_rows) == 128;
        let fir = [(HIDDEN, false), (KV_WIDTH, false), (KV_WIDTH, true)];
        let mut ops = Vec::new();
        for precision in [Precision::Bf16, Precision::F32] {
            for (width, dynamic) in fir {
                ops.push(Op::Convolution { width, dynamic, precision });
            }
            for (relative, key, global) in [
                (false, false, false),
                (false, true, false),
                (true, false, false),
                (false, false, true),
                (true, false, true),
            ] {
                ops.push(Op::Clip {
                    relative,
                    key,
                    global,
                    precision,
                });
            }
            ops.push(Op::Gate { precision });
            for value in [false, true] {
                ops.push(Op::Norm { value, precision });
            }
            ops.push(Op::KvPack { precision, fp8: false });
            if kv_format == InklingKvCache::Fp8 {
                ops.push(Op::KvPack { precision, fp8: true });
            }
        }
        for (queries, max_candidates) in [(1, 5), (3, 5), (4, 2)] {
            for addend in [false, true] {
                ops.push(Op::Residual {
                    queries,
                    max_candidates,
                    addend,
                });
            }
        }
        for (slots, broadcast_dest, broadcast_weight) in [
            (1, false, false),
            (3, true, true),
            (3, false, false),
        ] {
            ops.push(Op::LoopPairNorm { slots, broadcast_dest });
            ops.push(Op::LoopBlendNorm {
                slots,
                broadcast_dest,
                broadcast_weight,
            });
            ops.push(Op::LoopBlendNormFast {
                slots,
                broadcast_dest,
                broadcast_weight,
            });
        }
        ops.extend([
            Op::Router,
            Op::FastRouter,
            Op::FastNorm,
            Op::ResidualNorm { queries: 1 },
            Op::ResidualNorm { queries: 3 },
            Op::ResidualPairNorm,
            Op::AttnCrossPairNorm { broadcast_dest: true },
            Op::AttnCrossPairNorm { broadcast_dest: false },
        ]);
        let global_fp8 = kv_format == InklingKvCache::Fp8;
        for replay in [true, false] {
            ops.push(Op::AttnPrep {
                local: true,
                fp8: false,
                replay,
            });
            ops.push(Op::AttnPrep {
                local: false,
                fp8: global_fp8,
                replay,
            });
            for starts_block in [true, false] {
                ops.push(Op::PostAttnConv { starts_block, replay });
            }
            ops.push(Op::PostMlpConvResidual { replay });
        }
        ops.extend([Op::ControllerDown, Op::ControllerUp]);
        ops.extend([Op::AttnInProjLocal, Op::AttnInProjGlobal, Op::RouterProj]);
        let mut producers = BTreeMap::new();
        for op in ops {
            producers.insert(op.name()?, Producer::load(programs, op)?);
        }

        let quant = programs.program(
            "inkling2_mini_expert_input_quant",
            &["source_ptr", "values_ptr", "scale_rows_ptr", "scale_mma_ptr"],
        )?;
        let spec = quant.spec();
        ensure!(
            spec.capacity_rows == 4096
                && spec.scalar_kinds == b"ii"
                && spec.scratch.is_empty()
                && spec.geometry["hidden"] == HIDDEN
                && spec.geometry["row_bytes"] == 2376
                && spec.geometry["amax_floor"] == 0.0,
            "Inkling 2 Mini MXFP8 wire quantizer ABI/geometry differs"
        );

        let strided = fir
            .into_iter()
            .map(|g| Ok((g, cuteafd_ffi::fir_strided::Convolution::load(programs, g.0, g.1)?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let replay = fir
            .into_iter()
            .map(|g| Ok((g, cuteafd_ffi::fir_replay::Convolution::load_rows(programs, g.0, g.1, 64)?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let replay_128 = if load_128 {
            Some(
                fir.into_iter()
                    .map(|g| Ok((g, cuteafd_ffi::fir_replay::Convolution::load_rows(programs, g.0, g.1, 128)?)))
                    .collect::<Result<BTreeMap<_, _>>>()?,
            )
        } else {
            None
        };

        let projections_bf16 = DraftProjections::load_with_output(
            programs,
            &[
                (Q_WIDTH, HIDDEN),
                (64, 1024),
            ],
            4096,
            ProjectionOutput::Bf16,
        )?;
        let projections_f32 = DraftProjections::load_with_output(
            programs,
            &[],
            4096,
            ProjectionOutput::F32,
        )?;

        let fp8 = kv_format == InklingKvCache::Fp8;
        Ok(Self {
            projections_bf16,
            projections_f32,
            local: Attention::load(programs, Kind::Local, Route::Prefill, false)?,
            global: Attention::load(programs, Kind::Global, Route::Prefill, fp8)?,
            local_decode: Attention::load_rows(programs, Kind::Local, Route::Decode, false, 64)?,
            local_decode_128: if load_128 {
                Some(Attention::load_rows(programs, Kind::Local, Route::Decode, false, 128)?)
            } else {
                None
            },
            global_decode: Attention::load_rows(programs, Kind::Global, Route::Decode, fp8, 64)?,
            global_decode_128: if load_128 {
                Some(Attention::load_rows(programs, Kind::Global, Route::Decode, fp8, 128)?)
            } else {
                None
            },
            kv_format,
            producers,
            strided,
            replay,
            replay_128,
            add: programs.library().f32_add()?,
            narrow: programs.library().f32_bf16_narrow()?,
            copy: programs.library().device_row_copy()?,
            widen_add: programs.library().bf16_f32_add()?,
            quant,
            sms: programs.library().sm_count()? as usize,
        })
    }

    fn decode_attention(&self, local: bool, rows: usize) -> Result<&Attention<'a>> {
        if select_decode_capacity(rows) == 128 {
            let attn = if local {
                self.local_decode_128.as_ref()
            } else {
                self.global_decode_128.as_ref()
            };
            attn.context("Mini 128-row decode attention not bound")
        } else {
            Ok(if local { &self.local_decode } else { &self.global_decode })
        }
    }

    fn decode_replay(
        &self,
        width: usize,
        global: bool,
        rows: usize,
    ) -> Result<&cuteafd_ffi::fir_replay::Convolution<'a>> {
        let map = if select_decode_capacity(rows) == 128 {
            self.replay_128.as_ref().context("Mini 128-row replay conv not bound")?
        } else {
            &self.replay
        };
        map.get(&(width, global)).context("Mini replay conv not bound")
    }

    fn producer(&self, op: Op) -> Result<&Producer<'a>> {
        self.producers
            .get(&op.name()?)
            .with_context(|| format!("Inkling 2 Mini producer not bound: {op:?}"))
    }

    pub fn widen_zero(&self, owner: &impl UnitBuffers<'a>, src: Buffer, dst: Buffer, count: usize) -> Result<()> {
        let stream = owner.stream();
        // SAFETY: disjoint retained planes; +0 sum preserves exact BF16 values in F32.
        unsafe {
            if owner.decode_attention() {
                self.widen_add
                    .launch(owner.buffer("decode_zero_bf16")?, src, dst, count, stream.raw)
            } else {
                stream.device.library.cuda_zero_bytes_async(dst, count * 4, stream.raw)?;
                stream
                    .device
                    .library
                    .cuda_accumulate_bf16_to_f32_async(src, dst, count, stream.raw)
            }
        }
    }

    /// Rounds an F32 plane to exact BF16 and widens back to F32 (matches `.to(bf16).float()`).
    pub fn round_bf16_f32(
        &self,
        owner: &impl UnitBuffers<'a>,
        src_f32: Buffer,
        scratch_bf16: Buffer,
        dst_f32: Buffer,
        count: usize,
    ) -> Result<()> {
        let stream = owner.stream();
        // SAFETY: disjoint retained buffers on the same ordered stream.
        unsafe {
            self.narrow.launch(src_f32.ptr, scratch_bf16.ptr, count, stream.raw)?;
        }
        self.widen_zero(owner, scratch_bf16, dst_f32, count)
    }

    fn copy_residual(&self, owner: &impl UnitBuffers<'a>, dst: Buffer, src: Buffer, bytes: usize) -> Result<()> {
        let stream = owner.stream();
        // SAFETY: disjoint retained F32 arena planes.
        unsafe {
            if owner.decode_attention() {
                self.copy.launch_rows(dst, src, bytes, 1, bytes, bytes, stream.raw)
            } else {
                stream.device.library.copy_d2d_async(dst, src, bytes, stream.raw)
            }
        }
    }

    pub fn norm(
        &self,
        owner: &impl UnitBuffers<'a>,
        input: Buffer,
        weight: Buffer,
        output: Buffer,
        value: bool,
        precision: Precision,
        rows: usize,
    ) -> Result<()> {
        let op = if !value && precision == Precision::Bf16 {
            Op::FastNorm
        } else {
            Op::Norm { value, precision }
        };
        // SAFETY: disjoint typed planes on retained stream.
        unsafe {
            self.producer(op)?
                .launch(&[input, weight, output], rows, None, owner.stream().raw)
        }
    }

    pub fn residual_norm(
        &self,
        owner: &impl UnitBuffers<'a>,
        arena: Buffer,
        query: Buffer,
        weight: Buffer,
        output: Buffer,
        rows: usize,
        queries: usize,
        candidates: usize,
    ) -> Result<()> {
        // SAFETY: arena has pitch owner.layout().rows, out has live-row pitch rows.
        unsafe {
            self.producer(Op::ResidualNorm { queries })?.launch(
                &[arena, query, weight, output],
                rows,
                Some((owner.layout().rows, candidates)),
                owner.stream().raw,
            )
        }
    }

    pub fn convolution(
        &self,
        owner: &impl UnitBuffers<'a>,
        input: Buffer,
        weight: Buffer,
        dynamic: Buffer,
        output: Buffer,
        history: &str,
        width: usize,
        global: bool,
        rows: usize,
    ) -> Result<()> {
        if let Some((slot_stride, lag_stride, slots)) = owner.convolution_strides() {
            if let Some((replay, replay_stride)) = owner.convolution_replay(history)? {
                // SAFETY: retained verify replay owner covers packed rows; capacity selects m64 <= 64, m128 > 64.
                return unsafe {
                    self.decode_replay(width, global, rows)?.launch(
                            [
                                input,
                                weight,
                                dynamic,
                                owner.buffer(history)?,
                                owner.buffer("first")?,
                                owner.buffer("conv_slots")?,
                                output,
                                replay,
                            ],
                            rows,
                            slots,
                            slot_stride,
                            lag_stride,
                            replay_stride,
                            owner.stream().raw,
                        )
                };
            }
            // SAFETY: strided views of the contiguous 48-slot FIR backing.
            return unsafe {
                self.strided
                    .get(&(width, global))
                    .context("Mini strided conv not bound")?
                    .launch(
                        [
                            input,
                            weight,
                            dynamic,
                            owner.buffer(history)?,
                            owner.buffer("first")?,
                            owner.buffer("conv_slots")?,
                            output,
                        ],
                        rows,
                        slots,
                        slot_stride,
                        lag_stride,
                        owner.stream().raw,
                    )
            };
        }
        // SAFETY: cold standalone conv history.
        unsafe {
            self.producer(Op::Convolution {
                width,
                dynamic: global,
                precision: Precision::F32,
            })?
            .launch(
                &[
                    input,
                    weight,
                    dynamic,
                    owner.buffer(history)?,
                    owner.buffer("first")?,
                    owner.buffer("conv_slots")?,
                    output,
                ],
                rows,
                None,
                owner.stream().raw,
            )
        }
    }

    pub fn residual(
        &self,
        owner: &impl UnitBuffers<'a>,
        arena: Buffer,
        query: Buffer,
        addend: Buffer,
        output: Buffer,
        rows: usize,
        queries: usize,
        max_candidates: usize,
        candidates: usize,
        add: bool,
    ) -> Result<()> {
        // SAFETY: arena has pitch owner.layout().rows, out has live-row pitch rows.
        unsafe {
            self.producer(Op::Residual {
                queries,
                max_candidates,
                addend: add,
            })?
            .launch(
                &[arena, addend, query, output],
                rows,
                Some((owner.layout().rows, candidates)),
                owner.stream().raw,
            )
        }
    }

    pub fn project_bf16(
        &self,
        owner: &impl UnitBuffers<'a>,
        input: Buffer,
        weight: Buffer,
        output: Buffer,
        rows: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        ensure!(
            input.bytes >= rows * k * 2 && weight.bytes >= n * k * 2 && output.bytes >= rows * n * 2,
            "Mini BF16 projection exceeds admitted views ({rows}x{k}->{n})"
        );
        // SAFETY: checked BF16 input/weight/output extents on retained stream.
        unsafe {
            self.projections_bf16
                .launch_checked(input.ptr, weight.ptr, output, rows, k, n, owner.stream().raw)?;
        }
        Ok(())
    }

    pub fn project_f32(
        &self,
        owner: &impl UnitBuffers<'a>,
        input: Buffer,
        weight: Buffer,
        output: Buffer,
        rows: usize,
        k: usize,
        n: usize,
    ) -> Result<()> {
        ensure!(
            input.bytes >= rows * k * 2 && weight.bytes >= n * k * 2 && output.bytes >= rows * n * 4,
            "Mini F32 projection exceeds admitted views ({rows}x{k}->{n})"
        );
        // SAFETY: checked BF16 input/weight and F32 output extents on retained stream.
        unsafe {
            self.projections_f32
                .launch_checked(input.ptr, weight.ptr, output, rows, k, n, owner.stream().raw)?;
        }
        Ok(())
    }
}

fn slice(buffer: Buffer, offset: usize, bytes: usize) -> Result<Buffer> {
    ensure!(
        offset.checked_add(bytes).is_some_and(|end| end <= buffer.bytes),
        "Inkling 2 Mini view exceeds owner"
    );
    Ok(Buffer {
        ptr: buffer.ptr.cast::<u8>().wrapping_add(offset).cast(),
        bytes,
        ..buffer
    })
}

/// Executes attention + post-attention residual + MLP input preparation for one loop visit.
pub(crate) fn begin_visit<'a>(
    k: &MiniKernels<'a>,
    owner: &impl UnitBuffers<'a>,
    target: &MiniTargetWeights<'_>,
    cfg: &Inkling2MiniConfig,
    visit: LoopVisit,
    rows: usize,
) -> Result<()> {
    let layer = visit.layer;
    let local = visit.local;
    let site = residual_site(cfg, layer)?;
    let prefix = format!("model.llm.layers.{layer}");
    let w = |name: &str| -> Result<Buffer> { Ok(target.operand(&format!("{prefix}.{name}"))?.buffer) };
    let b = |name: &str| owner.buffer(name);

    // 1 & 2. Attention input AttnRes + cross-pass controller (pass > 0) or fused AttnRes+Norm (pass == 0).
    let (xq, xk, xv) = if visit.pass > 0 {
        ensure!(visit.in_core && layer > 0, "cross-pass controllers only run inside looped core");
        let query = target.operand(&format!("model.llm.attn_res.attn_queries.{}", layer - 1))?.buffer;
        let dest = b(if local { "mixed" } else { "global_mixes" })?;
        let paired = b("ctrl_paired")?;
        let rank = b("ctrl_rank")?;
        let fg = b("ctrl_fg")?;
        let out = b("ctrl_out")?;
        // SAFETY: fused AttnRes(dest) + CrossPass(4 queries) + LoopPairNorm(slots=3) in one launch.
        unsafe {
            k.producer(Op::AttnCrossPairNorm { broadcast_dest: local })?.launch(
                &[
                    b("residual_arena")?,
                    query,
                    b("prev_pass_arena")?,
                    w("cross_pass_queries")?,
                    dest,
                    b("cross_pass_msg")?,
                    paired,
                ],
                rows,
                Some((owner.layout().rows, site.attention_candidates)),
                owner.stream().raw,
            )?;
        }
        for (slot, stream_name) in [(0usize, "attn"), (1, "k"), (2, "v")] {
            let p_slot = slice(paired, slot * rows * 2 * HIDDEN * 2, rows * 2 * HIDDEN * 2)?;
            let r_slot = slice(rank, slot * rows * GATE_RANK * 2, rows * GATE_RANK * 2)?;
            let fg_slot = slice(fg, slot * rows * 2 * HIDDEN * 4, rows * 2 * HIDDEN * 4)?;
            // SAFETY: each slot's dense views and the reusable split-K owner survive the stream.
            unsafe {
                k.producer(Op::ControllerDown)?.launch(
                    &[p_slot, w(&format!("controllers.{stream_name}.down_weight"))?, r_slot, b("ctrl_down_partial")?],
                    rows, None, owner.stream().raw)?;
                k.producer(Op::ControllerUp)?.launch(
                    &[r_slot, w(&format!("controllers.{stream_name}.forget_gain_weight"))?, fg_slot],
                    rows, None, owner.stream().raw)?;
            }
        }
        let msg3 = slice(b("cross_pass_msg")?, 0, 3 * rows * HIDDEN * 4)?;
        let norm_w = w(if local { "attn_norm.weight" } else { "qkv_input_norm.weight" })?;
        // SAFETY: fast 768-thread weight-preloaded F32 tanh blend + weighted RMSNorm -> BF16 [3, rows, 2304].
        unsafe {
            k.producer(Op::LoopBlendNormFast {
                slots: 3,
                broadcast_dest: local,
                broadcast_weight: local,
            })?
            .launch(&[dest, msg3, fg, norm_w, out], rows, None, owner.stream().raw)?;
        }
        (
            slice(out, 0, rows * HIDDEN * 2)?,
            slice(out, rows * HIDDEN * 2, rows * HIDDEN * 2)?,
            slice(out, 2 * rows * HIDDEN * 2, rows * HIDDEN * 2)?,
        )
    } else if layer == 0 {
        k.norm(owner, b("partial")?, w("attn_norm.weight")?, b("projection_input")?, false, Precision::Bf16, rows)?;
        let x = b("projection_input")?;
        (x, x, x)
    } else if local {
        let query = target.operand(&format!("model.llm.attn_res.attn_queries.{}", layer - 1))?.buffer;
        k.residual_norm(
            owner,
            b("residual_arena")?,
            query,
            w("attn_norm.weight")?,
            b("projection_input")?,
            rows,
            1,
            site.attention_candidates,
        )?;
        let x = b("projection_input")?;
        (x, x, x)
    } else {
        let query = target.operand(&format!("model.llm.attn_res.attn_queries.{}", layer - 1))?.buffer;
        let out = b("ctrl_out")?;
        k.residual_norm(
            owner,
            b("residual_arena")?,
            query,
            w("qkv_input_norm.weight")?,
            out,
            rows,
            3,
            site.attention_candidates,
        )?;
        (
            slice(out, 0, rows * HIDDEN * 2)?,
            slice(out, rows * HIDDEN * 2, rows * HIDDEN * 2)?,
            slice(out, 2 * rows * HIDDEN * 2, rows * HIDDEN * 2)?,
        )
    };

    // 3..7. Fused Q/R/Gate/Sinks/K/V projections + fused decode/verify AttnPrep (or parallel prefill path).
    let fp8 = !local && k.kv_format == InklingKvCache::Fp8;
    // 3. Fused attention-input projections: Q / R / Gate / K / V (BF16) and Sinks [+ k/v dynamic] (F32).
    if local {
        // SAFETY: disjoint retained input, weight, and output planes on ordered stream.
        unsafe {
            k.producer(Op::AttnInProjLocal)?.launch(
                &[
                    xq, xk, xv,
                    w("attn.wq_du.weight")?, w("attn.wr_du.weight")?, w("attn.gate_attn.weight")?,
                    w("attn.attn_sink_proj.weight")?, w("attn.wk_dv.weight")?, w("attn.wv_dv.weight")?,
                    b("q")?, b("r")?, b("gate")?, b("sinks")?, b("k_bf16")?, b("v_bf16")?,
                ],
                rows, None, owner.stream().raw,
            )?;
        }
    } else {
        // SAFETY: disjoint retained input, weight, and output planes on ordered stream.
        unsafe {
            k.producer(Op::AttnInProjGlobal)?.launch(
                &[
                    xq, xk, xv,
                    w("attn.wq_du.weight")?, w("attn.wr_du.weight")?, w("attn.gate_attn.weight")?,
                    w("attn.attn_sink_proj.weight")?, w("attn.wk_dv.weight")?, w("attn.wv_dv.weight")?,
                    w("attn.k_sconv.weight_proj")?, w("attn.v_sconv.weight_proj")?,
                    b("q")?, b("r")?, b("gate")?, b("sinks")?, b("k_bf16")?, b("v_bf16")?,
                    b("k_dynamic")?, b("v_dynamic")?,
                ],
                rows, None, owner.stream().raw,
            )?;
        }
    }


    if rows <= 64 && owner.decode_attention() && owner.convolution_strides().is_some() {
        let (slot_stride, lag_stride, _slots) = owner.convolution_strides().unwrap();
        let k_rep = owner.convolution_replay("k_conv_history")?;
        let v_rep = owner.convolution_replay("v_conv_history")?;
        let (replay, k_replay_buf, v_replay_buf, replay_stride) = match (k_rep, v_rep) {
            (Some((kr, ks)), Some((vr, vs))) => {
                ensure!(ks == vs, "Mini K/V replay strides differ");
                (true, kr, vr, ks)
            }
            (None, None) => (false, b("k_f32")?, b("v_f32")?, 0),
            _ => anyhow::bail!("Mini K/V replay availability disagrees"),
        };
        let w_suffix = if local { "weight" } else { "static_w" };
        // SAFETY: fused Q/R clip + R zero-pad + K/V conv/commit/clip/norm/pack in one launch.
        unsafe {
            k.producer(Op::AttnPrep { local, fp8, replay })?.launch_attn_prep(
                &[
                    b("q")?,
                    w("attn.q_l2_clip.scale")?,
                    b("r")?,
                    w("attn.rel_l2_clip.scale")?,
                    b("positions")?,
                    b("q_clipped")?,
                    b("relative_input")?,
                    b("k_bf16")?,
                    w(&format!("attn.k_sconv.{w_suffix}"))?,
                    b("k_dynamic")?,
                    b("k_conv_history")?,
                    b("v_bf16")?,
                    w(&format!("attn.v_sconv.{w_suffix}"))?,
                    b("v_dynamic")?,
                    b("v_conv_history")?,
                    b("first")?,
                    b("conv_slots")?,
                    b("cache_slots")?,
                    if local { b("kv_step")? } else { b("global_cache")? },
                    k_replay_buf,
                    v_replay_buf,
                ],
                rows,
                slot_stride,
                lag_stride,
                replay_stride,
                owner.stream().raw,
            )?;
        }
        let mut first = 0;
        while first < rows * HEADS {
            let count = (rows * HEADS - first).min(4096);
            k.project_bf16(
                owner,
                slice(b("relative_input")?, first * 64 * 2, count * 64 * 2)?,
                w("attn.rel_logits_proj.padded")?,
                slice(b("relative")?, first * 1024 * 2, count * 1024 * 2)?,
                count,
                64,
                1024,
            )?;
            first += count;
        }
    } else {
        // 4. K & V short-convolution (BF16 -> F32 conv -> BF16).
        for (branch, bf16_name, f32_name, conv_name, history) in [
            ("k", "k_bf16", "k_f32", "conv_k", "k_conv_history"),
            ("v", "v_bf16", "v_f32", "conv_v", "v_conv_history"),
        ] {
            k.widen_zero(owner, b(bf16_name)?, b(f32_name)?, rows * KV_WIDTH)?;
            let dynamic = if local {
                b(f32_name)?
            } else {
                b(&format!("{branch}_dynamic"))?
            };
            k.convolution(
                owner,
                b(f32_name)?,
                w(&format!("attn.{branch}_sconv.{}", if local { "weight" } else { "static_w" }))?,
                dynamic,
                b(conv_name)?,
                history,
                KV_WIDTH,
                !local,
                rows,
            )?;
            // Exact `.to(bf16)` at model.py:245.
            // SAFETY: disjoint conv_k/conv_v (F32) and k_bf16/v_bf16 (BF16).
            unsafe {
                k.narrow.launch(b(conv_name)?.ptr, b(bf16_name)?.ptr, rows * KV_WIDTH, owner.stream().raw)?;
            }
        }

        // 5. L2 clip on Q, R, K (BF16 I/O, F32 internals) and unweighted RMSNorm on V (F32 input -> BF16).
        for (source, output, scale_name, relative, key) in [
            ("q", "q_clipped", Some("attn.q_l2_clip.scale"), false, false),
            ("r", "r_clipped", Some("attn.rel_l2_clip.scale"), true, false),
            ("k_bf16", "k_clipped", None, false, true),
        ] {
            let scale = match scale_name {
                Some(name) => w(name)?,
                None => b(source)?,
            };
            // SAFETY: BF16 I/O clip producers with F32 norm/scale/log-scale internals.
            unsafe {
                k.producer(Op::Clip {
                    relative,
                    key,
                    global: !local && !key,
                    precision: Precision::Bf16,
                })?
                .launch(
                    &[b(source)?, scale, b("positions")?, b(output)?],
                    rows,
                    None,
                    owner.stream().raw,
                )?;
            }
        }
        // Op::Norm reads F32; widen exact BF16-rounded conv_v into v_f32 first.
        k.widen_zero(owner, b("v_bf16")?, b("v_f32")?, rows * KV_WIDTH)?;
        k.norm(owner, b("v_f32")?, b("v_f32")?, b("v_norm")?, true, Precision::Bf16, rows)?;

        // 6. Pad R to K=64 and compute BF16 relative bias `[rows * 24, 1024]`.
        let lib = owner.stream().device.library;
        // SAFETY: disjoint retained unit buffers on the same ordered stream.
        unsafe {
            if owner.decode_attention() {
                k.copy
                    .launch_rows(b("relative_input")?, b("r_clipped")?, 16 * 2, rows * HEADS, 64 * 2, 16 * 2, owner.stream().raw)?;
            } else {
                lib.cuda_zero_bytes_async(b("relative_input")?, rows * HEADS * 64 * 2, owner.stream().raw)?;
                lib.copy_d2d_2d_async(
                    b("relative_input")?,
                    64 * 2,
                    b("r_clipped")?,
                    16 * 2,
                    16 * 2,
                    rows * HEADS,
                    owner.stream().raw,
                )?;
            }
        }
        let mut first = 0;
        while first < rows * HEADS {
            let count = (rows * HEADS - first).min(4096);
            k.project_bf16(
                owner,
                slice(b("relative_input")?, first * 64 * 2, count * 64 * 2)?,
                w("attn.rel_logits_proj.padded")?,
                slice(b("relative")?, first * 1024 * 2, count * 1024 * 2)?,
                count,
                64,
                1024,
            )?;
            first += count;
        }

        // 7. Pack KV.
        // SAFETY: disjoint k_clipped, v_norm, slots, and target cache plane.
        unsafe {
            k.producer(Op::KvPack {
                precision: Precision::Bf16,
                fp8,
            })?
            .launch(
                &[
                    b("k_clipped")?,
                    b("v_norm")?,
                    b("cache_slots")?,
                    if local { b("kv_step")? } else { b("global_cache")? },
                ],
                rows,
                None,
                owner.stream().raw,
            )?;
        }
    }
    let decode = owner.decode_attention();
    let tile_rows = if decode { DECODE_TILE_ROWS } else { PREFILL_TILE_ROWS };
    let (tiles, tables, table_width, ring_stride_records) = if let Some((unit, cache, _, members)) = owner.packed() {
        (
            members.iter().map(|m| m.rows.len().div_ceil(tile_rows)).sum(),
            members.len(),
            unit.layout.metadata.page_table_width,
            cache.layout.ring_stride_records(),
        )
    } else {
        (rows.div_ceil(tile_rows), 1, owner.layout().metadata.page_table_width, 1088)
    };
    let batch = Batch {
        rows,
        tiles,
        tile_rows: b("tile_rows")?,
        q: b("q_clipped")?,
        kv: if local { b("kv_step")? } else { b("global_cache")? },
        positions: b("positions")?,
        relative: b("relative")?,
        sinks: b("sinks")?,
        output: b("attended")?,
        scratch: b("attention_scratch")?,
    };
    if local {
        let attention = if decode { k.decode_attention(true, rows)? } else { &k.local };
        attention.local(
            batch,
            Local {
                ring: b("ring")?,
                ring_stride_records,
                ring_slots: b("ring_slots")?,
                first: b("first")?,
            },
            owner.stream().raw,
        )?;
    } else {
        let attention = if decode { k.decode_attention(false, rows)? } else { &k.global };
        attention.global(
            batch,
            Global {
                pages: b("global_cache")?.bytes / (64 * k.kv_format.record_bytes(KV_HEADS, HEAD_DIM)),
                page_table: b("page_table")?,
                table_width,
                tables,
                table_indices: b("table_indices")?,
                first: b("first")?,
            },
            owner.stream().raw,
        )?;
    }

    // 8 & 9. Output gate, wo_ud, fused post-attn conv+residual update, and MLP input (AttnRes+Norm or controller).
    // SAFETY: disjoint BF16 attended/gate/gated buffers.
    unsafe {
        k.producer(Op::Gate { precision: Precision::Bf16 })?
            .launch(&[b("attended")?, b("gate")?, b("gate")?, b("gated")?], rows, None, owner.stream().raw)?;
    }
    k.project_bf16(owner, b("gated")?, w("attn.wo_ud.weight")?, b("branch_bf16")?, rows, Q_WIDTH, HIDDEN)?;
    let partial = b("partial")?;
    let arena = b("residual_arena")?;
    let stride = owner.layout().rows * HIDDEN * 4;
    if rows <= 64 && owner.decode_attention() && owner.convolution_strides().is_some() {
        let (slot_stride, lag_stride, _slots) = owner.convolution_strides().unwrap();
        ensure!(
            !site.starts_block || site.attention_candidates + 1 == site.mlp_candidates,
            "Inkling 2 Mini block-start candidate indices differ"
        );
        let (replay, rep_buf, rep_stride) = match owner.convolution_replay("attn_conv_history")? {
            Some((buf, s)) => (true, buf, s),
            None => (false, b("mixed")?, 0),
        };
        let out_base = if site.starts_block { b("attention")? } else { b("branch")? };
        // SAFETY: fused BF16->attn_sconv->commit->BF16-round->add/copy->arena update in one launch.
        unsafe {
            k.producer(Op::PostAttnConv {
                starts_block: site.starts_block,
                replay,
            })?
            .launch_post_conv(
                &[
                    b("branch_bf16")?,
                    w("attn_sconv.weight")?,
                    b("attn_conv_history")?,
                    b("first")?,
                    b("conv_slots")?,
                    partial,
                    arena,
                    out_base,
                    rep_buf,
                ],
                rows,
                owner.layout().rows,
                site.mlp_candidates - 1,
                slot_stride,
                lag_stride,
                rep_stride,
                owner.stream().raw,
            )?;
        }
    } else {
        k.widen_zero(owner, b("branch_bf16")?, b("branch")?, rows * HIDDEN)?;
        k.convolution(
            owner,
            b("branch")?,
            w("attn_sconv.weight")?,
            b("branch")?,
            b("mixed")?,
            "attn_conv_history",
            HIDDEN,
            false,
            rows,
        )?;
        // Exact `conv = short_conv(attn, attn_sconv).to(bf16); out = conv.float()`.
        k.round_bf16_f32(owner, b("mixed")?, b("branch_bf16")?, b("attention")?, rows * HIDDEN)?;
        if site.starts_block {
            k.copy_residual(owner, slice(arena, (site.attention_candidates - 1) * stride, rows * HIDDEN * 4)?, partial, rows * HIDDEN * 4)?;
            k.copy_residual(owner, slice(arena, (site.mlp_candidates - 1) * stride, rows * HIDDEN * 4)?, b("attention")?, rows * HIDDEN * 4)?;
        } else {
            // SAFETY: partial + attention -> branch (F32, unrounded per model.py:328).
            unsafe {
                k.add.launch(partial, b("attention")?, b("branch")?, rows * HIDDEN, owner.stream().raw)?;
            }
            k.copy_residual(owner, slice(arena, (site.mlp_candidates - 1) * stride, rows * HIDDEN * 4)?, b("branch")?, rows * HIDDEN * 4)?;
        }
    }
    let mlp_query = target.operand(&format!("model.llm.attn_res.mlp_queries.{layer}"))?.buffer;
    if visit.pass > 0 {
        let msg_mlp = slice(b("cross_pass_msg")?, 3 * rows * HIDDEN * 4, rows * HIDDEN * 4)?;
        let p0 = slice(b("ctrl_paired")?, 0, rows * 2 * HIDDEN * 2)?;
        let r0 = slice(b("ctrl_rank")?, 0, rows * GATE_RANK * 2)?;
        let fg0 = slice(b("ctrl_fg")?, 0, rows * 2 * HIDDEN * 4)?;
        // SAFETY: fused MLP AttnRes + LoopPairNorm(slots=1) in one launch.
        unsafe {
            k.producer(Op::ResidualPairNorm)?.launch(
                &[arena, mlp_query, msg_mlp, b("mixed")?, p0],
                rows,
                Some((owner.layout().rows, site.mlp_candidates)),
                owner.stream().raw,
            )?;
        }
        // SAFETY: retained dense MLP controller views and unit-owned split-K workspace.
        unsafe {
            k.producer(Op::ControllerDown)?.launch(
                &[p0, w("controllers.mlp.down_weight")?, r0, b("ctrl_down_partial")?], rows, None, owner.stream().raw)?;
            k.producer(Op::ControllerUp)?.launch(
                &[r0, w("controllers.mlp.forget_gain_weight")?, fg0], rows, None, owner.stream().raw)?;
        }
        unsafe {
            k.producer(Op::LoopBlendNormFast {
                slots: 1,
                broadcast_dest: false,
                broadcast_weight: false,
            })?
            .launch(
                &[b("mixed")?, msg_mlp, fg0, w("mlp_norm.weight")?, b("projection_input")?],
                rows,
                None,
                owner.stream().raw,
            )?;
        }
    } else {
        k.residual_norm(
            owner,
            arena,
            mlp_query,
            w("mlp_norm.weight")?,
            b("projection_input")?,
            rows,
            1,
            site.mlp_candidates,
        )?;
    }
    Ok(())
}

/// Computes F32 router logits/ids/weights and optional MXFP8 wire quantization (layers 20..24).
pub(crate) fn prepare_routes<'a>(
    k: &MiniKernels<'a>,
    owner: &impl UnitBuffers<'a>,
    target: &MiniTargetWeights<'_>,
    experts: &dyn Dispatcher,
    layer: usize,
    rows: usize,
) -> Result<()> {
    let prefix = format!("model.llm.layers.{layer}");
    // SAFETY: retained BF16 input/weight and F32 route logits on ordered stream.
    unsafe {
        k.producer(Op::RouterProj)?.launch(
            &[
                owner.buffer("projection_input")?,
                target.operand(&format!("{prefix}.mlp.gate.weight"))?.buffer,
                owner.buffer("route_logits")?,
            ],
            rows,
            None,
            owner.stream().raw,
        )?;
    }
    // SAFETY: F32 router logits/bias -> I32 ids + F32 weights.
    // Keep stock 128-thread router at decode/verify rows (faster at M=1..64 than single-warp).
    let router_op = if rows > 64 { Op::FastRouter } else { Op::Router };
    unsafe {
        k.producer(router_op)?.launch(
            &[
                owner.buffer("route_logits")?,
                target.operand(&format!("{prefix}.mlp.gate.bias"))?.buffer,
                owner.buffer("route_ids")?,
                owner.buffer("route_weights")?,
            ],
            rows,
            None,
            owner.stream().raw,
        )?;
    }
    if experts.wire_input() {
        let wire = owner.buffer("expert_wire")?;
        let grid = (rows * (HIDDEN / 32).div_ceil(8)).div_ceil(8).min(k.sms * 4).max(1);
        // SAFETY: MXFP8 wire quantizer writes 2304 E4M3 + 72 UE8M0 bytes per row.
        unsafe {
            k.quant.launch(
                &[
                    owner.buffer("projection_input")?.ptr,
                    wire.ptr,
                    wire.ptr.cast::<u8>().wrapping_add(HIDDEN).cast(),
                    wire.ptr,
                ],
                &[
                    cuteafd_ffi::programs::Scalar::I32(rows as i32),
                    cuteafd_ffi::programs::Scalar::I32(grid as i32),
                ],
                owner.stream().raw,
            )?;
        }
    }
    Ok(())
}

pub(crate) fn dispatch_experts<'a>(
    owner: &impl UnitBuffers<'a>,
    experts: &dyn Dispatcher,
    rows: usize,
) -> Result<Buffer> {
    let input = owner.buffer(if experts.wire_input() {
        "expert_wire"
    } else {
        "projection_input"
    })?;
    experts.enqueue(
        input,
        owner.buffer("route_ids")?,
        owner.buffer("route_weights")?,
        owner.buffer("expert_output")?,
        rows,
        owner.stream(),
    )
}

/// Applies `mlp_sconv`, rounds layer output to exact BF16 state, updates `residual_arena`,
/// and performs loop boundary save/restart at visits 5, 17, and 29.
pub(crate) fn join_visit<'a>(
    k: &MiniKernels<'a>,
    owner: &impl UnitBuffers<'a>,
    target: &MiniTargetWeights<'_>,
    cfg: &Inkling2MiniConfig,
    visit: LoopVisit,
    expert_output: Buffer,
    rows: usize,
) -> Result<()> {
    let layer = visit.layer;
    let site = residual_site(cfg, layer)?;
    let b = |name: &str| owner.buffer(name);
    let arena = b("residual_arena")?;
    let stride = owner.layout().rows * HIDDEN * 4;
    let partial = b("partial")?;
    let base = if site.starts_block { b("attention")? } else { b("branch")? };

    if rows <= 64 && owner.decode_attention() && owner.convolution_strides().is_some() {
        let (slot_stride, lag_stride, _slots) = owner.convolution_strides().unwrap();
        let (replay, rep_buf, rep_stride) = match owner.convolution_replay("mlp_conv_history")? {
            Some((buf, s)) => (true, buf, s),
            None => (false, b("mixed")?, 0),
        };
        // SAFETY: fused BF16 expert_output -> mlp_sconv -> commit -> BF16 round -> add base -> BF16 round -> partial & arena.
        unsafe {
            k.producer(Op::PostMlpConvResidual { replay })?.launch_post_conv(
                &[
                    expert_output,
                    target.operand(&format!("model.llm.layers.{layer}.mlp_sconv.weight"))?.buffer,
                    b("mlp_conv_history")?,
                    b("first")?,
                    b("conv_slots")?,
                    base,
                    partial,
                    arena,
                    rep_buf,
                ],
                rows,
                owner.layout().rows,
                site.mlp_candidates - 1,
                slot_stride,
                lag_stride,
                rep_stride,
                owner.stream().raw,
            )?;
        }
    } else {
        k.widen_zero(owner, expert_output, b("mlp")?, rows * HIDDEN)?;
        k.convolution(
            owner,
            b("mlp")?,
            target.operand(&format!("model.llm.layers.{layer}.mlp_sconv.weight"))?.buffer,
            b("mlp")?,
            b("mixed")?,
            "mlp_conv_history",
            HIDDEN,
            false,
            rows,
        )?;
        // Exact `mlp = short_conv(moe, mlp_sconv).to(bf16)`.
        k.round_bf16_f32(owner, b("mixed")?, b("branch_bf16")?, b("mlp")?, rows * HIDDEN)?;

        // Exact `return (out + mlp.float()).to(bf16), blocks` (model.py:340).
        // SAFETY: base + mlp -> mixed (F32), then round BF16 -> partial (F32 holding exact BF16).
        unsafe {
            k.add.launch(base, b("mlp")?, b("mixed")?, rows * HIDDEN, owner.stream().raw)?;
        }
        k.round_bf16_f32(owner, b("mixed")?, b("branch_bf16")?, partial, rows * HIDDEN)?;
        k.copy_residual(
            owner,
            slice(arena, (site.mlp_candidates - 1) * stride, rows * HIDDEN * 4)?,
            partial,
            rows * HIDDEN * 4,
        )?;
    }

    // Loop boundary bookkeeping (exact reference/model.py:367-375):
    let row_bytes = rows * HIDDEN * 4;
    if !visit.in_core && layer + 1 == cfg.loop_cfg.core_start {
        // End of prelude (visit 5, layer 5): save pre-core (partial = h_after_L5, block0 = embed_normed).
        k.copy_residual(owner, b("pre_core_partial")?, partial, row_bytes)?;
        k.copy_residual(owner, b("pre_core_blocks")?, slice(arena, 0, row_bytes)?, row_bytes)?;
    } else if visit.in_core && layer + 1 == cfg.loop_cfg.core_end && visit.pass + 1 < cfg.loop_cfg.num_passes {
        // End of core pass 0 (visit 17) or pass 1 (visit 29):
        // Save prev_pass_arena = [h_after_L11 (arena[2]), h_after_L17 (partial)], then restart core!
        let prev = b("prev_pass_arena")?;
        k.copy_residual(owner, slice(prev, 0, row_bytes)?, slice(arena, 2 * stride, row_bytes)?, row_bytes)?;
        k.copy_residual(owner, slice(prev, stride, row_bytes)?, partial, row_bytes)?;
        k.copy_residual(owner, partial, b("pre_core_partial")?, row_bytes)?;
        k.copy_residual(owner, slice(arena, 0, row_bytes)?, b("pre_core_blocks")?, row_bytes)?;
        k.copy_residual(owner, slice(arena, stride, row_bytes)?, b("pre_core_partial")?, row_bytes)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_decode_capacity_selects_64_at_8_rows_and_128_at_80_rows() {
        for rows in [1usize, 8, 16, 64] {
            assert_eq!(select_decode_capacity(rows), 64, "rows={rows} must select m64");
        }
        for rows in [65usize, 80, 96, 128] {
            assert_eq!(select_decode_capacity(rows), 128, "rows={rows} must select m128");
        }
    }
}
