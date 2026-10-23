//! GLM 5.3 attention producers, o_proj and dense FFNs over an EXLR8 checkpoint's
//! own dense operands (`weights::GlmLayer::native`), composed from the shared
//! native dense linears and the format-neutral GLM attention stages. The
//! official fused programs are untouched; a native layer replaces exactly the
//! four entry points (`producer`, `index_producer`, `o_proj`, `ffn_on`) and
//! writes the same outputs (KV record, `query`, `q_resid`, FP8 index query and
//! head weights, `delta`, FFN output) into the same workspace buffers.
//!
//! Row routes: decode and verify (rows <= 64) run every projection on its native
//! kernel. Prefill (rows > 64): trellis projections take the exact FP8 unpack
//! baseline ([`Unpacked`]: the payload decoded once per chunk to E4M3 `[N, K]`
//! with unit 128x128 scales, then the block-FP8 W8A16 GEMM over the
//! BF16-rotated rows, F32 out); MXFP8 and int4 stay on their native kernels.
//!
//! Trellis transforms: `rotated = H128(x * suh)` before the core, `H128(acc) *
//! svh` after. Split projections apply svh per rank (o by input columns, down
//! by intermediate) before the existing BF16 partial exchange, the order the
//! shared trellis ABI permits ("transform each rank then reduce BF16 outputs").
use super::*;
use cuteafd_ffi::dense_trellis::{DenseTrellisGeometry, TrellisDType};
use cuteafd_ffi::native_dense::DenseLinear;
use cuteafd_ffi::native_dense_glue::Companion;
use cuteafd_ffi::CuteafdDeviceBuffer as Buffer;
use cuteafd_loader::plan::exlr8::dense::DenseEncoding;
use std::collections::HashMap;

/// Decode/verify rows run every native kernel; wider steps take the prefill route.
pub(super) const NATIVE_DECODE_ROWS: usize = 64;

/// One GPU's bound native programs (keyed by exported name).
pub(crate) struct NativePrograms<'a> {
    linears: HashMap<String, DenseLinear<'a>>,
    companions: HashMap<String, Companion<'a>>,
}

/// The exported names a rank binds: every native linear, transform and glue
/// program of the build's `exlr8_dense` table.
fn exported(programs: &Programs<'_>) -> Vec<String> {
    programs.names().filter(|n| n.starts_with("exlr8_dense_") && *n != "exlr8_dense_fp8_row_embedding"
        && *n != "exlr8_dense_fp8_row_lm_head").map(str::to_owned).collect()
}

impl<'a> NativePrograms<'a> {
    /// Binds on the current device (runtime SM count) before any capture.
    pub(crate) fn bind(programs: &'a Programs<'a>, manifest: &serde_json::Value) -> Result<Self> {
        let entries: HashMap<&str, &serde_json::Value> = manifest["programs"].as_array().context("manifest programs")?
            .iter().filter_map(|e| Some((e["name"].as_str()?, e))).collect();
        let (mut linears, mut companions) = (HashMap::new(), HashMap::new());
        for name in exported(programs) {
            let entry = entries.get(name.as_str()).with_context(|| format!("{name}: no manifest entry"))?;
            match cuteafd_ffi::native_dense_glue::op_of(entry)? {
                Some(op) => { companions.insert(name.clone(), Companion::bind(programs, &name, op)?); }
                None => { linears.insert(name.clone(), DenseLinear::bind(programs, &name)?); }
            }
        }
        ensure!(!linears.is_empty() && !companions.is_empty(), "the build carries no native dense programs");
        Ok(Self { linears, companions })
    }

    fn linear(&self, name: &str) -> Result<&DenseLinear<'a>> {
        self.linears.get(name).with_context(|| format!("native dense program {name} is not in this build"))
    }

    fn companion(&self, name: &str) -> Result<&Companion<'a>> {
        self.companions.get(name).with_context(|| format!("native dense program {name} is not in this build"))
    }
}

/// Per-lane native scratch: one rotated input, the trellis core's pointer
/// table/scales/partials, an F32 accumulator and BF16 intermediates.
pub(crate) struct NativeScratch<'a> {
    rotated: Dev<'a>,
    table: Dev<'a>,
    block_scales: Dev<'a>,
    partial: Dev<'a>,
    acc: Dev<'a>,
    qkv: Dev<'a>,
    q: Dev<'a>,
    gate: Dev<'a>,
    up: Dev<'a>,
    hidden: Dev<'a>,
    kw: Dev<'a>,
    iq: Dev<'a>,
    values: Dev<'a>,
}

/// One GPU's prefill workspace for the trellis unpack baseline: the largest
/// trellis payload of any native layer unpacked to E4M3 `[N, K]` (transient,
/// rewritten by every projection) and the all-ones 128x128 grid.
///
/// This owner is the per-GPU prefill route slot. A tensor-core MXFP8 route
/// replaces it: it holds instead one shared MXFP8 activation record
/// ([`mx8_activation_bytes`] at the largest K a projection reads), reused by
/// gate/up and every projection that shares an input, beside the per-lane
/// F16/BF16 rotated inputs in [`NativeScratch`]. Admission charges whichever
/// route is active (`GlmInputs::native_dense`).
pub(crate) struct Unpacked<'a> {
    weight: Dev<'a>,
    ones: Dev<'a>,
}

/// Bytes a lane's [`NativeScratch`] takes at `rows`, and the [`Unpacked`] bytes
/// of a GPU whose largest trellis weight is `[n, k]` (admission charges both).
pub(crate) fn scratch_bytes(cfg: &GlmDsaConfig, heads: usize, inter: usize, rows: usize) -> usize {
    let (h, q, ih) = (cfg.hidden, cfg.q_lora_rank, cfg.index_heads);
    let max_k = [h, q, heads * cfg.v_head_dim, inter].into_iter().max().unwrap_or(0);
    let max_n = [heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim), h, inter, ih * 128].into_iter().max().unwrap_or(0);
    let splits = max_k.div_ceil(1024);
    [rows * max_k * 2, 64, max_k / 128 * 2, rows.min(NATIVE_DECODE_ROWS) * splits * max_n * 4, rows * max_n * 4,
        rows * (q + cfg.kv_lora_rank + cfg.qk_rope_head_dim) * 2, rows * heads * 256 * 2, rows * inter * 2,
        rows * inter * 2, rows * inter * 2, rows * (128 + ih) * 2, rows * ih * 128 * 2, rows * heads * cfg.v_head_dim * 2]
        .into_iter().map(|b| b.max(256).next_multiple_of(256)).sum()
}

/// Bytes of one MXFP8 activation record of `rows` rows over `k` input
/// features: E4M3 values plus one UE8M0 scale per 32 (the MX8 prefill route's
/// per-GPU workspace; not allocated by the unpack baseline).
#[allow(dead_code)]
pub(crate) fn mx8_activation_bytes(rows: usize, k: usize) -> usize {
    rows * k + rows * k.div_ceil(32)
}

pub(crate) fn unpacked_bytes(n: usize, k: usize) -> usize {
    (n * k).max(256).next_multiple_of(256) + (n.div_ceil(128) * (k / 128) * 4).max(256).next_multiple_of(256)
}

impl<'a> NativeScratch<'a> {
    pub(crate) fn new(engine: &GlmEngine<'a>, heads: usize, inter: usize, rows: usize) -> Result<Self> {
        let _scope = cuteafd_ffi::memory_ledger::scope("native-dense-scratch");
        let cfg = &engine.cfg;
        let (h, q, ih) = (cfg.hidden, cfg.q_lora_rank, cfg.index_heads);
        let max_k = [h, q, heads * cfg.v_head_dim, inter].into_iter().max().unwrap_or(0);
        let max_n = [heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim), h, inter, ih * 128].into_iter().max().unwrap_or(0);
        let a = |bytes: usize| engine.alloc(bytes);
        let scratch = Self { rotated: a(rows * max_k * 2)?, table: a(64)?, block_scales: a(max_k / 128 * 2)?,
            // Split-K partials serve only the trellis decode core (<= 64 rows); prefill
            // trellis takes the unpack route and the scaled linears use no partials.
            partial: a(rows.min(NATIVE_DECODE_ROWS) * max_k.div_ceil(1024) * max_n * 4)?, acc: a(rows * max_n * 4)?,
            qkv: a(rows * (q + cfg.kv_lora_rank + cfg.qk_rope_head_dim) * 2)?, q: a(rows * heads * 256 * 2)?,
            gate: a(rows * inter * 2)?, up: a(rows * inter * 2)?, hidden: a(rows * inter * 2)?,
            kw: a(rows * (128 + ih) * 2)?, iq: a(rows * ih * 128 * 2)?, values: a(rows * heads * cfg.v_head_dim * 2)? };
        // The trellis core reads rotated inputs through a one-entry pointer table (rotation id 0).
        engine.library.copy_h2d(scratch.table.buffer, &(scratch.rotated.buffer.ptr as u64).to_le_bytes())?;
        Ok(scratch)
    }
}

impl<'a> Unpacked<'a> {
    pub(crate) fn new(engine: &GlmEngine<'a>, n: usize, k: usize) -> Result<Self> {
        let _scope = cuteafd_ffi::memory_ledger::scope("native-dense-prefill-staging");
        let ones = engine.alloc(n.div_ceil(128) * (k / 128) * 4)?;
        let values: Vec<u8> = std::iter::repeat_n(1.0f32.to_le_bytes(), n.div_ceil(128) * (k / 128)).flatten().collect();
        engine.library.copy_h2d(ones.buffer, &values)?;
        Ok(Self { weight: engine.alloc(n * k)?, ones })
    }
}

/// A buffer's first `bytes`.
fn head(buffer: Buffer, bytes: usize) -> Result<Buffer> {
    ensure!(bytes <= buffer.bytes, "native view of {bytes} B past a {} B buffer", buffer.bytes);
    Ok(Buffer { bytes, ..buffer })
}

/// Exported name stem of a trellis/int4 projection shape on this rank.
fn shape_name(projection: &str, split: bool) -> String {
    let base = match projection { "gate" | "up" => "gate_up", other => other };
    if split { format!("{base}_tp2") } else { base.to_owned() }
}

impl<'a> GlmEngine<'a> {
    fn native_programs(&self, rank: usize) -> Result<&NativePrograms<'a>> {
        self.native_dense.get(rank).and_then(Option::as_ref).context("native dense programs on this rank")
    }

    fn native_scratch<'w>(&self, _rank: usize, w: &'w Workspace<'_>) -> Result<&'w NativeScratch<'w>> {
        w.native_dense.as_ref().context("lane native dense scratch")
    }

    /// `out = x @ W^T` (F32 `[rows, N]`) for native projection `projection` of
    /// `layer` on `rank`, the trellis transforms included (`out_bf16`: the
    /// output transform's BF16 rows; else F32 core output for MXFP8/int4).
    /// Returns which buffer holds the result: F32 accumulator or BF16 `bf16_out`.
    #[allow(clippy::too_many_arguments)]
    fn native_linear(&self, rank: usize, w: &Workspace<'_>, layer: &GlmLayer<'_>, projection: &str,
        x: Buffer, rows: usize, bf16_out: Buffer, split: bool) -> Result<NativeOut> {
        let programs = self.native_programs(rank)?;
        let s = self.native_scratch(rank, w)?;
        let stream = self.stream_of(rank);
        let encoding = layer.native_encoding(projection)?;
        let key = |c: &str| layer.native(&crate::families::glm5::native_dense::key(projection, c));
        match encoding {
            DenseEncoding::Mxfp8 | DenseEncoding::Int4G32 => {
                let prefix = if encoding == DenseEncoding::Mxfp8 { "mxfp8" } else { "int4" };
                let name = format!("exlr8_dense_{prefix}_{}", if encoding == DenseEncoding::Mxfp8 { projection.to_owned() }
                    else { shape_name(projection, split) });
                let linear = programs.linear(&name)?;
                let (payload, scales) = if encoding == DenseEncoding::Mxfp8 { (key("fp8")?, key("e8m0")?) }
                    else { (key("q4")?, key("s4")?) };
                let n = linear.geometry().output_features as usize;
                // SAFETY: x holds rows x K BF16, the payload/scales are this layer's resident
                // operands of the validated geometry, acc holds rows x N F32; all on this rank.
                self.on(rank, || unsafe { linear.launch(x.ptr, payload.ptr, scales.ptr, s.partial.buffer.ptr,
                    s.partial.buffer.bytes as u64, s.acc.buffer.ptr, rows as u32, 0, stream) }.map_err(Into::into))?;
                Ok(NativeOut::F32 { n })
            }
            DenseEncoding::Trellis { bits: 4 } => {
                let stem = shape_name(projection, split);
                let core = programs.linear(&format!("exlr8_dense_trellis_{stem}"))?;
                let (k, n) = (core.geometry().input_features as usize, core.geometry().output_features as usize);
                let (payload, suh, svh) = (key("trellis")?, key("suh")?, key("svh")?);
                let prefill = rows > NATIVE_DECODE_ROWS;
                let input = programs.companion(&format!("exlr8_dense_trellis_input_k{k}_{}", if prefill { "bf16" } else { "f16" }))?;
                let output = programs.companion(&format!("exlr8_dense_trellis_output_n{n}"))?;
                let rotated = head(s.rotated.buffer, rows * k * 2)?;
                // SAFETY (each launch): rank-local live buffers of the companion/linear
                // contracts at `rows`; this rank's stream orders them.
                self.on(rank, || unsafe {
                    input.launch(&[x.ptr, suh.ptr, rotated.ptr, s.block_scales.buffer.ptr], rows as u32, 0, stream)?;
                    if prefill {
                        let (unpack, gemm) = (programs.companion(&format!("exlr8_dense_unpack_{stem}"))?,
                            programs.companion(&format!("exlr8_dense_prefill_{stem}"))?);
                        let staged = self.native_unpacked(rank)?;
                        unpack.launch(&[payload.ptr, staged.weight.buffer.ptr], 0, 0, stream)?;
                        gemm.launch(&[rotated.ptr, staged.weight.buffer.ptr, staged.ones.buffer.ptr, s.acc.buffer.ptr],
                            rows as u32, 0, stream)?;
                    } else {
                        core.launch(s.table.buffer.ptr, payload.ptr, s.block_scales.buffer.ptr, s.partial.buffer.ptr,
                            s.partial.buffer.bytes as u64, s.acc.buffer.ptr, rows as u32, 0, stream)?;
                    }
                    output.launch(&[s.acc.buffer.ptr, svh.ptr, head(bf16_out, rows * n * 2)?.ptr], rows as u32, 0, stream)?;
                    Ok::<_, anyhow::Error>(())
                })?;
                Ok(NativeOut::Bf16 { n })
            }
            other => anyhow::bail!("{projection}: no native GLM dense route for {other:?}"),
        }
    }

    /// F32 accumulator rows narrowed (RNE) into BF16 `out` (contiguous `[rows, n]`).
    fn native_narrow(&self, rank: usize, w: &Workspace<'_>, rows: usize, n: usize, out: Buffer) -> Result<()> {
        let s = self.native_scratch(rank, w)?;
        // SAFETY: acc holds rows x n F32, out rows x n BF16, both on this rank's stream.
        self.on(rank, || unsafe { self.library.cuda_f32_to_bf16_async(head(s.acc.buffer, rows * n * 4)?,
            head(out, rows * n * 2)?, rows * n, self.stream_of(rank)) })
    }

    /// Native MLA producer: q_a | kv_a -> rank_pack (KV record, q_resid) ->
    /// q_b -> absorb + query RoPE into `query` (rows `query_pitch` apart under context).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn native_producer(&self, rank: usize, w: &Workspace<'_>, layer: &GlmLayer<'_>, rows: usize,
        cap: &str, kv: Buffer, query: *mut c_void, context: bool) -> Result<()> {
        let s = self.native_scratch(rank, w)?;
        let programs = self.native_programs(rank)?;
        let stream = self.stream_of(rank);
        let (_, _, cos_sin) = self.rank_caches(rank, 0);
        let x = head(w.x.buffer, rows * self.cfg.hidden * 2)?;
        for (projection, pack) in [("q_a", "exlr8_dense_f32_narrow_pack_n2048_p2624"), ("kv_a", "exlr8_dense_f32_narrow_pack_n576_p2624")] {
            self.native_linear(rank, w, layer, projection, x, rows, s.q.buffer, false)?;
            let c0 = if projection == "q_a" { 0 } else { self.cfg.q_lora_rank as u32 };
            // SAFETY: acc holds the projection's F32 rows; qkv holds rows x 2624 BF16.
            self.on(rank, || unsafe { programs.companion(pack)?.launch(&[s.acc.buffer.ptr, s.qkv.buffer.ptr],
                rows as u32, c0, stream).map_err(anyhow::Error::from) })?;
        }
        let prefix = if layer.attention_split { "glm2" } else { "glm" };
        self.run_on(rank, &format!("{prefix}_stage_rank_pack_{cap}"), &[("qkv", s.qkv.buffer.ptr),
            ("q_norm", layer.ptr("q_a_norm")?), ("kv_norm", layer.ptr("kv_a_norm")?),
            ("positions", w.positions.buffer.ptr), ("slots", w.slots.buffer.ptr), ("cos_sin", cos_sin.buffer.ptr),
            ("q_resid", w.q_resid.buffer.ptr), ("cache", kv.ptr)], &[Scalar::I32(rows as i32)])?;
        let q_resid = head(w.q_resid.buffer, rows * self.cfg.q_lora_rank * 2)?;
        let heads = self.cfg.heads / if layer.attention_split { 2 } else { 1 };
        let q_bytes = rows * heads * 256 * 2;
        match self.native_linear(rank, w, layer, "q_b", q_resid, rows, s.q.buffer, layer.attention_split)? {
            NativeOut::F32 { n } => self.native_narrow(rank, w, rows, n, head(s.q.buffer, q_bytes)?)?,
            NativeOut::Bf16 { .. } => {}
        }
        let suffix = if context { "_context" } else { "" };
        self.run_on(rank, &format!("{prefix}_stage_absorb{suffix}_{cap}"), &[("a", s.q.buffer.ptr),
            ("w", layer.ptr("w_uk_fp8")?), ("scale", layer.ptr("w_uk_scale")?), ("out", query)], &[Scalar::I32(rows as i32)])?;
        self.run_on(rank, &format!("{prefix}_stage_query_rope{suffix}_{cap}"), &[("q", s.q.buffer.ptr),
            ("positions", w.positions.buffer.ptr), ("cos_sin", cos_sin.buffer.ptr), ("query", query)], &[Scalar::I32(rows as i32)])
    }

    /// Native DSA index producer: index_q over q_resid, index_k | weights over x,
    /// then index_post (keys into `index_cache`, FP8 query, head weights).
    pub(super) fn native_index_producer(&self, rank: usize, w: &Workspace<'_>, layer: &GlmLayer<'_>, rows: usize,
        cap: &str, index_cache: Buffer) -> Result<()> {
        let s = self.native_scratch(rank, w)?;
        let programs = self.native_programs(rank)?;
        let stream = self.stream_of(rank);
        let (_, _, cos_sin) = self.rank_caches(rank, 0);
        let ih = self.cfg.index_heads;
        let q_resid = head(w.q_resid.buffer, rows * self.cfg.q_lora_rank * 2)?;
        let n = match self.native_linear(rank, w, layer, "index_q", q_resid, rows, s.iq.buffer, false)? {
            NativeOut::F32 { n } => n, NativeOut::Bf16 { .. } => anyhow::bail!("index_q is MXFP8") };
        self.native_narrow(rank, w, rows, n, s.iq.buffer)?;
        let x = head(w.x.buffer, rows * self.cfg.hidden * 2)?;
        self.native_linear(rank, w, layer, "index_k", x, rows, s.q.buffer, false)?;
        // SAFETY (both): acc holds the F32 rows; kw holds rows x (128 + I) BF16; weights_proj is resident.
        self.on(rank, || unsafe { programs.companion("exlr8_dense_f32_narrow_pack_n128_p160")?
            .launch(&[s.acc.buffer.ptr, s.kw.buffer.ptr], rows as u32, 0, stream).map_err(anyhow::Error::from) })?;
        self.on(rank, || unsafe { programs.companion("exlr8_dense_bf16_index_weights")?
            .launch(&[x.ptr, layer.native(&crate::families::glm5::native_dense::key("index_w", "bf16"))?.ptr, s.acc.buffer.ptr],
                rows as u32, 0, stream)?;
            programs.companion("exlr8_dense_f32_narrow_pack_n32_p160")?
                .launch(&[s.acc.buffer.ptr, s.kw.buffer.ptr], rows as u32, 128, stream).map_err(anyhow::Error::from) })?;
        self.run_on(rank, &format!("glm_stage_index_post_{cap}"), &[("iq", s.iq.buffer.ptr), ("kw", s.kw.buffer.ptr),
            ("positions", w.positions.buffer.ptr), ("slots", w.slots.buffer.ptr), ("cos_sin", cos_sin.buffer.ptr),
            ("k_weight", layer.ptr("k_norm_w")?), ("k_bias", layer.ptr("k_norm_b")?), ("q_fp8", w.q_fp8.buffer.ptr),
            ("head_weights", w.head_weights.buffer.ptr), ("cache", index_cache.ptr)], &[Scalar::I32(rows as i32)])?;
        let _ = ih;
        Ok(())
    }

    /// Native W_UV + o_proj of this rank's heads over `attn` into `delta` (BF16;
    /// a rank's svh-transformed partial under a head split).
    pub(super) fn native_o_proj(&self, rank: usize, w: &Workspace<'_>, layer: &GlmLayer<'_>, rows: usize, cap: &str)
        -> Result<()> {
        let s = self.native_scratch(rank, w)?;
        let prefix = if layer.attention_split { "glm2" } else { "glm" };
        self.run_on(rank, &format!("{prefix}_stage_values_{cap}"), &[("a", w.attn.buffer.ptr),
            ("w", layer.ptr("w_uv_fp8")?), ("scale", layer.ptr("w_uv_scale")?), ("out", s.values.buffer.ptr)],
            &[Scalar::I32(rows as i32)])?;
        let heads = self.cfg.heads / if layer.attention_split { 2 } else { 1 };
        let values = head(s.values.buffer, rows * heads * self.cfg.v_head_dim * 2)?;
        match self.native_linear(rank, w, layer, "o", values, rows, w.delta.buffer, layer.attention_split)? {
            NativeOut::F32 { n } => self.native_narrow(rank, w, rows, n, w.delta.buffer),
            NativeOut::Bf16 { .. } => Ok(()),
        }
    }

    /// Native dense SwiGLU MLP (layers before the first MoE layer) into `out`.
    pub(super) fn native_ffn(&self, rank: usize, w: &Workspace<'_>, layer: &GlmLayer<'_>, rows: usize, out: Buffer)
        -> Result<()> {
        let s = self.native_scratch(rank, w)?;
        let programs = self.native_programs(rank)?;
        let x = head(w.x.buffer, rows * self.cfg.hidden * 2)?;
        let inter = self.cfg.dense_intermediate / if layer.split { 2 } else { 1 };
        for (projection, dst) in [("gate", s.gate.buffer), ("up", s.up.buffer)] {
            match self.native_linear(rank, w, layer, projection, x, rows, dst, layer.split)? {
                NativeOut::Bf16 { .. } => {}
                NativeOut::F32 { .. } => anyhow::bail!("dense {projection} is trellis"),
            }
        }
        // SAFETY: gate/up/hidden hold rows x inter BF16 on this rank's stream.
        self.on(rank, || unsafe { programs.companion(&format!("exlr8_dense_swiglu_split_i{inter}"))?
            .launch(&[s.gate.buffer.ptr, s.up.buffer.ptr, s.hidden.buffer.ptr], rows as u32, 0, self.stream_of(rank))
            .map_err(anyhow::Error::from) })?;
        match self.native_linear(rank, w, layer, "down", head(s.hidden.buffer, rows * inter * 2)?, rows, out, layer.split)? {
            NativeOut::Bf16 { .. } => Ok(()),
            NativeOut::F32 { .. } => anyhow::bail!("dense down is trellis"),
        }
    }
}

/// Where a native projection left its rows.
pub(super) enum NativeOut {
    /// F32 `[rows, n]` in the lane accumulator (MXFP8 / int4).
    F32 { n: usize },
    /// BF16 `[rows, n]` in the caller's output (trellis after its output transform).
    Bf16 { #[allow(dead_code)] n: usize },
}

/// Trellis geometries a GLM rank's native layers use (for admission and binding checks).
pub(crate) fn trellis_geometries(split: bool) -> Vec<(&'static str, DenseTrellisGeometry)> {
    let g = |input, output, side| DenseTrellisGeometry { input, output, bits: 4, side, rows: NATIVE_DECODE_ROWS,
        input_dtype: TrellisDType::Bf16, output_dtype: TrellisDType::Bf16 };
    if split {
        vec![("q_b_tp2", g(2048, 8192, 1)), ("o_tp2", g(8192, 6144, 2)), ("gate_up_tp2", g(6144, 6144, 1)),
            ("down_tp2", g(6144, 6144, 2))]
    } else {
        vec![("q_b", g(2048, 16384, 1)), ("o", g(16384, 6144, 2)), ("gate_up", g(6144, 12288, 1)), ("down", g(12288, 6144, 2))]
    }
}

#[cfg(test)]
mod charge_tests {
    use super::*;

    /// GLM 5.3 head split: the split-K partials are decode-sized at every lane
    /// width, so a 4096-row prefill lane charges no gigabyte-class partials.
    #[test]
    fn prefill_lane_scratch_charges_decode_sized_partials() {
        use cuteafd_loader::families::glm5::GlmIndexer;
        let cfg = GlmDsaConfig { vocab_size: 154880, hidden: 6144, layers: 78, heads: 64, q_lora_rank: 2048,
            kv_lora_rank: 512, qk_nope_head_dim: 192, qk_rope_head_dim: 64, v_head_dim: 256,
            index_heads: 32, index_head_dim: 128, index_topk: 2048, indexers: vec![GlmIndexer::Full; 78],
            first_moe_layer: 3, dense_intermediate: 12288, experts: 256, topk: 8, moe_intermediate: 2048,
            shared_experts: 1, routed_scale: 2.5, rope_theta: 8000000.0, rms_norm_eps: 1e-5,
            mtp_layers: 1, eos_tokens: vec![154820] };
        let (heads, inter) = (cfg.heads / 2, cfg.dense_intermediate / 2);
        let decode = scratch_bytes(&cfg, heads, inter, NATIVE_DECODE_ROWS);
        let prefill = scratch_bytes(&cfg, heads, inter, 4096);
        // Rows-proportional terms only: no 1 GiB (4096 x 8 x 8192 x 4) partial.
        assert!(prefill < 640 << 20, "prefill lane scratch {prefill}");
        assert!(decode < 32 << 20, "decode scratch {decode}");
    }
}
