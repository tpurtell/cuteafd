//! GLM 5.3 attention producers, o_proj and dense FFNs over an EXLR8 checkpoint's
//! own dense operands (`weights::GlmLayer::native`), composed from the shared
//! native dense linears and the format-neutral GLM attention stages. The
//! official fused programs are untouched; a native layer replaces exactly the
//! four entry points (`producer`, `index_producer`, `o_proj`, `ffn_on`) and
//! writes the same outputs (KV record, `query`, `q_resid`, FP8 index query and
//! head weights, `delta`, FFN output) into the same workspace buffers.
//!
//! Trellis dense prefill uses BF16 activations/MMA over exact E4M3 weights;
//! MXFP8 and int4 retain their native BF16-activation kernels. No activation
//! quantization records are admitted unless A8 prefill is explicitly selected.
//!
//! Trellis transforms: `rotated = H128(x * suh)` before the core, `H128(acc) *
//! svh` after. Split projections apply svh per rank (o by input columns, down
//! by intermediate) before the existing BF16 partial exchange, the order the
//! shared trellis ABI permits ("transform each rank then reduce BF16 outputs").
use super::*;
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

    /// Conservatively include every bound native route: attention-only entries
    /// may tighten exact packing, but never allow an unreviewed FFN transition.
    pub(super) fn packed_programs(&self) -> impl Iterator<Item = &str> {
        self.linears.keys().chain(self.companions.keys()).map(String::as_str)
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
    records: Option<Dev<'a>>,
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

impl<'a> NativeScratch<'a> {
    pub(crate) fn new(engine: &GlmEngine<'a>, heads: usize, inter: usize, rows: usize) -> Result<Self> {
        let _scope = cuteafd_ffi::memory_ledger::scope("native-dense-scratch");
        let cfg = &engine.cfg;
        let (h, q, ih) = (cfg.hidden, cfg.q_lora_rank, cfg.index_heads);
        let max_k = [h, q, heads * cfg.v_head_dim, inter].into_iter().max().unwrap_or(0);
        let max_n = [heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim), h, inter, ih * 128].into_iter().max().unwrap_or(0);
        let a = |bytes: usize| engine.alloc(bytes);
        let scratch = Self { rotated: a(rows * max_k * 2)?, table: a(64)?, block_scales: a(max_k / 128 * 2)?,
            records: if engine.exlr8_prefill_activations == super::super::Exlr8PrefillActivations::A8 && rows > NATIVE_DECODE_ROWS {
                Some(a(cuteafd_loader::plan::exlr8::dense::a8_scratch_bytes(cfg, heads, inter, rows))?)
            } else { None },
            // Exact trellis prefill and native scaled linears do not use partials.
            partial: a(rows.min(NATIVE_DECODE_ROWS) * max_k.div_ceil(1024) * max_n * 4)?, acc: a(rows * max_n * 4)?,
            qkv: a(rows * (q + cfg.kv_lora_rank + cfg.qk_rope_head_dim) * 2)?, q: a(rows * heads * 256 * 2)?,
            gate: a(rows * inter * 2)?, up: a(rows * inter * 2)?, hidden: a(rows * inter * 2)?,
            kw: a(rows * (128 + ih) * 2)?, iq: a(rows * ih * 128 * 2)?, values: a(rows * heads * cfg.v_head_dim * 2)? };
        // The trellis core reads rotated inputs through a one-entry pointer table (rotation id 0).
        engine.library.copy_h2d(scratch.table.buffer, &(scratch.rotated.buffer.ptr as u64).to_le_bytes())?;
        Ok(scratch)
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
        w.native_dense.as_deref().context("prefill native dense scratch")
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
                let a8 = self.exlr8_prefill_activations == super::super::Exlr8PrefillActivations::A8 && rows > NATIVE_DECODE_ROWS;
                let dtype = if rows > NATIVE_DECODE_ROWS && !a8 { "bf16" } else { "f16" };
                let suffix = if a8 { "_a8" } else if dtype == "bf16" { "_bf16" } else { "" };
                let core = programs.linear(&format!("exlr8_dense_trellis_{stem}{suffix}"))?;
                let (k, n) = (core.geometry().input_features as usize, core.geometry().output_features as usize);
                let (payload, suh, svh) = (key("trellis")?, key("suh")?, key("svh")?);
                let input = programs.companion(&format!("exlr8_dense_trellis_input_k{k}_{dtype}"))?;
                let output = programs.companion(&format!("exlr8_dense_trellis_output_n{n}"))?;
                let rotated = head(s.rotated.buffer, rows * k * 2)?;
                // SAFETY (each launch): rank-local live buffers of the companion/linear
                // contracts at `rows`; this rank's stream orders them.
                self.on(rank, || unsafe {
                    input.launch(&[x.ptr, suh.ptr, rotated.ptr, s.block_scales.buffer.ptr], rows as u32, 0, stream)?;
                    if a8 {
                        let records = s.records.as_ref().context("A8 prefill lane records")?.buffer;
                        programs.companion(&format!("exlr8_dense_quant_k{k}_a8"))?
                            .launch(&[s.table.buffer.ptr, records.ptr], rows as u32, 0, stream)?;
                        core.launch(s.table.buffer.ptr, payload.ptr, s.block_scales.buffer.ptr, records.ptr,
                            records.bytes as u64, s.acc.buffer.ptr, rows as u32, 0, stream)?;
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

impl<'a> GlmEngine<'a> {
    fn core_programs(&self, rank: usize) -> Result<&super::super::core_runtime::CorePrograms<'a>> {
        self.core_programs.get(rank).and_then(Option::as_ref).context("GLM K6 programs not bound")
    }

    fn core_scratch<'w>(&self, w: &'w Workspace<'_>) -> Result<&'w super::super::core_runtime::CoreScratch<'w>> {
        w.core.as_deref().context("GLM K6 scratch not admitted")
    }

    pub(super) fn core_split(layer: &GlmLayer<'_>, key: &str) -> bool {
        layer.core.get(key).is_some_and(|weight|
            weight.side == cuteafd_loader::formats::dense_trellis::TrellisSide::Input)
    }

    /// Native cores occupy a disjoint partition of each existing receive slot:
    /// the following routed/BF16 publication must not overwrite a live core.
    fn core_received(&self, rank: usize, slot: usize, rows: usize) -> Result<Buffer> {
        let exchange = self.exchange()?;
        let offset = self.prefill_rows.max(DECODE_ROWS) * self.cfg.hidden * self.tp2_exchange.element_bytes();
        let bytes = rows * self.cfg.hidden * 2;
        ensure!(offset + bytes <= exchange.slot_bytes(), "GLM K6 peer core exceeds admitted receive slot");
        let base = Buffer { ptr: exchange.recv(rank, slot)?, bytes: exchange.slot_bytes(),
            device_id: exchange.device(rank), flags: 0 };
        crate::shared::prefix::view(base, offset, bytes)
    }

    pub(super) fn publish_projection(&self, rank: usize, index: usize, lane: usize,
        w: &Workspace<'_>, layer: &GlmLayer<'_>, key: &str, rows: usize) -> Result<()> {
        let at = slot(index, key == "down", lane);
        if Self::core_split(layer, key) {
            let core = &w.core_rows.as_ref().context("GLM K6 retained core rows")?[usize::from(key == "down")];
            self.exchange()?.push_to(rank, at, core.buffer.ptr,
                self.core_received(1 - rank, at, rows)?.ptr, rows * self.cfg.hidden * 2)
        } else {
            self.exchange()?.push(rank, at, w.delta.buffer.ptr, rows * self.cfg.hidden * 2)
        }
    }

    fn reduce_projection(&self, rank: usize, index: usize, lane: usize, w: &Workspace<'_>,
        layer: &GlmLayer<'_>, key: &str, rows: usize, output: Buffer) -> Result<()> {
        let weights = layer.core.get(key).context("GLM K6 split weights")?;
        let programs = self.core_programs(rank)?;
        let retained = w.core_rows.as_ref().context("GLM K6 retained core rows")?;
        let local = retained[usize::from(key == "down")].buffer;
        let peer = self.core_received(rank, slot(index, key == "down", lane), rows)?;
        // Widen retained BF16 cores for rank-ordered F32 addition, then run
        // the one complete H128/svh egress.
        let (first, second) = if rank == 0 { (local, peer) } else { (peer, local) };
        self.on(rank, || programs.projection(weights)?.reduced(&programs.add, rows,
            first, second, retained[2].buffer, weights, output, self.stream_of(rank)))
    }

    /// The caller has consumed the projection's publication on this stream.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn projection_norm(&self, rank: usize, index: usize, lane: usize, w: &Workspace<'_>,
        layer: &GlmLayer<'_>, key: &str, weight: *mut c_void, rows: Scalar,
        first: *mut c_void, second: *mut c_void) -> Result<()> {
        if Self::core_split(layer, key) {
            self.reduce_projection(rank, index, lane, w, layer, key, scalar_rows(rows)?, w.delta.buffer)?;
            self.norm_full(rank, w, weight, 1, rows, w.delta.buffer.ptr, w.delta.buffer.ptr)
        } else { self.norm_full(rank, w, weight, 2, rows, first, second) }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn ffn_norm(&self, rank: usize, index: usize, lane: usize, w: &Workspace<'_>,
        layer: &GlmLayer<'_>, weight: *mut c_void, rows: Scalar,
        first: *mut c_void, second: *mut c_void) -> Result<()> {
        if layer.dense { self.projection_norm(rank, index, lane, w, layer, "down", weight, rows, first, second) }
        else { self.norm_full(rank, w, weight, 2, rows, first, second) }
    }

    pub(super) fn meet_shared_core(&self, rank: usize, index: usize, w: &Workspace<'_>,
        layer: &GlmLayer<'_>, rows: usize) -> Result<()> {
        if !Self::core_split(layer, "down") { return Ok(()); }
        ensure!(!layer.dense, "GLM shared meet used for a dense FFN");
        self.publish_projection(rank, index, w.lane, w, layer, "down", rows)?;
        self.exchange()?.wait(rank, slot(index, true, w.lane))?;
        if rank == 0 {
            self.reduce_projection(rank, index, w.lane, w, layer, "down", rows, w.shared.buffer)
        } else {
            // SAFETY: only rank 0 contributes shared; this rank's combine reads
            // its zeroed BF16 buffer after this operation on the same stream.
            self.on(rank, || unsafe { self.library.cuda_zero_bytes_async(w.shared.buffer,
                rows * self.cfg.hidden * 2, self.stream_of(rank)) })
        }
    }

    /// Input-side projections narrow F32 cores to retained BF16 before any
    /// following projection reuses the accumulator. Their caller owns the meet.
    fn core_linear(&self, rank: usize, w: &Workspace<'_>, layer: &GlmLayer<'_>, key: &str,
        input: Buffer, output: Buffer, rows: usize, retained: Option<Buffer>) -> Result<()> {
        let weights = layer.core.get(key).with_context(|| format!("GLM K6 missing {key}"))?;
        let programs = self.core_programs(rank)?;
        let scratch = self.core_scratch(w)?;
        let projection = programs.projection(weights)?;
        let stream = self.stream_of(rank);
        self.on(rank, || {
            if weights.side == cuteafd_loader::formats::dense_trellis::TrellisSide::Input {
                let core = projection.core(rows, input, weights, &scratch.projection, stream)?;
                let retained = retained.context("GLM K6 split core has no retained row storage")?;
                crate::shared::dense_trellis::retain_partial(&programs.narrow, core, retained,
                    rows * weights.shape[0], stream)
            } else { projection.local(rows, input, weights, &scratch.projection, output, stream) }
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn core_producer(&self, rank: usize, w: &Workspace<'_>, layer: &GlmLayer<'_>, rows: usize,
        cap: &str, kv: Buffer, query: *mut c_void, context: bool) -> Result<()> {
        let scratch = self.core_scratch(w)?;
        let programs = self.core_programs(rank)?;
        let (q, qkv) = (scratch.stage("q")?, scratch.stage("qkv")?);
        let stream = self.stream_of(rank);
        let pitch = self.cfg.q_lora_rank + self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim;
        for (key, logical, column) in [("q_a", self.cfg.q_lora_rank, 0),
            ("kv_a", self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim, self.cfg.q_lora_rank)] {
            let stored = layer.core[key].shape[0];
            self.core_linear(rank, w, layer, key, w.x.buffer, q, rows, None)?;
            let (source_pitch, destination_pitch, width, offset) = super::super::core_runtime::pack_layout(
                rows, stored, logical, pitch, column)?;
            let destination = crate::shared::prefix::view(qkv, offset, qkv.bytes - offset)?;
            // SAFETY: the complete stored-N output transform precedes this
            // logical crop, with checked disjoint rank-local pitched extents.
            self.on(rank, || unsafe { programs.copy.launch_rows(destination, q, width, rows,
                destination_pitch, source_pitch, stream) })?;
        }
        let (_, _, cos_sin) = self.rank_caches(rank, 0);
        let prefix = if layer.attention_split { "glm2" } else { "glm" };
        let row = Scalar::I32(rows as i32);
        self.run_on(rank, &format!("{prefix}_stage_rank_pack_{cap}"), &[("qkv", qkv.ptr),
            ("q_norm", layer.ptr("q_a_norm")?), ("kv_norm", layer.ptr("kv_a_norm")?),
            ("positions", w.positions.buffer.ptr), ("slots", w.slots.buffer.ptr), ("cos_sin", cos_sin.buffer.ptr),
            ("q_resid", w.q_resid.buffer.ptr), ("cache", kv.ptr)], &[row])?;
        self.core_linear(rank, w, layer, "q_b", w.q_resid.buffer, q, rows, None)?;
        let suffix = if context { "_context" } else { "" };
        self.run_on(rank, &format!("{prefix}_stage_absorb_bf16{suffix}_{cap}"), &[("a", q.ptr),
            ("w", layer.ptr("w_uk")?), ("out", query)], &[row])?;
        self.run_on(rank, &format!("{prefix}_stage_query_rope{suffix}_{cap}"), &[("q", q.ptr),
            ("positions", w.positions.buffer.ptr), ("cos_sin", cos_sin.buffer.ptr), ("query", query)], &[row])
    }

    /// Mistral Large 4's K6 producer: the direct query `wq(x)` scaled per head
    /// (RoPE on RoPE layers, the rotated tail written into the absorbed query),
    /// its nope part absorbed through W_UK; on latent owners `wkv_a(x)` cropped
    /// from its padded store and packed into the layer's FP8 record.
    #[allow(clippy::too_many_arguments)]
    /// `query` receives the rows' absorbed query of this rank's heads: `w.query` itself, or
    /// (`context`) its share of the combined two-GPU query, token rows 2 x heads apart.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ml4_core_producer(&self, rank: usize, w: &Workspace<'_>, index: usize, layer: &GlmLayer<'_>,
        rows: usize, cap: &str, kv: Buffer, query: *mut c_void, context: bool) -> Result<()> {
        let scratch = self.core_scratch(w)?;
        let programs = self.core_programs(rank)?;
        let (q, qkv) = (scratch.stage("q")?, scratch.stage("qkv")?);
        let stream = self.stream_of(rank);
        let rope = self.cfg.rope(index);
        let tag = if rope { "" } else { "_nope" };
        let row = Scalar::I32(rows as i32);
        let (_, _, cos_sin) = self.rank_caches(rank, 0);
        // A head split's share runs its 32-head programs (`ml42_*`); `_context` places it in
        // the combined query (NoPE query scale writes no tail, so it never needs one).
        let family = self.ml4_attention_family(layer);
        ensure!(!context || layer.attention_split, "a context query share needs a head-split layer");
        let placed = if context { "_context" } else { "" };
        let scale_placed = if context && rope { "_context" } else { "" };
        self.core_linear(rank, w, layer, "q_b", w.x.buffer, q, rows, None)?;
        self.run_on(rank, &format!("{family}_query_scale{tag}{scale_placed}_{cap}"), &[("q", q.ptr),
            ("positions", w.positions.buffer.ptr), ("cos_sin", cos_sin.buffer.ptr),
            ("s", layer.ptr("query_scale_s")?), ("b", layer.ptr("query_scale_b")?), ("query", query)], &[row])?;
        self.run_on(rank, &format!("{family}_absorb{tag}{placed}_{cap}"), &[("a", q.ptr), ("w", layer.ptr("w_uk")?),
            ("out", query)], &[row])?;
        if layer.core.contains_key("kv_a") {
            let width = self.cfg.kv_lora_rank + if rope { self.cfg.qk_rope_head_dim } else { 0 };
            let stored = layer.core["kv_a"].shape[0];
            self.core_linear(rank, w, layer, "kv_a", w.x.buffer, q, rows, None)?;
            let (source_pitch, destination_pitch, bytes, _) = super::super::core_runtime::pack_layout(
                rows, stored, width, width, 0)?;
            // SAFETY: the complete stored-N output transform precedes this crop
            // into the packed `[rows, width]` kv rows, rank-local and disjoint.
            self.on(rank, || unsafe { programs.copy.launch_rows(qkv, q, bytes, rows,
                destination_pitch, source_pitch, stream) })?;
            self.run_on(rank, &format!("ml4_rank_pack{tag}_{cap}"), &[("kv", qkv.ptr),
                ("kv_a_norm", layer.ptr("kv_a_norm")?), ("positions", w.positions.buffer.ptr),
                ("slots", w.slots.buffer.ptr), ("cos_sin", cos_sin.buffer.ptr), ("cache", kv.ptr)], &[row])?;
        }
        Ok(())
    }

    /// Mistral Large 4's index producer: `wq_b(LayerNorm(x))` scaled per index
    /// head, `[wk(x) | weights_proj(x)]`, then the shared epilogue (RoPE skipped
    /// on NoPE layers) into the index cache, the FP8 query and head weights.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ml4_core_index_producer(&self, rank: usize, w: &Workspace<'_>, index: usize, layer: &GlmLayer<'_>,
        rows: usize, cap: &str, index_cache: Buffer) -> Result<()> {
        // The K6 core's own stages (no EXLR8 native scratch exists on this path).
        let scratch = self.core_scratch(w)?;
        let (iq, kw) = (scratch.stage("iq")?, scratch.stage("kw")?);
        let row = Scalar::I32(rows as i32);
        let (_, _, cos_sin) = self.rank_caches(rank, 0);
        let rope = self.cfg.rope(index);
        // LayerNorm(x) into the q stage, then wq_b over it.
        let normed = scratch.stage("q")?;
        self.run_on(rank, &format!("ml4_index_layer_norm_{cap}"), &[("x", w.x.buffer.ptr),
            ("weight", layer.ptr("index_q_norm_w")?), ("bias", layer.ptr("index_q_norm_b")?),
            ("out", normed.ptr)], &[row])?;
        // wq_b in the checkpoint's own form: the K6 core (the shared trellis path), or
        // the shipped FP8 block-128 weight through its native projection (never re-quantized).
        if layer.core.contains_key("index_q") {
            self.core_linear(rank, w, layer, "index_q", normed, iq, rows, None)?;
        } else {
            let fp8 = [("x", normed.ptr), ("w_iq_fp8", layer.ptr("w_iq_fp8")?),
                ("w_iq_scale", layer.ptr("w_iq_scale")?), ("out", iq.ptr), ("scratch", w.scratch.buffer.ptr)];
            self.run_on(rank, &format!("ml4_index_q_{cap}"), &fp8, &self.projection_scalars(row, cap))?;
        }
        self.run_on(rank, &format!("ml4_index_query_scale_{cap}"), &[("q", iq.ptr),
            ("positions", w.positions.buffer.ptr), ("cos_sin", cos_sin.buffer.ptr),
            ("s", layer.ptr("index_scale_s")?), ("b", layer.ptr("index_scale_b")?),
            ("query", iq.ptr)], &[row])?;
        // [wk | weights_proj](x), BF16 [rows, 128 + I], both BF16 as shipped.
        self.run_on(rank, "ml4_index_kw", &[("x", w.x.buffer.ptr), ("w", layer.ptr("w_ik")?),
            ("out", kw.ptr)], &[row])?;
        let tag = if rope { "" } else { "_nope" };
        self.run_on(rank, &format!("ml4_index_post{tag}_{cap}"), &[
            ("iq", iq.ptr), ("kw", kw.ptr), ("positions", w.positions.buffer.ptr),
            ("slots", w.slots.buffer.ptr), ("cos_sin", cos_sin.buffer.ptr), ("k_weight", layer.ptr("k_norm_w")?),
            ("k_bias", layer.ptr("k_norm_b")?), ("q_fp8", w.q_fp8.buffer.ptr),
            ("head_weights", w.head_weights.buffer.ptr), ("cache", index_cache.ptr)], &[row])
    }

    pub(super) fn ml4_core_o_proj(&self, rank: usize, w: &Workspace<'_>, layer: &GlmLayer<'_>, rows: usize,
        cap: &str) -> Result<()> {
        let values = self.core_scratch(w)?.stage("values")?;
        self.run_on(rank, &format!("{}_values_{cap}", self.ml4_attention_family(layer)), &[("a", w.attn.buffer.ptr),
            ("w", layer.ptr("w_uv")?), ("out", values.ptr)], &[Scalar::I32(rows as i32)])?;
        self.core_linear(rank, w, layer, "o", values, w.delta.buffer, rows,
            w.core_rows.as_ref().map(|rows| rows[0].buffer))
    }

    pub(super) fn core_o_proj(&self, rank: usize, w: &Workspace<'_>, layer: &GlmLayer<'_>, rows: usize,
        cap: &str) -> Result<()> {
        let values = self.core_scratch(w)?.stage("values")?;
        let prefix = if layer.attention_split { "glm2" } else { "glm" };
        self.run_on(rank, &format!("{prefix}_stage_values_bf16_{cap}"), &[("a", w.attn.buffer.ptr),
            ("w", layer.ptr("w_uv")?), ("out", values.ptr)], &[Scalar::I32(rows as i32)])?;
        self.core_linear(rank, w, layer, "o", values, w.delta.buffer, rows,
            w.core_rows.as_ref().map(|rows| rows[0].buffer))
    }

    pub(super) fn core_ffn(&self, rank: usize, w: &Workspace<'_>, layer: &GlmLayer<'_>, rows: usize,
        out: Buffer) -> Result<()> {
        let scratch = self.core_scratch(w)?;
        let (gate, up, hidden) = (scratch.stage("gate")?, scratch.stage("up")?, scratch.stage("hidden")?);
        self.core_linear(rank, w, layer, "gate", w.x.buffer, gate, rows, None)?;
        self.core_linear(rank, w, layer, "up", w.x.buffer, up, rows, None)?;
        let intermediate = layer.core["gate"].shape[0];
        let name = match self.cfg.variant.ml4() {
            // Mistral Large 4's format-neutral SwiGLU (whole and two-GPU widths).
            Some(_) => format!("ml4_swiglu_i{intermediate}"),
            None => format!("{}_stage_swiglu_i{intermediate}", if layer.split { "glm2" } else { "glm" }),
        };
        self.run_on(rank, &name, &[("gate", gate.ptr),
            ("up", up.ptr), ("hidden", hidden.ptr)], &[Scalar::I32(rows as i32)])?;
        self.core_linear(rank, w, layer, "down", hidden, out, rows,
            w.core_rows.as_ref().map(|rows| rows[1].buffer))
    }
}

/// Where a native projection left its rows.
pub(super) enum NativeOut {
    /// F32 `[rows, n]` in the lane accumulator (MXFP8 / int4).
    F32 { n: usize },
    /// BF16 `[rows, n]` in the caller's output (trellis after its output transform).
    Bf16 { #[allow(dead_code)] n: usize },
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
            mtp_layers: 1, eos_tokens: vec![154820], variant: cuteafd_loader::families::glm5::GlmVariant::Glm };
        let (heads, inter) = (cfg.heads / 2, cfg.dense_intermediate / 2);
        let scratch_bytes = cuteafd_loader::plan::exlr8::dense::scratch_bytes;
        let decode = scratch_bytes(&cfg, heads, inter, NATIVE_DECODE_ROWS);
        let prefill = scratch_bytes(&cfg, heads, inter, 4096);
        // Rows-proportional terms only: no 1 GiB (4096 x 8 x 8192 x 4) partial.
        assert!(prefill < 640 << 20, "prefill lane scratch {prefill}");
        assert!(decode < 32 << 20, "decode scratch {decode}");
        let a8 = cuteafd_loader::plan::exlr8::dense::a8_scratch_bytes(&cfg, heads, inter, 4096);
        assert_eq!(a8, 256 * (8192 / 32) * 528);
        assert_eq!(cuteafd_loader::plan::exlr8::dense::a8_scratch_bytes(&cfg, heads, inter, 64), 0);
        assert!(prefill + a8 < 640 << 20);
    }
}
