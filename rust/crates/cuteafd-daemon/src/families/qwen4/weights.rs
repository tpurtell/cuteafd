//! Qwen 3.8 Flash Next coordinator weights, packed for the exported qwen4_* programs.
//!
//! Every Qwen checkpoint (BF16, FP8, NVFP4, EXL3) stores the coordinator
//! tensors in BF16. Packing (see the qwen4 program docstrings): each
//! hyper-connection site `w_di = [input_mix_weight_down; block_inject_weight]`
//! (the model's final mixer has no injection), GDN `w_in = [in_proj_qkv;
//! in_proj_z; in_proj_b; in_proj_a]` with FP32 `conv_w [10240, 4]`, `a_log`
//! and `dt_bias`, full attention `w_in = [q_proj; k_proj; v_proj;
//! indexer.index_qk_proj]`, the shared expert `w_gate_up = [gate_proj;
//! up_proj; shared_expert_gate; 15 zero rows]`, PLE `w_kv = [key_proj;
//! value_proj]` and FP32 `conv_w [10240, 4]`.
use crate::shared::memory::device::{Allocation, Device};
use cuteafd_loader::families::qwen4::weight_layout::rows_key;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::NativeLibrary;
use cuteafd_loader::plan::checkpoint::{Checkpoint, CheckpointTensor};
use cuteafd_loader::families::qwen4::resident::{self, fp8_operands, BF16_PROJECTION_OPERANDS};
use cuteafd_loader::families::qwen4::{Qwen4Attention, Qwen4Config};
use std::collections::HashMap;
use std::ffi::c_void;
use crate::shared::qwen::weights::WeightReader;

#[path = "weights_arena.rs"]
pub(super) mod arena;

pub(crate) const PREFIX: &str = "model.language_model.";
/// Rows of the shared expert's packed gate/up/gate-logit operand.
pub(crate) const SHARED_ROWS: usize = 1296;

pub(crate) struct Qwen4Layer<'a> {
    pub attention: Qwen4Attention,
    operands: HashMap<&'static str, Allocation<'a>>,
}

impl Qwen4Layer<'_> {
    pub fn ptr(&self, operand: &str) -> Result<*mut c_void> {
        Ok(self.operands.get(operand).with_context(|| format!("layer has no weight {operand}"))?.buffer.ptr)
    }

    pub fn has(&self, operand: &str) -> bool {
        self.operands.contains_key(operand)
    }

    /// Every operand by name (golden digests).
    #[cfg(test)]
    pub(crate) fn operands(&self) -> impl Iterator<Item = (&'static str, &Allocation<'_>)> {
        self.operands.iter().map(|(name, allocation)| (*name, allocation))
    }

    /// The device range of `operand`, when the layer has it.
    pub fn range(&self, operand: &str) -> Option<crate::shared::l2_prefetch::Range> {
        self.operands.get(operand).map(|a| (a.buffer.ptr.cast_const(), a.buffer.bytes))
    }

    pub fn bytes(&self) -> usize {
        self.operands.values().map(|a| a.buffer.bytes).sum()
    }

    pub fn device(&self) -> Result<i32> {
        let device = self.operands.values().next().context("Qwen layer without operands")?.device.id;
        ensure!(self.operands.values().all(|allocation| allocation.device.id == device),
            "Qwen whole-width layer operands span devices");
        Ok(device)
    }

    /// Bytes of the selectable projections, after checking that each is held in
    /// exactly one representation: BF16 only, or (`fp8`) E4M3 + scales only.
    pub fn check_single_residency(&self, fp8: bool) -> Result<usize> {
        let names: Vec<&str> = self.operands.keys().copied().collect();
        projection_residency(&names, fp8)?;
        Ok(BF16_PROJECTION_OPERANDS.iter().flat_map(|&name| {
            let (q, s) = fp8_operands(name).expect("projection operand");
            [name, q, s]
        }).filter_map(|name| self.operands.get(name)).map(|a| a.buffer.bytes).sum())
    }
}

/// Checks a layer's operand names: every selectable projection present in
/// exactly one representation (BF16, or with `fp8` its E4M3 values and scales).
pub(crate) fn projection_residency(names: &[&str], fp8: bool) -> Result<()> {
    let has = |n: &str| names.contains(&n);
    let mut found = 0;
    for name in BF16_PROJECTION_OPERANDS {
        let (q, s) = fp8_operands(name).expect("projection operand");
        let (bf16, fp8_pair) = (has(name), has(q) && has(s));
        ensure!(!(bf16 && (has(q) || has(s))), "{name} is resident as BF16 and as E4M3 ({q}/{s})");
        ensure!(has(q) == has(s), "{q} and {s} must be resident together");
        if fp8 {
            ensure!(!bf16, "--fp8-decode holds {name} as E4M3 only, but its BF16 copy is resident");
        } else {
            ensure!(!fp8_pair, "{name} is BF16 by default, but an E4M3 copy ({q}) is resident");
        }
        found += usize::from(bf16 || fp8_pair);
    }
    ensure!(found == 2, "a layer holds two selectable projections, found {found} in {names:?}");
    Ok(())
}

/// The LM head, shared by the target and the MTP drafts: exactly one resident
/// representation (the checkpoint's BF16, or an E4M3 copy with FP32 per-row x
/// 128-K scales made at load, run through `qwen4_head_fp8` in 16-row spans).
pub(crate) enum Qwen4Head<'a> {
    Bf16(Allocation<'a>),
    Fp8 { values: Allocation<'a>, scales: Allocation<'a> },
}

impl Qwen4Head<'_> {
    pub fn allocations(&self) -> Vec<&Allocation<'_>> {
        match self {
            Self::Bf16(w) => vec![w],
            Self::Fp8 { values, scales } => vec![values, scales],
        }
    }

    pub fn bytes(&self) -> usize {
        self.allocations().iter().map(|a| a.buffer.bytes).sum()
    }

    pub fn is_fp8(&self) -> bool {
        matches!(self, Self::Fp8 { .. })
    }
}

pub(crate) struct Qwen4Weights<'a> {
    pub layers: Vec<Qwen4Layer<'a>>,
    /// The final hyper-connection mixer: norm, w_down, w_up.
    pub mixer: [Allocation<'a>; 3],
    pub head: Qwen4Head<'a>,
    /// The native MTP layer (`mtp.*`), when loaded.
    pub mtp: Option<MtpWeights<'a>>,
}

impl Qwen4Weights<'_> {
    pub fn bytes(&self) -> usize {
        self.layers.iter().map(Qwen4Layer::bytes).sum::<usize>()
            + self.mtp.as_ref().map_or(0, MtpWeights::bytes)
            + self.mixer.iter().map(|a| a.buffer.bytes).sum::<usize>() + self.head.bytes()
    }

    /// Every layer (target and MTP) with its projections in the representation
    /// `fp8` selects and no other, and the head likewise: device bytes of the
    /// selectable weights as loaded (projections, head).
    pub fn check_single_residency(&self, fp8_projections: bool, fp8_head: bool) -> Result<(usize, usize)> {
        let mut projections = 0;
        for (index, layer) in self.layers.iter().chain(self.mtp.as_ref().map(|m| &m.layer)).enumerate() {
            let which = if index < self.layers.len() { format!("layer {index}") } else { "the MTP layer".into() };
            projections += layer.check_single_residency(fp8_projections).context(which)?;
        }
        ensure!(self.head.is_fp8() == fp8_head, "the LM head is {} but {} was selected",
            if self.head.is_fp8() { "FP8" } else { "BF16" }, if fp8_head { "FP8" } else { "BF16" });
        Ok((projections, self.head.bytes()))
    }
}

/// Qwen's MTP drafter (vLLM `Qwen4ExpMultiTokenPredictor`): one full-attention
/// decoder layer (`mtp.layers.0`, no PLE) fed by the `residual_linear_shared`
/// feedback of the target's pre-mixer streams and the next token's embedding
/// (shared `embed_tokens`), its own stream mixer, and the shared `lm_head`.
pub(crate) struct MtpWeights<'a> {
    pub layer: Qwen4Layer<'a>,
    /// `pre_fc_norm_hidden` [4H], `pre_fc_norm_embedding` [H], `fc_hidden`, `fc_embedding` [H, H].
    pub norm_hidden: Allocation<'a>,
    pub norm_embed: Allocation<'a>,
    pub fc_hidden: Allocation<'a>,
    pub fc_embed: Allocation<'a>,
    /// `mtp.hyper_connection_mixer`: norm, w_down, w_up. The drafts read the
    /// target's head ([`Qwen4Weights::head`]).
    pub mixer: [Allocation<'a>; 3],
}

impl MtpWeights<'_> {
    pub fn bytes(&self) -> usize {
        let [a, b, c] = &self.mixer;
        self.layer.bytes() + [&self.norm_hidden, &self.norm_embed, &self.fc_hidden, &self.fc_embed, a, b, c]
            .iter().map(|t| t.buffer.bytes).sum::<usize>()
    }
}

#[derive(Clone)]
pub(crate) struct Qwen4Loader<'a> {
    pub library: &'a NativeLibrary,
    pub checkpoint: &'a Checkpoint,
    /// Hold the GDN and attention in/out projections as E4M3 with FP32 128x128
    /// block scales only (the `qwen4_*_w8_*` programs); the BF16 source is
    /// device staging, freed once its copy is made.
    pub fp8_decode: bool,
    /// Scale rule of the E4M3 copies (projections, head).
    pub fp8_scales: crate::shared::fp8_linear::Fp8Scales,
    pub stream: *mut c_void,
    /// The coordinator's dense records: filled through the mount at once, or
    /// recorded for a push group (every transform runs on the GPU either way).
    pub dense: std::rc::Rc<crate::shared::experts::rtx::dense::DenseLoad<'a>>,
    pub(super) arenas: Option<std::rc::Rc<arena::Arenas<'a>>>,
}

impl<'a> Qwen4Loader<'a> {
    fn device(&self) -> Result<Device<'a>> {
        Ok(Device { library: self.library, id: self.library.cuda_get_device()? })
    }

    fn reader(&self) -> WeightReader<'a> {
        WeightReader { library: self.library, checkpoint: self.checkpoint }
    }

    pub fn tensor(&self, name: &str) -> Result<&CheckpointTensor> {
        self.reader().tensor(name)
    }

    pub fn raw(&self, name: &str) -> Result<(Vec<u8>, DType, Vec<usize>)> {
        self.reader().raw(name)
    }

    /// Rows `rows` of a tensor (whole rows) as a landing rectangle.
    fn rect(&self, name: &str, rows: Option<std::ops::Range<usize>>) -> Result<crate::shared::experts::rtx::dense::Rect> {
        let tensor = self.tensor(name)?;
        let total = if tensor.meta.shape.len() >= 2 { tensor.meta.shape[0] } else { 1 };
        let row_bytes = usize::try_from(tensor.meta.byte_length)? / total;
        Ok(crate::shared::experts::rtx::dense::Rect { name: name.into(), path: self.checkpoint.snapshot.join(&tensor.shard),
            base: tensor.meta.byte_offset, row_bytes, rows: rows.unwrap_or(0..total), cols: 0..row_bytes })
    }

    /// An allocation of `size` bytes (zeroed with `zero`) whose `lands` fill
    /// its first `bytes` unchanged (class 2); sizes match the mount loader's.
    fn landed(&self, name: &str, lands: Vec<(crate::shared::experts::rtx::dense::Rect, usize)>, bytes: usize, size: usize,
        zero: bool) -> Result<Allocation<'a>> {
        let device = self.device()?;
        let allocation = self.allocate(name, size)?;
        if zero { self.library.cuda_zero_bytes(allocation.buffer, allocation.buffer.bytes)?; }
        self.dense.operand(crate::shared::experts::rtx::dense::Operand { device: device.id, buffer: allocation.buffer,
            lands, bytes })?;
        Ok(allocation)
    }

    /// Records (or runs at once) one staged step on this GPU.
    fn step(&self, rects: Vec<crate::shared::experts::rtx::dense::Rect>,
        jobs: Vec<crate::shared::experts::rtx::dense::Job>) -> Result<()> {
        let device = self.device()?.id;
        self.dense.step(crate::shared::experts::rtx::dense::Step { device, rects, jobs }, self.stream)
    }

    /// BF16 matrices of `names` concatenated by rows, then `pad_rows` zero rows.
    fn rows(&self, names: &[String], pad_rows: usize) -> Result<Allocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        let (lands, bytes) = self.row_lands(names, pad_rows)?;
        // As the reader: exactly the packed bytes, zeroed (padding rows stay zero).
        let data = lands.iter().map(|(r, _)| r.bytes()).sum();
        self.landed(&names.join("+"), lands, data, bytes, true)
    }

    fn row_lands(&self, names: &[String], pad_rows: usize)
        -> Result<(Vec<(crate::shared::experts::rtx::dense::Rect, usize)>, usize)> {
        let mut lands = Vec::new();
        let (mut at, mut cols) = (0usize, None);
        for name in names {
            let tensor = self.tensor(name)?;
            let shape = &tensor.meta.shape;
            ensure!(tensor.meta.dtype == DType::Bf16 && shape.len() == 2 && shape[0] > 0
                && cols.is_none_or(|c| c == shape[1]), "Qwen packed rows require BF16 matrices sharing columns ({name})");
            cols = Some(shape[1]);
            let rect = self.rect(name, None)?;
            let bytes = rect.bytes();
            lands.push((rect, at));
            at += bytes;
        }
        let cols = cols.context("empty Qwen row concatenation")?;
        Ok((lands, at + pad_rows * cols * 2))
    }

    fn one(&self, name: &str) -> Result<Allocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        let tensor = self.tensor(name)?;
        ensure!(tensor.meta.dtype == DType::Bf16, "{name}: expected BF16, found {:?}", tensor.meta.dtype);
        let rect = self.rect(name, None)?;
        let bytes = rect.bytes();
        self.landed(name, vec![(rect, 0)], bytes, resident::native_operand_bytes(bytes), false)
    }

    /// Native FP32 values land unchanged; BF16 values widen exactly on the GPU.
    fn f32(&self, name: &str) -> Result<Allocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("f32");
        let tensor = self.tensor(name)?;
        let rect = self.rect(name, None)?;
        match tensor.meta.dtype {
            DType::F32 => { let bytes = rect.bytes(); self.landed(name, vec![(rect, 0)], bytes, resident::native_operand_bytes(bytes), false) }
            DType::Bf16 => {
                let width = rect.width();
                let out = self.allocate(name, resident::native_operand_bytes(rect.bytes() * 2))?;
                for (parts, window) in crate::shared::experts::rtx::dense::windows(&[rect], 1)? {
                    let dst = cuteafd_ffi::CuteafdDeviceBuffer {
                        // SAFETY: these rows' FP32 values lie inside `out` (twice the BF16 bytes).
                        ptr: unsafe { out.buffer.ptr.cast::<u8>().add(window.start * width * 2) }.cast(),
                        bytes: window.len() * width * 2, ..out.buffer };
                    self.step(parts, vec![crate::shared::experts::rtx::dense::Job::Widen { at: 0,
                        bytes: window.len() * width, dst }])?;
                }
                Ok(out)
            }
            ref other => anyhow::bail!("{name}: expected BF16 or FP32, found {other:?}"),
        }
    }

    /// The scale is a target-arena view. Initialize its entire native extent,
    /// including padding, before recording checkpoint conversion for push.
    fn ple_scale(&self, p: &str) -> Result<Allocation<'a>> {
        let dtype = cuteafd_loader::families::qwen4::weight_layout::ple_scale_dtype(self.checkpoint, p)?;
        let out = self.allocate(&format!("{p}.ple.scale"), 256)?;
        let mut initial = [0u8; 256];
        if dtype.is_none() { initial[..4].copy_from_slice(&1.0f32.to_le_bytes()); }
        self.library.copy_h2d(out.buffer, &initial)?;
        if let Some(dtype) = dtype {
            let name = format!("{p}.ple.ple_embedding.ngram_embedding.weight_scale");
            let rect = self.rect(&name, None)?;
            let dst = cuteafd_ffi::CuteafdDeviceBuffer { bytes: 4, ..out.buffer };
            match dtype {
                DType::F32 => self.dense.operand(crate::shared::experts::rtx::dense::Operand {
                    device: self.device()?.id, buffer: dst, lands: vec![(rect, 0)], bytes: 4 })?,
                DType::Bf16 => self.step(vec![rect], vec![crate::shared::experts::rtx::dense::Job::Widen {
                    at: 0, bytes: 2, dst }])?,
                _ => unreachable!("PLE scale headers already checked"),
            }
        }
        Ok(out)
    }

    /// The row concatenation of BF16 `names` as E4M3 values and FP32 128x128
    /// block scales, quantized on the GPU in whole 128-row windows (the BF16
    /// copy is never resident).
    fn block_fp8(&self, names: &[String], rows: usize, cols: usize) -> Result<(Allocation<'a>, Allocation<'a>)> {
        let (lands, bytes) = self.row_lands(names, 0)?;
        ensure!(bytes == resident::bf16_bytes(rows, cols), "{names:?}: BF16 [{rows}, {cols}] expected");
        let q = self.allocate(&format!("{}.fp8", names.join("+")), rows * cols)?;
        let scale = self.allocate(&format!("{}.scales", names.join("+")), resident::fp8_block_scale_bytes(rows, cols))?;
        let rects: Vec<_> = lands.into_iter().map(|(rect, _)| rect).collect();
        let kb = cols.div_ceil(128);
        for (parts, window) in crate::shared::experts::rtx::dense::windows_with_tail(&rects, 128)? {
            ensure!(window.start % 128 == 0, "FP8 block windows start on 128-row blocks");
            let n = window.len();
            let qw = cuteafd_ffi::CuteafdDeviceBuffer {
                // SAFETY: rows `window` of the E4M3 copy lie inside `q`.
                ptr: unsafe { q.buffer.ptr.cast::<u8>().add(window.start * cols) }.cast(), bytes: n * cols, ..q.buffer };
            let sw = cuteafd_ffi::CuteafdDeviceBuffer {
                // SAFETY: these rows' 128-row block scales lie inside `scale`.
                ptr: unsafe { scale.buffer.ptr.cast::<u8>().add(window.start / 128 * kb * 4) }.cast(),
                bytes: n.div_ceil(128) * kb * 4, ..scale.buffer };
            self.step(parts, vec![crate::shared::experts::rtx::dense::Job::RowQuant { rows: n, cols,
                rule: self.fp8_scales, q: qw, scale: sw, block: true }])?;
        }
        Ok((q, scale))
    }

    fn hc(&self, prefix: &str, inject: bool) -> Result<[Allocation<'a>; 3]> {
        let mut di = vec![format!("{prefix}.input_mix_weight_down.weight")];
        if inject {
            di.push(format!("{prefix}.block_inject_weight.weight"));
        }
        Ok([self.one(&format!("{prefix}.hc_norm.weight"))?, self.rows(&di, 0)?,
            self.one(&format!("{prefix}.input_mix_weight_up.weight"))?])
    }

    pub fn layer(&self, cfg: &Qwen4Config, layer: usize) -> Result<Qwen4Layer<'a>> {
        self.layer_at(cfg, &format!("{PREFIX}layers.{layer}"), cfg.attention[layer], cfg.ple_layers.contains(&layer))
    }

    /// The decoder layer under `p` (a target layer or `mtp.layers.0`).
    fn layer_at(&self, cfg: &Qwen4Config, p: &str, attention: Qwen4Attention, ple: bool) -> Result<Qwen4Layer<'a>> {
        let mut ops: HashMap<&'static str, Allocation<'a>> = HashMap::new();
        for (site, names) in [("attn_hyper_connection", ["attn.norm", "attn.w_di", "attn.w_up"]),
            ("mlp_hyper_connection", ["mlp.norm", "mlp.w_di", "mlp.w_up"])] {
            for (name, value) in names.into_iter().zip(self.hc(&format!("{p}.{site}"), true)?) {
                ops.insert(name, value);
            }
        }
        // The selectable projections' sources: BF16 operands, or with
        // `fp8_decode` their only resident copy, E4M3 with 128x128 block scales.
        let mut projections: Vec<(&'static str, Vec<String>)> = Vec::new();
        match attention {
            Qwen4Attention::Gdn => {
                let a = |name: &str| format!("{p}.linear_attn.{name}");
                projections.push(("w_in", ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]
                    .map(|n| a(&format!("{n}.weight"))).to_vec()));
                // [10240, 1, 4] -> FP32 [10240, 4].
                ops.insert("conv_w", self.f32(&a("conv1d.weight"))?);
                ops.insert("a_log", self.f32(&a("A_log"))?);
                ops.insert("dt_bias", self.f32(&a("dt_bias"))?);
                ops.insert("norm_w", self.one(&a("norm.weight"))?);
                projections.push(("w_out", vec![a("out_proj.weight")]));
            }
            Qwen4Attention::Full => {
                let a = |name: &str| format!("{p}.self_attn.{name}");
                projections.push(("w_in", vec![a("q_proj.weight"), a("k_proj.weight"), a("v_proj.weight"),
                    a("indexer.index_qk_proj.weight")]));
                ops.insert("q_norm", self.one(&a("q_norm.weight"))?);
                ops.insert("k_norm", self.one(&a("k_norm.weight"))?);
                ops.insert("iq_norm", self.one(&a("indexer.q_layernorm.weight"))?);
                ops.insert("ik_norm", self.one(&a("indexer.k_layernorm.weight"))?);
                projections.push(("w_o", vec![a("o_proj.weight")]));
            }
        }
        if self.fp8_decode {
            // One representation per weight: each projection's E4M3 copy is
            // quantized on the GPU straight from staged BF16 rows.
            for projection in resident::layer_projections(cfg, attention, None) {
                let names = &projections.iter().find(|(operand, _)| *operand == projection.operand)
                    .context("projection to quantize")?.1;
                let (q, scale) = self.block_fp8(names, projection.rows, projection.cols)
                    .with_context(|| format!("{p} {}", projection.operand))?;
                let (fp8, scales) = resident::fp8_operands(projection.operand).context("projection operand")?;
                ops.insert(fp8, q);
                ops.insert(scales, scale);
            }
        } else {
            for (operand, names) in &projections { ops.insert(operand, self.rows(names, 0)?); }
        }
        let m = |name: &str| format!("{p}.mlp.{name}");
        ops.insert("gate", self.one(&m("gate.weight"))?);
        let shared = [m("shared_expert.gate_proj.weight"), m("shared_expert.up_proj.weight"), m("shared_expert_gate.weight")];
        let packed = 2 * cfg.shared_intermediate + 1;
        ensure!(packed <= SHARED_ROWS, "shared expert rows exceed the packed operand");
        ops.insert("shared.w_gate_up", self.rows(&shared, SHARED_ROWS - packed)?);
        ops.insert("shared.w_down", self.one(&m("shared_expert.down_proj.weight"))?);
        if ple {
            let e = |name: &str| format!("{p}.ple.{name}");
            ops.insert("ple.w_kv", self.rows(&[e("key_proj.weight"), e("value_proj.weight")], 0)?);
            ops.insert("ple.norm_key", self.one(&e("norm_key.weight"))?);
            ops.insert("ple.norm_query", self.one(&e("norm_query.weight"))?);
            ops.insert("ple.norm_conv", self.one(&e("norm_conv.weight"))?);
            ops.insert("ple.conv_w", self.f32(&e("conv1d.weight"))?);
            ops.insert("ple.scale", self.ple_scale(p)?);
        }
        Ok(Qwen4Layer { attention, operands: ops })
    }

    /// Layers `0..layers` (all of them unless the caller stops early), the MTP
    /// layer with `mtp`, and the head shared by both: the checkpoint's BF16, or
    /// with `fp8_head` only an E4M3 copy (quantized on the host; the BF16 head
    /// never reaches the device).
    pub fn model(&self, cfg: &Qwen4Config, layers: usize, mtp: bool, fp8_head: bool) -> Result<Qwen4Weights<'a>> {
        let layers = layers.min(cfg.layers);
        self.model_placed(cfg, &vec![0; layers], &[(self.device()?, self.stream)], mtp, fp8_head)
    }

    /// Whole-width layers follow the global owner map; head and draft stay on the entry owner.
    pub fn model_placed(&self, cfg: &Qwen4Config, owners: &[usize],
        ranks: &[(Device<'a>, *mut c_void)], mtp: bool, fp8_head: bool) -> Result<Qwen4Weights<'a>> {
        ensure!(!ranks.is_empty() && owners.len() <= cfg.layers && owners.iter().all(|&owner| owner < ranks.len()),
            "invalid Qwen weight placement");
        ensure!(ranks.iter().all(|(device, _)| std::ptr::eq(device.library, self.library)),
            "Qwen weight placement crosses native libraries");
        if self.arenas.is_none() {
            let loader = self.with_arenas(cfg, owners, ranks, mtp, fp8_head, false)?;
            return loader.model_placed(cfg, owners, ranks, mtp, fp8_head);
        }
        let _entry = ranks[0].0.enter()?;
        let tensor = self.tensor("lm_head.weight")?;
        ensure!(tensor.meta.dtype == DType::Bf16 && tensor.meta.shape == [cfg.vocab_size, cfg.hidden],
            "lm_head must be BF16 [vocab, hidden]");
        let head = if fp8_head {
            // Per-row x 128-K E4M3 (Row128) quantized on the GPU from staged rows.
            let (rows, cols) = (cfg.vocab_size, cfg.hidden);
            ensure!(cols % 128 == 0, "the FP8 head needs whole 128-K scale blocks");
            let values = self.allocate("lm_head.weight.fp8", (rows * cols).max(256))?;
            let scales = self.allocate("lm_head.weight.scales", (rows * (cols / 128) * 4).max(256))?;
            let whole = self.rect("lm_head.weight", None)?;
            for (parts, window) in crate::shared::experts::rtx::dense::windows(&[whole], 1)? {
                let target = crate::shared::fp8_load_quant::Target {
                    // SAFETY: rows `window` of the head's values and scales lie inside them.
                    q: cuteafd_ffi::CuteafdDeviceBuffer { ptr: unsafe { values.buffer.ptr.cast::<u8>().add(window.start * cols) }
                        .cast(), bytes: window.len() * cols, ..values.buffer },
                    q_stride: cols,
                    scales: cuteafd_ffi::CuteafdDeviceBuffer { ptr: unsafe { scales.buffer.ptr.cast::<u8>()
                        .add(window.start * (cols / 128) * 4) }.cast(), bytes: window.len() * (cols / 128) * 4, ..scales.buffer },
                    kmajor: None,
                };
                self.step(parts, vec![crate::shared::experts::rtx::dense::Job::Quant { rows: 0..window.len(), pitch: cols,
                    col0: 0, cols, layout: crate::families::glm5_flash::fp8::Layout::Row128, rule: self.fp8_scales, target }])?;
            }
            tracing::info!("LM head (target and MTP) quantized to E4M3 on the GPU; no BF16 head is resident");
            Qwen4Head::Fp8 { values, scales }
        } else {
            Qwen4Head::Bf16(self.one("lm_head.weight")?)
        };
        let mtp = if mtp { Some(self.mtp(cfg)?) } else { None };
        Ok(Qwen4Weights {
            layers: owners.iter().enumerate().map(|(layer, &owner)| {
                let (device, stream) = ranks[owner];
                let loader = Qwen4Loader { library: self.library, checkpoint: self.checkpoint,
                    fp8_decode: self.fp8_decode, fp8_scales: self.fp8_scales, stream, dense: self.dense.clone(), arenas: self.arenas.clone() };
                device.run(|| loader.layer(cfg, layer))
            }).collect::<Result<_>>()?,
            mixer: self.hc(&format!("{PREFIX}hyper_connection_mixer"), false)?,
            head,
            mtp,
        })
    }

    pub fn mtp(&self, cfg: &Qwen4Config) -> Result<MtpWeights<'a>> {
        ensure!(cfg.mtp_layers == 1, "the MTP drafter runs one MTP layer (config has {})", cfg.mtp_layers);
        ensure!(self.tensor("mtp.fc_hidden.weight").is_ok(), "the checkpoint has no MTP weights (mtp.*)");
        Ok(MtpWeights {
            layer: self.layer_at(cfg, "mtp.layers.0", Qwen4Attention::Full, false)?,
            norm_hidden: self.one("mtp.pre_fc_norm_hidden.weight")?,
            norm_embed: self.one("mtp.pre_fc_norm_embedding.weight")?,
            fc_hidden: self.one("mtp.fc_hidden.weight")?,
            fc_embed: self.one("mtp.fc_embedding.weight")?,
            mixer: self.hc("mtp.hyper_connection_mixer", false)?,
        })
    }
}

#[cfg(test)]
#[path = "coordinator_golden_tests.rs"]
mod coordinator_golden_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn gdn_small_owners_match_admission_and_replay_payloads() -> Result<()> {
        use crate::shared::experts::rtx::dense::{self, DenseLoad};
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        use cuteafd_loader::plan::testing::{qwen4_config, t, write_snapshot};
        use std::os::unix::fs::FileExt;
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.configure_memory(&library, 0, 1 << 30)?;
        let mut config = qwen4_config(1);
        config["text_config"]["hidden_size"] = 128.into();
        let cfg = Qwen4Config::from_hf(&config)?;
        let p = "model.language_model.layers.0";
        let a = format!("{p}.linear_attn");
        let mut tensors = vec![
            t("lm_head.weight", "BF16", &[cfg.vocab_size, cfg.hidden]),
            t(format!("{a}.in_proj_qkv.weight"), "BF16", &[cfg.gdn_conv_width(), cfg.hidden]),
            t(format!("{a}.in_proj_z.weight"), "BF16", &[cfg.gdn_value_width(), cfg.hidden]),
            t(format!("{a}.in_proj_b.weight"), "BF16", &[48, cfg.hidden]),
            t(format!("{a}.in_proj_a.weight"), "BF16", &[48, cfg.hidden]),
            t(format!("{a}.out_proj.weight"), "BF16", &[cfg.hidden, cfg.gdn_value_width()]),
            t(format!("{a}.conv1d.weight"), "BF16", &[cfg.gdn_conv_width(), 1, 4]),
            t(format!("{a}.norm.weight"), "BF16", &[128]),
            t(format!("{p}.mlp.gate.weight"), "BF16", &[2, cfg.hidden]),
            t(format!("{p}.mlp.shared_expert.gate_proj.weight"), "BF16", &[640, cfg.hidden]),
            t(format!("{p}.mlp.shared_expert.up_proj.weight"), "BF16", &[640, cfg.hidden]),
            t(format!("{p}.mlp.shared_expert_gate.weight"), "BF16", &[1, cfg.hidden]),
            t(format!("{p}.mlp.shared_expert.down_proj.weight"), "BF16", &[cfg.hidden, 640]),
        ];
        for (prefix, inject) in [(format!("{p}.attn_hyper_connection"), true),
            (format!("{p}.mlp_hyper_connection"), true),
            ("model.language_model.hyper_connection_mixer".to_string(), false)] {
            tensors.extend([t(format!("{prefix}.hc_norm.weight"), "BF16", &[cfg.hidden]),
                t(format!("{prefix}.input_mix_weight_down.weight"), "BF16", &[4, cfg.hidden]),
                t(format!("{prefix}.input_mix_weight_up.weight"), "BF16", &[cfg.hidden, 4])]);
            if inject { tensors.push(t(format!("{prefix}.block_inject_weight.weight"), "BF16", &[4, cfg.hidden])); }
        }
        for dtype in ["BF16", "F32"] {
            let mut tensors = tensors.clone();
            tensors.extend([t(format!("{a}.A_log"), dtype, &[48]), t(format!("{a}.dt_bias"), dtype, &[48])]);
            write_snapshot(fixture.directory(), &config, &tensors, None);
            let checkpoint = Checkpoint::open(fixture.directory())?;
            for tensor in &checkpoint.tensors {
                let bytes = match tensor.meta.dtype {
                    DType::Bf16 => (0..tensor.meta.byte_length / 2).flat_map(|i|
                        (((0.125 * (1 + i % 7) as f32).to_bits() >> 16) as u16).to_le_bytes()).collect::<Vec<_>>(),
                    DType::F32 => (0..tensor.meta.byte_length / 4).flat_map(|i|
                        (0.125 * (1 + i % 7) as f32).to_le_bytes()).collect(),
                    _ => unreachable!(),
                };
                std::fs::File::options().write(true).open(checkpoint.snapshot.join(&tensor.shard))?
                    .write_all_at(&bytes, tensor.meta.byte_offset)?;
            }
            let loader = Qwen4Loader { library: &library, checkpoint: &checkpoint, fp8_decode: false,
                fp8_scales: crate::shared::fp8_linear::Fp8Scales::Amax, stream: std::ptr::null_mut(),
                dense: std::rc::Rc::new(DenseLoad::mount(&library)), arenas: None };
            let fp8_loader = Qwen4Loader { fp8_decode: true, ..loader.clone() };
            let layouts = cuteafd_loader::families::qwen4::weight_layout::coordinator_arenas(
                &checkpoint, &cfg, &[0], 1, false,
                resident::Qwen4Representation { fp8_projections: true, fp8_head: false }, 0)?;
            let payload = layouts.iter().map(|(_, layout)| layout.payload_bytes()).sum::<usize>();
            fixture.configure_memory(&library, 0, payload)?;
            let before = library.cuda_memory_info()?.0;
            let error = match fp8_loader.with_arenas(&cfg, &[0], &[(Device { library: &library, id: 0 },
                std::ptr::null_mut())], false, false, false) {
                Ok(_) => anyhow::bail!("startup admitted payload without live staging"),
                Err(error) => error,
            };
            assert!(error.to_string().contains("startup arenas and dense workspace"), "{error:#}");
            assert_eq!(library.cuda_memory_info()?.0, before, "refusal allocated a backing");
            fixture.configure_memory(&library, 0, 1 << 30)?;
            let mounted = loader.model(&cfg, 1, false, false)?;
            let admitted = resident::checkpoint_resident_bytes(&checkpoint, &cfg, 1, false,
                resident::Qwen4Representation::default())?;
            assert_eq!(mounted.bytes() as u64, admitted.target_bytes + admitted.head_bytes);
            let load = std::rc::Rc::new(DenseLoad::push(&library));
            let planning = Qwen4Loader { dense: load.clone(), ..loader };
            let (pushed, mut records) = dense::recording(|| planning.model(&cfg, 1, false, false))?;
            records.extend(load.sink.as_ref().unwrap().take_all());
            let sockets = tempfile::tempdir_in(fixture.directory())?;
            let _committed = crate::shared::experts::rtx::push::tests::receive_pushed(
                Device { library: &library, id: 0 }, None, &checkpoint.snapshot, sockets.path(), &records, None, 0, 0)?;
            assert_eq!(pushed.bytes(), mounted.bytes());
            let devices = cuteafd_loader::placement::planner::CoordinatorDevices::new(&[0])?;
            for model in [&mounted, &pushed] {
                let receipts = model.allocation_receipts(devices)?;
                let backings = receipts.iter().filter(|r| r.allocation.view.is_none()).collect::<Vec<_>>();
                assert_eq!(backings.len(), 1);
                assert!(backings[0].allocation.id.contains("qwen4/target/"));
                assert!(receipts.iter().filter(|r| r.allocation.view.is_some()).count() > 10);
                assert_eq!(backings[0].allocation.bytes, model.bytes() as u64);
            }
            for (name, allocation) in mounted.layers[0].operands() {
                let other = &pushed.layers[0].operands[name];
                // Only logical widened rows are initialized below the native min-256 extent.
                let bytes = if ["a_log", "dt_bias"].contains(&name) { 48 * 4 } else { allocation.buffer.bytes };
                let read = |a: &Allocation<'_>| -> Result<Vec<u8>> {
                    let mut out = vec![0; bytes];
                    library.copy_d2h(&mut out, cuteafd_ffi::CuteafdDeviceBuffer { bytes, ..a.buffer })?;
                    Ok(out)
                };
                assert_eq!(read(allocation)?, read(other)?, "{name}");
            }
            for key in ["a_log", "dt_bias"] {
                let mounted = &mounted.layers[0].operands[key];
                let pushed = &pushed.layers[0].operands[key];
                assert_eq!((mounted.buffer.bytes, pushed.buffer.bytes), (256, 256));
                let read = |allocation: &Allocation<'_>| -> Result<Vec<u8>> {
                    let mut bytes = vec![0; 48 * 4];
                    library.copy_d2h(&mut bytes, allocation.buffer)?;
                    Ok(bytes)
                };
                assert_eq!(read(mounted)?, read(pushed)?);
            }
        }
        Ok(())
    }

    #[test]
    fn ple_scale_is_an_owner_arena_view_with_identical_mount_and_push_bytes() -> Result<()> {
        use crate::shared::experts::rtx::dense::DenseLoad;
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        use cuteafd_loader::plan::testing::{qwen4_config, qwen4_coordinator, t, write_snapshot};
        use std::os::unix::fs::FileExt;
        let fixture = Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.quiet_events(&library, true)?;
        for gpu in [0, 1] { fixture.configure_memory(&library, gpu, 1 << 30)?; }
        let mut config = qwen4_config(2);
        config["text_config"]["hidden_size"] = 128.into();
        config["text_config"]["shared_expert_intermediate_size"] = 64.into();
        config["text_config"]["num_experts"] = 64.into();
        config["text_config"]["ple_layer_ids"] = serde_json::json!([1]);
        let cfg = Qwen4Config::from_hf(&config)?;
        let p = "model.language_model.layers.0";
        let scale_name = format!("{p}.ple.ple_embedding.ngram_embedding.weight_scale");
        for (table_dtype, scalar_dtype) in [("BF16", "BF16"), ("F8_E4M3", "BF16"), ("F8_E4M3", "F32")] {
            let mut headers = qwen4_coordinator(&config, false);
            headers.push(t(format!("{p}.ple.ple_embedding.ngram_embedding.shard_0.weight"), table_dtype, &[16, 160]));
            if table_dtype != "BF16" { headers.push(t(&scale_name, scalar_dtype, &[1])); }
            write_snapshot(fixture.directory(), &config, &headers, None);
            let checkpoint = Checkpoint::open(fixture.directory())?;
            let old_scalar = if table_dtype == "BF16" { 1.0f32 } else {
                let source = checkpoint.tensors.iter().find(|t| t.meta.name == scale_name).unwrap();
                let bytes = if scalar_dtype == "BF16" { 0x3eabu16.to_le_bytes().to_vec() }
                    else { 0.3125f32.to_le_bytes().to_vec() };
                std::fs::File::options().write(true).open(checkpoint.snapshot.join(&source.shard))?
                    .write_all_at(&bytes, source.meta.byte_offset)?;
                // The previous standalone scalar's exact conversion, not an approximation.
                if scalar_dtype == "BF16" { f32::from_bits(u32::from(u16::from_le_bytes(bytes.try_into().unwrap())) << 16) }
                    else { f32::from_le_bytes(bytes.try_into().unwrap()) }
            };
            let mut expected = vec![0; 256];
            expected[..4].copy_from_slice(&old_scalar.to_le_bytes());
            for gpu in [0, 1] {
                let device = Device { library: &library, id: gpu };
                let ranks = [(Device { library: &library, id: 0 }, std::ptr::null_mut()),
                    (Device { library: &library, id: 1 }, std::ptr::null_mut())];
                let load = std::rc::Rc::new(DenseLoad::push(&library));
                let loader = Qwen4Loader { library: &library, checkpoint: &checkpoint, fp8_decode: false,
                    fp8_scales: crate::shared::fp8_linear::Fp8Scales::Amax, stream: std::ptr::null_mut(),
                    dense: std::rc::Rc::new(DenseLoad::mount(&library)), arenas: None };
                let mounted = device.run(|| loader.with_arenas(&cfg, &[gpu as usize, 1 - gpu as usize],
                    &ranks, false, false, false)?.ple_scale(p))?;
                let planning = Qwen4Loader { dense: load.clone(), ..loader }
                    .with_arenas(&cfg, &[gpu as usize, 1 - gpu as usize], &ranks, false, false, false)?;
                let pushed = device.run(|| planning.ple_scale(p))?;
                assert_ne!(mounted.buffer.ptr, pushed.buffer.ptr);
                let records = load.sink.as_ref().unwrap().take_all();
                if table_dtype != "BF16" {
                    let sockets = tempfile::tempdir_in(fixture.directory())?;
                    crate::shared::experts::rtx::push::tests::receive_pushed(device, None,
                        &checkpoint.snapshot, sockets.path(), &records, None, 0, gpu as usize)?;
                }
                for allocation in [&mounted, &pushed] {
                    let mut actual = vec![0; 256];
                    device.run(|| library.copy_d2h(&mut actual, allocation.buffer))?;
                    assert_eq!(actual, expected, "table={table_dtype} scalar={scalar_dtype} gpu={gpu}");
                    let devices = cuteafd_loader::placement::planner::CoordinatorDevices::new(&[0, 1])?;
                    let mut inventory = crate::shared::memory::receipts::Inventory::new(devices, "ple-test",
                        cuteafd_loader::placement::planner::Lifetime::RESIDENT,
                        cuteafd_core::memory_layout::Category::Weights, "coordinator weights");
                    allocation.inventory(&mut inventory, "scale")?;
                    assert_eq!(inventory.receipts.iter().filter(|r| r.allocation.view.is_none()).count(), 1);
                    let view = inventory.receipts.iter().find(|r| r.allocation.view.is_some()).unwrap();
                    assert_eq!(view.allocation.bytes, 256);
                    assert!(inventory.receipts.iter().any(|r| r.allocation.id.contains(&format!("qwen4/target/rank{gpu}"))));
                    assert_eq!(allocation.buffer.device_id, gpu);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn a_projection_is_resident_in_exactly_one_representation() {
        projection_residency(&["w_in", "w_out", "conv_w"], false).unwrap();
        projection_residency(&["w_in", "w_o", "q_norm"], false).unwrap();
        projection_residency(&["w_in_fp8", "w_in_scale", "w_out_fp8", "w_out_scale"], true).unwrap();
        projection_residency(&["w_in_fp8", "w_in_scale", "w_o_fp8", "w_o_scale"], true).unwrap();
        // Dual copies (the old decode-only E4M3 route) are refused either way.
        for fp8 in [false, true] {
            let error = projection_residency(&["w_in", "w_in_fp8", "w_in_scale", "w_o", "w_o_fp8", "w_o_scale"], fp8)
                .unwrap_err().to_string();
            assert!(error.contains("BF16 and as E4M3"), "{error}");
        }
        assert!(projection_residency(&["w_in", "w_out"], true).is_err());
        assert!(projection_residency(&["w_in_fp8", "w_in_scale", "w_o_fp8", "w_o_scale"], false).is_err());
        assert!(projection_residency(&["w_in_fp8", "w_out_fp8", "w_out_scale"], true).is_err());
        assert!(projection_residency(&["w_in"], false).is_err());
    }
}
