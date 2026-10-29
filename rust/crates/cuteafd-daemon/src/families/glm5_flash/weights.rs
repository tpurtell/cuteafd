//! GLM 5.3 Flash coordinator weights, packed for the exported glmf_* programs.
//!
//! The MLA (`q_a|kv_a`, `q_b`, `o_proj`), dense and shared-expert projections
//! are FP8 only: the primary checkpoint's E4M3 bytes and FP32 128x128 block
//! scales, or 128x128 blocks quantized from its BF16 at load (the BF16 is
//! never resident). Every other matrix operand is BF16, except where an FP8
//! representation is selected instead: the KDA in/out projections
//! (`--kda-fp8 row128|channel`: E4M3 with per-row x 128-K scales stored
//! K-block major, `w_in_fp8`/`w_in_kscale`, `w_o_fp8`/`w_o_kscale`, no BF16
//! `w_in`/`w_o`) and the LM head (`--fp8-head`, [`super::head::GlmfHead::Fp8`]).
//! One resident copy each.
//! Packing (see the glmf program docstrings): KDA `w_in = [q; k; v; f_a; g_a;
//! b]`, `w_fg = [f_b; g_b]`, `conv_w` FP32 `[3D, 4]`; MLA `w_qkv_a = [q_a;
//! kv_a]`, `kv_b` split per head into `w_uk [N, 512, 256]` (transposed key
//! rows) and `w_uv [N, 256, 512]`; mHC `fn` widened to FP32; dense and shared
//! `w_gate_up = [gate; up]`. A ModelOpt NVFP4 dense MLP (nvidia/GLM-5.3-Flash-
//! NVFP4, layers 0-2) stays NVFP4: `nvfp4_w{1,3,2}` packed E2M1 and
//! `nvfp4_s{1,3,2}` its E4M3 scales then FP32 weight_scale_2 and input_scale,
//! the one-expert layout of the `fp8-glmfdense-nvfp4` package.
//!
//! Two-GPU head split ([`GlmfLoader::peers`]): rank `r` takes KDA heads and MLA heads
//! `r * N / 2 ..` (their in-projection, conv, gate, decay and state rows, q_b rows, W_UK / W_UV
//! heads and o_proj columns) and the matching half of the dense and shared-expert intermediate;
//! every rank the mHC weights, norms, the replicated low-rank KDA gates (`f_a`, `g_a`), the MLA
//! latent projection (`q_a | kv_a`), DSA indexer and router. Under explicit `layers`, MLA
//! operands and caches reside only on their selected owner. A ModelOpt NVFP4
//! dense MLP stays whole on rank 0 (its package has no half geometry).
//! With `kda_output_shard`, the output projection instead splits live token rows:
//! each rank owns one full FP8 `[H, D]` matrix and consumes complete gathered
//! KDA activations for its token rows. Quantization runs once before replication.
use crate::shared::experts::rtx::dense::{Job, Rect};
use crate::shared::fp8_load_quant::Target;
use crate::shared::memory::DeviceAllocation;
use crate::shared::peer_split::{Axis, RankDevice};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_loader::kda::{KdaGeometry, KdaOutputGate, KdaRole, KdaOperand, KdaError};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::families::glm5_flash::{GlmNextAttention, GlmNextConfig};
use cuteafd_loader::plan::checkpoint::{Checkpoint, CheckpointTensor};
use std::collections::HashMap;
use std::ffi::c_void;
use std::os::unix::fs::FileExt;

use crate::shared::placement::layers::AttentionShare;
use cuteafd_loader::placement::{FfnMode, LayerExecutionPlan, LayerMode};

pub(crate) const PREFIX: &str = "model.language_model.";

pub(crate) struct GlmfLayer<'a> {
    pub attention: GlmNextAttention,
    pub dense: bool,
    /// FFN width; whole attention still retains a split FFN on two GPUs.
    pub split: bool,
    pub attention_share: AttentionShare,
    pub(crate) core: std::collections::BTreeMap<String, crate::shared::dense_trellis::CoreWeights<'a>>,
    operands: HashMap<&'static str, DeviceAllocation<'a>>,
}

impl GlmfLayer<'_> {
    /// The device range of `operand`, when the layer has it.
    pub fn range(&self, operand: &str) -> Option<crate::shared::l2_prefetch::Range> {
        self.operands.get(operand).map(|a| (a.buffer.ptr.cast_const(), a.buffer.bytes))
    }

    pub fn ptr(&self, operand: &str) -> Result<*mut c_void> {
        Ok(self.operands.get(operand).with_context(|| format!("layer has no weight {operand}"))?.buffer.ptr)
    }

    /// An FP8 operand, or `fallback`'s pointer when the layer has none (the
    /// program then runs with `fp8_rows` 0 and never reads it).
    pub fn ptr_or(&self, operand: &str, fallback: &str) -> Result<*mut c_void> {
        match self.operands.get(operand) {
            Some(allocation) => Ok(allocation.buffer.ptr),
            None => self.ptr(fallback),
        }
    }

    pub fn has(&self, operand: &str) -> bool {
        self.operands.contains_key(operand)
    }

    pub fn bytes(&self) -> usize {
        self.operands.values().map(|a| a.buffer.bytes).sum::<usize>()
            + self.core.values().map(|w| w.payload.buffer.bytes + w.suh.buffer.bytes + w.svh.buffer.bytes).sum::<usize>()
    }
}

pub(crate) struct GlmfWeights<'a> {
    pub layers: Vec<GlmfLayer<'a>>,
    pub norm: DeviceAllocation<'a>,
    /// The one resident vocabulary head (BF16, or FP8 with --fp8-head).
    pub head: super::head::GlmfHead<'a>,
}

/// Resident bytes by representation of the weights with a selectable precision.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Residency {
    pub kda_bf16: usize,
    pub kda_fp8: usize,
    pub head_bf16: usize,
    pub head_fp8: usize,
}

impl GlmfWeights<'_> {
    pub fn residency(&self) -> Residency {
        let mut r = Residency::default();
        for layer in self.layers.iter().filter(|l| l.attention == GlmNextAttention::Kda) {
            r.kda_bf16 += ["w_in", "w_o"].iter().filter_map(|n| layer.range(n)).map(|(_, b)| b).sum::<usize>();
            r.kda_fp8 += ["w_in_fp8", "w_in_kscale", "w_o_fp8", "w_o_kscale"].iter()
                .filter_map(|n| layer.range(n)).map(|(_, b)| b).sum::<usize>();
        }
        match &self.head {
            super::head::GlmfHead::Bf16(w) => r.head_bf16 = w.buffer.bytes,
            head => r.head_fp8 = head.bytes(),
        }
        r
    }

    /// Single residency: no KDA layer and no head holds both a BF16 and an FP8
    /// copy, and the selected representation is the one resident.
    pub fn check_single_residency(&self, kda_fp8: super::fp8::KdaFp8, fp8_head: bool) -> Result<Residency> {
        for (index, layer) in self.layers.iter().enumerate().filter(|(_, l)| l.attention == GlmNextAttention::Kda) {
            let (bf16, fp8) = (layer.has("w_in") || layer.has("w_o"), layer.has("w_in_fp8") || layer.has("w_o_fp8"));
            ensure!(!(bf16 && fp8), "KDA layer {index} holds BF16 and FP8 in/out projections");
            let native = layer.core.contains_key("self_attn.q_proj");
            ensure!(!native || !(bf16 || fp8), "KDA layer {index} has native and converted copies");
            ensure!(native || fp8 == (kda_fp8 != super::fp8::KdaFp8::Off),
                "KDA layer {index}: --kda-fp8 {kda_fp8:?} but {} projections are resident", if fp8 { "FP8" } else { "BF16" });
        }
        ensure!(matches!(self.head, super::head::GlmfHead::K6(_))
            || matches!(self.head, super::head::GlmfHead::Fp8 { .. }) == fp8_head,
            "--fp8-head {fp8_head} but the resident head is {}", self.head.name());
        Ok(self.residency())
    }
}

#[derive(Clone)]
pub(crate) struct GlmfLoader<'a> {
    pub library: &'a NativeLibrary,
    pub checkpoint: &'a Checkpoint,
    pub stream: *mut c_void,
    pub kda_fp8: super::fp8::KdaFp8,
    /// Replicate FP8 KDA output weights for token-row sharding; requires two GPUs and FP8 KDA.
    pub kda_output_shard: bool,
    pub fp8_head: bool,
    /// Numerics gate only: KDA projections rounded through NVFP4 (Some(search)) and kept in BF16.
    pub kda_nvfp4: Option<bool>,
    /// Scale rule of copies quantized from BF16.
    pub fp8_scales: crate::shared::fp8_linear::Fp8Scales,
    /// This loader's device (rank 0 of a head split).
    pub device: i32,
    /// The other GPU of a head split (rank 1), if any.
    pub peers: Vec<RankDevice>,
    /// Class-2 and class-3 operands: run through the mount at once, or (push)
    /// recorded for the coordinator's push ranks. One runner per load.
    pub dense: std::rc::Rc<DenseLoad<'a>>,
}

pub(crate) use crate::shared::experts::rtx::dense::DenseLoad;

/// One part of a row concatenation quantized on the GPU: a checkpoint tensor
/// (read straight into pinned memory) or BF16 bytes the loader already sliced.
enum QuantPart<'b> {
    Tensor(&'b str),
    /// A window of a BF16 tensor (rows, byte columns).
    Rect(Rect),
}

/// Rows `rows`, bytes `cols` of each, of checkpoint tensor `name`.
fn rect(checkpoint: &Checkpoint, name: &str, rows: std::ops::Range<usize>, cols: std::ops::Range<usize>) -> Result<Rect> {
    let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
        .map_err(|_| anyhow::anyhow!("checkpoint has no tensor {name}"))?;
    let tensor = &checkpoint.tensors[at];
    let total = if tensor.meta.shape.len() >= 2 { tensor.meta.shape[0] } else { 1 };
    let row_bytes = tensor.meta.byte_length as usize / total;
    ensure!(rows.end <= total && cols.end <= row_bytes && !rows.is_empty() && !cols.is_empty(),
        "{name}: rectangle outside the tensor");
    Ok(Rect { name: name.into(), path: checkpoint.snapshot.join(&tensor.shard), base: tensor.meta.byte_offset,
        row_bytes, rows, cols })
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// FP32 scales `[n, kb]` (little-endian bytes) transposed to `[kb, n]`.
fn kmajor_scales(scales: &[u8], n: usize, kb: usize) -> Vec<u8> {
    assert_eq!(scales.len(), n * kb * 4);
    let mut kmajor = vec![0u8; scales.len()];
    for row in 0..n {
        for b in 0..kb {
            kmajor[(b * n + row) * 4..][..4].copy_from_slice(&scales[(row * kb + b) * 4..][..4]);
        }
    }
    kmajor
}

fn validate_kda_output_shard(ranks: usize, fp8: super::fp8::KdaFp8, output_shard: bool) -> Result<()> {
    ensure!(!output_shard || (ranks == 2 && fp8 != super::fp8::KdaFp8::Off),
        "KDA token-row output sharding requires two GPUs and FP8 KDA projections");
    Ok(())
}

/// Check the complete checkpoint header; `None` keeps the full matrix on each rank.
fn kda_output_geometry(hidden: usize, width: usize, shape: &[usize], ranks: usize, output_shard: bool)
    -> Result<(Option<Axis>, usize, usize)> {
    ensure!(ranks > 0 && hidden > 0 && width > 0 && shape == [hidden, width],
        "KDA o_proj: expected [{hidden}, {width}], found {shape:?}");
    let (axis, rows, cols) = if output_shard {
        ensure!(ranks == 2 && hidden % ranks == 0,
            "KDA token-row output sharding requires two GPUs and even hidden, got {ranks} GPUs / hidden={hidden}");
        (None, hidden, width)
    } else {
        ensure!(width % ranks == 0, "KDA o_proj: [{hidden}, {width}] does not split columns over {ranks} GPUs");
        (Some(Axis::Cols), hidden, width / ranks)
    };
    ensure!(cols % 128 == 0, "KDA o_proj: each rank's K={cols} must contain whole 128-K scale blocks");
    Ok((axis, rows, cols))
}

/// Quantize one KDA slice using the whole weight's per-row rule, with `[kb,n]` scales
/// (the host reference the GPU load quantization reproduces).
#[cfg(test)]
fn quantize_kda_part(bytes: &[u8], rows: usize, cols: usize, layout: super::fp8::Layout,
    rule: crate::shared::fp8_linear::Fp8Scales) -> (Vec<u8>, Vec<u8>) {
    let (values, scales) = super::fp8::quantize(bytes, rows, cols, layout, rule);
    (values, kmajor_scales(&f32_bytes(&scales), rows, cols / 128))
}

impl<'a> GlmfLoader<'a> {
    fn tensor(&self, name: &str) -> Result<&CheckpointTensor> {
        cuteafd_ffi::memory_ledger::tensor(name);
        let at = self.checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
            .map_err(|_| anyhow::anyhow!("checkpoint has no tensor {name}"))?;
        Ok(&self.checkpoint.tensors[at])
    }

    fn raw(&self, name: &str) -> Result<(Vec<u8>, DType, Vec<usize>)> {
        let tensor = self.tensor(name)?;
        let mut bytes = vec![0u8; tensor.meta.byte_length as usize];
        crate::shared::memory::load_timing::read(bytes.len(), || std::fs::File::open(self.checkpoint.snapshot.join(&tensor.shard))?
            .read_exact_at(&mut bytes, tensor.meta.byte_offset))
            .with_context(|| format!("reading {name}"))?;
        Ok((bytes, tensor.meta.dtype.clone(), tensor.meta.shape.clone()))
    }

    /// A ModelOpt NVFP4 dense MLP at `mlp` as the one-expert fp8_moe operands
    /// (see the module docstring); false when the checkpoint stores it otherwise.
    fn nvfp4_dense(&self, cfg: &GlmNextConfig, mlp: &str, ops: &mut HashMap<&'static str, DeviceAllocation<'a>>)
        -> Result<bool> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("nvfp4");
        if self.tensor(&format!("{mlp}.gate_proj.weight"))?.meta.dtype != DType::U8 {
            return Ok(false);
        }
        let (h, i) = (cfg.hidden, cfg.dense_intermediate);
        let mut gate_input_scale = None;
        for (proj, w_key, s_key, rows, cols) in [("gate_proj", "nvfp4_w1", "nvfp4_s1", i, h),
            ("up_proj", "nvfp4_w3", "nvfp4_s3", i, h), ("down_proj", "nvfp4_w2", "nvfp4_s2", h, i)] {
            let name = format!("{mlp}.{proj}");
            let (weight, dtype, shape) = self.raw(&format!("{name}.weight"))?;
            ensure!(dtype == DType::U8 && shape == [rows, cols / 2], "{name}.weight: expected packed E2M1 U8 [{rows}, {}], \
                found {dtype:?} {shape:?}", cols / 2);
            let (mut scales, dtype, shape) = self.raw(&format!("{name}.weight_scale"))?;
            ensure!(dtype == DType::F8E4M3 && shape == [rows, cols / 16], "{name}.weight_scale: expected E4M3 \
                [{rows}, {}], found {dtype:?} {shape:?}", cols / 16);
            let mut input_scale = [0u8; 4];
            for scalar in ["weight_scale_2", "input_scale"] {
                let (bytes, dtype, _) = self.raw(&format!("{name}.{scalar}"))?;
                ensure!(dtype == DType::F32 && bytes.len() == 4, "{name}.{scalar}: expected one FP32 value");
                if scalar == "input_scale" {
                    input_scale.copy_from_slice(&bytes);
                }
                scales.extend_from_slice(&bytes);
            }
            // The W4A4 dense FC1 kernel quantizes with the gate projection's
            // input_scale and dequantizes the up half with the up projection's;
            // require them bit-identical, as for the routed experts.
            match proj {
                "gate_proj" => gate_input_scale = Some(input_scale),
                "up_proj" => {
                    let gate = gate_input_scale.expect("gate_proj is packed before up_proj");
                    ensure!(gate == input_scale, "{mlp}: NVFP4 dense gate input_scale {} and up input_scale {} differ; \
                        the W4A4 FC1 kernel needs them bit-identical",
                        f32::from_le_bytes(gate), f32::from_le_bytes(input_scale));
                }
                _ => {}
            }
            ops.insert(w_key, self.upload(&weight)?);
            ops.insert(s_key, self.upload(&scales)?);
        }
        Ok(true)
    }

    fn upload(&self, bytes: &[u8]) -> Result<DeviceAllocation<'a>> {
        let allocation = DeviceAllocation::new(self.library, bytes.len().max(256))?;
        crate::shared::memory::load_timing::upload(bytes.len(), || self.library.copy_h2d(allocation.buffer, bytes))?;
        Ok(allocation)
    }

    /// The row-concatenation of 2-D `names` as E4M3 bytes and FP32 scales in
    /// `layout`: the primary checkpoint's own blocks when it stores the
    /// tensors as FP8 (block layout), else quantized from BF16 at load (the
    /// FP8 copy is the only resident one).
    fn fp8(&self, names: &[String], layout: super::fp8::Layout) -> Result<(DeviceAllocation<'a>, DeviceAllocation<'a>)> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("fp8");
        if layout == super::fp8::Layout::Block && self.all_native_blocks(names)? {
            return self.native_blocks(0, names, None);
        }
        if self.all_bf16(names)? {
            let parts = names.iter().map(|n| QuantPart::Tensor(n)).collect::<Vec<_>>();
            return self.gpu_fp8(&parts, layout, false, 0);
        }
        let (values, scales, _) = self.fp8_host(names, layout)?;
        Ok((self.upload(&values)?, self.upload(&scales)?))
    }

    /// Whether every name is a native E4M3 matrix with FP32 128x128 `_scale_inv`
    /// blocks in the source checkpoint.
    fn all_native_blocks(&self, names: &[String]) -> Result<bool> {
        let source = self.checkpoint;
        names.iter().try_fold(true, |all, name| {
            let tensor = Self::source_tensor(source, name)?;
            Ok(all && tensor.meta.dtype == DType::F8E4M3 && tensor.meta.shape.len() == 2)
        })
    }

    fn source_tensor(source: &'a Checkpoint, name: &str) -> Result<&'a CheckpointTensor> {
        let at = source.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
            .map_err(|_| anyhow::anyhow!("FP8 checkpoint has no tensor {name}"))?;
        Ok(&source.tensors[at])
    }

    /// The checkpoint's own E4M3 blocks and FP32 128x128 grids of `names`
    /// (row-concatenated), landing unchanged on rank `rank` (class 2). With
    /// `split = (axis, ranks)`, rank `rank`'s whole-128-block share of each
    /// tensor: rows, or a column window of every row (values and grid alike).
    fn native_blocks(&self, rank: usize, names: &[String], split: Option<(Axis, usize)>)
        -> Result<(DeviceAllocation<'a>, DeviceAllocation<'a>)> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("fp8");
        let source = self.checkpoint;
        let (mut values, mut grids) = (Vec::new(), Vec::new());
        let (mut at, mut grid_at, mut width) = (0usize, 0usize, None);
        for (i, name) in names.iter().enumerate() {
            let tensor = Self::source_tensor(source, name)?;
            let shape = &tensor.meta.shape;
            ensure!(tensor.meta.dtype == DType::F8E4M3 && shape.len() == 2, "{name}: expected an E4M3 matrix");
            let (rows, cols) = (shape[0], shape[1]);
            let scale_name = format!("{name}_scale_inv");
            let scale = Self::source_tensor(source, &scale_name)?;
            ensure!(scale.meta.dtype == DType::F32 && scale.meta.shape == [rows.div_ceil(128), cols.div_ceil(128)],
                "{name}: expected FP32 128x128 block scales, found {:?} {:?}", scale.meta.dtype, scale.meta.shape);
            let (kb, rb) = (cols.div_ceil(128), rows.div_ceil(128));
            let (row_range, col_range, grid_rows, grid_cols) = match split {
                None => {
                    ensure!(i + 1 == names.len() || rows % 128 == 0,
                        "{names:?}: FP8 rows must fill whole 128-row blocks but the last");
                    (0..rows, 0..cols, 0..rb, 0..kb)
                }
                Some((axis, ranks)) => {
                    ensure!(rows % 128 == 0 && cols % 128 == 0, "{name}: split FP8 blocks require whole 128x128 \
                        blocks, found {shape:?}");
                    let along = if axis == Axis::Rows { rows } else { cols };
                    ensure!(along % (128 * ranks) == 0, "{name}: [{rows}, {cols}] does not split into whole 128 \
                        blocks over {ranks} GPUs");
                    let share = along / ranks;
                    let window = rank * share..(rank + 1) * share;
                    let blocks = window.start / 128..window.end / 128;
                    match axis {
                        Axis::Rows => (window, 0..cols, blocks, 0..kb),
                        Axis::Cols => (0..rows, window, 0..rb, blocks),
                    }
                }
            };
            ensure!(width.is_none_or(|w| w == col_range.len()), "{names:?}: FP8 rows must share columns");
            width = Some(col_range.len());
            let value = Rect { name: name.clone(), path: source.snapshot.join(&tensor.shard),
                base: tensor.meta.byte_offset, row_bytes: cols, rows: row_range, cols: col_range };
            let grid = Rect { name: scale_name, path: source.snapshot.join(&scale.shard),
                base: scale.meta.byte_offset, row_bytes: kb * 4, rows: grid_rows, cols: grid_cols.start * 4..grid_cols.end * 4 };
            let (bytes, grid_bytes) = (value.bytes(), grid.bytes());
            values.push((value, at));
            grids.push((grid, grid_at));
            at += bytes;
            grid_at += grid_bytes;
        }
        Ok((self.landed(rank, values, at)?, self.landed(rank, grids, grid_at)?))
    }

    /// `parts` row-concatenated and quantized on the current device: E4M3 values
    /// and FP32 scales (row-major per unit, or K-block major `[K/128, N]`),
    /// allocated as the host path allocated them.
    fn gpu_fp8(&self, parts: &[QuantPart<'_>], layout: super::fp8::Layout, kmajor: bool, rank: usize)
        -> Result<(DeviceAllocation<'a>, DeviceAllocation<'a>)> {
        let source = self.checkpoint;
        let rects = parts.iter().map(|part| match part {
            QuantPart::Tensor(name) => {
                let at = source.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
                    .map_err(|_| anyhow::anyhow!("checkpoint has no tensor {name}"))?;
                let meta = &source.tensors[at].meta;
                ensure!(meta.dtype == DType::Bf16 && meta.shape.len() == 2, "{name}: GPU quantization takes BF16 matrices");
                rect(source, name, 0..meta.shape[0], 0..meta.shape[1] * 2)
            }
            QuantPart::Rect(rect) => Ok(rect.clone()),
        }).collect::<Result<Vec<_>>>()?;
        let cols = rects.first().context("no FP8 parts")?.width() / 2;
        ensure!(rects.iter().all(|r| r.width() == cols * 2), "FP8 rows must share columns");
        let total: usize = rects.iter().map(|r| r.rows.len()).sum();
        let (kb, unit) = (cols / 128, if layout == super::fp8::Layout::Block { 128 } else { 1 });
        let values = DeviceAllocation::new(self.library, (total * cols).max(256))?;
        let scales = DeviceAllocation::new(self.library, (total / unit * kb * 4).max(256))?;
        self.quant_steps(&rects, layout, rank, |rows| Target {
            // SAFETY: rows `rows` lie inside `values` (allocated for `total` rows).
            q: CuteafdDeviceBuffer { ptr: unsafe { values.buffer.ptr.cast::<u8>().add(rows.start * cols) }.cast(),
                bytes: rows.len() * cols, ..values.buffer },
            q_stride: cols,
            scales: if kmajor { scales.buffer } else { CuteafdDeviceBuffer {
                // SAFETY: these rows' scale units lie inside `scales`.
                ptr: unsafe { scales.buffer.ptr.cast::<u8>().add(rows.start / unit * kb * 4) }.cast(),
                bytes: rows.len() / unit * kb * 4, ..scales.buffer } },
            kmajor: kmajor.then_some((total, rows.start / unit)),
        }, 0, cols)?;
        Ok((values, scales))
    }

    /// Quantization steps over `rects` (BF16 rows, back to back) in windows of
    /// whole scale units: each window quantizes columns `col0..col0 + cols` of
    /// its staged rows into `target(rows of the whole)`.
    fn quant_steps(&self, rects: &[Rect], layout: super::fp8::Layout, rank: usize,
        target: impl Fn(std::ops::Range<usize>) -> Target, col0: usize, cols: usize) -> Result<()> {
        let unit = if layout == super::fp8::Layout::Block { 128 } else { 1 };
        let pitch = rects.first().context("no rows")?.width() / 2;
        for (parts, rows) in crate::shared::experts::rtx::dense::windows(rects, unit)? {
            let job = Job::Quant { rows: 0..rows.len(), pitch, col0, cols, layout, rule: self.fp8_scales,
                target: target(rows.clone()) };
            self.step(rank, parts, vec![job])?;
        }
        Ok(())
    }

    /// Records or runs one class-3 step on rank `rank`'s GPU.
    fn step(&self, rank: usize, rects: Vec<Rect>, jobs: Vec<Job>) -> Result<()> {
        let device = self.rank_device(rank)?;
        let step = crate::shared::experts::rtx::dense::Step { device, rects, jobs };
        let stream = self.rank_stream(rank)?;
        self.dense.step(step, stream)
    }

    /// Records or fills one class-2 operand of `bytes` on rank `rank`'s GPU.
    fn landed(&self, rank: usize, lands: Vec<(Rect, usize)>, bytes: usize) -> Result<DeviceAllocation<'a>> {
        let device = self.rank_device(rank)?;
        self.on_rank(rank, || {
            let allocation = DeviceAllocation::new(self.library, bytes.max(256))?;
            let operand = crate::shared::experts::rtx::dense::Operand { device, buffer: allocation.buffer, lands, bytes };
            self.dense.operand(operand)?;
            Ok(allocation)
        })
    }

    /// The device of rank `rank`.
    fn rank_device(&self, rank: usize) -> Result<i32> {
        Ok(if rank == 0 { self.device } else { self.peers.get(rank - 1).context("rank device")?.device })
    }

    /// A whole checkpoint tensor (one row for 1-D) landing unchanged on `rank`.
    fn whole(&self, rank: usize, name: &str) -> Result<DeviceAllocation<'a>> {
        let tensor = self.tensor(name)?;
        let rows = if tensor.meta.shape.len() >= 2 { tensor.meta.shape[0] } else { 1 };
        let bytes = tensor.meta.byte_length as usize;
        self.landed(rank, vec![(rect(self.checkpoint, name, 0..rows, 0..bytes / rows)?, 0)], bytes)
    }

    /// `make(rank)` on every rank: in push mode each rank receives its own copy;
    /// through the mount, rank 0's and device copies of it.
    fn on_every_rank(&self, make: impl Fn(usize) -> Result<DeviceAllocation<'a>>) -> Result<Vec<DeviceAllocation<'a>>> {
        if self.dense.pushing() { return (0..self.ranks()).map(make).collect(); }
        self.replicate(make(0)?)
    }

    /// [`Self::on_every_rank`] for a pair of operands (FP8 values and scales).
    #[allow(clippy::type_complexity)]
    fn on_every_rank_pair(&self, make: impl Fn(usize) -> Result<(DeviceAllocation<'a>, DeviceAllocation<'a>)>)
        -> Result<Vec<(DeviceAllocation<'a>, DeviceAllocation<'a>)>> {
        if self.dense.pushing() { return (0..self.ranks()).map(make).collect(); }
        let (q, s) = make(0)?;
        Ok(self.replicate(q)?.into_iter().zip(self.replicate(s)?).collect())
    }

    /// [`Self::fp8`] on rank `rank` (BF16 sources; FP8 sources load through the mount on rank 0).
    fn fp8_on(&self, rank: usize, names: &[String], layout: super::fp8::Layout)
        -> Result<(DeviceAllocation<'a>, DeviceAllocation<'a>)> {
        if rank == 0 { return self.fp8(names, layout); }
        if layout == super::fp8::Layout::Block && self.all_native_blocks(names)? {
            return self.native_blocks(rank, names, None);
        }
        ensure!(self.all_bf16(names)?, "{names:?}: FP8 {layout:?} replicas take BF16 or native block sources");
        let parts = names.iter().map(|n| QuantPart::Tensor(n)).collect::<Vec<_>>();
        self.on_rank(rank, || self.gpu_fp8(&parts, layout, false, rank))
    }

    /// Row-concatenate BF16 matrices on rank `rank`, unchanged.
    fn rows_on(&self, rank: usize, names: &[String]) -> Result<DeviceAllocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        let mut lands = Vec::new();
        let mut at = 0;
        for name in names {
            let tensor = self.tensor(name)?;
            ensure!(tensor.meta.dtype == DType::Bf16 && tensor.meta.shape.len() == 2,
                "{name}: pushed row concatenations take BF16 matrices");
            let r = rect(self.checkpoint, name, 0..tensor.meta.shape[0], 0..tensor.meta.shape[1] * 2)?;
            let bytes = r.bytes();
            lands.push((r, at));
            at += bytes;
        }
        ensure!(lands.windows(2).all(|w| w[0].0.width() == w[1].0.width()), "{names:?} do not share columns");
        self.landed(rank, lands, at)
    }

    /// Whether every name is a BF16 matrix in the source checkpoint (quantized on the GPU).
    fn all_bf16(&self, names: &[String]) -> Result<bool> {
        let source = self.checkpoint;
        names.iter().try_fold(true, |all, name| {
            let at = source.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
                .map_err(|_| anyhow::anyhow!("checkpoint has no tensor {name}"))?;
            Ok(all && source.tensors[at].meta.dtype == DType::Bf16 && source.tensors[at].meta.shape.len() == 2)
        })
    }

    /// Per-row FP8 `names` (`layout` Row128 or Channel) with its scales stored
    /// K-block major (`[K/128, N]`), the one layout the `kda_w8` decode and
    /// prefill programs both read: (values, K-major scales).
    fn fp8_rows_kmajor(&self, names: &[String], layout: super::fp8::Layout)
        -> Result<(DeviceAllocation<'a>, DeviceAllocation<'a>)> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("fp8");
        ensure!(layout != super::fp8::Layout::Block, "{names:?}: K-major scales are per-row scales");
        if self.all_bf16(names)? {
            let parts = names.iter().map(|n| QuantPart::Tensor(n)).collect::<Vec<_>>();
            return self.gpu_fp8(&parts, layout, true, 0);
        }
        let (values, scales, cols) = self.fp8_host(names, layout)?;
        Ok((self.upload(&values)?, self.upload(&kmajor_scales(&scales, values.len() / cols, cols / 128))?))
    }

    /// Host bytes of [`Self::fp8`]: E4M3 values, FP32 scales, and the column count.
    fn fp8_host(&self, names: &[String], layout: super::fp8::Layout) -> Result<(Vec<u8>, Vec<u8>, usize)> {
        use super::fp8::{quantize, Layout};
        let (mut values, mut scales) = (Vec::new(), Vec::<u8>::new());
        let mut cols = None;
        for name in names {
            let (bytes, dtype, shape) = self.raw(name)?;
            ensure!(shape.len() == 2 && cols.is_none_or(|c| c == shape[1]), "{name}: FP8 rows must share columns");
            cols = Some(shape[1]);
            match dtype {
                DType::F8E4M3 if layout == Layout::Block => {
                    let (scale, scale_dtype, scale_shape) = self.raw(&format!("{name}_scale_inv"))?;
                    ensure!(scale_dtype == DType::F32 && shape[0] % 128 == 0
                        && scale_shape == [shape[0] / 128, shape[1].div_ceil(128)],
                        "{name}: expected FP32 128x128 block scales, found {scale_dtype:?} {scale_shape:?}");
                    values.extend_from_slice(&bytes);
                    scales.extend_from_slice(&scale);
                }
                DType::Bf16 => {
                    let (q, s) = quantize(&bytes, shape[0], shape[1], layout, self.fp8_scales);
                    values.extend_from_slice(&q);
                    scales.extend(s.iter().flat_map(|v| v.to_le_bytes()));
                }
                other => anyhow::bail!("{name}: cannot make an FP8 {layout:?} copy of {other:?}"),
            }
        }
        Ok((values, scales, cols.context("no FP8 rows")?))
    }

    fn one(&self, name: &str) -> Result<DeviceAllocation<'a>> { self.one_on(0, name) }

    /// [`Self::one`] on rank `rank`.
    fn one_on(&self, rank: usize, name: &str) -> Result<DeviceAllocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        let tensor = self.tensor(name)?;
        ensure!(matches!(tensor.meta.dtype, DType::Bf16 | DType::F32),
            "{name}: unchanged coordinator operands require BF16 or FP32");
        self.whole(rank, name)
    }

    /// A BF16 (or FP32) tensor widened to FP32 (BF16 widens on the GPU).
    fn f32(&self, names: &[String]) -> Result<DeviceAllocation<'a>> { self.f32_on(0, names) }

    /// [`Self::f32`] on rank `rank`.
    fn f32_on(&self, rank: usize, names: &[String]) -> Result<DeviceAllocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("f32");
        let [name] = names else { anyhow::bail!("{names:?}: FP32 operands are one tensor") };
        let tensor = self.tensor(name)?;
        match &tensor.meta.dtype {
            DType::F32 => self.whole(rank, name),
            DType::Bf16 => {
                let bytes = tensor.meta.byte_length as usize;
                let rows = if tensor.meta.shape.len() >= 2 { tensor.meta.shape[0] } else { 1 };
                let out = self.on_rank(rank, || DeviceAllocation::new(self.library, (bytes * 2).max(256)))?;
                let whole = rect(self.checkpoint, name, 0..rows, 0..bytes / rows)?;
                for (parts, window) in crate::shared::experts::rtx::dense::windows(&[whole], 1)? {
                    let width = bytes / rows;
                    let dst = CuteafdDeviceBuffer {
                        // SAFETY: these rows' F32 values lie inside `out` (twice the BF16 bytes).
                        ptr: unsafe { out.buffer.ptr.cast::<u8>().add(window.start * width * 2) }.cast(),
                        bytes: window.len() * width * 2, ..out.buffer };
                    self.step(rank, parts, vec![Job::Widen { at: 0, bytes: window.len() * width, dst }])?;
                }
                Ok(out)
            }
            other => anyhow::bail!("{name}: expected BF16 or FP32, found {other:?}"),
        }
    }

    /// `kv_b_proj [N*(nope+v), 512]` -> `w_uk [N, 512, nope]` (key rows transposed per head)
    /// and `w_uv [N, v, 512]`, per rank its head share `r * N / ranks ..`, absorbed on
    /// each rank's GPU from its heads' staged rows.
    fn absorbed(&self, cfg: &GlmNextConfig, name: &str, ranks: usize)
        -> Result<Vec<(DeviceAllocation<'a>, DeviceAllocation<'a>)>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        let tensor = self.tensor(name)?;
        let (n, nope, v, lat) = (cfg.heads, cfg.qk_nope_dim, cfg.v_head_dim, cfg.kv_lora_rank);
        ensure!(tensor.meta.dtype == DType::Bf16 && tensor.meta.shape == [n * (nope + v), lat],
            "{name}: expected BF16 [{}, {lat}]", n * (nope + v));
        ensure!(n % ranks == 0, "{n} MLA heads do not split over {ranks} GPUs");
        let heads = n / ranks;
        // Steps of whole heads, each at most the step bound.
        let per = (crate::shared::experts::rtx::dense::STEP_BYTES / ((nope + v) * lat * 2)).max(1);
        (0..ranks).map(|rank| {
            let (uk, uv) = self.on_rank(rank, || Ok((
                DeviceAllocation::new(self.library, (heads * lat * nope * 2).max(256))?,
                DeviceAllocation::new(self.library, (heads * v * lat * 2).max(256))?)))?;
            let mut h = 0;
            while h < heads {
                let count = per.min(heads - h);
                let first = rank * heads + h;
                let rows = first * (nope + v)..(first + count) * (nope + v);
                let at = |buffer: CuteafdDeviceBuffer, per_head: usize| CuteafdDeviceBuffer {
                    // SAFETY: heads h..h + count of this rank lie inside its operand.
                    ptr: unsafe { buffer.ptr.cast::<u8>().add(h * per_head) }.cast(), bytes: count * per_head, ..buffer };
                self.step(rank, vec![rect(self.checkpoint, name, rows, 0..lat * 2)?], vec![Job::Absorb { heads: count,
                    nope, v, lat, uk: at(uk.buffer, lat * nope * 2), uv: at(uv.buffer, v * lat * 2) }])?;
                h += count;
            }
            Ok((uk, uv))
        }).collect()
    }

    fn mla_operands(&self, cfg: &GlmNextConfig, prefix: &str,
        ops: &mut HashMap<&'static str, DeviceAllocation<'a>>) -> Result<()> {
        let a = |name: &str| format!("{prefix}.self_attn.{name}");
        ops.insert("q_a_norm", self.one(&a("q_a_layernorm.weight"))?);
        ops.insert("kv_a_norm", self.one(&a("kv_a_layernorm.weight"))?);
        let (uk, uv) = self.absorbed(cfg, &a("kv_b_proj.weight"), 1)?.remove(0);
        ops.insert("w_uk", uk);
        ops.insert("w_uv", uv);
        let block = super::fp8::Layout::Block;
        let projections: Vec<_> = cuteafd_loader::families::glm5_flash::core_fp8_projections("mla")
            .iter().map(|name| a(name)).collect();
        let (q, s) = self.fp8(&projections[..2], block)?;
        ops.insert("w_qkv_a_fp8", q);
        ops.insert("w_qkv_a_scale", s);
        let (q, s) = self.fp8(&projections[2..3], block)?;
        ops.insert("w_q_b_fp8", q);
        ops.insert("w_q_b_scale", s);
        let (q, s) = self.fp8(&projections[3..4], block)?;
        ops.insert("w_o_fp8", q);
        ops.insert("w_o_scale", s);
        let i = |name: &str| a(&format!("indexer.{name}"));
        ops.insert("w_iq", self.one(&i("wq_b.weight"))?);
        ops.insert("w_ik", self.rows_on(0, &[i("wk.weight"), i("weights_proj.weight"),
            i("index_kpool_compress_gate")])?);
        ops.insert("k_norm_w", self.one(&i("k_norm.weight"))?);
        ops.insert("k_norm_b", self.one(&i("k_norm.bias"))?);
        ops.insert("ape", self.one(&i("index_kpool_compress_ape"))?);
        Ok(())
    }

    fn core_weights(&self, layout: cuteafd_loader::formats::trellis_core::CoreWeightLayout,
        rank: usize) -> Result<crate::shared::dense_trellis::CoreWeights<'a>> {
        let layout = crate::shared::dense_trellis::CoreWeightLayout::from_shared(layout);
        let loaded = layout.read(|name| {
            let (bytes, dtype, shape) = self.raw(name)?;
            let tensor = self.tensor(name)?;
            Ok(cuteafd_loader::LoadedTensor {
                info: cuteafd_core::TensorInfo { name: name.into(), file: tensor.shard.clone(), dtype, shape,
                    byte_offset: tensor.meta.byte_offset, byte_length: tensor.meta.byte_length,
                    role: cuteafd_core::TensorRole::Other, layer_id: None, expert_id: None,
                    is_quantization_metadata: false }, source_path: self.checkpoint.snapshot.join(&tensor.shard),
                bytes, elapsed_micros: 0, sha256: String::new(),
            })
        })?;
        self.on_rank(rank, || layout.upload(crate::shared::memory::device::Device {
            library: self.library, id: self.library.cuda_get_device()? }, loaded))
    }

    fn native_layer_shares(&self, cfg: &GlmNextConfig, layer: usize, mode: LayerMode,
        inventory: &cuteafd_loader::families::glm5_flash::core::CoreInventory) -> Result<Vec<GlmfLayer<'a>>> {
        let ranks = self.ranks();
        let attention = cfg.attention[layer];
        let dense = cfg.dense[layer];
        ensure!(self.kda_nvfp4.is_none(), "native K6 core cannot use the converted NVFP4 diagnostic");
        ensure!(attention == GlmNextAttention::Mla || ranks == 1 || mode == LayerMode::HeadSplit,
            "native KDA retains head ownership");
        ensure!(ranks == 1 || !matches!(mode, LayerMode::Whole { ffn: FfnMode::Owner, .. }),
            "native Flash FFNs retain intermediate ownership");
        let owner = match mode { LayerMode::Whole { gpu, .. } if ranks == 2 => Some(gpu), _ => None };
        let layouts = inventory.layer_weights(cfg, layer, ranks, owner)?;
        let attention_shares = AttentionShare::bind(mode, ranks)?;
        let p = format!("{PREFIX}layers.{layer}");
        let a = |leaf: &str| format!("{p}.self_attn.{leaf}");
        let mut ops: Vec<HashMap<&'static str, DeviceAllocation<'a>>> = (0..ranks).map(|_| HashMap::new()).collect();
        for rank in 0..ranks {
            for (site, keys) in [("attn", ["attn.fn", "attn.scale", "attn.base"]),
                ("ffn", ["ffn.fn", "ffn.scale", "ffn.base"])] {
                for (key, suffix) in keys.into_iter().zip(["fn", "scale", "base"]) {
                    ops[rank].insert(key, self.f32_on(rank, &[format!("{p}.hc_{site}_{suffix}")])?);
                }
            }
            for (key, leaf) in [("input_norm", "input_layernorm"), ("post_norm", "post_attention_layernorm")] {
                ops[rank].insert(key, self.one_on(rank, &format!("{p}.{leaf}.weight"))?);
            }
            if !dense {
                ops[rank].insert("gate", self.one_on(rank, &format!("{p}.mlp.gate.weight"))?);
                ops[rank].insert("gate.bias", self.f32_on(rank, &[format!("{p}.mlp.gate.e_score_correction_bias")])?);
            }
        }
        match attention {
            GlmNextAttention::Kda => {
                let beta = self.tensor(&a("b_proj.weight"))?;
                ensure!(beta.meta.dtype == DType::Bf16 && beta.meta.shape == [cfg.kda_heads, cfg.hidden],
                    "native KDA beta projection geometry");
                for rank in 0..ranks {
                    let mut lands = Vec::new();
                    let mut at = 0;
                    for leaf in ["f_a_proj.weight", "g_a_proj.weight", "b_proj.weight"] {
                        let name = a(leaf);
                        let tensor = self.tensor(&name)?;
                        ensure!(tensor.meta.dtype == DType::Bf16 && tensor.meta.shape.len() == 2
                            && tensor.meta.shape[1] == cfg.hidden, "{name}: native KDA auxiliary matrix");
                        let n = tensor.meta.shape[0];
                        let rows = if leaf == "b_proj.weight" { rank*n/ranks..(rank+1)*n/ranks } else { 0..n };
                        let source = rect(self.checkpoint, &name, rows, 0..cfg.hidden*2)?;
                        let bytes = source.bytes();
                        lands.push((source, at)); at += bytes;
                    }
                    ops[rank].insert("w_aux", self.landed(rank, lands, at)?);
                }
                for operand in [KdaOperand::WFg, KdaOperand::ConvW, KdaOperand::ALog,
                    KdaOperand::DtBias, KdaOperand::ONorm] {
                    for (rank, part) in self.kda_operand(cfg, &p, operand, ranks)?.into_iter().enumerate() {
                        ops[rank].insert(operand.name(), part);
                    }
                }
            }
            GlmNextAttention::Mla => {
                let mut local = self.clone();
                let parts = if let Some(owner) = owner {
                    let owner = usize::from(owner);
                    if owner > 0 { local.device = self.peers[owner-1].device; local.stream = self.peers[owner-1].stream; }
                    local.peers.clear();
                    self.on_rank(owner, || local.absorbed(cfg, &a("kv_b_proj.weight"), 1))?
                        .into_iter().map(|pair| (owner, pair)).collect::<Vec<_>>()
                } else { self.absorbed(cfg, &a("kv_b_proj.weight"), ranks)?.into_iter().enumerate().collect() };
                for (rank, (uk, uv)) in parts {
                    ops[rank].insert("w_uk", uk); ops[rank].insert("w_uv", uv);
                    for (key, leaf) in [("q_a_norm", "q_a_layernorm.weight"), ("kv_a_norm", "kv_a_layernorm.weight"),
                        ("w_iq", "indexer.wq_b.weight"), ("k_norm_w", "indexer.k_norm.weight"),
                        ("k_norm_b", "indexer.k_norm.bias"), ("ape", "indexer.index_kpool_compress_ape")] {
                        ops[rank].insert(key, self.one_on(rank, &a(leaf))?);
                    }
                    ops[rank].insert("w_ik", self.rows_on(rank, &[a("indexer.wk.weight"),
                        a("indexer.weights_proj.weight"), a("indexer.index_kpool_compress_gate")])?);
                }
            }
        }
        ops.into_iter().zip(attention_shares).zip(layouts).enumerate().map(|(rank, ((operands, attention_share), layouts))| {
            let expected = if owner.is_some_and(|owner| usize::from(owner) != rank) { 3 } else { 7 };
            ensure!(layouts.len() == expected, "native Flash requires complete attention and FFN operator groups");
            let core = layouts.into_iter().map(|(name, layout)| Ok((name, self.core_weights(layout, rank)?)))
                .collect::<Result<_>>()?;
            Ok(GlmfLayer { attention, dense, split: ranks == 2, attention_share, core, operands })
        }).collect()
    }

    pub fn layer(&self, cfg: &GlmNextConfig, layer: usize) -> Result<GlmfLayer<'a>> {
        let inventory = cuteafd_loader::families::glm5_flash::core::CoreInventory::from_checkpoint(cfg, self.checkpoint)?;
        if inventory.projections.keys().any(|stem| stem.starts_with(&format!("{PREFIX}layers.{layer}."))) {
            return self.native_layer_shares(cfg, layer, LayerMode::HeadSplit, &inventory)?
                .into_iter().next().context("native Flash lead layer");
        }
        ensure!(!self.kda_output_shard, "KDA output sharding loads through the two-GPU layer shares");
        let p = format!("{PREFIX}layers.{layer}");
        let attention = cfg.attention[layer];
        let dense = cfg.dense[layer];
        let mut ops: HashMap<&'static str, DeviceAllocation<'a>> = HashMap::new();
        for site in ["attn", "ffn"] {
            let (fn_, scale, base): (&'static str, &'static str, &'static str) = if site == "attn" {
                ("attn.fn", "attn.scale", "attn.base")
            } else {
                ("ffn.fn", "ffn.scale", "ffn.base")
            };
            ops.insert(fn_, self.f32(&[format!("{p}.hc_{site}_fn")])?);
            ops.insert(scale, self.f32(&[format!("{p}.hc_{site}_scale")])?);
            ops.insert(base, self.f32(&[format!("{p}.hc_{site}_base")])?);
        }
        ops.insert("input_norm", self.one(&format!("{p}.input_layernorm.weight"))?);
        ops.insert("post_norm", self.one(&format!("{p}.post_attention_layernorm.weight"))?);
        let a = |name: &str| format!("{p}.self_attn.{name}");
        match attention {
            GlmNextAttention::Kda => {
                let w_in: Vec<_> = Self::kda_geometry(cfg).roles(KdaOperand::WIn).into_iter()
                    .map(|role| Self::kda_name(&p, role)).collect();
                let kda_layout = match self.kda_fp8 {
                    super::fp8::KdaFp8::Off => None,
                    super::fp8::KdaFp8::Channel => Some(super::fp8::Layout::Channel),
                    super::fp8::KdaFp8::Row128 => Some(super::fp8::Layout::Row128),
                };
                if let Some(layout) = kda_layout {
                    // The FP8 in/out projections are the only resident copies (no BF16
                    // `w_in`/`w_o`); one K-major scale copy serves decode and prefill.
                    ensure!(self.kda_nvfp4.is_none(), "the KDA NVFP4 gate rounds BF16 projections; use --kda-fp8 off");
                    let (q, k) = self.fp8_rows_kmajor(&w_in, layout)?;
                    ops.insert("w_in_fp8", q);
                    ops.insert("w_in_kscale", k);
                    let (q, k) = self.fp8_rows_kmajor(&[a("o_proj.weight")], layout)?;
                    ops.insert("w_o_fp8", q);
                    ops.insert("w_o_kscale", k);
                } else if let Some(search) = self.kda_nvfp4 {
                    let mut bytes = Vec::new();
                    for name in w_in.iter().chain([a("o_proj.weight")].iter()) {
                        let (raw, dtype, shape) = self.raw(name)?;
                        ensure!(dtype == DType::Bf16, "{name}: NVFP4 gate needs BF16");
                        let rounded = super::fp8::nvfp4_roundtrip(&raw, shape[0], shape[1], search);
                        if name.ends_with("o_proj.weight") {
                            ops.insert("w_o_nvfp4", self.upload(&rounded)?);
                        } else {
                            bytes.extend(rounded);
                        }
                    }
                    ops.insert("w_in", self.upload(&bytes)?);
                } else {
                    ops.insert("w_in", self.kda_upload(cfg, &p, KdaOperand::WIn)?);
                }
                for operand in [KdaOperand::WFg, KdaOperand::ConvW, KdaOperand::ALog,
                    KdaOperand::DtBias, KdaOperand::ONorm] {
                    ops.insert(operand.name(), self.kda_upload(cfg, &p, operand)?);
                }
                if kda_layout.is_none() {
                    let w_o = match ops.remove("w_o_nvfp4") {
                        Some(rounded) => rounded,
                        None => self.one(&a("o_proj.weight"))?,
                    };
                    ops.insert("w_o", w_o);
                }
            }
            GlmNextAttention::Mla => self.mla_operands(cfg, &p, &mut ops)?,
        }
        let mlp = if dense { format!("{p}.mlp") } else { format!("{p}.mlp.shared_experts") };
        if !(dense && self.nvfp4_dense(cfg, &mlp, &mut ops)?) {
            let block = super::fp8::Layout::Block;
            let projections: Vec<_> = cuteafd_loader::families::glm5_flash::core_fp8_projections("mlp")
                .iter().map(|name| format!("{mlp}.{name}")).collect();
            let (q, s) = self.fp8(&projections[..2], block)?;
            ops.insert("w_gate_up_fp8", q);
            ops.insert("w_gate_up_scale", s);
            let (q, s) = self.fp8(&projections[2..3], block)?;
            ops.insert("w_down_fp8", q);
            ops.insert("w_down_scale", s);
        }
        if !dense {
            ops.insert("gate", self.one(&format!("{p}.mlp.gate.weight"))?);
            ops.insert("gate.bias", self.f32(&[format!("{p}.mlp.gate.e_score_correction_bias")])?);
        }
        Ok(GlmfLayer { attention, dense, split: false, attention_share: AttentionShare::Whole,
            core: Default::default(), operands: ops })
    }

    /// GPUs of the head split this loader fills (1: no split).
    pub fn ranks(&self) -> usize {
        1 + self.peers.len()
    }

    /// Runs `body` with rank `rank`'s device current.
    fn on_rank<T>(&self, rank: usize, body: impl FnOnce() -> Result<T>) -> Result<T> {
        if rank == 0 {
            return body();
        }
        let peer = self.peers.get(rank - 1).with_context(|| format!("no rank {rank}"))?;
        crate::shared::peer_split::on_device(self.library, peer.device, body)
    }

    /// `first` (on rank 0) and a device copy of it on every other rank (peer access to rank 0
    /// is enabled by the split setup).
    fn replicate(&self, first: DeviceAllocation<'a>) -> Result<Vec<DeviceAllocation<'a>>> {
        let mut out = vec![first];
        for (rank, peer) in self.peers.iter().enumerate() {
            let source = out[0].buffer;
            out.push(self.on_rank(rank + 1, || {
                let copy = DeviceAllocation::new(self.library, source.bytes)?;
                // SAFETY: both allocations hold `source.bytes`; the peer stream drains before return.
                unsafe {
                    self.library.copy_d2d_async(copy.buffer, source, source.bytes, peer.stream)?;
                    self.library.cuda_stream_synchronize(peer.stream)?;
                }
                Ok(copy)
            })?);
        }
        Ok(out)
    }

    fn kda_geometry(cfg: &GlmNextConfig) -> KdaGeometry {
        KdaGeometry { hidden: cfg.hidden, heads: cfg.kda_heads, head_dim: cfg.kda_head_dim,
            conv_kernel: 4, output_gate: KdaOutputGate::LowRank, o_norm_dtype: DType::Bf16 }
    }

    fn kda_name(prefix: &str, role: KdaRole) -> String {
        use KdaRole as R;
        let name = match role {
            R::Q => "q_proj.weight", R::K => "k_proj.weight", R::V => "v_proj.weight",
            R::Fa => "f_a_proj.weight", R::Fb => "f_b_proj.weight", R::Ga => "g_a_proj.weight",
            R::Gb => "g_b_proj.weight", R::G => "g_proj.weight", R::Beta => "b_proj.weight",
            R::ConvQ => "q_conv1d.weight", R::ConvK => "k_conv1d.weight", R::ConvV => "v_conv1d.weight",
            R::ALog => "A_log", R::DtBias => "dt_bias", R::ONorm => "o_norm.weight", R::O => "o_proj.weight",
        };
        format!("{prefix}.self_attn.{name}")
    }

    fn kda_upload(&self, cfg: &GlmNextConfig, prefix: &str, operand: KdaOperand) -> Result<DeviceAllocation<'a>> {
        Ok(self.kda_operand(cfg, prefix, operand, 1)?.remove(0))
    }

    /// The shared KDA plan of `operand` over `ranks`, from checkpoint headers.
    fn kda_plan(&self, cfg: &GlmNextConfig, prefix: &str, operand: KdaOperand, ranks: usize)
        -> Result<cuteafd_loader::kda::KdaPlan> {
        Ok(Self::kda_geometry(cfg).plan(operand, ranks, |role| {
            let tensor = self.tensor(&Self::kda_name(prefix, role)).map_err(|error| KdaError::Source(error.to_string()))?;
            Ok((tensor.meta.dtype.clone(), tensor.meta.shape.clone()))
        })?)
    }

    /// One rank's pieces of a KDA plan as source rectangles (BF16/F32 bytes).
    fn kda_rects(&self, prefix: &str, pieces: &[cuteafd_loader::kda::KdaPiece]) -> Result<Vec<Rect>> {
        pieces.iter().map(|piece| {
            let element = if piece.source == DType::F32 { 4 } else { 2 };
            let name = Self::kda_name(prefix, piece.role);
            // The plan views 1-D roles as [n, 1] and conv weights as [d, kernel];
            // the checkpoint rows those views map onto are the same bytes.
            let shape = &self.tensor(&name)?.meta.shape;
            if shape.len() == 1 {
                ensure!(piece.window == (0..1), "{name}: 1-D KDA pieces are whole elements");
                return rect(self.checkpoint, &name, 0..1, piece.rows.start * element..piece.rows.end * element);
            }
            rect(self.checkpoint, &name, piece.rows.clone(), piece.window.start * element..piece.window.end * element)
        }).collect()
    }

    /// A packed KDA operand on each rank's GPU (the shared plan): pieces land
    /// unchanged; BF16 conv/dt_bias pieces widen to FP32 there; A_log's padding
    /// is checked on the staged raw vector before its active heads land.
    fn kda_operand(&self, cfg: &GlmNextConfig, prefix: &str, operand: KdaOperand, ranks: usize)
        -> Result<Vec<DeviceAllocation<'a>>> {
        let plan = self.kda_plan(cfg, prefix, operand, ranks)?;
        let element = if plan.dtype == DType::F32 { 4 } else { 2 };
        plan.ranks.iter().enumerate().map(|(rank, pieces)| {
            let bytes = pieces.iter().map(|p| p.elements()).sum::<usize>() * element;
            let rects = self.kda_rects(prefix, pieces)?;
            if let Some(heads) = plan.a_log {
                // The raw vector staged whole (padding included), checked, then this
                // rank's active heads uploaded from the staging.
                let name = Self::kda_name(prefix, KdaRole::ALog);
                let raw = self.tensor(&name)?.meta.byte_length as usize;
                let out = self.on_rank(rank, || DeviceAllocation::new(self.library, bytes.max(256)))?;
                let piece = &pieces[0];
                self.step(rank, vec![rect(self.checkpoint, &name, 0..1, 0..raw)?], vec![Job::ALog { heads },
                    Job::Upload { at: piece.rows.start * 4, bytes, dst: out.buffer }])?;
                return Ok(out);
            }
            if pieces.iter().all(|p| p.source == plan.dtype) {
                let mut at = 0;
                let lands = rects.into_iter().map(|r| { let o = at; at += r.bytes(); (r, o) }).collect();
                return self.landed(rank, lands, bytes);
            }
            // Widened pieces: each a step writing its F32 elements.
            let out = self.on_rank(rank, || DeviceAllocation::new(self.library, bytes.max(256)))?;
            for (piece, whole) in pieces.iter().zip(rects) {
                ensure!(piece.source == DType::Bf16 && plan.dtype == DType::F32, "KDA piece needs a widening");
                let row_elements = piece.window.len();
                for (parts, window) in crate::shared::experts::rtx::dense::windows(&[whole], 1)? {
                    let dst = CuteafdDeviceBuffer {
                        // SAFETY: these elements of the piece lie inside the rank's operand.
                        ptr: unsafe { out.buffer.ptr.cast::<u8>().add((piece.at + window.start * row_elements) * 4) }.cast(),
                        bytes: window.len() * row_elements * 4, ..out.buffer };
                    self.step(rank, parts, vec![Job::Widen { at: 0, bytes: window.len() * row_elements * 2, dst }])?;
                }
            }
            Ok(out)
        }).collect()
    }

    /// One rank's KDA BF16 `operand` as quantization rectangles (W_in rows).
    fn kda_quant_rects(&self, cfg: &GlmNextConfig, prefix: &str, operand: KdaOperand, ranks: usize)
        -> Result<Vec<Vec<Rect>>> {
        let plan = self.kda_plan(cfg, prefix, operand, ranks)?;
        ensure!(plan.dtype == DType::Bf16, "KDA FP8 projections quantize BF16 sources");
        plan.ranks.iter().map(|pieces| self.kda_rects(prefix, pieces)).collect()
    }

    /// The selected source's 2-D `name` as E4M3 blocks, quantized from BF16
    /// when necessary: values, FP32 scale grid, rows, cols (the host
    /// reference of the GPU block quantization).
    #[cfg(test)]
    fn fp8_block_host(checkpoint: &Checkpoint, name: &str, scales: crate::shared::fp8_linear::Fp8Scales)
        -> Result<(Vec<u8>, Vec<u8>, usize, usize)> {
        let read = |name: &str| -> Result<(Vec<u8>, DType, Vec<usize>)> {
            let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
                .map_err(|_| anyhow::anyhow!("FP8 checkpoint has no tensor {name}"))?;
            let tensor = &checkpoint.tensors[at];
            let mut bytes = vec![0u8; tensor.meta.byte_length as usize];
            crate::shared::memory::load_timing::read(bytes.len(), || std::fs::File::open(checkpoint.snapshot.join(&tensor.shard))?
                .read_exact_at(&mut bytes, tensor.meta.byte_offset)).with_context(|| format!("reading {name}"))?;
            Ok((bytes, tensor.meta.dtype.clone(), tensor.meta.shape.clone()))
        };
        let (values, dtype, shape) = read(name)?;
        ensure!(shape.len() == 2 && shape[0] % 128 == 0 && shape[1] % 128 == 0,
            "{name}: split FP8 blocks require whole 128x128 blocks, found {shape:?}");
        if dtype == DType::Bf16 {
            // Match the unsplit loader: quantize the selected checkpoint once,
            // before slicing, so both ranks inherit exactly the same blocks.
            let (values, scales) = super::fp8::quantize(&values, shape[0], shape[1],
                super::fp8::Layout::Block, scales);
            let grid = scales.iter().flat_map(|v| v.to_le_bytes()).collect();
            return Ok((values, grid, shape[0], shape[1]));
        }
        ensure!(dtype == DType::F8E4M3, "{name}: split FP8 blocks require BF16 or E4M3, found {dtype:?}");
        let (grid, grid_dtype, grid_shape) = read(&format!("{name}_scale_inv"))?;
        ensure!(grid_dtype == DType::F32 && shape[0] % 128 == 0 && shape[1] % 128 == 0
            && grid_shape == [shape[0] / 128, shape[1] / 128],
            "{name}: expected FP32 128x128 block scales, found {grid_dtype:?} {grid_shape:?}");
        Ok((values, grid, shape[0], shape[1]))
    }

    /// The FP8 blocks of `names` concatenated by rows, sliced over `ranks` along `axis` in
    /// whole 128-row / 128-K blocks: per rank its E4M3 bytes and its part of the FP32 grid.
    fn fp8_split(&self, names: &[String], axis: Axis, ranks: usize)
        -> Result<Vec<(DeviceAllocation<'a>, DeviceAllocation<'a>)>> {
        ensure!(axis == Axis::Rows || names.len() == 1, "{names:?}: concatenated weights split by rows");
        if self.all_bf16(names)? {
            return self.gpu_fp8_split(names, axis, ranks);
        }
        ensure!(self.all_native_blocks(names)?, "{names:?}: split FP8 blocks require BF16 or native E4M3 sources");
        (0..ranks).map(|rank| self.native_blocks(rank, names, Some((axis, ranks)))).collect()
    }

    /// The load stream of rank `rank` (its device is current inside `on_rank`).
    fn rank_stream(&self, rank: usize) -> Result<*mut c_void> {
        Ok(if rank == 0 { self.stream } else { self.peers.get(rank - 1).context("rank stream")?.stream })
    }

    /// [`Self::fp8_split`] for BF16 sources on the GPU: each rank quantizes its
    /// own window of each tensor (whole 128-row or 128-column blocks, so the same
    /// blocks as the whole tensor's), staged in bounded row windows.
    fn gpu_fp8_split(&self, names: &[String], axis: Axis, ranks: usize)
        -> Result<Vec<(DeviceAllocation<'a>, DeviceAllocation<'a>)>> {
        let source = self.checkpoint;
        let tensor = |name: &str| -> Result<&CheckpointTensor> {
            let at = source.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
                .map_err(|_| anyhow::anyhow!("checkpoint has no tensor {name}"))?;
            Ok(&source.tensors[at])
        };
        let shapes = names.iter().map(|n| { let s = &tensor(n)?.meta.shape; Ok((s[0], s[1])) }).collect::<Result<Vec<_>>>()?;
        for (name, &(rows, cols)) in names.iter().zip(&shapes) {
            let along = if axis == Axis::Rows { rows } else { cols };
            ensure!(rows % 128 == 0 && cols % 128 == 0 && along % (128 * ranks) == 0,
                "{name}: [{rows}, {cols}] does not split into whole 128 blocks over {ranks} GPUs");
        }
        let cols = shapes[0].1;
        ensure!(shapes.iter().all(|&(_, c)| c == cols), "{names:?}: FP8 rows must share columns");
        let (rank_rows, rank_cols) = match axis {
            Axis::Rows => (shapes.iter().map(|&(r, _)| r / ranks).sum::<usize>(), cols),
            Axis::Cols => (shapes[0].0, cols / ranks),
        };
        let out = (0..ranks).map(|rank| self.on_rank(rank, || Ok((
            DeviceAllocation::new(self.library, (rank_rows * rank_cols).max(256))?,
            DeviceAllocation::new(self.library, (rank_rows / 128 * (rank_cols / 128) * 4).max(256))?,
        )))).collect::<Result<Vec<_>>>()?;
        for (rank, (values, grid)) in out.iter().enumerate() {
            let mut row = 0;
            for (name, &(rows, _)) in names.iter().zip(&shapes) {
                // Row splits stage only this rank's rows; column splits stage whole
                // rows (the scale rule spans 128 columns, inside this rank's window).
                let (window, col0) = match axis {
                    Axis::Rows => (rank * rows / ranks..(rank + 1) * rows / ranks, 0),
                    Axis::Cols => (0..rows, rank * rank_cols),
                };
                let base = row;
                let target = |rows: std::ops::Range<usize>| Target {
                    // SAFETY: these rows of the rank's share lie inside its operands.
                    q: CuteafdDeviceBuffer { ptr: unsafe { values.buffer.ptr.cast::<u8>().add((base + rows.start) * rank_cols) }.cast(),
                        bytes: rows.len() * rank_cols, ..values.buffer },
                    q_stride: rank_cols,
                    scales: CuteafdDeviceBuffer {
                        // SAFETY: these rows' scale blocks lie inside the rank's grid.
                        ptr: unsafe { grid.buffer.ptr.cast::<u8>().add((base + rows.start) / 128 * (rank_cols / 128) * 4) }.cast(),
                        bytes: rows.len() / 128 * (rank_cols / 128) * 4, ..grid.buffer },
                    kmajor: None,
                };
                let whole = rect(source, name, window.clone(), 0..cols * 2)?;
                self.quant_steps(&[whole], super::fp8::Layout::Block, rank, target, col0, rank_cols)?;
                row += window.len();
            }
        }
        Ok(out)
    }

    /// Layer `layer`, one share per rank of this loader's head split (see the module
    /// docstring); without a split, [`Self::layer`] alone.
    pub fn layer_shares(&self, cfg: &GlmNextConfig, layer: usize, mode: LayerMode) -> Result<Vec<GlmfLayer<'a>>> {
        let inventory = cuteafd_loader::families::glm5_flash::core::CoreInventory::from_checkpoint(cfg, self.checkpoint)?;
        if inventory.projections.keys().any(|stem| stem.starts_with(&format!("{PREFIX}layers.{layer}."))) {
            return self.native_layer_shares(cfg, layer, mode, &inventory);
        }
        let ranks = self.ranks();
        let attention_shares = AttentionShare::bind(mode, ranks)?;
        ensure!(ranks == 1 || !matches!(mode, LayerMode::Whole { ffn: FfnMode::Owner, .. }),
            "GLM Flash two-rank layers retain split FFNs");
        ensure!(ranks == 1 || cfg.attention[layer] == GlmNextAttention::Mla
            || mode == LayerMode::HeadSplit, "GLM Flash KDA layers must remain head-split");
        validate_kda_output_shard(ranks, self.kda_fp8, self.kda_output_shard)?;
        if ranks == 1 {
            return Ok(vec![self.layer(cfg, layer)?]);
        }
        ensure!(self.kda_nvfp4.is_none(), "the KDA NVFP4 numerics gate runs without a head split");
        ensure!(cfg.heads % ranks == 0 && cfg.kda_heads % ranks == 0, "{} MLA / {} KDA heads do not split over {ranks} \
            GPUs", cfg.heads, cfg.kda_heads);
        let p = format!("{PREFIX}layers.{layer}");
        let attention = cfg.attention[layer];
        let dense = cfg.dense[layer];
        let mut ops: Vec<HashMap<&'static str, DeviceAllocation<'a>>> = (0..ranks).map(|_| HashMap::new()).collect();
        let put = |ops: &mut Vec<HashMap<&'static str, DeviceAllocation<'a>>>, key: &'static str,
            parts: Vec<DeviceAllocation<'a>>| {
            for (map, part) in ops.iter_mut().zip(parts) {
                map.insert(key, part);
            }
        };
        for (site, keys) in [("attn", ["attn.fn", "attn.scale", "attn.base"]), ("ffn", ["ffn.fn", "ffn.scale", "ffn.base"])] {
            for (key, suffix) in keys.into_iter().zip(["fn", "scale", "base"]) {
                put(&mut ops, key, self.on_every_rank(|r| self.f32_on(r, &[format!("{p}.hc_{site}_{suffix}")]))?);
            }
        }
        put(&mut ops, "input_norm", self.on_every_rank(|r| self.one_on(r, &format!("{p}.input_layernorm.weight")))?);
        put(&mut ops, "post_norm", self.on_every_rank(|r| self.one_on(r, &format!("{p}.post_attention_layernorm.weight")))?);
        let a = |name: &str| format!("{p}.self_attn.{name}");
        match attention {
            GlmNextAttention::Kda => {
                // [q; k; v; f_a; g_a; b]: q, k, v and b by head, the low-rank f_a and g_a whole.
                let o_name = a("o_proj.weight");
                let o = self.tensor(&o_name)?;
                let (dtype, shape) = (o.meta.dtype.clone(), o.meta.shape.clone());
                ensure!(dtype == DType::Bf16,
                    "{o_name}: a head split requires a BF16 [H, D] o_proj source, found {dtype:?} {shape:?}");
                let (axis, o_rows, o_cols) = kda_output_geometry(cfg.hidden, cfg.kda_heads * cfg.kda_head_dim,
                    &shape, ranks, self.kda_output_shard)?;
                // Each rank's o_proj columns (or, token-row sharded, the whole matrix).
                let w_o: Vec<Vec<Rect>> = match axis {
                    Some(_) => (0..ranks).map(|rank| Ok(vec![rect(self.checkpoint, &o_name, 0..o_rows,
                        rank * o_cols * 2..(rank + 1) * o_cols * 2)?])).collect::<Result<_>>()?,
                    None => Vec::new(), // The full FP8 copy below is quantized once per rank.
                };
                let w_in = self.kda_quant_rects(cfg, &p, KdaOperand::WIn, ranks)?;
                // Single-copy FP8 (--kda-fp8), with K-major scales and no BF16 residency.
                // Row128 column slices retain the same bytes as the whole weight's copy.
                let layout = match self.kda_fp8 {
                    super::fp8::KdaFp8::Off => None,
                    super::fp8::KdaFp8::Channel => Some(super::fp8::Layout::Channel),
                    super::fp8::KdaFp8::Row128 => Some(super::fp8::Layout::Row128),
                };
                if let Some(layout) = layout {
                    for (parts, key) in [(&w_in, "w_in"), (&w_o, "w_o")] {
                        let (fp8, kscale) = if key == "w_in" { ("w_in_fp8", "w_in_kscale") } else { ("w_o_fp8", "w_o_kscale") };
                        if key == "w_o" && axis.is_none() {
                            let whole = rect(self.checkpoint, &o_name, 0..o_rows, 0..o_cols * 2)?;
                            let (q, k): (Vec<_>, Vec<_>) = self.on_every_rank_pair(|r| self.on_rank(r, ||
                                self.gpu_fp8(&[QuantPart::Rect(whole.clone())], layout, true, r)))?.into_iter().unzip();
                            put(&mut ops, fp8, q);
                            put(&mut ops, kscale, k);
                            continue;
                        }
                        // Each rank's head share quantized on its own GPU, K-major scales.
                        let (mut values, mut kmajor) = (Vec::new(), Vec::new());
                        for (rank, rects) in parts.iter().enumerate() {
                            let parts: Vec<_> = rects.iter().cloned().map(QuantPart::Rect).collect();
                            let (q, k) = self.on_rank(rank, || self.gpu_fp8(&parts, layout, true, rank))?;
                            values.push(q);
                            kmajor.push(k);
                        }
                        put(&mut ops, fp8, values);
                        put(&mut ops, kscale, kmajor);
                    }
                } else {
                    put(&mut ops, "w_in", self.kda_operand(cfg, &p, KdaOperand::WIn, ranks)?);
                }
                for operand in [KdaOperand::WFg, KdaOperand::ConvW, KdaOperand::ALog,
                    KdaOperand::DtBias, KdaOperand::ONorm] {
                    put(&mut ops, operand.name(), self.kda_operand(cfg, &p, operand, ranks)?);
                }
                if layout.is_none() {
                    put(&mut ops, "w_o", self.kda_operand(cfg, &p, KdaOperand::WO, ranks)?);
                }
            }
            GlmNextAttention::Mla if matches!(mode, LayerMode::Whole { .. }) => {
                let LayerMode::Whole { gpu, .. } = mode else { unreachable!() };
                let rank = usize::from(gpu);
                self.on_rank(rank, || {
                    // FP8 source conversions enqueue on the owning device's stream.
                    let mut loader = self.clone();
                    if rank > 0 {
                        loader.device = self.peers[rank - 1].device;
                        loader.stream = self.peers[rank - 1].stream;
                    }
                    loader.peers.clear();
                    loader.mla_operands(cfg, &p, &mut ops[rank])
                })?;
            }
            GlmNextAttention::Mla => {
                put(&mut ops, "q_a_norm", self.on_every_rank(|r| self.one_on(r, &a("q_a_layernorm.weight")))?);
                put(&mut ops, "kv_a_norm", self.on_every_rank(|r| self.one_on(r, &a("kv_a_layernorm.weight")))?);
                // W_UK / W_UV are head-major: rank r takes heads r * N / ranks ..
                let (uk, uv): (Vec<_>, Vec<_>) = self.absorbed(cfg, &a("kv_b_proj.weight"), ranks)?.into_iter().unzip();
                put(&mut ops, "w_uk", uk);
                put(&mut ops, "w_uv", uv);
                let projections: Vec<_> = cuteafd_loader::families::glm5_flash::core_fp8_projections("mla")
                    .iter().map(|name| a(name)).collect();
                let (q, s): (Vec<_>, Vec<_>) = self.on_every_rank_pair(|r| self.fp8_on(r, &projections[..2],
                    super::fp8::Layout::Block))?.into_iter().unzip();
                put(&mut ops, "w_qkv_a_fp8", q);
                put(&mut ops, "w_qkv_a_scale", s);
                let (q, s): (Vec<_>, Vec<_>) = self.fp8_split(&projections[2..3], Axis::Rows, ranks)?.into_iter().unzip();
                put(&mut ops, "w_q_b_fp8", q);
                put(&mut ops, "w_q_b_scale", s);
                let (q, s): (Vec<_>, Vec<_>) = self.fp8_split(&projections[3..4], Axis::Cols, ranks)?.into_iter().unzip();
                put(&mut ops, "w_o_fp8", q);
                put(&mut ops, "w_o_scale", s);
                let i = |name: &str| a(&format!("indexer.{name}"));
                put(&mut ops, "w_iq", self.on_every_rank(|r| self.one_on(r, &i("wq_b.weight")))?);
                put(&mut ops, "w_ik", self.on_every_rank(|r| self.rows_on(r, &[i("wk.weight"), i("weights_proj.weight"),
                    i("index_kpool_compress_gate")]))?);
                put(&mut ops, "k_norm_w", self.on_every_rank(|r| self.one_on(r, &i("k_norm.weight")))?);
                put(&mut ops, "k_norm_b", self.on_every_rank(|r| self.one_on(r, &i("k_norm.bias")))?);
                put(&mut ops, "ape", self.on_every_rank(|r| self.one_on(r, &i("index_kpool_compress_ape")))?);
            }
        }
        let mlp = if dense { format!("{p}.mlp") } else { format!("{p}.mlp.shared_experts") };
        // A ModelOpt NVFP4 dense MLP runs whole on rank 0 (rank 1 adds a zero partial).
        if !(dense && self.nvfp4_dense(cfg, &mlp, &mut ops[0])?) {
            let projections: Vec<_> = cuteafd_loader::families::glm5_flash::core_fp8_projections("mlp")
                .iter().map(|name| format!("{mlp}.{name}")).collect();
            let (q, s): (Vec<_>, Vec<_>) = self.fp8_split(&projections[..2], Axis::Rows, ranks)?.into_iter().unzip();
            put(&mut ops, "w_gate_up_fp8", q);
            put(&mut ops, "w_gate_up_scale", s);
            let (q, s): (Vec<_>, Vec<_>) = self.fp8_split(&projections[2..3], Axis::Cols, ranks)?
                .into_iter().unzip();
            put(&mut ops, "w_down_fp8", q);
            put(&mut ops, "w_down_scale", s);
        }
        if !dense {
            put(&mut ops, "gate", self.on_every_rank(|r| self.one_on(r, &format!("{p}.mlp.gate.weight")))?);
            put(&mut ops, "gate.bias", self.on_every_rank(|r| self.f32_on(r,
                &[format!("{p}.mlp.gate.e_score_correction_bias")]))?);
        }
        Ok(ops.into_iter().zip(attention_shares).map(|(operands, attention_share)|
            GlmfLayer { attention, dense, split: true, attention_share, core: Default::default(), operands }).collect())
    }

    /// The admitted layer plan: rank 0's weights (its
    /// layer shares, the norm and head) and with a head split the other rank's layer shares.
    #[allow(clippy::type_complexity)]
    pub fn model(&self, cfg: &GlmNextConfig, plan: &LayerExecutionPlan) -> Result<(GlmfWeights<'a>, Vec<Vec<GlmfLayer<'a>>>)> {
        let inventory = cuteafd_loader::families::glm5_flash::core::CoreInventory::from_checkpoint(cfg, self.checkpoint)?;
        if inventory.projections.is_empty() {
            validate_kda_output_shard(self.ranks(), self.kda_fp8, self.kda_output_shard)?;
        }
        let split = crate::shared::memory::load_timing::Split::start();
        let mut shares: Vec<Vec<GlmfLayer<'a>>> = (0..self.ranks()).map(|_| Vec::new()).collect();
        ensure!(plan.modes().len() <= cfg.layers, "attention plan exceeds checkpoint layers");
        for (layer, &mode) in plan.modes().iter().enumerate() {
            for (share, part) in shares.iter_mut().zip(self.layer_shares(cfg, layer, mode)?) {
                share.push(part);
            }
        }
        let mut shares = shares.into_iter();
        let weights = GlmfWeights {
            layers: shares.next().context("rank 0")?,
            norm: self.one(&format!("{PREFIX}norm.weight"))?,
            head: if let Some(head) = inventory.get_weight("lm_head") {
                super::head::GlmfHead::K6(self.core_weights(
                    cuteafd_loader::formats::trellis_core::CoreWeightLayout::new(head,
                        cuteafd_loader::formats::dense_trellis::TrellisSide::Output, 0, 1)?, 0)?)
            } else if self.fp8_head {
                let (values, scales) = self.fp8(&["lm_head.weight".to_string()], super::fp8::Layout::Row128)?;
                super::head::GlmfHead::Fp8 { values, scales }
            } else {
                super::head::GlmfHead::Bf16(self.one("lm_head.weight")?)
            },
        };
        crate::shared::memory::staging::release_staging();
        // The pinned GPU-quantization source and tables live only for the load.
        self.dense.release();
        split.log(if self.dense.pushing() { "glm5_flash dense planning" } else { "glm5_flash dense" });
        Ok((weights, shares.collect()))
    }
}

#[cfg(test)]
mod tests {
    use super::{kda_output_geometry, quantize_kda_part, validate_kda_output_shard};
    use crate::families::glm5_flash::fp8::{quantize, KdaFp8, Layout};
    use crate::shared::fp8_linear::Fp8Scales;
    use crate::shared::peer_split::{slice_2d, Axis};

    #[test]
    fn split_reads_bf16_blocks_without_a_companion_or_scale_tensors() {
        use cuteafd_loader::plan::checkpoint::{Checkpoint, CheckpointTensor};
        use cuteafd_loader::SafetensorsTensorMetadata;
        let dir = tempfile::tempdir().unwrap();
        let mut bytes = Vec::new();
        for row in 0..256 {
            for col in 0..256 {
                let value = [1.0f32, 2.0, 4.0, 8.0][row / 128 * 2 + col / 128];
                bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
            }
        }
        std::fs::write(dir.path().join("weights.bin"), &bytes).unwrap();
        let checkpoint = Checkpoint { snapshot: dir.path().into(), config: serde_json::json!({}),
            quantize_config: None, weight_map: Default::default(), missing_shards: vec![], shard_bytes: bytes.len() as u64,
            tensors: vec![CheckpointTensor { shard: "weights.bin".into(), meta: SafetensorsTensorMetadata {
                name: "projection.weight".into(), dtype: cuteafd_core::DType::Bf16, shape: vec![256, 256],
                byte_offset: 0, byte_length: bytes.len() as u64 } }] };
        let (values, grid, rows, cols) = super::GlmfLoader::fp8_block_host(&checkpoint, "projection.weight",
            crate::shared::fp8_linear::Fp8Scales::Amax).unwrap();
        assert_eq!((rows, cols), (256, 256));
        assert_eq!(values, vec![0x7e; 256 * 256]);
        let grid: Vec<f32> = grid.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        assert_eq!(grid, [1.0 / 448.0, 2.0 / 448.0, 4.0 / 448.0, 8.0 / 448.0]);
    }

    #[test]
    fn kmajor_scales_transpose_row_blocks() {
        let rows: Vec<f32> = (0..6).map(|i| i as f32).collect(); // [n=3, kb=2]
        let bytes: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
        let out: Vec<f32> = super::kmajor_scales(&bytes, 3, 2).chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        assert_eq!(out, [0.0, 2.0, 4.0, 1.0, 3.0, 5.0]);
    }

    #[test]
    fn kda_token_row_output_checks_geometry_and_resident_bytes() {
        let (hidden, width) = (4096, 8192);
        let old = kda_output_geometry(hidden, width, &[hidden, width], 2, false).unwrap();
        let shard = kda_output_geometry(hidden, width, &[hidden, width], 2, true).unwrap();
        assert_eq!(old, (Some(Axis::Cols), 4096, 4096));
        assert_eq!(shard, (None, 4096, 8192));
        let bytes = |(_, n, k): (Option<Axis>, usize, usize)| (n * k, n * (k / 128) * 4);
        assert_eq!(bytes(old), (16_777_216, 524_288));
        assert_eq!(bytes(shard), (33_554_432, 1_048_576));
        let old_bytes = bytes(old).0 + bytes(old).1;
        let full_bytes = bytes(shard).0 + bytes(shard).1;
        assert_eq!(full_bytes - old_bytes, 17_301_504);
        assert_eq!((full_bytes - old_bytes) * 34, 588_251_136); // 561 MiB per GPU.
        for shape in [&[4096, 4096][..], &[2048, 8192], &[4096, 8192, 1], &[]] {
            assert!(kda_output_geometry(hidden, width, shape, 2, true).is_err());
        }
        assert!(kda_output_geometry(hidden, width, &[hidden, width], 3, true).is_err());
        assert!(kda_output_geometry(hidden, width, &[hidden, width], 0, true).is_err());
        assert!(kda_output_geometry(hidden, 8193, &[hidden, 8193], 2, true).is_err());
        assert!(kda_output_geometry(4095, width, &[4095, width], 2, true).is_err());
        assert_eq!(kda_output_geometry(hidden, width, &[hidden, width], 1, false).unwrap(),
            (Some(Axis::Cols), hidden, width));
    }

    #[test]
    fn kda_output_shard_requires_two_fp8_ranks() {
        for fp8 in [KdaFp8::Row128, KdaFp8::Channel] {
            assert!(validate_kda_output_shard(2, fp8, true).is_ok());
            for ranks in [0, 1, 3, 4] {
                assert!(validate_kda_output_shard(ranks, fp8, true).is_err());
            }
        }
        assert!(validate_kda_output_shard(2, KdaFp8::Off, true).is_err());
        assert!(validate_kda_output_shard(1, KdaFp8::Off, false).is_ok());
        assert!(validate_kda_output_shard(2, KdaFp8::Off, false).is_ok());
    }

    #[test]
    fn kda_token_rows_keep_whole_weight_fp8_payload_and_kmajor_scale_bits() {
        // Non-128-aligned N catches confusing output rows with scale blocks.
        let (rows, cols) = (34, 512);
        let bytes: Vec<u8> = (0..rows * cols).flat_map(|at| {
            let (row, col) = (at / cols, at % cols);
            let x = ((row * 37 + col * 19) % 101) as f32 - 50.0;
            let x = x * (1 + row + 7 * (col / 128)) as f32 * 0.01;
            ((x.to_bits() >> 16) as u16).to_le_bytes()
        }).collect();
        let (axis, n, k) = kda_output_geometry(rows, cols, &[rows, cols], 2, true).unwrap();
        assert!(axis.is_none());
        for layout in [Layout::Row128, Layout::Channel] {
            for rule in [Fp8Scales::Amax, Fp8Scales::Pow2, Fp8Scales::Best] {
                let (whole, scales) = quantize(&bytes, rows, cols, layout, rule);
                let (values, kmajor) = quantize_kda_part(&bytes, n, k, layout, rule);
                assert_eq!(values, whole, "whole payload: {layout:?} {rule:?}");
                assert_eq!((values.len(), kmajor.len()), (rows * cols, rows * (cols / 128) * 4));
                for row in 0..rows {
                    for block in 0..cols / 128 {
                        let offset = (block * rows + row) * 4;
                        assert_eq!(&kmajor[offset..offset + 4], &scales[row * (cols / 128) + block].to_le_bytes(),
                            "whole scale bits: {layout:?} {rule:?} row {row} block {block}");
                    }
                }
            }
        }
    }

    #[test]
    fn kda_column_split_keeps_row128_blocks_and_scale_bits() {
        let (rows, cols) = (6, 512);
        let bytes: Vec<u8> = (0..rows * cols).flat_map(|at| {
            let x = ((at * 37) % 127) as f32 - 63.0;
            let x = x * (1 + at / cols + 9 * (at % cols / 128)) as f32 * 0.01;
            ((x.to_bits() >> 16) as u16).to_le_bytes()
        }).collect();
        let (axis, n, k) = kda_output_geometry(rows, cols, &[rows, cols], 2, false).unwrap();
        assert_eq!(axis, Some(Axis::Cols));
        for rule in [Fp8Scales::Amax, Fp8Scales::Pow2, Fp8Scales::Best] {
            let (whole, scales) = quantize(&bytes, rows, cols, Layout::Row128, rule);
            for rank in 0..2 {
                let part = slice_2d(&bytes, rows, cols, 2, axis.unwrap(), rank, 2);
                let (values, kmajor) = quantize_kda_part(&part, n, k, Layout::Row128, rule);
                assert_eq!(values, slice_2d(&whole, rows, cols, 1, Axis::Cols, rank, 2));
                for row in 0..rows {
                    for block in 0..k / 128 {
                        let offset = (block * rows + row) * 4;
                        let whole_block = rank * (k / 128) + block;
                        assert_eq!(&kmajor[offset..offset + 4], &scales[row * (cols / 128) + whole_block].to_le_bytes(),
                            "column scale bits: {rule:?} rank {rank} row {row} block {block}");
                    }
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "weights_golden_tests.rs"]
mod golden_tests;
