//! GLM 5.3 Flash coordinator weights, packed for the exported glmf_* programs.
//!
//! The MLA (`q_a|kv_a`, `q_b`, `o_proj`), dense and shared-expert projections
//! are FP8 only: the official FP8 release's E4M3 bytes and FP32 128x128 block
//! scales (`--fp8-snapshot` or native FP8 in the primary checkpoint), else
//! 128x128 blocks quantized from BF16 at load (the BF16 is never resident).
//! Every other matrix operand is BF16 (the EXL3
//! publications store the dense tensors in BF16, equal to the official BF16
//! release; FP8 checkpoint tensors are dequantized on the GPU with their FP32
//! 128x128 block scales), except where an FP8 representation is selected
//! instead: the KDA in/out projections (`--kda-fp8 row128|channel`: E4M3 with
//! per-row x 128-K scales stored K-block major, `w_in_fp8`/`w_in_kscale`,
//! `w_o_fp8`/`w_o_kscale`, no BF16 `w_in`/`w_o`) and the LM head
//! (`--fp8-head`, [`super::head::GlmfHead::Fp8`]). One resident copy each.
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
//! latent projection (`q_a | kv_a`) and the DSA indexer; rank 0 the router. A ModelOpt NVFP4
//! dense MLP stays whole on rank 0 (its package has no half geometry).
//! With `kda_output_shard`, the output projection instead splits live token rows:
//! each rank owns one full FP8 `[H, D]` matrix and consumes complete gathered
//! KDA activations for its token rows. Quantization runs once before replication.
use crate::shared::memory::DeviceAllocation;
use crate::shared::peer_split::{slice_2d, Axis, RankDevice};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::families::glm5_flash::{GlmNextAttention, GlmNextConfig};
use cuteafd_loader::plan::checkpoint::{Checkpoint, CheckpointTensor};
use std::collections::HashMap;
use std::ffi::c_void;
use std::os::unix::fs::FileExt;

pub(crate) const PREFIX: &str = "model.language_model.";

pub(crate) struct GlmfLayer<'a> {
    pub attention: GlmNextAttention,
    pub dense: bool,
    /// One GPU's share of a head split: it runs the split (`glmf2`) programs.
    pub split: bool,
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
        self.operands.values().map(|a| a.buffer.bytes).sum()
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
            ensure!(fp8 == (kda_fp8 != super::fp8::KdaFp8::Off),
                "KDA layer {index}: --kda-fp8 {kda_fp8:?} but {} projections are resident", if fp8 { "FP8" } else { "BF16" });
        }
        ensure!(matches!(self.head, super::head::GlmfHead::Fp8 { .. }) == fp8_head,
            "--fp8-head {fp8_head} but the resident head is {}", self.head.name());
        Ok(self.residency())
    }
}

pub(crate) struct GlmfLoader<'a> {
    pub library: &'a NativeLibrary,
    pub checkpoint: &'a Checkpoint,
    pub stream: *mut c_void,
    /// The official FP8 release: MLA/dense/shared weights (their only copies,
    /// 128x128 blocks) come from it when given, else from primary native FP8.
    pub fp8_source: Option<&'a Checkpoint>,
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
}

fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16)).collect()
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

/// Quantize one KDA slice using the whole weight's per-row rule, with `[kb,n]` scales.
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
        std::fs::File::open(self.checkpoint.snapshot.join(&tensor.shard))?
            .read_exact_at(&mut bytes, tensor.meta.byte_offset)
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
        for (proj, w_key, s_key, rows, cols) in [("gate_proj", "nvfp4_w1", "nvfp4_s1", i, h),
            ("up_proj", "nvfp4_w3", "nvfp4_s3", i, h), ("down_proj", "nvfp4_w2", "nvfp4_s2", h, i)] {
            let name = format!("{mlp}.{proj}");
            let (weight, dtype, shape) = self.raw(&format!("{name}.weight"))?;
            ensure!(dtype == DType::U8 && shape == [rows, cols / 2], "{name}.weight: expected packed E2M1 U8 [{rows}, {}], \
                found {dtype:?} {shape:?}", cols / 2);
            let (mut scales, dtype, shape) = self.raw(&format!("{name}.weight_scale"))?;
            ensure!(dtype == DType::F8E4M3 && shape == [rows, cols / 16], "{name}.weight_scale: expected E4M3 \
                [{rows}, {}], found {dtype:?} {shape:?}", cols / 16);
            for scalar in ["weight_scale_2", "input_scale"] {
                let (bytes, dtype, _) = self.raw(&format!("{name}.{scalar}"))?;
                ensure!(dtype == DType::F32 && bytes.len() == 4, "{name}.{scalar}: expected one FP32 value");
                scales.extend_from_slice(&bytes);
            }
            ops.insert(w_key, self.upload(&weight)?);
            ops.insert(s_key, self.upload(&scales)?);
        }
        Ok(true)
    }

    fn upload(&self, bytes: &[u8]) -> Result<DeviceAllocation<'a>> {
        let allocation = DeviceAllocation::new(self.library, bytes.len().max(256))?;
        self.library.copy_h2d(allocation.buffer, bytes)?;
        Ok(allocation)
    }

    /// The row-concatenation of 2-D `names` as one BF16 operand (FP8 blocks dequantized).
    fn rows(&self, names: &[String]) -> Result<DeviceAllocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        let tensors = names.iter().map(|n| self.raw(n).map(|t| (n, t))).collect::<Result<Vec<_>>>()?;
        let cols = tensors[0].1 .2[1];
        let rows: usize = tensors.iter().map(|(_, (_, _, shape))| shape[0]).sum();
        ensure!(tensors.iter().all(|(_, (_, _, s))| s.len() == 2 && s[1] == cols), "{names:?} do not share columns");
        let out = DeviceAllocation::new(self.library, rows * cols * 2)?;
        let mut row = 0;
        for (name, (bytes, dtype, shape)) in &tensors {
            let dest = |first: usize, count: usize| CuteafdDeviceBuffer {
                // SAFETY: rows row+first..row+first+count lie inside `out`.
                ptr: unsafe { out.buffer.ptr.cast::<u8>().add((row + first) * cols * 2) }.cast(),
                bytes: count * cols * 2,
                ..out.buffer
            };
            match dtype {
                DType::Bf16 => self.library.copy_h2d(dest(0, shape[0]), bytes)?,
                DType::F8E4M3 => {
                    let (scale, scale_dtype, scale_shape) = self.raw(&format!("{name}_scale_inv"))?;
                    ensure!(scale_dtype == DType::F32, "{name}: block scales must be FP32");
                    ensure!(scale_shape == [shape[0].div_ceil(128), cols.div_ceil(128)],
                        "{name}: unexpected scale grid {scale_shape:?}");
                    let (w, s) = (self.upload(bytes)?, self.upload(&scale)?);
                    // SAFETY: the FP8 weight, its scales and the destination rows are live;
                    // the stream drains before `w`/`s` drop.
                    unsafe {
                        self.library.fp8_block_dequant(w.buffer.ptr, s.buffer.ptr, dest(0, shape[0]).ptr, shape[0],
                            cols, self.stream)?;
                        self.library.cuda_stream_synchronize(self.stream)?;
                    }
                }
                other => anyhow::bail!("{name}: unsupported coordinator dtype {other:?}"),
            }
            row += shape[0];
        }
        Ok(out)
    }

    /// The row-concatenation of 2-D `names` as E4M3 bytes and FP32 scales in
    /// `layout`: the FP8 source checkpoint's own blocks when it stores the
    /// tensors as FP8 (block layout), else quantized from BF16 at load (the
    /// FP8 copy is the only resident one).
    fn fp8(&self, names: &[String], layout: super::fp8::Layout) -> Result<(DeviceAllocation<'a>, DeviceAllocation<'a>)> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("fp8");
        if layout == super::fp8::Layout::Block {
            if let Some(copy) = self.fp8_blocks(names)? {
                return Ok(copy);
            }
        }
        let (values, scales, _) = self.fp8_host(names, layout)?;
        Ok((self.upload(&values)?, self.upload(&scales)?))
    }

    /// The FP8 source's own E4M3 blocks and FP32 128x128 grids of `names`
    /// (row-concatenated) on the device, read through this thread's staging
    /// buffer; None when a part is not an FP8 tensor there (quantized instead).
    fn fp8_blocks(&self, names: &[String]) -> Result<Option<(DeviceAllocation<'a>, DeviceAllocation<'a>)>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("fp8");
        let checkpoint = self.fp8_source.unwrap_or(self.checkpoint);
        let find = |name: &str| -> Result<&'a CheckpointTensor> {
            let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
                .map_err(|_| anyhow::anyhow!("FP8 checkpoint has no tensor {name}"))?;
            Ok(&checkpoint.tensors[at])
        };
        let tensors = names.iter().map(|n| find(n)).collect::<Result<Vec<_>>>()?;
        if tensors.iter().any(|t| t.meta.dtype != DType::F8E4M3 || t.meta.shape.len() != 2) {
            return Ok(None);
        }
        let total: usize = tensors.iter().map(|t| t.meta.byte_length as usize).sum();
        crate::shared::memory::staging::with_staging(total, |values| {
            let (mut grid, mut at) = (Vec::new(), 0usize);
            for (i, (name, tensor)) in names.iter().zip(&tensors).enumerate() {
                let shape = &tensor.meta.shape;
                ensure!(shape[1] == tensors[0].meta.shape[1] && (i + 1 == names.len() || shape[0] % 128 == 0),
                    "{names:?}: FP8 rows must share columns and fill whole 128-row blocks but the last");
                let length = tensor.meta.byte_length as usize;
                std::fs::File::open(checkpoint.snapshot.join(&tensor.shard))?
                    .read_exact_at(&mut values[at..at + length], tensor.meta.byte_offset)
                    .with_context(|| format!("reading {name}"))?;
                at += length;
                let scale = find(&format!("{name}_scale_inv"))?;
                ensure!(scale.meta.dtype == DType::F32 && scale.meta.shape == [shape[0] / 128, shape[1].div_ceil(128)],
                    "{name}: expected FP32 128x128 block scales, found {:?} {:?}", scale.meta.dtype, scale.meta.shape);
                let mut bytes = vec![0u8; scale.meta.byte_length as usize];
                std::fs::File::open(checkpoint.snapshot.join(&scale.shard))?
                    .read_exact_at(&mut bytes, scale.meta.byte_offset)?;
                grid.extend_from_slice(&bytes);
            }
            Ok(Some((self.upload(values)?, self.upload(&grid)?)))
        })
    }

    /// Per-row FP8 `names` (`layout` Row128 or Channel) with its scales stored
    /// K-block major (`[K/128, N]`), the one layout the `kda_w8` decode and
    /// prefill programs both read: (values, K-major scales).
    fn fp8_rows_kmajor(&self, names: &[String], layout: super::fp8::Layout)
        -> Result<(DeviceAllocation<'a>, DeviceAllocation<'a>)> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("fp8");
        ensure!(layout != super::fp8::Layout::Block, "{names:?}: K-major scales are per-row scales");
        let (values, scales, cols) = self.fp8_host(names, layout)?;
        Ok((self.upload(&values)?, self.upload(&kmajor_scales(&scales, values.len() / cols, cols / 128))?))
    }

    /// Host bytes of [`Self::fp8`]: E4M3 values, FP32 scales, and the column count.
    fn fp8_host(&self, names: &[String], layout: super::fp8::Layout) -> Result<(Vec<u8>, Vec<u8>, usize)> {
        use super::fp8::{quantize, Layout};
        let source = |name: &str| -> Result<(Vec<u8>, DType, Vec<usize>)> {
            match self.fp8_source {
                Some(checkpoint) => {
                    let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
                        .map_err(|_| anyhow::anyhow!("FP8 checkpoint has no tensor {name}"))?;
                    let tensor = &checkpoint.tensors[at];
                    let mut bytes = vec![0u8; tensor.meta.byte_length as usize];
                    std::fs::File::open(checkpoint.snapshot.join(&tensor.shard))?
                        .read_exact_at(&mut bytes, tensor.meta.byte_offset)?;
                    Ok((bytes, tensor.meta.dtype.clone(), tensor.meta.shape.clone()))
                }
                None => self.raw(name),
            }
        };
        let (mut values, mut scales) = (Vec::new(), Vec::<u8>::new());
        let mut cols = None;
        for name in names {
            let (bytes, dtype, shape) = source(name)?;
            ensure!(shape.len() == 2 && cols.is_none_or(|c| c == shape[1]), "{name}: FP8 rows must share columns");
            cols = Some(shape[1]);
            match dtype {
                DType::F8E4M3 if layout == Layout::Block => {
                    let (scale, scale_dtype, scale_shape) = source(&format!("{name}_scale_inv"))?;
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

    fn one(&self, name: &str) -> Result<DeviceAllocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        let (bytes, dtype, shape) = self.raw(name)?;
        if shape.len() == 2 && dtype == DType::F8E4M3 {
            return self.rows(&[name.to_string()]);
        }
        self.upload(&bytes)
    }

    /// A BF16 (or FP32) tensor widened to FP32.
    fn f32(&self, names: &[String]) -> Result<DeviceAllocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("f32");
        let mut values = Vec::new();
        for name in names {
            let (bytes, dtype, _) = self.raw(name)?;
            match dtype {
                DType::Bf16 => values.extend(bf16_to_f32(&bytes)),
                DType::F32 => values.extend(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap()))),
                other => anyhow::bail!("{name}: expected BF16 or FP32, found {other:?}"),
            }
        }
        self.upload(&f32_bytes(&values))
    }

    /// `kv_b_proj [N*(nope+v), 512]` -> `w_uk [N, 512, nope]` (key rows transposed per head)
    /// and `w_uv [N, v, 512]`.
    fn absorbed(&self, cfg: &GlmNextConfig, name: &str) -> Result<(DeviceAllocation<'a>, DeviceAllocation<'a>)> {
        let (uk, uv) = self.absorbed_host(cfg, name)?;
        Ok((self.upload(&uk)?, self.upload(&uv)?))
    }

    /// [`Self::absorbed`]'s host bytes.
    fn absorbed_host(&self, cfg: &GlmNextConfig, name: &str) -> Result<(Vec<u8>, Vec<u8>)> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        let (bytes, dtype, shape) = self.raw(name)?;
        let (n, nope, v, lat) = (cfg.heads, cfg.qk_nope_dim, cfg.v_head_dim, cfg.kv_lora_rank);
        ensure!(dtype == DType::Bf16 && shape == [n * (nope + v), lat], "{name}: expected BF16 [{}, {lat}]",
            n * (nope + v));
        let at = |row: usize, col: usize| -> [u8; 2] {
            let i = (row * lat + col) * 2;
            [bytes[i], bytes[i + 1]]
        };
        let mut uk = Vec::with_capacity(n * lat * nope * 2);
        let mut uv = Vec::with_capacity(n * v * lat * 2);
        for h in 0..n {
            let base = h * (nope + v);
            for c in 0..lat {
                for d in 0..nope {
                    uk.extend_from_slice(&at(base + d, c));
                }
            }
            for r in 0..v {
                uv.extend_from_slice(&bytes[(base + nope + r) * lat * 2..(base + nope + r + 1) * lat * 2]);
            }
        }
        Ok((uk, uv))
    }

    pub fn layer(&self, cfg: &GlmNextConfig, layer: usize) -> Result<GlmfLayer<'a>> {
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
                let w_in = ["q_proj", "k_proj", "v_proj", "f_a_proj", "g_a_proj", "b_proj"].map(|n| a(&format!("{n}.weight")));
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
                    ops.insert("w_in", self.rows(&w_in)?);
                }
                ops.insert("w_fg", self.rows(&[a("f_b_proj.weight"), a("g_b_proj.weight")])?);
                // [3D, 1, 4] each -> FP32 [3D, 4].
                ops.insert("conv_w", self.f32(&["q", "k", "v"].map(|n| a(&format!("{n}_conv1d.weight"))))?);
                ops.insert("a_log", self.f32(&[a("A_log")])?);
                ops.insert("dt_bias", self.f32(&[a("dt_bias")])?);
                ops.insert("o_norm", self.one(&a("o_norm.weight"))?);
                if kda_layout.is_none() {
                    let w_o = match ops.remove("w_o_nvfp4") {
                        Some(rounded) => rounded,
                        None => self.one(&a("o_proj.weight"))?,
                    };
                    ops.insert("w_o", w_o);
                }
            }
            GlmNextAttention::Mla => {
                ops.insert("q_a_norm", self.one(&a("q_a_layernorm.weight"))?);
                ops.insert("kv_a_norm", self.one(&a("kv_a_layernorm.weight"))?);
                let (uk, uv) = self.absorbed(cfg, &a("kv_b_proj.weight"))?;
                ops.insert("w_uk", uk);
                ops.insert("w_uv", uv);
                let block = super::fp8::Layout::Block;
                let (q, s) = self.fp8(&[a("q_a_proj.weight"), a("kv_a_proj_with_mqa.weight")], block)?;
                ops.insert("w_qkv_a_fp8", q);
                ops.insert("w_qkv_a_scale", s);
                let (q, s) = self.fp8(&[a("q_b_proj.weight")], block)?;
                ops.insert("w_q_b_fp8", q);
                ops.insert("w_q_b_scale", s);
                let (q, s) = self.fp8(&[a("o_proj.weight")], block)?;
                ops.insert("w_o_fp8", q);
                ops.insert("w_o_scale", s);
                let i = |name: &str| a(&format!("indexer.{name}"));
                ops.insert("w_iq", self.one(&i("wq_b.weight"))?);
                ops.insert("w_ik", self.rows(&[i("wk.weight"), i("weights_proj.weight"),
                    i("index_kpool_compress_gate")])?);
                ops.insert("k_norm_w", self.one(&i("k_norm.weight"))?);
                ops.insert("k_norm_b", self.one(&i("k_norm.bias"))?);
                ops.insert("ape", self.one(&i("index_kpool_compress_ape"))?);
            }
        }
        let mlp = if dense { format!("{p}.mlp") } else { format!("{p}.mlp.shared_experts") };
        if !(dense && self.nvfp4_dense(cfg, &mlp, &mut ops)?) {
            let block = super::fp8::Layout::Block;
            let (q, s) = self.fp8(&[format!("{mlp}.gate_proj.weight"), format!("{mlp}.up_proj.weight")], block)?;
            ops.insert("w_gate_up_fp8", q);
            ops.insert("w_gate_up_scale", s);
            let (q, s) = self.fp8(&[format!("{mlp}.down_proj.weight")], block)?;
            ops.insert("w_down_fp8", q);
            ops.insert("w_down_scale", s);
        }
        if !dense {
            ops.insert("gate", self.one(&format!("{p}.mlp.gate.weight"))?);
            ops.insert("gate.bias", self.f32(&[format!("{p}.mlp.gate.e_score_correction_bias")])?);
        }
        Ok(GlmfLayer { attention, dense, split: false, operands: ops })
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
        crate::shared::peer_split::on_device(self.library, peer.device, self.device, body)
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

    /// Per rank its host bytes uploaded on that rank's GPU.
    fn upload_ranks(&self, parts: Vec<Vec<u8>>) -> Result<Vec<DeviceAllocation<'a>>> {
        parts.into_iter().enumerate().map(|(rank, bytes)| self.on_rank(rank, || self.upload(&bytes))).collect()
    }

    /// The row-concatenation of BF16 matrices `parts`, each either split into `ranks`
    /// contiguous row blocks (`true`: rank `r` takes block `r`) or whole on every rank.
    fn bf16_parts(&self, parts: &[(String, bool)], ranks: usize) -> Result<Vec<Vec<u8>>> {
        let mut out = vec![Vec::new(); ranks];
        for (name, split) in parts {
            let (bytes, dtype, shape) = self.raw(name)?;
            ensure!(dtype == DType::Bf16 && shape.len() == 2, "{name}: a head split slices BF16 matrices, found \
                {dtype:?} {shape:?}");
            ensure!(!split || shape[0] % ranks == 0, "{name}: {} rows do not split over {ranks} GPUs", shape[0]);
            for (rank, share) in out.iter_mut().enumerate() {
                share.extend_from_slice(if *split { &bytes[rank * bytes.len() / ranks..(rank + 1) * bytes.len() / ranks] }
                    else { &bytes[..] });
            }
        }
        Ok(out)
    }

    /// BF16 / FP32 tensors `names` widened to FP32, each split into `ranks` contiguous
    /// blocks along its first dimension; per rank its blocks back to back.
    fn f32_parts(&self, names: &[String], ranks: usize) -> Result<Vec<Vec<u8>>> {
        let mut out = vec![Vec::new(); ranks];
        for name in names {
            let (bytes, dtype, _) = self.raw(name)?;
            let values = match dtype {
                DType::Bf16 => bf16_to_f32(&bytes),
                DType::F32 => bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
                other => anyhow::bail!("{name}: expected BF16 or FP32, found {other:?}"),
            };
            ensure!(values.len() % ranks == 0, "{name}: {} values do not split over {ranks} GPUs", values.len());
            let share = values.len() / ranks;
            for (rank, part) in out.iter_mut().enumerate() {
                part.extend(f32_bytes(&values[rank * share..(rank + 1) * share]));
            }
        }
        Ok(out)
    }

    /// The selected source's 2-D `name` as E4M3 blocks, quantized from BF16
    /// when necessary: values, FP32 scale grid, rows, cols.
    fn fp8_block_host(checkpoint: &Checkpoint, name: &str, scales: crate::shared::fp8_linear::Fp8Scales)
        -> Result<(Vec<u8>, Vec<u8>, usize, usize)> {
        let read = |name: &str| -> Result<(Vec<u8>, DType, Vec<usize>)> {
            let at = checkpoint.tensors.binary_search_by(|t| t.meta.name.as_str().cmp(name))
                .map_err(|_| anyhow::anyhow!("FP8 checkpoint has no tensor {name}"))?;
            let tensor = &checkpoint.tensors[at];
            let mut bytes = vec![0u8; tensor.meta.byte_length as usize];
            std::fs::File::open(checkpoint.snapshot.join(&tensor.shard))?
                .read_exact_at(&mut bytes, tensor.meta.byte_offset).with_context(|| format!("reading {name}"))?;
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
        let (mut values, mut grids) = (vec![Vec::new(); ranks], vec![Vec::new(); ranks]);
        for name in names {
            let (v, g, rows, cols) = Self::fp8_block_host(self.fp8_source.unwrap_or(self.checkpoint), name,
                self.fp8_scales)?;
            let along = if axis == Axis::Rows { rows } else { cols };
            ensure!(along % (128 * ranks) == 0, "{name}: [{rows}, {cols}] does not split into whole 128 blocks over \
                {ranks} GPUs");
            for rank in 0..ranks {
                values[rank].extend(slice_2d(&v, rows, cols, 1, axis, rank, ranks));
                grids[rank].extend(slice_2d(&g, rows / 128, cols / 128, 4, axis, rank, ranks));
            }
        }
        values.into_iter().zip(grids).enumerate()
            .map(|(rank, (v, g))| self.on_rank(rank, || Ok((self.upload(&v)?, self.upload(&g)?)))).collect()
    }

    /// Layer `layer`, one share per rank of this loader's head split (see the module
    /// docstring); without a split, [`Self::layer`] alone.
    pub fn layer_shares(&self, cfg: &GlmNextConfig, layer: usize) -> Result<Vec<GlmfLayer<'a>>> {
        let ranks = self.ranks();
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
                put(&mut ops, key, self.replicate(self.f32(&[format!("{p}.hc_{site}_{suffix}")])?)?);
            }
        }
        put(&mut ops, "input_norm", self.replicate(self.one(&format!("{p}.input_layernorm.weight"))?)?);
        put(&mut ops, "post_norm", self.replicate(self.one(&format!("{p}.post_attention_layernorm.weight"))?)?);
        let a = |name: &str| format!("{p}.self_attn.{name}");
        match attention {
            GlmNextAttention::Kda => {
                // [q; k; v; f_a; g_a; b]: q, k, v and b by head, the low-rank f_a and g_a whole.
                let w_in: Vec<(String, bool)> = [("q_proj", true), ("k_proj", true), ("v_proj", true),
                    ("f_a_proj", false), ("g_a_proj", false), ("b_proj", true)].iter()
                    .map(|(n, split)| (a(&format!("{n}.weight")), *split)).collect();
                let w_in = self.bf16_parts(&w_in, ranks)?;
                let (bytes, dtype, shape) = self.raw(&a("o_proj.weight"))?;
                ensure!(dtype == DType::Bf16,
                    "{}: a head split requires a BF16 [H, D] o_proj source, found {dtype:?} {shape:?}", a("o_proj.weight"));
                let (axis, o_rows, o_cols) = kda_output_geometry(cfg.hidden, cfg.kda_heads * cfg.kda_head_dim,
                    &shape, ranks, self.kda_output_shard)?;
                let w_o: Vec<Vec<u8>> = match axis {
                    Some(axis) => (0..ranks)
                        .map(|rank| slice_2d(&bytes, shape[0], shape[1], 2, axis, rank, ranks)).collect(),
                    None => Vec::new(), // The full FP8 copy below is quantized and uploaded once.
                };
                // Single-copy FP8 (--kda-fp8), with K-major scales and no BF16 residency.
                // Row128 column slices retain the same bytes as the whole weight's copy.
                let layout = match self.kda_fp8 {
                    super::fp8::KdaFp8::Off => None,
                    super::fp8::KdaFp8::Channel => Some(super::fp8::Layout::Channel),
                    super::fp8::KdaFp8::Row128 => Some(super::fp8::Layout::Row128),
                };
                if let Some(layout) = layout {
                    let h = cfg.hidden;
                    for (parts, key, cols) in [(&w_in, "w_in", h), (&w_o, "w_o", o_cols)] {
                        let (fp8, kscale) = if key == "w_in" { ("w_in_fp8", "w_in_kscale") } else { ("w_o_fp8", "w_o_kscale") };
                        if key == "w_o" && axis.is_none() {
                            let (q, k) = quantize_kda_part(&bytes, o_rows, o_cols, layout, self.fp8_scales);
                            put(&mut ops, fp8, self.replicate(self.upload(&q)?)?);
                            put(&mut ops, kscale, self.replicate(self.upload(&k)?)?);
                            continue;
                        }
                        let (mut values, mut kmajor) = (Vec::new(), Vec::new());
                        for part in parts {
                            let n = part.len() / 2 / cols;
                            let (q, k) = quantize_kda_part(part, n, cols, layout, self.fp8_scales);
                            values.push(q);
                            kmajor.push(k);
                        }
                        put(&mut ops, fp8, self.upload_ranks(values)?);
                        put(&mut ops, kscale, self.upload_ranks(kmajor)?);
                    }
                } else {
                    put(&mut ops, "w_in", self.upload_ranks(w_in)?);
                }
                put(&mut ops, "w_fg", self.upload_ranks(self.bf16_parts(&[(a("f_b_proj.weight"), true),
                    (a("g_b_proj.weight"), true)], ranks)?)?);
                put(&mut ops, "conv_w", self.upload_ranks(self.f32_parts(&["q", "k", "v"]
                    .map(|n| a(&format!("{n}_conv1d.weight"))), ranks)?)?);
                put(&mut ops, "a_log", self.upload_ranks(self.f32_parts(&[a("A_log")], ranks)?)?);
                put(&mut ops, "dt_bias", self.upload_ranks(self.f32_parts(&[a("dt_bias")], ranks)?)?);
                put(&mut ops, "o_norm", self.replicate(self.one(&a("o_norm.weight"))?)?);
                if layout.is_none() {
                    put(&mut ops, "w_o", self.upload_ranks(w_o)?);
                }
            }
            GlmNextAttention::Mla => {
                put(&mut ops, "q_a_norm", self.replicate(self.one(&a("q_a_layernorm.weight"))?)?);
                put(&mut ops, "kv_a_norm", self.replicate(self.one(&a("kv_a_layernorm.weight"))?)?);
                // W_UK / W_UV are head-major: rank r takes heads r * N / ranks ..
                let (uk, uv) = self.absorbed_host(cfg, &a("kv_b_proj.weight"))?;
                let part = |bytes: &[u8], rank: usize| bytes[rank * bytes.len() / ranks..(rank + 1) * bytes.len() / ranks]
                    .to_vec();
                put(&mut ops, "w_uk", self.upload_ranks((0..ranks).map(|r| part(&uk, r)).collect())?);
                put(&mut ops, "w_uv", self.upload_ranks((0..ranks).map(|r| part(&uv, r)).collect())?);
                let (q, s) = self.fp8(&[a("q_a_proj.weight"), a("kv_a_proj_with_mqa.weight")], super::fp8::Layout::Block)?;
                put(&mut ops, "w_qkv_a_fp8", self.replicate(q)?);
                put(&mut ops, "w_qkv_a_scale", self.replicate(s)?);
                let (q, s): (Vec<_>, Vec<_>) = self.fp8_split(&[a("q_b_proj.weight")], Axis::Rows, ranks)?.into_iter().unzip();
                put(&mut ops, "w_q_b_fp8", q);
                put(&mut ops, "w_q_b_scale", s);
                let (q, s): (Vec<_>, Vec<_>) = self.fp8_split(&[a("o_proj.weight")], Axis::Cols, ranks)?.into_iter().unzip();
                put(&mut ops, "w_o_fp8", q);
                put(&mut ops, "w_o_scale", s);
                let i = |name: &str| a(&format!("indexer.{name}"));
                put(&mut ops, "w_iq", self.replicate(self.one(&i("wq_b.weight"))?)?);
                put(&mut ops, "w_ik", self.replicate(self.rows(&[i("wk.weight"), i("weights_proj.weight"),
                    i("index_kpool_compress_gate")])?)?);
                put(&mut ops, "k_norm_w", self.replicate(self.one(&i("k_norm.weight"))?)?);
                put(&mut ops, "k_norm_b", self.replicate(self.one(&i("k_norm.bias"))?)?);
                put(&mut ops, "ape", self.replicate(self.one(&i("index_kpool_compress_ape"))?)?);
            }
        }
        let mlp = if dense { format!("{p}.mlp") } else { format!("{p}.mlp.shared_experts") };
        // A ModelOpt NVFP4 dense MLP runs whole on rank 0 (rank 1 adds a zero partial).
        if !(dense && self.nvfp4_dense(cfg, &mlp, &mut ops[0])?) {
            let (q, s): (Vec<_>, Vec<_>) = self.fp8_split(&[format!("{mlp}.gate_proj.weight"),
                format!("{mlp}.up_proj.weight")], Axis::Rows, ranks)?.into_iter().unzip();
            put(&mut ops, "w_gate_up_fp8", q);
            put(&mut ops, "w_gate_up_scale", s);
            let (q, s): (Vec<_>, Vec<_>) = self.fp8_split(&[format!("{mlp}.down_proj.weight")], Axis::Cols, ranks)?
                .into_iter().unzip();
            put(&mut ops, "w_down_fp8", q);
            put(&mut ops, "w_down_scale", s);
        }
        if !dense {
            ops[0].insert("gate", self.one(&format!("{p}.mlp.gate.weight"))?);
            ops[0].insert("gate.bias", self.f32(&[format!("{p}.mlp.gate.e_score_correction_bias")])?);
        }
        Ok(ops.into_iter().map(|operands| GlmfLayer { attention, dense, split: true, operands }).collect())
    }

    /// Layers `0..layers` (all of them unless the caller stops early): rank 0's weights (its
    /// layer shares, the norm and head) and with a head split the other rank's layer shares.
    #[allow(clippy::type_complexity)]
    pub fn model(&self, cfg: &GlmNextConfig, layers: usize) -> Result<(GlmfWeights<'a>, Vec<Vec<GlmfLayer<'a>>>)> {
        validate_kda_output_shard(self.ranks(), self.kda_fp8, self.kda_output_shard)?;
        let mut shares: Vec<Vec<GlmfLayer<'a>>> = (0..self.ranks()).map(|_| Vec::new()).collect();
        for layer in 0..layers.min(cfg.layers) {
            for (share, part) in shares.iter_mut().zip(self.layer_shares(cfg, layer)?) {
                share.push(part);
            }
        }
        let mut shares = shares.into_iter();
        let weights = GlmfWeights {
            layers: shares.next().context("rank 0")?,
            norm: self.one(&format!("{PREFIX}norm.weight"))?,
            head: if self.fp8_head {
                let (values, scales) = self.fp8(&["lm_head.weight".to_string()], super::fp8::Layout::Row128)?;
                super::head::GlmfHead::Fp8 { values, scales }
            } else {
                super::head::GlmfHead::Bf16(self.one("lm_head.weight")?)
            },
        };
        crate::shared::memory::staging::release_staging();
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
