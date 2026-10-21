//! GLM 5.x coordinator weights, packed for the exported glm_* programs.
//!
//! The checkpoint's FP8 projections stay FP8 (ModelOpt per-tensor FP8 as the
//! same bytes under a uniform block grid; a ModelOpt release's BF16 projections
//! are quantized to 128x128 blocks at load): E4M3 bytes (`{w}_fp8`) with
//! their FP32 128x128 scale grids (`{w}_scale`, `[ceil(N/128), K/128]`) are
//! the only copies of `w_qkv_a`, `w_q_b`, `w_iq`, `w_o`, `w_gate_up` and
//! `w_down` (decode programs: FP8 GEMV up to 16 rows, W8A16 GEMMs above, bitwise
//! the BF16 programs over the dequantized weights; prefill programs: W8A8 or
//! W8A16). Prefill reads `q_a|kv_a` (2624 rows, the last block partial) with
//! per-row scales K-block major (`w_qkv_a_kscale`, `[K/128, N]`). `kv_b` is
//! split per head on the host into E4M3 `w_uk_fp8 [N, 512, 192]` (key rows
//! transposed) and `w_uv_fp8 [N, 256, 512]` with one FP32 scale per weight
//! row and 64-wide K tile (`w_uk_scale [N, 512, 3]`, `w_uv_scale [N, 256, 8]`:
//! heads span 3.5 blocks of the grid). Row-concatenated operands (`w_qkv_a`,
//! `w_gate_up`) concatenate their grids (every part but the last is a whole
//! number of 128-row blocks). BF16 operands: norms, the router, the LM head and
//! `w_ik` (`wk` FP8 dequantized on the GPU with its scales, `weights_proj` BF16).
use crate::shared::memory::DeviceAllocation;
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::families::glm5::{GlmDsaConfig, GlmIndexer};
use cuteafd_loader::OfficialV41Catalog;
use std::collections::HashMap;
use std::ffi::c_void;
use std::os::unix::fs::FileExt;
use crate::shared::peer_split::{slice_2d, Axis};
use super::landing::{self, Land};

pub(crate) struct GlmLayer<'a> {
    pub dense: bool,
    pub full_indexer: bool,
    pub mlp_numerics: cuteafd_loader::plan::checkpoint::ModuleNumerics,
    /// Dense/shared FFNs cover this GPU's intermediate slice (`glm2` programs).
    pub split: bool,
    /// q_b, kv_b and o_proj cover this GPU's heads, producing a partial sum.
    pub attention_split: bool,
    /// Whole attention and its cache exist only on this rank when present.
    pub attention_owner: Option<u8>,
    operands: HashMap<&'static str, DeviceAllocation<'a>>,
}

/// The `{w}_fp8` / `{w}_scale` operand names of an FP8 weight, and its
/// per-row K-block-major scales `{w}_kscale` (prefill `w_qkv_a` only).
pub(crate) fn fp8_operand_names(name: &str) -> (&'static str, &'static str, &'static str) {
    match name {
        "w_qkv_a" => ("w_qkv_a_fp8", "w_qkv_a_scale", "w_qkv_a_kscale"),
        "w_q_b" => ("w_q_b_fp8", "w_q_b_scale", ""),
        "w_iq" => ("w_iq_fp8", "w_iq_scale", ""),
        "w_o" => ("w_o_fp8", "w_o_scale", ""),
        "w_gate_up" => ("w_gate_up_fp8", "w_gate_up_scale", ""),
        "w_down" => ("w_down_fp8", "w_down_scale", ""),
        "w_uk" => ("w_uk_fp8", "w_uk_scale", ""),
        "w_uv" => ("w_uv_fp8", "w_uv_scale", ""),
        other => unreachable!("{other} has no FP8 operand"),
    }
}

impl GlmLayer<'_> {
    /// The device range of `operand`, when the layer has it.
    pub fn range(&self, operand: &str) -> Option<crate::shared::l2_prefetch::Range> {
        self.operands.get(operand).map(|a| (a.buffer.ptr.cast_const(), a.buffer.bytes))
    }

    pub fn ptr(&self, operand: &str) -> Result<*mut c_void> {
        Ok(self.operands.get(operand).with_context(|| format!("layer has no weight {operand}"))?.buffer.ptr)
    }

    /// Device bytes of the layer's operands.
    pub fn bytes(&self) -> usize {
        self.operands.values().map(|a| a.buffer.bytes).sum()
    }
}

pub(crate) struct GlmWeights<'a> {
    pub layers: Vec<GlmLayer<'a>>,
    pub norm: DeviceAllocation<'a>,
    pub head: DeviceAllocation<'a>,
}

#[derive(Clone)]
pub(crate) struct GlmLoader<'a> {
    pub library: &'a NativeLibrary,
    pub catalog: &'a OfficialV41Catalog,
    pub stream: *mut c_void,
    /// This loader's device (rank 0 of a head split).
    pub device: i32,
    /// The other GPU of a head split (rank 1), if any.
    pub peers: Vec<crate::shared::peer_split::RankDevice>,
    /// Push mode: class-2 operands are allocated and recorded, filled later by
    /// the push group; transforms of their bytes wait for the commit.
    pub push: Option<std::rc::Rc<super::landing::PushSink>>,
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; values.len() * 4];
    for (dst, v) in out.chunks_exact_mut(4).zip(values) {
        dst.copy_from_slice(&v.to_le_bytes());
    }
    out
}

impl<'a> GlmLoader<'a> {
    /// Replicate only admitted local routers; their bytes are charged to the TP2 expert arena.
    pub(crate) fn peer_routers(&self, layers: &mut [GlmLayer<'a>], range: std::ops::Range<usize>) -> Result<()> {
        let peer = *self.peers.first().context("TP2 router replica needs peer")?;
        for index in range {
            for (operand, suffix) in [("gate", "weight"), ("gate.bias", "e_score_correction_bias")] {
                let allocation = crate::shared::peer_split::on_device(self.library, peer.device, self.device, || {
                    let _scope = cuteafd_ffi::memory_ledger::scope("experts/TP2 router replica");
                    self.one(&format!("model.layers.{index}.mlp.gate.{suffix}"))
                })?;
                layers[index].operands.insert(operand, allocation);
            }
        }
        Ok(())
    }

    fn raw(&self, name: &str) -> Result<(Vec<u8>, DType, Vec<usize>)> {
        cuteafd_ffi::memory_ledger::tensor(name);
        let tensor = self.catalog.tensor(name)?;
        let mut bytes = vec![0u8; tensor.metadata.byte_length as usize];
        crate::shared::memory::load_timing::read(bytes.len(), || std::fs::File::open(self.catalog.snapshot().join(&tensor.shard))?
            .read_exact_at(&mut bytes, tensor.metadata.byte_offset))
            .with_context(|| format!("reading {name}"))?;
        Ok((bytes, tensor.metadata.dtype.clone(), tensor.metadata.shape.clone()))
    }

    /// One operand on this loader's current device filled from checkpoint
    /// rectangles that land unchanged (class 2).
    fn landed(&self, lands: &[Land], bytes: usize) -> Result<DeviceAllocation<'a>> {
        if let Some(push) = &self.push {
            let allocation = DeviceAllocation::new(self.library, bytes.max(256))?;
            push.record(self.library.cuda_get_device()?, allocation.buffer, lands, bytes)?;
            return Ok(allocation);
        }
        self.upload(&landing::read(self.catalog, lands, bytes)?)
    }

    /// `lands` replicated on every rank (one read, rank 0 upload, peer copies;
    /// pushed, each rank receives its own copy).
    fn landed_all(&self, ranks: usize, lands: &[Land], bytes: usize) -> Result<Vec<DeviceAllocation<'a>>> {
        if self.push.is_some() {
            return (0..ranks).map(|rank| self.on_rank(rank, |_| self.landed(lands, bytes))).collect();
        }
        self.upload_all(ranks, &landing::read(self.catalog, lands, bytes)?)
    }

    /// Whether every part is E4M3 with its own FP32 block grid.
    fn native_fp8(&self, names: &[String]) -> Result<bool> {
        names.iter().try_fold(true, |all, name| Ok(all && landing::native_fp8_block(self.catalog, name)?))
    }

    fn upload(&self, bytes: &[u8]) -> Result<DeviceAllocation<'a>> {
        let allocation = DeviceAllocation::new(self.library, bytes.len().max(256))?;
        crate::shared::memory::load_timing::upload(bytes.len(), || self.library.copy_h2d(allocation.buffer, bytes))?;
        Ok(allocation)
    }

    /// The row-concatenation of 2-D `names` as one BF16 operand (FP8 parts
    /// dequantized on the GPU with their FP32 block scales).
    fn rows(&self, names: &[String]) -> Result<DeviceAllocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        let tensors = names.iter().map(|n| self.raw(n).map(|t| (n, t))).collect::<Result<Vec<_>>>()?;
        let cols = tensors[0].1 .2[1];
        let rows: usize = tensors.iter().map(|(_, (_, _, shape))| shape[0]).sum();
        ensure!(tensors.iter().all(|(_, (_, _, s))| s.len() == 2 && s[1] == cols), "{names:?} do not share columns");
        let out = DeviceAllocation::new(self.library, rows * cols * 2)?;
        let k_blocks = cols.div_ceil(128);
        let mut row = 0;
        for (name, (bytes, dtype, shape)) in &tensors {
            let dest = CuteafdDeviceBuffer {
                // SAFETY: rows row..row+shape[0] lie inside `out`.
                ptr: unsafe { out.buffer.ptr.cast::<u8>().add(row * cols * 2) }.cast(),
                bytes: shape[0] * cols * 2,
                ..out.buffer
            };
            match dtype {
                DType::Bf16 => crate::shared::memory::load_timing::upload(bytes.len(), || self.library.copy_h2d(dest, bytes))?,
                DType::F8E4M3 => {
                    let (scale, scale_dtype, _) = self.raw(&format!("{}_scale_inv", name))?;
                    ensure!(scale_dtype == DType::F32, "{name}: block scales must be FP32");
                    ensure!(scale.len() == shape[0].div_ceil(128) * k_blocks * 4, "{name}: unexpected scale grid");
                    let (w, s) = (self.upload(bytes)?, self.upload(&scale)?);
                    // SAFETY: fp8 [shape], its scale grid and the destination rows
                    // are live; the stream is drained before the temporaries drop.
                    unsafe {
                        self.library.fp8_block_dequant(w.buffer.ptr, s.buffer.ptr, dest.ptr, shape[0], cols, self.stream)?;
                        self.library.cuda_stream_synchronize(self.stream)?;
                    }
                }
                other => anyhow::bail!("{name}: unsupported coordinator dtype {other:?}"),
            }
            row += shape[0];
        }
        Ok(out)
    }

    /// Reads tensor `name`'s bytes into `out[at..]` (one positioned read into the
    /// operand's buffer: no second copy) and returns its byte length, dtype and shape.
    fn read_into(&self, name: &str, out: &mut [u8], at: usize) -> Result<(usize, DType, Vec<usize>)> {
        cuteafd_ffi::memory_ledger::tensor(name);
        let tensor = self.catalog.tensor(name)?;
        let length = tensor.metadata.byte_length as usize;
        ensure!(at + length <= out.len(), "{name}: {length} bytes past the operand buffer");
        crate::shared::memory::load_timing::read(length, || std::fs::File::open(self.catalog.snapshot().join(&tensor.shard))?
            .read_exact_at(&mut out[at..at + length], tensor.metadata.byte_offset))
            .with_context(|| format!("reading {name}"))?;
        Ok((length, tensor.metadata.dtype.clone(), tensor.metadata.shape.clone()))
    }

    /// Reads the row-concatenation of the FP8 checkpoint weights `names` into
    /// this thread's staging buffer and hands `body` the E4M3 bytes `[N, K]`,
    /// the FP32 block grids `[ceil(N/128), K/128]` (every part but the last a
    /// whole number of 128-row blocks), N and K.
    fn with_fp8<T>(&self, names: &[String], body: impl FnOnce(&[u8], Vec<f32>, usize, usize) -> Result<T>)
        -> Result<T> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("fp8");
        // BF16 parts need room for their padded rows while they quantize in place.
        let total: u64 = names.iter().map(|n| self.catalog.tensor(n).map(|t| t.metadata.byte_length))
            .sum::<Result<u64>>()? + 128 * 8192 * 2;
        crate::shared::memory::staging::with_staging(total as usize, |values| {
            let mut grid = Vec::new();
            let (mut rows, mut cols, mut at) = (0usize, None, 0usize);
            for (i, name) in names.iter().enumerate() {
                let (length, dtype, shape) = self.read_into(name, values, at)?;
                ensure!(shape.len() == 2 && matches!(dtype, DType::F8E4M3 | DType::Bf16),
                    "{name}: the glm programs take this weight as FP8 (or BF16 quantized at load), found {dtype:?} {shape:?}");
                ensure!(cols.is_none_or(|c| c == shape[1]), "{names:?} do not share columns");
                ensure!(i + 1 == names.len() || shape[0] % 128 == 0,
                    "{names:?}: only the last concatenated weight may end inside a 128-row block");
                cols = Some(shape[1]);
                let (n, k) = (shape[0], shape[1]);
                let block_rows = n.div_ceil(128);
                if dtype == DType::Bf16 {
                    // An unquantized weight of a ModelOpt release (nvidia/GLM-5.3-NVFP4 keeps
                    // attention and the shared experts BF16): 128x128 E4M3 blocks at load,
                    // power-of-two scales (the glmf rule), rows padded to whole blocks.
                    let mut padded = values[at..at + length].to_vec();
                    padded.resize(block_rows * 128 * k * 2, 0);
                    let (q, scales) = crate::families::glm5_flash::fp8::quantize(&padded, block_rows * 128, k,
                        crate::families::glm5_flash::fp8::Layout::Block, crate::shared::fp8_linear::Fp8Scales::Pow2);
                    values[at..at + n * k].copy_from_slice(&q[..n * k]);
                    grid.extend(scales);
                } else if self.catalog.tensor(&format!("{name}_scale_inv")).is_ok() {
                    let (scale, scale_dtype, scale_shape) = self.raw(&format!("{name}_scale_inv"))?;
                    ensure!(scale_dtype == DType::F32 && scale_shape == [block_rows, k.div_ceil(128)],
                        "{name}: expected FP32 128x128 block scales, found {scale_dtype:?} {scale_shape:?}");
                    grid.extend(scale.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())));
                } else {
                    // ModelOpt per-tensor FP8 (one FP32 `weight_scale`): the same E4M3 bytes
                    // with every block scale equal to it, exactly (decode GEMVs and W8A16).
                    // Prefill W8A8 takes its static input_scale (`tensor_scales`).
                    let (scale, scale_dtype, _) = self.raw(&format!("{name}_scale"))
                        .with_context(|| format!("{name}: an FP8 weight needs weight_scale_inv blocks or a weight_scale"))?;
                    ensure!(scale_dtype == DType::F32 && scale.len() == 4, "{name}_scale: expected one FP32 value");
                    let value = f32::from_le_bytes(scale[..4].try_into().unwrap());
                    grid.extend(std::iter::repeat_n(value, block_rows * k.div_ceil(128)));
                }
                at += n * k;
                rows += n;
            }
            // The E4M3 bytes only (BF16 parts left their staging tail unused).
            body(&values[..at], grid, rows, cols.context("no FP8 rows")?)
        })
    }

    /// The prefill programs' FP32 `[12]` tensor-scale operand of row-concatenated ModelOpt
    /// per-tensor FP8 weights `parts` (E4M3, one FP32 `weight_scale`, one calibrated
    /// `input_scale` shared by the parts): `input_scale` at 0 and `input_scale * weight_scale`
    /// of part `i` at `4 * (i + 1)`. None unless every part is stored that way.
    fn tensor_scales(&self, parts: &[String]) -> Result<Option<Vec<u8>>> {
        ensure!(parts.len() <= 2, "{parts:?}: the tensor-scale operand holds two parts");
        let scalar = |name: &str| -> Result<Option<f32>> {
            if self.catalog.tensor(name).is_err() {
                return Ok(None);
            }
            let (bytes, dtype, _) = self.raw(name)?;
            ensure!(dtype == DType::F32 && bytes.len() == 4, "{name}: expected one FP32 value");
            Ok(Some(f32::from_le_bytes(bytes[..4].try_into().unwrap())))
        };
        let mut out = [0f32; 12];
        let mut input = None;
        for (i, name) in parts.iter().enumerate() {
            let Some(stem) = name.strip_suffix("weight") else { return Ok(None) };
            if self.catalog.tensor(name)?.metadata.dtype != DType::F8E4M3 {
                return Ok(None);
            }
            let (Some(weight_scale), Some(input_scale)) =
                (scalar(&format!("{name}_scale"))?, scalar(&format!("{stem}input_scale"))?) else { return Ok(None) };
            ensure!(input.is_none_or(|s| s == input_scale),
                "{parts:?}: concatenated per-tensor FP8 parts need one input_scale");
            input = Some(input_scale);
            out[4 * (i + 1)] = input_scale * weight_scale;
        }
        out[0] = input.context("no parts")?;
        Ok(Some(f32_bytes(&out)))
    }

    /// E4M3 `[N, K]` and the FP32 block grid of `names` on the device.
    fn fp8(&self, names: &[String]) -> Result<(DeviceAllocation<'a>, DeviceAllocation<'a>)> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("fp8");
        if let Some(((values, values_bytes), (grid, grid_bytes))) = self.fp8_lands(names)? {
            return Ok((self.landed(&values, values_bytes)?, self.landed(&grid, grid_bytes)?));
        }
        self.with_fp8(names, |values, grid, _, _| Ok((self.upload(values)?, self.upload(&f32_bytes(&grid))?)))
    }

    /// Native block FP8 `names` row-concatenated: the E4M3 values and the
    /// FP32 grids land unchanged. None for any other storage (quantized at load).
    #[allow(clippy::type_complexity)]
    fn fp8_lands(&self, names: &[String]) -> Result<Option<((Vec<Land>, usize), (Vec<Land>, usize))>> {
        if !self.native_fp8(names)? { return Ok(None); }
        let cols = names.iter().map(|n| Ok(self.catalog.tensor(n)?.metadata.shape[1])).collect::<Result<Vec<_>>>()?;
        ensure!(cols.windows(2).all(|w| w[0] == w[1]), "{names:?} do not share columns");
        for (i, name) in names.iter().enumerate() {
            ensure!(i + 1 == names.len() || self.catalog.tensor(name)?.metadata.shape[0] % 128 == 0,
                "{names:?}: only the last concatenated weight may end inside a 128-row block");
        }
        let grids = names.iter().map(|n| format!("{n}_scale_inv")).collect::<Vec<_>>();
        Ok(Some((landing::concat(self.catalog, names)?, landing::concat(self.catalog, &grids)?)))
    }

    /// `w_uk [N, 512, D]` / `w_uv [N, V, 512]` E4M3 slices of the FP8 `kv_b`
    /// with one scale per weight row and 64-wide K tile (`[N, 512, D/64]`,
    /// `[N, V, 512/64]`): `w_uk[h,c,d] = kv_b[h*(D+V)+d, c]`,
    /// `w_uv[h,v,c] = kv_b[h*(D+V)+D+v, c]`.
    /// Over `ranks` GPUs each takes its heads' slices (head-major, so contiguous).
    #[allow(clippy::type_complexity)]
    fn kv_b(&self, cfg: &GlmDsaConfig, prefix: &str, ranks: usize)
        -> Result<Vec<((DeviceAllocation<'a>, DeviceAllocation<'a>), (DeviceAllocation<'a>, DeviceAllocation<'a>))>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("fp8");
        let (heads, d, v, c) = (cfg.heads, cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank);
        self.with_fp8(&[format!("{prefix}.self_attn.kv_b_proj.weight")], |kv_b, grid, rows, cols| {
        ensure!(rows == heads * (d + v) && cols == c && d % 64 == 0 && c % 128 == 0,
            "{prefix}: kv_b_proj [{rows}, {cols}] does not match {heads} heads x ({d} + {v}) x {c}");
        let kb = c / 128;
        let scale = |row: usize, col: usize| grid[(row / 128) * kb + col / 128];
        // Each 64-row K tile of a key slice lies in one 128-row block (448 = 3.5 x 128).
        ensure!((d + v) % 64 == 0, "{prefix}: head rows must be whole 64-row tiles");
        let (mut uk, mut uk_s) = (vec![0u8; heads * c * d], vec![0f32; heads * c * d / 64]);
        let (mut uv, mut uv_s) = (vec![0u8; heads * v * c], vec![0f32; heads * v * c / 64]);
        // One head per task: key rows transposed in 64 x 64 tiles, value rows copied.
        std::thread::scope(|scope| {
            let parts = uk.chunks_mut(c * d).zip(uk_s.chunks_mut(c * d / 64)).zip(uv.chunks_mut(v * c))
                .zip(uv_s.chunks_mut(v * c / 64)).enumerate();
            let threads = std::thread::available_parallelism().map_or(8, |p| p.get()).clamp(1, 16);
            let mut groups: Vec<Vec<_>> = (0..threads).map(|_| Vec::new()).collect();
            for (h, part) in parts {
                groups[h % threads].push((h, part));
            }
            for group in groups {
                let (kv_b, scale) = (&kv_b, &scale);
                scope.spawn(move || {
                    for (h, (((uk, uk_s), uv), uv_s)) in group {
                        let base = h * (d + v);
                        for c0 in (0..c).step_by(64) {
                            for d0 in (0..d).step_by(64) {
                                for dd in d0..d0 + 64 {
                                    let src = &kv_b[(base + dd) * c..][..c];
                                    for col in c0..c0 + 64 {
                                        uk[col * d + dd] = src[col];
                                    }
                                }
                            }
                        }
                        for col in 0..c {
                            for t in 0..d / 64 {
                                uk_s[col * (d / 64) + t] = scale(base + t * 64, col);
                            }
                        }
                        uv.copy_from_slice(&kv_b[(base + d) * c..(base + d + v) * c]);
                        for r in 0..v {
                            for t in 0..c / 64 {
                                uv_s[r * (c / 64) + t] = scale(base + d + r, t * 64);
                            }
                        }
                    }
                });
            }
        });
        ensure!(heads % ranks == 0, "{prefix}: {heads} heads do not split over {ranks} GPUs");
        let part = |bytes: &[u8], rank: usize| -> Vec<u8> {
            bytes[rank * bytes.len() / ranks..(rank + 1) * bytes.len() / ranks].to_vec()
        };
        let (uk_s, uv_s) = (f32_bytes(&uk_s), f32_bytes(&uv_s));
        (0..ranks).map(|rank| self.on_rank(rank, |_| Ok((
            (self.upload(&part(&uk, rank))?, self.upload(&part(&uk_s, rank))?),
            (self.upload(&part(&uv, rank))?, self.upload(&part(&uv_s, rank))?))))).collect()
        })
    }

    /// Whether `names` run as the checkpoint's BF16 tensors (the BF16 programs): every one is
    /// BF16 and CUTEAFD_GLM_BF16=native asks for them ([`bf16_programs`]).
    fn bf16_native(&self, names: &[String]) -> Result<bool> {
        if !bf16_programs() {
            return Ok(false);
        }
        for name in names {
            if self.catalog.tensor(name)?.metadata.dtype != DType::Bf16 {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The row-concatenation of BF16 `names`: its bytes, rows and columns.
    fn bf16_rows(&self, names: &[String]) -> Result<(Vec<u8>, usize, usize)> {
        let mut out = Vec::new();
        let (mut rows, mut cols) = (0usize, None);
        for name in names {
            let (bytes, dtype, shape) = self.raw(name)?;
            ensure!(dtype == DType::Bf16 && shape.len() == 2, "{name}: expected a BF16 matrix, found {dtype:?} {shape:?}");
            ensure!(cols.is_none_or(|c| c == shape[1]), "{names:?} do not share columns");
            cols = Some(shape[1]);
            rows += shape[0];
            out.extend_from_slice(&bytes);
        }
        Ok((out, rows, cols.context("no BF16 parts")?))
    }

    /// The BF16 weights `names` concatenated by rows, sliced over `ranks` along `axis`:
    /// rank `r` takes part `r` of every concatenated weight's rows (so a head- or
    /// intermediate-split `gate | up` stays gate rows then up rows), or of the columns.
    fn bf16_split(&self, names: &[String], axis: Axis, ranks: usize) -> Result<Vec<DeviceAllocation<'a>>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        if ranks == 1 {
            return Ok(vec![self.upload(&self.bf16_rows(names)?.0)?]);
        }
        let parts = names.iter().map(|n| self.bf16_rows(std::slice::from_ref(n))).collect::<Result<Vec<_>>>()?;
        ensure!(axis == Axis::Rows || parts.len() == 1, "{names:?}: concatenated weights split by rows");
        (0..ranks).map(|rank| {
            let mut bytes = Vec::new();
            for (values, rows, cols) in &parts {
                ensure!(if axis == Axis::Rows { rows % ranks == 0 } else { cols % ranks == 0 },
                    "{names:?}: [{rows}, {cols}] does not split over {ranks} GPUs");
                bytes.extend(slice_2d(values, *rows, *cols, 2, axis, rank, ranks));
            }
            self.on_rank(rank, |_| self.upload(&bytes))
        }).collect()
    }

    /// `w_uk [N, 512, D]` / `w_uv [N, V, 512]` BF16 per-head slices of the BF16 `kv_b`
    /// (`w_uk[h,c,d] = kv_b[h*(D+V)+d, c]`, `w_uv[h,v,c] = kv_b[h*(D+V)+D+v, c]`), each rank
    /// its heads.
    fn kv_b_bf16(&self, cfg: &GlmDsaConfig, prefix: &str, ranks: usize)
        -> Result<Vec<(DeviceAllocation<'a>, DeviceAllocation<'a>)>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        let (heads, d, v, c) = (cfg.heads, cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank);
        let (kv_b, rows, cols) = self.bf16_rows(&[format!("{prefix}.self_attn.kv_b_proj.weight")])?;
        ensure!(rows == heads * (d + v) && cols == c && heads % ranks == 0,
            "{prefix}: kv_b_proj [{rows}, {cols}] does not match {heads} heads x ({d} + {v}) x {c} over {ranks} GPUs");
        let (mut uk, mut uv) = (vec![0u8; heads * c * d * 2], vec![0u8; heads * v * c * 2]);
        std::thread::scope(|scope| {
            for (h, (uk, uv)) in uk.chunks_mut(c * d * 2).zip(uv.chunks_mut(v * c * 2)).enumerate() {
                let kv_b = &kv_b;
                scope.spawn(move || {
                    let base = h * (d + v);
                    for dd in 0..d {
                        let src = &kv_b[(base + dd) * c * 2..][..c * 2];
                        for col in 0..c {
                            uk[(col * d + dd) * 2..][..2].copy_from_slice(&src[col * 2..col * 2 + 2]);
                        }
                    }
                    uv.copy_from_slice(&kv_b[(base + d) * c * 2..(base + d + v) * c * 2]);
                });
            }
        });
        let part = |bytes: &[u8], rank: usize| bytes[rank * bytes.len() / ranks..(rank + 1) * bytes.len() / ranks].to_vec();
        (0..ranks).map(|rank| self.on_rank(rank, |_| Ok((self.upload(&part(&uk, rank))?, self.upload(&part(&uv, rank))?))))
            .collect()
    }

    fn one(&self, name: &str) -> Result<DeviceAllocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        let meta = &self.catalog.tensor(name)?.metadata;
        if meta.shape.len() == 2 && meta.dtype == DType::F8E4M3 {
            return self.rows(&[name.to_string()]);
        }
        let land = Land::whole(self.catalog, name, 0)?;
        let bytes = land.bytes();
        self.landed(&[land], bytes)
    }

    /// GPUs of the head split this loader fills (1: no split).
    pub fn ranks(&self) -> usize {
        1 + self.peers.len()
    }

    /// Runs `body` with rank `rank`'s device current and its load stream.
    fn on_rank<T>(&self, rank: usize, body: impl FnOnce(*mut c_void) -> Result<T>) -> Result<T> {
        if rank == 0 {
            return body(self.stream);
        }
        let peer = self.peers.get(rank - 1).with_context(|| format!("no rank {rank}"))?;
        crate::shared::peer_split::on_device(self.library, peer.device, self.device, || body(peer.stream))
    }

    /// The FP8 weights `names` concatenated by rows, each sliced over `ranks`
    /// along `axis` in whole 128-row / 128-K blocks: per rank its E4M3 bytes and
    /// the matching part of the FP32 128x128 grid. Rows-sliced weights read once.
    fn fp8_split(&self, names: &[String], axis: Axis, ranks: usize)
        -> Result<Vec<(DeviceAllocation<'a>, DeviceAllocation<'a>)>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("fp8");
        if ranks == 1 {
            return Ok(vec![self.fp8(names)?]);
        }
        if let ([name], Axis::Cols) = (names, axis) {
            return self.fp8_cols(name, ranks);
        }
        ensure!(axis == Axis::Rows, "{names:?}: concatenated weights split by rows");
        if self.native_fp8(names)? {
            let grids = names.iter().map(|n| format!("{n}_scale_inv")).collect::<Vec<_>>();
            return (0..ranks).map(|rank| {
                let (values, values_bytes) = landing::row_shares(self.catalog, names, ranks, rank, 128)?;
                let (grid, grid_bytes) = landing::row_shares(self.catalog, &grids, ranks, rank, 1)?;
                self.on_rank(rank, |_| Ok((self.landed(&values, values_bytes)?, self.landed(&grid, grid_bytes)?)))
            }).collect();
        }
        // Per rank: the E4M3 rows and grid rows of every part, uploaded straight from the
        // staging buffer into the part's offset (whole 128-row blocks, so grids slice too).
        let mut out: Vec<Option<(DeviceAllocation<'a>, DeviceAllocation<'a>)>> = (0..ranks).map(|_| None).collect();
        let (mut row_at, mut grid_at) = (0usize, 0usize);
        let (total_rows, cols) = names.iter().try_fold((0usize, 0usize), |(rows, _), name| -> Result<_> {
            let shape = &self.catalog.tensor(name)?.metadata.shape;
            Ok((rows + shape[0], shape[1]))
        })?;
        for name in names {
            self.with_fp8(std::slice::from_ref(name), |values, grid, rows, part_cols| {
                ensure!(part_cols == cols && rows % (128 * ranks) == 0 && cols % 128 == 0,
                    "{name}: [{rows}, {cols}] does not split into whole 128-row blocks over {ranks} GPUs");
                let (share, kb) = (rows / ranks, cols / 128);
                for (rank, slot) in out.iter_mut().enumerate() {
                    self.on_rank(rank, |_| {
                        if slot.is_none() {
                            *slot = Some((DeviceAllocation::new(self.library, total_rows / ranks * cols)?,
                                DeviceAllocation::new(self.library, total_rows / ranks / 128 * kb * 4)?));
                        }
                        let (dv, dg) = slot.as_ref().context("rank buffers")?;
                        let at = |buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize| CuteafdDeviceBuffer {
                            // SAFETY: offset + bytes lie inside the rank's buffer (sized above).
                            ptr: unsafe { buffer.ptr.cast::<u8>().add(offset) }.cast(), bytes, ..buffer };
                        crate::shared::memory::load_timing::upload(share * cols, || self.library.copy_h2d(at(dv.buffer, row_at / ranks * cols, share * cols),
                            &values[rank * share * cols..(rank + 1) * share * cols]))?;
                        let grid_part = f32_bytes(&grid[rank * share / 128 * kb..(rank + 1) * share / 128 * kb]);
                        crate::shared::memory::load_timing::upload(grid_part.len(), || self.library.copy_h2d(at(dg.buffer, grid_at / ranks * 4, grid_part.len()), &grid_part))
                    })?;
                }
                row_at += rows;
                grid_at += rows / 128 * kb;
                Ok(())
            })?;
        }
        out.into_iter().map(|slot| slot.context("no FP8 parts")).collect()
    }

    /// One FP8 weight sliced by columns over `ranks`: the whole E4M3 weight goes up
    /// once to rank 0 and each rank's columns are a pitched device copy from it
    /// (over peer memory for rank 1); the grid's column blocks are sliced on the host.
    fn fp8_cols(&self, name: &str, ranks: usize) -> Result<Vec<(DeviceAllocation<'a>, DeviceAllocation<'a>)>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("fp8");
        if self.native_fp8(&[name.to_string()])? {
            let shape = &self.catalog.tensor(name)?.metadata.shape;
            ensure!(shape[0] % 128 == 0 && shape[1] % (128 * ranks) == 0,
                "{name}: {shape:?} does not split into whole 128-K blocks over {ranks} GPUs");
            return (0..ranks).map(|rank| {
                let (values, values_bytes) = landing::col_share(self.catalog, name, ranks, rank)?;
                let (grid, grid_bytes) = landing::col_share(self.catalog, &format!("{name}_scale_inv"), ranks, rank)?;
                self.on_rank(rank, |_| Ok((self.landed(&[values], values_bytes)?, self.landed(&[grid], grid_bytes)?)))
            }).collect();
        }
        let (whole, grid, rows, cols) = self.with_fp8(&[name.to_string()], |values, grid, rows, cols| {
            ensure!(rows % 128 == 0 && cols % (128 * ranks) == 0,
                "{name}: [{rows}, {cols}] does not split into whole 128-K blocks over {ranks} GPUs");
            Ok((self.upload(values)?, grid, rows, cols))
        })?;
        let width = cols / ranks;
        let grid_bytes = f32_bytes(&grid);
        let out = (0..ranks).map(|rank| {
            let part = slice_2d(&grid_bytes, rows / 128, cols / 128, 4, Axis::Cols, rank, ranks);
            self.on_rank(rank, |stream| {
                let values = DeviceAllocation::new(self.library, rows * width)?;
                let source = CuteafdDeviceBuffer {
                    // SAFETY: column block `rank` of every row lies inside the whole weight.
                    ptr: unsafe { whole.buffer.ptr.cast::<u8>().add(rank * width) }.cast(),
                    bytes: whole.buffer.bytes - rank * width,
                    ..whole.buffer
                };
                // SAFETY: both buffers are live and sized for these pitched spans; peer access
                // to rank 0 is enabled on every rank (the split setup); drained below, before
                // `whole` drops.
                unsafe {
                    self.library.copy_d2d_2d_async(values.buffer, width, source, cols, width, rows, stream)?;
                    self.library.cuda_stream_synchronize(stream)?;
                }
                Ok((values, self.upload(&part)?))
            })
        }).collect();
        drop(whole);
        out
    }

    /// Every rank's copy of a small operand read once.
    /// A whole non-FP8 tensor (a norm) on every rank, landing unchanged.
    fn replicated_tensor(&self, ranks: usize, name: &str) -> Result<Vec<DeviceAllocation<'a>>> {
        let meta = &self.catalog.tensor(name)?.metadata;
        ensure!(!(meta.shape.len() == 2 && meta.dtype == DType::F8E4M3), "{name}: replicated operands are not FP8");
        let land = Land::whole(self.catalog, name, 0)?;
        let bytes = land.bytes();
        self.landed_all(ranks, &[land], bytes)
    }

    fn replicated(&self, ranks: usize, read: impl FnOnce() -> Result<Vec<u8>>) -> Result<Vec<DeviceAllocation<'a>>> {
        self.upload_all(ranks, &read()?)
    }

    /// `bytes` on every rank: one upload to rank 0, peer copies to the others (faster than
    /// a second pageable host upload).
    fn upload_all(&self, ranks: usize, bytes: &[u8]) -> Result<Vec<DeviceAllocation<'a>>> {
        let first = self.upload(bytes)?;
        let mut out = Vec::with_capacity(ranks);
        for rank in 1..ranks {
            out.push(self.on_rank(rank, |stream| {
                let copy = DeviceAllocation::new(self.library, bytes.len().max(256))?;
                // SAFETY: both allocations hold `bytes.len()` bytes; peer access to rank 0 is
                // enabled (the split setup); the stream drains before return.
                unsafe {
                    self.library.copy_d2d_async(copy.buffer, first.buffer, bytes.len(), stream)?;
                    self.library.cuda_stream_synchronize(stream)?;
                }
                Ok(copy)
            })?);
        }
        out.insert(0, first);
        Ok(out)
    }

    fn attention_operands(&self, cfg: &GlmDsaConfig, layer: usize)
        -> Result<Vec<HashMap<&'static str, DeviceAllocation<'a>>>> {
        let ranks = self.ranks();
        let p = format!("model.layers.{layer}");
        let full_indexer = cfg.indexers[layer] == GlmIndexer::Full;
        let mut ops: Vec<HashMap<&'static str, DeviceAllocation<'a>>> = (0..ranks).map(|_| HashMap::new()).collect();
        let put = |ops: &mut Vec<HashMap<&'static str, DeviceAllocation<'a>>>, key: &'static str,
            parts: Vec<DeviceAllocation<'a>>| {
            for (map, part) in ops.iter_mut().zip(parts) {
                map.insert(key, part);
            }
        };
        // The checkpoint's E4M3 bytes and FP32 scale grid: the only copy (per rank's slice).
        let with_fp8 = |ops: &mut Vec<HashMap<&'static str, DeviceAllocation<'a>>>, name: &'static str,
            parts: &[String], axis: Axis| -> Result<()> {
            let (w8_name, scale_name, _) = fp8_operand_names(name);
            let split = if ranks > 1 { self.fp8_split(parts, axis, ranks)? } else { vec![self.fp8(parts)?] };
            for (map, (w8, scale)) in ops.iter_mut().zip(split) {
                map.insert(w8_name, w8);
                map.insert(scale_name, scale);
            }
            Ok(())
        };
        let raw = |name: &str| -> Result<Vec<u8>> {
            let (bytes, dtype, shape) = self.raw(name)?;
            ensure!(!(shape.len() == 2 && dtype == DType::F8E4M3), "{name}: replicated operands are not FP8");
            Ok(bytes)
        };
        let qkv_a = [format!("{p}.self_attn.q_a_proj.weight"), format!("{p}.self_attn.kv_a_proj_with_mqa.weight")];
        let attention = [&qkv_a[..], &[format!("{p}.self_attn.q_b_proj.weight"),
            format!("{p}.self_attn.kv_b_proj.weight"), format!("{p}.self_attn.o_proj.weight")]].concat();
        if self.bf16_native(&attention)? {
            // The checkpoint's BF16 attention as-is (the BF16 programs).
            put(&mut ops, "w_qkv_a", self.upload_all(ranks, &self.bf16_rows(&qkv_a)?.0)?);
            put(&mut ops, "q_a_norm", self.replicated_tensor(ranks, &format!("{p}.self_attn.q_a_layernorm.weight"))?);
            put(&mut ops, "kv_a_norm", self.replicated_tensor(ranks, &format!("{p}.self_attn.kv_a_layernorm.weight"))?);
            put(&mut ops, "w_q_b", self.bf16_split(&[format!("{p}.self_attn.q_b_proj.weight")], Axis::Rows, ranks)?);
            for (rank, (uk, uv)) in self.kv_b_bf16(cfg, &p, ranks)?.into_iter().enumerate() {
                ops[rank].insert("w_uk", uk);
                ops[rank].insert("w_uv", uv);
            }
            put(&mut ops, "w_o", self.bf16_split(&[format!("{p}.self_attn.o_proj.weight")], Axis::Cols, ranks)?);
        } else {
        let (values, scale, kscale) = if let Some(((values, values_bytes), (grid, grid_bytes))) = self.fp8_lands(&qkv_a)? {
            // Prefill: per-row scales K-block major, widened on each GPU from its landed grid.
            let rows: usize = values.iter().map(|land| land.rows.len()).sum();
            let kb = values[0].width().div_ceil(128);
            let (values, grid) = (self.landed_all(ranks, &values, values_bytes)?, self.landed_all(ranks, &grid, grid_bytes)?);
            let kscale = grid.iter().enumerate().map(|(rank, grid)| self.on_rank(rank, |stream| {
                let kscale = DeviceAllocation::new(self.library, (rows * kb * 4).max(256))?;
                let job = landing::Kmajor { device: self.library.cuda_get_device()?, grid: grid.buffer,
                    kmajor: kscale.buffer, rows, kb };
                match &self.push {
                    // The grid is not delivered yet: widen after the group commits.
                    Some(push) => push.defer(job)?,
                    None => job.run(self.library, stream)?,
                }
                Ok(kscale)
            })).collect::<Result<Vec<_>>>()?;
            (values, grid, kscale)
        } else { self.with_fp8(&qkv_a, |values, grid, rows, cols| {
            // Prefill: per-row scales K-block major (the block-FP8 GEMM wants whole 128-row blocks).
            let kb = cols.div_ceil(128);
            let grid_ref = &grid;
            let kscale: Vec<f32> = (0..kb).flat_map(|b| (0..rows).map(move |r| grid_ref[(r / 128) * kb + b])).collect();
            let each = |bytes: &[u8]| self.upload_all(ranks, bytes);
            Ok((each(values)?, each(&f32_bytes(&grid))?, each(&f32_bytes(&kscale))?))
        })? };
        put(&mut ops, "w_qkv_a_fp8", values);
        put(&mut ops, "w_qkv_a_scale", scale);
        put(&mut ops, "w_qkv_a_kscale", kscale);
        put(&mut ops, "q_a_norm", self.replicated_tensor(ranks, &format!("{p}.self_attn.q_a_layernorm.weight"))?);
        put(&mut ops, "kv_a_norm", self.replicated_tensor(ranks, &format!("{p}.self_attn.kv_a_layernorm.weight"))?);
        // q_b rows are head-major (each head's 192 + 64 query rows): whole heads per rank.
        with_fp8(&mut ops, "w_q_b", &[format!("{p}.self_attn.q_b_proj.weight")], Axis::Rows)?;
        for (rank, ((uk, uk_scale), (uv, uv_scale))) in self.kv_b(cfg, &p, ranks)?.into_iter().enumerate() {
            ops[rank].insert("w_uk_fp8", uk);
            ops[rank].insert("w_uk_scale", uk_scale);
            ops[rank].insert("w_uv_fp8", uv);
            ops[rank].insert("w_uv_scale", uv_scale);
        }
        with_fp8(&mut ops, "w_o", &[format!("{p}.self_attn.o_proj.weight")], Axis::Cols)?;
        }
        if full_indexer {
            let wq_b = [format!("{p}.self_attn.indexer.wq_b.weight")];
            if self.bf16_native(&wq_b)? {
                put(&mut ops, "w_iq", self.upload_all(ranks, &self.bf16_rows(&wq_b)?.0)?);
            } else {
                let (w8_name, scale_name, _) = fp8_operand_names("w_iq");
                if let Some(((values, values_bytes), (grid, grid_bytes))) = self.fp8_lands(&wq_b)? {
                    put(&mut ops, w8_name, self.landed_all(ranks, &values, values_bytes)?);
                    put(&mut ops, scale_name, self.landed_all(ranks, &grid, grid_bytes)?);
                } else {
                    let (values, grid) = self.with_fp8(&wq_b, |values, grid, _, _| Ok((values.to_vec(), f32_bytes(&grid))))?;
                    put(&mut ops, w8_name, self.replicated(ranks, || Ok(values))?);
                    put(&mut ops, scale_name, self.replicated(ranks, || Ok(grid))?);
                }
            }
            // `w_ik`: BF16 (FP8 `wk` dequantized) on rank 0, copied to the others.
            let w_ik = self.rows(&[format!("{p}.self_attn.indexer.wk.weight"),
                format!("{p}.self_attn.indexer.weights_proj.weight")])?;
            let mut copies = vec![];
            for rank in 1..ranks {
                copies.push(self.on_rank(rank, |stream| {
                    let copy = DeviceAllocation::new(self.library, w_ik.buffer.bytes)?;
                    // SAFETY: both allocations hold the whole operand; peer access to rank 0 is
                    // enabled (the split setup); the stream drains before return.
                    unsafe {
                        self.library.copy_d2d_async(copy.buffer, w_ik.buffer, w_ik.buffer.bytes, stream)?;
                        self.library.cuda_stream_synchronize(stream)?;
                    }
                    Ok(copy)
                })?);
            }
            put(&mut ops, "w_ik", std::iter::once(w_ik).chain(copies).collect());
            put(&mut ops, "k_norm_w", self.replicated_tensor(ranks, &format!("{p}.self_attn.indexer.k_norm.weight"))?);
            put(&mut ops, "k_norm_b", self.replicated_tensor(ranks, &format!("{p}.self_attn.indexer.k_norm.bias"))?);
        }
        Ok(ops)
    }

    /// FFNs are rank-split; attention is either head-split with replicated
    /// latent/indexer operands or loaded whole on `owner`. Residual norms stay
    /// replicated and routers live on rank 0 until local TP2 admission.
    fn layer(&self, cfg: &GlmDsaConfig, layer: usize,
        checkpoint: &cuteafd_loader::plan::checkpoint::Checkpoint, owner: Option<u8>,
        native_shared: bool) -> Result<Vec<GlmLayer<'a>>> {
        let ranks = self.ranks();
        let p = format!("model.layers.{layer}");
        let dense = layer < cfg.first_moe_layer;
        let full_indexer = cfg.indexers[layer] == GlmIndexer::Full;
        let mut ops: Vec<HashMap<&'static str, DeviceAllocation<'a>>> = (0..ranks).map(|_| HashMap::new()).collect();
        let put = |ops: &mut Vec<HashMap<&'static str, DeviceAllocation<'a>>>, key: &'static str,
            parts: Vec<DeviceAllocation<'a>>| {
            for (map, part) in ops.iter_mut().zip(parts) {
                map.insert(key, part);
            }
        };
        // The checkpoint's E4M3 bytes and FP32 scale grid: the only copy (per rank's slice).
        let with_fp8 = |ops: &mut Vec<HashMap<&'static str, DeviceAllocation<'a>>>, name: &'static str,
            parts: &[String], axis: Axis| -> Result<()> {
            let (w8_name, scale_name, _) = fp8_operand_names(name);
            let split = if ranks > 1 { self.fp8_split(parts, axis, ranks)? } else { vec![self.fp8(parts)?] };
            for (map, (w8, scale)) in ops.iter_mut().zip(split) {
                map.insert(w8_name, w8);
                map.insert(scale_name, scale);
            }
            Ok(())
        };
        let raw = |name: &str| -> Result<Vec<u8>> {
            let (bytes, dtype, shape) = self.raw(name)?;
            ensure!(!(shape.len() == 2 && dtype == DType::F8E4M3), "{name}: replicated operands are not FP8");
            Ok(bytes)
        };
        put(&mut ops, "input_norm", self.replicated_tensor(ranks, &format!("{p}.input_layernorm.weight"))?);
        put(&mut ops, "post_norm", self.replicated_tensor(ranks, &format!("{p}.post_attention_layernorm.weight"))?);
        let attention = if let Some(owner) = owner {
            let (device, stream) = if owner == 0 { (self.device, self.stream) }
                else { let peer = self.peers.get(usize::from(owner) - 1).context("attention owner")?; (peer.device, peer.stream) };
            let local = GlmLoader { library: self.library, catalog: self.catalog, device, stream, peers: Vec::new(),
                push: self.push.clone() };
            let mut maps = self.on_rank(usize::from(owner), |_| local.attention_operands(cfg, layer))?;
            ops[usize::from(owner)].extend(maps.pop().context("whole attention operands")?);
            None
        } else { Some(self.attention_operands(cfg, layer)?) };
        if let Some(attention) = attention {
            for (ops, attention) in ops.iter_mut().zip(attention) { ops.extend(attention); }
        }
        let mlp = if dense { format!("{p}.mlp") } else { format!("{p}.mlp.shared_experts") };
        let mlp_numerics = if native_shared && !dense {
            cuteafd_loader::plan::checkpoint::ModuleNumerics::CheckpointPrecision
        } else { checkpoint.module_numerics(&format!("{mlp}.gate_proj"))? };
        if dense || !native_shared {
        for projection in ["up_proj", "down_proj"] {
            ensure!(checkpoint.module_numerics(&format!("{mlp}.{projection}"))? == mlp_numerics,
                "{mlp}: mixed declared MLP arithmetic is unsupported");
        }
        let (gate_up, down) = ([format!("{mlp}.gate_proj.weight"), format!("{mlp}.up_proj.weight")],
            [format!("{mlp}.down_proj.weight")]);
        if !dense && self.bf16_native(&[&gate_up[..], &down[..]].concat())? {
            // The checkpoint's BF16 shared expert as-is.
            put(&mut ops, "w_gate_up", self.bf16_split(&gate_up, Axis::Rows, ranks)?);
            put(&mut ops, "w_down", self.bf16_split(&down, Axis::Cols, ranks)?);
        } else {
            with_fp8(&mut ops, "w_gate_up", &gate_up, Axis::Rows)?;
            with_fp8(&mut ops, "w_down", &down, Axis::Cols)?;
        }
        // Retain the checkpoint's static activation scales for every execution shape.
        if dense {
            if let (Some(gu), Some(d)) = (self.tensor_scales(&gate_up)?, self.tensor_scales(&down)?) {
                put(&mut ops, "w_gate_up_tscale", self.upload_all(ranks, &gu)?);
                put(&mut ops, "w_down_tscale", self.upload_all(ranks, &d)?);
            }
        }
        }
        if !dense {
            ops[0].insert("gate", self.one(&format!("{p}.mlp.gate.weight"))?);
            ops[0].insert("gate.bias", self.one(&format!("{p}.mlp.gate.e_score_correction_bias"))?);
        }
        if mlp_numerics == cuteafd_loader::plan::checkpoint::ModuleNumerics::W8A8 {
            ensure!(ops.iter().all(|operands| operands.contains_key("w_gate_up_tscale")
                && operands.contains_key("w_down_tscale")),
                "{mlp}: declared W8A8 needs native per-tensor activation scales");
        }
        Ok(ops.into_iter().map(|operands| GlmLayer { dense, full_indexer, mlp_numerics, split: ranks > 1, attention_split: ranks > 1 && owner.is_none(), attention_owner: owner, operands }).collect())
    }

    /// Layers `0..layers` (all of them unless the caller stops early): rank 0's
    /// weights (its layer shares, the norm and head) and with a head split the
    /// other rank's layer shares.
    #[allow(clippy::type_complexity)]
    pub fn model(&self, cfg: &GlmDsaConfig, layers: usize, owners: Option<&[u8]>, native_shared: bool)
        -> Result<(GlmWeights<'a>, Vec<Vec<GlmLayer<'a>>>)> {
        ensure!(owners.is_none_or(|o| o.len() == layers && self.ranks() == 2 && o.iter().all(|&r| r < 2)),
            "whole attention needs one owner per layer and two ranks");
        let split = crate::shared::memory::load_timing::Split::start();
        let checkpoint = cuteafd_loader::plan::checkpoint::Checkpoint::inventory(self.catalog.snapshot())?;
        let mut shares: Vec<Vec<GlmLayer<'a>>> = (0..self.ranks()).map(|_| Vec::new()).collect();
        for layer in 0..layers.min(cfg.layers) {
            for (share, part) in shares.iter_mut().zip(self.layer(cfg, layer, &checkpoint, owners.map(|o| o[layer]), native_shared)?) {
                share.push(part);
            }
        }
        let mut shares = shares.into_iter();
        let weights = GlmWeights {
            layers: shares.next().context("rank 0")?,
            norm: self.one("model.norm.weight")?,
            head: self.one("lm_head.weight")?,
        };
        crate::shared::memory::staging::release_staging();
        split.log("glm5 dense");
        Ok((weights, shares.collect()))
    }
}

/// BF16 checkpoint weights (nvidia/GLM-5.3-NVFP4's attention, indexer and shared experts) run
/// as-is on the BF16 programs with CUTEAFD_GLM_BF16=native; by default they are quantized to
/// 128x128 FP8 blocks at load (power-of-two scales) for the FP8 programs: BF16 reads twice
/// the bytes (one RTX PRO 6000, coordinator alone: C1 step 28.6 vs 21.0 ms, 8K prefill 3.35
/// vs 2.74 s), a decision pending in PLAN.md Phase 5.
pub(crate) fn bf16_programs() -> bool {
    static NATIVE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *NATIVE.get_or_init(|| std::env::var("CUTEAFD_GLM_BF16").is_ok_and(|v| v == "native"))
}

#[cfg(test)]
#[path = "weights_tests.rs"]
mod operand_tests;
