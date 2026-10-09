//! MiMo V2 coordinator weights, packed for the exported mimo_* programs.
//!
//! The checkpoint's FP8 projections, `w_qkv` and the dense `w_gate_up` /
//! `w_down`, stay FP8 and have no other copy: `{w}_fp8` (the E4M3 bytes,
//! `[N, K]`) with the block grid expanded to one FP32 scale per output row and
//! 128-wide K block (exact, the programs widen to `bf16(w * s)`), row major
//! for decode programs (`{w}_scale`, `[N, K/128]`) and K-block major for
//! prefill programs (`{w}_kscale`, `[K/128, N]`). Decode programs run the FP8 GEMVs up to
//! 32 rows and W8A16 GEMMs above (bitwise the former BF16 programs over the
//! dequantized weights); prefill programs run W8A8 or W8A16. Scale grids are
//! 128x128 blocks except the full-attention `k_proj` [768, 4096], whose
//! [8, 32] grid is per KV head (each 192-row head a 128-row then a 64-row
//! block). `w_qkv` is `[q_proj; k_proj; v_proj]` (V2.6 Pro: the fused,
//! TP-interleaved `qkv_proj` de-interleaved with every key padded to 256
//! rows, see `FusedQkvLayout`).
//!
//! Head and output projections have one immutable format across row shapes:
//! checkpoint BF16 or E4M3 by default, with explicit conversion available at
//! load. FP8 has one value allocation and row/K-major scale metadata, never a
//! resident BF16 fallback. Norms and sinks remain checkpoint BF16; the FP32
//! router weight (Flash) becomes `w_hilo = [bf16(w);
//! bf16(w - bf16(w))]` for the router program's two FP32-accumulated BF16
//! products, a BF16 one (V2.6 Pro) is `w_router` as stored.
use crate::shared::memory::DeviceAllocation;
use super::head::MimoHead;
use cuteafd_loader::families::mimo_v2::projection::{MimoProjectionLayout, MimoProjectionRepresentation, MIMO_PROJECTION_STAGING_ROWS};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::families::mimo_v2::{FusedQkvLayout, MimoAttention, MimoV2Config};
use cuteafd_loader::plan::checkpoint::{Checkpoint, CheckpointTensor};
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::os::unix::fs::FileExt;
use crate::shared::peer_split::{slice_2d, Axis, RankDevice};

#[cfg(test)]
#[path = "weights_retirement_tests.rs"]
mod retirement_tests;

pub(crate) struct MimoLayer<'a> {
    pub attention: MimoAttention,
    pub dense: bool,
    /// One GPU's share of a head split: its qkv/attention/o_proj cover its heads
    /// (o_proj and the dense MLP produce partial sums) and run the split programs.
    pub split: bool,
    operands: HashMap<&'static str, DeviceAllocation<'a>>,
}

impl MimoLayer<'_> {
    /// The device range of `operand`, when the layer has it.
    pub fn range(&self, operand: &str) -> Option<crate::shared::l2_prefetch::Range> {
        self.operands.get(operand).map(|a| (a.buffer.ptr.cast_const(), a.buffer.bytes))
    }

    pub fn ptr(&self, operand: &str) -> Result<*mut c_void> {
        Ok(self.operands.get(operand).with_context(|| format!("layer has no weight {operand}"))?.buffer.ptr)
    }

    pub fn has(&self, operand: &str) -> bool {
        self.operands.contains_key(operand)
    }

    /// The router scores program's weight operand: `w_hilo` (FP32 weight split
    /// into BF16 hi + lo, Flash) or `w_router` (BF16 as stored, V2.6 Pro).
    pub fn router_operand(&self) -> Result<(&'static str, *mut c_void)> {
        for name in ["w_hilo", "w_router"] {
            if let Some(weight) = self.operands.get(name) {
                return Ok((name, weight.buffer.ptr));
            }
        }
        anyhow::bail!("layer has no router weight")
    }

    /// `operand`, or `fallback` when the layer has no such weight (an FP8
    /// operand the program then does not read: it runs with `fp8_rows` 0).
    pub fn ptr_or(&self, operand: &str, fallback: &str) -> Result<*mut c_void> {
        self.operands.get(operand).map_or_else(|| self.ptr(fallback), |a| Ok(a.buffer.ptr))
    }

    /// Device bytes of the layer's operands.
    pub fn bytes(&self) -> usize {
        self.operands.values().map(|a| a.buffer.bytes).sum()
    }
}

pub(crate) struct MimoWeights<'a> {
    pub layers: Vec<MimoLayer<'a>>,
    pub norm: DeviceAllocation<'a>,
    pub head: MimoHead<'a>,
    pub output_fp8: bool,
}

pub(crate) struct MimoLoader<'a> {
    pub library: &'a NativeLibrary,
    pub checkpoint: &'a Checkpoint,
    pub stream: *mut c_void,
    /// The checkpoint's tensor-parallel degree (fused `qkv_proj` row shards).
    pub checkpoint_tp: usize,
    pub fp8_head: bool,
    pub output_formats: &'a BTreeMap<String, MimoProjectionRepresentation>,
    /// Scale rule of copies quantized from BF16 (o_proj, the LM head).
    pub fp8_scales: crate::shared::fp8_linear::Fp8Scales,
    /// This loader's device (rank 0 of a head split).
    pub device: i32,
    /// The other GPUs of a head split, ranks 1.. (empty: one GPU).
    pub peers: Vec<RankDevice>,
}

/// Scale-grid row of every weight row: uniform 128-row blocks, or per
/// 192-row head (a 128-row then a 64-row block).
pub(crate) fn scale_rows(name: &str, rows: usize, grid_rows: usize) -> Result<Vec<usize>> {
    if rows.div_ceil(128) == grid_rows {
        return Ok((0..rows).map(|r| r / 128).collect());
    }
    let head = 192;
    ensure!(rows % head == 0 && grid_rows == rows / head * head.div_ceil(128),
        "{name}: no block layout maps {rows} rows onto {grid_rows} scale rows");
    Ok((0..rows).map(|r| r / head * head.div_ceil(128) + r % head / 128).collect())
}

/// An FP8-only weight: E4M3 values and its per-row x 128-K scales row major
/// (`[N, K/128]`, decode programs) and K-block major (`[K/128, N]`, prefill).
struct Fp8Copy<'a> {
    values: DeviceAllocation<'a>,
    scale: DeviceAllocation<'a>,
    kscale: DeviceAllocation<'a>,
}

impl<'a> Fp8Copy<'a> {
    fn insert(self, ops: &mut HashMap<&'static str, DeviceAllocation<'a>>, names: [&'static str; 3]) {
        ops.insert(names[0], self.values);
        ops.insert(names[1], self.scale);
        ops.insert(names[2], self.kscale);
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut out = vec![0u8; values.len() * 4];
    for (dst, v) in out.chunks_exact_mut(4).zip(values) {
        dst.copy_from_slice(&v.to_le_bytes());
    }
    out
}

/// Row-major `[N, KB]` FP32 scales as K-block-major `[KB, N]` bytes.
fn kmajor(row_scales: &[f32], k_blocks: usize) -> Vec<u8> {
    let n = row_scales.len() / k_blocks;
    let mut out = vec![0u8; row_scales.len() * 4];
    for (b, block) in out.chunks_exact_mut(n * 4).enumerate() {
        for (r, value) in block.chunks_exact_mut(4).enumerate() {
            value.copy_from_slice(&row_scales[r * k_blocks + b].to_le_bytes());
        }
    }
    out
}

impl<'a> MimoLoader<'a> {
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

    /// Reads tensor `name`'s bytes into `out[at..]` (one positioned read into the
    /// operand's buffer: no second copy) and returns its byte length, dtype and shape.
    fn read_into(&self, name: &str, out: &mut [u8], at: usize) -> Result<(usize, DType, Vec<usize>)> {
        cuteafd_ffi::memory_ledger::tensor(name);
        let tensor = self.tensor(name)?;
        let length = tensor.meta.byte_length as usize;
        ensure!(at + length <= out.len(), "{name}: {length} bytes past the operand buffer");
        std::fs::File::open(self.checkpoint.snapshot.join(&tensor.shard))?
            .read_exact_at(&mut out[at..at + length], tensor.meta.byte_offset)
            .with_context(|| format!("reading {name}"))?;
        Ok((length, tensor.meta.dtype.clone(), tensor.meta.shape.clone()))
    }

    fn upload(&self, bytes: &[u8]) -> Result<DeviceAllocation<'a>> {
        let allocation = DeviceAllocation::new(self.library, bytes.len().max(256))?;
        self.library.copy_h2d(allocation.buffer, bytes)?;
        Ok(allocation)
    }

    /// One FP8 projection per head partition. Checkpoint FP8 values/scales
    /// remain exact; BF16 checkpoints are quantized through a bounded source
    /// block. No complete BF16 projection survives this operation.
    #[allow(clippy::type_complexity)]
    fn fp8_projection(&self, name: &str, ranks: usize, kmajor_scale: bool)
        -> Result<Vec<(DeviceAllocation<'a>, DeviceAllocation<'a>, Option<DeviceAllocation<'a>>)>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("fp8");
        let (bytes, dtype, shape) = self.raw(name)?;
        ensure!(shape.len() == 2 && shape[1] % (ranks *128) == 0,
            "{name}: FP8-only projection requires [N,K] with K divisible by128 per rank");
        ensure!(matches!(dtype, DType::Bf16 | DType::F8E4M3),
            "{name}: FP8-only projection needs BF16 or E4M3 checkpoint values, found {dtype:?}");
        let (rows, cols) = (shape[0], shape[1] /ranks);
        let layout = MimoProjectionLayout::new(rows as u64, cols as u64, MimoProjectionRepresentation::Fp8,
            if kmajor_scale { 2 } else { 1 })?;
        ensure!(bytes.len() as u64 == layout.values.checked_mul(ranks as u64)
            .and_then(|n| n.checked_mul(if dtype == DType::Bf16 { 2 } else { 1 })).context("projection source extent overflow")?,
            "{name}: checkpoint projection bytes disagree with its shape");
        let grid = if dtype == DType::F8E4M3 {
            let (bytes, dtype, shape) = self.raw(&format!("{name}_scale_inv"))?;
            ensure!(dtype == DType::F32 && shape.len() == 2 && shape[1] == cols /128 *ranks,
                "{name}: FP8-only projection needs checkpoint FP32 block scales");
            ensure!(bytes.len() == shape[0] *shape[1] *4, "{name}: checkpoint scale bytes disagree with its grid");
            Some((bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<_>>(),
                scale_rows(name, rows, shape[0])?))
        } else { None };
        (0..ranks).map(|rank| self.on_rank(rank, |stream| {
            let values = DeviceAllocation::new(self.library, usize::try_from(layout.values)?)?;
            let scales = DeviceAllocation::new(self.library, rows * (cols /128) *4)?;
            let region = |buffer: CuteafdDeviceBuffer, offset: usize, count: usize| CuteafdDeviceBuffer {
                ptr: buffer.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes: count, ..buffer
            };
            let row_scales = if let Some((grid, grid_rows)) = &grid {
                let values_host = slice_2d(&bytes, rows, shape[1], 1, Axis::Cols, rank, ranks);
                self.library.copy_h2d(values.buffer, &values_host)?;
                let kb = cols /128;
                let row_scales: Vec<f32> = grid_rows.iter().flat_map(|&r|
                    grid[r *kb *ranks + rank *kb..r *kb *ranks + (rank +1) *kb].iter().copied()).collect();
                self.library.copy_h2d(scales.buffer, &f32_bytes(&row_scales))?;
                row_scales
            } else {
                let staging = DeviceAllocation::new(self.library, usize::try_from(layout.max_load_staging)?)?;
                let queued = (|| -> Result<()> {
                    let mut block = vec![0u8; staging.buffer.bytes];
                    for first in (0..rows).step_by(MIMO_PROJECTION_STAGING_ROWS as usize) {
                        let n = (MIMO_PROJECTION_STAGING_ROWS as usize).min(rows -first);
                        for r in 0..n {
                            let offset = ((first +r) *shape[1] +rank *cols) *2;
                            block[r *cols *2..(r +1) *cols *2].copy_from_slice(&bytes[offset..offset +cols *2]);
                        }
                        self.library.copy_h2d(region(staging.buffer, 0, n *cols *2), &block[..n *cols *2])?;
                        // SAFETY: live source block and checked destinations;
                        // every block drains before the source buffer is reused.
                        unsafe {
                            self.library.fp8_quant_rule(staging.buffer.ptr,
                                region(values.buffer, first *cols, n *cols).ptr,
                                region(scales.buffer, first *cols /128 *4, n *cols /128 *4).ptr,
                                n, cols, true, self.fp8_scales.code(), stream)?;
                            self.library.cuda_stream_synchronize(stream)?;
                        }
                    }
                    Ok(())
                })();
                // SAFETY: all queued sources/destinations are still owned here,
                // including when a native launch returned an error.
                if let Err(drain) = unsafe { self.library.cuda_stream_synchronize(stream) } {
                    self.library.quarantine_module_after_failed_drain();
                    std::mem::forget(staging);
                    std::mem::forget(values);
                    std::mem::forget(scales);
                    return Err(queued.err().unwrap_or_else(|| anyhow::anyhow!("FP8-only projection completion failed"))
                        .context(format!("FP8-only projection owners and native module quarantined after failed packing drain: {drain}")));
                }
                queued?;
                drop(staging);
                if kmajor_scale {
                    let mut bytes = vec![0u8; scales.buffer.bytes];
                    self.library.copy_d2h(&mut bytes, scales.buffer)?;
                    bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect()
                } else { Vec::new() }
            };
            let kscale = if kmajor_scale { Some(self.upload(&kmajor(&row_scales, cols /128))?) } else { None };
            Ok((values, scales, kscale))
        })).collect()
    }

    /// The row-concatenation of 2-D `names` as one BF16 operand.
    fn rows(&self, names: &[String]) -> Result<DeviceAllocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        let tensors = names.iter().map(|n| self.raw(n).map(|t| (n, t))).collect::<Result<Vec<_>>>()?;
        ensure!(!tensors.is_empty() && tensors.iter().all(|(_, (_, _, s))| s.len() == 2 && s[0] > 0 && s[1] > 0),
            "{names:?}: row operands require nonempty two-dimensional tensors");
        let cols = tensors[0].1 .2[1];
        let rows = tensors.iter().try_fold(0usize, |rows, (_, (_, _, shape))| rows.checked_add(shape[0]))
            .context("row operand height overflow")?;
        ensure!(tensors.iter().all(|(_, (_, _, s))| s.len() == 2 && s[1] == cols), "{names:?} do not share columns");
        let k_blocks = cols.div_ceil(128);
        let bytes = rows.checked_mul(cols).and_then(|n| n.checked_mul(2)).context("row operand storage overflow")?;
        let out = DeviceAllocation::new(self.library, bytes)?;
        let at = |buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize| CuteafdDeviceBuffer {
            // SAFETY: callers keep offset + bytes inside the allocation.
            ptr: unsafe { buffer.ptr.cast::<u8>().add(offset) }.cast(),
            bytes,
            ..buffer
        };
        let mut row = 0;
        // Uploads the stream still reads; released after the final synchronize.
        let mut staged = Vec::new();
        let queued = (|| -> Result<()> {
            for (name, (bytes, dtype, shape)) in &tensors {
                let dest = |first: usize, count: usize| at(out.buffer, (row + first) * cols * 2, count * cols * 2);
                match dtype {
                    DType::Bf16 => {
                        ensure!(bytes.len() == shape[0] * cols * 2, "{name}: BF16 bytes disagree with row shape");
                        self.library.copy_h2d(dest(0, shape[0]), bytes)?;
                    }
                    DType::F8E4M3 => {
                        ensure!(bytes.len() == shape[0] * cols, "{name}: FP8 bytes disagree with row shape");
                        let (scale, scale_dtype, scale_shape) = self.raw(&format!("{name}_scale_inv"))?;
                        ensure!(scale_dtype == DType::F32, "{name}: block scales must be FP32");
                        ensure!(scale_shape.len() == 2 && scale_shape[1] == k_blocks,
                            "{name}: unexpected scale grid {scale_shape:?}");
                        ensure!(scale_shape[0].checked_mul(k_blocks).and_then(|n| n.checked_mul(4)) == Some(scale.len()),
                            "{name}: FP32 scale bytes disagree with the grid");
                        // Row blocks: uniform 128, or per 192-row head (128 + 64).
                        let (block, per_block_rows) = if shape[0].div_ceil(128) == scale_shape[0] {
                            (shape[0], shape[0].div_ceil(128))
                        } else {
                            let head = 192;
                            ensure!(shape[0] % head == 0 && scale_shape[0] == shape[0] / head * head.div_ceil(128),
                                "{name}: no block layout maps {} rows onto {} scale rows", shape[0], scale_shape[0]);
                            (head, head.div_ceil(128))
                        };
                        let (w, s) = (self.upload(bytes)?, self.upload(&scale)?);
                        // Install both owners before the first fallible launch.
                        staged.push((w, s));
                        let (w, s) = staged.last().expect("just inserted staging owners");
                        for chunk in 0..shape[0] / block {
                            // SAFETY: chunk rows of the FP8 weight, their scale rows and the
                            // destination rows are live; the stream drains before `w`/`s` drop.
                            unsafe {
                                self.library.fp8_block_dequant(
                                    w.buffer.ptr.cast::<u8>().add(chunk * block * cols).cast(),
                                    s.buffer.ptr.cast::<u8>().add(chunk * per_block_rows * k_blocks * 4).cast(),
                                    dest(chunk * block, block).ptr, block, cols, self.stream)?;
                            }
                        }
                    }
                    other => anyhow::bail!("{name}: unsupported coordinator dtype {other:?}"),
                }
                row += shape[0];
            }
        Ok(())
        })();
        // SAFETY: the loader owns this stream and every source/destination,
        // including after an error while processing a later tensor.
        if let Err(drain) = unsafe { self.library.cuda_stream_synchronize(self.stream) } {
            self.library.quarantine_module_after_failed_drain();
            std::mem::forget(staged);
            std::mem::forget(out);
            return Err(queued.err().unwrap_or_else(|| anyhow::anyhow!("row operand completion failed"))
                .context(format!("row operand owners and native module quarantined after failed drain: {drain}")));
        }
        queued?;
        drop(staged);
        Ok(out)
    }

    fn has(&self, name: &str) -> bool {
        self.checkpoint.contains_tensor(name)
    }

    /// The row-concatenation of the FP8 checkpoint weights `names` as their
    /// E4M3 bytes `[N, K]` and FP32 per-row x 128-K scales (each row's
    /// block-grid value; exact), row major and K-block major.
    fn fp8_kmajor(&self, names: &[String]) -> Result<Fp8Copy<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("fp8");
        let total: u64 = names.iter().map(|n| self.tensor(n).map(|t| t.meta.byte_length)).sum::<Result<u64>>()?;
        crate::shared::memory::staging::with_staging(total as usize, |values| {
            // Per row of the concatenation: its tensor's grid and that grid's block row.
            let (mut grids, mut rows) = (Vec::new(), Vec::<(usize, usize)>::new());
            let (mut cols, mut at) = (None, 0usize);
            for name in names {
                let (length, dtype, shape) = self.read_into(name, values, at)?;
                at += length;
                ensure!(dtype == DType::F8E4M3 && shape.len() == 2 && shape[1] % 128 == 0
                    && cols.is_none_or(|c| c == shape[1]),
                    "{name}: the mimo programs take this weight as an FP8 checkpoint tensor (the official FP8 \
                     release), found {dtype:?} {shape:?}");
                cols = Some(shape[1]);
                let k_blocks = shape[1] / 128;
                let (scale, scale_dtype, scale_shape) = self.raw(&format!("{name}_scale_inv"))?;
                ensure!(scale_dtype == DType::F32 && scale_shape.len() == 2 && scale_shape[1] == k_blocks,
                    "{name}: expected FP32 block scales, found {scale_dtype:?} {scale_shape:?}");
                grids.push(scale.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<f32>>());
                rows.extend(scale_rows(name, shape[0], scale_shape[0])?.into_iter().map(|r| (grids.len() - 1, r)));
            }
            let k_blocks = cols.context("no FP8 rows")? / 128;
            // Row major (decode) and K-block major (prefill) copies of each row's grid values.
            let n = rows.len();
            let mut scale = vec![0u8; k_blocks * n * 4];
            for (row, &(grid, r)) in scale.chunks_exact_mut(k_blocks * 4).zip(&rows) {
                for (value, v) in row.chunks_exact_mut(4).zip(&grids[grid][r * k_blocks..(r + 1) * k_blocks]) {
                    value.copy_from_slice(&v.to_le_bytes());
                }
            }
            let mut kscale = vec![0u8; k_blocks * n * 4];
            for (b, block) in kscale.chunks_exact_mut(n * 4).enumerate() {
                for (value, &(grid, r)) in block.chunks_exact_mut(4).zip(&rows) {
                    value.copy_from_slice(&grids[grid][r * k_blocks + b].to_le_bytes());
                }
            }
            Ok(Fp8Copy { values: self.upload(values)?, scale: self.upload(&scale)?, kscale: self.upload(&kscale)? })
        })
    }

    /// [`Self::fp8_kmajor`] over a head split's `ranks` GPUs: each tensor's rows
    /// split evenly (V2 Flash's q / k / v by heads: whole query heads and the KV
    /// heads they read), rank `r`'s slices concatenated in tensor order, every row
    /// with its own grid value. Read once.
    fn fp8_kmajor_ranks(&self, names: &[String], ranks: usize) -> Result<Vec<Fp8Copy<'a>>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("fp8");
        let total: u64 = names.iter().map(|n| self.tensor(n).map(|t| t.meta.byte_length)).sum::<Result<u64>>()?;
        crate::shared::memory::staging::with_staging(total as usize, |values| {
            // Per tensor: its first row in the concatenation and row count; per row: grid and block row.
            let (mut grids, mut rows, mut parts) = (Vec::new(), Vec::<(usize, usize)>::new(), Vec::new());
            let (mut cols, mut at) = (None, 0usize);
            for name in names {
                let (length, dtype, shape) = self.read_into(name, values, at)?;
                at += length;
                ensure!(dtype == DType::F8E4M3 && shape.len() == 2 && shape[1] % 128 == 0
                    && cols.is_none_or(|c| c == shape[1]) && shape[0] % ranks == 0,
                    "{name}: a head split takes this weight as an FP8 checkpoint tensor whose rows split over {ranks} \
                     GPUs, found {dtype:?} {shape:?}");
                cols = Some(shape[1]);
                let (scale, scale_dtype, scale_shape) = self.raw(&format!("{name}_scale_inv"))?;
                ensure!(scale_dtype == DType::F32 && scale_shape.len() == 2 && scale_shape[1] == shape[1] / 128,
                    "{name}: expected FP32 block scales, found {scale_dtype:?} {scale_shape:?}");
                grids.push(scale.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect::<Vec<f32>>());
                parts.push((rows.len(), shape[0]));
                rows.extend(scale_rows(name, shape[0], scale_shape[0])?.into_iter().map(|r| (grids.len() - 1, r)));
            }
            let cols = cols.context("no FP8 rows")?;
            let k_blocks = cols / 128;
            (0..ranks).map(|rank| {
                let picked: Vec<usize> = parts.iter()
                    .flat_map(|&(first, count)| first + rank * count / ranks..first + (rank + 1) * count / ranks).collect();
                let n = picked.len();
                let mut bytes = Vec::with_capacity(n * cols);
                let (mut scale, mut kscale) = (vec![0u8; k_blocks * n * 4], vec![0u8; k_blocks * n * 4]);
                for (i, &row) in picked.iter().enumerate() {
                    bytes.extend_from_slice(&values[row * cols..(row + 1) * cols]);
                    let (grid, r) = rows[row];
                    for b in 0..k_blocks {
                        let v = grids[grid][r * k_blocks + b].to_le_bytes();
                        scale[(i * k_blocks + b) * 4..][..4].copy_from_slice(&v);
                        kscale[(b * n + i) * 4..][..4].copy_from_slice(&v);
                    }
                }
                self.on_rank(rank, |_| Ok(Fp8Copy { values: self.upload(&bytes)?, scale: self.upload(&scale)?,
                    kscale: self.upload(&kscale)? }))
            }).collect()
        })
    }

    /// V2.6 Pro's fused `qkv_proj` (FP8, TP-interleaved row shards with their
    /// own whole-shard 128x128 grids) in the coordinator's `[q; k; v]` layout with keys
    /// `cfg.qkv_key_stride()` rows apart (256: each 192-row key zero-padded):
    /// the E4M3 rows and FP32 per-row x 128-K scales, row major and K-block
    /// major (exactly the checkpoint's values; padding rows keep zero values and scales).
    /// Over `ranks` GPUs (a head split), rank `r` takes checkpoint shards
    /// `r * tp / ranks ..`: its query heads and the KV heads they read, in the
    /// same layout at its share's geometry. Read once, one copy per rank.
    fn fused_qkv(&self, cfg: &MimoV2Config, attention: MimoAttention, name: &str, ranks: usize)
        -> Result<Vec<Fp8Copy<'a>>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("fp8");
        let full = FusedQkvLayout::new(cfg, attention, self.checkpoint_tp)?;
        ensure!(self.checkpoint_tp % ranks == 0, "{name}: checkpoint TP {} does not split over {ranks} GPUs",
            self.checkpoint_tp);
        let share = cfg.head_split(ranks)?;
        let layout = FusedQkvLayout::new(&share, attention, self.checkpoint_tp / ranks)?;
        ensure!(layout.rows() * ranks == full.rows() && layout.scale_rows() * ranks == full.scale_rows(),
            "{name}: checkpoint shards do not divide into {ranks} head groups");
        let source_bytes = self.tensor(name)?.meta.byte_length as usize;
        let (scale, scale_dtype, scale_shape) = self.raw(&format!("{name}_scale_inv"))?;
        let (width, segments) = layout.program_segments(&share)?;
        let cols = source_bytes / full.rows().max(1);
        crate::shared::memory::staging::with_staging_pair(width * cols, source_bytes, |values, bytes| {
            let (_, dtype, shape) = self.read_into(name, bytes, 0)?;
            let k_blocks = cols.div_ceil(128);
            ensure!(dtype == DType::F8E4M3 && shape == [full.rows(), cols] && scale_dtype == DType::F32
                && scale_shape == [full.scale_rows(), k_blocks] && cols % 128 == 0,
                "{name}: expected E4M3 [{}, {cols}] with FP32 [{}, {}] scales for checkpoint TP {}, found {dtype:?} \
                 {shape:?} / {scale_dtype:?} {scale_shape:?}", full.rows(), full.scale_rows(), k_blocks,
                self.checkpoint_tp);
            let grid: Vec<f32> = scale.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            (0..ranks).map(|rank| {
                // This rank's shards: a contiguous run of checkpoint rows and grid rows.
                let (source0, scale0) = (rank * layout.rows(), rank * layout.scale_rows());
                // The padded E4M3 rows and each row's shard-grid scales; padding rows are zero.
                let mut covered = vec![false; width];
                let mut row_scales = vec![0f32; width * k_blocks];
                for segment in &segments {
                    values[segment.dest_row * cols..][..segment.rows * cols]
                        .copy_from_slice(&bytes[(source0 + segment.source_row) * cols..][..segment.rows * cols]);
                    covered[segment.dest_row..segment.dest_row + segment.rows].fill(true);
                    for r in 0..segment.rows {
                        row_scales[(segment.dest_row + r) * k_blocks..][..k_blocks]
                            .copy_from_slice(&grid[(scale0 + segment.scale_row_of(r)) * k_blocks..][..k_blocks]);
                    }
                }
                for (row, _) in covered.iter().enumerate().filter(|(_, &c)| !c) {
                    values[row * cols..(row + 1) * cols].fill(0);
                }
                self.on_rank(rank, |_| Ok(Fp8Copy { values: self.upload(values)?,
                    scale: self.upload(&f32_bytes(&row_scales))?, kscale: self.upload(&kmajor(&row_scales, k_blocks))? }))
            }).collect()
        })
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
        self.library.cuda_set_device(peer.device)?;
        let out = body(peer.stream);
        self.library.cuda_set_device(self.device)?;
        out
    }

    /// The BF16 o_proj `[H, heads * v_head]` sliced by columns over `ranks`
    /// (rank `r`'s heads). The final operands own BF16 values only.
    fn o_proj(&self, name: &str, ranks: usize) -> Result<Vec<DeviceAllocation<'a>>> {
        let tensor = self.tensor(name)?;
        if ranks == 1 && tensor.meta.dtype != DType::Bf16 {
            return Ok(vec![self.rows(&[name.to_string()])?]);
        }
        let shape = tensor.meta.shape.clone();
        ensure!(tensor.meta.dtype == DType::Bf16 && shape.len() == 2 && (ranks == 1 || shape[1] % (ranks * 128) == 0),
            "{name}: a head split takes o_proj as BF16 [H, K] with K a multiple of {}, found {:?} {shape:?}",
            ranks * 128, tensor.meta.dtype);
        let (rows, cols) = (shape[0], shape[1] / ranks);
        // The whole weight goes up once to rank 0; each rank's columns are a pitched
        // device copy from it (over peer memory for the others).
        let whole = crate::shared::memory::staging::with_staging(tensor.meta.byte_length as usize, |bytes| {
            self.read_into(name, bytes, 0)?;
            self.upload(bytes)
        })?;
        let mut whole = Some(whole);
        let mut completed = Vec::with_capacity(ranks);
        for rank in 0..ranks {
            let operand = self.on_rank(rank, |stream| {
                if ranks == 1 {
                    // One GPU: the uploaded weight itself.
                    let out = whole.take().context("o_proj")?;
                    return Ok(out);
                }
                let source_allocation = whole.as_ref().context("o_proj")?;
                let out = DeviceAllocation::new(self.library, rows * cols * 2)?;
                let source = CuteafdDeviceBuffer {
                    // SAFETY: column block `rank` of every row lies inside the whole weight.
                    ptr: unsafe { source_allocation.buffer.ptr.cast::<u8>().add(rank * cols * 2) }.cast(),
                    bytes: source_allocation.buffer.bytes - rank * cols * 2,
                    ..source_allocation.buffer
                };
                // SAFETY: both buffers are live and sized for these pitched spans; peer access to
                // rank 0 is enabled on every rank (the engine's split setup), drained below.
                let queued = unsafe {
                    self.library.copy_d2d_2d_async(out.buffer, cols * 2, source, shape[1] * 2, cols * 2, rows, stream)
                };
                // SAFETY: the loader owns this stream; the copy reads `whole`, which drops after.
                if let Err(drain) = unsafe { self.library.cuda_stream_synchronize(stream) } {
                    self.library.quarantine_module_after_failed_drain();
                    std::mem::forget(out);
                    std::mem::forget(whole.take());
                    // cudaFree of a completed rank's independent output can still
                    // synchronize with this pending peer copy. Retain every split
                    // output on the unproved path so error return cannot block on
                    // unrelated allocation destruction.
                    std::mem::forget(std::mem::take(&mut completed));
                    return Err(queued.err().unwrap_or_else(|| anyhow::anyhow!("split projection completion failed"))
                        .context(format!("split projection owners and native module quarantined after failed drain: {drain}")));
                }
                queued?;
                Ok(out)
            })?;
            completed.push(operand);
        }
        drop(whole);
        Ok(completed)
    }

    /// The FP8 checkpoint weights `names` concatenated by rows (as
    /// `fp8_kmajor`), each sliced over `ranks` along `axis` (whole 128-row or
    /// 128-K blocks, so every slice keeps exactly its grid values).
    fn fp8_split(&self, names: &[String], axis: Axis, ranks: usize) -> Result<Vec<Fp8Copy<'a>>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("fp8");
        if ranks == 1 {
            return Ok(vec![self.fp8_kmajor(names)?]);
        }
        let tensors = names.iter().map(|name| -> Result<_> {
            let (bytes, dtype, shape) = self.raw(name)?;
            let (scale, scale_dtype, scale_shape) = self.raw(&format!("{name}_scale_inv"))?;
            ensure!(dtype == DType::F8E4M3 && shape.len() == 2 && scale_dtype == DType::F32
                && scale_shape == [shape[0].div_ceil(128), shape[1].div_ceil(128)]
                && shape[if axis == Axis::Rows { 0 } else { 1 }] % (ranks * 128) == 0 && shape[1] % 128 == 0,
                "{name}: a head split takes this weight as E4M3 with 128x128 FP32 blocks split into whole blocks, \
                 found {dtype:?} {shape:?} / {scale_dtype:?} {scale_shape:?}");
            let grid: Vec<f32> = scale.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            Ok((bytes, shape, grid))
        }).collect::<Result<Vec<_>>>()?;
        (0..ranks).map(|rank| {
            let (mut values, mut row_scales, mut k_blocks) = (Vec::new(), Vec::new(), 0);
            for (bytes, shape, grid) in &tensors {
                let (rows, cols) = (shape[0], shape[1]);
                let kb = cols / 128;
                values.extend(slice_2d(bytes, rows, cols, 1, axis, rank, ranks));
                let (first, count, kb0, kbn) = match axis {
                    Axis::Rows => (rank * rows / ranks, rows / ranks, 0, kb),
                    Axis::Cols => (0, rows, rank * kb / ranks, kb / ranks),
                };
                k_blocks = kbn;
                for r in first..first + count {
                    row_scales.extend_from_slice(&grid[(r / 128) * kb + kb0..][..kbn]);
                }
            }
            self.on_rank(rank, |_| Ok(Fp8Copy { values: self.upload(&values)?, scale: self.upload(&f32_bytes(&row_scales))?,
                kscale: self.upload(&kmajor(&row_scales, k_blocks))? }))
        }).collect()
    }

    fn one(&self, name: &str) -> Result<DeviceAllocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16");
        let (bytes, dtype, shape) = self.raw(name)?;
        if shape.len() == 2 && dtype == DType::F8E4M3 {
            return self.rows(&[name.to_string()]);
        }
        self.upload(&bytes)
    }

    /// `[bf16(w); bf16(w - bf16(w))]` of the FP32 router weight.
    fn router_hilo(&self, name: &str) -> Result<DeviceAllocation<'a>> {
        let _memory_format = cuteafd_ffi::memory_ledger::format("bf16-hilo");
        let (bytes, dtype, shape) = self.raw(name)?;
        ensure!(dtype == DType::F32 && shape.len() == 2, "{name}: the MiMo router weight is FP32 [E, H]");
        let bf16 = |x: f32| -> u16 {
            // Round to nearest even, as torch's float -> bfloat16.
            let bits = u64::from(x.to_bits());
            ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
        };
        let values: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let mut out = Vec::with_capacity(values.len() * 4);
        let hi: Vec<u16> = values.iter().map(|&x| bf16(x)).collect();
        for h in &hi {
            out.extend_from_slice(&h.to_le_bytes());
        }
        for (&x, &h) in values.iter().zip(&hi) {
            out.extend_from_slice(&bf16(x - f32::from_bits(u32::from(h) << 16)).to_le_bytes());
        }
        self.upload(&out)
    }

    /// Layer `layer`, one share per rank of this loader's head split (one
    /// element without a split).
    pub fn layer(&self, cfg: &MimoV2Config, layer: usize) -> Result<Vec<MimoLayer<'a>>> {
        self.block(cfg, &format!("model.layers.{layer}"), cfg.attention[layer], cfg.dense[layer],
            "post_attention_layernorm", self.ranks())
    }

    /// MTP layer `k` (`model.mtp.layers.{k}`): an SWA decoder layer with a
    /// dense MLP (`pre_mlp_layernorm` is its post-attention norm). Whole, on
    /// this loader's GPU (rank 0), under a head split too.
    pub fn mtp_layer(&self, cfg: &MimoV2Config, k: usize) -> Result<MimoLayer<'a>> {
        let layer = self.block(cfg, &format!("model.mtp.layers.{k}"), MimoAttention::Sliding, true, "pre_mlp_layernorm", 1);
        crate::shared::memory::staging::release_staging();
        Ok(layer?.remove(0))
    }

    /// One MTP block's extra weights: `eh_proj` (BF16 [H, 2H]), `enorm`,
    /// `hnorm` and `final_layernorm`.
    pub fn mtp_extras(&self, k: usize) -> Result<[DeviceAllocation<'a>; 4]> {
        let p = format!("model.mtp.layers.{k}");
        Ok([self.one(&format!("{p}.eh_proj.weight"))?, self.one(&format!("{p}.enorm.weight"))?,
            self.one(&format!("{p}.hnorm.weight"))?, self.one(&format!("{p}.final_layernorm.weight"))?])
    }

    /// One decoder block, one share per rank over `ranks` GPUs: rank `r` holds
    /// its heads' qkv rows, sinks and o_proj columns and its slice of the dense
    /// MLP's intermediate; every rank the norms; rank 0 the router.
    fn block(&self, cfg: &MimoV2Config, p: &str, attention: MimoAttention, dense: bool, post: &str, ranks: usize)
        -> Result<Vec<MimoLayer<'a>>> {
        let layer = p;
        let mut ops: Vec<HashMap<&'static str, DeviceAllocation<'a>>> = (0..ranks).map(|_| HashMap::new()).collect();
        // Small BF16 operands: read once, every rank's slice (`Some(axis)`) or a copy.
        let small = |ops: &mut Vec<HashMap<&'static str, DeviceAllocation<'a>>>, key: &'static str, name: &str,
            split: bool| -> Result<()> {
            let (bytes, _, _) = self.raw(name)?;
            ensure!(!split || bytes.len() % ranks == 0, "{name} does not split over {ranks} GPUs");
            for (rank, map) in ops.iter_mut().enumerate() {
                let part = if split { &bytes[rank * bytes.len() / ranks..(rank + 1) * bytes.len() / ranks] } else { &bytes };
                map.insert(key, self.on_rank(rank, |_| self.upload(part))?);
            }
            Ok(())
        };
        small(&mut ops, "input_norm", &format!("{p}.input_layernorm.weight"), false)?;
        small(&mut ops, "post_norm", &format!("{p}.{post}.weight"), false)?;
        let fused = format!("{p}.self_attn.qkv_proj.weight");
        let qkv = [format!("{p}.self_attn.q_proj.weight"), format!("{p}.self_attn.k_proj.weight"),
            format!("{p}.self_attn.v_proj.weight")];
        let copies = if self.has(&fused) {
            self.fused_qkv(cfg, attention, &fused, ranks)?
        } else if ranks > 1 {
            self.fp8_kmajor_ranks(&qkv, ranks)?
        } else {
            vec![self.fp8_kmajor(&qkv)?]
        };
        for (map, copy) in ops.iter_mut().zip(copies) {
            copy.insert(map, ["w_qkv_fp8", "w_qkv_scale", "w_qkv_kscale"]);
        }
        // Immutable output representation across prefill/decode/verify/MTP.
        // Activation/kernel choice never requires another resident weight.
        let name = format!("{p}.self_attn.o_proj.weight");
        ensure!(self.tensor(&name)?.meta.shape == [cfg.hidden, cfg.heads *cfg.v_head_dim],
            "{name}: target output projection does not match configured attention geometry");
        let output_format = self.output_formats.get(&name)
            .with_context(|| format!("{name}: output weight format was not resolved before loading"))?;
        if *output_format == MimoProjectionRepresentation::Fp8 {
            for (map, (q, s, ks)) in ops.iter_mut().zip(self.fp8_projection(&name, ranks, true)?) {
                map.insert("w_o_fp8", q);
                map.insert("w_o_scale", s);
                map.insert("w_o_kscale", ks.context("FP8-only output K-major scales")?);
            }
        } else {
            for (map, bf16) in ops.iter_mut().zip(self.o_proj(&name, ranks)?) {
                map.insert("w_o", bf16);
            }
        }
        let sinks = match attention {
            MimoAttention::Full => cfg.full_sinks,
            MimoAttention::Sliding => cfg.swa_sinks,
        };
        ensure!(sinks == (attention == MimoAttention::Sliding),
            "layer {layer}: the mimo programs take sinks on SWA layers only");
        if sinks {
            small(&mut ops, "sinks", &format!("{p}.self_attn.attention_sink_bias"), true)?;
        }
        if dense {
            let gate_up = self.fp8_split(&[format!("{p}.mlp.gate_proj.weight"), format!("{p}.mlp.up_proj.weight")],
                Axis::Rows, ranks)?;
            let down = self.fp8_split(&[format!("{p}.mlp.down_proj.weight")], Axis::Cols, ranks)?;
            for ((map, gate_up), down) in ops.iter_mut().zip(gate_up).zip(down) {
                gate_up.insert(map, ["w_gate_up_fp8", "w_gate_up_scale", "w_gate_up_kscale"]);
                down.insert(map, ["w_down_fp8", "w_down_scale", "w_down_kscale"]);
            }
        } else {
            let router = format!("{p}.mlp.gate.weight");
            let expected = if cfg.router_fp32 { DType::F32 } else { DType::Bf16 };
            ensure!(self.tensor(&router)?.meta.dtype == expected,
                "{router}: selected program requires {expected:?} router weights");
            if !cfg.router_fp32 {
                ops[0].insert("w_router", self.one(&router)?);
            } else {
                ops[0].insert("w_hilo", self.router_hilo(&router)?);
            }
            ops[0].insert("gate.bias", self.one(&format!("{p}.mlp.gate.e_score_correction_bias"))?);
        }
        Ok(ops.into_iter().map(|operands| MimoLayer { attention, dense, split: ranks > 1, operands }).collect())
    }

    /// Layers `0..layers` (all of them unless the caller stops early): rank 0's
    /// weights (its layer shares, the norm and head), and with a head split each
    /// other rank's layer shares.
    #[allow(clippy::type_complexity)]
    pub fn model(&self, cfg: &MimoV2Config, layers: usize) -> Result<(MimoWeights<'a>, Vec<Vec<MimoLayer<'a>>>)> {
        ensure!(self.tensor("lm_head.weight")?.meta.shape == [cfg.vocab_size, cfg.hidden],
            "lm_head.weight: target head does not match configured [vocab,hidden]");
        let head = if self.fp8_head {
            let (values, scales, _) = self.fp8_projection("lm_head.weight", 1, false)?
                .pop().context("FP8-only target head")?;
            MimoHead::Fp8 { values, scales }
        } else {
            MimoHead::Bf16(self.rows(&["lm_head.weight".to_string()])?)
        };
        let mut shares: Vec<Vec<MimoLayer<'a>>> = (0..self.ranks()).map(|_| Vec::new()).collect();
        for layer in 0..layers.min(cfg.layers) {
            for (share, part) in shares.iter_mut().zip(self.layer(cfg, layer)?) {
                share.push(part);
            }
        }
        let mut shares = shares.into_iter();
        let weights = MimoWeights {
            layers: shares.next().context("rank 0")?,
            norm: self.one("model.norm.weight")?,
            head,
            output_fp8: self.output_formats.values().any(|&format| format == MimoProjectionRepresentation::Fp8),
        };
        crate::shared::memory::staging::release_staging();
        Ok((weights, shares.collect()))
    }
}
