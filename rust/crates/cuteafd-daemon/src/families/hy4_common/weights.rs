//! Shared weight, buffer-slicing, and RoPE table helpers for `hy4` and `hy4_flash`.
use crate::shared::experts::rtx::dense::Rect;
use anyhow::{bail, ensure, Context, Result};
use cuteafd_core::DType;
use cuteafd_ffi::CuteafdDeviceBuffer;
use cuteafd_loader::formats::trellis_core::CoreSplit;
use cuteafd_loader::plan::checkpoint::Checkpoint;
use std::fs::File;
use std::ops::Range;
use std::os::unix::fs::FileExt;

pub(crate) fn slice_buf(
    buf: CuteafdDeviceBuffer,
    offset: usize,
    bytes: usize,
) -> Result<CuteafdDeviceBuffer> {
    ensure!(
        bytes > 0 && offset.checked_add(bytes).is_some_and(|end| end <= buf.bytes),
        "Hy4 buffer slice [{offset}..+{bytes}] exceeds allocation {}",
        buf.bytes
    );
    Ok(CuteafdDeviceBuffer {
        ptr: buf.ptr.cast::<u8>().wrapping_add(offset).cast(),
        bytes,
        ..buf
    })
}

pub(crate) fn read_tensor_bytes(
    checkpoint: &Checkpoint,
    name: &str,
) -> Result<(Vec<u8>, DType, Vec<usize>)> {
    let entry = checkpoint
        .tensors
        .iter()
        .find(|t| t.meta.name == name)
        .with_context(|| format!("Hy4 checkpoint missing tensor {name}"))?;
    let file = File::open(checkpoint.snapshot.join(&entry.shard))
        .with_context(|| format!("opening shard {} for {name}", entry.shard))?;
    let mut bytes = vec![0u8; usize::try_from(entry.meta.byte_length)?];
    file.read_exact_at(&mut bytes, entry.meta.byte_offset)?;
    Ok((bytes, entry.meta.dtype.clone(), entry.meta.shape.clone()))
}

pub(crate) fn dense_rect(
    checkpoint: &Checkpoint,
    name: &str,
    rank: usize,
    gpus: usize,
    split_fn: fn(&str) -> CoreSplit,
) -> Result<Rect> {
    let stem = name.strip_suffix(".weight").unwrap_or(name);
    let tensor = if checkpoint.contains_tensor(name) {
        name.to_owned()
    } else {
        format!("{name}.weight")
    };
    let header = checkpoint
        .header(&tensor)
        .with_context(|| format!("Hy4 checkpoint missing {tensor}"))?;
    let element = match header.dtype {
        DType::Bf16 | DType::F16 => 2,
        DType::F32 => 4,
        ref other => bail!("Hy4 unsupported target tensor dtype {other:?} for {tensor}"),
    };
    let share = |n: usize| -> Result<Range<usize>> {
        ensure!(n % gpus == 0, "Hy4 {tensor} cannot split extent {n} across {gpus} ranks");
        let width = n / gpus;
        Ok(rank * width..(rank + 1) * width)
    };
    let (rows, cols) = if stem.ends_with(".self_attn.learnable_sink_param") {
        ensure!(header.dtype == DType::F32 && header.shape.len() == 1, "bad sink shape for {tensor}");
        (None, Some(share(header.shape[0] * element)?))
    } else if gpus == 2 && header.shape.len() == 2 {
        let split = if stem == "lm_head" { CoreSplit::Output } else { split_fn(stem) };
        match split {
            CoreSplit::Output => (Some(share(header.shape[0])?), None),
            CoreSplit::Input => (None, Some(share(header.shape[1] * element)?)),
            CoreSplit::Whole => (None, None),
        }
    } else {
        (None, None)
    };
    Rect::tensor(checkpoint, &tensor, rows, cols)
}

pub(crate) fn pack_kv_b_absorbed(
    raw: &[u8],
    local_heads: usize,
    nope: usize,
    v_dim: usize,
    kv_lora: usize,
) -> Result<Vec<u8>> {
    let head_rows = nope + v_dim;
    ensure!(
        raw.len() == local_heads * head_rows * kv_lora * 2,
        "kv_b_proj raw length {} != expected {}",
        raw.len(),
        local_heads * head_rows * kv_lora * 2
    );
    let uk_bytes = local_heads * kv_lora * nope * 2;
    let uv_bytes = local_heads * v_dim * kv_lora * 2;
    let mut out = vec![0u8; uk_bytes + uv_bytes];
    for h in 0..local_heads {
        let src_h = h * head_rows * kv_lora * 2;
        let dst_uk = h * kv_lora * nope * 2;
        for d in 0..nope {
            for c in 0..kv_lora {
                let s = src_h + (d * kv_lora + c) * 2;
                let t = dst_uk + (c * nope + d) * 2;
                out[t] = raw[s];
                out[t + 1] = raw[s + 1];
            }
        }
        let src_v = src_h + nope * kv_lora * 2;
        let dst_uv = uk_bytes + h * v_dim * kv_lora * 2;
        let v_span = v_dim * kv_lora * 2;
        out[dst_uv..dst_uv + v_span].copy_from_slice(&raw[src_v..src_v + v_span]);
    }
    Ok(out)
}

pub(crate) fn inv_freq_bytes(rope_dim: usize, theta: f64) -> Vec<u8> {
    let half = rope_dim / 2;
    let mut bytes = Vec::with_capacity(half * 4);
    for i in 0..half {
        let freq = (1.0f64 / theta.powf((2 * i) as f64 / rope_dim as f64)) as f32;
        bytes.extend_from_slice(&freq.to_le_bytes());
    }
    bytes
}

/// Return `(w_uk [local_heads, kv_lora, nope], w_uv [local_heads, v_dim, kv_lora])` pointers inside `kv_b_proj`.
pub(crate) fn kv_b_absorbed_ptrs(
    buf: CuteafdDeviceBuffer,
    local_heads: usize,
    nope: usize,
    v_dim: usize,
    kv_lora: usize,
) -> Result<(*mut std::ffi::c_void, *mut std::ffi::c_void)> {
    let uk_bytes = local_heads * kv_lora * nope * 2;
    let uv_bytes = local_heads * v_dim * kv_lora * 2;
    ensure!(buf.bytes == uk_bytes + uv_bytes, "kv_b_proj absorbed view size mismatch");
    Ok((buf.ptr, buf.ptr.cast::<u8>().wrapping_add(uk_bytes).cast()))
}
