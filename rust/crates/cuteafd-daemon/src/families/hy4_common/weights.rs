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

/// Per-layer decode L2 prefetch ranges for Hy4 / Hy4-flash.
///
/// Prioritizes the three large attention projections (`q_b_proj`, `linear_gate`,
/// `o_proj`) in execution order before smaller attention, indexer, router and
/// FFN operands:
/// - On 1x RTX (128 MiB L2, 96 MiB auto budget) Hy4 K8 `q_b_proj` (32 MiB) and
///   `linear_gate` (64 of 96 MiB) warm without being evicted before use; putting
///   `o_proj` ahead of `linear_gate` would let `linear_gate`'s 96 MiB sweep evict
///   it before `o_proj` launches.
/// - On 2x RTX (TP2) and Hy4-flash, `q_b_proj` + `linear_gate` leave room in the
///   runtime L2 budget to warm `o_proj` (and remaining layer operands) as well.
pub(crate) fn layer_prefetch_ranges(
    layer: usize,
    extra_after_attention: &[&str],
    mut resolve: impl FnMut(&str) -> Vec<crate::shared::l2_prefetch::Range>,
) -> Vec<crate::shared::l2_prefetch::Range> {
    let p = format!("model.layers.{layer}");
    let mut out = Vec::new();
    for suffix in [
        "self_attn.q_b_proj",
        "self_attn.linear_gate",
        "self_attn.o_proj",
        "self_attn.q_a_proj",
        "self_attn.kv_a_proj_with_mqa",
        "self_attn.kv_b_proj",
    ] {
        out.extend(resolve(&format!("{p}.{suffix}")));
    }
    for suffix in extra_after_attention {
        out.extend(resolve(&format!("{p}.{suffix}")));
    }
    for suffix in [
        "mlp.gate",
        "mlp.shared_experts.gate_proj",
        "mlp.shared_experts.up_proj",
        "mlp.shared_experts.down_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ] {
        out.extend(resolve(&format!("{p}.{suffix}")));
    }
    out
}

/// Final-layer decode L2 prefetch ranges: `hc_head_fn` (when resident on this
/// rank) followed by the vocabulary head (`lm_head`).
pub(crate) fn head_prefetch_ranges(
    mut resolve: impl FnMut(&str) -> Vec<crate::shared::l2_prefetch::Range>,
) -> Vec<crate::shared::l2_prefetch::Range> {
    let mut out = Vec::new();
    out.extend(resolve("model.hc_head.hc_head_fn"));
    out.extend(resolve("lm_head"));
    out
}

/// Build the full `0..layers` decode L2 prefetch schedule: after layer `i`'s
/// Spark exchange, warm layer `i + 1` or, after the final layer, the head.
pub(crate) fn decode_prefetch_order(
    layers: usize,
    extra_after_attention: &[&str],
    mut resolve: impl FnMut(&str) -> Vec<crate::shared::l2_prefetch::Range>,
) -> Vec<Vec<crate::shared::l2_prefetch::Range>> {
    (0..layers)
        .map(|i| {
            if i + 1 < layers {
                layer_prefetch_ranges(i + 1, extra_after_attention, &mut resolve)
            } else {
                head_prefetch_ranges(&mut resolve)
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::l2_prefetch::{prefix, Range};
    use std::ffi::c_void;

    fn at(offset: usize) -> *const c_void {
        (0x2000_0000usize + offset) as *const c_void
    }

    #[test]
    fn decode_prefetch_order_prioritizes_qb_gate_o_and_head() {
        let mib = 1usize << 20;
        let resolve = |stem: &str| -> Vec<Range> {
            match stem {
                "model.layers.1.self_attn.q_b_proj" => vec![(at(1), 32 * mib)],
                "model.layers.1.self_attn.linear_gate" => vec![(at(2), 96 * mib)],
                "model.layers.1.self_attn.o_proj" => vec![(at(3), 96 * mib)],
                "model.layers.1.self_attn.q_a_proj" => vec![(at(4), 12 * mib)],
                "model.layers.1.self_attn.kv_b_proj" => vec![(at(5), 28 * mib)],
                "model.hc_head.hc_head_fn" => vec![(at(6), 384 << 10)],
                "lm_head" => vec![(at(7), 708 * mib)],
                _ => Vec::new(),
            }
        };
        let order = decode_prefetch_order(2, &[], resolve);
        assert_eq!(order.len(), 2);
        // After layer 0 -> layer 1: 96 MiB budget takes all of q_b (32 MiB) + 64 MiB of linear_gate.
        assert_eq!(prefix(&order[0], 96 * mib), vec![(at(1), 32 * mib), (at(2), 64 * mib)]);
        // After layer 1 (last) -> hc_head_fn (384 KiB) + remainder of lm_head.
        assert_eq!(
            prefix(&order[1], 96 * mib),
            vec![(at(6), 384 << 10), (at(7), 96 * mib - (384 << 10))]
        );
    }

    #[test]
    fn decode_prefetch_order_reaches_o_proj_on_tp2_and_flash() {
        let mib = 1usize << 20;
        let resolve = |stem: &str| -> Vec<Range> {
            match stem {
                "model.layers.1.self_attn.q_b_proj" => vec![(at(1), 16 * mib)],
                "model.layers.1.self_attn.linear_gate" => vec![(at(2), 48 * mib)],
                "model.layers.1.self_attn.o_proj" => vec![(at(3), 48 * mib)],
                _ => Vec::new(),
            }
        };
        let order = decode_prefetch_order(2, &[], resolve);
        assert_eq!(
            prefix(&order[0], 96 * mib),
            vec![(at(1), 16 * mib), (at(2), 48 * mib), (at(3), 32 * mib)]
        );
    }
}
