//! DFlash2 drafter kernels (GLM 5.3 and GLM 5.3 Flash) (`native/families/glm5/cuda/glm_dflash.cu`) and
//! the cuBLAS BF16 linear its GEMMs use. Every pointer is device memory of
//! the documented shape on the stream's device; the stream orders them.
//! With `fp8_kv`, retained K/V planes use `align16(kv_heads * (head_dim + 4))`
//! bytes per row: E4M3 codes, FP32 head descales, zero alignment padding.
//! Queries and speculative block K/V remain BF16 in either mode.
use crate::NativeLibrary;
use anyhow::{ensure, Result};
use std::ffi::c_void;

type P = *const c_void;
type M = *mut c_void;

fn check(status: i32, what: &str) -> Result<()> {
    ensure!(status == 0, "DFlash2 {what} failed with {status}");
    Ok(())
}

fn i(value: usize) -> Result<i32> {
    Ok(i32::try_from(value)?)
}

impl NativeLibrary {
    /// `out` [rows, n] BF16 = `x` [rows, k] BF16 @ `w`^T (`w` [n, k] BF16), FP32 accumulation.
    ///
    /// # Safety
    /// Pointers are live device buffers of those shapes.
    pub unsafe fn linear_bf16(&self, x: P, w: P, out: M, rows: usize, k: usize, n: usize, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, M, usize, usize, usize, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_cuda_linear_bf16_cublas_async") }?;
        check(unsafe { f(x, w, std::ptr::null(), out, rows, k, n, stream) }, "linear")
    }

    /// Qwen3 RMSNorm of `rows` rows of `width` BF16 values.
    ///
    /// # Safety
    /// As [`Self::linear_bf16`].
    pub unsafe fn glm_dflash_rmsnorm(&self, x: P, w: P, out: M, rows: usize, width: usize, eps: f32, stream: M)
        -> Result<()> {
        type F = unsafe extern "C" fn(P, P, M, i32, i32, f32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_rmsnorm") }?;
        check(unsafe { f(x, w, out, i(rows)?, i(width)?, eps, stream) }, "rmsnorm")
    }

    /// Side-0 dynamic convolution of block rows.
    ///
    /// # Safety
    /// As [`Self::linear_bf16`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glm_dflash_conv(&self, source: P, dynamic: P, base: P, out: M, rows: usize, block_rows: usize,
        hidden: usize, group: usize, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, M, i32, i32, i32, i32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_conv") }?;
        check(unsafe { f(source, dynamic, base, out, i(rows)?, i(block_rows)?, i(hidden)?, i(group)?, stream) }, "conv")
    }

    /// `residual_out = residual + conv1(source)`, `normalized = RMSNorm(residual_out) * w`.
    ///
    /// # Safety
    /// As [`Self::linear_bf16`]; `residual_out` may alias `residual`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glm_dflash_conv_residual_norm(&self, source: P, dynamic: P, base: P, residual: P, w: P,
        residual_out: M, normalized: M, rows: usize, block_rows: usize, hidden: usize, group: usize, eps: f32,
        stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, P, P, M, M, i32, i32, i32, i32, f32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_conv_residual_norm") }?;
        check(unsafe {
            f(source, dynamic, base, residual, w, residual_out, normalized, i(rows)?, i(block_rows)?, i(hidden)?,
                i(group)?, eps, stream)
        }, "conv residual norm")
    }

    /// Per-head RMSNorm + RoPE of `qkv` rows; k/v rows land at `slots[row]`
    /// (row when null, skipped when negative).
    ///
    /// # Safety
    /// As [`Self::linear_bf16`]; `positions` I64 [rows], `slots` I32 [rows].
    /// `head_dim` 128 or 64 selects the entry points (`_d64`: same operands,
    /// 64-wide heads and norm weights).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glm_dflash_qk_rope(&self, qkv: P, q_norm: P, k_norm: P, positions: P, slots: P, q_out: M,
        k_out: M, v_out: M, rows: usize, heads: usize, kv_heads: usize, theta: f32, eps: f32, fp8_kv: bool,
        head_dim: usize, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, P, P, M, M, M, i32, i32, i32, f32, f32, M) -> i32;
        let symbol: &[u8] = match (head_dim, fp8_kv) {
            (128, true) => b"cuteafd_glm_dflash_qk_rope_fp8",
            (128, false) => b"cuteafd_glm_dflash_qk_rope",
            (64, true) => b"cuteafd_glm_dflash_qk_rope_d64_fp8",
            (64, false) => b"cuteafd_glm_dflash_qk_rope_d64",
            _ => anyhow::bail!("DFlash2 qk rope takes 128- or 64-wide heads, not {head_dim}"),
        };
        // SAFETY: every width/format symbol has the native signature above.
        let f = *unsafe { self.lib.get::<F>(symbol) }?;
        check(unsafe {
            f(qkv, q_norm, k_norm, positions, slots, q_out, k_out, v_out, i(rows)?, i(heads)?, i(kv_heads)?, theta,
                eps, stream)
        }, "qk rope")
    }

    /// Workspace bytes of [`Self::glm_dflash_attention`].
    /// `head_dim` 128 or 64 selects the entry point.
    pub fn glm_dflash_attention_workspace(&self, sequences: usize, kv_heads: usize, max_keys: usize,
        head_dim: usize) -> Result<usize> {
        type F = unsafe extern "C" fn(i32, i32, i32) -> u64;
        let symbol: &[u8] = match head_dim {
            128 => b"cuteafd_glm_dflash_attention_workspace",
            64 => b"cuteafd_glm_dflash_attention_workspace_d64",
            _ => anyhow::bail!("DFlash2 attention takes 128- or 64-wide heads, not {head_dim}"),
        };
        let f = *unsafe { self.lib.get::<F>(symbol) }?;
        Ok(usize::try_from(unsafe { f(i(sequences)?, i(kv_heads)?, i(max_keys)?) })?)
    }

    /// Non-causal block attention over each sequence's ring context and block;
    /// `window` > 0 limits each block row to context positions within it.
    ///
    /// # Safety
    /// As [`Self::linear_bf16`]; the I32 tables hold one entry per sequence.
    /// `head_dim` 128 or 64 selects the entry points (`_d64`: identical
    /// operands, 64-wide heads).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glm_dflash_attention(&self, q: P, k_block: P, v_block: P, k_ring: P, v_ring: P, seq_slots: P,
        ctx_lengths: P, ctx_ends: P, out: M, workspace: M, sequences: usize, block_rows: usize, heads: usize,
        kv_heads: usize, ring: usize, max_keys: usize, window: usize, scale: f32, fp8_kv: bool, head_dim: usize,
        stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, P, P, P, P, P, M, M, i32, i32, i32, i32, i32, i32, i32, f32, M) -> i32;
        let symbol: &[u8] = match (head_dim, fp8_kv) {
            (128, true) => b"cuteafd_glm_dflash_attention_fp8",
            (128, false) => b"cuteafd_glm_dflash_attention",
            (64, true) => b"cuteafd_glm_dflash_attention_d64_fp8",
            (64, false) => b"cuteafd_glm_dflash_attention_d64",
            _ => anyhow::bail!("DFlash2 attention takes 128- or 64-wide heads, not {head_dim}"),
        };
        // SAFETY: every width/format symbol has the native signature above.
        let f = *unsafe { self.lib.get::<F>(symbol) }?;
        check(unsafe {
            f(q, k_block, v_block, k_ring, v_ring, seq_slots, ctx_lengths, ctx_ends, out, workspace, i(sequences)?,
                i(block_rows)?, i(heads)?, i(kv_heads)?, i(ring)?, i(max_keys)?, i(window)?, scale, stream)
        }, "attention")
    }

    /// `out` [rows, inter] = silu(gate) * up of `gate_up` [rows, 2 * inter].
    ///
    /// # Safety
    /// As [`Self::linear_bf16`].
    pub unsafe fn glm_dflash_silu_mul(&self, gate_up: P, out: M, rows: usize, inter: usize, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, M, i32, i32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_silu_mul") }?;
        check(unsafe { f(gate_up, out, i(rows)?, i(inter)?, stream) }, "silu mul")
    }

    /// Workspace bytes of [`Self::glm_dflash_topk`] for `rows` drafted rows.
    pub fn glm_dflash_topk_workspace(&self, rows: usize) -> Result<usize> {
        type F = unsafe extern "C" fn(i32) -> u64;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_topk_workspace") }?;
        Ok(usize::try_from(unsafe { f(i(rows)?) })?)
    }

    /// Top-16 (F32 values of BF16-rounded logits, I32 ids) of block rows
    /// 1..=drafts of each sequence.
    ///
    /// # Safety
    /// As [`Self::linear_bf16`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glm_dflash_topk(&self, logits: P, unary: M, candidates: M, workspace: M, sequences: usize,
        block_rows: usize, drafts: usize, vocab: usize, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, M, M, M, i32, i32, i32, i32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_topk") }?;
        check(unsafe {
            f(logits, unary, candidates, workspace, i(sequences)?, i(block_rows)?, i(drafts)?, i(vocab)?, stream)
        }, "top-k")
    }

    /// Greedy candidate-selector walk: tokens [sequences, drafts] U32 and
    /// features [sequences, drafts, 4] F32.
    ///
    /// # Safety
    /// As [`Self::linear_bf16`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glm_dflash_select(&self, pred_cb: P, succ_cb: P, projected: P, candidates: P, unary: P,
        anchors: P, tokens: M, features: M, sequences: usize, block_rows: usize, drafts: usize, rank: usize,
        stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, P, P, P, P, P, M, M, i32, i32, i32, i32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_select") }?;
        check(unsafe {
            f(pred_cb, succ_cb, projected, candidates, unary, anchors, tokens, features, i(sequences)?,
                i(block_rows)?, i(drafts)?, i(rank)?, stream)
        }, "select")
    }

    /// `dst[row, offset..offset + width] = src[row]` for BF16 rows.
    ///
    /// # Safety
    /// As [`Self::linear_bf16`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glm_dflash_tap(&self, src: P, dst: M, rows: usize, width: usize, stride: usize, offset: usize,
        stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, M, i32, i32, i32, i32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_tap") }?;
        check(unsafe { f(src, dst, i(rows)?, i(width)?, i(stride)?, i(offset)?, stream) }, "tap")
    }

    /// `dst[row, offset..offset + width] = bf16(mean of the row's hc streams)`
    /// for `src` [rows, hc, width] BF16 (the mHC contraction DFlash2 taps).
    ///
    /// # Safety
    /// As [`Self::linear_bf16`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn glm_dflash_tap_mean(&self, src: P, dst: M, rows: usize, width: usize, hc: usize, stride: usize,
        offset: usize, stream: M) -> Result<()> {
        type F = unsafe extern "C" fn(P, M, i32, i32, i32, i32, i32, M) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_glm_dflash_tap_mean") }?;
        check(unsafe { f(src, dst, i(rows)?, i(width)?, i(hc)?, i(stride)?, i(offset)?, stream) }, "tap mean")
    }
}
