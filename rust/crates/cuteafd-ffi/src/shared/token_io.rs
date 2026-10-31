//! Device token input and output (`native/shared/cuda/token_io.cu`): BF16
//! embedding rows gathered by device token ids, and greedy selection (with an
//! optional log-probability) from FP32 logits rows.
use crate::NativeLibrary;
use anyhow::{ensure, Result};
use std::ffi::c_void;

impl NativeLibrary {
    /// Writes `copies` contiguous copies of `table[ids[r]]` (BF16 `[vocab,
    /// hidden]`; `table[ids[index[r]]]` with a non-null `index`) for each of
    /// `rows` rows into `out` (`[rows * copies, hidden]`); an id `>= vocab`
    /// copies `fallback` (one BF16 row) or zeros without it.
    ///
    /// # Safety
    /// Every pointer is live device memory of those shapes on `stream`'s
    /// device; `ids` holds `rows` U32 ids by the time the stream reaches the
    /// kernel.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_embed_gather_bf16_async(&self, table: *const c_void, vocab: usize, hidden: usize,
        ids: *const c_void, index: *const c_void, rows: usize, copies: usize, fallback: *const c_void,
        out: *mut c_void, stream: *mut c_void) -> Result<()> {
        type F = unsafe extern "C" fn(*const u16, usize, usize, *const u32, *const u32, usize, usize, *const u16,
            *mut u16, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_cuda_embed_gather_bf16_async") }?;
        // SAFETY: the caller's contract.
        let status = unsafe { f(table.cast(), vocab, hidden, ids.cast(), index.cast(), rows, copies, fallback.cast(),
            out.cast(), stream) };
        ensure!(status == 0, "embedding gather of {rows} rows x {copies} (hidden {hidden}) failed with {status}");
        Ok(())
    }

    /// Greedy token of each of `rows` FP32 logits rows (`stride` floats apart):
    /// the lowest id among the largest values, NaN never chosen. `status[r]`
    /// is 1 when row `r` holds a non-finite logit; `logprob` (nullable) gets
    /// `log_softmax(row)[id]`.
    ///
    /// # Safety
    /// Every pointer is live device memory of those shapes on `stream`'s device.
    /// Final-logit soft cap in place over `rows` FP32 rows (`stride` floats
    /// apart): `x = tanh(x / cap) * cap` in FP32, the reference's arithmetic.
    ///
    /// # Safety
    /// `logits` is live device memory of `rows * stride` floats on `stream`'s device.
    pub unsafe fn cuda_logits_softcap_f32_async(&self, logits: *mut c_void, rows: usize, vocab: usize, stride: usize,
        cap: f32, stream: *mut c_void) -> Result<()> {
        type F = unsafe extern "C" fn(*mut f32, usize, usize, usize, f32, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_cuda_logits_softcap_f32_async") }?;
        // SAFETY: the caller's contract.
        let code = unsafe { f(logits.cast(), rows, vocab, stride, cap, stream) };
        ensure!(code == 0, "logit soft cap of {rows} rows x {vocab} failed with {code}");
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cuda_logits_greedy_f32_async(&self, logits: *const c_void, rows: usize, vocab: usize, stride: usize,
        ids: *mut c_void, logprob: *mut c_void, status: *mut c_void, stream: *mut c_void) -> Result<()> {
        type F = unsafe extern "C" fn(*const f32, usize, usize, usize, *mut u32, *mut f32, *mut u32, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_cuda_logits_greedy_f32_async") }?;
        // SAFETY: the caller's contract.
        let code = unsafe { f(logits.cast(), rows, vocab, stride, ids.cast(), logprob.cast(), status.cast(), stream) };
        ensure!(code == 0, "greedy selection of {rows} rows x {vocab} failed with {code}");
        Ok(())
    }
}
