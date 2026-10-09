//! Vocabulary head for a few rows (`native/shared/cuda/vocab_head_rows.cu`):
//! FP32-accumulated BF16 projection that reads the head once per 8 rows.
use crate::NativeLibrary;
use anyhow::{ensure, Result};
use std::ffi::c_void;

/// Most rows [`NativeLibrary::vocab_head_rows`] is used for (three passes;
/// wider heads stay on cuBLAS.
pub const VOCAB_HEAD_ROWS_MAX: usize = 24;

/// Rows one pass of [`NativeLibrary::vocab_head_rows`] takes (one read of the
/// head; `kMaxRows` in `vocab_head_rows.cu`).
pub const VOCAB_HEAD_ROWS_PASS: usize = 8;

impl NativeLibrary {
    /// `logits` [rows, vocab] FP32 = `x` [rows, width] BF16 times `weight`
    /// [vocab, width]^T BF16, FP32 products and accumulation, in passes of 8
    /// rows. `width` is a multiple of 256 and at most 6336.
    ///
    /// # Safety
    /// Every pointer is live device memory of those shapes on the stream's
    /// device, 16-byte aligned.
    pub unsafe fn vocab_head_rows(&self, x: *const c_void, weight: *const c_void, logits: *mut f32, rows: usize,
        width: usize, vocab: usize, stream: *mut c_void) -> Result<()> {
        type F = unsafe extern "C" fn(*const c_void, *const c_void, *mut f32, i32, i32, i32, *mut c_void) -> i32;
        let f = *unsafe { self.lib.get::<F>(b"cuteafd_vocab_head_rows") }?;
        let status = unsafe {
            f(x, weight, logits, i32::try_from(rows)?, i32::try_from(width)?, i32::try_from(vocab)?, stream)
        };
        ensure!(status == 0, "vocabulary head of {rows} rows failed with {status}");
        Ok(())
    }
}
