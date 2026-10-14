//! TP2 RTX expert combine kernels (`native/shared/cuda/route_reduce.cu`):
//! each rank's FP32 routed partial plus its BF16 shared/dense half, written as
//! the payload the head split's FFN exchange already carries, and the
//! rank-ordered sum of the two payloads. PLAN "v3 placement: design",
//! section 2, "Combine numerics".
use crate::NativeLibrary;
use anyhow::{ensure, Context, Result};
use std::ffi::c_void;

/// Element type of one rank's combined partial on the FFN exchange.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RtxPartialDtype {
    /// `BF16(routed_r + shared_r)`: today's payload; the sum rounds twice.
    Bf16,
    /// `FP32(routed_r + shared_r)`: 2x payload; the sum rounds once.
    F32,
}

impl RtxPartialDtype {
    pub fn element_bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}

type PartialFn = unsafe extern "C" fn(*const f32, *const u16, *mut c_void, u64, u32, *mut c_void) -> i32;
type SumFn = unsafe extern "C" fn(*const c_void, *const c_void, *mut u16, u64, u32, *mut c_void) -> i32;

/// `cuteafd_rtx_tp2_partial_async` / `cuteafd_rtx_tp2_sum_async`.
pub struct RtxTp2Combine<'a> {
    _library: &'a NativeLibrary,
    partial: PartialFn,
    sum: SumFn,
}

impl NativeLibrary {
    pub fn rtx_tp2_combine(&self) -> Result<RtxTp2Combine<'_>> {
        // SAFETY: symbol types match native/shared/include/cuteafd_experts.h.
        unsafe {
            Ok(RtxTp2Combine {
                _library: self,
                partial: *self.lib.get::<PartialFn>(b"cuteafd_rtx_tp2_partial_async")
                    .context("native library lacks cuteafd_rtx_tp2_partial_async (TP2 RTX combine)")?,
                sum: *self.lib.get::<SumFn>(b"cuteafd_rtx_tp2_sum_async")
                    .context("native library lacks cuteafd_rtx_tp2_sum_async (TP2 RTX combine)")?,
            })
        }
    }
}

impl RtxTp2Combine<'_> {
    /// `out[i] = dtype(FP32(routed[i]) + FP32(shared[i]))` over `count`
    /// elements; either input may be null (zero contribution).
    /// # Safety
    /// `routed` FP32 [count], `shared` BF16 [count] and `out` [count] of
    /// `dtype` are live on the current device, disjoint, and ordered on `stream`.
    pub unsafe fn partial(&self, routed: *const f32, shared: *const u16, out: *mut c_void, count: usize,
        dtype: RtxPartialDtype, stream: *mut c_void) -> Result<()> {
        ensure!(count > 0, "empty TP2 combine partial");
        let status = unsafe { (self.partial)(routed, shared, out, count as u64, dtype as u32, stream) };
        ensure!(status == 0, "TP2 combine partial failed with CUDA status {status}");
        Ok(())
    }

    /// `out[i] = BF16(FP32(rank0[i]) + FP32(rank1[i]))`: rank order fixed, so
    /// both GPUs produce the same bits from the same operands.
    /// # Safety
    /// `rank0`/`rank1` [count] of `dtype` and BF16 `out` [count] are live on
    /// the current device (a peer-received slot counts), ordered on `stream`;
    /// `out` does not overlap either input.
    pub unsafe fn sum(&self, rank0: *const c_void, rank1: *const c_void, out: *mut u16, count: usize,
        dtype: RtxPartialDtype, stream: *mut c_void) -> Result<()> {
        ensure!(count > 0, "empty TP2 combine sum");
        let status = unsafe { (self.sum)(rank0, rank1, out, count as u64, dtype as u32, stream) };
        ensure!(status == 0, "TP2 combine sum failed with CUDA status {status}");
        Ok(())
    }
    /// V4 fixed-bucket sum; rows with a negative device slot produce +0 and
    /// never read either rank payload. Other sum consumers use [`Self::sum`].
    /// # Safety
    /// As [`Self::sum`] for `rows * hidden` elements; `main_slots` is live
    /// I64 [rows] on the same device/stream, disjoint from `out`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dsv4_sum_masked(&self, rank0: *const c_void, rank1: *const c_void,
        out: *mut u16, main_slots: *const i64, rows: usize, hidden: usize,
        dtype: RtxPartialDtype, stream: *mut c_void) -> Result<()> {
        ensure!((1..=4096).contains(&rows) && (1..=16384).contains(&hidden),
            "invalid V4 masked sum geometry");
        type MaskedSum = unsafe extern "C" fn(*const c_void, *const c_void, *mut u16,
            *const i64, u32, u32, u32, *mut c_void) -> i32;
        // SAFETY: the new V4-only symbol matches cuteafd_experts.h; shared construction is unchanged.
        let sum = *unsafe { self._library.lib.get::<MaskedSum>(b"cuteafd_dsv4_rtx_tp2_sum_masked_async") }
            .context("native library lacks cuteafd_dsv4_rtx_tp2_sum_masked_async")?;
        // SAFETY: caller retains and orders all device arrays through completion.
        let status = unsafe { sum(rank0, rank1, out, main_slots, rows as u32, hidden as u32, dtype as u32, stream) };
        ensure!(status == 0, "V4 masked TP2 combine sum failed with CUDA status {status}");
        Ok(())
    }

}
