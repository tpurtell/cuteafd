//! How a TP2 layer's two routed halves meet.
//!
//! Fused combine numerics (PLAN section 2): each rank forms
//! `p_r = dtype(FP32(routed_r) + FP32(shared_r))`, the FFN exchange carries
//! `p_r` to the other GPU, and both GPUs compute `BF16(FP32(p_0) + FP32(p_1))`
//! in rank order, so both hold the same bits. Spark reduction instead sums
//! routed planes first and adds shared after; this is a numerics change with
//! its own fidelity gate per format.
use anyhow::Result;
use cuteafd_ffi::{NativeLibrary, RtxPartialDtype, RtxTp2Combine};
use std::ffi::c_void;

/// The dtype of the FFN exchange payload under [`Combine::FusedAllReduce`].
pub(crate) type ExchangeDtype = RtxPartialDtype;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Combine {
    /// HeadSplit / `Whole{ffn: Split}` layers: each rank adds its routed
    /// partial to its shared/dense half, then the existing FFN slot exchange
    /// sums the two in rank order on both GPUs.
    FusedAllReduce { exchange: ExchangeDtype },
    /// `Whole{ffn: Owner}` layers (V4.1): inputs and routes broadcast, routed
    /// then shared reduced in rank order onto the owner, added in BF16.
    OwnerReduce { owner: u8 },
}

impl Combine {
    /// `CUTEAFD_TP2_EXCHANGE=bf16|f32` overrides the fused exchange dtype.
    pub(crate) fn fused_from_env(default: ExchangeDtype) -> Result<Self> {
        let exchange = match std::env::var("CUTEAFD_TP2_EXCHANGE").ok().as_deref() {
            None | Some("") => default,
            Some("bf16") => ExchangeDtype::Bf16,
            Some("f32") => ExchangeDtype::F32,
            Some(other) => anyhow::bail!("CUTEAFD_TP2_EXCHANGE must be bf16 or f32, got {other:?}"),
        };
        Ok(Self::FusedAllReduce { exchange })
    }
}

/// The fused all-reduce's two kernels on one rank.
pub(crate) struct FusedCombine<'a> {
    kernels: RtxTp2Combine<'a>,
    pub exchange: ExchangeDtype,
}

impl<'a> FusedCombine<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, exchange: ExchangeDtype) -> Result<Self> {
        Ok(Self { kernels: library.rtx_tp2_combine()?, exchange })
    }

    /// Bytes of one rank's exchange payload for `rows` rows of `hidden`.
    pub(crate) fn payload_bytes(&self, rows: usize, hidden: usize) -> usize {
        rows * hidden * self.exchange.element_bytes()
    }

    /// This rank's payload: its routed partial (FP32, or null for a layer
    /// whose routed experts are not RTX-resident) plus its shared half.
    /// # Safety
    /// As [`RtxTp2Combine::partial`], on the current device.
    pub(crate) unsafe fn partial(&self, routed: *const f32, shared: *const u16, out: *mut c_void, rows: usize,
        hidden: usize, stream: *mut c_void) -> Result<()> {
        unsafe { self.kernels.partial(routed, shared, out, rows * hidden, self.exchange, stream) }
    }

    /// Both payloads summed in rank order into BF16 `out`.
    /// # Safety
    /// As [`RtxTp2Combine::sum`], on the current device.
    pub(crate) unsafe fn sum(&self, rank0: *const c_void, rank1: *const c_void, out: *mut u16, rows: usize,
        hidden: usize, stream: *mut c_void) -> Result<()> {
        unsafe { self.kernels.sum(rank0, rank1, out, rows * hidden, self.exchange, stream) }
    }
}
