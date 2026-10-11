//! TP2 RTX routed experts for every family (PLAN "v3 placement: design",
//! section 2). In a 2-RTX layout the routed experts resident on the RTX cards
//! are always TP2 halves: each GPU runs half of every resident expert (split on
//! the intermediate dimension) on its own bitwise-identical copy of the FFN
//! input and routes, and the halves meet in the head split's existing FFN
//! exchange ([`Combine::FusedAllReduce`]) or, for layer-range engines (V4.1),
//! on the owner ([`Combine::OwnerReduce`]).
//!
//! - [`RtxShard`], [`RtxExpertLayer`]: one rank's half of a set of routed
//!   layers and what it computes (no shared add, no inter-rank reduce).
//! - [`native::NativeTp2`]: the native FP8-K32 `rtx_tp2` kernels over any
//!   [`cuteafd_core::ExpertGeometry`] (V4.1, V4 Flash, V4 Pro), moved from
//!   V4.1's `RankWeights`/`RankWave`.
//! - [`exl3::Exl3Tp2`]: EXL3 `rtx-tp2` packages, raw FP32 partials.
//! - [`combine::Combine`]: how the two halves meet; [`combine::FusedCombine`]
//!   runs the fused all-reduce's two kernels.
//! - [`routes::RouteIdentity`]: replicated routers are checked byte-for-byte
//!   at startup, with the broadcast fallback.
use anyhow::Result;
use std::ffi::c_void;

pub(crate) mod combine;
pub(crate) mod exl3;
pub(crate) mod fp8moe;
pub(crate) mod native;
pub(crate) mod routes;

pub(crate) use combine::{Combine, ExchangeDtype, FusedCombine};
pub(crate) use routes::{RouteCheck, RouteIdentity, RouteSource};

/// Which part of each routed expert a resident layer set holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RtxShard {
    /// Complete experts on one GPU (single-RTX layouts).
    Whole,
    /// Half the intermediate dimension of every expert; rank 0 holds the
    /// first half of the gate/up rows and the matching down-projection columns.
    Tp2 { rank: u8 },
}

/// Element type of a rank's routed partial as its kernels publish it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PartialDtype {
    /// FP32 `[rows, hidden]` token sums (native atomic/route-summed, EXL3).
    F32,
    /// BF16 `[rows, hidden]` (FP8 MoE packages today).
    Bf16,
}

/// One rank's expert input: FP8-K32 wire rows (native, EXL3) or BF16 hidden
/// rows (FP8 MoE, W4A4), already on this rank's device.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ExpertInput {
    /// `rows * (hidden + hidden / 32)` bytes: FP8 values then UE8M0 K32 scales.
    Fp8K32(*mut c_void),
    Bf16(*mut c_void),
}

/// Routes for `rows` rows: `[rows, topk]` U32 expert ids and FP32 weights,
/// on the executing rank's device.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Routes {
    pub ids: *mut c_void,
    pub weights: *mut c_void,
}

/// One rank's routed half of a contiguous set of backbone layers.
pub(crate) trait RtxExpertLayer {
    fn shard(&self) -> RtxShard;
    /// Element type of [`Self::output`].
    fn partial(&self) -> PartialDtype;
    /// Backbone layers `first..first + count` this rank holds.
    fn layers(&self) -> std::ops::Range<usize>;
    /// The device this rank executes on.
    fn device(&self) -> i32;
    /// Exact device bytes of the execution workspace (scratch and the routed
    /// output) for up to `rows` rows; feeds `ExpertCost`.
    fn workspace_bytes(&self) -> usize;
    /// This rank's routed partial `[rows, hidden]` of [`Self::partial`],
    /// valid after the last [`Self::enqueue`] completes on its stream and
    /// until the next one.
    fn output(&self) -> *mut c_void;
    /// Initialize first-use kernels before queuing any peer waits. Native and
    /// EXL3 callers already prewarm their step packages; FP8 primes capacities.
    ///
    /// # Safety
    /// The input/routes contain the backend's maximum initialized row count on
    /// this rank; the startup stream has no pending peer waits and is drained.
    unsafe fn prime(&mut self, _input: ExpertInput, _routes: Routes, _stream: *mut c_void) -> Result<()> {
        Ok(())
    }
    /// This rank's routed sum for `rows` rows of `layer` into [`Self::output`]:
    /// no shared add, no inter-rank reduce, no finish or copy.
    ///
    /// # Safety
    /// `input` and `routes` hold `rows` complete rows on this rank's device,
    /// ordered before `stream` (which belongs to this rank's device) and
    /// unchanged until it drains; the previous use of the output is complete.
    unsafe fn enqueue(&mut self, layer: usize, rows: usize, input: ExpertInput, routes: Routes,
        stream: *mut c_void) -> Result<()>;
}
