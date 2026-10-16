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
pub(crate) mod draft;
pub(crate) mod exl3;
pub(crate) mod fp8moe;
pub(crate) mod native;
pub(crate) mod pair;
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

/// Kernel selection is part of the caller's arithmetic contract, not placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RtxSelectionPolicy {
    FirstCapacity,
    CanonicalDecode,
}

impl RtxSelectionPolicy {
    pub(super) fn execution_index(
        self,
        rows: usize,
        required: usize,
        capacities: impl Iterator<Item = usize>,
        canonical: impl FnOnce() -> Result<usize>,
    ) -> Result<usize> {
        anyhow::ensure!(
            rows > 0 && required >= rows,
            "invalid expert selection rows"
        );
        match self {
            Self::FirstCapacity => capacities
                .enumerate()
                .find(|&(_, capacity)| capacity >= required)
                .map(|(index, _)| index)
                .ok_or_else(|| anyhow::anyhow!("no expert execution for admitted rows")),
            Self::CanonicalDecode => canonical(),
        }
    }
}

/// Shape of a rank partial; dtype alone cannot distinguish token and route sums.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PartialLayout {
    Tokens,
    Routes { topk: usize },
}

/// Element type of a rank's routed partial as its kernels publish it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PartialDtype {
    /// FP32 token sums or route planes, as identified by PartialLayout.
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

/// Resident slot of an absolute backbone ID; gaps are never resident layers.
pub(super) fn resident_index(layers: &[usize], layer: usize) -> Result<usize> {
    layers.binary_search(&layer).map_err(|_| anyhow::anyhow!("TP2 layer {layer} not resident in {layers:?}"))
}

/// One rank's routed half of an exact ordered set of backbone layers.
pub(crate) trait RtxExpertLayer {
    fn shard(&self) -> RtxShard;
    /// Element type of [`Self::output`].
    fn partial(&self) -> PartialDtype;
    fn partial_layout(&self, _rows: usize) -> Result<PartialLayout> {
        Ok(PartialLayout::Tokens)
    }
    /// Strictly increasing absolute backbone layer IDs this rank holds.
    fn layers(&self) -> &[usize];
    /// The device this rank executes on.
    fn device(&self) -> i32;
    /// Exact device bytes of the execution workspace (scratch and the routed
    /// output) for up to `rows` rows; feeds `ExpertCost`.
    fn workspace_bytes(&self) -> usize;
    /// Route grouping of the selected live-row package; zero queries the compiled maximum.
    fn group_rows(&self, _rows: usize) -> Result<usize> {
        anyhow::bail!("expert backend does not publish installed grouping metadata")
    }
    /// One expert's installed FP8-family weight and scale read extent, excluding scratch.
    fn weight_read_bytes(&self) -> Result<usize> {
        anyhow::bail!("expert backend does not publish installed read-byte metadata")
    }
    /// This rank's routed partial of [`Self::partial`] and [`Self::partial_layout`],
    /// valid after the last [`Self::enqueue`] completes on its stream and
    /// until the next one.
    fn output(&self) -> *mut c_void;
    /// Initialize first-use kernels before queuing any peer waits. Native and
    /// EXL3 callers already prewarm their step packages; FP8 primes capacities.
    ///
    /// # Safety
    /// The input/routes contain the backend's maximum initialized row count on
    /// this rank; the startup stream has no pending peer waits and is drained.
    unsafe fn prime(
        &mut self,
        _input: ExpertInput,
        _routes: Routes,
        _stream: *mut c_void,
    ) -> Result<()> {
        Ok(())
    }
    /// This rank's routed sum for `rows` rows of `layer` into [`Self::output`]:
    /// no shared add, no inter-rank reduce, no finish or copy.
    ///
    /// # Safety
    /// `input` and `routes` hold `rows` complete rows on this rank's device,
    /// ordered before `stream` (which belongs to this rank's device) and
    /// unchanged until it drains; the previous use of the output is complete.
    /// On any error, `abort` publishes every associated peer exchange's abort
    /// words before returning (or terminates if publication fails). It must not
    /// wait on a compute stream. The caller then drains all consumers before
    /// releasing storage; a failed drain must terminate, not release owners.
    /// An independent qualification stream with no peer waits may use a no-op.
    unsafe fn enqueue(
        &mut self,
        layer: usize,
        rows: usize,
        input: ExpertInput,
        routes: Routes,
        stream: *mut c_void,
        abort: &mut dyn FnMut(),
    ) -> Result<()>;
}

#[cfg(test)]
mod selection_tests {
    use super::*;

    #[test]
    fn sparse_inventory_maps_absolute_ids_without_filling_gaps() -> Result<()> {
        let ids = [2, 5, 9];
        for (index, layer) in ids.into_iter().enumerate() {
            assert_eq!(resident_index(&ids, layer)?, index);
        }
        for missing in [0, 1, 3, 4, 6, 8, 10] {
            assert!(resident_index(&ids, missing).is_err());
        }
        assert!(resident_index(&[], 0).is_err());
        Ok(())
    }

    #[test]
    fn nearest_capacity_never_consults_canonical_reduction() -> Result<()> {
        for rows in 1..=256 {
            let capacities = [1, 16, 80, 256];
            let index = RtxSelectionPolicy::FirstCapacity.execution_index(
                rows,
                rows,
                capacities.into_iter(),
                || panic!("nearest consulted canonical"),
            )?;
            assert_eq!(
                index,
                capacities
                    .iter()
                    .position(|&capacity| capacity >= rows)
                    .unwrap()
            );
        }
        assert!(RtxSelectionPolicy::FirstCapacity
            .execution_index(0, 0, [1].into_iter(), || Ok(0))
            .is_err());
        assert!(RtxSelectionPolicy::FirstCapacity
            .execution_index(81, 81, [1, 16, 80].into_iter(), || Ok(0))
            .is_err());
        Ok(())
    }

    #[test]
    fn canonical_selection_remains_distinct_from_nearest_capacity() -> Result<()> {
        let variants = [(1, 64), (16, 192), (80, 64), (256, 192)];
        let plan = crate::shared::experts::ExpertReductionPlan::new(&variants)?;
        for rows in 1..=256 {
            let expected = plan.execution_index(rows, rows, variants.iter().map(|v| v.0))?;
            assert_eq!(
                RtxSelectionPolicy::CanonicalDecode.execution_index(
                    rows,
                    rows,
                    variants.iter().map(|v| v.0),
                    || Ok(expected)
                )?,
                expected
            );
        }
        assert_eq!(
            RtxSelectionPolicy::FirstCapacity.execution_index(
                16,
                16,
                variants.iter().map(|v| v.0),
                || Ok(2)
            )?,
            1
        );
        assert_eq!(
            RtxSelectionPolicy::CanonicalDecode.execution_index(
                16,
                16,
                variants.iter().map(|v| v.0),
                || plan.execution_index(16, 16, variants.iter().map(|v| v.0))
            )?,
            2
        );
        Ok(())
    }
}
