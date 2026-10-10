//! `NativeTp2`: one rank's half of the native FP8-K32 `rtx_tp2` expert
//! kernels (`cuteafd_{family}_tp2_expert_*`) over the process
//! [`cuteafd_core::ExpertGeometry`] (V4.1, V4 Flash, V4 Pro). Moved from
//! V4.1's `RankWeights`/`RankWave` (`v41_experts/tp2.rs`); per-device kernel
//! handles come from the native library (one module per device).
//!
//! SKELETON (work/v3-p4): the API is fixed; component C2 implements it.
use super::{ExpertInput, PartialDtype, Routes, RtxExpertLayer, RtxShard};
use crate::shared::memory::device::Device;
use anyhow::{bail, Result};
use std::ffi::c_void;
use std::ops::Range;

/// Rank `rank`'s half of backbone layers `layers` on `device`.
pub(crate) struct NativeTp2<'a> {
    device: Device<'a>,
    rank: u8,
    layers: Range<usize>,
}

impl<'a> NativeTp2<'a> {
    /// Exact device bytes of one rank's execution workspace for up to
    /// `max_rows` rows (kernel scratch over the compiled capacities plus the
    /// FP32 `[max_rows, hidden]` output), without loading weights.
    pub(crate) fn workspace_bytes_for(library: &cuteafd_ffi::NativeLibrary, max_rows: usize) -> Result<usize> {
        let _ = (library, max_rows);
        bail!("NativeTp2 is not implemented yet")
    }

    /// Loads rank `rank`'s half of `layers` on `device` within `budget` bytes
    /// (weights + workspace + the largest staging), the same plan as
    /// [`Self::workspace_bytes_for`] and the placement solver's `ExpertCost.half`.
    pub(crate) fn load(device: Device<'a>, catalog: &cuteafd_loader::OfficialV41Catalog, rank: u8,
        layers: Range<usize>, max_rows: usize, budget: usize) -> Result<Self> {
        let _ = (catalog, max_rows, budget);
        let _ = Self { device, rank, layers };
        bail!("NativeTp2 is not implemented yet")
    }
}

impl RtxExpertLayer for NativeTp2<'_> {
    fn shard(&self) -> RtxShard { RtxShard::Tp2 { rank: self.rank } }
    fn partial(&self) -> PartialDtype { PartialDtype::F32 }
    fn layers(&self) -> Range<usize> { self.layers.clone() }
    fn device(&self) -> i32 { self.device.id }
    fn workspace_bytes(&self) -> usize { 0 }
    fn output(&self) -> *mut c_void { std::ptr::null_mut() }
    unsafe fn enqueue(&mut self, _layer: usize, _rows: usize, _input: ExpertInput, _routes: Routes,
        _stream: *mut c_void) -> Result<()> {
        bail!("NativeTp2 is not implemented yet")
    }
}
