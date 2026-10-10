//! `Exl3Tp2`: one rank's half of an EXL3 checkpoint's routed layers through
//! the `exl3-<family>-k<tiers>/rtx-tp2` package (`residency(sel, 2, rank)` +
//! `launch_layer_into`), raw FP32 partials: no wire re-quantization, no
//! `reducer.finish`, no copy. Moved from V4.1's `RankWeights::load_exl3_pair`.
//!
//! SKELETON (work/v3-p4): the API is fixed; component C2 implements it.
use super::{ExpertInput, PartialDtype, Routes, RtxExpertLayer, RtxShard};
use crate::shared::memory::device::Device;
use anyhow::{bail, Result};
use std::ffi::c_void;
use std::ops::Range;
use std::path::Path;

pub(crate) struct Exl3Tp2<'a> {
    device: Device<'a>,
    rank: u8,
    layers: Range<usize>,
}

impl<'a> Exl3Tp2<'a> {
    /// Exact device bytes of one rank's execution workspace for up to
    /// `max_rows` rows from the package manifests in `package` (the
    /// `rtx-tp2` directory), plus the FP32 output.
    pub(crate) fn workspace_bytes_for(package: &Path, hidden: usize, max_rows: usize) -> Result<usize> {
        let _ = (package, hidden, max_rows);
        bail!("Exl3Tp2 is not implemented yet")
    }

    /// Loads both ranks' halves of `layers`, layer by layer (rank 0 then rank
    /// 1, so rank 1 reuses source pages rank 0 just read), within each rank's
    /// budget. Partial loads release every allocation on its own device.
    pub(crate) fn load_pair(devices: [Device<'a>; 2], catalog: &cuteafd_loader::OfficialV41Catalog,
        package: &Path, layers: Range<usize>, max_rows: usize, budgets: [usize; 2]) -> Result<[Self; 2]> {
        let _ = (catalog, package, max_rows, budgets);
        let _ = devices.map(|device| Self { device, rank: 0, layers: layers.clone() });
        bail!("Exl3Tp2 is not implemented yet")
    }
}

impl RtxExpertLayer for Exl3Tp2<'_> {
    fn shard(&self) -> RtxShard { RtxShard::Tp2 { rank: self.rank } }
    fn partial(&self) -> PartialDtype { PartialDtype::F32 }
    fn layers(&self) -> Range<usize> { self.layers.clone() }
    fn device(&self) -> i32 { self.device.id }
    fn workspace_bytes(&self) -> usize { 0 }
    fn output(&self) -> *mut c_void { std::ptr::null_mut() }
    unsafe fn enqueue(&mut self, _layer: usize, _rows: usize, _input: ExpertInput, _routes: Routes,
        _stream: *mut c_void) -> Result<()> {
        bail!("Exl3Tp2 is not implemented yet")
    }
}
