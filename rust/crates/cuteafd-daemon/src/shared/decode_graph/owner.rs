//! Device-scoped executable owner; P retains the resources captured by CUDA.
use crate::shared::memory::device::Device;
use anyhow::{ensure, Result};
use cuteafd_ffi::NativeLibrary;
use std::ffi::c_void;

pub(crate) struct GraphOwner<'a, P> {
    pub raw: *mut c_void,
    device: Device<'a>,
    pub pins: P,
}

impl<'a, P> GraphOwner<'a, P> {
    /// # Safety
    /// raw is exclusively owned, instantiated on device, and every captured
    /// allocation is kept alive by pins or the containing wave. The containing
    /// wave must drain launches before dropping this owner or its external pins.
    pub unsafe fn new(library: &'a NativeLibrary, device: i32, raw: *mut c_void, pins: P) -> Result<Self> {
        ensure!(!raw.is_null(), "null graph executable");
        Ok(Self { raw, device: Device { library, id: device }, pins })
    }
    pub fn device(&self) -> i32 { self.device.id }
}

impl<P> Drop for GraphOwner<'_, P> {
    fn drop(&mut self) {
        // SAFETY: the bank returns retired owners only after the owning streams
        // drain; live entries are dropped by their drained containing wave.
        if let Err(error) = self.device.run(|| unsafe { self.device.library.cuda_graph_exec_destroy(self.raw) }) {
            tracing::error!(%error, device = self.device.id, "destroying graph executable");
        }
    }
}
