//! Device-scoped executable owner; P retains the resources captured by CUDA.
use crate::shared::memory::device::Device;
use anyhow::{ensure, Result};
use cuteafd_ffi::NativeLibrary;
use std::ffi::c_void;

pub(crate) fn fatal_drain(result: Result<()>, site: &str) {
    if let Err(error) = result {
        tracing::error!(%error, site, "graph storage drain failed; aborting before storage release");
        // AGENTS: drain queued work before publishing or releasing storage.
        // A failed drain cannot prove captured pointers are no longer in use.
        std::process::abort();
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successful_drain_allows_storage_release() { fatal_drain(Ok(()), "fixture"); }

    #[test]
    #[cfg(unix)]
    fn failed_drop_drain_aborts_before_field_release() {
        use std::os::unix::process::ExitStatusExt;
        const CHILD: &str = "CUTEAFD_FATAL_GRAPH_DRAIN_CHILD";
        if std::env::var_os(CHILD).is_some() {
            struct Storage;
            impl Drop for Storage { fn drop(&mut self) { std::process::exit(91); } }
            struct Wave { _storage: Storage }
            impl Drop for Wave {
                fn drop(&mut self) { fatal_drain(Err(anyhow::anyhow!("fake drain failure")), "fixture wave"); }
            }
            drop(Wave { _storage: Storage });
            return;
        }
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command.args(["shared::decode_graph::owner::tests::failed_drop_drain_aborts_before_field_release", "--exact"])
            .env(CHILD, "1");
        let status = command.status().unwrap();
        assert_eq!(status.signal(), Some(libc::SIGABRT));
    }
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
