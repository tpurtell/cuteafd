//! Uninstantiated CUDA sources and transferred executable references.
use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphUpdateResult {
    pub cuda_status: i32,
    pub update_result: i32,
}
impl GraphUpdateResult {
    // Every driver error and every non-success CUDA update enum is rejected.
    pub fn accepted(self) -> bool { self.cuda_status == 0 && self.update_result == 0 }
}

pub struct GraphSource<'a> {
    library: &'a NativeLibrary,
    raw: *mut c_void,
    pub nodes: usize,
    pub width: usize,
    pub spans_devices: bool,
    pub signature: Vec<u64>,
    pub inspection_error: Option<String>,
}

impl NativeLibrary {
    /// # Safety
    /// The caller exclusively owns the active stream capture and all operands.
    pub unsafe fn cuda_graph_capture_source(&self, stream: *mut c_void) -> Result<GraphSource<'_>> {
        type End = unsafe extern "C" fn(*mut c_void, *mut *mut c_void, *mut usize,
            *mut usize, *mut i32, *mut *const u64, *mut usize) -> CuteafdStatus;
        // SAFETY: the fixed export ABI is owned by this loaded module.
        let end = unsafe { self.lib.get::<End>(b"cuteafd_cuda_graph_capture_source")? };
        let (mut raw, mut nodes, mut width, mut spans, mut signature, mut words) =
            (std::ptr::null_mut(), 0, 0, 0, std::ptr::null(), 0);
        // SAFETY: caller owns capture and all output pointers are valid locals.
        let status = unsafe { end(stream, &mut raw, &mut nodes, &mut width, &mut spans, &mut signature, &mut words) };
        if status == CUTEAFD_STATUS_GRAPH_TOO_SMALL { return Err(SmallGraphCapture { nodes }.into()) }
        self.status_to_result("cuda_graph_capture_source", status)?;
        // Read immediately: later instantiate/update calls overwrite native TLS.
        let inspection_error = (width == 0).then(|| self.last_error().unwrap_or_default());
        let signature = if words == 0 { Vec::new() } else {
            anyhow::ensure!(!signature.is_null(), "null graph topology signature");
            // SAFETY: source owns these words until its destruction below.
            unsafe { std::slice::from_raw_parts(signature, words) }.to_vec()
        };
        Ok(GraphSource { library: self, raw, nodes, width, spans_devices: spans != 0, signature, inspection_error })
    }

    /// Invalidates a cache-owned handle before removing its cache identity.
    pub fn cuda_graph_exec_invalidate(&self, exec: usize) -> Result<()> {
        type Invalidate = unsafe extern "C" fn(*mut c_void) -> CuteafdStatus;
        // SAFETY: fixed ABI; native validates the opaque handle against its registry.
        let invalidate = unsafe { self.lib.get::<Invalidate>(b"cuteafd_cuda_graph_exec_invalidate")? };
        // SAFETY: native validates without dereferencing unregistered handles.
        let status = unsafe { invalidate(exec as *mut c_void) };
        self.status_to_result("invalidate graph executable", status)
    }

    /// Asynchronously uploads an executable on `stream` so its next launch
    /// does not pay an implicit first-launch upload stall on the host.
    ///
    /// # Safety
    /// `exec` is a live executable owned on the current executor thread and
    /// `stream` is a live CUDA stream (or null for stream 0) on the same device.
    pub unsafe fn cuda_graph_upload(&self, exec: *mut c_void, stream: *mut c_void) -> Result<()> {
        type Upload = unsafe extern "C" fn(*mut c_void, *mut c_void) -> CuteafdStatus;
        // SAFETY: fixed ABI owned by the retained native module.
        let upload = unsafe { self.lib.get::<Upload>(b"cuteafd_cuda_graph_upload")? };
        // SAFETY: native validates tagged handles without dereferencing stale ones.
        let status = unsafe { upload(exec, stream) };
        self.status_to_result("cuteafd_cuda_graph_upload", status)
    }

}

impl GraphSource<'_> {
    /// # Safety
    /// The resulting executable's operands remain live through all consumers.
    pub fn instantiate(&self, stream: *mut c_void) -> Result<*mut c_void> {
        type Instantiate = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut *mut c_void) -> CuteafdStatus;
        let _failure_boundary = memory_ledger::fault::boundary();
        let _budget_guard = coordinator_gpu_budget().map(|_| GPU_BUDGET_ALLOCATION.lock().unwrap_or_else(|p| p.into_inner()));
        // SAFETY: fixed export ABI; this source keeps the module loaded.
        let instantiate = unsafe { self.library.lib.get::<Instantiate>(b"cuteafd_cuda_graph_source_instantiate")? };
        let mut exec = std::ptr::null_mut();
        // SAFETY: source and stream are live, and output is a local pointer.
        let status = unsafe { instantiate(self.raw, stream, &mut exec) };
        self.library.status_to_result("cuda_graph_source_instantiate", status)?;
        if coordinator_gpu_budget().is_some() {
            if let Err(error) = self.library.cuda_memory_info() {
                // SAFETY: this new executable has never launched.
                unsafe { self.library.cuda_graph_exec_destroy(exec)?; }
                return Err(error.context("graph executable exceeds coordinator GPU budget"));
            }
        }
        Ok(exec)
    }

    pub fn transfer(&self, donor: usize) -> Result<usize> {
        type Transfer = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut *mut c_void) -> CuteafdStatus;
        // SAFETY: fixed ABI; self owns source and native validates donor handle.
        let transfer = unsafe { self.library.lib.get::<Transfer>(b"cuteafd_cuda_graph_source_transfer")? };
        let mut handle = std::ptr::null_mut();
        // SAFETY: source remains live and output is a valid local pointer.
        let status = unsafe { transfer(self.raw, donor as *mut c_void, &mut handle) };
        self.library.status_to_result("transfer graph executable", status)?;
        Ok(handle as usize)
    }

    /// # Safety
    /// The caller holds the cache lock and proves the donor has no queued use.
    pub fn update(&self, exec: usize) -> Result<GraphUpdateResult> {
        type Update = unsafe extern "C" fn(*mut c_void, *mut c_void, *mut i32, *mut i32) -> CuteafdStatus;
        anyhow::ensure!(!self.signature.is_empty(), "only all-kernel topologies may transfer");
        // SAFETY: fixed export ABI; self retains source and module.
        let update = unsafe { self.library.lib.get::<Update>(b"cuteafd_cuda_graph_source_update")? };
        let (mut cuda_status, mut update_result) = (-1, -1);
        // SAFETY: exclusive donor and live source, valid result pointers.
        let status = unsafe { update(self.raw, exec as *mut c_void, &mut cuda_status, &mut update_result) };
        self.library.status_to_result("cuda_graph_source_update", status)?;
        Ok(GraphUpdateResult { cuda_status, update_result })
    }
}
impl Drop for GraphSource<'_> {
    fn drop(&mut self) {
        type Destroy = unsafe extern "C" fn(*mut c_void) -> CuteafdStatus;
        // SAFETY: fixed export ABI; only this owner releases its source.
        if let Ok(destroy) = unsafe { self.library.lib.get::<Destroy>(b"cuteafd_cuda_graph_source_destroy") } {
            // SAFETY: raw is the exclusively owned, uninstantiated source graph.
            let status = unsafe { destroy(self.raw) };
            if let Err(error) = self.library.status_to_result("cuda_graph_source_destroy", status) {
                tracing::error!(%error, "destroying graph source");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_non_success_update_result_is_rejected() {
        for cuda_status in [0, 1, 2, 910] {
            for update_result in -1..=16 {
                assert_eq!(GraphUpdateResult { cuda_status, update_result }.accepted(),
                    cuda_status == 0 && update_result == 0);
            }
        }
    }
    #[test]
    fn transferred_handle_cannot_launch_or_destroy_the_new_owners_exec() -> Result<()> {
        let fixture = crate::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let library = fixture.load()?;
        fixture.configure_memory_usage(&library, 0, 64 << 20, 0)?;
        // SAFETY: CPU fixture has no CUDA work or operand pointers.
        let source = unsafe { library.cuda_graph_capture_source(std::ptr::null_mut())? };
        let donor = source.instantiate(std::ptr::null_mut())?;
        // SAFETY: fixture submits no device work.
        unsafe { library.cuda_graph_launch(donor, std::ptr::null_mut())?; }
        let before = library.cuda_memory_info()?;
        assert!(source.update(donor as usize)?.accepted());
        let receiver = source.transfer(donor as usize)? as *mut c_void;
        assert_ne!(receiver, donor);
        // SAFETY: fixture handles are valid; donor must reject, receiver survives.
        unsafe {
            assert!(library.cuda_graph_upload(donor, std::ptr::null_mut()).is_err());
            assert!(library.cuda_graph_launch(donor, std::ptr::null_mut()).is_err());
            library.cuda_graph_exec_destroy(donor)?;
            library.cuda_graph_upload(receiver, std::ptr::null_mut())?;
            library.cuda_graph_upload(receiver, std::ptr::null_mut())?;
            library.cuda_graph_launch(receiver, std::ptr::null_mut())?;
        }
        assert_eq!(fixture.events()?, "GuLUguL");
        assert_eq!(before, library.cuda_memory_info()?);
        // SAFETY: receiver exclusively owns the still-live executable.
        unsafe { library.cuda_graph_exec_destroy(receiver)?; }
        Ok(())
    }
}
