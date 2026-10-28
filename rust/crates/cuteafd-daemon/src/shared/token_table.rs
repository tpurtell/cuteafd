//! Common token-table H2D issue contract. Call only after host readiness and
//! destination reuse gates are satisfied; never park these copies behind waits.
use anyhow::{ensure, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer, NativeLibrary};
use std::ffi::c_void;

pub(crate) struct CopyPlane { pub destination: CuteafdDeviceBuffer, pub offset: usize, pub bytes: usize }

/// # Safety
/// The source lease and every destination survive the upload fence; stream is
/// on their device, and the caller has observed its destination reuse gate.
pub(crate) unsafe fn enqueue(library: &NativeLibrary, source: CuteafdHostBuffer,
    planes: &[CopyPlane], stream: *mut c_void) -> Result<()> {
    for plane in planes {
        ensure!(plane.offset.checked_add(plane.bytes).is_some_and(|n| n <= source.bytes)
            && plane.bytes <= plane.destination.bytes, "token-table copy plane exceeds storage");
    }
    for plane in planes {
        if plane.bytes == 0 { continue; }
        let host = CuteafdHostBuffer { ptr: source.ptr.cast::<u8>().wrapping_add(plane.offset).cast(), bytes: plane.bytes, ..source };
        // SAFETY: validated disjoint source range and caller-owned destination;
        // the caller records a source fence even after a partial enqueue error.
        unsafe { library.copy_host_buffer_h2d_async(plane.destination, host, plane.bytes, stream)?; }
    }
    Ok(())
}
