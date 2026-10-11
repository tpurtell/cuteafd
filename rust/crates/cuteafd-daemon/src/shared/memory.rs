//! Native device/pinned allocation and stream owners shared by model components.
use anyhow::Result;
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer, NativeLibrary};
use std::ffi::c_void;

#[path = "memory/snapshot.rs"]
mod snapshot;
pub(crate) use snapshot::{SnapshotCopies, SnapshotPool, SnapshotStorage};
#[path = "memory/download.rs"]
mod download;
pub(crate) use download::RowDownload;
#[path = "memory/device.rs"]
pub(crate) mod device;
#[path = "memory/staging.rs"]
pub(crate) mod staging;
#[path = "memory/peer_publication.rs"]
pub(crate) mod peer_publication;
#[path = "memory/proposal_replica.rs"]
pub(crate) mod proposal_replica;
#[path = "memory/chain.rs"]
pub(crate) mod chain;

pub(crate) struct DeviceAllocation<'a> {
    pub(crate) library: &'a NativeLibrary,
    pub(crate) buffer: CuteafdDeviceBuffer,
}
impl<'a> DeviceAllocation<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, bytes: usize) -> Result<Self> {
        Ok(Self {
            library,
            buffer: library.alloc_device_buffer(bytes)?,
        })
    }
}
impl Drop for DeviceAllocation<'_> {
    fn drop(&mut self) {
        if self.library.is_quarantined_after_failed_drain() {
            // cudaFree may synchronize with unproved work on another stream.
            // Retain this allocation until process teardown on that path.
            return;
        }
        if let Err(error) = self.library.free_device_buffer(&mut self.buffer) {
            tracing::error!(%error, "freeing V4.1 device allocation");
        }
    }
}
pub(crate) struct HostAllocation<'a> {
    pub(crate) library: &'a NativeLibrary,
    pub(crate) buffer: CuteafdHostBuffer,
}
impl<'a> HostAllocation<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, bytes: usize) -> Result<Self> {
        let value = Self {
            library,
            buffer: library.alloc_host_buffer(bytes)?,
        };
        // Padding must also be initialized before copying the contiguous arena.
        unsafe {
            std::ptr::write_bytes(value.buffer.ptr.cast::<u8>(), 0, bytes);
        }
        Ok(value)
    }
    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.buffer.ptr.cast::<u8>(), self.buffer.bytes) }
    }
    /// Read-only view of the same pinned bytes; readers must have joined.
    pub(crate) fn bytes(&self) -> &[u8] {
        // SAFETY: this owner holds the allocation, and its mutable view requires an exclusive borrow.
        unsafe { std::slice::from_raw_parts(self.buffer.ptr.cast::<u8>(), self.buffer.bytes) }
    }
}
impl Drop for HostAllocation<'_> {
    fn drop(&mut self) {
        if self.library.is_quarantined_after_failed_drain() {
            // Pinned storage can still feed queued copies. Its irreversible
            // quarantine keeps it alive until process teardown.
            return;
        }
        if let Err(error) = self.library.free_host_buffer(&mut self.buffer) {
            tracing::error!(%error, "freeing V4.1 pinned staging");
        }
    }
}
/// Where the token embedding table lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub(crate) enum EmbedPlacement {
    /// A BF16 copy on the coordinator GPU.
    Gpu,
    /// One pinned mapped host copy, gathered without a host hop.
    Host,
}

/// One immutable weight allocation, either device-local or mapped pinned RAM.
/// Both expose a stable device pointer; host placement never allocates a GPU copy.
pub(crate) enum ResidentWeight<'a> {
    Device(DeviceAllocation<'a>),
    Host { storage: HostAllocation<'a>, alias: CuteafdDeviceBuffer },
}
impl<'a> ResidentWeight<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, bytes: usize, host: bool) -> Result<Self> {
        if host {
            let storage = HostAllocation::new(library, bytes)?;
            let alias = library.cuda_host_buffer_device_alias(storage.buffer)?;
            Ok(Self::Host { storage, alias })
        } else {
            Ok(Self::Device(DeviceAllocation::new(library, bytes)?))
        }
    }
    pub(crate) fn buffer(&self) -> CuteafdDeviceBuffer {
        match self { Self::Device(a) => a.buffer, Self::Host { alias, .. } => *alias }
    }
    pub(crate) fn is_host(&self) -> bool { matches!(self, Self::Host { .. }) }
    pub(crate) fn device_bytes(&self) -> usize {
        if self.is_host() { 0 } else { self.buffer().bytes }
    }
}

/// Pinned staging with one region per layer while V4.1 passes may run
/// device-ordered (`chain::device_enabled`), else one region: each layer's
/// queued uploads then read their own bytes, so the host can queue later
/// layers before earlier uploads ran (no staging fence).
pub(crate) struct LayerStaging<'a> {
    allocation: HostAllocation<'a>,
    region: usize,
    regions: usize,
}
impl<'a> LayerStaging<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, region: usize, layers: usize) -> Result<Self> {
        let regions = if chain::device_enabled() { layers.max(1) } else { 1 };
        Ok(Self { allocation: HostAllocation::new(library, region.max(1) * regions)?, region: region.max(1), regions })
    }
    /// Layer `layer`'s region: its host buffer and bytes.
    pub(crate) fn region(&mut self, layer: usize) -> (CuteafdHostBuffer, &mut [u8]) {
        let start = (layer % self.regions) * self.region;
        let buffer = CuteafdHostBuffer {
            // SAFETY: the region lies inside the allocation.
            ptr: unsafe { self.allocation.buffer.ptr.cast::<u8>().add(start) }.cast(),
            bytes: self.region, ..self.allocation.buffer
        };
        (buffer, &mut self.allocation.bytes_mut()[start..start + self.region])
    }
    pub(crate) fn region_bytes(&self) -> usize { self.region }
}
pub(crate) struct LoadStream<'a> {
    pub(crate) library: &'a NativeLibrary,
    pub(crate) raw: *mut c_void,
}
impl LoadStream<'_> {
    /// Rebinding requires a completed owner, not a hidden host-thread wait.
    #[track_caller]
    pub(crate) fn require_complete(&self) -> Result<()> {
        // A device-ordered pass orders every stage after the chain head, so a
        // rebound owner's next work cannot overtake its previous readers.
        if chain::deferred() {
            return Ok(());
        }
        let caller = std::panic::Location::caller();
        anyhow::ensure!(unsafe { self.library.cuda_stream_query(self.raw)? },
            "cannot rebind an unfinished V4.1 stream ({}:{})", caller.file(), caller.line());
        Ok(())
    }
    /// Yield the owner thread while retaining stream/buffer ownership. Cancellation
    /// and errors still drain before the caller can release queued input storage.
    pub(crate) async fn wait(&self) -> Result<()> {
        struct Drain<'s, 'a> { stream: &'s LoadStream<'a>, complete: bool }
        impl Drop for Drain<'_, '_> {
            fn drop(&mut self) {
                if !self.complete {
                    // SAFETY: the borrowed stream and queued storage remain owned by the caller.
                    crate::shared::decode_graph::fatal_drain(
                        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
                            .map_err(anyhow::Error::from),
                        "cancelled loading stream wait");
                }
            }
        }
        let mut guard = Drain { stream: self, complete: false };
        while !unsafe { self.library.cuda_stream_query(self.raw)? } {
            tokio::task::yield_now().await;
        }
        guard.complete = true;
        Ok(())
    }
}
impl Drop for LoadStream<'_> {
    fn drop(&mut self) {
        // Owners must drop this stream before releasing buffers used by queued work.
        // SAFETY: this owner retains the stream until its queued work has drained.
        crate::shared::decode_graph::fatal_drain(
            unsafe { self.library.cuda_stream_synchronize(self.raw) }.map_err(anyhow::Error::from),
            "loading stream");
        if let Err(error) = unsafe { self.library.cuda_stream_destroy(self.raw) } {
            tracing::error!(%error, "destroying V4.1 loading stream");
        }
    }
}

#[cfg(test)]
mod load_stream_lifetime_tests {
    use super::*;
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use std::{future::Future, task::{Context, Poll, Waker}};

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn failed_load_stream_drains_abort_before_release() -> Result<()> {
        use std::os::unix::process::ExitStatusExt;
        const CHILD: &str = "CUTEAFD_FATAL_LOAD_STREAM_CHILD";
        if let Ok(path) = std::env::var(CHILD) {
            let fixture = Fixture::build()?;
            std::fs::write(std::env::var("CUTEAFD_FATAL_LOAD_STREAM_EVIDENCE")?,
                fixture.directory().join("events").to_str().unwrap())?;
            let library = fixture.load()?;
            let weights = [DeviceAllocation::new(&library, 256)?, DeviceAllocation::new(&library, 256)?];
            let staging = [HostAllocation::new(&library, 256)?, HostAllocation::new(&library, 256)?];
            fixture.configure_pack(&library, 0, 1)?;
            match path.as_str() {
                "drop" => drop(LoadStream { library: &library, raw: std::ptr::null_mut() }),
                "cancel" => {
                    let stream = LoadStream { library: &library, raw: std::ptr::null_mut() };
                    let mut wait = Box::pin(stream.wait());
                    let mut context = Context::from_waker(Waker::noop());
                    assert!(matches!(wait.as_mut().poll(&mut context), Poll::Pending));
                    drop(wait);
                }
                "pair" => {
                    fixture.configure_drain_after(&library, 1)?;
                    let streams = [
                        device::Device { library: &library, id: 0 }
                            .own(|| Ok(LoadStream { library: &library, raw: std::ptr::null_mut() }))?,
                        device::Device { library: &library, id: 1 }
                            .own(|| Ok(LoadStream { library: &library, raw: std::ptr::null_mut() }))?,
                    ];
                    drop(streams);
                }
                _ => unreachable!(),
            }
            drop(staging);
            drop(weights);
            drop(library);
            panic!("failed loading-stream drain returned without abort");
        }
        let root = std::path::PathBuf::from(std::env::var("CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR")?);
        std::fs::create_dir_all(&root)?;
        for path in ["drop", "cancel", "pair"] {
            let evidence = root.join(format!("fatal-load-stream-{}-{path}", std::process::id()));
            let status = std::process::Command::new(std::env::current_exe()?)
                .args(["--exact", "shared::memory::load_stream_lifetime_tests::failed_load_stream_drains_abort_before_release", "--ignored"])
                .env(CHILD, path).env("CUTEAFD_FATAL_LOAD_STREAM_EVIDENCE", &evidence).status()?;
            assert_eq!(status.signal(), Some(libc::SIGABRT));
            let events = std::fs::read_to_string(std::fs::read_to_string(evidence)?)?;
            assert_eq!(events, if path == "pair" { "DDAASTS" } else { "DDAAS" },
                "failed drain must abort before its stream destroy, pinned/device free or module unload");
        }
        Ok(())
    }
}
