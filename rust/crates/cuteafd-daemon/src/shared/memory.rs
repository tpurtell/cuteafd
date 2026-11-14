//! Native device/pinned allocation and stream owners shared by model components.
use anyhow::Result;
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer, NativeLibrary};
pub(crate) use cuteafd_ffi::memory_ledger::{tag, MemoryKind, Tag};
/// Scratch allocated by this crate's tests; absent from every non-test build.
#[cfg(test)]
pub(crate) const TEST: Tag = Tag::workspace("test/scratch");
/// An immutable arena tag for this crate's tests; absent from every non-test build.
#[cfg(test)]
pub(crate) const TEST_ARENA: cuteafd_core::memory_layout::ArenaTag<cuteafd_core::memory_layout::arena_class::Immutable> =
    cuteafd_core::memory_layout::ArenaTag::weights("test/arena");
use std::ffi::c_void;

#[path = "memory/receipts.rs"]
pub(crate) mod receipts;
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
#[path = "memory/row_window.rs"]
pub(crate) mod row_window;

#[path = "memory/budget.rs"]
pub(crate) mod budget;
#[path = "memory/load_timing.rs"]
pub(crate) mod load_timing;
#[path = "memory/push_coordinator.rs"]
pub(crate) mod push_coordinator;
#[path = "memory/weight_arena.rs"]
pub(crate) mod weight_arena;
#[path = "memory/weight_validation.rs"]
pub(crate) mod weight_validation;
#[path = "memory/weight_views.rs"]
pub(crate) mod weight_views;
#[path = "memory/push_consumer.rs"]
pub(crate) mod push_consumer;
#[path = "memory/push_initialize.rs"]
pub(crate) mod push_initialize;
#[path = "memory/push_exl3.rs"]
pub(crate) mod push_exl3;
#[path = "memory/push_exlr8.rs"]
pub(crate) mod push_exlr8;
#[path = "memory/push_fp8.rs"]
pub(crate) mod push_fp8;
#[path = "memory/push_native.rs"]
pub(crate) mod push_native;

pub(crate) struct DeviceAllocation<'a> {
    pub(crate) library: &'a NativeLibrary,
    pub(crate) buffer: CuteafdDeviceBuffer,
    pub(crate) backing: Option<device::Allocation<'a>>,
}
impl<'a> DeviceAllocation<'a> {
    pub(crate) fn from_allocation(allocation: device::Allocation<'a>) -> Self {
        Self { library: allocation.device.library, buffer: allocation.buffer, backing: Some(allocation) }
    }

    pub(crate) fn inventory(&self, inventory: &mut receipts::Inventory<'_>, name: &str) -> Result<()> {
        match &self.backing {
            Some(allocation) => allocation.inventory(inventory, name),
            None => inventory.device(name, self.buffer),
        }
    }

    /// Attention value/scale planes retain 16-byte alignment under every row slice.
    pub(crate) fn attention_plane(library: &'a NativeLibrary, rows: usize, stride: usize,
        name: &str, tag: Tag) -> Result<Self> {
        let bytes = rows.checked_mul(stride).ok_or_else(|| anyhow::anyhow!("attention plane size overflow"))?;
        let allocation = Self::new(library, bytes, tag)?;
        anyhow::ensure!(allocation.buffer.ptr as usize % 16 == 0 && stride % 16 == 0,
            "V4.1 BF16 AOT attention load: {name} shape=[{rows},{stride}] on GPU {} requires 16-byte aligned base and row stride",
            allocation.buffer.device_id);
        Ok(allocation)
    }

    /// Device memory on the current device holding `tag`.
    #[track_caller]
    pub(crate) fn new(library: &'a NativeLibrary, bytes: usize, tag: Tag) -> Result<Self> {
        Ok(Self { library, buffer: library.alloc_device_buffer(bytes, tag)?, backing: None })
    }

    /// Kind and owner as the ledger recorded them (`None` for empty backing).
    pub(crate) fn tag(&self) -> Option<Tag> {
        match &self.backing { Some(allocation) => allocation.tag(), None => self.tag_of_buffer() }
    }

    fn tag_of_buffer(&self) -> Option<Tag> {
        cuteafd_ffi::memory_ledger::tag_of(self.buffer.ptr as usize)
    }
}
impl Drop for DeviceAllocation<'_> {
    fn drop(&mut self) {
        if self.backing.is_some() { return; }
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
/// Borrowed descriptor of an already admitted pinned owner, retained until load commit.
#[derive(Clone)]
pub(crate) struct HostLanding { address: usize, bytes: usize, pub(crate) ready: std::sync::Arc<std::sync::atomic::AtomicBool> }
impl HostLanding {
    pub(crate) fn new(storage: &mut HostAllocation<'_>) -> Self {
        Self { address: storage.buffer.ptr as usize, bytes: storage.buffer.bytes, ready: Default::default() }
    }
    pub(crate) fn bytes(&self) -> usize { self.bytes }
    pub(crate) fn fill(&self, body: impl FnOnce(&mut [u8]) -> Result<()>) -> Result<()> {
        // SAFETY: the load owns this pinned allocation exclusively through commit.
        body(unsafe { std::slice::from_raw_parts_mut(self.address as *mut u8, self.bytes) })
    }

    pub(crate) fn copy(&self, at: usize, source: &[u8]) -> Result<()> {
        anyhow::ensure!(at.checked_add(source.len()).is_some_and(|end| end <= self.bytes),
            "host landing outside admitted owner");
        // SAFETY: the recording loader retains this pinned owner until every rank
        // joins; checked fragments are disjoint and no reader runs before commit.
        unsafe { std::ptr::copy_nonoverlapping(source.as_ptr(), (self.address as *mut u8).add(at), source.len()); }
        Ok(())
    }
}
pub(crate) struct HostAllocation<'a> {
    pub(crate) library: &'a NativeLibrary,
    pub(crate) buffer: CuteafdHostBuffer,
}
impl<'a> HostAllocation<'a> {
    /// Zeroed pinned, device-mapped host memory holding `tag`.
    #[track_caller]
    pub(crate) fn new(library: &'a NativeLibrary, bytes: usize, tag: Tag) -> Result<Self> {
        Self::zeroed(library, library.alloc_host_buffer(bytes, tag)?, bytes)
    }

    fn zeroed(library: &'a NativeLibrary, buffer: CuteafdHostBuffer, bytes: usize) -> Result<Self> {
        let value = Self { library, buffer };
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
    /// `tag` must be an immutable kind: the same bytes serve from either placement.
    pub(crate) fn new(library: &'a NativeLibrary, bytes: usize, host: bool,
        tag: cuteafd_core::memory_layout::ArenaTag<cuteafd_core::memory_layout::arena_class::Immutable>) -> Result<Self> {
        if host {
            let storage = HostAllocation::new(library, bytes, tag.tag())?;
            let alias = library.cuda_host_buffer_device_alias(storage.buffer)?;
            Ok(Self::Host { storage, alias })
        } else {
            Ok(Self::Device(DeviceAllocation::new(library, bytes, tag.tag())?))
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

/// Pinned staging with one region per layer: V4.1 passes run device-ordered,
/// so each layer's queued uploads read their own bytes and the host can queue
/// later layers before earlier uploads ran (no staging fence).
pub(crate) struct LayerStaging<'a> {
    allocation: HostAllocation<'a>,
    region: usize,
    regions: usize,
}
impl<'a> LayerStaging<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, region: usize, layers: usize, owner: &'static str) -> Result<Self> {
        let regions = layers.max(1);
        let tag = Tag::new(MemoryKind::Staging, owner);
        Ok(Self { allocation: HostAllocation::new(library, region.max(1) * regions, tag)?, region: region.max(1), regions })
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
/// A per-step host wait for a stream's queued work (the sampling sync, a
/// draft round): under `--host-wait block` it polls one blocking-sync event
/// per device, recorded on the stream, for `BLOCK_SPIN`, then sleeps on it;
/// under `spin` it is a
/// stream synchronize, which the context's default scheduling spins.
pub(crate) struct BlockingWait<'a> {
    library: &'a NativeLibrary,
    events: Vec<(i32, *mut std::ffi::c_void)>,
}

impl<'a> BlockingWait<'a> {
    pub(crate) fn new(library: &'a NativeLibrary) -> Self { Self { library, events: Vec::new() } }

    /// Waits until everything queued on `stream` so far has completed.
    ///
    /// # Safety
    /// `stream` is live until this wait completes (its device need not be current).
    pub(crate) unsafe fn wait(&mut self, stream: *mut std::ffi::c_void) -> Result<()> {
        // SAFETY: forwarded from this function's contract.
        let device = unsafe { self.library.cuda_stream_device(stream)? };
        let _device = device::Device { library: self.library, id: device }.enter()?;
        if cuteafd_transport::wait_policy::wait_policy().wait == cuteafd_transport::wait_policy::HostWait::Spin {
            // SAFETY: forwarded from this function's contract.
            return unsafe { self.library.cuda_stream_synchronize(stream) };
        }
        let event = match self.events.iter().find(|(owner, _)| *owner == device) {
            Some(&(_, event)) => event,
            None => {
                let event = self.library.cuda_event_create_blocking()?;
                self.events.push((device, event));
                event
            }
        };
        // SAFETY: the event belongs to the current device, as does `stream`
        // (the caller's contract); it lives until this wait drops. Polls it
        // for the first `BLOCK_SPIN` (10 ms: every per-layer decode wait
        // completes within that, and the wake-up of a blocking sync would be
        // on the critical path), then sleeps in the driver.
        unsafe {
            self.library.cuda_event_record(event, stream)?;
            if cuteafd_transport::wait_policy::spin_window(|| self.library.cuda_event_query(event))? { return Ok(()); }
            self.library.cuda_event_synchronize(event)
        }
    }
}

impl Drop for BlockingWait<'_> {
    fn drop(&mut self) {
        for &(device, event) in &self.events {
            let previous = self.library.cuda_get_device();
            let _ = self.library.cuda_set_device(device);
            // SAFETY: created by this wait on `device`; synchronizing retires
            // its last record before the event is destroyed.
            unsafe {
                let _ = self.library.cuda_event_synchronize(event);
                let _ = self.library.cuda_event_destroy(event);
            }
            if let Ok(previous) = previous { let _ = self.library.cuda_set_device(previous); }
        }
    }
}

/// Standalone component qualification shares one lazily created stream per owner.
/// Device metadata and teardown never initialize it; serving uses explicit streams.
pub(crate) struct StandaloneStream<'a> {
    pub(crate) device: device::Device<'a>,
    stream: std::cell::OnceCell<device::Stream<'a>>,
}
impl<'a> StandaloneStream<'a> {
    pub(crate) fn new(device: device::Device<'a>) -> Self {
        Self { device, stream: std::cell::OnceCell::new() }
    }
    pub(crate) fn get(&self) -> Result<&device::Stream<'a>> {
        if self.stream.get().is_none() {
            let stream = device::Stream::new(self.device)?;
            let _ = self.stream.set(stream);
        }
        Ok(self.stream.get().expect("initialized standalone stream"))
    }
    pub(crate) fn drain(&self) -> Result<()> {
        self.stream.get().map_or(Ok(()), device::Stream::drain)
    }
    pub(crate) fn require_complete(&self) -> Result<()> {
        if let Some(stream) = self.stream.get() {
            anyhow::ensure!(stream.ready()?, "standalone component stream is unfinished");
        }
        Ok(())
    }
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
        // A completed wait is progress: the host-wait pacing around the
        // caller's loop restarts its spin (`wait_policy::Backoff`).
        cuteafd_transport::wait_policy::note_active();
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
pub(crate) fn cpu_stream_fixture() -> Result<(tempfile::TempDir, cuteafd_ffi::native_library_lifetime_fixture::Fixture)> {
    use anyhow::Context;
    let root = std::path::PathBuf::from(std::env::var_os("HOME").context("HOME is unset")?)
        .join(".cache/cuteafd/builds");
    let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let status = std::process::Command::new("python3")
        .arg(repo.join("scripts/build/assert-build-filesystem.py"))
        .arg(&root)
        .status()?;
    anyhow::ensure!(status.success(), "CPU stream fixture filesystem check failed");
    std::fs::create_dir_all(&root)?;
    let directory = tempfile::Builder::new().prefix("cpu-stream-fixture-").tempdir_in(root)?;
    let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_in(directory.path())?;
    Ok((directory, fixture))
}

#[cfg(test)]
mod load_stream_lifetime_tests {
    use super::*;
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
    use std::{future::Future, task::{Context, Poll, Waker}};

    #[test]
    fn standalone_stream_is_lazy_shared_and_drains_without_initializing() -> Result<()> {
        let (_directory, fixture) = cpu_stream_fixture()?;
        let library = fixture.load()?;
        let device = device::Device { library: &library, id: 0 };
        let owner = std::rc::Rc::new(StandaloneStream::new(device));
        let component = owner.clone();
        owner.drain()?;
        component.require_complete()?;
        assert!(!fixture.events()?.contains('V'), "metadata/drain must not create a stream");
        assert_eq!(owner.get()?.raw, component.get()?.raw);
        assert_eq!(fixture.events()?.matches('V').count(), 1);
        drop(component);
        drop(owner);
        assert_eq!(fixture.events()?.matches('V').count(), 1);
        Ok(())
    }

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
            let weights = [DeviceAllocation::new(&library, 256, crate::shared::memory::TEST)?, DeviceAllocation::new(&library, 256, crate::shared::memory::TEST)?];
            let staging = [HostAllocation::new(&library, 256, crate::shared::memory::TEST)?, HostAllocation::new(&library, 256, crate::shared::memory::TEST)?];
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

/// Test support for fake SparkNest push sources.
#[cfg(test)]
pub(crate) mod push_testing {
    pub(crate) fn bind(socket: &str) -> std::io::Result<cuteafd_loader::push_load::Listener> {
        cuteafd_loader::push_load::validate_socket_endpoint(socket)?;
        cuteafd_loader::push_load::Listener::bind(socket)
    }

    #[test]
    fn overlong_fixture_socket_is_rejected_before_bind() {
        let error = match bind(&"x".repeat(108)) {
            Ok(_) => panic!("overlong fixture socket accepted"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(error.get_ref().unwrap().is::<cuteafd_loader::push_load::SocketEndpointError>());
        assert!(error.to_string().contains("socket path is 108 bytes > 107 bytes"));
    }

    /// A fake SparkNest's `accept`, bounded: a rank that never connects fails the
    /// test after `timeout` with a clear error instead of hanging (Listener has no
    /// accept timeout; a waker connects at the deadline).
    pub(crate) fn accept_within(listener: &cuteafd_loader::push_load::Listener, socket: &str, timeout: std::time::Duration)
        -> anyhow::Result<cuteafd_loader::push_load::Conn> {
        let (accepted, watch) = std::sync::mpsc::channel::<()>();
        let fired = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (path, flag) = (socket.to_string(), fired.clone());
        let watchdog = std::thread::spawn(move || {
            if watch.recv_timeout(timeout).is_err() {
                flag.store(true, std::sync::atomic::Ordering::SeqCst);
                let _ = cuteafd_loader::push_load::Conn::connect(&path);
            }
        });
        let conn = listener.accept();
        let _ = accepted.send(());
        let _ = watchdog.join();
        anyhow::ensure!(!fired.load(std::sync::atomic::Ordering::SeqCst),
            "fake SparkNest: no rank connected to {socket} within {} s", timeout.as_secs());
        Ok(conn?)
    }
}
