//! Explicit device owners for the dual-RTX path. Legacy owners stay unchanged.
use super::*;
use anyhow::ensure;
use std::{future::Future, mem::ManuallyDrop, pin::Pin, task::{Context as TaskContext, Poll}};

#[derive(Clone, Copy)]
pub(crate) struct Device<'a> {
    pub library: &'a NativeLibrary,
    pub id: i32,
}

#[cfg(test)]
pub(crate) fn cache_test_device(library: &NativeLibrary) -> Result<Device<'_>> {
    let id: i32 = std::env::var("CUTEAFD_CACHE_TEST_DEVICE").unwrap_or_else(|_| "0".into()).parse()?;
    ensure!(matches!(id,0 | 1), "invalid cache fixture GPU");
    library.cuda_set_device(0)?;
    if id == 1 {
        for gpu in 0..2 { Device { library, id: gpu }.run(|| library.cuda_enable_peer(1-gpu))?; }
    }
    Ok(Device { library,id })
}

impl<'a> Device<'a> {
    /// Scope a synchronous method body. Never retain this guard across a yield.
    pub fn enter(&self) -> Result<DeviceScope<'a>> {
        let previous = self.library.cuda_get_device()?;
        // The runtime's default ordinal does not prove a context is current.
        self.library.cuda_set_device(self.id)?;
        Ok(DeviceScope { library: self.library, previous, changed: previous != self.id })
    }
    /// Scope every poll and cancellation cleanup, never an entire async wait.
    pub fn future<F: Future>(&self, future: F) -> DeviceFuture<'a, F> {
        DeviceFuture { device: *self, future: ManuallyDrop::new(future) }
    }

    /// Legacy components may own streams without storing their device ordinal.
    /// Construct and destroy the entire component on its selected device.
    pub fn own<T>(&self, initialize: impl FnOnce() -> Result<T>) -> Result<DeviceOwner<'a, T>> {
        Ok(DeviceOwner { device: *self, value: ManuallyDrop::new(self.run(initialize)?) })
    }
    /// Only synchronous enqueue/query work belongs inside this closure. The
    /// previous device is restored before the caller can yield its async task.
    pub fn run<T>(&self, work: impl FnOnce() -> Result<T>) -> Result<T> {
        struct Restore<'a> { library: &'a NativeLibrary, previous: i32, armed: bool }
        impl Drop for Restore<'_> {
            fn drop(&mut self) {
                if self.armed {
                    if let Err(error) = self.library.cuda_set_device(self.previous) {
                        tracing::error!(%error, "restoring CUDA device during unwind");
                    }
                }
            }
        }
        let previous = self.library.cuda_get_device()?;
        self.library.cuda_set_device(self.id)?;
        if previous == self.id { return work(); }
        let mut restore = Restore { library: self.library, previous, armed: true };
        let result = work();
        self.library.cuda_set_device(previous)?;
        restore.armed = false;
        result
    }
}

pub(crate) struct DeviceScope<'a> { library: &'a NativeLibrary, previous: i32, changed: bool }
impl Drop for DeviceScope<'_> {
    fn drop(&mut self) {
        if self.changed {
            if let Err(error) = self.library.cuda_set_device(self.previous) {
                tracing::error!(%error, "restoring synchronous CUDA device scope");
            }
        }
    }
}

pub(crate) struct DeviceOwner<'a, T> { pub device: Device<'a>, value: ManuallyDrop<T> }
impl<T> DeviceOwner<'_, T> {
    /// Metadata access is ordinary; GPU work must use this owner's device scope.
    pub fn get(&self) -> &T { &self.value }
    pub fn get_mut(&mut self) -> &mut T { &mut self.value }
    /// The owned value, to be dropped by the caller in this owner's device
    /// scope (moved into another device-scoped owner).
    pub fn into_inner(self) -> T {
        let mut this = std::mem::ManuallyDrop::new(self);
        // SAFETY: `this` is never dropped; the value is moved out exactly once.
        unsafe { ManuallyDrop::take(&mut this.value) }
    }
}
impl<T> std::ops::Deref for DeviceOwner<'_, T> {
    type Target = T;
    fn deref(&self) -> &T { self.get() }
}
impl<T> std::ops::DerefMut for DeviceOwner<'_, T> {
    fn deref_mut(&mut self) -> &mut T { self.get_mut() }
}
impl<T> Drop for DeviceOwner<'_, T> {
    fn drop(&mut self) {
        if let Err(error) = self.device.run(|| {
            unsafe { ManuallyDrop::drop(&mut self.value); }
            Ok(())
        }) { tracing::error!(%error, "dropping device-owned component"); }
    }
}

pub(crate) struct DeviceFuture<'a, F: Future> { device: Device<'a>, future: ManuallyDrop<F> }
impl<T, F: Future<Output = Result<T>>> Future for DeviceFuture<'_, F> {
    type Output = Result<T>;
    fn poll(self: Pin<&mut Self>, context: &mut TaskContext<'_>) -> Poll<Self::Output> {
        // The field is never moved after pinning; Drop destroys it in place.
        let this = unsafe { self.get_unchecked_mut() };
        match this.device.run(|| Ok(unsafe { Pin::new_unchecked(&mut *this.future) }.poll(context))) {
            Ok(poll) => poll,
            Err(error) => Poll::Ready(Err(error)),
        }
    }
}
impl<F: Future> Drop for DeviceFuture<'_, F> {
    fn drop(&mut self) {
        if let Err(error) = self.device.run(|| {
            unsafe { ManuallyDrop::drop(&mut self.future); }
            Ok(())
        }) { tracing::error!(%error, "dropping device-scoped future"); }
    }
}

pub(crate) struct Allocation<'a> {
    pub device: Device<'a>,
    pub buffer: CuteafdDeviceBuffer,
}
impl<'a> Allocation<'a> {
    pub fn attention_plane(device: Device<'a>, rows: usize, stride: usize, name: &str) -> Result<Self> {
        let bytes = rows.checked_mul(stride).ok_or_else(|| anyhow::anyhow!("attention plane size overflow"))?;
        let allocation = Self::new(device, bytes)?;
        anyhow::ensure!(allocation.buffer.ptr as usize % 16 == 0 && stride % 16 == 0,
            "V4.1 BF16 AOT attention load: {name} shape=[{rows},{stride}] on GPU {} requires 16-byte aligned base and row stride",
            device.id);
        Ok(allocation)
    }

    pub fn new(device: Device<'a>, bytes: usize) -> Result<Self> {
        let buffer = device.run(|| device.library.alloc_device_buffer(bytes))?;
        Ok(Self { device, buffer })
    }
}
impl Drop for Allocation<'_> {
    fn drop(&mut self) {
        if self.device.library.is_quarantined_after_failed_drain() {
            // Match legacy allocations: cudaFree may wait on unproved work.
            return;
        }
        if let Err(error) = self.device.run(|| self.device.library.free_device_buffer(&mut self.buffer)) {
            tracing::error!(%error, "freeing device-owned allocation");
        }
    }
}

#[cfg(test)]
mod allocation_lifetime_tests {
    use super::*;
    use cuteafd_ffi::native_library_lifetime_fixture::Fixture;

    #[test]
    fn scopes_bind_context_even_when_runtime_reports_the_owner_ordinal() {
        let source = include_str!("device.rs").split("#[cfg(test)]\nmod allocation_lifetime_tests").next().unwrap();
        let enter = source.split("pub fn enter(").nth(1).unwrap().split("pub fn future").next().unwrap();
        assert!(enter.contains("self.library.cuda_set_device(self.id)?;"));
        assert!(!enter.contains("if previous"));
        let run = source.split("pub fn run<T>").nth(1).unwrap().split("pub(crate) struct DeviceScope").next().unwrap();
        let bind = run.find("self.library.cuda_set_device(self.id)?;").unwrap();
        let same_ordinal = run.find("if previous == self.id { return work(); }").unwrap();
        assert!(bind < same_ordinal, "same-ordinal work must not bypass primary-context binding");
    }

    fn failed_sync_aborts_before_free(path: &str) -> Result<()> {
        use std::os::unix::process::ExitStatusExt;
        const CHILD: &str = "CUTEAFD_FATAL_SYNC_CHILD";
        if std::env::var_os(CHILD).is_some() {
            let fixture = Fixture::build()?;
            std::fs::write(std::env::var("CUTEAFD_FATAL_SYNC_EVIDENCE")?, fixture.directory().join("events").to_str().unwrap())?;
            let library = fixture.load()?;
            let device = Device { library: &library, id: 0 };
            let allocation = Allocation::new(device, 256)?;
            let stream = Stream { device, raw: std::ptr::null_mut() };
            fixture.configure_pack(&library, 0, 1)?;
            match path {
                "stream_drop" => drop(stream),
                "wait_cancel" => {
                    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
                    runtime.block_on(async {
                        let mut wait = std::pin::pin!(stream.wait());
                        std::future::poll_fn(|cx| {
                            assert!(wait.as_mut().poll(cx).is_pending());
                            std::task::Poll::Ready(())
                        }).await;
                        drop(wait);
                    });
                }
                _ => {
                    let site = match path {
                        "peer_cancel" => "cancelled peer transfer",
                        "tp2_shared_cancel" => "TP2 shared input upload",
                        "tp2_ffn_cancel" => "TP2 FFN input upload",
                        "tp2_rank_cancel" => "cancelled TP2 rank",
                        "tp2_shared_rank_cancel" => "shared TP2 rank",
                        _ => unreachable!(),
                    };
                    if matches!(path, "tp2_rank_cancel" | "tp2_shared_rank_cancel") { stream.drain_or_abort(site); }
                    else { drop(stream.cancellation_guard(site)); }
                }
            }
            drop(allocation);
            panic!("failed synchronize returned without abort");
        }
        let root = std::path::PathBuf::from(std::env::var("CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR")?);
        std::fs::create_dir_all(&root)?;
        let evidence = root.join(format!("fatal-sync-{}-{path}", std::process::id()));
        let status = std::process::Command::new(std::env::current_exe()?)
            .args(["--exact", &format!("shared::memory::device::allocation_lifetime_tests::{path}"), "--ignored"])
            .env(CHILD, "1").env("CUTEAFD_FATAL_SYNC_EVIDENCE", &evidence).status()?;
        assert_eq!(status.signal(), Some(libc::SIGABRT));
        let events = std::fs::read_to_string(std::fs::read_to_string(evidence)?)?;
        assert_eq!(events, "DS", "abort must precede cudaFree and module unload");
        Ok(())
    }

    macro_rules! failed_sync_test {
        ($name:ident) => {
            #[test]
            #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
            fn $name() -> Result<()> { failed_sync_aborts_before_free(stringify!($name)) }
        };
    }
    failed_sync_test!(stream_drop);
    failed_sync_test!(wait_cancel);
    failed_sync_test!(peer_cancel);
    failed_sync_test!(tp2_rank_cancel);
    failed_sync_test!(tp2_shared_rank_cancel);
    failed_sync_test!(tp2_shared_cancel);
    failed_sync_test!(tp2_ffn_cancel);

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn explicit_drain_error_remains_recoverable() -> Result<()> {
        let fixture = Fixture::build()?;
        let library = fixture.load()?;
        let device = Device { library: &library, id: 0 };
        let allocation = Allocation::new(device, 256)?;
        let stream = Stream { device, raw: std::ptr::null_mut() };
        fixture.configure_pack(&library, 0, 1)?;
        assert!(stream.drain().is_err());
        library.quarantine_module_after_failed_drain();
        // Deliberate error handling retains the raw stream as well as storage.
        std::mem::forget(stream);
        drop(allocation);
        drop(library);
        assert_eq!(fixture.events()?, "DS");
        Ok(())
    }

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn quarantined_allocation_drop_never_calls_cuda_free() -> Result<()> {
        for quarantined in [false, true] {
            let fixture = Fixture::build()?;
            let library = fixture.load()?;
            let allocation = Allocation::new(Device { library: &library, id: 1 }, 256)?;
            assert_eq!(library.cuda_get_device()?, 0, "allocation must restore the current device");
            if quarantined { library.quarantine_module_after_failed_drain(); }
            drop(allocation);
            assert_eq!(fixture.events()?, if quarantined { "D" } else { "Dd" });
            assert_eq!(library.cuda_get_device()?, 0, "drop must preserve the current device");
            drop(library);
            assert_eq!(fixture.events()?, if quarantined { "D" } else { "DdU" });
            assert_eq!(fixture.resident()?, quarantined);
        }
        Ok(())
    }
}

pub(crate) struct Stream<'a> {
    pub device: Device<'a>,
    pub raw: *mut c_void,
}
impl<'a> Stream<'a> {
    pub fn new(device: Device<'a>) -> Result<Self> {
        Ok(Self { device, raw: device.run(|| device.library.cuda_stream_create())? })
    }
    fn ready(&self) -> Result<bool> {
        self.device.run(|| unsafe { self.device.library.cuda_stream_query(self.raw) })
    }
    pub(crate) fn drain(&self) -> Result<()> {
        self.device.run(|| unsafe { self.device.library.cuda_stream_synchronize(self.raw) })
    }
    pub(crate) fn drain_or_abort(&self, site: &str) {
        crate::shared::decode_graph::fatal_drain(self.drain(), site);
    }
    pub(crate) fn cancellation_guard(&self, site: &'static str) -> CancellationDrain<'_, 'a> {
        CancellationDrain { stream: self, complete: false, site }
    }
    /// Completes this stream's queued work for later stages: a host wait
    /// ([`Self::wait`]), or under [`super::chain::deferred`] a chain merge
    /// (later stages join the chain head; no host wait).
    pub(crate) async fn complete(&self) -> Result<()> {
        if super::chain::deferred() {
            return self.device.run(|| unsafe { super::chain::finish(self.device.library, self.raw) });
        }
        self.wait().await
    }
    /// Orders this stream after the chain head (no-op outside a chain scope).
    pub(crate) fn join_chain(&self) -> Result<()> {
        self.device.run(|| unsafe { super::chain::join(self.device.library, self.raw) })
    }
    /// Retain queued work through cooperative completion or cancellation drain.
    pub(crate) async fn wait(&self) -> Result<()> {
        let mut guard = self.cancellation_guard("interrupted device stream");
        while !self.ready()? { tokio::task::yield_now().await; }
        guard.complete = true;
        Ok(())
    }
}
pub(crate) struct CancellationDrain<'s, 'a> {
    stream: &'s Stream<'a>,
    pub complete: bool,
    site: &'static str,
}
impl Drop for CancellationDrain<'_, '_> {
    fn drop(&mut self) {
        if !self.complete { self.stream.drain_or_abort(self.site); }
    }
}
impl Drop for Stream<'_> {
    fn drop(&mut self) {
        self.drain_or_abort("device-owned stream");
        if let Err(error) = self.device.run(|| unsafe { self.device.library.cuda_stream_destroy(self.raw) }) {
            tracing::error!(%error, "destroying device-owned stream");
        }
    }
}

pub(crate) struct Event<'a> { pub device: Device<'a>, pub raw: *mut c_void }
impl<'a> Event<'a> {
    pub fn new(device: Device<'a>) -> Result<Self> {
        Ok(Self { device, raw: device.run(|| device.library.cuda_event_create())? })
    }
    pub fn record(&mut self, producer: &Stream<'a>) -> Result<()> {
        ensure!(self.device.id == producer.device.id
            && std::ptr::eq(self.device.library, producer.device.library), "event producer device mismatch");
        self.device.run(|| unsafe { self.device.library.cuda_event_record(self.raw, producer.raw) })
    }
}
impl Drop for Event<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.device.run(|| unsafe { self.device.library.cuda_event_destroy(self.raw) }) {
            tracing::error!(%error, "destroying device-owned event");
        }
    }
}

/// The widest decode-width peer transfer: a decode lane's 80 rows
/// (`DECODE_LANE_ROWS`, the compiled cover of the 64-row verify width) of a
/// 10240-byte BF16 hidden row. Every decode/verify transfer is at most this;
/// every prefill unit (>= 256 rows, `V41_LIVE_MIN_ROWS`) is wider.
pub(crate) const DECODE_PEER_COPY_BYTES: usize = 80 * 10240;

/// Whether a device-ordered peer copy of `bytes` runs as an SM copy on the
/// issuing stream (ordered by its event waits) instead of on a copy engine.
/// Decode-width transfers do: a copy-engine copy queued behind an unmet event
/// wait parks that engine and head-of-line blocks every other copy on it
/// (Engram uploads, peer landings). Prefill-width transfers keep the copy
/// engines (prof12: SM copies cost ~1.2 s of GPU0 per 64K prefill on two RTX).
pub(crate) fn sm_peer_copy_for(deferred: bool, bytes: usize) -> bool {
    deferred && bytes <= DECODE_PEER_COPY_BYTES
}

#[cfg(test)]
mod peer_copy_choice_tests {
    /// Decode-width device-ordered transfers run as SM copies (never parked on
    /// a copy engine behind an unmet wait); prefill-width ones keep the copy
    /// engines; host-ordered passes always use the copy engines.
    #[test]
    fn decode_width_device_ordered_copies_use_sm() {
        use super::{sm_peer_copy_for, DECODE_PEER_COPY_BYTES};
        assert!(sm_peer_copy_for(true, 1 * 10240));
        assert!(sm_peer_copy_for(true, DECODE_PEER_COPY_BYTES));
        assert!(!sm_peer_copy_for(true, 2752 * 10240));
        assert!(!sm_peer_copy_for(true, DECODE_PEER_COPY_BYTES + 1));
        assert!(!sm_peer_copy_for(false, 1 * 10240));
    }
}

#[cfg(test)]
thread_local! {
    /// Test-only negative control: `copy_then_ordered` neither joins nor acquires.
    static SKIP_COPY_ORDER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// One transfer direction for one lane; stream and event are allocated at startup.
pub(crate) struct PeerTransfer<'a> {
    destination: Stream<'a>,
    ready: Event<'a>,
    /// SM copies for device-ordered passes ([`super::chain::deferred`]).
    sm: Option<cuteafd_ffi::V41PeerCopy<'a>>,
}
impl<'a> PeerTransfer<'a> {
    pub fn new(source: Device<'a>, destination: Device<'a>) -> Result<Self> {
        ensure!(std::ptr::eq(source.library, destination.library) && source.id != destination.id,
            "peer transfer requires distinct devices from one library");
        destination.run(|| destination.library.cuda_enable_peer(source.id))?;
        let stream = Stream::new(destination)?;
        let ready = Event { device: source,
            raw: source.run(|| source.library.cuda_event_create())? };
        let sm = Some(destination.run(|| destination.library.v41_peer_copy())?);
        Ok(Self { destination: stream, ready, sm })
    }

    /// # Safety
    /// All source writes must be ordered on `producer`; no aliases may access
    /// either allocation incompatibly while this future is live. Allocation and
    /// producer borrows survive until completion or cancellation drains the copy.
    pub async unsafe fn copy(&mut self, source: &Allocation<'a>, destination: &mut Allocation<'a>,
        producer: &Stream<'a>, bytes: usize) -> Result<()> {
        unsafe { self.copy_then(source, destination, producer, bytes, |_, _| Ok(())).await }
    }

    /// # Safety
    /// Same buffer contract as `copy`. `then` enqueues only on the supplied
    /// destination stream; its captured storage must survive completion. The
    /// callback itself is retained until the completion/cancellation drain.
    pub async unsafe fn copy_then(&mut self, source: &Allocation<'a>, destination: &mut Allocation<'a>,
        producer: &Stream<'a>, bytes: usize,
        then: impl FnMut(CuteafdDeviceBuffer, *mut c_void) -> Result<()>) -> Result<()> {
        unsafe { self.copy_then_ordered(source, destination, producer, bytes, None, then).await }
    }
    /// As [`Self::copy_then`]; under device ordering, `guard` (when given)
    /// replaces the join to the whole chain head. The destination stream then
    /// waits only on the producer's event and on the guard: the release of
    /// the last cross-stream reader of what `then` writes (the caller's
    /// write-after-read edge, `row_window::ScratchGuard`). Readers on this
    /// stream (the destination staging) are ordered by the stream.
    /// # Safety
    /// As `copy_then`; with a guard, every reader of `then`'s outputs on other
    /// streams releases the guard after its read.
    pub async unsafe fn copy_then_ordered(&mut self, source: &Allocation<'a>, destination: &mut Allocation<'a>,
        producer: &Stream<'a>, bytes: usize, guard: Option<&super::row_window::ScratchGuard<'a>>,
        mut then: impl FnMut(CuteafdDeviceBuffer, *mut c_void) -> Result<()>) -> Result<()> {
        let library = self.destination.device.library;
        ensure!(source.device.id == self.ready.device.id
            && producer.device.id == source.device.id
            && destination.device.id == self.destination.device.id
            && std::ptr::eq(library, source.device.library)
            && std::ptr::eq(library, destination.device.library)
            && std::ptr::eq(library, producer.device.library)
            && bytes <= source.buffer.bytes && bytes <= destination.buffer.bytes,
            "peer transfer owner or extent mismatch");
        let mut drain = self.destination.cancellation_guard("cancelled peer transfer");
        self.ready.device.run(|| unsafe { library.cuda_event_record(self.ready.raw, producer.raw) })?;
        let deferred = super::chain::deferred();
        let sm = self.sm.as_ref().filter(|_| sm_peer_copy_for(deferred, bytes));
        self.destination.device.run(|| unsafe {
            library.cuda_stream_wait_event(self.destination.raw, self.ready.raw)?;
            // The destination's previous readers precede this write: the
            // caller's guard, or the whole chain head.
            #[cfg(test)]
            let unordered = SKIP_COPY_ORDER.with(|skip| skip.get());
            #[cfg(not(test))]
            let unordered = false;
            match guard {
                _ if unordered => {}
                Some(guard) if deferred => guard.acquire(self.destination.raw)?,
                _ if deferred => super::chain::join(library, self.destination.raw)?,
                _ => {}
            }
            match sm {
                Some(sm) => sm.launch(destination.buffer, source.buffer, bytes, self.destination.raw)?,
                // Device-ordered passes copy on the copy engines too (#815,
                // prof13): this stream waits on events only, never a memory
                // wait, so it holds no other lane's DMA behind it.
                None => library.copy_peer_async(destination.buffer, source.buffer, bytes, self.destination.raw)?,
            }
            then(destination.buffer, self.destination.raw)
        })?;
        if deferred {
            self.destination.device.run(|| unsafe { super::chain::finish(library, self.destination.raw) })?;
            drain.complete = true;
            return Ok(());
        }
        while !self.destination.ready()? { tokio::task::yield_now().await; }
        drain.complete = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and two CUDA GPUs"]
    fn device_scoped_futures_restore_poll_and_cancellation_context() -> Result<()> {
        use std::{cell::Cell, rc::Rc, task::Waker};
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        let d1 = Device { library: &lib, id: 1 };
        let dropped_on = Rc::new(Cell::new(-1));
        struct Probe<'a> { library: &'a NativeLibrary, dropped_on: Rc<Cell<i32>> }
        impl Drop for Probe<'_> {
            fn drop(&mut self) { self.dropped_on.set(self.library.cuda_get_device().unwrap()); }
        }
        let mut owner = d1.own(|| {
            assert_eq!(lib.cuda_get_device()?,1);
            Ok(Probe { library: &lib, dropped_on: dropped_on.clone() })
        })?;
        assert_eq!(lib.cuda_get_device()?,0);
        let mut context = TaskContext::from_waker(Waker::noop());
        let future = d1.future(async {
            assert_eq!(lib.cuda_get_device()?,1);
            let _guard = Probe { library: &lib, dropped_on: dropped_on.clone() };
            let _stream = LoadStream { library: &lib, raw: lib.cuda_stream_create()? };
            tokio::task::yield_now().await;
            assert_eq!(lib.cuda_get_device()?,1);
            Ok(())
        });
        let mut future = Box::pin(future);
        assert!(future.as_mut().poll(&mut context).is_pending());
        assert_eq!(lib.cuda_get_device()?,0);
        drop(future);
        assert_eq!(dropped_on.get(),1);
        assert_eq!(lib.cuda_get_device()?,0);
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        let d0 = Device { library: &lib, id: 0 };
        runtime.block_on(async {
            tokio::try_join!(d0.future(async {
                for _ in 0..8 { assert_eq!(lib.cuda_get_device()?,0); tokio::task::yield_now().await; }
                Ok(())
            }), d1.future(async {
                for _ in 0..8 { assert_eq!(lib.cuda_get_device()?,1); tokio::task::yield_now().await; }
                Ok(())
            }))?;
            Ok::<_,anyhow::Error>(())
        })?;
        owner.get_mut().dropped_on.set(-1);
        drop(owner);
        assert_eq!(dropped_on.get(),1);
        assert_eq!(lib.cuda_get_device()?,0);
        Ok(())
    }
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, libcudart.so.13 and two CUDA devices"]
    fn cancelled_peer_chain_drains_before_releasing_borrows() -> Result<()> {
        use std::future::Future;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::task::{Context, Poll, Waker};
        struct RuntimeLibrary(*mut c_void);
        impl Drop for RuntimeLibrary {
            fn drop(&mut self) { unsafe { libc::dlclose(self.0); } }
        }
        unsafe extern "C" fn hold(data: *mut c_void) {
            let release = unsafe { &*data.cast::<AtomicBool>() };
            while !release.load(Ordering::Acquire) { std::thread::yield_now(); }
        }
        let cuda = RuntimeLibrary(unsafe { libc::dlopen(c"libcudart.so.13".as_ptr(), libc::RTLD_NOW) });
        ensure!(!cuda.0.is_null(), "CUDA runtime library unavailable");
        let symbol = unsafe { libc::dlsym(cuda.0, c"cudaLaunchHostFunc".as_ptr()) };
        ensure!(!symbol.is_null(), "CUDA callback API unavailable");
        let launch: unsafe extern "C" fn(*mut c_void, unsafe extern "C" fn(*mut c_void), *mut c_void) -> i32 =
            unsafe { std::mem::transmute(symbol) };
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        let d0 = Device { library: &lib, id: 0 };
        let d1 = Device { library: &lib, id: 1 };
        let source = Allocation::new(d0, 4096)?;
        let mut destination = Allocation::new(d1, 4096)?;
        let output = Allocation::new(d1, 4096)?;
        let producer = Stream::new(d0)?;
        let independent = Stream::new(d1)?;
        let mut transfer = PeerTransfer::new(d0, d1)?;
        lib.copy_h2d(source.buffer, &[73u8; 4096])?;
        let release = AtomicBool::new(false);
        // Always release during unwinding, before any stream owner drains.
        struct Release<'a>(&'a AtomicBool);
        impl Drop for Release<'_> { fn drop(&mut self) { self.0.store(true, Ordering::Release); } }
        let _release = Release(&release);
        ensure!(unsafe { launch(producer.raw, hold, (&release as *const AtomicBool).cast_mut().cast()) } == 0,
            "could not stall producer");
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        let _entered = runtime.enter();
        let mut future = Box::pin(unsafe { transfer.copy_then(&source, &mut destination, &producer, 4096,
            |copied, stream| lib.copy_d2d_async(output.buffer, copied, 4096, stream)) });
        // On assertion failure release before the future's cancellation guard.
        let _unwind_release = Release(&release);
        let mut context = Context::from_waker(Waker::noop());
        assert!(matches!(future.as_mut().poll(&mut context), Poll::Pending));
        assert_eq!(lib.cuda_get_device()?, 0);
        assert!(independent.ready()?);
        assert!(!release.load(Ordering::Acquire));
        std::thread::scope(|scope| {
            scope.spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(50));
                release.store(true, Ordering::Release);
            });
            drop(future);
            // Without cancellation draining, Drop returns while the gate is held.
            assert!(release.load(Ordering::Acquire));
        });
        assert!(producer.ready()?);
        assert_eq!(lib.cuda_get_device()?, 0);
        let mut bytes = [0u8; 4096];
        d1.run(|| lib.copy_d2h(&mut bytes, output.buffer))?;
        assert_eq!(bytes, [73u8; 4096]);
        Ok(())
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and two CUDA devices"]
    fn peer_owners_restore_device_and_copy_independent_lanes() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        let a = Device { library: &lib, id: 0 };
        let b = Device { library: &lib, id: 1 };
        let failed: Result<()> = b.run(|| anyhow::bail!("deliberate error"));
        assert!(failed.is_err());
        assert_eq!(lib.cuda_get_device()?, 0);
        const BYTES: usize = 1024 * 1024;
        let source_a = Allocation::new(a, BYTES)?;
        let source_b = Allocation::new(b, BYTES)?;
        let mut target_a = Allocation::new(a, BYTES)?;
        let mut target_b = Allocation::new(b, BYTES)?;
        let producer_a = Stream::new(a)?;
        let producer_b = Stream::new(b)?;
        let mut ab = PeerTransfer::new(a, b)?;
        let mut ba = PeerTransfer::new(b, a)?;
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        for seed in [13u8, 127, 241] {
            let input_a: Vec<u8> = (0..BYTES).map(|i| (i as u8).wrapping_add(seed)).collect();
            let input_b: Vec<u8> = input_a.iter().map(|v| !v).collect();
            a.run(|| lib.copy_h2d(source_a.buffer, &input_a))?;
            b.run(|| lib.copy_h2d(source_b.buffer, &input_b))?;
            runtime.block_on(async {
                let (x, y) = tokio::join!(
                    unsafe { ab.copy(&source_a, &mut target_b, &producer_a, BYTES) },
                    unsafe { ba.copy(&source_b, &mut target_a, &producer_b, BYTES) });
                x?; y?;
                Ok::<_, anyhow::Error>(())
            })?;
            assert_eq!(lib.cuda_get_device()?, 0);
            let mut output = vec![0u8; BYTES];
            a.run(|| lib.copy_d2h(&mut output, target_a.buffer))?;
            assert_eq!(output, input_b);
            b.run(|| lib.copy_d2h(&mut output, target_b.buffer))?;
            assert_eq!(output, input_a);
        }
        drop((ab, ba, producer_a, producer_b, source_a, source_b, target_a, target_b));
        assert_eq!(lib.cuda_get_device()?, 0);
        Ok(())
    }

    /// Hunt A 1E: a guarded peer transfer's `then` write waits for the last
    /// cross-stream reader of its output (released on that reader's chain),
    /// without joining the writer's whole chain head. Negative control: with
    /// no ordering, the write overtakes the held reader.
    fn guarded_copy_waits_for_the_output_reader(ordered: bool) -> Result<()> {
        use crate::shared::memory::chain::StageChain;
        use crate::shared::memory::row_window::ScratchGuard;
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        let (d0, d1) = (Device { library: &lib, id: 0 }, Device { library: &lib, id: 1 });
        let source = Allocation::new(d0, 4096)?;
        let mut staging = Allocation::new(d1, 4096)?;
        let out = Allocation::new(d1, 4096)?;
        let check = Allocation::new(d1, 4096)?;
        let (producer, reader) = (Stream::new(d0)?, Stream::new(d1)?);
        let mut transfer = PeerTransfer::new(d0, d1)?;
        let guard = ScratchGuard::new(&lib, &[0, 1])?;
        lib.copy_h2d(source.buffer, &[2u8; 4096])?;
        d1.run(|| lib.copy_h2d(out.buffer, &[1u8; 4096]))?;
        let written = d1.run(|| lib.cuda_event_create_ordering())?;
        let chains = [StageChain::on_devices(&lib, &[0, 1])?, StageChain::on_devices(&lib, &[0, 1])?];
        for chain in &chains { chain.set_device_order(true); }
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        lib.cuda_device_synchronize()?;
        let gate = d1.run(|| unsafe { cuteafd_ffi::test_support::CudaStreamGate::new(&lib, reader.raw) })?;
        // Unit A: the reader of `out` (held), joined into A's chain, then released.
        runtime.block_on(chains[0].handle().scope(async { d1.run(|| unsafe {
            super::chain::join(&lib, reader.raw)?;
            lib.copy_d2d_async(check.buffer, out.buffer, 4096, reader.raw)?;
            super::chain::finish(&lib, reader.raw)?;
            guard.release()
        }) }))?;
        // Unit B: the guarded transfer overwrites `out` in `then`.
        SKIP_COPY_ORDER.with(|skip| skip.set(!ordered));
        let queued = runtime.block_on(chains[1].handle().scope(async { unsafe {
            transfer.copy_then_ordered(&source, &mut staging, &producer, 4096, Some(&guard), |copied, stream| {
                lib.copy_d2d_async(out.buffer, copied, 4096, stream)?;
                lib.cuda_event_record(written, stream)
            }).await
        } }));
        SKIP_COPY_ORDER.with(|skip| skip.set(false));
        queued?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(if ordered { 2 } else { 10 });
        let mut done = false;
        while std::time::Instant::now() < deadline {
            if d1.run(|| unsafe { lib.cuda_event_query(written) })? { done = true; break; }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        drop(gate);
        for chain in &chains { chain.drain()?; }
        lib.cuda_device_synchronize()?;
        let mut seen = vec![0u8; 4096];
        d1.run(|| lib.copy_d2h(&mut seen, check.buffer))?;
        d1.run(|| unsafe { lib.cuda_event_destroy(written) })?;
        assert_eq!(done, !ordered, "the guarded write {} while the reader was held", if done { "completed" } else { "stayed pending" });
        assert_eq!(seen[0], if ordered { 1 } else { 2 }, "the reader saw {} data", if seen[0] == 2 { "the overwrite's" } else { "its own" });
        Ok(())
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and two CUDA GPUs"]
    fn guarded_peer_transfer_waits_for_its_output_reader() -> Result<()> { guarded_copy_waits_for_the_output_reader(true) }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and two CUDA GPUs"]
    fn unordered_peer_transfer_overwrites_a_held_reader() -> Result<()> { guarded_copy_waits_for_the_output_reader(false) }
}