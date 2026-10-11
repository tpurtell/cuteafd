//! Device-side ordering for the stages of one target pass.
//!
//! Each layer used to drain every stage on the host before the next stage was
//! submitted (mHC finish, query, window KV, attention, shared FFN, reduction),
//! leaving the RTX idle while the host issued the following launch. Inside a
//! chain scope, a stage instead joins the chain's CUDA event before its first
//! enqueue and re-records that event after its last one. Consecutive stages on
//! different streams are therefore ordered on the device, and the host waits
//! only where it must read results (routes before dispatch, the head output).
//!
//! Safety argument: every stage of a scoped pass joins the chain, so the chain
//! is a total order over the pass's GPU work. The per-layer route download is
//! a host wait on a stream joined after all earlier stages, hence every stage
//! submitted before it, including pinned-staging uploads, has completed when
//! the host reuses that staging in the next layer. Work outside a scope keeps
//! its original synchronous behaviour.
//!
//! The scope is a thread-local set only while polling the pass future (the
//! same pattern as the device-scoped futures), so interleaved lanes on one
//! executor thread keep independent chains.
use super::LoadStream;
use anyhow::Result;
use cuteafd_ffi::NativeLibrary;
use std::cell::Cell;
use std::ffi::c_void;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

#[derive(Clone)]
struct Current {
    /// One ordering event per device the pass runs on; (device, event).
    events: Rc<[(i32, *mut c_void)]>,
    /// Index of the event holding the chain head, if any stage finished.
    head: Rc<Cell<Option<usize>>>,
    /// Per-device fork events: a stage marks the point where its outputs that
    /// later producers read are complete, before its remaining work.
    forks: Rc<[(i32, *mut c_void)]>,
    fork: Rc<Cell<Option<usize>>>,
    /// Device-ordered passes: per device a stream and two events marking
    /// "everything chained so far" ([`fence_mark`]), and which device marked each.
    fences: Rc<[Fence]>,
    marked: Rc<Cell<[Option<usize>; 2]>>,
    /// This pass may run device-ordered ([`deferred`]).
    device: bool,
}

#[derive(Clone, Copy)]
struct Fence {
    device: i32,
    stream: *mut c_void,
    events: [*mut c_void; 2],
}

thread_local! {
    static CURRENT: std::cell::RefCell<Option<Current>> = const { std::cell::RefCell::new(None) };
}

/// Ordering events for one pass owner, one per participating device.
pub(crate) struct StageChain<'a> {
    library: &'a NativeLibrary,
    events: Rc<[(i32, *mut c_void)]>,
    head: Rc<Cell<Option<usize>>>,
    forks: Rc<[(i32, *mut c_void)]>,
    fork: Rc<Cell<Option<usize>>>,
    fences: Rc<[Fence]>,
    marked: Rc<Cell<[Option<usize>; 2]>>,
    device: Cell<bool>,
}
impl<'a> StageChain<'a> {
    /// Whether the next scoped pass may run device-ordered (with
    /// [`device_enabled`]); default on.
    pub fn set_device_order(&self, on: bool) {
        self.device.set(on);
    }
    /// A chain for the current device only.
    pub fn new(library: &'a NativeLibrary) -> Result<Self> {
        let device = library.cuda_get_device()?;
        Self::on_devices(library, &[device])
    }
    /// A chain spanning `devices`; each event is created on its own device.
    pub fn on_devices(library: &'a NativeLibrary, devices: &[i32]) -> Result<Self> {
        let previous = library.cuda_get_device()?;
        let mut events = Vec::with_capacity(devices.len());
        let mut forks = Vec::with_capacity(devices.len());
        let mut fences = Vec::with_capacity(devices.len());
        let created = (|| -> Result<()> {
            for &device in devices {
                library.cuda_set_device(device)?;
                events.push((device, library.cuda_event_create_ordering()?));
                forks.push((device, library.cuda_event_create_ordering()?));
                if device_enabled() {
                    fences.push(Fence { device, stream: library.cuda_stream_create()?,
                        events: [library.cuda_event_create_ordering()?, library.cuda_event_create_ordering()?] });
                }
            }
            Ok(())
        })();
        library.cuda_set_device(previous)?;
        if let Err(error) = created {
            drain_fences(library, &fences);
            for &(_, event) in events.iter().chain(&forks) { let _ = unsafe { library.cuda_event_destroy(event) }; }
            release_fences(library, &fences);
            return Err(error);
        }
        Ok(Self { library, events: events.into(), head: Rc::new(Cell::new(None)),
            forks: forks.into(), fork: Rc::new(Cell::new(None)), fences: fences.into(),
            marked: Rc::new(Cell::new([None; 2])), device: Cell::new(true) })
    }
    /// An owned handle that can wrap a future borrowing the chain's owner.
    pub fn handle(&self) -> ChainHandle<'a> {
        ChainHandle { library: self.library, current: Current {
            events: self.events.clone(), head: self.head.clone(),
            forks: self.forks.clone(), fork: self.fork.clone(), fences: self.fences.clone(),
            marked: self.marked.clone(), device: self.device.get() } }
    }
    /// Host wait for everything recorded so far, then forget the head. Call
    /// after the pass (or an aborted pass) before any unscoped consumer.
    pub fn drain(&self) -> Result<()> {
        self.fork.set(None);
        self.marked.set([None; 2]);
        if let Some(head) = self.head.replace(None) {
            unsafe { self.library.cuda_event_synchronize(self.events[head].1)?; }
        }
        Ok(())
    }
    /// [`Self::drain`] that gives up after `limit` (device-ordered passes,
    /// which wait on Spark and peer flags inside the chain): polls the head
    /// through this device's fence stream instead of blocking in the driver.
    /// Without fences it is [`Self::drain`].
    pub fn drain_bounded(&self, limit: std::time::Duration) -> Result<()> {
        let Some(head) = self.head.get() else { return self.drain() };
        let (device, event) = self.events[head];
        let Some(fence) = self.fences.iter().find(|f| f.device == device).copied() else { return self.drain() };
        let previous = self.library.cuda_get_device()?;
        self.library.cuda_set_device(device)?;
        let started = std::time::Instant::now();
        // SAFETY: the fence stream and head event belong to this chain's device.
        let result = (|| -> Result<()> {
            unsafe { self.library.cuda_stream_wait_event(fence.stream, event)?; }
            loop {
                if unsafe { self.library.cuda_stream_query(fence.stream)? } { return Ok(()); }
                anyhow::ensure!(started.elapsed() < limit,
                    "device-ordered pass did not complete within {limit:?} (a device wait never released)");
                std::thread::sleep(std::time::Duration::from_micros(50));
            }
        })();
        self.library.cuda_set_device(previous)?;
        result?;
        self.drain()
    }
}
impl Drop for StageChain<'_> {
    fn drop(&mut self) {
        crate::shared::decode_graph::fatal_drain(self.drain(), "target stage chain");
        drain_fences(self.library, &self.fences);
        for &(_, event) in self.events.iter().chain(self.forks.iter()) {
            if let Err(error) = unsafe { self.library.cuda_event_destroy(event) } {
                tracing::error!(%error, "destroying target stage chain event");
            }
        }
        release_fences(self.library, &self.fences);
    }
}

fn drain_fences(library: &NativeLibrary, fences: &[Fence]) {
    for fence in fences {
        // SAFETY: this chain retains every fence stream and event until all drain.
        crate::shared::decode_graph::fatal_drain(
            unsafe { library.cuda_stream_synchronize(fence.stream) }, "target stage chain fence");
    }
}

fn release_fences(library: &NativeLibrary, fences: &[Fence]) {
    for fence in fences {
        // SAFETY: created by this chain; all fences drained before destruction.
        unsafe {
            let _ = library.cuda_stream_destroy(fence.stream);
            for event in fence.events { let _ = library.cuda_event_destroy(event); }
        }
    }
}

/// Device-ordered passes: marks fence `slot` (0 or 1) at the current chain
/// head, without a host wait. A later [`fence_wait`] returns once every
/// stage chained before the mark has completed, so host staging those stages
/// uploaded from may be rewritten.
pub(crate) fn fence_mark(library: &NativeLibrary, slot: usize) -> Result<()> {
    let Some(current) = current() else { return Ok(()) };
    let Some(head) = current.head.get() else { return Ok(()) };
    let fence = current.fences.iter().position(|f| f.device == current.events[head].0)
        .map(|i| (i, current.fences[i]));
    let Some((index, fence)) = fence else { return Ok(()) };
    let previous = library.cuda_get_device()?;
    library.cuda_set_device(fence.device)?;
    // SAFETY: the fence stream and events belong to this chain's device.
    let marked = unsafe {
        library.cuda_stream_wait_event(fence.stream, current.events[head].1)
            .and_then(|()| library.cuda_event_record(fence.events[slot], fence.stream))
    };
    library.cuda_set_device(previous)?;
    marked?;
    let mut slots = current.marked.get();
    slots[slot] = Some(index);
    current.marked.set(slots);
    Ok(())
}

/// Waits for fence `slot` (no-op when it was not marked since the last wait),
/// yielding between polls so another lane on this thread keeps running.
pub(crate) async fn fence_wait(library: &NativeLibrary, slot: usize) -> Result<()> {
    let Some(current) = current() else { return Ok(()) };
    let mut slots = current.marked.get();
    let Some(index) = slots[slot].take() else { return Ok(()) };
    current.marked.set(slots);
    let fence = current.fences[index];
    // The fence stream holds only marks, and the next mark is queued after this
    // wait: the stream is idle exactly when this slot's mark has completed.
    loop {
        let previous = library.cuda_get_device()?;
        library.cuda_set_device(fence.device)?;
        // SAFETY: the fence stream belongs to this chain.
        let ready = unsafe { library.cuda_stream_query(fence.stream) };
        library.cuda_set_device(previous)?;
        if ready? {
            return Ok(());
        }
        tokio::task::yield_now().await;
    }
}

pub(crate) struct ChainHandle<'a> {
    library: &'a NativeLibrary,
    current: Current,
}
impl<'a> ChainHandle<'a> {
    /// Poll `future` with this chain installed as the current scope. Cancellation
    /// drains its recorded work before the future releases borrowed storage.
    pub fn scope<F: Future>(self, future: F) -> ChainScope<'a, F> {
        ChainScope { library: self.library, current: self.current, future, drain_on_drop: false }
    }
}

/// Whether target passes order their stages on the device (default) or drain
/// each stage on the host (`CUTEAFD_STAGE_CHAIN=0`, the pre-v14 behaviour).
pub(crate) fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("CUTEAFD_STAGE_CHAIN").map_or(true, |v| v != "0"))
}

/// Whether chained V4.1 passes also drop the host waits that only kept
/// copy-engine transfers from queuing behind unresolved events
/// (`CUTEAFD_V41_DEVICE=1`, PLAN.md device-driven exchange, stage D3): peer
/// transfers become SM copies ordered by device events and the host enqueues
/// the next stage at once. Off by default; the default path is unchanged.
pub(crate) fn device_enabled() -> bool {
    device_setting() > 0
}

/// Whether V4.1 remote verification waves also use the device-driven Spark
/// exchange (`CUTEAFD_V41_DEVICE=1`; `chain` keeps them on the host path).
pub(crate) fn device_exchange_enabled() -> bool {
    device_setting() > 1
}

/// `CUTEAFD_V41_DEVICE_LANES=1`: device-ordered passes also while both lanes
/// are busy (off: a lane runs device-ordered only while the other is idle).
pub(crate) fn device_with_lanes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("CUTEAFD_V41_DEVICE_LANES").as_deref(), Ok("1" | "on")))
}

/// `CUTEAFD_V41_STAGING_FENCE=0`: device-ordered passes do not wait for the
/// previous layer's staged uploads before preparing the next layer (stages
/// whose pinned staging differs by layer keep one region per layer,
/// `LayerStaging`), so the host can queue the pass ahead of the GPU.
pub(crate) fn staging_fence() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("CUTEAFD_V41_STAGING_FENCE").as_deref(), Ok("0" | "off")))
}

fn device_setting() -> u8 {
    static SETTING: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *SETTING.get_or_init(|| {
        let setting = match std::env::var("CUTEAFD_V41_DEVICE").as_deref() {
            Ok("1" | "on" | "true") => 2,
            Ok("chain") => 1,
            _ => 0,
        };
        if setting > 0 {
            tracing::info!(exchange = setting > 1, "V4.1 device-ordered passes: chained stages enqueue without host waits");
        }
        setting
    })
}

/// Device-ordered passes run under this watchdog: a pass still pending after
/// `CUTEAFD_V41_DEVICE_WATCHDOG_S` (default 60) seconds is a device wait that
/// never released (draining it would block forever), so the process logs every
/// device lane's sequences and exits instead of holding the GPUs.
pub(crate) async fn watchdog<F: Future>(future: F) -> F::Output {
    if !device_enabled() {
        return future.await;
    }
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let limit = *LIMIT.get_or_init(|| std::env::var("CUTEAFD_V41_DEVICE_WATCHDOG_S").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(60));
    let mut future = std::pin::pin!(future);
    tokio::select! {
        output = &mut future => output,
        () = tokio::time::sleep(std::time::Duration::from_secs(limit)) => {
            tracing::error!(limit_s = limit, lanes = %cuteafd_transport::expert::device_stuck_report(),
                "device-ordered pass stuck: exiting (a device wait never released)");
            std::process::exit(70);
        }
    }
}

/// Inside a chain scope with [`device_enabled`]: stages that used to wait on
/// the host for a producer stream join the chain instead.
pub(crate) fn deferred() -> bool {
    device_enabled() && CURRENT.with(|c| c.borrow().as_ref().is_some_and(|c| c.device))
}

pub(crate) struct ChainScope<'a, F> {
    library: &'a NativeLibrary,
    current: Current,
    future: F,
    drain_on_drop: bool,
}
impl<F: Future> Future for ChainScope<'_, F> {
    type Output = F::Output;
    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<F::Output> {
        // SAFETY: the future is never moved after pinning; only borrowed in place.
        let this = unsafe { self.get_unchecked_mut() };
        // Arm before polling so a panic also drains before future-owned staging
        // is dropped. Unpolled scopes have not submitted any work.
        this.drain_on_drop = true;
        let previous = CURRENT.with(|c| c.replace(Some(this.current.clone())));
        struct Restore(Option<Current>);
        impl Drop for Restore {
            fn drop(&mut self) { let previous = self.0.take(); CURRENT.with(|c| *c.borrow_mut() = previous); }
        }
        let _restore = Restore(previous);
        // SAFETY: `future` remains pinned with its containing scope.
        let result = unsafe { Pin::new_unchecked(&mut this.future) }.poll(context);
        if result.is_ready() {
            // The target-pass wrapper performs its existing fallible drain.
            this.drain_on_drop = false;
        }
        result
    }
}
impl<F> Drop for ChainScope<'_, F> {
    fn drop(&mut self) {
        if !self.drain_on_drop { return; }
        self.current.fork.set(None);
        if let Some(head) = self.current.head.replace(None) {
            // SAFETY: the pass retains the chain's events and library. All
            // recorded producers are already submitted (the Spark proxy runs
            // independently), so draining needs no progress from this future.
            // CUDA permits event synchronization from another current device;
            // this leaves the caller's device and thread-local scope unchanged.
            crate::shared::decode_graph::fatal_drain(
                unsafe { self.library.cuda_event_synchronize(self.current.events[head].1) },
                "cancelled target stage chain");
        }
        // Field destruction follows this body. Inner stream guards still drain
        // submissions not yet recorded in the chain before releasing their owners.
    }
}

fn current() -> Option<Current> {
    CURRENT.with(|c| c.borrow().clone())
}

/// Whether stages are currently device-ordered instead of host-drained.
pub(crate) fn active() -> bool {
    CURRENT.with(|c| c.borrow().is_some())
}

/// Order `stream` after the chain head. Call before a stage's first enqueue.
/// # Safety
/// `stream` is a live stream on the chain's device.
pub(crate) unsafe fn join(library: &NativeLibrary, stream: *mut c_void) -> Result<()> {
    let Some(current) = current() else { return Ok(()) };
    // A full join ends any fork window: later stages see the whole chain.
    current.fork.set(None);
    let Some(head) = current.head.get() else { return Ok(()) };
    // Cross-device event waits are permitted; the event lives on its own device.
    unsafe { library.cuda_stream_wait_event(stream, current.events[head].1) }
}

/// Mark a fork on `stream`: work queued so far on it (for example the query
/// stage's normalized layer input) is what fork joiners depend on, while the
/// stage's later work (the query projections) may overlap theirs.
/// # Safety
/// `stream` is a live stream on a chain device, already joined to the head.
pub(crate) unsafe fn mark_fork(library: &NativeLibrary, stream: *mut c_void) -> Result<()> {
    let Some(current) = current() else { return Ok(()) };
    let device = library.cuda_get_device()?;
    let Some(index) = current.forks.iter().position(|&(d, _)| d == device) else { return Ok(()) };
    unsafe { library.cuda_event_record(current.forks[index].1, stream)?; }
    current.fork.set(Some(index));
    Ok(())
}

/// Join the current fork instead of the head, so this stage overlaps the rest
/// of the forking stage. Its `finish` still merges the head. Without a fork
/// (none marked since the last full join) this is [`join`].
/// # Safety
/// The stage reads only outputs complete at the fork (plus its own state).
pub(crate) unsafe fn join_fork(library: &NativeLibrary, stream: *mut c_void) -> Result<()> {
    let Some(current) = current() else { return Ok(()) };
    match current.fork.get() {
        Some(fork) => unsafe { library.cuda_stream_wait_event(stream, current.forks[fork].1) },
        None => unsafe { join(library, stream) },
    }
}

/// Complete a stage: record the chain head on `stream` inside a scope,
/// otherwise drain `stream` on the host exactly as before.
/// # Safety
/// `stream` holds this stage's queued work on the chain's device.
pub(crate) unsafe fn finish(library: &NativeLibrary, stream: *mut c_void) -> Result<()> {
    let Some(current) = current() else { return unsafe { library.cuda_stream_synchronize(stream) } };
    let device = library.cuda_get_device()?;
    let Some(index) = current.events.iter().position(|&(d, _)| d == device) else {
        // A device outside this chain: complete the stage on the host.
        return unsafe { library.cuda_stream_synchronize(stream) };
    };
    // Merge: the stream first waits for the previous head, so parallel branches
    // (window, compressor, index projection) all precede the new head.
    if let Some(head) = current.head.get() {
        unsafe { library.cuda_stream_wait_event(stream, current.events[head].1)?; }
    }
    unsafe { library.cuda_event_record(current.events[index].1, stream)?; }
    current.head.set(Some(index));
    Ok(())
}

/// Cooperative form of [`finish`]: record inside a scope, otherwise yield
/// until the stream completes.
/// # Safety
/// Same as [`finish`].
pub(crate) async unsafe fn finish_cooperative(stream: &LoadStream<'_>) -> Result<()> {
    if active() {
        unsafe { finish(stream.library, stream.raw) }
    } else {
        stream.wait().await
    }
}

/// Host wait for all chained work before a host-synchronous operation (legacy
/// stream copies do not order with the non-blocking stage streams).
pub(crate) fn settle(library: &NativeLibrary) -> Result<()> {
    let Some(current) = current() else { return Ok(()) };
    if let Some(head) = current.head.get() {
        unsafe { library.cuda_event_synchronize(current.events[head].1)?; }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::memory::{HostAllocation, device::{Allocation, Device, Stream}};
    use cuteafd_ffi::test_support::CudaStreamGate;
    use std::sync::atomic::Ordering;
    use std::task::Waker;
    use std::time::Duration;

    #[test]
    #[ignore = "requires an allocated CPU build slot and explicit NVMe fixture directory"]
    fn failed_chain_drops_abort_before_release() -> Result<()> {
        use cuteafd_ffi::native_library_lifetime_fixture::Fixture;
        use std::os::unix::process::ExitStatusExt;
        const CHILD: &str = "CUTEAFD_FATAL_CHAIN_CHILD";
        if let Some(path) = std::env::var_os(CHILD) {
            let fixture = Fixture::build()?;
            std::fs::write(std::env::var("CUTEAFD_FATAL_CHAIN_EVIDENCE")?,
                fixture.directory().join("events").to_str().unwrap())?;
            let library = fixture.load()?;
            let allocation = Allocation::new(Device { library: &library, id: 0 }, 256)?;
            fixture.configure_pack(&library, 0, 1)?;
            if path == "fence" { fixture.configure_drain_after(&library, 1)?; }
            let chain = StageChain {
                library: &library,
                events: vec![(0, std::ptr::null_mut())].into(),
                head: Rc::new(Cell::new((path != "fence").then_some(0))),
                forks: vec![(0, std::ptr::null_mut())].into(), fork: Rc::new(Cell::new(None)),
                fences: if path == "fence" {
                    vec![Fence { device: 0, stream: std::ptr::null_mut(),
                        events: [std::ptr::null_mut(); 2] }; 2]
                } else { vec![] }.into(),
                marked: Rc::new(Cell::new([None; 2])), device: Cell::new(true),
            };
            if path == "cancel" {
                let mut scope = Box::pin(chain.handle().scope(async move {
                    let _allocation = allocation;
                    std::future::pending::<()>().await;
                }));
                let mut context = Context::from_waker(Waker::noop());
                assert!(scope.as_mut().poll(&mut context).is_pending());
                drop(scope);
            } else {
                drop(chain);
                drop(allocation);
            }
            panic!("failed chain drain returned without abort");
        }
        let root = std::path::PathBuf::from(std::env::var("CUTEAFD_NATIVE_LIFETIME_FIXTURE_DIR")?);
        std::fs::create_dir_all(&root)?;
        for path in ["head", "fence", "cancel"] {
            let evidence = root.join(format!("fatal-chain-{}-{path}", std::process::id()));
            let status = std::process::Command::new(std::env::current_exe()?)
                .args(["--exact", "shared::memory::chain::tests::failed_chain_drops_abort_before_release", "--ignored"])
                .env(CHILD, path).env("CUTEAFD_FATAL_CHAIN_EVIDENCE", &evidence).status()?;
            assert_eq!(status.signal(), Some(libc::SIGABRT));
            let events = std::fs::read_to_string(std::fs::read_to_string(evidence)?)?;
            let expected = if path == "fence" { "DSS" } else { "DS" };
            assert_eq!(events, expected, "abort must precede any event/fence/storage release");
        }
        Ok(())
    }

    /// Models staging returned to a pool by an inner future's destructor. It
    /// must already be reusable when that destructor starts, not just after the
    /// surrounding pass owner eventually drops or starts another execution.
    struct ReuseStaging<'s, 'a> {
        staging: &'s mut HostAllocation<'a>,
        final_stream: &'s Stream<'a>,
        observed: Rc<Cell<Option<(bool, i32)>>>,
    }
    impl Drop for ReuseStaging<'_, '_> {
        fn drop(&mut self) {
            let device = self.final_stream.device;
            let current = device.library.cuda_get_device().unwrap();
            // SAFETY: the borrowed stream outlives this staging lease.
            let ready = device.run(|| unsafe {
                device.library.cuda_stream_query(self.final_stream.raw)
            }).unwrap();
            self.observed.set(Some((ready, current)));
            // On the broken implementation, record the premature release
            // without introducing a test-side race with the pending DMA.
            if ready { self.staging.bytes_mut().fill(0xee); }
        }
    }

    fn queued_reuse(device_ids: &[i32], unwind: bool) -> Result<()> {
        const BYTES: usize = 4096;
        // SAFETY: the explicitly selected native library is retained by every
        // allocation, stream and chain until their destruction.
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let mut staging = HostAllocation::new(&library, BYTES)?;
        let mut landing = device_ids.iter().map(|_| HostAllocation::new(&library, BYTES))
            .collect::<Result<Vec<_>>>()?;
        let devices = device_ids.iter().map(|&id| Device { library: &library, id }).collect::<Vec<_>>();
        let output = devices.iter().map(|&device| Allocation::new(device, BYTES)).collect::<Result<Vec<_>>>()?;
        let streams = devices.iter().map(|&device| Stream::new(device)).collect::<Result<Vec<_>>>()?;
        let chain = StageChain::on_devices(&library, device_ids)?;
        let mut context = Context::from_waker(Waker::noop());
        let addresses = (staging.buffer.ptr, output.iter().map(|v| v.buffer.ptr).collect::<Vec<_>>());

        for seed in [37u8, 149] {
            staging.bytes_mut().fill(seed);
            for target in &mut landing { target.bytes_mut().fill(0); }
            // SAFETY: this stream remains live until after the gate is released
            // and drained; an independent OS thread supplies the release.
            let gate = unsafe { CudaStreamGate::new(&library, streams[0].raw)? };
            let release = gate.release_handle();
            let observed = Rc::new(Cell::new(None));
            let lease = ReuseStaging { staging: &mut staging, final_stream: streams.last().unwrap(),
                observed: observed.clone() };
            let work = async {
                for ((stream, allocation), target) in streams.iter().zip(&output).zip(&landing) {
                    stream.device.run(|| {
                        // SAFETY: pinned source/destination and device storage
                        // are retained across cancellation; each stage joins its
                        // predecessor before copying and records after its D2H.
                        unsafe {
                            join(&library, stream.raw)?;
                            library.copy_host_buffer_h2d_async(allocation.buffer, lease.staging.buffer, BYTES, stream.raw)?;
                            library.copy_d2h_host_buffer_async(target.buffer, allocation.buffer, BYTES, stream.raw)?;
                            finish(&library, stream.raw)
                        }
                    })?;
                }
                std::future::pending::<()>().await;
                drop(lease);
                Ok::<_, anyhow::Error>(())
            };
            let future = Box::pin(chain.handle().scope(work));
            std::thread::scope(|threads| -> Result<()> {
                let (begin_release, wait_for_drop) = std::sync::mpsc::channel();
                let release = release.clone();
                threads.spawn(move || {
                    // The timeout also releases the gate if setup/assertions
                    // fail before announcing cancellation.
                    let _ = wait_for_drop.recv_timeout(Duration::from_secs(2));
                    std::thread::sleep(Duration::from_millis(30));
                    release.store(true, Ordering::Release);
                });
                let mut future = future;
                assert!(future.as_mut().poll(&mut context).is_pending());
                assert!(!active(), "poll leaked the chain scope");
                assert_eq!(library.cuda_get_device()?, 0);
                begin_release.send(())?;
                if unwind {
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                        let _future = future;
                        panic!("exercise scoped future destruction while unwinding");
                    }));
                    assert!(result.is_err());
                } else {
                    drop(future);
                }
                // The lease's destructor executes before thread::scope joins
                // its release thread, so that join cannot mask a missing drain.
                assert_eq!(observed.get(), Some((true, 0)), "staging was released before the GPU chain completed");
                Ok(())
            })?;
            assert!(chain.head.get().is_none());
            assert!(chain.fork.get().is_none());
            assert!(!active());
            assert_eq!(library.cuda_get_device()?, 0);
            assert!(staging.bytes().iter().all(|&v| v == 0xee));
            for target in &landing { assert!(target.bytes().iter().all(|&v| v == seed)); }
            assert_eq!(staging.buffer.ptr, addresses.0);
            assert_eq!(output.iter().map(|v| v.buffer.ptr).collect::<Vec<_>>(), addresses.1);
            // Deliberately do not call chain.drain(): the next iteration reuses
            // the same host/device storage immediately after cancellation.
            drop(gate);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, libcudart.so.13 and one CUDA device"]
    fn cancelled_stage_chain_drains_before_staging_reuse() -> Result<()> {
        queued_reuse(&[0], false)
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, libcudart.so.13 and two CUDA devices"]
    fn cancelled_cross_device_stage_chain_drains_before_staging_reuse() -> Result<()> {
        queued_reuse(&[0, 1], false)
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, libcudart.so.13 and one CUDA device"]
    fn unwinding_stage_chain_drains_before_staging_reuse() -> Result<()> {
        queued_reuse(&[0], true)
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, libcudart.so.13 and one CUDA device"]
    fn completed_and_unpolled_stage_scopes_leave_the_explicit_drain() -> Result<()> {
        // SAFETY: all native owners are destroyed before the library.
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let device = Device { library: &library, id: 0 };
        let mut source = HostAllocation::new(&library, 4096)?;
        let target = Allocation::new(device, 4096)?;
        let output = HostAllocation::new(&library, 4096)?;
        let stream = Stream::new(device)?;
        let chain = StageChain::new(&library)?;
        source.bytes_mut().fill(81);
        // SAFETY: the gate is released and drained before the stream drops.
        let gate = unsafe { CudaStreamGate::new(&library, stream.raw)? };
        let release = gate.release_handle();
        std::thread::scope(|threads| -> Result<()> {
            let (begin_release, wait_for_check) = std::sync::mpsc::channel();
            let worker_release = release.clone();
            threads.spawn(move || {
                let _ = wait_for_check.recv_timeout(Duration::from_secs(2));
                worker_release.store(true, Ordering::Release);
            });
            let mut work = Box::pin(chain.handle().scope(async {
                // SAFETY: all buffers and the stream remain live until the
                // explicit drain below, including the normal Ready path.
                unsafe {
                    join(&library, stream.raw)?;
                    library.copy_host_buffer_h2d_async(target.buffer, source.buffer, 4096, stream.raw)?;
                    library.copy_d2h_host_buffer_async(output.buffer, target.buffer, 4096, stream.raw)?;
                    finish(&library, stream.raw)
                }
            }));
            let mut context = Context::from_waker(Waker::noop());
            match work.as_mut().poll(&mut context) {
                Poll::Ready(result) => result?,
                Poll::Pending => anyhow::bail!("submission unexpectedly suspended"),
            }
            drop(work);
            drop(chain.handle().scope(std::future::pending::<()>()));
            assert!(!release.load(Ordering::Acquire), "normal/unpolled drop waited for queued work");
            assert!(chain.head.get().is_some(), "normal/unpolled drop consumed the explicit drain");
            assert!(!active());
            begin_release.send(())?;
            chain.drain()?;
            assert!(output.bytes().iter().all(|&value| value == 81));
            Ok(())
        })
    }
}
