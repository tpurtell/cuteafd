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
use cuteafd_ffi::{event_waiters::BlockingEventWaiter, NativeLibrary};
use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll};

type FenceWaiter = Rc<RefCell<Option<BlockingEventWaiter>>>;

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
    /// Device-ordered passes: per device a stream and two blocking events marking
    /// "everything chained so far" ([`fence_mark`]), and which device marked each.
    fences: Rc<[Fence]>,
    marked: Rc<Cell<[Option<usize>; 2]>>,
    /// Lazily-spawned sleeping OS waiter thread for host [`fence_wait`].
    waiter: FenceWaiter,
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
    static CURRENT: RefCell<Option<Current>> = const { RefCell::new(None) };
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
    waiter: FenceWaiter,
    device: Cell<bool>,
}
impl<'a> StageChain<'a> {
    pub fn library(&self) -> &'a NativeLibrary { self.library }
    /// Whether the next scoped pass runs device-ordered; default on.
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
                fences.push(Fence { device, stream: library.cuda_stream_create()?,
                    events: [library.cuda_event_create_blocking()?, library.cuda_event_create_blocking()?] });
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
            marked: Rc::new(Cell::new([None; 2])), waiter: Rc::new(RefCell::new(None)),
            device: Cell::new(true) })
    }
    /// An owned handle that can wrap a future borrowing the chain's owner.
    pub fn handle(&self) -> ChainHandle<'a> {
        ChainHandle { library: self.library, current: Current {
            events: self.events.clone(), head: self.head.clone(),
            forks: self.forks.clone(), fork: self.fork.clone(), fences: self.fences.clone(),
            marked: self.marked.clone(), waiter: self.waiter.clone(), device: self.device.get() } }
    }
    /// Host wait for everything recorded so far, then forget the head. Call
    /// after the pass (or an aborted pass) before any unscoped consumer.
    pub fn drain(&self) -> Result<()> {
        self.fork.set(None);
        self.marked.set([None; 2]);
        if let Some(head) = self.head.replace(None) {
            unsafe { self.library.cuda_event_synchronize(self.events[head].1)?; }
        }
        if let Some(waiter) = self.waiter.borrow().as_ref() {
            waiter.flush()?;
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
        // Join the sleeping fence waiter before destroying any fence events.
        self.waiter.borrow_mut().take();
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

/// Orders `stream` on the device after fence `slot` (no-op when unmarked),
/// with no host wait or polling. Use this whenever the consumer is a CUDA stream.
/// # Safety
/// `stream` is a live stream on a participating device.
#[allow(dead_code)]
pub(crate) unsafe fn fence_wait_stream(library: &NativeLibrary, slot: usize, stream: *mut c_void) -> Result<()> {
    let Some(current) = current() else { return Ok(()) };
    let mut slots = current.marked.get();
    let Some(index) = slots[slot].take() else { return Ok(()) };
    current.marked.set(slots);
    let event = current.fences[index].events[slot];
    // SAFETY: `stream` and `event` are live; cross-device event waits are valid.
    unsafe { library.cuda_stream_wait_event(stream, event) }
}

/// Waits on the host for fence `slot` (no-op when it was not marked since the
/// last wait), using a blocking `cudaEventSynchronize` on a dedicated sleeping
/// waiter thread instead of busy-polling `cudaStreamQuery`, so another lane on
/// this runtime thread keeps running with zero idle CPU burn.
pub(crate) async fn fence_wait(library: &NativeLibrary, slot: usize) -> Result<()> {
    let Some(current) = current() else { return Ok(()) };
    let mut slots = current.marked.get();
    let Some(index) = slots[slot].take() else { return Ok(()) };
    current.marked.set(slots);
    let fence = current.fences[index];
    let event = fence.events[slot];
    // Fast path: one non-blocking query avoids a waiter-thread handoff when
    // the fence event already completed while waiting on Spark exchange.
    let previous = library.cuda_get_device()?;
    library.cuda_set_device(fence.device)?;
    // SAFETY: `event` belongs to `fence.device` and was recorded by `fence_mark`.
    let ready = unsafe { library.cuda_event_query(event) };
    library.cuda_set_device(previous)?;
    if ready? {
        return Ok(());
    }
    let (tx, rx) = tokio::sync::oneshot::channel();
    {
        let mut slot_waiter = current.waiter.borrow_mut();
        let waiter = match slot_waiter.as_ref() {
            Some(waiter) => waiter,
            None => slot_waiter.insert(BlockingEventWaiter::new(library, "chain-fence-wait")?),
        };
        // SAFETY: `StageChain` retains `event` until `drain()` / `Drop` flushes
        // and joins `waiter`, and callers do not re-mark `slot` before this wait
        // (or cancellation drain) returns.
        unsafe {
            waiter.submit(fence.device, event, Box::new(move |result| { let _ = tx.send(result); }))?;
        }
    }
    rx.await.map_err(|_| anyhow::anyhow!("fence waiter dropped before completion"))?
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

/// Device-ordered passes run under this watchdog: a pass still pending after
/// `CUTEAFD_V41_DEVICE_WATCHDOG_S` (default 60) seconds is a device wait that
/// never released (draining it would block forever), so the process logs it
/// and exits instead of holding the GPUs.
pub(crate) async fn watchdog<F: Future>(future: F) -> F::Output {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let limit = *LIMIT.get_or_init(|| std::env::var("CUTEAFD_V41_DEVICE_WATCHDOG_S").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(60));
    let mut future = std::pin::pin!(future);
    tokio::select! {
        output = &mut future => output,
        () = tokio::time::sleep(std::time::Duration::from_secs(limit)) => {
            tracing::error!(limit_s = limit, "device-ordered pass stuck: exiting (a device wait never released)");
            std::process::exit(70);
        }
    }
}

/// Inside a device-ordered chain scope: stages that used to wait on the host
/// for a producer stream join the chain instead. Scopes are device-ordered
/// unless the pass owner turned it off ([`StageChain::set_device_order`]:
/// both V4.1 lanes busy).
pub(crate) fn deferred() -> bool {
    CURRENT.with(|c| c.borrow().as_ref().is_some_and(|c| c.device))
}

/// Non-blocking `cudaEventQuery` check: returns `true` when `event` on
/// `device` has not yet signaled on the GPU (zero host sync).
/// # Safety
/// `event` is a live CUDA event owned by `device`.
pub(crate) unsafe fn event_unresolved(
    library: &NativeLibrary,
    device: i32,
    event: *mut c_void,
) -> Result<bool> {
    let previous = library.cuda_get_device()?;
    if previous != device {
        library.cuda_set_device(device)?;
    }
    // SAFETY: caller guarantees `event` is live on `device`.
    let ready = unsafe { library.cuda_event_query(event) };
    if previous != device {
        library.cuda_set_device(previous)?;
    }
    Ok(!ready?)
}

/// Non-blocking check for whether the current device-ordered chain has a
/// recorded head event that has not yet signaled on the GPU (zero host sync).
pub(crate) fn head_unresolved(library: &NativeLibrary) -> Result<bool> {
    let Some(current) = current() else { return Ok(false) };
    if !current.device {
        return Ok(false);
    }
    let Some(head) = current.head.get() else { return Ok(false) };
    let (device, event) = current.events[head];
    // SAFETY: `StageChain` retains all chain events for the scope's lifetime.
    unsafe { event_unresolved(library, device, event) }
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
        if let Some(waiter) = self.current.waiter.borrow().as_ref() {
            crate::shared::decode_graph::fatal_drain(waiter.flush(), "cancelled target stage chain fence waiter");
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

/// Complete an issued stage either on the chain or at its existing host wait.
/// # Safety
/// Same as [`finish`]; the caller retains all stage storage through completion.
#[inline(always)]
pub(crate) async unsafe fn finish_or_wait(stream: &LoadStream<'_>, chain_only: bool) -> Result<()> {
    if chain_only {
        // SAFETY: forwarded from this function's contract.
        unsafe { finish(stream.library, stream.raw) }
    } else {
        stream.wait().await
    }
}

/// The device whose event is the chain head (None outside a scope or before
/// any stage finished).
pub(crate) fn head_device() -> Option<i32> {
    let current = current()?;
    current.head.get().map(|head| current.events[head].0)
}

/// Records the current chain head into `event` (no host wait): another
/// chain's stages ordered after it with [`depend`] follow everything chained
/// here so far. No-op outside a scope or before any stage finished.
/// # Safety
/// `event` is a live event on the chain's head device.
pub(crate) unsafe fn record_into(library: &NativeLibrary, event: *mut c_void) -> Result<bool> {
    let Some(current) = current() else { return Ok(false) };
    let Some(head) = current.head.get() else { return Ok(false) };
    let device = current.events[head].0;
    let fence = current.fences.iter().find(|f| f.device == device).copied()
        .ok_or_else(|| anyhow::anyhow!("chain has no fence stream on device {device}"))?;
    let previous = library.cuda_get_device()?;
    library.cuda_set_device(device)?;
    // SAFETY: the fence stream and head event belong to this chain's device;
    // `event` lives on the same device (caller contract).
    let recorded = unsafe { library.cuda_stream_wait_event(fence.stream, current.events[head].1)
        .and_then(|()| library.cuda_event_record(event, fence.stream)) };
    library.cuda_set_device(previous)?;
    recorded.map(|()| true)
}

/// Orders the current chain after `event` (recorded by another chain with
/// [`record_into`]): every later stage of this scope joins a head that
/// includes it. No host wait.
/// # Safety
/// `event` is a live event recorded on the chain's current device.
#[cfg(test)]
thread_local! {
    /// Test-only negative control: `depend` leaves an open fork untouched (the
    /// pre-fix behaviour, hunt C #1).
    static SKIP_FORK_FOLD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
pub(crate) unsafe fn depend(library: &NativeLibrary, event: *mut c_void) -> Result<()> {
    let Some(current) = current() else { return Ok(()) };
    let device = library.cuda_get_device()?;
    let index = current.events.iter().position(|&(d, _)| d == device)
        .ok_or_else(|| anyhow::anyhow!("chain has no event on device {device}"))?;
    let fence = current.fences.iter().find(|f| f.device == device).copied()
        .ok_or_else(|| anyhow::anyhow!("chain has no fence stream on device {device}"))?;
    // SAFETY: the fence stream and events belong to this chain's device.
    unsafe {
        if let Some(head) = current.head.get() {
            library.cuda_stream_wait_event(fence.stream, current.events[head].1)?;
        }
        library.cuda_stream_wait_event(fence.stream, event)?;
        library.cuda_event_record(current.events[index].1, fence.stream)?;
        // An open fork window must include the dependency too: fork joiners
        // (`join_fork`) wait on the fork event, not the head, so without this
        // they would overtake `event` (hunt C #1). The fork event becomes
        // "old fork point + event", recorded on this device's fork slot.
        #[cfg(test)]
        let skip = SKIP_FORK_FOLD.with(|skip| skip.get());
        #[cfg(not(test))]
        let skip = false;
        if let Some(fork) = current.fork.get().filter(|_| !skip) {
            let slot = current.forks.iter().position(|&(d, _)| d == device)
                .ok_or_else(|| anyhow::anyhow!("chain has no fork event on device {device}"))?;
            library.cuda_stream_wait_event(fence.stream, current.forks[fork].1)?;
            library.cuda_event_record(current.forks[slot].1, fence.stream)?;
            current.fork.set(Some(slot));
        }
    }
    current.head.set(Some(index));
    Ok(())
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
            let allocation = Allocation::new(Device { library: &library, id: 0 }, 256, crate::shared::memory::TEST)?;
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
                marked: Rc::new(Cell::new([None; 2])), waiter: Rc::new(RefCell::new(None)),
                device: Cell::new(true),
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
        let mut staging = HostAllocation::new(&library, BYTES, crate::shared::memory::TEST)?;
        let mut landing = device_ids.iter().map(|_| HostAllocation::new(&library, BYTES, crate::shared::memory::TEST))
            .collect::<Result<Vec<_>>>()?;
        let devices = device_ids.iter().map(|&id| Device { library: &library, id }).collect::<Vec<_>>();
        let output = devices.iter().map(|&device| Allocation::new(device, BYTES, crate::shared::memory::TEST)).collect::<Result<Vec<_>>>()?;
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
        let mut source = HostAllocation::new(&library, 4096, crate::shared::memory::TEST)?;
        let target = Allocation::new(device, 4096, crate::shared::memory::TEST)?;
        let output = HostAllocation::new(&library, 4096, crate::shared::memory::TEST)?;
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

    /// Hunt C #1: two slots' chains, unit B's chain depends on unit A's
    /// recorded publication while B's fork window is open. A `join_fork`
    /// stage of B must wait for A (gated here). The negative control skips the
    /// fold, and that stage then completes while A is still held.
    fn fork_join_follows_a_cross_slot_dependency(fold: bool) -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let device = Device { library: &library, id: 0 };
        let [a, b_main, b_fork] = [Stream::new(device)?, Stream::new(device)?, Stream::new(device)?];
        let published = library.cuda_event_create_ordering()?;
        // Recorded right after the fork joiner's wait: the stage's own
        // ordering, before its `finish` merges the head (which includes A).
        let joined = library.cuda_event_create_ordering()?;
        let chains = [StageChain::new(&library)?, StageChain::new(&library)?];
        for chain in &chains { chain.set_device_order(true); }
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        // Warm every call ungated (lazy module loading) before gating A.
        runtime.block_on(chains[0].handle().scope(async { unsafe {
            super::finish(&library, a.raw)?; super::record_into(&library, published).map(|_| ())
        } }))?;
        unsafe { library.cuda_stream_synchronize(a.raw)?; }
        library.cuda_device_synchronize()?;
        let gate = unsafe { CudaStreamGate::new(&library, a.raw)? };
        // Unit A (slot 0): a stage held by the gate, then its publication.
        runtime.block_on(chains[0].handle().scope(async { unsafe {
            super::join(&library, a.raw)?;
            super::finish(&library, a.raw)?;
            super::record_into(&library, published).map(|_| ())
        } }))?;
        // Unit B (slot 1): marks a fork, depends on A's publication, then a
        // fork joiner stage.
        SKIP_FORK_FOLD.with(|skip| skip.set(!fold));
        let queued = runtime.block_on(chains[1].handle().scope(async { unsafe {
            super::join(&library, b_main.raw)?;
            super::mark_fork(&library, b_main.raw)?;
            super::finish(&library, b_main.raw)?;
            super::depend(&library, published)?;
            super::join_fork(&library, b_fork.raw)?;
            library.cuda_event_record(joined, b_fork.raw)?;
            super::finish(&library, b_fork.raw)
        } }));
        SKIP_FORK_FOLD.with(|skip| skip.set(false));
        queued?;
        let deadline = std::time::Instant::now() + Duration::from_secs(if fold { 2 } else { 10 });
        let mut done = false;
        while std::time::Instant::now() < deadline {
            if unsafe { library.cuda_event_query(joined)? } { done = true; break; }
            std::thread::sleep(Duration::from_millis(10));
        }
        drop(gate);
        for chain in &chains { chain.drain()?; }
        unsafe { library.cuda_event_destroy(published)?; library.cuda_event_destroy(joined)?; }
        assert_eq!(done, !fold, "the fork joiner {} while unit A's publication was held",
            if done { "completed" } else { "stayed pending" });
        Ok(())
    }

    #[test]
    fn fence_wait_and_fence_wait_stream_avoid_stream_query_spin() -> Result<()> {
        let (_directory, fixture) = crate::shared::memory::cpu_stream_fixture()?;
        let library = fixture.load()?;
        let device = Device { library: &library, id: 0 };
        let producer = Stream::new(device)?;
        let consumer = Stream::new(device)?;
        let chain = StageChain::new(&library)?;
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        runtime.block_on(chain.handle().scope(async {
            // SAFETY: streams and library live through the scope and drain.
            unsafe {
                join(&library, producer.raw)?;
                finish(&library, producer.raw)?;
                fence_mark(&library, 0)?;
                fence_wait_stream(&library, 0, consumer.raw)?;
                fence_mark(&library, 1)?;
            }
            fence_wait(&library, 1).await?;
            // Unmarked slot is a no-op.
            fence_wait(&library, 1).await?;
            Ok::<_, anyhow::Error>(())
        }))?;
        chain.drain()?;
        // Verify fence_wait does not spin on cuda_stream_query + yield_now.
        let src = include_str!("chain.rs");
        let body = &src[src.find("pub(crate) async fn fence_wait(").unwrap()..];
        let body = &body[..body.find("pub(crate) struct ChainHandle").unwrap()];
        assert!(!body.contains("cuda_stream_query"), "fence_wait must not poll cuda_stream_query");
        assert!(!body.contains("yield_now"), "fence_wait must not spin with yield_now");
        assert!(body.contains("BlockingEventWaiter"));
        Ok(())
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and a GPU"]
    fn fork_joiners_wait_for_a_cross_slot_dependency() -> Result<()> { fork_join_follows_a_cross_slot_dependency(true) }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and a GPU"]
    fn unfolded_fork_joiners_overtake_a_cross_slot_dependency() -> Result<()> { fork_join_follows_a_cross_slot_dependency(false) }
}