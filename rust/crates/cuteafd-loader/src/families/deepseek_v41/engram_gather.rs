//! Bounded background engram gathers on the shared [`TablePipeline`]: reusable
//! staging slots and request-owned cancellation.
use crate::{EngramBatchStaging, EngramGatherView, EngramTable, GatherFailure, GatherLease, GatherPoll, GatherTicket,
    shared::token_table::{TablePipeline, TableShape}};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{EngramBatch, ENGRAM_ROWS};
use std::sync::Arc;
use std::time::Instant;

pub const ENGRAM_DEVICE_ROW_BYTES: usize = 24 * (256 + 8 + 512) + 1;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum EngramStagingError {
    #[error("unsupported Engram shape: {units} units of {rows} rows")]
    Shape { units: usize, rows: usize },
    #[error("Engram shape needs {required} host slots, got {available}")]
    Slots { required: usize, available: usize },
}

/// Shape is a requirement, not a memory tuning knob. The executor can begin k
/// units before a slot returns, and layer 1 follows begin immediately. Start a
/// full k-unit burst ahead (including the initial burst before any begin).
/// Every unit's L1 and L14 gathers start together at prepare (the worker runs
/// every queued L1 before any L14) and each holds its staging until its H2D
/// completes: k active plus k lookahead units, two layers each, live maximum
/// = 4k. A stalled copy may attain this maximum.
/// OS fault latency is unbounded: readiness remains a correctness fence even
/// with the mandatory burst of lead time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EngramStagingShape { rows: usize, units: usize }
impl EngramStagingShape {
    pub fn new(rows: usize, units: usize) -> std::result::Result<Self, EngramStagingError> {
        if !(1..=4096).contains(&rows) || !(1..=4).contains(&units) {
            return Err(EngramStagingError::Shape { rows, units });
        }
        Ok(Self { rows, units })
    }
    pub fn rows(self) -> usize { self.rows }
    pub fn units(self) -> usize { self.units }
    pub fn decode_slots(self) -> usize { 2 * 2 }
    pub fn decode_rows(self) -> usize { self.rows.min(256) }
    pub fn decode_host_resident_bytes(self) -> usize {
        self.decode_slots() * EngramBatchStaging::storage_bytes(self.decode_rows()).expect("validated shape")
    }
    pub fn decode_host_workspace_bytes(self) -> usize {
        3 * EngramBatchStaging::worker_bytes(self.decode_rows()).expect("validated shape")
    }
    pub fn decode_host_bytes(self) -> usize { self.decode_host_resident_bytes() + self.decode_host_workspace_bytes() }
    pub fn prefetch_units(self) -> usize { self.units }
    pub fn host_slots(self) -> usize { TableShape::new(self.rows, self.units, 2).expect("validated shape").slots().expect("validated shape") }
    pub fn host_resident_bytes(self) -> usize {
        self.host_slots() * EngramBatchStaging::storage_bytes(self.rows).expect("validated shape")
    }
    pub fn host_workspace_bytes(self) -> usize { 3 * EngramBatchStaging::worker_bytes(self.rows).expect("validated shape") }
    /// Exact persistent pool plus preparation and two bounded finish workspaces.
    /// Decode has its own pool/worker, so concurrent decode cannot starve prefill.
    pub fn host_bytes(self) -> usize { self.host_resident_bytes() + self.host_workspace_bytes() }
    /// Preload runs before either staging pool or its lazy worker scratch is
    /// allocated. Keep their complete working set and the shared read owner.
    pub fn preload_budget(self, available: u64) -> u64 {
        available.saturating_sub((self.host_bytes() + self.decode_host_bytes() + self.cache_bytes(264)
            + crate::formats::mapped_table::TABLE_REACTOR_WORKSPACE_BYTES) as u64)
    }
    pub fn device_bytes(self) -> usize { self.units * self.rows * ENGRAM_DEVICE_ROW_BYTES }
    /// Row-cache slots per layer: prefill holds the look-ahead's window (two
    /// full prefills of this shape, [`window_units`](super::engram_lookahead::window_units))
    /// plus the unit being gathered and committed; decode holds two verify
    /// windows of every decode lane. Rows of 24 per token.
    pub fn cache_shares(self) -> [usize; 2] {
        let prefill = (super::engram_lookahead::window_units(self.units) + 1) * self.rows;
        let decode = 2 * cuteafd_core::V41_VERIFY_ROWS * crate::plan::decode_lanes::MAX_DECODE_LANES as usize;
        [prefill * 24, decode * 24]
    }
    /// Both layers' row caches for an encoding's row width (weight + scale bytes).
    pub fn cache_bytes(self, row_width: usize) -> usize {
        2 * crate::formats::mapped_table::row_cache::RowCache::host_bytes(row_width, self.cache_shares())
    }
    pub fn validate_slots(self, available: usize) -> std::result::Result<(), EngramStagingError> {
        if available != self.host_slots() { return Err(EngramStagingError::Slots {
            required: self.host_slots(), available }); }
        Ok(())
    }
}

/// Recorded only when the runtime's timing trace is enabled at submission.
pub type EngramGatherTiming = crate::GatherTiming;

/// One layer's gather: the exact immutable request batches, in wave order.
pub struct EngramGatherJob {
    table: Arc<EngramTable>,
    layer: usize,
    batches: Vec<Arc<EngramBatch>>,
}

/// A pinned gather source. After enqueue it is retained by the gatherer until
/// its H2D event completes, independently of wave cancellation or gate lifetime.
pub struct EngramGatherLease(GatherLease<EngramBatchStaging, EngramGatherJob>);
impl EngramGatherLease {
    pub fn view(&self) -> Result<EngramGatherView<'_>> {
        self.0.slot().context("engram gather lease is empty")?.view()
    }
    /// Exact immutable batches represented by the concatenated output rows.
    pub fn batches(&self) -> &[Arc<EngramBatch>] {
        &self.0.job().batches
    }
    pub fn timing(&self) -> Option<&EngramGatherTiming> {
        self.0.timing()
    }
    pub fn host_buffer(&self) -> Option<cuteafd_ffi::CuteafdHostBuffer> { self.0.slot()?.host_buffer() }
    /// # Safety
    /// All reads of this lease's pinned bytes are enqueued on the supplied stream.
    pub unsafe fn copied_on(&mut self, stream: *mut std::ffi::c_void) -> Result<()> {
        // SAFETY: caller provides the stream containing every queued reader.
        unsafe { self.0.slot_mut().context("empty gather lease")?.copied_on(stream) }
    }
}

pub enum EngramGatherPoll {
    Pending,
    Cancelled,
    Ready(EngramGatherLease),
}
/// Independent of scheduler slot reuse; drop cancels queued/in-progress work.
/// An in-progress OS page fault cannot be interrupted, but its result is discarded.
pub struct EngramGatherTicket(GatherTicket<EngramBatchStaging, EngramGatherJob, anyhow::Error>);
impl EngramGatherTicket {
    pub fn cancel(&mut self) {
        self.0.cancel();
    }
    /// Sleep until completion without retaining a request-bank borrow.
    pub async fn wait(&mut self) -> Result<EngramGatherPoll> {
        Self::result(self.0.wait().await)
    }
    /// Nonblocking polling for a CUDA or scheduler thread; consume at most once.
    pub fn poll(&mut self) -> Result<EngramGatherPoll> {
        Self::result(self.0.poll())
    }
    fn result(result: std::result::Result<GatherPoll<GatherLease<EngramBatchStaging, EngramGatherJob>>,
        GatherFailure<anyhow::Error>>) -> Result<EngramGatherPoll> {
        match result {
            Ok(GatherPoll::Pending) => Ok(EngramGatherPoll::Pending),
            Ok(GatherPoll::Cancelled) => Ok(EngramGatherPoll::Cancelled),
            Ok(GatherPoll::Ready(lease)) => Ok(EngramGatherPoll::Ready(EngramGatherLease(lease))),
            Err(GatherFailure::Job(error)) => Err(error),
            Err(GatherFailure::Worker(error)) => Err(anyhow::Error::new(error).context("engram gather")),
        }
    }
}

pub struct EngramGatherer {
    worker: TablePipeline<EngramBatchStaging, EngramGatherJob>,
    capacity: usize,
}
impl EngramGatherer {
    /// The slot pool bounds queued, in-flight and completed-but-unconsumed storage.
    pub fn new(slots: usize, capacity: usize, staging_budget: usize) -> Result<Self> {
        ensure!(
            slots > 0 && slots <= 32,
            "engram gather pool requires 1..32 slots"
        );
        let bytes = EngramBatchStaging::storage_bytes(capacity)?
            .checked_mul(slots)
            .context("engram gather pool budget overflow")?;
        ensure!(
            bytes <= staging_budget,
            "engram gather pool exceeds host staging budget"
        );
        let staging = (0..slots).map(|_| EngramBatchStaging::new(capacity)).collect::<Result<Vec<_>>>()?;
        Self::with_staging(capacity, staging, false)
    }
    pub fn new_pinned(library: &cuteafd_ffi::NativeLibrary, shape: EngramStagingShape) -> Result<Self> {
        let _scope = cuteafd_ffi::memory_ledger::scope("v41/engram-prefill");
        let staging = (0..shape.host_slots()).map(|_| EngramBatchStaging::new_pinned(library, shape.rows))
            .collect::<Result<Vec<_>>>()?;
        shape.validate_slots(staging.len())?;
        Self::with_staging(shape.rows, staging, false)
    }
    pub fn new_decode(library: &cuteafd_ffi::NativeLibrary, shape: EngramStagingShape) -> Result<Self> {
        let _scope = cuteafd_ffi::memory_ledger::scope("v41/engram-decode");
        let staging = (0..shape.decode_slots()).map(|_| EngramBatchStaging::new_pinned(library, shape.decode_rows()))
            .collect::<Result<Vec<_>>>()?;
        Self::with_staging(shape.decode_rows(), staging, true)
    }
    pub fn copy_capacity_stalled(&self) -> Result<bool> {
        self.worker.copy_capacity_stalled()
    }
    fn with_staging(capacity: usize, staging: Vec<EngramBatchStaging>, decode: bool) -> Result<Self> {
        use crate::formats::mapped_table::ReadPriority;
        let priority = if decode { ReadPriority::Decode } else { ReadPriority::Prefill };
        let worker = TablePipeline::split("engram-gather", staging,
            move |staging: &mut EngramBatchStaging, job: &EngramGatherJob| {
                let started = Instant::now();
                let batches: Vec<_> = job.batches.iter().map(Arc::as_ref).collect();
                Ok((staging.prepare(&job.table, &batches, job.layer, priority)?, started))
            },
            |staging: &mut EngramBatchStaging, job: &EngramGatherJob, (prepared, started)| {
                staging.finish(&job.table, prepared)?;
                let view = staging.view()?;
                let bytes = view.weights.len() + view.scales.len();
                job.table.weights().stats().record_gather(view.rows * 24, bytes, started.elapsed(), [0; 3]);
                Ok(())
            }, EngramBatchStaging::copy_ready)
        .context("starting engram gather worker")?;
        Ok(Self { worker, capacity })
    }
    pub fn retain_copy(&self, lease: EngramGatherLease) {
        self.worker.retain_copy(lease.0);
    }
    /// Called from existing request I/O polling, including cancelled waves.
    pub fn reclaim_copies(&self) -> Result<()> {
        self.worker.reclaim_copies()
    }
    /// Submit early when decode/prefill/verification token hashes become available.
    /// None is bounded backpressure; it performs no mapped reads or blocking wait.
    pub fn try_submit(
        &self,
        table: Arc<EngramTable>,
        batches: &[Arc<EngramBatch>],
        layer: usize,
    ) -> Result<Option<EngramGatherTicket>> {
        self.reclaim_copies()?;
        let job = self.job(table, batches, layer)?;
        // Layer-1 gathers run before every queued layer-14 one: an L14 issued
        // early never delays a unit's L1 deadline.
        Ok(self.worker.try_submit(job, layer == 0)?.map(EngramGatherTicket))
    }
    /// Retain the immutable job while sleeping for bounded staging admission.
    pub async fn submit(&self, table: Arc<EngramTable>, batches: &[Arc<EngramBatch>],
        layer: usize) -> Result<EngramGatherTicket> {
        self.reclaim_copies()?;
        let job = self.job(table, batches, layer)?;
        Ok(EngramGatherTicket(self.worker.submit(job, layer == 0).await?))
    }
    fn job(&self, table: Arc<EngramTable>, batches: &[Arc<EngramBatch>],
        layer: usize) -> Result<EngramGatherJob> {
        ensure!(
            !batches.is_empty() && batches.len() <= 16,
            "invalid engram request batch count"
        );
        let rows = batches.iter().try_fold(0usize, |sum, batch| {
            sum.checked_add(batch.hashes().len())
                .context("engram batch length overflow")
        })?;
        ensure!(
            rows > 0 && rows <= self.capacity,
            "engram gather exceeds row capacity"
        );
        ensure!(
            table.weights().rows() == *ENGRAM_ROWS.get(layer).context("invalid engram layer")?,
            "engram gather table belongs to another layer"
        );
        Ok(EngramGatherJob { table, layer, batches: batches.to_vec() })
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    use std::{cell::RefCell, rc::Rc, sync::atomic::{AtomicBool, Ordering}};

    struct CopyLease { ready: Arc<AtomicBool>, drops: Rc<RefCell<Vec<usize>>>, id: usize }
    impl Drop for CopyLease { fn drop(&mut self) { self.drops.borrow_mut().push(self.id); } }

    #[test]
    fn preload_budget_reserves_every_staging_owner_and_lazy_workspace() -> Result<()> {
        for (rows, units) in [(2752, 3), (4096, 2), (2048, 4)] {
            let shape = EngramStagingShape::new(rows, units)?;
            let future = (shape.host_bytes() + shape.decode_host_bytes() + shape.cache_bytes(264)
                + crate::formats::mapped_table::TABLE_REACTOR_WORKSPACE_BYTES) as u64;
            assert_eq!(shape.preload_budget(future - 1), 0);
            assert_eq!(shape.preload_budget(future), 0);
            assert_eq!(shape.preload_budget(future + 123), 123);
        }
        Ok(())
    }

    #[test]
    fn engram_copy_pending_pool_reclaims_on_completion_with_bounded_polls() -> Result<()> {
        for k in 1..=4 {
            let shape = EngramStagingShape::new(((8192usize.div_ceil(k.max(2)) + 63) / 64 * 64), k)?;
            let drops = Rc::new(RefCell::new(Vec::new()));
            let mut pool = crate::shared::token_table::CopyPool::new(shape.host_slots());
            let mut events = Vec::new();
            for id in 0..shape.host_slots() {
                let ready = Arc::new(AtomicBool::new(false));
                events.push(ready.clone()); pool.retain(CopyLease { ready, drops: drops.clone(), id });
            }
            assert!(pool.capacity_stalled(false));
            let mut polls = 0;
            pool.reclaim(|l| { polls += 1; Ok(l.ready.load(Ordering::Acquire)) })?;
            assert!(pool.capacity_stalled(false) && drops.borrow().is_empty());
            events[k].store(true, Ordering::Release);
            pool.reclaim(|l| { polls += 1; Ok(l.ready.load(Ordering::Acquire)) })?;
            assert!(!pool.capacity_stalled(true), "one completion admits another gather");
            assert_eq!(*drops.borrow(), [k]);
            assert!(polls <= 2 * shape.host_slots());
            for event in events { event.store(true, Ordering::Release); }
            pool.reclaim(|l| Ok(l.ready.load(Ordering::Acquire)))?;
            let mut released = drops.borrow().clone(); released.sort_unstable();
            assert_eq!(released, (0..shape.host_slots()).collect::<Vec<_>>());
        }
        Ok(())
    }

    fn complete<T>(future: impl std::future::Future<Output = T>) -> T {
        use std::task::{Context, Poll, Wake, Waker};
        struct ThreadWake(std::thread::Thread);
        impl Wake for ThreadWake { fn wake(self: Arc<Self>) { self.0.unpark(); } }
        let waker = Waker::from(Arc::new(ThreadWake(std::thread::current())));
        let mut future = std::pin::pin!(future);
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        loop {
            if let Poll::Ready(value) = future.as_mut().poll(&mut Context::from_waker(&waker)) { return value; }
            assert!(Instant::now() < deadline, "fixture wait timed out");
            std::thread::park_timeout(deadline.saturating_duration_since(Instant::now()));
        }
    }
    type TestTicket = GatherTicket<usize, (usize, usize), ()>;
    type TestLease = GatherLease<usize, (usize, usize)>;
    fn ready(ticket: &mut TestTicket) -> TestLease {
        match complete(ticket.wait()) { Ok(GatherPoll::Ready(lease)) => lease, _ => panic!("gather not ready") }
    }
    struct PendingCopy { lease: TestLease, done: Arc<AtomicBool> }

    #[test]
    fn engram_lifecycle_exact_pool_and_adversarial_layer14_admission() -> Result<()> {
        for k in 1..=4 {
            let rows = 8192usize.div_ceil(k.max(2)).div_ceil(64) * 64;
            let shape = EngramStagingShape::new(rows, k)?;
            let n = shape.host_slots();
            assert_eq!(n, 4 * k);
            assert_eq!(shape.host_resident_bytes(), n * EngramBatchStaging::storage_bytes(rows)?);
            assert_eq!(shape.host_bytes(), shape.host_resident_bytes() + 3 * (rows * 24 * 1024 + (1 << 20)));
            assert_eq!(shape.decode_host_bytes(), 4 * EngramBatchStaging::storage_bytes(256)? + 3 * (256 * 24 * 1024 + (1 << 20)));
            assert_eq!(shape.device_bytes(), k * rows * ENGRAM_DEVICE_ROW_BYTES);
            assert_eq!(shape.validate_slots(n - 1), Err(EngramStagingError::Slots { required: n, available: n - 1 }));
            let order = Arc::new(std::sync::Mutex::new(Vec::new()));
            let observed = order.clone();
            let worker = crate::GatherWorker::new("engram-lifecycle", (0..n).collect::<Vec<_>>(),
                move |_: &mut usize, job: &(usize, usize)| {
                    observed.lock().unwrap().push(*job); Ok::<_, ()>(())
                })?;
            let mut copies = crate::shared::token_table::CopyPool::new(n);
            let mut events = Vec::new();
            // k active units retain both layers until their copies complete;
            // k lookahead units hold both layers ready (L14 issued with L1).
            for unit in 0..k {
                for layer in 0..2 {
                    let mut ticket = worker.try_submit_with((unit, layer), false, layer == 0)?.unwrap();
                    let done = Arc::new(AtomicBool::new(false)); events.push(done.clone());
                    copies.retain(PendingCopy { lease: ready(&mut ticket), done });
                }
            }
            let mut future = (k..2*k).flat_map(|unit| [(unit, 0), (unit, 1)])
                .map(|(unit, layer)| worker.try_submit_with((unit, layer), false, layer == 0).map(Option::unwrap))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let future = future.iter_mut().map(ready).collect::<Vec<_>>();
            assert!(worker.try_submit((99, 0), false)?.is_none(), "computed live maximum attained");
            drop(future); // cancelled ready lookahead returns its slots
            for event in &events { event.store(true, Ordering::Release); }
            copies.reclaim(|copy| { let _ = copy.lease.slot(); Ok(copy.done.load(Ordering::Acquire)) })?;
            // Exactly n slots returned: every admission succeeds once, n+1 fails.
            let held = (0..n).map(|i| worker.try_submit((100+i, 0), false).map(Option::unwrap))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            assert!(worker.try_submit((999, 0), false)?.is_none());
            drop(held);

        }
        Ok(())
    }

    #[test]
    fn engram_cancel_queued_gathering_ready_and_copy_states_return_once() -> Result<()> {
        for k in 1..=4 {
            let n = EngramStagingShape::new(8192usize.div_ceil(k.max(2)).div_ceil(64) * 64, k)?.host_slots();
            // Block the first gather deterministically; the second is queued.
            let (started, seen) = std::sync::mpsc::sync_channel(1);
            let (release, wait) = std::sync::mpsc::sync_channel(1);
            let mut first = true;
            let worker = crate::GatherWorker::new("engram-cancel-states", (0..n).collect::<Vec<_>>(),
                move |_: &mut usize, _: &(usize, usize)| {
                    if first { first = false; let _ = started.send(()); let _ = wait.recv(); }
                    Ok::<_, ()>(())
                })?;
            let mut gathering = worker.try_submit((0, 0), false)?.unwrap();
            seen.recv_timeout(std::time::Duration::from_secs(5))?;
            let mut queued = worker.try_submit((1, 0), false)?.unwrap();
            gathering.cancel(); queued.cancel(); drop(gathering); drop(queued);
            release.send(())?;
            // FIFO barrier proves both cancelled leases have finished recycling.
            let mut barrier = worker.try_submit((2, 0), false)?.unwrap(); drop(ready(&mut barrier));
            let mut ticket = worker.try_submit((3, 0), false)?.unwrap();
            drop(ready(&mut ticket)); // ready cancellation
            let mut ticket = worker.try_submit((4, 0), false)?.unwrap();
            let done = Arc::new(AtomicBool::new(false));
            let mut copies = crate::shared::token_table::CopyPool::new(n);
            copies.retain(PendingCopy { lease: ready(&mut ticket), done: done.clone() });
            copies.reclaim(|copy| Ok(copy.done.load(Ordering::Acquire)))?;
            assert_eq!(copies.len(), 1, "cancel cannot return a copy-pending source");
            done.store(true, Ordering::Release);
            copies.reclaim(|copy| Ok(copy.done.load(Ordering::Acquire)))?;
            assert!(copies.is_empty());
            copies.reclaim(|_| panic!("copy-done lease returned twice"))?;
            let mut tickets = (0..n).map(|i| worker.try_submit((i, 1), false).map(Option::unwrap))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            assert!(worker.try_submit((n, 1), false)?.is_none());
            let mut ids = tickets.iter_mut().map(|t| *ready(t).slot().unwrap()).collect::<Vec<_>>();
            ids.sort_unstable(); assert_eq!(ids, (0..n).collect::<Vec<_>>());
        }
        Ok(())
    }

    #[test]
    fn engram_copy_completion_wakes_real_gather_admission() -> Result<()> {
        use std::{future::Future, task::{Context, Poll, Wake, Waker}};
        struct Counter(std::sync::atomic::AtomicUsize);
        impl Wake for Counter { fn wake(self: Arc<Self>) { self.0.fetch_add(1, Ordering::Release); } }
        for k in 1..=4 {
            let n = EngramStagingShape::new(8192usize.div_ceil(k.max(2)).div_ceil(64) * 64, k)?.host_slots();
            let worker = crate::GatherWorker::new("engram-copy-admit", (0..n).collect::<Vec<_>>(),
                |_: &mut usize, _: &(usize, usize)| Ok::<_, ()>(()))?;
            let mut copies = crate::shared::token_table::CopyPool::new(n);
            let event = Arc::new(AtomicBool::new(false));
            for i in 0..n {
                let mut ticket = worker.try_submit((i, 0), false)?.unwrap();
                copies.retain(PendingCopy { lease: ready(&mut ticket), done: event.clone() });
            }
            let wake = Arc::new(Counter(std::sync::atomic::AtomicUsize::new(0)));
            let waker = Waker::from(wake.clone());
            let mut cx = Context::from_waker(&waker);
            let mut admission = std::pin::pin!(worker.submit((n, 1), false));
            assert!(admission.as_mut().poll(&mut cx).is_pending());
            assert!(copies.capacity_stalled(worker.capacity_available()?));
            event.store(true, Ordering::Release);
            let mut polls = 0;
            copies.reclaim(|copy| { polls += 1; Ok(copy.done.load(Ordering::Acquire)) })?;
            assert!(polls <= n && wake.0.load(Ordering::Acquire) > 0);
            let Poll::Ready(ticket) = admission.as_mut().poll(&mut cx) else { panic!("copy completion did not admit gather") };
            let mut ticket = ticket?; drop(ready(&mut ticket));
        }
        Ok(())
    }

    #[test]
    fn engram_cancel_mid_copy_mixed_lookahead_admission_progress() -> Result<()> {
        use std::{future::Future, task::{Context, Poll, Wake, Waker}};
        struct Counter(std::sync::atomic::AtomicUsize);
        impl Wake for Counter { fn wake(self: Arc<Self>) { self.0.fetch_add(1, Ordering::Release); } }
        for k in 1..=4 {
            let n = EngramStagingShape::new(8192usize.div_ceil(k.max(2)).div_ceil(64) * 64, k)?.host_slots();
            let worker = crate::GatherWorker::new("engram-mixed-copy-admit", (0..n).collect::<Vec<_>>(),
                |_: &mut usize, _: &(usize, usize)| Ok::<_, ()>(()))?;
            let mut copies = crate::shared::token_table::CopyPool::new(n);
            let event = Arc::new(AtomicBool::new(false));
            let mut cancelled = worker.try_submit((0, 0), false)?.unwrap();
            copies.retain(PendingCopy { lease: ready(&mut cancelled), done: event.clone() });
            cancelled.cancel(); drop(cancelled); // lease now belongs to the copy pool
            let mut held = Vec::new();
            for i in 1..n {
                let mut ticket = worker.try_submit((i, 0), false)?.unwrap();
                held.push(ready(&mut ticket)); // refill lookahead and active ready gathers
            }
            let wake = Arc::new(Counter(std::sync::atomic::AtomicUsize::new(0)));
            let waker = Waker::from(wake.clone());
            let mut cx = Context::from_waker(&waker);
            let mut admission = std::pin::pin!(worker.submit((n, 1), false));
            assert!(admission.as_mut().poll(&mut cx).is_pending());
            assert!(copies.capacity_stalled(worker.capacity_available()?));
            copies.reclaim(|copy| Ok(copy.done.load(Ordering::Acquire)))?;
            assert_eq!(wake.0.load(Ordering::Acquire), 0, "worker cannot notify CUDA completion");
            event.store(true, Ordering::Release);
            let mut polls = 0;
            copies.reclaim(|copy| { polls += 1; Ok(copy.done.load(Ordering::Acquire)) })?;
            assert!(polls <= n && wake.0.load(Ordering::Acquire) > 0);
            assert!(!copies.capacity_stalled(worker.capacity_available()?));
            let Poll::Ready(ticket) = admission.as_mut().poll(&mut cx) else { panic!("mixed admission stuck") };
            let mut ticket = ticket?; drop(ready(&mut ticket));
            // A full worker-only pool must use notifications, not event polling.
            let mut ticket = worker.try_submit((n + 1, 0), false)?.unwrap();
            held.push(ready(&mut ticket));
            assert!(!worker.capacity_available()?);
            assert!(!copies.capacity_stalled(false));
        }
        Ok(())
    }

    /// An L1 gather admitted after a queued L14 runs first: the worker drains
    /// every pending priority (L1) job before any normal (L14) one, so an L14
    /// issued early never delays an L1 deadline. Negative control: without the
    /// priority the queue is FIFO and the L14 runs first.
    #[test]
    fn a_later_l1_gather_runs_before_a_queued_l14() -> Result<()> {
        for prioritized in [true, false] {
            let order = Arc::new(std::sync::Mutex::new(Vec::new()));
            let gate = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
            let started = Arc::new(AtomicBool::new(false));
            let (observed, held, running) = (order.clone(), gate.clone(), started.clone());
            let worker = crate::GatherWorker::new("engram-priority", (0..4usize).collect::<Vec<_>>(),
                move |_: &mut usize, job: &(usize, usize)| {
                    if job.0 == 0 {
                        // The first job holds the worker until both later ones are queued.
                        running.store(true, Ordering::Release);
                        let (lock, open) = &*held;
                        let mut opened = lock.lock().unwrap();
                        while !*opened { opened = open.wait(opened).unwrap(); }
                    }
                    observed.lock().unwrap().push(*job); Ok::<_, ()>(())
                })?;
            let mut busy = worker.try_submit_with((0, 0), false, true)?.unwrap();
            while !started.load(Ordering::Acquire) { std::thread::yield_now(); }
            let mut l14 = worker.try_submit_with((1, 1), false, false)?.unwrap();
            let mut l1 = worker.try_submit_with((2, 0), false, prioritized)?.unwrap();
            { let (lock, open) = &*gate; *lock.lock().unwrap() = true; open.notify_all(); }
            for ticket in [&mut busy, &mut l1, &mut l14] { let _ = ready(ticket); }
            let order = order.lock().unwrap().clone();
            let at = |job| order.iter().position(|x| *x == job).unwrap();
            assert_eq!(at((2, 0)) < at((1, 1)), prioritized, "prioritized={prioritized}: {order:?}");
        }
        Ok(())
    }

    #[test]
    fn engram_shape_refusal_is_typed_and_never_shrinks_lookahead() {
        for (rows, units) in [(0, 2), (4097, 2), (8192, 1), (4096, 0), (4096, 5)] {
            assert_eq!(EngramStagingShape::new(rows, units), Err(EngramStagingError::Shape { rows, units }));
        }
        for k in 1..=4 { assert_eq!(EngramStagingShape::new(4096, k).unwrap().prefetch_units(), k); }
    }
}
