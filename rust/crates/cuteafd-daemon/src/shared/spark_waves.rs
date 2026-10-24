//! The shared Spark expert exchange (#717 items 1–2, PLAN "Prefill contract"):
//! one persistent transport worker per link carries every wave, decode and
//! prefill alike, and the GPU-issuing thread never waits for it.
//!
//! A link's [`Waves`] owns a worker thread for the link's life (until it is
//! stopped to hand the transport back for a reset or terminal teardown). The
//! worker owns the link's QPs and CQs, so no queue pair is shared between
//! threads, waves stay FIFO on every rank, and there are no locks (one
//! channel in). It meets the engine's streams through monotone u32 counters
//! in pinned, device-mapped host memory, compared cyclically ([`reached`]):
//!
//! - `staged[slot]`: after a wave's route ids, gate weights and expert input
//!   rows copy into its slot's pinned staging on the engine stream, the
//!   stream writes the wave's sequence (`cuStreamWriteValue32`, fenced behind
//!   the copies). The worker polls the word; once it reaches the wave, the
//!   staging is complete. One word per slot: a slot's waves are ordered on
//!   the GPU (below), so the word never goes backwards.
//! - `landed`: once every rank's rows arrived, the worker stores the wave's
//!   sequence from the host. [`Waves::land`] queues `cuStreamWaitValue32
//!   (landed >= sequence)` on the consuming stream and returns at once; for
//!   ranks whose rows arrive in pinned host memory (no GPU landing) it also
//!   queues their upload from the slot's landing staging into the planes,
//!   behind the wait. Later work there (the reduce) sees the planes.
//! - `consumed[slot]`: after the reduce, the consuming stream writes the
//!   slot's use count; the worker posts the slot's next wave (whose rows
//!   overwrite the planes and landing staging) only once it is reached.
//!
//! The worker queues no GPU work at all. A stream memory wait stalls its
//! hardware queue, and streams share queues, so a wait released by GPU work
//! (an upload on a copy stream) could deadlock behind itself; here every wait
//! is released by a host store alone. Events are not used for these counters:
//! with a non-blocking `land`, the engine can re-record a slot's event for a
//! later wave before the worker queried the earlier record, losing a
//! completion. Counters never alias.
//!
//! Ordering without host waits. Wave `k` takes slot `k % waves`. Before its
//! staging copies, [`Waves::send`] queues a wait for `landed >= k - waves`
//! on the sending stream: the slot's previous wave (and with it every rank's
//! receipt of its request, which the NIC may have been reading from the same
//! staging) is complete before the copies overwrite it. A decode step
//! therefore queues all its layers back to back (send, land, reduce, next
//! layer...): the stream stops at each wait until the worker publishes,
//! while the host is already ahead. A link's sends, lands and consumes are
//! issued in one stream order (one stream, or streams ordered by events).
//!
//! Failure. A transport error records a terminal [`health`] failure, after
//! which every request refuses. The worker then releases every wait the
//! engine queued or will queue, by storing each sequence it is sent, until
//! the link stops it. The planes of those waves are garbage: every consumer
//! checks [`healthy`] after its step's sampling sync, so a released step
//! fails instead of producing tokens.
//!
//! [`health`]: cuteafd_transport::health
use crate::shared::experts::staging::RouteStaging;
use crate::shared::memory::HostAllocation;
use crate::shared::spark_intake::{copy_parallel, upload_offset, MAX_INTAKE_RANKS};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, CuteafdHostBuffer, NativeLibrary};
use cuteafd_transport::expert::{SparkExperts, SparkExpertWave, WaveReceipt};
use cuteafd_transport::{ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, StagingArena};
use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Builds a wave's request on the transport worker from its staged route
/// entries and its hidden payload (a view of the slot's staging arena).
pub(crate) type BuildRequest = Box<dyn FnOnce(Vec<ExpertProtocolV2RouteEntry>, bytes::Bytes)
    -> Result<ExpertProtocolV2Request> + Send>;

/// A wave handed to the worker; land it with [`Waves::land`], oldest first.
#[derive(Debug)]
#[must_use = "every sent wave must be landed in dispatch order"]
pub(crate) struct WaveTicket {
    sequence: u32,
    slot: usize,
    rows: usize,
}

impl WaveTicket {
    /// The return slot this wave's planes land in.
    pub(crate) fn slot(&self) -> usize { self.slot }
    /// The wave's rows.
    pub(crate) fn rows(&self) -> usize { self.rows }
}

/// Typed failures of the exchange.
#[derive(Debug, thiserror::Error)]
pub(crate) enum WaveError {
    #[error("Spark expert wire failed: {0}")]
    Failed(String),
    #[error("Spark exchange worker stopped before wave {0}")]
    Lost(u32),
    #[error("Spark wave {found} landed out of order (expected {expected})")]
    Order { expected: u32, found: u32 },
    #[error("Spark wave admission exceeded: {0} waves outstanding")]
    Admission(usize),
    #[error("Spark wave of {rows} rows exceeds the {capacity}-row return slot")]
    Rows { rows: usize, capacity: usize },
    #[error("return slot {0} takes a new wave before its reduce was marked consumed")]
    Unconsumed(usize),
}

/// Whether cyclic counter `value` has reached `target`.
pub(crate) fn reached(value: u32, target: u32) -> bool {
    (value.wrapping_sub(target) as i32) >= 0
}

/// Whether the expert wire is still usable: false once any exchange recorded
/// a terminal failure, after which waits were released over garbage planes.
/// Every step checks it after its sampling sync, before using its results.
pub(crate) fn healthy() -> Result<()> {
    match cuteafd_transport::health::failure_reason() {
        Some(reason) => Err(WaveError::Failed(reason).into()),
        None => Ok(()),
    }
}

/// One cache line per counter (written by different agents).
const LINE: usize = 64;

/// A counter word of the control block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Counter { Landed, Staged(usize), Consumed(usize) }

impl Counter {
    fn index(self, slots: usize) -> usize {
        match self {
            Self::Landed => 0,
            Self::Staged(slot) => { assert!(slot < slots, "staged counter of slot {slot} past {slots}"); 1 + 2 * slot }
            Self::Consumed(slot) => { assert!(slot < slots, "consumed counter of slot {slot} past {slots}"); 2 + 2 * slot }
        }
    }
    fn words(slots: usize) -> usize { 1 + 2 * slots }
}

/// The control block's words as raw addresses: pinned, device-mapped host
/// memory owned by the [`Waves`], which joins the worker before freeing it.
#[derive(Clone, Copy)]
pub(crate) struct Words { base: usize, slots: usize }

impl Words {
    pub(crate) fn ptr(self, counter: Counter) -> *mut u32 {
        (self.base + counter.index(self.slots) * LINE) as *mut u32
    }
    pub(crate) fn load(self, counter: Counter) -> u32 {
        // SAFETY: an aligned u32 of the live control block, accessed only
        // atomically (host atomics here, stream memory operations on the GPU).
        unsafe { (*self.ptr(counter).cast::<AtomicU32>()).load(Ordering::Acquire) }
    }
    pub(crate) fn store(self, counter: Counter, value: u32) {
        // SAFETY: as `load`.
        unsafe { (*self.ptr(counter).cast::<AtomicU32>()).store(value, Ordering::Release) }
    }
}

/// Where a wave's request comes from.
pub(crate) enum Source {
    /// Staged by the engine stream into the slot's arena; built on the worker.
    Staged { layout: RouteStaging, build: BuildRequest },
    /// Built on the host already (warm-ups, gates): posted as is.
    Built(ExpertProtocolV2Request),
}

pub(crate) struct Job {
    sequence: u32,
    slot: usize,
    rows: usize,
    source: Source,
    /// The slot's use count the worker waits for before posting (planes free).
    consumed: u32,
    sent: Instant,
}

/// A command to the worker.
pub(crate) enum Command {
    Wave(Job),
}

/// Host seconds of one wave on the worker (the transport-timing profile).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct WaveTimes {
    /// From the engine's send until its staged copies completed.
    pub staged: f64,
    /// Request build and post.
    pub posted: f64,
    /// From the post until the last rank's rows arrived (and were copied).
    pub received: f64,
}

/// What the worker needs from the GPU and the transport; [`Native`] in
/// serving, a fake in the CPU tests.
pub(crate) trait Backend {
    type Wave;
    /// The current value of `counter` (never blocks).
    fn counter(&self, counter: Counter) -> u32;
    /// Builds (from its slot's staging) and posts the job's request.
    fn post(&mut self, job: Job) -> Result<Self::Wave>;
    /// One progress pass over the oldest wave in `slot` of `rows` rows:
    /// copies arrived rows of pinned ranks into the slot's landing staging.
    /// `true` once every rank's rows are in place.
    fn poll(&mut self, wave: &mut Self::Wave, slot: usize, rows: usize) -> Result<bool>;
    /// Stores `landed = sequence` (also used to release waits after a failure).
    fn publish(&mut self, sequence: u32);
}

struct Inflight<W> {
    sequence: u32,
    slot: usize,
    rows: usize,
    wave: W,
    posted: Instant,
    times: WaveTimes,
}

/// With waves outstanding the worker spins this long between progress, then yields.
const SPIN: Duration = Duration::from_micros(20);
/// With nothing outstanding it keeps polling this long (a decode step's next
/// layer follows within a layer's time), yielding after [`SPIN`], then
/// blocks on its channel.
const IDLE_SPIN: Duration = Duration::from_micros(500);

/// The worker's loop until its channel closes or a transport error. It never
/// blocks on one condition while another could progress: it posts the oldest
/// job once its staging is complete and its slot's planes are free, and
/// progresses the oldest wave meanwhile. Returns the highest sequence sent.
pub(crate) fn run_worker<B: Backend>(backend: &mut B, inbox: &mpsc::Receiver<Command>, timeout: Duration,
    timing: bool) -> (Result<()>, Option<u32>) {
    let mut queued = VecDeque::<Job>::new();
    let mut inflight = VecDeque::<Inflight<B::Wave>>::new();
    let mut highest: Option<u32> = None;
    let mut idle_since: Option<Instant> = None;
    let result = (|| -> Result<()> {
        loop {
            let mut progressed = false;
            let mut closed = false;
            let empty = queued.is_empty() && inflight.is_empty();
            if empty && idle_since.is_some_and(|since| since.elapsed() >= IDLE_SPIN) {
                // Nothing outstanding: block until the engine sends.
                match inbox.recv() {
                    Ok(Command::Wave(job)) => { highest = Some(job.sequence); queued.push_back(job); progressed = true; }
                    Err(_) => return Ok(()),
                }
            }
            loop {
                match inbox.try_recv() {
                    Ok(Command::Wave(job)) => { highest = Some(job.sequence); queued.push_back(job); progressed = true; }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => { closed = true; break; }
                }
            }
            // Post the next job once its rows are staged and its slot's planes are free.
            if let Some(job) = queued.front() {
                let staged = match job.source {
                    Source::Staged { .. } => reached(backend.counter(Counter::Staged(job.slot)), job.sequence),
                    Source::Built(_) => true,
                };
                if staged && reached(backend.counter(Counter::Consumed(job.slot)), job.consumed) {
                    let job = queued.pop_front().expect("front job");
                    let (sequence, slot, rows) = (job.sequence, job.slot, job.rows);
                    let mut times = WaveTimes { staged: job.sent.elapsed().as_secs_f64(), ..Default::default() };
                    let started = Instant::now();
                    let wave = backend.post(job)?;
                    times.posted = started.elapsed().as_secs_f64();
                    inflight.push_back(Inflight { sequence, slot, rows, wave, posted: Instant::now(), times });
                    progressed = true;
                }
            }
            // Progress the oldest wave.
            if let Some(front) = inflight.front_mut() {
                if backend.poll(&mut front.wave, front.slot, front.rows)? {
                    let mut done = inflight.pop_front().expect("front wave");
                    done.times.received = done.posted.elapsed().as_secs_f64();
                    backend.publish(done.sequence);
                    if timing {
                        tracing::info!(target: "cuteafd::timing", sequence = done.sequence, slot = done.slot,
                            rows = done.rows, staged_us = (done.times.staged * 1e6) as u64,
                            posted_us = (done.times.posted * 1e6) as u64,
                            received_us = (done.times.received * 1e6) as u64, "spark exchange wave");
                    }
                    progressed = true;
                }
            }
            let empty = queued.is_empty() && inflight.is_empty();
            if closed {
                // The link closed the channel: at rest, or unwinding with
                // waves it will never land.
                ensure!(empty, "Spark exchange closed with {} waves outstanding", queued.len() + inflight.len());
                return Ok(());
            }
            if progressed {
                idle_since = None;
                continue;
            }
            let since = *idle_since.get_or_insert_with(Instant::now);
            if !empty {
                // Waiting on the GPU (staged copies, a pending reduce) or on a
                // reply, which carries its own transport deadline.
                ensure!(since.elapsed() < timeout, "Spark exchange made no progress for {timeout:?}");
            }
            if since.elapsed() < SPIN { std::hint::spin_loop(); } else { std::thread::yield_now(); }
        }
    })();
    (result, highest)
}

/// After a failure: release every wait queued for `highest` and every wave
/// the engine still sends, so no stream waits forever, until the channel closes.
fn release_until_closed<B: Backend>(backend: &mut B, inbox: &mpsc::Receiver<Command>, highest: Option<u32>) {
    if let Some(highest) = highest { backend.publish(highest); }
    while let Ok(Command::Wave(job)) = inbox.recv() {
        backend.publish(job.sequence);
    }
}

/// The Spark side of the worker: posts a request and progresses its wave.
/// [`SparkExperts`] in serving; an in-process echo in the GPU exchange test.
pub(crate) trait Transport: Send + 'static {
    type Wave: Send;
    fn dispatch(&mut self, request: &ExpertProtocolV2Request) -> Result<Self::Wave>;
    /// One non-blocking pass: arrived rows go to `sink(rank, first row,
    /// payload)`; `Some(landed ranks)` once every rank's rows arrived.
    fn poll(&mut self, wave: &mut Self::Wave, sink: &mut dyn FnMut(usize, u32, &[u8]) -> Result<()>)
        -> Result<Option<u8>>;
    /// The response deadline every wave is held to.
    fn timeout(&self) -> Duration;
    /// Gives up every outstanding wave (after a failure).
    fn abandon(&mut self);
}

impl Transport for SparkExperts {
    type Wave = (SparkExpertWave, WaveReceipt);
    fn dispatch(&mut self, request: &ExpertProtocolV2Request) -> Result<Self::Wave> {
        Ok((self.dispatch_wave(request)?, WaveReceipt::default()))
    }
    fn poll(&mut self, wave: &mut Self::Wave, sink: &mut dyn FnMut(usize, u32, &[u8]) -> Result<()>)
        -> Result<Option<u8>> {
        let (wave, receipt) = wave;
        let done = self.poll_wave(wave, receipt, |rank, first, payload| sink(rank, first, payload.as_ref()))?;
        Ok(done.then_some(receipt.landed))
    }
    fn timeout(&self) -> Duration { SparkExperts::timeout(self) }
    fn abandon(&mut self) { self.abandon_waves("Spark exchange failed with waves in flight"); }
}

/// One rank's landing destination in a slot: device plane, and the pinned
/// staging its rows arrive in when the rank does not land in device memory.
#[derive(Clone, Copy)]
struct Plane { device: usize, host: usize, bytes: usize }

/// The worker's backend over a [`Transport`]: the slots' request staging
/// and landing staging, and the control counters.
struct Native<T> {
    transport: T,
    arenas: Vec<StagingArena>,
    words: Words,
    /// Per slot, per rank.
    planes: Vec<Vec<Plane>>,
    ranks: usize,
    row_bytes: usize,
    /// Bit r: rank r lands in device memory; others arrive pinned.
    landing: u8,
}

impl<T: Transport> Backend for Native<T> {
    type Wave = T::Wave;

    fn counter(&self, counter: Counter) -> u32 { self.words.load(counter) }

    fn post(&mut self, job: Job) -> Result<Self::Wave> {
        let request = match job.source {
            Source::Built(request) => request,
            Source::Staged { layout, build } => {
                let arena = &self.arenas[job.slot];
                // The slot's previous request was released when its wave landed.
                ensure!(arena.writable(), "staging arena {} is still viewed by an unfinished request", job.slot);
                // SAFETY: `staged[slot]` reached this wave, so the engine's
                // copies into the arena are complete; the next copies into it
                // wait (on the GPU) for this wave's landing.
                let staged = unsafe { arena.read(layout.bytes)? };
                let routes = layout.routes(staged)?;
                let wire = arena.payload(2 * layout.route_bytes, layout.wire_bytes)?;
                build(routes, wire)?
            }
        };
        self.transport.dispatch(&request)
    }

    fn poll(&mut self, wave: &mut Self::Wave, slot: usize, rows: usize) -> Result<bool> {
        let (planes, ranks, row_bytes) = (&self.planes[slot], self.ranks, self.row_bytes);
        let plane_bytes = rows * row_bytes;
        let landing = self.landing;
        let done = self.transport.poll(wave, &mut |rank, first, payload| {
            let plane = planes.get(rank).copied().context("rank outside the intake planes")?;
            ensure!(plane.host != 0 && landing & (1 << rank) == 0,
                "rank {rank} arrived in host memory but its link lands it in device memory");
            ensure!(plane_bytes <= plane.bytes, "wave of {rows} rows exceeds its return slot");
            let bytes = payload.len();
            let offset = upload_offset(ranks, rank, first, row_bytes, bytes, plane_bytes)?;
            // SAFETY: the slot's landing staging holds `plane.bytes` for this
            // rank (checked above); the engine stream reads it only after this
            // wave's `landed`, and the slot's next wave is posted only after
            // that read (`consumed`), so nothing else touches it now.
            let target = unsafe { std::slice::from_raw_parts_mut((plane.host + offset) as *mut u8, bytes) };
            copy_parallel(target, payload);
            Ok(())
        })?;
        match done {
            Some(landed) => {
                ensure!(landed == self.landing,
                    "ranks {landed:#b} landed in device memory; the link expects {:#b}", self.landing);
                Ok(true)
            }
            None => Ok(false),
        }
    }

    fn publish(&mut self, sequence: u32) {
        // Ranks in device memory: the NIC's writes completed before their
        // completions were polled (GPUDirect writes ordered for consumers,
        // checked at the landing probe); pinned ranks: copied above.
        if !reached(self.words.load(Counter::Landed), sequence) {
            self.words.store(Counter::Landed, sequence);
        }
    }
}

/// What a link hands its exchange: the intake geometry the worker lands in.
pub(crate) struct Intake {
    pub ranks: usize,
    pub rows: usize,
    pub row_bytes: usize,
    pub waves: usize,
    /// Per slot: rank planes in rank order.
    pub planes: Vec<Vec<CuteafdDeviceBuffer>>,
    /// Bit r: rank r lands in device memory.
    pub landing: u8,
    /// The slot the transport's next wave lands in (its landing rotation).
    pub next_slot: usize,
}

/// A link's persistent exchange: the engine side of the worker.
pub(crate) struct Waves<'a> {
    library: &'a NativeLibrary,
    /// Dropped first: closing it ends the worker, joined in `Drop`.
    jobs: Option<mpsc::Sender<Command>>,
    worker: Option<std::thread::JoinHandle<(Box<dyn std::any::Any + Send>, Result<()>)>>,
    /// The counters; freed only after the worker joined.
    control: HostAllocation<'a>,
    words: Words,
    arenas: Vec<StagingArena>,
    /// Per slot: pinned landing staging of the ranks that arrive in host memory.
    landing_staging: Vec<Option<HostAllocation<'a>>>,
    landing: u8,
    planes: Vec<Vec<CuteafdDeviceBuffer>>,
    /// Per slot: the layout of its last staged wave (route bytes for probes).
    layouts: Vec<Option<RouteStaging>>,
    /// Per slot: landed and not yet marked consumed.
    unread: Vec<bool>,
    /// Per slot: waves consumed from it (the worker posts the slot's next
    /// wave once `consumed[slot]` reaches this).
    uses: Vec<u32>,
    ranks: usize,
    rows: usize,
    row_bytes: usize,
    waves: usize,
    next_slot: usize,
    sequence: u32,
    outstanding: VecDeque<(u32, usize)>,
    /// The workers' installed grouping, fixed at bootstrap.
    grouping: Vec<cuteafd_transport::worker_selection::InstalledExpertGrouping>,
}

impl<'a> Waves<'a> {
    /// Starts the worker on `transport` (bootstrapped; connections open) for
    /// `intake`'s slots, staging each wave's request in `arenas` (one per slot).
    pub(crate) fn start<T: Transport>(library: &'a NativeLibrary, intake: Intake, transport: T,
        grouping: Vec<cuteafd_transport::worker_selection::InstalledExpertGrouping>,
        arenas: Vec<StagingArena>, timing: bool) -> Result<Self> {
        let _scope = cuteafd_ffi::memory_ledger::scope("spark-intake");
        let waves = intake.waves;
        ensure!(arenas.len() == waves && intake.planes.len() == waves, "one staging arena and plane set per slot");
        ensure!(intake.planes.iter().all(|planes| planes.len() == intake.ranks), "one plane per rank");
        let control = HostAllocation::new(library, Counter::words(waves) * LINE)?;
        let words = Words { base: control.buffer.ptr as usize, slots: waves };
        let plane_bytes = intake.rows * intake.row_bytes;
        let all = if intake.ranks == 8 { u8::MAX } else { (1u8 << intake.ranks) - 1 };
        let pinned = all & !intake.landing;
        let landing_staging = (0..waves).map(|_| (pinned != 0)
            .then(|| HostAllocation::new(library, intake.ranks * plane_bytes)).transpose())
            .collect::<Result<Vec<_>>>()?;
        let planes = intake.planes.iter().zip(&landing_staging).map(|(planes, staging)| planes.iter().enumerate()
            .map(|(rank, plane)| Plane { device: plane.ptr as usize, bytes: plane_bytes,
                host: staging.as_ref().filter(|_| pinned & (1 << rank) != 0)
                    .map_or(0, |staging| staging.buffer.ptr as usize + rank * plane_bytes) })
            .collect()).collect();
        let timeout = transport.timeout();
        let mut native = Native { transport, arenas: arenas.clone(), words, planes, ranks: intake.ranks,
            row_bytes: intake.row_bytes, landing: intake.landing };
        tracing::info!(waves, ranks = intake.ranks, landing = format!("{:#b}", intake.landing),
            staging_bytes = arenas.first().map_or(0, StagingArena::bytes), "Spark exchange worker started");
        let (jobs, inbox) = mpsc::channel();
        let worker = std::thread::Builder::new().name("spark-exchange".into()).spawn(move || {
            let (result, highest) = run_worker(&mut native, &inbox, timeout, timing);
            if let Err(error) = &result {
                cuteafd_transport::health::record_failure(format!("Spark exchange: {error:#}"));
                tracing::error!(error = %format!("{error:#}"), "Spark exchange failed");
                native.transport.abandon();
                release_until_closed(&mut native, &inbox, highest);
            }
            (Box::new(native.transport) as Box<dyn std::any::Any + Send>, result)
        })?;
        Ok(Self { library, jobs: Some(jobs), worker: Some(worker), control, words, arenas, landing_staging,
            landing: intake.landing, planes: intake.planes, layouts: vec![None; waves], unread: vec![false; waves],
            uses: vec![0; waves], ranks: intake.ranks, rows: intake.rows, row_bytes: intake.row_bytes, waves,
            next_slot: intake.next_slot % waves, sequence: 0, outstanding: VecDeque::new(), grouping })
    }

    /// The workers' installed expert grouping (from the bootstrap).
    pub(crate) fn installed_grouping(&self) -> &[cuteafd_transport::worker_selection::InstalledExpertGrouping] {
        &self.grouping
    }

    /// Whether each slot's request staging holds `bytes`.
    pub(crate) fn fits(&self, bytes: usize) -> bool { self.arenas.iter().all(|arena| arena.bytes() >= bytes) }

    fn submit(&mut self, rows: usize, source: Source, stage: Option<(*mut c_void, RouteStaging,
        [CuteafdDeviceBuffer; 3])>) -> Result<WaveTicket> {
        ensure!(self.outstanding.len() < self.waves, WaveError::Admission(self.outstanding.len()));
        ensure!(rows > 0 && rows <= self.rows, WaveError::Rows { rows, capacity: self.rows });
        healthy()?;
        let slot = self.next_slot;
        ensure!(!self.unread[slot], WaveError::Unconsumed(slot));
        let sequence = self.sequence.wrapping_add(1);
        if let Some((stream, layout, sources)) = stage {
            let arena = &self.arenas[slot];
            ensure!(layout.bytes <= arena.bytes(), "wave staging of {} bytes exceeds its {}-byte arena",
                layout.bytes, arena.bytes());
            // SAFETY: a live stream of the link's device; the control words
            // outlive every queued wait and write (the worker stores every
            // sequence it is sent before the link frees them, and the link's
            // owner drains its streams before dropping it). The wait comes
            // first and is unconditional: the slot's previous wave (cyclically
            // already reached on a slot's first use) has landed before the
            // copies overwrite its staging, so the NIC no longer reads it. The
            // caller vouches for the sources (`send`'s contract).
            unsafe {
                self.library.cuda_stream_wait_value32(stream, self.words.ptr(Counter::Landed),
                    sequence.wrapping_sub(self.waves as u32))?;
                layout.enqueue(self.library, arena.host(), sources, stream)?;
                self.library.cuda_stream_write_value32(stream, self.words.ptr(Counter::Staged(slot)), sequence)?;
            }
            self.layouts[slot] = Some(layout);
        } else {
            self.layouts[slot] = None;
        }
        let job = Job { sequence, slot, rows, source, consumed: self.uses[slot], sent: Instant::now() };
        self.jobs.as_ref().context("Spark exchange stopped")?.send(Command::Wave(job))
            .map_err(|_| WaveError::Lost(sequence))?;
        self.sequence = sequence;
        self.outstanding.push_back((sequence, slot));
        self.next_slot = (slot + 1) % self.waves;
        Ok(WaveTicket { sequence, slot, rows })
    }

    /// Queues a `rows`-row wave into the next return slot: its route ids, gate
    /// weights and expert input rows (`sources`, in that order) copy into the
    /// slot's staging on `stream`, which then publishes `staged`; the worker
    /// builds, posts and lands it. The engine thread does not wait.
    ///
    /// # Safety
    /// `sources` are live device buffers of `layout`'s extents on `stream`'s
    /// device, written in `stream` order before this call (the copies read
    /// them in stream order, so later stream work may rewrite them).
    pub(crate) unsafe fn send(&mut self, rows: usize, layout: RouteStaging, sources: [CuteafdDeviceBuffer; 3],
        stream: *mut c_void, build: BuildRequest) -> Result<WaveTicket> {
        self.submit(rows, Source::Staged { layout, build }, Some((stream, layout, sources)))
    }

    /// Queues a wave whose request the host already built (warm-ups, gates):
    /// the worker posts it once its slot's planes are free.
    pub(crate) fn send_request(&mut self, request: ExpertProtocolV2Request) -> Result<WaveTicket> {
        let rows = request.header.row_count as usize;
        self.submit(rows, Source::Built(request), None)
    }

    /// Orders `stream` after `ticket`'s wave (the oldest) has landed in its
    /// slot's planes, which are readable on `stream` from here on. Never
    /// waits on the host. Mark the planes read with [`Self::consumed`] after
    /// the reduce, or reduce with [`Self::reduce`].
    pub(crate) fn land(&mut self, ticket: WaveTicket, stream: *mut c_void) -> Result<()> {
        let (expected, slot) = self.outstanding.pop_front().ok_or(WaveError::Lost(ticket.sequence))?;
        ensure!(expected == ticket.sequence && slot == ticket.slot,
            WaveError::Order { expected, found: ticket.sequence });
        let bytes = ticket.rows * self.row_bytes;
        // SAFETY: as `submit`'s waits. The uploads read the slot's landing
        // staging after the wait (the worker filled it before publishing) and
        // write the slot's planes, which no other stream reads before this
        // stream's `consumed`; both outlive the queued copies (the link's
        // owner drains its streams before dropping the link).
        unsafe {
            self.library.cuda_stream_wait_value32(stream, self.words.ptr(Counter::Landed), ticket.sequence)?;
            if let Some(staging) = &self.landing_staging[slot] {
                for (rank, plane) in self.planes[slot].iter().enumerate().filter(|(rank, _)| self.landing & (1 << rank) == 0) {
                    let source = CuteafdHostBuffer {
                        ptr: staging.buffer.ptr.cast::<u8>().add(rank * self.rows * self.row_bytes).cast(),
                        bytes, ..staging.buffer };
                    self.library.copy_host_buffer_h2d_async(CuteafdDeviceBuffer { bytes, ..*plane }, source, bytes,
                        stream)?;
                }
            }
        }
        self.unread[slot] = true;
        Ok(())
    }

    /// Plane pointers of return slot `slot` (rank order), for the reduce.
    pub(crate) fn planes(&self, slot: usize) -> [*const u16; MAX_INTAKE_RANKS] {
        let mut pointers = [std::ptr::null::<u16>(); MAX_INTAKE_RANKS];
        for (pointer, plane) in pointers.iter_mut().zip(&self.planes[slot]) { *pointer = plane.ptr.cast(); }
        pointers
    }

    pub(crate) fn ranks(&self) -> usize { self.ranks }

    /// Marks `slot`'s planes read once `stream` reaches this point (after the
    /// reduce): the worker posts the slot's next wave only after it.
    pub(crate) fn consumed(&mut self, slot: usize, stream: *mut c_void) -> Result<()> {
        ensure!(slot < self.waves && self.unread[slot], "return slot {slot} has no landed wave to consume");
        let uses = self.uses[slot].wrapping_add(1);
        // SAFETY: as `submit`'s writes.
        unsafe { self.library.cuda_stream_write_value32(stream, self.words.ptr(Counter::Consumed(slot)), uses)? };
        self.uses[slot] = uses;
        self.unread[slot] = false;
        Ok(())
    }

    /// Queues `output = shared + sum of slot`'s planes for `rows` rows on
    /// `stream` and marks the planes read (`shared` may be null).
    ///
    /// # Safety
    /// `shared` (when not null) and `output` are live `[rows, hidden]` BF16
    /// device buffers ordered on `stream`; `slot`'s wave landed on `stream`.
    pub(crate) unsafe fn reduce(&mut self, slot: usize, shared: *const u16, output: *mut u16, rows: usize,
        stream: *mut c_void) -> Result<()> {
        // SAFETY: forwarded from this function's contract.
        unsafe {
            self.library.v41_compact_reducer()?.reduce_planes(self.planes(slot), self.ranks as u32, shared,
                output, u32::try_from(rows)?, stream)?;
        }
        self.consumed(slot, stream)
    }

    /// Lands `ticket` on `stream`, marks it consumed (nothing reads the
    /// planes) and drains `stream`: warm-ups and gates.
    pub(crate) fn land_drained(&mut self, ticket: WaveTicket, stream: *mut c_void) -> Result<()> {
        let slot = ticket.slot();
        self.land(ticket, stream)?;
        self.consumed(slot, stream)?;
        // SAFETY: the caller's live stream.
        unsafe { self.library.cuda_stream_synchronize(stream)? };
        healthy()
    }

    /// The staged route ids (`rows * topk` little-endian u32) of `slot`'s last
    /// staged wave, for draft probes. Call only once the stream that landed
    /// the wave was synchronized; valid until the slot's next send.
    pub(crate) fn staged_ids(&self, slot: usize) -> Result<&[u8]> {
        let layout = self.layouts[slot].context("no wave staged in this slot")?;
        ensure!(self.outstanding.iter().all(|&(_, pending)| pending != slot), "slot {slot}'s wave has not landed");
        // SAFETY: the caller synchronized a stream ordered after the wave's
        // landing, so its staged copies completed; only this slot's next
        // send rewrites the arena.
        let bytes = unsafe { self.arenas[slot].read(layout.bytes)? };
        Ok(&bytes[..layout.route_bytes])
    }

    pub(crate) fn waves(&self) -> usize { self.waves }
    pub(crate) fn outstanding(&self) -> usize { self.outstanding.len() }
    /// The slot the next send takes (the transport's landing rotation).
    pub(crate) fn next_slot(&self) -> usize { self.next_slot }

    /// Stops the worker and returns the transport (after a failure, with its
    /// waves abandoned) and the slot its next wave lands in. Every sent wave
    /// must have been landed and consumed, and the consuming streams drained.
    pub(crate) fn stop<T: Transport>(mut self) -> Result<(T, usize)> {
        ensure!(self.outstanding.is_empty() && self.unread.iter().all(|unread| !unread),
            "Spark exchange stopped with waves not landed and consumed");
        drop(self.jobs.take());
        let worker = self.worker.take().context("Spark exchange already stopped")?;
        let (transport, result) = worker.join().map_err(|_| anyhow::anyhow!("Spark exchange worker panicked"))?;
        result?;
        let transport = transport.downcast::<T>().map_err(|_| anyhow::anyhow!("Spark exchange transport type differs"))?;
        Ok((*transport, self.next_slot))
    }
}

impl Drop for Waves<'_> {
    fn drop(&mut self) {
        // Closing the channel ends the worker. With waves outstanding (an
        // unwinding owner) it fails, which releases every sequence it was
        // sent, so no queued stream wait outlives the counters' use; the
        // owner drains its streams before freeing the link.
        drop(self.jobs.take());
        if let Some(worker) = self.worker.take() {
            if worker.join().is_err() {
                // Waits may still be queued against these words: keep them.
                tracing::error!("Spark exchange worker panicked; retaining its control block");
                self.library.quarantine_module_after_failed_drain();
            }
        }
    }
}

#[cfg(test)]
#[path = "spark_waves_tests.rs"]
mod tests;
