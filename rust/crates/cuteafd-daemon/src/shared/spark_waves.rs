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
//! - `arrived[rank]`: for a rank whose rows arrive in pinned host memory (no
//!   GPU landing), once its reply is in, the worker writes the reply frames'
//!   addresses into the slot's mapped pull descriptor for that rank and stores
//!   the wave's sequence. The frames stay in their registered receive slots:
//!   nothing copies them on the host.
//! - `landed`: once every rank's rows arrived, the worker stores the wave's
//!   sequence from the host. [`Waves::land`] queues, per pinned rank not yet
//!   pulled, `cuStreamWaitValue32(arrived[rank] >= sequence)` and an SM pull
//!   of that rank's frames into its plane (the kernel reads the descriptor when
//!   it runs), then `landed >= sequence`, and returns at once. Early ranks'
//!   pulls run while later ranks are still on the wire; a host-ordered owner
//!   queues them from its wait loop instead ([`Waves::pull_arrived`]). Small
//!   (decode) waves pull every rank in one launch behind `landed`. Later work
//!   on the stream (the reduce) sees the planes.
//! - `consumed[slot]`: after the reduce, the consuming stream writes the
//!   slot's use count; the worker releases the slot's retained reply frames
//!   (their receive slots go back to the NIC) and posts the slot's next wave
//!   (whose rows overwrite the planes and descriptors) only once it is reached.
//!   Each rank's reply is one frame, and a link has at most its admitted slots
//!   of waves between post and consumed, so a receive ring at least that deep
//!   never runs dry ([`Waves::start`] checks it).
//!
//! Every copy the engine queues around these waits (the staging copies in
//! [`Waves::send`], the pinned ranks' pulls in [`Waves::land`]) is an SM
//! copy through mapped host memory, never `cudaMemcpyAsync`. CUDA documents that ordering through stream memory
//! operations is invisible to its scheduler, and a small H2D may read its
//! host source when queued; a kernel reads and writes when it runs. A lazily
//! loaded kernel's first launch can wait for the device to idle, so a first
//! launch behind a pending wait stalls the engine thread until that wait is
//! released, and for good when the release depends on the stalled thread
//! (the first Engram dequantization of a cold-start request hung a
//! 0-RTX-layer prefill this way). [`Waves::start`] loads every registered
//! native kernel before any wait exists.
//!
//! The worker queues no GPU work at all. A stream memory wait stalls its
//! hardware queue, and streams share queues, so a wait released by GPU work
//! (an upload on a copy stream) could deadlock behind itself; here every wait
//! is released by a host store alone. Measured (#815, RTX PRO 6000): with four
//! streams each holding a kernel and an H2D copy behind a host-word wait, a
//! copy-engine H2D on a fresh stream never completes until the word is
//! released, so a worker-issued `cudaMemcpyAsync` landing could not gate
//! `landed`. Events are not used for these counters:
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
//! which every request refuses. The worker clears every pull descriptor (a
//! pull that runs after that copies nothing), then releases every wait the
//! engine queued or will queue, by storing each sequence it is sent, until
//! the link stops it. Retained reply frames go back with the transport
//! ([`Waves::stop`], [`Waves::close`]); the link keeps them until its streams
//! drained, since a pull may already have been reading them. The planes of those waves are garbage: every consumer
//! checks [`healthy`] after its step's sampling sync, so a released step
//! fails instead of producing tokens.
//!
//! [`health`]: cuteafd_transport::health
use crate::shared::experts::staging::RouteStaging;
use crate::shared::wave_geometry::SlotGeometry;
use crate::shared::memory::HostAllocation;
use crate::shared::spark_intake::{upload_offset, MAX_INTAKE_RANKS};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::peer_exchange::{PullEntry, PullFrames, PULL_MAX_FRAMES};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_transport::expert::{SparkExperts, SparkExpertWave, WaveReceipt};
use cuteafd_transport::{ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, StagingArena,
    VerbsHostProtocolV2ResponsePayload};
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
    #[error("rank {rank} keeps {depth} reply frames but the link admits {slots} waves")]
    RingDepth { rank: usize, depth: usize, slots: usize },
    #[error("rank {rank}'s reply arrived in {frames} frames (at most {PULL_MAX_FRAMES})")]
    Frames { rank: usize, frames: usize },
}

/// Whether cyclic counter `value` has reached `target`.
/// The landed sequence a wave's request staging waits for before its slot
/// is rewritten: the slot's previous wave, `sequence - waves`. A slot's first
/// use (the link's first `waves` sequences) waits for 0, landed at start.
/// (`cuStreamWaitValue32` GEQ is cyclic, so a wrapped value would also pass;
/// the explicit floor keeps the first use readable without that reasoning.)
pub(crate) fn staging_reuse(sequence: u32, waves: usize) -> u32 {
    sequence.checked_sub(waves as u32).unwrap_or(0)
}

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
/// A wave whose pinned ranks carry more than this lands rank by rank (a wait
/// and a pull each, overlapping the later ranks' replies); a smaller (decode)
/// wave pulls all its ranks in one launch behind `landed`.
const PULL_PER_RANK_BYTES: usize = 1 << 20;

/// A counter word of the control block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Counter { Landed, Staged(usize), Consumed(usize), Arrived(usize) }

impl Counter {
    fn index(self, slots: usize) -> usize {
        match self {
            Self::Landed => 0,
            Self::Staged(slot) => { assert!(slot < slots, "staged counter of slot {slot} past {slots}"); 1 + 2 * slot }
            Self::Consumed(slot) => { assert!(slot < slots, "consumed counter of slot {slot} past {slots}"); 2 + 2 * slot }
            Self::Arrived(rank) => { assert!(rank < MAX_INTAKE_RANKS, "arrived counter of rank {rank}"); 1 + 2 * slots + rank }
        }
    }
    fn words(slots: usize) -> usize { 1 + 2 * slots + MAX_INTAKE_RANKS }
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
    /// The use counts the worker waits for before posting (planes free): the
    /// slot's own last use.
    consumed: Vec<(usize, u32)>,
    /// Byte offset and extent of the wave's return slot in each rank's range
    /// (also its placed GPU landing slice).
    slice: (usize, usize),
    /// The slot's use count once this wave's reduce is marked consumed: its
    /// retained reply frames go back to the transport when `consumed[slot]`
    /// reaches it.
    release: u32,
    sent: Instant,
}

/// What the worker knows of a posted wave when it progresses it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WaveInfo {
    pub sequence: u32,
    pub slot: usize,
    pub rows: usize,
    /// Byte offset and extent of the wave's return slot in each rank's range.
    pub slice: (usize, usize),
    pub release: u32,
}

/// A command to the worker.
pub(crate) enum Command {
    Wave(Job),
    /// A deliberate stop (a link reset after a cancelled request): waves
    /// still outstanding are given up without marking the process unhealthy.
    Cancel,
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
    /// One progress pass over the oldest wave: a pinned rank whose rows are
    /// all in gets its pull descriptor and `arrived` published (its frames
    /// retained until the wave's `release`). `true` once every rank's rows
    /// are in place.
    fn poll(&mut self, wave: &mut Self::Wave, info: &WaveInfo) -> Result<bool>;
    /// Stores `landed = sequence`.
    fn publish(&mut self, sequence: u32);
    /// Returns retained reply frames whose wave's reduce was consumed; true
    /// when any went back.
    fn release_consumed(&mut self) -> bool { false }
    /// After a failure: clears every pull descriptor, then releases every
    /// wait up to `sequence` (`arrived` of every rank and `landed`).
    fn fail(&mut self, sequence: u32) { self.publish(sequence) }
}

struct Inflight<W> {
    info: WaveInfo,
    wave: W,
    posted: Instant,
    times: WaveTimes,
}

/// Idle passes the worker spins after progress before it may sleep (each
/// pass polls the counters and the CQs: a few tens of microseconds).
const SPIN_PASSES: u32 = 256;
/// Idle passes between clock reads (the no-progress deadline and the sleep
/// decisions are tens of microseconds and up).
const IDLE_CLOCK: u32 = 64;
/// One sleep while a wave cannot complete yet. With the thread's timer slack
/// at 1 us and shallow idle states this wakes within a few microseconds.
const SLEEP: Duration = Duration::from_micros(20);
/// A wave sleeps only through this share of its expected duration (the
/// running mean of the last waves), then spins: completion stays within a
/// spin of the event, while the long middle of a long Spark round trip or GPU
/// stage costs no CPU. A wait first spins through the hybrid window
/// ([`BLOCK_SPIN`](cuteafd_transport::wait_policy::BLOCK_SPIN), 10 ms), so a
/// decode wave (a per-layer round well under that) never sleeps.
const SLEEP_SHARE: f64 = 0.7;
/// Weight of the newest wave in the running means.
const MEAN_WEIGHT: f64 = 0.125;
/// With nothing outstanding it keeps polling this long (a step's layers are
/// queued in one burst), then blocks on its channel.
const IDLE_SPIN: Duration = Duration::from_micros(30);

/// Running means of how long waves take, which decide how long the worker
/// may sleep before a wave can complete.
#[derive(Default)]
struct Pace { round_trip: Option<f64>, staging: Option<f64> }
impl Pace {
    fn add(mean: &mut Option<f64>, sample: f64) {
        *mean = Some(mean.map_or(sample, |mean| mean + MEAN_WEIGHT * (sample - mean)));
    }
    /// Whether a wait that has lasted `age` seconds, of a kind whose mean is
    /// `mean`, may sleep: past the hybrid spin window and still early in its
    /// expected duration.
    fn early(mean: Option<f64>, age: f64) -> bool {
        !cuteafd_transport::wait_policy::block_spins(Duration::from_secs_f64(age))
            && mean.is_some_and(|mean| age < SLEEP_SHARE * mean)
    }
}

/// The worker's loop until its channel closes or a transport error. It never
/// blocks on one condition while another could progress: it posts the oldest
/// job once its staging is complete and its slot's planes are free, and
/// progresses the oldest wave meanwhile. Returns the highest sequence sent.
/// How a worker that did not fail stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stop {
    /// The link closed the channel at rest.
    Closed,
    /// The link cancelled (a reset): waves may still be outstanding.
    Cancelled,
}

pub(crate) fn run_worker<B: Backend>(backend: &mut B, inbox: &mpsc::Receiver<Command>, timeout: Duration,
    timing: bool) -> (Result<Stop>, Option<u32>) {
    let mut queued = VecDeque::<Job>::new();
    let mut inflight = VecDeque::<Inflight<B::Wave>>::new();
    let mut highest: Option<u32> = None;
    // When the current idle stretch began, read every IDLE_CLOCK idle passes
    // (a busy loop must not spend its time reading the clock).
    let mut idle_since: Option<Instant> = None;
    let mut idle_passes = 0u32;
    let mut pace = Pace::default();
    // `--host-wait block` spins through the first `BLOCK_SPIN` of a wave's
    // wait, then paces the polling to its completion; `spin` polls throughout
    // (lowest latency).
    let paced = cuteafd_transport::wait_policy::wait_policy().wait == cuteafd_transport::wait_policy::HostWait::Block;
    // Whether the current wait is early enough to sleep through, and whether
    // the last pass slept (the next one re-decides).
    let (mut early, mut slept) = (false, false);
    // When the last wave landed: a queued job's staging on the GPU cannot
    // finish before that (its slot or stream waits for the landing), so its
    // wait is timed from the later of this and its send (the host runs ahead).
    let mut last_landed = Instant::now();
    // Sleeps taken and seconds slept (the CPU the pacing saved).
    let mut sleeps = (0u64, 0.0f64);
    let result = (|| -> Result<Stop> {
        loop {
            let mut progressed = false;
            let mut closed = false;
            let empty = queued.is_empty() && inflight.is_empty();
            if empty && idle_passes % IDLE_CLOCK == 0 && idle_since.is_some_and(|since| since.elapsed() >= IDLE_SPIN) {
                // Nothing outstanding: block until the engine sends.
                match inbox.recv() {
                    Ok(Command::Wave(job)) => { highest = Some(job.sequence); queued.push_back(job); progressed = true; }
                    Ok(Command::Cancel) => { backend.release_consumed(); return Ok(Stop::Cancelled); }
                    Err(_) => {
                        // At rest: the frames of reduces already consumed go back now.
                        backend.release_consumed();
                        return Ok(Stop::Closed);
                    }
                }
            }
            loop {
                match inbox.try_recv() {
                    Ok(Command::Wave(job)) => { highest = Some(job.sequence); queued.push_back(job); progressed = true; }
                    // Outstanding waves are given up by the caller (`finish_worker`).
                    Ok(Command::Cancel) => return Ok(Stop::Cancelled),
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => { closed = true; break; }
                }
            }
            // Reply frames go back to the transport once their reduce ran
            // (before a post, which needs a free receive per rank).
            if backend.release_consumed() { progressed = true; }
            // Post the next job once its rows are staged and its slot's planes are free.
            if let Some(job) = queued.front() {
                let staged = match job.source {
                    Source::Staged { .. } => reached(backend.counter(Counter::Staged(job.slot)), job.sequence),
                    Source::Built(_) => true,
                };
                let free = job.consumed.iter().all(|&(slot, uses)| reached(backend.counter(Counter::Consumed(slot)), uses));
                if staged && free {
                    // After the fence check: the frames of every wave whose
                    // reduce that check saw consumed go back first, so the
                    // post finds their receives (counters only grow).
                    backend.release_consumed();
                    let job = queued.pop_front().expect("front job");
                    let info = WaveInfo { sequence: job.sequence, slot: job.slot, rows: job.rows, slice: job.slice,
                        release: job.release };
                    let mut times = WaveTimes { staged: job.sent.elapsed().as_secs_f64(), ..Default::default() };
                    Pace::add(&mut pace.staging, job.sent.max(last_landed).elapsed().as_secs_f64());
                    let started = Instant::now();
                    let wave = backend.post(job)?;
                    times.posted = started.elapsed().as_secs_f64();
                    inflight.push_back(Inflight { info, wave, posted: Instant::now(), times });
                    progressed = true;
                }
            }
            // Progress the oldest wave.
            if let Some(front) = inflight.front_mut() {
                if backend.poll(&mut front.wave, &front.info)? {
                    let mut done = inflight.pop_front().expect("front wave");
                    done.times.received = done.posted.elapsed().as_secs_f64();
                    Pace::add(&mut pace.round_trip, done.times.received);
                    last_landed = Instant::now();
                    backend.publish(done.info.sequence);
                    if timing {
                        tracing::info!(target: "cuteafd::timing", sequence = done.info.sequence, slot = done.info.slot,
                            rows = done.info.rows, staged_us = (done.times.staged * 1e6) as u64,
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
                backend.release_consumed();
                return Ok(Stop::Closed);
            }
            if progressed {
                idle_since = None;
                idle_passes = 0;
                (early, slept) = (false, false);
                continue;
            }
            idle_passes = idle_passes.wrapping_add(1);
            if idle_since.is_none() || idle_passes % IDLE_CLOCK == 0 {
                let since = *idle_since.get_or_insert_with(Instant::now);
                // Waiting on the GPU (staged copies, a pending reduce) or on a
                // reply, which carries its own transport deadline.
                ensure!(empty || since.elapsed() < timeout, "Spark exchange made no progress for {timeout:?}");
            }
            if idle_passes < SPIN_PASSES || empty || !paced {
                std::hint::spin_loop();
                continue;
            }
            // Sleep only while what blocks progress is early in its expected
            // duration: the oldest wave's round trip, or the next job's
            // staging on the GPU. Decided after a sleep or every IDLE_CLOCK
            // passes, so spinning near completion reads no clock.
            if slept || idle_passes % IDLE_CLOCK == 0 {
                early = match (inflight.front(), queued.front()) {
                    (Some(wave), _) => Pace::early(pace.round_trip, wave.posted.elapsed().as_secs_f64()),
                    (None, Some(job)) => Pace::early(pace.staging, job.sent.max(last_landed).elapsed().as_secs_f64()),
                    (None, None) => false,
                };
            }
            slept = early;
            if early {
                let started = Instant::now();
                std::thread::sleep(SLEEP);
                sleeps.0 += 1;
                sleeps.1 += started.elapsed().as_secs_f64();
            } else { std::hint::spin_loop(); }
        }
    })();
    tracing::debug!(sleeps = sleeps.0, slept_ms = sleeps.1 * 1e3, round_trip_us = pace.round_trip.map(|m| m * 1e6),
        staging_us = pace.staging.map(|m| m * 1e6), "Spark exchange worker pacing");
    #[cfg(test)]
    eprintln!("worker pacing: {} sleeps, {:.1} ms slept, round trip {:?} us, staging {:?} us", sleeps.0, sleeps.1 * 1e3,
        pace.round_trip.map(|m| (m * 1e6) as u64), pace.staging.map(|m| (m * 1e6) as u64));
    (result, highest)
}

/// After a failure: release every wait queued for `highest` and every wave
/// the engine still sends, so no stream waits forever, until the channel closes.
fn release_until_closed<B: Backend>(backend: &mut B, inbox: &mpsc::Receiver<Command>, highest: Option<u32>) {
    if let Some(highest) = highest { backend.fail(highest); }
    while let Ok(command) = inbox.recv() {
        if let Command::Wave(job) = command { backend.fail(job.sequence); }
    }
}

/// The worker's exit: a failure marks the process unhealthy (other lanes may
/// still own queued work from the failed pass) and abandons the transport's
/// waves; a cancel gives its outstanding waves up without a failure. Either
/// way every queued wait is released (no stream waits on a wave that will
/// never land) until the link closes the channel.
fn finish_worker<T: Transport>(native: &mut Native<T>, inbox: &mpsc::Receiver<Command>,
    result: Result<Stop>, highest: Option<u32>) -> Result<()> {
    match result {
        Ok(Stop::Closed) => Ok(()),
        Ok(Stop::Cancelled) => {
            native.transport.cancel();
            release_until_closed(native, inbox, highest);
            Ok(())
        }
        Err(error) => {
            cuteafd_transport::health::record_failure(format!("Spark exchange: {error:#}"));
            tracing::error!(error = %format!("{error:#}"), "Spark exchange failed");
            native.transport.abandon();
            release_until_closed(native, inbox, highest);
            Err(error)
        }
    }
}

/// A reply frame the worker keeps until its wave's reduce ran: the payload's
/// length and, when it lies in mapped pinned host memory, the address a
/// kernel reads it at (UVA: the host address, checked at [`Waves::start`]).
pub(crate) trait ReplyFrame: Send + 'static {
    fn len(&self) -> usize;
    fn device_address(&self) -> Option<u64>;
}

impl ReplyFrame for VerbsHostProtocolV2ResponsePayload {
    fn len(&self) -> usize { self.as_ref().len() }
    fn device_address(&self) -> Option<u64> {
        self.pinned_host_buffer().filter(|buffer| buffer.flags & cuteafd_ffi::CUTEAFD_HOST_BUFFER_FLAG_MAPPED != 0)
            .map(|buffer| buffer.ptr as u64)
    }
}

/// The Spark side of the worker: posts a request and progresses its wave.
/// [`SparkExperts`] in serving; an in-process echo in the GPU exchange test.
pub(crate) trait Transport: Send + 'static {
    type Wave: Send;
    type Frame: ReplyFrame;
    /// Posts `request`; `landing` places the payload of each GPU-landing rank
    /// at that slice of its registered range.
    fn dispatch(&mut self, request: &ExpertProtocolV2Request, landing: Option<&[cuteafd_transport::LandingSlice]>)
        -> Result<Self::Wave>;
    /// One non-blocking pass: arrived rows go to `sink(rank, first row,
    /// frame)`, which keeps the frame; `Some(landed ranks)` once every
    /// rank's rows arrived.
    fn poll(&mut self, wave: &mut Self::Wave, sink: &mut dyn FnMut(usize, u32, Self::Frame) -> Result<()>)
        -> Result<Option<u8>>;
    /// The response deadline every wave is held to.
    fn timeout(&self) -> Duration;
    /// Gives up every outstanding wave (after a failure).
    fn abandon(&mut self);
    /// Gives up every outstanding wave for a deliberate reset: nothing failed.
    fn cancel(&mut self) { self.abandon() }
    /// Per rank: reply frames its receive ring lets consumers keep at once
    /// (0: frames are copies, which keep no receive).
    fn retained_reply_slots(&self) -> Vec<usize>;
}

impl Transport for SparkExperts {
    type Wave = (SparkExpertWave, WaveReceipt);
    type Frame = VerbsHostProtocolV2ResponsePayload;
    fn dispatch(&mut self, request: &ExpertProtocolV2Request, landing: Option<&[cuteafd_transport::LandingSlice]>)
        -> Result<Self::Wave> {
        Ok((self.dispatch_wave_at(request, landing)?, WaveReceipt::default()))
    }
    fn poll(&mut self, wave: &mut Self::Wave, sink: &mut dyn FnMut(usize, u32, Self::Frame) -> Result<()>)
        -> Result<Option<u8>> {
        let (wave, receipt) = wave;
        let done = self.poll_wave(wave, receipt, |rank, first, payload| sink(rank, first, payload))?;
        Ok(done.then_some(receipt.landed))
    }
    fn timeout(&self) -> Duration { SparkExperts::timeout(self) }
    fn abandon(&mut self) { self.abandon_waves("Spark exchange failed with waves in flight"); }
    fn cancel(&mut self) { self.cancel_waves(); }
    fn retained_reply_slots(&self) -> Vec<usize> { SparkExperts::retained_reply_slots(self) }
}

/// Bytes of one rank's pull descriptor ([`PullFrames`]) in the mapped block.
const PULL_STRIDE: usize = 128;
const _: () = assert!(std::mem::size_of::<PullFrames>() <= PULL_STRIDE);

/// The mapped pull descriptors: one [`PullFrames`] per (admitted slot, rank),
/// written by the worker, read by pull kernels when they run.
#[derive(Clone, Copy)]
pub(crate) struct Pulls { base: usize }

impl Pulls {
    fn bytes(slots: usize) -> usize { slots * MAX_INTAKE_RANKS * PULL_STRIDE }
    fn at(self, slot: usize, rank: usize) -> *mut PullFrames {
        (self.base + (slot * MAX_INTAKE_RANKS + rank) * PULL_STRIDE) as *mut PullFrames
    }
    /// Publishes `rank`'s frames of the wave in `slot`.
    fn write(self, slot: usize, rank: usize, frames: &[PullEntry]) {
        let mut reply = PullFrames { count: frames.len() as u32, ..Default::default() };
        reply.frame[..frames.len()].copy_from_slice(frames);
        // SAFETY: an aligned descriptor of the live block; no pull of this
        // slot runs until `arrived[rank]` reaches the wave (stored after this),
        // and the slot's previous wave's pulls completed before its consumed.
        unsafe { std::ptr::write_volatile(self.at(slot, rank), reply) };
    }
    /// Clears every descriptor: a pull that runs afterwards copies nothing.
    /// W48-12 (2026-10-30): a link whose ranks all land in device memory has
    /// no descriptor block (`base` 0, see `Native`'s construction); clearing
    /// it wrote through a null pointer when a wave failed (p10.4.2 segfault
    /// in `spark-exchange` on a client disconnect during a parked prefill).
    fn clear(self, slots: usize) {
        if self.base == 0 { return; }
        for slot in 0..slots {
            for rank in 0..MAX_INTAKE_RANKS {
                // SAFETY: as `write`; a racing pull reads the old (retained)
                // frames or none (a zero source copies nothing).
                unsafe { std::ptr::write_volatile(std::ptr::addr_of_mut!((*self.at(slot, rank)).count), 0) };
            }
        }
    }
}

/// Every pinned rank's receive ring must hold a reply frame of every wave
/// between its post and its reduce. Every wave on a link, decode and prefill
/// alike, goes through its one exchange and takes one of its `slots` admitted
/// return slots, and a slot's next wave is posted only after the slot's
/// previous reduce released its frames, so at most `slots` waves hold frames
/// on a ring at once, whatever mix of decode steps and prefill units is in
/// flight. (Another lane's link has its own QPs, so its own rings.) A
/// shallower ring would leave a wave without a posted receive while the
/// frames that free one wait for a later wave's reduce.
pub(crate) fn check_reply_ring(depths: &[usize], pinned: u8, slots: usize) -> Result<()> {
    for (rank, &depth) in depths.iter().enumerate() {
        if pinned & (1 << rank) != 0 && depth != 0 && depth < slots {
            return Err(WaveError::RingDepth { rank, depth, slots }.into());
        }
    }
    Ok(())
}

/// A wave's reply frames, retained until its reduce was consumed.
struct Retained<F> {
    slot: usize,
    release: u32,
    frames: Vec<F>,
}

/// The worker's backend over a [`Transport`]: the slots' request staging,
/// the pull descriptors, the retained reply frames and the control counters.
struct Native<T: Transport> {
    transport: T,
    arenas: Vec<StagingArena>,
    words: Words,
    pulls: Pulls,
    /// Admitted slots (the descriptor block's extent).
    slots: usize,
    ranks: usize,
    row_bytes: usize,
    /// Bit r: rank r lands in device memory; others arrive pinned.
    landing: u8,
    /// Whether the GPU-landing ranks' ranges are placed (each wave's request
    /// names its slice); false only for legacy rotating landings.
    placed: bool,
    /// The oldest wave's arrived frames per rank: (bytes, frames).
    receiving: Vec<(usize, Vec<PullEntry>, Vec<T::Frame>)>,
    /// Landed waves' frames, oldest first, until their reduce was consumed.
    retained: VecDeque<Retained<T::Frame>>,
}

impl<T: Transport> Native<T> {
    /// Every frame still held (after the worker ended): the wave being received
    /// and the retained ones.
    fn frames(&mut self) -> Vec<T::Frame> {
        let mut frames: Vec<T::Frame> = self.retained.drain(..).flat_map(|retained| retained.frames).collect();
        for (_, _, receiving) in &mut self.receiving { frames.append(receiving); }
        frames
    }
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
        let slices = (self.placed && self.landing != 0).then(|| vec![cuteafd_transport::LandingSlice {
            offset: job.slice.0, bytes: job.slice.1 }; self.ranks]);
        self.transport.dispatch(&request, slices.as_deref())
    }

    fn poll(&mut self, wave: &mut Self::Wave, info: &WaveInfo) -> Result<bool> {
        let (ranks, row_bytes, landing, words, pulls) = (self.ranks, self.row_bytes, self.landing, self.words, self.pulls);
        let plane_bytes = info.rows * row_bytes;
        ensure!(plane_bytes <= info.slice.1, "wave of {} rows exceeds its return slot", info.rows);
        let receiving = &mut self.receiving;
        let done = self.transport.poll(wave, &mut |rank, first, frame| {
            ensure!(rank < ranks && landing & (1 << rank) == 0,
                "rank {rank} arrived in host memory but its link lands it in device memory");
            let bytes = frame.len();
            let offset = upload_offset(ranks, rank, first, row_bytes, bytes, plane_bytes)?;
            let source = frame.device_address()
                .with_context(|| format!("rank {rank}'s reply frame is not in mapped pinned memory"))?;
            let (received, entries, frames) = &mut receiving[rank];
            ensure!(entries.len() < PULL_MAX_FRAMES, WaveError::Frames { rank, frames: entries.len() + 1 });
            entries.push(PullEntry { source, offset: offset as u64, bytes: bytes as u64 });
            frames.push(frame);
            *received += bytes;
            if *received == plane_bytes {
                // The rank's rows are all in: its pull may run now. The
                // frames stay in their receive slots until the reduce.
                pulls.write(info.slot, rank, entries);
                if !reached(words.load(Counter::Arrived(rank)), info.sequence) {
                    words.store(Counter::Arrived(rank), info.sequence);
                }
            }
            Ok(())
        })?;
        let Some(landed) = done else { return Ok(false) };
        ensure!(landed == self.landing,
            "ranks {landed:#b} landed in device memory; the link expects {:#b}", self.landing);
        let mut frames = Vec::new();
        for (rank, (received, entries, held)) in self.receiving.iter_mut().enumerate().take(ranks) {
            if landing & (1 << rank) != 0 { continue; }
            ensure!(*received == plane_bytes, "rank {rank} finished with {received} of {plane_bytes} bytes");
            *received = 0;
            entries.clear();
            frames.append(held);
        }
        self.retained.push_back(Retained { slot: info.slot, release: info.release, frames });
        Ok(true)
    }

    fn publish(&mut self, sequence: u32) {
        // Ranks in device memory: the NIC's writes completed before their
        // completions were polled (GPUDirect writes ordered for consumers,
        // checked at the landing probe); pinned ranks: published per rank above.
        if !reached(self.words.load(Counter::Landed), sequence) {
            self.words.store(Counter::Landed, sequence);
        }
    }

    fn release_consumed(&mut self) -> bool {
        let words = self.words;
        let before = self.retained.len();
        // A wave's frames go back once its reduce (and so its pulls) ran.
        self.retained.retain(|retained| !reached(words.load(Counter::Consumed(retained.slot)), retained.release));
        self.retained.len() != before
    }

    fn fail(&mut self, sequence: u32) {
        // Descriptors first: a pull released below copies nothing. Frames stay
        // held (a pull may be reading them) until the link's streams drained.
        self.pulls.clear(self.slots);
        for rank in 0..self.ranks {
            if !reached(self.words.load(Counter::Arrived(rank)), sequence) {
                self.words.store(Counter::Arrived(rank), sequence);
            }
        }
        self.publish(sequence);
    }
}

/// What a link hands its exchange: each rank's reserve range and the
/// admitted/current cut of it ([`SlotGeometry`]) the worker lands in.
pub(crate) struct Intake {
    pub ranks: usize,
    pub row_bytes: usize,
    /// Per rank: the device range every return slot is a slice of.
    pub ranges: Vec<CuteafdDeviceBuffer>,
    pub geometry: SlotGeometry,
    /// Bit r: rank r lands in device memory.
    pub landing: u8,
    /// The GPU-landing ranks' ranges are placed: each request names its slice
    /// ([`cuteafd_transport::DeviceLanding::placed`]). False keeps a legacy
    /// rotating landing.
    pub placed: bool,
    /// The slot the transport's next wave lands in (a rotating landing's phase).
    pub next_slot: usize,
}

impl Intake {
    /// Rank planes of `slot` under the current cut, in rank order.
    pub(crate) fn slot_planes(&self, slot: usize) -> Vec<CuteafdDeviceBuffer> {
        let (offset, bytes) = self.geometry.slice(slot, self.row_bytes);
        self.ranges.iter().map(|range| CuteafdDeviceBuffer { ptr: (range.ptr as usize + offset) as *mut c_void,
            bytes, ..*range }).collect()
    }
}

/// Runs the compact reducer once over slot 0's planes on a scratch stream
/// with nothing pending, so its module is loaded before any decode step
/// queues a reduce behind an exchange wait.
fn warm_reducer(library: &NativeLibrary, intake: &Intake) -> Result<()> {
    let mut pointers = [std::ptr::null::<u16>(); MAX_INTAKE_RANKS];
    for (pointer, plane) in pointers.iter_mut().zip(&intake.slot_planes(0)) { *pointer = plane.ptr.cast(); }
    let output = crate::shared::memory::DeviceAllocation::new(library, intake.row_bytes.max(256))?;
    let stream = crate::shared::spark_intake::WarmStream::new(library)?;
    // SAFETY: slot 0's planes and the output hold one row; the stream is
    // drained (finish) before either can be released.
    unsafe {
        library.v41_compact_reducer()?.reduce_planes(pointers, intake.ranks as u32, std::ptr::null(),
            output.buffer.ptr.cast(), 1, stream.raw())?;
    }
    stream.finish()
}

/// A link's persistent exchange: the engine side of the worker.
pub(crate) struct Waves<'a> {
    library: &'a NativeLibrary,
    /// Dropped first: closing it ends the worker, joined in `Drop`.
    jobs: Option<mpsc::Sender<Command>>,
    worker: Option<std::thread::JoinHandle<(Box<dyn std::any::Any + Send>, Result<()>)>>,
    /// Owns the counters; freed only after the worker joined.
    _control: HostAllocation<'a>,
    words: Words,
    arenas: Vec<StagingArena>,
    /// Per slot: the request arena's device alias (mapped pinned memory).
    arena_alias: Vec<usize>,
    /// Owns the pull descriptors of the ranks that arrive in host memory
    /// (mapped; written by the worker, read by pull kernels); freed only after
    /// the worker joined.
    _pull_block: Option<HostAllocation<'a>>,
    pulls: Option<Pulls>,
    /// Bit r: rank r's rows arrive in pinned host memory and are pulled.
    pinned: u8,
    /// Per rank: the range its return slots live in.
    ranges: Vec<CuteafdDeviceBuffer>,
    geometry: SlotGeometry,
    /// Per slot: the layout of its last staged wave (route bytes for probes).
    layouts: Vec<Option<RouteStaging>>,
    /// Per slot: landed and not yet marked consumed.
    unread: Vec<bool>,
    /// Per slot: waves consumed from it (the worker posts the slot's next
    /// wave once `consumed[slot]` reaches this).
    uses: Vec<u32>,
    ranks: usize,
    row_bytes: usize,
    next_slot: usize,
    sequence: u32,
    /// Per outstanding wave: sequence, slot and the pinned ranks whose pull
    /// is already queued ([`Self::pull_arrived`]).
    outstanding: VecDeque<(u32, usize, u8)>,
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
        let slots = intake.geometry.slots();
        let range_bytes = intake.geometry.reserve_rows() * intake.row_bytes;
        ensure!(arenas.len() == slots, "one staging arena per admitted slot");
        ensure!(intake.ranges.len() == intake.ranks && intake.ranges.iter().all(|range| range.bytes >= range_bytes),
            "one reserve range of {range_bytes} bytes per rank");
        let control = HostAllocation::new(library, Counter::words(slots) * LINE)?;
        let words = Words { base: control.buffer.ptr as usize, slots };
        let all = if intake.ranks == 8 { u8::MAX } else { (1u8 << intake.ranks) - 1 };
        let pinned = all & !intake.landing;
        check_reply_ring(&transport.retained_reply_slots(), pinned, slots)?;
        // Pinned ranks reach the planes by SM pulls on the consuming stream
        // straight from their receive slots; the kernels read the frames'
        // host addresses, so mapped pinned memory must be device-visible at
        // its host address (UVA).
        let pull_block = (pinned != 0).then(|| HostAllocation::new(library, Pulls::bytes(slots))).transpose()?;
        if let Some(block) = &pull_block {
            let alias = library.cuda_host_buffer_device_alias(block.buffer)?;
            ensure!(alias.ptr == block.buffer.ptr,
                "mapped pinned memory is not device-visible at its host address (no unified addressing)");
        }
        let pulls = pull_block.as_ref().map(|block| Pulls { base: block.buffer.ptr as usize });
        let arena_alias = arenas.iter().map(|arena| library.cuda_host_buffer_device_alias(arena.host())
            .map(|alias| alias.ptr as usize)).collect::<Result<Vec<_>>>()?;
        // Every registered native kernel loads now, with nothing pending: a
        // lazily loaded kernel's first launch behind a pending wait stalls the
        // engine thread, for good when the wait's release depends on it.
        let started = std::time::Instant::now();
        library.tp2_native_initialize()?;
        tracing::info!(elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
            "native kernels loaded before Spark waits");
        library.peer_exchange_initialize()?;
        warm_reducer(library, &intake)?;
        let timeout = transport.timeout();
        let mut native = Native { transport, arenas: arenas.clone(), words,
            pulls: pulls.unwrap_or(Pulls { base: 0 }), slots, ranks: intake.ranks, row_bytes: intake.row_bytes,
            landing: intake.landing, placed: intake.placed,
            receiving: (0..intake.ranks).map(|_| (0, Vec::new(), Vec::new())).collect(), retained: VecDeque::new() };
        let cut = intake.geometry.cut();
        tracing::info!(slots, waves = cut.waves, rows = cut.rows, reserve_rows = intake.geometry.reserve_rows(),
            ranks = intake.ranks, landing = format!("{:#b}", intake.landing), placed = intake.placed,
            staging_bytes = arenas.first().map_or(0, StagingArena::bytes), "Spark exchange worker started");
        let (jobs, inbox) = mpsc::channel();
        let worker = std::thread::Builder::new().name("spark-exchange".into()).spawn(move || {
            // Short sleeps between a wave's post and its expected completion
            // must wake on time (see `SLEEP`).
            if let Err(error) = cuteafd_ffi::set_thread_timer_slack_ns(1_000) {
                tracing::warn!("Spark exchange worker keeps the default timer slack: {error:#}");
            }
            let (result, highest) = run_worker(&mut native, &inbox, timeout, timing);
            let result = finish_worker(&mut native, &inbox, result, highest);
            // Frames still held (a failure, or waves never consumed) go back
            // with the transport: the link keeps them until its streams drained.
            let frames = native.frames();
            (Box::new((native.transport, frames)) as Box<dyn std::any::Any + Send>, result)
        })?;
        Ok(Self { library, jobs: Some(jobs), worker: Some(worker), _control: control, words, arenas, arena_alias,
            _pull_block: pull_block, pulls, pinned, ranges: intake.ranges, layouts: vec![None; slots],
            unread: vec![false; slots], uses: vec![0; slots], ranks: intake.ranks, row_bytes: intake.row_bytes,
            next_slot: intake.next_slot % cut.waves, geometry: intake.geometry, sequence: 0,
            outstanding: VecDeque::new(), grouping })
    }

    /// Device planes of `slot` under the current cut, in rank order.
    fn slot_planes(&self, slot: usize) -> impl Iterator<Item = CuteafdDeviceBuffer> + '_ {
        let (offset, bytes) = self.geometry.slice(slot, self.row_bytes);
        self.ranges.iter().map(move |range| CuteafdDeviceBuffer {
            ptr: (range.ptr as usize + offset) as *mut c_void, bytes, ..*range })
    }

    /// The workers' installed expert grouping (from the bootstrap).
    pub(crate) fn installed_grouping(&self) -> &[cuteafd_transport::worker_selection::InstalledExpertGrouping] {
        &self.grouping
    }

    /// Whether each slot's request staging holds `bytes`.
    pub(crate) fn fits(&self, bytes: usize) -> bool { self.arenas.iter().all(|arena| arena.bytes() >= bytes) }

    fn submit(&mut self, rows: usize, source: Source, stage: Option<(*mut c_void, RouteStaging,
        [CuteafdDeviceBuffer; 3])>) -> Result<WaveTicket> {
        let cut = self.geometry.cut();
        ensure!(self.outstanding.len() < cut.waves, WaveError::Admission(self.outstanding.len()));
        ensure!(rows > 0 && rows <= cut.rows, WaveError::Rows { rows, capacity: cut.rows });
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
            // The staging copies are SM copies into the arena's mapped alias:
            // copy-engine work is not held by a stream memory wait, so a
            // `cudaMemcpyAsync` here could overwrite the arena while the
            // worker still reads the slot's previous wave.
            let alias = self.arena_alias[slot];
            let sizes = [layout.route_bytes, layout.route_bytes, layout.wire_bytes];
            let offsets = [0, layout.route_bytes, 2 * layout.route_bytes];
            ensure!(offsets[2] + sizes[2] == layout.bytes && layout.bytes <= arena.bytes(),
                "Spark staging regions run past their {}-byte arena", arena.bytes());
            let regions: Vec<(*mut c_void, *const c_void, u64)> = (0..3).map(|i| {
                ensure!(sources[i].bytes >= sizes[i] && !sources[i].ptr.is_null(), "Spark staging source is short or null");
                Ok(((alias + offsets[i]) as *mut c_void, sources[i].ptr.cast_const(), sizes[i] as u64))
            }).collect::<Result<_>>()?;
            unsafe {
                let previous = staging_reuse(sequence, cut.waves);
                self.library.cuda_stream_wait_value32(stream, self.words.ptr(Counter::Landed), previous)?;
                self.library.sm_copy_regions(&regions, stream)?;
                self.library.cuda_stream_write_value32(stream, self.words.ptr(Counter::Staged(slot)), sequence)?;
            }
            self.layouts[slot] = Some(layout);
        } else {
            self.layouts[slot] = None;
        }
        cuteafd_transport::wait_policy::note_active();
        let consumed = vec![(slot, self.uses[slot])];
        let slice = self.geometry.slice(slot, self.row_bytes);
        let release = self.uses[slot].wrapping_add(1);
        let job = Job { sequence, slot, rows, source, consumed, slice, release, sent: Instant::now() };
        self.jobs.as_ref().context("Spark exchange stopped")?.send(Command::Wave(job))
            .map_err(|_| WaveError::Lost(sequence))?;
        self.sequence = sequence;
        self.outstanding.push_back((sequence, slot, 0));
        self.next_slot = (slot + 1) % cut.waves;
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
    ///
    /// Pinned ranks: a wide wave queues, per rank not yet pulled
    /// ([`Self::pull_arrived`]), a wait for the rank's arrival and its pull, so
    /// early ranks cross PCIe while later ones are on the wire; a decode-sized
    /// wave pulls every rank in one launch behind `landed`.
    pub(crate) fn land(&mut self, ticket: WaveTicket, stream: *mut c_void) -> Result<()> {
        let (expected, slot, pulled) = *self.outstanding.front().ok_or(WaveError::Lost(ticket.sequence))?;
        ensure!(expected == ticket.sequence && slot == ticket.slot,
            WaveError::Order { expected, found: ticket.sequence });
        let cut = self.geometry.cut();
        ensure!(ticket.rows <= cut.rows, "Spark wave of {} rows lands past its {}-row planes", ticket.rows, cut.rows);
        let rest = self.pinned & !pulled;
        if rest != 0 && self.wide(ticket.rows, rest) {
            for rank in (0..self.ranks).filter(|rank| rest & (1 << rank) != 0) {
                // SAFETY: as `pull`'s contract; the wait precedes the pull.
                unsafe { self.pull(ticket.sequence, slot, ticket.rows, 1 << rank, true, stream)? };
            }
        }
        self.outstanding.pop_front();
        // SAFETY: as `submit`'s waits.
        unsafe { self.library.cuda_stream_wait_value32(stream, self.words.ptr(Counter::Landed), ticket.sequence)? };
        if rest != 0 && !self.wide(ticket.rows, rest) {
            // SAFETY: as `pull`'s contract; behind the `landed` wait.
            unsafe { self.pull(ticket.sequence, slot, ticket.rows, rest, false, stream)? };
        }
        self.unread[slot] = true;
        Ok(())
    }

    /// Whether `ranks` of a `rows`-row wave carry enough to pull rank by rank.
    fn wide(&self, rows: usize, ranks: u8) -> bool {
        rows * self.row_bytes * ranks.count_ones() as usize > PULL_PER_RANK_BYTES
    }

    /// For a host-ordered owner waiting on the host for `ticket`'s wave (the
    /// oldest): queues on `stream` the pull of every pinned rank of a wide
    /// wave whose rows arrived since the last call, so they cross PCIe while
    /// the rest are on the wire; [`Self::land`] then queues only what is
    /// left. Never waits. `stream` is the one the wave lands and reduces on.
    pub(crate) fn pull_arrived(&mut self, ticket: &WaveTicket, stream: *mut c_void) -> Result<()> {
        if !self.wide(ticket.rows, self.pinned) { return Ok(()); }
        let Some(front) = self.outstanding.front_mut() else { return Err(WaveError::Lost(ticket.sequence).into()) };
        ensure!(front.0 == ticket.sequence && front.1 == ticket.slot,
            WaveError::Order { expected: front.0, found: ticket.sequence });
        let mut arrived = 0u8;
        for rank in (0..self.ranks).filter(|rank| self.pinned & !front.2 & (1 << rank) != 0) {
            if reached(self.words.load(Counter::Arrived(rank)), ticket.sequence) { arrived |= 1 << rank; }
        }
        if arrived == 0 { return Ok(()); }
        front.2 |= arrived;
        // SAFETY: as `pull`'s contract; every rank's arrival is already
        // published (the wait it queues is satisfied).
        unsafe { self.pull(ticket.sequence, ticket.slot, ticket.rows, arrived, true, stream) }
    }

    /// Queues on `stream` the pull of `ranks`' published frames of wave
    /// `sequence` (in `slot`, `rows` rows) into their planes, each behind its
    /// own arrival wait when `wait`.
    ///
    /// # Safety
    /// `stream` is a live stream of the link's device, ordered after the
    /// slot's previous reduce; a pull without its waits sits behind `landed`.
    unsafe fn pull(&self, sequence: u32, slot: usize, rows: usize, ranks: u8, wait: bool, stream: *mut c_void)
        -> Result<()> {
        let pulls = self.pulls.context("no pinned ranks to pull")?;
        let planes: Vec<CuteafdDeviceBuffer> = self.slot_planes(slot).collect();
        let mut replies = Vec::new();
        let mut targets = Vec::new();
        for rank in (0..self.ranks).filter(|rank| ranks & (1 << rank) != 0) {
            if wait {
                // SAFETY: as `submit`'s waits (the worker stores every rank's
                // arrival, or releases it on failure).
                unsafe { self.library.cuda_stream_wait_value32(stream, self.words.ptr(Counter::Arrived(rank)), sequence)? };
            }
            replies.push(pulls.at(slot, rank).cast_const());
            targets.push((planes[rank].ptr, (rows * self.row_bytes) as u64));
        }
        // SAFETY: the kernel runs after the arrival (or `landed`) wait, when
        // the worker has published the rank's frames, which it keeps in their
        // receive slots until this slot's `consumed` (written after the
        // reduce that follows this pull on the stream); the planes are the
        // slot's, unread before the reduce. Descriptors and planes outlive the
        // launch (the link's owner drains its streams before dropping it).
        unsafe { self.library.pull_frames(&replies, &targets, stream) }
    }

    /// Whether `ticket`'s wave has landed (or the wire failed, which releases
    /// every wave): a host check, for host-ordered owners that hold a wave's
    /// [`Self::land`] until it completes so no stream sits blocked on a device
    /// wait through the round trip.
    pub(crate) fn landed(&self, ticket: &WaveTicket) -> bool {
        reached(self.words.load(Counter::Landed), ticket.sequence)
    }

    /// Plane pointers of return slot `slot` (rank order), for the reduce.
    pub(crate) fn planes(&self, slot: usize) -> [*const u16; MAX_INTAKE_RANKS] {
        let mut pointers = [std::ptr::null::<u16>(); MAX_INTAKE_RANKS];
        for (pointer, plane) in pointers.iter_mut().zip(self.slot_planes(slot)) { *pointer = plane.ptr.cast(); }
        pointers
    }

    pub(crate) fn ranks(&self) -> usize { self.ranks }

    /// The counter words' base address (tests: diagnosis from the worker).
    #[cfg(test)]
    pub(crate) fn words(&self) -> Words { self.words }

    /// Marks `slot`'s planes read once `stream` reaches this point (after the
    /// reduce): the worker posts the slot's next wave only after it.
    pub(crate) fn consumed(&mut self, slot: usize, stream: *mut c_void) -> Result<()> {
        ensure!(slot < self.geometry.cut().waves && self.unread[slot], "return slot {slot} has no landed wave to consume");
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


    /// Return slots of the current cut.
    pub(crate) fn waves(&self) -> usize { self.geometry.cut().waves }

    /// Stops the worker and returns the transport (after a failure, with its
    /// waves abandoned) and the slot its next wave lands in. Every sent wave
    /// must have been landed and consumed, and the consuming streams drained.
    pub(crate) fn stop<T: Transport>(mut self) -> Result<(T, Vec<T::Frame>, usize)> {
        ensure!(self.outstanding.is_empty() && self.unread.iter().all(|unread| !unread),
            "Spark exchange stopped with waves not landed and consumed");
        drop(self.jobs.take());
        let worker = self.worker.take().context("Spark exchange already stopped")?;
        let (transport, result) = worker.join().map_err(|_| anyhow::anyhow!("Spark exchange worker panicked"))?;
        result?;
        let (transport, frames) = *transport.downcast::<(T, Vec<T::Frame>)>()
            .map_err(|_| anyhow::anyhow!("Spark exchange transport type differs"))?;
        Ok((transport, frames, self.next_slot))
    }

    /// Terminal teardown (#784): closes the worker and returns its transport
    /// whatever state the exchange is in (at rest, failed, or with waves the
    /// engine will never land: the worker then abandons them and releases
    /// every queued wait), with the worker's result, so the link can quiesce
    /// and release the QPs itself. Only a panicked worker loses the transport.
    pub(crate) fn close<T: Transport>(self) -> Result<(T, Vec<T::Frame>, Result<()>)> {
        self.join_worker(false)
    }

    /// A reset's stop (a cancelled request, #784): as [`Self::close`], but
    /// waves still outstanding are given up without marking the process
    /// unhealthy; the worker releases every queued wait over them.
    pub(crate) fn cancel<T: Transport>(self) -> Result<(T, Vec<T::Frame>, Result<()>)> {
        self.join_worker(true)
    }

    fn join_worker<T: Transport>(mut self, cancel: bool) -> Result<(T, Vec<T::Frame>, Result<()>)> {
        if cancel {
            // A worker that already failed has hung up; its result says so.
            if let Some(jobs) = &self.jobs { let _ = jobs.send(Command::Cancel); }
        }
        drop(self.jobs.take());
        let worker = self.worker.take().context("Spark exchange already stopped")?;
        let (transport, result) = worker.join().map_err(|_| {
            // Waits may still be queued against these words: keep them.
            self.library.quarantine_module_after_failed_drain();
            anyhow::anyhow!("Spark exchange worker panicked")
        })?;
        let (transport, frames) = *transport.downcast::<(T, Vec<T::Frame>)>()
            .map_err(|_| anyhow::anyhow!("Spark exchange transport type differs"))?;
        Ok((transport, frames, result))
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
