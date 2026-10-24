//! Prefill expert waves on a transport worker (PLAN "Prefill contract").
//!
//! The engine thread only queues GPU work. A prefill call lends its one Spark
//! link to [`SparkLink::with_waves`], which runs a worker thread for the call.
//! The worker owns the link's QPs and CQs and does all host work for a wave:
//!
//! 1. [`Waves::send`] (engine thread) queues the wave's route ids, gate
//!    weights and expert input rows from the GPU into its return slot's
//!    pinned staging arena on the engine stream, records the slot's `staged`
//!    event and hands the worker a job. It never waits for the GPU.
//! 2. The worker waits without blocking until `staged` completes, plus the
//!    slot's `consumed` event when a reduce read the slot's planes since its
//!    last wave. It then builds the request from the arena (the hidden payload
//!    is a view of the arena, so there is no host copy) and posts it to every
//!    rank. It also progresses the oldest wave: pinned replies upload into the
//!    slot's planes on the intake copy stream, GPU-landed ranks need no upload,
//!    and the slot's `landed` event is recorded on the copy stream before the
//!    engine hears of the wave.
//! 3. [`Waves::land`] (engine thread) takes the worker's completion for the
//!    oldest wave and orders the engine stream after `landed`
//!    (`cudaStreamWaitEvent`). The family then reduces the slot's planes and
//!    calls [`Waves::consumed`] or uses [`Waves::reduce`].
//!
//! One worker drives one link, so no QP or CQ is shared between threads and
//! waves stay FIFO on every rank, with no locks. Waves land in dispatch order,
//! which is the executor's order (`prefill_schedule`).
//!
//! Ownership: a slot's arena is rewritten only by the slot's next send, after
//! its wave landed and no request views it. A reply's receive slot is held
//! until the copy stream has passed its upload. Failure: a transport error
//! ends the worker, which drains the copy stream and hands its held receive
//! slots back to the intake. Waves still in flight are abandoned (the link
//! resets them). The family's submission guard then drains its streams and
//! releases storage as for any failed wave.
use crate::shared::experts::staging::RouteStaging;
use crate::shared::spark_intake::{upload_offset, SparkLink, MAX_INTAKE_RANKS};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_transport::expert::{SparkExperts, SparkExpertWave, WaveReceipt};
use cuteafd_transport::{ExpertProtocolV2Request, ExpertProtocolV2RouteEntry, StagingArena,
    VerbsHostProtocolV2ResponsePayload};
use std::collections::VecDeque;
use std::ffi::c_void;
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
    sequence: u64,
    slot: usize,
    rows: usize,
}

impl WaveTicket {
    /// The return slot this wave's planes land in.
    pub(crate) fn slot(&self) -> usize { self.slot }
    /// The wave's rows.
    pub(crate) fn rows(&self) -> usize { self.rows }
}

/// Host seconds of the waves of one call, for the families' profiles.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct WaveTimes {
    /// Worker: from a job's arrival until its staged copies were complete.
    pub staged: f64,
    /// Worker: request build and post.
    pub posted: f64,
    /// Worker: from the post until the last rank's rows arrived.
    pub received: f64,
    /// Engine thread: waiting for the worker in [`Waves::land`].
    pub waited: f64,
}

impl WaveTimes {
    fn add(&mut self, other: Self) {
        self.staged += other.staged;
        self.posted += other.posted;
        self.received += other.received;
        self.waited += other.waited;
    }
}

/// Typed failures of the prefill transport stage.
#[derive(Debug, thiserror::Error)]
pub(crate) enum WaveError {
    #[error("prefill transport worker failed: {0:#}")]
    Worker(anyhow::Error),
    #[error("prefill transport worker exited before landing wave {0}")]
    Lost(u64),
    #[error("prefill wave {found} landed out of order (expected {expected})")]
    Order { expected: u64, found: u64 },
    #[error("prefill wave admission exceeded: {0} waves outstanding")]
    Admission(usize),
    #[error("prefill wave of {rows} rows exceeds the {capacity}-row return slot")]
    Rows { rows: usize, capacity: usize },
    #[error("prefill wave {0} did not land within {1:?}")]
    Timeout(u64, Duration),
}

/// A CUDA event or stream handle crossing to the worker. Handles are
/// process-wide; the link owns them and outlives the worker (scoped thread).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Handle(usize);
impl Handle {
    pub(crate) fn of(raw: *mut c_void) -> Self { Self(raw as usize) }
    pub(crate) fn raw(self) -> *mut c_void { self.0 as *mut c_void }
}

pub(crate) struct Job {
    sequence: u64,
    slot: usize,
    rows: usize,
    layout: RouteStaging,
    build: BuildRequest,
    /// The slot's previous reduce, when one read its planes since its last wave.
    consumed: Option<Handle>,
    arrived: Instant,
}

pub(crate) struct Landed {
    sequence: u64,
    times: WaveTimes,
}

/// What the worker needs from the GPU and the transport; [`Native`] in
/// serving, a fake in the CPU tests.
pub(crate) trait Backend {
    type Wave;
    type Payload;
    /// Whether `event`'s recorded work is complete (never blocks).
    fn complete(&self, event: Handle) -> Result<bool>;
    /// The slot's `staged` event.
    fn staged(&self, slot: usize) -> Handle;
    /// Builds and posts the job's request from its slot's arena.
    fn post(&mut self, job: Job) -> Result<Self::Wave>;
    /// One progress pass over the oldest wave; each arrived row range
    /// `(rank, first row, payload)` is appended to `arrived`. `Some(landed)`
    /// once complete (bit r: rank r landed in device memory).
    fn poll(&mut self, wave: &mut Self::Wave, arrived: &mut Vec<(usize, u32, Self::Payload)>)
        -> Result<Option<u8>>;
    /// Queues `payload`'s upload into `slot`'s plane of `rank` on the copy stream.
    fn upload(&self, slot: usize, rows: usize, rank: usize, first: u32, payload: &Self::Payload) -> Result<()>;
    /// Records `slot`'s `landed` event on the copy stream (after its uploads).
    fn landed(&self, slot: usize) -> Result<Handle>;
    /// Waits for every queued upload (failure and exit).
    fn drain(&self) -> Result<()>;
}

struct Inflight<W, P> {
    sequence: u64,
    slot: usize,
    rows: usize,
    wave: W,
    held: Vec<(usize, u32, P)>,
    posted: Instant,
    times: WaveTimes,
}

/// How the worker ended.
pub(crate) struct Retired<P> {
    /// Receive owners `(rank, first row, payload)` of queued uploads.
    pub held: Vec<(usize, u32, P)>,
    /// Whether the copy stream drained, so every held upload completed.
    pub drained: bool,
    /// Waves posted or queued that never landed (an error or an early close).
    pub abandoned: bool,
    pub error: Option<anyhow::Error>,
}

/// The worker's loop. It never blocks on one condition while another could
/// progress: it posts a job only once the job's GPU copies are complete,
/// keeps progressing the oldest wave meanwhile (the engine stream may be
/// waiting on that wave's `landed` before it reaches the next copies), and
/// returns receive slots as soon as their uploads pass. Idle, it blocks on
/// its job channel.
pub(crate) fn run_worker<B: Backend>(backend: &mut B, inbox: mpsc::Receiver<Job>,
    report: mpsc::Sender<Result<Landed>>, timeout: Duration) -> Retired<B::Payload> {
    let mut queued = VecDeque::<Job>::new();
    let mut inflight = VecDeque::<Inflight<B::Wave, B::Payload>>::new();
    let mut retiring = VecDeque::<(Handle, Vec<(usize, u32, B::Payload)>)>::new();
    let mut closed = false;
    let mut idle_since: Option<Instant> = None;
    let result = (|| -> Result<()> {
        loop {
            let mut progressed = false;
            // Jobs: block only when nothing else is pending.
            if queued.is_empty() && inflight.is_empty() && retiring.is_empty() && !closed {
                match inbox.recv() {
                    Ok(job) => { queued.push_back(job); progressed = true; }
                    Err(_) => closed = true,
                }
            }
            loop {
                match inbox.try_recv() {
                    Ok(job) => { queued.push_back(job); progressed = true; }
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => { closed = true; break; }
                }
            }
            if closed && (!queued.is_empty() || !inflight.is_empty()) {
                // The engine stopped before landing these: it is unwinding.
                return Ok(());
            }
            // Receive slots go back once the copy stream passed their uploads.
            while let Some((event, _)) = retiring.front() {
                if !backend.complete(*event)? { break; }
                retiring.pop_front();
                progressed = true;
            }
            // Post the next job once its rows are staged and its slot is free.
            if let Some(job) = queued.front() {
                let ready = backend.complete(backend.staged(job.slot))?
                    && job.consumed.map_or(Ok(true), |event| backend.complete(event))?;
                if ready {
                    let job = queued.pop_front().expect("front job");
                    let (sequence, slot, rows) = (job.sequence, job.slot, job.rows);
                    let mut times = WaveTimes { staged: job.arrived.elapsed().as_secs_f64(), ..Default::default() };
                    let started = Instant::now();
                    let wave = backend.post(job)?;
                    times.posted = started.elapsed().as_secs_f64();
                    inflight.push_back(Inflight { sequence, slot, rows, wave, held: Vec::new(),
                        posted: Instant::now(), times });
                    progressed = true;
                }
            }
            // Progress the oldest wave.
            if let Some(front) = inflight.front_mut() {
                let already = front.held.len();
                // Arrived owners are held before their uploads are queued: an
                // enqueue error after a copy was accepted keeps its source alive.
                let complete = backend.poll(&mut front.wave, &mut front.held);
                for (rank, first, payload) in &front.held[already..] {
                    backend.upload(front.slot, front.rows, *rank, *first, payload)?;
                    progressed = true;
                }
                if complete?.is_some() {
                    let mut done = inflight.pop_front().expect("front wave");
                    let event = backend.landed(done.slot)?;
                    done.times.received = done.posted.elapsed().as_secs_f64();
                    retiring.push_back((event, std::mem::take(&mut done.held)));
                    if report.send(Ok(Landed { sequence: done.sequence, times: done.times })).is_err() {
                        closed = true;
                    }
                    progressed = true;
                }
            }
            if closed && queued.is_empty() && inflight.is_empty() {
                return Ok(());
            }
            if progressed {
                idle_since = None;
            } else {
                let since = *idle_since.get_or_insert_with(Instant::now);
                // Staged copies and slot reuse wait on the GPU; replies have their
                // own transport deadline inside `poll`.
                ensure!(since.elapsed() < timeout, "prefill transport worker made no progress for {timeout:?}");
                std::hint::spin_loop();
                if since.elapsed() > Duration::from_micros(50) { std::thread::yield_now(); }
            }
        }
    })();
    let abandoned = !queued.is_empty() || !inflight.is_empty();
    // Every source must outlive its queued copy, also after an error: drain
    // first, then hand the owners back.
    let drain = backend.drain();
    let drained = drain.is_ok();
    let error = match (result, drain) {
        (Err(error), _) => Some(error),
        (Ok(()), Err(error)) => Some(error.context("draining prefill uploads")),
        (Ok(()), Ok(())) => None,
    };
    if let Some(error) = &error {
        let _ = report.send(Err(anyhow::anyhow!("{error:#}")));
    }
    let mut held: Vec<(usize, u32, B::Payload)> = retiring.into_iter().flat_map(|(_, held)| held).collect();
    held.extend(inflight.into_iter().flat_map(|wave| wave.held));
    Retired { held, drained, abandoned, error }
}

/// The engine thread's side of a running prefill transport.
pub(crate) struct Waves<'w, 'a> {
    library: &'a NativeLibrary,
    intake: &'w crate::shared::spark_intake::SparkIntake<'a>,
    jobs: Option<mpsc::Sender<Job>>,
    done: mpsc::Receiver<Result<Landed>>,
    staged: &'w [Handle],
    arenas: &'w [StagingArena],
    /// Per slot: the layout of its last sent wave (route bytes for probes).
    layouts: Vec<Option<RouteStaging>>,
    /// Per slot: landed and not yet marked read ([`Self::consumed`]); the
    /// slot takes no new wave until its reduce is ordered.
    unread: Vec<bool>,
    capacity: usize,
    waves: usize,
    next_slot: usize,
    sequence: u64,
    outstanding: VecDeque<(u64, usize)>,
    timeout: Duration,
    times: WaveTimes,
}

impl Waves<'_, '_> {
    /// Queues a `rows`-row wave into the next return slot: its route ids, gate
    /// weights and expert input rows (`sources`, in that order) copy into the
    /// slot's staging arena on `stream`, and the worker builds, posts and
    /// lands it once those copies complete. The engine thread does not wait.
    ///
    /// # Safety
    /// `sources` are live device buffers of `layout`'s extents on `stream`'s
    /// device, written in `stream` order before this call (the copies read
    /// them in stream order, so later stream work may rewrite them).
    pub(crate) unsafe fn send(&mut self, rows: usize, layout: RouteStaging, sources: [CuteafdDeviceBuffer; 3],
        stream: *mut c_void, build: BuildRequest) -> Result<WaveTicket> {
        ensure!(self.outstanding.len() < self.waves, WaveError::Admission(self.outstanding.len()));
        ensure!(rows > 0 && rows <= self.capacity, WaveError::Rows { rows, capacity: self.capacity });
        let slot = self.next_slot;
        let arena = &self.arenas[slot];
        // The slot's previous wave landed (admission and slot order), and its
        // request was released with its last reply.
        ensure!(arena.writable(), "prefill staging arena {slot} is still viewed by an unfinished request");
        ensure!(!self.unread[slot], "return slot {slot} takes a new wave before its reduce was marked consumed");
        ensure!(layout.bytes <= arena.bytes(), "wave staging of {} bytes exceeds its {}-byte arena",
            layout.bytes, arena.bytes());
        // SAFETY: the arena is pinned, long enough (checked) and unviewed, and
        // the copies that last read it (the previous request's) are done; the
        // caller vouches for the sources and the stream.
        unsafe {
            layout.enqueue(self.library, arena.host(), sources, stream)?;
            self.library.cuda_event_record(self.staged[slot].raw(), stream)?;
        }
        let sequence = self.sequence + 1;
        let consumed = self.intake.take_consumed(slot).map(Handle::of);
        let job = Job { sequence, slot, rows, layout, build, consumed, arrived: Instant::now() };
        self.jobs.as_ref().context("prefill transport worker closed")?.send(job)
            .map_err(|_| WaveError::Lost(sequence))?;
        self.sequence = sequence;
        self.layouts[slot] = Some(layout);
        self.outstanding.push_back((sequence, slot));
        self.next_slot = (slot + 1) % self.waves;
        Ok(WaveTicket { sequence, slot, rows })
    }

    /// Waits for `ticket`'s wave (the oldest) and orders `stream` after its
    /// planes, which are readable on `stream` from here on. Mark them read
    /// with [`Self::consumed`] after the reduce, or reduce with [`Self::reduce`].
    pub(crate) fn land(&mut self, ticket: WaveTicket, stream: *mut c_void) -> Result<WaveTimes> {
        let (expected, _) = self.outstanding.pop_front().ok_or(WaveError::Lost(ticket.sequence))?;
        ensure!(expected == ticket.sequence, WaveError::Order { expected, found: ticket.sequence });
        let waited = Instant::now();
        let landed = match self.done.recv_timeout(self.timeout) {
            Ok(result) => result.map_err(WaveError::Worker)?,
            Err(mpsc::RecvTimeoutError::Timeout) => return Err(WaveError::Timeout(ticket.sequence, self.timeout).into()),
            Err(mpsc::RecvTimeoutError::Disconnected) => return Err(WaveError::Lost(ticket.sequence).into()),
        };
        ensure!(landed.sequence == ticket.sequence, WaveError::Order { expected: ticket.sequence, found: landed.sequence });
        // SAFETY: live handles; the worker recorded the event on the copy
        // stream after the slot's uploads, before reporting the wave.
        unsafe { self.library.cuda_stream_wait_event(stream, self.intake.landed_event(ticket.slot))? };
        self.intake.landed_into(ticket.slot);
        self.unread[ticket.slot] = true;
        let mut times = landed.times;
        times.waited = waited.elapsed().as_secs_f64();
        self.times.add(times);
        Ok(times)
    }

    /// Plane pointers of return slot `slot` (rank order), for the reduce.
    pub(crate) fn planes(&self, slot: usize) -> [*const u16; MAX_INTAKE_RANKS] {
        self.intake.slot_pointers(slot)
    }

    pub(crate) fn ranks(&self) -> usize { self.intake.ranks() }

    /// Marks `slot`'s planes read once `stream` reaches this point (after the
    /// reduce): the slot's next wave is posted only after it.
    pub(crate) fn consumed(&mut self, slot: usize, stream: *mut c_void) -> Result<()> {
        ensure!(slot < self.waves && self.unread[slot], "return slot {slot} has no landed wave to consume");
        self.intake.mark_consumed(slot, stream)?;
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
            self.library.v41_compact_reducer()?.reduce_planes(self.planes(slot), self.ranks() as u32, shared,
                output, u32::try_from(rows)?, stream)?;
        }
        self.consumed(slot, stream)
    }

    /// The staged route ids (`rows * topk` little-endian u32) of `slot`'s last
    /// landed wave, for draft probes. Valid until the slot's next send.
    pub(crate) fn staged_ids(&self, slot: usize) -> Result<&[u8]> {
        let layout = self.layouts[slot].context("no wave staged in this slot")?;
        ensure!(self.outstanding.iter().all(|&(_, pending)| pending != slot), "slot {slot}'s wave has not landed");
        // SAFETY: the wave landed, so its staged copies completed (the worker
        // saw them before posting); only this slot's next send rewrites it.
        let bytes = unsafe { self.arenas[slot].read(layout.bytes)? };
        Ok(&bytes[..layout.route_bytes])
    }

    pub(crate) fn waves(&self) -> usize { self.waves }
    pub(crate) fn outstanding(&self) -> usize { self.outstanding.len() }
    /// Sums over every wave landed so far in this call.
    pub(crate) fn times(&self) -> WaveTimes { self.times }
}

/// The serving backend: the link's transport, its intake's planes, copy
/// stream and events, and the per-slot staging arenas.
struct Native<'t> {
    library: &'t NativeLibrary,
    device: i32,
    transport: &'t mut SparkExperts,
    arenas: &'t [StagingArena],
    staged: &'t [Handle],
    landed: Vec<Handle>,
    planes: Vec<Vec<(usize, usize)>>,
    ranks: usize,
    row_bytes: usize,
    copy_stream: Handle,
}

impl Backend for Native<'_> {
    type Wave = (SparkExpertWave, WaveReceipt);
    type Payload = VerbsHostProtocolV2ResponsePayload;

    fn complete(&self, event: Handle) -> Result<bool> {
        // SAFETY: a live event of the link's device (current on this thread).
        unsafe { self.library.cuda_event_query(event.raw()) }
    }

    fn staged(&self, slot: usize) -> Handle { self.staged[slot] }

    fn post(&mut self, job: Job) -> Result<Self::Wave> {
        let arena = &self.arenas[job.slot];
        let layout = job.layout;
        // SAFETY: the staged event completed, so the engine's copies into the
        // arena are done; nothing writes the arena until this wave lands.
        let staged = unsafe { arena.read(layout.bytes)? };
        let routes = layout.routes(staged)?;
        let wire = arena.payload(2 * layout.route_bytes, layout.wire_bytes)?;
        let request = (job.build)(routes, wire)?;
        let wave = self.transport.dispatch_wave(&request)?;
        Ok((wave, WaveReceipt::default()))
    }

    fn poll(&mut self, wave: &mut Self::Wave, arrived: &mut Vec<(usize, u32, Self::Payload)>)
        -> Result<Option<u8>> {
        let (wave, receipt) = wave;
        let done = self.transport.poll_wave(wave, receipt, |rank, first, payload| {
            arrived.push((rank, first, payload));
            Ok(())
        })?;
        Ok(done.then_some(receipt.landed))
    }

    fn upload(&self, slot: usize, rows: usize, rank: usize, first: u32, payload: &Self::Payload) -> Result<()> {
        let bytes = payload.as_ref().len();
        let plane_bytes = rows * self.row_bytes;
        let (base, capacity) = self.planes[slot].get(rank).copied().context("rank outside the intake planes")?;
        ensure!(plane_bytes <= capacity, "wave of {rows} rows exceeds its return slot");
        let offset = upload_offset(self.ranks, rank, first, self.row_bytes, bytes, plane_bytes)?;
        let destination = CuteafdDeviceBuffer { ptr: (base + offset) as *mut c_void, bytes,
            device_id: self.device, flags: 0 };
        match payload.pinned_host_buffer() {
            // SAFETY: the pinned receive slot is held until the copy stream
            // passes this copy (`landed`); the range lies inside the plane.
            Some(source) => unsafe {
                self.library.copy_host_buffer_h2d_async(destination, source, bytes, self.copy_stream.raw())
            },
            // SAFETY: a pageable source is staged by the driver before return.
            None => unsafe { self.library.copy_h2d_async(destination, payload.as_ref(), self.copy_stream.raw()) },
        }
    }

    fn landed(&self, slot: usize) -> Result<Handle> {
        // SAFETY: live event and stream of the link's device.
        unsafe { self.library.cuda_event_record(self.landed[slot].raw(), self.copy_stream.raw())? };
        Ok(self.landed[slot])
    }

    fn drain(&self) -> Result<()> {
        // SAFETY: the intake's copy stream; its sources are still held.
        unsafe { self.library.cuda_stream_synchronize(self.copy_stream.raw()) }
    }
}

/// Per-link staging for prefill waves: one pinned arena and one `staged`
/// event per return slot, allocated on a link's first prefill call.
pub(crate) struct PrefillStaging<'a> {
    library: &'a NativeLibrary,
    arenas: Vec<StagingArena>,
    staged: Vec<Handle>,
}

impl<'a> PrefillStaging<'a> {
    /// A single-wave link that already registered a request buffer large
    /// enough stages there, so its requests keep going out zero-copy.
    fn new(library: &'a NativeLibrary, waves: usize, bytes: usize, egress: Option<StagingArena>) -> Result<Self> {
        let _scope = cuteafd_ffi::memory_ledger::scope("spark-intake");
        let arenas = match egress.filter(|arena| waves == 1 && arena.bytes() >= bytes) {
            Some(arena) => vec![arena],
            None => (0..waves).map(|_| StagingArena::new(bytes)).collect::<Result<Vec<_>>>()?,
        };
        let staged = (0..waves).map(|_| library.cuda_event_create_ordering().map(Handle::of))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { library, arenas, staged })
    }
    fn fits(&self, waves: usize, bytes: usize) -> bool {
        self.arenas.len() == waves && self.arenas.iter().all(|arena| arena.bytes() >= bytes)
    }
}

impl Drop for PrefillStaging<'_> {
    fn drop(&mut self) {
        for event in &self.staged {
            // SAFETY: created here; synchronizing retires its last record.
            unsafe {
                let _ = self.library.cuda_event_synchronize(event.raw());
                let _ = self.library.cuda_event_destroy(event.raw());
            }
        }
    }
}

impl<'a> SparkLink<'a> {
    /// Runs `body` with a transport worker driving this link's waves, each
    /// staging up to `staging_bytes` (a [`RouteStaging`] of the widest wave).
    /// See [`with_link_waves`].
    pub(crate) fn with_waves<T>(&mut self, staging_bytes: usize,
        body: impl FnOnce(&mut Waves<'_, 'a>) -> Result<T>) -> Result<T> {
        with_link_waves(&mut [self], staging_bytes, |waves| body(&mut waves[0]))
    }

    fn prepare_waves(&mut self, staging_bytes: usize) -> Result<()> {
        let intake = &self.intake;
        ensure!(self.transport.open_waves() == 0, "a prefill transport needs a link with no wave in flight");
        ensure!(intake.dispatch_slot() == intake.receive_slot(), "a prefill transport needs a drained intake");
        // Receive slots of the link's last inline wave go back first.
        intake.release_uploads()?;
        let waves = intake.waves();
        if !self.prefill.as_ref().is_some_and(|staging| staging.fits(waves, staging_bytes)) {
            self.prefill = None;
            self.prefill = Some(PrefillStaging::new(intake.library(), waves, staging_bytes,
                self.transport.egress_arena())?);
        }
        Ok(())
    }
}

/// Runs `body` with one transport worker per link, each driving its link's
/// waves (index `i` of `body`'s slice is `links[i]`). Every wave `body` sends
/// must land before it returns. On any error each worker stops, drains its
/// copy stream and hands back its receive slots, and waves still in flight
/// are reset before the error propagates.
pub(crate) fn with_link_waves<'a, T>(links: &mut [&mut SparkLink<'a>], staging_bytes: usize,
    body: impl FnOnce(&mut [Waves<'_, 'a>]) -> Result<T>) -> Result<T> {
    for link in links.iter_mut() { link.prepare_waves(staging_bytes)?; }
    let intakes: Vec<_> = links.iter().map(|link| std::rc::Rc::clone(&link.intake)).collect();
    let mut natives = Vec::with_capacity(links.len());
    let mut stages = Vec::with_capacity(links.len());
    let mut channels = Vec::with_capacity(links.len());
    for (link, intake) in links.iter_mut().zip(&intakes) {
        let link: &mut SparkLink<'a> = link;
        let library = intake.library();
        let waves = intake.waves();
        let timeout = link.transport.timeout();
        let staging = link.prefill.as_ref().expect("prefill staging");
        natives.push(Native {
            library, device: intake.device(), transport: &mut link.transport,
            arenas: &staging.arenas, staged: &staging.staged,
            landed: (0..waves).map(|slot| Handle::of(intake.landed_event(slot))).collect(),
            planes: (0..waves).map(|slot| intake.slot_buffers(slot).into_iter()
                .map(|plane| (plane.ptr as usize, plane.bytes)).collect()).collect(),
            ranks: intake.ranks(), row_bytes: intake.row_bytes(), copy_stream: Handle::of(intake.copy_stream()),
        });
        let (jobs, inbox) = mpsc::channel::<Job>();
        let (report, done) = mpsc::channel::<Result<Landed>>();
        channels.push((inbox, report, timeout));
        stages.push(Waves {
            library, intake, jobs: Some(jobs), done, staged: &staging.staged, arenas: &staging.arenas,
            layouts: vec![None; waves], unread: vec![false; waves], capacity: intake.rows(), waves,
            next_slot: intake.dispatch_slot(),
            sequence: 0, outstanding: VecDeque::new(), timeout: timeout.saturating_mul(waves as u32 + 1),
            times: WaveTimes::default(),
        });
    }
    let (result, retired) = std::thread::scope(|scope| {
        let mut workers = Vec::with_capacity(natives.len());
        for (mut native, (inbox, report, timeout)) in natives.into_iter().zip(channels) {
            let spawned = std::thread::Builder::new().name("spark-waves".into()).spawn_scoped(scope, move || {
                if let Err(error) = native.library.cuda_set_device(native.device) {
                    let _ = report.send(Err(anyhow::anyhow!("{error:#}")));
                    return Retired { held: Vec::new(), drained: true, abandoned: false, error: Some(error) };
                }
                run_worker(&mut native, inbox, report, timeout)
            });
            match spawned {
                Ok(worker) => workers.push(worker),
                Err(error) => {
                    // Earlier workers end when their job channels close below.
                    for stage in &mut stages { stage.jobs.take(); }
                    for worker in workers { let _ = worker.join(); }
                    return (Err(anyhow::Error::from(error).context("spawning a prefill transport worker")), None);
                }
            }
        }
        let result = body(&mut stages).and_then(|value| {
            let outstanding: usize = stages.iter().map(|stage| stage.outstanding.len()).sum();
            ensure!(outstanding == 0, "prefill finished with {outstanding} waves outstanding");
            Ok(value)
        });
        // Closing a job channel ends its worker: at once when idle, or by
        // abandoning waves the engine will never land after an error.
        for stage in &mut stages { stage.jobs.take(); }
        let retired = workers.into_iter().map(|worker| worker.join()).collect::<Vec<_>>();
        (result, Some(retired))
    });
    let next: Vec<usize> = stages.iter().map(|stage| stage.next_slot).collect();
    drop(stages);
    let Some(retired) = retired else { return result };
    let mut failure = None;
    for (((link, intake), retired), next) in links.iter_mut().zip(&intakes).zip(retired).zip(next) {
        let Ok(retired) = retired else {
            failure.get_or_insert_with(|| anyhow::anyhow!("prefill transport worker panicked"));
            link.transport.abandon_waves("prefill transport worker panicked");
            continue;
        };
        // Drained uploads release their receive slots now. Without a proven
        // drain the intake keeps the owners until its terminal release.
        if retired.drained { drop(retired.held); } else { intake.adopt_uploads(retired.held)?; }
        intake.resume_at(next);
        if retired.abandoned || result.is_err() || retired.error.is_some() {
            link.transport.abandon_waves("prefill call ended with waves in flight");
        }
        if let Some(error) = retired.error { failure.get_or_insert(error.context("prefill transport worker")); }
    }
    match (result, failure) {
        (Err(error), _) => Err(error),
        (Ok(_), Some(error)) => Err(error),
        (Ok(value), None) => Ok(value),
    }
}

#[cfg(test)]
#[path = "spark_waves_tests.rs"]
mod tests;
