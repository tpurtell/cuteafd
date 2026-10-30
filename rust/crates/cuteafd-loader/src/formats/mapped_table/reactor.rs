//! One process-wide owner of miss reads; no caller memory crosses this thread.
use super::{invalid, uring, MappedTableError, ReadPriority, Result};
use io_uring::{opcode, types, IoUring};
use std::{collections::VecDeque, fs::File, os::fd::{AsRawFd, FromRawFd, OwnedFd},
    sync::{Arc, Condvar, Mutex, OnceLock, atomic::{AtomicBool, Ordering}, mpsc},
    time::{Duration, Instant}};

const SLOTS: usize = 16;
const DECODE_SLOTS: usize = 2;
pub(super) const BUFFER_BYTES: usize = 1 << 20;
const WAKE: u64 = u64::MAX;

#[derive(Clone)]
pub(super) struct Read { pub file: Arc<File>, pub offset: u64, pub slot: usize }
#[derive(Default)]
struct Buffer { bytes: Vec<u8>, reads: Vec<Read> }
struct Reply { buffer: Buffer, result: Result<uring::ReadStats> }
struct Request { buffer: Buffer, width: usize, priority: ReadPriority, reply: mpsc::SyncSender<Reply> }
#[derive(Default)]
struct Queue {
    decode: VecDeque<Request>, prefill: VecDeque<Request>, buffers: Vec<Buffer>,
    borrowed: usize, prefill_borrowed: usize, decode_waiters: usize,
}
struct Shared { queue: Mutex<Queue>, available: Condvar, wake: OwnedFd, ready: AtomicBool, failed: AtomicBool }
pub(super) struct Reactor { shared: Arc<Shared> }
static REACTOR: OnceLock<Mutex<Option<Arc<Reactor>>>> = OnceLock::new();

pub(super) fn started() -> bool {
    REACTOR.get().is_some_and(|global| global.lock().unwrap_or_else(|p| p.into_inner()).is_some())
}
pub(super) fn ready() -> bool {
    REACTOR.get().is_some_and(|global| global.lock().unwrap_or_else(|p| p.into_inner())
        .as_ref().is_some_and(|reactor| reactor.ready()))
}

fn error(context: &str, source: std::io::Error) -> MappedTableError {
    MappedTableError::Io { context: context.into(), source }
}
fn signal(fd: &OwnedFd) -> Result<()> {
    let value = 1u64;
    loop {
        // SAFETY: eventfd consumes exactly one initialized u64; fd stays owned.
        let result = unsafe { libc::write(fd.as_raw_fd(), (&value as *const u64).cast(), 8) };
        if result == 8 { return Ok(()); }
        let source = std::io::Error::last_os_error();
        if source.kind() == std::io::ErrorKind::WouldBlock { return Ok(()); }
        if source.kind() != std::io::ErrorKind::Interrupted { return Err(error("waking table read reactor", source)); }
    }
}
fn clear_signal(fd: &OwnedFd) {
    let mut value = 0u64;
    loop {
        // SAFETY: eventfd writes one u64 into live, sized storage.
        let result = unsafe { libc::read(fd.as_raw_fd(), (&mut value as *mut u64).cast(), 8) };
        if result == 8 { return; }
        if std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted { return; }
    }
}

pub(super) fn global() -> Result<Arc<Reactor>> {
    let mut global = REACTOR.get_or_init(|| Mutex::new(None)).lock().unwrap_or_else(|p| p.into_inner());
    if let Some(reactor) = global.as_ref() { return Ok(Arc::clone(reactor)); }
    // SAFETY: eventfd returns a fresh descriptor, with no borrowed pointers.
    let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
    if fd < 0 { return Err(error("creating table reactor eventfd", std::io::Error::last_os_error())); }
    // SAFETY: fd is freshly allocated and transferred exactly once.
    let wake = unsafe { OwnedFd::from_raw_fd(fd) };
    let shared = Arc::new(Shared { queue: Mutex::new(Queue::default()), available: Condvar::new(), wake, ready: AtomicBool::new(false), failed: AtomicBool::new(false) });
    let owner = Arc::clone(&shared);
    let (started, ready) = mpsc::sync_channel(1);
    std::thread::Builder::new().name("table-read-reactor".into()).spawn(move || run(owner, started))
        .map_err(|source| error("starting table read reactor", source))?;
    ready.recv().map_err(|_| invalid("table read reactor exited during startup"))?;
    let reactor = Arc::new(Reactor { shared });
    *global = Some(Arc::clone(&reactor));
    Ok(reactor)
}
/// Owns every submitted buffer until its CQEs have drained. A split gather
/// never lends its caller's destination to the reactor.
pub(super) struct ReadStream {
    owner: Arc<Reactor>, reads: Vec<Read>, next: usize, width: usize,
    priority: ReadPriority, window: usize, pending: VecDeque<mpsc::Receiver<Reply>>,
    stats: uring::ReadStats,
}
impl Reactor {
    pub(super) fn ready(&self) -> bool { self.shared.ready.load(Ordering::Acquire) }
    pub(super) fn wake(&self) -> Result<()> { signal(&self.shared.wake) }

    pub(super) fn start(self: &Arc<Self>, reads: Vec<Read>, width: usize,
        priority: ReadPriority, window: usize) -> Result<ReadStream> {
        if width == 0 || width > BUFFER_BYTES || !(1..=2).contains(&window) {
            return Err(invalid("table reactor row/window exceeds buffer budget"));
        }
        let mut stream = ReadStream { owner: Arc::clone(self), reads, next: 0, width,
            priority, window, pending: VecDeque::new(), stats: Default::default() };
        stream.fill()?;
        Ok(stream)
    }
    pub(super) fn read(self: &Arc<Self>, reads: &[uring::Read<'_>], output: &mut [u8], width: usize,
        priority: ReadPriority) -> Result<uring::ReadStats> {
        let reads = reads.iter().map(|read| Read { file: Arc::clone(&read.part.file),
            offset: read.part.file_offset + read.row * width as u64, slot: read.slot }).collect();
        let mut stream = self.start(reads, width, priority, 2)?;
        while !stream.finished() { stream.step(output)?; }
        Ok(stream.stats)
    }
}
impl ReadStream {
    fn fill(&mut self) -> Result<()> {
        let shared = &self.owner.shared;
        let decode = self.priority.is_decode();
        while self.next < self.reads.len() && self.pending.len() < self.window {
            let mut queue = shared.queue.lock().unwrap_or_else(|p| p.into_inner());
            if decode { queue.decode_waiters += 1; }
            while queue.borrowed == SLOTS || (!decode &&
                (queue.prefill_borrowed == SLOTS - DECODE_SLOTS || queue.decode_waiters > 0)) {
                if !self.pending.is_empty() {
                    if decode { queue.decode_waiters -= 1; }
                    return Ok(());
                }
                if shared.failed.load(Ordering::Acquire) {
                    if decode { queue.decode_waiters -= 1; }
                    return Err(invalid("table read reactor owner exited"));
                }
                queue = shared.available.wait(queue).unwrap_or_else(|p| p.into_inner());
            }
            if decode { queue.decode_waiters -= 1; } else { queue.prefill_borrowed += 1; }
            queue.borrowed += 1;
            let mut buffer = queue.buffers.pop().unwrap_or_default();
            drop(queue);
            let end = (self.next + uring::MAX_BATCH.min(BUFFER_BYTES / self.width)).min(self.reads.len());
            if buffer.bytes.is_empty() { buffer.bytes.resize(BUFFER_BYTES, 0); }
            buffer.reads.clear();
            buffer.reads.extend_from_slice(&self.reads[self.next..end]);
            self.next = end;
            let (send, receive) = mpsc::sync_channel(1);
            let request = Request { buffer, width: self.width, priority: self.priority, reply: send };
            {
                let mut queue = shared.queue.lock().unwrap_or_else(|p| p.into_inner());
                if shared.failed.load(Ordering::Acquire) {
                    let _ = request.reply.send(Reply { buffer: request.buffer,
                        result: Err(invalid("table read reactor owner exited")) });
                } else if decode { queue.decode.push_back(request); } else { queue.prefill.push_back(request); }
            }
            self.pending.push_back(receive);
            // Latched eventfd closes the inspect/arm race without polling.
            if let Err(error) = signal(&shared.wake) {
                tracing::error!(%error, "cannot wake table reactor; refusing an unbounded wait");
                std::process::abort();
            }
        }
        Ok(())
    }
    pub(super) fn finished(&self) -> bool { self.next == self.reads.len() && self.pending.is_empty() }
    pub(super) fn stats(&self) -> uring::ReadStats { self.stats }
    fn release(&self, mut buffer: Buffer) {
        buffer.reads.clear();
        let mut queue = self.owner.shared.queue.lock().unwrap_or_else(|p| p.into_inner());
        queue.buffers.push(buffer);
        queue.borrowed -= 1;
        if !self.priority.is_decode() { queue.prefill_borrowed -= 1; }
        self.owner.shared.available.notify_all();
    }
    pub(super) fn step(&mut self, output: &mut [u8]) -> Result<()> {
        self.fill()?;
        let Some(receive) = self.pending.pop_front() else { return Ok(()); };
        let reply = receive.recv().map_err(|_| invalid("table read reactor exited"))?;
        self.land(reply, output)
    }
    /// [`Self::step`] without waiting: lands the oldest buffer only when its
    /// reads have completed. True when it landed one (or refilled an idle stream).
    pub(super) fn try_step(&mut self, output: &mut [u8]) -> Result<bool> {
        let Some(receive) = self.pending.front() else {
            self.fill()?;
            return Ok(!self.pending.is_empty());
        };
        match receive.try_recv() {
            Ok(reply) => {
                self.pending.pop_front();
                self.land(reply, output)?;
                Ok(true)
            }
            Err(mpsc::TryRecvError::Empty) => Ok(false),
            Err(mpsc::TryRecvError::Disconnected) => Err(invalid("table read reactor exited")),
        }
    }
    fn land(&mut self, reply: Reply, output: &mut [u8]) -> Result<()> {
        if reply.result.is_ok() {
            for (read, bytes) in reply.buffer.reads.iter().zip(reply.buffer.bytes.chunks_exact(self.width)) {
                output[read.slot * self.width..(read.slot + 1) * self.width].copy_from_slice(bytes);
            }
        }
        self.release(reply.buffer);
        let batch = reply.result?;
        self.stats.miss_ns += batch.miss_ns;
        self.stats.miss_max_ns = self.stats.miss_max_ns.max(batch.miss_max_ns);
        for (total, value) in self.stats.miss_histogram.iter_mut().zip(batch.miss_histogram) { *total += value; }
        self.fill()
    }
}
impl Drop for ReadStream {
    fn drop(&mut self) {
        // Cancellation and errors drain already owned reads, never refill. No
        // channel receiver or buffer credit is abandoned while the kernel runs.
        while let Some(receive) = self.pending.pop_front() {
            match receive.recv() {
                Ok(reply) => self.release(reply.buffer),
                Err(_) => std::process::abort(),
            }
        }
    }
}

struct Job { request: Request, next: usize, left: usize,
    error: Option<MappedTableError>, stats: uring::ReadStats }
#[derive(Clone, Copy)]
struct Active { job: usize, row: usize, done: usize, submitted: Instant }
struct Engine {
    ring: IoUring, jobs: Vec<Option<Job>>, active: Vec<Option<Active>>,
    retry: VecDeque<Active>, free: Vec<usize>, outstanding: usize, wake_armed: bool, decode_streak: usize,
}
impl Engine {
    fn new(ring: IoUring) -> Self {
        let capacity = (ring.params().sq_entries() as usize - 1).min(uring::worker_limit().expect("validated ring worker limit") as usize);
        Self { ring, jobs: (0..SLOTS).map(|_| None).collect(), active: vec![None; capacity],
            retry: VecDeque::new(), free: (0..capacity).rev().collect(), outstanding: 0, wake_armed: false, decode_streak: 0 }
    }
    fn arm(&mut self, fd: &OwnedFd) -> Result<()> {
        if !self.wake_armed {
            let entry = opcode::PollAdd::new(types::Fd(fd.as_raw_fd()), libc::POLLIN as _).build().user_data(WAKE);
            // SAFETY: PollAdd holds no memory pointer; Shared owns the fd beyond this engine.
            unsafe { self.ring.submission().push(&entry) }.map_err(|_| invalid("reactor wake queue full"))?;
            self.wake_armed = true;
        }
        Ok(())
    }
    fn accept(&mut self, shared: &Shared) {
        let mut queue = shared.queue.lock().unwrap_or_else(|p| p.into_inner());
        while let Some(index) = self.jobs.iter().position(Option::is_none) {
            let take_first = |queue: &mut VecDeque<Request>, first| {
                let at = queue.iter().position(|request| request.priority == first).unwrap_or(0);
                queue.remove(at)
            };
            let Some(request) = take_first(&mut queue.decode, ReadPriority::DecodeFirst)
                .or_else(|| take_first(&mut queue.prefill, ReadPriority::PrefillFirst)) else { break; };
            let left = request.buffer.reads.len();
            self.jobs[index] = Some(Job { request, next: 0, left, error: None, stats: Default::default() });
        }
    }
    fn enqueue(&mut self, active: Active) -> Result<()> {
        let index = self.free.pop().ok_or_else(|| invalid("reactor capacity exceeded"))?;
        let job = self.jobs[active.job].as_mut().expect("live read job");
        let read = &job.request.buffer.reads[active.row];
        let width = job.request.width;
        // SAFETY: only this owner touches the buffer, it never resizes while in
        // flight, and each active row is disjoint. Engine::drop drains all SQEs
        // before any job buffer or Arc<File> can be released, including unwind.
        let pointer = unsafe { job.request.buffer.bytes.as_mut_ptr().add(active.row * width + active.done) };
        let entry = opcode::Read::new(types::Fd(read.file.as_raw_fd()), pointer, (width - active.done) as u32)
            .offset(read.offset + active.done as u64).build().flags(io_uring::squeue::Flags::ASYNC).user_data(index as u64);
        // SAFETY: the owned job and the draining engine retain fd and destination.
        unsafe { self.ring.submission().push(&entry) }.map_err(|_| invalid("reactor read queue full"))?;
        self.active[index] = Some(active);
        self.outstanding += 1;
        Ok(())
    }
    fn schedule(&mut self) -> Result<()> {
        let reserve = (self.active.len() / 8).clamp(1, 64);
        while self.outstanding < self.active.len() {
            let candidate = |priority| self.retry.iter().find(|active|
                self.jobs[active.job].as_ref().is_some_and(|job| job.request.priority == priority))
                .map(|active| active.job).or_else(|| self.jobs.iter().position(|job| job.as_ref().is_some_and(|job|
                    job.request.priority == priority && job.next < job.request.buffer.reads.len())));
            let decode = candidate(ReadPriority::DecodeFirst).or_else(|| candidate(ReadPriority::Decode));
            let prefill = candidate(ReadPriority::PrefillFirst).or_else(|| candidate(ReadPriority::Prefill));
            // Bound starvation under a continuous decode stream, without using
            // decode's reserved admission capacity for prefill.
            let prefill_allowed = self.outstanding < self.active.len() - reserve;
            let selected = if self.decode_streak >= 256 && prefill_allowed { prefill.or(decode) }
                else { decode.or(if prefill_allowed { prefill } else { None }) };
            // Pick a class/sub-priority before considering retries, so an
            // L14 short read cannot jump ahead of any ready L1 SQE. No join:
            // L14 fills capacity whenever L1 is waiting for CQEs or refill.
            let retry = selected.and_then(|id| self.retry.iter().position(|active|
                self.jobs[active.job].as_ref().is_some_and(|job|
                    job.request.priority == self.jobs[id].as_ref().unwrap().request.priority)));
            if let Some(at) = retry {
                let active = self.retry.remove(at).expect("selected retry");
                if self.jobs[active.job].as_ref().unwrap().request.priority.is_decode() {
                    self.decode_streak += 1;
                } else { self.decode_streak = 0; }
                self.enqueue(active)?;
            } else if let Some(id) = selected {
                let job = self.jobs[id].as_mut().expect("selected job");
                let row = job.next;
                job.next += 1;
                if job.request.priority.is_decode() { self.decode_streak += 1; } else { self.decode_streak = 0; }
                self.enqueue(Active { job: id, row, done: 0, submitted: Instant::now() })?;
            } else { break; }
        }
        Ok(())
    }
    fn complete(&mut self, shared: &Shared) -> Result<()> {
        while let Some(cqe) = self.ring.completion().next() {
            if cqe.user_data() == WAKE {
                self.wake_armed = false;
                clear_signal(&shared.wake);
                continue;
            }
            let id = usize::try_from(cqe.user_data()).map_err(|_| invalid("invalid reactor completion"))?;
            let mut active = self.active.get_mut(id).and_then(Option::take).ok_or_else(|| invalid("unknown reactor completion"))?;
            self.outstanding -= 1;
            self.free.push(id);
            let job = self.jobs[active.job].as_mut().expect("completion job retained");
            let count = cqe.result();
            if count > 0 && count as usize <= job.request.width - active.done {
                active.done += count as usize;
                if active.done < job.request.width { self.retry.push_back(active); continue; }
                let elapsed = active.submitted.elapsed().as_nanos() as u64;
                job.stats.miss_ns += elapsed;
                job.stats.miss_max_ns = job.stats.miss_max_ns.max(elapsed);
                job.stats.miss_histogram[uring::tier(elapsed)] += 1;
            } else {
                job.error.get_or_insert_with(|| error("table reactor read", if count < 0 {
                    std::io::Error::from_raw_os_error(-count)
                } else { std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "short immutable table") }));
            }
            job.left -= 1;
            if job.left == 0 {
                let job = self.jobs[active.job].take().expect("completed job retained");
                let result = job.error.map_or(Ok(job.stats), Err);
                // A cancelled caller may have dropped the receiver. All writes
                // are complete before its owned buffer is dropped here.
                let _ = job.request.reply.send(Reply { buffer: job.request.buffer, result });
            }
        }
        Ok(())
    }
    fn wait(&mut self) -> Result<()> {
        loop {
            match self.ring.submit_and_wait(1) {
                Ok(_) => return Ok(()),
                Err(source) if source.kind() == std::io::ErrorKind::Interrupted => {},
                Err(source) => return Err(error("waiting for table reactor", source)),
            }
        }
    }
}
impl Drop for Engine {
    fn drop(&mut self) {
        while self.outstanding > 0 {
            while let Some(cqe) = self.ring.completion().next() {
                if cqe.user_data() != WAKE { self.outstanding -= 1; }
            }
            if self.outstanding > 0 && self.wait().is_err() {
                tracing::error!("cannot drain owned table read buffers; refusing unsafe teardown");
                std::process::abort();
            }
        }
        for job in self.jobs.iter_mut().filter_map(Option::take) {
            let _ = job.request.reply.send(Reply { buffer: job.request.buffer, result: Err(invalid("table reactor batch failed; retry via mmap")) });
        }
    }
}
fn reject(shared: &Shared) {
    let mut queue = shared.queue.lock().unwrap_or_else(|p| p.into_inner());
    for request in queue.decode.drain(..).collect::<Vec<_>>().into_iter().chain(queue.prefill.drain(..)) {
        let _ = request.reply.send(Reply { buffer: request.buffer, result: Err(invalid("table reactor temporarily unavailable; retry via mmap")) });
    }
}
fn supervise(shared: &Shared, owner: impl FnOnce()) {
    if std::panic::catch_unwind(std::panic::AssertUnwindSafe(owner)).is_err() {
        tracing::error!("table read reactor owner panicked; rejecting queued work");
    }
    shared.failed.store(true, Ordering::Release);
    shared.ready.store(false, Ordering::Release);
    reject(shared);
    shared.available.notify_all();
}
fn run(shared: Arc<Shared>, started: mpsc::SyncSender<()>) {
    supervise(&shared, || run_owner(&shared, started));
}
fn run_owner(shared: &Shared, started: mpsc::SyncSender<()>) {
    let mut engine = match uring::new_ring() {
        Ok(ring) => { shared.ready.store(true, Ordering::Release); Some(Engine::new(ring)) },
        Err(error) => { tracing::warn!(%error, "table reactor setup failed; batches can retry"); None },
    };
    let _ = started.send(());
    let mut retry_after = Instant::now() + Duration::from_secs(1);
    loop {
        if let Some(owner) = engine.as_mut() {
            let step = owner.complete(&shared).and_then(|()| {
                owner.accept(&shared);
                owner.arm(&shared.wake)?;
                owner.schedule()?;
                owner.wait()
            });
            if let Err(error) = step {
                tracing::warn!(%error, "table reactor batch failed; draining before retry");
                shared.ready.store(false, Ordering::Release);
                drop(engine.take());
                reject(&shared);
                retry_after = Instant::now() + Duration::from_secs(1);
            }
        } else {
            // Setup can fail under transient memory pressure. Sleep in the
            // eventfd poll, then retry at most once per second of demand.
            let mut poll = libc::pollfd { fd: shared.wake.as_raw_fd(), events: libc::POLLIN, revents: 0 };
            // SAFETY: poll borrows one initialized descriptor for this call.
            let result = unsafe { libc::poll(&mut poll, 1, -1) };
            if result < 0 { continue; }
            clear_signal(&shared.wake);
            if Instant::now() >= retry_after {
                retry_after = Instant::now() + Duration::from_secs(1);
                match uring::new_ring() {
                    Ok(ring) => { engine = Some(Engine::new(ring)); shared.ready.store(true, Ordering::Release); },
                    Err(error) => tracing::warn!(%error, "table reactor setup retry failed"),
                }
            }
            if engine.is_none() { reject(&shared); }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn concurrent_callers_share_one_owner_and_stream_beyond_pool_volume() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut file = tempfile::NamedTempFile::new()?;
        let payload: Vec<_> = (0..4096).map(|i| (i % 251) as u8).collect();
        file.write_all(&payload)?;
        // SAFETY: the test keeps its immutable file alive until all callers join.
        let part = Arc::new(unsafe { super::super::MappedRows::open(file.path(), 0, 16, 256)? });
        let owner = global()?;
        if !owner.ready() { return Ok(()); }
        let mut threads = Vec::new();
        for caller in 0..4 {
            let part = Arc::clone(&part);
            let owner = Arc::clone(&owner);
            let expected = payload.clone();
            threads.push(std::thread::spawn(move || -> Result<()> {
                assert!(Arc::ptr_eq(&owner, &global()?));
                // One gather exceeds the entire 16 MiB pool; it must stream.
                let count = if caller == 0 { 70_000 } else { 257 };
                let reads: Vec<_> = (0..count).map(|slot| uring::Read { part: &part, row: (slot % 16) as u64, slot }).collect();
                let mut output = vec![0; count * 256];
                let priority = if caller == 0 { ReadPriority::Prefill } else { ReadPriority::Decode };
                owner.read(&reads, &mut output, 256, priority)?;
                for (slot, row) in output.chunks_exact(256).enumerate() {
                    assert_eq!(row, &expected[(slot % 16) * 256..(slot % 16 + 1) * 256]);
                }
                Ok(())
            }));
        }
        for thread in threads { thread.join().expect("read caller panic")?; }
        // Other tests in this process share the owner (e.g. Engram look-ahead
        // reads): every credit returns once their reads drain.
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut queue = owner.shared.queue.lock().unwrap();
        while (queue.borrowed, queue.prefill_borrowed, queue.decode_waiters) != (0, 0, 0) {
            assert!(Instant::now() < deadline, "reactor credits never returned");
            drop(queue);
            std::thread::sleep(Duration::from_millis(5));
            queue = owner.shared.queue.lock().unwrap();
        }
        assert!(queue.buffers.iter().map(|buffer| buffer.bytes.len()).sum::<usize>() <= SLOTS * BUFFER_BYTES);
        Ok(())
    }

    #[test]
    fn unavailable_owner_wake_is_latched_for_demand_driven_recovery() -> Result<()> {
        // Test the same eventfd wake used by GatherPool without depending on
        // io_uring availability or interfering with the process-wide owner.
        // SAFETY: eventfd takes no pointers and returns a new owned descriptor.
        let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
        if fd < 0 { return Err(error("test eventfd", std::io::Error::last_os_error())); }
        // SAFETY: this test owns the fresh descriptor exactly once.
        let wake = unsafe { OwnedFd::from_raw_fd(fd) };
        let owner = Reactor { shared: Arc::new(Shared { queue: Mutex::new(Queue::default()),
            available: Condvar::new(), wake, ready: AtomicBool::new(false), failed: AtomicBool::new(false) }) };
        assert!(!owner.ready());
        owner.wake()?;
        let mut poll = libc::pollfd { fd: owner.shared.wake.as_raw_fd(), events: libc::POLLIN, revents: 0 };
        // SAFETY: poll borrows one initialized descriptor and never retains it.
        assert_eq!(unsafe { libc::poll(&mut poll, 1, 0) }, 1);
        clear_signal(&owner.shared.wake);
        // SAFETY: the descriptor remains valid for this immediate readiness check.
        assert_eq!(unsafe { libc::poll(&mut poll, 1, 0) }, 0);
        Ok(())
    }

    #[test]
    fn two_prefill_windows_allow_decode_and_recycle_on_cancellation() -> Result<()> {
        let owner = global()?;
        if !owner.ready() { return Ok(()); }
        let file = Arc::new(File::open("/dev/zero").map_err(|source| error("test zero", source))?);
        let reads = || (0..8192).map(|slot| Read { file: Arc::clone(&file), offset: 0, slot }).collect();
        let first = owner.start(reads(), 1, ReadPriority::Prefill, 2)?;
        let second = owner.start(reads(), 1, ReadPriority::Prefill, 2)?;
        // Hold both finishers' completed-but-unreaped windows. Decode must
        // acquire and complete without either prefill caller making progress.
        let mut decode = owner.start(vec![Read { file, offset: 0, slot: 0 }], 1, ReadPriority::Decode, 2)?;
        let mut output = [255];
        while !decode.finished() { decode.step(&mut output)?; }
        assert_eq!(output, [0]);
        assert!(4 < SLOTS - DECODE_SLOTS);
        drop((first, second, decode));
        Ok(())
    }

    #[test]
    fn first_layer_precedes_ready_later_reads_without_excluding_refill_gaps() -> Result<()> {
        for (first, later) in [(ReadPriority::PrefillFirst, ReadPriority::Prefill),
            (ReadPriority::DecodeFirst, ReadPriority::Decode)] {
            let ring = match uring::new_ring() { Ok(ring) => ring, Err(_) => return Ok(()) };
            let mut engine = Engine::new(ring);
            let file = Arc::new(File::open("/dev/zero").map_err(|source| error("test data", source))?);
            let job = |priority, count| {
                let (reply, receiver) = mpsc::sync_channel(1);
                drop(receiver);
                Job { request: Request { buffer: Buffer { bytes: vec![0; count],
                    reads: (0..count).map(|_| Read { file: Arc::clone(&file), offset: 0, slot: 0 }).collect() },
                    width: 1, priority, reply }, next: 0, left: count, error: None, stats: Default::default() }
            };
            // A later-layer retry is already queued. Ready first-layer reads
            // still win; later reads fill immediately after the first two.
            engine.jobs[0] = Some(job(later, engine.active.len() * 2));
            engine.jobs[0].as_mut().unwrap().next = 1;
            engine.retry.push_back(Active { job: 0, row: 0, done: 0, submitted: Instant::now() });
            engine.jobs[1] = Some(job(first, 2));
            engine.schedule()?;
            assert_eq!(engine.active[0].unwrap().job, 1);
            assert_eq!(engine.active[1].unwrap().job, 1);
            assert_eq!(engine.jobs[1].as_ref().unwrap().next, 2);
            assert!(engine.jobs[0].as_ref().unwrap().next > 1, "no L1 drain/join exclusion");
            assert!(engine.retry.is_empty());
        }
        Ok(())
    }

    #[test]
    fn scheduling_reserves_decode_capacity_and_completes_abandoned_replies() -> Result<()> {
        // This isolated engine uses no global owner and is dropped after draining.
        let ring = match uring::new_ring() { Ok(ring) => ring, Err(_) => return Ok(()) };
        let mut engine = Engine::new(ring);
        let file = Arc::new(File::open("/dev/zero").map_err(|source| error("opening test data", source))?);
        let request = |priority, count| {
            let (reply, receiver) = mpsc::sync_channel(1);
            drop(receiver); // caller cancellation must never free in-flight storage
            let buffer = Buffer { bytes: vec![0; count], reads: (0..count).map(|_| Read { file: Arc::clone(&file), offset: 0, slot: 0 }).collect() };
            Job { request: Request { buffer, width: 1, priority, reply }, next: 0, left: count, error: None, stats: Default::default() }
        };
        let limit = engine.active.len();
        let reserve = (limit / 8).clamp(1, 64);
        engine.jobs[0] = Some(request(ReadPriority::Prefill, limit * 2));
        engine.schedule()?;
        assert_eq!(engine.outstanding, limit - reserve);
        engine.jobs[1] = Some(request(ReadPriority::Decode, reserve));
        engine.schedule()?;
        assert_eq!(engine.outstanding, limit);
        assert_eq!(engine.jobs[1].as_ref().unwrap().next, reserve);
        // Drop drains even though no receiver exists. This also tests shutdown
        // with buffered reads still queued rather than submitted.
        drop(engine);
        Ok(())
    }
    #[test]
    fn owner_panic_rejects_queued_and_later_requests() -> Result<()> {
        // SAFETY: eventfd returns a fresh descriptor with no borrowed pointers.
        let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
        if fd < 0 { return Err(error("test eventfd", std::io::Error::last_os_error())); }
        let shared = Arc::new(Shared { queue: Mutex::new(Queue::default()), available: Condvar::new(),
            // SAFETY: the descriptor is fresh and transferred exactly once.
            wake: unsafe { OwnedFd::from_raw_fd(fd) }, ready: AtomicBool::new(true), failed: AtomicBool::new(false) });
        let (send, reply) = mpsc::sync_channel(1);
        shared.queue.lock().unwrap().prefill.push_back(Request { buffer: Buffer::default(), width: 1,
            priority: ReadPriority::Prefill, reply: send });
        supervise(&shared, || panic!("injected owner panic"));
        assert!(reply.recv_timeout(Duration::from_secs(1)).unwrap().result.is_err());
        assert!(!shared.ready.load(Ordering::Acquire));
        let owner = Arc::new(Reactor { shared });
        let file = tempfile::NamedTempFile::new().map_err(|source| error("test file", source))?;
        file.as_file().set_len(1).map_err(|source| error("test file size", source))?;
        // SAFETY: immutable file remains live until the rejected request returns.
        let part = unsafe { super::super::MappedRows::open(file.path(), 0, 1, 1)? };
        let reads = [uring::Read { part: &part, row: 0, slot: 0 }];
        assert!(owner.read(&reads, &mut [0], 1, ReadPriority::Decode).is_err());
        let queue = owner.shared.queue.lock().unwrap();
        assert_eq!((queue.borrowed, queue.prefill_borrowed, queue.decode_waiters), (0, 0, 0));
        Ok(())
    }

    #[test]
    fn retries_preserve_first_submit_time_and_decode_fairness() -> Result<()> {
        let ring = match uring::new_ring() { Ok(ring) => ring, Err(_) => return Ok(()) };
        let mut engine = Engine::new(ring);
        let file = Arc::new(File::open("/dev/zero").map_err(|source| error("opening test data", source))?);
        let (send, _reply) = mpsc::sync_channel(1);
        engine.jobs[0] = Some(Job { request: Request { buffer: Buffer { bytes: vec![0; 2],
            reads: vec![Read { file, offset: 0, slot: 0 }] }, width: 2, priority: ReadPriority::Decode, reply: send },
            next: 1, left: 1, error: None, stats: Default::default() });
        let first = Instant::now() - Duration::from_millis(20);
        engine.retry.push_back(Active { job: 0, row: 0, done: 1, submitted: first });
        engine.schedule()?;
        assert_eq!(engine.decode_streak, 1);
        assert_eq!(engine.active.iter().flatten().next().unwrap().submitted, first);
        drop(engine);
        Ok(())
    }

    #[test]
    fn eventfd_wake_admits_decode_within_one_owner_iteration() -> Result<()> {
        let ring = match uring::new_ring() { Ok(ring) => ring, Err(_) => return Ok(()) };
        // SAFETY: eventfd returns a fresh descriptor with no memory references.
        let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
        if fd < 0 { return Err(error("test eventfd", std::io::Error::last_os_error())); }
        // SAFETY: this scope assumes unique ownership of the fresh descriptor.
        let shared = Shared { queue: Mutex::new(Queue::default()), available: Condvar::new(),
            wake: unsafe { OwnedFd::from_raw_fd(fd) }, ready: AtomicBool::new(true), failed: AtomicBool::new(false) };
        let mut engine = Engine::new(ring);
        engine.arm(&shared.wake)?;
        engine.ring.submit().map_err(|source| error("submitting test wake", source))?;
        let file = Arc::new(File::open("/dev/zero").map_err(|source| error("opening test data", source))?);
        let request = |priority, rows| {
            let (reply, receiver) = mpsc::sync_channel(1);
            let buffer = Buffer { bytes: vec![0; rows], reads: (0..rows).map(|_| Read { file: Arc::clone(&file), offset: 0, slot: 0 }).collect() };
            (Request { buffer, width: 1, priority, reply }, receiver)
        };
        let (prefill, _prefill_reply) = request(ReadPriority::Prefill, engine.active.len() * 2);
        let rows = prefill.buffer.reads.len();
        engine.jobs[0] = Some(Job { request: prefill, next: 0, left: rows, error: None, stats: Default::default() });
        engine.schedule()?;
        let before = engine.outstanding;
        // Keep prefill SQEs queued but unsubmitted, so only the already armed
        // eventfd can produce a CQE. A delayed producer wakes an idle owner.
        let (decode, _decode_reply) = request(ReadPriority::Decode, 1);
        std::thread::scope(|scope| -> Result<()> {
            scope.spawn(|| {
                std::thread::sleep(Duration::from_millis(10));
                shared.queue.lock().unwrap().decode.push_back(decode);
                signal(&shared.wake).expect("test reactor wake");
            });
            // SAFETY: zero submissions, GETEVENTS waits for the existing PollAdd;
            // no extended arguments or signal mask, all ring storage stays live.
            unsafe { engine.ring.submitter().enter::<libc::sigset_t>(0, 1, 1, None) }
                .map_err(|source| error("test eventfd-only wait", source))?;
            Ok(())
        })?;
        // Normal owner iteration: handle completions, accept, rearm, schedule.
        engine.complete(&shared)?;
        engine.accept(&shared);
        engine.arm(&shared.wake)?;
        engine.schedule()?;
        assert!(engine.jobs.iter().flatten().any(|job| job.request.priority == ReadPriority::Decode && job.next == 1));
        assert!(before > 0);
        drop(engine);
        Ok(())
    }

}
