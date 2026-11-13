//! Opt-in stall stack dump (`CUTEAFD_DIAG_STACK_ON_TIMEOUT=1`, #1878/#2016).
//!
//! A host thread that blocks inside a submission while a peer GPU spins on a
//! device wait unwinds once that wait times out, so the timeout path itself
//! never sees the blocked frame. A watcher thread instead dumps every thread's
//! stack while the stall is live: when a submission has issued no peer
//! push/wait for [`STALL`] (well inside the 60 s device wait timeout), and again
//! the first time the device timeout record appears.
//!
//! Other threads' stacks come from a `SIGUSR1` handler: the watcher signals each
//! task with `tgkill`, and the handler (async-signal-safe: glibc `backtrace`
//! primed at install, a pre-allocated slot, atomics only) stores raw return
//! addresses. The watcher then logs, per thread, its name, scheduler state and
//! kernel wait channel, plus each frame as `module+offset`, resolvable offline
//! with `addr2line -f -C -e <module> <offset>` against the sealed slot
//! binaries. Symbolizing inside a handler is not async-signal-safe, and the
//! release binary exports no dynamic Rust symbols, so offsets are exact while
//! in-process names would not be. No ptrace, no allocation in the handler, and
//! nothing at all (no thread, no handler, no counters read) unless enabled.
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// The private diagnostic's name; the launcher forwards every `CUTEAFD_DIAG_*`.
pub(crate) const STACK_ON_TIMEOUT: &str = "CUTEAFD_DIAG_STACK_ON_TIMEOUT";
/// No peer issue for this long inside a submission counts as a stall.
const STALL: Duration = Duration::from_secs(20);
const POLL: Duration = Duration::from_millis(250);
const FRAMES: usize = 64;
const SLOTS: usize = 512;

static ENABLED: AtomicBool = AtomicBool::new(false);
/// Peer push/wait issues so far (bumped only when enabled).
static PROGRESS: AtomicU64 = AtomicU64::new(0);
/// Submissions currently inside their body.
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);

struct Slot {
    tid: AtomicI32,
    depth: AtomicUsize,
    done: AtomicBool,
    frames: [AtomicUsize; FRAMES],
}

#[allow(clippy::declare_interior_mutable_const)]
const EMPTY: Slot = Slot { tid: AtomicI32::new(0), depth: AtomicUsize::new(0), done: AtomicBool::new(false),
    frames: [const { AtomicUsize::new(0) }; FRAMES] };
static SLOTS_TABLE: [Slot; SLOTS] = [EMPTY; SLOTS];
static NEXT_SLOT: AtomicUsize = AtomicUsize::new(0);

unsafe extern "C" {
    fn backtrace(buffer: *mut *mut libc::c_void, size: libc::c_int) -> libc::c_int;
}

pub(crate) fn enabled_by(value: Option<&str>) -> bool { value == Some("1") }

/// Start the watcher once per process when the diagnostic is on. Inert otherwise.
pub(crate) fn install() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        if !enabled_by(std::env::var(STACK_ON_TIMEOUT).ok().as_deref()) { return }
        if let Err(error) = install_handler() {
            tracing::warn!(%error, "stall stack dump unavailable");
            return;
        }
        ENABLED.store(true, Ordering::Release);
        if let Err(error) = std::thread::Builder::new().name("stall-stacks".into()).spawn(watch) {
            ENABLED.store(false, Ordering::Release);
            tracing::warn!(%error, "stall stack watcher not started");
            return;
        }
        tracing::info!(stall_s = STALL.as_secs(), "stall stack dump armed (CUTEAFD_DIAG_STACK_ON_TIMEOUT)");
    });
}

pub(crate) fn enabled() -> bool { ENABLED.load(Ordering::Relaxed) }

/// One peer push or wait was issued from the host.
#[inline]
pub(crate) fn progress() {
    if enabled() { PROGRESS.fetch_add(1, Ordering::Relaxed); }
}

/// Marks a submission body in flight until dropped.
pub(crate) struct InFlight(bool);
pub(crate) fn in_flight() -> InFlight {
    let on = enabled();
    if on { IN_FLIGHT.fetch_add(1, Ordering::AcqRel); PROGRESS.fetch_add(1, Ordering::Relaxed); }
    InFlight(on)
}
impl Drop for InFlight {
    fn drop(&mut self) { if self.0 { IN_FLIGHT.fetch_sub(1, Ordering::AcqRel); } }
}

fn install_handler() -> std::io::Result<()> {
    // Prime glibc's unwinder (libgcc load) outside any handler.
    let mut prime = [std::ptr::null_mut(); 4];
    // SAFETY: a live four-entry buffer.
    unsafe { backtrace(prime.as_mut_ptr(), prime.len() as libc::c_int) };
    // SAFETY: the handler only touches atomics and calls backtrace into a static slot.
    unsafe {
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = handler as usize;
        action.sa_flags = libc::SA_RESTART | libc::SA_SIGINFO;
        libc::sigemptyset(&mut action.sa_mask);
        if libc::sigaction(libc::SIGUSR1, &action, std::ptr::null_mut()) != 0 {
            return Err(std::io::Error::last_os_error());
        }
    }
    Ok(())
}

extern "C" fn handler(_: libc::c_int, _: *mut libc::siginfo_t, _: *mut libc::c_void) {
    // SAFETY: gettid is async-signal-safe.
    let tid = unsafe { libc::syscall(libc::SYS_gettid) } as i32;
    let Some(slot) = SLOTS_TABLE.iter().find(|slot| slot.tid.load(Ordering::Acquire) == tid) else { return };
    let mut frames = [std::ptr::null_mut(); FRAMES];
    // SAFETY: primed at install; a fixed stack buffer; no allocation.
    let depth = unsafe { backtrace(frames.as_mut_ptr(), FRAMES as libc::c_int) }.max(0) as usize;
    for (to, from) in slot.frames.iter().zip(&frames[..depth]) { to.store(*from as usize, Ordering::Relaxed); }
    slot.depth.store(depth, Ordering::Relaxed);
    slot.done.store(true, Ordering::Release);
}

/// The watcher's per-poll decision, separated from time and signals for tests.
#[derive(Debug, Default)]
struct Stall {
    seen: u64,
    quiet: Duration,
    dumped: bool,
}

impl Stall {
    /// One poll: `progress` is the issue counter, `in_flight` the submissions in their body,
    /// `elapsed` the time since the last poll. Returns true exactly once per stall: after
    /// [`STALL`] without progress inside a submission, until progress or an idle poll rearms it.
    fn poll(&mut self, progress: u64, in_flight: usize, elapsed: Duration) -> bool {
        if progress != self.seen || in_flight == 0 {
            self.seen = progress;
            self.quiet = Duration::ZERO;
            self.dumped = false;
            return false;
        }
        self.quiet += elapsed;
        if self.dumped || self.quiet < STALL { return false }
        self.dumped = true;
        true
    }
}

fn watch() {
    let mut stall = Stall { seen: PROGRESS.load(Ordering::Relaxed), ..Default::default() };
    let mut timeout_dumped = false;
    let mut last = Instant::now();
    loop {
        std::thread::sleep(POLL);
        let now = Instant::now();
        if stall.poll(PROGRESS.load(Ordering::Relaxed), IN_FLIGHT.load(Ordering::Acquire), now - last) {
            dump(&format!("no peer issue for {} s inside a submission", STALL.as_secs()));
        }
        last = now;
        if !timeout_dumped && device_timeout() {
            timeout_dumped = true;
            dump("device peer wait timeout recorded");
        }
    }
}

fn device_timeout() -> bool {
    TIMEOUT_PROBE.get().is_some_and(|probe| probe())
}

static TIMEOUT_PROBE: std::sync::OnceLock<Box<dyn Fn() -> bool + Send + Sync>> = std::sync::OnceLock::new();

/// Lets the watcher see the native timeout record (set once by the engine that owns the library).
pub(crate) fn watch_timeout_record(probe: impl Fn() -> bool + Send + Sync + 'static) {
    if enabled() { let _ = TIMEOUT_PROBE.set(Box::new(probe)); }
}

fn tasks() -> Vec<i32> {
    std::fs::read_dir("/proc/self/task").map(|entries| entries.filter_map(|entry| entry.ok()?
        .file_name().to_str()?.parse().ok()).collect()).unwrap_or_default()
}

fn task_file(tid: i32, name: &str) -> String {
    std::fs::read_to_string(format!("/proc/self/task/{tid}/{name}")).unwrap_or_default().trim().to_string()
}

/// `(start, end, offset, path)` executable mappings, to report frames as module+offset.
fn mappings() -> Vec<(usize, usize, usize, String)> {
    std::fs::read_to_string("/proc/self/maps").unwrap_or_default().lines().filter_map(|line| {
        let mut fields = line.split_whitespace();
        let (range, perms, offset) = (fields.next()?, fields.next()?, fields.next()?);
        if !perms.contains('x') { return None }
        let path = fields.nth(2)?.to_string();
        let (start, end) = range.split_once('-')?;
        Some((usize::from_str_radix(start, 16).ok()?, usize::from_str_radix(end, 16).ok()?,
            usize::from_str_radix(offset, 16).ok()?, path))
    }).collect()
}

fn frame(address: usize, maps: &[(usize, usize, usize, String)]) -> String {
    match maps.iter().find(|(start, end, ..)| (*start..*end).contains(&address)) {
        Some((start, _, offset, path)) => format!("{path}+{:#x}", address - start + offset),
        None => format!("{address:#x}"),
    }
}

fn dump(reason: &str) {
    // SAFETY: gettid has no preconditions.
    let me = unsafe { libc::syscall(libc::SYS_gettid) } as i32;
    let pid = std::process::id() as i32;
    let targets: Vec<i32> = tasks().into_iter().filter(|&tid| tid != me).take(SLOTS).collect();
    NEXT_SLOT.store(0, Ordering::Relaxed);
    let mut armed = Vec::new();
    for &tid in &targets {
        let index = NEXT_SLOT.fetch_add(1, Ordering::Relaxed);
        let slot = &SLOTS_TABLE[index];
        slot.done.store(false, Ordering::Relaxed);
        slot.depth.store(0, Ordering::Relaxed);
        slot.tid.store(tid, Ordering::Release);
        // SAFETY: signals one live task of this process; the handler is installed.
        if unsafe { libc::syscall(libc::SYS_tgkill, pid, tid, libc::SIGUSR1) } == 0 { armed.push(index); }
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline && armed.iter().any(|&i| !SLOTS_TABLE[i].done.load(Ordering::Acquire)) {
        std::thread::sleep(Duration::from_millis(10));
    }
    let maps = mappings();
    tracing::error!(reason, threads = targets.len(), "stall stack dump begin");
    for index in 0..NEXT_SLOT.load(Ordering::Relaxed) {
        let slot = &SLOTS_TABLE[index];
        let tid = slot.tid.load(Ordering::Acquire);
        let stat = task_file(tid, "stat");
        let state = stat.rsplit_once(") ").and_then(|(_, rest)| rest.split(' ').next()).unwrap_or("?").to_string();
        let frames = if slot.done.load(Ordering::Acquire) {
            (0..slot.depth.load(Ordering::Relaxed)).map(|i| frame(slot.frames[i].load(Ordering::Relaxed), &maps))
                .collect::<Vec<_>>().join(" <- ")
        } else { "(no reply: blocked with SIGUSR1 masked or uninterruptible)".into() };
        tracing::error!(tid, name = %task_file(tid, "comm"), state, wchan = %task_file(tid, "wchan"),
            frames = %frames, "stall stack");
        slot.tid.store(0, Ordering::Release);
    }
    tracing::error!(watcher = %std::backtrace::Backtrace::force_capture(), "stall stack dump end");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inert_unless_exactly_one() {
        for value in [None, Some(""), Some("0"), Some("on"), Some("true")] {
            assert!(!enabled_by(value), "{value:?}");
        }
        assert!(enabled_by(Some("1")));
        // Unset in the test process: install arms nothing and hooks stay no-ops.
        assert!(std::env::var(STACK_ON_TIMEOUT).is_err());
        install();
        assert!(!enabled());
        let before = PROGRESS.load(Ordering::Relaxed);
        progress();
        drop(in_flight());
        assert_eq!(PROGRESS.load(Ordering::Relaxed), before);
        assert_eq!(IN_FLIGHT.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_stalled_submission_dumps_exactly_once() {
        let mut stall = Stall::default();
        let tick = POLL;
        // Idle (nothing in flight) never fires, however long.
        for _ in 0..1000 { assert!(!stall.poll(0, 0, tick)); }
        // A submission that keeps issuing never fires.
        for issue in 1..=1000u64 { assert!(!stall.poll(issue, 1, tick)); }
        // Then it stops issuing: one dump once STALL has passed, none after.
        let polls = (STALL.as_millis() / tick.as_millis()) as usize;
        let fired: Vec<bool> = (0..polls * 4).map(|_| stall.poll(1000, 1, tick)).collect();
        assert_eq!(fired.iter().filter(|&&f| f).count(), 1);
        assert_eq!(fired.iter().position(|&f| f), Some(polls - 1));
        // Progress rearms it: a second stall dumps once more.
        assert!(!stall.poll(1001, 1, tick));
        assert_eq!((0..polls * 2).filter(|_| stall.poll(1001, 1, tick)).count(), 1);
        // A finished submission rearms too, without dumping.
        assert!(!stall.poll(1001, 0, tick));
    }

    #[test]
    fn frames_report_module_offsets() {
        let maps = vec![(0x1000, 0x2000, 0x400, "/opt/cuteafd/bin/cuteafd".to_string())];
        assert_eq!(frame(0x1010, &maps), "/opt/cuteafd/bin/cuteafd+0x410");
        assert_eq!(frame(0x9000, &maps), "0x9000");
    }
}
