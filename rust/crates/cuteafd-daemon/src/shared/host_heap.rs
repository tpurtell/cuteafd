//! Host-heap live-bytes probe (2026-10-26 residual-growth measurement).
//!
//! The 2026-10-25 leak fix bounded the host prefix-cache snapshot count, but a
//! load-proportional residual remained: ~1.3 MB per request in large anonymous
//! mappings (retained snapshot heap plus glibc arena extensions).  This module
//! makes the allocator's own view of that heap visible without a profiler.
//!
//! Two allocator backends, with DIFFERENT scopes — never compare their numbers
//! across builds:
//!
//! - default build (glibc): `mallinfo2()`.  **Calling thread's arena only.**
//!   glibc keeps a separate arena per thread (up to `M_ARENA_MAX`), and
//!   `mallinfo2` reports the arena of the thread that calls it — not the main
//!   arena and not the process.  The reporter thread may therefore see its own
//!   (near-empty) arena rather than the serving threads' heaps.  `live` =
//!   `uordblks + hblkhd` (in-use bytes in that arena plus mmapped chunks),
//!   `active` = `arena` (non-mmapped arena bytes), `mapped` = `arena + hblkhd`,
//!   `fragmentation` = `arena - uordblks`, and `hblkhd` is reported separately.
//!   `resident`, `retained` and `metadata` are 0.  A mutex serializes the
//!   `mallinfo2` calls (it is MT-Unsafe); that does NOT make it safe against
//!   concurrent `malloc`/`free` in other threads, so the numbers are a
//!   best-effort sample.
//! - `--features jemalloc`: jemalloc is the global allocator; all arenas are
//!   covered.  `live` = `stats.allocated`, `active` = `stats.active`,
//!   `mapped` = `stats.mapped` (**active extents only**), `retained` =
//!   `stats.retained` (retained virtual mappings), `metadata` =
//!   `stats.metadata`, `resident` = `stats.resident` (allocator-resident
//!   bytes, NOT process RSS), `fragmentation` = `active - allocated`.
//!
//! jemalloc is built with the `_rjem_` symbol prefix (tikv-jemallocator's
//! `--with-jemalloc-prefix=_rjem_`), so it only replaces Rust's global
//! allocator.  The prebuilt native library (`libcuteafd_native.so`) keeps
//! allocating from glibc, and that host heap is invisible to these stats.  Do
//! NOT enable the unprefixed override to "fix" this: it would interpose on the
//! native library and change its allocator behaviour.
//!
//! Always available: `rss` from `/proc/self/statm`, and a `host_heap` object in
//! `/v1/stats` (see `serving_stats`).  The snapshot is refreshed by a background
//! reporter (at most once per `CUTEAFD_HOST_HEAP_SECS`, or every 60 s when that
//! is 0); the serving path only reads the cached copy, so it may be up to one
//! reporter interval stale.  Logging is opt-in:
//!
//! - `CUTEAFD_HOST_HEAP_EVERY=N` — one `cuteafd::hostheap` line every N retired
//!   requests (0/unset = off).
//! - `CUTEAFD_HOST_HEAP_SECS=N` — one line every N seconds (0/unset = off; the
//!   cache is still refreshed every 60 s for `/v1/stats`).
//! - `CUTEAFD_JEMALLOC_STATS_SECS=N` (jemalloc build) — newline-delimited
//!   jemalloc stats every N seconds (0/unset = OFF; the default).
//! - `CUTEAFD_JEMALLOC_STATS_FILE=path` — append those dumps to `path` instead
//!   of stderr (the coordinator log).  A file needs a writable mount; the
//!   coordinator container has none by default.
//!
//! The probe never allocates on the serving path beyond the log line itself and
//! never blocks; `mallinfo2`/jemalloc reads are a handful of microseconds.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};
#[cfg(not(feature = "jemalloc"))]
use std::sync::Mutex;
use std::time::Duration;

/// Refresh cadence for the cached snapshot when `CUTEAFD_HOST_HEAP_SECS` is 0.
const DEFAULT_REFRESH_SECS: u64 = 60;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct HeapSnapshot {
    /// Process resident set size (`/proc/self/statm`).
    pub rss: u64,
    /// Bytes currently allocated by the application (live objects).
    pub live: u64,
    /// Bytes in active allocator pages (live + fragmentation).
    pub active: u64,
    /// Bytes mapped by the allocator (active extents; jemalloc `stats.mapped`).
    pub mapped: u64,
    /// Allocator-resident bytes (jemalloc `stats.resident`; 0 for glibc).
    /// This is NOT process RSS.
    pub resident: u64,
    /// Retained virtual mappings (jemalloc `stats.retained`; 0 for glibc).
    pub retained: u64,
    /// Allocator metadata bytes (jemalloc `stats.metadata`; 0 for glibc).
    pub metadata: u64,
    /// `active - live` (jemalloc) or `arena - uordblks` (glibc).
    pub fragmentation: u64,
    /// glibc mmapped-chunk bytes (`hblkhd`; 0 for jemalloc).
    pub hblkhd: u64,
    /// Arena limit (jemalloc only; 0 for glibc).
    pub arenas: u64,
    /// True for the glibc backend, which sees only the calling thread's arena.
    pub calling_thread_arena: bool,
}

fn page_size() -> u64 {
    static PAGE: OnceLock<u64> = OnceLock::new();
    *PAGE.get_or_init(|| {
        // SAFETY: sysconf(_SC_PAGESIZE) has no preconditions.
        let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if v > 0 { v as u64 } else { 4096 }
    })
}

fn rss_bytes() -> u64 {
    let Ok(text) = std::fs::read_to_string("/proc/self/statm") else { return 0 };
    let pages: u64 = text.split_whitespace().nth(1).and_then(|v| v.parse().ok()).unwrap_or(0);
    pages.saturating_mul(page_size())
}

#[cfg(not(feature = "jemalloc"))]
pub(crate) fn snapshot() -> HeapSnapshot {
    // mallinfo2 is MT-Unsafe: serialize the calls.  This does not make it safe
    // against concurrent malloc/free in other threads; it only prevents two
    // mallinfo2 calls from racing each other.
    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: mallinfo2 only reads this process's allocator state.
    let mi = unsafe { libc::mallinfo2() };
    let arena = mi.arena as u64;
    let hblkhd = mi.hblkhd as u64;
    let uordblks = mi.uordblks as u64;
    HeapSnapshot {
        rss: rss_bytes(),
        // uordblks excludes mmapped chunks; hblkhd is where they live.
        live: uordblks.saturating_add(hblkhd),
        active: arena,
        mapped: arena.saturating_add(hblkhd),
        resident: 0,
        retained: 0,
        metadata: 0,
        fragmentation: arena.saturating_sub(uordblks),
        hblkhd,
        arenas: 0,
        calling_thread_arena: true,
    }
}

#[cfg(feature = "jemalloc")]
pub(crate) fn snapshot() -> HeapSnapshot {
    use tikv_jemalloc_ctl::{arenas, epoch, stats};
    // Refresh the cached stats; a failed advance leaves the previous epoch.
    if let Err(e) = epoch::advance() {
        static WARNED: AtomicBool = AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::Relaxed) {
            tracing::warn!(target: "cuteafd::hostheap", error = %e, "jemalloc epoch advance failed");
        }
    }
    let read = |value: Result<usize, tikv_jemalloc_ctl::Error>| value.unwrap_or(0) as u64;
    let live = read(stats::allocated::read());
    let active = read(stats::active::read());
    HeapSnapshot {
        rss: rss_bytes(),
        live,
        active,
        mapped: read(stats::mapped::read()),
        resident: read(stats::resident::read()),
        retained: read(stats::retained::read()),
        metadata: read(stats::metadata::read()),
        fragmentation: active.saturating_sub(live),
        hblkhd: 0,
        arenas: arenas::narenas::read().unwrap_or(0) as u64,
        calling_thread_arena: false,
    }
}

pub(crate) fn allocator_name() -> &'static str {
    if cfg!(feature = "jemalloc") { "jemalloc" } else { "glibc" }
}

/// Cached snapshot, refreshed by the background reporter.  The serving path
/// only reads this; it may be up to one reporter interval stale.
static CACHE: RwLock<Option<HeapSnapshot>> = RwLock::new(None);

fn refresh_cache() -> HeapSnapshot {
    let s = snapshot();
    let mut w = CACHE.write().unwrap_or_else(|e| e.into_inner());
    *w = Some(s);
    s
}

fn cached() -> HeapSnapshot {
    let guard = CACHE.read().unwrap_or_else(|e| e.into_inner());
    (*guard).unwrap_or_else(|| HeapSnapshot {
        // A missing cache must not claim all-arena coverage on the glibc build.
        calling_thread_arena: !cfg!(feature = "jemalloc"),
        ..HeapSnapshot::default()
    })
}

/// The `/v1/stats` object.  Reads the cached snapshot only (never advances the
/// jemalloc epoch on the serving path).
pub(crate) fn snapshot_json() -> serde_json::Value {
    let s = cached();
    serde_json::json!({
        "allocator": allocator_name(),
        "scope": if s.calling_thread_arena { "calling_thread_arena" } else { "all_arenas" },
        "rss_bytes": s.rss,
        "live_bytes": s.live,
        "active_bytes": s.active,
        "mapped_bytes": s.mapped,
        "resident_bytes": s.resident,
        "retained_bytes": s.retained,
        "metadata_bytes": s.metadata,
        "fragmentation_bytes": s.fragmentation,
        "hblkhd_bytes": s.hblkhd,
        "arenas": s.arenas,
    })
}

fn log_snapshot(requests: u64, s: &HeapSnapshot) {
    tracing::info!(target: "cuteafd::hostheap",
        requests,
        allocator = allocator_name(),
        scope = if s.calling_thread_arena { "calling_thread_arena" } else { "all_arenas" },
        live_bytes = s.live,
        active_bytes = s.active,
        mapped_bytes = s.mapped,
        resident_bytes = s.resident,
        retained_bytes = s.retained,
        metadata_bytes = s.metadata,
        fragmentation_bytes = s.fragmentation,
        hblkhd_bytes = s.hblkhd,
        rss_bytes = s.rss,
        arenas = s.arenas,
        "host heap");
}

fn log_line(requests: u64) {
    log_snapshot(requests, &cached());
}

fn env_u64(name: &str) -> u64 {
    match std::env::var(name) {
        Ok(v) => match v.trim().parse() {
            Ok(n) => n,
            Err(_) => {
                tracing::warn!(target: "cuteafd::hostheap", var = name, value = %v,
                    "host-heap env var is not a number; using 0");
                0
            }
        },
        Err(_) => 0,
    }
}

static RETIRED: AtomicU64 = AtomicU64::new(0);
static EVERY: OnceLock<u64> = OnceLock::new();
static STARTED: AtomicBool = AtomicBool::new(false);

/// Called once per retired request (finished, cancelled or failed).
pub(crate) fn on_request_retired() {
    // Always count, even when per-request logging is off.
    let n = RETIRED.fetch_add(1, Ordering::Relaxed) + 1;
    let every = *EVERY.get_or_init(|| env_u64("CUTEAFD_HOST_HEAP_EVERY"));
    if every == 0 {
        return;
    }
    if n % every == 0 {
        log_line(n);
    }
}

/// Start the optional periodic reporters.  Call once, after tracing is installed.
/// Idempotent: a second call is a no-op.
pub(crate) fn start() {
    if STARTED.swap(true, Ordering::SeqCst) {
        return;
    }
    let secs = env_u64("CUTEAFD_HOST_HEAP_SECS");
    let every = env_u64("CUTEAFD_HOST_HEAP_EVERY");
    let _ = EVERY.set(every);
    // Prime the cache so /v1/stats is meaningful before the first tick.
    refresh_cache();
    let refresh = if secs > 0 { secs } else { DEFAULT_REFRESH_SECS };
    let spawned = std::thread::Builder::new().name("host-heap".into()).spawn(move || loop {
        std::thread::sleep(Duration::from_secs(refresh));
        let s = refresh_cache();
        if secs > 0 {
            log_snapshot(RETIRED.load(Ordering::Relaxed), &s);
        }
    });
    if let Err(e) = spawned {
        tracing::warn!(target: "cuteafd::hostheap", error = %e, "failed to spawn host-heap reporter");
    }
    #[cfg(feature = "jemalloc")]
    start_jemalloc_stats();
}

#[cfg(feature = "jemalloc")]
fn start_jemalloc_stats() {
    use std::io::Write;
    let secs = env_u64("CUTEAFD_JEMALLOC_STATS_SECS");
    if secs == 0 {
        return;
    }
    // Open the destination once; a failure is reported once, not every tick.
    let mut file = match std::env::var("CUTEAFD_JEMALLOC_STATS_FILE").ok() {
        Some(path) => match std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            Ok(file) => Some(file),
            Err(e) => {
                tracing::warn!(target: "cuteafd::hostheap", path = %path, error = %e,
                    "cannot open CUTEAFD_JEMALLOC_STATS_FILE; jemalloc stats disabled");
                return;
            }
        },
        None => None,
    };
    let spawned = std::thread::Builder::new().name("jemalloc-stats".into()).spawn(move || loop {
        std::thread::sleep(Duration::from_secs(secs));
        // Newline-delimited table, not one huge JSON line: journald's LineMax
        // splits a ~2.8 MB line and the dump becomes unreadable.
        let mut options = tikv_jemalloc_ctl::stats_print::Options::default();
        options.json_format = false;
        options.skip_constants = true;
        options.skip_per_arena = true;
        options.skip_bin_size_classes = true;
        options.skip_large_size_classes = true;
        options.skip_mutex_statistics = true;
        match file.as_mut() {
            Some(f) => {
                let _ = tikv_jemalloc_ctl::stats_print::stats_print(&mut *f, options);
                let _ = f.write_all(b"\n");
            }
            None => {
                let mut out = std::io::stderr();
                let _ = tikv_jemalloc_ctl::stats_print::stats_print(&mut out, options);
                let _ = out.write_all(b"\n");
            }
        }
    });
    if let Err(e) = spawned {
        tracing::warn!(target: "cuteafd::hostheap", error = %e, "failed to spawn jemalloc-stats reporter");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_reports_live_bytes_and_rss() {
        let held = vec![0u8; 4 << 20];
        let s = snapshot();
        assert!(s.rss > 0, "rss must be readable: {s:?}");
        assert!(s.live > 0, "live bytes must be nonzero: {s:?}");
        assert!(s.mapped >= s.active, "mapped {} < active {}: {s:?}", s.mapped, s.active);
        std::hint::black_box(&held);
    }

    #[test]
    fn env_zero_means_off() {
        assert_eq!(env_u64("CUTEAFD_HOST_HEAP_EVERY_UNSET_FOR_TEST"), 0);
    }

    #[test]
    fn json_names_the_allocator() {
        refresh_cache();
        let value = snapshot_json();
        assert_eq!(value["allocator"], allocator_name());
        assert!(value["live_bytes"].as_u64().unwrap_or(0) > 0);
        let expected = if cfg!(feature = "jemalloc") { "all_arenas" } else { "calling_thread_arena" };
        assert_eq!(value["scope"], expected);
    }

    // The old glibc path reported live = uordblks, which excludes mmapped
    // chunks; a > M_MMAP_THRESHOLD allocation then left `live` flat.  glibc's
    // dynamic threshold can rise to 32 MiB, so allocate 64 MiB to guarantee an
    // mmapped chunk.  Run with --test-threads=1 if this flakes: concurrent
    // frees can only shrink the delta, and the 64 MiB allocation leaves a
    // 32 MiB margin.
    #[cfg(not(feature = "jemalloc"))]
    #[test]
    fn glibc_live_includes_mmapped_chunks() {
        let before = snapshot();
        let held = vec![0u8; 64 << 20];
        std::hint::black_box(&held);
        let after = snapshot();
        let hblkhd_grew = after.hblkhd.saturating_sub(before.hblkhd);
        assert!(
            hblkhd_grew >= 32 << 20,
            "hblkhd grew only {hblkhd_grew} bytes for a 64 MiB mmapped allocation: before={before:?} after={after:?}"
        );
        let live_grew = after.live.saturating_sub(before.live);
        assert!(
            live_grew >= 32 << 20,
            "live grew only {live_grew} bytes for a 64 MiB mmapped allocation: before={before:?} after={after:?}"
        );
    }

    #[cfg(feature = "jemalloc")]
    #[test]
    fn jemalloc_reports_retained_and_metadata() {
        refresh_cache();
        let s = snapshot();
        assert!(s.metadata > 0, "jemalloc metadata must be nonzero: {s:?}");
        assert!(s.mapped >= s.active, "mapped {} < active {}: {s:?}", s.mapped, s.active);
        let v = snapshot_json();
        for key in [
            "allocator",
            "scope",
            "live_bytes",
            "active_bytes",
            "mapped_bytes",
            "resident_bytes",
            "retained_bytes",
            "metadata_bytes",
            "fragmentation_bytes",
            "hblkhd_bytes",
            "arenas",
        ] {
            assert!(v.get(key).is_some(), "missing {key}: {v}");
        }
        assert_eq!(v["allocator"], "jemalloc");
        assert_eq!(v["scope"], "all_arenas");
        // mapped is active extents only; retained is reported separately.
        assert!(v["mapped_bytes"].as_u64().unwrap_or(0) >= v["active_bytes"].as_u64().unwrap_or(0));
    }
}
