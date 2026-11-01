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
//! - default build (glibc): `mallinfo2()` sums every arena, locking each arena
//!   while walking its free lists, plus process-global mmapped chunks. `live` =
//!   `uordblks + hblkhd`, `active` = `arena` (non-mmapped arena bytes),
//!   `mapped` = `arena + hblkhd`, `fragmentation` = `arena - uordblks`.
//!   `resident`, `retained` and `metadata` are 0. Calls are serialized here;
//!   allocations continue between arena visits, so this is not an atomic
//!   process-wide snapshot. The lock walk can briefly delay malloc/free.
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
//! `/v1/stats` for every family. The snapshot is refreshed by a background
//! reporter (every `CUTEAFD_HOST_HEAP_SECS`, with a minimum/default of 60 s).
//! The serving path only reads the cached copy, so it may be up to one
//! reporter interval stale.  Logging is opt-in:
//!
//! - `CUTEAFD_HOST_HEAP_EVERY=N` — one `cuteafd::hostheap` line every N retired
//!   requests (0/unset = off).
//! - `CUTEAFD_HOST_HEAP_SECS=N` - log every max(N, 60) seconds (0/unset = off;
//!   the cache is still refreshed every 60 s for `/v1/stats`).
//! - `CUTEAFD_JEMALLOC_STATS_SECS=N` (jemalloc build) — newline-delimited
//!   jemalloc stats every N seconds (0/unset = OFF; the default).
//! - `CUTEAFD_JEMALLOC_STATS_FILE=path` — append those dumps to `path` instead
//!   of stderr (the coordinator log).  A file needs a writable mount; the
//!   coordinator container has none by default.
//!
//! The serving path reads only the cached snapshot. Allocator sampling and its
//! arena-lock walk happen in the reporter, no more than once a minute.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{OnceLock, RwLock};
#[cfg(not(feature = "jemalloc"))]
use std::sync::Mutex;
use std::time::Duration;

/// Minimum/default refresh cadence: probing glibc walks and locks every arena.
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
    // Serialize probes. glibc locks each arena while summing its free lists;
    // this is not an atomic snapshot across arenas or concurrent mmap activity.
    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: mallinfo2 only reads this process's allocator state.
    let mi = unsafe { libc::mallinfo2() };
    glibc_snapshot(mi.arena as u64, mi.uordblks as u64, mi.hblkhd as u64, rss_bytes())
}

#[cfg(not(feature = "jemalloc"))]
fn glibc_snapshot(arena: u64, uordblks: u64, hblkhd: u64, rss: u64) -> HeapSnapshot {
    HeapSnapshot {
        rss,
        // uordblks excludes mmapped chunks; hblkhd is where they live.
        live: uordblks.saturating_add(hblkhd),
        active: arena,
        mapped: arena.saturating_add(hblkhd),
        fragmentation: arena.saturating_sub(uordblks),
        hblkhd,
        ..HeapSnapshot::default()
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
    }
}

pub(crate) fn allocator_name() -> &'static str {
    if cfg!(feature = "jemalloc") { "jemalloc" } else { "glibc" }
}

fn allocator_scope() -> &'static str {
    // Prefixed jemalloc does not see the native library's glibc allocations.
    if cfg!(feature = "jemalloc") { "rust_all_arenas" } else { "all_arenas" }
}

/// Cached snapshot, refreshed by the background reporter.  The serving path
/// only reads this; it may be up to one reporter interval stale.
static CACHE: RwLock<Option<HeapSnapshot>> = RwLock::new(None);

fn refresh_cache() -> HeapSnapshot {
    let s = snapshot();
    let mut w = CACHE.write().unwrap_or_else(|e| e.into_inner());
    *w = Some(s);
    drop(w);
    cuteafd_api::openai::publish_host_heap(snapshot_json());
    s
}

fn cached() -> HeapSnapshot {
    let guard = CACHE.read().unwrap_or_else(|e| e.into_inner());
    (*guard).unwrap_or_default()
}

/// The `/v1/stats` object.  Reads the cached snapshot only (never advances the
/// jemalloc epoch on the serving path).
pub(crate) fn snapshot_json() -> serde_json::Value {
    let s = cached();
    serde_json::json!({
        "allocator": allocator_name(),
        "scope": allocator_scope(),
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
        scope = allocator_scope(),
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

fn refresh_secs(requested: u64) -> u64 {
    requested.max(DEFAULT_REFRESH_SECS)
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
    let refresh = refresh_secs(secs);
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
        std::hint::black_box(&held);
    }

    #[test]
    fn env_zero_means_off() {
        assert_eq!(refresh_secs(0), 60);
        assert_eq!(refresh_secs(1), 60);
        assert_eq!(refresh_secs(120), 120);
        assert_eq!(env_u64("CUTEAFD_HOST_HEAP_EVERY_UNSET_FOR_TEST"), 0);
    }

    #[test]
    fn json_names_the_allocator() {
        refresh_cache();
        let value = snapshot_json();
        assert_eq!(value["allocator"], allocator_name());
        assert!(value["live_bytes"].as_u64().unwrap_or(0) > 0);
        let expected = if cfg!(feature = "jemalloc") { "rust_all_arenas" } else { "all_arenas" };
        assert_eq!(value["scope"], expected);
    }

    #[cfg(not(feature = "jemalloc"))]
    #[test]
    fn glibc_live_includes_mmapped_chunks() {
        // Test the accounting directly: process-wide deltas race concurrent frees.
        let s = glibc_snapshot(1024, 768, 4096, 8192);
        assert_eq!(s.live, 4864);
        assert_eq!(s.active, 1024);
        assert_eq!(s.mapped, 5120);
        assert_eq!(s.fragmentation, 256);
        assert_eq!(s.hblkhd, 4096);
        assert_eq!(s.rss, 8192);
    }

    #[cfg(feature = "jemalloc")]
    #[test]
    fn jemalloc_reports_retained_and_metadata() {
        refresh_cache();
        let s = snapshot();
        assert!(s.metadata > 0, "jemalloc metadata must be nonzero: {s:?}");
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
        assert_eq!(v["scope"], "rust_all_arenas");
        // mapped is active extents only; retained is reported separately.
        assert!(v["mapped_bytes"].as_u64().unwrap_or(0) >= v["active_bytes"].as_u64().unwrap_or(0));
    }
}
