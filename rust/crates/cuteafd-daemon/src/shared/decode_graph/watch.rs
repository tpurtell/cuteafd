//! Process-wide capture observation, covering every verification return path.
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) struct CaptureWatch;

impl CaptureWatch {
    pub fn round() -> Self { Self }
    fn observe() {
        static ROUNDS: AtomicU64 = AtomicU64::new(0);
        static LAST: AtomicU64 = AtomicU64::new(0);
        static SITES: std::sync::Mutex<Vec<(String, u64)>> = std::sync::Mutex::new(Vec::new());
        let rounds = ROUNDS.fetch_add(1, Ordering::Relaxed) + 1;
        let interval = if tracing::enabled!(target: "cuteafd::graph_capture", tracing::Level::DEBUG) { 32 } else { 512 };
        if rounds % interval != 0 { return }
        let captures = cuteafd_ffi::graph_captures();
        let previous = LAST.swap(captures, Ordering::Relaxed);
        let now = cuteafd_ffi::graph_capture_sites();
        let mut before = SITES.lock().unwrap_or_else(|p| p.into_inner());
        let mut delta: Vec<_> = now.iter().map(|(site, n)| (site.rsplit('/').next().unwrap_or(site).to_string(),
            n.saturating_sub(before.iter().find(|(s, _)| s == site).map_or(0, |(_, m)| *m))))
            .filter(|(_, n)| *n > 0).collect();
        delta.sort_by(|a, b| b.1.cmp(&a.1));
        *before = now;
        tracing::info!(rounds, interval, captures = captures.saturating_sub(previous), sites = ?delta,
            "graph captures in recent verification rounds");
    }
}

impl Drop for CaptureWatch {
    fn drop(&mut self) { Self::observe(); }
}
