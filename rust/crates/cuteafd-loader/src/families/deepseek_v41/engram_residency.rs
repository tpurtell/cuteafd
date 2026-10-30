//! Background residency sweep for the Engram tables' residency hints: one
//! `mincore` per slice of each table's mapping, so gathers trust a bitmap
//! instead of a `mincore` per row page. A full sweep of a 101 GB table takes
//! ~365 ms of CPU (25M pages), so it runs in slices across a few seconds and
//! restarts at once when the gathers start taking late faults (stale bits).
use crate::EngramTable;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Pages per `mincore` slice (~15 ms) and the pause between slices: a full
/// sweep of both layers' tables every few seconds at a small fraction of a core.
const SLICE_PAGES: usize = 1 << 20;
const PAUSE: Duration = Duration::from_millis(100);

pub struct ResidencySweep { stop: Arc<AtomicBool>, thread: Option<std::thread::JoinHandle<()>> }

impl ResidencySweep {
    /// Enable the hint on every mapped table of `tables` and start sweeping.
    pub fn start(tables: [Arc<EngramTable>; 2]) -> std::io::Result<Self> {
        for table in &tables {
            table.weights().enable_residency_hint();
            if let Some(scales) = table.scales() { scales.enable_residency_hint(); }
        }
        let stop = Arc::new(AtomicBool::new(false));
        let running = Arc::clone(&stop);
        let thread = std::thread::Builder::new().name("engram-residency".into()).spawn(move || {
            let mut scratch = Vec::new();
            // The first sweep runs unpaused (well under a second of CPU), so the
            // hint reflects the page cache from the first requests on.
            let mut faults = u64::MAX;
            while !running.load(Ordering::Acquire) {
                for table in &tables {
                    for mapped in [Some(table.weights()), table.scales()].into_iter().flatten() {
                        let pages = mapped.residency_pages();
                        // Trust only a sweep no distrust interrupted (a distrust zeroes the bits).
                        let generation = mapped.stats().snapshot().residency_hint_distrusts;
                        let mut at = 0;
                        while at < pages && !running.load(Ordering::Acquire) {
                            match mapped.sweep_residency(at, SLICE_PAGES, &mut scratch) {
                                Ok(0) => break,
                                Ok(swept) => at += swept,
                                Err(error) => {
                                    tracing::warn!(%error, "Engram residency sweep failed; rows take the read path");
                                    return;
                                }
                            }
                            // Late faults mean stale bits: finish this sweep without pausing.
                            let now = tables.iter().map(|t| t.weights().stats().snapshot().late_major_faults).sum::<u64>();
                            if now == faults { std::thread::sleep(PAUSE); } else if faults != u64::MAX { faults = now; }
                        }
                        if at >= pages && mapped.stats().snapshot().residency_hint_distrusts == generation {
                            mapped.trust_residency_hint();
                        }
                    }
                }
                if faults == u64::MAX { faults = tables.iter().map(|t| t.weights().stats().snapshot().late_major_faults).sum(); }
            }
        })?;
        Ok(Self { stop, thread: Some(thread) })
    }
}

impl Drop for ResidencySweep {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() { let _ = thread.join(); }
    }
}
