//! Process-wide capture observation, covering every verification return path.
use std::sync::Mutex;

pub(crate) struct CaptureWatch;
pub(crate) struct CaptureSession;

#[derive(Default)]
struct WatchState {
    initialized: bool,
    rounds: u64,
    reported: u64,
    captures: u64,
    sites: Vec<(String, u64)>,
}

static STATE: Mutex<WatchState> = Mutex::new(WatchState {
    initialized: false, rounds: 0, reported: 0, captures: 0, sites: Vec::new(),
});

impl WatchState {
    fn reset(&mut self, captures: u64, sites: Vec<(String, u64)>) {
        *self = Self { initialized: true, captures, sites, ..Self::default() };
    }

    fn delta(&mut self, captures: u64, sites: Vec<(String, u64)>) -> (u64, u64, Vec<(String, u64)>) {
        let rounds = self.rounds - self.reported;
        let mut delta: Vec<_> = sites.iter().map(|(site, n)| (site.clone(),
            n.saturating_sub(self.sites.iter().find(|(s, _)| s == site).map_or(0, |(_, m)| *m))))
            .filter(|(_, n)| *n > 0).collect();
        delta.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let count = captures.saturating_sub(self.captures);
        self.reported = self.rounds;
        self.captures = captures;
        self.sites = sites;
        (rounds, count, delta)
    }
}

impl CaptureWatch {
    /// Start after startup warm-up; startup captures cannot leak into the first interval.
    pub fn session() -> CaptureSession {
        Self::reset();
        CaptureSession
    }

    pub fn reset() {
        STATE.lock().unwrap_or_else(|p| p.into_inner())
            .reset(cuteafd_ffi::graph_captures(), cuteafd_ffi::graph_capture_sites());
    }

    pub fn round() -> Self {
        let mut state = STATE.lock().unwrap_or_else(|p| p.into_inner());
        if !state.initialized {
            state.reset(cuteafd_ffi::graph_captures(), cuteafd_ffi::graph_capture_sites());
        }
        Self
    }

    /// Include the last partial interval, e.g. at a battery's idle or shutdown boundary.
    pub fn flush() {
        let mut state = STATE.lock().unwrap_or_else(|p| p.into_inner());
        if !state.initialized { return }
        Self::report(&mut state, true);
    }

    fn report(state: &mut WatchState, final_interval: bool) {
        let captures = cuteafd_ffi::graph_captures();
        if state.rounds == state.reported && captures == state.captures { return }
        let (rounds, count, sites) = state.delta(captures, cuteafd_ffi::graph_capture_sites());
        tracing::info!(rounds, total_rounds = state.rounds, captures = count, ?sites, final_interval,
            "graph captures in recent verification rounds");
    }
}

impl Drop for CaptureWatch {
    fn drop(&mut self) {
        let mut state = STATE.lock().unwrap_or_else(|p| p.into_inner());
        state.rounds += 1;
        let interval = if tracing::enabled!(target: "cuteafd::graph_capture", tracing::Level::DEBUG) { 32 } else { 512 };
        // Surface a miss immediately, even if the battery ends before the next periodic report.
        if state.rounds - state.reported >= interval || cuteafd_ffi::graph_captures() != state.captures {
            Self::report(&mut state, false);
        }
    }
}

impl Drop for CaptureSession {
    fn drop(&mut self) { CaptureWatch::flush(); }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warmup_baseline_and_partial_interval_are_exact() {
        let mut state = WatchState::default();
        state.reset(90, vec![("startup".into(), 90)]);
        state.rounds = 7;
        assert_eq!(state.delta(90, vec![("startup".into(), 90)]), (7, 0, vec![]));
        state.rounds = 9;
        assert_eq!(state.delta(93, vec![("startup".into(), 90), ("decode".into(), 3)]),
            (2, 3, vec![("decode".into(), 3)]));
        state.reset(93, vec![("startup".into(), 90), ("decode".into(), 3)]);
        state.rounds = 1;
        assert_eq!(state.delta(93, vec![("startup".into(), 90), ("decode".into(), 3)]), (1, 0, vec![]));
    }
}
