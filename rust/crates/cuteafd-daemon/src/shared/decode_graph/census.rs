//! Opt-in diagnostics; the live control file is read only at idle boundaries.
use std::sync::{atomic::{AtomicU8, Ordering}, OnceLock};

static ENABLED: OnceLock<bool> = OnceLock::new();
static STARTUP_EAGER: OnceLock<bool> = OnceLock::new();
static EAGER: AtomicU8 = AtomicU8::new(0);

pub(crate) fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var_os("CUTEAFD_GRAPH_CENSUS").is_some_and(|v| v == "1"))
}

fn bank_id(bank: &str) -> Option<u8> {
    Some(match bank { "baseline" => 0, "sparse" => 1, "index" => 2,
        "layer_row" => 3, "head" => 4, "dspark" => 5, "window" => 6, _ => return None })
}

pub(crate) fn idle() {
    if !enabled() { return }
    let Some(path) = std::env::var_os("CUTEAFD_GRAPH_CENSUS_CONTROL") else { return };
    match std::fs::read_to_string(path).ok().and_then(|s| bank_id(s.trim())) {
        Some(bank) => {
            if EAGER.swap(bank, Ordering::Relaxed) != bank {
                tracing::info!(target: "cuteafd::graph_capture", bank, "graph census idle arm switch");
            }
        }
        None => tracing::error!(target: "cuteafd::graph_capture", "invalid graph census control; retaining arm"),
    }
}

fn startup_eager(bank: &str, value: Option<&str>) -> bool {
    bank == "layer_row" && value == Some("layer_row")
}

pub(crate) fn eager(bank: &str) -> bool {
    // WIP launchers admit this startup-only arm without enabling census timing.
    let forced = bank == "layer_row" && *STARTUP_EAGER.get_or_init(|| {
        startup_eager("layer_row", std::env::var("CUTEAFD_GRAPH_EAGER").ok().as_deref())
    });
    forced || (enabled() && bank_id(bank).is_some_and(|id| id != 0 && id == EAGER.load(Ordering::Relaxed)))
}

pub(crate) fn arm() -> u8 { EAGER.load(Ordering::Relaxed) }

/// # Safety
/// Inputs and captured storage remain live through queued work on this stream.
pub(crate) unsafe fn dispatch(library: &cuteafd_ffi::NativeLibrary,
    stream: *mut std::ffi::c_void, bank: &std::ffi::CStr, rows: usize,
    graph: Option<*mut std::ffi::c_void>, eager_call: impl FnOnce() -> anyhow::Result<()>) -> anyhow::Result<()> {
    let replay = graph.is_some() && !eager(bank.to_str().expect("static census bank"));
    let call = || {
        if replay {
            // SAFETY: caller pins the executable and stream through queued use.
            unsafe { library.cuda_graph_launch(graph.unwrap(), stream) }
        } else { eager_call() }
    };
    if enabled() {
        // SAFETY: caller owns this stream and the closure preserves its device.
        unsafe { library.cuda_graph_census_call(stream, bank, rows, replay, arm(), call) }
    } else { call() }
}

/// Count-only coverage: no device events, synchronization or census timing.
/// Families record one dispatch per logical segment, including eager bypasses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Launch { Replay, Eager, CaptureLaunch }

#[derive(Debug, Clone, Copy, Default)]
struct Counts { replay: u64, eager: u64, capture_launch: u64 }

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct WidthKey {
    mode: &'static str,
    phase: &'static str,
    requests: usize,
    step_rows: usize,
    rank: usize,
    lane: usize,
    rows: usize,
    segment: &'static str,
    reason: &'static str,
}

#[derive(Default)]
struct CoverageState {
    active: bool,
    step: Option<(&'static str, usize, usize)>,
    counts: std::collections::BTreeMap<WidthKey, Counts>,
    steps: u64,
}

pub(crate) struct LaunchCoverage {
    enabled: bool,
    state: std::cell::RefCell<CoverageState>,
}

pub(crate) struct CoverageSession<'a>(&'a LaunchCoverage);
pub(crate) struct CoverageStep<'a> {
    census: &'a LaunchCoverage,
    previous: Option<(&'static str, usize, usize)>,
}

impl LaunchCoverage {
    pub fn new() -> Self {
        Self::with_enabled(std::env::var_os("CUTEAFD_DIAG_GRAPH_COVERAGE").is_some_and(|v| v == "1"))
    }

    fn with_enabled(enabled: bool) -> Self {
        Self { enabled, state: std::cell::RefCell::new(CoverageState::default()) }
    }

    pub fn enabled(&self) -> bool { self.enabled }

    /// Startup/check-routes activity is excluded, and unwind flushes partial counts.
    pub fn session(&self) -> CoverageSession<'_> {
        if self.enabled {
            *self.state.borrow_mut() = CoverageState { active: true, ..CoverageState::default() };
        }
        CoverageSession(self)
    }

    pub fn step(&self, phase: &'static str, requests: usize, rows: usize) -> CoverageStep<'_> {
        let previous = if self.enabled {
            self.state.borrow_mut().step.replace((phase, requests, rows))
        } else { None };
        CoverageStep { census: self, previous }
    }

    pub fn record(&self, mode: &'static str, segment: &'static str, rank: usize,
        lane: usize, rows: usize, launch: Launch, reason: &'static str) {
        if !self.enabled { return }
        let mut state = self.state.borrow_mut();
        if !state.active { return }
        let Some((phase, requests, step_rows)) = state.step else { return };
        let counts = state.counts.entry(WidthKey { mode, phase, requests, step_rows,
            rank, lane, rows, segment, reason }).or_default();
        match launch {
            Launch::Replay => counts.replay += 1,
            Launch::Eager => counts.eager += 1,
            Launch::CaptureLaunch => counts.capture_launch += 1,
        }
    }

    /// Periodic per-width deltas distinguish C1/C16 request counts without
    /// device timing, host polling or per-launch logging.
    pub fn flush(&self) {
        if !self.enabled { return }
        let counts = std::mem::take(&mut self.state.borrow_mut().counts);
        for (key, count) in counts {
            tracing::info!(target: "cuteafd::graph_capture", mode = key.mode, phase = key.phase,
                requests = key.requests, step_rows = key.step_rows, rank = key.rank,
                lane = key.lane, rows = key.rows, segment = key.segment, reason = key.reason,
                replay = count.replay, eager = count.eager, capture_launch = count.capture_launch,
                "graph launch coverage");
        }
    }
}

impl Drop for CoverageStep<'_> {
    fn drop(&mut self) {
        if self.census.enabled {
            let report = {
                let mut state = self.census.state.borrow_mut();
                state.steps += 1;
                state.step = self.previous;
                state.steps % 128 == 0
            };
            if report { self.census.flush(); }
        }
    }
}

impl Drop for CoverageSession<'_> {
    fn drop(&mut self) {
        if self.0.enabled {
            self.0.flush();
            self.0.state.borrow_mut().active = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Launch, LaunchCoverage};

    #[test]
    fn coverage_excludes_startup_and_separates_exact_c16_lane_widths() {
        let census = LaunchCoverage::with_enabled(true);
        let step = census.step("verify", 16, 64);
        census.record("layers", "front", 0, 0, 31, Launch::Replay, "hit");
        assert!(census.state.borrow().counts.is_empty());
        let session = census.session();
        drop(step);
        let step = census.step("verify", 16, 64);
        for (lane, rows, launch, reason) in [(0, 31, Launch::Eager, "unknown-key"),
            (1, 33, Launch::Replay, "hit"), (1, 33, Launch::CaptureLaunch, "first-miss")] {
            census.record("layers", "front", 0, lane, rows, launch, reason);
        }
        let state = census.state.borrow();
        assert_eq!(state.counts.len(), 3);
        assert!(state.counts.keys().all(|key| key.requests == 16 && key.step_rows == 64));
        assert_eq!(state.counts.values().map(|n| n.eager).sum::<u64>(), 1);
        assert_eq!(state.counts.values().map(|n| n.replay).sum::<u64>(), 1);
        assert_eq!(state.counts.values().map(|n| n.capture_launch).sum::<u64>(), 1);
        drop(state);
        drop(step);
        assert_eq!(census.state.borrow().counts.len(), 3);
        assert!(census.state.borrow().step.is_none());
        drop(session);
        assert!(census.state.borrow().counts.is_empty());
        assert!(!census.state.borrow().active);
    }

    #[test]
    fn disabled_coverage_has_no_counter_or_step_mutations() {
        let census = LaunchCoverage::with_enabled(false);
        let _session = census.session();
        let _step = census.step("verify", 16, 64);
        census.record("heads", "front", 1, 0, 64, Launch::Replay, "hit");
        assert!(census.state.borrow().counts.is_empty());
        assert!(census.state.borrow().step.is_none());
        assert!(!census.state.borrow().active);
    }
    #[test]
    fn startup_eager_only_selects_layer_row_without_timing() {
        assert!(super::startup_eager("layer_row", Some("layer_row")));
        for value in [None, Some(""), Some("sparse"), Some("all")] {
            assert!(!super::startup_eager("layer_row", value));
        }
        assert!(!super::startup_eager("sparse", Some("layer_row")));
    }

    #[test]
    fn startup_eager_does_not_enable_census_timing() {
        const CHILD: &str = "CUTEAFD_GRAPH_EAGER_TEST_CHILD";
        if std::env::var_os(CHILD).is_some() {
            assert!(super::eager("layer_row"));
            assert!(!super::enabled());
            for bank in ["sparse", "index", "head", "dspark", "window"] {
                assert!(!super::eager(bank));
            }
            return;
        }
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "shared::decode_graph::census::tests::startup_eager_does_not_enable_census_timing"])
            .env(CHILD, "1").env("CUTEAFD_GRAPH_EAGER", "layer_row")
            .env_remove("CUTEAFD_GRAPH_CENSUS").env_remove("CUTEAFD_GRAPH_CENSUS_CONTROL")
            .status().unwrap();
        assert!(status.success());
    }

    #[test]
    fn unknown_control_does_not_select_a_bank() {
        assert_eq!(super::bank_id("baseline"), Some(0));
        for name in ["sparse", "index", "layer_row", "head", "dspark", "window"] {
            assert!(super::bank_id(name).is_some_and(|id| id > 0));
        }
        assert_eq!(super::bank_id("all"), None);
        assert_eq!(super::bank_id("../sparse"), None);
    }
}
