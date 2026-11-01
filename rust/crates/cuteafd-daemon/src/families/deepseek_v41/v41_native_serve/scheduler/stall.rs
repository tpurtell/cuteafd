//! Diagnostic-only watchdog naming the serve-loop phase that stopped making
//! progress. It never changes scheduling, draft handling or request outcomes.
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::time::Duration;

static PROGRESS: AtomicU64 = AtomicU64::new(0);
static PHASE: AtomicU8 = AtomicU8::new(0);
static BUSY: AtomicUsize = AtomicUsize::new(0);
static DRAFT_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Serve-loop phases the watchdog reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Phase { Idle = 0, Retire = 1, Admission = 2, Decode = 3, Failed = 4 }

pub(crate) fn phase_name(phase: u8) -> &'static str {
    match phase { 1 => "retire", 2 => "admission/prefill", 3 => "decode round",
        4 => "failed-round cleanup", _ => "idle" }
}

/// Enter `phase` with `busy` live requests (active + parked); counts as progress.
pub(crate) fn mark(phase: Phase, busy: usize) {
    PHASE.store(phase as u8, Ordering::Relaxed);
    BUSY.store(busy, Ordering::Relaxed);
    tick();
}
/// Progress inside a long phase (a lane round, a prefill unit).
pub(crate) fn tick() { PROGRESS.fetch_add(1, Ordering::Relaxed); }

/// Seconds without progress before the watchdog reports a stall.
pub(crate) const STALL_SECS: u64 = 60;

/// Pure watchdog step: given the last seen progress count and how long it has
/// been unchanged, return the report (phase, busy, stalled seconds) if due.
///
/// W64 F3: a serve loop that marked `Idle, 0` while HTTP tickets stayed open
/// (admitted, never retired) is a stall too, so `open` is reported as
/// `"idle with open tickets"` with `busy = open`.
pub(crate) fn check(last: &mut (u64, u64), progress: u64, phase: u8, busy: usize, open: u64, step_secs: u64)
    -> Option<(&'static str, usize, u64)> {
    if progress != last.0 { *last = (progress, 0); return None; }
    let (name, count) = if busy > 0 && phase != Phase::Idle as u8 {
        (phase_name(phase), busy)
    } else if open > 0 {
        ("idle with open tickets", open as usize)
    } else {
        *last = (progress, 0);
        return None;
    };
    last.1 += step_secs;
    (last.1 >= STALL_SECS && last.1 % STALL_SECS < step_secs).then_some((name, count, last.1))
}

/// Scoped to one serve invocation, including error and unwind exits.
pub(crate) struct Watchdog {
    stop: std::sync::mpsc::Sender<()>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Watchdog {
    fn start(step: Duration, mut sample: impl FnMut() + Send + 'static) -> Self {
        let (stop, receive) = std::sync::mpsc::channel();
        let spawned = std::thread::Builder::new().name("serve-stall-watch".into()).spawn(move || {
            while matches!(receive.recv_timeout(step), Err(std::sync::mpsc::RecvTimeoutError::Timeout)) {
                sample();
            }
        });
        let thread = match spawned {
            Ok(thread) => Some(thread),
            Err(error) => {
                tracing::warn!(%error, "serve stall watchdog not started");
                None
            }
        };
        Self { stop, thread }
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        // Wake immediately, rather than waiting for the five-second sample interval.
        let _ = self.stop.send(());
        if let Some(thread) = self.thread.take() { let _ = thread.join(); }
        mark(Phase::Idle, 0);
    }
}

/// Start a fresh watchdog for this serve loop, not a detached process-lifetime thread.
pub(crate) fn spawn() -> Watchdog {
    mark(Phase::Idle, 0);
    let step = 5;
    let mut last = (u64::MAX, 0);
    Watchdog::start(Duration::from_secs(step), move || {
        let open = crate::shared::console::totals::active();
        if let Some((phase, busy, seconds)) = check(&mut last, PROGRESS.load(Ordering::Relaxed),
            PHASE.load(Ordering::Relaxed), BUSY.load(Ordering::Relaxed), open, step) {
            tracing::error!(phase, busy, open, seconds, "serve loop stalled: no round, unit or admission progress");
        }
    })
}

/// A failed draft proposal: the lane verifies each member's anchor alone.
pub(crate) fn anchor_only(seeds: &[(u64, u32, u64, usize)], lane: usize, error: &anyhow::Error) -> Vec<Vec<u32>> {
    let total = DRAFT_FALLBACKS.fetch_add(1, Ordering::Relaxed) + 1;
    tracing::warn!(lane, requests = seeds.len(), total, error = %format!("{error:#}"),
        "dSpark draft proposal failed; this lane round verifies anchors only");
    seeds.iter().map(|seed| vec![seed.1]).collect()
}

/// The liveness-gated draft fallback. A failed proposal verifies the anchors
/// alone only while the draft is still live; a terminal draft propagates its
/// error so the round fails instead of degrading forever. `live` is the
/// caller's [`DraftRuntime::require_live`] result.
pub(crate) fn draft_fallback(live: anyhow::Result<()>, seeds: &[(u64, u32, u64, usize)],
    lane: usize, error: &anyhow::Error,
) -> anyhow::Result<Vec<Vec<u32>>> {
    live?;
    Ok(anchor_only(seeds, lane, error))
}
pub(crate) fn draft_fallbacks() -> u64 { DRAFT_FALLBACKS.load(Ordering::Relaxed) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watchdog_reports_only_a_busy_phase_without_progress() {
        let mut last = (u64::MAX, 0);
        assert_eq!(check(&mut last, 7, Phase::Decode as u8, 48, 0, 5), None);
        let mut reports = Vec::new();
        for _ in 0..30 { if let Some(r) = check(&mut last, 7, Phase::Admission as u8, 48, 0, 5) { reports.push(r); } }
        // 150 s silent: reported at 60 s and 120 s, naming the phase.
        assert_eq!(reports, vec![("admission/prefill", 48, 60), ("admission/prefill", 48, 120)]);
        // Progress resets it; an idle loop (blocking on the queue) never reports.
        assert_eq!(check(&mut last, 8, Phase::Admission as u8, 48, 0, 5), None);
        assert_eq!(last, (8, 0));
        let mut idle = (3, 0);
        for _ in 0..30 { assert_eq!(check(&mut idle, 3, Phase::Idle as u8, 0, 0, 5), None); }
    }

    /// W64 F3: the serve loop marked `Idle, 0` while 31 HTTP tickets stayed
    /// open (admitted, never retired). That is a stall too.
    #[test]
    fn idle_with_open_tickets_is_reported() {
        let mut last = (u64::MAX, 0);
        assert_eq!(check(&mut last, 7, Phase::Idle as u8, 0, 31, 5), None);
        let mut reports = Vec::new();
        for _ in 0..30 { if let Some(r) = check(&mut last, 7, Phase::Idle as u8, 0, 31, 5) { reports.push(r); } }
        assert_eq!(reports, vec![("idle with open tickets", 31, 60), ("idle with open tickets", 31, 120)]);
    }

    #[test]
    fn idle_without_open_tickets_never_reports() {
        let mut last = (3, 0);
        for _ in 0..30 { assert_eq!(check(&mut last, 3, Phase::Idle as u8, 0, 0, 5), None); }
        assert_eq!(last, (3, 0));
    }

    #[test]
    fn progress_resets_the_open_ticket_report() {
        let mut last = (u64::MAX, 0);
        assert_eq!(check(&mut last, 7, Phase::Idle as u8, 0, 31, 5), None);
        for _ in 0..11 { assert_eq!(check(&mut last, 7, Phase::Idle as u8, 0, 31, 5), None); }
        assert_eq!(last, (7, 55));
        assert_eq!(check(&mut last, 8, Phase::Idle as u8, 0, 31, 5), None);
        assert_eq!(last, (8, 0));
        let mut reports = Vec::new();
        for _ in 0..12 { if let Some(r) = check(&mut last, 8, Phase::Idle as u8, 0, 31, 5) { reports.push(r); } }
        assert_eq!(reports, vec![("idle with open tickets", 31, 60)]);
    }

    #[test]
    fn retirement_to_idle_and_scoped_shutdown_clear_stale_state() {
        let snapshot = || (PROGRESS.load(Ordering::Relaxed), PHASE.load(Ordering::Relaxed),
            BUSY.load(Ordering::Relaxed));
        let watch = spawn();
        mark(Phase::Retire, 1);
        let (progress, _, _) = snapshot();
        let mut last = (progress, 55);
        // The last request has retired; mark Idle before waiting for a new one.
        mark(Phase::Idle, 0);
        for _ in 0..30 {
            let (progress, phase, busy) = snapshot();
            assert_eq!(check(&mut last, progress, phase, busy, 0, 5), None);
        }
        mark(Phase::Decode, 1);
        drop(watch);
        let (_, phase, busy) = snapshot();
        assert_eq!((phase, busy), (Phase::Idle as u8, 0));
        // The guard must stop/join even on an early error, and a new serve can restart it.
        let lifetime = std::sync::Arc::new(());
        let weak = std::sync::Arc::downgrade(&lifetime);
        let exit = || -> Result<(), ()> {
            let _watch = Watchdog::start(Duration::from_secs(3600), move || { let _ = &lifetime; });
            mark(Phase::Failed, 2);
            Err(())
        };
        assert!(exit().is_err());
        assert!(weak.upgrade().is_none(), "watchdog closure survived serve exit");
        let (_, phase, busy) = snapshot();
        assert_eq!((phase, busy), (Phase::Idle as u8, 0));
        drop(spawn());
    }

    /// Protect the ordering at the real blocking/admission sites without a GPU.
    #[test]
    fn serving_paths_are_diagnostic_only_and_instrument_slow_work() {
        let independent = include_str!("independent.rs");
        assert!(independent.contains("stall::tick();"));
        let scheduler = include_str!("../scheduler.rs");
        let single = &scheduler[scheduler.find("fn single_lane_round<").unwrap()..];
        assert!(single[..4000].contains("clock.draft(|| draft.propose(lib, lane, &seeds))"));
        let admission = &scheduler[scheduler.find("let mut held_slots =").unwrap()..];
        assert!(admission.find("stall::mark(stall::Phase::Admission").unwrap()
            < admission.find("loop {").unwrap());
        let blocking = &admission[admission.find("let job = if busy_slots == 0 && !closed").unwrap()..];
        assert!(blocking.find("stall::mark(stall::Phase::Idle, 0)").unwrap()
            < blocking.find("receive.blocking_recv()").unwrap());
        let resumed = &blocking[blocking.find("Some(job) => {").unwrap()..];
        assert!(resumed.find("stall::mark(stall::Phase::Admission, 1)").unwrap()
            < resumed.find("admission::Prepared::new").unwrap());
        for phase in ["Phase::Retire", "Phase::Admission", "Phase::Decode", "Phase::Failed", "Phase::Idle"] {
            assert!(scheduler.contains(&format!("stall::mark(stall::{phase}")), "{phase} not marked");
        }
        assert!(scheduler.contains("let _stall_watch = stall::spawn();"));
        let packed = &scheduler[scheduler.find("fn prefill_packed_admissions<").unwrap()
            ..scheduler.find("pub(crate) fn exercise_packed_prefill<").unwrap()];
        assert_eq!(packed.matches("stall::tick();").count(), 3);
        let serving = include_str!("../../v41_native_serve.rs");
        let score = &serving[serving.find("fn score<'a,").unwrap()
            ..serving.find("mod admitted_prefill_tests").unwrap()];
        assert_eq!(score.matches("scheduler::stall::tick();").count(), 2);
        assert!(include_str!("../prefill_target.rs").contains("super::scheduler::stall::tick();"));
    }

    /// The fallback keeps the lane live: a failed proposal verifies the anchors
    /// and the next round drafts again, so the lane never stalls on a
    /// draft-side bound.
    #[test]
    fn a_failed_proposal_keeps_the_lane_live_and_the_next_round_drafts_again() {
        let seeds = [(1, 11, 5, 9), (2, 22, 0, 1)];
        let mut attempts = 0;
        let mut rounds = 0;
        let mut fallbacks = 0;
        let mut drafted = 0;
        while rounds < 3 {
            rounds += 1;
            attempts += 1;
            let proposal: anyhow::Result<Vec<Vec<u32>>> = if attempts == 1 {
                Err(anyhow::anyhow!("invalid draft batch size"))
            } else {
                Ok(vec![vec![11, 12, 13], vec![22, 23]])
            };
            match proposal {
                Ok(inputs) => { drafted += 1; assert_eq!(inputs.len(), seeds.len()); }
                Err(error) => {
                    let inputs = draft_fallback(Ok(()), &seeds, 0, &error).unwrap();
                    fallbacks += 1;
                    assert_eq!(inputs, vec![vec![11], vec![22]]);
                }
            }
        }
        // Three rounds completed (no stall), one anchor-only fallback, and the
        // two later rounds drafted normally again.
        assert_eq!((rounds, fallbacks, drafted), (3, 1, 2));
    }

    /// A terminal draft is not a reusable target-only fallback: the liveness
    /// gate propagates the error instead of degrading every round to anchors.
    #[test]
    fn a_terminal_draft_propagates_instead_of_falling_back() {
        let seeds = [(1, 11, 5, 9)];
        let error = anyhow::anyhow!("invalid draft batch size");
        let terminal = Err(anyhow::anyhow!("draft peer wire unavailable until restart"));
        let result = draft_fallback(terminal, &seeds, 0, &error);
        assert!(result.is_err(), "a terminal draft must fail the round");
        assert!(result.unwrap_err().to_string().contains("unavailable until restart"));
    }

    /// A healthy draft path is unchanged: the `Ok` branch still returns the
    /// draft's own inputs, never the anchors, and both lane paths keep it.
    #[test]
    fn a_healthy_draft_path_is_unchanged() {
        let drafted = vec![vec![11, 12, 13], vec![22, 23]];
        let anchors = vec![vec![11], vec![22]];
        assert_ne!(drafted, anchors);
        let independent = include_str!("independent.rs");
        assert!(independent.contains("Ok(proposed) => proposed"));
        assert!(independent.contains("draft.require_live(lane)"));
        let scheduler = include_str!("../scheduler.rs");
        let single = &scheduler[scheduler.find("fn single_lane_round<").unwrap()..];
        assert!(single[..4000].contains("Ok(inputs) => inputs"));
        assert!(single[..4000].contains("draft.require_live(lane)"));
    }
}
