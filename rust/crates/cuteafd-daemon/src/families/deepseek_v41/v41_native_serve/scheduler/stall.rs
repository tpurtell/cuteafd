//! W64-1 (c48 wedge, 2026-10-31 04:23Z): after a failed decode round the serve
//! loop went silent (GPU idle, HTTP queue full, no log line) and the saved log
//! could not say where it waited. Two pieces:
//!
//! * a drafter failure no longer fails the lane round: the round verifies the
//!   anchors only ([`anchor_only`]) and the next round drafts again, so a
//!   draft-side bound can never fail (or wedge) every active request;
//! * a watchdog names the serve-loop phase that stopped making progress
//!   ([`mark`] / [`tick`]), so the next wedge is diagnosable from the log.
use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::time::Duration;

static PROGRESS: AtomicU64 = AtomicU64::new(0);
static PHASE: AtomicU8 = AtomicU8::new(0);
static BUSY: AtomicUsize = AtomicUsize::new(0);
static DRAFT_FALLBACKS: AtomicU64 = AtomicU64::new(0);

/// Serve-loop phases the watchdog reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Phase { Idle = 0, Retire = 1, Admission = 2, PrefillCall = 3, Decode = 4, Failed = 5 }

pub(crate) fn phase_name(phase: u8) -> &'static str {
    match phase { 1 => "retire", 2 => "admission/prefill", 3 => "parked prefill call", 4 => "decode round",
        5 => "failed-round cleanup", _ => "idle" }
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

/// Start the process's watchdog thread once.
pub(crate) fn spawn() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let spawned = std::thread::Builder::new().name("serve-stall-watch".into()).spawn(|| {
            let step = 5;
            let mut last = (u64::MAX, 0);
            loop {
                std::thread::sleep(Duration::from_secs(step));
                let open = crate::shared::console::totals::active();
                if let Some((phase, busy, seconds)) = check(&mut last, PROGRESS.load(Ordering::Relaxed),
                    PHASE.load(Ordering::Relaxed), BUSY.load(Ordering::Relaxed), open, step) {
                    tracing::error!(phase, busy, open, seconds, "serve loop stalled: no round, unit or admission progress");
                }
            }
        });
        if let Err(error) = spawned { tracing::warn!(%error, "serve stall watchdog not started"); }
    });
}

/// A failed draft proposal: the lane verifies each member's anchor alone.
pub(crate) fn anchor_only(seeds: &[(u64, u32, u64, usize)], lane: usize, error: &anyhow::Error) -> Vec<Vec<u32>> {
    let total = DRAFT_FALLBACKS.fetch_add(1, Ordering::Relaxed) + 1;
    tracing::warn!(lane, requests = seeds.len(), total, error = %format!("{error:#}"),
        "dSpark draft proposal failed; this lane round verifies anchors only");
    seeds.iter().map(|seed| vec![seed.1]).collect()
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
    fn a_failed_proposal_verifies_every_anchor() {
        let before = draft_fallbacks();
        let seeds = [(1, 11, 5, 9), (2, 22, 0, 1), (3, 33, 9, 4)];
        let inputs = anchor_only(&seeds, 1, &anyhow::anyhow!("invalid vocabulary projection rows"));
        assert_eq!(inputs, vec![vec![11], vec![22], vec![33]]);
        assert!(draft_fallbacks() > before);
    }

    /// Both lane paths fall back instead of propagating a draft error, and the
    /// serve loop marks every phase.
    #[test]
    fn serving_paths_use_the_fallback_and_the_watchdog() {
        let independent = include_str!("independent.rs");
        assert!(independent.contains("stall::anchor_only(&drafting, lane, &error)"));
        assert!(!independent.contains("draft.poll_propose(lane, &drafting)? }"));
        assert!(independent.contains("stall::tick();"));
        let scheduler = include_str!("../scheduler.rs");
        let single = &scheduler[scheduler.find("fn single_lane_round<").unwrap()..];
        assert!(single[..4000].contains("stall::anchor_only(&seeds, lane, &error)"));
        for phase in ["Phase::Retire", "Phase::Admission", "Phase::Decode", "Phase::Failed",
            "Phase::Idle"] {
            assert!(scheduler.contains(&format!("stall::mark(stall::{phase}")), "{phase} not marked");
        }
        assert!(scheduler.contains("stall::spawn();"));
        assert!(include_str!("../prefill_target.rs").contains("super::scheduler::stall::tick();"));
    }
}
