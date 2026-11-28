//! How host threads wait while the engine serves, and when they park once it
//! is idle (TJ 2026-10-25): one process-wide policy, read at startup from
//! `--host-wait` / `--idle-park-seconds` (`CUTEAFD_HOST_WAIT`,
//! `CUTEAFD_IDLE_PARK_SECONDS`), the same on the coordinator and the Spark
//! workers.
//!
//! - Busy waits ([`HostWait`]): `block` (default) spins for [`BLOCK_SPIN`]
//!   in each critical-path wait (every per-layer decode wait ends within it),
//!   then sleeps: per-step syncs on blocking-sync events, the rest in
//!   [`BLOCK_SLEEP`] slices at 1 us timer slack; the exchange worker spins
//!   through the same window and paces longer waits to each wave's expected
//!   duration. `spin` spins in
//!   host syncs and keeps the exchange worker polling between a wave's post
//!   and its replies, for the lowest latency at one busy core per waiting
//!   thread.
//! - Idle parking, in both modes: once no request has been active for
//!   [`WaitPolicy::idle_park`] (default 30 s), every polling thread blocks
//!   (the engine loop and the exchange workers on their channels, the Spark
//!   workers on their completion events), so an idle server uses no CPU. The
//!   GPU is idle too then: exchange waits exist only while a wave is
//!   outstanding. The first request after parking pays a one-time wake-up.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// How a host thread waits while the engine serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostWait {
    /// Sleep in per-step waits and pace polling: a less loaded machine (default).
    Block,
    /// Spin in host waits and polls: lowest latency, one busy core per waiting thread.
    Spin,
}

impl std::str::FromStr for HostWait {
    type Err = anyhow::Error;
    fn from_str(value: &str) -> anyhow::Result<Self> {
        match value {
            "spin" => Ok(Self::Spin),
            "block" => Ok(Self::Block),
            other => anyhow::bail!("host wait {other:?} is not spin or block"),
        }
    }
}

impl HostWait {
    pub fn name(self) -> &'static str {
        match self {
            Self::Spin => "spin",
            Self::Block => "block",
        }
    }
}

/// The process-wide wait policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaitPolicy {
    pub wait: HostWait,
    /// No active request for this long parks every polling thread.
    pub idle_park: Duration,
}

impl Default for WaitPolicy {
    fn default() -> Self { Self { wait: HostWait::Block, idle_park: Duration::from_secs(30) } }
}

static POLICY: OnceLock<WaitPolicy> = OnceLock::new();

/// Installs the policy once, at startup, and logs it.
pub fn set_wait_policy(policy: WaitPolicy) -> anyhow::Result<()> {
    anyhow::ensure!(!policy.idle_park.is_zero(), "idle park time must be positive");
    POLICY.set(policy).map_err(|_| anyhow::anyhow!("host wait policy was already installed"))?;
    tracing::info!(target: "cuteafd::fabric", host_wait = policy.wait.name(),
        idle_park_s = policy.idle_park.as_secs_f64(), "host wait policy");
    Ok(())
}

/// The installed policy, or the defaults.
pub fn wait_policy() -> WaitPolicy { POLICY.get().copied().unwrap_or_default() }

/// How long a critical-path wait under [`HostWait::Block`] polls before it
/// starts sleeping: the one hybrid spin window for every block-mode wait
/// ([`Backoff`], [`spin_window`], the Spark exchange worker's pacing).
///
/// 50 ms (TJ 2026-11-28), up from 10 ms: the end-of-step token-landing wait
/// must cover a whole decode step. Decode graphs finish host enqueue early,
/// so on Hy4-flash 2x RTX + 4 Sparks C1 the host reached that wait with
/// ~15 ms of GPU/Spark work left, spun the 10 ms window, then slept in
/// `cudaEventSynchronize` on 46 of 50 steps and paid a late, cold wake-up
/// (#2922). Every per-layer decode wait (a router, a Spark reply) still ends
/// well inside the window, so decode never pays a wake-up; only
/// prefill-length waits sleep, and an idle server still parks after
/// [`WaitPolicy::idle_park`].
///
/// Earlier: 10 ms (TJ 2026-10-26), up from 250 us. At 250 us Kimi K3 BF16 max,
/// whose per-layer Spark round is 0.6-0.7 ms, slept in every per-layer wait:
/// C8 basic 0.913 and sweep 0.946 of spin (#731 comment 6415231320).
pub const BLOCK_SPIN: Duration = Duration::from_millis(50);
/// One sleep of a critical-path wait past [`BLOCK_SPIN`].
pub const BLOCK_SLEEP: Duration = Duration::from_micros(20);

/// Whether a block-mode wait that has lasted `waited` still spins
/// (inside [`BLOCK_SPIN`]) rather than sleeps.
pub fn block_spins(waited: Duration) -> bool { waited < BLOCK_SPIN }

/// The spin phase of a block-mode wait: polls `ready` until it reports
/// `true` (returns `true`) or [`BLOCK_SPIN`] has passed (returns `false`; the
/// caller then sleeps until the condition holds, e.g. on a blocking event).
pub fn spin_window<E>(mut ready: impl FnMut() -> Result<bool, E>) -> Result<bool, E> {
    let started = Instant::now();
    loop {
        if ready()? { return Ok(true); }
        if !block_spins(started.elapsed()) { return Ok(false); }
        std::hint::spin_loop();
    }
}

/// A bounded backoff for one host wait on the critical path: under
/// [`HostWait::Spin`] it always spins; under [`HostWait::Block`] it spins for
/// [`BLOCK_SPIN`], then sleeps [`BLOCK_SLEEP`] per pass. The spin restarts on
/// progress: [`Self::reset`], or engine activity ([`note_active`], every Spark
/// wave) since the stretch began, so a backoff spanning many per-layer waits
/// (`serve::lanes::paced` around a whole decode loop) spins through each of
/// them instead of sleeping past every completion once its first 10 ms passed.
pub struct Backoff { block: bool, started: Option<Instant>, active: u64 }

thread_local! {
    /// Pacers polling on this thread right now ([`Paced`]).
    static PACED: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// While alive, an outer pacer owns this thread's waits: it polls a future
/// whose inner waits yield back to it, and sleeps once per pending poll
/// itself. An inner [`Backoff`] then only spins, so a pending poll pays one
/// [`BLOCK_SLEEP`], not one per nested backoff.
pub struct Paced(());

impl Paced {
    pub fn enter() -> Self {
        PACED.with(|depth| depth.set(depth.get() + 1));
        Self(())
    }
}

impl Drop for Paced {
    fn drop(&mut self) { PACED.with(|depth| depth.set(depth.get() - 1)); }
}

fn paced_here() -> bool { PACED.with(|depth| depth.get() > 0) }

impl Backoff {
    pub fn new() -> Self {
        Self { block: wait_policy().wait == HostWait::Block, started: None, active: 0 }
    }
    /// One pass of the wait: spin, or sleep once the spin budget is spent
    /// (unless an outer pacer sleeps for this poll, [`Paced`]).
    pub fn pause(&mut self) {
        if self.sleeps() && !paced_here() { std::thread::sleep(BLOCK_SLEEP) } else { std::hint::spin_loop() }
    }
    /// Whether this pass sleeps (block mode, past [`BLOCK_SPIN`] since the
    /// first pause after the last progress).
    fn sleeps(&mut self) -> bool {
        let active = LAST_ACTIVE.load(Ordering::Relaxed);
        if active != self.active { self.active = active; self.started = None; }
        self.block && !block_spins(self.started.get_or_insert_with(Instant::now).elapsed())
    }
    /// The wait made progress: the next pause spins again.
    pub fn reset(&mut self) { self.started = None; }
}

impl Default for Backoff {
    fn default() -> Self { Self::new() }
}

/// Monotonic nanoseconds since the process's activity epoch.
fn now_ns() -> u64 {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

/// When a request was last active (ns since the epoch); 0: never.
static LAST_ACTIVE: AtomicU64 = AtomicU64::new(0);

/// Records that a request is active now (admission, every step, every wave).
pub fn note_active() { LAST_ACTIVE.store(now_ns().max(1), Ordering::Relaxed); }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_wait_parses_its_two_modes_only() {
        assert_eq!("spin".parse::<HostWait>().unwrap(), HostWait::Spin);
        assert_eq!("block".parse::<HostWait>().unwrap(), HostWait::Block);
        assert!("yield".parse::<HostWait>().is_err());
        assert_eq!(WaitPolicy::default(), WaitPolicy { wait: HostWait::Block, idle_park: Duration::from_secs(30) });
    }

    #[test]
    fn block_spin_window_is_fifty_ms_and_covers_a_decode_step() {
        assert_eq!(BLOCK_SPIN, Duration::from_millis(50));
        // Kimi K3's per-layer Spark round (0.6-0.7 ms) and V4.1's must spin,
        // and so must a whole decode step's token-landing wait (~15 ms on
        // Hy4-flash 2x RTX + 4 Sparks C1, #2922).
        assert!(block_spins(Duration::ZERO));
        assert!(block_spins(Duration::from_micros(700)));
        assert!(block_spins(Duration::from_millis(15)));
        assert!(block_spins(Duration::from_micros(49_999)));
        assert!(!block_spins(BLOCK_SPIN));
        assert!(!block_spins(Duration::from_millis(100)));
        // Parking stays separate and unchanged.
        assert_eq!(WaitPolicy::default().idle_park, Duration::from_secs(30));
    }

    #[test]
    fn spin_window_returns_when_ready_and_gives_up_after_the_window() {
        // Ready after 1 ms of polling: caught inside the window, no sleep.
        let started = Instant::now();
        let ready = spin_window(|| Ok::<_, ()>(started.elapsed() >= Duration::from_millis(1))).unwrap();
        assert!(ready, "a sub-window wait completes while spinning");
        assert!(started.elapsed() < BLOCK_SPIN);
        // Never ready: spins the whole window, then hands back to the caller.
        let started = Instant::now();
        assert!(!spin_window(|| Ok::<_, ()>(false)).unwrap());
        assert!(started.elapsed() >= BLOCK_SPIN, "spins the full window before blocking");
        // Errors from the poll propagate at once.
        assert_eq!(spin_window(|| Err::<bool, _>("poll failed")), Err("poll failed"));
    }

    #[test]
    fn backoff_spins_through_the_window_then_sleeps_under_block() {
        let mut backoff = Backoff { block: true, started: None, active: LAST_ACTIVE.load(Ordering::Relaxed) };
        assert!(!backoff.sleeps(), "the first pause spins");
        // A Kimi K3 per-layer round (0.6-0.7 ms) in: still spinning.
        backoff.started = Instant::now().checked_sub(Duration::from_micros(700));
        assert!(!backoff.sleeps(), "a per-layer wait never sleeps");
        backoff.started = Instant::now().checked_sub(BLOCK_SPIN);
        assert!(backoff.sleeps(), "past the spin budget it sleeps");
        let late = Instant::now();
        backoff.pause();
        assert!(late.elapsed() >= BLOCK_SLEEP, "a sleeping pause sleeps");
        backoff.reset();
        assert!(!backoff.sleeps(), "progress restarts the spin");
        let mut spin = Backoff { block: false, started: Instant::now().checked_sub(Duration::from_secs(1)), active: 0 };
        assert!(!spin.sleeps(), "spin mode never sleeps");
    }

    /// A backoff spanning a whole decode loop (`serve::lanes::paced`): engine
    /// activity (every Spark wave notes it) restarts the spin, so each
    /// per-layer wait spins instead of sleeping past its completion once the
    /// loop's first 10 ms passed (V4.1 2-RTX max C1: +26 us per eager layer).
    #[test]
    fn engine_activity_restarts_the_spin() {
        // A stretch past its budget, begun before the latest recorded activity
        // (`active` holds an older reading than LAST_ACTIVE, which never equals
        // u64::MAX): the activity restarts the spin. The global is only read.
        let mut backoff = Backoff { block: true, started: Instant::now().checked_sub(BLOCK_SPIN), active: u64::MAX };
        assert!(!backoff.sleeps(), "activity since the stretch began restarts the spin");
        assert!(!backoff.sleeps(), "and the restarted stretch spins within its window");
        // With no new activity the same stretch sleeps once past its budget.
        backoff.started = Instant::now().checked_sub(BLOCK_SPIN);
        assert!(backoff.sleeps(), "no activity: past the budget it sleeps");
    }

    /// Bounded spin (AGENTS.md: no idle CPU burn): with no engine activity a
    /// stretch still reaches the sleep stage after BLOCK_SPIN and stays there;
    /// the waiting loop's own pauses never record activity, so an idle lane
    /// cannot keep itself spinning. Idle parking reads the same clock.
    #[test]
    fn an_idle_wait_still_sleeps_and_pausing_never_records_activity() {
        let before = LAST_ACTIVE.load(Ordering::Relaxed);
        let mut backoff = Backoff { block: true, started: None, active: before };
        assert!(!backoff.sleeps(), "the first pause spins");
        backoff.started = Instant::now().checked_sub(BLOCK_SPIN);
        // Another test may record activity concurrently; only assert the
        // sleep stage when the clock did not move under us.
        let moved = || LAST_ACTIVE.load(Ordering::Relaxed) != before;
        for _ in 0..3 {
            let slept = Instant::now();
            backoff.pause();
            if !moved() { assert!(slept.elapsed() >= BLOCK_SLEEP, "past the budget an idle wait sleeps"); }
        }
        if !moved() {
            assert_eq!(LAST_ACTIVE.load(Ordering::Relaxed), before, "pausing never records activity");
            assert!(backoff.sleeps(), "an idle stretch keeps sleeping");
        }
    }

    #[test]
    fn activity_moves_the_clock_backoffs_read() {
        let before = LAST_ACTIVE.load(Ordering::Relaxed);
        std::thread::sleep(Duration::from_micros(10));
        note_active();
        assert!(LAST_ACTIVE.load(Ordering::Relaxed) > before);
    }

    /// A wait nested in a pacer (an Engram or reply wait inside the lane
    /// loop's `paced`) yields its pending poll back to the pacer, which sleeps
    /// once; past its own budget the inner backoff only spins. Outside any
    /// pacer (prefill's plain `block_on`) it still sleeps.
    #[test]
    fn a_wait_inside_a_pacer_leaves_the_sleep_to_it() {
        let past_budget = || Backoff { block: true, started: Instant::now().checked_sub(BLOCK_SPIN),
            active: LAST_ACTIVE.load(Ordering::Relaxed) };
        let fast = |backoff: &mut Backoff| {
            // Another test may record activity concurrently (the inner then
            // spins anyway); only time the passes where it did not.
            let started = Instant::now();
            for _ in 0..50 { backoff.started = Instant::now().checked_sub(BLOCK_SPIN); backoff.pause(); }
            started.elapsed()
        };
        {
            let _outer = Paced::enter();
            let mut inner = past_budget();
            assert!(fast(&mut inner) < BLOCK_SLEEP * 50, "a paced inner wait never sleeps");
            {
                let _nested = Paced::enter();
                assert!(paced_here());
            }
            assert!(paced_here(), "leaving a nested pacer keeps the outer one");
        }
        assert!(!paced_here());
        let mut alone = past_budget();
        let before = LAST_ACTIVE.load(Ordering::Relaxed);
        let slept = fast(&mut alone);
        if LAST_ACTIVE.load(Ordering::Relaxed) == before {
            assert!(slept >= BLOCK_SLEEP * 50, "outside a pacer the wait sleeps");
        }
    }
}
