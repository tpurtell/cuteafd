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
/// 10 ms (TJ 2026-10-26), up from 250 us: every per-layer decode wait (a
/// router, a Spark reply) ends within it, so decode never pays a sleep's
/// wake-up; only prefill-length waits sleep, and an idle server still parks
/// after [`WaitPolicy::idle_park`]. At 250 us Kimi K3 BF16 max, whose
/// per-layer Spark round is 0.6-0.7 ms, slept in every per-layer wait: C8
/// basic 0.913 and sweep 0.946 of spin, and the timed Spark compute for
/// identical kernels grew 33.6 -> 38.9 ms (#731 comment 6415231320).
pub const BLOCK_SPIN: Duration = Duration::from_millis(10);
/// One sleep of a critical-path wait past [`BLOCK_SPIN`].
pub const BLOCK_SLEEP: Duration = Duration::from_micros(20);

/// Whether a block-mode wait that has lasted `waited` still spins
/// (inside [`BLOCK_SPIN`]) rather than sleeps.
pub fn block_spins(waited: Duration) -> bool { waited < BLOCK_SPIN }

/// Polls `ready` in a spin loop until it returns `true` or `limit` elapses on `now`.
fn spin_for<E>(
    mut now: impl FnMut() -> Duration,
    limit: Duration,
    mut ready: impl FnMut() -> Result<bool, E>,
) -> Result<bool, E> {
    let started = now();
    loop {
        if ready()? { return Ok(true); }
        if now().saturating_sub(started) >= limit { return Ok(false); }
        std::hint::spin_loop();
    }
}

/// The spin phase of a block-mode wait: polls `ready` until it reports
/// `true` (returns `true`) or [`BLOCK_SPIN`] has passed (returns `false`; the
/// caller then sleeps until the condition holds, e.g. on a blocking event).
pub fn spin_window<E>(ready: impl FnMut() -> Result<bool, E>) -> Result<bool, E> {
    spin_for(epoch_elapsed, BLOCK_SPIN, ready)
}

/// Spin budget while uncalibrated or re-stabilising under [`HostWait::Block`].
/// Much longer than [`BLOCK_SPIN`] (10 ms) so decode steps across C1..C16 and
/// long-context bands finish in spin without falling off a timeout cliff into
/// a late blocking wake, while large prefills and stalls still fall back to
/// blocking.
pub const PACE_STABILISE_SPIN: Duration = Duration::from_millis(100);
/// Positive-jitter spin headroom past expected completion after waking from
/// paced sleep, before falling back to blocking. Prevents a step that runs
/// moderately long from dropping off a 10 ms cliff into a blocking wake.
pub const PACE_POST_SLEEP_HEADROOM: Duration = Duration::from_millis(25);
/// Minimum lead time before expected completion when waking from paced sleep.
/// Brings the core out of C-states and back to boost clocks before completion
/// (well above the ~336 us measured wake + post-wake penalty).
pub const PACE_MIN_SPIN_TAIL: Duration = Duration::from_millis(2);
/// Upper bound on the adaptive spin tail (`PACE_MIN_SPIN_TAIL + 2 * dev`).
pub const PACE_MAX_SPIN_TAIL: Duration = Duration::from_millis(8);
/// Minimum sleep budget worth entering sleep for; shorter waits stay pure spin.
pub const PACE_MIN_SLEEP: Duration = Duration::from_millis(2);
/// Maximum single sleep slice inside the paced sleep prefix. Bounds how late
/// an abrupt drop in wait duration can be noticed to well within the spin tail.
pub const PACE_SLEEP_SLICE: Duration = Duration::from_micros(500);
/// Upper bound on paced wait durations. Longer waits (large prefills or stalls)
/// stay unpaced and fall back to blocking after [`PACE_STABILISE_SPIN`].
pub const PACE_MAX_WINDOW: Duration = Duration::from_millis(250);
/// Consecutive consistent waits required before entering the paced state.
pub const PACE_STABLE_ROUNDS: u32 = 3;
/// Idle gap between waits that resets pacing back to stabilising.
pub const PACE_IDLE_RESET: Duration = Duration::from_millis(500);

/// Counters recorded by [`WaitPace`] (per instance and optional shared bank).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct WaitPaceStats {
    /// Total waits completed under [`HostWait::Block`].
    pub waits: u64,
    /// Waits that slept for at least one paced slice before spinning.
    pub paced_sleeps: u64,
    /// Paced sleeps where `ready` was already true when a sleep slice ended (late wake).
    pub late_wakes: u64,
    /// Waits that exceeded the spin budget and invoked the blocking fallback.
    pub fallback_blocks: u64,
    /// Transitions from paced back to stabilising after a mis-estimate.
    pub restabilises: u64,
    /// Total nanoseconds spent inside paced sleep slices.
    pub sleep_ns: u64,
    /// Total nanoseconds spent actively spinning.
    pub spin_ns: u64,
    /// Sum of the final sleep-slice durations on late wakes (upper bound on
    /// wake latency past completion; hot-spin wakes notice completion within
    /// one poll and contribute 0).
    pub late_wake_slice_ns_sum: u64,
    /// Maximum final sleep-slice duration observed on a late wake.
    pub late_wake_slice_ns_max: u64,
}

/// Lock-free atomic counter bank for exporting [`WaitPace`] statistics (e.g. `/v1/stats`).
#[derive(Debug, Default)]
pub struct WaitPaceCounters {
    waits: AtomicU64,
    paced_sleeps: AtomicU64,
    late_wakes: AtomicU64,
    fallback_blocks: AtomicU64,
    restabilises: AtomicU64,
    sleep_ns: AtomicU64,
    spin_ns: AtomicU64,
    late_wake_slice_ns_sum: AtomicU64,
    late_wake_slice_ns_max: AtomicU64,
}

impl WaitPaceCounters {
    pub const fn new() -> Self {
        const ZERO: AtomicU64 = AtomicU64::new(0);
        Self {
            waits: ZERO,
            paced_sleeps: ZERO,
            late_wakes: ZERO,
            fallback_blocks: ZERO,
            restabilises: ZERO,
            sleep_ns: ZERO,
            spin_ns: ZERO,
            late_wake_slice_ns_sum: ZERO,
            late_wake_slice_ns_max: ZERO,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn record(
        &self,
        slept: bool,
        late_wake: bool,
        fallback_block: bool,
        restabilised: bool,
        sleep_ns: u64,
        spin_ns: u64,
        late_slice_ns: u64,
    ) {
        self.waits.fetch_add(1, Ordering::Relaxed);
        if slept { self.paced_sleeps.fetch_add(1, Ordering::Relaxed); }
        if late_wake {
            self.late_wakes.fetch_add(1, Ordering::Relaxed);
            self.late_wake_slice_ns_sum.fetch_add(late_slice_ns, Ordering::Relaxed);
            self.late_wake_slice_ns_max.fetch_max(late_slice_ns, Ordering::Relaxed);
        }
        if fallback_block { self.fallback_blocks.fetch_add(1, Ordering::Relaxed); }
        if restabilised { self.restabilises.fetch_add(1, Ordering::Relaxed); }
        if sleep_ns > 0 { self.sleep_ns.fetch_add(sleep_ns, Ordering::Relaxed); }
        if spin_ns > 0 { self.spin_ns.fetch_add(spin_ns, Ordering::Relaxed); }
    }

    pub fn snapshot(&self) -> WaitPaceStats {
        let get = |c: &AtomicU64| c.load(Ordering::Relaxed);
        WaitPaceStats {
            waits: get(&self.waits),
            paced_sleeps: get(&self.paced_sleeps),
            late_wakes: get(&self.late_wakes),
            fallback_blocks: get(&self.fallback_blocks),
            restabilises: get(&self.restabilises),
            sleep_ns: get(&self.sleep_ns),
            spin_ns: get(&self.spin_ns),
            late_wake_slice_ns_sum: get(&self.late_wake_slice_ns_sum),
            late_wake_slice_ns_max: get(&self.late_wake_slice_ns_max),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PacePhase {
    /// Default / re-stabilising: spin up to [`PACE_STABILISE_SPIN`], then `block`.
    Stabilising,
    /// Steady state: sleep up to `sleep_budget` in [`PACE_SLEEP_SLICE`] slices,
    /// then hot-spin across completion before falling back to `block`.
    Paced,
}

#[derive(Debug, Clone, Copy)]
enum WaitOutcome {
    Immediate,
    LateWake { slept_ns: u64, last_slice_ns: u64 },
    Spun { slept_ns: u64, spin_ns: u64 },
    Blocked { slept_ns: u64, spin_ns: u64 },
}

fn ema(current: Duration, sample: Duration, denom: u64) -> Duration {
    let cur = current.as_nanos();
    let sam = sample.as_nanos();
    let d = u128::from(denom);
    let next = (cur * (d - 1) + sam + d / 2) / d;
    Duration::from_nanos(next.min(u128::from(u64::MAX)) as u64)
}

/// Adaptive spin-first, paced-sleep-then-spin schedule for recurring host waits
/// under [`HostWait::Block`].
///
/// State is per wait site and per lane (never global):
/// - **Default (stabilising):** spins up to [`PACE_STABILISE_SPIN`] (100 ms)
///   and falls back to `block` for longer waits. Once [`PACE_STABLE_ROUNDS`]
///   consecutive waits agree within tolerance, transitions to paced mode.
/// - **Paced (steady state):** sleeps in bounded [`PACE_SLEEP_SLICE`] slices up
///   to `mean - spin_tail`, wakes early enough to spin hot across completion
///   without OS wake-up lag, and falls back to `block` only if spinning exceeds
///   `spin_tail + PACE_POST_SLEEP_HEADROOM` (capped at [`PACE_STABILISE_SPIN`]).
/// - **Re-stabilisation:** if an event is already ready when a sleep slice ends
///   (late wake) or finishes much later than predicted, switches back to
///   stabilising so the next few rounds spin and re-measure cleanly.
/// - Under [`HostWait::Spin`], spins directly until `ready` without sleeping.
#[derive(Debug, Clone, Copy)]
pub struct WaitPace {
    phase: PacePhase,
    mean: Option<Duration>,
    dev: Duration,
    stable_streak: u32,
    last_done: Option<Duration>,
    stats: WaitPaceStats,
    counters: Option<&'static WaitPaceCounters>,
}

impl Default for WaitPace {
    fn default() -> Self { Self::new() }
}

impl WaitPace {
    pub const fn new() -> Self {
        Self {
            phase: PacePhase::Stabilising,
            mean: None,
            dev: Duration::ZERO,
            stable_streak: 0,
            last_done: None,
            stats: WaitPaceStats {
                waits: 0,
                paced_sleeps: 0,
                late_wakes: 0,
                fallback_blocks: 0,
                restabilises: 0,
                sleep_ns: 0,
                spin_ns: 0,
                late_wake_slice_ns_sum: 0,
                late_wake_slice_ns_max: 0,
            },
            counters: None,
        }
    }

    pub const fn with_counters(counters: &'static WaitPaceCounters) -> Self {
        Self { counters: Some(counters), ..Self::new() }
    }

    pub const fn is_paced(&self) -> bool { matches!(self.phase, PacePhase::Paced) }

    pub const fn stats(&self) -> WaitPaceStats { self.stats }

    pub fn mean(&self) -> Option<Duration> { self.mean }

    /// Lead time before expected completion when paced sleep ends and hot spinning begins.
    pub fn spin_tail(&self) -> Duration {
        PACE_MIN_SPIN_TAIL
            .saturating_add(self.dev.saturating_mul(2))
            .min(PACE_MAX_SPIN_TAIL)
    }

    /// Planned sleep prefix for the next wait (zero while stabilising, for short
    /// waits below `spin_tail + PACE_MIN_SLEEP`, or above [`PACE_MAX_WINDOW`]).
    pub fn sleep_budget(&self) -> Duration {
        if self.phase != PacePhase::Paced {
            return Duration::ZERO;
        }
        let Some(mean) = self.mean else { return Duration::ZERO };
        if mean > PACE_MAX_WINDOW {
            return Duration::ZERO;
        }
        let budget = mean.saturating_sub(self.spin_tail());
        if budget < PACE_MIN_SLEEP { Duration::ZERO } else { budget }
    }

    fn reset_stabilising(&mut self) {
        self.phase = PacePhase::Stabilising;
        self.mean = None;
        self.dev = Duration::ZERO;
        self.stable_streak = 0;
    }

    fn seed_stabilising(&mut self, sample: Duration, keep_dev: Option<Duration>) {
        self.phase = PacePhase::Stabilising;
        if sample > PACE_MAX_WINDOW {
            self.mean = None;
            self.dev = Duration::ZERO;
            self.stable_streak = 0;
        } else {
            self.mean = Some(sample);
            self.dev = keep_dev.unwrap_or(Duration::ZERO);
            self.stable_streak = 1;
        }
    }

    fn record_stabilising(&mut self, sample: Duration) {
        if sample > PACE_MAX_WINDOW {
            self.reset_stabilising();
            return;
        }
        match self.mean {
            None => {
                self.mean = Some(sample);
                self.dev = Duration::ZERO;
                self.stable_streak = 1;
            }
            Some(mean) => {
                let diff = sample.abs_diff(mean);
                let tol = (mean / 4).max(PACE_MIN_SPIN_TAIL);
                if diff <= tol {
                    self.mean = Some(ema(mean, sample, 4));
                    self.dev = if diff > self.dev { ema(self.dev, diff, 2) } else { ema(self.dev, diff, 8) };
                    self.stable_streak += 1;
                    if self.stable_streak >= PACE_STABLE_ROUNDS {
                        self.phase = PacePhase::Paced;
                    }
                } else {
                    self.seed_stabilising(sample, None);
                }
            }
        }
    }

    fn finish(&mut self, elapsed: Duration, done: Duration, outcome: WaitOutcome) {
        self.last_done = Some(done);
        let (slept, late_wake, fallback_block, slept_ns, spin_ns, last_slice_ns) = match outcome {
            WaitOutcome::Immediate => (false, false, false, 0, 0, 0),
            WaitOutcome::LateWake { slept_ns, last_slice_ns } => (true, true, false, slept_ns, 0, last_slice_ns),
            WaitOutcome::Spun { slept_ns, spin_ns } => (slept_ns > 0, false, false, slept_ns, spin_ns, 0),
            WaitOutcome::Blocked { slept_ns, spin_ns } => (slept_ns > 0, false, true, slept_ns, spin_ns, 0),
        };
        let mut restabilised = false;
        match outcome {
            WaitOutcome::LateWake { .. } => {
                // Event was already done when sleep ended: preserve widened jitter
                // if this was a near-miss in the same regime so re-entry won't repeat it.
                let keep_dev = self.mean.and_then(|mean| {
                    let diff = elapsed.abs_diff(mean);
                    let tol = (mean / 4).max(PACE_MIN_SPIN_TAIL);
                    (diff <= tol).then(|| self.dev.max(diff))
                });
                restabilised = true;
                self.seed_stabilising(elapsed, keep_dev);
            }
            WaitOutcome::Blocked { .. } => {
                if self.phase == PacePhase::Paced {
                    restabilised = true;
                    self.seed_stabilising(elapsed, None);
                } else {
                    self.record_stabilising(elapsed);
                }
            }
            WaitOutcome::Spun { slept_ns, .. } => {
                if self.phase == PacePhase::Paced {
                    let mean = self.mean.unwrap_or(elapsed);
                    let late_tol = self.spin_tail().max(mean / 4);
                    let overran = elapsed > mean.saturating_add(late_tol)
                        || (slept_ns == 0 && elapsed.abs_diff(mean) > late_tol);
                    if overran {
                        restabilised = true;
                        self.seed_stabilising(elapsed, None);
                    } else {
                        let diff = elapsed.abs_diff(mean);
                        self.mean = Some(ema(mean, elapsed, 4));
                        self.dev = if diff > self.dev { ema(self.dev, diff, 2) } else { ema(self.dev, diff, 8) };
                    }
                } else {
                    self.record_stabilising(elapsed);
                }
            }
            WaitOutcome::Immediate => {
                if self.phase == PacePhase::Paced {
                    if !self.sleep_budget().is_zero() {
                        restabilised = true;
                        self.seed_stabilising(Duration::ZERO, None);
                    } else if let Some(mean) = self.mean {
                        self.mean = Some(ema(mean, Duration::ZERO, 4));
                        self.dev = ema(self.dev, mean, 8);
                    }
                } else {
                    self.record_stabilising(Duration::ZERO);
                }
            }
        }
        self.stats.waits += 1;
        if slept { self.stats.paced_sleeps += 1; }
        if late_wake {
            self.stats.late_wakes += 1;
            self.stats.late_wake_slice_ns_sum = self.stats.late_wake_slice_ns_sum.saturating_add(last_slice_ns);
            self.stats.late_wake_slice_ns_max = self.stats.late_wake_slice_ns_max.max(last_slice_ns);
        }
        if fallback_block { self.stats.fallback_blocks += 1; }
        if restabilised { self.stats.restabilises += 1; }
        self.stats.sleep_ns = self.stats.sleep_ns.saturating_add(slept_ns);
        self.stats.spin_ns = self.stats.spin_ns.saturating_add(spin_ns);
        if let Some(counters) = self.counters {
            counters.record(slept, late_wake, fallback_block, restabilised, slept_ns, spin_ns, last_slice_ns);
        }
    }

    /// Waits for `ready` under the installed [`wait_policy`].
    pub fn wait<E>(
        &mut self,
        ready: impl FnMut() -> Result<bool, E>,
        block: impl FnOnce() -> Result<(), E>,
    ) -> Result<(), E> {
        self.wait_for(wait_policy().wait, ready, block)
    }

    /// Waits for `ready` under an explicit [`HostWait`] mode.
    pub fn wait_for<E>(
        &mut self,
        mode: HostWait,
        ready: impl FnMut() -> Result<bool, E>,
        block: impl FnOnce() -> Result<(), E>,
    ) -> Result<(), E> {
        self.wait_with(mode, epoch_elapsed, std::thread::sleep, ready, block)
    }

    /// Injectable clock/sleep implementation for deterministic testing and reuse.
    pub fn wait_with<E>(
        &mut self,
        mode: HostWait,
        mut now: impl FnMut() -> Duration,
        mut sleep: impl FnMut(Duration),
        mut ready: impl FnMut() -> Result<bool, E>,
        block: impl FnOnce() -> Result<(), E>,
    ) -> Result<(), E> {
        if mode == HostWait::Spin {
            while !ready()? {
                std::hint::spin_loop();
            }
            return Ok(());
        }
        let started = now();
        if let Some(last) = self.last_done {
            if started.saturating_sub(last) >= PACE_IDLE_RESET {
                self.reset_stabilising();
            }
        }
        if ready()? {
            let done = now();
            self.finish(Duration::ZERO, done, WaitOutcome::Immediate);
            return Ok(());
        }
        let sleep_budget = self.sleep_budget();
        let mut slept_ns = 0u64;
        let mut cur = started;
        if !sleep_budget.is_zero() {
            loop {
                let elapsed = cur.saturating_sub(started);
                let Some(remaining) = sleep_budget.checked_sub(elapsed) else { break };
                if remaining.is_zero() { break; }
                sleep(remaining.min(PACE_SLEEP_SLICE));
                let after = now();
                let slice_ns = after.saturating_sub(cur).as_nanos() as u64;
                slept_ns = slept_ns.saturating_add(slice_ns);
                cur = after;
                if ready()? {
                    let elapsed = cur.saturating_sub(started);
                    self.finish(elapsed, cur, WaitOutcome::LateWake { slept_ns, last_slice_ns: slice_ns });
                    return Ok(());
                }
            }
        }
        let spin_limit = if slept_ns > 0 {
            self.spin_tail().saturating_add(PACE_POST_SLEEP_HEADROOM).min(PACE_STABILISE_SPIN)
        } else {
            PACE_STABILISE_SPIN
        };
        let spin_started = cur;
        if spin_for(&mut now, spin_limit, &mut ready)? {
            let done = now();
            let elapsed = done.saturating_sub(started);
            let spin_ns = done.saturating_sub(spin_started).as_nanos() as u64;
            self.finish(elapsed, done, WaitOutcome::Spun { slept_ns, spin_ns });
            return Ok(());
        }
        let spin_ns = now().saturating_sub(spin_started).as_nanos() as u64;
        block()?;
        let done = now();
        let elapsed = done.saturating_sub(started);
        self.finish(elapsed, done, WaitOutcome::Blocked { slept_ns, spin_ns });
        Ok(())
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

/// Monotonic duration since the process's activity epoch.
fn epoch_elapsed() -> Duration {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    EPOCH.get_or_init(Instant::now).elapsed()
}

/// Monotonic nanoseconds since the process's activity epoch.
fn now_ns() -> u64 { epoch_elapsed().as_nanos() as u64 }

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
    fn block_spin_window_is_ten_ms_and_covers_a_kimi_layer_round() {
        assert_eq!(BLOCK_SPIN, Duration::from_millis(10));
        // Kimi K3's per-layer Spark round (0.6-0.7 ms) and V4.1's must spin.
        assert!(block_spins(Duration::ZERO));
        assert!(block_spins(Duration::from_micros(700)));
        assert!(block_spins(Duration::from_micros(9_999)));
        assert!(!block_spins(BLOCK_SPIN));
        assert!(!block_spins(Duration::from_millis(50)));
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

    struct FakeClock {
        now: std::cell::Cell<Duration>,
    }

    impl FakeClock {
        fn new() -> Self { Self { now: std::cell::Cell::new(Duration::ZERO) } }
        fn get(&self) -> Duration { self.now.get() }
        fn advance(&self, delta: Duration) { self.now.set(self.now.get() + delta); }
    }

    struct SimRound {
        slept: Duration,
        sleep_calls: usize,
        polls: usize,
        blocked: bool,
        last_sleep_end: Option<Duration>,
        completion: Duration,
    }

    fn run_sim_round(pace: &mut WaitPace, clock: &FakeClock, mode: HostWait, duration: Duration) -> SimRound {
        let start = clock.get();
        let completion = start + duration;
        let mut slept = Duration::ZERO;
        let mut sleep_calls = 0usize;
        let mut polls = 0usize;
        let mut blocked = false;
        let last_sleep_end = std::cell::Cell::new(None);
        let just_slept = std::cell::Cell::new(false);
        let poll_step = Duration::from_micros(50);
        pace.wait_with(
            mode,
            || clock.get(),
            |d| {
                clock.advance(d);
                slept += d;
                sleep_calls += 1;
                last_sleep_end.set(Some(clock.get()));
                just_slept.set(true);
            },
            || {
                polls += 1;
                if clock.get() >= completion {
                    just_slept.set(false);
                    return Ok::<_, ()>(true);
                }
                // On the entry poll or the immediate post-sleep check, inspect the current
                // clock without advancing; on spin polls advance by `poll_step`.
                if polls > 1 && !just_slept.replace(false) {
                    let next = (clock.get() + poll_step).min(completion);
                    clock.now.set(next);
                }
                Ok(clock.get() >= completion)
            },
            || {
                blocked = true;
                if clock.get() < completion {
                    clock.now.set(completion);
                }
                Ok(())
            },
        )
        .unwrap();
        SimRound { slept, sleep_calls, polls, blocked, last_sleep_end: last_sleep_end.get(), completion }
    }

    #[test]
    fn wait_pace_converges_on_stable_duration() {
        let clock = FakeClock::new();
        let mut pace = WaitPace::new();
        let step = Duration::from_millis(15);

        // First PACE_STABLE_ROUNDS rounds spin-first without sleeping or blocking.
        for round in 0..PACE_STABLE_ROUNDS {
            assert!(!pace.is_paced(), "round {round} must stabilise in spin before pacing");
            let r = run_sim_round(&mut pace, &clock, HostWait::Block, step);
            assert_eq!(r.slept, Duration::ZERO);
            assert!(!r.blocked, "15 ms wait fits inside PACE_STABILISE_SPIN");
            clock.advance(Duration::from_millis(2)); // inter-step host work
        }
        assert!(pace.is_paced(), "converges to paced mode after PACE_STABLE_ROUNDS");
        let budget = pace.sleep_budget();
        assert_eq!(budget, step - PACE_MIN_SPIN_TAIL);

        // Steady state: sleeps for ~13 ms, wakes hot before completion, 0 late wakes.
        for _ in 0..10 {
            let r = run_sim_round(&mut pace, &clock, HostWait::Block, step);
            assert_eq!(r.slept, budget);
            assert!(!r.blocked);
            assert!(r.last_sleep_end.unwrap() < r.completion);
            clock.advance(Duration::from_millis(2));
        }
        let stats = pace.stats();
        assert_eq!(stats.waits, u64::from(PACE_STABLE_ROUNDS) + 10);
        assert_eq!(stats.paced_sleeps, 10);
        assert_eq!(stats.late_wakes, 0);
        assert_eq!(stats.fallback_blocks, 0);
        assert_eq!(stats.restabilises, 0);

        // Longer steady decode waits (> PACE_STABILISE_SPIN, e.g. 140 ms) also
        // converge after PACE_STABLE_ROUNDS and stop hitting the blocking fallback.
        let mut long_pace = WaitPace::new();
        let long_step = Duration::from_millis(140);
        for _ in 0..PACE_STABLE_ROUNDS {
            let r = run_sim_round(&mut long_pace, &clock, HostWait::Block, long_step);
            assert!(r.blocked, "140 ms stabilising wait falls back to block after PACE_STABILISE_SPIN");
            clock.advance(Duration::from_millis(2));
        }
        assert!(long_pace.is_paced());
        for _ in 0..5 {
            let r = run_sim_round(&mut long_pace, &clock, HostWait::Block, long_step);
            assert!(!r.blocked, "paced 140 ms wait catches completion in hot spin without blocking");
            assert_eq!(r.slept, long_step - PACE_MIN_SPIN_TAIL);
            clock.advance(Duration::from_millis(2));
        }
        assert_eq!(long_pace.stats().fallback_blocks, u64::from(PACE_STABLE_ROUNDS));
        assert_eq!(long_pace.stats().late_wakes, 0);
        assert_eq!(long_pace.stats().late_wake_slice_ns_max, 0);
    }

    #[test]
    fn wait_pace_restabilises_after_misestimate() {
        let clock = FakeClock::new();
        let mut pace = WaitPace::new();

        // Converge on 20 ms.
        for _ in 0..PACE_STABLE_ROUNDS {
            run_sim_round(&mut pace, &clock, HostWait::Block, Duration::from_millis(20));
            clock.advance(Duration::from_millis(2));
        }
        assert!(pace.is_paced());

        // 1) Abrupt drop (20 ms -> 6 ms): event finishes during sleep -> late wake,
        // drops back to stabilising, then re-converges on 6 ms with no further late wakes.
        let drop_r = run_sim_round(&mut pace, &clock, HostWait::Block, Duration::from_millis(6));
        assert!(!drop_r.blocked);
        assert!(!pace.is_paced(), "early finish during sleep must switch back to stabilising");
        assert_eq!(pace.stats().late_wakes, 1);
        assert_eq!(pace.stats().restabilises, 1);
        assert!(pace.stats().late_wake_slice_ns_max <= PACE_SLEEP_SLICE.as_nanos() as u64);
        clock.advance(Duration::from_millis(2));

        // Next 2 rounds confirm 6 ms in spin (seed from drop_r was streak=1).
        for _ in 0..(PACE_STABLE_ROUNDS - 1) {
            let r = run_sim_round(&mut pace, &clock, HostWait::Block, Duration::from_millis(6));
            assert_eq!(r.slept, Duration::ZERO);
            clock.advance(Duration::from_millis(2));
        }
        assert!(pace.is_paced(), "re-converges after early-finish mis-estimate");
        let steady = run_sim_round(&mut pace, &clock, HostWait::Block, Duration::from_millis(6));
        assert!(steady.slept > Duration::ZERO && !steady.blocked);
        assert_eq!(pace.stats().late_wakes, 1, "no further late wakes once re-stabilised");
        clock.advance(Duration::from_millis(2));

        // 2) Wait ends much later than predicted, still caught in spin (6 ms -> 20 ms).
        let late_spin = run_sim_round(&mut pace, &clock, HostWait::Block, Duration::from_millis(20));
        assert!(!late_spin.blocked, "20 ms finishes inside post-sleep spin headroom without blocking");
        assert!(!pace.is_paced(), "large overrun in spin must switch back to stabilising");
        assert_eq!(pace.stats().restabilises, 2);
        clock.advance(Duration::from_millis(2));
        for _ in 0..(PACE_STABLE_ROUNDS - 1) {
            run_sim_round(&mut pace, &clock, HostWait::Block, Duration::from_millis(20));
            clock.advance(Duration::from_millis(2));
        }
        assert!(pace.is_paced());

        // 3) Wait ends much later and hits the blocking fallback (20 ms -> 150 ms).
        let blocked_r = run_sim_round(&mut pace, &clock, HostWait::Block, Duration::from_millis(150));
        assert!(blocked_r.blocked);
        assert!(!pace.is_paced(), "blocking fallback from paced mode must re-stabilise");
        assert_eq!(pace.stats().restabilises, 3);
        clock.advance(Duration::from_millis(2));
        for _ in 0..(PACE_STABLE_ROUNDS - 1) {
            run_sim_round(&mut pace, &clock, HostWait::Block, Duration::from_millis(150));
            clock.advance(Duration::from_millis(2));
        }
        assert!(pace.is_paced());
        let paced_150 = run_sim_round(&mut pace, &clock, HostWait::Block, Duration::from_millis(150));
        assert!(!paced_150.blocked && paced_150.slept >= Duration::from_millis(140));
    }

    #[test]
    fn wait_pace_never_sleeps_past_completion_by_more_than_spin_tail() {
        let clock = FakeClock::new();
        let mut pace = WaitPace::new();

        // Test abrupt drops from several converged regimes, including non-multiple durations.
        for (steady, drop_to) in [
            (Duration::from_millis(50), Duration::from_millis(2)),
            (Duration::from_millis(100), Duration::from_micros(7_300)),
            (Duration::from_millis(20), Duration::from_micros(15_100)),
        ] {
            pace.reset_stabilising();
            for _ in 0..PACE_STABLE_ROUNDS {
                run_sim_round(&mut pace, &clock, HostWait::Block, steady);
                clock.advance(Duration::from_millis(2));
            }
            assert!(pace.is_paced());
            let tail = pace.spin_tail();
            let r = run_sim_round(&mut pace, &clock, HostWait::Block, drop_to);
            if let Some(sleep_end) = r.last_sleep_end {
                let overshoot = sleep_end.saturating_sub(r.completion);
                assert!(
                    overshoot <= PACE_SLEEP_SLICE && overshoot <= tail,
                    "sleep overshot completion by {overshoot:?} (slice={PACE_SLEEP_SLICE:?}, tail={tail:?})"
                );
            }
            clock.advance(Duration::from_millis(2));
        }

        // Jittery steady workload: adaptive spin tail widens after a near-miss so
        // steady-state rounds stop waking late.
        let mut jitter = WaitPace::new();
        for d in [15_000, 15_000, 15_000] {
            run_sim_round(&mut jitter, &clock, HostWait::Block, Duration::from_micros(d));
            clock.advance(Duration::from_millis(2));
        }
        assert!(jitter.is_paced());
        // One shorter step triggers a single late wake and preserves the widened dev.
        let miss = run_sim_round(&mut jitter, &clock, HostWait::Block, Duration::from_micros(12_500));
        assert!(miss.last_sleep_end.unwrap().saturating_sub(miss.completion) <= PACE_SLEEP_SLICE);
        assert_eq!(jitter.stats().late_wakes, 1);
        clock.advance(Duration::from_millis(2));
        // Re-stabilise across the jittery band, then verify zero further late wakes.
        for &d in &[15_000u64, 15_000, 12_500, 15_000, 16_000, 12_500, 15_000, 13_000] {
            let r = run_sim_round(&mut jitter, &clock, HostWait::Block, Duration::from_micros(d));
            if let Some(end) = r.last_sleep_end {
                assert!(end <= r.completion, "widened spin tail should catch {d} us without late wake");
            }
            clock.advance(Duration::from_millis(2));
        }
        assert_eq!(jitter.stats().late_wakes, 1);
    }

    #[test]
    fn wait_pace_short_long_idle_counters_and_errors() {
        let clock = FakeClock::new();
        static COUNTERS: WaitPaceCounters = WaitPaceCounters::new();
        let mut pace = WaitPace::with_counters(&COUNTERS);

        // Short waits (<= spin_tail + min_sleep) never sleep.
        for _ in 0..8 {
            let r = run_sim_round(&mut pace, &clock, HostWait::Block, Duration::from_millis(2));
            assert_eq!(r.slept, Duration::ZERO);
            assert!(!r.blocked);
            clock.advance(Duration::from_millis(1));
        }
        assert_eq!(pace.sleep_budget(), Duration::ZERO);
        assert_eq!(pace.stats().paced_sleeps, 0);

        // Long prefill-scale waits (> PACE_MAX_WINDOW) never enter paced sleep.
        for _ in 0..5 {
            let r = run_sim_round(&mut pace, &clock, HostWait::Block, Duration::from_millis(400));
            assert_eq!(r.slept, Duration::ZERO);
            assert!(r.blocked);
            clock.advance(Duration::from_millis(2));
        }
        assert!(!pace.is_paced());

        // Converge on 15 ms, then cross PACE_IDLE_RESET -> resets to stabilising.
        for _ in 0..PACE_STABLE_ROUNDS {
            run_sim_round(&mut pace, &clock, HostWait::Block, Duration::from_millis(15));
            clock.advance(Duration::from_millis(2));
        }
        assert!(pace.is_paced());
        clock.advance(PACE_IDLE_RESET);
        let after_idle = run_sim_round(&mut pace, &clock, HostWait::Block, Duration::from_millis(5));
        assert_eq!(after_idle.slept, Duration::ZERO, "idle gap resets pacing so short post-idle wait spins");
        assert!(!pace.is_paced());

        // Shared atomic counters match instance stats.
        assert_eq!(COUNTERS.snapshot(), pace.stats());

        // Poll and block errors propagate immediately.
        assert_eq!(
            pace.wait_with(HostWait::Block, || clock.get(), |_| {}, || Err::<bool, _>("poll"), || Ok(())),
            Err("poll")
        );
        assert_eq!(
            pace.wait_with(
                HostWait::Block,
                || { clock.advance(Duration::from_millis(1)); clock.get() },
                |_| {},
                || Ok(false),
                || Err::<(), _>("block"),
            ),
            Err("block")
        );
    }

    #[test]
    fn wait_pace_host_wait_spin_is_unchanged() {
        let clock = FakeClock::new();
        let mut pace = WaitPace::new();
        // Converge under Block first to prove Spin ignores any learned sleep budget too.
        for _ in 0..PACE_STABLE_ROUNDS {
            run_sim_round(&mut pace, &clock, HostWait::Block, Duration::from_millis(20));
            clock.advance(Duration::from_millis(2));
        }
        let before = pace.stats();
        for dur in [Duration::from_millis(20), Duration::from_millis(400)] {
            let r = run_sim_round(&mut pace, &clock, HostWait::Spin, dur);
            assert_eq!(r.slept, Duration::ZERO, "HostWait::Spin must never sleep");
            assert_eq!(r.sleep_calls, 0);
            assert!(!r.blocked, "HostWait::Spin must never invoke blocking fallback");
            assert!(r.polls > 1);
        }
        assert_eq!(pace.stats(), before, "HostWait::Spin does not alter Block-mode pacing stats");
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
