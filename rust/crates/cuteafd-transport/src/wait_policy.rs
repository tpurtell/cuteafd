//! How host threads wait while the engine serves, and when they park once it
//! is idle (TJ 2026-10-25): one process-wide policy, read at startup from
//! `--host-wait` / `--idle-park-seconds` (`CUTEAFD_HOST_WAIT`,
//! `CUTEAFD_IDLE_PARK_SECONDS`), the same on the coordinator and the Spark
//! workers.
//!
//! - Busy waits ([`HostWait`]): `block` (default) sleeps in per-step syncs
//!   (blocking-sync events) and paces the exchange worker's polling to each
//!   wave's expected duration, for a less loaded machine; `spin` spins in
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
/// starts sleeping. Most per-layer waits (a router, a Spark reply) complete
/// within this, so they never pay a sleep's wake-up; longer ones sleep.
pub const BLOCK_SPIN: Duration = Duration::from_micros(50);
/// One sleep of a critical-path wait past [`BLOCK_SPIN`].
pub const BLOCK_SLEEP: Duration = Duration::from_micros(20);

/// A bounded backoff for one host wait on the critical path: under
/// [`HostWait::Spin`] it always spins; under [`HostWait::Block`] it spins for
/// [`BLOCK_SPIN`], then sleeps [`BLOCK_SLEEP`] per pass. Start one per wait.
pub struct Backoff { block: bool, started: Option<Instant> }

impl Backoff {
    pub fn new() -> Self {
        Self { block: wait_policy().wait == HostWait::Block, started: None }
    }
    /// One pass of the wait: spin, or sleep once the spin budget is spent.
    pub fn pause(&mut self) {
        if !self.block { return std::hint::spin_loop(); }
        let started = *self.started.get_or_insert_with(Instant::now);
        if started.elapsed() < BLOCK_SPIN { std::hint::spin_loop() } else { std::thread::sleep(BLOCK_SLEEP) }
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

/// How long no request has been active (the whole process lifetime if none was).
pub fn idle_for() -> Duration {
    let now = now_ns();
    Duration::from_nanos(now.saturating_sub(LAST_ACTIVE.load(Ordering::Relaxed)))
}

/// Whether the engine has been idle long enough for polling threads to park.
pub fn parked() -> bool { idle_for() >= wait_policy().idle_park }

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
    fn backoff_spins_first_then_sleeps_under_block() {
        let mut backoff = Backoff { block: true, started: None };
        let started = Instant::now();
        backoff.pause();
        assert!(started.elapsed() < BLOCK_SLEEP, "the first pause spins");
        std::thread::sleep(BLOCK_SPIN);
        let late = Instant::now();
        backoff.pause();
        assert!(late.elapsed() >= BLOCK_SLEEP, "past the spin budget it sleeps");
        backoff.reset();
        let again = Instant::now();
        backoff.pause();
        assert!(again.elapsed() < BLOCK_SLEEP, "progress restarts the spin");
    }

    #[test]
    fn activity_resets_the_idle_clock() {
        note_active();
        assert!(idle_for() < Duration::from_secs(1));
        assert!(!parked(), "just active: not parked under the default 30 s");
        std::thread::sleep(Duration::from_millis(5));
        assert!(idle_for() >= Duration::from_millis(5));
    }
}
