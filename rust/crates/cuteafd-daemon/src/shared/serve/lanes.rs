//! Independent decode lanes polled on one executor thread.
//!
//! Each lane owns its mutable execution state (workspace, transport wave) and
//! borrows the immutable weights; the lanes share only request banks behind
//! `RefCell`s, borrowed for synchronous work and never across an await. A
//! [`LaneSet`] polls the lanes with `join!`, not `try_join!`: a lane that fails
//! asks its peers to stop at their next completed boundary, but a peer future is
//! never dropped mid-round, because it may own queued device or transport work.
//!
//! A lane round ends with a [`LaneCommit`]: the target's cache commit and the
//! drafter's commit are enqueued together, polled until both have drained, then
//! published; on any failure both are drained before their storage is revoked.
use anyhow::Result;
use std::cell::Cell;
use std::future::Future;
use std::time::{Duration, Instant};

/// How long a lane commit may wait for its device copies before the round fails.
pub(crate) const COMMIT_DEADLINE: Duration = Duration::from_secs(120);
/// How long a lane may wait for a queued prefix retain before giving up.
pub(crate) const RETAIN_DEADLINE: Duration = Duration::from_secs(30);
/// How long a lane may wait for the drafter's proposal before giving up.
pub(crate) const DRAFT_DEADLINE: Duration = Duration::from_secs(120);

/// The outcome of a bounded wait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WaitOutcome {
    /// The copy/commit drained.
    Ready,
    /// The deadline expired before it drained.
    TimedOut,
}

/// Poll `ready` until it reports true, bounded by `deadline`. A device copy or
/// commit that never completes must not spin the lane at 100% CPU forever
/// (W64 F3): the wait uses the shared host-wait policy
/// ([`cuteafd_transport::wait_policy`]) so it spins on the critical path and
/// sleeps once the policy says so, and it gives up at `deadline` so the caller
/// can quarantine the copy instead of synchronizing on it.
///
/// The already-ready path is the hot path: it polls once and never reads the
/// clock or sleeps, so a normal round pays no timer cost.
pub(crate) async fn wait_ready(mut ready: impl FnMut() -> Result<bool>, deadline: Duration) -> Result<WaitOutcome> {
    if ready()? { return Ok(WaitOutcome::Ready); }
    let started = Instant::now();
    let mut backoff = cuteafd_transport::wait_policy::Backoff::new();
    loop {
        if started.elapsed() >= deadline { return Ok(WaitOutcome::TimedOut); }
        backoff.pause();
        tokio::task::yield_now().await;
        if ready()? { return Ok(WaitOutcome::Ready); }
    }
}

/// The stop flag the lanes of one [`LaneSet::join`] share.
#[derive(Default)]
pub(crate) struct LaneSet {
    stop: Cell<bool>,
}

/// One lane's handle on its set.
#[derive(Clone, Copy)]
pub(crate) struct LaneLease<'s> {
    index: usize,
    stop: &'s Cell<bool>,
}

impl LaneSet {
    pub fn lease(&self, index: usize) -> LaneLease<'_> {
        LaneLease { index, stop: &self.stop }
    }

    /// Construct each lane's owners with the same identities used by `run`.
    pub fn try_map<T>(&self, mut setup: impl FnMut(LaneLease<'_>) -> Result<T>) -> Result<[T; 2]> {
        Ok([setup(self.lease(0))?, setup(self.lease(1))?])
    }

    /// Run both lane loops with shared failure propagation, without cancelling
    /// an outstanding device or transport operation.
    pub async fn run<'s, F, G>(&'s self,
        first: impl FnOnce(LaneLease<'s>) -> F,
        second: impl FnOnce(LaneLease<'s>) -> G,
    ) -> Result<()>
    where F: Future<Output = Result<()>>, G: Future<Output = Result<()>> {
        self.join(self.lease(0).run(first(self.lease(0))),
            self.lease(1).run(second(self.lease(1)))).await
    }

    /// Run a family's only decode lane (`LaneSupport::One`): lane 0 under the
    /// same lease, failure stop and host-wait pacing as a pair.
    pub async fn run_one<'s, T, F>(&'s self, lane: impl FnOnce(LaneLease<'s>) -> F) -> Result<T>
    where F: Future<Output = Result<T>> {
        let lease = self.lease(0);
        paced(std::pin::pin!(lease.run(lane(lease)))).await
    }

    /// Poll two lanes to completion. Neither is cancelled when the other fails;
    /// a failing lane stops its peers at their next boundary ([`LaneLease::run`]).
    /// The first lane's error wins when both fail.
    pub async fn join(&self, first: impl Future<Output = Result<()>>, second: impl Future<Output = Result<()>>)
        -> Result<()> {
        let joined = std::pin::pin!(async { tokio::join!(first, second) });
        let (first, second) = paced(joined).await;
        first.and(second)
    }
}

/// Lanes wait by yielding (stream queries, reply and commit polls), so the
/// executor re-polls at once: `--host-wait spin` keeps that; under `block` a
/// pending stretch spins for its first [`BLOCK_SPIN`] (10 ms: every
/// per-layer wait ends within it, and they are the critical path), then sleeps
/// [`BLOCK_SLEEP`] per poll. A poll that completes restarts the spin.
///
/// [`BLOCK_SPIN`]: cuteafd_transport::wait_policy::BLOCK_SPIN
/// [`BLOCK_SLEEP`]: cuteafd_transport::wait_policy::BLOCK_SLEEP
async fn paced<F: Future>(mut future: std::pin::Pin<&mut F>) -> F::Output {
    let mut backoff = cuteafd_transport::wait_policy::Backoff::new();
    std::future::poll_fn(move |context| {
        // Inner waits yield their pending polls to this pacer, which sleeps
        // once per pending poll (they only spin while it polls).
        let poll = { let _paced = cuteafd_transport::wait_policy::Paced::enter(); future.as_mut().poll(context) };
        if poll.is_pending() { backoff.pause() } else { backoff.reset() }
        poll
    }).await
}

/// Balance a drained request cohort before admission. Preserve the first
/// survivor migration and lane-zero tie break. Call only after both lanes
/// release their outstanding device and transport work.
pub(crate) fn rebalance<T>(entries: &mut [T], lane: impl Fn(&T) -> Option<usize>,
    set_lane: impl Fn(&mut T, usize)) -> [usize; 2] {
    let mut loads = [0usize; 2];
    for entry in entries.iter() {
        if let Some(index) = lane(entry) { loads[index] += 1; }
    }
    while loads[0].abs_diff(loads[1]) > 1 {
        let heavy = usize::from(loads[1] > loads[0]);
        let Some(entry) = entries.iter_mut().find(|entry| lane(entry) == Some(heavy)) else { break };
        set_lane(entry, 1 - heavy);
        loads[heavy] -= 1;
        loads[1 - heavy] += 1;
    }
    loads
}

impl LaneLease<'_> {
    pub fn index(&self) -> usize { self.index }
    /// A peer failed or a lane asked the set to stop (e.g. for admission).
    pub fn stopping(&self) -> bool { self.stop.get() }
    /// Ask every lane of the set to return at its next completed boundary.
    pub fn stop(&self) { self.stop.set(true); }
    /// Run this lane's loop; an error stops the peers.
    pub async fn run<T>(self, lane: impl Future<Output = Result<T>>) -> Result<T> {
        let result = lane.await;
        if result.is_err() { self.stop(); }
        result
    }
}

/// A lane round's commit: target and drafter halves that complete
/// asynchronously on the device.
pub(crate) trait LaneCommit {
    /// Enqueue both halves. A failure may leave either half partly enqueued.
    fn begin(&mut self) -> Result<()>;
    /// Whether both enqueued halves have drained.
    fn ready(&mut self) -> Result<bool>;
    /// Publish the drained commit.
    fn publish(&mut self) -> Result<()>;
    /// Drain whatever was enqueued and revoke the round's batch, leaving every
    /// owner releasable. Runs after any failure of the steps above.
    fn abort_and_drain(&mut self) -> Result<()>;
    /// Abandon a commit whose copies did not drain within the deadline. Must NOT
    /// synchronize on the abandoned stream: that would block the serving thread
    /// (V4.1 native serve runs on a current-thread runtime). The default leaves
    /// the owners in place for the caller to quarantine.
    fn quarantine(&mut self) -> Result<()> { Ok(()) }
}

/// A commit that timed out and was quarantined. Callers must NOT discard or
/// release its storage: the device may still be reading it. The owners stay
/// held until the abandoned stream completes (or the process restarts).
#[derive(Debug)]
pub(crate) struct CommitQuarantined;
impl std::fmt::Display for CommitQuarantined {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "commit did not drain; quarantined")
    }
}
impl std::error::Error for CommitQuarantined {}

/// What [`commit`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommitOutcome {
    /// The commit drained and published.
    Done,
    /// The commit did not drain within the deadline. Its owners (workspace,
    /// slot, window bits, snapshot buffers) are still held; the caller must
    /// keep them owned and poll the abandoned stream until it completes, then
    /// release. Never synchronize on it from the host.
    Quarantined,
}

/// Begin, wait (yielding to peer lanes) and publish one commit; on failure
/// drain it first and return the original error. A timed-out commit is
/// quarantined instead of drained: synchronizing on the stream it just gave up
/// on would freeze the whole server.
pub(crate) async fn commit(commit: &mut impl LaneCommit) -> Result<CommitOutcome> {
    commit_with_deadline(commit, COMMIT_DEADLINE).await
}

/// [`commit`] with an explicit deadline (tests use a short one).
async fn commit_with_deadline(commit: &mut impl LaneCommit, deadline: Duration) -> Result<CommitOutcome> {
    let mut timed_out = false;
    let result = match commit.begin() {
        Ok(()) => match wait_ready(|| commit.ready(), deadline).await {
            Ok(WaitOutcome::Ready) => commit.publish(),
            Ok(WaitOutcome::TimedOut) => {
                timed_out = true;
                Err(anyhow::anyhow!("lane commit did not drain within {deadline:?}"))
            }
            Err(error) => Err(error),
        },
        Err(error) => Err(error),
    };
    if let Err(error) = result {
        if timed_out {
            // Leave the owners in place: the caller quarantines them and polls
            // the abandoned stream instead of synchronizing on it.
            if let Err(cleanup) = commit.quarantine() {
                tracing::error!(%cleanup, "quarantining a timed-out lane commit");
            }
            return Ok(CommitOutcome::Quarantined);
        }
        if let Err(cleanup) = commit.abort_and_drain() {
            tracing::error!(%cleanup, "draining a failed lane commit");
        }
        return Err(error);
    }
    Ok(CommitOutcome::Done)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn block_on<T>(future: impl Future<Output = T>) -> T {
        tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap().block_on(future)
    }

    async fn peer(lease: LaneLease<'_>, rounds: &Cell<usize>, finished: &Cell<bool>) -> Result<()> {
        while !lease.stopping() {
            rounds.set(rounds.get() + 1);
            tokio::task::yield_now().await;
        }
        finished.set(true);
        Ok(())
    }
    async fn failing(lease: LaneLease<'_>) -> Result<()> {
        for _ in 0..3 { tokio::task::yield_now().await; }
        anyhow::ensure!(lease.index() == 0, "lane failed");
        Ok(())
    }

    #[test]
    fn owner_setup_and_execution_share_lane_identities() -> Result<()> {
        let set = LaneSet::default();
        let identities = set.try_map(|lease| Ok(lease.index()))?;
        let ran = RefCell::new(Vec::new());
        block_on(set.run(
            |lease| record(&ran, lease.index(), false),
            |lease| record(&ran, lease.index(), false),
        ))?;
        assert_eq!(identities, [0, 1]);
        assert_eq!(*ran.borrow(), identities);
        let mut attempted = Vec::new();
        assert!(set.try_map::<()>(|lease| {
            attempted.push(lease.index());
            anyhow::bail!("setup failed")
        }).is_err());
        assert_eq!(attempted, [0]);
        Ok(())
    }

    #[test]
    fn a_single_lane_runs_as_lane_zero_and_a_failure_stops_the_set() {
        let set = LaneSet::default();
        let value = block_on(set.run_one(|lease| async move {
            for _ in 0..3 { tokio::task::yield_now().await; }
            Ok((lease.index(), lease.stopping()))
        })).unwrap();
        assert_eq!(value, (0, false));
        assert!(!set.lease(0).stopping());
        let error = block_on(set.run_one(failing_one)).unwrap_err();
        assert_eq!(error.to_string(), "single lane failed");
        assert!(set.lease(0).stopping());
    }
    async fn failing_one(lease: LaneLease<'_>) -> Result<()> {
        anyhow::ensure!(lease.index() != 0, "single lane failed");
        Ok(())
    }

    #[test]
    fn a_failing_lane_stops_its_peer_at_a_boundary_without_cancelling_it() {
        let set = LaneSet::default();
        let (rounds, finished) = (Cell::new(0), Cell::new(false));
        let error = block_on(set.run(|lease| peer(lease, &rounds, &finished), failing)).unwrap_err();
        assert_eq!(error.to_string(), "lane failed");
        // The peer ran to its own boundary and returned; it was not dropped.
        assert!(finished.get() && rounds.get() >= 3);
    }

    #[test]
    fn drained_cohorts_migrate_first_survivors_and_keep_balanced_assignments() {
        for original in [vec![], vec![None], vec![Some(0); 8], vec![Some(1); 7],
            vec![Some(0), None, Some(1), Some(0)]] {
            let mut expected = original.clone();
            let mut loads = [0usize; 2];
            for index in expected.iter().flatten() { loads[*index] += 1; }
            while loads[0].abs_diff(loads[1]) > 1 {
                let heavy = usize::from(loads[1] > loads[0]);
                *expected.iter_mut().flatten().find(|index| **index == heavy).unwrap() = 1 - heavy;
                loads[heavy] -= 1;
                loads[1 - heavy] += 1;
            }
            let mut actual = original;
            assert_eq!(rebalance(&mut actual, |entry| *entry,
                |entry, index| *entry = Some(index)), loads);
            assert_eq!(actual, expected);
            let unchanged = actual.clone();
            assert_eq!(rebalance(&mut actual, |entry| *entry,
                |entry, index| *entry = Some(index)), loads);
            assert_eq!(actual, unchanged);
        }
    }

    async fn record(ran: &RefCell<Vec<usize>>, index: usize, fail: bool) -> Result<()> {
        ran.borrow_mut().push(index);
        anyhow::ensure!(!fail, "lane {index}");
        Ok(())
    }

    #[test]
    fn both_lanes_complete_and_the_first_error_wins() {
        let set = LaneSet::default();
        let ran = RefCell::new(Vec::new());
        assert!(block_on(set.join(record(&ran, 0, false), record(&ran, 1, false))).is_ok());
        let error = block_on(set.join(record(&ran, 0, true), record(&ran, 1, true))).unwrap_err();
        assert_eq!(error.to_string(), "lane 0");
        assert_eq!(*ran.borrow(), [0, 1, 0, 1]);
    }

    #[derive(Default)]
    struct Fake {
        fail_begin: bool,
        fail_publish: bool,
        polls: usize,
        log: Vec<&'static str>,
    }
    impl LaneCommit for Fake {
        fn begin(&mut self) -> Result<()> {
            self.log.push("begin");
            anyhow::ensure!(!self.fail_begin, "begin failed");
            Ok(())
        }
        fn ready(&mut self) -> Result<bool> {
            self.polls += 1;
            Ok(self.polls >= 3)
        }
        fn publish(&mut self) -> Result<()> {
            self.log.push("publish");
            anyhow::ensure!(!self.fail_publish, "publish failed");
            Ok(())
        }
        fn abort_and_drain(&mut self) -> Result<()> {
            self.log.push("abort");
            Ok(())
        }
    }

    #[test]
    fn commit_waits_for_both_halves_and_drains_on_failure() {
        let mut ok = Fake::default();
        assert_eq!(block_on(commit(&mut ok)).unwrap(), CommitOutcome::Done);
        assert_eq!((ok.log.as_slice(), ok.polls), (&["begin", "publish"][..], 3));
        let mut early = Fake { fail_begin: true, ..Fake::default() };
        assert_eq!(block_on(commit(&mut early)).unwrap_err().to_string(), "begin failed");
        assert_eq!((early.log.as_slice(), early.polls), (&["begin", "abort"][..], 0));
        let mut late = Fake { fail_publish: true, ..Fake::default() };
        assert_eq!(block_on(commit(&mut late)).unwrap_err().to_string(), "publish failed");
        assert_eq!(late.log, ["begin", "publish", "abort"]);
    }

    /// W64 F3: a device copy that never completes must not spin the lane at
    /// 100% CPU forever. The bounded wait gives up after its deadline.
    #[test]
    fn a_wait_that_never_becomes_ready_gives_up_instead_of_spinning() {
        let started = Instant::now();
        let outcome = block_on(wait_ready(|| Ok(false), Duration::from_millis(20))).unwrap();
        assert_eq!(outcome, WaitOutcome::TimedOut);
        assert!(started.elapsed() < Duration::from_secs(2), "the wait must not spin past its deadline");
    }
    #[test]
    fn a_wait_that_becomes_ready_returns_ready() {
        let mut polls = 0;
        let outcome = block_on(wait_ready(|| { polls += 1; Ok(polls >= 3) }, Duration::from_secs(1))).unwrap();
        assert_eq!(outcome, WaitOutcome::Ready);
        assert_eq!(polls, 3);
    }
    /// Finding 4: the already-ready path is the hot path. It must poll exactly
    /// once and never enter the timer/sleep loop, so a normal round pays no
    /// timer stall.
    #[test]
    fn an_already_ready_wait_polls_once_and_never_sleeps() {
        let mut polls = 0;
        let outcome = block_on(wait_ready(|| { polls += 1; Ok(true) }, Duration::ZERO)).unwrap();
        assert_eq!(outcome, WaitOutcome::Ready);
        assert_eq!(polls, 1, "the ready path must not poll again");
    }
    #[test]
    fn a_wait_propagates_a_poll_error() {
        let error = block_on(wait_ready(|| -> Result<bool> { anyhow::bail!("poll failed") },
            Duration::from_secs(1))).unwrap_err();
        assert_eq!(error.to_string(), "poll failed");
    }
    /// Finding 1: a timed-out commit must be quarantined, not drained. Draining
    /// synchronizes on the stream the wait just gave up on, which blocks the
    /// current-thread serving runtime.
    #[test]
    fn a_timed_out_commit_is_quarantined_not_drained() {
        struct Stuck { log: Vec<&'static str> }
        impl LaneCommit for Stuck {
            fn begin(&mut self) -> Result<()> { self.log.push("begin"); Ok(()) }
            fn ready(&mut self) -> Result<bool> { Ok(false) }
            fn publish(&mut self) -> Result<()> { self.log.push("publish"); Ok(()) }
            fn abort_and_drain(&mut self) -> Result<()> { self.log.push("abort"); Ok(()) }
            fn quarantine(&mut self) -> Result<()> { self.log.push("quarantine"); Ok(()) }
        }
        let mut stuck = Stuck { log: Vec::new() };
        // A zero deadline makes the first non-ready poll time out immediately.
        let outcome = block_on(commit_with_deadline(&mut stuck, Duration::ZERO)).unwrap();
        assert_eq!(outcome, CommitOutcome::Quarantined);
        assert_eq!(stuck.log, ["begin", "quarantine"], "a timed-out commit must not drain");
    }

    /// Engine-level lifecycle: a commit that times out keeps its owners (slot,
    /// window bit, prefix lane) held while the abandoned stream is still
    /// pending, and releases them only once the device-side completion fires.
    /// A second, independent slot keeps working throughout, so the engine does
    /// not freeze on the abandoned stream.
    #[test]
    fn a_quarantined_commit_holds_its_owners_until_the_stream_completes() {
        /// Models the three owners a decode commit holds: the request slot, the
        /// window's `writing` bit and the prefix capture lane.
        #[derive(Default)]
        struct Owners {
            slot: Cell<bool>,
            writing: Cell<bool>,
            prefix_lane: Cell<bool>,
            device_done: Cell<bool>,
            log: RefCell<Vec<&'static str>>,
        }
        impl LaneCommit for Owners {
            fn begin(&mut self) -> Result<()> {
                self.slot.set(true);
                self.writing.set(true);
                self.prefix_lane.set(true);
                self.log.borrow_mut().push("begin");
                Ok(())
            }
            fn ready(&mut self) -> Result<bool> { Ok(self.device_done.get()) }
            fn publish(&mut self) -> Result<()> {
                self.slot.set(false);
                self.writing.set(false);
                self.prefix_lane.set(false);
                self.log.borrow_mut().push("publish");
                Ok(())
            }
            fn abort_and_drain(&mut self) -> Result<()> {
                self.slot.set(false);
                self.writing.set(false);
                self.prefix_lane.set(false);
                self.log.borrow_mut().push("abort");
                Ok(())
            }
            fn quarantine(&mut self) -> Result<()> {
                // Keep every owner: the device may still be reading them.
                self.log.borrow_mut().push("quarantine");
                Ok(())
            }
        }
        let mut owners = Owners::default();
        let outcome = block_on(commit_with_deadline(&mut owners, Duration::ZERO)).unwrap();
        assert_eq!(outcome, CommitOutcome::Quarantined);
        assert!(owners.slot.get() && owners.writing.get() && owners.prefix_lane.get(),
            "a quarantined commit must keep its slot, window bit and prefix lane owned");
        assert_eq!(owners.log.borrow().as_slice(), ["begin", "quarantine"]);

        // A second, independent slot is usable while the abandoned stream is
        // still pending: the engine keeps serving.
        let mut other = Owners::default();
        other.device_done.set(true);
        assert_eq!(block_on(commit_with_deadline(&mut other, Duration::from_secs(1))).unwrap(),
            CommitOutcome::Done);
        assert!(!other.slot.get() && !other.writing.get() && !other.prefix_lane.get());

        // The abandoned stream completes: the owners are released and reusable.
        owners.device_done.set(true);
        assert!(owners.ready().unwrap());
        owners.publish().unwrap();
        assert!(!owners.slot.get() && !owners.writing.get() && !owners.prefix_lane.get(),
            "the owners are released once the device-side completion fires");
        assert_eq!(owners.log.borrow().as_slice(), ["begin", "quarantine", "publish"]);
    }
}
