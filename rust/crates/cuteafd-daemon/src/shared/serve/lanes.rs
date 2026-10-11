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

    /// Poll two lanes to completion. Neither is cancelled when the other fails;
    /// a failing lane stops its peers at their next boundary ([`LaneLease::run`]).
    /// The first lane's error wins when both fail.
    pub async fn join(&self, first: impl Future<Output = Result<()>>, second: impl Future<Output = Result<()>>)
        -> Result<()> {
        let (first, second) = tokio::join!(first, second);
        first.and(second)
    }
}

impl LaneLease<'_> {
    pub fn index(&self) -> usize { self.index }
    /// A peer failed or a lane asked the set to stop (e.g. for admission).
    pub fn stopping(&self) -> bool { self.stop.get() }
    /// Ask every lane of the set to return at its next completed boundary.
    pub fn stop(&self) { self.stop.set(true); }
    /// Run this lane's loop; an error stops the peers.
    pub async fn run(self, lane: impl Future<Output = Result<()>>) -> Result<()> {
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
}

/// Begin, wait (yielding to peer lanes) and publish one commit; on failure
/// drain it first and return the original error.
pub(crate) async fn commit(commit: &mut impl LaneCommit) -> Result<()> {
    let result = async {
        commit.begin()?;
        while !commit.ready()? { tokio::task::yield_now().await; }
        commit.publish()
    }.await;
    if let Err(error) = result {
        if let Err(cleanup) = commit.abort_and_drain() {
            tracing::error!(%cleanup, "draining a failed lane commit");
        }
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn block_on<T>(future: impl Future<Output = T>) -> T {
        tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(future)
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
    fn a_failing_lane_stops_its_peer_at_a_boundary_without_cancelling_it() {
        let set = LaneSet::default();
        let (rounds, finished) = (Cell::new(0), Cell::new(false));
        let error = block_on(set.join(set.lease(0).run(peer(set.lease(0), &rounds, &finished)),
            set.lease(1).run(failing(set.lease(1))))).unwrap_err();
        assert_eq!(error.to_string(), "lane failed");
        // The peer ran to its own boundary and returned; it was not dropped.
        assert!(finished.get() && rounds.get() >= 3);
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
        block_on(commit(&mut ok)).unwrap();
        assert_eq!((ok.log.as_slice(), ok.polls), (&["begin", "publish"][..], 3));
        let mut early = Fake { fail_begin: true, ..Fake::default() };
        assert_eq!(block_on(commit(&mut early)).unwrap_err().to_string(), "begin failed");
        assert_eq!((early.log.as_slice(), early.polls), (&["begin", "abort"][..], 0));
        let mut late = Fake { fail_publish: true, ..Fake::default() };
        assert_eq!(block_on(commit(&mut late)).unwrap_err().to_string(), "publish failed");
        assert_eq!(late.log, ["begin", "publish", "abort"]);
    }
}
