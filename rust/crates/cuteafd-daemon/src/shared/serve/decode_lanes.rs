//! Independent decode lanes as one shared mechanism (PLAN "Two decode lanes").
//!
//! A family that serves two independent decode batches composes these parts:
//!
//! - [`RoundBank`]: the serving owner keeps every active request in a stable
//!   slot; a lane round extracts only its own cohort ([`OwnedRound`]) and the
//!   guard restores each request to its exact slot on success, error, panic or
//!   cancellation. The bank is never borrowed across an await.
//! - [`Aggregate`]: the failure boundary of both lanes. A lane's whole future
//!   runs inside [`Aggregate::lane`]; the first failure (error, panic while
//!   constructing or polling) publishes every view's abort synchronously, so a
//!   peer blocked in a device-side wait is released before anything drains a
//!   shared rank stream. Neither future is dropped mid-round
//!   ([`super::lanes::LaneSet`] joins, it never cancels).
//! - [`retire_views`]: phase-major teardown of several execution views over
//!   shared rank streams: every abort, then every rank drain, link quiescence,
//!   copy drain and graph release. A failed proof retains every view.
//! - [`assign`]: the lane a newly admitted request joins (the lighter lane,
//!   lane zero on a tie), and [`super::lanes::rebalance`] at drained cohorts.
//!
//! The family supplies only its execution views (attention step, MoE send and
//! land on the lane's own decode link, sampling) and their
//! [`SubmissionOwners`]. Nothing here allocates device memory: every lane-two
//! byte is a family planner charge made before allocation; the lane count
//! and each family's support come from `cuteafd_loader::plan::decode_lanes`.
use crate::shared::experts::rtx::pair::{teardown, SubmissionOwners};
use crate::shared::peer_split::{TerminalPeerError, TerminalState};
use anyhow::Result;
use std::cell::{Cell, RefCell};
use std::future::{poll_fn, Future};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::task::Poll;

/// Most independent decode lanes one coordinator runs.
pub(crate) const MAX_DECODE_LANES: usize = cuteafd_loader::plan::decode_lanes::MAX_DECODE_LANES as usize;

/// Stable request slots owned by the serving loop. A lane round temporarily
/// owns only its cohort; never borrow the bank across an await or expose its
/// slot vector.
pub(crate) struct RoundBank<T> {
    slots: RefCell<Vec<Option<T>>>,
}

impl<T> Default for RoundBank<T> {
    fn default() -> Self { Self { slots: RefCell::new(Vec::new()) } }
}

impl<T> RoundBank<T> {
    /// Install a drained cohort for the next pair of lane loops.
    pub(crate) fn install(&mut self, requests: Vec<T>) {
        assert!(self.slots.get_mut().is_empty(), "previous serving cohort not returned");
        *self.slots.get_mut() = requests.into_iter().map(Some).collect();
    }

    /// Every request, back in admission order, once both lanes returned.
    pub(crate) fn take_restored(&mut self) -> Vec<T> {
        let slots = self.slots.get_mut();
        assert!(slots.iter().all(Option::is_some), "lane still owns a request");
        slots.drain(..).map(Option::unwrap).collect()
    }

    /// Extract the requests `selected` picks. The restore guard is installed
    /// before the (fallible or panicking) selector runs.
    pub(crate) fn extract(&self, mut selected: impl FnMut(&T) -> bool) -> OwnedRound<'_, T> {
        let mut round = OwnedRound { bank: self, requests: Vec::new() };
        let mut slots = self.slots.borrow_mut();
        for (index, slot) in slots.iter_mut().enumerate() {
            if slot.as_ref().is_some_and(&mut selected) {
                round.requests.push((index, slot.take().expect("selected live request")));
            }
        }
        round
    }
}

/// One lane round's cohort; dropping it restores every request to its slot.
pub(crate) struct OwnedRound<'b, T> {
    bank: &'b RoundBank<T>,
    requests: Vec<(usize, T)>,
}

impl<T> OwnedRound<'_, T> {
    pub(crate) fn is_empty(&self) -> bool { self.requests.is_empty() }
    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.requests.iter_mut().map(|(_, request)| request)
    }
}

impl<T> Drop for OwnedRound<'_, T> {
    fn drop(&mut self) {
        let mut slots = self.bank.slots.borrow_mut();
        for (index, request) in self.requests.drain(..) {
            debug_assert!(slots[index].is_none(), "round slot reused before restoration");
            slots[index] = Some(request);
        }
    }
}

/// The lane a newly admitted request joins with `lanes` lanes running: the
/// loader's shared rule ([`cuteafd_loader::plan::decode_lanes::assign`]);
/// with one lane every request joins lane zero.
pub(crate) fn assign(loads: [usize; MAX_DECODE_LANES], lanes: usize) -> usize {
    if lanes < 2 { 0 } else { cuteafd_loader::plan::decode_lanes::assign(loads) }
}

/// The shared failure boundary of a set of independent lanes.
///
/// `abort` publishes every view's terminal abort (device-side flags a peer's
/// blocked wait observes); it must not drain or synchronize a shared stream.
/// It runs at most once, synchronously, at the first failure. The failure
/// flag outlives one run of the lanes ([`LaneFlags`]): a failed set of views
/// refuses every later round until it is retired.
pub(crate) struct Aggregate<'f, A: Fn() -> Result<()>> {
    flags: &'f LaneFlags,
    abort: A,
}

/// A lane set's flags: `failed` is permanent, `stop` is per run.
#[derive(Default)]
pub(crate) struct LaneFlags {
    failed: Cell<bool>,
    stop: Cell<bool>,
}

impl LaneFlags {
    /// Refuse new work once the set failed.
    pub(crate) fn live(&self) -> Result<()> {
        anyhow::ensure!(!self.failed.get(), "decode lanes already failed");
        Ok(())
    }
    /// A lane asked the set to return at its next completed boundary (for
    /// admission, a finished request or a failure).
    pub(crate) fn stopping(&self) -> bool { self.stop.get() }
    pub(crate) fn stop(&self) { self.stop.set(true); }
}

impl<'f, A: Fn() -> Result<()>> Aggregate<'f, A> {
    /// One run of the lanes over `flags`; clears a previous run's stop.
    pub(crate) fn new(flags: &'f LaneFlags, abort: A) -> Self {
        flags.stop.set(false);
        Self { flags, abort }
    }

    /// Publish every abort once and stop the peers.
    pub(crate) fn fail(&self) {
        self.flags.stop.set(true);
        if !self.flags.failed.replace(true) {
            if let Err(error) = (self.abort)() {
                tracing::error!(%error, "decode lane abort publication failed; every view is retained");
            }
        }
    }

    /// Run one lane's whole future. Construction and polling are panic-safe;
    /// the future (and every round owner it holds) is dropped before the
    /// abort is published, so restored owners never race the peer's drain.
    pub(crate) async fn lane<T, F: Future<Output = Result<T>>>(&self, body: impl FnOnce() -> F) -> Result<T> {
        self.flags.live()?;
        let mut guard = Publication { aggregate: self, completed: false };
        let future = catch_unwind(AssertUnwindSafe(body))
            .map_err(|_| anyhow::anyhow!("decode lane panicked while constructing its future"))?;
        let mut future = Box::pin(future);
        let result = poll_fn(|cx| match catch_unwind(AssertUnwindSafe(|| future.as_mut().poll(cx))) {
            Ok(poll) => poll,
            Err(_) => Poll::Ready(Err(anyhow::anyhow!("decode lane panicked while polling its future"))),
        }).await;
        drop(future);
        guard.completed = result.is_ok();
        result
    }
}

struct Publication<'s, 'f, A: Fn() -> Result<()>> {
    aggregate: &'s Aggregate<'f, A>,
    completed: bool,
}

impl<A: Fn() -> Result<()>> Drop for Publication<'_, '_, A> {
    fn drop(&mut self) {
        if !self.completed { self.aggregate.fail(); }
    }
}

/// Active and already-aborting views use the same retirement phases; a
/// drained view needs none; a retained one cannot be retired again.
pub(crate) fn begin_retirement(state: &Cell<TerminalState>) -> Result<bool> {
    match state.get() {
        TerminalState::Drained => Ok(false),
        TerminalState::Active | TerminalState::Aborting => {
            state.set(TerminalState::Aborting);
            Ok(true)
        }
        retained => Err(TerminalPeerError::Closed(retained).into()),
    }
}

/// Retire views that share rank streams phase-major: every view publishes its
/// abort before any drains, and a failed proof retains all of them,
/// including views whose own drain succeeded.
pub(crate) fn retire_views(views: &mut [&mut dyn SubmissionOwners]) -> Result<()> {
    struct Views<'v, 'o>(&'v mut [&'o mut dyn SubmissionOwners]);
    impl Views<'_, '_> {
        fn all(&mut self, mut phase: impl FnMut(&mut dyn SubmissionOwners) -> Result<()>) -> Result<()> {
            let mut failures = Vec::new();
            for (lane, view) in self.0.iter_mut().enumerate() {
                if let Err(error) = phase(*view) { failures.push(format!("lane {lane}: {error:#}")); }
            }
            anyhow::ensure!(failures.is_empty(), "{}", failures.join("; "));
            Ok(())
        }
    }
    impl SubmissionOwners for Views<'_, '_> {
        fn abort_all(&mut self) -> Result<()> { self.all(|view| view.abort_all()) }
        fn drain_ranks(&mut self) -> Result<()> { self.all(|view| view.drain_ranks()) }
        fn quiesce_links(&mut self) -> Result<()> { self.all(|view| view.quiesce_links()) }
        fn drain_copies(&mut self) -> Result<()> { self.all(|view| view.drain_copies()) }
        fn retire_graphs(&mut self) -> Result<()> { self.all(|view| view.retire_graphs()) }
        fn finish(&mut self, drained: bool) {
            for view in self.0.iter_mut() { view.finish(drained); }
        }
    }
    teardown(&mut Views(views))
}

#[cfg(test)]
#[path = "decode_lanes_tests.rs"]
mod tests;
