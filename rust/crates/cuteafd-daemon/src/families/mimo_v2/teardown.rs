//! Preserve the forward failure while attempting every owned cleanup phase.
use anyhow::Result;
use crate::shared::experts::rtx::pair::{SubmissionOwners, teardown};

/// Active and already-aborted views use the same retirement phases.
pub(super) fn begin_retirement(state: &std::cell::Cell<crate::shared::peer_split::TerminalState>) -> Result<bool> {
    use crate::shared::peer_split::{TerminalPeerError, TerminalState};
    match state.get() {
        TerminalState::Drained => Ok(false),
        TerminalState::Active | TerminalState::Aborting => {
            state.set(TerminalState::Aborting);
            Ok(true)
        },
        retained => Err(TerminalPeerError::Closed(retained).into()),
    }
}

/// Every view must publish its abort before touching shared rank streams.
/// A failed proof retains all views, including peers whose own drain succeeded.
pub(super) fn retire_views(views: &mut [&mut dyn SubmissionOwners]) -> Result<()> {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CleanupStep {
    LeadStream,
    PeerStream,
    RestoreDevice,
}

/// An outer-owned stream may predate its installation in the engine. Retire
/// it after engine queues, or retain all owners if either proof fails. Do not
/// touch a potentially blocked outer queue after engine retirement failed.
pub(super) fn retire_engine(has_uninstalled_peer: bool, engine: impl FnOnce() -> Result<()>,
    peer: impl FnOnce() -> Result<()>, retain: impl FnOnce()) -> Result<()> {
    let result = engine().and_then(|()| if has_uninstalled_peer { peer() } else { Ok(()) });
    if result.is_err() { retain(); }
    result
}

/// Cleanup errors must not replace the error that made the engine stop.
/// This does not cancel queued peer waits; callers still own that drain.
pub(super) fn finish<T>(body: Result<T>, has_peer: bool,
    mut cleanup: impl FnMut(CleanupStep) -> Result<()>) -> Result<T> {
    let mut first_failure = None;
    for step in [CleanupStep::LeadStream, CleanupStep::PeerStream, CleanupStep::RestoreDevice] {
        if step == CleanupStep::PeerStream && !has_peer {
            continue;
        }
        if let Err(error) = cleanup(step) {
            tracing::error!(?step, error = %format!("{error:#}"), "MiMo engine cleanup failed");
            if first_failure.is_none() {
                first_failure = Some(error.context(format!("MiMo engine cleanup {step:?}")));
            }
        }
    }
    match body {
        Err(primary) => Err(primary),
        Ok(value) => first_failure.map_or(Ok(value), Err),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, thiserror::Error)]
    #[error("forward capacity failure")]
    struct ForwardFailure;

    #[test]
    fn one_and_two_lane_retirement_accept_active_and_aborting_states() {
        use std::cell::{Cell, RefCell};
        use crate::shared::{experts::rtx::pair::Teardown, peer_split::TerminalState};
        for count in [1, 2] {
            for initial in [TerminalState::Active, TerminalState::Aborting] {
                let states: Vec<_> = (0..count).map(|_| Cell::new(initial)).collect();
                let events = RefCell::new(Vec::new());
                let mut owners: Vec<_> = states.iter().enumerate().filter_map(|(lane, state)| {
                    let events = &events;
                    begin_retirement(state).unwrap().then(move || Teardown {
                        abort: move || { events.borrow_mut().push((lane, "abort")); Ok(()) },
                        ranks: move || { events.borrow_mut().push((lane, "ranks")); Ok(()) },
                        links: || Ok(()), copies: || Ok(()), graphs: || Ok(()),
                        finish: move |drained| state.set(if drained { TerminalState::Drained } else { TerminalState::Retained }),
                    })
                }).collect();
                let mut views: Vec<&mut dyn SubmissionOwners> = owners.iter_mut().map(|owner| owner as _).collect();
                retire_views(&mut views).unwrap();
                assert!(states.iter().all(|state| state.get() == TerminalState::Drained));
                let events = events.borrow();
                assert!(events[..count].iter().all(|(_, phase)| *phase == "abort"));
                assert!(events[count..].iter().all(|(_, phase)| *phase == "ranks"));
                assert!(!begin_retirement(&states[0]).unwrap());
            }
        }
        assert!(begin_retirement(&Cell::new(TerminalState::Retained)).is_err());
    }

    struct Lane<'a> {
        index: usize,
        events: &'a std::cell::RefCell<Vec<(usize, &'static str)>>,
        aborted: &'a [std::cell::Cell<bool>; 2],
        drained: &'a [std::cell::Cell<bool>; 2],
        fail_drain: bool,
    }
    impl SubmissionOwners for Lane<'_> {
        fn abort_all(&mut self) -> Result<()> {
            self.events.borrow_mut().push((self.index, "abort"));
            self.aborted[self.index].set(true);
            Ok(())
        }
        fn drain_ranks(&mut self) -> Result<()> {
            assert!(self.aborted.iter().all(std::cell::Cell::get), "peer wait needs both aborts");
            self.events.borrow_mut().push((self.index, "ranks"));
            anyhow::ensure!(!self.fail_drain, "injected rank drain failure");
            self.drained[self.index].set(true);
            Ok(())
        }
        fn quiesce_links(&mut self) -> Result<()> {
            self.events.borrow_mut().push((self.index, "spark-quiesce")); Ok(())
        }
        fn drain_copies(&mut self) -> Result<()> {
            assert!(self.drained.iter().all(std::cell::Cell::get));
            self.events.borrow_mut().push((self.index, "copies")); Ok(())
        }
        fn retire_graphs(&mut self) -> Result<()> {
            assert!(self.drained.iter().all(std::cell::Cell::get));
            self.events.borrow_mut().push((self.index, "release")); Ok(())
        }
        fn finish(&mut self, drained: bool) {
            self.events.borrow_mut().push((self.index, if drained { "drained" } else { "retained" }));
        }
    }

    #[test]
    fn mid_layer_failure_aborts_both_views_before_releasing_either_spark_wait_owner() {
        use std::cell::{Cell, RefCell};
        for fail_drain in [false, true] {
            let events = RefCell::new(vec![(1, "spark-wait"), (0, "mid-layer-failure")]);
            let aborted = [Cell::new(false), Cell::new(false)];
            let drained = [Cell::new(false), Cell::new(false)];
            let mut first = Lane { index: 0, events: &events, aborted: &aborted, drained: &drained, fail_drain };
            let mut second = Lane { index: 1, events: &events, aborted: &aborted, drained: &drained, fail_drain: false };
            let result = retire_views(&mut [&mut first, &mut second]);
            assert_eq!(result.is_err(), fail_drain);
            let events = events.borrow();
            assert_eq!(&events[2..6], [(0, "abort"), (1, "abort"), (0, "ranks"), (1, "ranks")]);
            if fail_drain {
                assert!(!events.iter().any(|(_, event)| matches!(*event, "release" | "copies")));
                assert_eq!(&events[events.len()-2..], [(0, "retained"), (1, "retained")]);
            } else {
                let first_release = events.iter().position(|(_, event)| *event == "release").unwrap();
                let last_copy = events.iter().rposition(|(_, event)| *event == "copies").unwrap();
                assert!(first_release > last_copy);
                assert_eq!(&events[events.len()-2..], [(0, "drained"), (1, "drained")]);
            }
        }
    }

    #[test]
    fn partial_peer_initialization_retires_both_owned_stream_phases() {
        let seen = std::cell::RefCell::new(Vec::new());
        retire_engine(true, || { seen.borrow_mut().push("engine"); Ok(()) },
            || { seen.borrow_mut().push("uninstalled-peer"); Ok(()) },
            || seen.borrow_mut().push("retained")).unwrap();
        assert_eq!(*seen.borrow(), ["engine", "uninstalled-peer"]);
    }

    #[test]
    fn failed_engine_proof_retains_without_entering_possibly_blocked_peer() {
        let seen = std::cell::RefCell::new(Vec::new());
        let error = retire_engine(true, || { seen.borrow_mut().push("engine-failed"); Err(ForwardFailure.into()) },
            || panic!("unproven peer drain may block"), || seen.borrow_mut().push("retained")).unwrap_err();
        assert!(error.downcast_ref::<ForwardFailure>().is_some());
        assert_eq!(*seen.borrow(), ["engine-failed", "retained"]);
    }

    #[test]
    fn failed_uninstalled_peer_proof_retains_its_typed_error() {
        let seen = std::cell::RefCell::new(Vec::new());
        let error = retire_engine(true, || { seen.borrow_mut().push("engine"); Ok(()) },
            || { seen.borrow_mut().push("peer-failed"); Err(ForwardFailure.into()) },
            || seen.borrow_mut().push("retained")).unwrap_err();
        assert!(error.downcast_ref::<ForwardFailure>().is_some());
        assert_eq!(*seen.borrow(), ["engine", "peer-failed", "retained"]);
    }

    #[test]
    fn installed_peer_is_already_owned_by_engine_retirement() {
        retire_engine(false, || Ok(()), || panic!("duplicate outer stream retirement"),
            || panic!("successful retirement retained owners")).unwrap();
    }

    #[test]
    fn forward_error_survives_every_cleanup_failure() {
        let mut seen = Vec::new();
        let result: Result<()> = finish(Err(ForwardFailure.into()), true, |step| {
            seen.push(step);
            anyhow::bail!("cleanup {step:?} failed")
        });
        assert!(result.unwrap_err().downcast_ref::<ForwardFailure>().is_some());
        assert_eq!(seen, [CleanupStep::LeadStream, CleanupStep::PeerStream, CleanupStep::RestoreDevice]);
    }

    #[test]
    fn lead_destroy_failure_still_attempts_peer_and_restores_device() {
        let mut seen = Vec::new();
        let result = finish(Ok(17), true, |step| {
            seen.push(step);
            if step == CleanupStep::LeadStream { anyhow::bail!("lead destroy failed"); }
            Ok(())
        });
        assert!(format!("{:#}", result.unwrap_err()).contains("lead destroy failed"));
        assert_eq!(seen, [CleanupStep::LeadStream, CleanupStep::PeerStream, CleanupStep::RestoreDevice]);
    }

    #[test]
    fn peer_destroy_failure_still_restores_device() {
        let mut seen = Vec::new();
        let result = finish(Ok(17), true, |step| {
            seen.push(step);
            if step == CleanupStep::PeerStream { anyhow::bail!("peer destroy failed"); }
            Ok(())
        });
        assert!(format!("{:#}", result.unwrap_err()).contains("peer destroy failed"));
        assert_eq!(seen.last(), Some(&CleanupStep::RestoreDevice));
    }

    #[test]
    fn successful_body_reports_failed_device_restore() {
        let result = finish(Ok(17), false, |step| {
            if step == CleanupStep::RestoreDevice { anyhow::bail!("device restore failed"); }
            Ok(())
        });
        assert!(format!("{:#}", result.unwrap_err()).contains("device restore failed"));
    }

    #[test]
    fn successful_single_gpu_cleanup_preserves_body_value() {
        let mut seen = Vec::new();
        let result = finish(Ok(17), false, |step| { seen.push(step); Ok(()) });
        assert_eq!(result.unwrap(), 17);
        assert_eq!(seen, [CleanupStep::LeadStream, CleanupStep::RestoreDevice]);
    }
}
