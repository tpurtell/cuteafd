//! Preserve the forward failure while attempting every owned cleanup phase.
use anyhow::Result;

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
