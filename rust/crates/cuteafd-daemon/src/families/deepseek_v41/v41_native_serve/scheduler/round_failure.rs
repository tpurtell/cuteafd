//! FR-01 (W64 F2 / W48 F6): a failed decode/verify round must fail and retire
//! ITS requests instead of wedging the serve loop.
//!
//! Live, after `native decode round failed` the failed-round branch marked every
//! active request `finished/failed`, but the next Retire pass never completed
//! (`serve loop stalled ... phase="retire" busy=48`). The loop called
//! `retire_request(...)?`, so the first release error propagated out of `serve`
//! and the remaining members were stranded with their console tickets, KV leases
//! and draft slots still held.
//!
//! The bookkeeping is extracted here, generic over a small [`RoundMember`] trait
//! (the scheduler implements it for [`Active`]; the tests use plain structs) and
//! a [`Releaser`] (the scheduler's adapter calls `retire_request`). There is no
//! CPU fake of `ServingTarget`/`Requests`: the helpers only touch the member
//! lifecycle, never the GPU path.
//!
//! * [`fail_round`] marks every member failed and sends the error once.
//! * [`retire_done`] takes every done member out and releases EACH one even if an
//!   earlier release errors; it logs each error with the request id and never
//!   returns early. A per-request failure must not end the engine (FX-F6).

use super::super::prefix::V41Cache as PrefixCache;
use super::super::speculative::{DraftChain, DraftRuntime};
use super::{retire_request, Active};
use crate::families::deepseek_v41::v41_requests::Requests;
use crate::families::deepseek_v41::v41_backbone_cache::CacheLease;

/// One member of a decode/verify round whose lifecycle the failure path owns.
pub(super) trait RoundMember {
    /// Report `message` to the client (once) and mark the member terminal.
    fn fail(&mut self, message: &str);
    /// The member is ready to leave the scheduler (finished or client gone).
    fn is_done(&self) -> bool;
    /// The request id, for logging a release error.
    fn id(&self) -> u64;
}

/// Releases a retired member's owners (console ticket, KV lease, draft slot).
pub(super) trait Releaser<M> {
    fn release(&mut self, member: M) -> anyhow::Result<()>;
}

/// What one [`retire_done`] pass did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct RetireReport {
    /// Members taken out of the round and released without error.
    pub retired: usize,
    /// Members whose release returned an error (still taken out of the round).
    pub errors: usize,
}

/// Fail every member of a round that returned an error: send the error once and
/// mark each member terminal so the next Retire pass releases it.
pub(super) fn fail_round<M: RoundMember>(active: &mut [Option<M>], error: &anyhow::Error) {
    let message = format!("{error:#}");
    for member in active.iter_mut().flatten() {
        member.fail(&message);
    }
}

/// Take every done member out of the round and release it. A release error is
/// logged with the request id and counted; it never stops the pass, so one bad
/// release cannot strand the rest (today `retire_request(...)?` did).
pub(super) fn retire_done<M: RoundMember, R: Releaser<M>>(active: &mut [Option<M>],
    releaser: &mut R) -> RetireReport {
    let mut report = RetireReport::default();
    for entry in active.iter_mut() {
        if !entry.as_ref().is_some_and(RoundMember::is_done) { continue; }
        let id = entry.as_ref().unwrap().id();
        let member = entry.take().unwrap();
        match releaser.release(member) {
            Ok(()) => report.retired += 1,
            Err(error) => {
                report.errors += 1;
                tracing::error!(request_id = id, %error,
                    "retired request was not fully released; the serve loop continues");
            }
        }
    }
    report
}

/// RV3-4: the serve loop is exiting (a fatal `?` or the channel closed). Fail
/// every member that is not done yet (once) and release EVERY member, so no
/// console ticket, KV lease or draft slot outlives `serve`. Release errors are
/// logged and counted, never returned.
pub(super) fn drain_on_exit<M: RoundMember, R: Releaser<M>>(active: &mut [Option<M>],
    releaser: &mut R, message: &str) -> RetireReport {
    for member in active.iter_mut().flatten() {
        if !member.is_done() { member.fail(message); }
    }
    retire_done(active, releaser)
}

/// RV3-5: release both owners of an admission that never entered the round
/// (KV lease, then draft slot). The second runs even when the first errors; the
/// first error is returned for the caller to log. A per-request release error
/// must not end `serve` (FX-F6).
pub(super) fn release_both(kv: impl FnOnce() -> anyhow::Result<()>,
    draft: impl FnOnce() -> anyhow::Result<()>) -> anyhow::Result<()> {
    let first = kv();
    let second = draft();
    first.and(second)
}

/// RV3-5: the scheduler's admission release: [`release_both`] over the KV lease
/// and the draft slot, logging (never returning) an error.
pub(super) fn release_admission<'w, 'a, C: DraftChain<'a>>(requests: &mut Requests<'a>,
    draft: Option<&mut DraftRuntime<'w, 'a, C>>, lease: CacheLease, id: u64) {
    let result = release_both(|| requests.release_if_present(lease),
        || draft.map_or(Ok(()), |draft| draft.release(id)));
    if let Err(error) = result {
        tracing::error!(request_id = id, %error, "admission release failed; the serve loop continues");
    }
}

impl RoundMember for Active<'_> {
    fn fail(&mut self, message: &str) {
        let _ = self.job.events.send(Err(message.into()));
        self.finished = true;
        self.cacheable = false;
        self.failed = true;
    }
    fn is_done(&self) -> bool { self.finished || self.job.events.is_closed() }
    fn id(&self) -> u64 { self.ticket.id() }
}

/// The scheduler's [`Releaser`]: the only place the Retire loop calls
/// `retire_request`. A release error is returned to [`retire_done`], which logs
/// it and keeps going instead of propagating out of `serve`.
pub(super) struct RetireAdapter<'r, 'p, 'd, 'w, 'a, C: DraftChain<'a>> {
    pub(super) requests: &'r mut Requests<'a>,
    pub(super) prefixes: &'p mut PrefixCache<'a>,
    pub(super) draft: Option<&'d mut DraftRuntime<'w, 'a, C>>,
}

impl<'a, C: DraftChain<'a>> Releaser<Active<'a>> for RetireAdapter<'_, '_, '_, '_, 'a, C> {
    fn release(&mut self, member: Active<'a>) -> anyhow::Result<()> {
        retire_request(member, self.requests, self.prefixes, self.draft.as_deref_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::{Arc, Mutex};

    type ErrorLog = Arc<Mutex<Vec<(u64, String)>>>;

    struct FakeMember {
        id: u64,
        /// The client left before the failure: `fail` cannot deliver an error.
        closed: bool,
        failed: bool,
        done: bool,
        errors: ErrorLog,
    }

    impl FakeMember {
        fn new(id: u64, closed: bool, errors: &ErrorLog) -> Self {
            Self { id, closed, failed: false, done: closed, errors: Arc::clone(errors) }
        }
    }

    impl RoundMember for FakeMember {
        fn fail(&mut self, message: &str) {
            self.failed = true;
            self.done = true;
            if !self.closed { self.errors.lock().unwrap().push((self.id, message.to_owned())); }
        }
        fn is_done(&self) -> bool { self.done }
        fn id(&self) -> u64 { self.id }
    }

    struct FakeReleaser {
        leases: HashSet<u64>,
        retired: usize,
        fail_on: Option<u64>,
    }

    impl Releaser<FakeMember> for FakeReleaser {
        fn release(&mut self, member: FakeMember) -> anyhow::Result<()> {
            self.leases.remove(&member.id);
            if self.fail_on == Some(member.id) {
                anyhow::bail!("release failed for request {}", member.id);
            }
            self.retired += 1;
            Ok(())
        }
    }

    fn round(count: u64, closed: u64, errors: &ErrorLog) -> Vec<Option<FakeMember>> {
        (0..count).map(|id| Some(FakeMember::new(id, id < closed, errors))).collect()
    }

    fn leases(count: u64) -> HashSet<u64> { (0..count).collect() }

    fn fault(message: &str) -> anyhow::Error { anyhow::anyhow!("{message}") }

    /// 48 members, one failed round: every member gets exactly one error event.
    #[test]
    fn failed_round_errors_every_member_once() {
        let errors: ErrorLog = Arc::new(Mutex::new(Vec::new()));
        let mut active = round(48, 0, &errors);
        fail_round(&mut active, &fault("invalid vocabulary projection rows"));
        let logged = errors.lock().unwrap();
        assert_eq!(logged.len(), 48);
        let ids: HashSet<u64> = logged.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids.len(), 48, "each member is answered exactly once");
        assert!(logged.iter().all(|(_, message)| message == "invalid vocabulary projection rows"));
        assert!(active.iter().flatten().all(|member| member.failed && member.done));
    }

    /// The next Retire pass takes every failed member out: slots `None`, the
    /// console tickets retire (48) and the KV leases are released.
    #[test]
    fn failed_round_members_retire_next_pass() {
        let errors: ErrorLog = Arc::new(Mutex::new(Vec::new()));
        let mut active = round(48, 0, &errors);
        fail_round(&mut active, &fault("window batch shape differs"));
        let mut releaser = FakeReleaser { leases: leases(48), retired: 0, fail_on: None };
        let report = retire_done(&mut active, &mut releaser);
        assert!(active.iter().all(Option::is_none), "every slot is released");
        assert_eq!(report, RetireReport { retired: 48, errors: 0 });
        assert_eq!(releaser.retired, 48);
        assert!(releaser.leases.is_empty(), "every KV lease is released");
    }

    /// One member's release error must not strand the other 47.
    #[test]
    fn release_error_does_not_strand_the_rest() {
        let errors: ErrorLog = Arc::new(Mutex::new(Vec::new()));
        let mut active = round(48, 0, &errors);
        fail_round(&mut active, &fault("invalid vocabulary projection rows"));
        let mut releaser = FakeReleaser { leases: leases(48), retired: 0, fail_on: Some(3) };
        let report = retire_done(&mut active, &mut releaser);
        assert!(active.iter().all(Option::is_none), "the failing member still leaves its slot");
        assert_eq!(report, RetireReport { retired: 47, errors: 1 });
        assert_eq!(releaser.retired, 47);
        assert!(releaser.leases.is_empty(), "the other 47 leases are released");
    }

    /// Half the clients disconnected before the failure: they still retire, and
    /// only the connected half receives an error event.
    #[test]
    fn failed_round_with_closed_clients_still_retires() {
        let errors: ErrorLog = Arc::new(Mutex::new(Vec::new()));
        let mut active = round(48, 24, &errors);
        fail_round(&mut active, &fault("window batch shape differs"));
        assert_eq!(errors.lock().unwrap().len(), 24, "closed clients get no error event");
        let mut releaser = FakeReleaser { leases: leases(48), retired: 0, fail_on: None };
        let report = retire_done(&mut active, &mut releaser);
        assert!(active.iter().all(Option::is_none));
        assert_eq!(report, RetireReport { retired: 48, errors: 0 });
        assert!(releaser.leases.is_empty());
    }

    /// RV3-4: on exit, 24 live members get one error each, all 48 are released.
    #[test]
    fn exit_drain_releases_every_member() {
        let errors: ErrorLog = Arc::new(Mutex::new(Vec::new()));
        let mut active = round(48, 24, &errors);
        let mut releaser = FakeReleaser { leases: leases(48), retired: 0, fail_on: None };
        let report = drain_on_exit(&mut active, &mut releaser, "serve loop stopped");
        assert_eq!(errors.lock().unwrap().len(), 24, "only the not-done members are answered");
        assert!(active.iter().all(Option::is_none), "every slot is released");
        assert_eq!(report, RetireReport { retired: 48, errors: 0 });
        assert!(releaser.leases.is_empty(), "every KV lease is released");
    }

    /// RV3-4: one release error on exit does not stop the drain.
    #[test]
    fn exit_drain_release_error_does_not_stop_the_drain() {
        let errors: ErrorLog = Arc::new(Mutex::new(Vec::new()));
        let mut active = round(48, 0, &errors);
        let mut releaser = FakeReleaser { leases: leases(48), retired: 0, fail_on: Some(7) };
        let report = drain_on_exit(&mut active, &mut releaser, "serve loop stopped");
        assert_eq!(errors.lock().unwrap().len(), 48);
        assert!(active.iter().all(Option::is_none));
        assert_eq!(report, RetireReport { retired: 47, errors: 1 });
        assert!(releaser.leases.is_empty());
    }

    /// RV3-4: the failed-round branch answers and marks its members BEFORE any
    /// `?` on a reset can leave `serve`; the exit block drains `active` once.
    #[test]
    fn fail_round_precedes_the_resets() {
        let source = include_str!("../scheduler.rs");
        let body = &source[source.find("\"native decode round failed\"").expect("failed-round branch")..];
        let fail = body.find("round_failure::fail_round(").expect("fail_round in the branch");
        let reset = body.find("P::reset_connections(first_transport)?").expect("reset in the branch");
        assert!(fail < reset, "fail_round must run before the resets");
        assert_eq!(source.matches("round_failure::drain_on_exit(").count(), 1);
    }

    /// RV3-5: the draft slot is released even when the KV release errors.
    #[test]
    fn admission_release_runs_both_owners_after_an_error() {
        let mut draft_released = false;
        let result = release_both(|| anyhow::bail!("kv release failed"),
            || { draft_released = true; Ok(()) });
        assert!(draft_released, "the second owner is still released");
        assert_eq!(format!("{}", result.unwrap_err()), "kv release failed");
        assert!(release_both(|| Ok(()), || Ok(())).is_ok());
    }

    /// RV3-5: no admission release in `serve` can `?` out of the loop.
    #[test]
    fn serve_admission_releases_never_propagate() {
        let source = include_str!("../scheduler.rs");
        let start = source.find(concat!("pub(super) fn ", "serve<")).expect("serve");
        let end = start + source[start..].find(concat!("media.stop(", "requests")).expect("end of serve");
        let body = &source[start..end];
        assert!(!body.contains(concat!("release_if_present(lease)", "?")), "KV release `?` in serve");
        assert!(!body.contains(concat!("draft.release(id)", "?")), "draft release `?` in serve");
        assert!(body.contains("round_failure::release_admission("));
    }

    /// The serve loop wires each helper exactly once (no second failure path).
    #[test]
    fn scheduler_wires_the_round_failure_helpers_once() {
        let source = include_str!("../scheduler.rs");
        assert_eq!(source.matches("round_failure::fail_round(").count(), 1);
        assert_eq!(source.matches("round_failure::retire_done(").count(), 1);
    }
}
