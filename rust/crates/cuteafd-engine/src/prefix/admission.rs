//! A bounded FIFO waiter for page pressure or lifetime admission.
//! Waiting owns no KV or state slot.
use super::PrefixError;

/// Per-waiter lifetime counters; `deferred` counts accepted jobs, including retries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct AdmissionStats {
    pub output_shrinks: u64,
    pub output_tokens_withheld: u64,
    pub deferred: u64,
}

pub enum AdmissionPoll<T> {
    Empty,
    Blocked,
    Ready(T),
}

struct Pending<T> {
    job: T,
    retry: Retry,
}

enum Retry {
    Release { free: usize, epoch: u64 },
    Retirement { epoch: u64 },
}

pub struct DeferredAdmission<T> {
    pending: Option<Pending<T>>,
    stats: AdmissionStats,
}

impl<T> Default for DeferredAdmission<T> {
    fn default() -> Self {
        Self { pending: None, stats: AdmissionStats::default() }
    }
}

impl<T> DeferredAdmission<T> {
    /// Keep at most one job, ahead of the scheduler's existing bounded input
    /// queue. Permanent capacity failures and non-allocation errors return the
    /// job to the caller for its normal error handling.
    pub fn defer(&mut self, job: T, error: &PrefixError, busy: bool, release_epoch: u64) -> Result<(), T> {
        let PrefixError::Pages(pages) = error else { return Err(job) };
        if self.pending.is_some() || !busy || pages.needed > pages.capacity || pages.needed <= pages.free {
            return Err(job);
        }
        self.pending = Some(Pending { job, retry: Retry::Release { free: pages.free, epoch: release_epoch } });
        self.stats.deferred = self.stats.deferred.saturating_add(1);
        Ok(())
    }

    /// Retry a lifetime budget after a retirement or when the pool becomes idle.
    pub fn defer_until_release(&mut self, job: T, busy: bool, release_epoch: u64) -> Result<(), T> {
        if self.pending.is_some() || !busy { return Err(job); }
        self.pending = Some(Pending { job, retry: Retry::Retirement { epoch: release_epoch } });
        self.stats.deferred = self.stats.deferred.saturating_add(1);
        Ok(())
    }

    /// Non-consuming readiness check; cancellation is handled only by `poll`.
    pub fn retry_due(&self, free: usize, release_epoch: u64, busy: bool) -> bool {
        self.pending.as_ref().is_some_and(|pending| !busy || match pending.retry {
            Retry::Release { free: previous, epoch } => free > previous || release_epoch != epoch,
            Retry::Retirement { epoch } => release_epoch != epoch,
        })
    }

    /// Retry when references are released, including pages that become
    /// evictable retained snapshots, or after all running work finishes. Do
    /// not repeat prefix eviction/restore on every decode step. A retirement
    /// waiter ignores free-page changes until a release or an idle pool.
    /// A disconnected waiter is dropped so it cannot block later requests.
    pub fn poll(&mut self, free: usize, release_epoch: u64, busy: bool,
        cancelled: impl FnOnce(&T) -> bool) -> AdmissionPoll<T> {
        let Some(pending) = self.pending.as_ref() else { return AdmissionPoll::Empty };
        if cancelled(&pending.job) {
            self.pending = None;
            return AdmissionPoll::Empty;
        }
        if !self.retry_due(free, release_epoch, busy) {
            return AdmissionPoll::Blocked;
        }
        match self.pending.take() {
            Some(pending) => AdmissionPoll::Ready(pending.job),
            None => AdmissionPoll::Empty,
        }
    }

    /// An idle request whose prompt plus `requested` output tokens does not fit the KV pool gets
    /// the longest output allowance that `fits` the `room`, and `take` reserves it there. The
    /// caller first frees what only a cached prompt costs (`PrefixCache::release_copies`), so a
    /// cached prompt gets what the same request gets cold. Every shrink is counted for `stats`.
    /// `fits` must be monotone. `None`, with nothing taken, when not even one output token fits.
    pub fn shrink_output<R, E>(&mut self, room: &mut R, requested: usize,
        mut fits: impl FnMut(&R, usize) -> Result<bool, E>,
        take: impl FnOnce(&mut R, usize) -> Result<(), E>) -> Result<Option<usize>, E> {
        let Some(granted) = fit_output(requested, |output| fits(room, output))? else {
            return Ok(None);
        };
        take(room, granted)?;
        if granted < requested {
            self.stats.output_shrinks = self.stats.output_shrinks.saturating_add(1);
            self.stats.output_tokens_withheld = self.stats.output_tokens_withheld
                .saturating_add((requested - granted) as u64);
        }
        Ok(Some(granted))
    }

    pub fn pending(&self) -> Option<&T> {
        self.pending.as_ref().map(|pending| &pending.job)
    }

    pub fn stats(&self) -> AdmissionStats {
        self.stats
    }

    pub fn len(&self) -> usize {
        usize::from(self.pending.is_some())
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_none()
    }
}

/// Largest output allowance up to `maximum` that `fits`; `None` when not even one token does.
/// A request with active peers waits instead; call only for an otherwise idle pool.
/// `fits` must be monotone: once an allowance fails, every larger allowance fails.
pub fn fit_output<E>(maximum: usize,
    mut fits: impl FnMut(usize) -> Result<bool, E>) -> Result<Option<usize>, E> {
    if maximum == 0 || !fits(1)? { return Ok(None); }
    let (mut low, mut high) = (1, maximum);
    while low < high {
        let mid = low + (high - low).div_ceil(2);
        if fits(mid)? { low = mid; } else { high = mid - 1; }
    }
    debug_assert!(fits(low)?, "returned output allowance must fit");
    Ok(Some(low))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prefix::{PoolExhausted, RefPagePool};

    #[test]
    fn until_release_waits_for_epoch_progress_or_idle() {
        let mut waiter = DeferredAdmission::default();
        waiter.defer_until_release(1, true, 0).unwrap();
        for (free, epoch) in [(0, 0), (4, 0), (usize::MAX, 0)] {
            assert!(matches!(waiter.poll(free, epoch, true, |_| false), AdmissionPoll::Blocked));
            assert_eq!(waiter.len(), 1);
        }
        assert!(matches!(waiter.poll(0, 0, false, |_| false), AdmissionPoll::Ready(1)));
        assert!(waiter.is_empty());
        waiter.defer_until_release(2, true, 7).unwrap();
        assert!(matches!(waiter.poll(0, 8, true, |_| false), AdmissionPoll::Ready(2)));
    }

    #[test]
    fn until_release_shares_the_single_slot_and_counts_only_accepted_jobs() {
        let mut waiter = DeferredAdmission::default();
        let error = PrefixError::from(PoolExhausted { needed: 3, free: 1, capacity: 4 });
        assert_eq!(waiter.defer_until_release(0, false, 0), Err(0));
        assert_eq!(waiter.defer(0, &error, false, 0), Err(0));
        assert_eq!(waiter.stats(), AdmissionStats::default());
        waiter.defer_until_release(1, true, 0).unwrap();
        assert_eq!(waiter.defer_until_release(2, true, 0), Err(2));
        assert_eq!(waiter.defer(3, &error, true, 0), Err(3));
        assert!(matches!(waiter.poll(0, 0, false, |_| false), AdmissionPoll::Ready(1)));
        waiter.defer(4, &error, true, 0).unwrap();
        assert_eq!(waiter.defer_until_release(5, true, 0), Err(5));
        assert!(matches!(waiter.poll(1, 0, true, |_| true), AdmissionPoll::Empty));
        assert_eq!(waiter.stats(), AdmissionStats { deferred: 2, ..Default::default() });
        assert_eq!(DeferredAdmission::<()>::default().stats(), AdmissionStats::default());
        assert_eq!(serde_json::to_value(waiter.stats()).unwrap()["deferred"], 2);
    }

    #[test]
    fn retry_due_agrees_with_poll_over_free_epoch_and_busy_inputs() {
        let error = PrefixError::from(PoolExhausted { needed: 3, free: 1, capacity: 4 });
        for mode in 0..3 {
            for free in [0, 1, 2, usize::MAX] {
                for epoch in [6, 7, 8, u64::MAX] {
                    for busy in [false, true] {
                        let mut waiter = DeferredAdmission::default();
                        match mode {
                            1 => waiter.defer(1, &error, true, 7).unwrap(),
                            2 => waiter.defer_until_release(1, true, 7).unwrap(),
                            _ => (),
                        }
                        let expected = mode != 0 && (!busy || epoch != 7 || (mode == 1 && free > 1));
                        let due = waiter.retry_due(free, epoch, busy);
                        assert_eq!(due, expected);
                        assert_eq!(waiter.len(), usize::from(mode != 0));
                        assert_eq!(waiter.retry_due(free, epoch, busy), due);
                        assert_eq!(matches!(waiter.poll(free, epoch, busy, |_| false), AdmissionPoll::Ready(1)), due);
                    }
                }
            }
        }
    }

    #[test]
    fn pending_borrows_without_consuming_and_clears_after_ready_or_cancel() {
        let mut waiter = DeferredAdmission::default();
        assert_eq!(waiter.pending(), None);
        waiter.defer_until_release(7, true, 0).unwrap();
        assert_eq!(waiter.pending(), Some(&7));
        assert_eq!(waiter.pending(), Some(&7));
        assert!(matches!(waiter.poll(0, 0, false, |_| false), AdmissionPoll::Ready(7)));
        assert_eq!(waiter.pending(), None);
        waiter.defer_until_release(8, true, 0).unwrap();
        assert!(matches!(waiter.poll(0, 0, true, |_| true), AdmissionPoll::Empty));
        assert_eq!(waiter.pending(), None);
    }

    #[test]
    fn non_monotone_fit_returns_a_fitting_allowance_but_need_not_be_maximal() {
        let fits = |output| Ok::<_, &'static str>(output == 1 || output == 10);
        assert_eq!(fit_output(10, fits), Ok(Some(1)));
        assert_eq!(fits(1), Ok(true));
        assert_eq!(fits(10), Ok(true), "a gap violates the binary search contract");
    }

    #[test]
    fn output_reservation_is_the_longest_that_fits() -> Result<(), &'static str> {
        for maximum in [0, 1, 2, 1000, usize::MAX] {
            for available in [0, 1, 2, 17, 1000, usize::MAX] {
                assert_eq!(fit_output(maximum, |output| Ok(output <= available))?,
                    (maximum > 0 && available > 0).then_some(maximum.min(available)));
            }
        }
        Ok(())
    }

    /// A retained source sharing its tail costs `copy` more output tokens until released.
    struct Room {
        free: usize,
        copy: usize,
        taken: Option<usize>,
    }
    impl Room {
        fn release_copies(&mut self) { self.copy = 0; }
        fn fits(&self, output: usize) -> Result<bool, &'static str> {
            Ok(output + self.copy <= self.free)
        }
        fn take(&mut self, output: usize) -> Result<(), &'static str> {
            if !self.fits(output)? { return Err("output does not fit"); }
            self.taken = Some(output);
            Ok(())
        }
    }

    #[test]
    fn shrink_grants_a_cached_prompt_what_the_same_request_gets_cold_and_counts_it() -> Result<(), &'static str> {
        let mut waiter = DeferredAdmission::<()>::default();
        let mut cold = Room { free: 700, copy: 0, taken: None };
        let mut cached = Room { free: 700, copy: 512, taken: None };
        assert_eq!(fit_output(1000, |output| cached.fits(output))?, Some(188));
        cached.release_copies();
        for room in [&mut cold, &mut cached] {
            let granted = waiter.shrink_output(room, 1000, Room::fits, Room::take)?;
            assert_eq!((granted, room.taken), (Some(700), Some(700)));
        }
        let mut whole = Room { free: 700, copy: 0, taken: None };
        assert_eq!(waiter.shrink_output(&mut whole, 600, Room::fits, Room::take)?, Some(600));
        let mut full = Room { free: 0, copy: 0, taken: None };
        assert_eq!(waiter.shrink_output(&mut full, 1000, Room::fits, Room::take)?, None);
        assert_eq!(full.taken, None);
        assert_eq!(waiter.shrink_output(&mut whole, 0,
            |_, _| -> Result<bool, &'static str> { unreachable!() }, |_, _| unreachable!())?, None);
        let mut failing = Room { free: 50, copy: 0, taken: None };
        assert_eq!(waiter.shrink_output(&mut failing, 1000, Room::fits, |_, _| Err("take failed")), Err("take failed"));
        assert_eq!(failing.taken, None);
        assert_eq!(waiter.stats(), AdmissionStats { output_shrinks: 2, output_tokens_withheld: 600, deferred: 0 });
        Ok(())
    }

    #[test]
    fn shrink_propagates_fit_errors_without_taking_or_counting() {
        let mut waiter = DeferredAdmission::<()>::default();
        for fail_at in [1, 6] {
            assert_eq!(waiter.shrink_output(&mut (), 10,
                |_, output| if output == fail_at { Err("fit failed") } else { Ok(true) },
                |_, _| panic!("failed fit must not take")), Err("fit failed"));
        }
        assert_eq!(waiter.stats(), AdmissionStats::default());
    }

    #[test]
    fn two_large_jobs_run_in_order_after_the_first_releases_its_pages() {
        let mut pool = RefPagePool::new(4, 256);
        let first = pool.alloc(3).unwrap();
        let second_error = PrefixError::from(pool.alloc(3).unwrap_err());
        let mut waiter = DeferredAdmission::default();
        waiter.defer(2, &second_error, true, pool.release_epoch()).unwrap();
        // Thousands of decode steps cause no additional allocation attempts.
        for _ in 0..1024 {
            assert!(matches!(waiter.poll(pool.free(), pool.release_epoch(), true, |_| false), AdmissionPoll::Blocked));
        }
        pool.release(&first);
        assert!(matches!(waiter.poll(pool.free(), pool.release_epoch(), false, |_| false), AdmissionPoll::Ready(2)));
        let second = pool.alloc(3).unwrap();
        assert_eq!(pool.free(), 1);
        pool.release(&second);
        assert_eq!(pool.free(), 4);
        assert_eq!(waiter.len(), 0);
    }

    #[test]
    fn cancellation_unblocks_the_fifo_and_drops_the_waiter_once() {
        use std::cell::Cell;
        struct Job<'a>(&'a Cell<u32>);
        impl Drop for Job<'_> {
            fn drop(&mut self) { self.0.set(self.0.get() + 1); }
        }
        let drops = Cell::new(0);
        let mut waiter = DeferredAdmission::default();
        let error = PrefixError::from(PoolExhausted { needed: 3, free: 1, capacity: 4 });
        assert!(waiter.defer(Job(&drops), &error, true, 0).is_ok());
        assert!(matches!(waiter.poll(1, 0, true, |_| true), AdmissionPoll::Empty));
        assert_eq!(drops.get(), 1);
        assert_eq!(waiter.len(), 0);
        assert!(matches!(waiter.poll(1, 0, true, |_| false), AdmissionPoll::Empty));
        assert_eq!(drops.get(), 1);
        assert!(waiter.defer_until_release(Job(&drops), true, 0).is_ok());
        assert!(!waiter.retry_due(usize::MAX, 0, true));
        assert!(matches!(waiter.poll(usize::MAX, u64::MAX, true, |_| true), AdmissionPoll::Empty));
        assert_eq!(drops.get(), 2);
        assert!(matches!(waiter.poll(0, 0, false, |_| false), AdmissionPoll::Empty));
        assert_eq!(drops.get(), 2);
    }

    #[test]
    fn never_fit_idle_and_non_memory_failures_are_not_queued() {
        let mut waiter = DeferredAdmission::default();
        let error = |needed, free| PrefixError::from(PoolExhausted { needed, free, capacity: 4 });
        assert_eq!(waiter.defer(1, &error(5, 1), true, 0), Err(1));
        assert_eq!(waiter.defer(2, &error(3, 1), false, 0), Err(2));
        assert_eq!(waiter.defer(3, &error(3, 3), true, 0), Err(3));
        assert_eq!(waiter.defer(4, &PrefixError::Host("copy failed".into()), true, 0), Err(4));
        assert_eq!(waiter.len(), 0);
    }

    #[test]
    fn one_waiter_is_bounded_and_preserves_the_earlier_job() {
        let mut waiter = DeferredAdmission::default();
        let error = PrefixError::from(PoolExhausted { needed: 3, free: 1, capacity: 4 });
        waiter.defer(1, &error, true, 0).unwrap();
        assert_eq!(waiter.defer(2, &error, true, 0), Err(2));
        assert_eq!(waiter.len(), 1);
        assert!(matches!(waiter.poll(4, 0, true, |_| false), AdmissionPoll::Ready(1)));
    }

    #[test]
    fn partial_release_retries_once_and_another_failure_waits_for_further_progress() {
        let mut waiter = DeferredAdmission::default();
        let error = |free| PrefixError::from(PoolExhausted { needed: 4, free, capacity: 4 });
        waiter.defer(1, &error(0), true, 0).unwrap();
        assert!(matches!(waiter.poll(2, 0, true, |_| false), AdmissionPoll::Ready(1)));
        waiter.defer(1, &error(2), true, 0).unwrap();
        assert!(matches!(waiter.poll(2, 0, true, |_| false), AdmissionPoll::Blocked));
        assert!(matches!(waiter.poll(4, 0, false, |_| false), AdmissionPoll::Ready(1)));
    }
}
