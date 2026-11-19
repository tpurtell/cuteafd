//! V4.1 prompts on the shared decode share (`shared::prefill_share`, #953).
//!
//! An admitted prompt (leased, restored) waits in the serve loop's
//! [`PrefillQueue`] holding its request slot. Each round runs one chunk of
//! every waiting prompt: with `--decode-share 0` the whole prompt (the
//! previous rule, `run_to_end`), otherwise one call of the prefill driver
//! (`prefill_target::ONE_CALL`: one replay window, single unit, stream call or
//! the decoder replay, the boundaries `prompt_units` cuts). A prompt that
//! finishes joins the running requests in its own chunk; one that fails or
//! whose client left is answered and released in its own chunk, so the next
//! prompt's call starts on reset connections, as before.
//!
//! Links are reset only when a call could have left a wave in flight: a failed
//! call (or a failed packed pass, recovered once in the round's prologue). A
//! client that left is answered at a drained boundary, never mid-call, so it
//! releases only its own owners (#2475).
use crate::shared::prefill_share::{run_to_end, run_unit, Chunk, PrefillDriver};
use anyhow::{Context, Result};

/// A waiting prompt: its request slot, the request (taken when it leaves the
/// queue) and its driver cursor (request-owned state between calls).
pub(in crate::families::deepseek_v41::v41_native_serve) struct Queued<T, C> {
    pub slot: usize,
    pub request: Option<T>,
    pub cursor: C,
    /// When the prompt entered the queue (`prefill_queue_wait_ns_total`).
    pub queued_at: std::time::Instant,
    /// Whether its first chunk has run; the wait is recorded exactly once.
    pub started: bool,
}

impl<T, C> Queued<T, C> {
    pub fn request(&self) -> Option<&T> { self.request.as_ref() }
}

/// What a round's chunk does with a prompt, implemented by the serve loop
/// (and by fakes in tests).
pub(in crate::families::deepseek_v41::v41_native_serve) trait Host<T, C> {
    type Output;
    /// The prompt's next chunk; `Some` once it is prefilled.
    fn run(&mut self, request: &mut T, cursor: &mut C) -> Result<Option<Self::Output>>;
    /// The prefilled prompt joins the running requests. An error is fatal to
    /// the serve loop (a release that failed), as before the queue.
    fn admit(&mut self, slot: usize, request: T, output: Self::Output) -> Result<()>;
    /// Answer the client and release the request's owners (lease, draft).
    fn fail(&mut self, request: T, error: anyhow::Error) -> Result<()>;
    /// Stop, reset and restart every expert link (both decode lanes and the
    /// prefill slots): a failed call may have left a wave unlanded.
    fn reset_links(&mut self) -> Result<()>;
    /// The request's client left.
    fn closed(&self, request: &T) -> bool;
    /// A packed group's stored result for this prompt, taken.
    fn take_stored(&mut self, request: &mut T) -> Option<Result<Self::Output>>;
    /// Mark a stored packed failure as recovered (its links were reset); true
    /// when it was an unrecovered failure.
    fn mark_recovered(&mut self, request: &mut T) -> bool;
    /// A fresh cursor: the left prompt's state (its suffix arena) drops now.
    fn rest(&self) -> C;
}

/// A packed pass that failed (every member stores the error) may have left a
/// wave unlanded on the prefill link. Reset once, here, before any other
/// prompt's call or a decode round reuses the links; its members are then
/// answered in their own chunks without a second reset.
pub(in crate::families::deepseek_v41::v41_native_serve) fn recover_packed<'q, T: 'q, C: 'q, H: Host<T, C>>(
    host: &mut H, waiting: impl Iterator<Item = &'q mut Queued<T, C>>) -> Result<()> {
    let mut failed = false;
    for entry in waiting {
        if let Some(request) = entry.request.as_mut() { failed |= host.mark_recovered(request); }
    }
    if failed { host.reset_links()?; }
    Ok(())
}


/// One round chunk of `entry`; a prompt that leaves (admitted or answered)
/// reports `Done`, so the queue never holds a released request. `Err` only
/// for a fatal serve error (the loop stops; nothing else was released).
pub(in crate::families::deepseek_v41::v41_native_serve) fn chunk<T, C, H: Host<T, C>>(host: &mut H,
    entry: &mut Queued<T, C>) -> Result<Chunk> {
    let request = entry.request.as_mut().context("a queued prompt already left")?;
    // The prompt's wait in the queue, recorded once at its first chunk.
    if !entry.started {
        entry.started = true;
        super::phase_stats::record_prefill_queue_wait(entry.queued_at.elapsed().as_nanos() as u64);
    }
    // A packed group's stored result first, so a packed fault is reported as
    // itself; then a client that left, at this drained boundary (no call of
    // this prompt is in flight between chunks).
    let outcome = match host.take_stored(request).transpose() {
        Err(error) => Err(error),
        Ok(_) if host.closed(request) => Err(super::super::ClientGone.into()),
        Ok(Some(output)) => Ok(Some(output)),
        Ok(None) => host.run(request, &mut entry.cursor),
    };
    // Watchdog progress: a chunk (a call, or a whole prompt) completed.
    super::stall::tick();
    if matches!(outcome, Ok(None)) { return Ok(Chunk::More); }
    let request = entry.request.take().expect("checked above");
    entry.cursor = host.rest();
    match outcome {
        Ok(Some(output)) => host.admit(entry.slot, request, output)?,
        Err(error) => {
            let reset = super::round_failure::needs_reset(&error);
            host.fail(request, error)?;
            if reset { host.reset_links()?; }
        }
        Ok(None) => unreachable!(),
    }
    Ok(Chunk::Done)
}

/// One queue chunk of the driver: the whole prompt with share 0 (exactly the
/// previous `run_to_end`), otherwise one call.
pub(in crate::families::deepseek_v41::v41_native_serve) fn advance<D: PrefillDriver>(driver: &mut D,
    cursor: &mut D::Cursor, whole: bool) -> Result<Option<D::Output>> {
    if whole { run_to_end(driver, cursor).map(Some) }
    else { run_unit(driver, cursor, super::super::prefill_target::ONE_CALL) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::prefill_share::{ChunkBudget, PrefillQueue, PrefillUnit};
    use std::cell::RefCell;
    use std::collections::BTreeSet;
    use std::ops::Range;
    use std::rc::Rc;
    use std::time::Duration;

    /// A V4.1-shaped driver: the real `prompt_units` call plan, recording
    /// each executed call's rows, `(prompt, start..end)`.
    struct Driver { tokens: usize, rows: usize, waves: usize, id: usize, log: Rc<RefCell<Vec<(usize, Range<usize>)>>>,
        sleep: Duration }
    #[derive(Default)]
    struct Cursor { next: usize, done: bool, suffix: Option<Rc<()>> }
    impl PrefillDriver for Driver {
        type Cursor = Cursor;
        type Unit = Range<usize>;
        type Output = usize;
        fn plan(&self, cursor: &Cursor, budget: ChunkBudget) -> Result<PrefillUnit<Range<usize>>> {
            anyhow::ensure!(!cursor.done, "prefill already finished");
            let whole = budget.target.is_none() && budget.row_cap == usize::MAX;
            let work = if cursor.next == self.tokens { self.tokens..self.tokens }
                else if whole { cursor.next..self.tokens }
                else {
                    let spans = super::super::super::prefill_target::one_call_spans(self.tokens, 0, self.rows, self.waves);
                    spans.into_iter().find(|span| span.start == cursor.next).context("no call starts here")?
                };
            let finalizes = work.is_empty();
            Ok(PrefillUnit { rows: work.len(), work, estimate: Duration::ZERO, finalizes })
        }
        fn execute(&mut self, cursor: &mut Cursor, unit: &PrefillUnit<Range<usize>>) -> Result<Option<usize>> {
            std::thread::sleep(self.sleep);
            cursor.suffix.get_or_insert_with(|| Rc::new(()));
            self.log.borrow_mut().push((self.id, unit.work.clone()));
            if unit.work.is_empty() { cursor.done = true; cursor.suffix = None; return Ok(Some(self.id)); }
            cursor.next = unit.work.end;
            Ok(None)
        }
        fn observe(&mut self, _: &PrefillUnit<Range<usize>>, _: Duration) {}
        fn abort_and_drain(&mut self, _: &mut Cursor) -> Result<()> { Ok(()) }
    }

    /// A prompt of `tokens`; `closed` once its client left; `packed` a packed
    /// group's stored result.
    struct Prompt { id: usize, tokens: usize, closed: Rc<RefCell<BTreeSet<usize>>>, packed: Option<Result<usize>> }
    /// The links as the serve loop sees them: a call that fails part-way may
    /// leave a wave unlanded (`dirty`) until `reset_links`; a call on dirty
    /// links is the bug #2475's review found, so it panics.
    struct FakeHost { whole: bool, rows: usize, waves: usize, log: Rc<RefCell<Vec<(usize, Range<usize>)>>>,
        leases: BTreeSet<usize>, admitted: Vec<(usize, usize)>, failed: Vec<(usize, String)>, sleep: Duration,
        resets: usize, dirty: bool, fault: Option<usize> }
    impl Host<Prompt, Cursor> for FakeHost {
        type Output = usize;
        fn run(&mut self, request: &mut Prompt, cursor: &mut Cursor) -> Result<Option<usize>> {
            assert!(!self.dirty, "prompt {}'s call dispatched on links a failed pass left unreset", request.id);
            if self.fault == Some(request.id) {
                self.dirty = true;
                anyhow::bail!("Spark exchange failed with waves in flight");
            }
            let mut driver = Driver { tokens: request.tokens, rows: self.rows, waves: self.waves, id: request.id,
                log: self.log.clone(), sleep: self.sleep };
            advance(&mut driver, cursor, self.whole)
        }
        fn admit(&mut self, slot: usize, request: Prompt, output: usize) -> Result<()> {
            assert_eq!(request.id, output);
            self.admitted.push((slot, request.id));
            Ok(())
        }
        fn fail(&mut self, request: Prompt, error: anyhow::Error) -> Result<()> {
            assert!(self.leases.remove(&request.id), "released twice");
            self.failed.push((request.id, format!("{error:#}")));
            Ok(())
        }
        fn reset_links(&mut self) -> Result<()> {
            self.resets += 1;
            self.dirty = false;
            Ok(())
        }
        fn closed(&self, request: &Prompt) -> bool { request.closed.borrow().contains(&request.id) }
        fn take_stored(&mut self, request: &mut Prompt) -> Option<Result<usize>> { request.packed.take() }
        fn mark_recovered(&mut self, request: &mut Prompt) -> bool {
            super::super::round_failure::mark_recovered(&mut request.packed)
        }
        fn rest(&self) -> Cursor { Cursor::default() }
    }

    fn host(share: f64, rows: usize, waves: usize) -> FakeHost {
        FakeHost { whole: share == 0.0, rows, waves, log: Rc::default(), leases: BTreeSet::new(), admitted: Vec::new(),
            failed: Vec::new(), sleep: Duration::ZERO, resets: 0, dirty: false, fault: None }
    }
    fn queue(host: &mut FakeHost, share: f64, prompts: &[usize], closed: &Rc<RefCell<BTreeSet<usize>>>)
        -> PrefillQueue<Queued<Prompt, Cursor>> {
        let mut queue = PrefillQueue::new(share);
        for (id, &tokens) in prompts.iter().enumerate() {
            host.leases.insert(id);
            queue.push(Queued { slot: 10 + id, request: Some(Prompt { id, tokens, closed: closed.clone(), packed: None }),
                cursor: Cursor::default(), queued_at: std::time::Instant::now(), started: false });
        }
        queue
    }
    fn round(queue: &mut PrefillQueue<Queued<Prompt, Cursor>>, host: &mut FakeHost) {
        for (entry, result) in queue.round(|entry| chunk(host, entry)) {
            result.unwrap();
            assert!(entry.request.is_none() && entry.cursor.suffix.is_none(), "a left prompt holds nothing");
        }
    }

    /// A prompt's wait in the queue is charged once, at its first chunk — not once per chunk,
    /// and not again when it rotates back for a later round.
    #[test]
    fn a_prompt_charges_its_queue_wait_once_at_its_first_chunk() {
        let _guard = super::super::phase_stats::TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let mut host = host(0.2, 2752, 3);
        let closed = Rc::default();
        let mut queue = queue(&mut host, 0.2, &[20000, 20000], &closed);
        // Both prompts have been waiting 5 ms when the first round runs.
        for entry in queue.iter_mut() {
            entry.queued_at = std::time::Instant::now() - Duration::from_millis(5);
        }
        let wait = || super::super::phase_stats::snapshot()["prefill_queue_wait_ns_total"].as_u64().unwrap();
        let before = wait();
        round(&mut queue, &mut host);
        let after_first = wait();
        assert!(after_first - before >= 10_000_000,
            "both prompts charged their wait: {} ns", after_first - before);
        // A second round runs more chunks of the same prompts: nothing new is charged.
        round(&mut queue, &mut host);
        assert_eq!(wait(), after_first, "a prompt's wait is charged once, not per chunk");
    }

    /// Share 0 makes the previous calls in the previous order: each prompt
    /// whole (`run_to_end`) in arrival order, before the next step.
    #[test]
    fn share_zero_makes_todays_calls_in_todays_order() {
        let prompts = [20000, 900, 8256 + 1];
        let mut today = Vec::new();
        for (id, &tokens) in prompts.iter().enumerate() {
            let log = Rc::new(RefCell::new(Vec::new()));
            let mut driver = Driver { tokens, rows: 2752, waves: 3, id, log: log.clone(), sleep: Duration::ZERO };
            run_to_end(&mut driver, &mut Cursor::default()).unwrap();
            today.extend(log.take());
        }
        let mut host = host(0.0, 2752, 3);
        let closed = Rc::default();
        let mut queue = queue(&mut host, 0.0, &prompts, &closed);
        assert!(queue.due(true), "share 0 prefills before the next step");
        round(&mut queue, &mut host);
        assert!(queue.is_empty());
        queue.settle(true);
        assert!(!queue.due(true) && queue.decode_seconds() == 0.0, "share 0 owes nothing");
        assert_eq!(*host.log.borrow(), today);
        assert_eq!(host.admitted, [(10, 0), (11, 1), (12, 2)]);
    }

    /// Positive share: one call per waiting prompt per round in arrival
    /// order, prompts that ran rotating behind; the running requests are owed
    /// their share of each round before the next.
    #[test]
    fn rounds_take_one_call_per_prompt_in_order_and_owe_the_share() {
        let mut host = host(0.2, 2752, 3);
        host.sleep = Duration::from_millis(2);
        let closed = Rc::default();
        let mut queue = queue(&mut host, 0.2, &[20000, 900], &closed);
        assert!(queue.due(true), "nothing owed yet");
        round(&mut queue, &mut host);
        assert_eq!(*host.log.borrow(), [(0, 0..8256), (1, 0..900)]);
        queue.settle(true);
        assert!(!queue.due(true), "the running requests are owed their share");
        assert!(queue.due(false), "nothing decoding: prefill continues");
        let owed = queue.decode_seconds();
        assert!(owed > 0.0);
        queue.stepped(owed / 2.0);
        assert!(!queue.due(true));
        queue.stepped(owed);
        assert!(queue.due(true));
        round(&mut queue, &mut host);
        // Prompt 1's decoder replay ran (it joined); prompt 0 took its next call.
        assert_eq!(host.log.borrow()[2..], [(0, 8256..16512), (1, 900..900)]);
        assert_eq!(host.admitted, [(11, 1)]);
        assert_eq!(queue.len(), 1);
    }

    /// A prompt split across rounds resumes at the next call's first unit:
    /// its calls, concatenated over rounds, are `prompt_units`' calls and
    /// match the whole prompt's rows exactly.
    #[test]
    fn a_split_prompt_resumes_at_the_next_call() {
        for (rows, waves, tokens) in [(2752, 3, 65536), (1024, 3, 5000), (4096, 2, 8193), (80, 1, 300)] {
            let mut host = host(0.2, rows, waves);
            let closed = Rc::default();
            let mut queue = queue(&mut host, 0.2, &[tokens], &closed);
            let mut rounds = 0;
            while !queue.is_empty() {
                round(&mut queue, &mut host);
                rounds += 1;
            }
            let calls: Vec<_> = host.log.borrow().iter().map(|(_, span)| span.clone()).collect();
            assert_eq!(calls, super::super::super::prefill_target::one_call_spans(tokens, 0, rows, waves));
            assert_eq!(rounds, calls.len(), "one call per round");
            assert_eq!(host.admitted, [(10, 0)]);
        }
    }

    /// A client that leaves mid-prefill fails at its next chunk: its lease is
    /// released once, it leaves the queue (its slot frees) and its cursor
    /// drops (the suffix arena frees); the other prompt is unaffected.
    #[test]
    fn cancelling_mid_prefill_releases_the_lease_and_slot() {
        let mut host = host(0.2, 2752, 3);
        let closed: Rc<RefCell<BTreeSet<usize>>> = Rc::default();
        let mut queue = queue(&mut host, 0.2, &[20000, 20000], &closed);
        round(&mut queue, &mut host);
        assert!(queue.iter().all(|entry| entry.cursor.suffix.is_some()), "both mid-encoder");
        closed.borrow_mut().insert(0);
        round(&mut queue, &mut host);
        assert_eq!(host.failed, [(0, "client disconnected".to_owned())]);
        assert_eq!(host.resets, 0, "a client that left between calls resets no link");
        assert!(!host.leases.contains(&0) && host.leases.contains(&1));
        assert_eq!(queue.iter().map(|entry| entry.slot).collect::<Vec<_>>(), [11], "slot 10 is free");
        while !queue.is_empty() { round(&mut queue, &mut host); }
        assert_eq!(host.admitted, [(11, 1)]);
        assert!(host.log.borrow().iter().filter(|(id, _)| *id == 0).count() == 1, "no call after the cancel");
    }

    fn ids(failed: &[(usize, String)]) -> Vec<usize> { failed.iter().map(|(id, _)| *id).collect() }

    /// A client that left before its prompt's first call: no call runs and no
    /// link resets (dev reset all three lanes here, #2475). Whole and shared.
    #[test]
    fn a_client_gone_before_dispatch_resets_nothing() {
        for share in [0.0, 0.2] {
            let mut host = host(share, 2752, 3);
            let closed: Rc<RefCell<BTreeSet<usize>>> = Rc::new(RefCell::new(BTreeSet::from([0])));
            let mut queue = queue(&mut host, share, &[20000, 900], &closed);
            while !queue.is_empty() { round(&mut queue, &mut host); }
            assert_eq!(host.failed, [(0, "client disconnected".to_owned())]);
            assert_eq!(host.resets, 0);
            assert!(host.log.borrow().iter().all(|(id, _)| *id == 1), "the closed prompt dispatched nothing");
            assert_eq!(host.admitted, [(11, 1)]);
        }
    }

    /// A call that fails part-way resets every link once, before the next
    /// prompt's call; that prompt is unaffected.
    #[test]
    fn a_transport_fault_resets_once_before_the_next_call() {
        let mut host = host(0.2, 2752, 3);
        host.fault = Some(0);
        let closed = Rc::default();
        let mut queue = queue(&mut host, 0.2, &[20000, 900], &closed);
        while !queue.is_empty() { round(&mut queue, &mut host); }
        assert_eq!(ids(&host.failed), [0]);
        assert_eq!(host.resets, 1);
        assert_eq!(host.admitted, [(11, 1)]);
    }

    /// A packed pass that failed and a member whose client left: the links
    /// reset once in the round's prologue, before the ineligible prompt packed
    /// across (`C`) dispatches; every member reports the packed fault itself,
    /// never "client disconnected", and nobody resets again. The review's
    /// scenario on #2475: `A`, `C`, `B` queued, `A`+`B` packed, `A` left.
    #[test]
    fn a_packed_fault_with_a_departed_member_resets_once_before_any_call() {
        for share in [0.0, 0.2] {
            let mut host = host(share, 2752, 3);
            let closed: Rc<RefCell<BTreeSet<usize>>> = Rc::new(RefCell::new(BTreeSet::from([0])));
            let mut queue = queue(&mut host, share, &[600, 20000, 700], &closed);
            for entry in queue.iter_mut().filter(|entry| entry.slot != 11) {
                entry.request.as_mut().unwrap().packed = Some(Err(anyhow::anyhow!("packed prefill of 2 prompts: wire down")));
            }
            host.dirty = true;
            recover_packed(&mut host, queue.iter_mut()).unwrap();
            assert_eq!(host.resets, 1, "the failed group resets in the prologue");
            while !queue.is_empty() { round(&mut queue, &mut host); }
            assert_eq!(host.resets, 1, "its members are answered without a second reset");
            assert_eq!(host.failed, [(0, "packed prefill of 2 prompts: wire down".to_owned()),
                (2, "packed prefill of 2 prompts: wire down".to_owned())]);
            assert_eq!(host.admitted, [(11, 1)]);
        }
    }

    /// A packed pass that succeeded for a member whose client left: answered
    /// as client-gone in its own chunk, no reset; its peer is admitted.
    #[test]
    fn a_departed_member_of_a_good_packed_pass_resets_nothing() {
        let mut host = host(0.2, 2752, 3);
        let closed: Rc<RefCell<BTreeSet<usize>>> = Rc::new(RefCell::new(BTreeSet::from([0])));
        let mut queue = queue(&mut host, 0.2, &[600, 700], &closed);
        for entry in queue.iter_mut() {
            let request = entry.request.as_mut().unwrap();
            request.packed = Some(Ok(request.id));
        }
        recover_packed(&mut host, queue.iter_mut()).unwrap();
        round(&mut queue, &mut host);
        assert_eq!(host.failed, [(0, "client disconnected".to_owned())]);
        assert_eq!(host.admitted, [(11, 1)]);
        assert_eq!(host.resets, 0);
        assert!(host.log.borrow().is_empty(), "packed members make no call of their own");
    }
}
