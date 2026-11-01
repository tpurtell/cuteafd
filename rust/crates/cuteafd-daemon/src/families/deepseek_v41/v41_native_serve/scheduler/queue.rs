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
use crate::shared::prefill_share::{run_to_end, run_unit, Chunk, PrefillDriver};
use anyhow::{Context, Result};

/// A waiting prompt: its request slot, the request (taken when it leaves the
/// queue) and its driver cursor (request-owned state between calls).
pub(in crate::families::deepseek_v41::v41_native_serve) struct Queued<T, C> {
    pub slot: usize,
    pub request: Option<T>,
    pub cursor: C,
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
    /// Answer the client and release every owner (lease, draft, connections).
    fn fail(&mut self, request: T, error: anyhow::Error) -> Result<()>;
    /// A fresh cursor: the left prompt's state (its suffix arena) drops now.
    fn rest(&self) -> C;
}

/// One round chunk of `entry`; a prompt that leaves (admitted or answered)
/// reports `Done`, so the queue never holds a released request. `Err` only
/// for a fatal serve error (the loop stops; nothing else was released).
pub(in crate::families::deepseek_v41::v41_native_serve) fn chunk<T, C, H: Host<T, C>>(host: &mut H,
    entry: &mut Queued<T, C>) -> Result<Chunk> {
    let request = entry.request.as_mut().context("a queued prompt already left")?;
    let outcome = host.run(request, &mut entry.cursor);
    // Watchdog progress: a chunk (a call, or a whole prompt) completed.
    super::stall::tick();
    if matches!(outcome, Ok(None)) { return Ok(Chunk::More); }
    let request = entry.request.take().expect("checked above");
    entry.cursor = host.rest();
    match outcome {
        Ok(Some(output)) => host.admit(entry.slot, request, output)?,
        Err(error) => host.fail(request, error)?,
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

    /// A prompt of `tokens`; `closed` once its client left.
    struct Prompt { id: usize, tokens: usize, closed: Rc<RefCell<BTreeSet<usize>>> }
    struct FakeHost { whole: bool, rows: usize, waves: usize, log: Rc<RefCell<Vec<(usize, Range<usize>)>>>,
        leases: BTreeSet<usize>, admitted: Vec<(usize, usize)>, failed: Vec<usize>, sleep: Duration }
    impl Host<Prompt, Cursor> for FakeHost {
        type Output = usize;
        fn run(&mut self, request: &mut Prompt, cursor: &mut Cursor) -> Result<Option<usize>> {
            anyhow::ensure!(!request.closed.borrow().contains(&request.id), "client disconnected");
            let mut driver = Driver { tokens: request.tokens, rows: self.rows, waves: self.waves, id: request.id,
                log: self.log.clone(), sleep: self.sleep };
            advance(&mut driver, cursor, self.whole)
        }
        fn admit(&mut self, slot: usize, request: Prompt, output: usize) -> Result<()> {
            assert_eq!(request.id, output);
            self.admitted.push((slot, request.id));
            Ok(())
        }
        fn fail(&mut self, request: Prompt, _: anyhow::Error) -> Result<()> {
            assert!(self.leases.remove(&request.id), "released twice");
            self.failed.push(request.id);
            Ok(())
        }
        fn rest(&self) -> Cursor { Cursor::default() }
    }

    fn host(share: f64, rows: usize, waves: usize) -> FakeHost {
        FakeHost { whole: share == 0.0, rows, waves, log: Rc::default(), leases: BTreeSet::new(), admitted: Vec::new(),
            failed: Vec::new(), sleep: Duration::ZERO }
    }
    fn queue(host: &mut FakeHost, share: f64, prompts: &[usize], closed: &Rc<RefCell<BTreeSet<usize>>>)
        -> PrefillQueue<Queued<Prompt, Cursor>> {
        let mut queue = PrefillQueue::new(share);
        for (id, &tokens) in prompts.iter().enumerate() {
            host.leases.insert(id);
            queue.push(Queued { slot: 10 + id, request: Some(Prompt { id, tokens, closed: closed.clone() }),
                cursor: Cursor::default() });
        }
        queue
    }
    fn round(queue: &mut PrefillQueue<Queued<Prompt, Cursor>>, host: &mut FakeHost) {
        for (entry, result) in queue.round(|entry| chunk(host, entry)) {
            result.unwrap();
            assert!(entry.request.is_none() && entry.cursor.suffix.is_none(), "a left prompt holds nothing");
        }
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
        assert_eq!(host.failed, [0]);
        assert!(!host.leases.contains(&0) && host.leases.contains(&1));
        assert_eq!(queue.iter().map(|entry| entry.slot).collect::<Vec<_>>(), [11], "slot 10 is free");
        while !queue.is_empty() { round(&mut queue, &mut host); }
        assert_eq!(host.admitted, [(11, 1)]);
        assert!(host.log.borrow().iter().filter(|(id, _)| *id == 0).count() == 1, "no call after the cancel");
    }
}
