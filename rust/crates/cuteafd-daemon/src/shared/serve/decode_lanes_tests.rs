//! CPU fault injection for the shared decode-lane owners.
use super::*;
use crate::shared::experts::rtx::pair::Teardown;
use crate::shared::serve::lanes::LaneSet;
use crate::shared::spark_intake::ReceiveDrain;

fn block_on<T>(future: impl Future<Output = T>) -> T {
    tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(future)
}

/// A request owner whose drop must follow the aggregate drain (grammar
/// matchers and request media address queued device work).
struct DropOwner<'s> {
    kind: &'static str,
    drained: &'s Cell<bool>,
    events: &'s RefCell<Vec<&'static str>>,
}
impl Drop for DropOwner<'_> {
    fn drop(&mut self) {
        assert!(self.drained.get(), "request owner dropped before aggregate drain");
        self.events.borrow_mut().push(self.kind);
    }
}

#[test]
fn assignment_fills_the_lighter_lane_and_one_lane_keeps_lane_zero() {
    assert_eq!(assign([0, 0], 2), 0);
    assert_eq!(assign([1, 0], 2), 1);
    assert_eq!(assign([1, 1], 2), 0);
    assert_eq!(assign([3, 5], 2), 0);
    assert_eq!(assign([5, 3], 1), 0);
    // Admission then rebalance keeps |load0 - load1| <= 1 for any arrival order.
    let mut loads = [0usize; 2];
    for _ in 0..17 {
        loads[assign(loads, 2)] += 1;
        assert!(loads[0].abs_diff(loads[1]) <= 1);
    }
}

#[test]
fn round_restores_in_place_on_success_and_selector_unwind() {
    let mut bank = RoundBank::default();
    bank.install(vec![1, 2, 3]);
    {
        let mut round = bank.extract(|request| *request != 2);
        assert_eq!(round.iter_mut().count(), 2);
        for request in round.iter_mut() { *request += 10; }
        assert_eq!(*bank.slots.borrow(), [None, Some(2), None]);
    }
    assert_eq!(*bank.slots.borrow(), [Some(11), Some(2), Some(13)]);
    let result = catch_unwind(AssertUnwindSafe(|| drop(bank.extract(|request| {
        assert!(*request != 2, "injected selector unwind");
        true
    }))));
    assert!(result.is_err());
    assert_eq!(*bank.slots.borrow(), [Some(11), Some(2), Some(13)]);
    assert_eq!(bank.take_restored(), [11, 2, 13]);
    bank.install(vec![4]);
}

#[test]
fn two_lanes_own_disjoint_cohorts_and_interleave_at_awaits() {
    let mut bank = RoundBank::default();
    bank.install((0..5).map(|id| (id, id % 2)).collect::<Vec<(usize, usize)>>());
    let order = RefCell::new(Vec::new());
    let flags = LaneFlags::default();
    let aggregate = Aggregate::new(&flags, || Ok(()));
    let lane = |index: usize| {
        let (bank, order, flags) = (&bank, &order, &flags);
        async move {
            for round in 0..3 {
                let mut owned = bank.extract(|request| request.1 == index);
                let ids: Vec<_> = owned.iter_mut().map(|request| request.0).collect();
                assert!(ids.iter().all(|id| id % 2 == index), "lane {index} took a peer request");
                order.borrow_mut().push((index, round, "issue"));
                // The Spark wait: the peer lane runs here.
                tokio::task::yield_now().await;
                order.borrow_mut().push((index, round, "land"));
                drop(owned);
                if flags.stopping() { break; }
            }
            Ok(())
        }
    };
    block_on(LaneSet::default().run(|_| aggregate.lane(|| lane(0)), |_| aggregate.lane(|| lane(1)))).unwrap();
    let order = order.into_inner();
    // Lane 1 issues while lane 0 waits on its exchange, every round.
    for round in 0..3 {
        let issue1 = order.iter().position(|e| *e == (1, round, "issue")).unwrap();
        let land0 = order.iter().position(|e| *e == (0, round, "land")).unwrap();
        assert!(issue1 < land0, "{order:?}");
    }
    assert_eq!(bank.take_restored().len(), 5);
}

#[test]
fn a_failing_lane_publishes_abort_before_the_waiting_peer_resumes() {
    for panics in [false, true] {
        let events = RefCell::new(Vec::new());
        let drained = Cell::new(false);
        let (waiting, aborted, resumed) = (Cell::new(false), Cell::new(false), Cell::new(false));
        let mut bank = RoundBank::default();
        bank.install((0..2).map(|_| (DropOwner { kind: "constraint-drop", drained: &drained, events: &events },
            DropOwner { kind: "media-drop", drained: &drained, events: &events })).collect::<Vec<_>>());
        let flags = LaneFlags::default();
        let aggregate = Aggregate::new(&flags, || {
            // Lane A has restored its slot while B still owns a receive round.
            let slots = bank.slots.borrow();
            assert!(slots[0].is_some() && slots[1].is_none());
            events.borrow_mut().push("abort");
            aborted.set(true);
            Ok(())
        });
        let result = block_on(LaneSet::default().run(|_| aggregate.lane(|| async {
            let mut first = true;
            let mut round = bank.extract(|_| std::mem::replace(&mut first, false));
            assert_eq!(round.iter_mut().count(), 1);
            while !waiting.get() { tokio::task::yield_now().await; }
            events.borrow_mut().push("lane-error");
            if panics { panic!("injected lane panic after extracting its round"); }
            anyhow::bail!("injected lane error after extracting its round")
        }), |_| aggregate.lane(|| async {
            let round = bank.extract(|_| true);
            // An aggregate receive never drains a shared stream locally.
            let receive = ReceiveDrain::when(false, || panic!("local receive drain"));
            waiting.set(true);
            events.borrow_mut().push("spark-wait");
            while !aborted.get() { tokio::task::yield_now().await; }
            drop(receive);
            resumed.set(true);
            drop(round);
            Ok(())
        })));
        assert!(result.is_err() && resumed.get() && flags.live().is_err());
        assert!(bank.slots.borrow().iter().all(Option::is_some));
        let mut owners: Vec<_> = (0..2).map(|_| Teardown {
            abort: || { assert!(aborted.get()); Ok(()) },
            ranks: || { assert!(resumed.get()); events.borrow_mut().push("rank-drain"); Ok(()) },
            links: || Ok(()), copies: || Ok(()), graphs: || Ok(()),
            finish: |success: bool| { assert!(success); events.borrow_mut().push("lane-drained"); },
        }).collect();
        let mut views: Vec<&mut dyn SubmissionOwners> = owners.iter_mut().map(|owner| owner as _).collect();
        retire_views(&mut views).unwrap();
        drained.set(true);
        drop(bank);
        let events = events.borrow();
        let drain_end = events.iter().rposition(|event| *event == "lane-drained").unwrap();
        for kind in ["media-drop", "constraint-drop"] {
            assert_eq!(events.iter().filter(|event| **event == kind).count(), 2);
            assert!(events.iter().position(|event| *event == kind).unwrap() > drain_end);
        }
        assert!(events.iter().position(|e| *e == "abort").unwrap() < events.iter().position(|e| *e == "rank-drain").unwrap());
    }
}

#[test]
fn host_failure_publishes_before_join_returns() {
    let (waiting, resumed, aborted) = (Cell::new(false), Cell::new(false), Cell::new(false));
    let flags = LaneFlags::default();
    let aggregate = Aggregate::new(&flags, || { aborted.set(true); Ok(()) });
    let result = block_on(LaneSet::default().run(|_| aggregate.lane(|| async {
        while !waiting.get() { tokio::task::yield_now().await; }
        anyhow::bail!("injected host-side metadata failure")
    }), |_| aggregate.lane(|| async {
        waiting.set(true);
        while !aborted.get() { tokio::task::yield_now().await; }
        resumed.set(true);
        Ok(())
    })));
    assert!(result.unwrap_err().to_string().contains("host-side metadata"));
    assert!(resumed.get() && aborted.get());
    // A failed aggregate admits no further lane futures.
    assert!(block_on(aggregate.lane(|| async { Ok(()) })).is_err());
}

#[test]
fn a_constructor_panic_publishes_while_the_peer_waits_and_refuses_later_lanes() {
    let (resumed, aborted) = (Cell::new(false), Cell::new(false));
    let flags = LaneFlags::default();
    let aggregate = Aggregate::new(&flags, || { aborted.set(true); Ok(()) });
    let result = block_on(LaneSet::default().run(|_| aggregate.lane(|| async {
        // Lane 0 queues its wave first, then waits for its reply.
        while !aborted.get() { tokio::task::yield_now().await; }
        resumed.set(true);
        Ok(())
    }), |_| aggregate.lane(|| -> std::future::Ready<Result<()>> { panic!("injected constructor panic") })));
    assert!(result.unwrap_err().to_string().contains("constructing"));
    assert!(resumed.get() && aborted.get() && flags.live().is_err());
}

#[test]
fn one_lane_runs_without_a_peer_and_still_publishes_on_failure() {
    let aborted = Cell::new(0);
    let flags = LaneFlags::default();
    let aggregate = Aggregate::new(&flags, || { aborted.set(aborted.get() + 1); Ok(()) });
    assert_eq!(block_on(aggregate.lane(|| async { Ok(7) })).unwrap(), 7);
    assert_eq!(aborted.get(), 0);
    assert!(block_on(aggregate.lane(|| async { anyhow::bail!("x") as Result<()> })).is_err());
    aggregate.fail();
    assert_eq!(aborted.get(), 1, "abort publishes once");
}

#[test]
fn retirement_is_phase_major_and_a_failed_drain_retains_every_view() {
    struct Lane<'a> {
        index: usize,
        events: &'a RefCell<Vec<(usize, &'static str)>>,
        aborted: &'a [Cell<bool>; 2],
        drained: &'a [Cell<bool>; 2],
        fail_drain: bool,
    }
    impl SubmissionOwners for Lane<'_> {
        fn abort_all(&mut self) -> Result<()> {
            self.events.borrow_mut().push((self.index, "abort"));
            self.aborted[self.index].set(true);
            Ok(())
        }
        fn drain_ranks(&mut self) -> Result<()> {
            assert!(self.aborted.iter().all(Cell::get), "peer wait needs both aborts");
            self.events.borrow_mut().push((self.index, "ranks"));
            anyhow::ensure!(!self.fail_drain, "injected rank drain failure");
            self.drained[self.index].set(true);
            Ok(())
        }
        fn quiesce_links(&mut self) -> Result<()> { self.events.borrow_mut().push((self.index, "links")); Ok(()) }
        fn drain_copies(&mut self) -> Result<()> {
            assert!(self.drained.iter().all(Cell::get));
            self.events.borrow_mut().push((self.index, "copies"));
            Ok(())
        }
        fn retire_graphs(&mut self) -> Result<()> {
            assert!(self.drained.iter().all(Cell::get));
            self.events.borrow_mut().push((self.index, "release"));
            Ok(())
        }
        fn finish(&mut self, drained: bool) {
            self.events.borrow_mut().push((self.index, if drained { "drained" } else { "retained" }));
        }
    }
    for fail_drain in [false, true] {
        let events = RefCell::new(Vec::new());
        let aborted = [Cell::new(false), Cell::new(false)];
        let drained = [Cell::new(false), Cell::new(false)];
        let mut first = Lane { index: 0, events: &events, aborted: &aborted, drained: &drained, fail_drain };
        let mut second = Lane { index: 1, events: &events, aborted: &aborted, drained: &drained, fail_drain: false };
        assert_eq!(retire_views(&mut [&mut first, &mut second]).is_err(), fail_drain);
        let events = events.borrow();
        assert_eq!(&events[..4], [(0, "abort"), (1, "abort"), (0, "ranks"), (1, "ranks")]);
        if fail_drain {
            assert!(!events.iter().any(|(_, event)| matches!(*event, "release" | "copies")));
            assert_eq!(&events[events.len() - 2..], [(0, "retained"), (1, "retained")]);
        } else {
            assert!(events.iter().position(|(_, e)| *e == "release").unwrap()
                > events.iter().rposition(|(_, e)| *e == "copies").unwrap());
            assert_eq!(&events[events.len() - 2..], [(0, "drained"), (1, "drained")]);
        }
    }
}

#[test]
fn one_and_two_view_retirement_accept_active_and_aborting_states() {
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

#[test]
fn stop_resets_per_run_but_a_failure_refuses_every_later_run() {
    let flags = LaneFlags::default();
    {
        let _aggregate = Aggregate::new(&flags, || Ok(()));
        flags.stop();
        assert!(flags.stopping() && flags.live().is_ok());
    }
    let aggregate = Aggregate::new(&flags, || Ok(()));
    assert!(!flags.stopping(), "a new run clears the previous run's stop");
    aggregate.fail();
    drop(aggregate);
    let later = Aggregate::new(&flags, || Ok(()));
    assert!(flags.live().is_err());
    assert!(block_on(later.lane(|| async { Ok(()) })).is_err());
}
