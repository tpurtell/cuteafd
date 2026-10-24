//! Concurrent callers get an operation-only handle, never a raw engine view.
use super::{MimoEngine, MimoPlacement};
use crate::shared::serve::lanes::LaneSet;
use crate::shared::token_io::{DeviceLogits, RowResult, SelectBatch, TokenSelector};
use anyhow::{Context, Result};
use std::{cell::{Cell, RefCell}, future::{Future, poll_fn}, panic::{AssertUnwindSafe, catch_unwind}, task::Poll};

/// Stable slots stay in the serving owner; a round temporarily owns only its
/// cohort. Never borrow the bank across an await or expose its slot vector.
#[derive(Default)]
pub(in super::super) struct RoundBank<T> {
    slots: RefCell<Vec<Option<T>>>,
}
impl<T> RoundBank<T> {
    pub(in super::super) fn from_requests(requests: Vec<T>) -> Self {
        Self { slots: RefCell::new(requests.into_iter().map(Some).collect()) }
    }

    pub(in super::super) fn extract(&self, mut selected: impl FnMut(&T) -> bool) -> OwnedRound<'_, T> {
        // Install the restore guard before calling a fallible/panicking selector.
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

pub(in super::super) struct OwnedRound<'b, T> {
    bank: &'b RoundBank<T>,
    requests: Vec<(usize, T)>,
}
impl<T> OwnedRound<'_, T> {
    pub(in super::super) fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
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

/// Publication is synchronous, retirement is deferred until both futures return.
/// In particular, a pending receive still owns its link's mutable borrow.
struct Boundary<A: Fn() -> Result<()>> {
    failed: Cell<bool>,
    abort: A,
}
impl<A: Fn() -> Result<()>> Boundary<A> {
    fn new(abort: A) -> Self { Self { failed: Cell::new(false), abort } }

    fn fail(&self) {
        if !self.failed.replace(true) {
            if let Err(error) = (self.abort)() {
                tracing::error!(%error, "decode aggregate abort publication failed; retaining both views");
            }
        }
    }

    fn operation<T>(&self, body: impl FnOnce() -> Result<T>) -> Result<T> {
        anyhow::ensure!(!self.failed.get(), "decode aggregate already failed");
        let mut guard = PublicationGuard { boundary: self, completed: false };
        let result = body();
        guard.completed = result.is_ok();
        result
    }

    async fn future<T, F: Future<Output = Result<T>>>(&self, body: impl FnOnce() -> F) -> Result<T> {
        anyhow::ensure!(!self.failed.get(), "decode aggregate already failed");
        let mut guard = PublicationGuard { boundary: self, completed: false };
        let future = catch_unwind(AssertUnwindSafe(body))
            .map_err(|_| anyhow::anyhow!("decode lane panicked while constructing its future"))?;
        let mut future = Box::pin(future);
        let result = poll_fn(|cx| match catch_unwind(AssertUnwindSafe(|| future.as_mut().poll(cx))) {
            Ok(poll) => poll,
            Err(_) => Poll::Ready(Err(anyhow::anyhow!("decode lane panicked while polling its future"))),
        }).await;
        // Restore every round owner before publishing abort, including owners
        // left in the future after a caught poll panic. The peer must still join.
        drop(future);
        guard.completed = result.is_ok();
        result
    }
}
struct PublicationGuard<'s, A: Fn() -> Result<()>> {
    boundary: &'s Boundary<A>,
    completed: bool,
}
impl<A: Fn() -> Result<()>> Drop for PublicationGuard<'_, A> {
    fn drop(&mut self) { if !self.completed { self.boundary.fail(); } }
}

/// Own lane one until aggregate retirement proves both views releasable.
/// No Deref or accessor exposes either underlying view to concurrent callers.
pub(in super::super) struct DecodeLanes<'e, 'a> {
    first: &'e MimoEngine<'a>,
    second: Option<MimoEngine<'a>>,
    failed: Cell<bool>,
    stop: Cell<bool>,
}
impl<'e, 'a> DecodeLanes<'e, 'a> {
    pub(in super::super) fn new(first: &'e MimoEngine<'a>, second: Option<MimoEngine<'a>>) -> Self {
        Self { first, second, failed: Cell::new(false), stop: Cell::new(false) }
    }

    /// Capture every admitted view before readiness, including a lane that is
    /// idle at runtime C1. No serving operation can capture a new shape.
    pub(in super::super) fn prepare(&self, probe: bool) -> Result<usize> {
        self.operation(|| {
            let mut graphs = 0;
            graphs += self.first.capture_decode_graphs(super::DECODE_ROWS)?;
            self.first.check_tp2_routers()?;
            if probe { self.first.arm_draft_probe()?; }
            if let Some(second) = &self.second {
                graphs += second.capture_decode_graphs(super::DECODE_ROWS)?;
                second.check_tp2_decode_routers(self.first)?;
                if probe { second.arm_draft_probe()?; }
                anyhow::ensure!(second.workspace.borrow().is_none()
                    && second.prefill_kv_wide.borrow().is_none()
                    && second.peer.as_ref().is_none_or(|peer| peer.workspace.borrow().is_none()
                        && peer.prefill_kv_wide.borrow().is_none()),
                    "decode-only view acquired an unadmitted prefill allocation");
            }
            Ok(graphs)
        })
    }

    pub(in super::super) fn has_second(&self) -> bool { self.second.is_some() }

    pub(in super::super) fn retire(&self) -> Result<()> {
        self.first.terminal_shutdown_views(self.second.as_ref())
    }

    pub(in super::super) fn retain(&self) {
        self.first.retain_serving_storage();
        if let Some(second) = &self.second { second.retain_serving_storage(); }
    }

    fn abort(&self) -> Result<()> {
        use crate::shared::{experts::rtx::pair::SubmissionOwners, peer_split::TerminalState};
        let mut errors = Vec::new();
        for view in std::iter::once(self.first).chain(self.second.as_ref()) {
            view.submission_failed.set(true);
            if view.terminal.get() == TerminalState::Active { view.terminal.set(TerminalState::Aborting); }
            if let Err(error) = view.terminal_owners().abort_all() { errors.push(format!("{error:#}")); }
        }
        if !errors.is_empty() {
            self.first.retain_serving_storage();
            if let Some(second) = &self.second { second.retain_serving_storage(); }
            anyhow::bail!("{}", errors.join("; "));
        }
        Ok(())
    }

    fn fail(&self) {
        if !self.failed.replace(true) {
            if let Err(error) = self.abort() {
                tracing::error!(%error, "decode aggregate abort publication failed");
            }
        }
        self.stop.set(true);
    }

    fn operation<T>(&self, body: impl FnOnce() -> Result<T>) -> Result<T> {
        anyhow::ensure!(!self.failed.get(), "decode aggregate already failed");
        // Only the whole-future boundary publishes failure, after its owned
        // round has restored media and grammar owners to the serving bank.
        body()
    }

    pub(in super::super) async fn run<'s, F, G>(&'s self,
        first: impl FnOnce(DecodeLane<'s, 'e, 'a>) -> F,
        second: impl FnOnce(DecodeLane<'s, 'e, 'a>) -> G) -> Result<()>
    where F: Future<Output = Result<()>>, G: Future<Output = Result<()>> {
        self.stop.set(false);
        let set = LaneSet::default();
        let boundary = Boundary::new(|| { self.fail(); Ok(()) });
        let result = if self.second.is_some() {
            set.run(|_| boundary.future(|| first(DecodeLane { owner: self, index: 0 })),
                |_| boundary.future(|| second(DecodeLane { owner: self, index: 1 }))).await
        } else {
            boundary.future(|| first(DecodeLane { owner: self, index: 0 })).await
        };
        if result.is_err() {
            self.fail();
            if let Err(error) = self.first.terminal_shutdown_views(self.second.as_ref()) {
                tracing::error!(%error, "failed decode aggregate retained after both lane futures returned");
            }
        }
        result
    }

}
impl Drop for DecodeLanes<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.first.terminal_shutdown_views(self.second.as_ref()) {
            self.first.retain_serving_storage();
            if let Some(second) = self.second.take() {
                second.retain_serving_storage();
                std::mem::forget(second);
            }
            tracing::error!(%error, "retaining both decode views after failed aggregate retirement");
        }
    }
}

pub(in super::super) struct DecodeLane<'s, 'e, 'a> {
    owner: &'s DecodeLanes<'e, 'a>,
    index: usize,
}
impl<'a> DecodeLane<'_, '_, 'a> {
    pub(in super::super) fn stopping(&self) -> bool { self.owner.stop.get() }
    pub(in super::super) fn stop(&self) { self.owner.stop.set(true); }
    fn view(&self) -> &MimoEngine<'a> {
        if self.index == 0 { self.owner.first } else { self.owner.second.as_ref().expect("admitted lane one") }
    }

    pub(in super::super) fn draft(&self, sequences: &[super::super::dflash::DraftSeq])
        -> Result<Vec<super::super::dflash::Draft>> {
        self.owner.operation(|| {
            let view = self.view();
            view.require_live()?;
            view.drafter.as_ref().context("lane has no DFlash drafter")?.draft(sequences, &view.embedding, view.head())
        })
    }

    pub(in super::super) fn sample(&self, selector: &mut TokenSelector<'a>, logits: &DeviceLogits,
        batch: &SelectBatch) -> Result<Vec<RowResult>> {
        self.owner.operation(|| { self.view().require_live()?; selector.select(logits, batch) })
    }

    pub(in super::super) fn update_context(&self, rows: &[super::super::dflash::ContextRow]) -> Result<()> {
        self.owner.operation(|| {
            self.view().require_live()?;
            self.view().drafter.as_ref().context("lane has no DFlash drafter")?.update(rows)
        })
    }

    pub(in super::super) async fn verify(&self, sequences: &mut [(&mut MimoPlacement, usize)], tokens: &[u32])
        -> Result<DeviceLogits> {
        anyhow::ensure!(!self.owner.failed.get(), "decode aggregate already failed");
        self.view().verify_device_async(sequences, tokens).await?
            .context("decode needs every layer")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{experts::rtx::pair::{SubmissionOwners, Teardown}, spark_intake::ReceiveDrain};
    use std::cell::RefCell;

    struct RequestOwner<'s> {
        media: cuteafd_engine::media::RequestMedia,
        constraint: DropOwner<'s>,
        media_owner: DropOwner<'s>,
    }
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
    fn extracted_round_restores_before_abort_and_keeps_requests_through_peer_drain() {
        for panics in [false, true] {
            let events = RefCell::new(Vec::new());
            let drained = Cell::new(false);
            let waiting = Cell::new(false);
            let resumed = Cell::new(false);
            let aborted = Cell::new(false);
            let bank = RoundBank::from_requests((0..2).map(|_| RequestOwner {
                media: cuteafd_engine::media::RequestMedia::new(Vec::new(), 1, 0).unwrap(),
                constraint: DropOwner { kind: "constraint-drop", drained: &drained, events: &events },
                media_owner: DropOwner { kind: "media-drop", drained: &drained, events: &events },
            }).collect());
            let boundary = Boundary::new(|| {
                // Lane A has restored its slot while B still owns a receive round.
                let slots = bank.slots.borrow();
                assert!(slots[0].is_some() && slots[1].is_none());
                events.borrow_mut().push("abort");
                aborted.set(true);
                Ok(())
            });
            let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
            let result = runtime.block_on(LaneSet::default().run(|_| boundary.future(|| async {
                let mut first = true;
                let mut round = bank.extract(|_| { let take = first; first = false; take });
                assert_eq!(round.iter_mut().count(), 1);
                while !waiting.get() { tokio::task::yield_now().await; }
                events.borrow_mut().push("lane-error");
                if panics { panic!("injected lane panic after extracting its round"); }
                anyhow::bail!("injected lane error after extracting its round")
            }), |_| boundary.future(|| async {
                let round = bank.extract(|_| true);
                let receive_drain = ReceiveDrain::when(false, || panic!("local receive drain"));
                waiting.set(true);
                events.borrow_mut().push("spark-wait");
                while !aborted.get() { tokio::task::yield_now().await; }
                assert!(!drained.get());
                drop(receive_drain);
                resumed.set(true);
                events.borrow_mut().push("receive-return");
                drop(round);
                Ok(())
            })));
            assert!(result.is_err());
            assert!(resumed.get());
            assert!(bank.slots.borrow().iter().all(Option::is_some));
            let mut owners: Vec<_> = (0..2).map(|_| Teardown {
                abort: || { assert!(aborted.get()); Ok(()) },
                ranks: || { assert!(resumed.get()); events.borrow_mut().push("rank-drain"); Ok(()) },
                links: || Ok(()), copies: || Ok(()), graphs: || Ok(()),
                finish: |success: bool| { assert!(success); events.borrow_mut().push("lane-drained"); },
            }).collect();
            let mut views: Vec<&mut dyn SubmissionOwners> = owners.iter_mut().map(|owner| owner as _).collect();
            super::super::super::teardown::retire_views(&mut views).unwrap();
            drained.set(true);
            // All real media values and both tracked owner types survive the drain.
            for request in bank.slots.borrow().iter().flatten() {
                assert!(request.media.spans().is_empty());
                assert_eq!(request.constraint.kind, "constraint-drop");
                assert_eq!(request.media_owner.kind, "media-drop");
            }
            drop(bank);
            let events = events.borrow();
            let drain_end = events.iter().rposition(|event| *event == "lane-drained").unwrap();
            assert_eq!(events.iter().filter(|event| **event == "media-drop").count(), 2);
            assert_eq!(events.iter().filter(|event| **event == "constraint-drop").count(), 2);
            assert!(events.iter().position(|event| *event == "media-drop").unwrap() > drain_end);
            assert!(events.iter().position(|event| *event == "constraint-drop").unwrap() > drain_end);
        }
    }

    #[test]
    fn round_restores_in_place_on_success_and_selector_unwind() {
        let bank = RoundBank::from_requests(vec![1, 2, 3]);
        {
            let mut round = bank.extract(|request| *request != 2);
            for request in round.iter_mut() { *request += 10; }
            assert_eq!(*bank.slots.borrow(), [None, Some(2), None]);
        }
        assert_eq!(*bank.slots.borrow(), [Some(11), Some(2), Some(13)]);
        let result = catch_unwind(AssertUnwindSafe(|| bank.extract(|request| {
            assert!(*request != 2, "injected selector unwind");
            true
        })));
        assert!(result.is_err());
        assert_eq!(*bank.slots.borrow(), [Some(11), Some(2), Some(13)]);
    }

    #[test]
    fn host_failure_in_whole_lane_future_publishes_before_join_returns() {
        let waiting = Cell::new(false);
        let resumed = Cell::new(false);
        let aborted = Cell::new(false);
        let boundary = Boundary::new(|| { aborted.set(true); Ok(()) });
        let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
        let result = runtime.block_on(LaneSet::default().run(|_| boundary.future(|| async {
            while !waiting.get() { tokio::task::yield_now().await; }
            anyhow::bail!("injected host-side metadata failure")
        }), |_| boundary.future(|| async {
            waiting.set(true);
            while !aborted.get() { tokio::task::yield_now().await; }
            resumed.set(true);
            Ok(())
        })));
        assert!(result.unwrap_err().to_string().contains("host-side metadata"));
        assert!(resumed.get() && aborted.get());
    }

    #[test]
    fn synchronous_draft_failure_and_receive_failure_abort_both_before_shared_drain() {
        for receive_failure in [false, true] {
            for drain_failure in [false, true] {
                let events = RefCell::new(Vec::new());
                let waiting = Cell::new(false);
                let resumed = Cell::new(false);
                let aborts = [Cell::new(false), Cell::new(false)];
                let boundary = Boundary::new(|| {
                    for lane in 0..2 {
                        events.borrow_mut().push((lane, "abort"));
                        aborts[lane].set(true);
                    }
                    Ok(())
                });
                let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
                let result = runtime.block_on(async {
                    LaneSet::default().run(|_| async {
                        while !waiting.get() { tokio::task::yield_now().await; }
                        if receive_failure {
                            while !boundary.failed.get() { tokio::task::yield_now().await; }
                            Ok(())
                        } else {
                            boundary.operation(|| -> Result<()> {
                                events.borrow_mut().push((0, "draft-failure"));
                                anyhow::bail!("injected synchronous drafter failure")
                            })
                        }
                    }, |_| async {
                        // Same disabled guard used by aggregate Spark receive:
                        // a partial upload must not trigger a local stream drain.
                        let receive_drain = ReceiveDrain::when(false, || panic!("local receive drain"));
                        waiting.set(true);
                        events.borrow_mut().push((1, "spark-wait"));
                        if receive_failure {
                            let result = boundary.operation(|| -> Result<()> {
                                events.borrow_mut().push((1, "receive-failure"));
                                anyhow::bail!("injected receive failure after upload")
                            });
                            drop(receive_drain);
                            resumed.set(true);
                            result
                        } else {
                            while !boundary.failed.get() { tokio::task::yield_now().await; }
                            assert!(aborts.iter().all(Cell::get));
                            drop(receive_drain);
                            events.borrow_mut().push((1, "receive-return"));
                            resumed.set(true);
                            Ok(())
                        }
                    }).await
                });
                assert!(result.is_err());
                assert!(resumed.get());
                let mut owners: Vec<_> = (0..2).map(|lane| {
                    let (events, resumed, aborts) = (&events, &resumed, &aborts);
                    Teardown {
                    abort: || {
                        assert!(aborts.iter().all(Cell::get));
                        Ok(())
                    },
                    ranks: move || {
                        assert!(resumed.get() && aborts.iter().all(Cell::get));
                        events.borrow_mut().push((lane, "drain"));
                        anyhow::ensure!(!drain_failure || lane != 0, "injected drain failure");
                        Ok(())
                    },
                    links: || Ok(()), copies: || Ok(()),
                    graphs: move || { events.borrow_mut().push((lane, "release")); Ok(()) },
                    finish: move |drained| events.borrow_mut().push((lane, if drained { "drained" } else { "retained" })),
                    }
                }).collect();
                let mut views: Vec<&mut dyn SubmissionOwners> = owners.iter_mut().map(|owner| owner as _).collect();
                assert_eq!(super::super::super::teardown::retire_views(&mut views).is_err(), drain_failure);
                let events = events.borrow();
                let first_drain = events.iter().position(|(_, event)| *event == "drain").unwrap();
                for lane in 0..2 {
                    assert!(events.iter().position(|event| *event == (lane, "abort")).unwrap() < first_drain);
                }
                if drain_failure { assert!(!events.iter().any(|(_, event)| *event == "release")); }
            }
        }
    }
}
