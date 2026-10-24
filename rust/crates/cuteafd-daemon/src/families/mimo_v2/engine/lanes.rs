//! Concurrent callers get an operation-only handle, never a raw engine view.
use super::{MimoEngine, MimoPlacement};
use crate::shared::serve::lanes::LaneSet;
use crate::shared::token_io::{DeviceLogits, RowResult, SelectBatch, TokenSelector};
use anyhow::{Context, Result};
use std::{cell::Cell, future::Future};

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
        let result = body().await;
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
        let boundary = Boundary::new(|| { self.fail(); Ok(()) });
        boundary.operation(body)
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
        let boundary = Boundary::new(|| { self.owner.fail(); Ok(()) });
        let mut guard = PublicationGuard { boundary: &boundary, completed: false };
        let result = self.view().verify_device_async(sequences, tokens).await?
            .context("decode needs every layer");
        guard.completed = result.is_ok();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::{experts::rtx::pair::{SubmissionOwners, Teardown}, spark_intake::ReceiveDrain};
    use std::cell::RefCell;

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
