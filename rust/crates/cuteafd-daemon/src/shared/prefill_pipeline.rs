//! Causal chunk ordering for prefill lanes polled on one executor thread.
//!
//! These are level-triggered dependencies, not ownership of execution storage.
//! Reserve/commit waits and their marks are separate so the family can perform its
//! synchronous cache mutation before releasing the successor. Stage-only users
//! can take `chunk` handles without participating in reservation or commit order.
//!
//! Dropping a permit cancels it without marking any unfinished dependency as
//! successful. The caller must propagate errors with `try_join!`, which drops a
//! peer waiting on that cancelled predecessor. A peer awaited independently can
//! remain pending forever; cancellation does not authorize it to use stale state.
//! Family guards remain responsible for revoking admission and draining work.

use std::cell::Cell;
use tokio::sync::Notify;

struct Event {
    ready: Cell<bool>,
    notify: Notify,
}

impl Event {
    fn new() -> Self {
        Self {
            ready: Cell::new(false),
            notify: Notify::new(),
        }
    }

    async fn wait(&self) {
        while !self.ready.get() {
            // No other lane can publish between the check and polling this wait:
            // all lanes are polled on the same executor thread.
            self.notify.notified().await;
        }
    }

    fn publish(&self) {
        self.ready.set(true);
        self.notify.notify_waiters();
    }
}

pub(crate) struct PipelineOrder {
    chunks: usize,
    stages: usize,
    // Per chunk: reservation, each stage, then commit. One allocation per prompt.
    events: Vec<Event>,
}

impl PipelineOrder {
    pub fn new(chunks: usize, stages: usize) -> Self {
        let event_count = stages
            .checked_add(2)
            .and_then(|width| chunks.checked_mul(width))
            .expect("prefill pipeline dimensions overflow");
        Self {
            chunks,
            stages,
            events: (0..event_count).map(|_| Event::new()).collect(),
        }
    }

    /// A stage-only handle, or a handle whose cache reservation is external.
    pub fn chunk(&self, index: usize) -> ChunkPermit<'_> {
        assert!(index < self.chunks, "prefill chunk index out of bounds");
        ChunkPermit { order: self, index }
    }

    /// Wait before reserving cache storage; mark only after that mutation succeeds.
    pub async fn wait_reserve_turn(&self, index: usize) -> ChunkPermit<'_> {
        let permit = self.chunk(index);
        permit.wait_previous(0).await;
        permit
    }

    fn event(&self, index: usize, offset: usize) -> &Event {
        &self.events[index * (self.stages + 2) + offset]
    }
}

pub(crate) struct ChunkPermit<'a> {
    order: &'a PipelineOrder,
    index: usize,
}

impl ChunkPermit<'_> {
    pub fn index(&self) -> usize {
        self.index
    }

    async fn wait_previous(&self, offset: usize) {
        if let Some(previous) = self.index.checked_sub(1) {
            self.order.event(previous, offset).wait().await;
        }
    }

    pub fn mark_reserved(&self) {
        assert!(
            self.index == 0 || self.order.event(self.index - 1, 0).ready.get(),
            "prefill reservation out of order"
        );
        self.order.event(self.index, 0).publish();
    }

    pub async fn wait_predecessor(&self, stage: usize) {
        assert!(stage < self.order.stages, "prefill stage out of bounds");
        self.wait_previous(stage + 1).await;
    }

    pub fn publish(&self, stage: usize) {
        assert!(stage < self.order.stages, "prefill stage out of bounds");
        self.order.event(self.index, stage + 1).publish();
    }

    /// Wait before suffix capture, boundary publication and history commit.
    pub async fn wait_commit_turn(&self) {
        self.wait_previous(self.order.stages + 1).await;
    }

    /// Release the successor only after the family's whole commit succeeds.
    pub fn commit(self) {
        let offset = self.order.stages + 1;
        assert!(
            self.index == 0 || self.order.event(self.index - 1, offset).ready.get(),
            "prefill commit out of order"
        );
        self.order.event(self.index, offset).publish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::time::Duration;

    async fn jitter(seed: &mut u64) {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        for _ in 0..(*seed >> 32) % 4 {
            tokio::task::yield_now().await;
        }
    }

    async fn lane(
        order: &PipelineOrder,
        parity: usize,
        stride: usize,
        mut seed: u64,
        log: &RefCell<Vec<(usize, usize)>>,
    ) {
        for index in (parity..order.chunks).step_by(stride) {
            jitter(&mut seed).await;
            let permit = order.wait_reserve_turn(index).await;
            assert_eq!(permit.index(), index);
            jitter(&mut seed).await;
            log.borrow_mut().push((index, 0));
            permit.mark_reserved();
            for stage in 0..order.stages {
                jitter(&mut seed).await;
                permit.wait_predecessor(stage).await;
                log.borrow_mut().push((index, stage + 1));
                permit.publish(stage);
            }
            jitter(&mut seed).await;
            permit.wait_commit_turn().await;
            jitter(&mut seed).await;
            log.borrow_mut().push((index, order.stages + 1));
            permit.commit();
        }
    }

    fn check_order(order: &PipelineOrder, log: &RefCell<Vec<(usize, usize)>>) {
        let log = log.borrow();
        assert_eq!(log.len(), order.chunks * (order.stages + 2));
        for dependency in 0..order.stages + 2 {
            let indices: Vec<_> = log
                .iter()
                .filter_map(|&(index, event)| (event == dependency).then_some(index))
                .collect();
            assert_eq!(indices, (0..order.chunks).collect::<Vec<_>>());
        }
        for index in 0..order.chunks {
            let events: Vec<_> = log
                .iter()
                .filter_map(|&(chunk, event)| (chunk == index).then_some(event))
                .collect();
            assert_eq!(events, (0..order.stages + 2).collect::<Vec<_>>());
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn two_lanes_order_five_chunks_with_randomized_yields() {
        for seed in 0..32 {
            let order = PipelineOrder::new(5, 4);
            let log = RefCell::new(Vec::new());
            tokio::time::timeout(Duration::from_secs(1), async {
                tokio::join!(
                    lane(&order, 0, 2, seed, &log),
                    lane(&order, 1, 2, seed + 1, &log)
                );
            })
            .await
            .expect("pipeline deadlocked");
            check_order(&order, &log);
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn three_lanes_preserve_dependency_order() {
        let order = PipelineOrder::new(8, 3);
        let log = RefCell::new(Vec::new());
        tokio::time::timeout(Duration::from_secs(1), async {
            tokio::join!(
                lane(&order, 0, 3, 7, &log),
                lane(&order, 1, 3, 11, &log),
                lane(&order, 2, 3, 13, &log)
            );
        })
        .await
        .expect("pipeline deadlocked");
        check_order(&order, &log);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn stage_publication_is_level_triggered_and_stage_specific() {
        let order = PipelineOrder::new(2, 2);
        let first = order.chunk(0);
        let second = order.chunk(1);
        first.wait_predecessor(1).await;
        first.publish(0);
        second.wait_predecessor(0).await;
        second.wait_predecessor(0).await;
        let waiting = second.wait_predecessor(1);
        tokio::pin!(waiting);
        tokio::select! {
            biased;
            _ = &mut waiting => panic!("another stage released the wait"),
            _ = tokio::task::yield_now() => {}
        }
        first.publish(1);
        tokio::time::timeout(Duration::from_secs(1), waiting)
            .await
            .expect("stage publication was lost");
        second.wait_predecessor(1).await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn reservation_and_commit_marks_follow_external_mutations() {
        let order = PipelineOrder::new(2, 0);
        let first = order.wait_reserve_turn(0).await;
        let second = order.wait_reserve_turn(1);
        tokio::pin!(second);
        tokio::select! {
            biased;
            _ = &mut second => panic!("reservation released before its mark"),
            _ = tokio::task::yield_now() => {}
        }
        first.mark_reserved();
        let second = second.await;
        second.mark_reserved();
        {
            let waiting = second.wait_commit_turn();
            tokio::pin!(waiting);
            tokio::select! {
                biased;
                _ = &mut waiting => panic!("commit released before its mark"),
                _ = tokio::task::yield_now() => {}
            }
            first.commit();
            waiting.await;
        }
        second.wait_commit_turn().await;
        second.commit();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn error_mid_chunk_drops_two_peer_lanes() {
        let order = PipelineOrder::new(3, 1);
        let entered = Cell::new(0);
        let result = tokio::time::timeout(Duration::from_secs(1), async {
            tokio::try_join!(
                biased;
                async {
                    {
                        let first = order.wait_reserve_turn(0).await;
                        first.mark_reserved();
                        tokio::task::yield_now().await;
                        assert_eq!(entered.get(), 2);
                    }
                    Err::<(), _>("encoder failed mid-chunk")
                },
                async {
                    let second = order.wait_reserve_turn(1).await;
                    second.mark_reserved();
                    entered.set(entered.get() + 1);
                    second.wait_predecessor(0).await;
                    Ok::<(), &str>(())
                },
                async {
                    let third = order.wait_reserve_turn(2).await;
                    third.mark_reserved();
                    entered.set(entered.get() + 1);
                    third.wait_commit_turn().await;
                    Ok::<(), &str>(())
                }
            )
        })
        .await
        .expect("try_join did not cancel both peer lanes");
        assert_eq!(result, Err("encoder failed mid-chunk"));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn erroring_lane_drops_peer_waiting_on_each_dependency() {
        for dependency in 0..3 {
            let order = PipelineOrder::new(2, 1);
            let entered = Cell::new(false);
            let result = tokio::time::timeout(Duration::from_secs(1), async {
                tokio::try_join!(
                    biased;
                    async {
                        {
                            let first = order.wait_reserve_turn(0).await;
                            if dependency != 0 {
                                first.mark_reserved();
                            }
                            tokio::task::yield_now().await;
                            assert!(entered.get());
                        }
                        Err::<(), _>("encoder failed")
                    },
                    async {
                        entered.set(true);
                        let second = order.wait_reserve_turn(1).await;
                        if dependency == 1 {
                            second.wait_predecessor(0).await;
                        } else {
                            second.wait_commit_turn().await;
                        }
                        panic!("cancelled predecessor released its successor");
                        #[allow(unreachable_code)]
                        Ok::<(), &str>(())
                    }
                )
            })
            .await
            .expect("try_join did not propagate the lane error");
            assert_eq!(result, Err("encoder failed"));
        }
    }
}
