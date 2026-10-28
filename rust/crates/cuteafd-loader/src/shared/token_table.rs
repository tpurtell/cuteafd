//! Shared bounded token-table work: priority gathers and completion-held staging.
use anyhow::{ensure, Context, Result};
use crate::{GatherLease, GatherTicket, GatherWorker, MappedTable, GatherPool};
use std::sync::Mutex;

/// Small gathers stay on the persistent owner; large mmap gathers retain fault
/// parallelism. io_uring retains its batched backend at either size.
pub const PARALLEL_GATHER_BYTES: usize = 1 << 20;

pub struct RowGather { pool: GatherPool }
impl RowGather {
    pub fn new(name: &str, threads: usize) -> Result<Self> { Ok(Self { pool: GatherPool::new(name, threads)? }) }
    pub fn parallel(bytes: usize) -> bool { bytes >= PARALLEL_GATHER_BYTES }
    pub fn gather(&self, table: &MappedTable, rows: &[u64], output: &mut [u8]) -> Result<()> {
        if Self::parallel(output.len()) { self.pool.gather(table, rows, output)?; }
        else {
            let started = std::time::Instant::now();
            table.gather_into(rows, output)?;
            table.stats().record_gather(rows.len(), output.len(), started.elapsed(), [0; 3]);
        }
        Ok(())
    }
    pub fn gather_slots(&self, table: &MappedTable, rows: &[u64], slots: &[usize], output: &mut [u8]) -> Result<()> {
        if Self::parallel(rows.len().saturating_mul(table.row_bytes())) { self.pool.gather_slots(table, rows, slots, output)?; }
        else { table.gather_slots(rows, slots, output)?; }
        Ok(())
    }

}

/// Exact active + equally deep lookahead slots, independently per consumer.
#[derive(Clone, Copy, Debug)]
pub struct TableShape { pub rows: usize, pub depth: usize, pub consumers: usize }
impl TableShape {
    pub fn new(rows: usize, depth: usize, consumers: usize) -> Result<Self> {
        ensure!(rows > 0 && (1..=4).contains(&depth) && consumers > 0, "invalid token-table shape");
        let shape = Self { rows, depth, consumers };
        shape.slots()?.checked_mul(rows).context("token-table row extent overflow")?;
        Ok(shape)
    }
    pub fn slots(self) -> Result<usize> { self.depth.checked_mul(2).and_then(|n| n.checked_mul(self.consumers)).context("token-table slot overflow") }
    pub fn resident_bytes(self, slot_bytes: usize) -> Result<usize> { self.slots()?.checked_mul(slot_bytes).context("token-table staging overflow") }
}

pub struct CopyPool<L> { pending: Vec<L>, capacity: usize }
impl<L> CopyPool<L> {
    pub fn new(capacity: usize) -> Self { Self { pending: Vec::with_capacity(capacity), capacity } }
    pub fn len(&self) -> usize { self.pending.len() }
    pub fn is_empty(&self) -> bool { self.pending.is_empty() }
    pub fn retain(&mut self, lease: L) {
        assert!(self.pending.len() < self.capacity, "copy pool exceeds its gather slots");
        self.pending.push(lease);
    }
    pub fn capacity_stalled(&self, capacity_available: bool) -> bool { !capacity_available && !self.pending.is_empty() }
    pub fn reclaim(&mut self, mut ready: impl FnMut(&mut L) -> Result<bool>) -> Result<()> {
        let mut index = 0;
        while index < self.pending.len() {
            if ready(&mut self.pending[index])? { self.pending.swap_remove(index); }
            else { index += 1; }
        }
        Ok(())
    }
}

/// Family adapters describe a job and native slot. All scheduling, cancellation,
/// priority, admission and deferred-copy retention live here.
pub struct TablePipeline<S: Send + 'static, J: Send + 'static> {
    worker: GatherWorker<S, J, anyhow::Error>,
    copies: Mutex<CopyPool<GatherLease<S, J>>>,
    copy_ready: fn(&mut S) -> Result<bool>,
}
impl<S: Send + 'static, J: Send + 'static> TablePipeline<S, J> {
    pub fn new(name: &str, slots: Vec<S>, gather: impl FnMut(&mut S, &J) -> Result<()> + Send + 'static,
        copy_ready: fn(&mut S) -> Result<bool>) -> Result<Self> {
        let capacity = slots.len();
        Ok(Self { worker: GatherWorker::new(name, slots, gather)?, copies: Mutex::new(CopyPool::new(capacity)), copy_ready })
    }
    pub fn reclaim_copies(&self) -> Result<()> {
        self.copies.lock().unwrap_or_else(|p| p.into_inner()).reclaim(|lease|
            (self.copy_ready)(lease.slot_mut().context("empty table lease")?))
    }
    pub fn retain_copy(&self, lease: GatherLease<S, J>) { self.copies.lock().unwrap_or_else(|p| p.into_inner()).retain(lease); }
    pub fn copy_capacity_stalled(&self) -> Result<bool> {
        Ok(self.copies.lock().unwrap_or_else(|p| p.into_inner()).capacity_stalled(self.worker.capacity_available()?))
    }
    pub fn try_submit(&self, job: J, priority: bool) -> Result<Option<GatherTicket<S, J, anyhow::Error>>> {
        self.reclaim_copies()?;
        Ok(self.worker.try_submit_with(job, tracing::enabled!(target: "cuteafd::timing", tracing::Level::DEBUG), priority)?)
    }
    /// Only synchronous consumer boundaries use this; early submission never waits.
    pub fn submit_timeout(&self, job: J, priority: bool, timeout: std::time::Duration) -> Result<GatherTicket<S, J, anyhow::Error>> {
        use std::{future::Future, task::{Context, Poll, Wake, Waker}};
        struct ThreadWake(std::thread::Thread);
        impl Wake for ThreadWake { fn wake(self: std::sync::Arc<Self>) { self.0.unpark(); } }
        let waker = Waker::from(std::sync::Arc::new(ThreadWake(std::thread::current())));
        let mut cx = Context::from_waker(&waker);
        let deadline = std::time::Instant::now() + timeout;
        let mut future = std::pin::pin!(self.submit(job, priority));
        loop {
            self.reclaim_copies()?;
            if let Poll::Ready(result) = future.as_mut().poll(&mut cx) { return result; }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            ensure!(!remaining.is_zero(), "token-table admission timed out");
            // CUDA completion does not wake a Rust waker. A bounded sleep lets
            // the owner reclaim copied slots without a hot spin loop.
            std::thread::park_timeout(remaining.min(std::time::Duration::from_millis(1)));
        }
    }
    pub async fn submit(&self, job: J, priority: bool) -> Result<GatherTicket<S, J, anyhow::Error>> {
        self.reclaim_copies()?;
        Ok(self.worker.submit_with(job, tracing::enabled!(target: "cuteafd::timing", tracing::Level::DEBUG), priority).await?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GatherPoll;
    use std::{sync::{Arc, atomic::{AtomicBool, Ordering}}, time::Duration};

    #[test]
    fn cancelled_request_cannot_publish_into_reused_slot() -> Result<()> {
        use std::sync::mpsc;
        let (entered_tx, entered_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        let pipeline = TablePipeline::new("shared-cancel-test", vec![0usize], move |slot, job: &usize| {
            if *job == 11 {
                entered_tx.send(()).unwrap();
                release_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            }
            *slot = *job;
            Ok(())
        }, |_| Ok(true))?;
        let mut old = pipeline.try_submit(11, false)?.unwrap();
        entered_rx.recv_timeout(Duration::from_secs(5))?;
        old.cancel();
        assert!(matches!(old.poll(), Ok(GatherPoll::Cancelled)));
        assert!(pipeline.try_submit(22, true)?.is_none(), "in-progress cancelled source is still owned");
        release_tx.send(())?;
        let mut new = pipeline.submit_timeout(22, true, Duration::from_secs(5))?;
        let GatherPoll::Ready(lease) = new.wait_timeout(Duration::from_secs(5))
            .map_err(|_| anyhow::anyhow!("replacement gather failed"))? else {
            anyhow::bail!("replacement was cancelled");
        };
        assert_eq!(*lease.slot().unwrap(), 22);
        assert_eq!(*lease.job(), 22);
        Ok(())
    }

    #[test]
    fn shared_pipeline_retains_copies_and_recovers_after_failure() -> Result<()> {
        struct Slot { copied: Arc<AtomicBool>, value: usize }
        for depth in 1..=4 {
            let shape = TableShape::new(64, depth, 1)?;
            let done = Arc::new(AtomicBool::new(false));
            let pipeline = TablePipeline::new("shared-lifecycle-test", (0..shape.slots()?).map(|_| Slot {
                copied: done.clone(), value: 0 }).collect(), |slot, value: &usize| {
                    ensure!(*value != usize::MAX, "injected gather error"); slot.value = *value; Ok(())
                }, |slot| Ok(slot.copied.load(Ordering::Acquire)))?;
            for n in 0..shape.slots()? {
                let mut ticket = pipeline.try_submit(n, true)?.expect("exact capacity");
                let GatherPoll::Ready(lease) = ticket.wait_timeout(Duration::from_secs(5)).map_err(|_| anyhow::anyhow!("gather failed"))? else {
                    anyhow::bail!("unexpected cancellation");
                };
                assert_eq!(lease.slot().unwrap().value, n);
                pipeline.retain_copy(lease);
            }
            assert!(pipeline.try_submit(0, true)?.is_none());
            assert!(pipeline.copy_capacity_stalled()?);
            done.store(true, Ordering::Release);
            pipeline.reclaim_copies()?;
            let mut ticket = pipeline.try_submit(usize::MAX, true)?.unwrap();
            assert!(ticket.wait_timeout(Duration::from_secs(5)).is_err());
            let mut ticket = pipeline.try_submit(7, true)?.unwrap();
            assert!(matches!(ticket.wait_timeout(Duration::from_secs(5)), Ok(GatherPoll::Ready(_))));
        }
        assert!(TableShape::new(usize::MAX, 4, 2).is_err());
        Ok(())
    }
}
