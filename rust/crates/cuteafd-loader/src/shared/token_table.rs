//! Shared bounded token-table work: priority gathers and completion-held staging.
use anyhow::{ensure, Context, Result};
use crate::{GatherLease, GatherTicket, GatherWorker};
use std::sync::Mutex;

/// Inclusive upper bound for device-ordered mapped-pinned metadata copies.
/// Larger payloads use the host-triggered copy engine after the previous reader.
pub const METADATA_SM_COPY_BYTES: usize = 4 << 20;

/// Exact pinned bytes and completion records for an admitted upload schedule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UploadShape { pub bytes: usize, pub ranges: usize, largest: usize }
impl UploadShape {
    pub fn append(&mut self, planes: &[usize], repetitions: usize) -> Result<()> {
        for &bytes in planes.iter().filter(|&&bytes| bytes > 0) {
            let aligned = bytes.checked_add(15).map(|n| n / 16 * 16).context("upload alignment overflow")?;
            if repetitions > 0 { self.largest = self.largest.max(aligned); }
            self.bytes = aligned.checked_mul(repetitions).and_then(|n| self.bytes.checked_add(n))
                .context("retained upload bytes overflow")?;
            self.ranges = self.ranges.checked_add(repetitions).context("retained upload fences overflow")?;
        }
        Ok(())
    }
    /// Split the same admitted payload over several calls without multiplying
    /// its bytes. Each extra fragment needs a fence and at most 15 padding bytes.
    pub fn fragment(&mut self, planes: usize, extra_calls: usize) -> Result<()> {
        let ranges = planes.checked_mul(extra_calls).context("upload fragment count overflow")?;
        self.ranges = self.ranges.checked_add(ranges).context("upload fragment fences overflow")?;
        self.bytes = ranges.checked_mul(16).and_then(|n| self.bytes.checked_add(n))
            .context("upload fragment padding overflow")?;
        Ok(())
    }

    /// One circular tail gap can strand fewer than one largest aligned plane.
    pub fn capacity_bytes(self) -> Result<usize> {
        self.bytes.checked_add(self.largest.saturating_sub(16)).context("upload wrap slack overflow")
    }
}

/// CPU ownership model: no range can wrap over an unfinished GPU source read.
/// Completion records are FIFO because one arena serves one consumer stream.
pub struct UploadCursor {
    shape: UploadShape,
    offset: usize,
    pending: std::collections::VecDeque<(usize, std::ops::Range<usize>)>,
    free: Vec<usize>,
}
impl UploadCursor {
    pub fn new(shape: UploadShape) -> Result<Self> {
        ensure!(shape.bytes > 0 && shape.ranges > 0, "empty retained upload arena");
        let shape = UploadShape { bytes: shape.capacity_bytes()?, ..shape };
        Ok(Self { shape, offset: 0, pending: std::collections::VecDeque::with_capacity(shape.ranges),
            free: (0..shape.ranges).rev().collect() })
    }
    pub fn oldest(&self) -> Option<usize> { self.pending.front().map(|entry| entry.0) }
    pub fn retire(&mut self, fence: usize) -> Result<()> {
        ensure!(self.oldest() == Some(fence), "retained upload retired out of source-fence order");
        self.pending.pop_front();
        self.free.push(fence);
        Ok(())
    }
    pub fn reserve(&mut self, bytes: usize) -> Result<Option<(usize, std::ops::Range<usize>)>> {
        ensure!(bytes > 0, "empty retained upload");
        let aligned = bytes.checked_add(15).map(|n| n / 16 * 16).context("upload alignment overflow")?;
        ensure!(aligned <= self.shape.bytes, "upload exceeds retained arena admission");
        if self.pending.is_empty() { self.offset = 0; }
        let at = if self.offset <= self.shape.bytes - aligned { self.offset } else { 0 };
        let range = at..at + aligned;
        let blocked = self.pending.front().is_some_and(|(_, oldest)| {
            if self.offset <= oldest.start { range.start < self.offset || range.end > oldest.start }
            else if range.start == 0 { range.end > oldest.start }
            else { false }
        });
        if self.free.is_empty() || blocked { return Ok(None); }
        let fence = self.free.pop().expect("checked fence capacity");
        self.pending.push_back((fence, range.clone()));
        self.offset = range.end;
        Ok(Some((fence, range)))
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
    pub(crate) fn split<P: Send + 'static>(name: &str, slots: Vec<S>,
        prepare: impl FnMut(&mut S, &J) -> Result<P> + Send + 'static,
        finish: impl Fn(&mut S, &J, P) -> Result<()> + Send + Sync + 'static,
        copy_ready: fn(&mut S) -> Result<bool>) -> Result<Self> {
        let capacity = slots.len();
        Ok(Self { worker: GatherWorker::split(name, slots, prepare, finish)?,
            copies: Mutex::new(CopyPool::new(capacity)), copy_ready })
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
    fn retained_upload_adversarial_schedule_never_reuses_live_sources() -> Result<()> {
        let planes = [4096 * 8, 4096 * 12, 4096 * 12, 4096 * 8, 4096 * 4,
            4096 * 4, 4096 * 8, 4096 * 4, 16384 * 4, 4096 * 4];
        let groups = 3 * (2 + 3 * 48);
        let mut shape = UploadShape::default();
        shape.append(&planes, groups)?;
        shape.append(&[4096 * 4], 3)?;
        assert_eq!(shape.bytes, 143_572_992);
        let mut cursor = UploadCursor::new(shape)?;
        for _ in 0..groups {
            for bytes in planes { assert!(cursor.reserve(bytes)?.is_some()); }
        }
        for _ in 0..3 { assert!(cursor.reserve(4096 * 4)?.is_some()); }
        assert!(cursor.reserve(16)?.is_none(), "unfinished SM source read must refuse wrap");
        let first = cursor.oldest().unwrap();
        assert!(cursor.retire(first + 1).is_err(), "out-of-order event cannot retire source");
        cursor.retire(first)?;
        assert!(cursor.reserve(16)?.is_some());
        assert_eq!(METADATA_SM_COPY_BYTES, 4_194_304);
        assert!(shape.append(&[usize::MAX], 1).is_err());
        Ok(())
    }

    #[test]
    fn fragmented_uploads_charge_fences_and_alignment_without_multiplying_payload() -> Result<()> {
        let mut shape = UploadShape::default();
        shape.append(&[257], 1)?;
        shape.fragment(1, 256)?;
        assert_eq!(shape.ranges, 257);
        assert_eq!(shape.bytes, 272 + 256 * 16);
        let mut cursor = UploadCursor::new(shape)?;
        for _ in 0..257 { assert!(cursor.reserve(1)?.is_some()); }
        assert!(cursor.reserve(1)?.is_none());
        assert!(shape.fragment(usize::MAX, 2).is_err());
        Ok(())
    }

    #[test]
    fn retained_upload_wrap_never_overlaps_an_unfinished_range() -> Result<()> {
        let mut shape = UploadShape::default();
        shape.append(&[16, 32, 48, 64], 1)?;
        assert_eq!(shape.capacity_bytes()?, 208);
        let mut cursor = UploadCursor::new(shape)?;
        let mut live = std::collections::VecDeque::new();
        for n in 0..4096 {
            let bytes = [16, 64, 32, 48][n % 4];
            if let Some((fence, range)) = cursor.reserve(bytes)? {
                assert!(live.iter().all(|(_, old): &(usize, std::ops::Range<usize>)|
                    range.end <= old.start || old.end <= range.start));
                live.push_back((fence, range));
            } else {
                let (fence, _) = live.pop_front().expect("only live sources can block admission");
                cursor.retire(fence)?;
            }
        }
        Ok(())
    }

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
