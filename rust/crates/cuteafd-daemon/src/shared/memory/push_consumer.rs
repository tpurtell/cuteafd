//! Push-only copy-out. Publication remains behind Done and semantic initialization.
#[cfg(test)]
#[path = "push_consumer_tests.rs"]
mod tests;
use super::{
    device::Device,
    weight_arena::{DeliveredArena, LoadingArena},
};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{event_waiters::EventWaiters, push_ring::RegisteredRing};
use cuteafd_loader::push_load::{proto::Event, slot, Poll, PushRank};
use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::c_void,
};

/// Where a rank's fragments land: a logical address space whose first
/// `device_bytes` map to device memory and the rest to a host tail, with
/// delivered coverage checked against the plan. `LoadingArena` is the Spark
/// rank's single admitted arena; coordinator ranks may map it differently.
pub(crate) trait PushDestination {
    type Delivered;
    fn belongs_to(&self, device: Device<'_>) -> bool;
    fn logical_bytes(&self) -> usize;
    fn device_bytes(&self) -> usize;
    fn check_fragment(&self, offset: u64, bytes: u64) -> Result<()>;
    fn destination(&self, offset: usize, bytes: usize) -> Result<cuteafd_ffi::CuteafdDeviceBuffer>;
    fn copy_validation(&mut self, offset: u64, bytes: &[u8]) -> Result<()>;
    /// Called only after a fragment's copy completed, before its slot is released.
    fn copied(&mut self, offset: u64, bytes: u64) -> Result<()>;
    /// Rank Done observed and every copy drained.
    fn done(self) -> Result<Self::Delivered>;
}

impl<'a> PushDestination for LoadingArena<'a> {
    type Delivered = DeliveredArena<'a>;
    fn belongs_to(&self, device: Device<'_>) -> bool { LoadingArena::belongs_to(self, device) }
    fn logical_bytes(&self) -> usize { LoadingArena::logical_bytes(self) }
    fn device_bytes(&self) -> usize { LoadingArena::device_bytes(self) }
    fn check_fragment(&self, offset: u64, bytes: u64) -> Result<()> { LoadingArena::check_fragment(self, offset, bytes) }
    fn destination(&self, offset: usize, bytes: usize) -> Result<cuteafd_ffi::CuteafdDeviceBuffer> {
        LoadingArena::destination(self, offset, bytes)
    }
    fn copy_validation(&mut self, offset: u64, bytes: &[u8]) -> Result<()> {
        LoadingArena::copy_validation(self, offset, bytes)
    }
    fn copied(&mut self, offset: u64, bytes: u64) -> Result<()> { LoadingArena::copied(self, offset, bytes) }
    fn done(self) -> Result<DeliveredArena<'a>> { LoadingArena::done(self) }
}

pub(crate) trait RankEvents {
    fn fd(&self) -> Option<std::os::fd::BorrowedFd<'_>> { None }
    fn next(&self) -> std::io::Result<Event>;
    /// The group's first event: no hub event (not even Progress) precedes the
    /// group's start, which waits for every rank to join.
    fn first(&self) -> std::io::Result<Event> { self.next() }
    fn try_next(&self) -> std::io::Result<Poll>;
    fn release(&self, slot: u32) -> std::io::Result<()>;
}
impl RankEvents for PushRank {
    fn fd(&self) -> Option<std::os::fd::BorrowedFd<'_>> { Some(self.fd()) }
    fn next(&self) -> std::io::Result<Event> {
        self.next()
    }
    fn first(&self) -> std::io::Result<Event> {
        use cuteafd_loader::push_load::{PUSH_IDLE_TIMEOUT, PUSH_START_TIMEOUT};
        self.first(PUSH_START_TIMEOUT, PUSH_IDLE_TIMEOUT)
    }
    fn try_next(&self) -> std::io::Result<Poll> {
        self.try_next()
    }
    fn release(&self, slot: u32) -> std::io::Result<()> {
        self.release(slot)
    }
}

const COPY_STREAMS: usize = 4;

struct Copies<'a> {
    device: Device<'a>,
    streams: Vec<*mut c_void>,
    next_stream: usize,
    waiters: EventWaiters,
    events: Vec<*mut c_void>,
    pending: BTreeMap<u32, Vec<(u64, u64)>>,
    occupied: BTreeSet<u32>,
}
impl<'a> Copies<'a> {
    fn new(device: Device<'a>) -> Result<Self> {
        let mut copies = Self {
            device,
            streams: Vec::new(),
            next_stream: 0,
            waiters: EventWaiters::new(device.library, device.id, COPY_STREAMS)?,
            events: Vec::new(),
            pending: BTreeMap::new(),
            occupied: BTreeSet::new(),
        };
        for _ in 0..COPY_STREAMS {
            copies.streams.push(device.run(|| device.library.cuda_stream_create())?);
        }
        Ok(copies)
    }
    fn complete(&mut self, arena: &mut impl PushDestination, rank: &impl RankEvents) -> Result<()> {
        // Every stream has a sleeping waiter: one slow stream cannot hold the
        // credits of a different stream whose event has already completed.
        let slot = self.waiters.complete()?;
        let fragments = self.pending.get(&slot).context("unknown completed slot")?;
        for &(offset, bytes) in fragments {
            arena.copied(offset, bytes)?;
        }
        rank.release(slot)?;
        self.occupied.remove(&slot);
        self.pending.remove(&slot);
        Ok(())
    }
}
impl Drop for Copies<'_> {
    fn drop(&mut self) {
        let library = self.device.library;
        if let Err(error) = self.waiters.join() {
            library.quarantine_module_after_failed_drain();
            tracing::error!(%error, "joining push copy waiters");
        }
        let _ = self
            .device
            .run(|| {
                let mut failed = library.is_quarantined_after_failed_drain();
                for &stream in &self.streams {
                    // SAFETY: all streams are live; ring/arena outlive this guard.
                    if let Err(error) = unsafe { library.cuda_stream_synchronize(stream) } {
                        library.quarantine_module_after_failed_drain();
                        tracing::error!(%error, "retaining push buffers after failed drain");
                        failed = true;
                    }
                }
                if failed { return Ok(()); }
                for &event in &self.events {
                    // SAFETY: stream has drained, so events have no pending users.
                    unsafe {
                        library.cuda_event_destroy(event)?;
                    }
                }
                for &stream in &self.streams {
                    // SAFETY: all streams drained and every waiter joined.
                    unsafe { library.cuda_stream_destroy(stream)?; }
                }
                Ok(())
            })
            .map_err(|error| {
                library.quarantine_module_after_failed_drain();
                tracing::error!(%error, "push copy teardown failed");
            });
    }
}

/// Own ring and arena through cancellation. No partial load escapes this function.
/// A queued slot has its own sleeping event wait; completion order is independent across streams.
pub(crate) fn consume<'a, A: PushDestination>(
    device: Device<'a>,
    mut arena: A,
    ring: RegisteredRing<'a>,
    rank: impl RankEvents,
    window: usize,
) -> Result<A::Delivered> {
    ensure!(
        window > 0 && window <= ring.slots(),
        "invalid push copy window"
    );
    ensure!(
        ring.belongs_to(device.library, device.id),
        "push ring device mismatch"
    );
    ensure!(arena.belongs_to(device), "push arena device mismatch");
    let mut copies = Copies::new(device)?;
    // What this rank received, for the cause of any failure below.
    let joined = std::time::Instant::now();
    let (mut slots, mut payload, mut last) = (0u64, 0u64, None::<std::time::Instant>);
    let delivered = |slots: u64, payload: u64, last: Option<std::time::Instant>| match last {
        Some(at) => format!("after {slots} slots ({payload} B) delivered; last event {:.1} s ago, \
            {:.1} s after join", at.elapsed().as_secs_f64(), (at - joined).as_secs_f64()),
        None => format!("with nothing delivered {:.1} s after join", joined.elapsed().as_secs_f64()),
    };
    let mut done = false;
    let mut started = false;
    let looped = (|| -> Result<()> {
    while !done {
        if copies.pending.len() == window {
            copies.complete(&mut arena, &rank)?;
        }
        let event = if !started {
            started = true;
            rank.first()?
        } else if copies.pending.is_empty() {
            rank.next()?
        } else {
            match rank.try_next()? {
                Poll::Ready(event) => event,
                Poll::Empty => {
                    if copies.waiters.wait(rank.fd())? {
                        copies.complete(&mut arena, &rank)?;
                    }
                    continue;
                }
                Poll::Closed => anyhow::bail!("push closed before Done"),
            }
        };
        last = Some(std::time::Instant::now());
        if let Event::Filled { len, .. } = &event { slots += 1; payload += u64::from(*len); }
        match event {
            Event::Progress { .. } => {}
            Event::Done => done = true,
            Event::Filled { slot: index, len } => {
                ensure!(
                    copies.occupied.insert(index),
                    "push reused an unreleased slot"
                );
                // SAFETY: Filled transfers the credit, duplicate credits are rejected,
                // and Release follows the slot's completed event (including teardown).
                let data = unsafe { ring.slot(index, len)? };
                let parsed = slot::parse(data, arena.logical_bytes() as u64)?;
                while copies.events.len() <= index as usize {
                    copies
                        .events
                        .push(device.run(|| device.library.cuda_event_create_blocking())?);
                }
                let event = copies.events[index as usize];
                let mut fragments = Vec::new();
                // One batched copy per slot: a fine-grained plan (thousands of
                // small fragments a slot) must not pay a copy call per fragment.
                let (mut dsts, mut srcs, mut lens) = (Vec::new(), Vec::new(), Vec::new());
                let mut source = 0;
                for (offset, bytes) in slot::Fragments::new(data, parsed) {
                    let bytes = bytes as usize;
                    arena.check_fragment(offset, bytes as u64)?;
                    let device_bytes = bytes.min(
                        arena
                            .device_bytes()
                            .saturating_sub(usize::try_from(offset)?),
                    );
                    if device_bytes > 0 {
                        dsts.push(arena.destination(usize::try_from(offset)?, device_bytes)?);
                        srcs.push(ring.payload(index, source, device_bytes)?);
                        lens.push(device_bytes);
                        fragments.push((offset, device_bytes as u64));
                    }
                    if device_bytes < bytes {
                        arena.copy_validation(
                            offset + device_bytes as u64,
                            &data[source + device_bytes..source + bytes],
                        )?;
                    }
                    source += bytes;
                }
                let stream_index = copies.next_stream;
                copies.next_stream = (stream_index + 1) % COPY_STREAMS;
                let stream = copies.streams[stream_index];
                // SAFETY: checked live arena/ring views, retained through the slot's
                // event or the teardown drain; the copies stream is non-default.
                device.run(|| unsafe {
                    device.library.copy_host_buffers_h2d_batch_async(&dsts, &srcs, &lens, stream)
                })?;
                // SAFETY: event and stream belong to device and remain retained by Copies.
                device.run(|| unsafe { device.library.cuda_event_record(event, stream) })?;
                copies.pending.insert(index, fragments);
                // SAFETY: the event is recorded, is not reused until its completion,
                // and Copies joins all waiters before destroying events or storage.
                unsafe { copies.waiters.submit(stream_index, event, index)?; }
            }
            other => anyhow::bail!("unexpected push event {other:?}"),
        }
    }
    while !copies.pending.is_empty() {
        copies.complete(&mut arena, &rank)?;
    }
    Ok(())
    })();
    looped.map_err(|error| error.context(delivered(slots, payload, last)))?;
    drop(copies);
    ensure!(
        !device.library.is_quarantined_after_failed_drain(),
        "push copy drain failed"
    );
    drop(rank);
    drop(ring);
    arena.done().map_err(|error| error.context(delivered(slots, payload, last)))
}
