//! Push-only copy-out. Publication remains behind Done and semantic initialization.
#[cfg(test)]
#[path = "push_consumer_tests.rs"]
mod tests;
use super::{
    device::Device,
    weight_arena::{DeliveredArena, LoadingArena},
};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::push_ring::RegisteredRing;
use cuteafd_loader::push_load::{proto::Event, slot, Poll, PushRank};
use std::{
    collections::{BTreeSet, VecDeque},
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
    fn next(&self) -> std::io::Result<Event>;
    fn try_next(&self) -> std::io::Result<Poll>;
    fn release(&self, slot: u32) -> std::io::Result<()>;
}
impl RankEvents for PushRank {
    fn next(&self) -> std::io::Result<Event> {
        self.next()
    }
    fn try_next(&self) -> std::io::Result<Poll> {
        self.try_next()
    }
    fn release(&self, slot: u32) -> std::io::Result<()> {
        self.release(slot)
    }
}

struct Pending {
    slot: u32,
    event: *mut c_void,
    fragments: Vec<(u64, u64)>,
}
struct Copies<'a> {
    device: Device<'a>,
    stream: *mut c_void,
    events: Vec<*mut c_void>,
    pending: VecDeque<Pending>,
    occupied: BTreeSet<u32>,
}
impl<'a> Copies<'a> {
    fn new(device: Device<'a>) -> Result<Self> {
        let stream = device.run(|| device.library.cuda_stream_create())?;
        Ok(Self {
            device,
            stream,
            events: Vec::new(),
            pending: VecDeque::new(),
            occupied: BTreeSet::new(),
        })
    }
    fn complete(&mut self, arena: &mut impl PushDestination, rank: &impl RankEvents) -> Result<()> {
        let pending = self.pending.front().context("empty copy queue")?;
        // SAFETY: event belongs to this stream and follows every copy from its retained slot.
        self.device
            .run(|| unsafe { self.device.library.cuda_event_synchronize(pending.event) })?;
        for &(offset, bytes) in &pending.fragments {
            arena.copied(offset, bytes)?;
        }
        rank.release(pending.slot)?;
        self.occupied.remove(&pending.slot);
        self.pending.pop_front();
        Ok(())
    }
}
impl Drop for Copies<'_> {
    fn drop(&mut self) {
        let library = self.device.library;
        let _ = self
            .device
            .run(|| {
                // SAFETY: stream is live, and ring/arena owners outlive this guard on every exit.
                if let Err(error) = unsafe { library.cuda_stream_synchronize(self.stream) } {
                    library.quarantine_module_after_failed_drain();
                    tracing::error!(%error, "retaining push buffers after failed drain");
                    return Ok(());
                }
                for &event in &self.events {
                    // SAFETY: stream has drained, so events have no pending users.
                    unsafe {
                        library.cuda_event_destroy(event)?;
                    }
                }
                // SAFETY: all copies have drained before stream destruction.
                unsafe { library.cuda_stream_destroy(self.stream) }
            })
            .map_err(|error| {
                library.quarantine_module_after_failed_drain();
                tracing::error!(%error, "push copy teardown failed");
            });
    }
}

/// Own ring and arena through cancellation. No partial load escapes this function.
/// A queued slot has its own event; Empty waits on the oldest rather than spinning.
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
    let mut done = false;
    while !done {
        if copies.pending.len() == window {
            copies.complete(&mut arena, &rank)?;
        }
        let event = if copies.pending.is_empty() {
            rank.next()?
        } else {
            match rank.try_next()? {
                Poll::Ready(event) => event,
                Poll::Empty => {
                    copies.complete(&mut arena, &rank)?;
                    continue;
                }
                Poll::Closed => anyhow::bail!("push closed before Done"),
            }
        };
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
                        .push(device.run(|| device.library.cuda_event_create_ordering())?);
                }
                let event = copies.events[index as usize];
                let mut fragments = Vec::new();
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
                        let dst = arena.destination(usize::try_from(offset)?, device_bytes)?;
                        let src = ring.payload(index, source, device_bytes)?;
                        // SAFETY: checked live arena/ring, retained through event or teardown drain.
                        device.run(|| unsafe {
                            device.library.copy_host_buffer_h2d_async(
                                dst,
                                src,
                                device_bytes,
                                copies.stream,
                            )
                        })?;
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
                // SAFETY: event and stream belong to device and remain retained by Copies.
                device.run(|| unsafe { device.library.cuda_event_record(event, copies.stream) })?;
                copies.pending.push_back(Pending {
                    slot: index,
                    event,
                    fragments,
                });
            }
            other => anyhow::bail!("unexpected push event {other:?}"),
        }
    }
    while !copies.pending.is_empty() {
        copies.complete(&mut arena, &rank)?;
    }
    drop(copies);
    ensure!(
        !device.library.is_quarantined_after_failed_drain(),
        "push copy drain failed"
    );
    drop(rank);
    drop(ring);
    arena.done()
}
