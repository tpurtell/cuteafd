//! Final response slots stay unposted until their GPU consumers release them.
use super::*;

pub(super) struct RegisteredResponseFrame {
    // Keep the MR, allocation and QP alive even if the session is reset.
    _endpoint: Arc<NativeRdmaEndpoint>,
    buffer: CuteafdHostBuffer,
    slot: usize,
    recycle: mpsc::Sender<usize>,
}
// No endpoint operation is exposed through this owner. A completed, unposted
// slot is immutable after publication; drop only returns its index. Last-owner
// endpoint destruction cannot race session operations, and pinned memory is
// portable across threads. The consumer must drain asynchronous reads first.
unsafe impl Send for RegisteredResponseFrame {}
unsafe impl Sync for RegisteredResponseFrame {}
impl RegisteredResponseFrame {
    pub(super) fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.buffer.ptr.cast(), self.buffer.bytes) }
    }
    pub(super) fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.buffer.ptr.cast(), self.buffer.bytes) }
    }
    pub(super) fn payload_host_buffer(&self, start: usize, end: usize) -> CuteafdHostBuffer {
        debug_assert!(start <= end && end <= self.buffer.bytes);
        CuteafdHostBuffer {
            ptr: unsafe { self.buffer.ptr.cast::<u8>().add(start).cast() },
            bytes: end - start,
            flags: self.buffer.flags,
        }
    }
}
impl Drop for RegisteredResponseFrame {
    fn drop(&mut self) {
        // A closed channel means the session was reset. The Arc still owns the
        // registration through this drop; no repost occurs on an abandoned QP.
        let _ = self.recycle.send(self.slot);
    }
}

/// Receive WRs complete in post order and a completion's slot is its sequence
/// modulo depth, so slots must go back to the receive queue in the order they
/// were consumed. With several waves outstanding a retained (still read)
/// slot holds every later one, retained or copied, until it is released.
#[derive(Debug)]
pub(super) struct RepostOrder {
    order: VecDeque<usize>,
    held: Vec<bool>,
    released: Vec<bool>,
}
impl RepostOrder {
    pub(super) fn new(depth: usize) -> Self {
        Self { order: VecDeque::with_capacity(depth), held: vec![false; depth], released: vec![false; depth] }
    }
    pub(super) fn held(&self, slot: usize) -> bool { self.held[slot] }
    /// `slot`'s frame stays with a consumer until `release`.
    pub(super) fn retain(&mut self, slot: usize) -> Result<()> {
        anyhow::ensure!(slot < self.held.len() && !self.held[slot], "invalid retained receive slot");
        self.held[slot] = true;
        self.order.push_back(slot);
        Ok(())
    }
    pub(super) fn release(&mut self, slot: usize) -> Result<()> {
        anyhow::ensure!(slot < self.held.len() && self.held[slot] && !self.released[slot],
            "invalid retained receive slot recycle");
        self.released[slot] = true;
        Ok(())
    }
    /// `slot`'s frame was copied out: post it now (`Some`) unless an earlier
    /// slot is still retained, else it queues behind it (`None`).
    pub(super) fn copied(&mut self, slot: usize) -> Result<Option<usize>> {
        anyhow::ensure!(slot < self.held.len() && !self.held[slot], "copied receive slot is retained");
        if self.order.is_empty() { return Ok(Some(slot)); }
        self.retain(slot)?;
        self.released[slot] = true;
        Ok(None)
    }
    /// Slots to post now, oldest first.
    pub(super) fn ready(&mut self) -> Vec<usize> {
        let mut ready = Vec::new();
        while let Some(&slot) = self.order.front().filter(|&&slot| self.released[slot]) {
            self.order.pop_front();
            self.held[slot] = false;
            self.released[slot] = false;
            ready.push(slot);
        }
        ready
    }
    pub(super) fn unposted(&self) -> usize { self.order.len() }
}

pub(super) struct RegisteredResponseRing {
    view: CuteafdRdmaRcEndpointBufferView,
    order: RepostOrder,
    tx: mpsc::Sender<usize>,
    rx: mpsc::Receiver<usize>,
}
impl RegisteredResponseRing {
    pub(super) fn new(endpoint: &NativeRdmaEndpoint, ring: VerbsHostRdmaRing) -> Result<Self> {
        let view = endpoint.recv_buffer_view()?;
        validate_mapped_endpoint_buffer_view(view, ring, "retained response")?;
        anyhow::ensure!(
            view.host_flags & CUTEAFD_HOST_BUFFER_FLAG_PINNED != 0,
            "retained response ring must be CUDA pinned"
        );
        let (tx, rx) = mpsc::channel();
        Ok(Self { view, order: RepostOrder::new(ring.depth), tx, rx })
    }

    fn post(endpoint: &NativeRdmaEndpoint, ring: VerbsHostRdmaRing, slots: Vec<usize>) -> Result<()> {
        for slot in slots {
            endpoint.post_recv_at(ring.slot_offset(slot), ring.slot_capacity_bytes,
                VERBS_HOST_RECV_WR_ID + slot as u64)?;
        }
        Ok(())
    }

    /// Reposts released slots, oldest first ([`RepostOrder`]).
    pub(super) fn drain(&mut self, endpoint: &NativeRdmaEndpoint, ring: VerbsHostRdmaRing) -> Result<()> {
        for slot in self.rx.try_iter() { self.order.release(slot)?; }
        Self::post(endpoint, ring, self.order.ready())
    }

    /// A frame copied out of `slot`: repost now or queue it behind an earlier
    /// retained slot.
    pub(super) fn copied(&mut self, endpoint: &NativeRdmaEndpoint, ring: VerbsHostRdmaRing,
        slot: usize) -> Result<()> {
        if let Some(slot) = self.order.copied(slot)? { return Self::post(endpoint, ring, vec![slot]); }
        self.drain(endpoint, ring)
    }

    /// Before a new request: repost what was released and keep at least one
    /// receive posted for its response.
    pub(super) fn reclaim_before_request(
        &mut self,
        endpoint: &NativeRdmaEndpoint,
        ring: VerbsHostRdmaRing,
    ) -> Result<()> {
        self.drain(endpoint, ring)?;
        anyhow::ensure!(
            self.order.unposted() < self.order.held.len(),
            "every local TP4 response slot is still owned by a consumer"
        );
        Ok(())
    }

    pub(super) fn try_retain_final(
        &mut self,
        endpoint: &Arc<NativeRdmaEndpoint>,
        ring: VerbsHostRdmaRing,
        sequence: usize,
        bytes: usize,
    ) -> Result<Option<RegisteredResponseFrame>> {
        let slot = mapped_ring_slot(self.view, ring, sequence as u64)?;
        anyhow::ensure!(
            bytes <= slot.capacity_bytes && !self.order.held(slot.slot_index),
            "invalid retained receive slot extent or ownership"
        );
        let data = unsafe { std::slice::from_raw_parts(slot.host_ptr.cast_const(), bytes) };
        let view = ExpertProtocolV2ResponseView::parse(data)?;
        // Earlier streamed chunks must be copied and reposted immediately:
        // consumers may retain all uploads until the final reduction drains.
        if view.more_chunks() {
            return Ok(None);
        }
        self.order.retain(slot.slot_index)?;
        Ok(Some(RegisteredResponseFrame {
            _endpoint: Arc::clone(endpoint),
            buffer: CuteafdHostBuffer {
                ptr: slot.host_ptr.cast(),
                bytes,
                flags: self.view.host_flags,
            },
            slot: slot.slot_index,
            recycle: self.tx.clone(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::RepostOrder;

    #[test]
    fn depth_two_slots_recycle_with_two_waves_across_many_rotations() -> anyhow::Result<()> {
        let mut order = RepostOrder::new(2);
        // An eager poll completes A and B before A's upload owner is released.
        order.retain(0)?;
        order.retain(1)?;
        for sequence in 0..4096 {
            let slot = sequence % 2;
            assert_eq!(order.unposted(), 2);
            assert!(order.ready().is_empty());
            order.release(slot)?;
            assert_eq!(order.ready(), [slot]);
            assert_eq!(order.unposted(), 1, "replacement request has no receive slot");
            assert!(order.held(1 - slot), "later wave must keep its response");
            // The replacement response completes before the other wave's
            // consumer releases it: the receive ring stays fully occupied.
            order.retain(slot)?;
        }
        order.release(1)?;
        assert!(order.ready().is_empty(), "later response reposted out of order");
        order.release(0)?;
        assert_eq!(order.ready(), [0, 1]);
        assert_eq!(order.unposted(), 0);
        Ok(())
    }

    #[test]
    fn slots_return_in_consumption_order_across_outstanding_waves() -> anyhow::Result<()> {
        // Wave A's final frame (slot 0) stays with its consumer while wave B's
        // frames (1 copied, 2 retained) arrive behind it.
        let mut order = RepostOrder::new(8);
        order.retain(0)?;
        assert_eq!(order.copied(1)?, None, "a copied slot jumped ahead of a retained one");
        order.retain(2)?;
        assert!(order.ready().is_empty());
        order.release(2)?;
        assert!(order.ready().is_empty(), "slot 2 posted before slot 0");
        order.release(0)?;
        assert_eq!(order.ready(), [0, 1, 2]);
        // Single outstanding wave: copies go straight back.
        assert_eq!(order.copied(3)?, Some(3));
        assert!(order.release(3).is_err() && order.retain(9).is_err());
        order.retain(4)?;
        assert!(order.retain(4).is_err());
        assert_eq!(order.unposted(), 1);
        Ok(())
    }
}
