//! Zero-copy request egress. A transport may own one pinned, device-mapped
//! host buffer that the engine writes a wave's expert input into (a D2H copy
//! on its own stream); requests whose hidden payload is a view of that buffer
//! ([`EgressBuffer::payload`]) are sent by every rank's session as a gathered
//! SEND: the header from the session's send-ring slot, the payload straight
//! from the shared buffer. Without it each session copied the whole payload
//! into its own send ring (six 26 MB copies per MiMo V2.6 Pro prefill wave).
use super::*;
use bytes::Bytes;

/// The pinned buffer, freed when the transport and every payload view are gone.
pub(crate) struct EgressBuffer {
    library: Arc<NativeLibrary>,
    host: CuteafdHostBuffer,
    /// Cloned into every payload view; the buffer may be rewritten only when
    /// no view is left ([`Self::writable`]).
    views: Vec<Arc<()>>,
    slot_bytes: usize,
}

// The allocation is owned here and only read through views (or written by the
// engine's D2H copy once no view and no send is left); the raw pointer is the
// only non-Send part.
unsafe impl Send for EgressBuffer {}
unsafe impl Sync for EgressBuffer {}

impl EgressBuffer {
    pub(crate) fn new(slot_bytes: usize, slots: usize) -> Result<Arc<Self>> {
        anyhow::ensure!(slots > 0, "empty egress slot ring");
        let bytes = slot_bytes.checked_mul(slots).context("egress slot extent overflow")?;
        anyhow::ensure!(bytes > 0, "empty egress buffer");
        let library = load_verbs_host_native_library()?;
        let host = library.alloc_host_buffer(bytes)?;
        Ok(Arc::new(Self { library, host, views: (0..slots).map(|_| Arc::new(())).collect(), slot_bytes }))
    }

    pub(crate) fn host(&self) -> CuteafdHostBuffer {
        self.host
    }

    /// Whether `payload` lies inside this buffer; its offset if so.
    pub(crate) fn offset_of(&self, payload: &[u8]) -> Option<usize> {
        let base = self.host.ptr as usize;
        let start = payload.as_ptr() as usize;
        (start >= base && start + payload.len() <= base + self.host.bytes).then(|| start - base)
    }

    /// The slot a payload at buffer offset `offset` lies in.
    pub(crate) fn slot_of(&self, offset: usize) -> usize { offset / self.slot_bytes }

    /// The slot is writable only after its last request owner has gone.
    pub(crate) fn writable(&self, slot: usize) -> bool {
        self.views.get(slot).is_some_and(views_writable)
    }

    pub(crate) fn target(&self, slot: usize) -> Result<CuteafdHostBuffer> {
        anyhow::ensure!(slot < self.views.len(), "egress slot outside its ring");
        Ok(CuteafdHostBuffer {
            // SAFETY: every slot is a disjoint region of the owned allocation.
            ptr: unsafe { self.host.ptr.cast::<u8>().add(slot * self.slot_bytes) }.cast(),
            bytes: self.slot_bytes,
            ..self.host
        })
    }

    /// A payload view retains only its slot, not every slot in the allocation.
    pub(crate) fn payload(self: &Arc<Self>, slot: usize, len: usize) -> Result<Bytes> {
        anyhow::ensure!(slot < self.views.len() && len <= self.slot_bytes,
            "egress payload exceeds its slot");
        Ok(Bytes::from_owner(EgressView { buffer: Arc::clone(self), _view: Arc::clone(&self.views[slot]),
            offset: slot * self.slot_bytes, len }))
    }

}

/// A pinned, device-mapped host region one prefill wave slot stages into:
/// the engine's stream copies the wave's routes and expert input rows here,
/// and the slot's transport worker builds the request from it, the hidden
/// payload a view of the region. When the region is a slot of a link's
/// registered egress ring, every rank sends (or RDMA-WRITEs) it straight from
/// these bytes: no per-rank copy into a send ring. The region may be
/// rewritten only once no request views it ([`Self::writable`]) and no send
/// still reads it (the slot's wave landed, so every rank received it).
#[derive(Clone)]
pub struct StagingArena { buffer: Arc<EgressBuffer>, slot: usize }

impl StagingArena {
    /// A standalone pinned arena of `bytes` (not registered on any session).
    pub fn new(bytes: usize) -> Result<Self> {
        Ok(Self { buffer: EgressBuffer::new(bytes, 1)?, slot: 0 })
    }
    /// Slot `slot` of a link's registered egress ring.
    pub(crate) fn of(buffer: &Arc<EgressBuffer>, slot: usize) -> Result<Self> {
        anyhow::ensure!(slot < buffer.views.len(), "egress slot outside its ring");
        Ok(Self { buffer: Arc::clone(buffer), slot })
    }
    pub fn host(&self) -> CuteafdHostBuffer {
        self.buffer.target(self.slot).expect("validated egress slot")
    }
    pub fn bytes(&self) -> usize { self.buffer.slot_bytes }
    /// No request payload views the arena.
    pub fn writable(&self) -> bool { self.buffer.writable(self.slot) }
    /// `len` bytes at `offset` as a request payload that keeps the arena alive.
    pub fn payload(&self, offset: usize, len: usize) -> Result<Bytes> {
        anyhow::ensure!(offset.checked_add(len).is_some_and(|end| end <= self.bytes()),
            "payload of {len} bytes at {offset} exceeds the {}-byte arena", self.bytes());
        Ok(Bytes::from_owner(EgressView { buffer: Arc::clone(&self.buffer),
            _view: Arc::clone(&self.buffer.views[self.slot]), offset: self.slot * self.buffer.slot_bytes + offset, len }))
    }
    /// The arena's first `len` bytes.
    ///
    /// # Safety
    /// The stream copies that wrote them completed and nothing writes the
    /// arena while the slice lives.
    pub unsafe fn read(&self, len: usize) -> Result<&[u8]> {
        anyhow::ensure!(len <= self.bytes(), "arena read of {len} bytes exceeds the {}-byte arena", self.bytes());
        // SAFETY: in bounds (checked); quiescence is the caller's contract.
        Ok(unsafe { std::slice::from_raw_parts(self.host().ptr.cast::<u8>(), len) })
    }
}

fn views_writable(views: &Arc<()>) -> bool {
    Arc::strong_count(views) == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Payload {
        bytes: Vec<u8>,
        _view: Arc<()>,
    }
    impl AsRef<[u8]> for Payload {
        fn as_ref(&self) -> &[u8] { &self.bytes }
    }

    #[test]
    fn outstanding_request_clones_block_egress_reuse_until_last_reply() {
        let views = Arc::new(());
        assert!(views_writable(&views));
        let request = Bytes::from_owner(Payload { bytes: vec![1; 16], _view: views.clone() });
        let pending: Vec<_> = (0..4).map(|_| request.clone()).collect();
        drop(request);
        // Dropping the dispatch-local request does not release pending replies.
        assert!(!views_writable(&views));
        let mut pending = pending.into_iter();
        for _ in 0..3 { drop(pending.next()); }
        assert!(!views_writable(&views));
        drop(pending);
        assert!(views_writable(&views));
    }
    #[test]
    fn depth_two_and_three_payload_slots_rotate_without_releasing_neighbors() {
        for depth in [2, 3] {
            let views: Vec<_> = (0..depth).map(|_| Arc::new(())).collect();
            let mut requests: Vec<Option<Bytes>> = (0..depth).map(|_| None).collect();
            for sequence in 0..4096 {
                let slot = sequence % depth;
                drop(requests[slot].take());
                assert!(views_writable(&views[slot]));
                let bytes = vec![(sequence % 251) as u8; 16];
                requests[slot] = Some(Bytes::from_owner(Payload { bytes, _view: views[slot].clone() }));
                for (other, pending) in requests.iter().enumerate() {
                    assert_eq!(views_writable(&views[other]), pending.is_none());
                }
                assert_eq!(requests[slot].as_ref().unwrap()[0], (sequence % 251) as u8);
            }
            drop(requests);
            assert!(views.iter().all(views_writable));
        }
    }

}

impl Drop for EgressBuffer {
    fn drop(&mut self) {
        let _ = self.library.free_host_buffer(&mut self.host);
    }
}

struct EgressView {
    buffer: Arc<EgressBuffer>,
    _view: Arc<()>,
    offset: usize,
    len: usize,
}

impl AsRef<[u8]> for EgressView {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: the allocation lives as long as `buffer`; the engine writes
        // it only while no view exists (`EgressBuffer::writable`), so the
        // bytes are not mutated while this shared view is alive.
        unsafe { std::slice::from_raw_parts(self.buffer.host.ptr.cast::<u8>().add(self.offset), self.len) }
    }
}
