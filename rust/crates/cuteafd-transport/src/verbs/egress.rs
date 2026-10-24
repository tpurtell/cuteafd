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
    views: Arc<()>,
}

// The allocation is owned here and only read through views (or written by the
// engine's D2H copy once no view and no send is left); the raw pointer is the
// only non-Send part.
unsafe impl Send for EgressBuffer {}
unsafe impl Sync for EgressBuffer {}

impl EgressBuffer {
    pub(crate) fn new(bytes: usize) -> Result<Arc<Self>> {
        anyhow::ensure!(bytes > 0, "empty egress buffer");
        let library = load_verbs_host_native_library()?;
        let host = library.alloc_host_buffer(bytes)?;
        Ok(Arc::new(Self { library, host, views: Arc::new(()) }))
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

    /// No payload view is alive (requests holding one have been dropped).
    pub(crate) fn writable(&self) -> bool {
        views_writable(&self.views)
    }

    /// The first `len` bytes as a request payload.
    pub(crate) fn payload(self: &Arc<Self>, len: usize) -> Result<Bytes> {
        anyhow::ensure!(len <= self.host.bytes, "egress payload of {len} bytes exceeds the {}-byte buffer",
            self.host.bytes);
        self.payload_at(0, len)
    }

    /// `len` bytes at `offset` as a request payload.
    pub(crate) fn payload_at(self: &Arc<Self>, offset: usize, len: usize) -> Result<Bytes> {
        anyhow::ensure!(offset.checked_add(len).is_some_and(|end| end <= self.host.bytes),
            "payload of {len} bytes at {offset} exceeds the {}-byte buffer", self.host.bytes);
        Ok(Bytes::from_owner(EgressView { buffer: Arc::clone(self), _view: Arc::clone(&self.views), offset, len }))
    }
}

/// A pinned, device-mapped host arena one prefill wave slot stages into: the
/// engine's stream copies the wave's routes and expert input rows here, and
/// the slot's transport worker builds the request from it, the hidden
/// payload a view of the arena (no host copy). The arena may be rewritten
/// only once no request views it ([`Self::writable`]).
#[derive(Clone)]
pub struct StagingArena(Arc<EgressBuffer>);

impl StagingArena {
    pub fn new(bytes: usize) -> Result<Self> {
        EgressBuffer::new(bytes).map(Self)
    }
    pub(crate) fn of(buffer: &Arc<EgressBuffer>) -> Self { Self(Arc::clone(buffer)) }
    pub fn host(&self) -> CuteafdHostBuffer { self.0.host() }
    pub fn bytes(&self) -> usize { self.0.host.bytes }
    /// No request payload views the arena.
    pub fn writable(&self) -> bool { self.0.writable() }
    /// `len` bytes at `offset` as a request payload that keeps the arena alive.
    pub fn payload(&self, offset: usize, len: usize) -> Result<Bytes> { self.0.payload_at(offset, len) }
    /// The arena's first `len` bytes.
    ///
    /// # Safety
    /// The stream copies that wrote them completed and nothing writes the
    /// arena while the slice lives.
    pub unsafe fn read(&self, len: usize) -> Result<&[u8]> {
        anyhow::ensure!(len <= self.bytes(), "arena read of {len} bytes exceeds the {}-byte arena", self.bytes());
        // SAFETY: in bounds (checked); quiescence is the caller's contract.
        Ok(unsafe { std::slice::from_raw_parts(self.0.host.ptr.cast::<u8>(), len) })
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
