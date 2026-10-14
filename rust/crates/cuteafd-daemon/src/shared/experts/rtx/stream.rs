//! The stream a GPU's locally resident routed experts run on ([`ExpertStream`]).
//!
//! A pipeline that queues every unit of a GPU on one stream — GLM 5.3 Flash's prefill lanes, a
//! layer-range engine's segments — runs a local layer's experts behind the attention that shares
//! its stream, while a remote layer's wave overlaps it. Giving the local experts a stream of their
//! own restores that overlap: the launches, their inputs and their arithmetic are unchanged, only
//! which stream runs them, so the result is bit-identical and the per-unit order is kept.
//!
//! [`ExpertStream::begin`] and [`ExpertStream::end`] carry the two ordering edges. `begin(engine)`
//! records `input` on the engine stream where the routes and the expert input rows are complete and
//! makes the expert stream wait on it; `end(engine)` records `output` on the expert stream once a
//! unit's launches are queued and makes the engine stream wait on it, before the first consumer of
//! the routed output. Buffers the expert stream writes must not be read or written by the engine
//! stream's overlapped work: the caller gives it its own scratch where a shared one would be
//! touched.
use anyhow::Result;
use cuteafd_ffi::NativeLibrary;
use std::ffi::c_void;

/// The calls an [`ExpertStream`] makes (the native library's; a fake in tests).
pub(crate) trait ExpertStreamLibrary {
    fn create_stream(&self) -> Result<*mut c_void>;
    /// A timing-disabled event, for ordering only.
    fn create_event(&self) -> Result<*mut c_void>;
    /// # Safety
    /// `event` and `stream` are live objects of one device, with no pending record.
    unsafe fn record(&self, event: *mut c_void, stream: *mut c_void) -> Result<()>;
    /// # Safety
    /// `event` is a live event and `stream` a live stream of the same device.
    unsafe fn wait_event(&self, stream: *mut c_void, event: *mut c_void) -> Result<()>;
    /// # Safety
    /// `stream` is a live stream this library created on the current device.
    unsafe fn synchronize_stream(&self, stream: *mut c_void) -> Result<()>;
    /// # Safety
    /// `stream` is a live stream this library created on the current device; it is not used again.
    unsafe fn destroy_stream(&self, stream: *mut c_void) -> Result<()>;
    /// # Safety
    /// `event` is a live event this library created on the current device; no queued work waits on
    /// it and it is not used again.
    unsafe fn destroy_event(&self, event: *mut c_void) -> Result<()>;
}

impl ExpertStreamLibrary for NativeLibrary {
    fn create_stream(&self) -> Result<*mut c_void> {
        self.cuda_stream_create()
    }
    fn create_event(&self) -> Result<*mut c_void> {
        self.cuda_event_create_ordering()
    }
    unsafe fn record(&self, event: *mut c_void, stream: *mut c_void) -> Result<()> {
        // SAFETY: the caller vouches for the event and the stream.
        unsafe { self.cuda_event_record(event, stream) }
    }
    unsafe fn wait_event(&self, stream: *mut c_void, event: *mut c_void) -> Result<()> {
        // SAFETY: the caller vouches for the stream and the event.
        unsafe { self.cuda_stream_wait_event(stream, event) }
    }
    unsafe fn synchronize_stream(&self, stream: *mut c_void) -> Result<()> {
        // SAFETY: the caller vouches for the stream.
        unsafe { self.cuda_stream_synchronize(stream) }
    }
    unsafe fn destroy_stream(&self, stream: *mut c_void) -> Result<()> {
        // SAFETY: the caller vouches for the stream.
        unsafe { self.cuda_stream_destroy(stream) }
    }
    unsafe fn destroy_event(&self, event: *mut c_void) -> Result<()> {
        // SAFETY: the caller vouches for the event.
        unsafe { self.cuda_event_destroy(event) }
    }
}

/// A stream of its own for local expert layers, over its own scratch, tied to the engine stream by
/// one event in each direction. Dropping it drains the stream (an error is logged, not lost) and
/// then destroys the stream and both events.
pub(crate) struct ExpertStream<'a, L: ExpertStreamLibrary = NativeLibrary> {
    library: &'a L,
    stream: *mut c_void,
    /// Recorded on the engine stream by `begin`: the expert stream waits for the routes and inputs.
    input: *mut c_void,
    /// Recorded on the expert stream by `end`: the engine stream waits for the routed output.
    output: *mut c_void,
}

impl<'a, L: ExpertStreamLibrary> ExpertStream<'a, L> {
    /// The stream and its two ordering events, on the current device. Declare the owner before
    /// every buffer the stream serves, so it drops first and its queued launches drain while their
    /// operands are alive.
    pub(crate) fn new(library: &'a L) -> Result<Self> {
        let mut this = Self { library, stream: library.create_stream()?, input: std::ptr::null_mut(), output: std::ptr::null_mut() };
        // Either event's failure drops `this`: it drains and destroys the stream, and skips the
        // event that was never created.
        this.input = library.create_event()?;
        this.output = library.create_event()?;
        Ok(this)
    }

    /// The stream local expert layers launch on.
    pub(crate) fn stream(&self) -> *mut c_void {
        self.stream
    }

    /// Drain the stream. The engine stream reaches every completed unit through [`Self::end`]; this
    /// is the backstop for a queueing failure, whose unit no [`Self::end`] follows.
    pub(crate) fn synchronize(&self) -> Result<()> {
        // SAFETY: the stream is live and this owner's.
        unsafe { self.library.synchronize_stream(self.stream) }
    }

    /// Order the expert stream after `engine`'s queued work: the routes and the expert input rows a
    /// local layer's experts read are complete when it runs.
    pub(crate) fn begin(&self, engine: *mut c_void) -> Result<()> {
        // SAFETY: both events and both streams are this device's and live for the engine's
        // lifetime; the events are the caller's ordering edges, recorded here and waited on there.
        unsafe { self.library.record(self.input, engine)? };
        // SAFETY: as above.
        unsafe { self.library.wait_event(self.stream, self.input) }
    }

    /// Order `engine` after the expert launches queued since [`Self::begin`]: their output is
    /// complete when the engine's next consumer of it runs.
    pub(crate) fn end(&self, engine: *mut c_void) -> Result<()> {
        // SAFETY: as in `begin`.
        unsafe { self.library.record(self.output, self.stream)? };
        // SAFETY: as in `begin`; a later `begin` re-recording `input` does not disturb this wait,
        // which is on the event's state as recorded here.
        unsafe { self.library.wait_event(engine, self.output) }
    }
}

impl<L: ExpertStreamLibrary> Drop for ExpertStream<'_, L> {
    fn drop(&mut self) {
        // SAFETY: the stream was created in `new` and the engine stream drained (it waits on every
        // `output`) before this owner is released; each object is destroyed exactly once.
        if let Err(error) = unsafe { self.library.synchronize_stream(self.stream) } {
            tracing::error!(%error, "draining a local experts' stream");
        }
        for event in [self.input, self.output].into_iter().filter(|event| !event.is_null()) {
            // SAFETY: as above; an event the engine recorded no longer has a queued wait (the
            // engine stream drained) and `new`'s early return leaves the others null.
            if let Err(error) = unsafe { self.library.destroy_event(event) } {
                tracing::error!(%error, "destroying a local experts' ordering event");
            }
        }
        // SAFETY: as above; the stream is not used again.
        if let Err(error) = unsafe { self.library.destroy_stream(self.stream) } {
            tracing::error!(%error, "destroying a local experts' stream");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{ensure, Result};
    use std::{cell::Cell, cell::RefCell};

    /// Records every call in order; the calls named in `fail` return an error (the second
    /// `create_event` for `"create_event"`, so the first event still exists when the second fails).
    #[derive(Default)]
    struct Fake { calls: RefCell<Vec<String>>, events: Cell<usize>, fail: Vec<&'static str> }

    impl Fake {
        fn call(&self, name: &'static str, objects: &[*mut c_void]) -> Result<()> {
            let mut line = name.to_owned();
            for object in objects {
                line.push_str(&format!(" {:#x}", *object as usize));
            }
            self.calls.borrow_mut().push(line);
            ensure!(!self.fail.contains(&name), "injected {name} failure");
            Ok(())
        }
    }

    impl ExpertStreamLibrary for Fake {
        fn create_stream(&self) -> Result<*mut c_void> {
            Ok(0x1 as *mut c_void)
        }
        fn create_event(&self) -> Result<*mut c_void> {
            self.events.set(self.events.get() + 1);
            if self.events.get() == 2 {
                self.call("create_event", &[])?;
            } else {
                self.calls.borrow_mut().push("create_event".to_owned());
            }
            Ok((0x1 + self.events.get()) as *mut c_void)
        }
        unsafe fn record(&self, event: *mut c_void, stream: *mut c_void) -> Result<()> {
            self.call("record", &[event, stream])
        }
        unsafe fn wait_event(&self, stream: *mut c_void, event: *mut c_void) -> Result<()> {
            self.call("wait_event", &[stream, event])
        }
        unsafe fn synchronize_stream(&self, stream: *mut c_void) -> Result<()> {
            self.call("synchronize_stream", &[stream])
        }
        unsafe fn destroy_stream(&self, stream: *mut c_void) -> Result<()> {
            self.call("destroy_stream", &[stream])
        }
        unsafe fn destroy_event(&self, event: *mut c_void) -> Result<()> {
            self.call("destroy_event", &[event])
        }
    }

    fn log(fake: &Fake) -> String {
        fake.calls.borrow().join("; ")
    }

    /// The two edges: the unit waits for the engine's inputs, the engine waits for its output.
    #[test]
    fn a_unit_is_tied_to_the_engine_stream_in_both_directions() {
        let fake = Fake::default();
        let stream = ExpertStream::new(&fake).unwrap();
        assert_eq!(stream.stream(), 0x1 as *mut c_void);
        stream.begin(0x2 as *mut c_void).unwrap();
        let began = log(&fake);
        // The engine stream records `input` where the routes and the expert inputs are out and the
        // expert stream waits on it there; the expert stream records `output` once the unit's
        // launches are queued and the engine stream waits on it.
        assert_eq!(began, "create_event; create_event; record 0x2 0x2; wait_event 0x1 0x2");
        stream.end(0x2 as *mut c_void).unwrap();
        assert_eq!(log(&fake),
            "create_event; create_event; record 0x2 0x2; wait_event 0x1 0x2; record 0x3 0x1; \
                wait_event 0x2 0x3");
        drop(stream);
        assert_eq!(log(&fake),
            "create_event; create_event; record 0x2 0x2; wait_event 0x1 0x2; record 0x3 0x1; \
                wait_event 0x2 0x3; synchronize_stream 0x1; destroy_event 0x2; destroy_event 0x3; \
                destroy_stream 0x1");
    }

    /// A failed second event leaves no object behind: the stream is drained and destroyed.
    #[test]
    fn a_failed_event_creation_still_destroys_the_stream() {
        let fake = Fake { fail: vec!["create_event"], ..Default::default() };
        assert!(ExpertStream::new(&fake).is_err());
        assert_eq!(log(&fake),
            "create_event; create_event; synchronize_stream 0x1; destroy_event 0x2; \
                destroy_stream 0x1");
    }
}
