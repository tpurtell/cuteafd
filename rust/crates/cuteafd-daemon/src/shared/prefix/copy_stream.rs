//! One prefix copy stream per rank: snapshot captures, restores, tail copies,
//! replica copies and slot initialization run on it instead of on the compute
//! streams, ordered against those streams by events (`#717` item 14).
//!
//! [`PrefixCopies::ordered`] is a two-sided fence around a batch of copies:
//! every reader stream (each compute stream that reads or writes the rank's
//! KV pages or prefix state) records a `produced` event and the copy stream
//! waits on all of them; the copies run on the copy stream; it records
//! `landed`, and every reader waits on that before its next queued work. A
//! copy therefore sees everything the readers queued before it, and nothing a
//! reader queues after it can overtake it, with no host wait.
//!
//! [`PrefixCopies::drain`] is an empty `ordered` (a fence joining every
//! reader) followed by a host synchronization of the copy stream alone.
//!
//! Safety argument for the prefix cache's drain contract
//! (`cuteafd_engine::prefix::PrefixFamily`: no queued copy, and no queued
//! compute work, may still touch storage the cache releases, stores to the
//! host tier or promotes). The fence makes the copy stream wait for every
//! reader's last queued work, so when the copy stream's synchronization
//! returns, every copy and every reader operation queued before the drain has
//! completed, exactly what a synchronization of each reader stream gave. Work
//! queued after the drain on a reader follows its `landed` wait, so it cannot
//! read a released page before the cache hands that page to someone else
//! (that hand-off happens on the host, after the drain returned).
//!
//! The reader list must name every stream that touches the rank's prefix
//! storage; a stream missing from it is not ordered and would race.
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::NativeLibrary;
use std::cell::RefCell;
use std::ffi::c_void;

/// A raw CUDA stream of `device`, owned elsewhere and live for the copies'
/// lifetime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Reader {
    pub device: i32,
    pub stream: *mut c_void,
}

struct Rank {
    device: i32,
    stream: *mut c_void,
    readers: Vec<*mut c_void>,
    produced: Vec<*mut c_void>,
    landed: *mut c_void,
}

/// One copy stream and its join events per rank (see the module docs).
pub(crate) struct PrefixCopies<'a> {
    library: &'a NativeLibrary,
    ranks: Vec<RefCell<Rank>>,
}

impl<'a> PrefixCopies<'a> {
    /// A copy stream on each rank's device, ordered against that rank's
    /// `readers` (rank order; every reader stream on that rank's device).
    pub(crate) fn new(library: &'a NativeLibrary, readers: Vec<Vec<Reader>>) -> Result<Self> {
        let _memory_scope = cuteafd_ffi::memory_ledger::scope("prefix/copy-stream");
        let previous = library.cuda_get_device()?;
        let mut copies = Self { library, ranks: Vec::with_capacity(readers.len()) };
        let built = (|| -> Result<()> {
            for list in readers {
                let device = list.first().context("a prefix copy rank has no reader stream")?.device;
                ensure!(list.iter().all(|reader| reader.device == device && !reader.stream.is_null()),
                    "prefix copy readers of one rank must be live streams of one device");
                library.cuda_set_device(device)?;
                let mut rank = Rank { device, stream: std::ptr::null_mut(), readers: Vec::new(),
                    produced: Vec::new(), landed: std::ptr::null_mut() };
                rank.stream = library.cuda_stream_create()?;
                // Pushed before the rest so a later failure still destroys it.
                copies.ranks.push(RefCell::new(rank));
                let mut rank = copies.ranks.last().expect("pushed").borrow_mut();
                rank.landed = library.cuda_event_create_ordering()?;
                for reader in list {
                    if rank.readers.contains(&reader.stream) { continue; }
                    rank.produced.push(library.cuda_event_create_ordering()?);
                    rank.readers.push(reader.stream);
                }
            }
            Ok(())
        })();
        library.cuda_set_device(previous)?;
        built.map(|()| copies)
    }

    pub(crate) fn ranks(&self) -> usize { self.ranks.len() }

    /// Runs `copies` on rank `rank`'s copy stream (the argument), fenced
    /// against every reader on both sides; `copies` runs with the rank's
    /// device current. Even when `copies` fails part way, the `landed` join is
    /// still queued, so whatever it queued stays ordered before every reader.
    pub(crate) fn ordered<T>(&self, rank: usize, copies: impl FnOnce(*mut c_void) -> Result<T>) -> Result<T> {
        let state = self.ranks.get(rank).context("prefix copy rank out of range")?.borrow();
        let library = self.library;
        let previous = library.cuda_get_device()?;
        library.cuda_set_device(state.device)?;
        let result = (|| -> Result<T> {
            for (&event, &reader) in state.produced.iter().zip(&state.readers) {
                // SAFETY: the reader streams are live for this owner's lifetime
                // (constructor contract), and the events were created here on
                // their device; recording and waiting only order device work.
                unsafe {
                    library.cuda_event_record(event, reader)?;
                    library.cuda_stream_wait_event(state.stream, event)?;
                }
            }
            let result = copies(state.stream);
            // SAFETY: as above; the landed join orders every reader after
            // whatever the copies queued, including a partial failure's.
            unsafe {
                library.cuda_event_record(state.landed, state.stream)?;
                for &reader in &state.readers {
                    library.cuda_stream_wait_event(reader, state.landed)?;
                }
            }
            result
        })();
        library.cuda_set_device(previous)?;
        result
    }

    /// Queues device-to-device `copies` `(rank, destination, source)`, one
    /// fenced batch per rank that has any (rank order, then queue order).
    ///
    /// # Safety
    /// Every view lies inside a live allocation of its rank's device and stays
    /// allocated until a later [`Self::drain`]; destinations of one batch do
    /// not overlap a source or another destination of it.
    pub(crate) unsafe fn copy_d2d(&self, copies: &[(usize, cuteafd_ffi::CuteafdDeviceBuffer,
        cuteafd_ffi::CuteafdDeviceBuffer)]) -> Result<()> {
        for rank in 0..self.ranks.len() {
            if !copies.iter().any(|&(owner, ..)| owner == rank) { continue; }
            self.ordered(rank, |stream| {
                for &(_, dst, src) in copies.iter().filter(|&&(owner, ..)| owner == rank) {
                    ensure!(dst.bytes == src.bytes, "prefix copy extents differ");
                    // SAFETY: forwarded from this function's contract; the
                    // fence orders the copy against every reader stream.
                    unsafe { self.library.copy_d2d_async(dst, src, src.bytes, stream)? };
                }
                Ok(())
            })?;
        }
        Ok(())
    }

    /// Waits for every copy queued so far and for every reader's work queued
    /// before this call (see the module docs); never synchronizes a reader.
    pub(crate) fn drain(&self) -> Result<()> {
        let mut failures = Vec::new();
        for rank in 0..self.ranks.len() {
            let drained = self.ordered(rank, |_| Ok(())).and_then(|()| self.synchronize(rank));
            if let Err(error) = drained { failures.push(format!("rank {rank}: {error:#}")); }
        }
        ensure!(failures.is_empty(), "prefix copy drain failed: {}", failures.join("; "));
        Ok(())
    }

    /// Waits for rank `rank`'s queued copies only (no reader fence): what a
    /// host read of bytes those copies wrote needs.
    pub(crate) fn synchronize(&self, rank: usize) -> Result<()> {
        let state = self.ranks.get(rank).context("prefix copy rank out of range")?.borrow();
        let previous = self.library.cuda_get_device()?;
        self.library.cuda_set_device(state.device)?;
        // SAFETY: the copy stream was created by this owner and is live.
        let result = unsafe { self.library.cuda_stream_synchronize(state.stream) };
        self.library.cuda_set_device(previous)?;
        result
    }
}

impl Drop for PrefixCopies<'_> {
    fn drop(&mut self) {
        let previous = self.library.cuda_get_device();
        for rank in &self.ranks {
            let rank = rank.borrow();
            let _ = self.library.cuda_set_device(rank.device);
            if !rank.stream.is_null() {
                // SAFETY: created by this owner; queued copies retire before the
                // stream and events are destroyed.
                crate::shared::decode_graph::fatal_drain(
                    unsafe { self.library.cuda_stream_synchronize(rank.stream) }, "prefix copy stream teardown");
                // SAFETY: as above, drained just now.
                unsafe { let _ = self.library.cuda_stream_destroy(rank.stream); }
            }
            for &event in rank.produced.iter().chain((!rank.landed.is_null()).then_some(&rank.landed)) {
                // SAFETY: created by this owner; every wait on it was queued on a
                // stream that drained or is drained by its own owner before reuse.
                unsafe { let _ = self.library.cuda_event_destroy(event); }
            }
        }
        if let Ok(previous) = previous { let _ = self.library.cuda_set_device(previous); }
    }
}

/// Source-contract checks shared by every family's tests: a family's
/// `PrefixFamily::drain` body (between `fn drain(` and the next `fn `) must
/// drain through [`PrefixCopies`], never a compute stream.
#[cfg(test)]
pub(crate) fn drain_body(source: &str) -> &str {
    let start = source.find("fn drain(&self) -> Result<(), BoxError>").expect("a PrefixFamily drain");
    let body = &source[start + 3..];
    &body[..body.find("\n    fn ").unwrap_or(body.len())]
}

#[cfg(test)]
mod tests {
    use super::drain_body;

    /// Every `PrefixFamily` implementation drains its copy stream (with the
    /// reader fence), not the engine, lane or bank streams themselves.
    #[test]
    fn no_family_drain_synchronizes_a_compute_stream() {
        // (family, PrefixFamily source, what its drain must call)
        let families = [
            ("deepseek_v4", include_str!("../../families/deepseek_v4/prefix.rs"), "copies.drain()"),
            ("deepseek_v41", include_str!("../../families/deepseek_v41/v41_native_serve/prefix.rs"), "prefix_drain()"),
            ("glm5", include_str!("../../families/glm5/prefix.rs"), "copies.drain()"),
            ("glm5_flash", include_str!("../../families/glm5_flash/prefix.rs"), "copies.drain()"),
            ("mimo_v2", include_str!("../../families/mimo_v2/prefix.rs"), "copies.drain()"),
            ("qwen3_5", include_str!("../../families/qwen3_5/prefix.rs"), "copies.drain()"),
            ("kimi_k3", include_str!("../../families/kimi_k3/prefix.rs"), "copies.drain()"),
            ("qwen4", include_str!("../../families/qwen4/prefix.rs"), "copies.drain()"),
            // Step-5's per-rank cache owner drains through its own lane fence (checked below).
            ("step5", include_str!("../../families/step5/prefix.rs"), "owner.borrow().drain()"),
        ];
        let forbidden = [["cuda_stream", "_synchronize"].concat(), ["stream", ".drain()"].concat(),
            ["engine", ".synchronize()"].concat(), ["drain", "_state()"].concat(), ["drain", "_prefix_stream"].concat()];
        for (family, source, required) in families {
            let body = drain_body(source);
            assert!(body.contains(required), "{family}: drain does not go through the prefix copy stream:\n{body}");
            for needle in &forbidden {
                assert!(!body.contains(needle.as_str()), "{family}: drain calls {needle}:\n{body}");
            }
        }
        // Step-5: the fence, then the maintenance stream alone; no lane is synchronized.
        let cache = include_str!("../../families/step5/cache.rs");
        let start = cache.find("    pub fn drain(&self) -> Result<()> {").expect("Step-5 cache drain");
        let body = &cache[start..start + cache[start..].find("\n    }\n").expect("drain end")];
        assert!(body.contains("self.ordered(|| Ok(()))?") && body.contains("self.stream.drain()"), "{body}");
        assert!(!body.contains("lane_streams") && !body.contains("upgrade"), "Step-5 drain synchronizes lanes:\n{body}");
        // V4.1: its drain waits on the prefix stream only, and a queued capture joins it on the
        // device rather than draining it.
        let v41 = include_str!("../../families/deepseek_v41/v41_backbone_cache/prefix.rs");
        let start = v41.find("pub fn prefix_drain(&self)").expect("V4.1 prefix drain");
        assert!(v41[start..start + 200].contains("self.prefix_stream.raw"));
        let serve = include_str!("../../families/deepseek_v41/v41_native_serve/prefix.rs");
        let queued = &serve[serve.find("fn queue_capture(").expect("V4.1 queue_capture")..];
        let queued = &queued[..queued.find("\n    fn ").unwrap_or(queued.len())];
        assert!(!queued.contains("prefix_drain") && !queued.contains("_synchronize"), "{queued}");
    }
}
