//! Windowed prefill row planes (PLAN "Prefill contract", #762, #744 (b)).
//!
//! A prefill call of `waves` units cuts one reserve of rows (slot `s` at rows
//! `s * unit`). Each prefill slot keeps its own component
//! objects (host phase, weight bindings, its unit's [`RowWindow`]). Every
//! slot's objects on a GPU share one storage set through `Rc`: the streams,
//! the row planes sized for the reserve, and scratch sized for the widest
//! unit. Decode lanes own private storage at their own widths. Kimi's lane
//! windows the same way (`kimi_k3::device::Lane::window`).
//!
//! # Contract (every component that serves prefill units)
//!
//! **Row planes.** A buffer that another *hook* of the same unit reads is a
//! [`RowPlane`] sized for the reserve (`waves * unit` rows): the
//! HC residual and boundary state, the FFN input, router ids/weights/wire,
//! the shared result, window proposals, compressor destinations and taps. A
//! unit only touches its own rows (`RowWindow`), so in-flight units never
//! share a byte of a row plane.
//!
//! **Scratch.** A buffer written and fully consumed inside one hook is
//! scratch, sized once for the unit (`PrefillDepth::unit_rows`)
//! and shared by all slots. There are two kinds:
//! - Plain scratch: written and read only on the component's own shared
//!   stream (GEMM workspaces, split-K partials). That stream's order is the
//!   guard.
//! - Guarded scratch: also read by other streams inside the hook, the fork
//!   readers (query outputs read by the window, compressor, index and sparse
//!   attention), on this GPU or the other. Units sit on different layers and
//!   their stage chains are separate, so unit B's next write on the shared
//!   stream is not ordered after unit A's fork readers. Use a
//!   [`ScratchGuard`]:
//!   - the hook calls `release` after its last reader joined the unit's chain
//!     (`chain::finish` merges forks into the head; the guard records the
//!     joined head, on whichever GPU holds it);
//!   - every writer calls `acquire` (a stream wait, cross-device if needed)
//!     before writing.
//!
//!   A reader that runs in a *later* hook makes the buffer a row plane, not
//!   guarded scratch.
//!
//! **Pinned staging.** A pinned buffer an async H2D reads is never rewritten
//! until that copy has completed. Either it is per unit (a row plane in host
//! memory), or the rewrite is ordered after the copy (stream order on the same
//! stream, or a `ScratchGuard`). Never rewrite it on the host while a queued
//! copy may still read it (`v41_window` `commit_staging`, #815).
//!
//! **Who records, who waits.** Within a unit, stages order on the unit's stage
//! chain (`chain::join`/`finish`). Across units, each slot records its
//! per-layer publication (`chain::record_into`), and the next unit's same
//! layer waits on it (`chain::depend`). Plain scratch needs nothing more,
//! because of the shared stream order. Guarded scratch adds its
//! release/acquire. Row planes need nothing, because they are disjoint. A
//! slot is drained before it is reused (`begin_unit`).
//!
//! **Host state.** Phase fields (begun, ready, pending, tokens, prepared
//! descriptors) and per-layer weight bindings stay on the slot's own
//! component object, never in the shared storage: units interleave between
//! hooks at different layers. A slot selects its window at rest
//! (`begin_unit`) and keeps it for the whole unit.
//!
//! **Graphs.** Windowed launches are eager. Graphs bake row-0 pointers and
//! serve decode/verify widths only, outside any window (`RowWindow::NONE`).
use anyhow::{ensure, Result};
use cuteafd_ffi::CuteafdDeviceBuffer;
use std::ops::Range;

/// A slot's window changed while one of its owners still holds pending
/// work (an ordering bug: the slot was not drained and at rest). Finished
/// outputs are dropped at rest; pending work always refuses, loudly.
#[derive(Debug, thiserror::Error)]
#[error("{owner} window changed while {state} (the slot is not at rest)")]
pub(crate) struct WindowNotAtRest { pub owner: &'static str, pub state: &'static str }

/// Refuses a window change while `pending`.
pub(crate) fn require_rest(pending: bool, owner: &'static str, state: &'static str) -> Result<()> {
    if pending { return Err(WindowNotAtRest { owner, state }.into()); }
    Ok(())
}

/// The widest reserve a row plane spans (three 2752-row units; four 2048-row
/// units span 8192).
pub(crate) const MAX_RESERVE_ROWS: usize = 8256;

fn view(buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
    ensure!(offset.checked_add(bytes).is_some_and(|end| end <= buffer.bytes),
        "row view {offset}+{bytes} past a {}-byte plane", buffer.bytes);
    Ok(CuteafdDeviceBuffer { ptr: buffer.ptr.cast::<u8>().wrapping_add(offset).cast(), bytes, ..buffer })
}

/// The rows a unit owns in every row plane (a call-relative range), or none:
/// decode/verify and graph capture use rows from 0.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct RowWindow(Option<Range<usize>>);

impl RowWindow {
    pub(crate) const NONE: Self = Self(None);
    pub(crate) fn rows(range: Range<usize>) -> Self { Self(Some(range)) }
    pub(crate) fn range(&self) -> Option<&Range<usize>> { self.0.as_ref() }
    /// First row of the window (0 outside one).
    pub(crate) fn start(&self) -> usize { self.0.as_ref().map_or(0, |range| range.start) }
    pub(crate) fn is_windowed(&self) -> bool { self.0.is_some() }
    /// A window inside a plane of `capacity` rows.
    pub(crate) fn within(range: Range<usize>, capacity: usize) -> Result<Self> {
        ensure!(range.start < range.end && range.end <= capacity,
            "row window {range:?} outside {capacity} reserve rows");
        Ok(Self(Some(range)))
    }
    /// The window's view of an allocation of `stride` bytes per row that holds
    /// the window (checked by [`Self::within`] against the allocation's rows).
    /// Outside a window, the allocation itself.
    pub(crate) fn apply(&self, buffer: CuteafdDeviceBuffer, stride: usize) -> CuteafdDeviceBuffer {
        match &self.0 {
            None => buffer,
            Some(range) => {
                debug_assert!(range.end * stride <= buffer.bytes, "window past its allocation");
                CuteafdDeviceBuffer { ptr: buffer.ptr.cast::<u8>().wrapping_add(range.start * stride).cast(),
                    bytes: range.len() * stride, ..buffer }
            }
        }
    }
    /// `rows` of work fit this window (exactly its length inside one).
    pub(crate) fn admits(&self, rows: usize) -> bool { self.0.as_ref().is_none_or(|range| range.len() == rows) }
}

/// One cross-hook buffer of `stride` bytes per row over `rows` reserve rows.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RowPlane { buffer: CuteafdDeviceBuffer, stride: usize }

impl RowPlane {
    /// `buffer` holds `rows` rows of `stride` bytes (its allocation).
    pub(crate) fn new(buffer: CuteafdDeviceBuffer, stride: usize) -> Result<Self> {
        ensure!(stride > 0 && buffer.bytes % stride == 0, "row plane of {} bytes is not whole {stride}-byte rows",
            buffer.bytes);
        Ok(Self { buffer, stride })
    }
    pub(crate) fn rows(&self) -> usize { self.buffer.bytes / self.stride }
    pub(crate) fn stride(&self) -> usize { self.stride }
    /// The whole plane (rows from 0): decode, verify and graph capture.
    pub(crate) fn whole(&self) -> CuteafdDeviceBuffer { self.buffer }
    /// `rows` rows at the window's start. The base keeps the plane's alignment
    /// for any stride that is a multiple of 16 bytes.
    pub(crate) fn at(&self, window: &RowWindow, rows: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(window.admits(rows), "{rows} rows outside window {:?}", window.range());
        view(self.buffer, window.start() * self.stride, rows * self.stride)
    }
    /// The window's whole extent (all `rows` admitted rows from its start;
    /// outside a window, the plane).
    pub(crate) fn window(&self, window: &RowWindow) -> Result<CuteafdDeviceBuffer> {
        match window.range() {
            Some(range) => view(self.buffer, range.start * self.stride, range.len() * self.stride),
            None => Ok(self.buffer),
        }
    }
}

/// Unit-cap scratch that streams other than its writer read inside one hook
/// (fork readers on this GPU or the other one), shared by every slot.
///
/// - The hook that used it calls [`Self::release`] after its last reader
///   joined the unit's stage chain (`chain::finish` merges the forks into the
///   head). That records the joined head: every reader, on either GPU.
/// - The next writer, in any slot, calls [`Self::acquire`] on its stream
///   before writing. That is a stream wait on the recorded event, cross-device
///   when the head was on the other GPU.
///
/// There are no host waits. Outside a chain scope both are no-ops: host-ordered
/// passes drain every stage.
pub(crate) struct ScratchGuard<'a> {
    library: &'a cuteafd_ffi::NativeLibrary,
    /// One ordering event per device a chain head may sit on.
    events: Vec<(i32, *mut std::ffi::c_void)>,
    /// The event holding the last release, if any.
    recorded: std::cell::Cell<Option<usize>>,
}

impl<'a> ScratchGuard<'a> {
    pub(crate) fn new(library: &'a cuteafd_ffi::NativeLibrary, devices: &[i32]) -> Result<Self> {
        ensure!(!devices.is_empty(), "scratch guard needs a device");
        let previous = library.cuda_get_device()?;
        let mut guard = Self { library, events: Vec::with_capacity(devices.len()), recorded: std::cell::Cell::new(None) };
        for &device in devices {
            let created = library.cuda_set_device(device).and_then(|()| library.cuda_event_create_ordering());
            match created {
                Ok(event) => guard.events.push((device, event)),
                Err(error) => { let _ = library.cuda_set_device(previous); return Err(error); }
            }
        }
        library.cuda_set_device(previous)?;
        Ok(guard)
    }
    /// Orders `writer` (a live stream on the current device) after the last
    /// release.
    /// # Safety
    /// `writer` is a live stream of the current device.
    pub(crate) unsafe fn acquire(&self, writer: *mut std::ffi::c_void) -> Result<()> {
        let Some(index) = self.recorded.get() else { return Ok(()) };
        // SAFETY: caller contract; the event outlives the wait (owned here),
        // and cross-device event waits are permitted.
        unsafe { self.library.cuda_stream_wait_event(writer, self.events[index].1) }
    }
    /// Non-blocking check for whether the last recorded release event has not
    /// yet signaled on the GPU (zero host sync).
    pub(crate) fn unresolved(&self) -> Result<bool> {
        let Some(index) = self.recorded.get() else { return Ok(false) };
        let (device, event) = self.events[index];
        // SAFETY: owned by `self` on `device`.
        unsafe { super::chain::event_unresolved(self.library, device, event) }
    }
    /// Records the joined chain head: the scratch is free once it completes.
    /// Call after the hook's last reader finished on the chain.
    pub(crate) fn release(&self) -> Result<()> {
        if !super::chain::active() { return Ok(()); }
        let Some(device) = super::chain::head_device() else { return Ok(()) };
        let index = self.events.iter().position(|&(d, _)| d == device)
            .ok_or_else(|| anyhow::anyhow!("scratch guard has no event on chain head device {device}"))?;
        // SAFETY: the event lives on the head's device (record_into contract).
        if unsafe { super::chain::record_into(self.library, self.events[index].1)? } {
            self.recorded.set(Some(index));
        }
        Ok(())
    }
}

impl Drop for ScratchGuard<'_> {
    fn drop(&mut self) {
        for &(_, event) in &self.events {
            // SAFETY: owned; the owner drains its streams before dropping guards.
            if let Err(error) = unsafe { self.library.cuda_event_destroy(event) } {
                tracing::error!(%error, "destroying scratch guard event");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(rows: usize, stride: usize) -> RowPlane {
        RowPlane::new(CuteafdDeviceBuffer { ptr: 0x1000 as *mut _, bytes: rows * stride, device_id: 0, ..Default::default() }, stride).unwrap()
    }

    #[test]
    fn windows_tile_the_reserve_at_every_pinned_depth() -> Result<()> {
        use cuteafd_loader::serving_capacity::prefill_waves::unit_rows;
        for k in 1..=4u64 {
            let unit = unit_rows(k) as usize;
            let reserve = k as usize * unit;
            assert!(reserve <= MAX_RESERVE_ROWS, "depth {k} exceeds the row planes");
            let p = plane(reserve, 40960);
            let mut end = 0;
            for slot in 0..k as usize {
                let window = RowWindow::rows(slot * unit..(slot + 1) * unit);
                let view = p.at(&window, unit)?;
                assert_eq!(view.ptr as usize, 0x1000 + slot * unit * 40960);
                assert_eq!(view.bytes, unit * 40960);
                end = (slot + 1) * unit;
            }
            assert!(end <= reserve, "depth {k} overruns the reserve");
        }
        Ok(())
    }

    #[test]
    fn apply_offsets_by_window_and_keeps_the_allocation_outside_one() -> Result<()> {
        let buffer = CuteafdDeviceBuffer { ptr: 0x1000 as *mut _, bytes: 100 * 16, device_id: 1, ..Default::default() };
        let window = RowWindow::within(10..30, 100)?;
        let view = window.apply(buffer, 16);
        assert_eq!((view.ptr as usize, view.bytes, view.device_id), (0x1000 + 160, 320, 1));
        assert_eq!(RowWindow::NONE.apply(buffer, 16).bytes, 1600);
        assert!(RowWindow::within(90..110, 100).is_err());
        assert!(RowWindow::within(5..5, 100).is_err());
        Ok(())
    }

    #[test]
    fn windows_refuse_other_widths_and_the_plane_end() {
        let p = plane(128, 64);
        assert!(p.at(&RowWindow::rows(64..128), 64).is_ok());
        assert!(p.at(&RowWindow::rows(64..128), 32).is_err(), "width differs from the window");
        assert!(p.at(&RowWindow::rows(96..160), 64).is_err(), "window past the plane");
        assert_eq!(p.at(&RowWindow::NONE, 16).unwrap().ptr as usize, 0x1000);
        assert!(RowPlane::new(CuteafdDeviceBuffer { ptr: 0x1000 as *mut _, bytes: 100, device_id: 0, ..Default::default() }, 64).is_err());
    }
}
