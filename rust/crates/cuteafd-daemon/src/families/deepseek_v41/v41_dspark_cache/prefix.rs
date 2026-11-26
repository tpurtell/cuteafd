//! A dSpark window's part of a prefix snapshot: its ring's initialized rows (at most 128),
//! copied into storage the caller owns (the prefix engine's mark arena), described by the ring's
//! end. A restored request drafts warm.
use super::*;

fn slice(mut buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> CuteafdDeviceBuffer {
    debug_assert!(offset + bytes <= buffer.bytes);
    buffer.ptr = unsafe { buffer.ptr.cast::<u8>().add(offset).cast() };
    buffer.bytes = bytes;
    buffer
}

impl<'a> DsparkWindow<'a> {
    /// Bytes a retained ring of a window ending at `end` holds.
    pub fn ring_bytes(end: u64) -> usize {
        end.min(128) as usize * V41DsparkCache::ROW_BYTES
    }
    /// Copy `lease`'s ring into `destination` (`SLOT_BYTES`) synchronously; returns its end.
    pub fn capture_ring(&mut self, lease: WindowLease, destination: CuteafdDeviceBuffer) -> Result<u64> {
        let end = self.copy_ring(lease, destination, self.stream.raw)?;
        self.synchronize()?;
        Ok(end)
    }
    /// [`DsparkWindow::capture_ring`] queued on capture lane `lane`'s stream; the slot stays
    /// readable only (no writer may run) until the copy lands or is aborted.
    pub fn queue_ring(&mut self, lane: usize, lease: WindowLease, destination: CuteafdDeviceBuffer) -> Result<u64> {
        let copies = self.prefix_copies.get(lane).context("invalid draft snapshot lane")?;
        ensure!(copies.pending.is_none(), "draft snapshot lane is occupied");
        let stream = copies.stream.raw;
        let slot = self.validate(lease)?;
        let mut used = [false; cuteafd_core::V41_MAX_REQUESTS]; used[slot] = true;
        let reservation = self.access.reserve(used)?;
        let end = self.copy_ring(lease, destination, stream)?;
        self.prefix_copies[lane].pending = Some((lease, reservation));
        Ok(end)
    }
    /// Whether lane `lane`'s queued ring copy landed; the lane and its reservation free once
    /// it has.
    pub fn ring_ready(&mut self, lane: usize) -> Result<bool> {
        let copies = self.prefix_copies.get(lane).context("invalid draft snapshot lane")?;
        ensure!(copies.pending.is_some(), "no draft snapshot is pending on this lane");
        let ready = copies.ready()?;
        if ready { self.prefix_copies[lane].pending = None; }
        Ok(ready)
    }
    /// Whether a ring copy is queued on lane `lane`.
    pub fn lane_pending(&self, lane: usize) -> bool {
        self.prefix_copies.get(lane).is_some_and(|copies| copies.pending.is_some())
    }
    pub fn abort_ring(&mut self, lane: usize) -> Result<()> {
        self.prefix_copies.get_mut(lane).context("invalid draft snapshot lane")?.abort()
    }
    fn copy_ring(&self, lease: WindowLease, destination: CuteafdDeviceBuffer, stream: *mut c_void) -> Result<u64> {
        let slot = self.validate(lease)?;
        self.access.readable(slot)?;
        let end = self.slots[slot].end.context("cannot retain an unseeded draft window")?;
        let bytes = Self::ring_bytes(end);
        ensure!(bytes > 0 && bytes <= destination.bytes, "draft ring does not fit its snapshot storage");
        let copied = unsafe {
            self.stream.library.copy_d2d_async(
                slice(destination, 0, bytes),
                slice(self.ring.buffer, slot * V41DsparkCache::SLOT_BYTES, bytes),
                bytes,
                stream,
            )
        };
        if let Err(error) = copied {
            unsafe { self.stream.library.cuda_stream_synchronize(stream)?; }
            return Err(error);
        }
        Ok(end)
    }
    /// Restore a ring captured at `end` from `source` into a fresh window lease.
    pub fn restore_ring(&mut self, lease: WindowLease, end: u64, source: CuteafdDeviceBuffer) -> Result<()> {
        let slot = self.validate(lease)?;
        self.access.writable(slot)?;
        let bytes = Self::ring_bytes(end);
        ensure!(self.slots[slot].end.is_none() && bytes > 0 && bytes <= source.bytes,
            "nonfresh draft window or short ring storage");
        let copied = unsafe {
            self.stream.library.copy_d2d_async(
                slice(self.ring.buffer, slot * V41DsparkCache::SLOT_BYTES, bytes),
                slice(source, 0, bytes),
                bytes,
                self.stream.raw,
            )
        };
        let drained = self.synchronize();
        // Preserve the slot owner on error; an unproved drain poisons the window and
        // must never expose its ring storage to a replacement request.
        copied.and(drained)?;
        self.slots[slot].end = Some(end);
        Ok(())
    }
    pub fn owner(&self) -> u64 {
        self.owner
    }
    pub fn library(&self) -> &'a NativeLibrary {
        self.stream.library
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn poisoned_restore_cleanup_drains_or_retains_without_abort() -> Result<()> {
        const CHILD: &str = "CUTEAFD_DSPARK_DROP_CHILD";
        let Some(mode) = std::env::var_os(CHILD) else {
            for mode in ["recovered", "persistent"] {
                let status = std::process::Command::new(std::env::current_exe()?)
                    .args(["--exact", "families::deepseek_v41::v41_dspark_cache::prefix::tests::poisoned_restore_cleanup_drains_or_retains_without_abort", "--nocapture"])
                    .env(CHILD, mode).status()?;
                ensure!(status.success(), "{mode} poisoned-window cleanup failed or aborted");
            }
            return Ok(());
        };
        let persistent = mode == "persistent";
        let fixture = cuteafd_ffi::native_library_lifetime_fixture::Fixture::build_for_tests()?;
        let lib = fixture.load()?;
        fixture.immediate_copies(&lib)?;
        fixture.configure_memory(&lib, 0, 1 << 30)?;
        let (free_before, _) = lib.cuda_memory_info()?;
        let mut window = DsparkWindow::new(&lib, 1, 8, usize::MAX)?;
        let lease = window.begin_request(0, 1)?;
        let saved = DeviceAllocation::new(&lib, V41DsparkCache::SLOT_BYTES, crate::shared::memory::TEST)?;
        fixture.configure_pack(&lib, 0, 1)?;
        assert!(window.restore_ring(lease, 63, saved.buffer).is_err());
        if !persistent {
            fixture.configure_pack(&lib, 0, 0)?;
            window.synchronize()?;
            assert!(window.begin_request(0, 2).is_err(), "later drain never clears poison");
        }
        let (free_live, _) = lib.cuda_memory_info()?;
        drop(window);
        let (free_dropped, _) = lib.cuda_memory_info()?;
        if persistent {
            assert_eq!(free_dropped, free_live, "unproved ring copies retain all buffers");
            assert!(lib.is_quarantined_after_failed_drain());
        } else {
            assert!(free_dropped > free_live, "proved later drain permits buffer cleanup");
            assert!(!lib.is_quarantined_after_failed_drain());
        }
        drop(saved);
        if !persistent { assert_eq!(lib.cuda_memory_info()?.0, free_before); }
        drop(lib);
        assert_eq!(fixture.resident()?, persistent, "only unproved copies retain the module");
        Ok(())
    }

    #[test]
    fn retained_ring_rows_match_cold_prefill_across_wraps_and_slot_reuse() {
        for end in [1usize, 63, 64, 65, 127, 128, 129, 255, 256, 257, 4097] {
            let row = V41DsparkCache::ROW_BYTES;
            let mut cold = vec![0u8; V41DsparkCache::SLOT_BYTES];
            let first = end.saturating_sub(128);
            for position in first..end {
                cold[(position % 128) * row..(position % 128 + 1) * row]
                    .fill((position % 251 + 1) as u8);
            }
            let bytes = DsparkWindow::ring_bytes(end as u64);
            let saved = cold[..bytes].to_vec();
            let mut reused = vec![0xa5; V41DsparkCache::SLOT_BYTES];
            reused[..bytes].copy_from_slice(&saved);
            for position in first..end {
                let at = (position % 128) * row;
                assert_eq!(&reused[at..at + row], &cold[at..at + row]);
            }
            // The next producer overwrites the same absolute-position ring row.
            let at = (end % 128) * row;
            reused[at..at + row].fill(0x42);
            cold[at..at + row].fill(0x42);
            let initialized = DsparkWindow::ring_bytes((end + 1) as u64);
            assert_eq!(&reused[..initialized], &cold[..initialized]);
        }
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and CUDA"]
    fn native_queued_draft_rings_preserve_peers_and_abort() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let mut window = DsparkWindow::new(&lib, 2, 128, usize::MAX)?;
        let storage = DeviceAllocation::new(&lib, 3 * V41DsparkCache::SLOT_BYTES, crate::shared::memory::TEST)?;
        let at = |i: usize| slice(storage.buffer, i * V41DsparkCache::SLOT_BYTES, V41DsparkCache::SLOT_BYTES);
        for end in [5u64, 128, 129, 1_000_000] {
            let first = window.begin_request(0, 1)?;
            let second = window.begin_request(1, 2)?;
            window.slots[0].end = Some(end); window.slots[1].end = Some(end);
            let bytes = DsparkWindow::ring_bytes(end);
            let original: Vec<u8> = (0..bytes).map(|i| (i % 251) as u8).collect();
            lib.copy_h2d(slice(window.ring.buffer, 0, bytes), &original)?;
            lib.copy_h2d(slice(window.ring.buffer, V41DsparkCache::SLOT_BYTES, bytes), &vec![29; bytes])?;
            assert_eq!(window.capture_ring(first, at(0))?, end);
            window.queue_ring(0, first, at(1))?;
            window.queue_ring(1, second, at(2))?;
            assert!(window.release(first).is_err());
            assert!(window.access.writable(0).is_err());
            assert!(window.queue_ring(1, first, at(2)).is_err());
            while !window.ring_ready(1)? { std::thread::yield_now(); }
            let mut actual = vec![0; bytes];
            lib.copy_d2h_quiescent(&mut actual, slice(at(2), 0, bytes))?;
            assert_eq!(actual, vec![29; bytes]);
            window.queue_ring(1, second, at(2))?;
            window.abort_ring(1)?;
            window.release(second)?;
            assert!(window.release(first).is_err());
            while !window.ring_ready(0)? { std::thread::yield_now(); }
            lib.copy_d2h_quiescent(&mut actual, slice(at(1), 0, bytes))?;
            assert_eq!(actual, original);
            let mut direct = vec![0; bytes];
            lib.copy_d2h_quiescent(&mut direct, slice(at(0), 0, bytes))?;
            assert_eq!(actual, direct);
            window.release(first)?;
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and CUDA"]
    fn native_draft_ring_survives_slot_reuse() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let mut window = DsparkWindow::new(&lib, 2, 128, usize::MAX)?;
        let storage = DeviceAllocation::new(&lib, V41DsparkCache::SLOT_BYTES, crate::shared::memory::TEST)?;
        for end in [5u64, 128, 129, 1000000] {
            let old = window.begin_request(0, 1)?;
            assert!(window.capture_ring(old, storage.buffer).is_err());
            window.slots[0].end = Some(end);
            let bytes = DsparkWindow::ring_bytes(end);
            let original: Vec<u8> = (0..bytes).map(|i| (i % 251) as u8).collect();
            lib.copy_h2d(slice(window.ring.buffer, 0, bytes), &original)?;
            let saved = window.capture_ring(old, storage.buffer)?;
            window.release(old)?;
            let replacement = window.begin_request(0, 2)?;
            lib.copy_h2d(slice(window.ring.buffer, 0, bytes), &vec![0xff; bytes])?;
            let resumed = window.begin_request(1, 3)?;
            window.restore_ring(resumed, saved, storage.buffer)?;
            let mut restored = vec![0; bytes];
            lib.copy_d2h_quiescent(&mut restored, slice(window.ring.buffer, V41DsparkCache::SLOT_BYTES, bytes))?;
            assert_eq!(restored, original);
            assert_eq!(window.committed_end(resumed)?, Some(end));
            assert!(window.validate(old).is_err());
            assert!(window.restore_ring(resumed, saved, storage.buffer).is_err());
            window.release(replacement)?;
            window.release(resumed)?;
        }
        Ok(())
    }
}
