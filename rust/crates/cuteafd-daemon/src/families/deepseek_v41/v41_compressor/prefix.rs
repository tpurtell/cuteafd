//! A compressed source's part of a prefix snapshot: its committed frontier and, at an odd
//! ratio-two frontier, the pending group's two FP32 projections (the carry). The source rows
//! themselves live in pages the prefix engine owns and forks; nothing here holds pages.
use super::*;

pub(crate) const COMPRESSOR_PREFIX_BYTES: usize = 4096;

fn slice(mut buffer: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> CuteafdDeviceBuffer {
    debug_assert!(offset + bytes <= buffer.bytes);
    buffer.ptr = unsafe { buffer.ptr.cast::<u8>().add(offset).cast() };
    buffer.bytes = bytes;
    buffer
}

impl CompressorState<'_> {
    /// Start a fresh lease at the complete-group frontier `end` of pages the prefix engine
    /// forked and bound: no pending group exists there, so no carry is read. This is the global
    /// backing for a later bounded local-window reconstruction.
    pub fn restore_compressed_prefix(&mut self, lease: CompressorLease, end: u64) -> Result<()> {
        let slot = self.validate(lease)?;
        let step = ratio(self.layer)? as u64;
        let rows = (end / step) as usize;
        ensure!(
            end % step == 0
                && self.slots[slot].end == 0
                && self.slots[slot].version == 0
                && self.index.binding(slot).1 == rows,
            "unaligned or nonfresh compressed prefix restore, or bound rows differ"
        );
        self.slots[slot].end = end;
        self.slots[slot].version = 1;
        Ok(())
    }

    /// # Safety
    /// Producers are drained. Destination remains live until stream completion,
    /// including on error. Only odd ratio-two frontiers have a live pending row.
    pub unsafe fn retain_prefix(
        &self,
        lease: CompressorLease,
        destination: CuteafdDeviceBuffer,
        stream: *mut c_void,
    ) -> Result<u64> {
        let slot = self.validate(lease)?;
        ensure!(
            destination.bytes == COMPRESSOR_PREFIX_BYTES,
            "compressor prefix storage size differs"
        );
        let end = self.slots[slot].end;
        if end % 2 == 1 {
            if let Some(pending) = &self.pending {
                for (i, buffer) in pending.iter().enumerate() {
                    unsafe {
                        buffer.library.copy_d2d_async(
                            slice(destination, i * 2048, 2048),
                            slice(buffer.buffer, slot * 2048, 2048),
                            2048,
                            stream,
                        )?;
                    }
                }
            }
        }
        Ok(end)
    }

    /// # Safety
    /// Drain the stream before observing or releasing the restored request. A
    /// partially failed restore must revoke the enclosing backbone request. The lease's pages
    /// hold the snapshot's source rows through `end`.
    pub unsafe fn restore_prefix(
        &mut self,
        lease: CompressorLease,
        end: u64,
        source: CuteafdDeviceBuffer,
        stream: *mut c_void,
    ) -> Result<()> {
        let slot = self.validate(lease)?;
        ensure!(
            self.slots[slot].end == 0
                && self.slots[slot].version == 0
                && source.bytes == COMPRESSOR_PREFIX_BYTES
                && self.index.binding(slot).1 == end as usize / ratio(self.layer)?,
            "nonfresh compressor destination or bound rows differ from the snapshot"
        );
        if end % 2 == 1 {
            if let Some(pending) = &self.pending {
                for (i, buffer) in pending.iter().enumerate() {
                    unsafe {
                        buffer.library.copy_d2d_async(
                            slice(buffer.buffer, slot * 2048, 2048),
                            slice(source, i * 2048, 2048),
                            2048,
                            stream,
                        )?;
                    }
                }
            }
        }
        self.slots[slot].end = end;
        self.slots[slot].version = 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and CUDA"]
    fn compressed_prefix_restore_requires_complete_groups_and_bounds_views() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let device = crate::shared::memory::device::cache_test_device(&lib)?;
        let saved = DeviceAllocation::new(&lib, COMPRESSOR_PREFIX_BYTES)?;
        let stream = LoadStream {
            library: &lib,
            raw: lib.cuda_stream_create()?,
        };
        for layer in [2, 20] {
            let mut state = device.own(|| CompressorState::new(&lib, layer, 2, 4, usize::MAX))?;
            assert_eq!(state.index.kv_values.buffer.device_id,device.id);
            assert_eq!(saved.buffer.device_id,0);
            let original = state.begin_request(0, 1)?;
            let rows = 601 / ratio(layer)?;
            let pages: Vec<u32> = (0..rows.div_ceil(PAGE_ROWS) as u32).collect();
            state.bind(original, &pages, 0)?;
            let plan = state.index.reserve(&[(0, 0, rows)])?;
            for buffer in [
                state.index.packed.buffer,
                state.index.scales.buffer,
                state.index.kv_values.buffer,
                state.index.kv_scales.buffer,
            ] {
                lib.copy_h2d(buffer, &vec![0x11; buffer.bytes])?;
            }
            unsafe {
                state.index.upload(&plan, stream.raw)?;
                lib.cuda_stream_synchronize(stream.raw)?;
            }
            state.index.apply(plan);
            state.slots[0].end = 601;
            if let Some(pending) = &state.pending {
                for buffer in pending {
                    lib.copy_h2d(buffer.buffer, &vec![0x33; buffer.buffer.bytes])?;
                }
            }
            assert_eq!(unsafe { state.retain_prefix(original, saved.buffer, stream.raw)? }, 601);
            unsafe {
                lib.cuda_stream_synchronize(stream.raw)?;
            }
            state.release(original)?;
            let resumed = state.begin_request(1, 2)?;
            let end = if layer == 2 { 514 } else { 513 };
            // The engine forks the snapshot's pages through `end` (shared full pages).
            state.bind(resumed, &pages, end as usize / ratio(layer)?)?;
            assert!(state.restore_compressed_prefix(resumed, 602).is_err());
            if layer == 2 {
                assert!(state.restore_compressed_prefix(resumed, 513).is_err());
            }
            state.restore_compressed_prefix(resumed, end)?;
            assert_eq!(state.committed_end(resumed)?, end);
            assert_eq!(
                state.index_cache(resumed)?.rows,
                end as usize / ratio(layer)?
            );
            assert_eq!(state.kv_cache(resumed)?.rows, end as usize / ratio(layer)?);
            assert!(state.committed_proposal(resumed, 0..end + 1, 1).is_err());
            assert!(state.committed_proposal(resumed, end - 128..end, 1).is_ok());
            assert!(state.restore_compressed_prefix(resumed, end).is_err());
            state.release(resumed)?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB and CUDA"]
    fn native_compressor_prefix_preserves_pending_odd_row() -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let device = crate::shared::memory::device::cache_test_device(&lib)?;
        let mut state = device.own(|| CompressorState::new(&lib, 2, 2, 4, usize::MAX))?;
        assert_eq!(state.index.kv_values.buffer.device_id,device.id);
        let saved = DeviceAllocation::new(&lib, COMPRESSOR_PREFIX_BYTES)?;
        let stream = LoadStream {
            library: &lib,
            raw: lib.cuda_stream_create()?,
        };
        let original = state.begin_request(0, 1)?;
        // One accepted token: no completed source row, both FP32 projections
        // must survive for the following token to complete the ratio-two group.
        state.slots[0].end = 1;
        for (i, buffer) in state.pending.as_ref().unwrap().iter().enumerate() {
            lib.copy_h2d(slice(buffer.buffer, 0, 2048), &vec![0x31 + i as u8; 2048])?;
        }
        let end = unsafe { state.retain_prefix(original, saved.buffer, stream.raw)? };
        unsafe {
            lib.cuda_stream_synchronize(stream.raw)?;
        }
        state.release(original)?;
        let replacement = state.begin_request(0, 2)?;
        for buffer in state.pending.as_ref().unwrap() {
            lib.copy_h2d(slice(buffer.buffer, 0, 2048), &vec![0xff; 2048])?;
        }
        let resumed = state.begin_request(1, 3)?;
        unsafe {
            state.restore_prefix(resumed, end, saved.buffer, stream.raw)?;
            lib.cuda_stream_synchronize(stream.raw)?;
        }
        assert_eq!(state.committed_end(resumed)?, 1);
        assert_eq!(state.index_cache(resumed)?.rows, 0);
        for (i, buffer) in state.pending.as_ref().unwrap().iter().enumerate() {
            let mut bytes = vec![0; 2048];
            lib.copy_d2h(&mut bytes, slice(buffer.buffer, 2048, 2048))?;
            assert_eq!(bytes, vec![0x31 + i as u8; 2048]);
        }
        assert!(state.validate(original).is_err());
        assert!(
            unsafe { state.restore_prefix(resumed, end, saved.buffer, stream.raw) }.is_err()
        );
        state.release(replacement)?;
        state.release(resumed)?;
        Ok(())
    }
}

