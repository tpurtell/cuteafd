//! Bounded gathered native rows; run page-faulting gathers on an I/O worker.
use crate::{EngramEncoding, EngramTable};
use anyhow::{ensure, Context, Result};
use cuteafd_core::{EngramBatch, ENGRAM_ROWS};

enum StagingBytes {
    Host(Vec<u8>),
    Pinned(cuteafd_ffi::PinnedBytes),
}
impl StagingBytes {
    fn bytes(&self) -> &[u8] { match self { Self::Host(v) => v, Self::Pinned(v) => v.bytes() } }
    fn bytes_mut(&mut self) -> &mut [u8] { match self { Self::Host(v) => v, Self::Pinned(v) => v.bytes_mut() } }
}

pub struct EngramBatchStaging {
    capacity: usize,
    payload: StagingBytes,
    order: Vec<(u64, usize)>,
    ready: Option<(usize, usize, EngramEncoding, f32)>,
}
/// Token-major native rows: 24 embeddings of width 256 per token.
/// Image embeddings are zero, scales are one, and the text mask is zero.
pub struct EngramGatherView<'a> {
    pub encoding: EngramEncoding,
    pub global_scale: f32,
    pub weights: &'a [u8],
    pub scales: &'a [u8],
    pub text_mask: &'a [u8],
    pub rows: usize,
    pub layer_index: usize,
}
impl EngramBatchStaging {
    /// Vec payload budget, excluding allocator metadata and request-owned batches.
    pub fn storage_bytes(capacity: usize) -> Result<usize> {
        ensure!(
            capacity > 0 && capacity <= 4096,
            "invalid engram staging capacity"
        );
        capacity
            .checked_mul(24 * (256 + 8 + std::mem::size_of::<(u64, usize)>()) + 1)
            .context("engram staging budget overflow")
    }
    pub fn new(capacity: usize) -> Result<Self> {
        ensure!(
            capacity > 0 && capacity <= 4096,
            "invalid engram staging capacity"
        );
        Ok(Self {
            capacity,
            payload: StagingBytes::Host(vec![0; Self::payload_bytes(capacity)]),
            order: Vec::with_capacity(capacity * 24),
            ready: None,
        })
    }
    /// Peak per gather worker, not per staging lease. At most 24 distinct table
    /// rows per token. Charge 1024 bytes per lookup for the address/slot vectors,
    /// mapped-table validation tree + read descriptors, selected/unsupported
    /// rows, residency flags and retained TLS page vectors, and io_uring's slot
    /// hash set. This conservative bound includes vector growth (at most 2x),
    /// two pages per <=256-byte table row, and tree/hash allocation slack.
    ///
    /// The fixed 1 MiB additionally covers MAX_BATCH=4096: one SQ ring (64-byte
    /// SQEs + 4-byte indices), default 2x CQ (16-byte CQEs), rounded ring headers,
    /// retry/done/nowait/missed arrays (18 bytes/entry), 4096 mincore bytes and
    /// <=16 borrowed batch references. Weights/scales and L1/L14 run serially,
    /// so these terms occur once per worker. Thread stacks, request batches,
    /// allocator bookkeeping and OS page cache are not staging workspace.
    pub fn worker_bytes(rows: usize) -> Result<usize> {
        Self::storage_bytes(rows)?;
        Ok(rows * 24 * 1024 + (1 << 20))
    }
    pub fn payload_bytes(rows: usize) -> usize { rows * (24 * (256 + 8) + 1) }
    pub fn new_pinned(library: &cuteafd_ffi::NativeLibrary, capacity: usize) -> Result<Self> {
        Self::storage_bytes(capacity)?;
        Ok(Self { capacity, payload: StagingBytes::Pinned(library.pinned_bytes(Self::payload_bytes(capacity))?),
            order: Vec::with_capacity(capacity * 24), ready: None })
    }
    pub fn host_buffer(&self) -> Option<cuteafd_ffi::CuteafdHostBuffer> {
        match &self.payload { StagingBytes::Pinned(p) => Some(p.buffer()), _ => None }
    }
    /// # Safety
    /// The copy stream contains every H2D reader of this slot's pinned bytes.
    pub unsafe fn copied_on(&mut self, stream: *mut std::ffi::c_void) -> Result<()> {
        match &mut self.payload {
            StagingBytes::Pinned(p) => {
                // SAFETY: forwarded from this method's stream ownership contract.
                unsafe { p.copied_on(stream) }
            },
            _ => anyhow::bail!("asynchronous gather upload requires pinned staging"),
        }
    }
    pub fn copy_ready(&mut self) -> Result<bool> {
        match &mut self.payload { StagingBytes::Pinned(p) => p.copy_ready(), _ => Ok(true) }
    }
    /// Concatenate request-owned batches without committing their histories.
    /// Read each distinct table row once, in address order, then scatter duplicates
    /// back to their canonical token/head slots using preallocated storage.
    /// Any failure invalidates the view; no eager table load is performed.
    pub fn gather(
        &mut self,
        table: &EngramTable,
        batches: &[&EngramBatch],
        layer_index: usize,
    ) -> Result<EngramGatherView<'_>> {
        self.gather_on(table, batches, layer_index, None)
    }
    pub(crate) fn gather_on(&mut self, table: &EngramTable, batches: &[&EngramBatch], layer_index: usize,
        gather: Option<&crate::shared::token_table::RowGather>) -> Result<EngramGatherView<'_>> {
        self.ready = None;
        ensure!(
            !batches.is_empty() && batches.len() <= 16,
            "engram wave requires 1..16 request batches"
        );
        let expected_rows = *ENGRAM_ROWS
            .get(layer_index)
            .context("invalid engram layer index")?;
        ensure!(
            table.weights().rows() == expected_rows,
            "engram table belongs to another layer"
        );
        let rows = batches.iter().try_fold(0usize, |sum, batch| {
            sum.checked_add(batch.hashes().len())
                .context("engram batch length overflow")
        })?;
        ensure!(
            rows > 0 && rows <= self.capacity,
            "engram wave exceeds staging capacity"
        );
        let payload = self.payload.bytes_mut();
        let (weights, rest) = payload.split_at_mut(self.capacity * 24 * 256);
        let (scales, text_mask) = rest.split_at_mut(self.capacity * 24 * 8);
        self.order.clear();
        let mut token = 0;
        for batch in batches {
            for (row, hashes) in batch.hashes().iter().enumerate() {
                let text = batch.is_image(row) == Some(false);
                text_mask[token] = u8::from(text);
                if text {
                    for (head, &address) in hashes[layer_index].iter().enumerate() {
                        ensure!(address < expected_rows, "engram hash is outside its table");
                        self.order.push((address, token * 24 + head));
                    }
                }
                token += 1;
            }
        }
        let encoding = table.encoding();
        let weight_bytes = encoding.weight_bytes();
        let scale_bytes = encoding.scale_bytes();
        // FP4's packed weights and scales fit in the original FP8 weight arena.
        // Keep the FP8 allocation/budget unchanged and never expand mapped rows.
        let scale_base = self.capacity * 24 * 128;
        weights[..rows * 24 * weight_bytes].fill(0);
        if encoding == EngramEncoding::Fp8 {
            scales[..rows * 24 * scale_bytes].fill(127);
        } else {
            weights[scale_base..scale_base + rows * 24 * scale_bytes].fill(0x38);
        }
        self.order.sort_unstable();
        let mut addresses = Vec::with_capacity(self.order.len());
        let mut slots = Vec::with_capacity(self.order.len());
        for &(address, slot) in &self.order {
            if addresses.last() != Some(&address) {
                addresses.push(address);
                slots.push(slot);
            }
        }
        // Both backends read the same distinct rows into the same canonical
        // slots; batching is not an io_uring-only optimization.
        let read = |table: &crate::MappedTable, output: &mut [u8]| -> Result<()> {
            match gather { Some(gather) => gather.gather_slots(table, &addresses, &slots, output),
                None => Ok(table.gather_slots(&addresses, &slots, output)?) }
        };
        read(table.weights(), &mut weights[..rows * 24 * weight_bytes])?;
        if encoding == EngramEncoding::Fp8 {
            read(table.scales(), &mut scales[..rows * 24 * scale_bytes])?;
        } else {
            read(table.scales(), &mut weights[scale_base..scale_base + rows * 24 * scale_bytes])?;
        }
        let mut previous = None;
        for &(address, slot) in &self.order {
            if let Some((last_address, source)) = previous {
                if last_address == address {
                    weights.copy_within(source * weight_bytes..(source + 1) * weight_bytes, slot * weight_bytes);
                    if encoding == EngramEncoding::Fp8 {
                        scales.copy_within(source * scale_bytes..(source + 1) * scale_bytes, slot * scale_bytes);
                    } else {
                        weights.copy_within(scale_base + source * scale_bytes..scale_base + (source + 1) * scale_bytes,
                            scale_base + slot * scale_bytes);
                    }
                    continue;
                }
            }
            previous = Some((address, slot));
        }
        self.ready = Some((rows, layer_index, encoding, table.global_scale()));
        self.view()
    }
    pub fn view(&self) -> Result<EngramGatherView<'_>> {
        let (rows, layer_index, encoding, global_scale) = self.ready.context("engram staging is not complete")?;
        let scale_base = self.capacity * 24 * 128;
        let (weights, rest) = self.payload.bytes().split_at(self.capacity * 24 * 256);
        let (scales, text_mask) = rest.split_at(self.capacity * 24 * 8);
        Ok(EngramGatherView {
            encoding, global_scale,
            weights: &weights[..rows * 24 * encoding.weight_bytes()],
            scales: if encoding == EngramEncoding::Fp8 { &scales[..rows * 24 * 8] }
                else { &weights[scale_base..scale_base + rows * 24 * 16] },
            text_mask: &text_mask[..rows],
            rows,
            layer_index,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::FileExt;

    #[test]
    fn engram_worker_bound_covers_row_scaled_and_fixed_backend_scratch() -> Result<()> {
        let word = std::mem::size_of::<usize>();
        // Sum simultaneous capacities (including geometric growth), rather than
        // multiplying a transient worker allocation by the number of leases.
        let per_lookup = 8 + word // distinct addresses and canonical slots
            + 256 + 2 * (std::mem::size_of::<u64>() + 2 * word) // validation tree, reads
            + 2 * word + 2 * 8 + 2 // unsupported indices, selected rows, hit flags
            + 2 * (2 * 2 * word + 2) + 2 // TLS pages/residency and returned hits
            + 2 * (8 + 2 * word) + 2 + 64; // pending reads/hits and hash slots
        let fixed = 4096 * (64 + 4 + 2 * 16 + 18) + 3 * 65536 + 4096 + 16 * word;
        assert!(per_lookup <= 1024);
        assert!(fixed <= 1 << 20);
        for rows in [1, 256, 2048, 2752, 4096] {
            assert_eq!(EngramBatchStaging::worker_bytes(rows)?, rows * 24 * 1024 + (1 << 20));
            assert!(EngramBatchStaging::worker_bytes(rows)? >= rows * 24 * per_lookup + fixed);
        }
        assert!(EngramBatchStaging::worker_bytes(0).is_err());
        assert!(EngramBatchStaging::worker_bytes(4097).is_err());
        Ok(())
    }

    #[test]
    fn packed_ple_gather_preserves_rows_duplicates_masks_and_reuse() -> Result<()> {
        let history = cuteafd_core::EngramHistory::new(0)?;
        let batch = history.prepare(0, &[Some(7), None, Some(31)], 3)?;
        let mut staging = EngramBatchStaging::new(8)?;
        let addresses_before = staging.payload.bytes().as_ptr();
        for encoding in [EngramEncoding::Fp8, EngramEncoding::Nvfp4, EngramEncoding::Fp8] {
            let weights = tempfile::NamedTempFile::new()?;
            let scales = tempfile::NamedTempFile::new()?;
            let wb = encoding.weight_bytes(); let sb = encoding.scale_bytes();
            // Sparse files exercise real 64-bit row offsets without allocating tables.
            weights.as_file().set_len(ENGRAM_ROWS[0] * wb as u64)?;
            scales.as_file().set_len(ENGRAM_ROWS[0] * sb as u64)?;
            let pattern = |address: u64, n: usize| -> Vec<u8> {
                (0..n).map(|col| ((address + col as u64) % 113) as u8).collect()
            };
            for hashes in batch.hashes() {
                for &address in &hashes[0] {
                    weights.as_file().write_all_at(&pattern(address, wb), address * wb as u64)?;
                    scales.as_file().write_all_at(&pattern(address + 3, sb), address * sb as u64)?;
                }
            }
            let map = |file: &tempfile::NamedTempFile, stride| unsafe {
                crate::MappedTable::single(file.path(), 0, ENGRAM_ROWS[0],
                    crate::RowFormat { dtype: cuteafd_core::DType::U8, width: stride, row_bytes: stride })
            };
            let table = match encoding {
                EngramEncoding::Fp8 => EngramTable::new(map(&weights, wb)?, map(&scales, sb)?)?,
                EngramEncoding::Nvfp4 => EngramTable::new_nvfp4(map(&weights, wb)?, map(&scales, sb)?, 0.125)?,
            };
            for copies in [2,1,2] {
                let batches = vec![&batch; copies];
                let view = staging.gather(&table, &batches, 0)?;
                assert_eq!(view.encoding, encoding);
                assert_eq!(view.global_scale, table.global_scale());
                for token in 0..copies*3 {
                    let source = token%3;
                    assert_eq!(view.text_mask[token], u8::from(source != 1));
                    for head in 0..24 {
                        let slot=token*24+head;
                        let w=&view.weights[slot*wb..(slot+1)*wb];
                        let s=&view.scales[slot*sb..(slot+1)*sb];
                        if source==1 {
                            assert!(w.iter().all(|&v| v==0));
                            assert!(s.iter().all(|&v| v==if encoding==EngramEncoding::Fp8 {127} else {0x38}));
                        } else {
                            let address=batch.hashes()[source][0][head];
                            assert_eq!(w, pattern(address,wb));
                            assert_eq!(s, pattern(address+3,sb));
                        }
                    }
                }
            }
            assert!(staging.gather(&table, &[],0).is_err());
            assert!(staging.view().is_err());
            assert_eq!(addresses_before,staging.payload.bytes().as_ptr());
        }
        Ok(())
    }
}
