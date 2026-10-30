//! Bounded gathered native rows: a lookup in the layer's row cache, then a copy
//! into the pinned upload staging. Rows the cache does not hold are read here
//! (and cached when the class has room); rows another reader is filling are waited for.
use super::engram_prefetch::RowRead;
use crate::formats::mapped_table::row_cache::{Lookup, Pins, RowClass};
use crate::formats::mapped_table::ReadPriority;
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
/// One layer's gather between preparation and completion: the row-cache pins,
/// rows still being filled elsewhere, and this gather's own reads.
pub(crate) struct PreparedEngram {
    pins: Pins,
    /// (cache slot, canonical staging slot, row) of rows another reader is filling.
    pending: Vec<(u32, usize, u64)>,
    /// Rows this gather reads itself (row-cache misses), with each one's cache
    /// slot when it reserved one and its canonical staging slot.
    read: Option<(RowRead, Vec<(Option<u32>, usize)>)>,
    rows: usize, layer: usize, encoding: EngramEncoding, global_scale: f32,
}
/// Copy one cached row (weight bytes then scale bytes) into its canonical slot.
fn place(weights: &mut [u8], scales: &mut [u8], scale_base: usize, encoding: EngramEncoding, slot: usize, row: &[u8]) {
    let (w, s) = (encoding.weight_bytes(), encoding.scale_bytes());
    weights[slot * w..(slot + 1) * w].copy_from_slice(&row[..w]);
    if encoding == EngramEncoding::Fp8 { scales[slot * s..(slot + 1) * s].copy_from_slice(&row[w..w + s]); }
    else { weights[scale_base + slot * s..scale_base + (slot + 1) * s].copy_from_slice(&row[w..w + s]); }
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
    /// two pages per <=264-byte table row, and tree/hash allocation slack.
    ///
    /// The fixed 1 MiB covers caller-side residency scratch and metadata.
    /// The process-wide reactor budget is charged separately by the staging
    /// shape, once rather than once per prefill/decode worker. Thread stacks,
    /// request batches, allocator bookkeeping and OS page cache are excluded.
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
        let prepared = self.prepare(table, batches, layer_index, ReadPriority::Prefill)?;
        self.finish(table, prepared)?;
        self.view()
    }
    /// Look up every distinct row of the batches: copy published rows now,
    /// start reading the rows the cache lacks, and pin rows being filled elsewhere.
    pub(crate) fn prepare(&mut self, table: &EngramTable, batches: &[&EngramBatch], layer_index: usize,
        priority: ReadPriority) -> Result<PreparedEngram> {
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
        let encoding = table.encoding();
        let (weight_bytes, scale_bytes) = (encoding.weight_bytes(), encoding.scale_bytes());
        ensure!(table.row_width() == weight_bytes + scale_bytes, "engram row cache width differs from its encoding");
        let capacity = self.capacity;
        let payload = self.payload.bytes_mut();
        let (weights, rest) = payload.split_at_mut(capacity * 24 * 256);
        let (scales, text_mask) = rest.split_at_mut(capacity * 24 * 8);
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
        // FP4's packed weights and scales fit in the original FP8 weight arena.
        // Keep the FP8 allocation/budget unchanged and never expand mapped rows.
        let scale_base = capacity * 24 * 128;
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
        let class = if priority.is_decode() { RowClass::Decode } else { RowClass::Prefill };
        let (pins, lookups) = table.cache().acquire(&addresses, class);
        let mut pending = Vec::new();
        let mut misses = Vec::new();
        for ((lookup, &address), &slot) in lookups.iter().zip(&addresses).zip(&slots) {
            match *lookup {
                Lookup::Ready(cached) => place(weights, scales, scale_base, encoding, slot, pins.row(cached)),
                // Decode never waits on another reader: its own hybrid read is a
                // copy when the row is resident and joins the device read when not.
                Lookup::Wait(_) if class == RowClass::Decode => misses.push((address, None, slot)),
                Lookup::Wait(cached) => pending.push((cached, slot, address)),
                Lookup::Fill(cached) => misses.push((address, Some(cached), slot)),
                Lookup::Uncached => misses.push((address, None, slot)),
            }
        }
        let read = if misses.is_empty() { None } else {
            let ids: Vec<u64> = misses.iter().map(|m| m.0).collect();
            let priority = if layer_index == 0 { priority.first() } else { priority };
            Some((table.start_read(&ids, priority)?, misses.into_iter().map(|(_, cached, slot)| (cached, slot)).collect()))
        };
        Ok(PreparedEngram { pins, pending, read, rows, layer: layer_index, encoding, global_scale: table.global_scale() })
    }
    /// Land this gather's reads (publishing the ones it reserved), wait for rows
    /// filled elsewhere, then scatter duplicates to their token/head slots.
    pub(crate) fn finish(&mut self, table: &EngramTable, prepared: PreparedEngram) -> Result<()> {
        let PreparedEngram { mut pins, pending, read, rows, layer, encoding, global_scale } = prepared;
        let (weight_bytes, scale_bytes) = (encoding.weight_bytes(), encoding.scale_bytes());
        let scale_base = self.capacity * 24 * 128;
        let (weights, rest) = self.payload.bytes_mut().split_at_mut(self.capacity * 24 * 256);
        let (scales, _) = rest.split_at_mut(self.capacity * 24 * 8);
        let (mut fallback, mut device) = (0, 0);
        if let Some((mut read, targets)) = read {
            read.finish(table)?;
            fallback = targets.len();
            device = read.misses();
            let mut row = [0u8; 280];
            pins.publish_all(targets.iter().enumerate().filter_map(|(i, (cached, _))| cached.map(|c| (c, read.row(i)))))?;
            for (i, &(_, slot)) in targets.iter().enumerate() {
                let [w, s] = read.row(i);
                row[..w.len()].copy_from_slice(w);
                row[w.len()..w.len() + s.len()].copy_from_slice(s);
                place(weights, scales, scale_base, encoding, slot, &row[..w.len() + s.len()]);
            }
        }
        let mut failed = Vec::new();
        for &(cached, slot, address) in &pending {
            if pins.wait(cached) { place(weights, scales, scale_base, encoding, slot, pins.row(cached)); }
            else { failed.push((slot, address)); }
        }
        if !failed.is_empty() {
            // Their filler stopped (cancelled request or failed read): read them here.
            let ids: Vec<u64> = failed.iter().map(|f| f.1).collect();
            let mut read = table.start_read(&ids, ReadPriority::Prefill)?;
            read.finish(table)?;
            let mut row = [0u8; 280];
            for (i, &(slot, _)) in failed.iter().enumerate() {
                let [w, s] = read.row(i);
                row[..w.len()].copy_from_slice(w);
                row[w.len()..w.len() + s.len()].copy_from_slice(s);
                place(weights, scales, scale_base, encoding, slot, &row[..w.len() + s.len()]);
            }
            fallback += failed.len();
            device += read.misses();
        }
        let waited = pending.len() - failed.len();
        if fallback > 0 || waited > 0 {
            let decode = pins.class() == RowClass::Decode;
            table.weights().stats().record_gather_misses(decode, fallback, waited);
            if device > 0 && !decode {
                static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    tracing::warn!(layer, rows = device, "Engram prefill gather read rows the look-ahead had not \
                        fetched; later ones are counted in stall_prefill_fallback_rows");
                }
            }
        }
        drop(pins);
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
        self.ready = Some((rows, layer, encoding, global_scale));
        Ok(())
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
            + 264 // optional interleaved readback, split after gathering
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
    fn strided_gather_matches_paired_bytes_across_parts_duplicates_images_and_gather_interfaces() -> Result<()> {
        let batch = cuteafd_core::EngramHistory::new(0)?.prepare(0, &[Some(7), None, Some(31)], 3)?;
        let weight = tempfile::NamedTempFile::new()?;
        let scale = tempfile::NamedTempFile::new()?;
        let packed = tempfile::NamedTempFile::new()?;
        weight.as_file().set_len(ENGRAM_ROWS[0] * 256)?;
        scale.as_file().set_len(ENGRAM_ROWS[0] * 8)?;
        packed.as_file().set_len(ENGRAM_ROWS[0] * 264)?;
        for hashes in batch.hashes() {
            for &address in &hashes[0] {
                let row: Vec<_> = (0..264).map(|col| ((address + col) % 251) as u8).collect();
                weight.as_file().write_all_at(&row[..256], address * 256)?;
                scale.as_file().write_all_at(&row[256..], address * 8)?;
                packed.as_file().write_all_at(&row, address * 264)?;
            }
        }
        let map = |file: &tempfile::NamedTempFile, width| {
            // SAFETY: test owns immutable files throughout all gathers.
            unsafe { crate::MappedTable::single(file.path(), 0, ENGRAM_ROWS[0],
                crate::RowFormat::of(cuteafd_core::DType::U8, width)?) }
        };
        let paired = EngramTable::new(map(&weight, 256)?, map(&scale, 8)?)?;
        let boundary = batch.hashes().iter().flat_map(|hashes| hashes[0].iter()).copied().min().unwrap() + 1;
        let parts = [crate::TablePart { path: packed.path().into(), offset: 0, rows: boundary },
            crate::TablePart { path: packed.path().into(), offset: boundary * 264, rows: ENGRAM_ROWS[0] - boundary }];
        // SAFETY: test owns the immutable packed file throughout both parts' lifetime.
        let strided = EngramTable::new_strided(unsafe { crate::MappedTable::open(&parts,
            crate::RowFormat::of(cuteafd_core::DType::U8, 264)?)? })?;
        assert!(strided.scales().is_none());
        let mut before = EngramBatchStaging::new(8)?;
        let mut after = EngramBatchStaging::new(8)?;
        for copies in [2, 1, 2] {
            let batches = vec![&batch; copies];
            let expected = before.gather(&paired, &batches, 0)?;
            for _ in 0..2 {
                let actual = after.gather(&strided, &batches, 0)?;
                assert_eq!(actual.weights, expected.weights);
                assert_eq!(actual.scales, expected.scales);
                assert_eq!(actual.text_mask, expected.text_mask);
                assert_eq!(actual.global_scale.to_bits(), expected.global_scale.to_bits());
                assert_eq!(actual.encoding, expected.encoding);
                assert_eq!(strided.weights().stats().snapshot().gathers, 0,
                    "staging must leave logical gather accounting to its owner");
                assert_eq!(actual.weights.len(), expected.weights.len());
            }
        }
        assert!(EngramTable::new_strided(map(&weight, 256)?).is_err());
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
