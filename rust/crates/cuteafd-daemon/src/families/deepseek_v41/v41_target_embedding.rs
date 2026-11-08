//! Token initialization and image replacement before the first target mHC block.
use crate::shared::memory::Tag;
use crate::shared::memory::{DeviceAllocation, HostAllocation, LoadStream};
use crate::families::deepseek_v41::v41_tensors::NativeRtxTensors;
use crate::shared::decode_graph::RowGraphs;
use anyhow::{Context, Result, ensure};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary, V41AttentionOps};
use std::{ffi::c_void, marker::PhantomData, rc::Rc};
use crate::shared::memory::{chain, row_window::{RowWindow, ScratchGuard, MAX_RESERVE_ROWS}};

#[cfg(test)]
thread_local! {
    static SKIP_ACQUIRE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
pub(crate) struct TargetEmbedding<'a> {
    pub residual: CuteafdDeviceBuffer,
    pub pre: CuteafdDeviceBuffer,
    pub token_ids: &'a [u32],
    pub positions: &'a [u64],
    _owner: PhantomData<&'a ()>,
}

/// Unit-cap input-hook scratch. Standalone residual/pre and caller-stream ids
/// are guarded; release after their block reader joins. Direct producers write
/// residual/pre into the block planes. Image scratch stays on the shared stream.
/// Pinned uploads and token/position/ready state stay per slot.
pub(crate) struct TargetEmbeddingStorage<'a> {
    stream: LoadStream<'a>,
    guard: ScratchGuard<'a>,
    ids: DeviceAllocation<'a>,
    residual: DeviceAllocation<'a>,
    pre: DeviceAllocation<'a>,
    image_features: DeviceAllocation<'a>,
    image_indices: DeviceAllocation<'a>,
    rows: usize,
    unit: usize,
}
impl Drop for TargetEmbeddingStorage<'_> {
    fn drop(&mut self) {
        // SAFETY: this storage owns its stream and all queued allocations.
        crate::shared::decode_graph::fatal_drain(unsafe {
            self.stream.library.cuda_stream_synchronize(self.stream.raw)
        }, "embedding storage");
    }
}
pub(crate) struct TargetEmbeddingWave<'w, 'a> {
    storage: Rc<TargetEmbeddingStorage<'a>>,
    window: RowWindow,
    ops: V41AttentionOps<'a>,
    table: &'w NativeRtxTensors<'a>,
    staging: HostAllocation<'a>,
    image_staging: HostAllocation<'a>,
    capacity: usize,
    graphs: RowGraphs<'a>,
    tokens: Vec<u32>,
    positions: Vec<u64>,
    ready: bool,
    destinations: Option<[CuteafdDeviceBuffer; 2]>,
}
impl<'w, 'a> TargetEmbeddingWave<'w, 'a> {
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid target embedding capacity"
        );
        Ok(capacity * (4 + 40960 + 16 + 10240 + 4))
    }
    pub fn new(
        library: &'a NativeLibrary,
        table: &'w NativeRtxTensors<'a>,
        capacity: usize,
        budget: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(capacity)? <= budget,
            "target embedding exceeds budget"
        );
        Self::on_storage(library, table, Self::storage(library, capacity, capacity)?)
    }
    pub(crate) fn storage(library: &'a NativeLibrary, rows: usize, unit: usize)
        -> Result<Rc<TargetEmbeddingStorage<'a>>> {
        ensure!((1..=MAX_RESERVE_ROWS).contains(&rows) && (1..=4096).contains(&unit)
            && unit <= rows, "invalid embedding storage dimensions");
        Ok(Rc::new(TargetEmbeddingStorage {
            stream: LoadStream { library, raw: library.cuda_stream_create()? },
            guard: ScratchGuard::new(library, &[library.cuda_get_device()?])?,
            ids: DeviceAllocation::new(library, unit * 4, Tag::workspace("v41/target-embedding"))?,
            residual: DeviceAllocation::new(library, unit * 40960, Tag::workspace("v41/target-embedding"))?,
            pre: DeviceAllocation::new(library, unit * 16, Tag::workspace("v41/target-embedding"))?,
            image_features: DeviceAllocation::new(library, unit * 10240, Tag::workspace("v41/target-embedding"))?,
            image_indices: DeviceAllocation::new(library, unit * 4, Tag::workspace("v41/target-embedding"))?, rows, unit,
        }))
    }
    pub(crate) fn on_storage(library: &'a NativeLibrary, table: &'w NativeRtxTensors<'a>,
        storage: Rc<TargetEmbeddingStorage<'a>>) -> Result<Self> {
        ensure!(table.get("embed.weight")?.bytes == 129280 * 5120 * 2
            && table.get("embed.weight")?.device_id == storage.ids.buffer.device_id,
            "invalid embedding table extent or device");
        let capacity = storage.unit;
        Ok(Self { storage, window: RowWindow::NONE,
            ops: library.v41_attention_ops()?, table,
            staging: HostAllocation::new(library, capacity * 4, Tag::staging("v41/target-embedding"))?,
            image_staging: HostAllocation::new(library, capacity * (10240 + 4), Tag::staging("v41/target-embedding"))?, capacity,
            graphs: RowGraphs::new(library, "target_embedding", cuteafd_core::V41_VERIFY_ROWS, "deepseek_v41"),
            tokens: Vec::with_capacity(capacity), positions: Vec::with_capacity(capacity), ready: false, destinations: None,
        })
    }
    /// Caller drains the slot before selecting its next unit's rows.
    pub(crate) fn select_window(&mut self, window: RowWindow) -> Result<()> {
        if let Some(range) = window.range() {
            ensure!(range.start < range.end && range.end <= self.storage.rows
                && range.len() <= self.capacity, "embedding window outside storage or unit");
        }
        self.invalidate();
        self.window = window;
        Ok(())
    }
    fn invalidate(&mut self) {
        self.ready = false;
        self.destinations = None;
        self.tokens.clear();
        self.positions.clear();
    }
    /// Release after this input hook's block reader has joined the unit chain.
    pub(crate) fn release_scratch(&self) -> Result<()> { self.storage.guard.release() }
    #[cfg(test)]
    pub(crate) fn test_writer(&self, skip: bool) -> *mut std::ffi::c_void {
        SKIP_ACQUIRE.with(|value| value.set(skip));
        self.storage.stream.raw
    }
    unsafe fn acquire_scratch(&self, stream: *mut std::ffi::c_void) -> Result<()> {
        #[cfg(test)]
        if SKIP_ACQUIRE.with(|skip| skip.get()) { return Ok(()); }
        // SAFETY: caller supplies a live writer stream on this storage's device.
        unsafe { self.storage.guard.acquire(stream) }
    }
    fn buffers(&self) -> [CuteafdDeviceBuffer; 2] {
        self.destinations.unwrap_or([self.storage.residual.buffer, self.storage.pre.buffer])
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.storage.stream.library.cuda_stream_synchronize(self.storage.stream.raw) }
    }
    unsafe fn enqueue(&self, rows: usize) -> Result<()> {
        unsafe {
            self.ops.target_embed(
                self.table.get("embed.weight")?,
                self.storage.ids.buffer,
                self.buffers()[0],
                self.buffers()[1],
                rows,
                self.storage.stream.raw,
            )
        }
    }
    /// Initializes text rows in caller order. Positions may repeat across
    /// requests; request ownership remains the scheduler's responsibility.
    /// Retains decode row shapes over the same fixed weights and stable storage.
    pub fn execute(&mut self, tokens: &[u32], positions: &[u64]) -> Result<TargetEmbedding<'_>> {
        self.execute_to(tokens, positions, None)
    }
    fn execute_to(&mut self, tokens: &[u32], positions: &[u64],
        destinations: Option<[CuteafdDeviceBuffer; 2]>) -> Result<TargetEmbedding<'_>> {
        self.invalidate();
        self.destinations = destinations;
        let rows = tokens.len();
        ensure!(
            rows > 0
                && rows <= self.capacity && self.window.admits(rows)
                && rows == positions.len()
                && tokens.iter().all(|&id| id < 129280)
                && positions.iter().all(|&p| p < 1048576),
            "invalid target embedding tokens or positions"
        );
        for (dst, token) in self.staging.bytes_mut().chunks_exact_mut(4).zip(tokens) {
            dst.copy_from_slice(&token.to_ne_bytes());
        }
        // SAFETY: staging is per slot and is retained until the slot is drained.
        unsafe {
            chain::join(self.storage.stream.library, self.storage.stream.raw)?;
            self.acquire_scratch(self.storage.stream.raw)?;
            self.storage.stream.library.copy_host_buffer_h2d_async(
                self.storage.ids.buffer, self.staging.buffer,
                rows * 4, self.storage.stream.raw)?;
        }
        if self.destinations.is_some() || self.window.is_windowed() || !super::graph_policy::captures_shape(rows as u32)
            || (self.graphs.get(rows).is_none() && !self.graphs.capture_allowed(rows))
            || crate::shared::decode_graph::census::eager("layer_row") {
            if self.destinations.is_none() && !self.window.is_windowed()
                && super::graph_policy::captures_shape(rows as u32)
                && self.graphs.get(rows).is_none() && !self.graphs.capture_allowed(rows) {
                super::graph_policy::note_miss(format!("target_embedding rows={rows}"),
                    || format!("capacity={} destinations={}", self.capacity, self.destinations.is_some()));
            }
            let launched = unsafe { crate::shared::decode_graph::census::dispatch(
                self.storage.stream.library, self.storage.stream.raw, c"layer_row", rows, None, || self.enqueue(rows)) };
            if let Err(error) = launched { self.synchronize()?; return Err(error); }
            // SAFETY: every output write precedes this chain publication.
            unsafe { chain::finish(self.storage.stream.library, self.storage.stream.raw)?; }
            self.tokens.extend_from_slice(tokens);
            self.positions.extend_from_slice(positions);
            self.ready = true;
            return self.output();
        }
        if self.graphs.get(rows).is_none() {
            let warmup = unsafe { self.enqueue(rows) };
            warmup.and(self.synchronize())?;
            unsafe {
                self.storage.stream
                    .library
                    .cuda_graph_begin_capture(self.storage.stream.raw)?;
            }
            let launched = unsafe { self.enqueue(rows) };
            let captured = unsafe { self.storage.stream.library.cuda_graph_end_capture(self.storage.stream.raw) };
            match (launched, captured) {
                (Ok(()), Ok(graph)) => {
                    // SAFETY: eager warmup drained; capture has not been launched.
                    if let Err(error) = unsafe { self.graphs.insert(rows, graph) } {
                        // SAFETY: this new graph has no queued launches.
                        unsafe { self.storage.stream.library.cuda_graph_exec_destroy(graph)?; }
                        return Err(error);
                    }
                },
                (Err(e), Ok(graph)) => {
                    unsafe {
                        self.storage.stream.library.cuda_graph_exec_destroy(graph)?;
                    }
                    return Err(e);
                }
                (Err(e), Err(_)) | (Ok(()), Err(e)) => return Err(e),
            }
        }
        let graph = self.graphs.get(rows).context("target embedding graph missing")?;
        let launched = unsafe {
            crate::shared::decode_graph::census::dispatch(self.storage.stream.library, self.storage.stream.raw, c"layer_row", rows,
                Some(graph), || self.enqueue(rows))
        };
        launched.and(self.synchronize())?;
        self.tokens.extend_from_slice(tokens);
        self.positions.extend_from_slice(positions);
        self.ready = true;
        self.output()
    }
    /// # Safety
    /// The caller owns the supplied stream and destinations and drains it before
    /// reusing this embedding owner, its staging, or either destination. Text
    /// embeddings are produced directly into the block, without an intermediate copy.
    pub unsafe fn enqueue_into(&mut self, tokens: &[u32], stream: *mut c_void,
        destination: [CuteafdDeviceBuffer; 2]) -> Result<()> {
        self.invalidate();
        ensure!(!tokens.is_empty() && tokens.len() <= self.capacity && self.window.admits(tokens.len())
            && tokens.iter().all(|&id| id < 129280)
            && destination[0].bytes >= tokens.len()*40960 && destination[1].bytes >= tokens.len()*16
            && destination.iter().all(|b| b.device_id == self.storage.ids.buffer.device_id),
            "invalid direct embedding producer input");
        for (dst, token) in self.staging.bytes_mut().chunks_exact_mut(4).zip(tokens) {
            dst.copy_from_slice(&token.to_ne_bytes());
        }
        unsafe {
            chain::join(self.storage.stream.library, stream)?;
            self.acquire_scratch(stream)?;
            self.storage.stream.library.copy_host_buffer_h2d_async(self.storage.ids.buffer, self.staging.buffer,
                tokens.len()*4, stream)?;
            self.ops.target_embed(self.table.get("embed.weight")?, self.storage.ids.buffer,
                destination[0], destination[1], tokens.len(), stream)
        }
    }
    /// Complete image span rows include learned delimiters. The row indices are
    /// sorted, unique positions within this flattened multi-request batch.
    pub fn execute_with_images(&mut self, tokens: &[u32], positions: &[u64],
        images: &[(usize, &[u8])]) -> Result<TargetEmbedding<'_>> {
        self.execute_images_to(tokens, positions, images, None)
    }
    /// # Safety
    /// Destinations are this unit's block planes, retained through its chain drain.
    /// Call release_scratch after the input hook's last reader joins.
    pub(crate) unsafe fn execute_with_images_into(&mut self, tokens: &[u32], positions: &[u64],
        images: &[(usize, &[u8])], destinations: [CuteafdDeviceBuffer; 2]) -> Result<TargetEmbedding<'_>> {
        self.invalidate();
        ensure!(destinations[0].bytes >= tokens.len() * 40960
            && destinations[1].bytes >= tokens.len() * 16
            && destinations.iter().all(|b| b.device_id == self.storage.ids.buffer.device_id),
            "direct embedding destination extent or device differs");
        self.execute_images_to(tokens, positions, images, Some(destinations))
    }
    fn execute_images_to(&mut self, tokens: &[u32], positions: &[u64],
        images: &[(usize, &[u8])], destinations: Option<[CuteafdDeviceBuffer; 2]>) -> Result<TargetEmbedding<'_>> {
        self.invalidate();
        ensure!(images.iter().enumerate().all(|(i, (row, bytes))|
            *row < tokens.len() && tokens[*row] == cuteafd_loader::V41_IMAGE_TOKEN_ID
                && bytes.len() == 10240 && (i == 0 || images[i-1].0 < *row)),
            "invalid target image rows or feature extent");
        self.execute_to(tokens, positions, destinations)?;
        if images.is_empty() { return self.output(); }
        self.ready = false;
        let count = images.len();
        let staged = self.image_staging.bytes_mut();
        for (i, &(row, data)) in images.iter().enumerate() {
            staged[i * 10240..(i + 1) * 10240].copy_from_slice(data);
            staged[count * 10240 + i * 4..count * 10240 + (i + 1) * 4]
                .copy_from_slice(&(row as u32).to_ne_bytes());
        }
        let mut indices = self.image_staging.buffer;
        indices.ptr = indices.ptr.cast::<u8>().wrapping_add(count * 10240).cast();
        indices.bytes = count * 4;
        let launched = (|| unsafe {
            chain::join(self.storage.stream.library, self.storage.stream.raw)?;
            self.acquire_scratch(self.storage.stream.raw)?;
            self.storage.stream.library.copy_host_buffer_h2d_async(self.storage.image_features.buffer,
                self.image_staging.buffer, count * 10240, self.storage.stream.raw)?;
            self.storage.stream.library.copy_host_buffer_h2d_async(self.storage.image_indices.buffer,
                indices, count * 4, self.storage.stream.raw)?;
            self.storage.stream.library.v41_vision_embed(self.storage.image_features.buffer,
                self.storage.image_indices.buffer, self.buffers()[0],
                count, tokens.len(), self.storage.stream.raw)
        })();
        if let Err(error) = launched { self.synchronize()?; return Err(error); }
        // SAFETY: the image writes are on the same stream as the scratch uploads.
        unsafe { chain::finish(self.storage.stream.library, self.storage.stream.raw)?; }
        self.ready = true;
        self.output()
    }
    pub fn output(&self) -> Result<TargetEmbedding<'_>> {
        ensure!(self.ready, "target embeddings unpublished");
        let mut residual = self.buffers()[0];
        let mut pre = self.buffers()[1];
        residual.bytes = self.tokens.len() * 40960;
        pre.bytes = self.tokens.len() * 16;
        Ok(TargetEmbedding {
            residual,
            pre,
            token_ids: &self.tokens,
            positions: &self.positions,
            _owner: PhantomData,
        })
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.invalidate();
        self.synchronize()?;
        // SAFETY: synchronize completed all uses of the fixed embedding storage.
        unsafe { self.graphs.clear() }
    }
}
impl Drop for TargetEmbeddingWave<'_, '_> {
    fn drop(&mut self) {
        if let Err(e) = self.clear_graph() {
            crate::shared::decode_graph::fatal_drain(Err(e), "target embedding");
        }
    }
}

#[cfg(test)]
mod tests;
