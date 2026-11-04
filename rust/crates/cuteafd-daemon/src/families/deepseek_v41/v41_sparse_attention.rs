//! Captured sparse attention bound to live window/source/selection proposals.
use crate::families::deepseek_v41::v41_attention_binding::QueryBinding;
use crate::families::deepseek_v41::v41_attention_query::AttentionQueryOutput;
use crate::families::deepseek_v41::v41_compressor::IndexProposal;
use crate::families::deepseek_v41::v41_index_selection::IndexSelectionOutput;
use crate::shared::memory::{DeviceAllocation, HostAllocation, LayerStaging, LoadStream};
use crate::families::deepseek_v41::v41_window::WindowProposal;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{
    CuteafdDeviceBuffer, NativeLibrary, V41SparseAttention, V41SparseBatch, V41SparseSource, V41SparseWindow, V41Kv, V41PeerCopy,
};
use std::{ffi::c_void, marker::PhantomData};
use crate::shared::decode_graph::{GraphBank, GraphOwner};
pub(crate) mod dual;

pub(crate) struct AttentionRequest<'a> {
    pub window: &'a WindowProposal<'a>,
    pub source: Option<&'a IndexProposal<'a>>,
    pub positions: &'a [u64],
}
pub(crate) struct SparseAttentionOutput<'a> {
    query: Option<QueryBinding>,
    tokens: Option<&'a [u64]>,
    pub values: CuteafdDeviceBuffer,
    pub layer: usize,
    pub rows: usize,
    _inputs: PhantomData<&'a ()>,
}
impl SparseAttentionOutput<'_> {
    pub fn binding(&self) -> Result<QueryBinding> {
        self.tokens()?;
        self.query.context("attention query missing")
    }

    pub fn tokens(&self) -> Result<&[u64]> {
        let query = self.query.context("attention has no query origin")?;
        ensure!(query.layer() == self.layer, "attention query layer differs");
        self.tokens.context("attention tokens missing")
    }
}
/// Internal view: producer is queued on the supplied stream, not host-ready.
pub(crate) struct QueuedSparseAttention {
    pub values: CuteafdDeviceBuffer,
    pub layer: usize,
    pub rows: usize,
}

/// The containing lane retains captured consumer storage and immutable weights
/// until sparse graphs are destroyed. No callback may suspend during capture.
pub(crate) trait AttentionGraphTail {
    fn identity(&self) -> Vec<usize>;
    unsafe fn prepare(&mut self, stream: *mut c_void) -> Result<()>;
    unsafe fn enqueue(&mut self, attention: &QueuedSparseAttention, stream: *mut c_void) -> Result<()>;
    unsafe fn restore_warmup(&mut self) -> Result<()>;
    unsafe fn replay_state(&mut self) -> Result<()>;
}
pub(crate) type SparseAttentionWave<'a> = LocalSparseAttentionWave<'a,64>;
pub(crate) type CompactSparseAttentionWave<'a> = LocalSparseAttentionWave<'a,32>;
/// Query/output are guarded unit scratch: fork readers on either GPU consume
/// them inside the attention hook. Other device buffers are plain unit scratch
/// on this storage's one stream. Pinned LayerStaging, graphs and cold plans are
/// per slot. release_scratch must follow the hook's final reader join.
#[cfg(test)]
thread_local! { static SKIP_STORAGE_ACQUIRE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
pub(crate) struct SparseAttentionStorage<'a, const HEADS: usize> {
    stream: LoadStream<'a>,
    kernel: V41SparseAttention<'a>,
    query: DeviceAllocation<'a>,
    output: DeviceAllocation<'a>,
    split_scratch: DeviceAllocation<'a>,
    descriptors: DeviceAllocation<'a>,
    replay_begins: DeviceAllocation<'a>,
    metadata: DeviceAllocation<'a>,
    guard: crate::shared::memory::row_window::ScratchGuard<'a>,
    unit: usize,
}
impl<'a, const HEADS: usize> SparseAttentionStorage<'a, HEADS> {
    fn allocation_sizes(unit: usize) -> Result<[usize; 6]> {
        ensure!((1..=4096).contains(&unit) && matches!(HEADS, 32 | 64), "invalid sparse storage unit/heads");
        let batch = unit.min(cuteafd_core::V41_VERIFY_ROWS);
        Ok([unit * HEADS * 1024, unit * HEADS * 1024,
            V41SparseAttention::split_scratch_bytes(batch, 10)? / 64 * HEADS,
            V41SparseAttention::batch_descriptor_bytes(batch)?, unit * 8, unit * 80])
    }
    pub(crate) fn storage_bytes(unit: usize) -> Result<usize> {
        Ok(Self::allocation_sizes(unit)?.into_iter().sum())
    }
    pub(crate) fn new(library: &'a NativeLibrary, unit: usize, readers: &[i32]) -> Result<std::rc::Rc<Self>> {
        let [query, output, split, descriptors, replay, metadata] = Self::allocation_sizes(unit)?;
        Ok(std::rc::Rc::new(Self {
            stream: LoadStream { library, raw: library.cuda_stream_create()? },
            kernel: if HEADS == 64 { library.v41_sparse_attention()? } else { library.v41_sparse_attention_heads32()? },
            query: DeviceAllocation::new(library, query)?, output: DeviceAllocation::new(library, output)?,
            split_scratch: DeviceAllocation::new(library, split)?, descriptors: DeviceAllocation::new(library, descriptors)?,
            replay_begins: DeviceAllocation::new(library, replay)?, metadata: DeviceAllocation::new(library, metadata)?,
            guard: crate::shared::memory::row_window::ScratchGuard::new(library, readers)?, unit,
        }))
    }
}
impl<const HEADS: usize> Drop for SparseAttentionStorage<'_, HEADS> {
    fn drop(&mut self) {
        // SAFETY: the shared stream is owned here and drains before its buffers.
        crate::shared::decode_graph::fatal_drain(unsafe {
            self.stream.library.cuda_stream_synchronize(self.stream.raw)
        }, "sparse attention storage");
    }
}
pub(crate) struct LocalSparseAttentionWave<'a,const HEADS:usize> {
    storage: std::rc::Rc<SparseAttentionStorage<'a, HEADS>>,
    window: crate::shared::memory::row_window::RowWindow,
    /// Per-layer pinned staging (see `LayerStaging`): metadata, descriptors
    /// and replay begins differ by layer and may still be queued.
    descriptor_staging: LayerStaging<'a>,
    staging: LayerStaging<'a>,
    replay_staging: LayerStaging<'a>,
    capacity: usize,
    batch_rows: usize,
    attention_plan: Option<cuteafd_core::coordinator_programs::AttentionReductionPlan>,
    decode: bool,
    // Batched keys retain row count and stable wave/selection/sink storage;
    // current descriptors are validated and uploaded before EVERY replay.
    // For other paths, complete external-pointer and launch-geometry fingerprints are checked
    // against live proposals before every replay. Inactive graphs never launch.
    // Adaptive mode keeps at most batch_rows variants per layer, including prefill;
    // request layouts can have many more combinations than total row counts.
    graphs: GraphBank<(usize, Vec<usize>), GraphOwner<'a, ()>>,
    graph_limit: usize,
    warmed_kernels: u16,
    cold: Option<ColdSparse>,
}
struct RequestLaunch {
    window: V41SparseWindow,
    source: Option<V41SparseSource>,
    rows: usize,
    width: usize,
}
// Owned launch description; the containing lane guard retains all referenced
// query/cache/selection allocations until warmup, replay and consumers finish.
struct ColdSparse {
    layer: usize,
    rows: usize,
    sink: CuteafdDeviceBuffer,
    launches: Vec<RequestLaunch>,
    selected: Option<CuteafdDeviceBuffer>,
    batch: Option<V41SparseBatch>,
    fingerprint: Vec<usize>,
    needed: u16,
    tail_identity: Option<Vec<usize>>,
}
fn validate_plan_installation(installed: bool, graph_count: usize, pending: bool,
    batch_rows: usize, capacity_rows: usize) -> Result<()> {
    ensure!(!installed && graph_count == 0 && !pending,
        "attention reduction plan is immutable after load");
    ensure!(batch_rows >= capacity_rows, "attention scratch must cover full execution capacity");
    Ok(())
}

/// Exact-pointer (non-descriptor) keys at decode-band rows come only from short
/// prefill and replay passes: they embed one request's cache and proposal
/// addresses and never repeat. Capturing them filled the per-layer bank and
/// evicted warmed descriptor keys, so every new request recaptured decode
/// graphs (#500). They execute eagerly; the bank stays decode-only below the
/// full-chunk and decoder-replay shapes, which graph_policy still captures.
fn exact_small_shape(batched: bool, rows: usize, batch_rows: usize) -> bool {
    !batched && rows <= batch_rows
}
fn slice(mut b: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> CuteafdDeviceBuffer {
    debug_assert!(offset + bytes <= b.bytes);
    b.ptr = unsafe { b.ptr.cast::<u8>().add(offset).cast() };
    b.bytes = bytes;
    b
}
impl<'a,const HEADS:usize> LocalSparseAttentionWave<'a,HEADS> {
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        SparseAttentionStorage::<'_, HEADS>::storage_bytes(capacity)
    }
    pub fn new(library: &'a NativeLibrary, capacity: usize, budget: usize) -> Result<Self> {
        ensure!(Self::device_bytes(capacity)? <= budget, "sparse attention exceeds budget");
        Self::on_storage(SparseAttentionStorage::new(library, capacity, &[library.cuda_get_device()?])?)
    }
    pub(crate) fn on_storage(storage: std::rc::Rc<SparseAttentionStorage<'a, HEADS>>) -> Result<Self> {
        let library = storage.stream.library;
        let capacity = storage.unit;
        let batch = capacity.min(cuteafd_core::V41_VERIFY_ROWS);
        Ok(Self {
            descriptor_staging: LayerStaging::new(library, V41SparseAttention::batch_descriptor_bytes(batch)?, 40)?,
            staging: LayerStaging::new(library, capacity * 80, 40)?,
            replay_staging: LayerStaging::new(library, capacity * 8, 40)?,
            storage, window: crate::shared::memory::row_window::RowWindow::NONE,
            capacity, batch_rows: batch, attention_plan: None, decode: false,
            graphs: GraphBank::new(None), graph_limit: 1, warmed_kernels: 0, cold: None,
        })
    }
    pub(crate) fn select_window(&mut self, window: crate::shared::memory::row_window::RowWindow) -> Result<()> {
        crate::shared::memory::row_window::require_rest(self.cold.is_some(), "sparse attention", "cold preparation is pending")?;
        if let Some(range) = window.range() {
            ensure!(range.start < range.end && range.len() <= self.capacity, "sparse window exceeds unit");
        }
        self.window = window; Ok(())
    }
    pub(crate) fn release_scratch(&self) -> Result<()> { self.storage.guard.release() }
    fn acquire_scratch(&self) -> Result<()> {
        #[cfg(test)]
        if SKIP_STORAGE_ACQUIRE.with(|skip| skip.get()) { return Ok(()); }
        // SAFETY: writer is this storage's live stream on the current device.
        unsafe { self.storage.guard.acquire(self.storage.stream.raw) }
    }
    pub fn install_attention_plan(&mut self,
        plan: cuteafd_core::coordinator_programs::AttentionReductionPlan) -> Result<()> {
        validate_plan_installation(self.attention_plan.is_some(), self.graphs.len(), self.cold.is_some(),
            self.batch_rows, plan.capacity_rows())?;
        self.storage.kernel.validate_plan(plan, &[self.storage.query.buffer, self.storage.output.buffer, self.storage.split_scratch.buffer,
            self.storage.descriptors.buffer, self.storage.replay_begins.buffer, self.storage.metadata.buffer])?;
        self.attention_plan = Some(plan);
        Ok(())
    }
    pub fn set_decode(&mut self, decode: bool) { self.decode = decode; }
    /// Reserve wider decode batching before any execution and before KV sizing.
    /// Serving allocates the full verifier extent regardless of neural width.
    pub fn reserve_decode_rows(&mut self, rows: usize) -> Result<()> {
        ensure!(rows > 0 && rows <= self.capacity && rows <= cuteafd_core::V41_VERIFY_ROWS,
            "invalid sparse decode reservation");
        ensure!(self.cold.is_none() && self.graphs.len() == 0,
            "sparse decode reservation requires an unused wave");
        if rows <= self.batch_rows { return Ok(()); }
        ensure!(self.attention_plan.is_none(), "attention workspace is immutable after plan admission");
        let library = self.storage.stream.library;
        let scratch = DeviceAllocation::new(library, V41SparseAttention::split_scratch_bytes(rows, 10)?/64*HEADS)?;
        let bytes = V41SparseAttention::batch_descriptor_bytes(rows)?;
        let descriptors = DeviceAllocation::new(library, bytes)?;
        let staging = LayerStaging::new(library, bytes, 40)?;
        let additional_device_bytes = scratch.buffer.bytes + descriptors.buffer.bytes
            - self.storage.split_scratch.buffer.bytes - self.storage.descriptors.buffer.bytes;
        let storage = std::rc::Rc::get_mut(&mut self.storage).context("cannot resize shared sparse storage")?;
        storage.split_scratch = scratch;
        storage.descriptors = descriptors;
        self.descriptor_staging = staging;
        self.batch_rows = rows;
        if self.graph_limit > 1 { self.graph_limit = rows; }
        tracing::info!(device=self.storage.query.buffer.device_id, rows, additional_device_bytes,
            "sparse decode batch reservation before KV sizing");
        Ok(())
    }
    pub fn enable_small_graph_shapes(&mut self) { self.graph_limit = self.batch_rows; }
    pub fn input(&self) -> CuteafdDeviceBuffer {
        self.storage.query.buffer
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.storage.stream.library.cuda_stream_synchronize(self.storage.stream.raw) }
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.drain_chain()?;
        self.graphs.retire_all();
        drop(self.graphs.drain_retired(|| Ok(()))?);
        Ok(())
    }
    fn graph_count(&self, layer: usize) -> usize { self.graphs.count(|(l, _)| *l == layer) }
    pub(crate) fn declare_startup(&mut self, keys: &[(usize, Vec<usize>)]) -> Result<()> {
        ensure!((self.decode || keys.is_empty()) && !self.window.is_windowed() && self.cold.is_none(),
            "sparse startup requires an idle owner and decode recipes or an empty eager inventory");
        self.storage.stream.require_complete()?;
        self.graphs.declare_startup(keys)
    }
    pub(crate) fn finish_startup(&mut self) -> Result<()> {
        ensure!(self.cold.is_none(), "cannot seal sparse bank with pending capture");
        self.storage.stream.require_complete()?;
        self.graphs.finish_startup()
    }

    pub(crate) fn startup_key(&self, layer: usize, rows: usize, sink: CuteafdDeviceBuffer,
        selected: Option<CuteafdDeviceBuffer>, aligned: bool, tail: Option<&dyn AttentionGraphTail>) -> Result<(usize, Vec<usize>)> {
        ensure!(layer < 40 && rows <= self.batch_rows && self.window.admits(rows),
            "sparse startup shape exceeds owned storage");
        let plan = self.attention_plan.context("sparse startup requires an admitted reduction plan")?;
        let recipe = self.storage.kernel.startup_batch_recipe(plan, rows, layer >= 2, aligned)?;
        ensure!(selected.is_some() == (layer >= 2), "sparse startup selection differs from layer");
        for (buffer, bytes) in std::iter::once((sink, HEADS * 4))
            .chain(selected.map(|buffer| (buffer, rows * 2048))) {
            ensure!(!buffer.ptr.is_null() && buffer.bytes >= bytes
                && buffer.device_id == self.storage.query.buffer.device_id,
                "sparse startup binding device or capacity differs");
        }
        let mut fingerprint = vec![usize::MAX, layer, rows, sink.ptr as usize,
            selected.map_or(0, |buffer| buffer.ptr as usize), recipe.backend_key()];
        if let Some(tail) = tail {
            fingerprint.push(usize::MAX - 1);
            fingerprint.extend(tail.identity());
        }
        Ok((layer, fingerprint))
    }

    /// # Safety
    /// Stable selection, sink, tail weights and tail buffers must outlive this
    /// bank. The tail must already have capture-only host phase prepared; it must
    /// not upload request data or publish/commit state. Restore that host phase
    /// after this call, including errors. No recorded device work executes here.
    pub(crate) unsafe fn capture_startup(&mut self, layer: usize, rows: usize,
        sink: CuteafdDeviceBuffer, selected: Option<CuteafdDeviceBuffer>, aligned: bool,
        mut tail: Option<&mut dyn AttentionGraphTail>) -> Result<()> {
        ensure!(self.decode && !self.window.is_windowed() && self.cold.is_none(),
            "sparse startup requires an idle full decode owner");
        self.storage.stream.require_complete()?;
        let key = self.startup_key(layer, rows, sink, selected, aligned,
            tail.as_ref().map(|tail| &**tail as &dyn AttentionGraphTail))?;
        ensure!(self.graphs.capture_allowed(&key), "sparse startup capture outside declared inventory");
        ensure!(!self.graphs.contains(&key), "sparse startup key already captured");
        let plan = self.attention_plan.context("sparse startup reduction plan absent")?;
        let recipe = self.storage.kernel.startup_batch_recipe(plan, rows, layer >= 2, aligned)?;
        // SAFETY: this owner is drained and the caller retains every bound plane.
        unsafe { self.storage.stream.library.cuda_graph_begin_capture(self.storage.stream.raw)?; }
        let queued = (|| unsafe {
            self.storage.kernel.record_startup_batch(&recipe, self.storage.query.buffer, sink,
                self.storage.metadata.buffer, selected, self.storage.output.buffer, self.storage.descriptors.buffer,
                self.storage.replay_begins.buffer, self.storage.split_scratch.buffer, self.storage.stream.raw)?;
            if let Some(tail) = tail.as_deref_mut() {
                tail.enqueue(&QueuedSparseAttention { values: slice(self.storage.output.buffer, 0, rows * HEADS * 1024),
                    layer, rows }, self.storage.stream.raw)?;
            }
            Ok(())
        })();
        // End capture even if a tail failed, so the owner is not left capturing.
        let captured = unsafe { self.storage.stream.library.cuda_graph_end_capture(self.storage.stream.raw) };
        match (queued, captured) {
            (Ok(()), Ok(graph)) => {
                if let Err(error) = self.insert_graph(key.0, key.1, graph) {
                    unsafe { self.storage.stream.library.cuda_graph_exec_destroy(graph)?; }
                    return Err(error);
                }
                Ok(())
            }
            (Err(error), Ok(graph)) => {
                unsafe { self.storage.stream.library.cuda_graph_exec_destroy(graph)?; }
                Err(error)
            }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
        }
    }
    fn insert_graph(&mut self, layer: usize, fingerprint: Vec<usize>, graph: *mut c_void) -> Result<()> {
        ensure!(self.graphs.capture_allowed(&(layer, fingerprint.clone())),
            "sparse graph capture outside startup inventory or after seal");
        let device = self.storage.stream.library.cuda_get_device()?;
        // SAFETY: the lane guard retains all captured cache/selection/tail storage.
        let owner = unsafe { GraphOwner::new(self.storage.stream.library, device, graph, ())? };
        tracing::debug!(target: "cuteafd::graph_capture", site = "sparse_attention", device, layer,
            bank = self as *const Self as usize, key = ?fingerprint, "native binding graph captured");
        self.graphs.insert((layer, fingerprint), owner, None);
        Ok(())
    }
    unsafe fn enqueue(
        &self,
        sink: CuteafdDeviceBuffer,
        launches: &[RequestLaunch],
        selected: Option<CuteafdDeviceBuffer>,
        batch: Option<&V41SparseBatch>,
    ) -> Result<()> {
        if let Some(batch) = batch {
            return unsafe { self.storage.kernel.launch_batch(batch, self.storage.query.buffer, sink,
                self.storage.metadata.buffer, selected, self.storage.output.buffer, self.storage.descriptors.buffer,
                self.storage.replay_begins.buffer, self.storage.split_scratch.buffer, self.storage.stream.raw) };
        }
        let mut offset = 0;
        for l in launches {
            unsafe {
                self.storage.kernel.launch(
                    slice(self.storage.query.buffer, offset * (HEADS*1024), l.rows * (HEADS*1024)),
                    sink,
                    slice(self.storage.metadata.buffer, offset * 80, l.rows * 80),
                    selected.map(|b| slice(b, offset * 2048, l.rows * 2048)),
                    &l.window,
                    l.source.as_ref(),
                    slice(self.storage.output.buffer, offset * (HEADS*1024), l.rows * (HEADS*1024)),
                    l.rows,
                    l.width,
                    // All request launches share this arena on one stream;
                    // each merge finishes before the next partial write.
                    (l.rows <= 16).then_some((
                        self.storage.split_scratch.buffer,
                        if l.source.is_some() { 10 } else { 2 },
                    )),
                    self.storage.stream.raw,
                )?;
            }
            offset += l.rows;
        }
        Ok(())
    }
    /// # Safety
    /// No external writes race query, cache/selection views or this wave. Sink
    /// is the matching layer's finite checkpoint sink on this device.
    pub unsafe fn execute_query<'s>(
        &'s mut self,
        query: &'s AttentionQueryOutput<'s>,
        sink: CuteafdDeviceBuffer,
        requests: &'s [AttentionRequest<'s>],
        selection: Option<&'s IndexSelectionOutput<'s>>,
    ) -> Result<SparseAttentionOutput<'s>> {
        let binding = query.binding()?;
        let tokens = query.tokens()?;
        ensure!(
            query.rows == tokens.len()
                && query.rows <= self.capacity && self.window.admits(query.rows)
                && query.rotated.bytes == query.rows*(HEADS*1024)
            && query.rotated.device_id == self.storage.query.buffer.device_id
                && requests
                    .iter()
                    .flat_map(|r| r.positions.iter().copied())
                    .eq(tokens.iter().copied()),
            "attention query token order or device differs"
        );
        if let Some(s) = selection {
            s.validate_query(binding)?;
        }
        self.acquire_scratch()?;
        let copied = unsafe {
            self.storage.stream.library.copy_d2d_async(
                self.storage.query.buffer, query.rotated, query.rotated.bytes, self.storage.stream.raw,
            )
        };
        if let Err(error) = copied {
            self.synchronize()?;
            return Err(error);
        }
        let mut out = unsafe { self.execute(query.layer, sink, requests, selection)? };
        out.query = Some(binding);
        out.tokens = Some(tokens);
        Ok(out)
    }
    /// # Safety
    /// Input contains finite, rotated BF16 attention queries in request/token
    /// order, with producer writes drained. Sink is this layer's finite FP32
    /// checkpoint tensor on the initialized device. No external writes race
    /// inputs or cache owners. Output retains proposal/selection borrows until
    /// its consumers finish, preventing accepted commits or producer reuse.
    pub unsafe fn execute<'s>(
        &'s mut self,
        layer: usize,
        sink: CuteafdDeviceBuffer,
        requests: &'s [AttentionRequest<'s>],
        selection: Option<&'s IndexSelectionOutput<'s>>,
    ) -> Result<SparseAttentionOutput<'s>> {
        let library = self.storage.stream.library;
        let stream = self.storage.stream.raw;
        let result = unsafe { self.execute_staged(layer, sink, requests, selection) };
        let drained = unsafe { library.cuda_stream_synchronize(stream) };
        let queued = result.and_then(|v| drained.map(|()| v))?;
        Ok(SparseAttentionOutput { query: None, tokens: None, values: queued.values,
            layer: queued.layer, rows: queued.rows, _inputs: PhantomData })
    }
    /// # Safety
    /// Same completed query/cache inputs as execute_query. The continuation may
    /// enqueue consumers only on the supplied stream, retaining every allocation
    /// until this method returns. Returned values must not publish GPU results
    /// until the successful drain performed here. No asynchronous suspension.
    pub unsafe fn execute_query_then<T>(
        &mut self, query: &AttentionQueryOutput<'_>, sink: CuteafdDeviceBuffer,
        requests: &[AttentionRequest<'_>], selection: Option<&IndexSelectionOutput<'_>>,
        consume: impl FnOnce(QueuedSparseAttention, *mut c_void) -> Result<T>,
    ) -> Result<T> {
        unsafe { self.query_then(query, sink, requests, selection, consume) }
    }
    /// Complete the queued attention chain: ordered inside a stage chain,
    /// otherwise awaited on the host.
    pub async fn wait_chain(&self) -> Result<()> {
        unsafe { crate::shared::memory::chain::finish_cooperative(&self.storage.stream).await }
    }
    pub fn drain_chain(&mut self) -> Result<()> {
        let drained = self.synchronize(); self.cold = None; drained
    }
    pub fn chain_stream(&self) -> *mut c_void { self.storage.stream.raw }
    /// # Safety
    /// Retain query/cache/selection buffers until warmup and downstream consumers
    /// complete or drain. No new submission may reuse this owner while pending.
    pub unsafe fn enqueue_query_prepared(&mut self, query: &AttentionQueryOutput<'_>,
        sink: CuteafdDeviceBuffer, requests: &[AttentionRequest<'_>],
        selection: Option<&IndexSelectionOutput<'_>>, defer_warmup: bool,
        tail: Option<&mut dyn AttentionGraphTail>) -> Result<Option<QueuedSparseAttention>> {
        ensure!(self.cold.is_none(), "cold attention preparation already pending");
        let binding = query.binding()?;
        let tokens = query.tokens()?;
        ensure!(query.rows == tokens.len() && query.rows <= self.capacity && self.window.admits(query.rows)
            && query.rotated.bytes == query.rows*(HEADS*1024)
            && query.rotated.device_id == self.storage.query.buffer.device_id
            && requests.iter().flat_map(|r| r.positions.iter().copied()).eq(tokens.iter().copied()),
            "attention query token order or device differs");
        if let Some(s) = selection { s.validate_query(binding)?; }
        let result = (|| unsafe {
            self.acquire_scratch()?;
            crate::shared::memory::chain::join(self.storage.stream.library, self.storage.stream.raw)?;
            self.storage.stream.library.copy_d2d_async(self.storage.query.buffer, query.rotated, query.rotated.bytes, self.storage.stream.raw)?;
            self.execute_staged_inner(query.layer, sink, requests, selection, defer_warmup, tail)
        })();
        if result.is_err() { self.drain_chain()?; }
        result
    }
    /// # Safety
    /// The caller retains the external inputs represented by the owned cold plan.
    /// Returned attention is queued, ready for consumers on chain_stream().
    pub async unsafe fn finish_prepare(&mut self, mut tail: Option<&mut dyn AttentionGraphTail>) -> Result<QueuedSparseAttention> {
        let plan = self.cold.take().context("cold attention preparation absent")?;
        #[cfg(test)]
        eprintln!("queued sparse warmup layer={} rows={} batched={}", plan.layer, plan.rows, plan.batch.is_some());
        self.storage.stream.wait().await?;
        ensure!(plan.tail_identity == tail.as_ref().map(|t| t.identity()), "cold attention tail identity changed");
        if let Some(tail) = tail.as_deref_mut() { unsafe { tail.restore_warmup()?; } }
        self.warmed_kernels |= plan.needed;
        let graph = unsafe { self.capture_plan(&plan, &mut tail)? };
        self.insert_graph(plan.layer, plan.fingerprint, graph)?;
        let launched = unsafe { self.storage.stream.library.cuda_graph_launch(graph, self.storage.stream.raw) };
        if let Err(error) = launched { self.synchronize()?; return Err(error); }
        Ok(QueuedSparseAttention { values: slice(self.storage.output.buffer, 0, plan.rows * (HEADS*1024)),
            layer: plan.layer, rows: plan.rows })
    }
    unsafe fn capture_plan(&self, plan: &ColdSparse, tail: &mut Option<&mut dyn AttentionGraphTail>) -> Result<*mut c_void> {
        unsafe { self.storage.stream.library.cuda_graph_begin_capture(self.storage.stream.raw)?; }
        let launched = (|| unsafe {
            self.enqueue(plan.sink, &plan.launches, plan.selected, plan.batch.as_ref())?;
            if let Some(tail) = tail.as_deref_mut() { tail.enqueue(&QueuedSparseAttention {
                values: slice(self.storage.output.buffer, 0, plan.rows * (HEADS*1024)), layer: plan.layer, rows: plan.rows }, self.storage.stream.raw)?; }
            Ok(())
        })();
        let captured = unsafe { self.storage.stream.library.cuda_graph_end_capture(self.storage.stream.raw) };
        match (launched, captured) {
            (Ok(()), Ok(graph)) => Ok(graph),
            (Err(error), Ok(graph)) => { unsafe { self.storage.stream.library.cuda_graph_exec_destroy(graph)?; } Err(error) }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// Retain all query/cache/selection and tail owners through this direct drain.
    pub unsafe fn execute_query_graph(&mut self, query: &AttentionQueryOutput<'_>,
        sink: CuteafdDeviceBuffer, requests: &[AttentionRequest<'_>], selection: Option<&IndexSelectionOutput<'_>>,
        tail: &mut dyn AttentionGraphTail) -> Result<()> {
        struct Drain<'a>(&'a NativeLibrary, *mut c_void, bool);
        impl Drop for Drain<'_> {
            fn drop(&mut self) { if !self.2 {
                if let Err(error) = unsafe { self.0.cuda_stream_synchronize(self.1) } {
                    crate::shared::decode_graph::fatal_drain(Err(error), "direct attention unwind");
                }
            } }
        }
        let mut drain = Drain(self.storage.stream.library, self.storage.stream.raw, false);
        let queued = unsafe { self.enqueue_query_prepared(query, sink, requests, selection, false, Some(tail)) };
        let drained = if queued.is_err() { self.synchronize() }
            else { unsafe { crate::shared::memory::chain::finish(self.storage.stream.library, self.storage.stream.raw) } };
        drain.2 = true;
        queued.and_then(|value| value.context("direct graph unexpectedly deferred")).and(drained)
    }
    unsafe fn query_then<T>(&mut self, query: &AttentionQueryOutput<'_>, sink: CuteafdDeviceBuffer,
        requests: &[AttentionRequest<'_>], selection: Option<&IndexSelectionOutput<'_>>,
        consume: impl FnOnce(QueuedSparseAttention, *mut c_void) -> Result<T>) -> Result<T> {
        let binding = query.binding()?;
        let tokens = query.tokens()?;
        ensure!(query.rows == tokens.len() && query.rows <= self.capacity && self.window.admits(query.rows)
            && query.rotated.bytes == query.rows*(HEADS*1024)
            && query.rotated.device_id == self.storage.query.buffer.device_id
            && requests.iter().flat_map(|r| r.positions.iter().copied()).eq(tokens.iter().copied()),
            "attention query token order or device differs");
        if let Some(s) = selection { s.validate_query(binding)?; }
        let library = self.storage.stream.library;
        let stream = self.storage.stream.raw;
        // Also drain on unwinding before the caller can release consumer owners.
        struct Drain<'a>(&'a NativeLibrary, *mut c_void, bool);
        impl Drop for Drain<'_> {
            fn drop(&mut self) { if !self.2 {
                if let Err(error) = unsafe { self.0.cuda_stream_synchronize(self.1) } {
                    crate::shared::decode_graph::fatal_drain(Err(error), "attention continuation unwind");
                }
            }}
        }
        let mut drain = Drain(library, stream, false);
        let result = (|| unsafe {
            self.acquire_scratch()?;
            library.copy_d2d_async(self.storage.query.buffer, query.rotated, query.rotated.bytes, stream)?;
            let queued = self.execute_staged(query.layer, sink, requests, selection)?;
            consume(queued, stream)
        })();
        let drained = unsafe { library.cuda_stream_synchronize(stream) };
        drain.2 = true;
        result.and_then(|v| drained.map(|()| v))
    }
    unsafe fn execute_staged<'s>(
        &'s mut self,
        layer: usize,
        sink: CuteafdDeviceBuffer,
        requests: &'s [AttentionRequest<'s>],
        selection: Option<&'s IndexSelectionOutput<'s>>,
    ) -> Result<QueuedSparseAttention> {
        unsafe { self.execute_staged_inner(layer, sink, requests, selection, false, None)? }
            .context("direct attention unexpectedly deferred")
    }
    unsafe fn execute_staged_inner(&mut self, layer: usize, sink: CuteafdDeviceBuffer,
        requests: &[AttentionRequest<'_>], selection: Option<&IndexSelectionOutput<'_>>,
        defer_warmup: bool, mut tail: Option<&mut dyn AttentionGraphTail>) -> Result<Option<QueuedSparseAttention>> {
        self.acquire_scratch()?;
        ensure!(self.cold.is_none(), "attention warmup is pending");
        ensure!(
            layer < 40 && !requests.is_empty() && requests.len() <= cuteafd_core::V41_MAX_REQUESTS,
            "invalid attention layer or request count"
        );
        let rows: usize = requests.iter().map(|r| r.positions.len()).sum();
        ensure!(
            rows > 0 && rows <= self.capacity && self.window.admits(rows),
            "attention rows exceed capacity or differ from slot window"
        );
        ensure!(
            sink.bytes >= HEADS*4
                && !sink.ptr.is_null()
                && sink.device_id == self.storage.query.buffer.device_id,
            "attention sink differs"
        );
        ensure!(
            selection.is_some() == (layer >= 2),
            "attention selection presence differs"
        );
        let source_layer = if layer >= 20 {
            20
        } else if layer >= 14 {
            14
        } else if layer >= 8 {
            8
        } else {
            2
        };
        let mut bindings = Vec::with_capacity(rows);
        let mut launches = Vec::with_capacity(requests.len());
        let mut metadata = Vec::with_capacity(rows * 10);
        let mut fingerprint = vec![layer, rows, sink.ptr as usize];
        for (i, r) in requests.iter().enumerate() {
            let w = r.window;
            ensure!(
                w.layer == layer && w.binding.same_pool(requests[0].window.binding),
                "attention window layer or owner differs"
            );
            ensure!(
                !r.positions.is_empty() && r.positions.windows(2).all(|p| p[0] < p[1]),
                "empty or unordered attention request"
            );
            ensure!(
                requests[..i].iter().all(|p| p.window.request != w.request),
                "duplicate attention request"
            );
            ensure!(
                r.source.is_some() == (layer >= 2),
                "attention source presence differs"
            );
            // Device metadata supplies each query's causal width on replay,
            // independent of the selected draft suffix and its row count.
            let width = 0;
            for &position in r.positions {
                let wm = w.metadata(position)?;
                metadata.extend(wm);
                if let Some(s) = r.source {
                    ensure!(
                        s.request_id() == w.request
                            && s.source_layer == source_layer
                            && s.first_token() == wm[0],
                        "attention source request, layer or committed position differs"
                    );
                    ensure!(
                        s.binding()
                            .same_pool(requests[0].source.context("source absent")?.binding()),
                        "attention source owner differs"
                    );
                    metadata.extend(s.metadata(position)?);
                    bindings.push((s.binding(), position));
                } else {
                    metadata.extend([0; 6]);
                }
            }
            let window = V41SparseWindow {
                values: w.cache.values,
                scales: w.cache.scales,
                proposals: w.values,
                proposal_scales: w.scales,
                end: w.cache.device_end,
                proposal_capacity: w.capacity,
                replay_begins: (HEADS==32 || w.cache.begin != 0).then(|| slice(self.storage.replay_begins.buffer,
                    (metadata.len() / 10 - r.positions.len()) * 8, r.positions.len() * 8)),
            };
            let source = r.source.map(|s| V41SparseSource {
                values: s.kv_cache.values,
                scales: s.kv_cache.scales,
                proposals: s.kv_values,
                proposal_scales: s.kv_scales,
                pages: s.kv_cache.device_pages,
                end: s.kv_cache.device_rows,
                capacity: s.kv_cache.values.bytes / V41Kv::COMPRESSED_VALUE_BYTES,
                proposal_capacity: s.capacity,
                page_stride: s.kv_cache.device_pages.bytes / 4,
            });
            fingerprint.extend([r.positions.len(), width, window.proposal_capacity]);
            let mut buffers = vec![
                window.values,
                window.scales,
                window.proposals,
                window.proposal_scales,
                window.end,
            ];
            fingerprint.push(usize::from(window.replay_begins.is_some()));
            if let Some(bounds) = window.replay_begins { buffers.push(bounds); }
            if let Some(s) = &source {
                fingerprint.extend([s.capacity, s.proposal_capacity, s.page_stride]);
                buffers.extend([
                    s.values,
                    s.scales,
                    s.proposals,
                    s.proposal_scales,
                    s.pages,
                    s.end,
                ]);
            }
            for b in buffers {
                ensure!(
                    b.device_id == self.storage.query.buffer.device_id && !b.ptr.is_null(),
                    "attention proposal device differs"
                );
                fingerprint.extend([b.ptr as usize, b.bytes]);
            }
            launches.push(RequestLaunch {
                window,
                source,
                rows: r.positions.len(),
                width,
            });
        }
        let selected = if let Some(s) = selection {
            s.validate_attention(layer, &bindings)?;
            ensure!(
                s.selected.device_id == self.storage.query.buffer.device_id
                    && s.selected.bytes == rows * 2048,
                "attention selection device or size differs"
            );
            fingerprint.push(s.selected.ptr as usize);
            Some(s.selected)
        } else {
            None
        };
        let fixed_plan = self.decode.then_some(self.attention_plan).flatten();
        ensure!(fixed_plan.is_none() || rows <= self.batch_rows,
            "decode rows exceed admitted attention execution capacity");
        // Decode always uses descriptors, including C1; prefill remains explicit legacy dispatch.
        let batch = if fixed_plan.is_some() || (self.storage.kernel.supports_batch() && launches.len() > 1
            && rows <= self.batch_rows && launches.iter().all(|launch| launch.rows <= 16)) {
            let bindings: Vec<_> = launches.iter()
                .map(|launch| (&launch.window, launch.source.as_ref(), launch.rows)).collect();
            let batch = if let Some(plan) = fixed_plan {
                self.storage.kernel.prepare_batch_planned(self.storage.query.buffer, sink, self.storage.metadata.buffer,
                    selected, &bindings, self.storage.output.buffer, self.storage.descriptors.buffer,
                    self.storage.replay_begins.buffer, self.storage.split_scratch.buffer, plan)?
            } else { self.storage.kernel.prepare_batch(self.storage.query.buffer, sink, self.storage.metadata.buffer,
                selected, &bindings, self.storage.output.buffer, self.storage.descriptors.buffer,
                self.storage.replay_begins.buffer, self.storage.split_scratch.buffer)? };
            let (host, bytes) = self.descriptor_staging.region(layer);
            bytes[..batch.bytes().len()].copy_from_slice(batch.bytes());
            unsafe { self.storage.stream.library.copy_host_buffer_h2d_async(self.storage.descriptors.buffer,
                host, batch.bytes().len(), self.storage.stream.raw)?; }
            // The kernel reads current descriptors rather than capturing external
            // cache pointers or request row counts. Selection is lane-owned and
            // stable; still include its address to guard any future owner change.
            fingerprint = vec![usize::MAX, layer, rows, sink.ptr as usize,
                selected.map_or(0, |buffer| buffer.ptr as usize), batch.backend_key()];
            Some(batch)
        } else { None };
        let (host, bytes) = self.staging.region(layer);
        for (i, m) in metadata.into_iter().enumerate() {
            bytes[i * 8..i * 8 + 8].copy_from_slice(&m.to_ne_bytes());
        }
        unsafe {
            self.storage.stream.library.copy_host_buffer_h2d_async(
                self.storage.metadata.buffer, host, rows * 80, self.storage.stream.raw,
            )?;
        }
        if HEADS==32 || batch.is_some() || requests.iter().any(|r| r.window.cache.begin != 0) {
            let mut row = 0;
            let (host, bytes) = self.replay_staging.region(layer);
            for request in requests {
                for _ in request.positions {
                    bytes[row * 8..row * 8 + 8]
                        .copy_from_slice(&request.window.cache.begin.to_ne_bytes());
                    row += 1;
                }
            }
            unsafe {
                self.storage.stream.library.copy_host_buffer_h2d_async(
                    self.storage.replay_begins.buffer, host, rows * 8, self.storage.stream.raw,
                )?;
            }
        }
        let queued = QueuedSparseAttention { values: slice(self.storage.output.buffer, 0, rows * (HEADS*1024)), layer, rows };
        let tail_identity = tail.as_ref().map(|t| t.identity());
        if let Some(identity) = &tail_identity {
            fingerprint.push(usize::MAX - 1); fingerprint.extend(identity);
            unsafe { tail.as_deref_mut().unwrap().prepare(self.storage.stream.raw)?; }
        }
        let key = (layer, fingerprint.clone());
        let cached = if self.window.is_windowed() { None } else { self.graphs.get(&key).map(|owner| owner.raw) };
        if crate::shared::decode_graph::census::eager("sparse") {
            // Retain every executable; only dispatch changes between census arms.
            unsafe {
                self.storage.stream.library.cuda_graph_census_call(self.storage.stream.raw, c"sparse", rows, false,
                    crate::shared::decode_graph::census::arm(), || {
                        self.enqueue(sink, &launches, selected, batch.as_ref())?;
                        if let Some(tail) = tail.as_deref_mut() { tail.enqueue(&queued, self.storage.stream.raw)?; }
                        Ok(())
                    })?;
            }
            return Ok(Some(queued));
        }
        let fixed_limit = super::graph_policy::fixed_binding_limit();
        if cached.is_none() && (self.window.is_windowed() || !super::graph_policy::captures_shape(rows as u32)
            || !self.graphs.capture_allowed(&key)
            || fixed_limit.is_some_and(|limit| self.graph_count(layer) >= limit)
            || exact_small_shape(batch.is_some(), rows, self.batch_rows)) {
            // Exact fingerprints can include launch recipes beyond row count.
            // Freeze the bounded bank rather than evicting and recapturing them.
            tracing::debug!(target: "cuteafd::target_step", layer, rows,
                retained=self.graph_count(layer), "V4.1 sparse attention eager binding");
            unsafe {
                self.enqueue(sink, &launches, selected, batch.as_ref())?;
                if let Some(tail) = tail.as_deref_mut() { tail.enqueue(&queued, self.storage.stream.raw)?; }
            }
            return Ok(Some(queued));
        }
        let graph = if let Some(graph) = cached {
            self.graphs.launch(&key);
            if let Some(tail) = tail.as_deref_mut() { unsafe { tail.replay_state()?; } }
            if crate::shared::decode_graph::census::enabled() {
                unsafe {
                    self.storage.stream.library.cuda_graph_census_call(self.storage.stream.raw, c"sparse", rows, true,
                        crate::shared::decode_graph::census::arm(), || self.storage.stream.library.cuda_graph_launch(graph, self.storage.stream.raw))?;
                }
                return Ok(Some(queued));
            }
            graph
        } else {
            tracing::debug!(target: "cuteafd::graph_capture", site="sparse_attention", layer, rows,
                retained=self.graph_count(layer), limit=self.graph_limit, key=?fingerprint,
                "graph cache miss");
            tracing::debug!(target: "cuteafd::timing", layer, rows, batched = batch.is_some(), "sparse graph capture");
            if !defer_warmup { self.synchronize()?; }
            let limit = fixed_limit.unwrap_or_else(|| if batch.is_some() { self.batch_rows } else { self.graph_limit });
            if self.graph_count(layer) >= limit {
                self.graphs.retire_oldest(|(l, _)| *l == layer);
                // Preserve ownership until a later stream drain, including deferred warmup.
                let library = self.storage.stream.library; let stream = self.storage.stream.raw;
                drop(self.graphs.drain_retired(|| unsafe { library.cuda_stream_synchronize(stream) })?);
            }
            // Native dispatch has four row recipes per cache format: split,
            // unsplit single-group, two-group and four-group. Changing pointers
            // or launch dimensions does not require executing a warmed recipe
            // again before capture. Keep the fixed-policy path unchanged.
            let needed = if batch.is_some() { 1u16 << (8 + usize::from(selected.is_some())) } else { launches.iter().fold(0u16, |mask, launch| {
                let recipe = if launch.rows <= 16 { 0 } else if launch.rows < 128 { 1 }
                    else if launch.rows < 256 { 2 } else { 3 };
                mask | (1 << (recipe + 4 * usize::from(launch.source.is_some())))
            }) };
            let needs_warmup = tail.is_some() || (batch.is_none() && self.graph_limit == 1) || self.warmed_kernels & needed != needed;
            let plan = ColdSparse { layer, rows, sink, launches, selected, batch, fingerprint, needed, tail_identity };
            if needs_warmup {
                let launched = (|| unsafe {
                    self.enqueue(plan.sink, &plan.launches, plan.selected, plan.batch.as_ref())?;
                    if let Some(tail) = tail.as_deref_mut() { tail.enqueue(&queued, self.storage.stream.raw)?; }
                    Ok(())
                })();
                if defer_warmup {
                    launched?;
                    self.cold = Some(plan);
                    return Ok(None);
                }
                launched.and(self.synchronize())?;
                if let Some(tail) = tail.as_deref_mut() { unsafe { tail.restore_warmup()?; } }
                self.warmed_kernels |= needed;
            }
            // Metadata uploads remain ordered on this stream; capture itself
            // contains no await. Previous uses of an evicted graph are complete.
            let graph = unsafe { self.capture_plan(&plan, &mut tail)? };
            self.insert_graph(layer, plan.fingerprint, graph)?;
            graph
        };
        let launched = unsafe {
            self.storage.stream
                .library
                .cuda_graph_launch(graph, self.storage.stream.raw)
        };
        launched?;
        Ok(Some(QueuedSparseAttention {
            values: slice(self.storage.output.buffer, 0, rows * (HEADS*1024)), layer, rows,
        }))
    }
}
impl LocalSparseAttentionWave<'_,32> {
    /// Split the live producer's full-width query directly into this wave's
    /// compact input. Keep the original query identity for selection validation;
    /// its other tensors remain on their producer device.
    ///
    /// # Safety
    /// The current device and copy kernel must match this wave. Peer access must
    /// be enabled when needed. Producer completion and replica publication must
    /// precede this stream's work through lane-local events. Retain the original
    /// query and all proposal/selection/tail owners through warmup and consumers,
    /// including error paths. Sink contains the corresponding 32 checkpoint heads.
    pub unsafe fn enqueue_split_query_prepared(&mut self,
        query: &AttentionQueryOutput<'_>, first_head: usize, copy: &V41PeerCopy<'_>,
        sink: CuteafdDeviceBuffer, requests: &[AttentionRequest<'_>],
        selection: Option<&IndexSelectionOutput<'_>>, defer_warmup: bool,
        tail: Option<&mut dyn AttentionGraphTail>) -> Result<Option<QueuedSparseAttention>> {
        ensure!(self.cold.is_none(), "cold attention preparation already pending");
        ensure!(first_head == 0 || first_head == 32, "invalid compact attention head half");
        let binding = query.binding()?;
        let tokens = query.tokens()?;
        ensure!(binding.layer() == query.layer && query.rows > 0
            && query.rows == tokens.len() && query.rows <= self.capacity && self.window.admits(query.rows)
            && query.rows.checked_mul(64*1024) == Some(query.rotated.bytes)
            && requests.iter().flat_map(|r| r.positions.iter().copied()).eq(tokens.iter().copied()),
            "split attention query layer, extent or token order differs");
        if let Some(s) = selection { s.validate_query(binding)?; }
        let offset = first_head*1024;
        let input = slice(query.rotated, offset, query.rotated.bytes-offset);
        let result = (|| unsafe {
            self.acquire_scratch()?;
            copy.launch_rows(self.storage.query.buffer, input, 32*1024, query.rows,
                32*1024, 64*1024, self.storage.stream.raw)?;
            self.execute_staged_inner(query.layer, sink, requests, selection, defer_warmup, tail)
        })();
        if result.is_err() { self.drain_chain()?; }
        result
    }
}
impl<const HEADS:usize> Drop for LocalSparseAttentionWave<'_,HEADS> {
    fn drop(&mut self) {
        if let Err(e) = self.clear_graph() {
            crate::shared::decode_graph::fatal_drain(Err(e), "sparse attention");
        }
    }
}

#[cfg(test)]
#[path = "v41_sparse_attention/prefix_tests.rs"]
mod prefix_tests;

#[cfg(test)]
mod plan_installation_tests {
    use super::validate_plan_installation;
    #[test]
    fn plan_is_load_only_and_reserves_full_execution_capacity() {
        assert!(validate_plan_installation(false, 0, false, 64, 64).is_ok());
        assert!(validate_plan_installation(true, 0, false, 64, 64).is_err());
        assert!(validate_plan_installation(false, 1, false, 64, 64).is_err());
        assert!(validate_plan_installation(false, 0, true, 64, 64).is_err());
        assert!(validate_plan_installation(false, 0, false, 6, 64).is_err());
    }
}

#[cfg(test)]
mod exact_small_shape_tests {
    use super::exact_small_shape;
    #[test]
    fn prefill_exact_keys_never_enter_the_decode_bank() {
        for rows in [1, 6, 52, 57, 64] {
            assert!(exact_small_shape(false, rows, 64), "exact prefill rows {rows} must run eagerly");
            assert!(!exact_small_shape(true, rows, 64), "descriptor decode rows {rows} must stay capturable");
        }
        // Full prefill chunks and the decoder replay keep their own retained graphs.
        for rows in [128, 2048] { assert!(!exact_small_shape(false, rows, 64)); }
    }
}

#[cfg(test)]
mod storage_bytes_tests {
    use super::*;
    #[test]
    fn sparse_storage_bytes_equal_planned_allocations() -> Result<()> {
        for unit in [4096, 2752, 2048] {
            assert_eq!(SparseAttentionStorage::<64>::storage_bytes(unit)?,
                SparseAttentionStorage::<64>::allocation_sizes(unit)?.into_iter().sum::<usize>());
            assert_eq!(SparseAttentionStorage::<32>::storage_bytes(unit)?,
                SparseAttentionStorage::<32>::allocation_sizes(unit)?.into_iter().sum::<usize>());
        }
        Ok(())
    }
}

#[cfg(test)]
mod shared_storage_tests {
    use super::*;
    use crate::shared::memory::{chain::StageChain, device::{Allocation, Device, Stream}};
    fn run<const HEADS: usize>(guarded: bool) -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        let device = Device { library: &lib, id: 0 };
        let storage = SparseAttentionStorage::<HEADS>::new(&lib, 4, &[0,1])?;
        let mut slots = [LocalSparseAttentionWave::on_storage(storage.clone())?, LocalSparseAttentionWave::on_storage(storage.clone())?];
        for (i,slot) in slots.iter_mut().enumerate() { slot.select_window(crate::shared::memory::row_window::RowWindow::within(i*4..i*4+4,8)?)?; }
        let scales=Allocation::new(device,128*16)?; let end=Allocation::new(device,8)?;
        let sink=Allocation::new(device,HEADS*4)?;
        let sources=[Allocation::new(device,128*512)?,Allocation::new(device,128*512)?];
        lib.copy_h2d(sources[0].buffer,&vec![0x38;128*512])?;
        lib.copy_h2d(sources[1].buffer,&vec![0x40;128*512])?;
        lib.copy_h2d(scales.buffer,&vec![127;128*16])?;lib.copy_h2d(end.buffer,&[0;8])?;
        lib.copy_h2d(sink.buffer,&vec![0;HEADS*4])?;
        let metadata:Vec<u8>=(0..4).flat_map(|pos| [0u64,0,4,pos,0,0,0,0,0,0].into_iter().flat_map(u64::to_ne_bytes)).collect();
        lib.copy_h2d(storage.metadata.buffer,&metadata)?;
        lib.copy_h2d(storage.replay_begins.buffer,&[0;32])?;
        lib.copy_h2d(storage.query.buffer,&vec![0;4*HEADS*1024])?;
        let enqueue=|slot: &LocalSparseAttentionWave<'_, HEADS>, seed: usize| -> Result<()> {
            slot.acquire_scratch()?;
            let window=V41SparseWindow { values:sources[seed].buffer,scales:scales.buffer,
                proposals:sources[seed].buffer,proposal_scales:scales.buffer,end:end.buffer,
                proposal_capacity:128,replay_begins:Some(storage.replay_begins.buffer) };
            unsafe { storage.kernel.launch(storage.query.buffer,sink.buffer,storage.metadata.buffer,None,
                &window,None,storage.output.buffer,4,0,None,storage.stream.raw) }
        };
        let output_bytes=4*HEADS*1024;
        // Private reference storage launches the identical native kernel.
        let private=SparseAttentionStorage::<HEADS>::new(&lib,4,&[0])?;
        lib.copy_h2d(private.metadata.buffer,&metadata)?;lib.copy_h2d(private.replay_begins.buffer,&[0;32])?;
        lib.copy_h2d(private.query.buffer,&vec![0;output_bytes])?;
        let window=V41SparseWindow { values:sources[0].buffer,scales:scales.buffer,
            proposals:sources[0].buffer,proposal_scales:scales.buffer,end:end.buffer,
            proposal_capacity:128,replay_begins:Some(private.replay_begins.buffer) };
        unsafe { private.kernel.launch(private.query.buffer,sink.buffer,private.metadata.buffer,None,
            &window,None,private.output.buffer,4,0,None,private.stream.raw)?;
            lib.cuda_stream_synchronize(private.stream.raw)?; }
        let mut want=vec![0;output_bytes];lib.copy_d2h(&mut want,private.output.buffer)?;
        for seed in 0..2 { enqueue(&slots[seed],seed)?; unsafe { lib.cuda_stream_synchronize(storage.stream.raw)?; } }
        let reader=Stream::new(device)?;let saved=Allocation::new(device,output_bytes)?;
        unsafe { lib.copy_d2d_async(saved.buffer,storage.output.buffer,output_bytes,reader.raw)?; }
        reader.drain()?;
        let chains=[StageChain::on_devices(&lib,&[0,1])?,StageChain::on_devices(&lib,&[0,1])?];
        let runtime=tokio::runtime::Builder::new_current_thread().build()?;
        let gate=unsafe { cuteafd_ffi::test_support::CudaStreamGate::new(&lib,reader.raw)? };
        runtime.block_on(chains[0].handle().scope(async {
            enqueue(&slots[0],0)?;
            unsafe { crate::shared::memory::chain::finish(&lib,storage.stream.raw)?;
                crate::shared::memory::chain::join(&lib,reader.raw)?;
                lib.copy_d2d_async(saved.buffer,storage.output.buffer,output_bytes,reader.raw)?;
                crate::shared::memory::chain::finish(&lib,reader.raw)?; }
            slots[0].release_scratch()
        }))?;
        SKIP_STORAGE_ACQUIRE.with(|flag|flag.set(!guarded));
        let queued=runtime.block_on(chains[1].handle().scope(async {
            enqueue(&slots[1],1)?;
            unsafe { crate::shared::memory::chain::finish(&lib,storage.stream.raw) }
        }));
        SKIP_STORAGE_ACQUIRE.with(|flag|flag.set(false)); queued?;
        let deadline=std::time::Instant::now()+std::time::Duration::from_secs(if guarded {2} else {10});
        let mut done=false;
        while std::time::Instant::now()<deadline {
            if unsafe { lib.cuda_stream_query(storage.stream.raw)? } { done=true;break; }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        drop(gate); for chain in &chains {chain.drain()?;} reader.drain()?;
        assert_eq!(done,!guarded,"sparse B completion before releasing A reader");
        let mut got=vec![0;output_bytes];lib.copy_d2h(&mut got,saved.buffer)?;
        if guarded {assert_eq!(got,want);} else {assert_ne!(got,want,"negative control did not clobber");}
        Ok(())
    }
    #[test]
    #[ignore = "requires native sparse CUDA kernels"]
    fn sparse_shared_scratch_preserves_gated_reader() -> Result<()> { for _ in 0..3 {run::<64>(true)?;} Ok(()) }
    #[test]
    #[ignore = "requires native sparse CUDA kernels"]
    fn sparse_without_guard_clobbers_gated_reader() -> Result<()> { for _ in 0..3 {run::<64>(false)?;} Ok(()) }
}
