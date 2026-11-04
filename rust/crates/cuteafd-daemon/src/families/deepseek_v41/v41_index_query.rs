//! Learned index-query projections and fused rotary/FP4 preparation on the RTX.
use crate::families::deepseek_v41::v41_attention_binding::QueryBinding;
use crate::families::deepseek_v41::v41_attention_query::AttentionQueryOutput;
use crate::shared::decode_graph::LayerGraphs;
use crate::shared::memory::{DeviceAllocation, LoadStream};
use crate::families::deepseek_v41::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary, V41AttentionOps, V41Compressor, V41Fp8Plan};
use cuteafd_loader::OfficialV41Catalog;
use std::{marker::PhantomData, rc::Rc};
use crate::shared::memory::row_window::{RowWindow, ScratchGuard};

#[cfg(test)]
thread_local! {
    pub(super) static SKIP_ACQUIRE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) struct IndexQueryWeights<'a> {
    library: &'a NativeLibrary,
    layer: usize,
    names: [String; 3],
    tensors: NativeRtxTensors<'a>,
    scales: DeviceAllocation<'a>,
}
impl<'a> IndexQueryWeights<'a> {
    fn names(layer: usize) -> Result<[String; 3]> {
        ensure!(
            [2, 8, 14, 20, 24, 28, 32, 36].contains(&layer),
            "layer has no learned index query producer"
        );
        Ok([
            format!("layers.{layer}.attn.indexer.wq_b.weight"),
            format!("layers.{layer}.attn.indexer.wq_b.scale"),
            format!("layers.{layer}.attn.indexer.weights_proj.weight"),
        ])
    }
    pub fn device_bytes(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
    ) -> Result<usize> {
        let resident = NativeRtxTensors::plan(catalog, &Self::names(layer)?)?;
        ensure!(
            resident == 5_575_680,
            "unexpected native index query tensor sizes"
        );
        Ok(resident
            + library
                .v41_fp8_matrix_info(1, 1280, 4096)?
                .packed_weight_scale_bytes as usize)
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
        budget: usize,
        staging_bytes: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(library, catalog, layer)? <= budget,
            "index query weights exceed budget"
        );
        let names = Self::names(layer)?;
        let tensors = NativeRtxTensors::load(library, catalog, &names, budget, staging_bytes)?;
        let kernel = library.v41_fp8_matrix_kernel(1, 1280, 4096)?;
        let scales =
            DeviceAllocation::new(library, kernel.info().packed_weight_scale_bytes as usize)?;
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        let launched =
            unsafe { crate::shared::experts::rtx::dense::v41_pack_scales(&kernel, tensors.get(&names[1])?, scales.buffer, stream.raw) };
        let drained = unsafe { library.cuda_stream_synchronize(stream.raw) };
        launched.and(drained)?;
        Ok(Self {
            library,
            layer,
            names,
            tensors,
            scales,
        })
    }
    pub fn wave(&self, capacity: u32, budget: usize) -> Result<IndexQueryWave<'_, 'a>> {
        ensure!(
            IndexQueryWave::device_bytes(self.library, capacity)? <= budget,
            "index query wave exceeds budget"
        );
        self.on_storage(self.storage(capacity, capacity, &[self.library.cuda_get_device()?])?)
    }
    pub fn storage(&self, rows: u32, unit: u32, readers: &[i32]) -> Result<Rc<IndexQueryStorage<'a>>> {
        ensure!(unit > 0 && unit <= rows, "index query unit outside reserve");
        let fp8 = self.library.v41_fp8_matrix_plan(unit, 1280, 4096)?;
        let scratch = DeviceAllocation::new(self.library, fp8.info().scratch_bytes as usize)?;
        let workspace = DeviceAllocation::new(self.library, V41Compressor::WORKSPACE_BYTES)?;
        ensure!(workspace.buffer.device_id == self.tensors.get(&self.names[0])?.device_id,
            "index query weight device differs");
        // SAFETY: workspace is retained by storage until its stream drains.
        let dense = unsafe { self.library.v41_compressor(workspace.buffer)? };
        let storage = IndexQueryStorage {
            stream: LoadStream { library: self.library, raw: self.library.cuda_stream_create()? },
            fp8, dense, _workspace: workspace, scratch,
            alpha: DeviceAllocation::new(self.library, 4)?,
            qr: DeviceAllocation::new(self.library, unit as usize * 2560)?,
            hidden: DeviceAllocation::new(self.library, unit as usize * 10240)?,
            positions: DeviceAllocation::new(self.library, unit as usize * 8)?,
            frequencies: DeviceAllocation::new(self.library, unit as usize * 256)?,
            projected: DeviceAllocation::new(self.library, unit as usize * 8192)?,
            head_projected: DeviceAllocation::new(self.library, unit as usize * 64)?,
            packed: DeviceAllocation::new(self.library, unit as usize * 2048)?,
            scales: DeviceAllocation::new(self.library, unit as usize * 128)?,
            head_weights: DeviceAllocation::new(self.library, unit as usize * 64)?,
            guard: ScratchGuard::new(self.library, readers)?, rows, unit,
        };
        // SAFETY: scratch and alpha belong to this stream and outlive initialization.
        unsafe { storage.fp8.initialize_scratch(storage.scratch.buffer, storage.alpha.buffer, storage.stream.raw)?;
            self.library.cuda_stream_synchronize(storage.stream.raw)?; }
        Ok(Rc::new(storage))
    }
    pub fn on_storage(&self, storage: Rc<IndexQueryStorage<'a>>) -> Result<IndexQueryWave<'_, 'a>> {
        ensure!(std::ptr::eq(storage.stream.library, self.library)
            && storage.hidden.buffer.device_id == self.tensors.get(&self.names[0])?.device_id,
            "index query storage library or device differs");
        Ok(IndexQueryWave {
            capacity: storage.unit, storage, window: RowWindow::NONE,
            norm: self.library.v41_attention_ops()?, weights: self,
            graphs: LayerGraphs::new(self.library, cuteafd_core::V41_VERIFY_ROWS as u32),
            ready: None, pending: None, origin: None, tokens: Vec::new(),
        })
    }
}
/// Index-query storage shared by all slots. Packed/scales/head_weights are
/// guarded unit-cap scratch: selection and sparse attention read them on fork
/// streams within the attention hook. Release after its last reader joins.
/// Plain unit-cap scratch: qr, hidden, positions, frequencies, projected,
/// head_projected, GEMM scratch, alpha and compressor workspace; one stream
/// consumes these within the projection hook. No cross-hook planes here.
/// Bindings, phase and graphs remain per slot; no host-staged metadata here.
pub(crate) struct IndexQueryStorage<'a> {
    guard: ScratchGuard<'a>,
    stream: LoadStream<'a>,
    fp8: V41Fp8Plan<'a>,
    dense: V41Compressor<'a>,
    _workspace: DeviceAllocation<'a>,
    scratch: DeviceAllocation<'a>,
    alpha: DeviceAllocation<'a>,
    qr: DeviceAllocation<'a>,
    hidden: DeviceAllocation<'a>,
    positions: DeviceAllocation<'a>,
    frequencies: DeviceAllocation<'a>,
    projected: DeviceAllocation<'a>,
    head_projected: DeviceAllocation<'a>,
    packed: DeviceAllocation<'a>,
    scales: DeviceAllocation<'a>,
    head_weights: DeviceAllocation<'a>,
    rows: u32,
    unit: u32,
}
impl Drop for IndexQueryStorage<'_> {
    fn drop(&mut self) {
        // SAFETY: owned stream; all its allocations are still live.
        crate::shared::decode_graph::fatal_drain(unsafe {
            self.stream.library.cuda_stream_synchronize(self.stream.raw)
        }.map_err(Into::into), "index query storage");
    }
}

#[derive(Clone, Copy)]
pub(crate) struct IndexQueryGeometry {
    pub layer: usize,
    pub packed: CuteafdDeviceBuffer,
    pub scales: CuteafdDeviceBuffer,
    pub head_weights: CuteafdDeviceBuffer,
}
pub(crate) struct IndexQueryOutput<'a> {
    origin: Option<QueryBinding>,
    tokens: &'a [u64],
    pub layer: usize,
    pub packed: CuteafdDeviceBuffer,
    pub scales: CuteafdDeviceBuffer,
    pub head_weights: CuteafdDeviceBuffer,
    _owner: PhantomData<&'a ()>,
}
impl IndexQueryOutput<'_> {
    pub(crate) fn geometry(&self) -> IndexQueryGeometry {
        IndexQueryGeometry { layer: self.layer, packed: self.packed, scales: self.scales,
            head_weights: self.head_weights }
    }
    pub fn origin(&self) -> Option<QueryBinding> {
        self.origin
    }
    pub fn bound_tokens(&self) -> Result<&[u64]> {
        ensure!(self.origin.is_some(), "index query has no origin");
        Ok(self.tokens)
    }
}
pub(crate) struct IndexQueryWave<'w, 'a> {
    storage: Rc<IndexQueryStorage<'a>>,
    window: RowWindow,
    norm: V41AttentionOps<'a>,
    weights: &'w IndexQueryWeights<'a>,
    capacity: u32,
    graphs: LayerGraphs<'w, 'a, IndexQueryWeights<'a>>,
    ready: Option<u32>,
    pending: Option<(u32, QueryBinding, bool)>,
    origin: Option<QueryBinding>,
    tokens: Vec<u64>,
}
impl<'w, 'a> IndexQueryWave<'w, 'a> {
    /// Retain captured weight owners while reusing this lane's fixed buffers.
    pub fn rebind(&mut self, weights: &'w IndexQueryWeights<'a>) -> Result<()> {
        ensure!(self.pending.is_none(), "index query pending");
        self.ready = None;
        self.origin = None;
        self.tokens.clear();
        ensure!(
            std::ptr::eq(self.storage.stream.library, weights.library)
                && weights.tensors.get(&weights.names[0])?.device_id
                    == self.storage.hidden.buffer.device_id,
            "index query rebound library or device differs"
        );
        self.storage.stream.require_complete()?;
        self.weights = weights;
        Ok(())
    }
}
impl IndexQueryWave<'_, '_> {
    pub(crate) fn startup_geometry(&self, rows: usize) -> Result<IndexQueryGeometry> {
        ensure!(rows > 0 && rows <= self.capacity as usize && !self.window.is_windowed()
            && self.pending.is_none() && self.ready.is_none(), "startup index query geometry requires idle full owner");
        let sized = |mut b: CuteafdDeviceBuffer, stride| { b.bytes = rows * stride; b };
        Ok(IndexQueryGeometry { layer: self.weights.layer,
            packed: sized(self.storage.packed.buffer, 2048),
            scales: sized(self.storage.scales.buffer, 128),
            head_weights: sized(self.storage.head_weights.buffer, 64) })
    }
    #[cfg(test)]
    pub(super) fn scratch_stream(&self) -> *mut std::ffi::c_void { self.storage.stream.raw }
    pub(crate) fn release_scratch(&self) -> Result<()> { self.storage.guard.release() }
    fn acquire_scratch(&self) -> Result<()> {
        #[cfg(test)]
        if SKIP_ACQUIRE.with(|skip| skip.get()) { return Ok(()); }
        // SAFETY: the owned writer stream is live on the current device.
        unsafe { self.storage.guard.acquire(self.storage.stream.raw) }
    }
    pub(crate) fn select_window(&mut self, window: RowWindow) -> Result<()> {
        crate::shared::memory::row_window::require_rest(self.pending.is_some() || self.ready.is_some(), "index query", "a query is pending or unconsumed")?;
        if let Some(range) = window.range() {
            ensure!(range.start < range.end && range.end <= self.storage.rows as usize
                && range.len() <= self.capacity as usize, "index query window outside reserve or unit");
        }
        self.window = window;
        Ok(())
    }
    pub fn storage_bytes(library: &NativeLibrary, rows: u32, unit: u32) -> Result<usize> {
        ensure!(unit > 0 && unit <= rows, "index query unit outside reserve");
        Self::device_bytes(library, unit)
    }
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        let info = library.v41_fp8_matrix_plan_info(capacity, 1280, 4096)?;
        Ok(V41Compressor::WORKSPACE_BYTES
            + info.scratch_bytes as usize
            + 4
            + capacity as usize * 23560)
    }
    /// Hook-local BF16 query-rank staging [unit,1280]; attention is copied on our stream.
    pub fn query_rank(&self) -> CuteafdDeviceBuffer {
        self.storage.qr.buffer
    }
    /// BF16 attention-input hidden states [capacity,5120], in the same row order.
    pub fn hidden(&self) -> CuteafdDeviceBuffer {
        self.storage.hidden.buffer
    }
    /// U64 token positions below the official context limit, in the same row order.
    pub fn positions(&self) -> CuteafdDeviceBuffer {
        self.storage.positions.buffer
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.storage.stream.library.cuda_stream_synchronize(self.storage.stream.raw) }
    }
    fn validate(&mut self, rows: u32) -> Result<()> {
        ensure!(self.pending.is_none(), "index query pending");
        self.ready = None;
        self.origin = None;
        self.tokens.clear();
        ensure!(
            rows > 0 && rows <= self.capacity && self.window.admits(rows as usize),
            "index query rows exceed capacity"
        );
        Ok(())
    }
    unsafe fn enqueue(&self, rows: u32) -> Result<()> {
        unsafe {
            self.norm.backbone_frequencies(
                self.storage.positions.buffer,
                self.storage.frequencies.buffer,
                rows,
                self.weights.layer as u32,
                self.storage.stream.raw,
            )?;
            self.storage.fp8.launch(
                self.storage.qr.buffer,
                self.weights.tensors.get(&self.weights.names[0])?,
                self.weights.scales.buffer,
                self.storage.scratch.buffer,
                self.storage.alpha.buffer,
                self.storage.projected.buffer,
                rows,
                self.storage.stream.raw,
            )?;
            self.storage.dense.weights_project(
                self.storage.hidden.buffer,
                self.weights.tensors.get(&self.weights.names[2])?,
                self.storage.head_projected.buffer,
                rows as usize,
                self.storage.stream.raw,
            )?;
            self.storage.dense.query_prepare(
                self.storage.projected.buffer,
                self.storage.frequencies.buffer,
                self.storage.head_projected.buffer,
                self.storage.packed.buffer,
                self.storage.scales.buffer,
                self.storage.head_weights.buffer,
                rows as usize,
                self.storage.stream.raw,
            )?;
        }
        Ok(())
    }
    /// # Safety
    /// Inputs are finite, initialized in matching row order on this device, with
    /// producer writes complete. Positions are below 1048576. No writes may race
    /// this wave; callers bind outputs to the matching request/query snapshot.
    pub unsafe fn execute(&mut self, rows: u32) -> Result<IndexQueryOutput<'_>> {
        self.validate(rows)?;
        // SAFETY: the caller retains inputs through the queued projection.
        unsafe { crate::shared::memory::chain::join(self.storage.stream.library, self.storage.stream.raw)?; }
        self.acquire_scratch()?;
        let launched = unsafe { self.enqueue(rows) };
        let completed = if self.window.is_windowed() {
            // SAFETY: publication retains this stream and its output planes.
            unsafe { crate::shared::memory::chain::finish(self.storage.stream.library, self.storage.stream.raw) }
        } else { self.synchronize() };
        launched.and(completed)?;
        self.ready = Some(rows);
        self.output()
    }
    /// # Safety
    /// Same inputs as execute. Warmup is drained but not published after capture.
    pub unsafe fn capture(&mut self, rows: u32) -> Result<()> {
        ensure!(!self.window.is_windowed(), "windowed index query cannot capture");
        self.ready = None;
        self.origin = None;
        self.tokens.clear();
        ensure!(
            self.graphs.get_shape(self.weights.layer, self.weights, rows).is_none(),
            "index query graph already captured"
        );
        unsafe {
            self.execute(rows)?;
        }
        self.ready = None;
        self.origin = None;
        self.tokens.clear();
        unsafe { self.capture_ready(rows) }
    }
    pub(crate) fn declare_startup(&mut self, keys: &[(usize, u32)]) -> Result<()> {
        self.storage.stream.require_complete()?;
        ensure!(self.pending.is_none() && self.ready.is_none(), "index query startup requires an idle owner");
        self.graphs.declare_startup(keys)
    }
    pub(crate) fn capture_startup(&mut self, rows: u32) -> Result<()> {
        self.validate(rows)?;
        self.storage.stream.require_complete()?;
        // SAFETY: capture records only owned addresses and the rebound weight
        // owner, without executing or publishing a query/proposal snapshot.
        unsafe { self.capture_ready(rows) }
    }
    pub(crate) fn finish_startup(&mut self) -> Result<()> { self.graphs.finish_startup() }
    unsafe fn capture_ready(&mut self, rows: u32) -> Result<()> {
        unsafe {
            self.storage.stream
                .library
                .cuda_graph_begin_capture(self.storage.stream.raw)?;
        }
        let launched = unsafe { self.enqueue(rows) };
        let captured = unsafe { self.storage.stream.library.cuda_graph_end_capture(self.storage.stream.raw) };
        match (launched, captured) {
            (Ok(()), Ok(graph)) => {
                if let Err(error) = unsafe {
                    self.graphs
                        .insert(self.weights.layer, self.weights, rows, graph)
                } {
                    unsafe {
                        self.storage.stream.library.cuda_graph_exec_destroy(graph)?;
                    }
                    return Err(error);
                }
                Ok(())
            }
            (Err(e), Ok(graph)) => {
                unsafe {
                    self.storage.stream.library.cuda_graph_exec_destroy(graph)?;
                }
                Err(e)
            }
            (Err(e), Err(_)) | (Ok(()), Err(e)) => Err(e),
        }
    }
    /// # Safety
    /// Same matching initialized inputs as execute. Captured live row count is fixed.
    pub unsafe fn replay(&mut self, rows: u32) -> Result<IndexQueryOutput<'_>> {
        ensure!(!self.window.is_windowed(), "windowed index query cannot replay");
        self.validate(rows)?;
        let (graph, count) = self
            .graphs
            .get_shape(self.weights.layer, self.weights, rows)
            .context("index query graph missing")?;
        ensure!(count == rows, "index query capture row count differs");
        let launched = unsafe {
            self.storage.stream
                .library
                .cuda_graph_launch(graph, self.storage.stream.raw)
        };
        launched.and(self.synchronize())?;
        self.ready = Some(rows);
        self.output()
    }
    /// # Safety
    /// No external writes race the completed query or this index wave.
    pub unsafe fn execute_attention(
        &mut self,
        query: &AttentionQueryOutput<'_>,
    ) -> Result<IndexQueryOutput<'_>> {
        if self.window.is_windowed() {
            // SAFETY: same input lifetime contract; finish publishes onto a chain
            // or drains outside one, without legacy-stream copies.
            unsafe { self.enqueue_attention(query)?;
                crate::shared::memory::chain::finish(self.storage.stream.library, self.storage.stream.raw)?; }
            self.poll_pending()?;
            return self.output();
        }
        self.ready = None;
        self.origin = None;
        self.tokens.clear();
        let binding = query.binding()?;
        let tokens = query.tokens()?;
        ensure!(
            binding.layer() == self.weights.layer
                && query.layer == self.weights.layer
                && query.rows == tokens.len()
                && query.rows <= self.capacity as usize
                && query.hidden.device_id == self.storage.hidden.buffer.device_id,
            "index query origin differs"
        );
        for (dst, src) in [
            (self.storage.qr.buffer, query.normalized_rank),
            (self.storage.hidden.buffer, query.hidden),
            (self.storage.positions.buffer, query.positions),
        ] {
            self.storage.stream.library.copy_d2d(dst, src, src.bytes)?;
        }
        let rows = query.rows as u32;
        if !super::graph_policy::captures_shape(rows)
            || (self.graphs.get_shape(self.weights.layer, self.weights, rows).is_none()
                && !self.graphs.capture_allowed(self.weights.layer, rows)) {
            unsafe { self.execute(rows)?; }
            self.origin = Some(binding);
            self.tokens.extend_from_slice(tokens);
            return self.output();
        }
        if self
            .graphs
            .get_shape(self.weights.layer, self.weights, rows)
            .is_none()
        {
            unsafe {
                self.capture(rows)?;
            }
        }
        unsafe {
            self.replay(rows)?;
        }
        self.origin = Some(binding);
        self.tokens.extend_from_slice(tokens);
        self.output()
    }
    /// # Safety
    /// Retain the main query and this wave until polling completes or abort drains.
    pub unsafe fn enqueue_attention(&mut self, query: &AttentionQueryOutput<'_>) -> Result<()> {
        self.validate(query.rows as u32)?;
        let binding = query.binding()?;
        let tokens = query.tokens()?;
        ensure!(query.rows <= self.capacity as usize && query.rows == tokens.len()
            && binding.layer() == self.weights.layer && query.layer == self.weights.layer
            && query.hidden.device_id == self.storage.hidden.buffer.device_id, "queued index origin differs");
        self.tokens.extend_from_slice(tokens);
        let rows = query.rows as u32;
        let graph = if self.window.is_windowed() { None } else { self.graphs.get_shape(self.weights.layer, self.weights, rows) };
        self.pending = Some((rows, binding, !self.window.is_windowed() && graph.is_none() && super::graph_policy::captures_shape(rows)
            && self.graphs.capture_allowed(self.weights.layer, rows)
            && !crate::shared::decode_graph::census::eager("layer_row")));
        let result = (|| -> Result<()> { unsafe {
            crate::shared::memory::chain::join(self.storage.stream.library, self.storage.stream.raw)?;
            self.acquire_scratch()?;
            for (dst, src) in [(self.storage.qr.buffer, query.normalized_rank),
                (self.storage.hidden.buffer, query.hidden), (self.storage.positions.buffer, query.positions)] {
                self.storage.stream.library.copy_d2d_async(dst, src, src.bytes, self.storage.stream.raw)?;
            }
            crate::shared::decode_graph::census::dispatch(self.storage.stream.library, self.storage.stream.raw, c"layer_row",
                rows as usize, graph.map(|(raw,_)|raw), || self.enqueue(rows))
        }})();
        if result.is_err() { self.abort_pending()?; }
        result
    }
    pub fn poll_pending(&mut self) -> Result<bool> {
        let result = (|| -> Result<bool> {
            let (rows, binding, capture) = self.pending.context("no pending index query")?;
            let chained = crate::shared::memory::chain::active();
            if !chained && !unsafe { self.storage.stream.library.cuda_stream_query(self.storage.stream.raw)? } { return Ok(false); }
            if capture {
                // The cold shape already ran eagerly (enqueue_query): record
                // the graph for later calls without launching it again
                // (hunt C #3; as the window and attention query do).
                unsafe { self.capture_ready(rows)?; }
                self.pending.as_mut().unwrap().2 = false;
            }
            if chained { unsafe { crate::shared::memory::chain::finish(self.storage.stream.library, self.storage.stream.raw)?; } }
            self.pending = None;
            self.ready = Some(rows);
            self.origin = Some(binding);
            Ok(true)
        })();
        if result.is_err() { self.abort_pending()?; }
        result
    }
    pub fn abort_pending(&mut self) -> Result<()> {
        if self.pending.is_some() { self.synchronize()?; }
        self.pending = None; self.ready = None; self.origin = None; self.tokens.clear();
        Ok(())
    }
    pub fn output(&self) -> Result<IndexQueryOutput<'_>> {
        let rows = self.ready.context("index query output unpublished")? as usize;
        let sized = |mut b: CuteafdDeviceBuffer, n: usize| {
            b.bytes = rows * n;
            b
        };
        Ok(IndexQueryOutput {
            origin: self.origin,
            tokens: &self.tokens,
            layer: self.weights.layer,
            packed: sized(self.storage.packed.buffer, 2048),
            scales: sized(self.storage.scales.buffer, 128),
            head_weights: sized(self.storage.head_weights.buffer, 64),
            _owner: PhantomData,
        })
    }
    pub fn enable_small_graph_shapes(&mut self) { self.graphs.enable_small_shapes(); }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.ready = None;
        self.origin = None;
        self.tokens.clear();
        self.synchronize()?;
        unsafe { self.graphs.remove(self.weights.layer) }
    }
}
impl Drop for IndexQueryWave<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self
            .synchronize()
            .and_then(|_| unsafe { self.graphs.clear() })
        {
            crate::shared::decode_graph::fatal_drain(Err(error), "index query");
        }
    }
}

#[cfg(test)]
mod reuse_tests {
    use super::*;
    fn download(library: &NativeLibrary, buffer: CuteafdDeviceBuffer) -> Result<Vec<u8>> {
        let mut bytes = vec![0; buffer.bytes];
        library.copy_d2h(&mut bytes, buffer)?;
        Ok(bytes)
    }
    #[test]
    fn real_index_queries_rebind_like_fresh_owners() -> Result<()> {
        let Some(path) = std::env::var_os("CUTEAFD_INDEX_REUSE_LIBRARY") else {
            eprintln!("skip index reuse GPU test: CUTEAFD_INDEX_REUSE_LIBRARY unset");
            return Ok(());
        };
        let model = std::env::var_os("CUTEAFD_INDEX_REUSE_MODEL")
            .context("CUTEAFD_INDEX_REUSE_MODEL required")?;
        let library = unsafe { NativeLibrary::load(path)? };
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&model),
        )?;
        let weights = [2, 8, 14, 20, 24, 28, 32, 36]
            .into_iter()
            .map(|layer| IndexQueryWeights::load(&library, &catalog, layer, 5_739_520, 1024 * 1024))
            .collect::<Result<Vec<_>>>()?;
        let budget = IndexQueryWave::device_bytes(&library, 4096)?;
        let mut reused = weights[0].wave(4096, budget)?;
        let pointers = [
            reused.query_rank().ptr,
            reused.hidden().ptr,
            reused.positions().ptr,
        ];
        let mut comparisons = 0;
        for rows in [1u32, 80, 4096] {
            for cycle in 0..2usize {
                for (index, weight) in weights.iter().enumerate() {
                    let mut fresh = weight.wave(4096, budget)?;
                    reused.rebind(weight)?;
                    assert!(reused.output().is_err());
                    assert_eq!(
                        pointers,
                        [
                            reused.query_rank().ptr,
                            reused.hidden().ptr,
                            reused.positions().ptr
                        ]
                    );
                    for (a, b, width) in [
                        (fresh.query_rank(), reused.query_rank(), 1280),
                        (fresh.hidden(), reused.hidden(), 5120),
                    ] {
                        let bytes = (0..rows as usize * width)
                            .flat_map(|i| {
                                let x = (((i * 17 + index * 31 + cycle * 13) % 257) as f32 - 128.0)
                                    / 256.0;
                                ((x.to_bits() >> 16) as u16).to_ne_bytes()
                            })
                            .collect::<Vec<_>>();
                        library.copy_h2d(a, &bytes)?;
                        library.copy_h2d(b, &bytes)?;
                    }
                    let positions = (0..rows as u64)
                        .flat_map(|i| (i + 8192 * cycle as u64).to_ne_bytes())
                        .collect::<Vec<_>>();
                    library.copy_h2d(fresh.positions(), &positions)?;
                    library.copy_h2d(reused.positions(), &positions)?;
                    let previous = reused.graphs.get(weight.layer, weight);
                    if previous.is_none_or(|(_, n)| n != rows) {
                        reused.clear_graph()?;
                        unsafe {
                            reused.capture(rows)?;
                        }
                    }
                    let handle = reused.graphs.get(weight.layer, weight).unwrap().0;
                    if cycle == 1 {
                        assert_eq!(Some((handle, rows)), previous);
                    }
                    let expected = unsafe { fresh.execute(rows)? };
                    let actual = unsafe { reused.replay(rows)? };
                    for (a, b) in [
                        (expected.packed, actual.packed),
                        (expected.scales, actual.scales),
                        (expected.head_weights, actual.head_weights),
                    ] {
                        assert_eq!(
                            download(&library, a)?,
                            download(&library, b)?,
                            "index layer={} rows={rows} cycle={cycle}",
                            weight.layer
                        );
                    }
                    let head = download(&library, actual.head_weights)?;
                    assert!(head
                        .chunks_exact(4)
                        .all(|b| f32::from_ne_bytes(b.try_into().unwrap()).is_finite()));
                    comparisons += 1;
                }
            }
        }
        assert!(unsafe { reused.replay(0) }.is_err());
        assert!(reused.output().is_err());
        reused.rebind(&weights[0])?;
        unsafe {
            reused.replay(4096)?;
        }
        assert_eq!(comparisons, 48);
        eprintln!("PASS 48 real index-query fresh/rebound comparisons, all eight layers, rows 1/80/4096, changed inputs, cached handles and invalid-row recovery");
        Ok(())
    }
}

#[cfg(test)]
impl IndexQueryWeights<'_> {
    pub(crate) fn visit(&self, prefix: &str, f: &mut dyn FnMut(String, cuteafd_ffi::CuteafdDeviceBuffer)) {
        self.tensors.visit(prefix, f);
        f(format!("{prefix}packed_scale"), self.scales.buffer);
    }
}
