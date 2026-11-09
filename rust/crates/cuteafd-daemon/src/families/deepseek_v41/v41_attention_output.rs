//! Backbone inverse rotary, grouped FP8 wo_a and native FP8 wo_b.
use crate::families::deepseek_v41::v41_attention_binding::QueryBinding;
use crate::shared::decode_graph::{CacheAttached, LayerGraphs};
use crate::shared::memory::Tag;
use crate::shared::memory::{DeviceAllocation, HostAllocation, LoadStream};
use crate::families::deepseek_v41::v41_sparse_attention::SparseAttentionOutput;
use crate::families::deepseek_v41::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{
    CuteafdDeviceBuffer, NativeLibrary, V41AttentionOps, V41Fp8Plan,
};
use cuteafd_loader::OfficialV41Catalog;
use std::marker::PhantomData;
const ROW_BYTES: [usize; 5] = [65536, 16384, 10240, 8, 256];
#[cfg(test)]
thread_local! { static IGNORE_WINDOW: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
pub(crate) struct AttentionOutputWeights<'a> {
    library: &'a NativeLibrary,
    layer: usize,
    names: [String; 4],
    tensors: NativeRtxTensors<'a>,
    grouped_scales: DeviceAllocation<'a>,
    scales: Option<DeviceAllocation<'a>>,
}
impl<'a> AttentionOutputWeights<'a> {
    pub(crate) fn inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>) -> Result<()> {
        self.tensors.inventory(inventory)?;
        self.grouped_scales.inventory(inventory, &super::weight_layout::packed_scale_name(&self.names[0]))?;
        if let Some(scales) = &self.scales {
            scales.inventory(inventory, &super::weight_layout::packed_scale_name(&self.names[2]))?;
        }
        Ok(())
    }
    pub(super) fn names(layer: usize) -> Result<[String; 4]> {
        ensure!(layer < 40, "invalid backbone output layer");
        Ok([
            format!("layers.{layer}.attn.wo_a.weight"),
            format!("layers.{layer}.attn.wo_a.scale"),
            format!("layers.{layer}.attn.wo_b.weight"),
            format!("layers.{layer}.attn.wo_b.scale"),
        ])
    }
    /// Official FP8 weights and packed scales for both output projections.
    pub fn device_bytes(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
    ) -> Result<usize> {
        Self::device_bytes_with_split(library,catalog,layer,false)
    }
    pub fn device_bytes_with_split(library:&NativeLibrary,catalog:&OfficialV41Catalog,
        layer:usize,split_output_b:bool)->Result<usize> {
        let names=Self::names(layer)?;
        Ok(NativeRtxTensors::plan(catalog,&names[..if split_output_b {2} else {4}])?
            + library.v41_fp8_matrix_info(1,32768,8192)?.packed_weight_scale_bytes as usize
            + if split_output_b {0} else {library.v41_fp8_matrix_info(1,8192,5120)?.packed_weight_scale_bytes as usize})
    }
    #[cfg(test)]
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
        budget: usize,
        staging: usize,
    ) -> Result<Self> {
        Self::load_with_split(library,catalog,layer,budget,staging,false)
    }
    #[cfg(test)]
    pub fn load_with_split(library:&'a NativeLibrary,catalog:&OfficialV41Catalog,layer:usize,
        budget:usize,staging:usize,split_output_b:bool)->Result<Self> {
        Self::load_in(library, catalog, layer, budget, staging, split_output_b, None)
    }
    pub fn load_in(library: &'a NativeLibrary, catalog: &OfficialV41Catalog, layer: usize,
        budget: usize, staging: usize, split_output_b: bool,
        arena: Option<&crate::shared::memory::weight_arena::DeviceWeightArena<'a>>) -> Result<Self> {
        ensure!(
            Self::device_bytes_with_split(library, catalog, layer,split_output_b)? <= budget,
            "attention output weights exceed budget"
        );
        let names = Self::names(layer)?;
        let grouped_kernel = library.v41_fp8_matrix_kernel(1, 32768, 8192)?;
        let grouped_scales = super::weight_layout::allocation(library, arena,
            &super::weight_layout::packed_scale_name(&names[0]), grouped_kernel.info().packed_weight_scale_bytes as usize)?;
        let stream = LoadStream { library, raw: library.cuda_stream_create()? };
        let kernel = library.v41_fp8_matrix_kernel(1, 8192, 5120)?;
        let scales=if split_output_b {None} else {
            Some(super::weight_layout::allocation(library, arena,
                &super::weight_layout::packed_scale_name(&names[2]), kernel.info().packed_weight_scale_bytes as usize)?)
        };
        let tensors = NativeRtxTensors::load_in(
            library, catalog, &names[..if split_output_b {2} else {4}],
            budget - grouped_scales.buffer.bytes - scales.as_ref().map_or(0,|s|s.buffer.bytes),
            staging, arena,
        )?;
        let launched = (|| unsafe {
            crate::shared::experts::rtx::dense::v41_pack_scales(&grouped_kernel, tensors.get(&names[1])?, grouped_scales.buffer, stream.raw)?;
            if let Some(scales)=&scales {
                crate::shared::experts::rtx::dense::v41_pack_scales(&kernel, tensors.get(&names[3])?,scales.buffer,stream.raw)?;
            }
            Ok(())
        })();
        launched.and(unsafe { library.cuda_stream_synchronize(stream.raw) })?;
        Ok(Self {
            library,
            layer,
            names,
            tensors,
            grouped_scales,
            scales,
        })
    }
    /// The buffers a decode step reads, in order (L2 prefetch): grouped wo_a and
    /// its packed scales, then wo_b and its scales unless TP2 splits it.
    pub(crate) fn decode_reads(&self) -> Vec<CuteafdDeviceBuffer> {
        [self.tensors.get(&self.names[0]).ok(), Some(self.grouped_scales.buffer),
            self.tensors.get(&self.names[2]).ok(), self.scales.as_ref().map(|scales| scales.buffer)]
            .into_iter().flatten().collect()
    }
    pub fn wave(&self, capacity: u32, budget: usize) -> Result<AttentionOutputWave<'_, 'a>> {
        ensure!(
            AttentionOutputWave::device_bytes_with_split(self.library, capacity,self.scales.is_none())? <= budget,
            "attention output wave exceeds budget"
        );
        self.on_storage(self.storage(capacity, capacity)?)
    }
    /// Shared storage: row planes over `rows` reserve rows, GEMM plans and
    /// scratch for units of up to `unit` rows.
    pub fn storage(&self, rows: u32, unit: u32) -> Result<std::rc::Rc<AttentionOutputStorage<'a>>> {
        ensure!(unit >= 1 && unit <= rows, "attention output unit {unit} outside {rows} rows");
        let stream = LoadStream {
            library: self.library,
            raw: self.library.cuda_stream_create()?,
        };
        let kernel=self.scales.as_ref().map(|_|self.library.v41_fp8_matrix_plan(unit,8192,5120)).transpose()?;
        let grouped = self.library.v41_fp8_matrix_plan(unit, 32768, 8192)?;
        let (grouped_bytes, down_bytes, alpha_bytes, planes) = AttentionOutputStorage::allocation_sizes(
            grouped.info().scratch_bytes as usize, kernel.as_ref().map(|k|k.info().scratch_bytes as usize).unwrap_or(0), rows);
        let storage = AttentionOutputStorage {
            stream,
            grouped,
            grouped_scratch: DeviceAllocation::new(self.library, grouped_bytes, Tag::workspace("v41/attention-output"))?,
            scratch: kernel.as_ref().map(|_|DeviceAllocation::new(self.library, down_bytes, Tag::workspace("v41/attention-output"))).transpose()?,
            kernel,
            alpha: DeviceAllocation::new(self.library, alpha_bytes, Tag::workspace("v41/attention-output"))?,
            buffers: planes
                .into_iter()
                .map(|n| DeviceAllocation::new(self.library, n, Tag::workspace("v41/attention-output")))
                .collect::<Result<Vec<_>>>()?,
            rows,
            unit,
        };
        let launched = (|| unsafe {
            storage.grouped.initialize_scratch(storage.grouped_scratch.buffer, storage.alpha.buffer, storage.stream.raw)?;
            if let (Some(kernel),Some(scratch))=(&storage.kernel,&storage.scratch) {
                kernel.initialize_scratch(scratch.buffer,storage.alpha.buffer,storage.stream.raw)?;
            }
            Ok(())
        })();
        launched.and(unsafe { self.library.cuda_stream_synchronize(storage.stream.raw) })?;
        Ok(std::rc::Rc::new(storage))
    }
    /// A slot's projection over shared `storage` (one per GPU for prefill slots).
    pub fn on_storage(&self, storage: std::rc::Rc<AttentionOutputStorage<'a>>) -> Result<AttentionOutputWave<'_, 'a>> {
        ensure!(storage.kernel.is_some() == self.scales.is_some(), "attention output storage plans differ from the weights");
        let value = AttentionOutputWave {
            norm: self.library.v41_attention_ops()?,
            position_staging: HostAllocation::new(self.library, storage.unit as usize * 8, Tag::staging("v41/attention-output"))?,
            weights: self,
            capacity: storage.unit,
            graphs: LayerGraphs::new(self.library, cuteafd_core::V41_VERIFY_ROWS as u32, "deepseek_v41"),
            ready: None,
            origin: None,
            window: crate::shared::memory::row_window::RowWindow::NONE,
            storage,
        };
        ensure!(
            value.b(0).device_id == self.grouped_scales.buffer.device_id,
            "output weight device differs"
        );
        Ok(value)
    }
}
pub(crate) struct AttentionOutput<'a> {
    origin: Option<QueryBinding>,
    pub layer: usize,
    pub rows: usize,
    pub input: CuteafdDeviceBuffer,
    pub grouped: CuteafdDeviceBuffer,
    pub projected: CuteafdDeviceBuffer,
    pub positions: CuteafdDeviceBuffer,
    pub frequencies: CuteafdDeviceBuffer,
    _owner: PhantomData<&'a ()>,
}
impl AttentionOutput<'_> {
    pub fn binding(&self) -> Result<QueryBinding> {
        let b = self
            .origin
            .context("attention projection has no query origin")?;
        ensure!(
            b.layer() == self.layer,
            "attention projection layer differs"
        );
        Ok(b)
    }
}
/// The attention output projection's stream, GEMM plans/scratch and row
/// planes (`shared::memory::row_window`). Every row buffer is a plane: sparse
/// attention writes the input, positions and frequencies on its stream, the
/// mHC post reads the projection on its own, and the grouped intermediate
/// (16 KB/row) is kept with them. Prefill slots share one per GPU.
pub(crate) struct AttentionOutputStorage<'a> {
    stream: LoadStream<'a>,
    grouped: V41Fp8Plan<'a>,
    grouped_scratch: DeviceAllocation<'a>,
    kernel: Option<V41Fp8Plan<'a>>,
    scratch: Option<DeviceAllocation<'a>>,
    alpha: DeviceAllocation<'a>,
    buffers: Vec<DeviceAllocation<'a>>,
    rows: u32,
    unit: u32,
}
impl AttentionOutputStorage<'_> {
    fn allocation_sizes(grouped: usize, down: usize, rows: u32) -> (usize, usize, usize, [usize; 5]) {
        (grouped, down, 4, ROW_BYTES.map(|n|n * rows as usize))
    }

    pub(crate) fn storage_bytes(library: &NativeLibrary, rows: u32, unit: u32, split: bool) -> Result<usize> {
        ensure!(unit > 0 && unit <= 4096 && unit <= rows
            && rows as usize <= crate::shared::memory::row_window::MAX_RESERVE_ROWS, "invalid output storage geometry");
        let (grouped, down, alpha, planes) = Self::allocation_sizes(
            library.v41_fp8_matrix_plan_info(unit, 32768, 8192)?.scratch_bytes as usize,
            if split { 0 } else { library.v41_fp8_matrix_plan_info(unit, 8192, 5120)?.scratch_bytes as usize }, rows);
        Ok(grouped + down + alpha + planes.into_iter().sum::<usize>())
    }
}
impl Drop for AttentionOutputStorage<'_> {
    fn drop(&mut self) {
        // SAFETY: the stream is owned here; queued work reads only this storage.
        if let Err(error) = unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) } {
            tracing::error!(%error, "draining attention output storage");
        }
    }
}
pub(crate) struct AttentionOutputWave<'w, 'a> {
    storage: std::rc::Rc<AttentionOutputStorage<'a>>,
    window: crate::shared::memory::row_window::RowWindow,
    norm: V41AttentionOps<'a>,
    /// Per slot: rewritten on the host for each unit (pinned staging rule).
    position_staging: HostAllocation<'a>,
    weights: &'w AttentionOutputWeights<'a>,
    capacity: u32,
    graphs: LayerGraphs<'w, 'a, AttentionOutputWeights<'a>>,
    ready: Option<u32>,
    origin: Option<QueryBinding>,
}
impl<'w, 'a> AttentionOutputWave<'w, 'a> {
    pub(crate) fn visit_graph_banks(&mut self, f: &mut dyn FnMut(i32, &'static str, bool, &mut dyn CacheAttached) -> Result<()>) -> Result<()> {
        let device = self.b(0).device_id;
        f(device, "v41.attention.output", false, &mut self.graphs)
    }

    /// Reuse this lane's storage with another layer's weights. Existing output
    /// borrows must end first. Cached graphs retain their original weight owners.
    pub fn rebind(&mut self, weights: &'w AttentionOutputWeights<'a>) -> Result<()> {
        self.ready = None;
        self.origin = None;
        ensure!(
            std::ptr::eq(self.storage.stream.library, weights.library)
                && self.storage.kernel.is_some()==weights.scales.is_some()
                && weights.grouped_scales.buffer.device_id == self.b(0).device_id,
            "attention rebound weight library or device differs"
        );
        if !crate::shared::memory::chain::active() { self.storage.stream.require_complete()?; }
        self.weights = weights;
        Ok(())
    }
}
impl AttentionOutputWave<'_, '_> {
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        Self::device_bytes_with_split(library,capacity,false)
    }
    pub fn device_bytes_with_split(library:&NativeLibrary,capacity:u32,split_output_b:bool)->Result<usize> {
        AttentionOutputStorage::storage_bytes(library, capacity, capacity, split_output_b)
    }
    /// Graphs bake in row-plane addresses (row_window: capture uses rows from
    /// 0), so a selected prefill window always executes eagerly.
    fn graphed(&self, rows: u32) -> bool {
        #[cfg(test)]
        if IGNORE_WINDOW.with(|ignore| ignore.get()) { return super::graph_policy::captures_shape(rows); }
        !self.window.is_windowed() && super::graph_policy::captures_shape(rows)
    }
    /// `execute_attention`'s graph decision over inputs already in `input()`
    /// and `positions()` (the window test drives exactly that branch).
    #[cfg(test)]
    unsafe fn run_unit(&mut self, rows: u32) -> Result<AttentionOutput<'_>> {
        if self.graphed(rows) {
            if self.graphs.get_shape(self.weights.layer, self.weights, rows).is_none() {
                unsafe { self.capture_ready_on(rows, self.storage.stream.raw)?; }
            }
            let Some((graph, _)) = self.graphs.cached_launch(self.weights.layer, self.weights, rows) else {
                return unsafe { self.execute(rows) };
            };
            unsafe { self.storage.stream.library.cuda_graph_launch(graph, self.storage.stream.raw)?; }
            self.synchronize()?;
            self.ready = Some(rows);
            self.output()
        } else {
            unsafe { self.execute(rows) }
        }
    }
    fn b(&self, i: usize) -> CuteafdDeviceBuffer {
        self.window.apply(self.storage.buffers[i].buffer, ROW_BYTES[i])
    }
    /// Selects this slot's unit rows in the shared planes, at rest
    /// (`shared::memory::row_window`).
    /// Drops a finished output (the slot is at rest: its readers drained).
    pub(crate) fn discard_ready(&mut self) { self.ready = None; }
    pub(crate) fn select_window(&mut self, window: crate::shared::memory::row_window::RowWindow) -> Result<()> {
        crate::shared::memory::row_window::require_rest(self.ready.is_some(), "attention output", "an output is unconsumed")?;
        if let Some(range) = window.range() {
            ensure!(range.end <= self.storage.rows as usize && range.len() <= self.capacity as usize,
                "attention output window {range:?} outside {} rows or wider than the {}-row unit",
                self.storage.rows, self.capacity);
        }
        self.window = window;
        Ok(())
    }
    pub fn input(&self) -> CuteafdDeviceBuffer {
        self.b(0)
    }
    pub fn positions(&self) -> CuteafdDeviceBuffer {
        self.b(3)
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.storage.stream.library.cuda_stream_synchronize(self.storage.stream.raw) }
    }
    fn validate(&mut self, rows: u32) -> Result<()> {
        self.ready = None;
        self.origin = None;
        ensure!(
            rows > 0 && rows <= self.capacity && self.window.admits(rows as usize),
            "attention output rows exceed capacity or the unit's window"
        );
        Ok(())
    }
    unsafe fn enqueue(&mut self, rows: u32) -> Result<()> {
        unsafe { self.enqueue_on(rows, self.storage.stream.raw) }
    }
    unsafe fn enqueue_grouped_on(&mut self, rows: u32, stream: *mut std::ffi::c_void) -> Result<()> {
        unsafe {
            self.norm.backbone_frequencies(
                self.b(3),
                self.b(4),
                rows,
                self.weights.layer as u32,
                stream,
            )?;
            self.storage.grouped.launch_rope(
                self.b(0),
                self.b(4),
                self.weights.tensors.get(&self.weights.names[0])?,
                self.weights.grouped_scales.buffer,
                self.storage.grouped_scratch.buffer,
                self.storage.alpha.buffer,
                self.b(1),
                rows,
                stream,
            )?;
        }
        Ok(())
    }
    unsafe fn enqueue_on(&mut self,rows:u32,stream:*mut std::ffi::c_void)->Result<()> {
        ensure!(self.storage.kernel.is_some(),"split output-B requires TP2 execution");
        unsafe {
            self.enqueue_grouped_on(rows,stream)?;
            self.storage.kernel.as_ref().context("full output-B kernel absent")?.launch(
                self.b(1),
                self.weights.tensors.get(&self.weights.names[2])?,
                self.weights.scales.as_ref().context("full output-B scales absent")?.buffer,
                self.storage.scratch.as_ref().context("full output-B scratch absent")?.buffer,
                self.storage.alpha.buffer,
                self.b(2),
                rows,
                stream,
            )?;
        }
        Ok(())
    }
    /// # Safety
    /// Inputs are finite, initialized in matching row order on this device, with
    /// producer writes complete. Positions are below 1048576. No writes may race
    /// this wave; callers bind outputs to the matching request/query snapshot.
    pub unsafe fn execute(&mut self, rows: u32) -> Result<AttentionOutput<'_>> {
        self.validate(rows)?;
        let launched = unsafe { self.enqueue(rows) };
        launched.and(self.synchronize())?;
        self.ready = Some(rows);
        self.output()
    }
    /// # Safety
    /// Same inputs as execute. Warmup is drained but not published after capture.
    pub unsafe fn capture(&mut self, rows: u32) -> Result<()> {
        ensure!(!self.window.is_windowed(), "attention output graphs never capture a row window");
        self.ready = None;
        self.origin = None;
        ensure!(
            self.graphs.get_shape(self.weights.layer, self.weights, rows).is_none(),
            "attention output graph already captured"
        );
        unsafe {
            self.execute(rows)?;
        }
        unsafe { self.capture_ready_on(rows, self.storage.stream.raw) }
    }
    unsafe fn capture_ready_on(&mut self, rows: u32, stream: *mut std::ffi::c_void) -> Result<()> {
        let Some(ticket) = self.graphs.cache_ticket(&(self.weights.layer, rows), super::graph_policy::pinned_shape(rows), None) else {
            super::graph_policy::note_miss(format!("attention_output rows={rows} layer={}", self.weights.layer), || "graph cache selected eager".into());
            return Ok(());
        };
        self.ready = None;
        self.origin = None;
        unsafe {
            self.storage.stream
                .library
                .cuda_graph_begin_capture(stream)?;
        }
        let launched = unsafe { self.enqueue_on(rows, stream) };
        let captured = unsafe { self.storage.stream.library.cuda_graph_end_capture_topology(stream) };
        match (launched, captured) {
            (Ok(()), Ok((graph, nodes, width, spans, _))) => {
                if let Err(error) = unsafe {
                    self.graphs.cache_insert(self.weights.layer, self.weights, rows, graph, ticket, nodes as u64, width, spans)
                } {
                    unsafe {
                        self.storage.stream.library.cuda_graph_exec_destroy(graph)?;
                    }
                    return Err(error);
                }
                Ok(())
            }
            (Err(e), Ok((graph, _, _, _, _))) => {
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
    pub unsafe fn replay(&mut self, rows: u32) -> Result<AttentionOutput<'_>> {
        ensure!(!self.window.is_windowed(), "attention output graphs never replay into a row window");
        self.validate(rows)?;
        let Some((graph, count)) = self.graphs.cached_launch(self.weights.layer, self.weights, rows) else {
            return unsafe { self.execute(rows) };
        };
        ensure!(count == rows, "attention output capture row count differs");
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
    /// No external writes race this wave or the completed attention output.
    pub unsafe fn execute_attention(
        &mut self,
        attention: &SparseAttentionOutput<'_>,
    ) -> Result<AttentionOutput<'_>> {
        self.ready = None;
        self.origin = None;
        let tokens = attention.tokens()?;
        ensure!(
            attention.layer == self.weights.layer
                && attention.rows == tokens.len()
                && attention.rows <= self.capacity as usize
                && attention.values.device_id == self.b(0).device_id,
            "attention output origin differs"
        );
        for (dst, token) in self.position_staging.bytes_mut().chunks_exact_mut(8).zip(tokens) {
            dst.copy_from_slice(&token.to_ne_bytes());
        }
        let executed = (|| -> Result<()> {
            unsafe {
                self.storage.stream.library.copy_d2d_async(
                    self.b(0), attention.values, attention.values.bytes, self.storage.stream.raw,
                )?;
                self.storage.stream.library.copy_host_buffer_h2d_async(
                    self.positions(), self.position_staging.buffer, tokens.len() * 8, self.storage.stream.raw,
                )?;
            }
            let rows = attention.rows as u32;
            if self.graphed(rows) {
                if self.graphs.get_shape(self.weights.layer, self.weights, rows).is_none() {
                    unsafe { self.capture(rows)?; }
                }
                unsafe { self.replay(rows)?; }
            } else {
                unsafe { self.execute(rows)?; }
            }
            Ok(())
        })();
        if let Err(error) = executed {
            // Drain staged copies even if graph preparation/replay rejects them.
            self.synchronize()?;
            self.ready = None;
            self.origin = None;
            return Err(error);
        }
        self.origin = Some(attention.binding()?);
        self.output()
    }
    /// # Safety
    /// Attention is queued on stream. All owners, including this projection's
    /// graph and pinned staging, stay alive and exclusive until that stream is
    /// drained by the enclosing sparse-attention continuation, including errors.
    /// This method never publishes AttentionOutput or marks this wave ready.
    pub unsafe fn enqueue_attention(
        &mut self, attention: &crate::families::deepseek_v41::v41_sparse_attention::QueuedSparseAttention,
        tokens: &[u64], stream: *mut std::ffi::c_void,
    ) -> Result<CuteafdDeviceBuffer> {
        self.ready = None;
        self.origin = None;
        ensure!(attention.layer == self.weights.layer && attention.rows == tokens.len()
            && attention.rows > 0 && attention.rows <= self.capacity as usize
            && attention.values.device_id == self.b(0).device_id,
            "queued attention output origin differs");
        for (dst, token) in self.position_staging.bytes_mut().chunks_exact_mut(8).zip(tokens) {
            dst.copy_from_slice(&token.to_ne_bytes());
        }
        unsafe {
            self.storage.stream.library.copy_d2d_async(self.b(0), attention.values, attention.values.bytes, stream)?;
            self.storage.stream.library.copy_host_buffer_h2d_async(self.positions(), self.position_staging.buffer,
                tokens.len() * 8, stream)?;
        }
        let rows = attention.rows as u32;
        if !self.graphed(rows) || crate::shared::decode_graph::census::eager("layer_row") {
            let library = self.storage.stream.library;
            unsafe { crate::shared::decode_graph::census::dispatch(library, stream, c"layer_row", rows as usize,
                None, || self.enqueue_on(rows, stream))?; }
            let mut output = self.b(2);
            output.bytes = attention.rows * ROW_BYTES[2];
            return Ok(output);
        }
        if self.graphs.get_shape(self.weights.layer, self.weights, rows).is_none() {
            // Cold setup uses the existing drained capture path; its input copies
            // must complete before capture's private-stream warmup can read them.
            unsafe { self.storage.stream.library.cuda_stream_synchronize(stream)?; self.capture(rows)?; }
        }
        let graph = self.graphs.cached_launch(self.weights.layer, self.weights, rows).map(|(raw, _)| raw);
        let library = self.storage.stream.library;
        unsafe { crate::shared::decode_graph::census::dispatch(library, stream, c"layer_row", rows as usize,
            graph, || self.enqueue_on(rows, stream))?; }
        let mut output = self.b(2);
        output.bytes = attention.rows * ROW_BYTES[2];
        Ok(output)
    }
    /// # Safety
    /// Retain attention and this wave through the containing stream's completion.
    /// None means warmup is queued and finish_prepared must follow a cooperative wait.
    pub unsafe fn enqueue_attention_prepared(&mut self,
        attention: &crate::families::deepseek_v41::v41_sparse_attention::QueuedSparseAttention, tokens: &[u64],
        stream: *mut std::ffi::c_void) -> Result<Option<CuteafdDeviceBuffer>> {
        self.ready = None;
        self.origin = None;
        ensure!(attention.layer == self.weights.layer && attention.rows == tokens.len()
            && attention.rows > 0 && attention.rows <= self.capacity as usize
            && attention.values.device_id == self.b(0).device_id,
            "queued attention output origin differs");
        for (dst, token) in self.position_staging.bytes_mut().chunks_exact_mut(8).zip(tokens) {
            dst.copy_from_slice(&token.to_ne_bytes());
        }
        unsafe {
            self.storage.stream.library.copy_d2d_async(self.b(0), attention.values, attention.values.bytes, stream)?;
            self.storage.stream.library.copy_host_buffer_h2d_async(self.positions(), self.position_staging.buffer,
                tokens.len() * 8, stream)?;
        }
        let rows = attention.rows as u32;
        if self.window.is_windowed() {
            // A windowed (shared-storage) unit: no captured addresses, eager.
            unsafe { self.enqueue_on(rows, stream)?; }
            let mut output = self.b(2); output.bytes = attention.rows * ROW_BYTES[2];
            return Ok(Some(output));
        }
        let graph = self.graphs.cached_launch(self.weights.layer, self.weights, rows).map(|(raw, _)| raw);
        let library = self.storage.stream.library;
        unsafe { crate::shared::decode_graph::census::dispatch(library, stream, c"layer_row", rows as usize,
            graph, || self.enqueue_on(rows, stream))?; }
        if graph.is_none() && super::graph_policy::captures_shape(rows)
            && !crate::shared::decode_graph::census::eager("layer_row") {
            return Ok(None);
        }
        let mut output = self.b(2); output.bytes = attention.rows * ROW_BYTES[2];
        Ok(Some(output))
    }
    /// # Safety
    /// Queued warmup on stream has completed; retain storage through capture.
    pub unsafe fn finish_prepared(&mut self, rows: u32,
        stream: *mut std::ffi::c_void) -> Result<CuteafdDeviceBuffer> {
        // The warmup already produced these rows. Capture only records a future
        // invocation; a declined ticket must not make this completed output fail.
        unsafe { self.capture_ready_on(rows, stream)?; }
        let mut output = self.b(2); output.bytes = rows as usize * ROW_BYTES[2];
        Ok(output)
    }

    /// Prepare only mutable metadata outside a containing graph. The caller
    /// drains stream before releasing or reusing this wave or pinned staging.
    pub unsafe fn prepare_chain_graph(&mut self, tokens: &[u64], stream: *mut std::ffi::c_void) -> Result<()> {
        self.validate(tokens.len() as u32)?;
        ensure!(tokens.iter().all(|&p| p < 1048576), "projection graph positions exceed context");
        for (dst, token) in self.position_staging.bytes_mut().chunks_exact_mut(8).zip(tokens) {
            dst.copy_from_slice(&token.to_ne_bytes());
        }
        unsafe { self.storage.stream.library.copy_host_buffer_h2d_async(self.positions(), self.position_staging.buffer,
            tokens.len() * 8, stream) }
    }
    /// # Safety
    /// Matching sparse output is ordered on stream. Capture retains all weights
    /// and storage through graph destruction, and owners stay exclusive in flight.
    pub unsafe fn enqueue_chain_graph(&mut self, attention: &crate::families::deepseek_v41::v41_sparse_attention::QueuedSparseAttention,
        stream: *mut std::ffi::c_void) -> Result<CuteafdDeviceBuffer> {
        ensure!(attention.layer == self.weights.layer && attention.rows > 0
            && attention.rows <= self.capacity as usize
            && attention.values.device_id == self.b(0).device_id, "projection graph origin differs");
        unsafe {
            self.storage.stream.library.copy_d2d_async(self.b(0), attention.values, attention.values.bytes, stream)?;
            self.enqueue_on(attention.rows as u32, stream)?;
        }
        let mut value = self.b(2); value.bytes = attention.rows * 10240; Ok(value)
    }
    /// # Safety
    /// Same capture/producer contract as enqueue_chain_graph. This queues only
    /// inverse rotary and grouped output-A; output-B must follow before FFN.
    pub unsafe fn enqueue_grouped_chain_graph(&mut self,
        attention:&crate::families::deepseek_v41::v41_sparse_attention::QueuedSparseAttention,
        stream:*mut std::ffi::c_void)->Result<CuteafdDeviceBuffer> {
        ensure!(self.storage.kernel.is_none() && attention.layer==self.weights.layer && attention.rows>0
            && attention.rows<=self.capacity as usize && attention.values.device_id==self.b(0).device_id,
            "split output prefix differs");
        unsafe {
            self.storage.stream.library.copy_d2d_async(self.b(0),attention.values,attention.values.bytes,stream)?;
            self.enqueue_grouped_on(attention.rows as u32,stream)?;
        }
        Ok(CuteafdDeviceBuffer {bytes:attention.rows*ROW_BYTES[1],..self.b(1)})
    }
    /// # Safety
    /// Grouped output-A has been queued on producer (or is complete when absent).
    /// Retain prefix producer, this wave and consumer storage until completion
    /// or drained cancellation. Consumer must enqueue only on the supplied stream.
    pub async unsafe fn finish_tp2_then<T>(&mut self,rows:u32,
        projection:&mut crate::families::deepseek_v41::v41_projection_tp2::Wave<'_, '_>,producer:Option<*mut std::ffi::c_void>,
        consume:impl FnOnce(CuteafdDeviceBuffer,*mut std::ffi::c_void)->Result<T>)->Result<T> {
        self.validate(rows)?;
        ensure!(self.storage.kernel.is_none() && projection.kind()==crate::families::deepseek_v41::v41_projection_tp2::Kind::OutputB
            && projection.output_device().id==self.b(0).device_id
            && std::ptr::eq(projection.output_device().library,self.storage.stream.library),"output projection owner differs");
        unsafe { projection.execute_after(self.weights.layer,rows,
            CuteafdDeviceBuffer {bytes:rows as usize*ROW_BYTES[1],..self.b(1)},producer,|output,stream| {
                self.storage.stream.library.copy_d2d_async(self.b(2),output,output.bytes,stream)?;
                consume(CuteafdDeviceBuffer {bytes:output.bytes,..self.b(2)},stream)
            }).await }
    }
    pub fn chain_graph_identity(&self) -> [usize; 2] {
        [self.b(0).ptr as usize, self.weights as *const _ as usize]
    }
    pub fn output(&self) -> Result<AttentionOutput<'_>> {
        let rows = self.ready.context("attention output unpublished")? as usize;
        let b = |i| {
            let mut b = self.b(i);
            b.bytes = rows * ROW_BYTES[i];
            b
        };
        Ok(AttentionOutput {
            origin: self.origin,
            layer: self.weights.layer,
            rows,
            input: b(0),
            grouped: b(1),
            projected: b(2),
            positions: b(3),
            frequencies: b(4),
            _owner: PhantomData,
        })
    }
    /// Evict all shapes for the current layer; other layers remain cached.
    pub fn clear_graph(&mut self) -> Result<()> {
        self.ready = None;
        self.origin = None;
        self.synchronize()?;
        unsafe {
            self.graphs.remove(self.weights.layer)?;
        }
        Ok(())
    }
}
impl Drop for AttentionOutputWave<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self
            .synchronize()
            .and_then(|()| unsafe { self.graphs.clear() })
        {
            crate::shared::decode_graph::fatal_drain(Err(error), "attention output");
        }
    }
}

#[cfg(test)]
mod window_graph_tests {
    use super::*;
    /// Two equal-row units at two window offsets on one shared-storage slot
    /// each equal a private eager run of the same input. With the window gate
    /// removed (negative control) the second unit replays the first unit's
    /// captured graph, which reads and writes the first window's rows.
    fn run(gated: bool) -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        let layer = 2;
        let weights = AttentionOutputWeights::load(&library, &catalog, layer,
            AttentionOutputWeights::device_bytes(&library, &catalog, layer)?, 1 << 20)?;
        let rows = 16u32;
        let input = |seed: u8| -> Vec<u8> { (0..rows as usize * 32768).flat_map(|i| {
            let v = ((i * 37 + seed as usize * 101) % 61) as u16; (0x3c00u16 ^ (v << 3)).to_ne_bytes() }).collect() };
        let positions: Vec<u8> = (0..rows as u64).flat_map(|p| (p * 3 + 100).to_ne_bytes()).collect();
        let mut private = weights.wave(rows, usize::MAX)?;
        let mut expect = Vec::new();
        for seed in [1u8, 2] {
            library.copy_h2d(private.input(), &input(seed))?;
            library.copy_h2d(private.positions(), &positions)?;
            let out = unsafe { private.execute(rows)? };
            let mut bytes = vec![0; out.projected.bytes]; library.copy_d2h(&mut bytes, out.projected)?;
            expect.push(bytes);
        }
        assert_ne!(expect[0], expect[1], "distinct inputs must project differently");
        let storage = weights.storage(4 * rows, rows)?;
        let mut slot = weights.on_storage(storage)?;
        slot.visit_graph_banks(&mut |device, name, spans, bank| bank.attach(device, 8 << 30,
            crate::shared::decode_graph::device::Site { name,
                cycle: crate::shared::decode_graph::device::Cycle::Other,
                preference: crate::shared::decode_graph::device::Preference::Capture,
                bootstrap: 2 << 20, basis: crate::shared::decode_graph::device::Basis::Bytes,
                spans_devices: spans }))?;
        IGNORE_WINDOW.with(|ignore| ignore.set(!gated));
        let result = (|| -> Result<Vec<Vec<u8>>> {
            let mut got = Vec::new();
            for (seed, offset) in [(1u8, 2 * rows as usize), (2, rows as usize)] {
                // As the lane does before a window change: the previous output is consumed.
                slot.discard_ready();
                slot.select_window(crate::shared::memory::row_window::RowWindow::within(offset..offset + rows as usize,
                    4 * rows as usize)?)?;
                library.copy_h2d(slot.input(), &input(seed))?;
                library.copy_h2d(slot.positions(), &positions)?;
                let out = unsafe { slot.run_unit(rows)? };
                let mut bytes = vec![0; out.projected.bytes]; library.copy_d2h(&mut bytes, out.projected)?;
                got.push(bytes);
            }
            Ok(got)
        })();
        IGNORE_WINDOW.with(|ignore| ignore.set(false));
        let got = result?;
        eprintln!("attention output window graphs gated={gated}: unit A exact={} unit B exact={}",
            got[0] == expect[0], got[1] == expect[1]);
        assert_eq!(got[0], expect[0], "first windowed unit differs from its private run");
        if gated { assert_eq!(got[1], expect[1], "second windowed unit differs from its private run"); }
        else { assert_ne!(got[1], expect[1], "ungated replay at a moved window did not go stale"); }
        Ok(())
    }
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_SNAPSHOT and a CUDA GPU"]
    fn windowed_units_never_replay_a_graph_captured_at_another_offset() -> Result<()> { for _ in 0..3 { run(true)?; } Ok(()) }
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_SNAPSHOT and a CUDA GPU"]
    fn without_the_window_gate_a_moved_window_replays_stale_rows() -> Result<()> { for _ in 0..3 { run(false)?; } Ok(()) }
}

#[cfg(test)]
mod tp2_tests {
    use super::*;
    use crate::shared::memory::device::{Allocation,Device};
    use crate::families::deepseek_v41::v41_projection_tp2::{Kind,Weights,Wave};
    #[test]
    #[ignore = "requires checkpoint, two GPUs and projection shard AOT"]
    fn checkpoint_tp2_output_preserves_grouped_rotary() -> Result<()> {
        let library=unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let catalog=cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        let devices=[Device {library:&library,id:0},Device {library:&library,id:1}];
        let runtime=tokio::runtime::Builder::new_current_thread().build()?;
        for (owner,layer) in [(0,2),(1,20)] {
            let device=devices[owner];let capacity=16;
            let weights=device.own(||AttentionOutputWeights::load(&library,&catalog,layer,
                AttentionOutputWeights::device_bytes(&library,&catalog,layer)?,1<<20))?;
            let split=device.own(||AttentionOutputWeights::load_with_split(&library,&catalog,layer,
                AttentionOutputWeights::device_bytes_with_split(&library,&catalog,layer,true)?,1<<20,true))?;
            assert!(split.scales.is_none() && split.tensors.get(&split.names[2]).is_err());
            let mut reference=device.own(||weights.wave(capacity,usize::MAX))?;
            let mut output=device.own(||split.wave(capacity,usize::MAX))?;
            assert!(output.storage.scratch.is_none() && output.rebind(&weights).is_err());
            let halves=[vec![Weights::load(devices[0],&catalog,layer,Kind::OutputB,0,
                Weights::load_peak_device_bytes(Kind::OutputB))?],
                vec![Weights::load(devices[1],&catalog,layer,Kind::OutputB,1,
                Weights::load_peak_device_bytes(Kind::OutputB))?]];
            let mut projection=Wave::new([&halves[0],&halves[1]],capacity,owner,
                Wave::device_bytes(&library,Kind::OutputB,capacity,owner)?)?;
            let input=Allocation::new(device,capacity as usize*ROW_BYTES[0], crate::shared::memory::TEST)?;
            let read=|buffer:CuteafdDeviceBuffer|->Result<Vec<u8>> {
                let mut bytes=vec![0;buffer.bytes];device.run(||library.copy_d2h(&mut bytes,buffer))?;Ok(bytes)
            };
            for (cycle,rows) in [1,6,16,6,1].into_iter().enumerate() {
                let host:Vec<u8>=(0..capacity as usize*32768).flat_map(|i| {
                    let value=((i+cycle*5)%29) as f32/64.0-0.25;
                    ((value.to_bits()>>16) as u16).to_ne_bytes()
                }).collect();
                device.run(||library.copy_h2d(input.buffer,&host))?;
                let tokens:Vec<u64>=(0..rows).map(|i|(cycle*8192+i) as u64).collect();
                let attention=crate::families::deepseek_v41::v41_sparse_attention::QueuedSparseAttention {
                    values:CuteafdDeviceBuffer {bytes:rows*ROW_BYTES[0],..input.buffer},rows,layer};
                let stream=reference.storage.stream.raw;
                let expected=device.run(||unsafe {
                    reference.prepare_chain_graph(&tokens,stream)?;
                    let result=reference.enqueue_chain_graph(&attention,stream)?;
                    library.cuda_stream_synchronize(stream)?;read(result)
                })?;
                let stream=output.storage.stream.raw;
                device.run(||unsafe {
                    output.prepare_chain_graph(&tokens,stream)?;
                    output.enqueue_grouped_chain_graph(&attention,stream)?;Ok(())
                })?;
                runtime.block_on(async {
                    let mut pending=std::pin::pin!(device.future(unsafe {
                        output.finish_tp2_then(rows as u32,&mut projection,Some(stream),|value,_|Ok(value))
                    }));
                    std::future::poll_fn(|cx| {
                        use std::future::Future;
                        let _=pending.as_mut().poll(cx);std::task::Poll::Ready(())
                    }).await;
                });
                let actual=runtime.block_on(device.future(unsafe {
                    output.finish_tp2_then(rows as u32,&mut projection,Some(stream),|value,_|Ok(value))
                }))?;
                let actual=read(actual)?;let mut max_error=0f32;let mut max_ulp=0u16;let mut changed=0usize;
                for index in [0,1,3,4] {
                    let slice=|buffer:CuteafdDeviceBuffer|CuteafdDeviceBuffer {bytes:rows*ROW_BYTES[index],..buffer};
                    assert_eq!(read(slice(reference.b(index)))?,read(slice(output.b(index)))?);
                }
                for (a,b) in actual.chunks_exact(2).zip(expected.chunks_exact(2)) {
                    let decode=|v:&[u8]|f32::from_bits(u32::from(u16::from_ne_bytes([v[0],v[1]]))<<16);
                    let (a,b)=(decode(a),decode(b));
                    ensure!(a.is_finite() && b.is_finite(),"nonfinite output projection");
                    let ulp=((a.to_bits()>>16) as u16).abs_diff((b.to_bits()>>16) as u16);
                    // The full M1 export uses split-K=2, while each half uses
                    // split-K=1. Different FP32 accumulation order can straddle
                    // a BF16 rounding boundary; allow one adjacent BF16 value.
                    ensure!((a-b).abs()<=1e-4 || ulp<=1,
                        "TP2 output differs beyond rounding: layer={layer} rows={rows} {a} vs {b}");
                    max_ulp=max_ulp.max(ulp);changed+=usize::from(a!=b);
                    max_error=max_error.max((a-b).abs());
                }
                eprintln!("TP2 complete output layer={layer} rows={rows} cycle={cycle} max_error={max_error} max_ulp={max_ulp} changed={changed}");

            }
        }
        Ok(())
    }
}

#[cfg(test)]
impl AttentionOutputWeights<'_> {
    pub(crate) fn visit(&self, prefix: &str, f: &mut dyn FnMut(String, cuteafd_ffi::CuteafdDeviceBuffer)) {
        self.tensors.visit(prefix, f);
        f(format!("{prefix}grouped_scale"), self.grouped_scales.buffer);
        if let Some(s) = &self.scales { f(format!("{prefix}packed_scale"), s.buffer); }
    }
}

#[cfg(test)]
mod storage_bytes_tests {
    use super::*;
    #[test]
    fn output_storage_bytes_equal_allocation_requests() {
        for rows in [2048, 2752, 4096, 8192] {
            for down in [0, 456] {
                let (grouped, down, alpha, planes) = AttentionOutputStorage::allocation_sizes(123, down, rows);
                assert_eq!(planes, [65536, 16384, 10240, 8, 256].map(|n|n * rows as usize));
                assert_eq!(grouped + down + alpha + planes.into_iter().sum::<usize>(),
                    123 + down + 4 + rows as usize * 92424);
            }
        }
    }
}

#[cfg(test)]
mod bounded_graph_oracle {
    use super::*;
    use crate::shared::decode_graph::device::{Basis, Cycle, Preference, Site};

    #[test]
    #[ignore = "requires V4.1 checkpoint, AOT native library and CUDA GPU"]
    fn prepared_eager_output_is_byte_exact_to_cached_replay() -> Result<()> {
        // SAFETY: library and checkpoint owners outlive all waves and launches.
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(
            &std::env::var("CUTEAFD_SNAPSHOT")?))?;
        let weights = AttentionOutputWeights::load(&library, &catalog, 2,
            AttentionOutputWeights::device_bytes(&library, &catalog, 2)?, 1 << 20)?;
        let mut wave = weights.wave(16, usize::MAX)?;
        wave.visit_graph_banks(&mut |device, name, spans, bank| bank.attach(device, 256 << 20,
            Site { name, cycle: Cycle::Wave, preference: Preference::Capture, bootstrap: 2 << 20,
                basis: Basis::Bytes, spans_devices: spans }))?;
        for rows in [1, 16] {
            let input: Vec<_> = (0..rows as usize * (ROW_BYTES[0] / 2)).flat_map(|i| {
                let value = ((i * 37 % 61) as f32 - 30.) / 32.;
                ((value.to_bits() >> 16) as u16).to_ne_bytes()
            }).collect();
            let positions: Vec<_> = (0..rows as u64).flat_map(|p| (p * 3 + 100).to_ne_bytes()).collect();
            library.copy_h2d(wave.input(), &input)?;
            library.copy_h2d(wave.positions(), &positions)?;
            // SAFETY: finite matching inputs are staged above and the wave
            // retains the exact same output buffer through eager/capture/replay.
            let expected = unsafe {
                wave.enqueue_on(rows, wave.storage.stream.raw)?;
                wave.synchronize()?;
                let output = wave.finish_prepared(rows, wave.storage.stream.raw)?;
                let mut bytes = vec![0; output.bytes];
                library.copy_d2h(&mut bytes, output)?;
                bytes
            };
            let (graph, _) = wave.graphs.cached_launch(weights.layer, &weights, rows)
                .context("oracle requires a successful capture, not eager fallback")?;
            // SAFETY: executable, weights, workspace and unchanged inputs stay
            // alive until the owned stream completes below.
            unsafe { library.cuda_graph_launch(graph, wave.storage.stream.raw)?; }
            wave.synchronize()?;
            let mut actual = vec![0; expected.len()];
            library.copy_d2h(&mut actual, wave.b(2))?;
            assert_eq!(actual, expected, "prepared eager/replay differs at rows={rows}");
            wave.graphs.cache_measured(0);
        }
        Ok(())
    }
}
