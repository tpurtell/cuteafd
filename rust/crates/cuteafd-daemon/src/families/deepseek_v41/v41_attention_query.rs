//! Real backbone low-rank query projection, normalization and rotary graphs.
use crate::families::deepseek_v41::v41_attention_binding::QueryBinding;
use crate::shared::decode_graph::{CacheAttached, LayerGraphs};
use crate::shared::memory::Tag;
use crate::shared::memory::{DeviceAllocation, HostAllocation, LoadStream};
use crate::families::deepseek_v41::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary, V41AttentionOps, V41Fp8Plan};
use cuteafd_loader::OfficialV41Catalog;
use std::marker::PhantomData;
#[cfg(test)]
thread_local! {
    /// Test-only: the negative control skips the scratch guard's acquire.
    static SKIP_ACQUIRE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
const MATRICES: [(u32, u32); 2] = [(5120, 1280), (1280, 32768)];
const ROW_BYTES: [usize; 7] = [10240, 2560, 2560, 65536, 65536, 8, 256];
fn part(mut b: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> CuteafdDeviceBuffer {
    debug_assert!(offset + bytes <= b.bytes);
    b.ptr = unsafe { b.ptr.cast::<u8>().add(offset).cast() };
    b.bytes = bytes;
    b
}
pub(crate) struct AttentionQueryWeights<'a> {
    library: &'a NativeLibrary,
    layer: usize,
    names: [String; 5],
    tensors: NativeRtxTensors<'a>,
    scales: Vec<DeviceAllocation<'a>>,
    split_query_b: bool,
}
impl<'a> AttentionQueryWeights<'a> {
    pub(crate) fn inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>) -> Result<()> {
        self.tensors.inventory(inventory)?;
        for (index, scales) in self.scales.iter().enumerate() {
            scales.inventory(inventory, &cuteafd_loader::placement::families::deepseek_v41::weight_layout::packed_scale_name(&self.names[index * 2]))?;
        }
        Ok(())
    }
    pub(super) fn names(layer: usize) -> Result<[String; 5]> {
        ensure!(layer < 40, "invalid backbone query layer");
        Ok([
            format!("layers.{layer}.attn.wq_a.weight"),
            format!("layers.{layer}.attn.wq_a.scale"),
            format!("layers.{layer}.attn.wq_b.weight"),
            format!("layers.{layer}.attn.wq_b.scale"),
            format!("layers.{layer}.attn.q_norm.weight"),
        ])
    }
    pub fn device_bytes(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
    ) -> Result<usize> {
        Self::device_bytes_with_split(library,catalog,layer,false)
    }
    pub fn device_bytes_with_split(library:&NativeLibrary,catalog:&OfficialV41Catalog,
        layer:usize,split_query_b:bool)->Result<usize> {
        let names=Self::names(layer)?;
        let selected:Vec<_>=names.iter().enumerate().filter(|(i,_)|!split_query_b || !matches!(i,2|3))
            .map(|(_,name)|name.clone()).collect();
        let mut bytes=NativeRtxTensors::plan(catalog,&selected)?;
        for (k,n) in MATRICES.into_iter().take(if split_query_b {1} else {2}) {
            bytes+=library.v41_fp8_matrix_info(1,k,n)?.packed_weight_scale_bytes as usize;
        }
        Ok(bytes)
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
        budget:usize,staging:usize,split_query_b:bool)->Result<Self> {
        Self::load_in(library, catalog, layer, budget, staging, split_query_b, None)
    }
    pub fn load_in(library: &'a NativeLibrary, catalog: &OfficialV41Catalog, layer: usize,
        budget: usize, staging: usize, split_query_b: bool,
        arena: Option<&crate::shared::memory::weight_arena::DeviceWeightArena<'a>>) -> Result<Self> {
        ensure!(
            Self::device_bytes_with_split(library, catalog, layer,split_query_b)? <= budget,
            "attention query weights exceed budget"
        );
        let names = Self::names(layer)?;
        let selected:Vec<_>=names.iter().enumerate().filter(|(i,_)|!split_query_b || !matches!(i,2|3))
            .map(|(_,name)|name.clone()).collect();
        let tensors = NativeRtxTensors::load_in(library, catalog, &selected, budget, staging, arena)?;
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        let mut scales = vec![];
        for (i, (k, n)) in MATRICES.into_iter().take(if split_query_b {1} else {2}).enumerate() {
            let kernel = library.v41_fp8_matrix_kernel(1, k, n)?;
            let scale = super::v41_tensors::allocation(library, arena,
                &cuteafd_loader::placement::families::deepseek_v41::weight_layout::packed_scale_name(&names[i * 2]),
                kernel.info().packed_weight_scale_bytes as usize)?;
            let launched = unsafe {
                crate::shared::experts::rtx::dense::v41_pack_scales(&kernel, tensors.get(&names[i * 2 + 1])?, scale.buffer, stream.raw)
            };
            launched.and(unsafe { library.cuda_stream_synchronize(stream.raw) })?;
            scales.push(scale);
        }
        Ok(Self {
            library,
            layer,
            names,
            tensors,
            scales,
            split_query_b,
        })
    }
    /// The buffers a decode step reads, in order (L2 prefetch): wq_a and its
    /// packed scales, q_norm, then wq_b and its scales unless TP2 splits it.
    pub(crate) fn decode_reads(&self) -> Vec<CuteafdDeviceBuffer> {
        let mut scales = self.scales.iter().map(|scale| scale.buffer);
        [self.tensors.get(&self.names[0]).ok(), scales.next(), self.tensors.get(&self.names[4]).ok(),
            self.tensors.get(&self.names[2]).ok(), scales.next()].into_iter().flatten().collect()
    }
    pub fn wave(&self, capacity: u32, budget: usize) -> Result<AttentionQueryWave<'_, 'a>> {
        ensure!(
            AttentionQueryWave::device_bytes_with_split(self.library, capacity,self.split_query_b)? <= budget,
            "attention query wave exceeds budget"
        );
        self.on_storage(self.storage(capacity, &[self.library.cuda_get_device()?])?)
    }
    /// Shared storage for units of up to `unit` rows; `readers` are the
    /// devices whose streams read the outputs (fork readers; both GPUs on two
    /// RTX), where a joined chain head may sit.
    pub fn storage(&self, unit: u32, readers: &[i32]) -> Result<std::rc::Rc<AttentionQueryStorage<'a>>> {
        ensure!((1..=4096).contains(&unit), "invalid attention query unit {unit}");
        let stream = LoadStream {
            library: self.library,
            raw: self.library.cuda_stream_create()?,
        };
        let kernels = MATRICES
            .into_iter().take(if self.split_query_b {1} else {2})
            .map(|(k, n)| self.library.v41_fp8_matrix_plan(unit, k, n))
            .collect::<Result<Vec<_>>>()?;
        let scratch = kernels
            .iter()
            .map(|k| DeviceAllocation::new(self.library, k.info().scratch_bytes as usize, Tag::workspace("v41/attention-query")))
            .collect::<Result<Vec<_>>>()?;
        let storage = AttentionQueryStorage {
            stream,
            kernels,
            scratch,
            alpha: DeviceAllocation::new(self.library, 4, Tag::workspace("v41/attention-query"))?,
            buffers: ROW_BYTES
                .into_iter().enumerate()
                .map(|(i, n)| {
                    if i == 4 && !cfg!(test) { Ok(None) }
                    else { DeviceAllocation::new(self.library, n * unit as usize, Tag::workspace("v41/attention-query")).map(Some) }
                })
                .collect::<Result<Vec<_>>>()?,
            guard: crate::shared::memory::row_window::ScratchGuard::new(self.library, readers)?,
            unit,
        };
        for i in 0..storage.kernels.len() {
            let launched = unsafe {
                storage.kernels[i].initialize_scratch(
                    storage.scratch[i].buffer,
                    storage.alpha.buffer,
                    storage.stream.raw,
                )
            };
            launched.and(unsafe { self.library.cuda_stream_synchronize(storage.stream.raw) })?;
        }
        Ok(std::rc::Rc::new(storage))
    }
    /// A slot's query over shared `storage` (one per GPU for prefill slots).
    pub fn on_storage(&self, storage: std::rc::Rc<AttentionQueryStorage<'a>>) -> Result<AttentionQueryWave<'_, 'a>> {
        ensure!(storage.kernels.len() == self.scales.len(), "attention query storage plans differ from the weights");
        let value = AttentionQueryWave {
            position_staging: HostAllocation::new(self.library, storage.unit as usize * 8, Tag::staging("v41/attention-query"))?,
            norm: self.library.v41_attention_ops()?,
            weights: self,
            capacity: storage.unit,
            graphs: LayerGraphs::new(self.library, cuteafd_core::V41_VERIFY_ROWS as u32, "deepseek_v41"),
            window: crate::shared::memory::row_window::RowWindow::NONE,
            tp2_prefix_graphs: LayerGraphs::new(self.library, cuteafd_core::V41_VERIFY_ROWS as u32, "deepseek_v41"),
            ready: None,
            binding: None,
            tokens: Vec::new(),
            storage,
        };
        ensure!(
            value.b(0).device_id == self.tensors.get(&self.names[0])?.device_id,
            "query weight device differs"
        );
        Ok(value)
    }
}
pub(crate) struct AttentionQueryOutput<'a> {
    binding: Option<QueryBinding>,
    tokens: &'a [u64],
    pub layer: usize,
    pub rows: usize,
    pub hidden: CuteafdDeviceBuffer,
    pub raw_rank: CuteafdDeviceBuffer,
    pub normalized_rank: CuteafdDeviceBuffer,
    #[cfg(test)]
    pub projected: CuteafdDeviceBuffer,
    #[cfg(test)]
    pub qb_scratch: Option<CuteafdDeviceBuffer>,
    pub rotated: CuteafdDeviceBuffer,
    pub positions: CuteafdDeviceBuffer,
    pub frequencies: CuteafdDeviceBuffer,
    _owner: PhantomData<&'a ()>,
}
impl AttentionQueryOutput<'_> {
    pub fn binding(&self) -> Result<QueryBinding> {
        self.binding.context("query has no token binding")
    }
    pub fn tokens(&self) -> Result<&[u64]> {
        self.binding()?;
        Ok(self.tokens)
    }
}
/// The attention query's stream, GEMM scratch and outputs
/// (`shared::memory::row_window` contract). Every output buffer (hidden,
/// ranks, rotated, positions, frequencies) is consumed inside the attention
/// hook: by this stream and by its fork readers (window, compressor, index,
/// sparse attention), all joined into the unit's chain before the hook ends.
/// So they are guarded scratch at the unit cap, shared by every slot:
/// [`AttentionQueryWave::release_scratch`] after the hook's join, and an
/// `acquire` before every write. The GEMM plans' scratch and alpha are plain
/// scratch, ordered by this stream. Prefill slots share one per GPU.
pub(crate) struct AttentionQueryStorage<'a> {
    stream: LoadStream<'a>,
    kernels: Vec<V41Fp8Plan<'a>>,
    scratch: Vec<DeviceAllocation<'a>>,
    alpha: DeviceAllocation<'a>,
    buffers: Vec<Option<DeviceAllocation<'a>>>,
    /// Orders every write of the outputs after the previous unit's joined
    /// readers (`ScratchGuard`).
    guard: crate::shared::memory::row_window::ScratchGuard<'a>,
    unit: u32,
}
impl Drop for AttentionQueryStorage<'_> {
    fn drop(&mut self) {
        // SAFETY: the stream is owned here; queued work reads only this storage.
        if let Err(error) = unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) } {
            tracing::error!(%error, "draining attention query storage");
        }
    }
}
pub(crate) struct AttentionQueryWave<'w, 'a> {
    storage: std::rc::Rc<AttentionQueryStorage<'a>>,
    /// Per slot: rewritten on the host for each unit (pinned staging rule).
    position_staging: HostAllocation<'a>,
    norm: V41AttentionOps<'a>,
    weights: &'w AttentionQueryWeights<'a>,
    capacity: u32,
    window: crate::shared::memory::row_window::RowWindow,
    graphs: LayerGraphs<'w, 'a, AttentionQueryWeights<'a>>,
    tp2_prefix_graphs: LayerGraphs<'w, 'a, AttentionQueryWeights<'a>>,
    ready: Option<u32>,
    binding: Option<QueryBinding>,
    tokens: Vec<u64>,
}
impl<'w, 'a> AttentionQueryWave<'w, 'a> {
    pub(crate) fn visit_graph_banks(&mut self, f: &mut dyn FnMut(i32, &'static str, bool, &mut dyn CacheAttached) -> Result<()>) -> Result<()> {
        let device = self.input().device_id;
        f(device, "v41.attention.query", false, &mut self.graphs)?;
        f(device, "v41.attention.query.tp2_prefix", true, &mut self.tp2_prefix_graphs)
    }

    /// Reuse this lane's storage with another layer's weights. Existing output
    /// borrows must end first. Cached graphs retain their original weight owners.
    pub fn rebind(&mut self, weights: &'w AttentionQueryWeights<'a>) -> Result<()> {
        self.ready = None;
        self.binding = None;
        self.tokens.clear();
        ensure!(
            std::ptr::eq(self.storage.stream.library, weights.library)
                && self.storage.kernels.len()==weights.scales.len()
                && weights.tensors.get(&weights.names[0])?.device_id == self.b(0).device_id,
            "attention rebound weight library or device differs"
        );
        if !crate::shared::memory::chain::active() { self.storage.stream.require_complete()?; }
        self.weights = weights;
        Ok(())
    }
}
impl AttentionQueryWave<'_, '_> {
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        Self::device_bytes_with_split(library,capacity,false)
    }
    pub fn device_bytes_with_split(library:&NativeLibrary,capacity:u32,split_query_b:bool)->Result<usize> {
        let mut bytes = 4 + capacity as usize * ROW_BYTES.iter().enumerate()
            .filter(|(i, _)| *i != 4 || cfg!(test)).map(|(_, bytes)| bytes).sum::<usize>();
        for (k, n) in MATRICES.into_iter().take(if split_query_b {1} else {2}) {
            bytes += library.v41_fp8_matrix_plan_info(capacity, k, n)?.scratch_bytes as usize;
        }
        Ok(bytes)
    }
    fn b(&self, i: usize) -> CuteafdDeviceBuffer {
        self.storage.buffers[i].as_ref().expect("query buffer is allocated").buffer
    }
    /// Drops a finished output (the slot is at rest: its readers drained).
    pub(crate) fn discard_ready(&mut self) { self.ready = None; }
    pub(crate) fn select_window(&mut self, window: crate::shared::memory::row_window::RowWindow) -> Result<()> {
        if let Some(range) = window.range() {
            ensure!(range.start < range.end && range.len() <= self.capacity as usize, "query window exceeds unit");
        }
        self.ready = None; self.binding = None; self.tokens.clear(); self.window = window; Ok(())
    }
    /// Releases the outputs for the next unit's writer, in any slot: call
    /// after the attention hook's last reader joined the unit's stage chain
    /// (`shared::memory::row_window::ScratchGuard`). No-op outside a chain.
    pub(crate) fn release_scratch(&self) -> Result<()> { self.storage.guard.release() }
    /// Orders this stream's next writes after the last release.
    fn acquire_scratch(&self) -> Result<()> {
        #[cfg(test)]
        if SKIP_ACQUIRE.with(|skip| skip.get()) { return Ok(()); }
        // SAFETY: the storage's own live stream, on the current device.
        unsafe { self.storage.guard.acquire(self.storage.stream.raw) }
    }
    pub fn layer(&self) -> usize {
        self.weights.layer
    }
    pub fn input(&self) -> CuteafdDeviceBuffer {
        self.b(0)
    }
    pub fn positions(&self) -> CuteafdDeviceBuffer {
        self.b(5)
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.storage.stream.library.cuda_stream_synchronize(self.storage.stream.raw) }
    }
    fn validate(&mut self, rows: u32) -> Result<()> {
        self.ready = None;
        self.binding = None;
        self.tokens.clear();
        ensure!(
            rows > 0 && rows <= self.capacity && self.window.admits(rows as usize),
            "query rows exceed capacity or differ from slot window"
        );
        Ok(())
    }
    unsafe fn enqueue_rank(&self, rows: u32) -> Result<()> {
        unsafe {
            self.norm.backbone_frequencies(
                self.b(5),
                self.b(6),
                rows,
                self.weights.layer as u32,
                self.storage.stream.raw,
            )?;
            self.storage.kernels[0].launch(
                self.b(0),
                self.weights.tensors.get(&self.weights.names[0])?,
                self.weights.scales[0].buffer,
                self.storage.scratch[0].buffer,
                self.storage.alpha.buffer,
                self.b(1),
                rows,
                self.storage.stream.raw,
            )?;
            self.norm.norm(
                self.b(1),
                self.weights.tensors.get(&self.weights.names[4])?,
                None,
                self.b(2),
                rows,
                1280,
                self.storage.stream.raw,
            )?;
        }
        Ok(())
    }
    unsafe fn enqueue(&self, rows: u32) -> Result<()> {
        ensure!(!self.weights.split_query_b,"split query-B weights require TP2 execution");
        unsafe {
            self.enqueue_rank(rows)?;
            self.storage.kernels[1].launch(
                self.b(2),
                self.weights.tensors.get(&self.weights.names[2])?,
                self.weights.scales[1].buffer,
                self.storage.scratch[1].buffer,
                self.storage.alpha.buffer,
                self.b(3),
                rows,
                self.storage.stream.raw,
            )?;
            // Preserve the pre-RoPE tensor only for numerical trace fixtures.
            // Serving rotates the completed projection in place, saving one
            // 64-head BF16 buffer per lane without changing the kernel math.
            #[cfg(test)]
            self.storage.stream.library.copy_d2d_async(self.b(4), self.b(3),
                rows as usize * ROW_BYTES[3], self.storage.stream.raw)?;
            self.norm.rope(
                self.b(3),
                self.b(6),
                self.b(3),
                rows,
                64,
                false,
                self.storage.stream.raw,
            )?;
        }
        Ok(())
    }
    /// # Safety
    /// Inputs are finite, initialized in matching row order on this device, with
    /// producer writes complete. Positions are below 1048576. No writes may race
    /// this wave; callers bind outputs to the matching request/query snapshot.
    pub unsafe fn execute(&mut self, rows: u32) -> Result<AttentionQueryOutput<'_>> {
        self.validate(rows)?;
        // Launch and return under a stage chain; outside one, `finish` drains.
        let launched = unsafe { self.enqueue(rows) };
        launched.and(unsafe { crate::shared::memory::chain::finish(self.storage.stream.library, self.storage.stream.raw) })?;
        self.ready = Some(rows);
        self.output()
    }
    /// # Safety
    /// Same inputs as execute. Warmup is drained but not published after capture.
    pub unsafe fn capture(&mut self, rows: u32) -> Result<()> {
        self.ready = None;
        self.binding = None;
        self.tokens.clear();
        ensure!(
            self.graphs.get_shape(self.weights.layer, self.weights, rows).is_none(),
            "attention query graph already captured"
        );
        unsafe {
            self.execute(rows)?;
        }
        self.ready = None;
        self.binding = None;
        self.tokens.clear();
        unsafe { self.capture_ready(rows) }
    }
    pub(crate) fn capture_startup(&mut self, rows: u32) -> Result<()> {
        self.validate(rows)?;
        self.storage.stream.require_complete()?;
        // SAFETY: capture records only this owner's stable buffers and bound
        // weights. It does not execute kernels or publish a query binding.
        unsafe { self.capture_ready_kind(rows, self.weights.split_query_b) }
    }
    /// Capture only after the input producer and warmup have completed.
    unsafe fn capture_ready(&mut self, rows: u32) -> Result<()> {
        unsafe { self.capture_ready_kind(rows, false) }
    }
    unsafe fn capture_ready_kind(&mut self, rows: u32, prefix: bool) -> Result<()> {
        let graphs = if prefix { &mut self.tp2_prefix_graphs } else { &mut self.graphs };
        let Some(ticket) = graphs.cache_ticket(&(self.weights.layer, rows), super::graph_policy::pinned_shape(rows), None) else {
            super::graph_policy::note_miss(format!("attention_query rows={rows} layer={}", self.weights.layer), || "graph cache selected eager".into());
            return Ok(());
        };
        unsafe {
            self.storage.stream
                .library
                .cuda_graph_begin_capture(self.storage.stream.raw)?;
        }
        let launched = unsafe { if prefix { self.enqueue_rank(rows) } else { self.enqueue(rows) } };
        let captured = unsafe { self.storage.stream.library.cuda_graph_end_capture_topology(self.storage.stream.raw) };
        match (launched, captured) {
            (Ok(()), Ok((graph, nodes, width, spans, _))) => {
                let graphs = if prefix { &mut self.tp2_prefix_graphs } else { &mut self.graphs };
                if let Err(error) = unsafe {
                    graphs.cache_insert(self.weights.layer, self.weights, rows, graph, ticket, nodes as u64, width, spans || prefix)
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
    pub unsafe fn replay(&mut self, rows: u32) -> Result<AttentionQueryOutput<'_>> {
        self.validate(rows)?;
        let Some((graph, count)) = self.graphs.cached_launch(self.weights.layer, self.weights, rows) else {
            return unsafe { self.execute(rows) };
        };
        ensure!(count == rows, "attention query capture row count differs");
        let launched = unsafe {
            self.storage.stream
                .library
                .cuda_graph_launch(graph, self.storage.stream.raw)
        };
        launched.and(unsafe { crate::shared::memory::chain::finish(self.storage.stream.library, self.storage.stream.raw) })?;
        self.ready = Some(rows);
        self.output()
    }
    /// # Safety
    /// Hidden rows are finite attention inputs in the supplied token order, with
    /// producer writes drained. No external writes race this wave.
    pub unsafe fn execute_tokens(&mut self, tokens: &[u64]) -> Result<AttentionQueryOutput<'_>> {
        unsafe { self.execute_tokens_prepared(tokens, |_, _| Ok(())) }
    }
    /// Enqueue an input producer before the query graph on its owned stream.
    ///
    /// # Safety
    /// The producer must initialize finite hidden rows in token order, retain
    /// its allocations through this call, and use only the supplied stream.
    /// Success and producer/query errors both drain before releasing its borrow.
    /// Preparation is outside capture; cached graphs contain query operations only.
    pub(crate) unsafe fn execute_tokens_prepared(
        &mut self,
        tokens: &[u64],
        prepare: impl FnOnce(*mut std::ffi::c_void, CuteafdDeviceBuffer) -> Result<()>,
    ) -> Result<AttentionQueryOutput<'_>> {
        self.ready = None;
        self.binding = None;
        self.tokens.clear();
        ensure!(
            !tokens.is_empty()
                && tokens.len() <= self.capacity as usize
                && tokens.iter().all(|&p| p < 1048576),
            "invalid query tokens"
        );
        let binding = QueryBinding::new(self.weights.layer)?;
        for (dst, token) in self.position_staging.bytes_mut().chunks_exact_mut(8).zip(tokens) {
            dst.copy_from_slice(&token.to_ne_bytes());
        }
        let prepared_and_executed = (|| -> Result<()> {
            unsafe {
                crate::shared::memory::chain::join(self.storage.stream.library, self.storage.stream.raw)?;
                self.acquire_scratch()?;
                self.storage.stream.library.copy_host_buffer_h2d_async(
                    self.positions(), self.position_staging.buffer, tokens.len() * 8, self.storage.stream.raw,
                )?;
            }
            prepare(self.storage.stream.raw, self.input())?;
            // The normalized input is complete here; cache producers fork from it.
            unsafe { crate::shared::memory::chain::mark_fork(self.storage.stream.library, self.storage.stream.raw)?; }
            let rows = tokens.len() as u32;
            if self.window.is_windowed() || !super::graph_policy::captures_shape(rows) {
                unsafe { self.execute(rows)?; }
                return Ok(());
            }
            if self
                .graphs
                .get_shape(self.weights.layer, self.weights, rows)
                .is_none()
            {
                unsafe { self.capture(rows)?; }
            }
            unsafe { self.replay(rows)?; }
            Ok(())
        })();
        if let Err(error) = prepared_and_executed {
            // Even a partially submitted producer must finish before its owner
            // can be reset or reused. Successful replay already drained.
            let drained = self.synchronize();
            self.ready = None;
            self.binding = None;
            if let Err(drain_error) = drained {
                tracing::error!(%drain_error, "draining failed query preparation");
            }
            return Err(error);
        }
        self.tokens.extend_from_slice(tokens);
        self.binding = Some(binding);
        self.output()
    }
    /// # Safety
    /// Same producer/ownership contract as execute_tokens_prepared, retained
    /// across suspension. All producer and query work uses this one stream.
    pub(crate) async unsafe fn execute_tokens_prepared_cooperative(&mut self, tokens: &[u64],
        prepare: impl FnOnce(*mut std::ffi::c_void, CuteafdDeviceBuffer) -> Result<()>)
        -> Result<AttentionQueryOutput<'_>> {
        ensure!(tokens.len() <= self.capacity as usize, "query rows exceed capacity");
        self.validate(tokens.len() as u32)?;
        ensure!(tokens.iter().all(|&p| p < 1048576), "invalid query tokens");
        let binding = QueryBinding::new(self.weights.layer)?;
        let rows = tokens.len() as u32;
        for (dst, token) in self.position_staging.bytes_mut().chunks_exact_mut(8).zip(tokens) {
            dst.copy_from_slice(&token.to_ne_bytes());
        }
        let graph = if self.window.is_windowed() { None } else { self.graphs.cached_launch(self.weights.layer, self.weights, rows) };
        let queued = (|| -> Result<()> { unsafe {
            crate::shared::memory::chain::join(self.storage.stream.library, self.storage.stream.raw)?;
            self.acquire_scratch()?;
            self.storage.stream.library.copy_host_buffer_h2d_async(self.positions(), self.position_staging.buffer,
                tokens.len()*8, self.storage.stream.raw)?;
            prepare(self.storage.stream.raw, self.input())?;
            // The normalized input is complete here; cache producers fork from it.
            crate::shared::memory::chain::mark_fork(self.storage.stream.library, self.storage.stream.raw)?;
            crate::shared::decode_graph::census::dispatch(self.storage.stream.library, self.storage.stream.raw, c"layer_row",
                rows as usize, graph.map(|(raw, _)| raw), || self.enqueue(rows))
        } })();
        let drained = if queued.is_err() { self.storage.stream.wait().await }
            else { unsafe { crate::shared::memory::chain::finish_cooperative(&self.storage.stream).await } };
        queued.and(drained)?;
        if graph.is_none() {
            // Eager output is complete; capture only records the next execution.
            // Replaying now would duplicate work whenever a shape was evicted.
            if !self.window.is_windowed() && super::graph_policy::captures_shape(rows)
                && !crate::shared::decode_graph::census::eager("layer_row") {
                unsafe { self.capture_ready(rows)?; }
            }
        }
        self.ready = Some(rows);
        self.tokens.extend_from_slice(tokens);
        self.binding = Some(binding);
        self.output()
    }
    /// Query-A/norm stays on the layer owner; query-B uses both RTX shards.
    /// The projection owner appends rotary before its final wait, preserving
    /// the ordinary query output/binding contract for attention and the indexer.
    /// # Safety
    /// Same producer ownership contract as execute_tokens_prepared_cooperative.
    /// Projection weights must be this layer's checkpoint query-B shards and
    /// its gathered output device must match this query owner.
    pub(crate) async unsafe fn execute_tokens_tp2_prepared_cooperative(
        &mut self, tokens: &[u64], projection: &mut crate::families::deepseek_v41::v41_projection_tp2::Wave<'_, '_>,
        prepare: impl FnOnce(*mut std::ffi::c_void, CuteafdDeviceBuffer)->Result<()>)
        ->Result<AttentionQueryOutput<'_>> {
        ensure!(tokens.len()<=self.capacity as usize,"query rows exceed capacity");
        self.validate(tokens.len() as u32)?;
        ensure!(tokens.iter().all(|&p|p<1048576)
            && projection.kind()==crate::families::deepseek_v41::v41_projection_tp2::Kind::QueryB
            && projection.output_device().id==self.input().device_id
            && std::ptr::eq(projection.output_device().library,self.storage.stream.library),"TP2 query origin differs");
        let rows=tokens.len() as u32;
        let binding=QueryBinding::new(self.weights.layer)?;
        for (dst,token) in self.position_staging.bytes_mut().chunks_exact_mut(8).zip(tokens) {
            dst.copy_from_slice(&token.to_ne_bytes());
        }
        struct Drain<'s,'a> { stream:&'s LoadStream<'a>, complete:bool }
        impl Drop for Drain<'_,'_> {
            fn drop(&mut self) {
                if !self.complete {
                    if let Err(error)=unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) } {
                        crate::shared::decode_graph::fatal_drain(Err(error), "TP2 query unwind");
                    }
                }
            }
        }
        let storage = std::rc::Rc::clone(&self.storage);
        let mut guard=Drain { stream:&storage.stream,complete:false };
        let graph=if self.window.is_windowed() { None } else { self.tp2_prefix_graphs.cached_launch(self.weights.layer,self.weights,rows) };
        unsafe {
            // The projection's own stream writes `b(3)` after this stream's
            // rank stage, so this acquire orders it too.
            self.acquire_scratch()?;
            self.storage.stream.library.copy_host_buffer_h2d_async(self.positions(),self.position_staging.buffer,
                tokens.len()*8,self.storage.stream.raw)?;
            prepare(self.storage.stream.raw,self.input())?;
            crate::shared::decode_graph::census::dispatch(self.storage.stream.library, self.storage.stream.raw, c"layer_row",
                rows as usize, graph.map(|(raw, _)| raw), || self.enqueue_rank(rows))?;
            projection.execute_after(self.weights.layer,rows,self.b(2),Some(self.storage.stream.raw),|projected,stream| {
                self.storage.stream.library.copy_d2d_async(self.b(3),projected,rows as usize*ROW_BYTES[3],stream)?;
                #[cfg(test)]
                self.storage.stream.library.copy_d2d_async(self.b(4),projected,rows as usize*ROW_BYTES[3],stream)?;
                self.norm.rope(self.b(3),self.b(6),self.b(3),rows,64,false,stream)
            }).await?;
        }
        if !self.window.is_windowed() && graph.is_none() && super::graph_policy::captures_shape(rows)
            && !crate::shared::decode_graph::census::eager("layer_row") {
            unsafe { self.capture_ready_kind(rows, true)?; }
        }
        guard.complete=true;drop(guard);
        self.ready=Some(rows);self.tokens.extend_from_slice(tokens);self.binding=Some(binding);
        self.output()
    }
    pub fn output(&self) -> Result<AttentionQueryOutput<'_>> {
        let rows = self.ready.context("attention query output unpublished")? as usize;
        let b = |i| part(self.b(i), 0, rows * ROW_BYTES[i]);
        Ok(AttentionQueryOutput {
            binding: self.binding,
            tokens: &self.tokens,
            layer: self.weights.layer,
            rows,
            hidden: b(0),
            raw_rank: b(1),
            normalized_rank: b(2),
            #[cfg(test)]
            projected: b(4),
            #[cfg(test)]
            qb_scratch: self.storage.scratch.get(1).map(|scratch|scratch.buffer),
            rotated: b(3),
            positions: b(5),
            frequencies: b(6),
            _owner: PhantomData,
        })
    }
    /// Evict all shapes for the current layer; other layers remain cached.
    pub fn clear_graph(&mut self) -> Result<()> {
        self.ready = None;
        self.binding = None;
        self.tokens.clear();
        self.synchronize()?;
        unsafe {
            self.graphs.remove(self.weights.layer)?;
            self.tp2_prefix_graphs.remove(self.weights.layer)?;
        }
        Ok(())
    }
}
impl Drop for AttentionQueryWave<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self
            .synchronize()
            .and_then(|()| unsafe {
                let first=self.graphs.clear();let second=self.tp2_prefix_graphs.clear();first.and(second)
            })
        {
            crate::shared::decode_graph::fatal_drain(Err(error), "attention query");
        }
    }
}

#[cfg(test)]
mod tp2_tests {
    use super::*;
    use crate::shared::memory::device::{Allocation,Device};
    use crate::families::deepseek_v41::v41_projection_tp2::{Kind,Weights,Wave};
    #[test]
    #[ignore = "requires checkpoint, two GPUs and projection shard AOT"]
    fn checkpoint_tp2_query_preserves_rank_and_rotary() -> Result<()> {
        let library=unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let catalog=cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        let devices=[Device { library:&library,id:0 },Device { library:&library,id:1 }];
        let runtime=tokio::runtime::Builder::new_current_thread().build()?;
        let read=|device:Device,output:AttentionQueryOutput<'_>|->Result<Vec<Vec<u8>>> {
            [output.hidden,output.raw_rank,output.normalized_rank,output.projected,
                output.rotated,output.positions,output.frequencies].into_iter().map(|view| {
                    let mut bytes=vec![0;view.bytes];device.run(||library.copy_d2h(&mut bytes,view))?;Ok(bytes)
                }).collect()
        };
        for (owner,layer) in [(0,2),(1,20)] {
            let device=devices[owner];let capacity=16;
            let weights=device.own(||AttentionQueryWeights::load(&library,&catalog,layer,
                AttentionQueryWeights::device_bytes(&library,&catalog,layer)?,1<<20))?;
            let mut reference=device.own(||weights.wave(capacity,usize::MAX))?;
            let split_weights=device.own(||AttentionQueryWeights::load_with_split(&library,&catalog,layer,
                AttentionQueryWeights::device_bytes_with_split(&library,&catalog,layer,true)?,1<<20,true))?;
            assert!(split_weights.tensors.get(&split_weights.names[2]).is_err());
            assert_eq!(split_weights.scales.len(),1);
            let mut query=device.own(||split_weights.wave(capacity,usize::MAX))?;
            assert_eq!(query.storage.scratch.len(),1);
            assert!(query.rebind(&weights).is_err());
            let halves=[vec![Weights::load(devices[0],&catalog,layer,Kind::QueryB,0,
                Weights::load_peak_device_bytes(Kind::QueryB))?],
                vec![Weights::load(devices[1],&catalog,layer,Kind::QueryB,1,
                Weights::load_peak_device_bytes(Kind::QueryB))?]];
            let mut projection=Wave::new([&halves[0],&halves[1]],capacity,owner,
                Wave::device_bytes(&library,Kind::QueryB,capacity,owner)?)?;
            let input=Allocation::new(device,capacity as usize*10240, crate::shared::memory::TEST)?;
            for (cycle,rows) in [1,6,16,6,1].into_iter().enumerate() {
                let host:Vec<u8>=(0..capacity as usize*5120).flat_map(|i| {
                    let value=((i+cycle*3)%23) as f32/32.0-0.25;
                    ((value.to_bits()>>16) as u16).to_ne_bytes()
                }).collect();
                device.run(||library.copy_h2d(input.buffer,&host))?;
                let tokens:Vec<u64>=(0..rows).map(|i|(cycle*8192+i) as u64).collect();
                let prepare=|stream,destination|unsafe {
                    library.copy_d2d_async(destination,input.buffer,rows*10240,stream)
                };
                let full=runtime.block_on(device.future(unsafe {
                    reference.execute_tokens_prepared_cooperative(&tokens,prepare)
                }))?;
                let expected=read(device,full)?;
                // A dropped future cannot publish a partially completed query.
                let cancelled=runtime.block_on(async {
                    let mut pending=std::pin::pin!(device.future(unsafe {
                        query.execute_tokens_tp2_prepared_cooperative(&tokens,&mut projection,prepare)
                    }));
                    std::future::poll_fn(|cx| {
                        use std::future::Future;
                        std::task::Poll::Ready(pending.as_mut().poll(cx).is_pending())
                    }).await
                });
                if cancelled { assert!(query.output().is_err()); }
                let actual=runtime.block_on(device.future(unsafe {
                    query.execute_tokens_tp2_prepared_cooperative(&tokens,&mut projection,prepare)
                }))?;
                assert_eq!(actual.tokens()?,tokens);
                actual.binding()?;
                let actual=read(device,actual)?;
                for index in [0,1,2,5,6] { assert_eq!(actual[index],expected[index]); }
                let mut max_error=0f32;
                for index in [3,4] {
                    for (a,b) in actual[index].chunks_exact(2).zip(expected[index].chunks_exact(2)) {
                        let decode=|v:&[u8]|f32::from_bits(u32::from(u16::from_ne_bytes([v[0],v[1]]))<<16);
                        let (a,b)=(decode(a),decode(b));
                        ensure!(a.is_finite() && b.is_finite() && (a-b).abs()<=1e-4,
                            "TP2 query differs: layer={layer} rows={rows} output={index} {a} vs {b}");
                        max_error=max_error.max((a-b).abs());
                    }
                }
                eprintln!("TP2 complete query layer={layer} rows={rows} cycle={cycle} max_error={max_error}");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
impl AttentionQueryWeights<'_> {
    pub(crate) fn visit(&self, prefix: &str, f: &mut dyn FnMut(String, cuteafd_ffi::CuteafdDeviceBuffer)) {
        self.tensors.visit(prefix, f);
        for (i, s) in self.scales.iter().enumerate() { f(format!("{prefix}packed_scale{i}"), s.buffer); }
    }
}

/// Reference GPU check of the row_window contract's guarded scratch. Two
/// units share one query storage. Unit A's output is read by a fork reader on
/// another stream (this GPU, or the other GPU on two RTX). That reader is held
/// by a gate until unit B has been queued, then joins A's chain, and A
/// releases. Unit B acquires before writing, so it must wait for A's reader:
/// A's copy has to equal a private run. Without the guard, B overwrites the
/// shared outputs first.
#[cfg(test)]
mod guarded_scratch_tests {
    use super::*;
    use crate::shared::memory::chain::StageChain;
    use crate::shared::memory::device::{Device, Stream};

    /// `guarded`: false skips unit B's acquire (test-only), which must let B
    /// overwrite A's outputs before A's reader: the test can fail.
    fn run(reader_device: i32, guarded: bool) -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        library.cuda_set_device(0)?;
        let weights = AttentionQueryWeights::load(&library, &catalog, 2,
            AttentionQueryWeights::device_bytes(&library, &catalog, 2)?, 1 << 20)?;
        let (unit, rows) = (16u32, 13usize);
        let devices: Vec<i32> = if reader_device == 0 { vec![0] } else { vec![0, reader_device] };
        if reader_device != 0 { library.cuda_enable_peer(reader_device)?; }
        let shared = weights.storage(unit, &devices)?;
        let mut slots = [weights.on_storage(shared.clone())?, weights.on_storage(shared.clone())?];
        let input = |seed: usize| -> Vec<u8> { (0..rows * 5120).flat_map(|i| {
            let value = ((i * 7 + seed * 11) % 29) as f32 / 32.0 - 0.4;
            ((value.to_bits() >> 16) as u16).to_ne_bytes()
        }).collect() };
        let tokens = |seed: usize| -> Vec<u64> { (0..rows).map(|i| (seed * 4096 + i * 3) as u64).collect() };
        let sources = [input(0), input(1)].map(|host| -> Result<crate::shared::memory::DeviceAllocation<'_>> {
            let buffer = crate::shared::memory::DeviceAllocation::new(&library, host.len(), crate::shared::memory::TEST)?;
            library.copy_h2d(buffer.buffer, &host)?;
            Ok(buffer)
        }).into_iter().collect::<Result<Vec<_>>>()?;
        // The fork reader: a stream on `reader_device`, copying A's rotated rows.
        let reader = Stream::new(Device { library: &library, id: reader_device })?;
        let rotated_bytes = rows * ROW_BYTES[3];
        library.cuda_set_device(reader_device)?;
        let copy = crate::shared::memory::DeviceAllocation::new(&library, rotated_bytes, crate::shared::memory::TEST)?;
        library.cuda_set_device(0)?;
        let chains = [StageChain::on_devices(&library, &devices)?, StageChain::on_devices(&library, &devices)?];
        for chain in &chains { chain.set_device_order(true); }
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        // Warm every kernel and copy path ungated first: with lazy module
        // loading, a first launch inside the gated interval loads its module,
        // which synchronizes the context behind the gated stream and hangs.
        // The private run is also the expected value.
        let want = {
            let mut private = weights.wave(unit, usize::MAX)?;
            library.copy_h2d(private.input(), &input(0))?;
            let expected = unsafe { private.execute_tokens(&tokens(0))? };
            let mut want = vec![0; rotated_bytes];
            library.copy_d2h(&mut want, expected.rotated)?;
            want
        };
        for (seed, slot) in slots.iter_mut().enumerate() {
            let source = sources[seed].buffer;
            unsafe { slot.execute_tokens_prepared(&tokens(seed), |stream, destination| {
                library.copy_d2d_async(destination, source, source.bytes, stream)
            })?; }
        }
        reader.device.run(|| unsafe {
            let warm = shared.buffers[3].as_ref().unwrap().buffer;
            if reader_device == 0 { library.copy_d2d_async(copy.buffer, warm, rotated_bytes, reader.raw)?; } else { library.copy_peer_async(copy.buffer, warm, rotated_bytes, reader.raw)?; }
            library.cuda_stream_synchronize(reader.raw)
        })?;
        unsafe { library.cuda_stream_synchronize(shared.stream.raw)?; }
        let reader_gate = unsafe { cuteafd_ffi::test_support::CudaStreamGate::new(&library, reader.raw)? };
        // Unit A: query, then its fork reader (gated), joined into A's chain, release.
        runtime.block_on(chains[0].handle().scope(async {
            let source = sources[0].buffer;
            let output = unsafe { slots[0].execute_tokens_prepared(&tokens(0), |stream, destination| {
                library.copy_d2d_async(destination, source, source.bytes, stream)
            })? };
            let rotated = output.rotated;
            reader.device.run(|| unsafe {
                crate::shared::memory::chain::join(&library, reader.raw)?;
                if reader_device == 0 { library.copy_d2d_async(copy.buffer, rotated, rotated_bytes, reader.raw)?; } else { library.copy_peer_async(copy.buffer, rotated, rotated_bytes, reader.raw)?; }
                crate::shared::memory::chain::finish(&library, reader.raw)
            })?;
            slots[0].release_scratch()
        }))?;
        // Unit B (another slot, its own chain): acquires, then overwrites.
        if !guarded { SKIP_ACQUIRE.with(|skip| skip.set(true)); }
        runtime.block_on(chains[1].handle().scope(async {
            let source = sources[1].buffer;
            unsafe { slots[1].execute_tokens_prepared(&tokens(1), |stream, destination| {
                library.copy_d2d_async(destination, source, source.bytes, stream)
            })?; }
            anyhow::Ok(())
        }))?;
        // B is queued. Give it every chance to run: unordered (no guard) it
        // completes while A's reader is held; guarded it must still be
        // pending. Asserting both makes the outcome deterministic, not a race
        // between B's launch and the gate's release.
        let deadline = std::time::Instant::now()
            + std::time::Duration::from_secs(if guarded { 2 } else { 10 });
        let mut b_done = false;
        while std::time::Instant::now() < deadline {
            if unsafe { library.cuda_stream_query(shared.stream.raw)? } { b_done = true; break; }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(b_done, !guarded, "unit B {} while A's fork reader was held",
            if b_done { "completed" } else { "stayed pending" });
        // Only now may A's reader run.
        drop(reader_gate);
        for chain in &chains { chain.drain()?; }
        unsafe { library.cuda_stream_synchronize(shared.stream.raw)?; }
        reader.device.run(|| unsafe { library.cuda_stream_synchronize(reader.raw) })?;
        let mut got = vec![0; rotated_bytes];
        reader.device.run(|| library.copy_d2h(&mut got, copy.buffer))?;
        SKIP_ACQUIRE.with(|skip| skip.set(false));
        if guarded {
            assert_eq!(got, want, "unit B overwrote shared query outputs before unit A's fork reader joined");
        } else {
            assert_ne!(got, want, "without the guard unit B should overwrite first: the test cannot detect a race");
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_SNAPSHOT and a GPU"]
    fn guarded_query_scratch_waits_for_a_same_gpu_fork_reader() -> Result<()> { run(0, true) }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_SNAPSHOT and a GPU"]
    fn unguarded_query_scratch_races_a_gated_fork_reader() -> Result<()> { run(0, false) }

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_SNAPSHOT and two GPUs"]
    fn guarded_query_scratch_waits_for_a_fork_reader_on_the_other_gpu() -> Result<()> { run(1, true) }
}

#[cfg(test)]
mod bounded_graph_tests {
    use super::*;
    use crate::shared::decode_graph::{device::{Basis, Cycle, Preference, Site}, RowGraphs};

    #[test]
    fn primitive_wrappers_full_reserve_dispatch_eager_without_refusal() -> Result<()> {
        let (_directory, fixture) = crate::shared::memory::cpu_stream_fixture()?;
        let library = fixture.load()?;
        let site = Site { name: "v41.primitive.full-reserve", cycle: Cycle::Wave,
            preference: Preference::Capture, bootstrap: 10, basis: Basis::Bytes, spans_devices: false };
        let weights = ();
        let mut layers = LayerGraphs::new(&library, 64, "deepseek_v41");
        let mut rows = RowGraphs::new(&library, "engram", 64, "deepseek_v41");
        layers.attach_device(-88441, 20, site)?;
        rows.attach_device(-88441, 20, site)?;
        let ticket = layers.cache_ticket(&(0, 1), true, None).expect("pinned seed fits");
        // SAFETY: the CPU fixture records no device work. Weights outlive the
        // fake executable and every consumer completes before bank destruction.
        unsafe {
            library.cuda_graph_begin_capture(std::ptr::null_mut())?;
            let raw = library.cuda_graph_end_capture(std::ptr::null_mut())?;
            layers.cache_insert(0, &weights, 1, raw, ticket, 1, 1, false)?;
        }
        layers.cache_measured(20);
        let mut eager_runs = 0;
        for _ in 0..3 {
            assert!(layers.cache_ticket(&(0, 16), false, None).is_none());
            assert!(rows.cache_ticket(&16, false, None).is_none());
            assert!(layers.get_shape(0, &weights, 1).is_some(), "full reserve cannot evict its pin");
            let graph = layers.cached_launch(0, &weights, 16).map(|(raw, _)| raw);
            // SAFETY: eager closure has no device operands; the cache miss must
            // choose it rather than trying to launch a nonexistent executable.
            unsafe {
                crate::shared::decode_graph::census::dispatch(&library, std::ptr::null_mut(), c"layer_row", 16,
                    graph, || { eager_runs += 1; Ok(()) })?;
                crate::shared::decode_graph::census::dispatch(&library, std::ptr::null_mut(), c"layer_row", 16,
                    rows.cached_launch(16), || { eager_runs += 1; Ok(()) })?;
            }
        }
        assert_eq!(eager_runs, 6);
        assert_eq!(layers.len(), 1);
        assert_eq!(rows.len(), 0);
        assert!(layers.cached_launch(0, &weights, 1).is_some());
        Ok(())
    }

    #[test]
    fn primitive_native_capture_inventory_uses_ticket_topology_and_visitor() {
        let sites = [
            (include_str!("v41_attention_query.rs"), 1),
            (include_str!("v41_attention_output.rs"), 1),
            (include_str!("v41_backbone_router.rs"), 1),
            (include_str!("v41_backbone_shared.rs"), 1),
            (include_str!("v41_index_query.rs"), 1),
            (include_str!("v41_projection_tp2.rs"), 2),
            (include_str!("v41_engram/layer.rs"), 1),
        ];
        for (source, count) in sites {
            // Only count the production declarations, excluding CPU fixture
            // captures and this source-inventory test itself.
            let production = source.split("#[cfg(test)]\nmod bounded_graph_tests").next().unwrap();
            assert_eq!(production.matches(".cuda_graph_begin_capture(").count(), count);
            assert_eq!(production.matches(".cuda_graph_end_capture_topology(").count(), count);
            assert_eq!(production.matches(".cache_ticket(").count(), count);
            assert_eq!(production.matches(".cache_insert(").count(), count);
            assert!(production.contains("fn visit_graph_banks("));
            assert!(production.contains(".cached_launch("));
            assert!(!production.contains(".capture_allowed("));
            assert!(!production.contains(".declare_startup("));
            assert!(!production.contains(".finish_startup("));
        }
    }
}
