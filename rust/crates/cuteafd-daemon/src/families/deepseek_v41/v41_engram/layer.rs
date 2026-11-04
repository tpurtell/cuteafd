//! Native engram projection carriers and fused residual gate ownership.
use super::EngramDeviceView;
use std::rc::Rc;
use crate::shared::memory::{chain, row_window::{RowWindow, ScratchGuard, MAX_RESERVE_ROWS}};
use crate::shared::memory::{DeviceAllocation, LoadStream};
use crate::families::deepseek_v41::v41_tensors::NativeRtxTensors;
use crate::shared::decode_graph::RowGraphs;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary, V41Fp8Plan};
use cuteafd_loader::OfficialV41Catalog;
#[cfg(test)]
thread_local! {
    static SKIP_ACQUIRE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) struct EngramLayerWeights<'a> {
    tensors: NativeRtxTensors<'a>,
    packed_scales: DeviceAllocation<'a>,
    library: &'a NativeLibrary,
    layer_index: usize,
    names: [String; 4],
}
impl<'a> EngramLayerWeights<'a> {
    pub fn device_bytes(library:&NativeLibrary,catalog:&OfficialV41Catalog,layer_index:usize)->Result<usize> {
        let layer=*cuteafd_core::ENGRAM_LAYERS.get(layer_index).context("invalid engram layer index")?;
        let names=["wkv.weight","wkv.scale","q_weight","k_weight"].map(|suffix| format!("layers.{layer}.engram.{suffix}"));
        NativeRtxTensors::plan(catalog,&names)?.checked_add(library.v41_fp8_kernel(16)?.info().packed_weight_scale_bytes as usize)
            .context("engram resident budget overflow")
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer_index: usize,
        device_budget: usize,
        staging_bytes: usize,
    ) -> Result<Self> {
        let layer = *cuteafd_core::ENGRAM_LAYERS
            .get(layer_index)
            .context("invalid engram layer index")?;
        let names = ["wkv.weight", "wkv.scale", "q_weight", "k_weight"]
            .map(|suffix| format!("layers.{layer}.engram.{suffix}"));
        let kernel = library.v41_fp8_kernel(16)?;
        let packed_bytes = usize::try_from(kernel.info().packed_weight_scale_bytes)?;
        let required = NativeRtxTensors::plan(catalog, &names)?
            .checked_add(packed_bytes)
            .context("engram resident budget overflow")?;
        ensure!(
            required <= device_budget,
            "engram weights and packed scales exceed device budget"
        );
        let tensors = NativeRtxTensors::load(
            library,
            catalog,
            &names,
            device_budget - packed_bytes,
            staging_bytes,
        )?;
        let packed_scales = DeviceAllocation::new(library, packed_bytes)?;
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        unsafe {
            crate::shared::experts::rtx::dense::v41_pack_scales(&kernel, tensors.get(&names[1])?, packed_scales.buffer, stream.raw)?;
            library.cuda_stream_synchronize(stream.raw)?;
        }
        Ok(Self {
            tensors,
            packed_scales,
            library,
            layer_index,
            names,
        })
    }
    pub fn resident_bytes(&self) -> usize {
        self.tensors.resident_bytes() + self.packed_scales.buffer.bytes
    }
}

/// Unit-cap gate scratch: residual/embeddings/mask, projected, GEMM workspace
/// and alpha are consumed on the gate stream within the input hook. Output is
/// guarded scratch: the block copies it on its attention stream in that hook.
/// Release after that reader joins. Weight bindings/graphs/ready are per slot.
pub(crate) struct EngramGateStorage<'a> {
    stream: LoadStream<'a>,
    guard: ScratchGuard<'a>,
    output: DeviceAllocation<'a>,
    residual: DeviceAllocation<'a>,
    embeddings: DeviceAllocation<'a>,
    text_mask: DeviceAllocation<'a>,
    projected: DeviceAllocation<'a>,
    scratch: DeviceAllocation<'a>,
    alpha: DeviceAllocation<'a>,
    kernel: V41Fp8Plan<'a>,
    rows: usize,
    unit: usize,
}
impl Drop for EngramGateStorage<'_> {
    fn drop(&mut self) {
        // SAFETY: owned stream; all scratch users precede its drain.
        crate::shared::decode_graph::fatal_drain(unsafe {
            self.stream.library.cuda_stream_synchronize(self.stream.raw)
        }, "Engram gate storage");
    }
}
pub(crate) struct EngramGate<'weights, 'library> {
    storage: Rc<EngramGateStorage<'library>>,
    window: RowWindow,
    graphs: RowGraphs<'library>,
    weights: &'weights EngramLayerWeights<'library>,
    capacity: usize,
    ready_rows: Option<usize>,
}
impl<'weights, 'library> EngramGate<'weights, 'library> {
    pub fn layer(&self) -> usize {
        cuteafd_core::ENGRAM_LAYERS[self.weights.layer_index] as usize
    }
    /// # Safety
    /// Same input ownership and row-order contract as execute. Captures only
    /// this gate's owned buffers and retains the finite decode shape set.
    pub unsafe fn execute_captured(
        &mut self,
        residual: CuteafdDeviceBuffer,
        gathered: &EngramDeviceView,
    ) -> Result<CuteafdDeviceBuffer> {
        self.ready_rows = None;
        if self.window.is_windowed() || !crate::families::deepseek_v41::graph_policy::captures_shape(gathered.rows as u32)
            || (self.graphs.get(gathered.rows).is_none() && !self.graphs.capture_allowed(gathered.rows)) {
            return unsafe { self.execute(residual, gathered) };
        }
        if self.graphs.get(gathered.rows).is_none() {
            unsafe { self.capture(residual, gathered)?; }
        }
        unsafe { self.replay(residual, gathered) }
    }
    pub fn device_bytes(library:&NativeLibrary,capacity:usize)->Result<usize> {
        ensure!(capacity>0 && capacity<=4096,"invalid engram gate capacity");
        let scratch=library.v41_fp8_matrix_plan(capacity as u32,6144,25600)?.info().scratch_bytes as usize;
        (capacity*(40960*2+51200+24*512+1)).checked_add(scratch).and_then(|b| b.checked_add(4))
            .context("engram execution budget overflow")
    }
    pub fn new(
        weights: &'weights EngramLayerWeights<'library>,
        capacity: usize,
        device_budget: usize,
    ) -> Result<Self> {
        ensure!(Self::device_bytes(weights.library, capacity)? <= device_budget,
            "engram execution exceeds device budget");
        Self::on_storage(weights, Self::storage(weights.library, capacity, capacity)?)
    }
    pub(crate) fn storage(library: &'library NativeLibrary, rows: usize, unit: usize)
        -> Result<Rc<EngramGateStorage<'library>>> {
        ensure!((1..=MAX_RESERVE_ROWS).contains(&rows) && (1..=4096).contains(&unit)
            && unit <= rows, "invalid Engram gate storage dimensions");
        let kernel = library.v41_fp8_matrix_plan(unit as u32, 6144, 25600)?;
        let scratch_bytes = kernel.info().scratch_bytes as usize;
        let storage = EngramGateStorage {
            stream: LoadStream { library, raw: library.cuda_stream_create()? },
            guard: ScratchGuard::new(library, &[library.cuda_get_device()?])?,
            output: DeviceAllocation::new(library, unit * 40960)?,
            residual: DeviceAllocation::new(library, unit * 40960)?,
            embeddings: DeviceAllocation::new(library, unit * 24 * 512)?,
            text_mask: DeviceAllocation::new(library, unit)?,
            projected: DeviceAllocation::new(library, unit * 51200)?,
            scratch: DeviceAllocation::new(library, scratch_bytes)?,
            alpha: DeviceAllocation::new(library, 4)?, kernel, rows, unit,
        };
        // SAFETY: workspace is owned by storage and initialization drains here.
        unsafe { storage.kernel.initialize_scratch(storage.scratch.buffer,
            storage.alpha.buffer, storage.stream.raw)?;
            library.cuda_stream_synchronize(storage.stream.raw)?; }
        Ok(Rc::new(storage))
    }
    pub(crate) fn on_storage(weights: &'weights EngramLayerWeights<'library>,
        storage: Rc<EngramGateStorage<'library>>) -> Result<Self> {
        ensure!(weights.tensors.get(&weights.names[0])?.device_id == storage.output.buffer.device_id,
            "Engram weights and storage devices differ");
        Ok(Self { capacity: storage.unit, storage, weights, window: RowWindow::NONE,
            graphs: RowGraphs::new(weights.library, "engram", cuteafd_core::V41_VERIFY_ROWS), ready_rows: None })
    }
    /// Slot reuse, including all cross-stream readers, is drained by the caller.
    pub(crate) fn select_window(&mut self, window: RowWindow) -> Result<()> {
        if let Some(range) = window.range() {
            ensure!(range.start < range.end && range.end <= self.storage.rows
                && range.len() <= self.capacity, "Engram gate window outside storage or unit");
        }
        self.ready_rows = None;
        self.window = window;
        Ok(())
    }
    /// Release after the block copied the output and joined the unit chain.
    pub(crate) fn release_scratch(&self) -> Result<()> { self.storage.guard.release() }
    #[cfg(test)]
    pub(crate) fn test_writer(&self, skip: bool) -> *mut std::ffi::c_void {
        SKIP_ACQUIRE.with(|value| value.set(skip));
        self.storage.stream.raw
    }
    fn acquire_scratch(&self) -> Result<()> {
        #[cfg(test)]
        if SKIP_ACQUIRE.with(|skip| skip.get()) { return Ok(()); }
        // SAFETY: this storage owns the live writer stream.
        unsafe { self.storage.guard.acquire(self.storage.stream.raw) }
    }
    /// Quantize gathered rows by K32, project with native FP8 weights, then gate.
    ///
    /// # Safety
    /// Residual BF16 [rows,4,5120], gathered BF16 [rows,24,256] and text mask
    /// must belong to this device and remain valid through this call; producer
    /// writes must be complete. Inputs may use their corresponding `inputs()`
    /// destination; otherwise they must not alias owned workspace.
    pub unsafe fn execute(
        &mut self,
        residual: CuteafdDeviceBuffer,
        gathered: &EngramDeviceView,
    ) -> Result<CuteafdDeviceBuffer> {
        self.ready_rows = None;
        let launched = unsafe {
            chain::join(self.weights.library, self.storage.stream.raw)?;
            self.acquire_scratch()?;
            self.enqueue_inputs(residual, gathered)?;
            self.enqueue(gathered.rows)
        };
        if let Err(error) = launched { self.synchronize()?; return Err(error); }
        // SAFETY: the gate output is produced before the chain head is recorded.
        unsafe { chain::finish(self.weights.library, self.storage.stream.raw)?; }
        self.ready_rows = Some(gathered.rows);
        self.output()
    }
    /// Stable input destinations: residual BF16, embeddings BF16, text-mask bytes.
    /// Never free or retain after this owner drops; finish writes before execution.
    pub fn inputs(&self) -> [CuteafdDeviceBuffer; 3] {
        [
            self.storage.residual.buffer,
            self.storage.embeddings.buffer,
            self.storage.text_mask.buffer,
        ]
    }
    unsafe fn stage_inputs(
        &self,
        residual: CuteafdDeviceBuffer,
        gathered: &EngramDeviceView,
    ) -> Result<()> {
        self.synchronize()?;
        // SAFETY: a block handoff may still be queued on its non-blocking
        // attention stream, even for a private decode gate inside a chain.
        unsafe { chain::join(self.weights.library, self.storage.stream.raw)?;
            self.acquire_scratch()?;
            self.enqueue_inputs(residual, gathered)?; }
        self.synchronize()
    }
    unsafe fn enqueue_inputs(&self, residual: CuteafdDeviceBuffer, gathered: &EngramDeviceView) -> Result<()> {
        ensure!(
            gathered.layer_index == self.weights.layer_index,
            "engram gate layer mismatch"
        );
        ensure!(
            gathered.rows > 0 && gathered.rows <= self.capacity && self.window.admits(gathered.rows),
            "engram gate exceeds capacity"
        );
        let sources = [residual, gathered.embeddings, gathered.text_mask];
        let destinations = self.inputs();
        let sizes = [
            gathered.rows * 4 * 5120 * 2,
            gathered.rows * 24 * 512,
            gathered.rows,
        ];
        // Validate all views and cross-copy hazards before enqueuing any work.
        for (index, source) in sources.iter().enumerate() {
            ensure!(
                !source.ptr.is_null() && source.bytes >= sizes[index],
                "engram input is null or too small"
            );
            let start = source.ptr as usize;
            let end = start
                .checked_add(sizes[index])
                .context("engram input extent overflow")?;
            for (slot, destination) in destinations.iter().enumerate() {
                let dst = destination.ptr as usize;
                let dst_end = dst
                    .checked_add(destination.bytes)
                    .context("engram destination extent overflow")?;
                ensure!(
                    (slot == index && start == dst) || end <= dst || start >= dst_end,
                    "engram input overlaps another owned input destination"
                );
            }
        }
        let copies = (|| -> Result<()> {
            for index in 0..3 {
                if sources[index].ptr != destinations[index].ptr {
                    unsafe {
                        self.weights.library.copy_d2d_async(
                            destinations[index],
                            sources[index],
                            sizes[index],
                            self.storage.stream.raw,
                        )?;
                    }
                }
            }
            Ok(())
        })();
        if let Err(error) = copies { self.synchronize()?; return Err(error); }
        Ok(())
    }
    unsafe fn enqueue(&self, rows: usize) -> Result<()> {
        unsafe {
            self.storage.kernel.launch(
                self.storage.embeddings.buffer,
                self.weights.tensors.get(&self.weights.names[0])?,
                self.weights.packed_scales.buffer,
                self.storage.scratch.buffer,
                self.storage.alpha.buffer,
                self.storage.projected.buffer,
                u32::try_from(rows)?,
                self.storage.stream.raw,
            )?;
            self.weights.library.cuda_engram_gate_bf16_async(
                self.storage.residual.buffer,
                self.storage.projected.buffer,
                self.weights.tensors.get(&self.weights.names[2])?,
                self.weights.tensors.get(&self.weights.names[3])?,
                Some(self.storage.text_mask.buffer),
                self.storage.output.buffer,
                i32::try_from(rows)?,
                self.storage.stream.raw,
            )?;
        }
        Ok(())
    }
    /// # Safety
    /// Same initialized-input contract as execute. Captures only owned addresses;
    /// the supplied external views need not remain alive after this call returns.
    pub unsafe fn capture(
        &mut self,
        residual: CuteafdDeviceBuffer,
        gathered: &EngramDeviceView,
    ) -> Result<()> {
        ensure!(!self.window.is_windowed(), "windowed gate cannot capture");
        ensure!(self.graphs.get(gathered.rows).is_none(), "engram graph is already captured");
        unsafe {
            self.execute(residual, gathered)?;
        }
        // Decode graph warmup must drain even inside a device-ordered chain.
        self.synchronize()?;
        unsafe { self.capture_ready(gathered.rows) }
    }
    pub(crate) fn capture_startup(&mut self, rows: &[usize]) -> Result<()> {
        ensure!(self.ready_rows.is_none() && !self.window.is_windowed(), "Engram startup requires an idle owner");
        self.storage.stream.require_complete()?;
        self.graphs.declare_startup(rows)?;
        for &row in rows {
            ensure!(row <= self.capacity, "Engram startup row exceeds capacity");
            // SAFETY: recording owned gate buffers does not execute kernels,
            // gather table rows, or publish residuals or request state.
            unsafe { self.capture_ready(row)?; }
        }
        self.graphs.finish_startup()
    }
    unsafe fn capture_ready(&mut self, rows: usize) -> Result<()> {
        self.ready_rows = None;
        ensure!(!self.window.is_windowed(), "windowed gate cannot capture");
        unsafe {
            self.weights
                .library
                .cuda_graph_begin_capture(self.storage.stream.raw)?;
        }
        let launched = unsafe { self.enqueue(rows) };
        // End on both paths so a failed enqueue cannot leave the stream capturing.
        let captured = unsafe { self.weights.library.cuda_graph_end_capture(self.storage.stream.raw) };
        match (launched, captured) {
            (Ok(()), Ok(graph)) => {
                // SAFETY: this capture has not launched; a graph the insert
                // retires had its launches drained by the caller (`would_retire`).
                if let Err(error) = unsafe { self.graphs.insert(rows, graph) } {
                    // SAFETY: the rejected graph has no queued launches.
                    unsafe { self.weights.library.cuda_graph_exec_destroy(graph)?; }
                    return Err(error);
                }
                Ok(())
            }
            (Err(error), Ok(graph)) => {
                if let Err(cleanup) = unsafe { self.weights.library.cuda_graph_exec_destroy(graph) }
                {
                    tracing::error!(%cleanup, "destroying failed engram capture");
                }
                Err(error)
            }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// Same initialized-input contract as execute; row count must match capture.
    pub unsafe fn replay(
        &mut self,
        residual: CuteafdDeviceBuffer,
        gathered: &EngramDeviceView,
    ) -> Result<CuteafdDeviceBuffer> {
        self.ready_rows = None;
        let rows = gathered.rows;
        ensure!(!self.window.is_windowed(), "windowed gate cannot replay");
        let graph = self.graphs.get(rows).context("engram graph has not been captured for these rows")?;
        let launched = unsafe {
            self.stage_inputs(residual, gathered)?;
            self.weights
                .library
                .cuda_graph_launch(graph, self.storage.stream.raw)
        };
        launched.and(self.synchronize())?;
        self.ready_rows = Some(rows);
        self.output()
    }
    /// # Safety
    /// Retain gathered storage and exclusive residual ownership through completion
    /// or cancellation drain. The final residual copy is queued after the gate.
    pub async unsafe fn execute_into_cooperative(&mut self, residual: CuteafdDeviceBuffer,
        gathered: &EngramDeviceView) -> Result<()> {
        self.ready_rows = None;
        let graph = if self.window.is_windowed() { None } else { self.graphs.get(gathered.rows) };
        let cold = graph.is_none();
        let launched = (|| unsafe {
            crate::shared::memory::chain::join(self.weights.library, self.storage.stream.raw)?;
            self.acquire_scratch()?;
            self.enqueue_inputs(residual, gathered)?;
            crate::shared::decode_graph::census::dispatch(self.weights.library, self.storage.stream.raw, c"layer_row",
                gathered.rows, graph, || self.enqueue(gathered.rows))
        })();
        if let Err(error) = launched { self.synchronize()?; return Err(error); }
        if cold && !self.window.is_windowed() && crate::families::deepseek_v41::graph_policy::captures_shape(gathered.rows as u32)
            && self.graphs.capture_allowed(gathered.rows)
            && !crate::shared::decode_graph::census::eager("layer_row") {
            // The cold shape already ran eagerly above: record the graph for
            // later calls without a second launch (hunt C #3). Only when the
            // insert retires another graph must its queued launches drain
            // first (a rare new large shape), so the host waits only then.
            if self.graphs.would_retire(gathered.rows) { self.storage.stream.wait().await?; }
            if let Err(error) = unsafe { self.capture_ready(gathered.rows) } { self.synchronize()?; return Err(error); }
        }
        let copied = unsafe { self.weights.library.copy_d2d_async(residual, self.storage.output.buffer,
            gathered.rows * 40960, self.storage.stream.raw) };
        if let Err(error) = copied { self.synchronize()?; return Err(error); }
        unsafe { crate::shared::memory::chain::finish_cooperative(&self.storage.stream).await?; }
        self.ready_rows = Some(gathered.rows);
        Ok(())
    }
    #[cfg(test)]
    pub async unsafe fn check_cooperative(&mut self, residual: CuteafdDeviceBuffer,
        gathered: &EngramDeviceView) -> Result<()> {
        use std::{future::Future, task::Poll};
        let lib = self.weights.library;
        let mut residual = residual; residual.bytes = gathered.rows * 40960;
        let read = |buffer: CuteafdDeviceBuffer| -> Result<Vec<u8>> {
            let mut bytes = vec![0; buffer.bytes]; lib.copy_d2h(&mut bytes, buffer)?; Ok(bytes)
        };
        let original = read(residual)?;
        let expected = read(unsafe { self.execute_captured(residual, gathered)? })?;
        self.clear_graph()?;
        let cancelled = {
            let mut work = std::pin::pin!(unsafe { self.execute_into_cooperative(residual, gathered) });
            std::future::poll_fn(|cx| Poll::Ready(work.as_mut().poll(cx).is_pending())).await
        };
        if cancelled { assert!(self.output().is_err()); }
        for _ in 0..2 {
            lib.copy_h2d(residual, &original)?;
            unsafe { self.execute_into_cooperative(residual, gathered).await?; }
            assert_eq!(read(residual)?, expected);
        }
        lib.copy_h2d(residual, &original)?;
        eprintln!("PASS queued Engram gate {}: exact gate/residual parity, pending cancellation={cancelled}, reuse", self.layer());
        Ok(())
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.ready_rows = None;
        self.synchronize()?;
        // SAFETY: explicit synchronization drains all queued graph/storage uses.
        unsafe { self.graphs.clear() }
    }
    /// Borrowed BF16 residual; never free or retain across reuse/drop.
    pub fn output(&self) -> Result<CuteafdDeviceBuffer> {
        let rows = self
            .ready_rows
            .context("engram gate output is not complete")?;
        let mut output = self.storage.output.buffer;
        output.bytes = rows * 4 * 5120 * 2;
        Ok(output)
    }
    fn synchronize(&self) -> Result<()> {
        unsafe {
            self.weights
                .library
                .cuda_stream_synchronize(self.storage.stream.raw)
        }
    }
}
impl Drop for EngramGate<'_, '_> {
    fn drop(&mut self) {
        if let Err(error) = self.synchronize() {
            crate::shared::decode_graph::fatal_drain(Err(error), "Engram gate");
        }
        // SAFETY: destruction follows the containing stream's drain.
        if let Err(error) = unsafe { self.graphs.clear() } {
            tracing::error!(%error, "destroying native engram graphs");
        }
    }
}

#[cfg(test)]
impl EngramLayerWeights<'_> {
    pub(crate) fn visit(&self, prefix: &str, f: &mut dyn FnMut(String, cuteafd_ffi::CuteafdDeviceBuffer)) {
        self.tensors.visit(prefix, f);
        f(format!("{prefix}packed_scale"), self.packed_scales.buffer);
    }
}
