//! Real backbone shared experts, with immutable weights and exclusive captured waves.
#[path = "v41_backbone_shared/tp2.rs"]
pub(crate) mod tp2;
use crate::families::deepseek_v41::v41_attention_binding::QueryBinding;
use crate::families::deepseek_v41::v41_block::FfnInput;
use crate::shared::decode_graph::{CacheAttached, LayerGraphs};
use crate::shared::memory::Tag;
use crate::shared::memory::{DeviceAllocation, LoadStream};
use crate::families::deepseek_v41::v41_shared_ffn::{SharedFfn, SharedFfnStorage};
use crate::families::deepseek_v41::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::OfficialV41Catalog;
use std::marker::PhantomData;
use std::rc::Rc;
use crate::shared::memory::row_window::RowWindow;
#[cfg(test)]
thread_local! {
    static PRIVATE_STREAM: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Input/result are reserve planes: the Spark reduce reads the result in a
/// later hook. Inner SwiGLU/GEMM scratch is unit-cap plain scratch, serialized
/// on this storage's one stream. Weights, phases and graphs remain per slot.
pub(crate) struct BackboneSharedStorage<'a> {
    stream: LoadStream<'a>,
    inner: Rc<SharedFfnStorage<'a>>,
    input: DeviceAllocation<'a>,
    result: DeviceAllocation<'a>,
    rows: u32,
    unit: u32,
}
impl<'a> BackboneSharedStorage<'a> {
    pub(crate) fn storage_bytes(library: &NativeLibrary, rows: u32, unit: u32) -> Result<usize> {
        ensure!(unit > 0 && unit <= 4096 && unit <= rows
            && rows as usize <= crate::shared::memory::row_window::MAX_RESERVE_ROWS,
            "invalid shared expert storage rows/unit");
        Ok(SharedFfnStorage::storage_bytes(library, unit)? + Self::plane_sizes(rows).into_iter().sum::<usize>())
    }
    fn plane_sizes(rows: u32) -> [usize; 2] { [rows as usize * 10240; 2] }
    pub(crate) fn new(library: &'a NativeLibrary, rows: u32, unit: u32) -> Result<Rc<Self>> {
        Self::storage_bytes(library, rows, unit)?;
        let [input, result] = Self::plane_sizes(rows);
        Ok(Rc::new(Self {
            stream: LoadStream { library, raw: library.cuda_stream_create()? },
            inner: SharedFfnStorage::new(library, unit)?,
            input: DeviceAllocation::new(library, input, Tag::workspace("v41/backbone-shared"))?,
            result: DeviceAllocation::new(library, result, Tag::workspace("v41/backbone-shared"))?, rows, unit,
        }))
    }
}
impl Drop for BackboneSharedStorage<'_> {
    fn drop(&mut self) {
        // SAFETY: drain the storage's stream before freeing scratch and planes.
        crate::shared::decode_graph::fatal_drain(unsafe {
            self.stream.library.cuda_stream_synchronize(self.stream.raw)
        }, "shared expert storage");
    }
}

pub(crate) struct BackboneSharedWeights<'a> {
    library: &'a NativeLibrary,
    layer: usize,
    tensors: NativeRtxTensors<'a>,
    scales: Vec<DeviceAllocation<'a>>,
}
impl<'a> BackboneSharedWeights<'a> {
    pub(crate) fn inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>) -> Result<()> {
        self.tensors.inventory(inventory)?;
        let names = Self::names(self.layer)?;
        for (index, scales) in self.scales.iter().enumerate() {
            scales.inventory(inventory, &super::weight_layout::packed_scale_name(&names[index * 2]))?;
        }
        Ok(())
    }
    pub(super) fn names(layer: usize) -> Result<Vec<String>> {
        ensure!(layer < 40, "invalid backbone shared FFN layer");
        Ok(["w1", "w3", "w2"]
            .into_iter()
            .flat_map(|n| {
                ["weight", "scale"].map(|s| format!("layers.{layer}.ffn.shared_experts.{n}.{s}"))
            })
            .collect())
    }
    pub fn device_bytes(
        library: &NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
    ) -> Result<usize> {
        Ok(NativeRtxTensors::plan(catalog, &Self::names(layer)?)?
            + (2 * library
                .v41_fp8_matrix_info(1, 5120, 2304)?
                .packed_weight_scale_bytes
                + library
                    .v41_fp8_matrix_info(1, 2304, 5120)?
                    .packed_weight_scale_bytes) as usize)
    }
    #[cfg(test)]
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        layer: usize,
        budget: usize,
        staging: usize,
    ) -> Result<Self> {
        Self::load_in(library, catalog, layer, budget, staging, None)
    }
    pub fn load_in(library: &'a NativeLibrary, catalog: &OfficialV41Catalog, layer: usize,
        budget: usize, staging: usize, arena: Option<&crate::shared::memory::weight_arena::DeviceWeightArena<'a>>) -> Result<Self> {
        ensure!(
            Self::device_bytes(library, catalog, layer)? <= budget,
            "backbone shared weights exceed budget"
        );
        let names = Self::names(layer)?;
        let tensors = NativeRtxTensors::load_in(library, catalog, &names, budget, staging, arena)?;
        let mut scales = Vec::with_capacity(3);
        let stream = LoadStream {
            library,
            raw: library.cuda_stream_create()?,
        };
        for (i, (k, n)) in [(5120, 2304), (5120, 2304), (2304, 5120)]
            .into_iter()
            .enumerate()
        {
            let kernel = library.v41_fp8_matrix_kernel(1, k, n)?;
            scales.push(super::weight_layout::allocation(library, arena,
                &super::weight_layout::packed_scale_name(&names[2 * i]),
                kernel.info().packed_weight_scale_bytes as usize)?);
            unsafe {
                crate::shared::experts::rtx::dense::v41_pack_scales(&kernel, 
                    tensors.get(&names[2 * i + 1])?,
                    scales[i].buffer,
                    stream.raw,
                )?;
            }
        }
        unsafe {
            library.cuda_stream_synchronize(stream.raw)?;
        }
        Ok(Self {
            library,
            layer,
            tensors,
            scales,
        })
    }
    /// The buffers a decode step reads, in order (L2 prefetch): w1, w3 (gate
    /// and up) and w2 (down), each followed by its packed scales.
    pub(crate) fn decode_reads(&self) -> Vec<CuteafdDeviceBuffer> {
        ["w1", "w3", "w2"].into_iter().zip(&self.scales).flat_map(|(name, scales)| [
            self.tensors.get(&format!("layers.{}.ffn.shared_experts.{name}.weight", self.layer)).ok(),
            Some(scales.buffer),
        ]).flatten().collect()
    }
    pub fn wave(&self, capacity: u32, budget: usize) -> Result<BackboneSharedWave<'_, 'a>> {
        ensure!(
            BackboneSharedWave::device_bytes(self.library, capacity)? <= budget,
            "backbone shared wave exceeds budget"
        );
        self.on_storage(BackboneSharedStorage::new(self.library, capacity, capacity)?)
    }
    pub(crate) fn on_storage(&self, storage: Rc<BackboneSharedStorage<'a>>) -> Result<BackboneSharedWave<'_, 'a>> {
        ensure!(std::ptr::eq(storage.stream.library, self.library)
            && storage.input.buffer.device_id == self.library.cuda_get_device()?, "shared expert storage device/library differs");
        let capacity = storage.unit;
        Ok(BackboneSharedWave {
            inner: SharedFfn::on_storage(self.library, &self.tensors,
                &format!("layers.{}.ffn.shared_experts", self.layer), &self.scales, storage.inner.clone())?,
            #[cfg(test)]
            private_stream: if PRIVATE_STREAM.with(|flag| flag.get()) {
                Some(LoadStream { library: self.library, raw: self.library.cuda_stream_create()? })
            } else { None },
            storage, window: RowWindow::NONE,
            layer: self.layer,
            capacity,
            graphs: LayerGraphs::new(self.library, cuteafd_core::V41_VERIFY_ROWS as u32, "deepseek_v41"),
            weights: self,
            ready: None,
            origin: None,
        })
    }
}
pub(crate) struct SharedOutput<'a> {
    pub layer: usize,
    pub rows: u32,
    pub input: CuteafdDeviceBuffer,
    pub intermediates: [CuteafdDeviceBuffer; 3],
    pub values: CuteafdDeviceBuffer,
    origin: Option<QueryBinding>,
    _owner: PhantomData<&'a ()>,
}
impl SharedOutput<'_> {
    pub fn binding(&self) -> Result<QueryBinding> {
        self.origin.context("shared FFN output has no block origin")
    }
}
pub(crate) struct BackboneSharedWave<'w, 'a> {
    storage: Rc<BackboneSharedStorage<'a>>,
    #[cfg(test)]
    private_stream: Option<LoadStream<'a>>,
    window: RowWindow,
    inner: SharedFfn<'w, 'a>,
    layer: usize,
    capacity: u32,
    graphs: LayerGraphs<'w, 'a, BackboneSharedWeights<'a>>,
    weights: &'w BackboneSharedWeights<'a>,
    ready: Option<u32>,
    origin: Option<QueryBinding>,
}
impl<'w, 'a> BackboneSharedWave<'w, 'a> {
    pub(crate) fn visit_graph_banks(&mut self, f: &mut dyn FnMut(i32, &'static str, bool, &mut dyn CacheAttached) -> Result<()>) -> Result<()> {
        let device = self.input().device_id;
        f(device, "v41.backbone.shared", false, &mut self.graphs)
    }

    /// Reuse this lane's shared-expert storage with another backbone layer.
    pub fn rebind(&mut self, weights: &'w BackboneSharedWeights<'a>) -> Result<()> {
        self.invalidate();
        ensure!(
            std::ptr::eq(self.stream().library, weights.library),
            "shared FFN rebound weight library differs"
        );
        // Chained shared work is ordered before the reduction that consumes it;
        // later enqueues with the new weights follow it on this stream.
        if !crate::shared::memory::chain::active() { self.stream().require_complete()?; }
        // LayerGraphs retains the full owner, including packed scales, for
        // every cached graph. Only this drained stream consumes the workspace.
        unsafe {
            self.inner.rebind(
                &weights.tensors,
                &format!("layers.{}.ffn.shared_experts", weights.layer),
                &weights.scales,
            )?;
        }
        self.weights = weights;
        self.layer = weights.layer;
        Ok(())
    }
}
impl<'a> BackboneSharedWave<'_, 'a> {
    fn stream(&self) -> &LoadStream<'a> {
        #[cfg(test)]
        if let Some(stream) = &self.private_stream { return stream; }
        &self.storage.stream
    }

    pub(crate) fn select_window(&mut self, window: RowWindow) -> Result<()> {
        if let Some(range) = window.range() {
            ensure!(range.start < range.end && range.end <= self.storage.rows as usize
                && range.len() <= self.capacity as usize, "shared expert window outside storage/unit");
        }
        self.invalidate(); self.window = window; Ok(())
    }

    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        BackboneSharedStorage::storage_bytes(library, capacity, capacity)
    }
    pub fn input(&self) -> CuteafdDeviceBuffer {
        self.window.apply(self.storage.input.buffer, 10240)
    }
    fn result(&self) -> CuteafdDeviceBuffer { self.window.apply(self.storage.result.buffer, 10240) }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream().library.cuda_stream_synchronize(self.stream().raw) }
    }
    fn invalidate(&mut self) {
        self.ready = None;
        self.origin = None;
    }
    fn validate(&mut self, rows: u32) -> Result<()> {
        self.invalidate();
        ensure!(
            rows > 0 && rows <= self.capacity && self.window.admits(rows as usize),
            "shared FFN rows exceed capacity"
        );
        Ok(())
    }
    unsafe fn enqueue(&mut self, rows: u32) -> Result<()> {
        unsafe {
            self.inner
                .enqueue(self.input(), self.result(), rows, self.stream().raw)
        }
    }
    /// # Safety
    /// Input is finite, initialized and exclusively owned until the call drains.
    pub unsafe fn execute(&mut self, rows: u32) -> Result<SharedOutput<'_>> {
        self.validate(rows)?;
        let launched = unsafe { self.enqueue(rows) };
        launched.and(self.synchronize())?;
        self.ready = Some(rows);
        self.output()
    }
    /// # Safety
    /// Same contract as execute. Warmup is drained before capture.
    pub unsafe fn capture(&mut self, rows: u32) -> Result<()> {
        self.invalidate();
        ensure!(
            self.graphs.get_shape(self.layer, self.weights, rows).is_none(),
            "shared FFN graph already captured"
        );
        unsafe {
            self.execute(rows)?;
        }
        unsafe { self.capture_ready(rows) }
    }
    // Warmup has completed. Capture never yields with the stream in capture mode.
    unsafe fn capture_ready(&mut self, rows: u32) -> Result<()> {
        let Some(ticket) = self.graphs.cache_ticket(&(self.layer, rows), super::graph_policy::pinned_shape(rows), None) else {
            super::graph_policy::note_miss(format!("backbone_shared rows={rows} layer={}", self.layer), || "graph cache selected eager".into());
            return Ok(());
        };
        self.invalidate();
        unsafe {
            self.stream()
                .library
                .cuda_graph_begin_capture(self.stream().raw)?;
        }
        let launched = unsafe { self.enqueue(rows) };
        let captured = unsafe { self.stream().library.cuda_graph_end_capture_topology(self.stream().raw) };
        match (launched, captured) {
            (Ok(()), Ok((graph, nodes, width, spans, _))) => {
                if let Err(error) =
                    unsafe { self.graphs.cache_insert(self.layer, self.weights, rows, graph, ticket, nodes as u64, width, spans) }
                {
                    unsafe {
                        self.stream().library.cuda_graph_exec_destroy(graph)?;
                    }
                    return Err(error);
                }
                Ok(())
            }
            (Err(e), Ok((graph, _, _, _, _))) => {
                unsafe {
                    self.stream().library.cuda_graph_exec_destroy(graph)?;
                }
                Err(e)
            }
            (Err(e), Err(_)) | (Ok(()), Err(e)) => Err(e),
        }
    }
    /// # Safety
    /// Same contract as execute; graph live row count must match.
    pub unsafe fn replay(&mut self, rows: u32) -> Result<SharedOutput<'_>> {
        self.validate(rows)?;
        let Some((graph, count)) = self.graphs.cached_launch(self.layer, self.weights, rows) else {
            return unsafe { self.execute(rows) };
        };
        ensure!(count == rows, "shared FFN captured rows differ");
        let launched = unsafe {
            self.stream()
                .library
                .cuda_graph_launch(graph, self.stream().raw)
        };
        launched.and(unsafe { crate::shared::memory::chain::finish(self.stream().library, self.stream().raw) })?;
        self.ready = Some(rows);
        self.output()
    }
    /// # Safety
    /// The completed block input stays immutable through the copy. This wave has
    /// exclusive storage; returned values are only the shared contribution.
    pub unsafe fn execute_ffn(&mut self, input: &FfnInput<'_>) -> Result<SharedOutput<'_>> {
        self.invalidate();
        ensure!(
            input.layer == self.layer
                && input.binding().layer() == self.layer
                && !input.tokens.is_empty()
                && input.tokens.len() <= self.capacity as usize
                && input.values.bytes == input.tokens.len() * 10240
                && input.values.device_id == self.input().device_id,
            "shared FFN block input differs"
        );
        // Order the input copy on this stream after the chained attention output
        // (a legacy-stream copy would not order with the non-blocking streams).
        unsafe {
            crate::shared::memory::chain::join(self.stream().library, self.stream().raw)?;
            if let Err(error) = self.stream().library.copy_d2d_async(self.input(), input.values,
                input.values.bytes, self.stream().raw) { self.synchronize()?; return Err(error); }
        }
        let rows = input.tokens.len() as u32;
        if self.window.is_windowed() || !super::graph_policy::captures_shape(rows) {
            unsafe { self.execute(rows)?; }
            self.origin = Some(input.binding());
            return self.output();
        }
        if self
            .graphs
            .get_shape(self.layer, self.weights, rows)
            .is_none()
        {
            unsafe {
                self.capture(rows)?;
            }
        }
        unsafe {
            self.replay(rows)?;
        }
        self.origin = Some(input.binding());
        self.output()
    }
    /// # Safety
    /// Same input contract as execute_ffn; the future retains this wave and input
    /// until completion. Cancellation drains before their storage can be reused.
    pub async unsafe fn execute_ffn_cooperative(&mut self, input: &FfnInput<'_>) -> Result<SharedOutput<'_>> {
        self.invalidate();
        ensure!(
            input.layer == self.layer
                && input.binding().layer() == self.layer
                && !input.tokens.is_empty()
                && input.tokens.len() <= self.capacity as usize
                && input.values.bytes == input.tokens.len() * 10240
                && input.values.device_id == self.input().device_id,
            "shared FFN block input differs"
        );
        let rows = input.tokens.len() as u32;
        let cold = self.graphs.get_shape(self.layer, self.weights, rows).is_none();
        let launched = (|| unsafe {
            crate::shared::memory::chain::join(self.stream().library, self.stream().raw)?;
            self.stream().library.copy_d2d_async(self.input(), input.values, input.values.bytes, self.stream().raw)?;
            let graph = if self.window.is_windowed() { None } else { self.graphs.cached_launch(self.layer, self.weights, rows).map(|(raw, _)| raw) };
            let library = self.stream().library; let stream = self.stream().raw;
            crate::shared::decode_graph::census::dispatch(library, stream, c"layer_row", rows as usize,
                graph, || self.enqueue(rows))
        })();
        if let Err(error) = launched { self.synchronize()?; return Err(error); }
        unsafe { crate::shared::memory::chain::finish_cooperative(&self.stream()).await?; }
        if cold && !self.window.is_windowed() && super::graph_policy::captures_shape(rows)
            && !crate::shared::decode_graph::census::eager("layer_row") {
            // The eager execution above already completed these inputs. Capture
            // records future launches without executing them; publish that result
            // instead of running the same work again on every cache miss.
            unsafe { self.capture_ready(rows)?; }
        }
        self.ready = Some(rows);
        self.origin = Some(input.binding());
        self.output()
    }
    pub fn output(&self) -> Result<SharedOutput<'_>> {
        let rows = self.ready.context("shared FFN output unpublished")?;
        let mut input = self.input();
        input.bytes = rows as usize * 10240;
        let mut values = self.result();
        values.bytes = input.bytes;
        let intermediates = self.inner.intermediates().map(|mut b| {
            b.bytes = rows as usize * 4608;
            b
        });
        Ok(SharedOutput {
            layer: self.layer,
            rows,
            input,
            intermediates,
            values,
            origin: self.origin,
            _owner: PhantomData,
        })
    }
    /// Evict all shapes for the current layer; other layers remain cached.
    pub fn clear_graph(&mut self) -> Result<()> {
        self.invalidate();
        self.synchronize()?;
        unsafe {
            self.graphs.remove(self.layer)?;
        }
        Ok(())
    }
}
impl Drop for BackboneSharedWave<'_, '_> {
    fn drop(&mut self) {
        if let Err(e) = self
            .synchronize()
            .and_then(|()| unsafe { self.graphs.clear() })
        {
            crate::shared::decode_graph::fatal_drain(Err(e), "shared FFN");
        }
    }
}

#[cfg(test)]
impl BackboneSharedWeights<'_> {
    pub(crate) fn visit(&self, prefix: &str, f: &mut dyn FnMut(String, cuteafd_ffi::CuteafdDeviceBuffer)) {
        self.tensors.visit(prefix, f);
        for (i, s) in self.scales.iter().enumerate() { f(format!("{prefix}packed_scale{i}"), s.buffer); }
    }
}

#[cfg(test)]
mod storage_bytes_tests {
    use super::*;
    #[test]
    fn shared_expert_planes_equal_planned_allocations() {
        assert_eq!(BackboneSharedStorage::plane_sizes(8192).into_iter().sum::<usize>(), 8192 * 20480);
    }
}

#[cfg(test)]
mod shared_storage_tests {
    use super::*;
    use crate::shared::memory::chain::StageChain;
    fn run(ordered: bool) -> Result<()> {
        let lib = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        lib.cuda_set_device(0)?;
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(&std::env::var("CUTEAFD_SNAPSHOT")?))?;
        let weights = BackboneSharedWeights::load(&lib, &catalog, 2, usize::MAX, 1 << 20)?;
        let (rows, unit, reserve) = (13usize, 16u32, 48u32);
        let inputs = [0, 1].map(|seed| -> Result<_> {
            let hidden: Vec<u8> = (0..rows * 5120).flat_map(|i| {
                let f = ((i * 7 + seed * 11) % 29) as f32 / 32.0 - 0.4;
                ((f.to_bits() >> 16) as u16).to_ne_bytes()
            }).collect();
            let source = DeviceAllocation::new(&lib, hidden.len(), crate::shared::memory::TEST)?;
            lib.copy_h2d(source.buffer, &hidden)?; Ok(source)
        }).into_iter().collect::<Result<Vec<_>>>()?;
        let read = |buffer| -> Result<Vec<u8>> {
            let mut bytes = vec![0; rows * 10240]; lib.copy_d2h(&mut bytes, buffer)?; Ok(bytes)
        };
        let mut expected = Vec::new();
        for source in &inputs {
            let mut wave = weights.wave(unit, usize::MAX)?;
            unsafe { lib.copy_d2d_async(wave.input(), source.buffer, source.buffer.bytes, wave.stream().raw)?; wave.execute(rows as u32)?; }
            expected.push(read(wave.result())?);
        }
        assert_ne!(expected[0], expected[1]);
        let storage = BackboneSharedStorage::new(&lib, reserve, unit)?;
        PRIVATE_STREAM.with(|flag| flag.set(!ordered));
        let built = (|| -> Result<_> { Ok([weights.on_storage(storage.clone())?, weights.on_storage(storage.clone())?]) })();
        PRIVATE_STREAM.with(|flag| flag.set(false));
        let mut slots = built?;
        for (i, slot) in slots.iter_mut().enumerate() {
            slot.select_window(RowWindow::within(i * unit as usize..i * unit as usize + rows, reserve as usize)?)?;
            unsafe { lib.copy_d2d_async(slot.input(), inputs[i].buffer, rows * 10240, slot.stream().raw)?; slot.enqueue(rows as u32)?; }
            slot.synchronize()?;
        }
        let chains = [StageChain::new(&lib)?, StageChain::new(&lib)?];
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        let mut gate = None;
        for i in 0..2 {
            runtime.block_on(chains[i].handle().scope(async {
                let slot = &slots[i]; let stream = slot.stream().raw;
                // SAFETY: source, scratch and row planes outlive both chains.
                unsafe {
                    crate::shared::memory::chain::join(&lib, stream)?;
                    lib.copy_d2d_async(slot.input(), inputs[i].buffer, rows * 10240, stream)?;
                    slot.inner.enqueue_up(slot.input(), rows as u32, stream)?;
                    if i == 0 { gate = Some(cuteafd_ffi::test_support::CudaStreamGate::new(&lib, stream)?); }
                    slot.inner.enqueue_down(slot.result(), rows as u32, stream)?;
                    crate::shared::memory::chain::finish(&lib, stream)
                }
            }))?;
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(if ordered { 2 } else { 10 });
        let mut done = false;
        while std::time::Instant::now() < deadline {
            if unsafe { lib.cuda_stream_query(slots[1].stream().raw)? } { done = true; break; }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(done, !ordered, "shared FFN B completion before gate release");
        drop(gate);
        for chain in &chains { chain.drain()?; }
        for (i, slot) in slots.iter().enumerate() {
            slot.synchronize()?;
            let got = read(slot.result())?;
            if ordered || i == 1 { assert_eq!(got, expected[i]); }
            else { assert_ne!(got, expected[i], "negative control failed to detect scratch clobber"); }
        }
        eprintln!("PASS shared FFN storage ordered={ordered}, B_completed_before_gate={done}");
        Ok(())
    }
    #[test]
    #[ignore = "requires snapshot, native library and CUDA"]
    fn shared_expert_and_swiglu_storage_match_private_runs_behind_gate() -> Result<()> {
        for _ in 0..3 { run(true)?; } Ok(())
    }
    #[test]
    #[ignore = "requires snapshot, native library and CUDA"]
    fn shared_expert_private_streams_clobber_gated_swiglu_scratch() -> Result<()> {
        for _ in 0..3 { run(false)?; } Ok(())
    }
}
