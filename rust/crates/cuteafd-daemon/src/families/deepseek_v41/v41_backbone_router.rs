//! Official backbone routing with block-bound inputs and canonical TP4 requests.
use crate::families::deepseek_v41::v41_attention_binding::QueryBinding;
use crate::families::deepseek_v41::v41_block::FfnInput;
use crate::shared::decode_graph::LayerGraphs;
use crate::shared::memory::Tag;
use crate::shared::memory::{DeviceAllocation, HostAllocation, LoadStream};
use crate::families::deepseek_v41::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary, V41ExpertInputQuantizer, V41Router};
use cuteafd_loader::OfficialV41Catalog;
use std::marker::PhantomData;
use std::rc::Rc;
use crate::shared::memory::row_window::RowWindow;
#[cfg(test)]
thread_local! {
    static PRIVATE_ALIAS: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
const ROW_BYTES: [usize; 6] = [10240, 1, 1536, 24, 24, 5280];

unsafe fn record_v41_routes(usage: Option<&cuteafd_ffi::expert_usage::ExpertUsage<'_>>,
    device: i32, layer: usize, ids: *const std::ffi::c_void, rows: usize,
    phase: cuteafd_ffi::expert_usage::UsagePhase, stream: *mut std::ffi::c_void) -> Result<()> {
    let Some(usage) = usage else { return Ok(()); };
    let tally = usage.routes(device, layer, phase)?;
    // Count the live rows in one launch outside the router's phase-less graph.
    // SAFETY: the caller retains rows * 6 ids on the router's producer stream.
    unsafe { tally.record(ids, rows, None, 511, true, stream) }
}

/// Reserve planes: hidden, mask, scores, ids, routing, wire. These outputs
/// may survive hooks or be read on fork streams. Pinned staging and the ring
/// remain per-slot: host reads its [40][unit][6] ring only after its stream
/// drains, so another slot's queued D2H can never write the read allocation.
pub(crate) struct BackboneRouterStorage<'a> {
    stream: LoadStream<'a>,
    buffers: Vec<DeviceAllocation<'a>>,
    rows: u32,
    unit: u32,
}
impl<'a> BackboneRouterStorage<'a> {
    pub(crate) fn storage_bytes(rows: u32, unit: u32) -> Result<usize> {
        ensure!(unit > 0 && unit <= 4096 && unit <= rows
            && rows as usize <= crate::shared::memory::row_window::MAX_RESERVE_ROWS,
            "invalid router storage rows/unit");
        Ok(Self::allocation_sizes(rows).into_iter().sum())
    }
    fn allocation_sizes(rows: u32) -> [usize; 6] { ROW_BYTES.map(|n| n * rows as usize) }
    pub(crate) fn new(library: &'a NativeLibrary, rows: u32, unit: u32) -> Result<Rc<Self>> {
        Self::storage_bytes(rows, unit)?;
        Ok(Rc::new(Self {
            stream: LoadStream { library, raw: library.cuda_stream_create()? },
            buffers: Self::allocation_sizes(rows).into_iter()
                .map(|n| DeviceAllocation::new(library, n, Tag::workspace("v41/backbone-router"))).collect::<Result<_>>()?,
            rows, unit,
        }))
    }
}
impl Drop for BackboneRouterStorage<'_> {
    fn drop(&mut self) {
        // SAFETY: this storage owns its stream; drain before freeing planes.
        crate::shared::decode_graph::fatal_drain(unsafe {
            self.stream.library.cuda_stream_synchronize(self.stream.raw)
        }, "router storage");
    }
}

pub(crate) struct BackboneRouterWeights<'a> {
    library: &'a NativeLibrary,
    layer: usize,
    tensors: NativeRtxTensors<'a>,
    nvfp4: bool,
}

fn request_hidden_format(nvfp4: bool) -> (cuteafd_transport::ExpertV2Dtype, usize) {
    if nvfp4 {
        (cuteafd_transport::ExpertV2Dtype::Bf16, 10240)
    } else {
        (cuteafd_transport::ExpertV2Dtype::Fp8E4m3Ue8m0K32, 5280)
    }
}
impl<'a> BackboneRouterWeights<'a> {
    pub(crate) fn inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>) -> Result<()> {
        self.tensors.inventory(inventory)
    }
    pub(super) fn names(layer: usize) -> Result<Vec<String>> {
        ensure!(layer < 40, "invalid backbone router layer");
        Ok(["weight", "bias", "bias_vl"]
            .map(|n| format!("layers.{layer}.ffn.gate.{n}"))
            .to_vec())
    }
    pub fn device_bytes(catalog: &OfficialV41Catalog, layer: usize) -> Result<usize> {
        let bytes = NativeRtxTensors::plan(catalog, &Self::names(layer)?)?;
        ensure!(
            bytes == 3_935_232,
            "unexpected backbone router weight geometry"
        );
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
        Self::load_in(library, catalog, layer, budget, staging, None)
    }
    pub fn load_in(library: &'a NativeLibrary, catalog: &OfficialV41Catalog, layer: usize,
        budget: usize, staging: usize, arena: Option<&crate::shared::memory::weight_arena::DeviceWeightArena<'a>>) -> Result<Self> {
        ensure!(
            Self::device_bytes(catalog, layer)? <= budget,
            "backbone router weights exceed budget"
        );
        Ok(Self {
            library,
            layer,
            nvfp4: catalog.nvfp4().is_some(),
            tensors: NativeRtxTensors::load_in(
                library, catalog, &Self::names(layer)?, budget, staging, arena,
            )?,
        })
    }

    /// The buffers a decode step reads (L2 prefetch): gate weight and biases.
    pub(crate) fn decode_reads(&self) -> Vec<CuteafdDeviceBuffer> {
        Self::names(self.layer).into_iter().flatten().filter_map(|name| self.tensors.get(&name).ok()).collect()
    }
    pub fn wave(&self, capacity: u32, budget: usize) -> Result<BackboneRouterWave<'_, 'a>> {
        ensure!(
            BackboneRouterWave::device_bytes(capacity)? <= budget,
            "backbone router wave exceeds budget"
        );
        self.on_storage(BackboneRouterStorage::new(self.library, capacity, capacity)?)
    }
    pub(crate) fn on_storage(&self, storage: Rc<BackboneRouterStorage<'a>>) -> Result<BackboneRouterWave<'_, 'a>> {
        ensure!(std::ptr::eq(storage.stream.library, self.library)
            && storage.buffers[0].buffer.device_id == self.library.cuda_get_device()?, "router storage device/library differs");
        let capacity = storage.unit;
        Ok(BackboneRouterWave {
            #[cfg(test)]
            private_stream: if PRIVATE_ALIAS.with(|flag| flag.get()) {
                Some(LoadStream { library: self.library, raw: self.library.cuda_stream_create()? })
            } else { None },
            storage,
            expert_usage: None,
            window: RowWindow::NONE,
            kernel: self.library.v41_router()?,
            input_quantizer: self.library.v41_expert_input_quantizer()?,
            // Decode/verification benefits from batching small DMA transfers.
            // Large prefill retains direct downloads to avoid another host copy.
            // Route ids (24 B per row); also the non-deferred mask staging.
            request_staging: HostAllocation::new(self.library, capacity as usize * 24, Tag::staging("v41/backbone-router"))?,
            mask_staging: HostAllocation::new(self.library, capacity as usize, Tag::staging("v41/backbone-router"))?,
            route_ring: Some(HostAllocation::new(self.library, 40 * capacity as usize * 24, Tag::staging("v41/backbone-router"))?),
            ring_capture: false,
            ring_pending: Vec::with_capacity(40),
            weights: self,
            tokens: Vec::new(),
            layer: self.layer,
            capacity,
            graphs: LayerGraphs::new(self.library, cuteafd_core::V41_VERIFY_ROWS as u32, "deepseek_v41"),
            other_graphs: LayerGraphs::new(self.library, cuteafd_core::V41_VERIFY_ROWS as u32, "deepseek_v41"),
            full_request: true,
            ready: None,
            origin: None,
            wait: std::cell::RefCell::new(crate::shared::memory::BlockingWait::new(self.library)),
        })
    }
}
pub(crate) struct RouterOutput<'a> {
    pub layer: usize,
    pub rows: u32,
    pub input: CuteafdDeviceBuffer,
    pub expert_input: CuteafdDeviceBuffer,
    request_nvfp4: bool,
    pub mask: CuteafdDeviceBuffer,
    pub scores: CuteafdDeviceBuffer,
    pub ids: CuteafdDeviceBuffer,
    pub routing: CuteafdDeviceBuffer,
    pub tokens: &'a [u64],
    request_staging: cuteafd_ffi::CuteafdHostBuffer,
    full_request: bool,
    origin: Option<QueryBinding>,
    /// The router's stream, on which these outputs complete (a remote layer's
    /// wave stages from it).
    pub stream: *mut std::ffi::c_void,
    _owner: PhantomData<&'a ()>,
}
impl RouterOutput<'_> {
    /// # Safety
    /// The completed router output stays live and unchanged through this copy.
    pub unsafe fn capture_route_ids(&self, library: &NativeLibrary, output: &mut Vec<[u32; 6]>) -> Result<()> {
        ensure!(self.rows > 0 && self.rows <= 4096 && self.ids.bytes == self.rows as usize * 24,
            "invalid route capture extent");
        output.resize(self.rows as usize, [0; 6]);
        // Arrays have contiguous u32 layout; every bit pattern is valid. The
        // vector is exclusively borrowed and fully initialized before copying.
        let bytes = unsafe { std::slice::from_raw_parts_mut(output.as_mut_ptr().cast::<u8>(), self.ids.bytes) };
        ensure!(bytes.len() <= self.request_staging.bytes, "route capture staging extent differs");
        // This execution already drained the graph's route-ID download.
        unsafe { std::ptr::copy_nonoverlapping(self.request_staging.ptr.cast::<u8>(), bytes.as_mut_ptr(), bytes.len()); }
        let _ = library;
        Ok(())
    }
    pub fn binding(&self) -> Result<QueryBinding> {
        self.origin
            .context("backbone router output has no block origin")
    }
}

#[cfg(test)]
mod reuse_tests {
    use super::*;
    use cuteafd_loader::read_official_v41_catalog;

    #[test]
    fn expert_request_wire_format_tracks_nvfp4_checkpoint() {
        use cuteafd_transport::ExpertV2Dtype;
        assert_eq!(request_hidden_format(false), (ExpertV2Dtype::Fp8E4m3Ue8m0K32, 5280));
        assert_eq!(request_hidden_format(true), (ExpertV2Dtype::Bf16, 10240));
        for nvfp4 in [false, true] {
            for rows in [1usize, 16, 80, 4096] {
                let (_, width) = request_hidden_format(nvfp4);
                let hidden = rows * width;
                let ids = rows * 24;
                let arena = vec![0xa5u8; rows * (width + 48)];
                let (h, routes) = arena.split_at(hidden);
                let (i, w) = routes.split_at(ids);
                assert_eq!((h.len(), i.len(), w.len()), (rows * width, rows * 24, rows * 24));
            }
        }
    }
    fn bytes(library: &NativeLibrary, buffer: CuteafdDeviceBuffer) -> Result<Vec<u8>> {
        let mut result = vec![0; buffer.bytes];
        library.copy_d2h(&mut result, buffer)?;
        Ok(result)
    }
    #[test]
    fn local_router_graphs_preserve_ids_without_remote_payload() -> Result<()> {
        let Some(path) = std::env::var_os("CUTEAFD_ROUTER_REUSE_LIBRARY") else {
            eprintln!("skip local router GPU test: CUTEAFD_ROUTER_REUSE_LIBRARY unset");
            return Ok(());
        };
        let model = std::env::var_os("CUTEAFD_ROUTER_REUSE_MODEL").context("CUTEAFD_ROUTER_REUSE_MODEL required")?;
        let library = unsafe { NativeLibrary::load(path)? };
        let catalog = read_official_v41_catalog(std::path::Path::new(&model))?;
        let weights = (0..2).map(|layer| BackboneRouterWeights::load(&library, &catalog, layer, 1 << 30, 16 << 20))
            .collect::<Result<Vec<_>>>()?;
        for rows in [1, 16, 80, 256] {
            let mut lane = weights[0].wave(rows, BackboneRouterWave::device_bytes(rows)?)?;
            let mut handles = std::collections::HashMap::new();
            for (cycle, local) in [false, true, false, true].into_iter().enumerate() {
                for layer in 0..2 {
                    lane.rebind(&weights[layer])?;
                    lane.set_local_mode(local)?;
                    let hidden: Vec<u8> = (0..rows as usize * 5120).flat_map(|i| {
                        let x = ((i + cycle * 3 + layer) % 17) as f32 / 32.;
                        ((x.to_bits() >> 16) as u16).to_ne_bytes()
                    }).collect();
                    let mask: Vec<u8> = (0..rows).map(|i| (i % 2) as u8).collect();
                    let [input, modality] = lane.inputs();
                    library.copy_h2d(input, &hidden)?;
                    library.copy_h2d(modality, &mask)?;
                    let mut fresh = weights[layer].wave(rows, BackboneRouterWave::device_bytes(rows)?)?;
                    let [fresh_input, fresh_mask] = fresh.inputs();
                    library.copy_h2d(fresh_input, &hidden)?;
                    library.copy_h2d(fresh_mask, &mask)?;
                    if local {
                        unsafe { std::ptr::write_bytes(lane.request_staging.buffer.ptr, 0xa5, lane.request_staging.buffer.bytes); }
                    }
                    let actual = unsafe { lane.execute_captured(rows)? };
                    let reference = unsafe { fresh.execute(rows)? };
                    if local {
                        let staging = unsafe { std::slice::from_raw_parts(actual.request_staging.ptr.cast::<u8>(), actual.request_staging.bytes) };
                        assert!(staging[rows as usize * 24..].iter().all(|&b| b == 0xa5));
                    }
                    for (a,b) in [(actual.ids,reference.ids),(actual.routing,reference.routing),
                                  (actual.expert_input,reference.expert_input)] {
                        assert_eq!(bytes(&library,a)?,bytes(&library,b)?);
                    }
                    let mut ids = vec![[u32::MAX;6]; rows as usize + 5];
                    unsafe { actual.capture_route_ids(&library, &mut ids)?; }
                    assert_eq!(ids.len(),rows as usize);
                    let captured: Vec<u8> = ids.iter().flatten().flat_map(|v| v.to_ne_bytes()).collect();
                    assert_eq!(captured,bytes(&library,reference.ids)?);
                    assert_eq!(unsafe { actual.download_request(&library) }.is_err(),local);
                    let handle = lane.graphs.get(layer,&weights[layer]).unwrap().0;
                    if let Some(previous) = handles.insert((local,layer),handle) { assert_eq!(previous,handle); }
                    if let Some(other) = handles.get(&(!local,layer)) { assert_ne!(*other,handle); }
                }
            }
        }
        eprintln!("PASS 32 local/remote router graph transitions with exact IDs, routes and FP8 input");
        Ok(())
    }

    #[test]
    fn real_router_rebinding_matches_fresh_owners() -> Result<()> {
        let Some(path) = std::env::var_os("CUTEAFD_ROUTER_REUSE_LIBRARY") else {
            eprintln!("skip router reuse GPU test: CUTEAFD_ROUTER_REUSE_LIBRARY unset");
            return Ok(());
        };
        let model = std::env::var_os("CUTEAFD_ROUTER_REUSE_MODEL")
            .context("CUTEAFD_ROUTER_REUSE_MODEL required")?;
        let library = unsafe { NativeLibrary::load(path)? };
        let catalog =
            read_official_v41_catalog(std::path::Path::new(&model))?;
        let weights = (0..40)
            .map(|layer| {
                BackboneRouterWeights::load(&library, &catalog, layer, 3_935_232, 1024 * 1024)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut comparisons = 0;
        for rows in [1u32, 6, 16, 80, 81, 4096] {
            let layers: Vec<usize> = if rows > 80 {
                vec![0, 20, 39]
            } else {
                (0..40).collect()
            };
            let mut lane = weights[0].wave(rows, BackboneRouterWave::device_bytes(rows)?)?;
            let pointers = lane.inputs().map(|b| b.ptr);
            let mut handles = std::collections::HashMap::new();
            for cycle in 0..2usize {
                for &layer in &layers {
                    lane.rebind(&weights[layer])?;
                    assert!(lane.output().is_err());
                    assert_eq!(lane.inputs().map(|b| b.ptr), pointers);
                    let hidden: Vec<u8> = (0..rows as usize * 5120)
                        .flat_map(|i| {
                            let value =
                                (((i * 7 + cycle * 19 + layer * 3) % 31) as i32 - 15) as f32 / 128.;
                            ((value.to_bits() >> 16) as u16).to_ne_bytes()
                        })
                        .collect();
                    let mask: Vec<u8> = (0..rows as usize)
                        .map(|i| ((i + cycle + layer) % 2) as u8)
                        .collect();
                    let [input, modality] = lane.inputs();
                    library.copy_h2d(input, &hidden)?;
                    library.copy_h2d(modality, &mask)?;
                    let mut fresh =
                        weights[layer].wave(rows, BackboneRouterWave::device_bytes(rows)?)?;
                    let [fresh_input, fresh_mask] = fresh.inputs();
                    library.copy_d2d(fresh_input, input, input.bytes)?;
                    library.copy_d2d(fresh_mask, modality, modality.bytes)?;
                    // Poison both GPU inputs and the shared host arena before
                    // staging from an independent completed device input.
                    library.copy_h2d(input, &vec![0xff; input.bytes])?;
                    library.copy_h2d(modality, &vec![0xff; modality.bytes])?;
                    unsafe {
                        std::ptr::write_bytes(
                            lane.request_staging.buffer.ptr,
                            0xa5,
                            lane.request_staging.buffer.bytes,
                        );
                        lane.stage_inputs(fresh_input, &mask)?;
                    }
                    let actual = unsafe { lane.execute_captured(rows)? };
                    let reference = unsafe { fresh.execute(rows)? };
                    assert_eq!(actual.layer, layer);
                    for (a, b) in [
                        (actual.scores, reference.scores),
                        (actual.ids, reference.ids),
                        (actual.routing, reference.routing),
                        (actual.expert_input, reference.expert_input),
                    ] {
                        assert_eq!(bytes(&library, a)?, bytes(&library, b)?);
                    }
                    let (hidden, ids, routing) = unsafe { actual.download_request(&library)? };
                    assert_eq!(hidden, bytes(&library,
                        if reference.request_nvfp4 { reference.input } else { reference.expert_input })?);
                    assert_eq!(ids, bytes(&library, reference.ids)?);
                    assert_eq!(routing, bytes(&library, reference.routing)?);
                    for buffer in [actual.scores, actual.routing] {
                        assert!(bytes(&library, buffer)?
                            .chunks_exact(4)
                            .all(|b| f32::from_ne_bytes(b.try_into().unwrap()).is_finite()));
                    }
                    let handle = lane.graphs.get(layer, &weights[layer]).unwrap().0;
                    if cycle == 0 {
                        handles.insert(layer, handle);
                    } else {
                        assert_eq!(handles[&layer], handle);
                    }
                    comparisons += 1;
                }
                eprintln!("PASS real router rows={rows} cycle={cycle} layers={} scores/ids/routing/encoded-input exact; stable inputs and cached handles",layers.len());
            }
            assert!(unsafe { lane.execute_captured(0) }.is_err());
            assert!(lane.output().is_err());
            lane.rebind(&weights[0])?;
            unsafe {
                lane.execute_captured(rows)?;
            }
            lane.clear_graph()?;
            assert!(lane.output().is_err());
        }
        assert_eq!(comparisons, 332);
        eprintln!("PASS 332 real-weight router comparisons and invalid-row recovery");
        Ok(())
    }
}
pub(crate) struct BackboneRouterWave<'w, 'a> {
    expert_usage: Option<Rc<cuteafd_ffi::expert_usage::ExpertUsage<'a>>>,
    storage: Rc<BackboneRouterStorage<'a>>,
    #[cfg(test)]
    private_stream: Option<LoadStream<'a>>,
    window: RowWindow,
    kernel: V41Router<'a>,
    input_quantizer: V41ExpertInputQuantizer<'a>,
    weights: &'w BackboneRouterWeights<'a>,
    request_staging: HostAllocation<'a>,
    /// The modality mask's own pinned source under device-ordered passes, where
    /// the host stages the next layer while this layer's route download into
    /// the request arena may still be queued.
    mask_staging: HostAllocation<'a>,
    /// Device-ordered passes: each local layer's route ids, downloaded into
    /// its own slot (`[40][capacity][6]` u32) and read after the pass drains.
    route_ring: Option<HostAllocation<'a>>,
    ring_capture: bool,
    ring_pending: Vec<(usize, u32)>,
    tokens: Vec<u64>,
    layer: usize,
    capacity: u32,
    graphs: LayerGraphs<'w, 'a, BackboneRouterWeights<'a>>,
    other_graphs: LayerGraphs<'w, 'a, BackboneRouterWeights<'a>>,
    full_request: bool,
    ready: Option<u32>,
    origin: Option<QueryBinding>,
    /// The per-layer host wait (`--host-wait`: block spins briefly, then sleeps).
    wait: std::cell::RefCell<crate::shared::memory::BlockingWait<'a>>,
}
impl<'w, 'a> BackboneRouterWave<'w, 'a> {
    pub fn rebind(&mut self, weights: &'w BackboneRouterWeights<'a>) -> Result<()> {
        self.invalidate();
        ensure!(
            std::ptr::eq(self.stream().library, weights.library),
            "router rebound library differs"
        );
        ensure!(self.weights.nvfp4 == weights.nvfp4,
            "router cannot rebind across expert wire formats");
        for name in BackboneRouterWeights::names(weights.layer)? {
            ensure!(
                weights.tensors.get(&name)?.device_id == self.b(0).device_id,
                "router rebound weight device differs"
            );
        }
        if !crate::shared::memory::chain::active() { self.stream().require_complete()?; }
        self.weights = weights;
        self.layer = weights.layer;
        Ok(())
    }
}
impl<'a> BackboneRouterWave<'_, 'a> {
    fn stream(&self) -> &LoadStream<'a> {
        #[cfg(test)]
        if let Some(stream) = &self.private_stream { return stream; }
        &self.storage.stream
    }

    pub(crate) fn select_window(&mut self, window: RowWindow) -> Result<()> {
        crate::shared::memory::row_window::require_rest(!self.ring_pending.is_empty(), "router", "its route ring is undrained")?;
        if let Some(range) = window.range() {
            ensure!(range.start < range.end && range.end <= self.storage.rows as usize
                && range.len() <= self.capacity as usize, "router window outside storage/unit");
        }
        self.invalidate();
        self.window = window;
        Ok(())
    }

    /// Device-ordered passes keep captured local routes in the ring.
    pub fn set_ring_capture(&mut self, enabled: bool) {
        self.ring_capture = enabled && self.route_ring.is_some();
        self.ring_pending.clear();
    }
    /// Whether this wave captures this layer's routes into its ring (no host
    /// read now): every remote layer, and local layers of device-ordered passes.
    /// Whether this layer's routes are read on the device only, so the router
    /// need not complete on the host: remote layers (unless route-policy
    /// tracing reads them from the staging) and, in device-ordered passes,
    /// local layers. Requires an active chain (later stages join its head).
    fn device_read(&self) -> bool {
        let trace = tracing::enabled!(target: "cuteafd::route_policy", tracing::Level::DEBUG);
        crate::shared::memory::chain::active() && if self.full_request { self.ring_capture || !trace }
            else { crate::shared::memory::chain::deferred() }
    }
    pub fn ring_capture(&self) -> bool {
        self.ring_capture && (self.full_request || crate::shared::memory::chain::deferred())
    }
    /// After a failed or cancelled pass: its ring copies complete (the pinned
    /// ring is reused) and its captures are dropped, so the router can move
    /// to another window; a failed pass's routes describe nothing.
    pub fn discard_ring(&mut self) -> Result<()> {
        if self.ring_pending.is_empty() { return Ok(()); }
        self.synchronize()?;
        self.ring_pending.clear();
        Ok(())
    }
    /// After the pass drained: every ring-captured layer's routes into `output[layer]`.
    pub fn drain_ring(&mut self, output: &mut [Vec<[u32; 6]>]) -> Result<Vec<usize>> {
        self.synchronize()?;
        let ring = match &self.route_ring { Some(ring) => ring.bytes(), None => return Ok(Vec::new()) };
        let mut layers = Vec::with_capacity(self.ring_pending.len());
        for (layer, rows) in self.ring_pending.drain(..) {
            let slot = &ring[layer * self.capacity as usize * 24..][..rows as usize * 24];
            let rows_out = output.get_mut(layer).context("ring layer outside the capture")?;
            rows_out.clear();
            rows_out.extend(slot.chunks_exact(24).map(|row| std::array::from_fn(|i|
                u32::from_ne_bytes(row[4 * i..4 * i + 4].try_into().unwrap()))));
            layers.push(layer);
        }
        Ok(layers)
    }
    pub fn set_local_mode(&mut self, local: bool) -> Result<()> {
        if self.full_request == !local { return Ok(()); }
        self.stream().require_complete()?;
        self.invalidate();
        std::mem::swap(&mut self.graphs, &mut self.other_graphs);
        self.full_request = !local;
        Ok(())
    }
    pub fn device_bytes(capacity: u32) -> Result<usize> {
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid backbone router capacity"
        );
        BackboneRouterStorage::storage_bytes(capacity, capacity)
    }
    pub(crate) fn install_expert_usage(&mut self, usage: Option<Rc<cuteafd_ffi::expert_usage::ExpertUsage<'a>>>) {
        self.expert_usage = usage;
    }
    unsafe fn record_routes(&self, rows: u32, phase: Option<cuteafd_ffi::expert_usage::UsagePhase>) -> Result<()> {
        let (Some(usage), Some(phase)) = (&self.expert_usage, phase) else { return Ok(()); };
        let ids = self.b(3);
        // SAFETY: the wave owns the real router rows until this stream drains.
        unsafe { record_v41_routes(Some(usage), ids.device_id, self.layer, ids.ptr,
            rows as usize, phase, self.stream().raw) }
    }
    fn b(&self, i: usize) -> CuteafdDeviceBuffer {
        {
            #[cfg(test)]
            if self.private_stream.is_some() { return self.storage.buffers[i].buffer; }
            self.window.apply(self.storage.buffers[i].buffer, ROW_BYTES[i])
        }
    }
    pub fn inputs(&self) -> [CuteafdDeviceBuffer; 2] {
        [self.b(0), self.b(1)]
    }
    fn synchronize(&self) -> Result<()> {
        // SAFETY: the router's live stream, on the current device.
        unsafe { self.wait.borrow_mut().wait(self.stream().raw) }
    }
    fn invalidate(&mut self) {
        self.ready = None;
        self.origin = None;
        self.tokens.clear();
    }
    fn validate(&mut self, rows: u32) -> Result<()> {
        self.invalidate();
        ensure!(
            rows > 0 && rows <= self.capacity && self.window.admits(rows as usize),
            "backbone router rows exceed capacity"
        );
        Ok(())
    }
    unsafe fn enqueue(&mut self, rows: u32) -> Result<()> {
        let names = BackboneRouterWeights::names(self.layer)?;
        unsafe {
            self.kernel.launch(
                self.b(0),
                self.weights.tensors.get(&names[0])?,
                self.weights.tensors.get(&names[1])?,
                self.weights.tensors.get(&names[2])?,
                Some(self.b(1)),
                self.b(2),
                self.b(3),
                self.b(4),
                rows as usize,
                384,
                self.stream().raw,
            )?;
            // NVFP4 consumes BF16 directly; its FP8 buffer has no consumer.
            if !self.weights.nvfp4 {
                self.input_quantizer
                    .launch(self.b(0), self.b(5), rows, self.stream().raw)?;
            }
            // Route ids only (route capture, route-policy tracing). A remote
            // layer's wave stages its hidden rows, ids and weights from the
            // device buffers (`wave_sources`), so nothing else comes down (P4).
            let bytes = rows as usize * 24;
            ensure!(bytes <= self.request_staging.buffer.bytes, "route staging too small");
            let ids = std::slice::from_raw_parts_mut(self.request_staging.buffer.ptr.cast::<u8>(), bytes);
            self.stream().library.copy_d2h_async(ids, self.b(3), self.stream().raw)?;
            Ok(())
        }
    }
    /// # Safety
    /// Hidden input is finite and initialized; mask bytes are 0 or 1 in the same
    /// row order. Both inputs are exclusively owned until the call drains.
    pub unsafe fn execute(&mut self, rows: u32) -> Result<RouterOutput<'_>> {
        unsafe { self.execute_counted(rows, None) }
    }
    unsafe fn execute_counted(&mut self, rows: u32, phase: Option<cuteafd_ffi::expert_usage::UsagePhase>) -> Result<RouterOutput<'_>> {
        self.validate(rows)?;
        // The ring copy (if captured) completes with the host wait.
        let launched = unsafe { self.enqueue(rows) }.and_then(|()| unsafe { self.record_routes(rows, phase) })
            .and_then(|()| unsafe { self.ring_download(rows) });
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
            "backbone router graph already captured"
        );
        unsafe {
            self.execute(rows)?;
        }
        unsafe { self.capture_ready(rows) }
    }
    pub(crate) fn declare_startup(&mut self, keys: &[(usize, u32)]) -> Result<()> {
        self.stream().require_complete()?;
        self.graphs.declare_startup(keys)?;
        self.other_graphs.declare_startup(keys)
    }
    pub(crate) fn capture_startup(&mut self, rows: u32) -> Result<()> {
        self.validate(rows)?;
        self.stream().require_complete()?;
        // SAFETY: capture records stable owned buffers and borrowed weights;
        // no kernel executes and no routes are published during startup.
        let local = !self.full_request;
        let result = (|| -> Result<()> {
            // Runtime swaps banks when expert placement changes the handoff mode.
            for mode in [local, !local] {
                self.set_local_mode(mode)?;
                // SAFETY: both banks record the same stable owner-local enqueue.
                unsafe { self.capture_ready(rows)?; }
            }
            Ok(())
        })();
        let restored = self.set_local_mode(local);
        result.and(restored)
    }
    pub(crate) fn finish_startup(&mut self) -> Result<()> {
        self.graphs.finish_startup()?;
        self.other_graphs.finish_startup()
    }
    // Warmup has completed. Capture never yields with the stream in capture mode.
    unsafe fn capture_ready(&mut self, rows: u32) -> Result<()> {
        self.invalidate();
        unsafe {
            self.stream()
                .library
                .cuda_graph_begin_capture(self.stream().raw)?;
        }
        let launched = unsafe { self.enqueue(rows) };
        let captured = unsafe { self.stream().library.cuda_graph_end_capture(self.stream().raw) };
        match (launched, captured) {
            (Ok(()), Ok(graph)) => {
                if let Err(error) =
                    unsafe { self.graphs.insert(self.layer, self.weights, rows, graph) }
                {
                    unsafe {
                        self.stream().library.cuda_graph_exec_destroy(graph)?;
                    }
                    return Err(error);
                }
                Ok(())
            }
            (Err(e), Ok(graph)) => {
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
    pub unsafe fn replay(&mut self, rows: u32) -> Result<RouterOutput<'_>> {
        unsafe { self.replay_counted(rows, None) }
    }
    unsafe fn replay_counted(&mut self, rows: u32, phase: Option<cuteafd_ffi::expert_usage::UsagePhase>) -> Result<RouterOutput<'_>> {
        self.validate(rows)?;
        let (graph, count) = self
            .graphs
            .get_shape(self.layer, self.weights, rows)
            .context("backbone router graph missing")?;
        ensure!(count == rows, "backbone router captured rows differ");
        let launched = unsafe {
            self.stream()
                .library
                .cuda_graph_launch(graph, self.stream().raw)
        };
        // Device-read routes ([`Self::device_read`]): later stages join the
        // chain; the remote layer's collect is its host wait.
        let device_read = self.device_read();
        launched.and_then(|()| unsafe { self.record_routes(rows, phase) })
            .and_then(|()| unsafe { self.ring_download(rows) })
            .and_then(|()| if device_read { unsafe { crate::shared::memory::chain::finish(self.stream().library, self.stream().raw) } }
                else { self.synchronize() })
            .or_else(|error| self.synchronize().and(Err(error)))?;
        self.ready = Some(rows);
        self.output()
    }
    /// Captured routes of ring layers (remote layers, whose routes go to the
    /// Spark exchange from the device, and local layers of a device-ordered
    /// pass) download to this layer's ring slot, read once the pass drained,
    /// on every path including a cold shape.
    ///
    /// # Safety
    /// The router's outputs for `rows` rows are queued on its stream.
    unsafe fn ring_download(&mut self, rows: u32) -> Result<()> {
        if !self.ring_capture() { return Ok(()); }
        let ring = self.route_ring.as_ref().context("route ring absent")?.buffer;
        let slot = self.layer * self.capacity as usize * 24;
        let target = cuteafd_ffi::CuteafdHostBuffer {
            // SAFETY: the slot lies inside the [40][capacity][6] u32 ring.
            ptr: unsafe { ring.ptr.cast::<u8>().add(slot) }.cast(), bytes: rows as usize * 24, ..ring };
        // SAFETY: the ring is pinned and owned by this wave; read only after the pass drains.
        unsafe { self.stream().library.copy_d2h_host_buffer_async(target, self.b(3), rows as usize * 24,
            self.stream().raw)?; }
        // A cold shape runs eagerly, then replays: one entry per layer.
        self.ring_pending.retain(|&(layer, _)| layer != self.layer);
        self.ring_pending.push((self.layer, rows));
        Ok(())
    }
    /// Stage completed input on the router stream. The mask uses the beginning
    /// of the pinned request arena; H2D consumes it before graph D2H overwrites it.
    /// The caller must drain even if subsequent graph preparation fails.
    unsafe fn stage_inputs(&mut self, input: CuteafdDeviceBuffer, mask: &[u8]) -> Result<()> {
        ensure!(
            !mask.is_empty()
                && mask.len() <= self.capacity as usize
                && input.bytes == mask.len() * 10240
                && input.device_id == self.b(0).device_id
                && mask.iter().all(|&v| v <= 1)
                && mask.len() <= self.request_staging.buffer.bytes,
            "invalid router staged input"
        );
        // Deferred passes stage every layer's (identical) mask in its own
        // buffer: rewriting the same bytes cannot race the queued H2D.
        let source = if crate::shared::memory::chain::deferred() {
            self.mask_staging.buffer
        } else {
            self.request_staging.buffer
        };
        unsafe {
            std::ptr::copy_nonoverlapping(
                mask.as_ptr(),
                source.ptr.cast::<u8>(),
                mask.len(),
            );
        }
        let staged = (|| unsafe {
            crate::shared::memory::chain::join(self.stream().library, self.stream().raw)?;
            self.stream()
                .library
                .copy_d2d_async(self.b(0), input, input.bytes, self.stream().raw)?;
            self.stream().library.copy_host_buffer_h2d_async(
                self.b(1),
                source,
                mask.len(),
                self.stream().raw,
            )
        })();
        if staged.is_err() {
            return staged.and(self.synchronize());
        }
        Ok(())
    }
    /// # Safety
    /// The completed block input stays immutable through the copy. This wave has
    /// exclusive storage; image mask matches those tokens (0=text, 1=image).
    pub unsafe fn execute_ffn(
        &mut self,
        input: &FfnInput<'_>,
        image_mask: &[u8],
        phase: cuteafd_ffi::expert_usage::UsagePhase,
    ) -> Result<RouterOutput<'_>> {
        self.invalidate();
        ensure!(
            input.layer == self.layer
                && input.binding().layer() == self.layer
                && !input.tokens.is_empty()
                && input.tokens.len() <= self.capacity as usize
                && input.values.bytes == input.tokens.len() * 10240
                && input.values.device_id == self.b(0).device_id
                && image_mask.len() == input.tokens.len()
                && image_mask.iter().all(|&v| v <= 1),
            "backbone router block input differs"
        );
        let rows = input.tokens.len() as u32;
        let executed = (|| unsafe {
            self.stage_inputs(input.values, image_mask)?;
            self.execute_selected(rows, Some(phase)).map(|_| ())
        })();
        if executed.is_err() {
            // A staged copy may precede a graph preparation failure. Always
            // drain before permitting the input or pinned arena to be reused.
            return executed.and(self.synchronize()).and_then(|_| self.output());
        }
        self.origin = Some(input.binding());
        self.tokens.extend_from_slice(input.tokens);
        self.output()
    }
    /// # Safety
    /// Same initialized finite hidden and binary-mask contract as execute.
    pub unsafe fn execute_captured(&mut self, rows: u32) -> Result<RouterOutput<'_>> {
        unsafe { self.execute_selected(rows, None) }
    }
    unsafe fn execute_selected(&mut self, rows: u32, phase: Option<cuteafd_ffi::expert_usage::UsagePhase>) -> Result<RouterOutput<'_>> {
        self.invalidate();
        if self.window.is_windowed() || !super::graph_policy::captures_shape(rows)
            || (self.graphs.get_shape(self.layer, self.weights, rows).is_none()
                && !self.graphs.capture_allowed(self.layer, rows)) {
            if !self.window.is_windowed() && super::graph_policy::captures_shape(rows)
                && self.graphs.get_shape(self.layer, self.weights, rows).is_none()
                && !self.graphs.capture_allowed(self.layer, rows) {
                super::graph_policy::note_miss(format!("backbone_router rows={rows} layer={}", self.layer),
                    || format!("capacity={} windowed={}", self.capacity, self.window.is_windowed()));
            }
            return unsafe { self.execute_counted(rows, phase) };
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
        unsafe { self.replay_counted(rows, phase) }
    }
    /// # Safety
    /// Same input contract as execute_ffn; the future retains this wave and input
    /// until completion. Cancellation drains before their storage can be reused.
    pub async unsafe fn execute_ffn_cooperative(&mut self, input: &FfnInput<'_>, image_mask: &[u8], phase: cuteafd_ffi::expert_usage::UsagePhase) -> Result<RouterOutput<'_>> {
        self.invalidate();
        ensure!(
            input.layer == self.layer
                && input.binding().layer() == self.layer
                && !input.tokens.is_empty()
                && input.tokens.len() <= self.capacity as usize
                && input.values.bytes == input.tokens.len() * 10240
                && input.values.device_id == self.b(0).device_id
                && image_mask.len() == input.tokens.len()
                && image_mask.iter().all(|&v| v <= 1),
            "backbone router block input differs"
        );
        let rows = input.tokens.len() as u32;
        let cold = self.graphs.get_shape(self.layer, self.weights, rows).is_none();
        let launched = (|| unsafe {
            self.stage_inputs(input.values, image_mask)?;
            let graph = if self.window.is_windowed() { None } else { self.graphs.get_shape(self.layer, self.weights, rows).map(|(raw, _)| raw) };
            let library = self.stream().library; let stream = self.stream().raw;
            crate::shared::decode_graph::census::dispatch(library, stream, c"layer_row", rows as usize,
                graph, || self.enqueue(rows))?;
            self.record_routes(rows, Some(phase))
        })();
        if let Err(error) = launched { self.synchronize()?; return Err(error); }
        unsafe { self.ring_download(rows)?; }
        // Every expert that reads the routes on the device lets later stages
        // join the chain instead of the host waiting here: local layers of a
        // device-ordered pass, and every remote layer (the shared exchange
        // stages its request from the device). A host-ordered pass (1 RTX)
        // then host-waits once per remote layer at its collect, after the
        // request, shared FFN and prefetch are queued; that wait, behind this
        // router in the chain, is what its owners' per-layer rebinds rely on
        // (`LoadStream::require_complete`). Local layers of a host-ordered
        // pass keep this wait (their experts complete on the host). A cold
        // shape always completes here: its eager run precedes the capture.
        // Route-policy tracing reads routes from the pinned staging on the
        // host, so it keeps the wait too.
        let device_read = self.device_read();
        // A shape that is never captured (a prompt's tail unit) has no capture
        // to follow its eager run, so it chains like a replay instead of
        // stalling the issuing thread on every layer.
        let captures = !self.window.is_windowed() && super::graph_policy::captures_shape(rows)
            && self.graphs.capture_allowed(self.layer, rows)
            && !crate::shared::decode_graph::census::eager("layer_row");
        // SAFETY: this wave retains all queued input and output storage.
        unsafe { crate::shared::memory::chain::finish_or_wait(self.stream(),
            (!cold || !captures) && device_read).await?; }
        if cold && !self.window.is_windowed() && super::graph_policy::captures_shape(rows)
            && !self.graphs.capture_allowed(self.layer, rows) {
            super::graph_policy::note_miss(format!("backbone_router rows={rows} layer={}", self.layer),
                || format!("capacity={} windowed={}", self.capacity, self.window.is_windowed()));
        }
        if cold && !self.window.is_windowed() && super::graph_policy::captures_shape(rows)
            && self.graphs.capture_allowed(self.layer, rows)
            && !crate::shared::decode_graph::census::eager("layer_row") {
            // The eager execution above already completed these inputs. Capture
            // records future launches without executing them; publish that result
            // instead of running the same work again on every cache miss.
            unsafe { self.capture_ready(rows)?; }
        }
        self.ready = Some(rows);
        self.origin = Some(input.binding());
        self.tokens.extend_from_slice(input.tokens);
        self.output()
    }
    pub fn output(&self) -> Result<RouterOutput<'_>> {
        let rows = self.ready.context("backbone router output unpublished")?;
        let b = |i: usize, n: usize| {
            let mut b = self.b(i);
            b.bytes = rows as usize * n;
            b
        };
        Ok(RouterOutput {
            layer: self.layer,
            rows,
            input: b(0, 10240),
            expert_input: b(5, 5280),
            request_nvfp4: self.weights.nvfp4,
            mask: b(1, 1),
            scores: b(2, 1536),
            ids: b(3, 24),
            routing: b(4, 24),
            tokens: &self.tokens,
            request_staging: self.request_staging.buffer,
            full_request: self.full_request,
            origin: self.origin,
            stream: self.stream().raw,
            _owner: PhantomData,
        })
    }
    /// Clear only this layer; other layers retain their captured shape.
    pub fn enable_small_graph_shapes(&mut self) {
        self.graphs.enable_small_shapes();
        self.other_graphs.enable_small_shapes();
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.invalidate();
        self.synchronize()?;
        unsafe {
            let current = self.graphs.remove(self.layer);
            let other = self.other_graphs.remove(self.layer);
            current.and(other)?;
        }
        Ok(())
    }
}
impl Drop for BackboneRouterWave<'_, '_> {
    fn drop(&mut self) {
        crate::shared::decode_graph::fatal_drain(self.synchronize(), "backbone router");
        let current = unsafe { self.graphs.clear() };
        let other = unsafe { self.other_graphs.clear() };
        if let Err(e) = current.and(other) {
            tracing::error!(%e,"draining backbone router");
        }
    }
}

/// The next native expert request id (one process-wide sequence, so every
/// remote request is distinct).
fn next_request_id() -> Result<u64> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .ok().context("native expert request IDs exhausted")
}

/// Scheduler metadata in exactly the block's token order.
pub(crate) struct ExpertRow {
    pub request_id: u64,
    pub position: u64,
    pub kind: cuteafd_transport::ExpertV2SourceKind,
}
impl RouterOutput<'_> {
    /// Whether the Spark request carries BF16 rows (NVFP4 experts) instead of FP8 wire rows.
    pub fn request_nvfp4(&self) -> bool {
        self.request_nvfp4
    }
    pub fn validate_request_rows(&self, rows: &[ExpertRow]) -> Result<QueryBinding> {
        let binding = self.binding()?;
        ensure!(
            binding.layer() == self.layer
                && rows.len() == self.rows as usize
                && self.tokens.len() == rows.len()
                && rows.iter().zip(self.tokens).all(|(r, &p)| r.position == p),
            "router request rows differ from block"
        );
        Ok(binding)
    }
    /// The remote request rows from the device buffers (tests; serving sends
    /// them from the device via `wave_sources`).
    /// # Safety
    /// The completed output and its owning stream remain exclusively borrowed
    /// until all transfers drain, including after a partial enqueue failure.
    #[cfg(test)]
    unsafe fn download_request(
        &self,
        library: &NativeLibrary,
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        ensure!(self.full_request, "local router output has no remote request payload");
        let hidden_row_bytes = request_hidden_format(self.request_nvfp4).1;
        let mut hidden = vec![0; self.rows as usize * hidden_row_bytes];
        let mut ids = vec![0; self.rows as usize * 24];
        let mut weights = vec![0; self.rows as usize * 24];
        library.copy_d2h(&mut hidden, if self.request_nvfp4 { self.input } else { self.expert_input })?;
        library.copy_d2h(&mut ids, self.ids)?;
        library.copy_d2h(&mut weights, self.routing)?;
        Ok((hidden, ids, weights))
    }

    /// The shared exchange's staging of this remote layer's wave: `rows x 6`
    /// route ids and gate weights, then the expert input rows in the request's
    /// wire format (FP8 K32, or BF16 for NVFP4 experts), all on the router's
    /// device and complete in the router stream's order.
    pub fn wave_sources(&self) -> Result<(crate::shared::experts::staging::RouteStaging, [CuteafdDeviceBuffer; 3])> {
        ensure!(self.full_request, "local router output has no remote request payload");
        let width = request_hidden_format(self.request_nvfp4).1;
        let layout = crate::shared::experts::staging::RouteStaging::new(self.rows as usize, 6, width)?;
        let input = if self.request_nvfp4 { self.input } else { self.expert_input };
        Ok((layout, [self.ids, self.routing, input]))
    }

    /// The worker-side builder of this layer's Spark request: the same request
    /// request bytes (one request id per
    /// call, taken now, in issue order), from the staged routes and wire rows.
    pub fn request_builder(&self, placement: u64, rows: &[ExpertRow])
        -> Result<(QueryBinding, crate::shared::spark_waves::BuildRequest)> {
        use cuteafd_transport::{ExpertProtocolV2Request, ExpertProtocolV2RowDescriptor};
        let binding = self.validate_request_rows(rows)?;
        ensure!(self.full_request, "local router output has no remote request payload");
        let request_id = next_request_id()?;
        let descriptors: Vec<_> = rows.iter().enumerate().map(|(i, r)| ExpertProtocolV2RowDescriptor {
            row_id: i as u64, source_kind: r.kind, source_request_id: r.request_id, token_position: r.position,
            route_offset: i as u32 * 6, route_count: 6,
        }).collect();
        let (layer, count, dtype) = (self.layer as u32, self.rows, request_hidden_format(self.request_nvfp4).0);
        Ok((binding, Box::new(move |routes, wire| {
            let mut request = ExpertProtocolV2Request::new_bytes(request_id, placement, layer, 5120, dtype,
                descriptors, routes, wire)?;
            request.header.flags |= cuteafd_transport::expert::EXPERT_PROTOCOL_V2_FLAG_V41_COMPACT_BF16;
            cuteafd_transport::expert::BackboneRequest::validate_owned(&request, count)?;
            Ok(request)
        })))
    }

}

#[cfg(test)]
impl BackboneRouterWeights<'_> {
    pub(crate) fn visit(&self, prefix: &str, f: &mut dyn FnMut(String, cuteafd_ffi::CuteafdDeviceBuffer)) {
        self.tensors.visit(prefix, f);
    }
}

#[cfg(test)]
mod host_order_tests {
    use crate::shared::memory::{chain::{self, StageChain}, DeviceAllocation, LoadStream};
    use anyhow::Result;
    #[test]
    fn host_ordered_passes_wait_once_per_layer_after_issuing_it() -> Result<()> {
        let (_directory, fixture) = crate::shared::memory::cpu_stream_fixture()?;
        let library = fixture.load()?;
        fixture.immediate_copies(&library)?;
        fixture.trace_stream_queries(&library)?;
        let stream = LoadStream { library: &library, raw: library.cuda_stream_create()? };
        let storage = DeviceAllocation::new(&library, 4, crate::shared::memory::TEST)?;
        let pass = StageChain::new(&library)?;
        pass.set_device_order(false);
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        runtime.block_on(pass.handle().scope(async {
            for layer in 0..2u8 {
                let before = fixture.events()?.len();
                // Router, request issue, shared FFN and reduce use the real
                // chain/wait helpers over a CPU fake stream, in serving order.
                // SAFETY: storage, source bytes and stream outlive every copy.
                unsafe {
                    library.copy_h2d_async(storage.buffer, &[layer; 4], stream.raw)?;
                    chain::finish_or_wait(&stream, true).await?;
                    library.copy_h2d_async(storage.buffer, &[layer; 4], stream.raw)?;
                    library.copy_h2d_async(storage.buffer, &[layer; 4], stream.raw)?;
                    assert!(!fixture.events()?[before..].contains('q'),
                        "remote router must not wait before request/shared issue");
                    library.copy_h2d_async(storage.buffer, &[layer; 4], stream.raw)?;
                    chain::finish_cooperative(&stream).await?;
                    chain::finish_or_wait(&stream, false).await?;
                }
                let events = fixture.events()?;
                let work = &events[before..];
                assert_eq!(work.chars().filter(|&v| v == 'q').count(), 1,
                    "one host wait per completed remote layer: {work}");
                assert!(work.rfind('Q').unwrap() < work.find('q').unwrap(),
                    "completion wait must follow all issued work: {work}");
                assert_eq!(work.chars().filter(|&v| v == 'Q').count(), 4);
                stream.require_complete()?;
            }
            let before = fixture.events()?.len();
            // Local layers retain the router's host completion point.
            // SAFETY: the same live owners retain this queued copy.
            unsafe {
                library.copy_h2d_async(storage.buffer, &[0; 4], stream.raw)?;
                chain::finish_or_wait(&stream, false).await?;
            }
            assert_eq!(&fixture.events()?[before..], "Qq");
            Ok::<_, anyhow::Error>(())
        }))?;
        pass.drain()?;
        stream.require_complete()?;
        Ok(())
    }
}

#[cfg(test)]
mod expert_usage_tests {
    use super::*;
    use cuteafd_core::expert_usage::{Geometry, Reader};
    use cuteafd_ffi::expert_usage::{ExpertUsage, UsageDevice, UsagePhase};

    #[test]
    fn disabled_expert_usage_never_touches_router_operands_or_captures() {
        let captures = cuteafd_ffi::graph_capture_sites();
        for phase in [UsagePhase::Prefill, UsagePhase::Decode] {
            // SAFETY: disabled telemetry must not inspect these deliberately invalid operands.
            unsafe { record_v41_routes(None, -1, usize::MAX, std::ptr::null(),
                usize::MAX, phase, std::ptr::null_mut()).unwrap(); }
        }
        assert_eq!(cuteafd_ffi::graph_capture_sites(), captures);
    }

    #[test]
    #[ignore = "requires two CUDA GPUs, CUTEAFD_NATIVE_LIB and CUTEAFD_SNAPSHOT"]
    fn v41_expert_usage_real_router_phase_replay_and_bounded_rows() -> Result<()> {
        // SAFETY: the matching native library outlives every fixture allocation and graph.
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        library.cuda_set_device(0)?;
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(
            &std::env::var("CUTEAFD_SNAPSHOT")?))?;
        let reader = Reader::default();
        reader.configure(true, false).map_err(anyhow::Error::msg)?;
        let usage = Rc::new(ExpertUsage::new(&library, Geometry { layers: 40, experts: 384,
            topk: 6, routed_layers: (0..40).collect() },
            &[UsageDevice { logical: 0, cuda: 0 }, UsageDevice { logical: 1, cuda: 1 }], reader.clone())?);
        let mut expected_prefill = vec![0u64; 40 * 384];
        let mut expected_decode = vec![0u64; 40 * 384];
        for (device, layer) in [(0, 2), (1, 20)] {
            library.cuda_set_device(device)?;
            let weights = BackboneRouterWeights::load(&library, &catalog, layer, usize::MAX, 1 << 20)?;
            let capacity = 2768;
            let mut wave = weights.wave(capacity, usize::MAX)?;
            wave.install_expert_usage(Some(usage.clone()));
            wave.set_local_mode(true)?;
            let [input, mask] = wave.inputs();
            let hidden = (0..capacity as usize * 5120).flat_map(|i| {
                let x = ((i * 7 % 31) as f32 - 15.0) / 128.0;
                ((x.to_bits() >> 16) as u16).to_ne_bytes()
            }).collect::<Vec<_>>();
            library.copy_h2d(input, &hidden)?;
            library.copy_h2d(mask, &vec![0; capacity as usize])?;
            // Warmup and cold capture use the uncounted low-level entrypoint.
            // SAFETY: all capacity rows and masks are initialized and wave-owned.
            unsafe { wave.execute_captured(13)?; }
            let graph = wave.graphs.get_shape(layer, &weights, 13).unwrap().0;
            let captures = cuteafd_ffi::graph_capture_sites();
            for phase in [UsagePhase::Prefill, UsagePhase::Decode, UsagePhase::Decode, UsagePhase::Decode] {
                // SAFETY: the real launch uses 13 initialized rows, never capacity 144.
                let output = unsafe { wave.execute_selected(13, Some(phase))? };
                let mut ids = vec![0; 13 * 24];
                library.copy_d2h(&mut ids, output.ids)?;
                let expected = match phase { UsagePhase::Prefill => &mut expected_prefill,
                    UsagePhase::Decode => &mut expected_decode };
                for bytes in ids.chunks_exact(4) {
                    let expert = u32::from_ne_bytes(bytes.try_into().unwrap()) & 511;
                    ensure!(expert < 384, "router selected invalid expert");
                    expected[layer * 384 + expert as usize] += 1;
                }
                assert_eq!(wave.graphs.get_shape(layer, &weights, 13).unwrap().0, graph);
            }
            assert_eq!(cuteafd_ffi::graph_capture_sites(), captures);
            // Each tally covers only live rows, including a full prefill unit;
            // metadata and poison in the capacity suffix must not affect counts.
            for rows in [0, 129, 2752] {
                let words = (0..capacity as usize * 6).map(|i| if i < rows * 6 {
                    (i % 384) as u32 | 0xa000_0000
                } else { 383 }).collect::<Vec<_>>();
                let bytes = words.iter().flat_map(|n| n.to_ne_bytes()).collect::<Vec<_>>();
                library.copy_h2d(wave.b(3), &bytes)?;
                // SAFETY: rows complete ID rows remain live on this wave's stream.
                unsafe { wave.record_routes(rows as u32, Some(UsagePhase::Prefill))?; }
                wave.synchronize()?;
                for &word in &words[..rows * 6] { expected_prefill[layer * 384 + (word & 511) as usize] += 1; }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(1500));
        let value = reader.snapshot();
        ensure!(value["error"].is_null(), "snapshot worker failed");
        let total = &value["windows"]["total"];
        assert_eq!(total["prefill_routed_rows"], serde_json::json!(expected_prefill));
        assert_eq!(total["decode_routed_rows"], serde_json::json!(expected_decode));
        assert_eq!(expected_prefill.iter().sum::<u64>(), 2 * (13 + 129 + 2752) * 6);
        assert_eq!(expected_decode.iter().sum::<u64>(), 2 * 3 * 13 * 6);
        library.cuda_set_device(0)?;
        Ok(())
    }
}

#[cfg(test)]
mod storage_bytes_tests {
    use super::*;
    #[test]
    fn router_storage_bytes_equal_planned_allocations() -> Result<()> {
        for unit in [4096, 2752, 2048] {
            assert_eq!(BackboneRouterStorage::storage_bytes(8192, unit)?,
                BackboneRouterStorage::allocation_sizes(8192).into_iter().sum::<usize>());
        }
        Ok(())
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
        let weights = BackboneRouterWeights::load(&lib, &catalog, 2, usize::MAX, 1 << 20)?;
        let (rows, unit, reserve) = (13usize, 16u32, 48u32);
        let inputs = [0, 1].map(|seed| -> Result<_> {
            let hidden: Vec<u8> = (0..rows * 5120).flat_map(|i| {
                let f = ((i * 7 + seed * 11) % 29) as f32 / 32.0 - 0.4;
                ((f.to_bits() >> 16) as u16).to_ne_bytes()
            }).collect();
            let source = DeviceAllocation::new(&lib, hidden.len(), crate::shared::memory::TEST)?;
            lib.copy_h2d(source.buffer, &hidden)?; Ok(source)
        }).into_iter().collect::<Result<Vec<_>>>()?;
        let read = |wave: &BackboneRouterWave<'_, '_>| -> Result<Vec<Vec<u8>>> {
            (0..6).map(|i| {
                let mut bytes = vec![0; rows * ROW_BYTES[i]];
                lib.copy_d2h(&mut bytes, wave.b(i))?; Ok(bytes)
            }).collect()
        };
        let mut expected = Vec::new();
        for source in &inputs {
            let mut wave = weights.wave(unit, usize::MAX)?;
            lib.copy_h2d(wave.b(0), &vec![0; unit as usize * 10240])?;
            lib.copy_h2d(wave.b(1), &vec![0; unit as usize])?;
            unsafe { lib.copy_d2d_async(wave.b(0), source.buffer, source.buffer.bytes, wave.stream().raw)?; }
            unsafe { wave.execute(rows as u32)?; }
            expected.push(read(&wave)?);
        }
        assert_ne!(expected[0][2], expected[1][2]);
        let storage = BackboneRouterStorage::new(&lib, reserve, unit)?;
        PRIVATE_ALIAS.with(|flag| flag.set(!ordered));
        let built = (|| -> Result<_> { Ok([weights.on_storage(storage.clone())?, weights.on_storage(storage.clone())?]) })();
        PRIVATE_ALIAS.with(|flag| flag.set(false));
        let mut slots = built?;
        for (i, slot) in slots.iter_mut().enumerate() {
            slot.select_window(RowWindow::within(i * unit as usize..i * unit as usize + rows, reserve as usize)?)?;
            slot.set_ring_capture(true);
            lib.copy_h2d(slot.b(1), &vec![0; rows])?;
        }
        let chains = [StageChain::new(&lib)?, StageChain::new(&lib)?];
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        // Warm the exact shared kernels and pinned D2H paths before gating.
        for (i, slot) in slots.iter_mut().enumerate() {
            unsafe {
                lib.copy_d2d_async(slot.b(0), inputs[i].buffer, rows * 10240, slot.stream().raw)?;
                slot.enqueue(rows as u32)?;
                slot.ring_download(rows as u32)?;
            }
            slot.synchronize()?;
            slot.ring_pending.clear();
        }
        let mut gate = None;
        for i in 0..2 {
            runtime.block_on(chains[i].handle().scope(async {
                let slot = &mut slots[i]; let stream = slot.stream().raw;
                // SAFETY: sources and planes outlive both chains and the gate.
                unsafe {
                    crate::shared::memory::chain::join(&lib, stream)?;
                    lib.copy_d2d_async(slot.b(0), inputs[i].buffer, rows * 10240, stream)?;
                    if i == 0 { gate = Some(cuteafd_ffi::test_support::CudaStreamGate::new(&lib, stream)?); }
                    slot.enqueue(rows as u32)?;
                    slot.ring_download(rows as u32)?;
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
        assert_eq!(done, !ordered, "router B completion before gate release");
        drop(gate);
        for chain in &chains { chain.drain()?; }
        for (i, slot) in slots.iter_mut().enumerate() {
            slot.synchronize()?;
            let got = read(slot)?;
            if ordered || i == 1 { assert_eq!(got, expected[i]); }
            else { assert_ne!(got, expected[i], "negative control failed to detect aliased planes"); }
            let mut ring = vec![Vec::new(); 40];
            assert_eq!(slot.drain_ring(&mut ring)?, vec![2]);
            let route_bytes: Vec<u8> = ring[2].iter().flat_map(|row| row.iter().flat_map(|id| id.to_ne_bytes())).collect();
            assert_eq!(route_bytes, got[3]);
        }
        eprintln!("PASS router shared storage ordered={ordered}, B_completed_before_gate={done}");
        Ok(())
    }
    #[test]
    #[ignore = "requires snapshot, native library and CUDA"]
    fn router_shared_planes_match_private_runs_behind_gate() -> Result<()> {
        for _ in 0..3 { run(true)?; } Ok(())
    }
    #[test]
    #[ignore = "requires snapshot, native library and CUDA"]
    fn router_private_streams_and_aliased_planes_clobber_gated_run() -> Result<()> {
        for _ in 0..3 { run(false)?; } Ok(())
    }
}
