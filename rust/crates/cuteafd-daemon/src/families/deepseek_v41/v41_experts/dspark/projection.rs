//! Native dSpark main/attention FP8 matrix ownership.
use crate::shared::decode_graph::{fatal_drain, BindingGraphs};
use super::DsparkWeights;
use crate::shared::memory::Tag;
use crate::shared::memory::{DeviceAllocation, LoadStream};
use crate::families::deepseek_v41::v41_tensors::NativeRtxTensors;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary, V41Fp8Plan};
use std::ffi::c_void;

#[derive(Clone, Copy, Debug)]
pub(crate) enum ProjectionKind {
    Main,
    QueryA(usize),
    QueryB(usize),
    Kv(usize),
    OutputB(usize),
}
impl ProjectionKind {
    fn binding(self) -> Result<(usize, String, u32, u32)> {
        let (stage, offset, suffix, k, n) = match self {
            Self::Main => return Ok((0, "mtp.0.main_proj".into(), 15360, 5120)),
            Self::QueryA(s) => (s, 0, "wq_a", 5120, 1280),
            Self::QueryB(s) => (s, 1, "wq_b", 1280, 32768),
            Self::Kv(s) => (s, 2, "wkv", 5120, 512),
            Self::OutputB(s) => (s, 3, "wo_b", 8192, 5120),
        };
        ensure!(stage < 3, "invalid dSpark projection stage");
        Ok((
            1 + stage * 4 + offset,
            format!("mtp.{stage}.attn.{suffix}"),
            k,
            n,
        ))
    }
}
const KINDS: [ProjectionKind; 13] = [
    ProjectionKind::Main,
    ProjectionKind::QueryA(0),
    ProjectionKind::QueryB(0),
    ProjectionKind::Kv(0),
    ProjectionKind::OutputB(0),
    ProjectionKind::QueryA(1),
    ProjectionKind::QueryB(1),
    ProjectionKind::Kv(1),
    ProjectionKind::OutputB(1),
    ProjectionKind::QueryA(2),
    ProjectionKind::QueryB(2),
    ProjectionKind::Kv(2),
    ProjectionKind::OutputB(2),
];
pub(super) fn packed_bytes(library: &NativeLibrary) -> Result<usize> {
    KINDS.iter().try_fold(0usize, |bytes, kind| {
        let (_, _, k, n) = kind.binding()?;
        bytes
            .checked_add(
                library
                    .v41_fp8_matrix_info(16, k, n)?
                    .packed_weight_scale_bytes as usize,
            )
            .context("dSpark projection scale budget overflow")
    })
}
pub(super) fn wave_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
    KINDS.iter().try_fold(0usize, |bytes, kind| {
        bytes
            .checked_add(DsparkProjection::device_bytes(library, *kind, capacity)?)
            .context("dSpark projection wave budget overflow")
    })
}
pub(super) fn pack_scales<'a>(
    library: &'a NativeLibrary,
    tensors: &NativeRtxTensors<'a>,
    arena: &crate::shared::memory::weight_arena::DeviceWeightArena<'a>,
) -> Result<[DeviceAllocation<'a>; 13]> {
    let mut scales = Vec::with_capacity(13);
    // On error the stream drains before the vector releases pending destinations.
    let stream = LoadStream {
        library,
        raw: library.cuda_stream_create()?,
    };
    for kind in KINDS {
        let (_, name, k, n) = kind.binding()?;
        let kernel = library.v41_fp8_matrix_kernel(16, k, n)?;
        scales.push(crate::families::deepseek_v41::weight_layout::allocation(
            library, Some(arena),
            &crate::families::deepseek_v41::weight_layout::packed_scale_name(&format!("{name}.weight")),
            kernel.info().packed_weight_scale_bytes as usize,
        )?);
        unsafe {
            crate::shared::experts::rtx::dense::v41_pack_scales(&kernel, 
                tensors.get(&format!("{name}.scale"))?,
                scales.last().unwrap().buffer,
                stream.raw,
            )?;
        }
    }
    unsafe {
        library.cuda_stream_synchronize(stream.raw)?;
    }
    scales
        .try_into()
        .ok()
        .context("expected thirteen dSpark projection scales")
}
pub(crate) struct DsparkProjection<'weights, 'library> {
    standalone: &'weights crate::shared::memory::StandaloneStream<'library>,
    kernel: V41Fp8Plan<'library>,
    _weights: &'weights DsparkWeights<'library>,
    weight: CuteafdDeviceBuffer,
    scales: CuteafdDeviceBuffer,
    scratch: DeviceAllocation<'library>,
    alpha: DeviceAllocation<'library>,
    input: CuteafdDeviceBuffer,
    _owned_input: Option<DeviceAllocation<'library>>,
    output: DeviceAllocation<'library>,
    graph: BindingGraphs<'library, u32>,
    ready: Option<u32>,
}
impl<'library> DsparkWeights<'library> {
    pub fn projection(
        &self,
        kind: ProjectionKind,
        capacity: u32,
        budget: usize,
    ) -> Result<DsparkProjection<'_, 'library>> {
        unsafe { self.projection_input(kind, capacity, None, budget) }
    }
    /// Borrow an existing producer buffer instead of allocating unused private input.
    /// # Safety
    /// Input remains on this device and live through this projection's destruction;
    /// the containing owner serializes producer writes and projection reads.
    pub unsafe fn projection_from(
        &self, kind: ProjectionKind, capacity: u32, input: CuteafdDeviceBuffer, budget: usize,
    ) -> Result<DsparkProjection<'_, 'library>> {
        unsafe { self.projection_input(kind, capacity, Some(input), budget) }
    }
    unsafe fn projection_input(
        &self, kind: ProjectionKind, capacity: u32, external: Option<CuteafdDeviceBuffer>, budget: usize,
    ) -> Result<DsparkProjection<'_, 'library>> {
        let library = self.library;
        ensure!(
            (if external.is_some() { DsparkProjection::external_input_bytes(library, kind, capacity)? }
                else { DsparkProjection::device_bytes(library, kind, capacity)? }) <= budget,
            "dSpark projection exceeds budget"
        );
        let (index, name, k, n) = kind.binding()?;
        let kernel = library.v41_fp8_matrix_plan(capacity, k, n)?;
        let owned_input = if external.is_none() { Some(DeviceAllocation::new(library, capacity as usize*k as usize*2, Tag::drafter_workspace("v41/draft-projection"))?) } else { None };
        let input = external.unwrap_or_else(|| owned_input.as_ref().unwrap().buffer);
        ensure!(!input.ptr.is_null() && input.bytes >= capacity as usize*k as usize*2
            && input.device_id == self.tensor(&format!("{name}.weight"))?.device_id,
            "projection input extent or device differs");
        let value = DsparkProjection {
            standalone: &self.standalone,
            scratch: DeviceAllocation::new(library, kernel.info().scratch_bytes as usize, Tag::drafter_workspace("v41/draft-projection"))?,
            kernel,
            _weights: self,
            weight: self.tensor(&format!("{name}.weight"))?,
            scales: self.projection_scales[index].buffer,
            alpha: DeviceAllocation::new(library, 4, Tag::drafter_workspace("v41/draft-projection"))?,
            input,
            _owned_input: owned_input,
            output: DeviceAllocation::new(library, capacity as usize * n as usize * 2, Tag::drafter_workspace("v41/draft-projection"))?,
            graph: BindingGraphs::new(library, "dspark_projection")?,
            ready: None,
        };
        let stream = LoadStream { library, raw: library.cuda_stream_create()? };
        // SAFETY: load-only stream drains before projection scratch can be released.
        unsafe {
            value.kernel.initialize_scratch(
                value.scratch.buffer,
                value.alpha.buffer,
                stream.raw,
            )?;
        }
        // SAFETY: initialization and its storage belong to this load scope.
        unsafe { library.cuda_stream_synchronize(stream.raw)?; }
        Ok(value)
    }
}
impl DsparkProjection<'_, '_> {
    pub fn device_bytes(
        library: &NativeLibrary,
        kind: ProjectionKind,
        capacity: u32,
    ) -> Result<usize> {
        let (_, _, k, n) = kind.binding()?;
        let info = library.v41_fp8_matrix_plan_info(capacity, k, n)?;
        usize::try_from(
            info.scratch_bytes + u64::from(capacity) * (u64::from(k) + u64::from(n)) * 2 + 4,
        )
        .context("dSpark projection workspace overflow")
    }
    pub fn external_input_bytes(library: &NativeLibrary, kind: ProjectionKind, capacity: u32) -> Result<usize> {
        let (_, _, k, _) = kind.binding()?;
        Self::device_bytes(library, kind, capacity)?
            .checked_sub(capacity as usize*k as usize*2).context("projection input budget underflow")
    }
    // Storage access for an exclusive containing owner that tracks completion itself.
    pub(super) fn output_storage(&self) -> CuteafdDeviceBuffer {
        self.output.buffer
    }
    pub fn input(&self) -> CuteafdDeviceBuffer {
        self.input
    }
    fn synchronize(&self) -> Result<()> {
        self.standalone.drain()
    }
    /// For a containing attention owner: supplied views must be distinct initialized
    /// buffers on the stream device; drain before releasing this exclusive borrow.
    pub(super) unsafe fn enqueue(
        &mut self,
        input: CuteafdDeviceBuffer,
        output: CuteafdDeviceBuffer,
        rows: u32,
        stream: *mut c_void,
    ) -> Result<()> {
        self.ready = None;
        unsafe {
            self.kernel.launch(
                input,
                self.weight,
                self.scales,
                self.scratch.buffer,
                self.alpha.buffer,
                output,
                rows,
                stream,
            )
        }
    }
    /// # Safety
    /// Initialize finite BF16 input rows and finish producer writes; serialize view reuse.
    pub unsafe fn execute(&mut self, rows: u32) -> Result<CuteafdDeviceBuffer> {
        let launched =
            unsafe { self.enqueue(self.input, self.output.buffer, rows, self.standalone.get()?.raw) };
        let drained = self.synchronize();
        launched.and(drained)?;
        self.ready = Some(rows);
        self.output()
    }
    /// # Safety
    /// Same initialized input contract as execute; captures only owned addresses.
    pub unsafe fn capture(&mut self, rows: u32) -> Result<()> {
        unsafe {
            self.execute(rows)?;
        }
        self.ready = None;
        let key = rows;
        let Some(ticket) = self.graph.cache_ticket(&key, false, None) else { return Ok(()); };
        unsafe {
            self.standalone.device
                .library
                .cuda_graph_begin_capture(self.standalone.get()?.raw)?;
        }
        let launched =
            unsafe { self.enqueue(self.input, self.output.buffer, rows, self.standalone.get()?.raw) };
        let captured = unsafe { self.standalone.device.library.cuda_graph_end_capture_topology(self.standalone.get()?.raw) };
        match (launched, captured) {
            (Ok(()), Ok((graph, nodes, width, spans, inspection))) => {
                super::chain::warn_topology(inspection);
                // SAFETY: wave pins captured storage through drained teardown.
                unsafe { self.graph.cache_insert(key, graph, ticket, nodes as u64, width, spans)?; }
                Ok(())
            }
            (Err(error), Ok((graph, _, _, _, _))) => {
                if let Err(cleanup) = unsafe { self.standalone.device.library.cuda_graph_exec_destroy(graph) }
                {
                    tracing::error!(%cleanup,"destroying failed projection capture");
                }
                Err(error)
            }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// Same input contract as execute, with the captured row count.
    pub unsafe fn replay(&mut self, rows: u32) -> Result<CuteafdDeviceBuffer> {
        self.ready = None;
        let Some(graph) = self.graph.cached_launch(&rows) else {
            // SAFETY: replay and eager execution have the same input contract.
            return unsafe { self.execute(rows) };
        };
        let launched = unsafe {
            self.standalone.device
                .library
                .cuda_graph_launch(graph, self.standalone.get()?.raw)
        };
        let drained = self.synchronize();
        launched.and(drained)?;
        self.ready = Some(rows);
        self.output()
    }
    pub fn output(&self) -> Result<CuteafdDeviceBuffer> {
        let rows = self
            .ready
            .context("dSpark projection output is incomplete")?;
        let mut output = self.output.buffer;
        output.bytes = rows as usize * self.kernel.info().output_dim as usize * 2;
        Ok(output)
    }
}
impl Drop for DsparkProjection<'_, '_> {
    fn drop(&mut self) {
        fatal_drain(self.synchronize(), "dSpark projection");
        // SAFETY: owning stream has drained before executable retirement.
        fatal_drain(unsafe { self.graph.clear() }, "dSpark projection graphs");
    }
}

impl DsparkProjection<'_, '_> {
    pub(crate) fn visit_graph_banks(&mut self, f: &mut super::chain::GraphVisitor<'_>) -> Result<()> {
        f(self.standalone.device.id, "dspark_projection", false, &mut self.graph)?;
        Ok(())
    }
}
