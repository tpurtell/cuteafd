//! Inverse RoPE -> grouped FP8 wo_a -> native FP8 wo_b in caller stream order.
use crate::shared::decode_graph::{fatal_drain, BindingGraphs};
use super::{DsparkProjection, DsparkWeights, ProjectionKind};
use crate::shared::memory::Tag;
use crate::shared::memory::{DeviceAllocation, LoadStream};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{
    CuteafdDeviceBuffer, NativeLibrary, V41Fp8Plan,
};
use std::ffi::c_void;
pub(crate) struct DsparkAttentionOutput<'weights, 'library> {
    standalone: &'weights crate::shared::memory::StandaloneStream<'library>,
    grouped: V41Fp8Plan<'library>,
    grouped_scratch: DeviceAllocation<'library>,
    alpha: DeviceAllocation<'library>,
    scales: CuteafdDeviceBuffer,
    projection: DsparkProjection<'weights, 'library>,
    weight: CuteafdDeviceBuffer,
    input: DeviceAllocation<'library>,
    frequencies: DeviceAllocation<'library>,
    capacity: u32,
    graph: BindingGraphs<'library, u32>,
    ready: Option<u32>,
}
impl<'library> DsparkWeights<'library> {
    pub fn attention_output(
        &self,
        stage: usize,
        capacity: u32,
        budget: usize,
    ) -> Result<DsparkAttentionOutput<'_, 'library>> {
        ensure!(stage < 3, "invalid dSpark attention output stage");
        let library = self.library;
        ensure!(
            DsparkAttentionOutput::device_bytes(library, capacity)? <= budget,
            "dSpark attention output exceeds budget"
        );
        let grouped = library.v41_fp8_matrix_plan(capacity, 32768, 8192)?;
        let grouped_scratch = DeviceAllocation::new(library, grouped.info().scratch_bytes as usize, Tag::drafter_workspace("v41/draft-attention-output"))?;
        let alpha = DeviceAllocation::new(library, 4, Tag::drafter_workspace("v41/draft-attention-output"))?;
        let stream = LoadStream { library, raw: library.cuda_stream_create()? };
        let initialized = unsafe { grouped.initialize_scratch(grouped_scratch.buffer, alpha.buffer, stream.raw) };
        initialized.and(unsafe { library.cuda_stream_synchronize(stream.raw) })?;
        let projection = self.projection(
            ProjectionKind::OutputB(stage),
            capacity,
            DsparkProjection::device_bytes(library, ProjectionKind::OutputB(stage), capacity)?,
        )?;
        Ok(DsparkAttentionOutput {
            standalone: &self.standalone,
            grouped,
            grouped_scratch,
            alpha,
            scales: self.grouped_output_scales[stage].buffer,
            projection,
            weight: self.auxiliary.get(&format!("mtp.{stage}.attn.wo_a.weight"))?,
            input: DeviceAllocation::new(library, capacity as usize * 65536, Tag::drafter_workspace("v41/draft-attention-output"))?,
            frequencies: DeviceAllocation::new(library, capacity as usize * 256, Tag::drafter_workspace("v41/draft-attention-output"))?,
            capacity,
            graph: BindingGraphs::new(library, "dspark_attention_output")?,
            ready: None,
        })
    }
}
impl DsparkAttentionOutput<'_, '_> {
    // Extra storage beyond the already-budgeted OutputB projection owner.
    pub fn additional_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid dSpark attention output capacity"
        );
        Ok(library.v41_fp8_matrix_plan_info(capacity, 32768, 8192)?.scratch_bytes as usize + 4 + capacity as usize * (65536 + 256))
    }
    pub fn device_bytes(library: &NativeLibrary, capacity: u32) -> Result<usize> {
        DsparkProjection::device_bytes(library, ProjectionKind::OutputB(0), capacity)?
            .checked_add(Self::additional_bytes(library, capacity)?)
            .context("dSpark attention output storage overflow")
    }
    pub(super) fn output_storage(&self)->CuteafdDeviceBuffer {self.projection.output_storage()}
    pub fn input(&self) -> CuteafdDeviceBuffer {
        self.input.buffer
    }
    pub fn frequencies(&self) -> CuteafdDeviceBuffer {
        self.frequencies.buffer
    }
    fn synchronize(&self) -> Result<()> {
        self.standalone.drain()
    }
    /// # Safety
    /// Inputs/frequencies must be initialized on this device. A containing owner
    /// must drain the supplied stream before ending the exclusive borrow.
    pub(super) unsafe fn enqueue(&mut self, rows: u32, stream: *mut c_void) -> Result<()> {
        self.ready = None;
        ensure!(
            (1..=self.capacity).contains(&rows),
            "invalid dSpark attention output rows"
        );
        unsafe {
            self.grouped.launch_rope(
                self.input.buffer,
                self.frequencies.buffer,
                self.weight,
                self.scales,
                self.grouped_scratch.buffer,
                self.alpha.buffer,
                self.projection.input(),
                rows,
                stream,
            )?;
            self.projection.enqueue(
                self.projection.input(),
                self.projection.output_storage(),
                rows,
                stream,
            )
        }
    }
    /// # Safety
    /// Initialize finite BF16 attention outputs and per-row FP32 complex frequencies,
    /// finish producer writes, and serialize raw view reuse through completion.
    pub unsafe fn execute(&mut self, rows: u32) -> Result<CuteafdDeviceBuffer> {
        let launched = unsafe { self.enqueue(rows, self.standalone.get()?.raw) };
        let drained = self.synchronize();
        launched.and(drained)?;
        self.ready = Some(rows);
        self.output()
    }
    /// # Safety
    /// Same input contract as execute; warmup and capture use owned storage only.
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
        let launched = unsafe { self.enqueue(rows, self.standalone.get()?.raw) };
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
                    tracing::error!(%cleanup,"destroying failed attention output capture");
                }
                Err(error)
            }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// Same input contract as execute, using the captured row count.
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
            .context("dSpark attention output is incomplete")?;
        let mut output = self.projection.output_storage();
        output.bytes = rows as usize * 10240;
        Ok(output)
    }
}
impl Drop for DsparkAttentionOutput<'_, '_> {
    fn drop(&mut self) {
        fatal_drain(self.synchronize(), "dSpark attention_output");
        // SAFETY: owning stream has drained before executable retirement.
        fatal_drain(unsafe { self.graph.clear() }, "dSpark attention_output graphs");
    }
}

pub(super) fn pack_grouped_scales<'a>(
    library: &'a NativeLibrary,
    tensors: &crate::families::deepseek_v41::v41_tensors::NativeRtxTensors<'a>,
    arena: &crate::shared::memory::weight_arena::DeviceWeightArena<'a>,
) -> Result<[DeviceAllocation<'a>; 3]> {
    let mut output = Vec::with_capacity(3);
    // Drain before destinations are released on any failed load.
    let stream = LoadStream {
        library,
        raw: library.cuda_stream_create()?,
    };
    for stage in 0..3 {
        output.push(crate::families::deepseek_v41::weight_layout::allocation(library, Some(arena),
            &crate::families::deepseek_v41::weight_layout::packed_scale_name(
                &format!("mtp.{stage}.attn.wo_a.weight")), 1048576)?);
        unsafe {
            crate::shared::experts::rtx::dense::v41_pack_scales(&library.v41_fp8_matrix_kernel(1, 32768, 8192)?,
                tensors.get(&format!("mtp.{stage}.attn.wo_a.scale"))?,
                output.last().unwrap().buffer,
                stream.raw,
            )?;
        }
    }
    unsafe {
        library.cuda_stream_synchronize(stream.raw)?;
    }
    output
        .try_into()
        .ok()
        .context("expected three grouped output scale allocations")
}

impl DsparkAttentionOutput<'_, '_> {
    pub(crate) fn visit_graph_banks(&mut self, f: &mut super::chain::GraphVisitor<'_>) -> Result<()> {
        f(self.standalone.device.id, "dspark_attention_output", false, &mut self.graph)?;
        self.projection.visit_graph_banks(f)?;
        Ok(())
    }
}
