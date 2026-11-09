use crate::shared::decode_graph::{fatal_drain, BindingGraphs};
use super::DsparkWeights;
use crate::shared::memory::Tag;
use crate::shared::memory::{DeviceAllocation};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, V41DsparkConfidence};
use std::ffi::c_void;

/// Stable per-wave confidence inputs/output; borrowed weights outlive its graph.
pub(crate) struct DsparkConfidence<'weights, 'library> {
    standalone: &'weights crate::shared::memory::StandaloneStream<'library>,
    hidden: DeviceAllocation<'library>,
    markov: DeviceAllocation<'library>,
    output: DeviceAllocation<'library>,
    kernel: V41DsparkConfidence<'library>,
    weights: &'weights DsparkWeights<'library>,
    graph: BindingGraphs<'library, usize>,
    capacity: usize,
    ready_rows: Option<usize>,
}
impl<'library> DsparkWeights<'library> {
    pub fn confidence(
        &self,
        capacity: usize,
        available_device_bytes: usize,
    ) -> Result<DsparkConfidence<'_, 'library>> {
        let bytes = DsparkConfidence::device_bytes(capacity)?;
        ensure!(
            bytes <= available_device_bytes,
            "dSpark confidence exceeds device budget"
        );
        let library = self.library;
        let kernel = library.v41_dspark_confidence()?;
        // Resolve the final-stage checkpoint weight before allocating workspace.
        self.tensor("mtp.2.confidence_head.proj.weight")?;
        Ok(DsparkConfidence {
            standalone: &self.standalone,
            hidden: DeviceAllocation::new(library, capacity * 10240, Tag::drafter_workspace("v41/draft-confidence"))?,
            markov: DeviceAllocation::new(library, capacity * 512, Tag::drafter_workspace("v41/draft-confidence"))?,
            output: DeviceAllocation::new(library, capacity * 4, Tag::drafter_workspace("v41/draft-confidence"))?,
            kernel,
            weights: self,
            graph: BindingGraphs::new(library, "dspark_confidence")?,
            capacity,
            ready_rows: None,
        })
    }
}
impl DsparkConfidence<'_, '_> {
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        ensure!(
            (1..=4096).contains(&capacity),
            "invalid confidence capacity"
        );
        Ok(capacity * (10240 + 512 + 4))
    }
    /// BF16 hidden [capacity,5120] and preceding-token Markov embedding
    /// [capacity,256], with identical row order. Never free or retain after drop.
    /// Finish producer writes before execute/replay; serialize reuse per owner.
    pub fn inputs(&self) -> [CuteafdDeviceBuffer; 2] {
        [self.hidden.buffer, self.markov.buffer]
    }
    fn validate_rows(&self, rows: usize) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.capacity,
            "confidence rows exceed capacity"
        );
        Ok(())
    }
    unsafe fn enqueue(&self, rows: usize) -> Result<()> {
        unsafe { self.enqueue_on(rows, self.standalone.get()?.raw) }
    }
    pub(super) unsafe fn enqueue_on(&self, rows: usize, stream: *mut c_void) -> Result<()> {
        self.validate_rows(rows)?;
        unsafe {
            self.kernel.launch(
                self.hidden.buffer,
                self.markov.buffer,
                self.weights.tensor("mtp.2.confidence_head.proj.weight")?,
                self.output.buffer,
                rows,
                stream,
            )
        }
    }
    pub(super) fn storage(&self) -> [CuteafdDeviceBuffer; 1] {
        [self.output.buffer]
    }
    fn synchronize(&self) -> Result<()> {
        self.standalone.drain()
    }
    /// # Safety
    /// Both owned inputs must be initialized for these rows and producer work
    /// must be complete; input writes must not race this execution.
    pub unsafe fn execute(&mut self, rows: usize) -> Result<CuteafdDeviceBuffer> {
        self.ready_rows = None;
        self.validate_rows(rows)?;
        unsafe {
            self.enqueue(rows)?;
        }
        self.synchronize()?;
        self.ready_rows = Some(rows);
        self.output()
    }
    /// # Safety
    /// Same input contract as execute; warms the launch before graph capture.
    pub unsafe fn capture(&mut self, rows: usize) -> Result<()> {
        unsafe {
            self.execute(rows)?;
        }
        self.ready_rows = None;
        let key = rows;
        let Some(ticket) = self.graph.cache_ticket(&key, false, None) else { return Ok(()); };
        unsafe {
            self.standalone.device
                .library
                .cuda_graph_begin_capture(self.standalone.get()?.raw)?;
        }
        let launch = unsafe { self.enqueue(rows) };
        let capture = unsafe { self.standalone.device.library.cuda_graph_end_capture_topology(self.standalone.get()?.raw) };
        match (launch, capture) {
            (Ok(()), Ok((graph, nodes, width, spans, inspection))) => {
                super::chain::warn_topology(inspection);
                // SAFETY: wave pins captured storage through drained teardown.
                unsafe { self.graph.cache_insert(key, graph, ticket, nodes as u64, width, spans)?; }
                Ok(())
            }
            (Err(error), Ok((graph, _, _, _, _))) => {
                if let Err(cleanup) = unsafe { self.standalone.device.library.cuda_graph_exec_destroy(graph) }
                {
                    tracing::error!(%cleanup, "destroying failed confidence capture");
                }
                Err(error)
            }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// Same input contract as execute; rows must match the captured shape.
    pub unsafe fn replay(&mut self, rows: usize) -> Result<CuteafdDeviceBuffer> {
        self.ready_rows = None;
        let Some(graph) = self.graph.cached_launch(&rows) else {
            // SAFETY: replay and eager execution have the same input contract.
            return unsafe { self.execute(rows) };
        };
        unsafe {
            self.standalone.device
                .library
                .cuda_graph_launch(graph, self.standalone.get()?.raw)?;
        }
        self.synchronize()?;
        self.ready_rows = Some(rows);
        self.output()
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.ready_rows = None;
        self.synchronize()?;
        // SAFETY: owning stream has drained before executable retirement.
        unsafe { self.graph.clear()?; }
        Ok(())
    }
    /// Raw FP32 scores, without sigmoid; borrowed until reuse/drop.
    pub fn output(&self) -> Result<CuteafdDeviceBuffer> {
        let rows = self
            .ready_rows
            .context("confidence output is not complete")?;
        let mut output = self.output.buffer;
        output.bytes = rows * 4;
        Ok(output)
    }
}
impl Drop for DsparkConfidence<'_, '_> {
    fn drop(&mut self) {
        fatal_drain(self.synchronize(), "dSpark confidence");
        // SAFETY: owning stream has drained before executable retirement.
        fatal_drain(unsafe { self.graph.clear() }, "dSpark confidence graphs");
    }
}

impl DsparkConfidence<'_, '_> {
    pub(crate) fn visit_graph_banks(&mut self, f: &mut super::chain::GraphVisitor<'_>) -> Result<()> {
        f(self.standalone.device.id, "dspark_confidence", false, &mut self.graph)?;
        Ok(())
    }
}
