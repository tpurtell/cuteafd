//! One complete dSpark FFN boundary on its owned expert wave stream.
use crate::shared::decode_graph::{fatal_drain, BindingGraphs};
use super::{DsparkRouter, DsparkSharedFfn, DsparkWeights, HcSublayer};
use super::expert_backend::DraftExperts;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
use std::ffi::c_void;

pub(crate) struct DsparkFfn<'weights, 'library> {
    // Drop drains this stream before any sibling workspace can be released.
    experts: DraftExperts<'weights, 'library>,
    boundary: HcSublayer<'weights, 'library>,
    router: DsparkRouter<'weights, 'library>,
    shared: DsparkSharedFfn<'weights, 'library>,
    device_bytes: usize,
    library: &'library NativeLibrary,
    standalone: &'weights crate::shared::memory::StandaloneStream<'library>,
    graph: BindingGraphs<'library, u32>,
}
impl<'library> DsparkWeights<'library> {
    pub fn ffn_bytes(&self, capacity: u32) -> Result<usize> { self.shape().ffn_bytes(capacity) }
    /// Replaces one stage's separate FFN mHC/router/shared/expert wave owners.
    /// These allocations are already included in DsparkBudget, not additional.
    pub fn ffn(
        &self,
        stage: usize,
        capacity: u32,
        budget: usize,
    ) -> Result<DsparkFfn<'_, 'library>> {
        ensure!(stage < 3, "invalid dSpark FFN stage");
        let bytes = self.ffn_bytes(capacity)?;
        ensure!(bytes <= budget, "dSpark FFN exceeds device budget");
        let library = self.library;
        Ok(DsparkFfn {
            experts: self.expert_wave(stage, capacity)?,
            boundary: self.hc_sublayer(
                stage,
                false,
                capacity as usize,
                HcSublayer::device_bytes(capacity as usize)?,
            )?,
            router: self.router(
                stage,
                capacity as usize,
                DsparkRouter::device_bytes(capacity as usize)?,
            )?,
            shared: self.shared_ffn(
                stage,
                capacity,
                DsparkSharedFfn::device_bytes(library, capacity)?,
            )?,
            standalone: &self.standalone,
            device_bytes: bytes,
            library,
            graph: BindingGraphs::new(library, "dspark_ffn")?,
        })
    }
}
impl<'a> DsparkFfn<'_, 'a> {
    pub(super) fn peer_stream(&self) -> Option<std::rc::Rc<crate::shared::memory::device::Stream<'a>>> {
        self.experts.peer_stream()
    }
    #[cfg(test)]
    pub(super) fn expert_diagnostics(&self)->[CuteafdDeviceBuffer;4] {
        let [hidden,ids,routing]=self.experts.inputs();
        [hidden,ids,routing,self.experts.output().expect("draft output")]
    }
    pub(super) fn stream(&self) -> Result<*mut c_void> {
        Ok(self.standalone.get()?.raw)
    }
    pub(super) fn synchronize(&self) -> Result<()> {
        self.standalone.drain().and(self.experts.synchronize())
    }
    pub(super) fn invalidate(&mut self) {
        self.boundary.invalidate();
    }
    /// The containing stage has drained its complete graph successfully.
    pub(super) unsafe fn complete_replay(&mut self, rows: u32) -> Result<[CuteafdDeviceBuffer; 2]> {
        unsafe { self.boundary.complete_replay(rows as usize) }
    }

    pub fn device_bytes(&self) -> usize {
        self.device_bytes
    }
    /// BF16 residual [capacity,4,5120] and incoming FP32 pre-mix [capacity,4].
    /// The incoming mix is the preceding attention sublayer's generated pre.
    pub fn inputs(&self) -> [CuteafdDeviceBuffer; 2] {
        self.boundary.inputs()
    }

    /// # Safety
    /// Initialize both inputs with finite states and valid pre-mix coefficients on
    /// this device, finish producer writes, and serialize all reuse of borrowed views.
    pub unsafe fn execute(&mut self, rows: u32) -> Result<[CuteafdDeviceBuffer; 2]> {
        let launched = unsafe { self.enqueue(rows) };
        let drained = self.synchronize();
        if let Err(error) = crate::families::deepseek_v41::v41_experts::dspark::settle(launched, drained) {
            self.boundary.invalidate();
            return Err(error);
        }
        unsafe { self.boundary.complete() }
    }
    /// Scratch owners and borrowed weights must live through stream completion.
    pub(super) unsafe fn enqueue(&mut self, rows: u32) -> Result<()> {
        unsafe { self.enqueue_on(rows, self.stream()?) }
    }
    pub(super) fn output_storage(&self) -> [CuteafdDeviceBuffer; 2] {
        self.boundary.output_storage()
    }
    /// The containing owner serializes and drains this supplied stream.
    pub(super) unsafe fn enqueue_on(&mut self, rows: u32, stream: *mut c_void) -> Result<()> {
        self.boundary.invalidate();
        let normalized = self.experts.inputs()[0];
        let result = self
            .experts
            .output()
            .context("dSpark FFN requires RTX output")?;
        unsafe {
            // Write normalization directly into the expert wave, avoiding D2D copies.
            self.boundary
                .enqueue_begin(rows as usize, Some(normalized), stream)?;
            self.experts
                .enqueue_draft_ffn_on(&mut self.router, &mut self.shared, rows, stream)?;
            self.boundary.enqueue_finish(Some(result), stream)
        }
    }
    /// # Safety
    /// Same initialized-input contract as execute; captures only owned addresses.
    pub unsafe fn capture(&mut self, rows: u32) -> Result<()> {
        unsafe {
            self.execute(rows)?;
        }
        self.boundary.invalidate();
        let key = rows;
        let Some(ticket) = self.graph.cache_ticket(&key, false, None) else { return Ok(()); };
        unsafe {
            self.library
                .cuda_graph_begin_capture(self.stream()?)?;
        }
        let launched = unsafe { self.enqueue(rows) };
        let captured = unsafe { self.library.cuda_graph_end_capture_topology(self.stream()?) };
        self.boundary.invalidate();
        match (launched, captured) {
            (Ok(()), Ok((graph, nodes, width, spans, inspection))) => {
                super::chain::warn_topology(inspection);
                // SAFETY: wave pins captured storage through drained teardown.
                unsafe { self.graph.cache_insert(key, graph, ticket, nodes as u64, width, spans || self.peer_stream().is_some())?; }
                Ok(())
            }
            (Err(error), Ok((graph, _, _, _, _))) => {
                if let Err(cleanup) = unsafe { self.library.cuda_graph_exec_destroy(graph) } {
                    tracing::error!(%cleanup,"destroying failed dSpark FFN capture");
                }
                Err(error)
            }
            (Err(first), Err(second)) => Err(crate::families::deepseek_v41::v41_experts::dspark::prefer_device(first, second)),
            (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// Same initialized-input contract as execute; rows must match capture.
    pub unsafe fn replay(&mut self, rows: u32) -> Result<[CuteafdDeviceBuffer; 2]> {
        self.boundary.invalidate();
        let Some(graph) = self.graph.cached_launch(&rows) else {
            // SAFETY: replay and eager execution have the same input contract.
            return unsafe { self.execute(rows) };
        };
        let launched = unsafe { self.library.cuda_graph_launch(graph, self.stream()?) };
        let drained = self.synchronize();
        crate::families::deepseek_v41::v41_experts::dspark::settle(launched, drained)?;
        unsafe { self.boundary.complete_replay(rows as usize) }
    }
    /// Completed residual streams and next pre-mix; invalid after a failed execute.
    pub fn output(&self) -> Result<[CuteafdDeviceBuffer; 2]> {
        self.boundary.output()
    }
}
impl Drop for DsparkFfn<'_, '_> {
    fn drop(&mut self) {
        fatal_drain(self.synchronize(), "dSpark ffn");
        // SAFETY: owning stream has drained before executable retirement.
        fatal_drain(unsafe { self.graph.clear() }, "dSpark ffn graphs");
    }
}

impl DsparkFfn<'_, '_> {
    pub(crate) fn visit_graph_banks(&mut self, f: &mut super::chain::GraphVisitor<'_>) -> Result<()> {
        f(self.standalone.device.id, "dspark_ffn", self.peer_stream().is_some(), &mut self.graph)?;
        Ok(())
    }
}
