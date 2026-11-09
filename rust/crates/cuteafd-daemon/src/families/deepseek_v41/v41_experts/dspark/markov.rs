use crate::shared::decode_graph::{fatal_drain, BindingGraphs};
use super::DsparkWeights;
use crate::shared::memory::Tag;
use crate::shared::memory::{DeviceAllocation};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, V41VocabularyProjection};
use std::ffi::c_void;

/// One sequential draft position's stable token, embedding and logits buffers.
/// Returned views are valid until this owner is reused; preserve embeddings for
/// the final confidence projection before advancing to the next draft position.
pub(crate) struct DsparkMarkov<'weights, 'library> {
    standalone: &'weights crate::shared::memory::StandaloneStream<'library>,
    kernel: V41VocabularyProjection<'library>,
    _workspace: DeviceAllocation<'library>,
    tokens: DeviceAllocation<'library>,
    embedding: DeviceAllocation<'library>,
    logits: DeviceAllocation<'library>,
    weights: &'weights DsparkWeights<'library>,
    graph: BindingGraphs<'library, usize>,
    capacity: usize,
    ready_rows: Option<usize>,
}
impl<'library> DsparkWeights<'library> {
    pub fn markov(
        &self,
        capacity: usize,
        available_device_bytes: usize,
    ) -> Result<DsparkMarkov<'_, 'library>> {
        let bytes = DsparkMarkov::device_bytes(capacity)?;
        ensure!(
            bytes <= available_device_bytes,
            "dSpark Markov exceeds device budget"
        );
        let library = self.library;
        self.tensor("mtp.2.markov_head.embed.weight")?;
        self.tensor("mtp.2.markov_head.head.weight")?;
        let workspace = DeviceAllocation::new(library, V41VocabularyProjection::WORKSPACE_BYTES, Tag::drafter_workspace("v41/draft-markov"))?;
        let kernel = unsafe { library.v41_dspark_markov(workspace.buffer)? };
        Ok(DsparkMarkov {
            kernel,
            _workspace: workspace,
            standalone: &self.standalone,
            tokens: DeviceAllocation::new(library, capacity * 4, Tag::drafter_workspace("v41/draft-markov"))?,
            embedding: DeviceAllocation::new(library, capacity * 512, Tag::drafter_workspace("v41/draft-markov"))?,
            logits: DeviceAllocation::new(library, capacity * 129280 * 4, Tag::drafter_workspace("v41/draft-markov"))?,
            weights: self,
            graph: BindingGraphs::new(library, "dspark_markov")?,
            capacity,
            ready_rows: None,
        })
    }
}
impl DsparkMarkov<'_, '_> {
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        ensure!(
            (1..=cuteafd_core::V41_MAX_LANE_MEMBERS).contains(&capacity),
            "Markov request capacity exceeds native lane capacity"
        );
        Ok(capacity * (4 + 512 + 129280 * 4) + V41VocabularyProjection::WORKSPACE_BYTES)
    }
    /// Preceding token IDs [capacity], u32, each less than 129280. Never free or
    /// retain after drop; finish producer writes before execute/replay.
    pub fn tokens(&self) -> CuteafdDeviceBuffer {
        self.tokens.buffer
    }
    fn validate_rows(&self, rows: usize) -> Result<()> {
        ensure!(
            rows > 0 && rows <= self.capacity,
            "markov rows exceed capacity"
        );
        Ok(())
    }
    unsafe fn enqueue(&self, rows: usize) -> Result<()> {
        unsafe { self.enqueue_on(rows, self.standalone.get()?.raw) }
    }
    pub(super) unsafe fn enqueue_on(&self, rows: usize, stream: *mut c_void) -> Result<()> {
        self.validate_rows(rows)?;
        unsafe {
            self.standalone.device.library.cuda_embedding_lookup_bf16_async(
                self.weights.tensor("mtp.2.markov_head.embed.weight")?,
                self.tokens.buffer,
                self.embedding.buffer,
                rows,
                129280,
                256,
                stream,
            )?;
            // BF16 checkpoint weights and embeddings are exact FP32 values;
            // accumulate and expose FP32 logits without a BF16 output rounding.
            self.kernel.launch(
                self.embedding.buffer,
                self.weights.tensor("mtp.2.markov_head.head.weight")?,
                self.logits.buffer,
                rows,
                stream,
            )
        }
    }
    pub(super) fn storage(&self) -> [CuteafdDeviceBuffer; 2] {
        [self.embedding.buffer, self.logits.buffer]
    }
    fn synchronize(&self) -> Result<()> {
        self.standalone.drain()
    }
    /// # Safety
    /// Owned token IDs must be initialized and less than 129280; producer work
    /// must be complete; input writes must not race this execution.
    pub unsafe fn execute(&mut self, rows: usize) -> Result<[CuteafdDeviceBuffer; 2]> {
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
                    tracing::error!(%cleanup, "destroying failed markov capture");
                }
                Err(error)
            }
            (Err(error), Err(_)) | (Ok(()), Err(error)) => Err(error),
        }
    }
    /// # Safety
    /// Same input contract as execute; rows must match the captured shape.
    pub unsafe fn replay(&mut self, rows: usize) -> Result<[CuteafdDeviceBuffer; 2]> {
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
    /// BF16 preceding-token embeddings and FP32 logits bias, in matching row
    /// order, borrowed until reuse/drop. No sampling or logits addition here.
    pub fn output(&self) -> Result<[CuteafdDeviceBuffer; 2]> {
        let rows = self.ready_rows.context("Markov output is not complete")?;
        let mut embedding = self.embedding.buffer;
        embedding.bytes = rows * 512;
        let mut logits = self.logits.buffer;
        logits.bytes = rows * 129280 * 4;
        Ok([embedding, logits])
    }
}
impl Drop for DsparkMarkov<'_, '_> {
    fn drop(&mut self) {
        fatal_drain(self.synchronize(), "dSpark markov");
        // SAFETY: owning stream has drained before executable retirement.
        fatal_drain(unsafe { self.graph.clear() }, "dSpark markov graphs");
    }
}

impl DsparkMarkov<'_, '_> {
    pub(crate) fn visit_graph_banks(&mut self, f: &mut super::chain::GraphVisitor<'_>) -> Result<()> {
        f(self.standalone.device.id, "dspark_markov", false, &mut self.graph)?;
        Ok(())
    }
}
