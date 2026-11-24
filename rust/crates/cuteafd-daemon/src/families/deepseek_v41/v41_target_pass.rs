//! One target pass from request-owned text/image rows through layer 39 and logits.
use crate::families::deepseek_v41::v41_backbone_execution::BackboneExecution;
use crate::families::deepseek_v41::v41_backbone_cache::CacheStage;
use crate::families::deepseek_v41::v41_block::{BlockOutput, EncoderSuffix};
use crate::families::deepseek_v41::v41_backbone_lane::BackboneLane;
use crate::families::deepseek_v41::v41_engram::{layer::EngramGate, EngramDeviceRows};
use crate::families::deepseek_v41::v41_experts::coordinator::NativeTp4Wave;
use crate::families::deepseek_v41::v41_index_lane::IndexLane;
use crate::families::deepseek_v41::v41_requests::{RequestBatch, Requests};
use crate::families::deepseek_v41::v41_target_embedding::TargetEmbeddingWave;
use crate::families::deepseek_v41::v41_target_head::{SampledTargetRows, TargetHeadWave, TargetLogits, TargetSamplingRowRequest};
use anyhow::{ensure, Context, Result};
use std::time::{Duration, Instant};
// Expand scheduling around concrete layer bodies: receive/FFN futures keep
// their original types and lexical borrows, including the direct C1 path.
macro_rules! layer_schedule {
    ($stage:expr; $layer:ident, $first:ident; { $($setup:tt)* }; $library:expr; $body:block) => {{
        let range = $stage.windows();
        let $first = range.start;
        for $layer in range {
            $($setup)*
            if $layer != $first && crate::shared::memory::chain::deferred() {
                crate::shared::memory::chain::fence_wait($library, ($layer - 1) % 2).await?;
                tokio::task::yield_now().await;
            }
            $body
        }
    }};
}

mod taps;
mod distributed;
pub(crate) use distributed::DistributedTargetPass;
mod verification;
pub(crate) use verification::VerificationTarget;
mod encoder_units;
pub(crate) use taps::{TargetTapStorage, TargetTapWave, TargetTaps};

/// Target cache publication used by the independent scheduler's queued dSpark
/// transaction. Static dispatch keeps ordinary serving on its existing path.
pub(crate) trait TargetCache<'a> {
    fn taps(&self, batch: &RequestBatch) -> Result<TargetTaps<'_>>;
    fn commit(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        accepted: &[u32]) -> Result<()>;
}
impl<'a> TargetCache<'a> for TargetPass<'_, 'a> {
    fn taps(&self, batch: &RequestBatch) -> Result<TargetTaps<'_>> { TargetPass::taps(self, batch) }
    fn commit(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        accepted: &[u32]) -> Result<()> { TargetPass::commit(self, requests, batch, accepted) }
}
impl<'a> TargetCache<'a> for DistributedTargetPass<'_, 'a> {
    fn taps(&self, batch: &RequestBatch) -> Result<TargetTaps<'_>> { DistributedTargetPass::taps(self, batch) }
    fn commit(&mut self, requests: &mut Requests<'a>, batch: &mut RequestBatch,
        accepted: &[u32]) -> Result<()> { DistributedTargetPass::commit(self, requests, batch, accepted) }
}

// A remote FFN wait owns its prepared lane state, never a request-bank borrow.
// Both ordinary serving and independent lane scheduling use the same execution
// body, so cache production and numerical operation order remain identical.
trait RequestAccess<'a> {
    fn cooperative_completion(&self) -> bool { false }
    fn with_requests<T>(&self, operation: impl FnOnce(&Requests<'a>) -> T) -> T;
}
impl<'a> RequestAccess<'a> for Requests<'a> {
    fn with_requests<T>(&self, operation: impl FnOnce(&Requests<'a>) -> T) -> T { operation(self) }
}
impl<'a> RequestAccess<'a> for std::cell::RefCell<&mut Requests<'a>> {
    fn cooperative_completion(&self) -> bool { true }
    fn with_requests<T>(&self, operation: impl FnOnce(&Requests<'a>) -> T) -> T {
        operation(&self.borrow())
    }
}

/// Which head terminal `execute_phase` should run for a decode/verification
/// pass. The compact greedy lane keeps its own terminal byte-for-byte; the
/// regular terminal returns the logits for the CPU path; the sampled terminal
/// runs the v4.1 GPU target-sampler after the head graph.
enum HeadTerminal<'m> {
    Regular,
    Greedy,
    Sampled {
        requests: &'m [TargetSamplingRowRequest],
        masks: Option<&'m [u32]>,
        mask_words: usize,
        /// Whether any row of this batch needs K3/K4/K5. The caller decides it
        /// per row (`SamplingRoute::is_ordered`); a purely greedy or purely
        /// fast-path batch keeps the chunk-1/chunk-2 two-stage sequence.
        ordered_rows: bool,
    },
}

#[derive(Default, Debug, PartialEq, Eq)]
enum State {
    #[default]
    Idle,
    Running,
    Ready(u64),
    Encoded(u64),
}
impl State {
    fn begin(&mut self) -> Result<()> {
        ensure!(
            *self == Self::Idle,
            "target pass must be committed or discarded before reuse"
        );
        *self = Self::Running;
        Ok(())
    }
    fn ready(&self, batch: u64) -> Result<()> {
        ensure!(
            *self == Self::Ready(batch),
            "target pass has no completed logits for this batch"
        );
        Ok(())
    }
}
/// Cancelling the async future must cancel mapped I/O and invalidate the batch.
struct BatchGuard<'a> {
    batch: &'a mut RequestBatch,
    completed: bool,
}
impl Drop for BatchGuard<'_> {
    fn drop(&mut self) {
        if !self.completed {
            self.batch.cancel();
        }
    }
}

/// Each alternating wave owns a separate instance; immutable weights are shared.
/// The CUDA-owning executor must poll this owner on one thread. Transport remains
/// external so a scheduler can assign a separate connection set to each wave.
pub(crate) struct Pass<'a, R> {
    resources: R,
    state: State,
    sampled: Option<SampledTargetRows>,
    chain: Option<crate::shared::memory::chain::StageChain<'a>>,
}

pub(crate) type TargetPass<'w, 'a> = Pass<'a, SingleResources<'w, 'a>>;

pub(crate) struct SingleResources<'w, 'a> {
    embedding: TargetEmbeddingWave<'w, 'a>,
    lane: BackboneLane<'w, 'a>,
    index: IndexLane<'w, 'a>,
    execution: BackboneExecution<'w, 'a>,
    upload: EngramDeviceRows<'a>,
    gates: [EngramGate<'w, 'a>; 2],
    head: TargetHeadWave<'w, 'a>,
    taps: TargetTapWave<'a>,
    engram_timeout: Duration,
}

impl<'a, R> Pass<'a, R> {
    /// Enter this pass's device-ordered stage chain when enabled.
    fn begin_chain(&mut self, enabled: bool) -> Result<Option<crate::shared::memory::chain::ChainHandle<'a>>> {
        if let Some(chain) = &self.chain {
            chain.set_device_order(true);
        }
        let Some(chain) = self.chain.as_ref().filter(|_| enabled) else { return Ok(None) };
        // Cancellation may have left work queued on this owner.
        chain.drain()?;
        Ok(Some(chain.handle()))
    }

    /// A prefill unit's device-ordered chain (shared executor): every hook of
    /// the unit polls in this scope, so its stages order on the device and no
    /// hook waits on the host for its own GPU or Spark work (#815). Drained
    /// with [`Self::drain_chain`] at the unit's end.
    pub(super) fn unit_chain(&self) -> Result<crate::shared::memory::chain::ChainHandle<'a>> {
        let chain = self.chain.as_ref().context("prefill units need the stage chain (CUTEAFD_STAGE_CHAIN)")?;
        chain.set_device_order(true);
        Ok(chain.handle())
    }
    /// The pass's native library (its stage chain's).
    pub(super) fn library(&self) -> Result<&'a cuteafd_ffi::NativeLibrary> {
        Ok(self.chain.as_ref().context("prefill units need the stage chain (CUTEAFD_STAGE_CHAIN)")?.library())
    }
    pub(super) fn drain_chain(&self) -> Result<()> {
        let chain = self.chain.as_ref().context("target pass chain absent")?;
        chain.drain_bounded(Duration::from_secs(60))
    }
}
impl<'w, 'a> TargetPass<'w, 'a> {
    pub(crate) fn capture_startup_prefill(&mut self, bank: &crate::families::deepseek_v41::v41_backbone_cache::BackboneCache<'a>, local: &[bool; 40]) -> Result<()> {
        ensure!(self.state == State::Idle, "startup prefill requires idle target pass");
        self.select_window(crate::shared::memory::row_window::RowWindow::NONE)?;
        let resources = &mut self.resources;
        let rows = resources.lane.startup_prefill_rows();
        resources.execution.capture_startup(bank, &rows)?;
        resources.index.capture_startup(&rows, |layer, row| resources.execution.startup_source(bank, layer, row))?;
        resources.lane.capture_startup_prefill(local)?;
        for gate in &mut resources.gates { gate.capture_startup(&rows)?; }
        resources.head.capture_startup(&crate::families::deepseek_v41::graph_policy::PREFILL_HEAD_PREWARM_ROWS)
    }
    pub(crate) fn capture_startup(&mut self, bank: &crate::families::deepseek_v41::v41_backbone_cache::BackboneCache<'a>, local: &[bool; 40]) -> Result<()> {
        ensure!(self.state == State::Idle, "startup graph capture requires idle target pass");
        let rows: Vec<_> = cuteafd_loader::serving_capacity::v41_graphs::v41_prewarm_rows(crate::families::deepseek_v41::graph_policy::ranks(), 0,
            crate::families::deepseek_v41::graph_policy::max_decode_rows() as usize).into_iter().map(|entry| entry.rows).collect();
        let resources = &mut self.resources;
        resources.execution.capture_startup(bank, &rows)?;
        resources.index.capture_startup(&rows, |layer, row| resources.execution.startup_source(bank, layer, row))?;
        let selections = resources.index.startup_selections();
        resources.lane.capture_startup(resources.execution.startup_sinks(), &selections, true, local)?;
        for gate in &mut resources.gates { gate.capture_startup(&rows)?; }
        resources.head.capture_startup(&rows)?;
        unsafe { // SAFETY: V4.1 graph owners borrow persistent engine storage only.
            crate::shared::decode_graph::device::observe_completed(resources.head.library(), |f| self.visit_graph_banks(f)) }
    }

    pub(crate) fn visit_graph_banks(&mut self, f: &mut dyn FnMut(i32, &'static str, bool, &mut dyn crate::shared::decode_graph::CacheAttached) -> Result<()>) -> Result<()> {
        self.resources.lane.visit_graph_banks(f)?;
        self.resources.index.visit_graph_banks(f)?;
        self.resources.execution.visit_graph_banks(f)?;
        for gate in &mut self.resources.gates { gate.visit_graph_banks(f)?; }
        self.resources.head.visit_graph_banks(f)
    }
    pub fn bind_execution_plan(&self, placement: &cuteafd_loader::placement::Placement) -> Result<()> {
        let plan = cuteafd_loader::placement::LayerExecutionPlan::new(placement,
            cuteafd_loader::placement::families::deepseek_v41::execution_hops(1))?;
        let map = crate::families::deepseek_v41::v41_backbone_cache::CachePlacement::from_execution_plan(&plan)?;
        ensure!((0..40).all(|layer| map.attention(layer).ok() == Some(0)) && plan.hops().is_empty(),
            "single RTX execution differs from admitted ownership");
        Ok(())
    }
    pub fn set_route_capture(&mut self, enabled: bool) {
        self.resources.lane.set_route_capture(enabled);
    }
    /// Route ids and/or pass timing for the next pass (see `BackboneLane::set_capture`).
    pub fn set_capture(&mut self, capture: crate::families::deepseek_v41::v41_backbone_lane::Capture) {
        self.resources.lane.set_capture(capture);
    }
    pub fn captured_routes(&self) -> &[Vec<[u32; 6]>] { self.resources.lane.captured_routes() }
    /// Host FFN stage split of the last captured pass.
    pub fn captured_ffn_split(&self) -> crate::families::deepseek_v41::v41_backbone_lane::FfnSplit { self.resources.lane.captured_ffn_split() }
    /// Device time per layer (FFN finish to FFN finish) of the last captured
    /// pass; layer 0 has no predecessor.
    pub fn captured_layer_us(&self) -> Vec<Option<f64>> {
        self.resources.lane.layer_clock().read()
    }
    pub fn install_attention_plan(&mut self, plan: cuteafd_core::coordinator_programs::AttentionReductionPlan) -> Result<()> {
        self.resources.lane.install_attention_plan(plan)
    }
    pub(crate) fn install_expert_usage(&mut self, usage: Option<std::rc::Rc<cuteafd_ffi::expert_usage::ExpertUsage<'a>>>) {
        self.resources.lane.install_expert_usage(usage);
    }
    pub fn set_attention_decode(&mut self, decode: bool) { self.resources.lane.set_attention_decode(decode); }
    pub fn reserve_sparse_decode_rows(&mut self, rows: usize) -> Result<()> {
        self.resources.lane.reserve_sparse_decode_rows(rows)
    }
    pub fn new(
        embedding: TargetEmbeddingWave<'w, 'a>,
        lane: BackboneLane<'w, 'a>,
        index: IndexLane<'w, 'a>,
        execution: BackboneExecution<'w, 'a>,
        upload: EngramDeviceRows<'a>,
        gates: [EngramGate<'w, 'a>; 2],
        head: TargetHeadWave<'w, 'a>,
        taps: TargetTapWave<'a>,
        engram_timeout: Duration,
    ) -> Result<Self> {
        ensure!(
            gates[0].layer() == 1 && gates[1].layer() == 14,
            "target engram gates out of order"
        );
        ensure!(!engram_timeout.is_zero(), "engram timeout must be positive");
        let chain = crate::shared::memory::chain::enabled()
            .then(|| crate::shared::memory::chain::StageChain::new(upload.library())).transpose()?;
        Ok(Self {
            resources: SingleResources {
                embedding,
                lane,
                index,
                execution,
                upload,
                gates,
                head,
                taps,
                engram_timeout,
            },
            state: State::Idle,
            sampled: None,
            chain,
        })
    }
    /// `--l2-prefetch` with `budget` bytes per layer (see [`BackboneLane::enable_l2_prefetch`]).
    pub fn enable_l2_prefetch(&mut self, budget: usize) -> Result<()> {
        self.resources.lane.enable_l2_prefetch(budget)
    }
    /// Selects this prefill slot's rows in every component's shared row planes
    /// (`shared::memory::row_window`), at rest: before a unit begins (its
    /// window) and before non-unit prefill work (`RowWindow::NONE`). The slot's
    /// previous work has drained ([`Pass::drain_chain`]).
    pub(crate) fn select_window(&mut self, window: crate::shared::memory::row_window::RowWindow) -> Result<()> {
        crate::shared::memory::row_window::require_rest(self.state != State::Idle, "prefill slot", "a pass is running or uncommitted")?;
        let resources = &mut self.resources;
        // At rest the previous unit's finished outputs are dropped (the next
        // use restarts every owner); pending work still refuses.
        resources.index.restart()?;
        resources.execution.restart();
        resources.lane.select_window(window.clone())?;
        resources.index.select_window(window.clone())?;
        resources.execution.select_window(window.clone())?;
        resources.embedding.select_window(window.clone())?;
        resources.upload.select_window(window.clone())?;
        for gate in &mut resources.gates { gate.select_window(window.clone())?; }
        resources.taps.select_window(window)
    }
    /// # Safety
    /// All components belong to the same device, capacity and official model.
    /// The caller exclusively owns CUDA buffers and polls on the owning thread.
    /// Selected rows are in this batch's flattened request order, at most 80.
    /// Produces private dSpark taps for every input row, including prepared image spans.
    pub async unsafe fn execute(
        &mut self,
        requests: &Requests<'a>,
        batch: &mut RequestBatch,
        transport: &mut NativeTp4Wave<'a>,
        placement: u64,
        selected: &[usize],
    ) -> Result<TargetLogits<'_>> {
        self.set_attention_decode(false);
        ensure!(batch.cache()?.stage() == CacheStage::Full, "ordinary target execute requires full phase");
        unsafe { self.execute_phase(requests, batch, transport, placement, selected, None, None, HeadTerminal::Regular).await?; }
        self.resources.head.output()
    }
    /// Full-logit decode uses the same fixed attention as device-scored verification.
    /// # Safety
    /// Same ownership and lifetime contract as `execute`.
    pub async unsafe fn execute_decode(&mut self, requests: &Requests<'a>, batch: &mut RequestBatch,
        transport: &mut NativeTp4Wave<'a>, placement: u64, selected: &[usize]) -> Result<TargetLogits<'_>> {
        self.set_attention_decode(true);
        ensure!(batch.cache()?.stage() == CacheStage::Full, "decode target execute requires full phase");
        unsafe { self.execute_phase(requests, batch, transport, placement, selected, None, None, HeadTerminal::Regular).await?; }
        self.resources.head.output()
    }
    /// Execute one full verification while allowing unrelated request commits
    /// during remote waits. The caller owns disjoint leases and drains queued
    /// lane work before releasing any request or device storage.
    pub async unsafe fn execute_shared(&mut self,
        requests: &std::cell::RefCell<&mut Requests<'a>>, batch: &mut RequestBatch,
        transport: &mut NativeTp4Wave<'a>, placement: u64, selected: &[usize],
    ) -> Result<TargetLogits<'_>> {
        self.set_attention_decode(true);
        ensure!(batch.cache()?.stage() == CacheStage::Full, "shared target execute requires full phase");
        unsafe { self.execute_phase(requests, batch, transport, placement, selected, None, None, HeadTerminal::Regular).await?; }
        self.resources.head.output()
    }
    pub async unsafe fn execute_greedy(&mut self, requests: &Requests<'a>, batch: &mut RequestBatch,
        transport: &mut NativeTp4Wave<'a>, placement: u64, selected: &[usize]) -> Result<Vec<(u32, f32)>> {
        self.set_attention_decode(true);
        ensure!(batch.cache()?.stage() == CacheStage::Full, "greedy target execute requires full phase");
        unsafe { self.execute_phase(requests, batch, transport, placement, selected, None, None, HeadTerminal::Greedy).await?; }
        self.resources.head.greedy_output()
    }
    pub async unsafe fn execute_shared_greedy(&mut self,
        requests: &std::cell::RefCell<&mut Requests<'a>>, batch: &mut RequestBatch,
        transport: &mut NativeTp4Wave<'a>, placement: u64, selected: &[usize]) -> Result<Vec<(u32, f32)>> {
        self.set_attention_decode(true);
        ensure!(batch.cache()?.stage() == CacheStage::Full, "shared greedy target execute requires full phase");
        unsafe { self.execute_phase(requests, batch, transport, placement, selected, None, None, HeadTerminal::Greedy).await?; }
        self.resources.head.greedy_output()
    }
    /// Run one verification pass and select every row on the device with the
    /// v4.1 target-sampler. The caller names the per-row parameters and, for
    /// constrained rows, the packed mask arena; `sampled_rows` then exposes the
    /// device's choices.
    pub async unsafe fn execute_sampled(&mut self, requests: &Requests<'a>,
        batch: &mut RequestBatch, transport: &mut NativeTp4Wave<'a>, placement: u64,
        selected: &[usize], sampling: &[TargetSamplingRowRequest], masks: Option<&[u32]>,
        mask_words: usize, ordered_rows: bool) -> Result<()> {
        self.set_attention_decode(true);
        ensure!(batch.cache()?.stage() == CacheStage::Full, "sampled target execute requires full phase");
        let terminal = HeadTerminal::Sampled { requests: sampling, masks, mask_words, ordered_rows };
        unsafe { self.execute_phase(requests, batch, transport, placement, selected, None, None, terminal).await?; }
        ensure!(self.sampled.is_some(), "sampled target pass published no rows");
        Ok(())
    }
    /// The cooperative (independent-lane) twin of [`Self::execute_sampled`].
    pub async unsafe fn execute_shared_sampled(&mut self,
        requests: &std::cell::RefCell<&mut Requests<'a>>, batch: &mut RequestBatch,
        transport: &mut NativeTp4Wave<'a>, placement: u64, selected: &[usize],
        sampling: &[TargetSamplingRowRequest], masks: Option<&[u32]>, mask_words: usize,
        ordered_rows: bool,
    ) -> Result<()> {
        self.set_attention_decode(true);
        ensure!(batch.cache()?.stage() == CacheStage::Full, "shared sampled target execute requires full phase");
        let terminal = HeadTerminal::Sampled { requests: sampling, masks, mask_words, ordered_rows };
        unsafe { self.execute_phase(requests, batch, transport, placement, selected, None, None, terminal).await?; }
        ensure!(self.sampled.is_some(), "shared sampled target pass published no rows");
        Ok(())
    }
    /// Take the device-selected rows published by [`Self::execute_sampled`].
    pub fn sampled_rows(&mut self) -> Result<SampledTargetRows> {
        self.sampled.take().context("target pass has no sampled rows")
    }
    /// Download full logits for a subset of the sampled rows, in the same order.
    pub async fn download_sampled_rows(&mut self, rows: &SampledTargetRows,
        selection: &[usize]) -> Result<Vec<u8>> {
        self.resources.head.download_sampled_rows(rows, selection).await
    }
    pub async unsafe fn execute_encoder(&mut self, requests: &Requests<'a>, batch: &mut RequestBatch,
        transport: &mut NativeTp4Wave<'a>, placement: u64, suffix: &mut EncoderSuffix<'a>) -> Result<()> {
        self.set_attention_decode(false);
        ensure!(batch.cache()?.stage() == CacheStage::Encoder, "encoder execute phase differs");
        unsafe { self.execute_phase(requests, batch, transport, placement, &[], Some(suffix), None, HeadTerminal::Regular).await }
    }
    pub async unsafe fn execute_encoder_replay(
        &mut self,
        requests: &Requests<'a>,
        batch: &mut RequestBatch,
        transport: &mut NativeTp4Wave<'a>,
        placement: u64,
        suffix: &mut EncoderSuffix<'a>,
    ) -> Result<()> {
        self.set_attention_decode(false);
        ensure!(
            batch.cache()?.stage() == CacheStage::EncoderReplay,
            "encoder replay phase differs"
        );
        unsafe {
            self.execute_phase(
                requests,
                batch,
                transport,
                placement,
                &[],
                Some(suffix),
                None,
                HeadTerminal::Regular,
            )
            .await
        }
    }
    pub async unsafe fn execute_replay(
        &mut self,
        requests: &Requests<'a>,
        batch: &mut RequestBatch,
        transport: &mut NativeTp4Wave<'a>,
        placement: u64,
        selected: &[usize],
        encoder: &BlockOutput<'_>,
    ) -> Result<TargetLogits<'_>> {
        self.set_attention_decode(false);
        ensure!(
            batch.cache()?.stage() == CacheStage::Replay,
            "decoder execute phase differs"
        );
        unsafe {
            self.execute_phase(
                requests,
                batch,
                transport,
                placement,
                selected,
                None,
                Some(encoder),
                HeadTerminal::Regular,
            )
            .await?;
        }
        self.resources.head.output()
    }
    async unsafe fn execute_phase(&mut self, requests: &impl RequestAccess<'a>, batch: &mut RequestBatch,
        transport: &mut NativeTp4Wave<'a>, placement: u64, selected: &[usize],
        suffix: Option<&mut EncoderSuffix<'a>>, encoder: Option<&BlockOutput<'_>>,
        terminal: HeadTerminal<'_>) -> Result<()> {
        let cycle = if !self.resources.lane.attention_decode() {
            crate::shared::decode_graph::device::Cycle::Prefill
        } else if batch.cache()?.positions().len() == batch.cache()?.request_ids().len() {
            crate::shared::decode_graph::device::Cycle::Decode
        } else { crate::shared::decode_graph::device::Cycle::Verify };
        super::graph_policy::cycle(cycle, |f| self.visit_graph_banks(f))?;
        // Only verification passes (greedy or device-sampled head) are chained.
        // Prefill, encoder and replay passes keep host-drained stages: their
        // many-row paths publish KV and encoder state through host-ordered copies.
        let verification = matches!(terminal, HeadTerminal::Greedy | HeadTerminal::Sampled { .. });
        let step_sample = if verification && tracing::enabled!(target: "cuteafd::target_step", tracing::Level::DEBUG) {
            Some((batch.cache()?.positions().len(), std::time::Instant::now()))
        } else { None };
        let Some(handle) = self.begin_chain(verification)? else {
            let result = unsafe { self.execute_phase_inner(requests, batch, transport, placement, selected,
                suffix, encoder, terminal).await };
            // A failed Spark wire released its waits over garbage planes.
            let result = result.and_then(|()| crate::shared::spark_waves::healthy());
            if result.is_ok() { self.resources.lane.drain_route_ring()?; }
            let measured = unsafe { // SAFETY: V4.1 graph owners borrow persistent engine storage only.
                crate::shared::decode_graph::device::observe_completed(
                self.resources.head.library(), |f| self.visit_graph_banks(f)) };
            return result.and(measured);
        };
        let result = crate::shared::memory::chain::watchdog(unsafe { handle.scope(self.execute_phase_inner(
            requests, batch, transport, placement, selected, suffix, encoder, terminal)) }).await;
        // Every consumer after the pass (commit, dSpark, logits downloads) is
        // unscoped, so the chained work must be complete before returning.
        let drained = self.drain_chain();
        // A failed Spark wire released its waits over garbage planes.
        let result = result.and(drained).and_then(|()| crate::shared::spark_waves::healthy());
        if result.is_ok() {
            // Device-read layers left their captured routes in the router's ring.
            self.resources.lane.drain_route_ring()?;
        }
        if let Some((rows, started)) = step_sample {
            tracing::debug!(target: "cuteafd::target_step", rows,
                eager=!super::graph_policy::captures_shape(rows as u32),
                elapsed_us=started.elapsed().as_micros() as u64, success=result.is_ok(), "V4.1 target verification step");
        }
        let measured = unsafe { // SAFETY: V4.1 graph owners borrow persistent engine storage only.
            crate::shared::decode_graph::device::observe_completed(
            self.resources.head.library(), |f| self.visit_graph_banks(f)) };
        result.and(measured)
    }
    async unsafe fn execute_phase_inner(&mut self, requests: &impl RequestAccess<'a>, batch: &mut RequestBatch,
        transport: &mut NativeTp4Wave<'a>, placement: u64, selected: &[usize],
        mut suffix: Option<&mut EncoderSuffix<'a>>, encoder: Option<&BlockOutput<'_>>,
        terminal: HeadTerminal<'_>) -> Result<()> {
        requests.with_requests(|requests| requests.validate(batch))?;
        let id = batch.cache()?.identity();
        let rows = batch.cache()?.positions().len();
        let stage = batch.cache()?.stage();
        let activation_trace = activation_trace_directory(&batch.cache()?.positions(),
            || format!("batch{id}-{stage:?}-{rows}rows"))?;
        ensure!(
            (stage.is_encoder() || !selected.is_empty())
                && selected.len() <= self.resources.head.capacity()
                && selected.iter().all(|&i| i < rows)
                && selected
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    == selected.len(),
            "invalid target head row selection"
        );
        self.state.begin()?;
        self.sampled = None;
        let mut guard = BatchGuard {
            batch,
            completed: false,
        };
        if !stage.is_encoder() {
            self.resources.taps.begin(guard.batch.cache()?)?;
        }
        self.resources.upload.reset()?;
        self.resources.execution.restart_for(stage);
        if stage == CacheStage::Replay {
            let encoder = encoder.context("missing retained encoder suffix")?;
            ensure!(encoder.tokens == guard.batch.cache()?.positions(), "encoder suffix/replay row order differs");
            self.resources.lane.restart_decoder(encoder)?;
            self.resources.index.restart_decoder()?;
            unsafe { if requests.cooperative_completion() { self.resources.lane.begin_prepared_cooperative().await?; }
                    else { self.resources.lane.begin_prepared()?; } }
        } else {
            self.resources.lane.restart()?;
            self.resources.index.restart()?;
            let text = if requests.cooperative_completion() {
                requests.with_requests(|requests| requests.text_embedding_input(guard.batch))?
            } else { None };
            if let Some((tokens, positions)) = text {
                unsafe { self.resources.lane.begin_tokens_cooperative(&mut self.resources.embedding, tokens, &positions).await?; }
            } else {
                unsafe { requests.with_requests(|requests| requests.begin_input(guard.batch, &mut self.resources.embedding, &mut self.resources.lane))?; }
            }
        }
        layer_schedule!(stage; layer, first; {}; self.resources.upload.library(); {

            if layer != first {
                let prepare_timing = Instant::now();
                self.resources.lane.advance()?;
                let advance_us = prepare_timing.elapsed().as_micros() as u64;
                if (2..14).contains(&layer) && !self.resources.upload.staged(1)
                    && self.resources.upload.can_enqueue(1)? {
                    if let cuteafd_loader::EngramGatherPoll::Ready(lease) = requests.with_requests(|r| r.poll_engram_layer(guard.batch, 1))? {
                        self.resources.upload.enqueue_lease(lease, guard.batch.engram_wave()?)?;
                    }
                }
                if let Some(gate) = [1, 14].iter().position(|&l| l == layer) {
                    if !self.resources.upload.staged(gate) {
                        guard.batch.wait_engram_gather(gate, self.resources.engram_timeout).await?;
                    self.resources.upload.wait_enqueue(gate, self.resources.engram_timeout).await?;
                    }
                    // Revalidate only after waking; no request-bank borrow spans
                    // admission or mapped I/O. Keep each route's upload/gate order.
                    if requests.cooperative_completion() || crate::shared::memory::chain::active() {
                        if !self.resources.upload.staged(gate) {
                            match requests.with_requests(|r| r.poll_engram_gather(guard.batch, &self.resources.lane))? {
                                cuteafd_loader::EngramGatherPoll::Ready(lease) => self.resources.upload.enqueue_lease(lease, guard.batch.engram_wave()?)?,
                                cuteafd_loader::EngramGatherPoll::Cancelled => anyhow::bail!("engram gather cancelled"),
                                cuteafd_loader::EngramGatherPoll::Pending => anyhow::bail!("engram gather wait returned pending"),
                            }
                        }
                        let rows = self.resources.upload.consume(gate)?;
                        unsafe { self.resources.lane.apply_engram_cooperative(&mut self.resources.gates[gate], &rows).await?; }
                        self.resources.upload.release_scratch()?;
                    } else {
                        crate::shared::memory::chain::settle(self.resources.upload.library())?;
                        ensure!(unsafe { requests.with_requests(|requests| requests.poll_engram(
                            guard.batch, &mut self.resources.upload, &mut self.resources.gates[gate],
                            &mut self.resources.lane))? }, "engram gather wait returned pending");
                    }
                }
                let engram_us = prepare_timing.elapsed().as_micros() as u64 - advance_us;
                if layer >= 37 {
                    unsafe {
                        if requests.cooperative_completion() {
                            self.resources.taps.capture_cooperative(guard.batch.cache()?, &self.resources.lane.prepared_input()?).await?;
                        } else { self.resources.taps.capture(guard.batch.cache()?, &self.resources.lane.prepared_input()?)?; }
                    }
                }
                let tapped_us = prepare_timing.elapsed().as_micros() as u64;
                unsafe {
                    if requests.cooperative_completion() { self.resources.lane.begin_prepared_cooperative().await?; }
                    else { self.resources.lane.begin_prepared()?; }
                }
                tracing::debug!(target: "cuteafd::timing", layer, rows, advance_us, engram_us, taps_us=tapped_us-advance_us-engram_us, begin_us=prepare_timing.elapsed().as_micros() as u64-tapped_us, "target layer preparation");
            }
            unsafe {
                let cooperative = requests.cooperative_completion();
                if cooperative {
                    let mut production = requests.with_requests(|requests| self.resources.execution.enqueue_production_and_index(
                        requests.cache(), guard.batch.cache()?, &self.resources.lane, &mut self.resources.index))?;
                    while !requests.with_requests(|requests| production.poll(requests.cache(), guard.batch.cache()?))? {
                        tokio::task::yield_now().await;
                    }
                }
                let prepared = requests.with_requests(|requests| {
                    if cooperative {
                        self.resources.execution.prepare_layer_cooperative(requests.cache(), guard.batch.cache()?,
                            &mut self.resources.lane, &mut self.resources.index)
                    } else {
                        self.resources.execution.prepare_layer(requests.cache(), guard.batch.cache()?,
                            &mut self.resources.lane, &mut self.resources.index)
                    }
                })?;
                // No RefCell guard or bank reference survives into this await.
                let completed = prepared.execute(transport, placement, guard.batch.image_mask()).await?;
                if cooperative {
                    self.resources.execution.complete_layer_cooperative(guard.batch.cache()?, &mut self.resources.lane, completed).await?;
                } else { self.resources.execution.complete_layer(guard.batch.cache()?, &mut self.resources.lane, completed)?; }
            }
            if let Some(directory) = &activation_trace {
                self.resources.lane.trace_output(directory)?;
            }
        });
        if stage.is_encoder() {
            suffix
                .as_mut()
                .context("missing encoder retention owner")?
                .capture(&self.resources.lane.output()?)?;
            if stage == CacheStage::Encoder {
                self.resources.lane.advance()?;
                unsafe {
                    if requests.cooperative_completion() { self.resources.lane.begin_prepared_cooperative().await?; }
                    else { self.resources.lane.begin_prepared()?; }
                    requests.with_requests(|requests| self.resources.execution.produce_decoder_source(
                        requests.cache(),
                        guard.batch.cache()?,
                        &self.resources.lane,
                    ))?;
                }
            }
            self.state = State::Encoded(id);
        } else {
            self.resources.taps.output(guard.batch.cache()?)?;
            let output = self.resources.lane.output()?;
            match terminal {
                HeadTerminal::Greedy => {
                    unsafe { self.resources.head.execute_block_greedy(&output, selected, requests.cooperative_completion()).await?; }
                }
                HeadTerminal::Regular => {
                    if requests.cooperative_completion() {
                        unsafe { self.resources.head.execute_block_cooperative(&output, selected).await?; }
                    } else {
                        unsafe { self.resources.head.execute_block(&output, selected)?; }
                    }
                }
                HeadTerminal::Sampled { requests: rows, masks, mask_words, ordered_rows } => {
                    let sampled = unsafe {
                        self.resources.head
                            .execute_block_sampled(&output, selected, rows, masks, mask_words,
                                ordered_rows, requests.cooperative_completion())
                            .await?
                    };
                    self.sampled = Some(sampled);
                }
            }
            self.state = State::Ready(id);
        }
        guard.completed = true;
        Ok(())
    }
    pub fn output(&self, batch: &RequestBatch) -> Result<TargetLogits<'_>> {
        self.state.ready(batch.cache()?.identity())?;
        self.resources.head.output()
    }
    pub async fn download_logits(&mut self, batch: &RequestBatch, rows: &[usize]) -> Result<Vec<u8>> {
        self.state.ready(batch.cache()?.identity())?;
        self.resources.head.download_rows(rows).await
    }
    pub fn taps(&self, batch: &RequestBatch) -> Result<TargetTaps<'_>> {
        self.state.ready(batch.cache()?.identity())?;
        self.resources.taps.output(batch.cache()?)
    }
    /// The scheduler samples/validates logits before publishing accepted input
    /// prefixes. A failed commit consumes this pass; discard before reuse.
    pub fn enqueue_cache_commit(&mut self, requests: &Requests<'a>, batch: &RequestBatch,
        accepted: &[u32]) -> Result<()> {
        let id = batch.cache()?.identity();
        self.state.ready(id)?;
        requests.validate_acceptance(batch, accepted)?;
        unsafe { self.resources.execution.enqueue_cache_commit(requests.cache(), batch.cache()?, accepted) }
    }
    pub fn poll_cache_commit(&self) -> Result<bool> { self.resources.execution.poll_cache_commit() }
    pub fn abort_cache_commit(&mut self, requests: &mut Requests<'a>) -> Result<()> {
        requests.abort_cache_commit(&mut self.resources.execution)
    }
    pub fn commit(
        &mut self,
        requests: &mut Requests<'a>,
        batch: &mut RequestBatch,
        accepted: &[u32],
    ) -> Result<()> {
        let id = batch.cache()?.identity();
        ensure!(self.state == State::Ready(id) || self.state == State::Encoded(id), "target commit phase incomplete");
        self.state = State::Running;
        self.resources.taps.reset();
        requests.commit(batch, &mut self.resources.execution, accepted)?;
        self.state = State::Idle;
        Ok(())
    }
    /// Publish the same accepted prefixes to dSpark, backbone and engram owners.
    /// Preflight errors retain the pass; failures after publication starts revoke
    /// every participating admission, including requests accepting zero rows.
    /// # Safety
    /// Proposal input belongs to this completed target batch. All windows and
    /// producer outputs are on this pass's CUDA device. Producer scratch is
    /// exclusive; reserved draft readers of disjoint window slots may remain active.
    pub unsafe fn commit_with_dspark(
        &mut self,
        requests: &mut Requests<'a>,
        batch: &mut RequestBatch,
        proposal: &mut crate::families::deepseek_v41::v41_experts::dspark::MainProposal<'_, '_, '_>,
        windows: &mut [&mut crate::families::deepseek_v41::v41_dspark_cache::DsparkWindow<'_>; 3],
        leases: [&[crate::families::deepseek_v41::v41_dspark_cache::WindowLease]; 3],
        accepted: &[u32],
    ) -> Result<()> {
        let id = batch.cache()?.identity();
        self.state.ready(id)?;
        requests.validate_acceptance(batch, accepted)?;
        proposal.validate_commit(id, windows, leases, accepted)?;
        self.state = State::Running;
        self.resources.taps.reset();
        let result = (|| -> Result<()> {
            unsafe {
                proposal.commit(id, windows, leases, accepted)?;
            }
            requests.commit(batch, &mut self.resources.execution, accepted)
        })();
        if let Err(error) = result {
            requests.revoke_batch(batch);
            for stage in 0..3 {
                for &lease in leases[stage] {
                    if windows[stage].request_id(lease).is_ok() {
                        if let Err(cleanup) = windows[stage].release(lease) {
                            tracing::error!(%cleanup, "releasing failed dSpark transaction");
                        }
                    }
                }
            }
            return Err(error);
        }
        self.state = State::Idle;
        Ok(())
    }
    /// Call only after execution future/borrowed outputs and transport consumers
    /// have been dropped. Request admission survives discarded private proposals.
    pub fn discard(&mut self, batch: &mut RequestBatch) -> Result<()> {
        self.sampled = None;
        batch.cancel();
        self.resources.taps.reset();
        self.state = State::Running;
        self.resources.lane.restart()?;
        self.resources.index.restart()?;
        self.resources.execution.restart();
        self.state = State::Idle;
        Ok(())
    }
}

/// The debug activation trace's batch directory, created with its positions;
/// `None` unless the trace target is on and the trace position selects this batch.
fn activation_trace_directory(positions: &[u64], name: impl FnOnce() -> String)
    -> Result<Option<std::path::PathBuf>> {
    if !tracing::enabled!(target: "cuteafd::activation_trace", tracing::Level::DEBUG) { return Ok(None) }
    let Some(directory) = crate::shared::activation_trace::batch_directory(positions.first().copied(), name)?
        else { return Ok(None) };
    std::fs::create_dir_all(&directory)?;
    std::fs::write(directory.join("positions.json"), serde_json::to_vec(positions)?)?;
    Ok(Some(directory))
}

#[cfg(test)]
mod tests {
    use super::State;

    #[test]
    fn debug_logging_without_trace_variables_runs_untraced() -> anyhow::Result<()> {
        if std::env::var_os("CUTEAFD_ACTIVATION_TRACE_POSITION").is_some() { return Ok(()) }
        let subscriber = tracing_subscriber::fmt().with_max_level(tracing::Level::DEBUG)
            .with_test_writer().finish();
        tracing::subscriber::with_default(subscriber, || {
            assert!(tracing::enabled!(target: "cuteafd::activation_trace", tracing::Level::DEBUG));
            assert!(super::activation_trace_directory(&[0, 1, 2], || unreachable!("no trace selected"))?.is_none());
            Ok(())
        })
    }
    fn cpu_library() -> &'static cuteafd_ffi::NativeLibrary {
        unreachable!("CPU schedules do not enter a device chain")
    }

    #[test]
    fn concrete_topologies_share_stage_order_without_receive_dispatch() -> anyhow::Result<()> {
        use super::CacheStage;
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        for stage in [CacheStage::Full, CacheStage::Encoder, CacheStage::EncoderReplay, CacheStage::Replay] {
            let rows = runtime.block_on(async {
                let mut visited = Vec::new();
                layer_schedule!(stage; layer, first; {
                    visited.push((layer, "owner"));
                }; cpu_library(); {
                    if layer != first { visited.push((layer, "advance")); }
                    visited.push((layer, "receive"));
                    visited.push((layer, "complete"));
                });
                anyhow::Ok(visited)
            })?;
            let mut expected = Vec::new();
            for layer in stage.windows() {
                expected.push((layer, "owner"));
                if layer != stage.windows().start { expected.push((layer, "advance")); }
                expected.push((layer, "receive"));
                expected.push((layer, "complete"));
            }
            assert_eq!(rows, expected);
        }
        Ok(())
    }

    #[test]
    fn placed_lane_interleavings_drain_owner_reduce_and_direct_hops() -> anyhow::Result<()> {
        use crate::shared::peer_split::order::{check, host::{check_host, HostSchedule}};
        use super::CacheStage;
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        for stage in [CacheStage::Full, CacheStage::Encoder, CacheStage::EncoderReplay, CacheStage::Replay] {
            // Record the real scheduler's layer order, then interleave independent
            // lanes at its await boundaries. Each direct-input copy and owner
            // reduction queues its producer before its destination wait.
            let layers = runtime.block_on(async {
                let mut layers = Vec::new();
                layer_schedule!(stage; layer, first; {}; cpu_library(); {
                    layers.push((layer, first));
                });
                anyhow::Ok(layers)
            })?;
            for placed in [false, true] {
                for leading in 0..2 {
                    for chunk in [1, 2, 7, 20, 40] {
                        let mut schedule = HostSchedule::default();
                        let mut next = [0; 2];
                        while next.iter().any(|&at| at < layers.len()) {
                            for lane in [leading, 1 - leading] {
                                for _ in 0..chunk {
                                    let Some(&(layer, first)) = layers.get(next[lane]) else { break };
                                    next[lane] += 1;
                                    let owner = usize::from(placed && layer >= 20);
                                    if placed && layer == 20 {
                                        // Replay imports the retained encoder suffix before
                                        // its first layer; full passes advance across L19/20.
                                        assert!(layer != first || stage == CacheStage::Replay);
                                        schedule.push_to(0, "direct-input", 0, lane, "residual and pre ready");
                                        schedule.wait_for(1, "direct-input", 0, lane, "land block inputs");
                                        schedule.sync(1, "direct-input copy complete");
                                    }
                                    if placed {
                                        schedule.push_to(1 - owner, "owner-reduce", layer, lane, "remote expert partial ready");
                                        schedule.wait_for(owner, "owner-reduce", layer, lane, "owner consumes remote partial");
                                        schedule.sync(owner, "owner reduction complete");
                                    }
                                }
                            }
                        }
                        assert_eq!(check(&schedule.streams()), Ok(()),
                            "stage={stage:?} placed={placed} leading={leading} chunk={chunk}");
                        assert_eq!(check_host(&schedule), Ok(()),
                            "stage={stage:?} placed={placed} leading={leading} chunk={chunk}");
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn pass_lifecycle_is_independent_of_concrete_resources() -> anyhow::Result<()> {
        struct CpuResources;
        let mut pass = super::Pass {
            resources: CpuResources,
            state: State::Idle,
            sampled: None,
            chain: None,
        };
        assert!(pass.begin_chain(false)?.is_none());
        assert!(pass.begin_chain(true)?.is_none());
        assert!(pass.drain_chain().is_err());
        pass.state.begin()?;
        assert!(pass.state.begin().is_err());
        Ok(())
    }

    #[test]
    fn two_lane_chains_stay_device_ordered_and_isolated() -> anyhow::Result<()> {
        use crate::shared::memory::chain::{self, StageChain};
        let (_directory, fixture) = crate::shared::memory::cpu_stream_fixture()?;
        let library = fixture.load()?;
        let mut passes = [
            super::Pass { resources: (), state: State::Idle, sampled: None, chain: Some(StageChain::new(&library)?) },
            super::Pass { resources: (), state: State::Idle, sampled: None, chain: Some(StageChain::new(&library)?) },
        ];
        // Even if a prior caller flipped a chain off, begin_chain restores device order.
        passes[1].chain.as_ref().unwrap().set_device_order(false);
        let h0 = passes[0].begin_chain(true)?.expect("lane 0 chain");
        let h1 = passes[1].begin_chain(true)?.expect("lane 1 chain");
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        runtime.block_on(async {
            let (a, b) = tokio::join!(
                h0.scope(async {
                    assert!(chain::active() && chain::deferred());
                    tokio::task::yield_now().await;
                    assert!(chain::active() && chain::deferred());
                    anyhow::Ok(())
                }),
                h1.scope(async {
                    assert!(chain::active() && chain::deferred());
                    tokio::task::yield_now().await;
                    assert!(chain::active() && chain::deferred());
                    anyhow::Ok(())
                }),
            );
            a?; b?;
            anyhow::Ok(())
        })?;
        for pass in &passes { pass.drain_chain()?; }
        let independent = include_str!("v41_native_serve/scheduler/independent.rs");
        let independent = &independent[..independent.find("#[cfg(test)]").unwrap()];
        assert!(!independent.contains("set_device_order"));
        assert!(!independent.contains("set_l2_prefetch"));
        let lane = include_str!("v41_backbone_lane.rs");
        let lane = &lane[..lane.find("#[cfg(test)]").unwrap()];
        assert!(!lane.contains("set_l2_prefetch"));
        let router = include_str!("v41_backbone_router.rs");
        let set_local = &router[router.find("pub fn set_local_mode(").unwrap()..];
        let set_local = &set_local[..set_local.find("pub fn device_bytes(").unwrap()];
        assert!(set_local.contains("if !crate::shared::memory::chain::active()"));
        let index_query = include_str!("v41_index_query.rs");
        let rebind = &index_query[index_query.find("pub fn rebind(").unwrap()..];
        let rebind = &rebind[..rebind.find("pub(crate) fn startup_geometry(").unwrap()];
        assert!(rebind.contains("if !crate::shared::memory::chain::active()"));
        Ok(())
    }

    #[test]
    fn incomplete_foreign_and_consumed_passes_cannot_publish() {
        let mut state = State::default();
        assert!(state.ready(1).is_err());
        state.begin().unwrap();
        assert!(state.begin().is_err());
        assert!(state.ready(1).is_err());
        state = State::Ready(1);
        state.ready(1).unwrap();
        assert!(state.ready(2).is_err());
        assert!(state.begin().is_err());
        state = State::Idle;
        assert!(state.ready(1).is_err());
        state.begin().unwrap();
    }
}

#[cfg(test)]
mod distributed_tests;
