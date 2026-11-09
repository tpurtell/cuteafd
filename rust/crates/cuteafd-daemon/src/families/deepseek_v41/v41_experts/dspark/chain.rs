//! Draft transformer chain with optional shared embedding and terminal heads.
use crate::shared::decode_graph::{fatal_drain, GraphBank, GraphOwner};
use super::{DsparkStage, DsparkTerminal, DsparkWeights};
use crate::families::deepseek_v41::v41_dspark_cache::{DsparkWindow, WindowGeometry, WindowLease, WindowRead};
use crate::shared::memory::Tag;
use crate::shared::memory::{DeviceAllocation, HostAllocation, LoadStream};
use crate::families::deepseek_v41::v41_tensors::{NativeRtxTensors, VocabularyHead};
use anyhow::{ensure, Context, Result};
use cuteafd_core::DsparkRng;
use cuteafd_ffi::{CuteafdDeviceBuffer, V41AttentionOps};
use std::ffi::c_void;
mod distributed;
pub(crate) use distributed::DistributedDsparkChain;

pub(crate) type GraphVisitor<'f> = dyn FnMut(i32, &'static str, bool,
    &mut dyn crate::shared::decode_graph::CacheAttached) -> Result<()> + 'f;

pub(super) fn warn_topology(error: Option<String>) {
    if let Some(error) = error {
        tracing::warn!(%error, "dSpark graph topology unavailable; disabling chain reuse");
    }
}

/// Whether an error is a device/driver failure (a native-library status return)
/// rather than a host-side invariant failure. Only device failures are terminal
/// for a draft chain: a CPU-side guard must not latch the lane.
pub(crate) fn is_device_failure(error: &anyhow::Error) -> bool {
    error.chain().any(|cause| cause.is::<cuteafd_ffi::NativeStatusError>())
}

/// Keep the device-tagged cause when only one side has it, and carry the other
/// as context so neither failure is lost.
pub(crate) fn prefer_device(first: anyhow::Error, second: anyhow::Error) -> anyhow::Error {
    if is_device_failure(&second) && !is_device_failure(&first) {
        second.context(format!("first failure: {first:#}"))
    } else {
        first.context(format!("second failure: {second:#}"))
    }
}

/// Combine a launch result with the drain or capture result that follows it.
/// Both are inspected: a plain launch error must not hide a tagged device error
/// from the drain, which is what the liveness gate classifies on.
pub(crate) fn settle(launched: Result<()>, drained: Result<()>) -> Result<()> {
    match (launched, drained) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(first), Err(second)) => Err(prefer_device(first, second)),
    }
}

/// A sticky terminal failure for a draft chain. A CUDA or replay error is not
/// recoverable in-process, so the lane must fail the round instead of falling
/// back to anchors forever with the error masked. Host-side invariant failures
/// are deliberately not recorded: they are recoverable and must not latch.
#[derive(Default)]
pub(crate) struct DraftFailure(Option<String>);
impl DraftFailure {
    pub(crate) fn record(&mut self, error: &anyhow::Error) {
        if self.0.is_none() && is_device_failure(error) { self.0 = Some(format!("{error:#}")); }
    }
    pub(crate) fn check(&self, what: &str) -> Result<()> {
        match &self.0 {
            Some(reason) => anyhow::bail!("{what} failed until restart: {reason}"),
            None => Ok(()),
        }
    }
    #[cfg(test)]
    pub(crate) fn is_failed(&self) -> bool { self.0.is_some() }
}

pub(crate) struct DsparkChain<'weights, 'library> {
    stream: LoadStream<'library>,
    timing: crate::shared::draft::timing::DraftTiming<'library>,
    /// Live draft width; storage and the maximum are fixed at load time.
    width: usize,
    maximum: usize,
    stages: [DsparkStage<'weights, 'library>; 3],
    /// Captured complete drafts keyed by (request count, width).
    graphs: GraphBank<(usize, usize), GraphOwner<'library, [u64; 3]>>,
    ready: Option<usize>,
    pending: Option<PendingDraft<'library>>,
    download: HostAllocation<'library>,
    embedding: Option<&'weights NativeRtxTensors<'library>>,
    tokens: DeviceAllocation<'library>,
    token_staging: HostAllocation<'library>,
    token_count: Option<usize>,
    terminal: Option<DsparkTerminal<'weights, 'library>>,
    ops: V41AttentionOps<'library>,
    /// A CUDA/replay failure is terminal for this chain.
    failure: DraftFailure,
}
/// Own reservations before the first upload. Errors and unwinding drain before
/// pinned inputs or cache ring slots may be reused. Successful polls disarm it.
struct PendingDraft<'a> {
    count: usize,
    reads: [WindowRead; 3],
    warming: bool,
    armed: bool,
    library: &'a cuteafd_ffi::NativeLibrary,
    stream: *mut c_void,
    peers: [Option<std::rc::Rc<crate::shared::memory::device::Stream<'a>>>; 3],
}
impl Drop for PendingDraft<'_> {
    fn drop(&mut self) {
        if self.armed {
            // A partial enqueue may fail before the root joins the peer. Drain
            // every peer even when root synchronization reports an error.
            let mut drained = unsafe { self.library.cuda_stream_synchronize(self.stream) };
            for peer in self.peers.iter().flatten() {
                let result = peer.drain();
                drained = drained.and(result);
            }
            fatal_drain(drained, "pending draft readers");
        }
    }
}
impl<'library> DsparkWeights<'library> {
    pub fn draft_bytes(&self, requests: u32) -> Result<usize> {
        self.chain_bytes(requests)?
            .checked_add(DsparkTerminal::device_bytes_with_width(requests as usize, self.draft_width)?)
            .context("dSpark complete draft budget overflow")
    }
    /// Complete seed embedding, three-stage transformer and terminal proposal
    /// graph. Shared coordinator embedding and vocabulary owners are borrowed.
    pub fn draft<'weights>(
        &'weights self,
        embedding: &'weights NativeRtxTensors<'library>,
        head: &'weights VocabularyHead<'library>,
        requests: u32,
        budget: usize,
    ) -> Result<DsparkChain<'weights, 'library>> {
        ensure!(
            self.draft_bytes(requests)? <= budget,
            "dSpark complete draft exceeds budget"
        );
        let mut chain = self.embedded_chain(embedding, requests, self.chain_bytes(requests)?)?;
        chain.terminal = Some(self.terminal(
            head,
            requests as usize,
            DsparkTerminal::device_bytes_with_width(requests as usize, self.draft_width)?,
        )?);
        Ok(chain)
    }

    /// Borrow the coordinator's existing shared embedding table, never duplicate it.
    pub fn embedded_chain<'weights>(
        &'weights self,
        embedding: &'weights NativeRtxTensors<'library>,
        requests: u32,
        budget: usize,
    ) -> Result<DsparkChain<'weights, 'library>> {
        ensure!(
            embedding.get("embed.weight")?.bytes == 129280 * 5120 * 2,
            "invalid shared embedding extent"
        );
        let mut chain = self.chain(requests, budget)?;
        let table_device = embedding.get("embed.weight")?.device_id;
        if table_device != chain.tokens.buffer.device_id {
            // Plan peer access once. Draft graphs only read the shared table;
            // tokens, residuals and all transformer work stay on this device.
            chain.stream.library.cuda_enable_peer(table_device)?;
        }
        chain.embedding = Some(embedding);
        Ok(chain)
    }

    pub fn chain_bytes(&self, requests: u32) -> Result<usize> { self.shape().chain_bytes(requests) }
    pub fn chain(&self, requests: u32, budget: usize) -> Result<DsparkChain<'_, 'library>> {
        ensure!(
            self.chain_bytes(requests)? <= budget,
            "dSpark chain exceeds budget"
        );
        let library = self.library;
        let bytes = self.stage_bytes(requests)?;
        Ok(DsparkChain {
            timing: crate::shared::draft::timing::DraftTiming::from_env(library, "v41-dspark")?,
            width: self.draft_width,
            maximum: self.draft_width,
            stream: LoadStream {
                library,
                raw: library.cuda_stream_create()?,
            },
            stages: [
                self.stage(0, requests, bytes)?,
                self.stage(1, requests, bytes)?,
                self.stage(2, requests, bytes)?,
            ],
            graphs: GraphBank::new(None),
            ready: None,
            pending: None,
            download: HostAllocation::new(library, requests as usize * (2 * self.draft_width + 1) * 4, Tag::staging("v41/draft-chain"))?,
            embedding: None,
            tokens: DeviceAllocation::new(library, requests as usize * 4, Tag::drafter_workspace("v41/draft-chain"))?,
            token_staging: HostAllocation::new(library, requests as usize * 4, Tag::staging("v41/draft-chain"))?,
            token_count: None,
            terminal: None,
            ops: library.v41_attention_ops_width(self.draft_width)?,
            failure: DraftFailure::default(),
        })
    }
}
impl DsparkChain<'_, '_> {
    /// Reserve a fresh attempt in packed seed/cache row order. Failed attempts
    /// consume admitted ranges; replay reuses them until prepared again.
    pub fn prepare_sampling(
        &mut self,
        rngs: &mut [&mut DsparkRng],
        temperatures: &[f32],
    ) -> Result<()> {
        ensure!(self.pending.is_none(), "dSpark chain replay still pending");
        self.invalidate();
        self.terminal
            .as_mut()
            .context("dSpark chain has no terminal")?
            .prepare_sampling(rngs, temperatures)
    }
    pub fn stage_sampling(&mut self, rngs: &mut [&mut DsparkRng], temperatures: &[f32]) -> Result<()> {
        ensure!(self.pending.is_none(), "dSpark chain replay still pending");
        self.invalidate();
        self.terminal.as_mut().context("dSpark chain has no terminal")?
            .stage_sampling(rngs, temperatures)
    }
    pub(crate) fn capture_startup(&mut self, windows: [&DsparkWindow<'_>; 3]) -> Result<()> {
        ensure!(self.pending.is_none() && self.ready.is_none(), "draft startup requires idle owner");
        self.synchronize()?;
        let geometry = windows.map(DsparkWindow::startup_geometry);
        ensure!(geometry.iter().all(|g| g.ring.device_id == self.tokens.buffer.device_id),
            "draft startup ring device differs");
        let counts = self.tokens.buffer.bytes / 4;
        let widths: Vec<_> = [5, 7].into_iter().filter(|&w| w <= self.maximum).collect();
        let original = self.width;
        let recorded = (|| -> Result<()> {
            for width in widths {
                self.set_width(width)?;
                for count in 1..=counts {
                    // SAFETY: real ring/weight owners stay pinned; capture only
                    // records commands and never reads uninitialized metadata.
                    unsafe { self.capture_geometry(&geometry, count)?; }
                }
            }
            Ok(())
        })();
        let restored = self.set_width(original);
        recorded.and(restored)
    }
    pub fn has_graph(&self, count: usize) -> bool { self.graphs.contains(&(count, self.width)) }
    pub fn width(&self) -> usize { self.width }
    /// Select the draft width (5 or 7, within the loaded maximum) for the next
    /// draft. All storage is sized for the maximum; each width keeps its own
    /// captured graphs, so switching is free after both are warm.
    pub fn set_width(&mut self, width: usize) -> Result<()> {
        ensure!(self.pending.is_none(), "dSpark chain replay still pending");
        ensure!(matches!(width, 5 | 7) && width <= self.maximum, "draft width must be five or seven within the loaded width");
        if width == self.width { return Ok(()); }
        self.invalidate();
        for stage in &mut self.stages { stage.set_width(width, self.maximum)?; }
        if let Some(terminal) = &mut self.terminal { terminal.set_width(width)?; }
        self.ops = self.stream.library.v41_attention_ops_width(width)?;
        self.width = width;
        Ok(())
    }
    /// Borrowed anchor + K tokens [K+1,R], corrected raw logits [K,R,V] and
    /// raw confidence [K,R], live until reuse/drop. No target history is committed.
    pub fn draft_output(&self) -> Result<[CuteafdDeviceBuffer; 3]> {
        let requests = self.ready.context("dSpark draft output incomplete")?;
        self.terminal
            .as_ref()
            .context("dSpark chain has no terminal")?
            .output_storage(requests)
    }

    /// Seed IDs follow cache-binding row order; invalid input clears publication
    /// and token readiness so a subsequent execution cannot consume old IDs.
    pub fn set_tokens(&mut self, tokens: &[i32]) -> Result<()> {
        self.stage_tokens(tokens)?;
        let uploaded = self.stream.library.copy_h2d(self.tokens.buffer,
            &self.token_staging.bytes_mut()[..tokens.len() * 4]);
        if uploaded.is_err() { self.token_count = None; }
        uploaded
    }
    /// CPU-only seed staging; pending work excludes writes to this pinned owner.
    pub fn stage_tokens(&mut self, tokens: &[i32]) -> Result<()> {
        ensure!(self.pending.is_none(), "dSpark chain replay still pending");
        self.invalidate();
        self.token_count = None;
        ensure!(
            self.embedding.is_some(),
            "dSpark chain has no shared embedding binding"
        );
        ensure!(
            !tokens.is_empty()
                && tokens.len() <= cuteafd_core::V41_MAX_LANE_MEMBERS
                && tokens.len() <= self.tokens.buffer.bytes / 4,
            "invalid dSpark seed count"
        );
        ensure!(
            tokens.iter().all(|&id| (0..129280).contains(&id)),
            "invalid dSpark seed token"
        );
        let bytes = self.token_staging.bytes_mut();
        for (i, id) in tokens.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&id.to_ne_bytes());
        }
        self.token_count = Some(tokens.len());
        Ok(())
    }

    pub fn inputs(&self) -> [CuteafdDeviceBuffer; 2] {
        self.stages[0].inputs()
    }
    fn invalidate(&mut self) {
        self.ready = None;
        for stage in &mut self.stages {
            stage.invalidate();
        }
    }
    fn synchronize(&self) -> Result<()> {
        let mut drained = unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) };
        for stage in &self.stages {
            let result = stage.synchronize();
            drained = drained.and(result);
        }
        drained
    }
    fn prepare(
        &mut self,
        windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3],
    ) -> Result<[WindowRead; 3]> {
        ensure!(self.pending.is_none(), "dSpark chain replay still pending");
        self.invalidate();
        ensure!(
            self.embedding.is_none() || self.token_count == Some(bindings[0].len()),
            "dSpark seed count differs from cache bindings"
        );
        if let Some(terminal) = &self.terminal {
            terminal.validate_sampling(bindings[0].len())?;
        }
        let reads = [
            self.stages[0].prepare(windows[0], bindings[0])?,
            self.stages[1].prepare(windows[1], bindings[1])?,
            self.stages[2].prepare(windows[2], bindings[2])?,
        ];
        ensure!(
            reads[0].owner != reads[1].owner
                && reads[0].owner != reads[2].owner
                && reads[1].owner != reads[2].owner,
            "dSpark stages require independent cache owners"
        );
        for stage in 1..3 {
            ensure!(
                bindings[stage].len() == bindings[0].len(),
                "dSpark chain stage counts differ"
            );
            for (&(first, end), &(lease, stage_end)) in bindings[0].iter().zip(bindings[stage]) {
                ensure!(
                    end == stage_end
                        && windows[0].request_id(first)? == windows[stage].request_id(lease)?,
                    "dSpark chain request or committed position differs"
                );
            }
        }
        Ok(reads)
    }
    fn upload(
        &mut self,
        reads: &[WindowRead; 3],
        bindings: [&[(WindowLease, u64)]; 3],
    ) -> Result<()> {
        for stage in 0..3 {
            self.stages[stage].upload(&reads[stage], bindings[stage])?;
        }
        Ok(())
    }
    unsafe fn enqueue(&mut self, reads: &[WindowRead; 3], requests: usize) -> Result<()> {
        unsafe { self.enqueue_geometry(&reads.each_ref().map(WindowRead::geometry), requests) }
    }
    unsafe fn enqueue_geometry(&mut self, reads: &[WindowGeometry; 3], requests: usize) -> Result<()> {
        let forward = self.timing.begin_retained("forward", "forward", None, requests * self.width, self.stream.raw)?;
        if let Some(embedding) = self.embedding {
            let output = self.inputs();
            unsafe {
                self.ops.embed(
                    embedding.get("embed.weight")?,
                    self.tokens.buffer,
                    output[0],
                    output[1],
                    requests as u32,
                    self.stream.raw,
                )?;
            }
        }
        for stage in 0..3 {
            if stage > 0 {
                let source = self.stages[stage - 1].output_storage();
                let destination = self.stages[stage].inputs();
                unsafe {
                    self.stream.library.copy_d2d_async(
                        destination[0],
                        source[0],
                        requests * self.width * 40960,
                        self.stream.raw,
                    )?;
                    self.stream.library.copy_d2d_async(
                        destination[1],
                        source[1],
                        requests * self.width * 16,
                        self.stream.raw,
                    )?;
                }
            }
            let layer = self.timing.begin_retained("forward", "layer", Some(stage), requests * self.width, self.stream.raw)?;
            unsafe {
                self.stages[stage].enqueue_geometry(&reads[stage], requests, self.stream.raw)?;
            }
            self.timing.end_retained(layer, self.stream.raw)?;
        }
        self.timing.end_retained(forward, self.stream.raw)?;
        if let Some(terminal) = &self.terminal {
            let source = self.stages[2].output_storage();
            let target = terminal.inputs();
            unsafe {
                self.ops.terminal_layout(
                    source[0],
                    source[1],
                    target[0],
                    target[1],
                    requests as u32,
                    self.stream.raw,
                )?;
                self.stream.library.copy_d2d_async(
                    target[2],
                    self.tokens.buffer,
                    requests * 4,
                    self.stream.raw,
                )?;
                terminal.enqueue_on(requests, self.stream.raw)?;
            }
        }
        Ok(())
    }
    /// # Safety
    /// For a plain chain, initialize finite BF16 residuals and FP32 pre-mix in
    /// input row order. An embedded chain initializes these from set_tokens.
    /// Cache bindings across stages describe the same requests, in seed order.
    /// For a complete draft, prepare sampling in the same packed request order.
    /// All buffers are on this device and serialized through graph completion.
    pub unsafe fn execute(
        &mut self,
        windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3],
    ) -> Result<[CuteafdDeviceBuffer; 2]> {
        let reads = self.prepare(windows, bindings)?;
        self.upload(&reads, bindings)?;
        let launched = unsafe { self.enqueue(&reads, bindings[0].len()) };
        let drained = self.synchronize();
        if let Err(error) = settle(launched, drained) {
            self.invalidate();
            return Err(error);
        }
        self.ready = Some(bindings[0].len());
        self.output()
    }
    /// # Safety
    /// Same contract as execute; captures all three independent window owners.
    pub unsafe fn capture(
        &mut self,
        windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3],
    ) -> Result<()> {
        ensure!(self.pending.is_none(), "dSpark chain replay still pending");
        self.invalidate();
        ensure!(!self.has_graph(bindings[0].len()), "dSpark chain count already captured");
        unsafe {
            self.execute(windows, bindings)?;
        }
        let reads = self.prepare(windows, bindings)?;
        unsafe { self.capture_ready(&reads, bindings[0].len()) }
    }
    /// Warmup completed; never suspend while capturing.
    unsafe fn capture_ready(&mut self, reads: &[WindowRead; 3], count: usize) -> Result<()> {
        unsafe { self.capture_geometry(&reads.each_ref().map(WindowRead::geometry), count) }
    }
    unsafe fn capture_geometry(&mut self, reads: &[WindowGeometry; 3], count: usize) -> Result<()> {
        let key = (count, self.width);
        if self.graphs.contains(&key) { return Ok(()); }
        let Some(ticket) = self.graphs.cache_ticket(&key, false, None) else { return Ok(()); };
        unsafe {
            self.stream
                .library
                .cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launched = unsafe { self.enqueue_geometry(reads, count) };
        let captured = unsafe { self.stream.library.cuda_graph_end_capture_topology(self.stream.raw) };
        self.invalidate();
        match (launched, captured) {
            (Ok(()), Ok((graph, nodes, width, spans, inspection))) => {
                let device = self.tokens.buffer.device_id;
                // SAFETY: chain pins all captured weights/workspaces and drains on Drop.
                let owner = unsafe { GraphOwner::new(self.stream.library, device, graph,
                    reads.each_ref().map(|r| r.owner))? };
                warn_topology(inspection);
                self.graphs.cache_commit(key, owner, ticket, nodes as u64, width,
                    spans || self.spans_devices());
                tracing::debug!(target: "cuteafd::graph_capture", count, width=self.width,
                    bank=self as *const Self as usize, retained=self.graphs.len(),
                    "native dSpark chain graph captured");
                Ok(())
            }
            (Err(error), Ok((graph, _, _, _, _))) => {
                unsafe {
                    self.stream.library.cuda_graph_exec_destroy(graph)?;
                }
                Err(error)
            }
            (Ok(()), Err(error)) => Err(error),
            (Err(launched), Err(captured)) => Err(prefer_device(launched, captured)),
        }
    }
    /// # Safety
    /// Same contract as execute; owner IDs and request count must match capture.
    pub unsafe fn replay(
        &mut self,
        windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3],
    ) -> Result<[CuteafdDeviceBuffer; 2]> {
        let reads = self.prepare(windows, bindings)?;
        let count = bindings[0].len();
        let Some(captured) = self.graphs.cached_launch(&(count, self.width)) else {
            // SAFETY: eager execution has the same retained-input contract.
            return unsafe { self.execute(windows, bindings) };
        };
        let (graph, owners) = (captured.raw, captured.pins);
        ensure!(
            count == bindings[0].len() && owners == reads.each_ref().map(|r| r.owner),
            "dSpark chain capture binding differs"
        );
        self.upload(&reads, bindings)?;
        let launched = unsafe {
            self.stream
                .library
                .cuda_graph_launch(graph, self.stream.raw)
        };
        let drained = self.synchronize();
        settle(launched, drained)?;
        self.ready = Some(count);
        self.output()
    }
    /// Queue warmup or replay plus compact copies. The caller keeps window
    /// owners alive; this owner retains reservations and pinned inputs throughout.
    /// # Safety
    /// Same inputs as replay; no raw consumer may mutate reserved ring slots.
    pub unsafe fn begin_replay(&mut self, windows: [&DsparkWindow<'_>; 3],
        bindings: [&[(WindowLease, u64)]; 3]) -> Result<()> {
        let reads = self.prepare(windows, bindings)?;
        let count = bindings[0].len();
        if let Some(captured) = self.graphs.get(&(count, self.width)) {
            ensure!(captured.pins == reads.each_ref().map(|r| r.owner), "dSpark chain capture binding differs");
        }
        self.terminal.as_ref().context("dSpark chain has no terminal")?.output_storage(count)?;
        let pending = PendingDraft { count, reads, warming: !self.has_graph(count), armed: true,
            library: self.stream.library, stream: self.stream.raw,
            peers: self.stages.each_ref().map(DsparkStage::peer_stream) };
        // The local guard already owns reservations before any partial upload.
        unsafe {
            self.stream.library.copy_h2d_async(self.tokens.buffer,
                &self.token_staging.bytes_mut()[..count * 4], self.stream.raw)?;
            self.terminal.as_mut().unwrap().upload_sampling_on(self.stream.raw)?;
            for stage in 0..3 {
                self.stages[stage].upload_on(&pending.reads[stage], bindings[stage], self.stream.raw)?;
            }
            if pending.warming { self.enqueue(&pending.reads, count)?; }
            else { self.launch_download(count)?; }
        }
        self.pending = Some(pending);
        Ok(())
    }
    unsafe fn launch_download(&mut self, count: usize) -> Result<()> {
        let graph = self.graphs.cached_launch(&(count, self.width)).map(|owner| owner.raw);
        // A denied capture already executed eagerly in the completed warm pass.
        // Copy that result rather than sampling a second time or requiring a graph.
        let output = self.terminal.as_ref().context("dSpark chain has no terminal")?.output_storage(count)?;
        unsafe {
            let span = self.timing.begin("graph_forward_terminal", None, count * self.width, self.stream.raw)?;
            if let Some(graph) = graph {
                self.stream.library.cuda_graph_launch(graph, self.stream.raw)?;
            }
            self.timing.end(span, self.stream.raw)?;
            let bytes = self.download.bytes_mut();
            self.stream.library.copy_d2h_async(&mut bytes[..count * (self.width + 1) * 4], output[0], self.stream.raw)?;
            self.stream.library.copy_d2h_async(&mut bytes[count * (self.width + 1) * 4..count * (2 * self.width + 1) * 4], output[2], self.stream.raw)
        }
    }
    /// Cold warmup and final replay both poll. Capture itself never suspends.
    pub fn poll_replay(&mut self) -> Result<Option<(Vec<u32>, Vec<f32>)>> {
        ensure!(self.pending.is_some(), "dSpark replay not pending");
        match unsafe { self.stream.library.cuda_stream_query(self.stream.raw) } {
            Ok(false) => return Ok(None),
            Ok(true) => {},
            Err(error) => { self.pending = None; return Err(error); },
        }
        let mut pending = self.pending.take().unwrap();
        let count = pending.count;
        if pending.warming {
            unsafe {
                self.capture_ready(&pending.reads, count)?;
                self.launch_download(count)?;
            }
            pending.warming = false;
            self.pending = Some(pending);
            return Ok(None);
        }
        self.timing.publish_completed()?;
        self.timing.publish_retained("forward", count * self.width)?;
        self.terminal.as_ref().unwrap().publish_timing(count)?;
        pending.armed = false;
        self.ready = Some(count);
        let bytes = self.download.bytes_mut();
        let tokens = bytes[..count * (self.width + 1) * 4].chunks_exact(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap())).collect();
        let confidence = bytes[count * (self.width + 1) * 4..count * (2 * self.width + 1) * 4].chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap())).collect();
        Ok(Some((tokens, confidence)))
    }
    /// Record a CUDA/replay failure as terminal for this chain.
    pub(crate) fn note_failure(&mut self, error: &anyhow::Error) { self.failure.record(error); }
    /// A terminal draft is not a reusable target-only fallback.
    pub fn require_live(&self) -> Result<()> { self.failure.check("dSpark draft chain") }
    /// Copy a draft output buffer to the host on this chain's stream.
    pub fn copy_d2h(&self, dst: &mut [u8], buffer: CuteafdDeviceBuffer) -> Result<()> {
        self.stream.library.copy_d2h(dst, buffer)
    }
    /// Drop a pending replay so its requests can be released.
    pub fn cancel_replay(&mut self) {
        self.pending = None; // guard drains before releasing reads
        self.invalidate();
    }
    #[cfg(test)]
    pub fn cancel_pending_for_test(&mut self) { self.cancel_replay(); }
    pub fn output(&self) -> Result<[CuteafdDeviceBuffer; 2]> {
        let requests = self.ready.context("dSpark chain output incomplete")?;
        let mut output = self.stages[2].output_storage();
        output[0].bytes = requests * self.width * 40960;
        output[1].bytes = requests * self.width * 16;
        Ok(output)
    }
}
impl Drop for DsparkChain<'_, '_> {
    fn drop(&mut self) {
        fatal_drain(self.synchronize(), "dSpark chain");
        if let Some(pending) = &mut self.pending { pending.armed = false; }
        self.pending = None; // GPU work has drained before read reservations release.
        if self.graphs.cache_attached() {
            self.graphs.cache_clear_drained();
        } else {
            self.graphs.retire_all();
            fatal_drain(self.graphs.drain_retired(|| Ok(())).map(drop), "dSpark chain graphs");
        }
    }
}

#[cfg(test)]
mod failure_tests {
    use super::*;
    #[test]
    fn dspark_full_reserve_miss_runs_eager_without_refusal() -> Result<()> {
        use crate::shared::decode_graph::device::{Basis, Cycle, Preference, Site};
        let mut bank = GraphBank::<(usize, usize), usize>::new(None);
        bank.attach_device(-164_401, 200, Site {
            name: "dspark_chain", cycle: Cycle::Wave, preference: Preference::Capture,
            bootstrap: 100, basis: Basis::Bytes, spans_devices: false,
        })?;
        let ticket = bank.cache_ticket(&(1, 5), true, None).expect("initial reserve fits");
        bank.cache_commit((1, 5), 17, ticket, 1, 1, false);
        bank.cache_measured(200);
        let mut eager_runs = 0;
        for _ in 0..3 {
            assert!(bank.cached_launch(&(2, 7)).is_none());
            if bank.cache_ticket(&(2, 7), false, None).is_none() {
                eager_runs += 1;
            } else {
                anyhow::bail!("a full dSpark reserve must not capture");
            }
        }
        assert_eq!(eager_runs, 3);
        assert_eq!(bank.cached_launch(&(1, 5)), Some(&17));
        Ok(())
    }

    #[test]
    fn dspark_failed_capture_cancels_ticket_and_preserves_eager_fallback() -> Result<()> {
        use crate::shared::decode_graph::device::{Basis, Cycle, Preference, Site};
        let mut bank = GraphBank::<(usize, usize), usize>::new(None);
        bank.attach_device(-164_402, 200, Site {
            name: "dspark_chain", cycle: Cycle::Other, preference: Preference::Capture,
            bootstrap: 100, basis: Basis::Bytes, spans_devices: true,
        })?;
        let ticket = bank.cache_ticket(&(1, 5), false, None).expect("initial reserve fits");
        drop(ticket); // Simulates any failure between begin and owner insertion.
        bank.cache_measured(0);
        assert!(bank.cached_launch(&(1, 5)).is_none());
        assert!(bank.cache_ticket(&(1, 5), false, None).is_some());
        Ok(())
    }

    fn device() -> anyhow::Error {
        anyhow::Error::new(cuteafd_ffi::NativeStatusError { context: "capture".into(), status: 98,
            detail: "invalid device function".into() }).context("capture failed")
    }
    #[test]
    fn settling_never_masks_a_device_failure_in_either_order() -> Result<()> {
        settle(Ok(()), Ok(()))?;
        for (launch, drain) in [(Err(anyhow::anyhow!("host guard")), Err(device())),
            (Err(device()), Err(anyhow::anyhow!("host drain"))), (Ok(()), Err(device())),
            (Err(device()), Ok(()))] {
            let error = settle(launch, drain).unwrap_err();
            assert!(is_device_failure(&error));
            let mut failure = DraftFailure::default();
            failure.record(&error);
            assert!(failure.check("draft").is_err());
        }
        let mut failure = DraftFailure::default();
        failure.record(&settle(Err(anyhow::anyhow!("host")), Err(anyhow::anyhow!("host drain"))).unwrap_err());
        failure.check("draft")?;
        failure.record(&device());
        let first = failure.check("draft").unwrap_err().to_string();
        failure.record(&anyhow::anyhow!("later host failure"));
        assert_eq!(failure.check("draft").unwrap_err().to_string(), first);
        Ok(())
    }
}

impl DsparkChain<'_, '_> {
    fn spans_devices(&self) -> bool {
        self.stages.iter().any(|stage| stage.peer_stream().is_some())
            || self.embedding.is_some_and(|embedding| embedding.get("embed.weight")
                .is_ok_and(|table| table.device_id != self.tokens.buffer.device_id))
    }
    pub(crate) fn visit_graph_banks(&mut self, f: &mut GraphVisitor<'_>) -> Result<()> {
        f(self.tokens.buffer.device_id, "dspark_chain", self.spans_devices(), &mut self.graphs)?;
        for stage in &mut self.stages { stage.visit_graph_banks(f)?; }
        if let Some(terminal) = &mut self.terminal { terminal.visit_graph_banks(f)?; }
        Ok(())
    }
}
