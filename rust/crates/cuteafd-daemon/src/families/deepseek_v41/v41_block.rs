//! Shifted backbone attention/FFN sequencing with exact query-result identity.
use crate::families::deepseek_v41::v41_attention_binding::QueryBinding;
use crate::families::deepseek_v41::v41_attention_output::AttentionOutput;
use crate::families::deepseek_v41::v41_attention_query::{AttentionQueryOutput, AttentionQueryWave};
use crate::families::deepseek_v41::v41_hc::HcSublayer;
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{CuteafdDeviceBuffer, NativeLibrary};
mod encoder_suffix;
mod transfer;
pub(crate) use encoder_suffix::EncoderSuffix;
#[derive(Clone, Copy)]
enum Phase {
    Idle,
    StartupAttention(usize),
    Prepared(QueryBinding, usize, bool),
    Attention(QueryBinding, usize),
    QueuedFfn(QueryBinding, usize),
    Ffn(QueryBinding, usize),
    Ready(QueryBinding, usize, bool),
}
/// Next-layer input after required engram work, before attention or dSpark taps.
pub(crate) struct PreparedBlockInput<'a> {
    pub residual: CuteafdDeviceBuffer,
    pub pre: CuteafdDeviceBuffer,
    pub layer: usize,
    pub tokens: &'a [u64],
    previous: QueryBinding,
}
impl PreparedBlockInput<'_> {
    pub fn previous_binding(&self) -> QueryBinding { self.previous }
}
pub(crate) struct FfnInput<'a> {
    pub residual: CuteafdDeviceBuffer,
    pub incoming_pre: CuteafdDeviceBuffer,
    pub values: CuteafdDeviceBuffer,
    pub layer: usize,
    pub tokens: &'a [u64],
    binding: QueryBinding,
}
impl FfnInput<'_> {
    pub fn binding(&self) -> QueryBinding {
        self.binding
    }
}
pub(crate) struct BlockOutput<'a> {
    binding: QueryBinding,
    pub residual: CuteafdDeviceBuffer,
    pub pre: CuteafdDeviceBuffer,
    pub layer: usize,
    pub tokens: &'a [u64],
}
impl BlockOutput<'_> {
    pub fn binding(&self) -> QueryBinding {
        self.binding
    }
}
pub(crate) struct BackboneBlockWave<'w, 'a> {
    library: &'a NativeLibrary,
    layer: usize,
    attention: HcSublayer<'w, 'a>,
    ffn: HcSublayer<'w, 'a>,
    capacity: usize,
    tokens: Vec<u64>,
    phase: Phase,
}
impl<'w, 'a> BackboneBlockWave<'w, 'a> {
    #[cfg(test)]
    pub fn trace_stream(&self) -> *mut std::ffi::c_void { self.ffn.stream_raw() }
    /// The stream carrying each layer's FFN finish (mHC post, next inputs).
    pub(crate) fn finish_stream(&self) -> *mut std::ffi::c_void { self.ffn.stream_raw() }
    /// Reuse this lane's two mHC workspaces for the adjacent layer. Completed
    /// residual/pre values are copied before installing both validated bindings.
    /// No device allocation or graph capture occurs here.
    pub fn advance(
        &mut self,
        next: &'w crate::families::deepseek_v41::v41_backbone_hc::BackboneHcWeights<'a>,
    ) -> Result<()> {
        let result = (|| -> Result<()> {
            let (binding, rows, copied_next) = match self.phase {
                Phase::Ready(binding, rows, copied_next) => (binding, rows, copied_next),
                _ => anyhow::bail!("block advance requires a completed FFN"),
            };
            ensure!(self.layer < 39 && next.layer() == self.layer + 1
                && binding.layer() == self.layer && self.tokens.len() == rows,
                "block advance layer or identity differs");
            let [attention, ffn] = next.prepare_bindings(&self.attention, &self.ffn)?;
            if !copied_next {
                let copied = unsafe { self.enqueue_next_inputs(rows) };
                let drained = unsafe { self.library.cuda_stream_synchronize(self.ffn.stream_raw()) };
                copied.and(drained)?;
            }
            // Backbone mHC executes on its own drained streams; it has no
            // external captured mHC graphs to invalidate during this rebind.
            unsafe {
                self.attention.install_binding(attention);
                self.ffn.install_binding(ffn);
            }
            self.layer = next.layer();
            self.phase = Phase::Prepared(binding, rows, ![1,14].contains(&self.layer));
            Ok(())
        })();
        if result.is_err() { self.reset(); }
        result
    }
    /// Start another sequence after completion, or after an explicit reset on
    /// cancellation. Token initialization is required before attention resumes.
    pub fn restart(
        &mut self,
        first: &'w crate::families::deepseek_v41::v41_backbone_hc::BackboneHcWeights<'a>,
    ) -> Result<()> {
        let result = (|| -> Result<()> {
            ensure!(first.layer() == 0
                && (matches!(self.phase, Phase::Idle)
                    || (self.layer == 39 && matches!(self.phase, Phase::Ready(..)))),
                "block restart requires layer zero and an idle or completed sequence");
            let [attention, ffn] = first.prepare_bindings(&self.attention, &self.ffn)?;
            unsafe {
                self.attention.install_binding(attention);
                self.ffn.install_binding(ffn);
            }
            self.layer = 0;
            self.reset();
            Ok(())
        })();
        if result.is_err() { self.reset(); }
        result
    }
    /// Bind drained startup storage without constructing a completed query.
    pub(crate) fn rebind_startup(
        &mut self,
        weights: &'w crate::families::deepseek_v41::v41_backbone_hc::BackboneHcWeights<'a>,
    ) -> Result<()> {
        ensure!(matches!(self.phase, Phase::Idle) && self.tokens.is_empty(),
            "startup mHC rebind requires an idle block");
        let [attention, ffn] = weights.prepare_bindings(&self.attention, &self.ffn)?;
        // SAFETY: startup has no submitted consumers; both bindings were drained.
        unsafe {
            self.attention.install_binding(attention);
            self.ffn.install_binding(ffn);
        }
        self.layer = weights.layer();
        Ok(())
    }
    /// # Safety
    /// Only a capture-only tail may follow. No work is executed or published;
    /// reset must run after capture on both success and error.
    pub(crate) unsafe fn prepare_startup_ffn(&mut self, rows: usize) -> Result<()> {
        ensure!(matches!(self.phase, Phase::Idle) && self.tokens.is_empty(),
            "startup FFN requires an idle block");
        // SAFETY: this changes host phase only; the recorded graph is not run.
        unsafe { self.attention.graph_begin_state(rows)?; }
        self.ffn.invalidate();
        self.phase = Phase::StartupAttention(rows);
        Ok(())
    }
    /// Rebind an idle lane to decoder layer 20 and restore retained encoder rows.
    pub fn initialize_decoder(&mut self, decoder: &'w crate::families::deepseek_v41::v41_backbone_hc::BackboneHcWeights<'a>,
        encoder: &BlockOutput<'_>) -> Result<()> {
        self.reset();
        let result = (|| -> Result<()> {
            ensure!(decoder.layer() == 20 && encoder.layer == 19, "invalid encoder/decoder boundary");
            let [attention, ffn] = decoder.prepare_bindings(&self.attention, &self.ffn)?;
            unsafe { self.attention.install_binding(attention); self.ffn.install_binding(ffn); }
            self.layer = 20;
            self.initialize_previous(encoder)
        })();
        if result.is_err() { self.reset(); }
        result
    }
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        Ok(2 * HcSublayer::device_bytes(capacity)?)
    }
    pub(crate) fn new(
        library: &'a NativeLibrary,
        layer: usize,
        attention: HcSublayer<'w, 'a>,
        ffn: HcSublayer<'w, 'a>,
        capacity: usize,
    ) -> Self {
        Self {
            library,
            layer,
            attention,
            ffn,
            capacity,
            tokens: Vec::with_capacity(capacity),
            phase: Phase::Idle,
        }
    }
    /// Residual [capacity,4,5120] and incoming FP32 pre [capacity,4]. Engram
    /// contributions and dSpark tap reads precede begin_attention at the caller.
    pub fn inputs(&self) -> [CuteafdDeviceBuffer; 2] {
        self.attention.inputs()
    }
    /// Selects this slot's unit rows in both boundaries' shared planes
    /// (`shared::memory::row_window`), at rest.
    pub(crate) fn select_window(&mut self, window: crate::shared::memory::row_window::RowWindow) -> Result<()> {
        crate::shared::memory::row_window::require_rest(!matches!(self.phase, Phase::Idle), "mHC block", "a layer is in progress")?;
        self.attention.select_window(window.clone())?;
        self.ffn.select_window(window)
    }
    pub fn reset(&mut self) {
        self.phase = Phase::Idle;
        self.tokens.clear();
        self.attention.invalidate();
        self.ffn.invalidate();
    }
    /// Copy a completed adjacent layer into stable next-block input storage.
    /// Layers 1 and 14 remain unavailable until their engram update completes.
    pub fn initialize_previous(&mut self, previous: &BlockOutput<'_>) -> Result<()> {
        self.reset();
        let rows = previous.tokens.len();
        ensure!(previous.layer < 39 && self.layer == previous.layer + 1
            && previous.binding.layer() == previous.layer
            && rows > 0 && rows <= self.capacity
            && previous.tokens.iter().all(|&p| p < 1048576)
            && previous.residual.bytes == rows * 40960 && previous.pre.bytes == rows * 16,
            "previous block layer, tokens or extents differ");
        let sources = [previous.residual, previous.pre];
        let destinations = self.inputs();
        ensure!(sources.iter().zip(destinations).all(|(s,d)| s.device_id == d.device_id),
            "previous block device differs");
        self.copy_inputs_ordered(&sources, &destinations)?;
        self.tokens.extend_from_slice(previous.tokens);
        self.phase = Phase::Prepared(previous.binding, rows, ![1,14].contains(&self.layer));
        Ok(())
    }
    /// # Safety
    /// Gathered engram rows/masks correspond to this block's exact request and
    /// token order, with current history leases and completed producer writes.
    /// Gather and gate storage remain live and exclusive until the call drains.
    pub unsafe fn apply_engram(
        &mut self,
        gate: &mut crate::families::deepseek_v41::v41_engram::layer::EngramGate<'_, '_>,
        gathered: &crate::families::deepseek_v41::v41_engram::EngramDeviceView,
    ) -> Result<()> {
        let result = (|| -> Result<()> {
            let (binding, rows) = match self.phase {
                Phase::Prepared(binding, rows, false) => (binding, rows),
                _ => anyhow::bail!("block is not awaiting engram"),
            };
            ensure!(gate.layer() == self.layer && gathered.rows == rows,
                "block engram layer or rows differ");
            let output = unsafe { gate.execute_captured(self.inputs()[0], gathered) }?;
            self.copy_inputs_ordered(&[output], &[self.inputs()[0]])?;
            gate.release_scratch()?;
            self.phase = Phase::Prepared(binding, rows, true);
            Ok(())
        })();
        if result.is_err() { self.reset(); }
        result
    }
    /// # Safety
    /// Keep gathered upload storage and this block exclusive through the gate/copy.
    pub async unsafe fn apply_engram_cooperative(&mut self,
        gate: &mut crate::families::deepseek_v41::v41_engram::layer::EngramGate<'_, '_>,
        gathered: &crate::families::deepseek_v41::v41_engram::EngramDeviceView) -> Result<()> {
        let Phase::Prepared(binding, rows, false) = std::mem::replace(&mut self.phase, Phase::Idle) else {
            anyhow::bail!("block is not awaiting engram");
        };
        ensure!(gate.layer() == self.layer && gathered.rows == rows, "block engram layer or rows differ");
        unsafe { gate.execute_into_cooperative(self.inputs()[0], gathered).await?; }
        gate.release_scratch()?;
        self.phase = Phase::Prepared(binding, rows, true);
        Ok(())
    }
    /// Metadata for gather association before engram makes query inputs ready.
    pub fn pending_engram(&self) -> Result<(usize, &[u64])> {
        ensure!(matches!(self.phase, Phase::Prepared(_, _, false)),
            "block is not awaiting engram");
        Ok((self.layer, &self.tokens))
    }
    pub fn prepared_input(&self) -> Result<PreparedBlockInput<'_>> {
        let (previous, rows) = match self.phase {
            Phase::Prepared(binding, rows, true) => (binding, rows),
            _ => anyhow::bail!("block input is not prepared; engram may be pending"),
        };
        let [mut residual, mut pre] = self.inputs();
        residual.bytes = rows * 40960;
        pre.bytes = rows * 16;
        Ok(PreparedBlockInput { residual, pre, layer: self.layer,
            tokens: &self.tokens, previous })
    }
    /// # Safety
    /// Query and block storage have exclusive use on the same device. Required
    /// dSpark tap reads must finish before this call overwrites any input.
    pub unsafe fn begin_prepared_attention<'q>(
        &mut self,
        query: &'q mut AttentionQueryWave<'_, '_>,
    ) -> Result<AttentionQueryOutput<'q>> {
        let tokens = match self.prepared_input() {
            Ok(input) => input.tokens.to_vec(),
            Err(e) => { self.reset(); return Err(e); }
        };
        unsafe { self.begin_attention(query, &tokens) }
    }
    /// Copy text embeddings into the first block's stable inputs. This leaves
    /// output unpublished. Image replacement belongs to the separate vision path.
    pub fn initialize_embedding(
        &mut self,
        embedding: &crate::families::deepseek_v41::v41_target_embedding::TargetEmbedding<'_>,
    ) -> Result<()> {
        self.reset();
        let rows = embedding.positions.len();
        ensure!(self.layer == 0 && rows > 0
            && rows <= self.capacity && embedding.token_ids.len() == rows
            && embedding.residual.bytes == rows * 40960 && embedding.pre.bytes == rows * 16,
            "initial block embedding geometry differs");
        self.copy_inputs_ordered(&[embedding.residual, embedding.pre], &self.inputs())?;
        Ok(())
    }
    /// A prefill unit's layer-0 input, without its query: the embedding lands in
    /// this block's planes (its window, a row plane) and the block waits
    /// prepared at layer 0. The unit's attention hook runs the query there, so
    /// no other unit's hook writes the shared query scratch in between
    /// (`shared::memory::row_window`: the query outputs are guarded scratch
    /// consumed within the attention hook).
    pub fn prepare_embedding(
        &mut self,
        embedding: &crate::families::deepseek_v41::v41_target_embedding::TargetEmbedding<'_>,
        binding: QueryBinding,
    ) -> Result<()> {
        self.initialize_embedding(embedding)?;
        self.tokens.extend_from_slice(embedding.positions);
        self.phase = Phase::Prepared(binding, embedding.positions.len(), true);
        Ok(())
    }
    /// Join producers before copying into the windowed input planes. The legacy
    /// stream does not order with our non-blocking component streams.
    fn copy_inputs_ordered(&self, sources: &[CuteafdDeviceBuffer],
        destinations: &[CuteafdDeviceBuffer]) -> Result<()> {
        ensure!(sources.len() == destinations.len() && sources.iter().zip(destinations)
            .all(|(source, destination)| source.device_id == destination.device_id
                && source.bytes <= destination.bytes), "block input copy extent or device differs");
        let stream = self.attention.stream_raw();
        // SAFETY: inputs belong to this block; the chain joins all producer
        // writes and keeps both owners alive until the slot is drained.
        let launched = (|| unsafe {
            crate::shared::memory::chain::join(self.library, stream)?;
            for (&source, &destination) in sources.iter().zip(destinations) {
                if source.ptr != destination.ptr {
                    self.library.copy_d2d_async(destination, source, source.bytes, stream)?;
                }
            }
            Ok(())
        })();
        if let Err(error) = launched {
            // SAFETY: drain any partially enqueued copies before returning.
            unsafe { self.library.cuda_stream_synchronize(stream)?; }
            return Err(error);
        }
        // SAFETY: every copy precedes publication; outside a chain this drains.
        unsafe { crate::shared::memory::chain::finish(self.library, stream) }
    }
    /// # Safety
    /// Query and block storage have exclusive use on the embedding device.
    /// Embeddings are finite text inputs; image replacement uses the vision path.
    pub unsafe fn begin_embedded_attention<'q>(
        &mut self,
        query: &'q mut AttentionQueryWave<'_, '_>,
        embedding: &crate::families::deepseek_v41::v41_target_embedding::TargetEmbedding<'_>,
    ) -> Result<AttentionQueryOutput<'q>> {
        self.reset();
        ensure!(query.layer() == 0, "initial block query layer differs");
        self.initialize_embedding(embedding)?;
        unsafe { self.begin_attention(query, embedding.positions) }
    }
    /// # Safety
    /// Inputs are finite in token order with all producer writes complete. Query
    /// storage and both boundary owners have exclusive use on this device.
    pub unsafe fn begin_attention<'q>(
        &mut self,
        query: &'q mut AttentionQueryWave<'_, '_>,
        tokens: &[u64],
    ) -> Result<AttentionQueryOutput<'q>> {
        self.reset();
        ensure!(
            query.layer() == self.layer
                && !tokens.is_empty()
                && tokens.len() <= self.capacity
                && tokens.iter().all(|&p| p < 1048576),
            "block attention layer or tokens differ"
        );
        ensure!(
            query.input().device_id == self.inputs()[0].device_id,
            "block query device differs"
        );
        let timing = std::time::Instant::now();
        let out = match unsafe {
            query.execute_tokens_prepared(tokens, |stream, input| {
                self.attention.enqueue_begin(tokens.len(), Some(input), stream)?;
                Ok(())
            })
        } {
            Ok(o) => o,
            Err(e) => {
                self.reset();
                return Err(e);
            }
        };
        self.tokens.extend_from_slice(tokens);
        tracing::debug!(target: "cuteafd::timing", layer=self.layer, rows=tokens.len(), total_us=timing.elapsed().as_micros() as u64, "target query preparation");
        self.phase = Phase::Attention(out.binding()?, tokens.len());
        Ok(out)
    }
    /// # Safety
    /// Producer, block and query storage stay owned through completion. The
    /// producer initializes block inputs on the query stream before mHC/query.
    pub async unsafe fn begin_attention_cooperative<'q>(&mut self,
        query: &'q mut AttentionQueryWave<'_, '_>, tokens: &[u64],
        prepare: impl FnOnce(*mut std::ffi::c_void, [CuteafdDeviceBuffer; 2]) -> Result<()>)
        -> Result<AttentionQueryOutput<'q>> {
        unsafe { self.begin_attention_with_projection_cooperative(query,tokens,None,prepare).await }
    }
    /// # Safety
    /// Same contract as begin_attention_cooperative; retain the optional TP2
    /// projection owner through completion or drained cancellation.
    pub async unsafe fn begin_attention_with_projection_cooperative<'q>(&mut self,
        query: &'q mut AttentionQueryWave<'_, '_>, tokens: &[u64],
        projection: Option<&mut crate::families::deepseek_v41::v41_projection_tp2::Wave<'_, '_>>,
        prepare: impl FnOnce(*mut std::ffi::c_void, [CuteafdDeviceBuffer; 2]) -> Result<()>)
        -> Result<AttentionQueryOutput<'q>> {
        self.reset();
        ensure!(query.layer() == self.layer && !tokens.is_empty() && tokens.len() <= self.capacity
            && tokens.iter().all(|&p| p < 1048576), "block attention layer or tokens differ");
        ensure!(query.input().device_id == self.inputs()[0].device_id, "block query device differs");
        let prepare_query = |stream, input| unsafe {
            prepare(stream, self.inputs())?;
            self.attention.enqueue_begin(tokens.len(), Some(input), stream)?;
            Ok(())
        };
        let out = unsafe { match projection {
            Some(projection) => query.execute_tokens_tp2_prepared_cooperative(tokens,projection,prepare_query).await?,
            None => query.execute_tokens_prepared_cooperative(tokens,prepare_query).await?,
        } };
        self.tokens.extend_from_slice(tokens);
        self.phase = Phase::Attention(out.binding()?, tokens.len());
        Ok(out)
    }
    /// # Safety
    /// Same contract as begin_prepared_attention, retained across completion waits.
    pub async unsafe fn begin_prepared_attention_cooperative<'q>(&mut self,
        query: &'q mut AttentionQueryWave<'_, '_>) -> Result<AttentionQueryOutput<'q>> {
        unsafe { self.begin_prepared_attention_with_projection_cooperative(query,None).await }
    }
    /// # Safety
    /// Same contract as begin_attention_with_projection_cooperative.
    pub async unsafe fn begin_prepared_attention_with_projection_cooperative<'q>(&mut self,
        query: &'q mut AttentionQueryWave<'_, '_>,
        projection: Option<&mut crate::families::deepseek_v41::v41_projection_tp2::Wave<'_, '_>>)
        -> Result<AttentionQueryOutput<'q>> {
        let tokens = match self.prepared_input() {
            Ok(input) => input.tokens.to_vec(),
            Err(error) => { self.reset(); return Err(error); }
        };
        unsafe { self.begin_attention_with_projection_cooperative(query, &tokens, projection, |_, _| Ok(())).await }
    }
    /// # Safety
    /// Attention output is complete and immutable; no external writes race the
    /// preserved residual or mixing coefficients. A rejected finish resets phase.
    pub unsafe fn begin_ffn(&mut self, output: &AttentionOutput<'_>) -> Result<FfnInput<'_>> {
        let result = (|| -> Result<(CuteafdDeviceBuffer, QueryBinding)> {
            let Phase::Attention(binding, rows) = std::mem::replace(&mut self.phase, Phase::Idle)
            else {
                anyhow::bail!("block attention is not pending");
            };
            ensure!(
                output.binding()? == binding
                    && output.layer == self.layer
                    && output.rows == rows
                    && output.projected.device_id == self.inputs()[0].device_id,
                "block attention result differs"
            );
            let stream = self.attention.stream_raw();
            let enqueued = (|| -> Result<CuteafdDeviceBuffer> {
                unsafe { self.attention.enqueue_finish(Some(output.projected), stream)?; }
                // Keep the existing residual/pre buffers, but copy on the same
                // stream after attention post and before FFN pre/mixing.
                for ((dst, src), bytes) in self.ffn.inputs().into_iter()
                    .zip(self.attention.output_storage())
                    .zip([rows * 40960, rows * 16]) {
                    unsafe { self.library.copy_d2d_async(dst, src, bytes, stream)?; }
                }
                unsafe { self.ffn.enqueue_begin(rows, None, stream) }
            })();
            // Always drain, including partial enqueue failure: both mHC owners
            // must be safe to reset/rebind when this function returns.
            let drained = unsafe { self.library.cuda_stream_synchronize(stream) };
            let normalized = enqueued.and_then(|value| drained.map(|()| value))?;
            unsafe { self.attention.complete()?; }
            self.phase = Phase::Ffn(binding, rows);
            Ok((normalized, binding))
        })();
        match result {
            Ok((values, binding)) => Ok(FfnInput {
                residual: {
                    let mut b = self.ffn.inputs()[0];
                    b.bytes = values.bytes * 4;
                    b
                },
                incoming_pre: {
                    let mut b = self.ffn.inputs()[1];
                    b.bytes = values.bytes / 10240 * 16;
                    b
                },
                values,
                layer: self.layer,
                tokens: &self.tokens,
                binding,
            }),
            Err(e) => {
                self.reset();
                Err(e)
            }
        }
    }
    /// # Safety
    /// Projection is ordered on stream with the matching query binding. Keep
    /// both mHC owners exclusive and alive until stream drains, including errors.
    /// Only complete_queued_ffn may publish the resulting normalized input.
    pub unsafe fn enqueue_ffn(
        &mut self, binding: QueryBinding, rows: usize, projected: CuteafdDeviceBuffer,
        stream: *mut std::ffi::c_void,
    ) -> Result<CuteafdDeviceBuffer> {
        let Phase::Attention(expected, count) = self.phase else {
            anyhow::bail!("block attention is not pending for queued FFN");
        };
        ensure!(binding == expected && rows == count && binding.layer() == self.layer
            && projected.device_id == self.inputs()[0].device_id
            && !projected.ptr.is_null() && projected.bytes == rows * 10240,
            "queued block attention result differs");
        self.phase = Phase::QueuedFfn(binding, rows);
        // SAFETY: runtime identity and projection geometry were checked above.
        unsafe { self.enqueue_ffn_geometry(rows, projected, stream) }
    }
    /// # Safety
    /// Stream is capturing, never executing this uninitialized startup storage.
    /// The owner must be reset after capture, including errors.
    pub(crate) unsafe fn record_startup_ffn(
        &mut self, rows: usize, projected: CuteafdDeviceBuffer,
        stream: *mut std::ffi::c_void,
    ) -> Result<CuteafdDeviceBuffer> {
        ensure!(matches!(self.phase, Phase::StartupAttention(n) if n == rows)
            && projected.device_id == self.inputs()[0].device_id
            && !projected.ptr.is_null() && projected.bytes == rows * 10240,
            "startup FFN geometry differs");
        // SAFETY: the startup caller retains stable storage and never runs it.
        unsafe { self.enqueue_ffn_geometry(rows, projected, stream) }
    }
    unsafe fn enqueue_ffn_geometry(
        &mut self, rows: usize, projected: CuteafdDeviceBuffer,
        stream: *mut std::ffi::c_void,
    ) -> Result<CuteafdDeviceBuffer> {
        // SAFETY: either a validated live query or a capture-only owner supplies
        // these same stable planes. Both paths record exactly these operations.
        unsafe { self.attention.enqueue_finish(Some(projected), stream)?; }
        for ((dst, src), bytes) in self.ffn.inputs().into_iter()
            .zip(self.attention.output_storage()).zip([rows * 40960, rows * 16]) {
            unsafe { self.library.copy_d2d_async(dst, src, bytes, stream)?; }
        }
        unsafe { self.ffn.enqueue_begin(rows, None, stream) }
    }
    /// # Safety
    /// A graph with enqueue_ffn's exact operations is about to replay on the
    /// enclosing stream. Its weights/buffers match this block and rows.
    pub unsafe fn prepare_ffn_graph_replay(&mut self, binding: QueryBinding, rows: usize) -> Result<()> {
        ensure!(matches!(self.phase, Phase::Attention(b, n) if b == binding && n == rows),
            "FFN graph replay query differs");
        unsafe { self.attention.graph_post_state(rows)?; self.ffn.graph_begin_state(rows)?; }
        self.phase = Phase::QueuedFfn(binding, rows); Ok(())
    }
    /// # Safety
    /// Only the unpublished warmup of this same graph has completed. Attention
    /// residual and mixing coefficients were read, not overwritten, by warmup.
    pub unsafe fn restore_ffn_graph_warmup(&mut self, binding: QueryBinding, rows: usize) -> Result<()> {
        ensure!(matches!(self.phase, Phase::QueuedFfn(b, n) if b == binding && n == rows),
            "FFN graph warmup query differs");
        unsafe { self.attention.graph_begin_state(rows)?; }
        self.ffn.invalidate(); self.phase = Phase::Attention(binding, rows); Ok(())
    }
    pub fn chain_graph_identity(&self) -> Vec<usize> {
        self.attention.graph_identity().into_iter().chain(self.ffn.graph_identity()).collect()
    }
    pub fn graph_normalized_storage(&self, rows: usize) -> CuteafdDeviceBuffer {
        self.ffn.normalized_storage(rows)
    }
    /// # Safety
    /// The enclosing chain has successfully drained its stream. Values are the
    /// buffer returned by this block's enqueue_ffn, without intervening reuse.
    pub unsafe fn complete_queued_ffn(&mut self, values: CuteafdDeviceBuffer) -> Result<FfnInput<'_>> {
        let Phase::QueuedFfn(binding, rows) = self.phase else {
            anyhow::bail!("block has no queued FFN");
        };
        ensure!(values.bytes == rows * 10240 && values.device_id == self.inputs()[0].device_id,
            "queued normalized FFN extent differs");
        unsafe { self.attention.complete()?; }
        self.phase = Phase::Ffn(binding, rows);
        let mut residual = self.ffn.inputs()[0]; residual.bytes = rows * 40960;
        let mut incoming_pre = self.ffn.inputs()[1]; incoming_pre.bytes = rows * 16;
        Ok(FfnInput { residual, incoming_pre, values, layer: self.layer, tokens: &self.tokens, binding })
    }
    /// Reborrow a completed attention tail without extending the original lane
    /// borrow across executor hooks. The binding rejects stale continuations.
    pub fn ffn_input(&self, binding: QueryBinding) -> Result<FfnInput<'_>> {
        let Phase::Ffn(expected, rows) = self.phase else {
            anyhow::bail!("block has no prepared FFN input");
        };
        ensure!(binding == expected && binding.layer() == self.layer
            && rows == self.tokens.len(), "prepared FFN binding differs");
        let mut residual = self.ffn.inputs()[0]; residual.bytes = rows * 40960;
        let mut incoming_pre = self.ffn.inputs()[1]; incoming_pre.bytes = rows * 16;
        Ok(FfnInput { residual, incoming_pre, values: self.ffn.normalized_storage(rows),
            layer: self.layer, tokens: &self.tokens, binding })
    }
    /// # Safety
    /// Result is the finite completed shared+routed FFN output for the returned
    /// FfnInput binding, in its token order, on this device. It stays immutable
    /// through the copy. Transport/expert reduction must establish this contract.
    pub unsafe fn finish_ffn(&mut self, binding: QueryBinding, result: CuteafdDeviceBuffer) -> Result<BlockOutput<'_>> {
        // Inside a stage chain the next-layer input copies share the ordered
        // mHC stream, so advance() needs no separate drained copy.
        let copy_next = crate::shared::memory::chain::active() && self.layer < 39;
        let rows = unsafe { self.enqueue_finish_ffn(binding, result, copy_next)? };
        if let Err(error) = unsafe { crate::shared::memory::chain::finish(self.library, self.ffn.stream_raw()) } {
            self.reset(); return Err(error);
        }
        unsafe { self.publish_finished_ffn(binding, rows, copy_next) }
    }
    /// # Safety
    /// Retain the completed transport result and exclusive block storage through
    /// completion or cancellation drain. Next-layer copies share the mHC stream.
    pub async unsafe fn finish_ffn_cooperative(&mut self, binding: QueryBinding,
        result: CuteafdDeviceBuffer) -> Result<BlockOutput<'_>> {
        unsafe { self.finish_ffn_with_next_copy(binding,result,self.layer < 39).await }
    }
    /// # Safety
    /// Same completed-result ownership as finish_ffn_cooperative. A peer handoff
    /// consumes the output directly, so no local next-input copy is required.
    pub async unsafe fn finish_ffn_for_handoff_cooperative(&mut self,binding:QueryBinding,
        result:CuteafdDeviceBuffer)->Result<BlockOutput<'_>> {
        unsafe { self.finish_ffn_with_next_copy(binding,result,false).await }
    }
    async unsafe fn finish_ffn_with_next_copy(&mut self,binding:QueryBinding,
        result:CuteafdDeviceBuffer,copy_next:bool)->Result<BlockOutput<'_>> {
        let rows = unsafe { self.enqueue_finish_ffn(binding, result, copy_next)? };
        if let Err(error) = self.ffn.wait_chain().await { self.reset(); return Err(error); }
        unsafe { self.publish_finished_ffn(binding, rows, copy_next) }
    }
    unsafe fn enqueue_next_inputs(&self, rows: usize) -> Result<()> {
        for ((source, destination), bytes) in self.ffn.output_storage().into_iter()
            .zip(self.inputs()).zip([rows * 40960, rows * 16]) {
            unsafe { self.library.copy_d2d_async(destination, source, bytes, self.ffn.stream_raw())?; }
        }
        Ok(())
    }
    unsafe fn enqueue_finish_ffn(&mut self, binding: QueryBinding,
        result: CuteafdDeviceBuffer, copy_next: bool) -> Result<usize> {
        let submitted = (|| -> Result<usize> {
            let Phase::Ffn(expected, rows) = std::mem::replace(&mut self.phase, Phase::Idle) else {
                anyhow::bail!("block FFN is not pending");
            };
            ensure!(binding == expected && result.device_id == self.inputs()[0].device_id
                && result.bytes >= rows * 10240 && !result.ptr.is_null(), "block FFN result binding differs");
            unsafe { crate::shared::memory::chain::join(self.library, self.ffn.stream_raw())?; }
            unsafe { self.ffn.enqueue_finish(Some(result), self.ffn.stream_raw())?; }
            if copy_next { unsafe { self.enqueue_next_inputs(rows)?; } }
            Ok(rows)
        })();
        if submitted.is_err() {
            let drained = unsafe { self.library.cuda_stream_synchronize(self.ffn.stream_raw()) };
            self.reset();
            return submitted.and_then(|rows| drained.map(|()| rows));
        }
        submitted
    }
    unsafe fn publish_finished_ffn(&mut self, binding: QueryBinding,
        rows: usize, copied_next: bool) -> Result<BlockOutput<'_>> {
        if let Err(error) = unsafe { self.ffn.complete() } { self.reset(); return Err(error); }
        self.phase = Phase::Ready(binding, rows, copied_next);
        self.output()
    }
    #[cfg(test)]
    pub async unsafe fn check_queued_finish(&mut self, binding: QueryBinding,
        result: CuteafdDeviceBuffer) -> Result<()> {
        use std::{future::Future, task::Poll};
        let Phase::Ffn(expected, rows) = self.phase else { anyhow::bail!("test requires pending FFN"); };
        let read = |library: &NativeLibrary, buffers: &[CuteafdDeviceBuffer]| -> Result<Vec<Vec<u8>>> {
            buffers.iter().map(|&buffer| { let mut out = vec![0; buffer.bytes];
                library.copy_d2h(&mut out, buffer)?; Ok(out) }).collect()
        };
        let library = self.library;
        let reference = { let out = unsafe { self.finish_ffn(binding, result)? };
            read(library, &[out.residual, out.pre])? };
        self.ffn.rearm_finish_for_test(rows); self.phase = Phase::Ffn(expected, rows);
        let cancelled = {
            let mut future = std::pin::pin!(unsafe { self.finish_ffn_cooperative(binding, result) });
            std::future::poll_fn(|cx| Poll::Ready(future.as_mut().poll(cx).is_pending())).await
        };
        for _ in 0..2 {
            self.ffn.rearm_finish_for_test(rows); self.phase = Phase::Ffn(expected, rows);
            let actual = unsafe { self.finish_ffn_cooperative(binding, result).await? };
            assert_eq!(read(library, &[actual.residual, actual.pre])?, reference);
            if self.layer < 39 {
                let mut input = self.inputs(); input[0].bytes = rows * 40960; input[1].bytes = rows * 16;
                assert_eq!(read(library, &input)?, reference, "queued next-layer inputs differ");
            }
        }
        self.ffn.rearm_finish_for_test(rows); self.phase = Phase::Ffn(expected, rows);
        eprintln!("PASS queued layer finish {}: exact mHC/next-input parity, pending cancellation={cancelled}, reuse", self.layer);
        Ok(())
    }
    pub fn output(&self) -> Result<BlockOutput<'_>> {
        let (binding, rows) = match self.phase {
            Phase::Ready(binding, rows, _) => (binding, rows),
            _ => return Err(anyhow::anyhow!("block output unpublished")),
        };
        ensure!(self.tokens.len() == rows, "block token count differs");
        let [residual, pre] = self.ffn.output().context("block FFN output unavailable")?;
        Ok(BlockOutput {
            binding,
            residual,
            pre,
            layer: self.layer,
            tokens: &self.tokens,
        })
    }
}

#[cfg(test)]
impl<'a> BlockOutput<'a> {
    /// # Safety
    /// Test buffers contain complete residual/pre rows and outlive the output.
    pub(crate) unsafe fn from_test_buffers(layer: usize, tokens: &'a [u64],
        residual: CuteafdDeviceBuffer, pre: CuteafdDeviceBuffer) -> Result<Self> {
        Ok(Self { binding: QueryBinding::new(layer)?, layer, tokens, residual, pre })
    }
}

#[cfg(test)]
mod input_storage_tests {
    use super::*;
    use crate::families::deepseek_v41::{v41_engram::{EngramDeviceRows, layer::EngramGate},
        v41_hc::HcStorage, v41_target_embedding::TargetEmbeddingWave,
        v41_tensors::NativeRtxTensors};
    use crate::shared::memory::{chain::StageChain, row_window::RowWindow};

    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_SNAPSHOT and a GPU"]
    fn windowed_input_units_match_private_runs_behind_a_gated_stream() -> Result<()> {
        // SAFETY: the library outlives all allocations, streams and weights.
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(
            &std::env::var("CUTEAFD_SNAPSHOT")?))?;
        let names = ["embed.weight".to_owned()];
        let table = NativeRtxTensors::load(&library, &catalog, &names,
            NativeRtxTensors::plan(&catalog, &names)?, 1 << 20)?;
        let hc_names = |attention: bool| {
            let kind = if attention { "attn" } else { "ffn" };
            [format!("layers.0.hc_{kind}_fn"), format!("layers.0.hc_{kind}_scale"),
                format!("layers.0.hc_{kind}_base"), format!("layers.0.{kind}_norm.weight")]
        };
        let names = [hc_names(true), hc_names(false)].concat();
        let hc = NativeRtxTensors::load(&library, &catalog, &names,
            NativeRtxTensors::plan(&catalog, &names)?, 1 << 20)?;
        let weights = crate::families::deepseek_v41::v41_engram::layer::EngramLayerWeights::load(
            &library, &catalog, 0, usize::MAX, 1 << 20)?;
        let (unit, reserve) = (16, 48);
        let embedding_storage = TargetEmbeddingWave::storage(&library, reserve, unit)?;
        let upload_storage = EngramDeviceRows::storage(&library, reserve, unit)?;
        let gate_storage = EngramGate::storage(&library, reserve, unit)?;
        let boundaries = [HcStorage::new(&library, reserve)?, HcStorage::new(&library, reserve)?,
            HcStorage::new(&library, reserve)?, HcStorage::new(&library, reserve)?];
        let block = |layer, attention, ffn| -> Result<BackboneBlockWave<'_, '_>> {
            Ok(BackboneBlockWave::new(&library, layer,
                HcSublayer::on_storage(&library, &hc, hc_names(true), attention, unit)?,
                HcSublayer::on_storage(&library, &hc, hc_names(false), ffn, unit)?, unit))
        };
        let mut embeddings = [TargetEmbeddingWave::on_storage(&library, &table, embedding_storage.clone())?,
            TargetEmbeddingWave::on_storage(&library, &table, embedding_storage.clone())?];
        let mut uploads = [EngramDeviceRows::on_storage(upload_storage.clone())?,
            EngramDeviceRows::on_storage(upload_storage.clone())?];
        let mut gates = [EngramGate::on_storage(&weights, gate_storage.clone())?,
            EngramGate::on_storage(&weights, gate_storage.clone())?];
        let mut first = [block(0, boundaries[0].clone(), boundaries[1].clone())?,
            block(0, boundaries[0].clone(), boundaries[1].clone())?];
        let mut next = [block(1, boundaries[2].clone(), boundaries[3].clone())?,
            block(1, boundaries[2].clone(), boundaries[3].clone())?];
        let read = |buffers: &[CuteafdDeviceBuffer]| -> Result<Vec<Vec<u8>>> {
            buffers.iter().map(|&buffer| {
                let mut bytes = vec![0; buffer.bytes]; library.copy_d2h(&mut bytes, buffer)?; Ok(bytes)
            }).collect()
        };
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        // Capture widths, odd eager widths, both encodings, and reuse with
        // swapped windows. Every enqueue must return while the GPU is gated.
        for (cycle, rows) in [1, 13, 7].into_iter().enumerate() {
            let mut tokens: [Vec<u32>; 2] = std::array::from_fn(|seed| (0..rows)
                .map(|i| (42 + i * 7919 + seed * 103 + cycle * 97) as u32 % 129280).collect());
            let features: [Vec<u8>; 2] = std::array::from_fn(|seed| (0..5120)
                .flat_map(|i| (0x3f00u16 + ((i + seed * 17) % 128) as u16).to_ne_bytes()).collect());
            if cycle == 2 { for ids in &mut tokens { ids[rows - 1] = cuteafd_loader::V41_IMAGE_TOKEN_ID; } }
            let images = |seed: usize| if cycle == 2 { vec![(rows - 1, features[seed].as_slice())] }
                else { Vec::new() };
            let positions: [Vec<u64>; 2] = std::array::from_fn(|seed| (0..rows)
                .map(|i| (seed * 4096 + cycle * 31 + i) as u64).collect());
            let packed: [Vec<u8>; 2] = std::array::from_fn(|seed| vec![0x28 + seed as u8;
                rows * 24 * if cycle == 1 { 128 } else { 256 }]);
            let scales: [Vec<u8>; 2] = std::array::from_fn(|seed| vec![if cycle == 1 { 0x38 + seed as u8 } else { 127 };
                rows * 24 * if cycle == 1 { 16 } else { 8 }]);
            let masks: [Vec<u8>; 2] = std::array::from_fn(|seed| (0..rows).map(|i| ((seed + i) % 2) as u8).collect());
            let gathered = |seed: usize| cuteafd_loader::EngramGatherView {
                weights: &packed[seed], scales: &scales[seed], text_mask: &masks[seed], rows,
                layer_index: 0, encoding: if cycle == 1 { cuteafd_loader::EngramEncoding::Nvfp4 }
                    else { cuteafd_loader::EngramEncoding::Fp8 }, global_scale: 1.0,
            };
            for slot in 0..2 {
                let start = if (slot + cycle) % 2 == 0 { 0 } else { unit };
                let window = RowWindow::rows(start..start + rows);
                embeddings[slot].select_window(window.clone())?;
                uploads[slot].select_window(window.clone())?;
                gates[slot].select_window(window.clone())?;
                first[slot].reset(); first[slot].select_window(window.clone())?;
                next[slot].reset(); next[slot].select_window(window)?;
            }
            // Run the exact pass ungated first: first-use CUDA module loading
            // may synchronize the context. Re-poison afterward so warm outputs
            // cannot mask a missing write in the gated pass.
            for gated in [false, true] {
                // Poison all public row planes before the gated production.
                let mut planes = Vec::new();
                for seed in 0..2 {
                    planes.extend(gates[seed].inputs());
                    planes.extend(first[seed].inputs());
                    planes.extend(next[seed].inputs());
                    if gated {
                        let embedding = embeddings[seed].output()?;
                        planes.extend([embedding.residual, embedding.pre]);
                        let upload = uploads[seed].view()?;
                        planes.extend([upload.embeddings, upload.text_mask, gates[seed].output()?]);
                    }
                }
                for plane in planes { library.copy_h2d(plane, &vec![0xa5; plane.bytes])?; }
                let chain = StageChain::new(&library)?;
                chain.set_device_order(true);
                // Gate the first block's stream; embedding joins it before its
                // pinned upload. Subsequent slots cannot make host progress by
                // accidentally synchronizing a legacy-stream copy.
                runtime.block_on(chain.handle().scope(async {
                    // SAFETY: stream remains owned until gate release and drain.
                    let gate = if gated { Some(unsafe { cuteafd_ffi::test_support::CudaStreamGate::new(
                        &library, first[0].attention.stream_raw())? }) } else { None };
                    // SAFETY: publish the gate as the first producer in this chain.
                    unsafe { crate::shared::memory::chain::finish(&library, first[0].attention.stream_raw())?; }
                    for seed in 0..2 {
                        // SAFETY: each block owns a disjoint window until chain drain.
                        let embedding = unsafe { embeddings[seed].execute_with_images_into(
                            &tokens[seed], &positions[seed], &images(seed), first[seed].inputs())? };
                        first[seed].initialize_embedding(&embedding)?;
                        embeddings[seed].release_scratch()?;
                        let [mut residual, mut pre] = first[seed].inputs();
                        residual.bytes = rows * 40960; pre.bytes = rows * 16;
                        let previous = BlockOutput { binding: QueryBinding::new(0)?, residual, pre,
                            layer: 0, tokens: &positions[seed] };
                        next[seed].initialize_previous(&previous)?;
                    }
                    // Upload and consume inside one input hook, then release the
                    // guarded scratch after the block reader has joined.
                    for seed in 0..2 {
                        uploads[seed].upload_cooperative(&gathered(seed)).await?;
                        // SAFETY: windows and gather staging remain owned through drain.
                        unsafe { next[seed].apply_engram(&mut gates[seed], &uploads[seed].view()?)?; }
                        uploads[seed].release_scratch()?;
                    }
                    drop(gate);
                    anyhow::Ok(())
                }))?;
                chain.drain()?;
            }
            for seed in 0..2 {
                let mut private_embedding = TargetEmbeddingWave::new(&library, &table, unit, usize::MAX)?;
                let private = private_embedding.execute_with_images(&tokens[seed], &positions[seed], &images(seed))?;
                let actual = embeddings[seed].output()?;
                assert_eq!(read(&[actual.residual, actual.pre])?, read(&[private.residual, private.pre])?);
                let mut private_upload = EngramDeviceRows::new(&library, unit, usize::MAX)?;
                let view = private_upload.upload(&gathered(seed))?;
                let mut private_gate = EngramGate::new(&weights, unit, usize::MAX)?;
                // SAFETY: private inputs are complete and remain live through execution.
                let expected = unsafe { private_gate.execute_captured(private.residual, &view)? };
                let prepared = next[seed].prepared_input()?;
                assert_eq!(read(&[prepared.residual, prepared.pre])?, read(&[expected, private.pre])?,
                    "input handoff differs for cycle {cycle}, slot {seed}");
            }
            eprintln!("PASS windowed embedding/Engram/block handoffs: cycle={cycle} rows={rows}");
        }
        Ok(())
    }
    // Each arm holds A's fork reader while an independent chain writes B.
    // The negative arm must finish B before release, then observe clobber.
    fn guarded_input_scratch(kind: usize, guarded: bool) -> Result<()> {
        use crate::shared::memory::{DeviceAllocation, device::{Device, Stream}};
        // SAFETY: the library outlives every allocation and stream below.
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let catalog = cuteafd_loader::read_official_v41_catalog(std::path::Path::new(
            &std::env::var("CUTEAFD_SNAPSHOT")?))?;
        let names = ["embed.weight".to_owned()];
        let table = NativeRtxTensors::load(&library, &catalog, &names,
            NativeRtxTensors::plan(&catalog, &names)?, 1 << 20)?;
        let weights = crate::families::deepseek_v41::v41_engram::layer::EngramLayerWeights::load(
            &library, &catalog, 0, usize::MAX, 1 << 20)?;
        let (unit, reserve) = (16, 48);
        let shared_embedding = TargetEmbeddingWave::storage(&library, reserve, unit)?;
        let shared_upload = EngramDeviceRows::storage(&library, reserve, unit)?;
        let shared_gate = EngramGate::storage(&library, reserve, unit)?;
        let mut embeddings = [TargetEmbeddingWave::on_storage(&library, &table, shared_embedding.clone())?,
            TargetEmbeddingWave::on_storage(&library, &table, shared_embedding)?];
        let mut uploads = [EngramDeviceRows::on_storage(shared_upload.clone())?,
            EngramDeviceRows::on_storage(shared_upload)?];
        let mut gates = [EngramGate::on_storage(&weights, shared_gate.clone())?,
            EngramGate::on_storage(&weights, shared_gate)?];
        let widths = [13, 7];
        let tokens: [Vec<u32>; 2] = std::array::from_fn(|seed| (0..widths[seed])
            .map(|i| (42 + i * 7919 + seed * 103) as u32 % 129280).collect());
        let positions: [Vec<u64>; 2] = std::array::from_fn(|seed| (0..widths[seed])
            .map(|i| (seed * 4096 + i) as u64).collect());
        let packed: [Vec<u8>; 2] = std::array::from_fn(|seed| vec![0x28 + seed as u8; widths[seed] * 24 * 256]);
        let scales: [Vec<u8>; 2] = std::array::from_fn(|seed| vec![127; widths[seed] * 24 * 8]);
        let masks: [Vec<u8>; 2] = std::array::from_fn(|seed| vec![1; widths[seed]]);
        let gathered = |seed: usize| cuteafd_loader::EngramGatherView {
            weights: &packed[seed], scales: &scales[seed], text_mask: &masks[seed], rows: widths[seed],
            layer_index: 0, encoding: cuteafd_loader::EngramEncoding::Fp8, global_scale: 1.0,
        };
        // Gate inputs are private and fully prepared before any reader is gated.
        let mut private_uploads = [EngramDeviceRows::new(&library, unit, usize::MAX)?,
            EngramDeviceRows::new(&library, unit, usize::MAX)?];
        let mut residuals = Vec::new();
        for seed in 0..2 {
            private_uploads[seed].upload(&gathered(seed))?;
            let output = embeddings[seed].execute(&tokens[seed], &positions[seed])?;
            let allocation = DeviceAllocation::new(&library, output.residual.bytes)?;
            library.copy_d2d(allocation.buffer, output.residual, output.residual.bytes)?;
            residuals.push(allocation);
            let window = RowWindow::rows(seed * unit..seed * unit + widths[seed]);
            embeddings[seed].select_window(window.clone())?;
            uploads[seed].select_window(window.clone())?;
            gates[seed].select_window(window)?;
        }
        let reader = Stream::new(Device { library: &library, id: library.cuda_get_device()? })?;
        let stride = if kind == 1 { 24 * 512 } else { 40960 };
        let bytes = widths[0] * stride;
        let copy = DeviceAllocation::new(&library, bytes)?;
        let produce = |seed: usize, embeddings: &mut [TargetEmbeddingWave<'_, '_>; 2],
            uploads: &mut [EngramDeviceRows<'_>; 2], gates: &mut [EngramGate<'_, '_>; 2]|
            -> Result<CuteafdDeviceBuffer> {
            Ok(match kind {
                0 => embeddings[seed].execute(&tokens[seed], &positions[seed])?.residual,
                1 => uploads[seed].upload(&gathered(seed))?.embeddings,
                // SAFETY: private inputs are complete, disjoint, and retained.
                _ => unsafe { gates[seed].execute_captured(residuals[seed].buffer,
                    &private_uploads[seed].view()?)? },
            })
        };
        let mut want = vec![0; bytes];
        {
            let mut private_embedding = TargetEmbeddingWave::new(&library, &table, unit, usize::MAX)?;
            let mut private_gate = EngramGate::new(&weights, unit, usize::MAX)?;
            let expected = match kind {
                0 => private_embedding.execute(&tokens[0], &positions[0])?.residual,
                1 => private_uploads[0].view()?.embeddings,
                // SAFETY: both private inputs are initialized and retained.
                _ => unsafe { private_gate.execute_captured(residuals[0].buffer,
                    &private_uploads[0].view()?)? },
            };
            library.copy_d2h(&mut want, expected)?;
        }
        let first = produce(0, &mut embeddings, &mut uploads, &mut gates)?;
        produce(1, &mut embeddings, &mut uploads, &mut gates)?;
        // Warm the exact fork copy on the same reader, then poison its output.
        unsafe { library.copy_d2d_async(copy.buffer, first, bytes, reader.raw)?;
            library.cuda_stream_synchronize(reader.raw)?; }
        library.copy_h2d(copy.buffer, &vec![0xa5; bytes])?;
        let chains = [StageChain::new(&library)?, StageChain::new(&library)?];
        for chain in &chains { chain.set_device_order(true); }
        let runtime = tokio::runtime::Builder::new_current_thread().build()?;
        let reader_gate = unsafe { cuteafd_ffi::test_support::CudaStreamGate::new(&library, reader.raw)? };
        runtime.block_on(chains[0].handle().scope(async {
            let output = produce(0, &mut embeddings, &mut uploads, &mut gates)?;
            // SAFETY: source/copy/reader remain owned through both chain drains.
            unsafe {
                crate::shared::memory::chain::join(&library, reader.raw)?;
                library.copy_d2d_async(copy.buffer, output, bytes, reader.raw)?;
                crate::shared::memory::chain::finish(&library, reader.raw)?;
            }
            match kind { 0 => embeddings[0].release_scratch(), 1 => uploads[0].release_scratch(),
                _ => gates[0].release_scratch() }
        }))?;
        let writer = match kind { 0 => embeddings[1].test_writer(!guarded),
            1 => uploads[1].test_writer(!guarded), _ => gates[1].test_writer(!guarded) };
        let launched = runtime.block_on(chains[1].handle().scope(async {
            produce(1, &mut embeddings, &mut uploads, &mut gates)?;
            anyhow::Ok(())
        }));
        match kind { 0 => { embeddings[1].test_writer(false); },
            1 => { uploads[1].test_writer(false); }, _ => { gates[1].test_writer(false); } }
        launched?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(if guarded { 2 } else { 10 });
        let mut done = false;
        while std::time::Instant::now() < deadline {
            if unsafe { library.cuda_stream_query(writer)? } { done = true; break; }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(done, !guarded, "scratch kind {kind}: B completion before reader release");
        drop(reader_gate);
        for chain in &chains { chain.drain()?; }
        unsafe { library.cuda_stream_synchronize(reader.raw)?; }
        let mut got = vec![0; bytes];
        library.copy_d2h(&mut got, copy.buffer)?;
        if guarded { assert_eq!(got, want, "scratch kind {kind}: guarded reader clobbered"); }
        else { assert_ne!(got, want, "scratch kind {kind}: negative control did not clobber"); }
        eprintln!("PASS input scratch kind={kind} guarded={guarded} writer_done_before_release={done}");
        Ok(())
    }
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_SNAPSHOT and a GPU"]
    fn guarded_input_scratch_waits_for_fork_readers() -> Result<()> {
        for kind in 0..3 { guarded_input_scratch(kind, true)?; }
        Ok(())
    }
    #[test]
    #[ignore = "requires CUTEAFD_NATIVE_LIB, CUTEAFD_SNAPSHOT and a GPU"]
    fn unguarded_input_scratch_clobbers_fork_readers() -> Result<()> {
        for kind in 0..3 { guarded_input_scratch(kind, false)?; }
        Ok(())
    }

}
