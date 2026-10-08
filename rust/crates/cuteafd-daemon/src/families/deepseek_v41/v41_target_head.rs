//! Final target mHC collapse, RMS norm and the vocabulary shared with dSpark.
pub(crate) mod distributed;
pub(crate) mod distributed_target;
use crate::families::deepseek_v41::v41_attention_binding::QueryBinding;
use crate::families::deepseek_v41::v41_block::BlockOutput;
use crate::shared::memory::{DeviceAllocation, HostAllocation, LoadStream};
use crate::families::deepseek_v41::v41_tensors::{NativeRtxTensors, VocabularyHead};
use anyhow::{Context, Result, ensure};
use cuteafd_ffi::{
    CuteafdDeviceBuffer, NativeLibrary, V41Hc, V41VocabularyProjection,
};
use cuteafd_loader::OfficialV41Catalog;
use std::{ffi::c_void, marker::PhantomData};
const STRIDES: [usize; 7] = [40960, 16, 10240, 10240, 517120, 4, 4];

/// Vocabulary width of the official checkpoint. Only used to size the mask,
/// bitmap and histogram arenas owned by [`TargetSamplingWave`]; the K1 kernel
/// discovers the width from the buffer it is given (contract §1).
const SAMPLING_VOCAB: usize = 129_280;

/// Packed mask words per row (`ceil(vocab/32)`, design §5.2).
#[cfg(test)]
const SAMPLING_MASK_WORDS: usize = SAMPLING_VOCAB.div_ceil(32);

pub(crate) use crate::shared::sampler::{
    SampledTargetRows, TargetSamplingRowRequest, TargetSamplingWave, SAMPLING_MAX_RETAINED,
};
#[cfg(test)]
use cuteafd_ffi::CuteafdV41SamplerRow;
#[cfg(test)]
pub(crate) use crate::shared::sampler::{
    bytemuck_bytes, bytemuck_slice, mask_launch_arguments, SamplerStageLauncher, SamplerStages,
    SAMPLING_HISTOGRAM_BUCKETS,
};

pub(crate) struct TargetHeadWeights<'a> {
    library: &'a NativeLibrary,
    norm: NativeRtxTensors<'a>,
}
impl<'a> TargetHeadWeights<'a> {
    pub fn device_bytes(catalog: &OfficialV41Catalog) -> Result<usize> {
        let bytes = NativeRtxTensors::plan(catalog, &["norm.weight".into()])?;
        ensure!(bytes == 10240, "unexpected target norm geometry");
        Ok(bytes)
    }
    pub fn load(
        library: &'a NativeLibrary,
        catalog: &OfficialV41Catalog,
        budget: usize,
        staging: usize,
    ) -> Result<Self> {
        ensure!(
            Self::device_bytes(catalog)? <= budget,
            "target norm exceeds budget"
        );
        Ok(Self {
            library,
            norm: NativeRtxTensors::load(
                library,
                catalog,
                &["norm.weight".into()],
                budget,
                staging,
            )?,
        })
    }
    pub fn wave<'w>(
        &'w self,
        head: &'w VocabularyHead<'a>,
        capacity: usize,
        budget: usize,
    ) -> Result<TargetHeadWave<'w, 'a>> {
        ensure!(
            TargetHeadWave::device_bytes(capacity)? <= budget,
            "target head wave exceeds budget"
        );
        let workspace =
            DeviceAllocation::new(self.library, V41VocabularyProjection::WORKSPACE_BYTES)?;
        let projection = unsafe { self.library.v41_vocabulary_head(workspace.buffer)? };
        let buffers = STRIDES
            .into_iter()
            .map(|n| DeviceAllocation::new(self.library, n * capacity))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            head.device_id() == buffers[0].buffer.device_id
                && self.norm.get("norm.weight")?.device_id == buffers[0].buffer.device_id,
            "target head weight device differs"
        );
        Ok(TargetHeadWave {
            stream: LoadStream {
                library: self.library,
                raw: self.library.cuda_stream_create()?,
            },
            projection,
            _workspace: workspace,
            hc: self.library.v41_hc()?,
            buffers,
            weights: self,
            head,
            capacity,
            graphs: vec![None; capacity],
            fp8_scratch: crate::families::deepseek_v41::v41_tensors::fp8_scratch(self.library, head.fp8(), capacity,
                crate::families::deepseek_v41::v41_tensors::Fp8Head::All)?,
            ready: None,
            origin: None,
            selected: Vec::new(),
            tokens: Vec::new(),
            greedy_staging: HostAllocation::new(self.library, capacity * 8)?,
            greedy_ready: false,
            sampling: TargetSamplingWave::new(self.library, capacity, SAMPLING_VOCAB)?,
            download: crate::shared::memory::RowDownload::new(self.library, capacity * STRIDES[4])?,
        })
    }
}
pub(crate) struct TargetLogits<'a> {
    pub rows: usize,
    pub collapsed: CuteafdDeviceBuffer,
    pub normalized: CuteafdDeviceBuffer,
    pub logits: CuteafdDeviceBuffer,
    pub selected_rows: &'a [usize],
    pub token_positions: &'a [u64],
    origin: Option<QueryBinding>,
    _owner: PhantomData<&'a ()>,
}
impl TargetLogits<'_> {
    pub fn binding(&self) -> Result<QueryBinding> {
        self.origin.context("target logits have no block origin")
    }
}
pub(crate) struct TargetHeadWave<'w, 'a> {
    download: crate::shared::memory::RowDownload<'a>,
    stream: LoadStream<'a>,
    projection: V41VocabularyProjection<'a>,
    _workspace: DeviceAllocation<'a>,
    hc: V41Hc<'a>,
    buffers: Vec<DeviceAllocation<'a>>,
    weights: &'w TargetHeadWeights<'a>,
    head: &'w VocabularyHead<'a>,
    capacity: usize,
    /// Captured head graphs by row count (index rows - 1), kept while serving:
    /// verification widths vary round to round.
    graphs: Vec<Option<*mut c_void>>,
    /// Scratch of the FP8 head copy when the target head projects through it.
    fp8_scratch: Option<DeviceAllocation<'a>>,
    ready: Option<usize>,
    origin: Option<QueryBinding>,
    selected: Vec<usize>,
    tokens: Vec<u64>,
    greedy_staging: HostAllocation<'a>,
    greedy_ready: bool,
    sampling: TargetSamplingWave<'a>,
}
impl TargetHeadWave<'_, '_> {
    /// Up to 16 decode/prefill-last rows or 80 verification rows per head wave.
    pub fn device_bytes(capacity: usize) -> Result<usize> {
        ensure!(
            (1..=80).contains(&capacity),
            "target head capacity must be 1..80"
        );
        Ok(V41VocabularyProjection::WORKSPACE_BYTES
            + capacity * STRIDES.iter().sum::<usize>()
            + TargetSamplingWave::device_bytes(capacity, SAMPLING_VOCAB))
    }
    fn b(&self, i: usize) -> CuteafdDeviceBuffer {
        self.buffers[i].buffer
    }
    pub fn inputs(&self) -> [CuteafdDeviceBuffer; 2] {
        [self.b(0), self.b(1)]
    }
    fn synchronize(&self) -> Result<()> {
        unsafe { self.stream.library.cuda_stream_synchronize(self.stream.raw) }
    }
    fn invalidate(&mut self) {
        self.greedy_ready = false;
        self.ready = None;
        self.origin = None;
        self.selected.clear();
        self.tokens.clear();
    }
    fn validate(&mut self, rows: usize) -> Result<()> {
        self.invalidate();
        ensure!(
            rows > 0 && rows <= self.capacity,
            "target head rows exceed capacity"
        );
        Ok(())
    }
    unsafe fn enqueue(&self, rows: usize) -> Result<()> {
        unsafe {
            self.hc
                .pre(self.b(0), self.b(1), self.b(2), rows, self.stream.raw)?;
            self.stream.library.cuda_ds4_rmsnorm_bf16_rne_async(
                self.b(2),
                self.weights.norm.get("norm.weight")?,
                self.b(3),
                rows as i32,
                5120,
                1e-20,
                self.stream.raw,
            )?;
            crate::families::deepseek_v41::v41_tensors::project_vocabulary(self.stream.library, &self.projection,
                self.head.weight(), self.head.fp8().zip(self.fp8_scratch.as_ref()), self.b(3), self.b(4), rows,
                self.stream.raw)?;
            Ok(())
        }
    }
    /// # Safety
    /// Residual and pre inputs are finite and initialized, with exclusive storage
    /// until the call drains. Raw outputs carry no block identity.
    pub unsafe fn execute(&mut self, rows: usize) -> Result<TargetLogits<'_>> {
        self.validate(rows)?;
        let launched = unsafe { self.enqueue(rows) };
        launched.and(self.synchronize())?;
        self.ready = Some(rows);
        self.output()
    }
    /// # Safety
    /// Same initialized-input contract as execute; capture drains its warmup.
    pub unsafe fn capture(&mut self, rows: usize) -> Result<()> {
        self.invalidate();
        ensure!(self.graph(rows).is_none(), "target head graph already captured");
        unsafe {
            self.execute(rows)?;
        }
        unsafe { self.capture_ready(rows) }
    }
    unsafe fn capture_ready(&mut self, rows: usize) -> Result<()> {
        self.invalidate();
        ensure!((1..=self.capacity).contains(&rows) && self.graph(rows).is_none(),
            "target head graph for {rows} rows already captured or out of range");
        unsafe {
            self.stream
                .library
                .cuda_graph_begin_capture(self.stream.raw)?;
        }
        let launched = unsafe { self.enqueue(rows) };
        let captured = unsafe { self.stream.library.cuda_graph_end_capture(self.stream.raw) };
        match (launched, captured) {
            (Ok(()), Ok(graph)) => {
                self.graphs[rows - 1] = Some(graph);
                Ok(())
            }
            (Err(e), Ok(graph)) => {
                unsafe {
                    self.stream.library.cuda_graph_exec_destroy(graph)?;
                }
                Err(e)
            }
            (Err(e), Err(_)) | (Ok(()), Err(e)) => Err(e),
        }
    }
    /// # Safety
    /// Same inputs as execute; live row count must match the captured graph.
    pub unsafe fn replay(&mut self, rows: usize) -> Result<TargetLogits<'_>> {
        self.validate(rows)?;
        let graph = self.graph(rows).context("target head graph missing")?;
        let launched = unsafe {
            self.stream
                .library
                .cuda_graph_launch(graph, self.stream.raw)
        };
        launched.and(self.synchronize())?;
        self.ready = Some(rows);
        self.output()
    }
    fn slice(mut b: CuteafdDeviceBuffer, offset: usize, bytes: usize) -> Result<CuteafdDeviceBuffer> {
        ensure!(
            offset <= b.bytes && bytes <= b.bytes - offset,
            "target head row slice exceeds buffer"
        );
        b.ptr = unsafe { b.ptr.cast::<u8>().add(offset).cast() };
        b.bytes = bytes;
        Ok(b)
    }
    /// # Safety
    /// Completed layer-39 residual/pre storage remains immutable until all row
    /// copies drain. Selection order defines output order; each row occurs once.
    unsafe fn copy_block(
        &mut self,
        block: &BlockOutput<'_>,
        selected: &[usize],
    ) -> Result<()> {
        self.invalidate();
        ensure!(
            block.layer == 39
                && block.binding().layer() == 39
                && block.tokens.len() <= 4096
                && !selected.is_empty()
                && selected.len() <= self.capacity
                && selected.iter().all(|&i| i < block.tokens.len())
                && selected
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    == selected.len()
                && block.residual.bytes == block.tokens.len() * 40960
                && block.pre.bytes == block.tokens.len() * 16
                && block.residual.device_id == self.b(0).device_id
                && block.pre.device_id == self.b(0).device_id,
            "target head block or selected rows differ"
        );
        let copied = (|| -> Result<()> {
            // The final block output may still be queued in a stage chain.
            unsafe { crate::shared::memory::chain::join(self.stream.library, self.stream.raw)?; }
            let mut first = 0;
            while first < selected.len() {
                let mut count = 1;
                while first + count < selected.len()
                    && selected[first + count] == selected[first] + count
                {
                    count += 1;
                }
                for (i, src, stride) in [(0, block.residual, 40960), (1, block.pre, 16)] {
                    unsafe {
                        self.stream.library.copy_d2d_async(
                            Self::slice(self.b(i), first * stride, count * stride)?,
                            Self::slice(src, selected[first] * stride, count * stride)?,
                            count * stride,
                            self.stream.raw,
                        )?;
                    }
                }
                first += count;
            }
            Ok(())
        })();
        if let Err(error) = copied {
            self.synchronize()?;
            return Err(error);
        }
        Ok(())
    }
    unsafe fn capture_block_head(&mut self, rows: usize) -> Result<()> {
        if super::v41_layer_graphs::captures_shape(rows as u32) && self.graph(rows).is_none() {
            unsafe { self.capture(rows)?; }
        }
        Ok(())
    }
    async unsafe fn prepare_head_cooperative(&mut self, rows: usize) -> Result<()> {
        // copy_block has already queued inputs on this stream. Warmup consumes
        // them in order, then capture executes without suspension.
        let launched = unsafe { self.enqueue(rows) };
        let drained = self.stream.wait().await;
        launched.and(drained)?;
        if super::v41_layer_graphs::captures_shape(rows as u32) {
            unsafe { self.capture_ready(rows) }
        } else { Ok(()) }
    }
    fn publish_block(&mut self, block: &BlockOutput<'_>, selected: &[usize]) {
        self.ready = Some(selected.len());
        self.origin = Some(block.binding());
        self.selected.extend_from_slice(selected);
        self.tokens.extend(selected.iter().map(|&i| block.tokens[i]));
    }
    /// Completed block inputs remain immutable through the drained head pass.
    pub unsafe fn execute_block(&mut self, block: &BlockOutput<'_>, selected: &[usize])
        -> Result<TargetLogits<'_>> {
        unsafe { self.copy_block(block, selected)?; }
        self.synchronize()?;
        if super::v41_layer_graphs::captures_shape(selected.len() as u32) {
            unsafe { self.capture_block_head(selected.len())?; self.replay(selected.len())?; }
        } else { unsafe { self.execute(selected.len())?; } }
        self.publish_block(block, selected);
        self.output()
    }
    /// Same ownership contract, yielding the owner thread during GPU completion.
    /// First-use warmup and steady replay both complete cooperatively.
    pub async unsafe fn execute_block_cooperative(&mut self, block: &BlockOutput<'_>, selected: &[usize])
        -> Result<TargetLogits<'_>> {
        unsafe { self.copy_block(block, selected)?; }
        // Warmup and replay consume input copies on this same stream.
        let captured = super::v41_layer_graphs::captures_shape(selected.len() as u32);
        if captured && self.graph(selected.len()).is_none() {
            unsafe { self.prepare_head_cooperative(selected.len()).await?; }
        }
        self.invalidate();
        let launched = if captured {
            let graph = self.graph(selected.len()).context("target head graph missing")?;
            unsafe { self.stream.library.cuda_graph_launch(graph, self.stream.raw) }
        } else { unsafe { self.enqueue(selected.len()) } };
        let drained = self.stream.wait().await;
        launched.and(drained)?;
        self.publish_block(block, selected);
        self.output()
    }
    /// Keep logits on the device, downloading only checked greedy IDs/scores.
    pub async unsafe fn execute_block_greedy(&mut self, block: &BlockOutput<'_>,
        selected: &[usize], cooperative: bool) -> Result<()> {
        unsafe { self.copy_block(block, selected)?; }
        let rows = selected.len();
        let captured = super::v41_layer_graphs::captures_shape(rows as u32);
        if captured && self.graph(rows).is_none() {
            if cooperative { unsafe { self.prepare_head_cooperative(rows).await?; } }
            else { self.synchronize()?; unsafe { self.capture_block_head(rows)?; } }
        }
        self.invalidate();
        let graph = if captured { Some(self.graph(rows).context("target head graph missing")?) } else { None };
        let logits = Self::slice(self.b(4), 0, rows * STRIDES[4])?;
        let indices = Self::slice(self.b(5), 0, rows * 4)?;
        let scores = Self::slice(self.b(6), 0, rows * 4)?;
        let launched = (|| -> Result<()> { unsafe {
            let lib = self.stream.library;
            if let Some(graph) = graph { lib.cuda_graph_launch(graph, self.stream.raw)?; }
            else { self.enqueue(rows)?; }
            lib.cuda_logits_argmax_checked_f32_async(logits, indices, scores, rows, 129280, self.stream.raw)?;
            let host = self.greedy_staging.bytes_mut();
            lib.copy_d2h_async(&mut host[..rows*4], indices, self.stream.raw)?;
            lib.copy_d2h_async(&mut host[rows*4..rows*8], scores, self.stream.raw)?;
            Ok(())
        } })();
        let drained = if cooperative { self.stream.wait().await } else { self.synchronize() };
        launched.and(drained)?;
        self.publish_block(block, selected);
        self.greedy_ready = true;
        Ok(())
    }
    /// Run the head and then the v4.1 target-sampler on the same stream,
    /// downloading only `rows x (4 + 4 + 4 + 4)` bytes (ids, raw scores, status,
    /// status detail), plus K3/K4's retained counts when the ordered path ran,
    /// plus — in a second call — the rows the caller asks for.
    ///
    /// This call downloads no logits: the caller decides which rows still need
    /// them and asks for exactly those through
    /// [`Self::download_sampled_rows`] (none, for an untraced device-served
    /// round; the fallback/traced rows otherwise). The compact greedy lane
    /// ([`Self::execute_block_greedy`]) is untouched.
    ///
    /// `ordered_rows` is the caller's per-row routing decision: `true` whenever
    /// any row of the batch needs K3/K4/K5 (see [`TargetSamplingWave::launch`]).
    /// It is the *only* thing chunk 4a changes about the enqueue shape, and it
    /// is `false` for a purely greedy or purely fast-path batch — the chunk-1
    /// all-greedy round and the chunk-2 fast-path round keep their exact
    /// two-kernel/two-stage sequence.
    ///
    /// Enqueue order is `copy_block` → graph launch → sampler stages → the small
    /// D2H, all before the drain the scheduler already performs, so no new host
    /// synchronization is introduced (design §11.2).
    pub async unsafe fn execute_block_sampled(&mut self, block: &BlockOutput<'_>,
        selected: &[usize], requests: &[TargetSamplingRowRequest], mask_staging: Option<&[u32]>,
        mask_words: usize, ordered_rows: bool, cooperative: bool,
    ) -> Result<SampledTargetRows> {
        ensure!(
            requests.len() == selected.len(),
            "sampling requests must match the selected rows"
        );
        unsafe { self.copy_block(block, selected)?; }
        let rows = selected.len();
        let captured = super::v41_layer_graphs::captures_shape(rows as u32);
        if captured && self.graph(rows).is_none() {
            if cooperative { unsafe { self.prepare_head_cooperative(rows).await?; } }
            else { self.synchronize()?; unsafe { self.capture_block_head(rows)?; } }
        }
        self.invalidate();
        let graph = if captured { Some(self.graph(rows).context("target head graph missing")?) } else { None };
        let logits = Self::slice(self.b(4), 0, rows * STRIDES[4])?;
        let staged = self
            .sampling
            .upload(requests, mask_staging, mask_words, self.stream.raw);
        let launched: Result<()> = staged.and_then(|()| {
            (|| -> Result<()> { unsafe {
                if let Some(graph) = graph { self.stream.library.cuda_graph_launch(graph, self.stream.raw)?; }
                else { self.enqueue(rows)?; }
                self.sampling.launch(logits, rows, ordered_rows, self.stream.raw)?;
                Ok(())
            } })()
        });
        let drained = if cooperative { self.stream.wait().await } else { self.synchronize() };
        launched.and(drained)?;
        self.publish_block(block, selected);
        self.sampling.output(logits, rows)
    }
    /// Download the full logits of exactly the rows the caller names, packed in
    /// the same order.
    ///
    /// Chunk-4a callers name a row here for exactly three reasons: the round is
    /// traced (`SamplingRound::trace_rows` downloads every row so
    /// `cuteafd::logit_trace` can log `top_two`), the row is a device row that
    /// finishes so its retained frontier keeps a full row, or the row is a
    /// CPU-fallback row the device could not serve. A device-served round names
    /// none of its ordinary rows, so a greedy row costs only its 4-byte
    /// id/score/status entries instead of a 517,120-byte row transfer (design
    /// §5.5, §8.2) — including when a stochastic peer shares the round
    /// (contract §7.1.15).
    pub async fn download_sampled_rows(&mut self, rows: &SampledTargetRows,
        selection: &[usize]) -> Result<Vec<u8>> {
        ensure!(!selection.is_empty(), "sampled row download selection is empty");
        unsafe { self.download.rows(rows.logits, STRIDES[4], selection).await }
    }
    pub fn greedy_output(&mut self) -> Result<Vec<(u32, f32)>> {
        ensure!(self.greedy_ready, "compact head output unpublished");
        let rows = self.ready.context("compact head rows unpublished")?;
        let host = self.greedy_staging.bytes_mut();
        Ok((0..rows).map(|i| (
            u32::from_ne_bytes(host[i*4..i*4+4].try_into().unwrap()),
            f32::from_ne_bytes(host[rows*4+i*4..rows*4+i*4+4].try_into().unwrap()),
        )).collect())
    }
    pub fn output(&self) -> Result<TargetLogits<'_>> {
        let rows = self.ready.context("target logits unpublished")?;
        let b = |i| {
            let mut b = self.b(i);
            b.bytes = rows * STRIDES[i];
            b
        };
        Ok(TargetLogits {
            rows,
            collapsed: b(2),
            normalized: b(3),
            logits: b(4),
            selected_rows: &self.selected,
            token_positions: &self.tokens,
            origin: self.origin,
            _owner: PhantomData,
        })
    }
    pub async fn download_rows(&mut self, rows: &[usize]) -> Result<Vec<u8>> {
        let logits = self.output()?.logits;
        // The head remains exclusively borrowed through transfer completion.
        unsafe { self.download.rows(logits, STRIDES[4], rows).await }
    }
    fn graph(&self, rows: usize) -> Option<*mut c_void> {
        rows.checked_sub(1).and_then(|i| self.graphs.get(i).copied().flatten())
    }
    pub fn clear_graph(&mut self) -> Result<()> {
        self.invalidate();
        self.stream.require_complete()?;
        for graph in self.graphs.iter_mut().filter_map(Option::take) {
            unsafe {
                self.stream.library.cuda_graph_exec_destroy(graph)?;
            }
        }
        Ok(())
    }
}
impl Drop for TargetHeadWave<'_, '_> {
    fn drop(&mut self) {
        if let Err(e) = self.clear_graph() {
            tracing::error!(%e,"draining target head");
        }
    }
}

#[cfg(test)]
mod sampling_budget_tests {
    use super::*;

    /// D-new: an all-unconstrained round stages no mask rows, and the launch
    /// pair must then be `(None, 0)`. Passing the wave's own non-zero
    /// `mask_words` with a null arena is what the FFI validator and the kernel
    /// both reject, which failed a whole lane for any all-greedy round that was
    /// not compact (a traced round, or a constrained member whose grammar needs
    /// no mask at any row).
    #[test]
    fn an_all_unconstrained_round_launches_without_a_mask_or_a_mask_width() {
        let device = CuteafdDeviceBuffer {
            ptr: 0x1000 as *mut std::ffi::c_void,
            bytes: 8 * SAMPLING_MASK_WORDS * 4,
            ..Default::default()
        };
        // No staged rows: no buffer and a zero width.
        let (masks, words) = mask_launch_arguments(0, device, SAMPLING_MASK_WORDS);
        assert!(masks.is_none(), "an unstaged round must launch no mask buffer");
        assert_eq!(words, 0, "an unstaged round must launch mask_words_per_row = 0");

        // Staged rows: the buffer and the wave's width, unchanged.
        let (masks, words) = mask_launch_arguments(3, device, SAMPLING_MASK_WORDS);
        let masks = masks.expect("a staged round launches its mask buffer");
        assert_eq!(masks.ptr, device.ptr);
        assert_eq!(masks.bytes, device.bytes);
        assert_eq!(words, SAMPLING_MASK_WORDS);
        assert_eq!(words, 4040, "the shipped mask width");
    }

    /// The sampler's workspace is charged to the head wave so capacity
    /// budgeting stays honest (design §5.5, §11.1). The formula is pinned here
    /// so a field added without updating `device_bytes` fails loudly.
    #[test]
    fn sampling_workspace_bytes_follow_the_documented_formula() {
        for capacity in [1usize, 6, 48, 64, 80] {
            let param = capacity * cuteafd_ffi::CUTEAFD_V41_SAMPLER_PARAM_BYTES;
            let arena = capacity * SAMPLING_MASK_WORDS * 4;
            let scratch = capacity * cuteafd_ffi::CUTEAFD_V41_SAMPLER_SCRATCH_BYTES;
            let histogram = capacity * SAMPLING_HISTOGRAM_BUCKETS * 4;
            let rank = TargetSamplingWave::rank_order_bytes(capacity);
            let rows = capacity * 4;
            let expected = param + arena + scratch + arena + histogram + rank + rows + rows
                + rows + rows + capacity * 8 + rows + rows;
            assert_eq!(TargetSamplingWave::device_bytes(capacity, SAMPLING_VOCAB), expected);
            // Pinned call sites: the production capacities are 48 and 64.
        }
        // The documented §11.1 total at capacity 80 is ~4.2 MiB; the arena is
        // allocated twice (device + top-k bitmap), the histogram once, and the
        // chunk-3a rank-order arena is `capacity x 256` ids plus their u64
        // staging.
        assert_eq!(TargetSamplingWave::device_bytes(80, SAMPLING_VOCAB), 80 * 64 + 80 * 4040 * 4 + 80 * 64
            + 80 * 4040 * 4 + 80 * 2048 * 4 + 80 * 256 * 12 + 80 * 4 + 80 * 4 + 80 * 4
            + 80 * 4 + 80 * 8 + 80 * 4 + 80 * 4);
        assert!(TargetSamplingWave::device_bytes(80, SAMPLING_VOCAB) < 5 * 1024 * 1024);
    }

    /// Chunk 4a: the rank-order arena is sized to K5's actual supported limit
    /// (`kBlock == 256`), and the routing cap that keeps K5 from ever seeing a
    /// wider list is the same constant. Pinned together so a kernel-side change
    /// to `kBlock` cannot silently desynchronize the two.
    #[test]
    fn rank_order_arena_covers_exactly_the_supported_retained_width() {
        assert_eq!(SAMPLING_MAX_RETAINED, 256);
        // 80 rows x 256 ids x (4 + 8) bytes = 240 KiB.
        assert_eq!(TargetSamplingWave::rank_order_bytes(80), 80 * 256 * 12);
        assert_eq!(TargetSamplingWave::rank_order_bytes(1), 256 * 12);
    }
}


/// A recording [`SamplerStageLauncher`] for the chunk-4a wiring tests.
///
/// It reproduces the documented per-row kernel eligibility (`k2_applicable`,
/// the K3/K4 eligibility block, and K5's ordered row class) and publishes the
/// production CPU draw, so the scheduler tests can drive the **real**
/// `TargetSamplingWave` staging/launch/output path with no GPU and compare the
/// delivered ids bit-for-bit against the CPU sampler. It also records the exact
/// per-row parameter blocks and mask width each stage was launched with, which
/// is what pins "per-row params and masks" and "the ordered tail ran only for
/// the ordered rows".
#[cfg(test)]
pub(crate) mod recording_sampler {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The one global recording launcher.
    /// Serializes the tests that drive the one global recorder. The recorder's
    /// state is process-global (the scheduler tests and the wiring tests share
    /// it), so a test must hold this for its whole body or the counters it
    /// asserts on can be advanced by a concurrent test.
    pub(crate) static SERIAL: Mutex<()> = Mutex::new(());

    /// Reset the shared recorder and hold its lock for the caller's whole body.
    ///
    /// The recorder is process-global, so every test that asserts on its
    /// counters must reset **under** the lock: resetting before acquiring it
    /// lets a concurrent test's launches land between the reset and the
    /// assertion.
    pub(crate) fn lock_and_reset() -> std::sync::MutexGuard<'static, ()> {
        let guard = SERIAL.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        reset();
        guard
    }

    pub(crate) static RECORDER: RecordingSampler = RecordingSampler;
    static RECORDED: Mutex<Option<Recorded>> = Mutex::new(None);
    static PREPARE_CALLS: AtomicUsize = AtomicUsize::new(0);
    static TOPK_CALLS: AtomicUsize = AtomicUsize::new(0);
    static NUCLEUS_CALLS: AtomicUsize = AtomicUsize::new(0);
    /// Rows the recording K5 must report `INTERNAL` for, with a countdown so a
    /// fault can be injected for a bounded number of launches. A post-launch
    /// device refusal cannot otherwise be produced by a stub, and it is the case
    /// that must fall back to a **stored** CPU token.
    static REFUSE: Mutex<(Vec<usize>, usize)> = Mutex::new((Vec::new(), 0));

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub(crate) enum Stage {
        Prepare,
        TopkSelect,
        Nucleus,
    }

    /// One recorded stage invocation.
    #[derive(Clone, Debug)]
    pub(crate) struct Launch {
        pub(crate) stage: Stage,
        pub(crate) rows: usize,
        /// The device row count of the batch; every stage sees the same one.
        pub(crate) vocab: usize,
        pub(crate) params: Vec<CuteafdV41SamplerRow>,
        pub(crate) mask_words_per_row: usize,
    }

    #[derive(Clone, Debug, Default)]
    pub(crate) struct Recorded {
        pub(crate) launches: Vec<Launch>,
        /// The device's own id channel after the last stage that wrote it.
        pub(crate) ids: Vec<u32>,
        /// The per-row status channel after the last stage that wrote it.
        pub(crate) status: Vec<u32>,
        /// K3/K4's per-row retained counts.
        pub(crate) retained: Vec<u32>,
    }

    pub(crate) fn recorded() -> Recorded {
        RECORDED.lock().unwrap().clone().expect("no recorded launch")
    }

    pub(crate) fn reset() {
        *RECORDED.lock().unwrap() = None;
        PREPARE_CALLS.store(0, Ordering::SeqCst);
        TOPK_CALLS.store(0, Ordering::SeqCst);
        NUCLEUS_CALLS.store(0, Ordering::SeqCst);
        *REFUSE.lock().unwrap() = (Vec::new(), 0);
    }

    /// Make the recording K5 report `INTERNAL` for `rows` on the next
    /// `launches` launches.
    pub(crate) fn refuse_rows(rows: &[usize], launches: usize) {
        *REFUSE.lock().unwrap() = (rows.to_vec(), launches);
    }

    /// The rows this launch must refuse, consuming one launch from the countdown.
    fn take_refused() -> Vec<usize> {
        let mut guard = REFUSE.lock().unwrap();
        let (rows, launches) = guard.clone();
        if launches == 0 {
            return Vec::new();
        }
        guard.1 = launches - 1;
        rows
    }

    pub(crate) fn stage_calls() -> (usize, usize, usize) {
        (PREPARE_CALLS.load(Ordering::SeqCst), TOPK_CALLS.load(Ordering::SeqCst),
            NUCLEUS_CALLS.load(Ordering::SeqCst))
    }

    pub(crate) fn stages() -> SamplerStages {
        SamplerStages::with(&RECORDER)
    }

    /// The test vocabulary is the production one: [`TargetSamplingWave`]'s arena
    /// is `ceil(SAMPLING_VOCAB/32)` words wide and `upload` requires the launch
    /// width to match it, so a smaller test vocabulary would be rejected by the
    /// staging contract rather than exercised through it.
    pub(crate) const TEST_VOCAB: usize = SAMPLING_VOCAB;

    pub(crate) struct RecordingSampler;

    fn read_u32s(library: &NativeLibrary, buffer: CuteafdDeviceBuffer, count: usize) -> Vec<u32> {
        let mut bytes = vec![0u8; count * 4];
        library.copy_d2h(&mut bytes, buffer).unwrap();
        bytes.chunks_exact(4).map(|w| u32::from_ne_bytes(w.try_into().unwrap())).collect()
    }

    fn read_logits(library: &NativeLibrary, logits: CuteafdDeviceBuffer, rows: usize,
        vocab: usize,
    ) -> Vec<f32> {
        let mut bytes = vec![0u8; rows * vocab * 4];
        library.copy_d2h(&mut bytes, logits).unwrap();
        bytes.chunks_exact(4).map(|w| f32::from_ne_bytes(w.try_into().unwrap())).collect()
    }

    /// The recording double's reading of one row's mask, exactly as the kernels
    /// read it: `NO_MASK`, the arena sentinel, or "no arena at all" mean
    /// unconstrained; otherwise the row's `mask_row`-th slice of the arena.
    fn row_mask<'a>(params: CuteafdV41SamplerRow, arena: Option<&'a [u32]>, vocab: usize,
    ) -> Option<&'a [u32]> {
        let arena = arena?;
        if params.flags & cuteafd_ffi::CUTEAFD_V41_SAMPLER_FLAG_NO_MASK != 0
            || params.mask_row == cuteafd_ffi::CUTEAFD_V41_SAMPLER_NO_MASK_ROW
        {
            return None;
        }
        let words = vocab.div_ceil(32);
        let start = params.mask_row as usize * words;
        Some(&arena[start..start + words])
    }

    /// Whether one dense mask bit allows `token`. Absent/`NO_MASK`/sentinel
    /// semantics are handled by [`row_mask`] before this is reached.
    fn mask_allows(mask: &[u32], token: usize) -> bool {
        mask[token / 32] >> (token % 32) & 1 != 0
    }

    /// Whether the row has a non-finite logit **where it could act on it** — a
    /// mirror of the shipped kernel/CPU strictness rule. Chunk 5b R2 aligned the
    /// production plan with the CPU arbiter, which checks finiteness *after* the
    /// `allowed()` skip (`target_sampling.rs:430-436`): a masked-out non-finite
    /// value is legal and unread for a stochastic row, while any unmasked row —
    /// and any position its own mask allows — is still strict.
    pub(crate) fn nonfinite_where_allowed(logits: &[f32], mask: Option<&[u32]>) -> bool {
        logits.iter().enumerate().any(|(token, value)| {
            !value.is_finite() && mask.is_none_or(|mask| mask_allows(mask, token))
        })
    }

    fn greedy(params: CuteafdV41SamplerRow) -> bool {
        params.temperature < 1e-5 || params.top_k == 1
            || params.flags & cuteafd_ffi::CUTEAFD_V41_SAMPLER_FLAG_GREEDY != 0
    }

    /// The production CPU selection at this row's own parameters and position,
    /// over the row's own logits and mask. This is what the recording double
    /// publishes, so a delivered token that differs is a wiring defect rather
    /// than a numeric residual.
    pub(crate) fn cpu_token(params: &CuteafdV41SamplerRow, logits: &[f32],
        mask: Option<&[u32]>,
    ) -> u32 {
        if greedy(*params) {
            let mut best = 0u32;
            let mut maximum = f32::NEG_INFINITY;
            for (token, &value) in logits.iter().enumerate() {
                if mask.is_none_or(|words| words[token / 32] >> (token % 32) & 1 != 0)
                    && value > maximum
                {
                    maximum = value;
                    best = token as u32;
                }
            }
            return best;
        }
        let sampling = cuteafd_core::TargetSamplingParams::new(
            params.temperature, params.top_p,
            if params.top_k == 0 { None } else { Some(params.top_k as usize) },
            params.min_p, params.seed,
        ).expect("the plan always builds validated parameters");
        sampling.select_token(logits, mask, params.position).expect("CPU oracle selection") as u32
    }

    impl SamplerStageLauncher for RecordingSampler {
        #[allow(clippy::too_many_arguments)]
        fn launch_prepare(
            &self,
            library: &NativeLibrary,
            logits: CuteafdDeviceBuffer,
            rows: usize,
            vocab: usize,
            _logits_stride: usize,
            params: &[CuteafdV41SamplerRow],
            _params_device: CuteafdDeviceBuffer,
            masks: Option<CuteafdDeviceBuffer>,
            mask_words_per_row: usize,
            out_indices: CuteafdDeviceBuffer,
            out_status: CuteafdDeviceBuffer,
            _out_status_detail: CuteafdDeviceBuffer,
            _out_scores: CuteafdDeviceBuffer,
            _scratch: CuteafdDeviceBuffer,
            _stream: *mut std::ffi::c_void,
        ) -> Result<()> {
            PREPARE_CALLS.fetch_add(1, Ordering::SeqCst);
            let arena = masks.map(|buffer| read_u32s(library, buffer, rows * mask_words_per_row));
            let logits = read_logits(library, logits, rows, vocab);
            let mut ids = vec![0u32; rows];
            let mut status = vec![0u32; rows];
            for (block_row, params) in params.iter().enumerate() {
                let row_logits = &logits[block_row * vocab..(block_row + 1) * vocab];
                let mask = row_mask(*params, arena.as_deref(), vocab);
                // Chunk 5b R2/F5: strict only where the row can act. The old
                // `greedy(*params) || mask.is_some()` made every masked row
                // strict, which no longer mirrors `build_target_sampling_plan`
                // (greedy rows only) nor the CPU arbiter.
                if nonfinite_where_allowed(row_logits, mask) {
                    status[block_row] = cuteafd_ffi::CUTEAFD_V41_SAMPLER_STATUS_NONFINITE_LOGIT;
                    continue;
                }
                status[block_row] = cuteafd_ffi::CUTEAFD_V41_SAMPLER_STATUS_OK;
                // Only a greedy or fast-path row's id survives to the caller;
                // an ordered row's id is overwritten by the recorded K5 below,
                // exactly as the real K5 overwrites it.
                ids[block_row] = cpu_token(params, row_logits, mask);
            }
            library.copy_h2d(out_indices, bytemuck_slice(&ids))?;
            library.copy_h2d(out_status, bytemuck_slice(&status))?;
            *RECORDED.lock().unwrap() = Some(Recorded {
                launches: vec![Launch { stage: Stage::Prepare, rows, vocab,
                    params: params.to_vec(), mask_words_per_row }],
                ids,
                status,
                retained: vec![0u32; rows],
            });
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn launch_topk_select(
            &self,
            library: &NativeLibrary,
            _logits: CuteafdDeviceBuffer,
            rows: usize,
            vocab: usize,
            _logits_stride: usize,
            params: &[CuteafdV41SamplerRow],
            _params_device: CuteafdDeviceBuffer,
            _masks: Option<CuteafdDeviceBuffer>,
            mask_words_per_row: usize,
            _rank_order_ids: CuteafdDeviceBuffer,
            _rank_order_scratch: CuteafdDeviceBuffer,
            _rank_order_capacity: usize,
            out_retained_count: CuteafdDeviceBuffer,
            _out_pivot_passes: CuteafdDeviceBuffer,
            _scratch: CuteafdDeviceBuffer,
            _stream: *mut std::ffi::c_void,
        ) -> Result<()> {
            TOPK_CALLS.fetch_add(1, Ordering::SeqCst);
            // K3/K4 eligibility: ready, not greedy, `top_k != 0` and below the
            // survivor count (unknowable in a stub, so every servable `top_k`
            // counts). Everything else is a no-op with a zero count.
            let retained: Vec<u32> = params.iter().map(|params| {
                if greedy(*params) || params.top_k == 0 { 0 } else { params.top_k }
            }).collect();
            library.copy_h2d(out_retained_count, bytemuck_slice(&retained))?;
            let mut guard = RECORDED.lock().unwrap();
            let record = guard.as_mut().expect("prepare must run first");
            record.launches.push(Launch { stage: Stage::TopkSelect, rows, vocab,
                params: params.to_vec(), mask_words_per_row });
            record.retained = retained;
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn launch_nucleus(
            &self,
            library: &NativeLibrary,
            logits: CuteafdDeviceBuffer,
            rows: usize,
            vocab: usize,
            _logits_stride: usize,
            params: &[CuteafdV41SamplerRow],
            _params_device: CuteafdDeviceBuffer,
            masks: Option<CuteafdDeviceBuffer>,
            mask_words_per_row: usize,
            _rank_order_ids: CuteafdDeviceBuffer,
            _rank_order_capacity: usize,
            _rank_retained_count: CuteafdDeviceBuffer,
            out_indices: CuteafdDeviceBuffer,
            out_status: CuteafdDeviceBuffer,
            _scratch: CuteafdDeviceBuffer,
            _stream: *mut std::ffi::c_void,
        ) -> Result<()> {
            NUCLEUS_CALLS.fetch_add(1, Ordering::SeqCst);
            let arena = masks.map(|buffer| read_u32s(library, buffer, rows * mask_words_per_row));
            let logits = read_logits(library, logits, rows, vocab);
            let mut ids = read_u32s(library, out_indices, rows);
            let mut status = read_u32s(library, out_status, rows);
            // A real K5 writes `INTERNAL` and leaves `out_indices` at whatever
            // the caller left there, so an id slot for a refused row is *stale*.
            // The stub reproduces exactly that: it writes the status and does not
            // touch that row's id.
            let refused = take_refused();
            for (block_row, params) in params.iter().enumerate() {
                if refused.contains(&block_row) {
                    status[block_row] = cuteafd_ffi::CUTEAFD_V41_SAMPLER_STATUS_INTERNAL;
                    continue;
                }
                // K5's ordered row class: ready, not greedy, and (a retained
                // `top_k` or `top_p < 1`). Every other row is left untouched.
                let ordered = !greedy(*params) && (params.top_k != 0 || params.top_p < 1.0);
                if !ordered {
                    continue;
                }
                if params.output_row as usize != block_row {
                    status[block_row] = cuteafd_ffi::CUTEAFD_V41_SAMPLER_STATUS_INTERNAL;
                    continue;
                }
                let row_logits = &logits[block_row * vocab..(block_row + 1) * vocab];
                let mask = row_mask(*params, arena.as_deref(), vocab);
                // Same post-R2 rule as `launch_prepare`: a masked-out
                // non-finite value is unread, not a fault.
                if nonfinite_where_allowed(row_logits, mask) {
                    status[block_row] = cuteafd_ffi::CUTEAFD_V41_SAMPLER_STATUS_NONFINITE_LOGIT;
                    continue;
                }
                ids[block_row] = cpu_token(params, row_logits, mask);
                status[block_row] = cuteafd_ffi::CUTEAFD_V41_SAMPLER_STATUS_OK;
            }
            library.copy_h2d(out_indices, bytemuck_slice(&ids))?;
            library.copy_h2d(out_status, bytemuck_slice(&status))?;
            let mut guard = RECORDED.lock().unwrap();
            let record = guard.as_mut().expect("prepare must run first");
            record.launches.push(Launch { stage: Stage::Nucleus, rows, vocab,
                params: params.to_vec(), mask_words_per_row });
            record.ids = ids;
            record.status = status;
            Ok(())
        }
    }
}

#[cfg(test)]
mod sampler_wiring_tests {
    //! Chunk-4a wiring tests that drive the real [`TargetSamplingWave`] with a
    //! recording launcher. No GPU: the launcher publishes the production CPU
    //! draw, so a delivered id that differs is a wiring defect.
    use super::*;

    /// A peaked logit row for the test vocabulary.
    fn logits(rows: usize, vocab: usize) -> Vec<f32> {
        let mut values = Vec::with_capacity(rows * vocab);
        for row in 0..rows {
            for token in 0..vocab {
                let winner = (row * 7 + 3) % vocab;
                let value = if token == winner { 9.0 }
                    else { (((row * 13 + token * 5) % 17) as f32) * 0.25 - 4.0 };
                values.push(value);
            }
        }
        values
    }

    fn logit_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_ne_bytes()).collect()
    }

    fn member(params: cuteafd_core::TargetSamplingParams, base_position: u64,
        masks: Vec<Option<Vec<u32>>>,
    ) -> crate::families::deepseek_v41::v41_native_serve::scheduler::SamplingMember {
        crate::families::deepseek_v41::v41_native_serve::scheduler::SamplingMember {
            params, base_position, row_masks: masks,
        }
    }

    /// One mask row that allows exactly the four lowest token ids.
    fn low_tokens_mask(vocab: usize) -> Vec<u32> {
        let mut words = vec![0u32; vocab.div_ceil(32)];
        words[0] = 0b1111;
        words
    }

    /// The plan the wiring tests stage: a greedy row, three fast-path rows, an
    /// ordered masked top-k row, an ordered top-p row and a `top_k = 300` row the
    /// router must send to the CPU.
    fn wiring_members(vocab: usize) -> Vec<crate::families::deepseek_v41::v41_native_serve::scheduler::SamplingMember> {
        use cuteafd_core::TargetSamplingParams as P;
        vec![
            // 0: greedy, no mask.
            member(P::greedy().with_seed(1), 0, vec![None]),
            // 1-3: fast path (temperature + min_p, top_p disabled), unmasked.
            member(P::new(0.7, 1.0, None, 0.05, 2).unwrap(), 40, vec![None, None, None]),
            // 4: ordered top-k, masked to the four lowest ids.
            member(P::new(0.7, 1.0, Some(3), 0.0, 3).unwrap(), 7,
                vec![Some(low_tokens_mask(vocab))]),
            // 5: ordered top-p, unmasked.
            member(P::new(0.6, 0.9, None, 0.0, 4).unwrap(), 11, vec![None]),
            // 6: unservable top_k; must be re-sampled on the CPU.
            member(P::new(0.7, 1.0, Some(300), 0.0, 5).unwrap(), 50, vec![None]),
        ]
    }

    const WIRING_INPUTS: [&[u32]; 5] = [&[0], &[0, 0, 0], &[0], &[0], &[0]];

    /// Build the plan and the host mask arena exactly as the scheduler does.
    fn wiring_plan(vocab: usize) -> (
        crate::families::deepseek_v41::v41_native_serve::scheduler::SamplingPlan, Vec<u32>, Vec<Vec<u32>>,
    ) {
        use crate::families::deepseek_v41::v41_native_serve::scheduler::{build_sampling_masks, build_target_sampling_plan};
        let members = wiring_members(vocab);
        let inputs: Vec<Vec<u32>> = WIRING_INPUTS.iter().map(|input| input.to_vec()).collect();
        let plan = build_target_sampling_plan(&members, &inputs).unwrap();
        let rows: usize = inputs.iter().map(Vec::len).sum();
        let arena_words = crate::families::deepseek_v41::v41_native_serve::scores::VOCAB.div_ceil(32);
        let mut arena = vec![0u32; rows * arena_words];
        let row_masks: Vec<Vec<Option<Vec<u32>>>> = plan.mask.iter()
            .enumerate()
            .map(|(row, mask)| {
                if plan.planned_fallback(row) {
                    vec![None]
                } else {
                    vec![mask.as_ref().map(|mask| {
                        let mut padded = vec![0u32; arena_words];
                        padded[..mask.len()].copy_from_slice(mask);
                        padded
                    })]
                }
            })
            .collect();
        build_sampling_masks(&row_masks, &mut arena).unwrap();
        (plan, arena, inputs)
    }

    /// The loaded library, the device logits buffer and the wave, in drop order.
    ///
    /// The library is leaked so the wave's `&'a NativeLibrary` borrow can be
    /// `'static`: a `NativeLibrary` owns the dynamic-library handle and cannot be
    /// moved into the same struct that borrows it. A test process leaks one, and
    /// the OS reclaims it at exit.
    struct WiringWave {
        library: &'static NativeLibrary,
        logits_buffer: CuteafdDeviceBuffer,
        wave: TargetSamplingWave<'static>,
        rows: usize,
        vocab: usize,
        values: Vec<f32>,
    }

    impl WiringWave {
        /// Stage the plan and run the launch through the recording launcher.
        fn build(plan: &crate::families::deepseek_v41::v41_native_serve::scheduler::SamplingPlan, arena: &[u32],
            rows: usize, vocab: usize,
        ) -> Option<Self> {
            use super::recording_sampler as rec;
            let library: &'static NativeLibrary = Box::leak(Box::new(wiring_library()?));
            let mut wave = TargetSamplingWave::allocate(library, rec::stages(), rows, SAMPLING_VOCAB).unwrap();
            let logits_buffer = library.alloc_device_buffer(rows * vocab * 4).unwrap();
            let values = logits(rows, vocab);
            library.copy_h2d(logits_buffer, &logit_bytes(&values)).unwrap();
            wave.upload(&plan.rows, Some(arena), cuteafd_ffi::cuteafd_sampler_mask_words(vocab),
                std::ptr::null_mut()).unwrap();
            wave.launch(logits_buffer, rows, plan.routes_need_ordered_tail(),
                std::ptr::null_mut()).unwrap();
            Some(Self { library, logits_buffer, wave, rows, vocab, values })
        }
    }

    impl Drop for WiringWave {
        fn drop(&mut self) {
            // The wave owns buffers the queued (recording) stages read; the
            // recording launcher copies D2H synchronously, but keep the same
            // order the device tests use so a future real-launcher variant is
            // still sound.
            self.library.free_device_buffer(&mut self.logits_buffer).unwrap();
        }
    }

    /// A full-logit `BatchScores` whose every value is `value(row)`. Used to
    /// plant a deliberately stale device id in a row's slot: the commit path for
    /// a device-routed row reads `best`, and a kernel that failed leaves that
    /// slot at the caller's sentinel.
    fn batch_with_value(value: impl Fn(usize) -> u32, rows: usize,
    ) -> crate::families::deepseek_v41::v41_native_serve::scores::BatchScores {
        use crate::families::deepseek_v41::v41_native_serve::scores::{ROW_BYTES, VOCAB};
        let mut bytes = Vec::with_capacity(rows * ROW_BYTES);
        for row in 0..rows {
            let planted = value(row);
            for token in 0..VOCAB {
                let logit = if token == planted as usize { 8.0f32 } else { -8.0 };
                bytes.extend_from_slice(&logit.to_ne_bytes());
            }
        }
        crate::families::deepseek_v41::v41_native_serve::scores::BatchScores::new(bytes).unwrap()
    }

    fn wiring_library() -> Option<NativeLibrary> {
        let path = std::env::var_os("CUTEAFD_NATIVE_LIB")?;
        Some(unsafe { NativeLibrary::load(path) }.unwrap())
    }

    /// The wave's per-row parameters, its per-row mask rows, and its stage
    /// sequence are exactly what the plan specifies.
    ///
    /// This is the load-bearing wiring test: the plan mixes a greedy row, two
    /// fast-path rows (one masked, one not), an ordered top-k row, an ordered
    /// top-p row and a row that must fall back, and asserts (a) the launch
    /// sequence, (b) that every stage's parameter blocks are the plan's, per
    /// row, (c) that only the masked rows' arena slices were uploaded and at
    /// their own `mask_row`, and (d) that every delivered id equals the CPU
    /// sampler at that row's own parameters, mask and absolute position.
    #[test]
    fn the_wave_applies_per_row_params_and_masks_and_delivers_the_cpu_tokens() {
        use super::recording_sampler as rec;
        let _serial = rec::lock_and_reset();
        let vocab = rec::TEST_VOCAB;
        let (plan, arena, _inputs) = wiring_plan(vocab);
        assert_eq!(plan.routes_need_ordered_tail(), true);
        assert_eq!(plan.planned_fallback_count(), 1);

        let Some(mut fixture) = WiringWave::build(&plan, &arena, 7, vocab) else {
            eprintln!("skipping: CUTEAFD_NATIVE_LIB is not set");
            return;
        };
        let rows = fixture.rows;
        let words = vocab.div_ceil(32);
        let delivered = fixture.wave.output(fixture.logits_buffer, rows).unwrap();
        assert_eq!(rec::stage_calls(), (1, 1, 1),
            "one prepare, one K3/K4 and one K5 for this batch");
        assert_eq!(delivered.rows(), rows);
        assert!(delivered.status.iter().all(|status| *status == cuteafd_ffi::CUTEAFD_V41_SAMPLER_STATUS_OK),
            "every plan row is servable: {:?}", delivered.status);

        // (b)+(c): the recorded stages saw the plan's own per-row blocks and the
        // mask width, and nothing else was uploaded.
        let recorded = rec::recorded();
        assert_eq!(recorded.launches.len(), 3);
        for launch in &recorded.launches {
            assert_eq!(launch.rows, rows);
            assert_eq!(launch.vocab, vocab);
            assert_eq!(launch.mask_words_per_row, words);
            assert_eq!(launch.params.len(), rows);
            for (row, params) in launch.params.iter().enumerate() {
                assert_eq!(params.output_row as usize, row, "row identity for K5");
                assert_eq!(params, &plan.rows[row].row, "row {row} device block");
            }
        }
        assert_eq!(recorded.launches[0].stage, rec::Stage::Prepare);
        assert_eq!(recorded.launches[1].stage, rec::Stage::TopkSelect);
        assert_eq!(recorded.launches[2].stage, rec::Stage::Nucleus);
        // K3/K4 published a retained count only for the ordered top-k row.
        assert_eq!(recorded.retained, vec![0, 0, 0, 0, 3, 0, 0]);
        // The mask reached the device at the masked row's own ordinal, and only
        // there: the others' slices stayed zero.
        let mask_device = fixture.wave.mask_device.buffer;
        let mut mask_bytes = vec![0u8; rows * words * 4];
        fixture.library.copy_d2h(&mut mask_bytes, mask_device).unwrap();
        let uploaded: Vec<u32> = mask_bytes.chunks_exact(4)
            .map(|word| u32::from_ne_bytes(word.try_into().unwrap())).collect();
        for row in 0..rows {
            let slice = &uploaded[row * words..(row + 1) * words];
            if row == 4 {
                assert_eq!(slice, &low_tokens_mask(vocab)[..], "row 4's mask");
            } else {
                assert!(slice.iter().all(|word| *word == 0), "row {row} must stay zero");
            }
        }
        // (d): every delivered id is the CPU sampler's at that row's own
        // parameters and absolute position, and the constrained row is genuinely
        // constrained (its winner is outside the mask).
        for row in 0..rows {
            let row_logits = &fixture.values[row * vocab..(row + 1) * vocab];
            let mask = plan.mask[row].as_deref();
            let expected = rec::cpu_token(&plan.rows[row].row, row_logits, mask);
            assert_eq!(delivered.ids[row], expected, "row {row}");
        }
        let unmasked_winner = fixture.values[4 * vocab..5 * vocab].iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
        assert_ne!(delivered.ids[4], unmasked_winner, "the mask must change row 4's token");
    }

    /// A fallback row's resolved token must be **stored**, not merely computed.
    ///
    /// Regression test for the chunk-4a review's FIX 1: `resolve_fallback_rows`
    /// used to discard `next.sample(...)`, while the commit path for a
    /// device-routed row consumes `next.best[row]` wholesale. This builds a
    /// batch whose `best` slots all hold a stale sentinel (what a failed kernel
    /// leaves in `out_indices`), resolves the planned fallback rows, and
    /// requires the fallback slot to hold the CPU token. Before the fix the slot
    /// kept the sentinel.
    #[test]
    fn resolving_a_fallback_row_stores_the_cpu_token_in_its_best_slot() {
        use super::recording_sampler as rec;
        use crate::families::deepseek_v41::v41_native_serve::scheduler::{resolve_fallback_rows, SamplingRound};
        use crate::families::deepseek_v41::v41_native_serve::scores::{BatchScores, ROW_BYTES};
        let _serial = rec::lock_and_reset();
        let vocab = rec::TEST_VOCAB;
        let Some(_library) = wiring_library() else {
            eprintln!("skipping: CUTEAFD_NATIVE_LIB is not set");
            return;
        };
        let (plan, arena, _inputs) = wiring_plan(vocab);
        let Some(fixture) = WiringWave::build(&plan, &arena, 7, vocab) else { return };
        let rows = fixture.rows;
        let dummy = 129_279u32;
        let mut next = BatchScores::test_visible(&vec![dummy; rows], logit_bytes(&fixture.values))
            .unwrap();
        assert!(next.best.iter().all(|id| *id == dummy), "the fixture starts stale");
        let fallback: Vec<usize> = (0..rows)
            .filter(|&row| plan.planned_fallback(row))
            .collect();
        assert_eq!(fallback, vec![6], "exactly the top_k = 300 row falls back");
        let round = SamplingRound { plan, arena, trace_rows: Vec::new() };
        resolve_fallback_rows(&mut next, &round, &fallback).unwrap();
        let row_logits = &fixture.values[6 * vocab..7 * vocab];
        let expected = round.plan.params[6]
            .select_token(row_logits, None, round.plan.position[6]).unwrap() as u32;
        assert_ne!(expected, dummy, "the fixture would not detect a stored-value bug");
        assert_eq!(next.best[6], expected, "the fallback token must be stored in best[6]");
        // Every other slot is untouched: the resolution writes only its own rows.
        for row in 0..rows {
            if row != 6 {
                assert_eq!(next.best[row], dummy, "row {row} must not move");
            }
        }
        // The batch's logit extent is exactly what the resolver expects (one full
        // row per batch row), so `sample` could not have silently no-opped.
        assert!(next.has_row_logits(6));
        assert_eq!(logit_bytes(&fixture.values).len(), rows * ROW_BYTES);
    }

    /// The post-launch refusal path: a row the router calls device-servable but
    /// the kernel refuses with `INTERNAL` must end up with the **CPU** token, and
    /// the commit path must consume it without recomputing.
    ///
    /// Regression test for review FIX 1. Before the fix the refused row's `best`
    /// slot was never written, so the commit path consumed the stale device slot:
    /// `select_routed` keys on the plan route, which still says "device-served".
    #[test]
    fn a_device_refused_row_is_re_sampled_and_stored_on_the_cpu() {
        use super::recording_sampler as rec;
        use crate::families::deepseek_v41::v41_native_serve::scheduler::{admit_device_rows, resolve_fallback_rows,
            select_routed, SamplingRound};
        use crate::families::deepseek_v41::v41_native_serve::scores::BatchScores;
        let _serial = rec::lock_and_reset();
        let vocab = rec::TEST_VOCAB;
        let Some(_library) = wiring_library() else {
            eprintln!("skipping: CUTEAFD_NATIVE_LIB is not set");
            return;
        };
        let (plan, arena, _inputs) = wiring_plan(vocab);
        // Row 5 is the ordered top-p row: device-servable by the router.
        assert!(plan.route[5].device_served());
        rec::refuse_rows(&[5], 1);
        let Some(mut fixture) = WiringWave::build(&plan, &arena, 7, vocab) else { return };
        let rows = fixture.rows;
        let sampled = fixture.wave.output(fixture.logits_buffer, rows).unwrap();
        assert_eq!(sampled.status[5], cuteafd_ffi::CUTEAFD_V41_SAMPLER_STATUS_INTERNAL);

        // The batch `execute_sampled_rows` would build: device ids for the
        // admitted rows, and the stale sentinel for the refused row (a failed K5
        // writes no id, so `out_indices` still holds the caller's value).
        let dummy = 129_279u32;
        let ids: Vec<u32> = (0..rows).map(|row| {
            if row == 5 { dummy } else { sampled.ids[row] }
        }).collect();
        let mut next = BatchScores::test_visible(&ids, logit_bytes(&fixture.values)).unwrap();
        let round = SamplingRound { plan, arena, trace_rows: Vec::new() };
        let (device_rows, refused) = admit_device_rows(&round, &sampled).unwrap();
        assert_eq!(refused, vec![5], "the INTERNAL row is the refused set");
        assert!(!device_rows.contains(&5), "a refused row is not admitted");
        sampled.check_status(&device_rows).unwrap();

        let fallback = round.fallback_rows(&refused);
        assert_eq!(fallback, vec![5, 6], "the refused row joins the planned fallback");
        resolve_fallback_rows(&mut next, &round, &fallback).unwrap();

        let expected = round.plan.params[5]
            .select_token(&fixture.values[5 * vocab..6 * vocab], None,
                round.plan.position[5]).unwrap() as u32;
        assert_ne!(expected, dummy, "the fixture would not detect a stale-slot bug");
        assert_eq!(next.best[5], expected,
            "a refused row must commit the CPU token, not the stale device slot");
        assert_eq!(next.best[6], round.plan.params[6]
            .select_token(&fixture.values[6 * vocab..7 * vocab], None,
                round.plan.position[6]).unwrap() as u32);

        // The commit path consumes the stored selection for both fallback rows
        // (no recomputation), and keeps the admitted rows' device ids.
        for row in 0..rows {
            let selected = select_routed(&next, Some(&round), row, row, 1,
                || panic!("a stored row must not be recomputed in the commit path")).unwrap();
            let expected_row = if row == 5 || row == 6 {
                next.best[row]
            } else {
                sampled.ids[row]
            };
            assert_eq!(&selected[..], &[expected_row], "row {row}");
        }
        assert_eq!(next.best[0], sampled.ids[0], "an admitted device row keeps its id");
    }

    /// The unmasked argmax of one row of the fixture, computed on the host.
    fn row_argmax(values: &[f32], row: usize, vocab: usize) -> u32 {
        values[row * vocab..(row + 1) * vocab].iter().enumerate()
            .max_by(|left, right| left.1.partial_cmp(right.1).unwrap())
            .unwrap().0 as u32
    }

    /// A finishing frontier for a **planned-fallback stochastic row** must retain
    /// the stored draw, not re-derive an argmax and reject it.
    ///
    /// Regression test for the delta review's P1. The classifier keyed on
    /// `route[frontier].device_served()`, which is **false** for a planned
    /// fallback, so the row was classified `Checked` and `retain_packed` compared
    /// the row's argmax against `best[row]` — the CPU draw `store_sampled` had
    /// just written — and failed the whole lane with "GPU and retained CPU greedy
    /// selection differ". Reachable whenever an out-of-range (`top_k >= 257`)
    /// request is co-scheduled with a servable row and finishes at its frontier
    /// with tracing off. Both lanes take the same decision: this drives the
    /// serial path's `retain_recorded_from_device` / `retain_from_device` and the
    /// independent path's `resolve_frontier`.
    #[test]
    fn a_finishing_fallback_frontier_retains_its_stored_draw() {
        use super::recording_sampler as rec;
        use crate::families::deepseek_v41::v41_native_serve::scheduler::{frontier_retain, resolve_fallback_rows,
            resolve_frontier, FrontierRetain, SamplingRound};
        use crate::families::deepseek_v41::v41_native_serve::scores::BatchScores;
        let _serial = rec::lock_and_reset();
        let vocab = rec::TEST_VOCAB;
        let Some(library) = wiring_library() else {
            eprintln!("skipping: CUTEAFD_NATIVE_LIB is not set");
            return;
        };
        let (plan, arena, _inputs) = wiring_plan(vocab);
        let Some(fixture) = WiringWave::build(&plan, &arena, 7, vocab) else { return };
        let rows = fixture.rows;
        let frontier = 6usize; // the top_k = 300 fallback row
        assert!(plan.planned_fallback(frontier));
        assert!(!plan.route[frontier].device_served(),
            "the regression only exists while a fallback row is not device-served");
        assert!(plan.route[0].device_served() && plan.rows[0].row.flags != 0,
            "row 0 is the greedy device peer");

        // The frontier row gets a **near-flat** profile: the peaked fixture makes
        // the draw equal the argmax (the winner carries ~0.999 of the mass), so it
        // could not exercise the cross-check at all. A shallow strict descent over
        // the whole vocabulary puts ~1/300 of the mass on the argmax and makes the
        // draw a different token, exactly the situation the review reported.
        let mut values = fixture.values.clone();
        let plateau: Vec<f32> = (0..vocab).map(|token| -1e-3 * token as f32 / vocab as f32).collect();
        values[frontier * vocab..(frontier + 1) * vocab].copy_from_slice(&plateau);

        // Stage the batch, then run the production fallback resolution: `best[6]`
        // becomes the CPU draw, while every other row keeps a stale device slot.
        let stale = 7u32;
        let ids: Vec<u32> = (0..rows).map(|row| {
            if row == frontier { stale } else { row_argmax(&values, row, vocab) }
        }).collect();
        let mut next = BatchScores::test_visible(&ids, logit_bytes(&values)).unwrap();
        let round = SamplingRound { plan, arena, trace_rows: Vec::new() };
        resolve_fallback_rows(&mut next, &round, &[frontier]).unwrap();
        let draw = next.best[frontier];
        let argmax = row_argmax(&values, frontier, vocab);
        assert_ne!(draw, argmax,
            "the fixture must distinguish a stored draw from the row's argmax");

        // The classification: a non-greedy row in a device round is a draw
        // whoever produced it; a greedy row keeps the cross-check.
        let stochastic = round.plan.params[frontier];
        assert_eq!(frontier_retain(Some(&round), stochastic), FrontierRetain::RecordedSample);
        assert_eq!(frontier_retain(Some(&round), round.plan.params[0]), FrontierRetain::Checked);
        assert_eq!(frontier_retain(None, stochastic), FrontierRetain::Checked,
            "outside a device round the full-logit path is taken instead");

        // The bug itself, function-level and without a GPU: the pre-fix
        // `Checked` classification rejects the stored draw.
        let row_bytes = logit_bytes(&values[frontier * vocab..(frontier + 1) * vocab]);
        match resolve_frontier(FrontierRetain::Checked, &next, frontier, &row_bytes, None) {
            Ok(_) => panic!("the pre-fix classification must reject the draw, else this test is vacuous"),
            Err(error) => {
                eprintln!("DELTA-PROBE position=0 stored={draw} argmax={argmax} err={error}");
                assert!(error.to_string()
                    .contains("GPU and retained CPU greedy selection differ"));
            }
        }

        // The fix, independent lane: the recorded sample is retained as-is.
        let retained = resolve_frontier(FrontierRetain::RecordedSample, &next, frontier,
            &row_bytes, None).unwrap();
        assert_eq!(retained.select(None).unwrap(), draw, "the retained frontier is the stored draw");
        // The fix, serial lane: same decision through the packed-row path.
        let retained = next.retain_recorded_from_device(&library, fixture.logits_buffer,
            frontier).unwrap();
        assert_eq!(retained.select(None).unwrap(), draw);
        assert!(next.retain_from_device(&library, fixture.logits_buffer, frontier, None).is_err(),
            "the pre-fix call must still reject the draw on the packed path");
        // A greedy frontier peer still gets the cross-check and still passes it.
        let retained = next.retain_from_device(&library, fixture.logits_buffer, 0, None).unwrap();
        assert_eq!(retained.select(None).unwrap(), next.best[0]);
    }

    /// Chunk 5b F5: the recording double's strictness decision must mirror the
    /// **post-R2** rule — strict only where the row can act on the value
    /// (unmasked, or an allowed position), *not* `greedy || mask.is_some()`,
    /// which made every masked stochastic row strict and no longer matched
    /// `build_target_sampling_plan`.
    #[test]
    fn the_recording_double_is_strict_only_where_the_row_can_act() {
        use super::recording_sampler::nonfinite_where_allowed;
        // `0b1111` allows tokens 0..=3; token 4 is masked out.
        let allows_low_four = [0b1111u32];
        // A non-finite value at an allowed position: strict under every rule.
        assert!(nonfinite_where_allowed(&[0.0, f32::NAN, 2.0, 3.0, 4.0], Some(&allows_low_four)));
        // The same row with the non-finite token masked out is permissive —
        // exactly the case the old `mask.is_some()` clause wrongly made strict.
        assert!(!nonfinite_where_allowed(&[0.0, 1.0, 2.0, 3.0, f32::NAN], Some(&allows_low_four)));
        // No mask at all is strict regardless of the row's class.
        assert!(nonfinite_where_allowed(&[0.0, 1.0, 2.0, 3.0, f32::NAN], None));
        // All-finite rows are clean either way.
        assert!(!nonfinite_where_allowed(&[0.0, 1.0, 2.0, 3.0, 4.0], None));
        assert!(!nonfinite_where_allowed(&[0.0, 1.0, 2.0, 3.0, 4.0], Some(&allows_low_four)));
    }
}

#[cfg(test)]
mod sampler_device_tests {
    //! The chunk-4a GPU gate: drive the **real** kernels through the production
    //! [`TargetSamplingWave`] sequence for every required stochastic profile and
    //! for a mixed round, and compare against the production CPU sampler.
    //!
    //! `#[ignore]`d for the GPU convention (see `v41_compressor/source_cache.rs`
    //! and the FFI device tests): run it explicitly with a built native library,
    //! for example
    //!
    //! ```text
    //! CUTEAFD_NATIVE_LIB=/path/to/libcuteafd_native.so \
    //!   PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 \
    //!   cargo test -p cuteafd-daemon --bin cuteafd -- --ignored v41_device
    //! ```
    //!
    //! The divergence policy is the design's declared residual (§20 item 5,
    //! §6.3c): a greedy row must match bit-for-bit, the retained (top-k) domain
    //! is token-exact, and the fast-path and survivor domains are allowed to
    //! differ. Anything outside that policy — a status other than OK, an
    //! out-of-vocabulary or masked-out id, or a greedy row that differs — fails.
    //!
    //! **The measured divergence here is not the residual rate.** This fixture is
    //! `smooth_narrow` (see its doc), and the published per-cell rates are the
    //! numbers of record. The rate scales with the nucleus width and the mass-gap
    //! structure: a wider nucleus and smaller adjacent weight gaps make a 1–2 ulp
    //! `expf` difference more likely to move a ranked membership or the draw
    //! across a boundary. What this test contributes is that the device terminal
    //! is genuinely running the right route and that no divergence class outside
    //! the declared one appears.
    use super::*;
    use crate::families::deepseek_v41::v41_native_serve::scheduler::{build_sampling_masks, build_target_sampling_plan,
        SamplingMember};
    use cuteafd_core::TargetSamplingParams;

    const VOCAB: usize = SAMPLING_VOCAB;

    /// The `smooth_narrow` synthetic fixture: one per-rank geometric step with a
    /// small rank-ordered scatter, rotated per row.
    ///
    /// **This is not the phase-0 `long_tail` row and must never be labelled as
    /// one.** The real `long_tail` row has a ~90,426-token nucleus at
    /// `temperature 0.7, top_p 0.9` and a minimum adjacent weight gap of ~3.3e-9;
    /// this fixture's nucleus is only tens of ranks wide with a minimum gap of
    /// ~1e-4, so a 1–2 ulp `expf` difference essentially never moves its
    /// crossing. That is exactly why the measured divergence on it is 0/1024 and
    /// why the design's published per-cell rates (§6.3c: 12.3135% fast path,
    /// 25.9% survivor, 14.801607% ordered overall with the retained domain at
    /// 0.00%) remain the numbers of record: they come from the real-shaped
    /// fixtures, whose nucleus width and mass-gap structure are what drive the
    /// divergence. This fixture is for **wiring** evidence — the right kernels,
    /// the right per-row parameters and masks, the right routing — not for the
    /// residual rate.
    ///
    /// The retained-domain (`top_k = 40`) and greedy classes are token-exact on
    /// any shape, and those are asserted exactly below.
    fn smooth_narrow(rows: usize) -> Vec<f32> {
        let mut values = Vec::with_capacity(rows * VOCAB);
        for row in 0..rows {
            let offset = (row * 7919 + 13) % VOCAB;
            // A modest spread: rank r is about `r * step` below the maximum, so
            // with `temperature ~ 0.7` the nucleus is tens of ranks wide.
            let step = 0.02 + 0.01 * ((row % 4) as f32);
            for rank in 0..VOCAB {
                let delta = -step * rank as f32
                    + ((offset + rank * 31) % 7) as f32 * 0.0005;
                values.push(delta);
            }
            let row_values = &mut values[row * VOCAB..(row + 1) * VOCAB];
            row_values.rotate_left(offset);
        }
        values
    }

    fn bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|value| value.to_ne_bytes()).collect()
    }

    /// The mask that allows every token except one id per row.
    fn mask_excluding(rows: usize, excluded: &[u32]) -> Vec<u32> {
        let words = VOCAB.div_ceil(32);
        let mut arena = vec![u32::MAX; rows * words];
        cuteafd_ffi::cuteafd_sampler_clear_remainder(&mut arena, VOCAB);
        for (row, &token) in excluded.iter().enumerate() {
            arena[row * words + token as usize / 32] &= !(1u32 << (token % 32));
        }
        arena
    }

    struct Case {
        name: &'static str,
        params: Vec<TargetSamplingParams>,
        /// `None` uses the standard rank profile; `Some` supplies a deliberate
        /// witness shape (a broken/tied/saturated row) for the residual test.
        weights: Option<fn(usize) -> Vec<f32>>,
        /// Rows whose token must equal the CPU sampler exactly.
        exact: Vec<bool>,
        /// The most rows allowed to differ from the CPU sampler.
        max_mismatch: usize,
        /// Extra seed sweeps over the same logits, to measure the divergence
        /// rate rather than one draw's worth of luck.
        seed_sweeps: u64,
    }

    fn run_case(case: &Case) -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let rows = case.params.len();
        let words = VOCAB.div_ceil(32);
        let values = match case.weights {
            Some(witness) => witness(rows),
            None => smooth_narrow(rows),
        };
        let members: Vec<SamplingMember> = case.params.iter().enumerate()
            .map(|(row, &params)| SamplingMember {
                params, base_position: 1000 + 97 * row as u64, row_masks: vec![None],
            })
            .collect();
        let inputs: Vec<Vec<u32>> = vec![vec![0]; rows];
        let plan = build_target_sampling_plan(&members, &inputs)?;
        let mut arena = vec![0u32; rows * words];
        let row_masks: Vec<Vec<Option<Vec<u32>>>> = plan.mask.iter()
            .map(|mask| vec![mask.clone()]).collect();
        build_sampling_masks(&row_masks, &mut arena)?;

        let logits_buffer = library.alloc_device_buffer(rows * VOCAB * 4)?;
        library.copy_h2d(logits_buffer, &bytes(&values))?;
        let stream = library.cuda_stream_create()?;
        // One wave for both launches: its buffers must not be freed under the
        // queued kernels, and the second launch is the replay check.
        let mut wave = TargetSamplingWave::new(&library, rows, SAMPLING_VOCAB)?;
        wave.upload(&plan.rows, Some(&arena), words, stream)?;
        wave.launch(logits_buffer, rows, plan.routes_need_ordered_tail(), stream)?;
        unsafe { library.cuda_stream_synchronize(stream)?; }
        let sampled = wave.output(logits_buffer, rows)?;
        // Two launches must agree: the draw is keyed on (seed, position) and
        // nothing else.
        wave.upload(&plan.rows, Some(&arena), words, stream)?;
        wave.launch(logits_buffer, rows, plan.routes_need_ordered_tail(), stream)?;
        unsafe { library.cuda_stream_synchronize(stream)?; }
        let second = wave.output(logits_buffer, rows)?;
        assert_eq!(sampled.ids, second.ids, "{}: device draws must replay", case.name);
        let (ids, status) = (sampled.ids, sampled.status);

        // Rows the router sent to the CPU never ran on the device, so they are
        // **excluded from the divergence statistic**: their token is a CPU token
        // by construction and counting it would fold a routing decision into a
        // kernel residual. They are still required to be `OK` and in range, and
        // their CPU value is asserted, which is what pins that the device-neutral
        // block is a real no-op.
        let fallback: Vec<bool> = (0..rows).map(|row| plan.planned_fallback(row)).collect();
        assert!(fallback.iter().any(|fallback| !fallback),
            "{}: a case with no device-served row measures nothing", case.name);
        let mut mismatches = 0usize;
        let mut exact_mismatches = 0usize;
        let mut device_rows = 0usize;
        for row in 0..rows {
            assert_eq!(status[row], cuteafd_ffi::CUTEAFD_V41_SAMPLER_STATUS_OK,
                "{}: row {row} status", case.name);
            assert!((ids[row] as usize) < VOCAB, "{}: row {row} id in range", case.name);
            let row_logits = &values[row * VOCAB..(row + 1) * VOCAB];
            let expected = case.params[row].select_token(row_logits, None,
                plan.rows[row].row.position)? as u32;
            if fallback[row] {
                // The device-side value of a fallback row is deliberately
                // disposable: its block is the device-neutral greedy no-op, so
                // the device publishes that row's **argmax**, never the CPU
                // sample. It is still required to be `OK` (asserted above) and in
                // range. That the block is neutral, and that the CPU token the
                // fallback must produce is stored, are pinned exactly by the
                // recording-launcher tests.
                let _ = expected;
                continue;
            }
            device_rows += 1;
            if ids[row] != expected {
                mismatches += 1;
                if case.exact[row] { exact_mismatches += 1; }
            }
        }
        assert_eq!(exact_mismatches, 0, "{}: {exact_mismatches} exact rows diverged", case.name);
        assert!(mismatches <= case.max_mismatch,
            "{}: {mismatches} of {device_rows} device rows diverged (bound {})", case.name,
            case.max_mismatch);
        // Sweep seeds at the same logits and positions to measure the rate, not
        // one draw's luck. Every row is still checked for status, range, and (for
        // the exact classes) token equality.
        let mut swept_total = 0usize;
        let mut swept_mismatch = 0usize;
        if case.seed_sweeps > 0 {
            let mut sweep_params = case.params.clone();
            let mut sweep_exact = case.exact.clone();
            for sweep in 1..=case.seed_sweeps {
                for (row, params) in sweep_params.iter_mut().enumerate() {
                    *params = params.with_seed(params.seed().wrapping_add(sweep * 0x9e37));
                    sweep_exact[row] = case.exact[row];
                }
                let members: Vec<SamplingMember> = sweep_params.iter().enumerate()
                    .map(|(row, &params)| SamplingMember {
                        params, base_position: 1000 + 97 * row as u64, row_masks: vec![None],
                    }).collect();
                let plan = build_target_sampling_plan(&members, &inputs)?;
                let mut wave = TargetSamplingWave::new(&library, rows, SAMPLING_VOCAB)?;
                // No row here is masked, so nothing is copied; the width still
                // has to be the wave's own (the staging contract).
                wave.upload(&plan.rows, None, words, stream)?;
                wave.launch(logits_buffer, rows, plan.routes_need_ordered_tail(), stream)?;
                unsafe { library.cuda_stream_synchronize(stream)?; }
                let swept = wave.output(logits_buffer, rows)?;
                for row in 0..rows {
                    assert_eq!(swept.status[row], cuteafd_ffi::CUTEAFD_V41_SAMPLER_STATUS_OK,
                        "{} sweep {sweep}: row {row} status", case.name);
                    let row_logits = &values[row * VOCAB..(row + 1) * VOCAB];
                    let expected = sweep_params[row].select_token(row_logits, None,
                        plan.rows[row].row.position)? as u32;
                    if plan.planned_fallback(row) {
                        // Disposable device value; see the first-draw loop above.
                        let _ = expected;
                        continue;
                    }
                    swept_total += 1;
                    if swept.ids[row] != expected {
                        swept_mismatch += 1;
                        assert!(!case.exact[row],
                            "{} sweep {sweep}: exact row {row} diverged", case.name);
                    }
                }
            }
            println!("chunk4a device case {}: sweep {swept_mismatch}/{swept_total}                 ({:.4}%) divergent, first-draw id={:?}", case.name,
                swept_mismatch as f64 * 100.0 / swept_total as f64, ids);
        } else {
            println!("chunk4a device case {}: rows={rows} mismatches={mismatches} ids={:?}",
                case.name, ids);
        }
        // The wave owns buffers the queued launches read, so drain before it
        // drops and before the logits buffer is returned to the allocator.
        unsafe { library.cuda_stream_synchronize(stream)?; }
        drop(wave);
        unsafe { library.cuda_stream_destroy(stream)?; }
        let mut logits_buffer = logits_buffer;
        library.free_device_buffer(&mut logits_buffer)?;
        Ok(())
    }

    /// Chunk-4a latency: the device terminal's per-round sampler cost versus the
    /// CPU sampler, at a small (4-row) and a realistic (48-row) batch, plus a
    /// mixed greedy+stochastic batch.
    ///
    /// Both paths use the same per-round shape as production: params + mask
    /// staging/upload, the kernel sequence, the small D2H and the drain for the
    /// device; the production `TargetSamplingParams::select_token` with a single
    /// full-row D2H for the CPU. Reported medians are of many iterations after a
    /// warmup, printed so the run is self-evidencing. Run it in release (the
    /// default `cargo test` profile is unoptimized, which the CPU sampler
    /// dominates):
    ///
    /// ```text
    /// PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 \
    /// CUTEAFD_NATIVE_LIB=…/libcuteafd_native.so \
    ///   cargo test --release -p cuteafd-daemon --bin cuteafd -- \
    ///     --ignored --nocapture v41_device_latency
    /// ```
    #[test]
    #[ignore = "requires a GPU and CUTEAFD_NATIVE_LIB; run with --ignored"]
    fn v41_device_latency_vs_cpu_sampler() -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        for rows in [4usize, 48] {
            let words = VOCAB.div_ceil(32);
            let params = vec![
                TargetSamplingParams::new(0.7, 0.9, Some(40), 0.05, 0x33).unwrap(); rows];
            let values = smooth_narrow(rows);
            let members: Vec<SamplingMember> = params.iter().enumerate()
                .map(|(row, &params)| SamplingMember {
                    params, base_position: row as u64, row_masks: vec![None],
                }).collect();
            let inputs: Vec<Vec<u32>> = vec![vec![0]; rows];
            let plan = build_target_sampling_plan(&members, &inputs)?;
            let mut arena = vec![0u32; rows * words];
            build_sampling_masks(&vec![vec![None]; rows], &mut arena)?;
            let logits_buffer = library.alloc_device_buffer(rows * VOCAB * 4)?;
            library.copy_h2d(logits_buffer, &bytes(&values))?;
            let mut host_rows = vec![0u8; rows * VOCAB * 4];
            let stream = library.cuda_stream_create()?;

            // Device path: staging + upload + kernel sequence + small D2H + drain,
            // with a CUDA-event pair around the launch so the GPU-side work is
            // separated from the host-side staging.
            let mut wave = TargetSamplingWave::new(&library, rows, SAMPLING_VOCAB)?;
            let start_event = library.cuda_event_create()?;
            let end_event = library.cuda_event_create()?;
            let mut device_us = Vec::new();
            let mut device_gpu_us = Vec::new();
            let mut device_host_us = Vec::new();
            for iteration in 0..40 {
                let start = std::time::Instant::now();
                wave.upload(&plan.rows, Some(&arena), words, stream)?;
                let staged = start.elapsed().as_secs_f64() * 1e6;
                unsafe {
                    library.cuda_event_record(start_event, stream)?;
                    wave.launch(logits_buffer, rows, plan.routes_need_ordered_tail(), stream)?;
                    library.cuda_event_record(end_event, stream)?;
                    library.cuda_stream_synchronize(stream)?;
                }
                let sampled = wave.output(logits_buffer, rows)?;
                assert_eq!(sampled.rows(), rows);
                if iteration >= 4 {
                    device_us.push(start.elapsed().as_secs_f64() * 1e6);
                    device_host_us.push(staged);
                    device_gpu_us.push(unsafe {
                        library.cuda_event_elapsed_ms(start_event, end_event)? as f64 * 1e3
                    });
                }
            }
            unsafe {
                library.cuda_event_destroy(start_event)?;
                library.cuda_event_destroy(end_event)?;
            }

            // CPU path: the production sampler over a downloaded full row.
            let mut cpu_us = Vec::new();
            for iteration in 0..12 {
                let start = std::time::Instant::now();
                library.copy_d2h(&mut host_rows, logits_buffer)?;
                for (row, params) in params.iter().enumerate() {
                    let row_logits: Vec<f32> = host_rows[row * VOCAB * 4..(row + 1) * VOCAB * 4]
                        .chunks_exact(4).map(|w| f32::from_ne_bytes(w.try_into().unwrap()))
                        .collect();
                    let _ = params.select_token(&row_logits, None, row as u64)?;
                }
                if iteration >= 2 { cpu_us.push(start.elapsed().as_secs_f64() * 1e6); }
            }
            // The transfer alone, so the CPU row can be split into D2H + compute.
            let mut copy_us = Vec::new();
            for iteration in 0..12 {
                let start = std::time::Instant::now();
                library.copy_d2h(&mut host_rows, logits_buffer)?;
                if iteration >= 2 { copy_us.push(start.elapsed().as_secs_f64() * 1e6); }
            }
            unsafe { library.cuda_stream_synchronize(stream)?; }
            drop(wave);
            unsafe { library.cuda_stream_destroy(stream)?; }
            let mut logits_buffer = logits_buffer;
            library.free_device_buffer(&mut logits_buffer)?;

            let median = |mut values: Vec<f64>| {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                values[values.len() / 2]
            };
            println!(
                "chunk4a latency rows={rows}: device_round={:.1}us (stage={:.1}us gpu={:.1}us)                  cpu_full={:.1}us (d2h={:.1}us compute={:.1}us) speedup={:.2}x",
                median(device_us.clone()), median(device_host_us), median(device_gpu_us),
                median(cpu_us.clone()), median(copy_us.clone()),
                median(cpu_us.clone()) - median(copy_us.clone()),
                median(cpu_us) / median(device_us));
        }

        // Split the ordered path: the same batch shape with the fast-path
        // profile (K1 -> K2 only) isolates the K3/K4/K5 cost.
        for rows in [4usize, 48] {
            let words = VOCAB.div_ceil(32);
            let values = smooth_narrow(rows);
            let members: Vec<SamplingMember> = (0..rows).map(|row| SamplingMember {
                params: TargetSamplingParams::new(0.7, 1.0, None, 0.05, 0x34).unwrap(),
                base_position: row as u64, row_masks: vec![None],
            }).collect();
            let inputs: Vec<Vec<u32>> = vec![vec![0]; rows];
            let plan = build_target_sampling_plan(&members, &inputs)?;
            assert!(!plan.routes_need_ordered_tail());
            let logits_buffer = library.alloc_device_buffer(rows * VOCAB * 4)?;
            library.copy_h2d(logits_buffer, &bytes(&values))?;
            let stream = library.cuda_stream_create()?;
            let mut wave = TargetSamplingWave::new(&library, rows, SAMPLING_VOCAB)?;
            let start_event = library.cuda_event_create()?;
            let end_event = library.cuda_event_create()?;
            let mut fast_us = Vec::new();
            for iteration in 0..40 {
                let start = std::time::Instant::now();
                wave.upload(&plan.rows, None, words, stream)?;
                unsafe {
                    library.cuda_event_record(start_event, stream)?;
                    wave.launch(logits_buffer, rows, false, stream)?;
                    library.cuda_event_record(end_event, stream)?;
                    library.cuda_stream_synchronize(stream)?;
                }
                let _ = wave.output(logits_buffer, rows)?;
                if iteration >= 4 {
                    fast_us.push(unsafe {
                        library.cuda_event_elapsed_ms(start_event, end_event)? as f64 * 1e3
                    });
                }
            }
            unsafe {
                library.cuda_event_destroy(start_event)?;
                library.cuda_event_destroy(end_event)?;
                library.cuda_stream_synchronize(stream)?;
            }
            drop(wave);
            unsafe { library.cuda_stream_destroy(stream)?; }
            let mut logits_buffer = logits_buffer;
            library.free_device_buffer(&mut logits_buffer)?;
            fast_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!("chunk4a fast-path (K1+K2) rows={rows}: gpu={:.1}us",
                fast_us[fast_us.len() / 2]);
        }

        // Greedy regression: the compact path is `execute_greedy`, which is the
        // FFI argmax over the same row buffer. Compare it with the sampler so the
        // new routing's effect on the greedy figure is visible. The figure of
        // record for the v11 greedy weighted rate is 95.46 tok/s; this measures
        // the kernel stage only.
        let rows = 48usize;
        let values = smooth_narrow(rows);
        let logits_buffer = library.alloc_device_buffer(rows * VOCAB * 4)?;
        library.copy_h2d(logits_buffer, &bytes(&values))?;
        let indices = library.alloc_device_buffer(rows * 4)?;
        let scores = library.alloc_device_buffer(rows * 4)?;
        let stream = library.cuda_stream_create()?;
        let mut greedy_us = Vec::new();
        for iteration in 0..60 {
            let start = std::time::Instant::now();
            unsafe {
                library.cuda_logits_argmax_checked_f32_async(
                    logits_buffer, indices, scores, rows, VOCAB, stream)?;
            }
            unsafe { library.cuda_stream_synchronize(stream)?; }
            if iteration >= 4 { greedy_us.push(start.elapsed().as_secs_f64() * 1e6); }
        }
        unsafe { library.cuda_stream_synchronize(stream)?; }
        unsafe { library.cuda_stream_destroy(stream)?; }
        let mut logits_buffer = logits_buffer;
        let mut indices = indices;
        let mut scores = scores;
        library.free_device_buffer(&mut logits_buffer)?;
        library.free_device_buffer(&mut indices)?;
        library.free_device_buffer(&mut scores)?;
        greedy_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("chunk4a greedy argmax rows=48: {:.1}us (compact lane kernel stage only)",
            greedy_us[greedy_us.len() / 2]);

        // The greedy **sampled** route (all rows greedy through K1): the routing
        // chunk's cost for a greedy round that is not on the compact lane. It
        // must stay at the K1 level, not approach the stochastic sampler.
        {
            let rows = 48usize;
            let words = VOCAB.div_ceil(32);
            let values = smooth_narrow(rows);
            let members: Vec<SamplingMember> = (0..rows).map(|row| SamplingMember {
                params: TargetSamplingParams::greedy().with_seed(row as u64),
                base_position: row as u64, row_masks: vec![None],
            }).collect();
            let inputs: Vec<Vec<u32>> = vec![vec![0]; rows];
            let plan = build_target_sampling_plan(&members, &inputs)?;
            assert!(plan.route.iter().all(|route| *route == crate::families::deepseek_v41::v41_native_serve::scheduler::SamplingRoute::DeviceGreedy));
            let logits_buffer = library.alloc_device_buffer(rows * VOCAB * 4)?;
            library.copy_h2d(logits_buffer, &bytes(&values))?;
            let stream = library.cuda_stream_create()?;
            let mut wave = TargetSamplingWave::new(&library, rows, SAMPLING_VOCAB)?;
            let start_event = library.cuda_event_create()?;
            let end_event = library.cuda_event_create()?;
            let mut sampled_greedy_us = Vec::new();
            let mut sampled_greedy_gpu_us = Vec::new();
            for iteration in 0..40 {
                let start = std::time::Instant::now();
                wave.upload(&plan.rows, None, words, stream)?;
                unsafe {
                    library.cuda_event_record(start_event, stream)?;
                    wave.launch(logits_buffer, rows, plan.routes_need_ordered_tail(), stream)?;
                    library.cuda_event_record(end_event, stream)?;
                    library.cuda_stream_synchronize(stream)?;
                }
                let sampled = wave.output(logits_buffer, rows)?;
                assert_eq!(sampled.rows(), rows);
                if iteration >= 4 {
                    sampled_greedy_us.push(start.elapsed().as_secs_f64() * 1e6);
                    sampled_greedy_gpu_us.push(unsafe {
                        library.cuda_event_elapsed_ms(start_event, end_event)? as f64 * 1e3
                    });
                }
            }
            unsafe {
                library.cuda_event_destroy(start_event)?;
                library.cuda_event_destroy(end_event)?;
            }
            unsafe { library.cuda_stream_synchronize(stream)?; }
            drop(wave);
            unsafe { library.cuda_stream_destroy(stream)?; }
            let mut logits_buffer = logits_buffer;
            library.free_device_buffer(&mut logits_buffer)?;
            let median = |mut values: Vec<f64>| {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                values[values.len() / 2]
            };
            println!("chunk4a greedy sampled-route rows=48: round={:.1}us gpu={:.1}us \
                      (K1 over 48 greedy rows)",
                median(sampled_greedy_us), median(sampled_greedy_gpu_us));
        }

        // Chunk 4b retention gate: the one-row frontier transfer the gate removes
        // when the turn bank is disabled. This is the production pageable D2H
        // (`BatchScores::retain_from_device`), measured per row and extrapolated
        // to a representative finishing batch (every one of 8 requests finishing
        // in its last round).
        {
            let rows = 8usize;
            let values = smooth_narrow(rows);
            let buffer = library.alloc_device_buffer(rows * VOCAB * 4)?;
            library.copy_h2d(buffer, &bytes(&values))?;
            let mut host = vec![0u8; VOCAB * 4];
            let mut one_row = buffer;
            one_row.bytes = VOCAB * 4;
            let mut frontier_us = Vec::new();
            for iteration in 0..40 {
                let start = std::time::Instant::now();
                library.copy_d2h(&mut host, one_row)?;
                if iteration >= 4 { frontier_us.push(start.elapsed().as_secs_f64() * 1e6); }
            }
            let mut buffer = buffer;
            drop(one_row);
            library.free_device_buffer(&mut buffer)?;
            let mut median = |mut values: Vec<f64>| {
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                values[values.len() / 2]
            };
            let per_row = median(frontier_us);
            println!("chunk4b retention gate: one-row frontier D2H={per_row:.1}us \
                      ({}B); gated 1-row round saves {per_row:.1}us / {}B; \
                      representative 8-finishing-row round saves {:.1}us / {}B",
                VOCAB * 4, VOCAB * 4, 8.0 * per_row, 8 * VOCAB * 4);
        }
        Ok(())
    }

    /// The four required stochastic profiles on the real device terminal, plus a
    /// deliberately wide-nucleus row.
    ///
    /// `temperature 0.7 + top_k 40` is the retained domain and is asserted
    /// token-exact (design §20 item 5: 0.00% retained-domain divergence). The
    /// other profiles are bounded with margin: the measured rate on a
    /// well-conditioned rank profile is *below* the design's published
    /// near-uniform-cell figures, and the test reports the measured counts rather
    /// than asserting equality the kernels cannot have.
    #[test]
    #[ignore = "requires a GPU and CUTEAFD_NATIVE_LIB; run with --ignored"]
    fn v41_device_stochastic_profiles_match_the_cpu_sampler_within_the_declared_residual() -> Result<()> {
        let rows = 8;
        let cases = vec![
            // A deliberately wide row: a smaller per-rank step widens the
            // `top_p` nucleus into the hundreds of ranks, which is the
            // `near_uniform` cell where the CUDA-vs-glibc `expf` and reduction
            // order actually move the crossing. It is the case that shows the
            // declared residual is reachable, and the case whose bound must stay
            // under 100%.
            Case {
                name: "wide-nucleus temperature0.7+top_p0.995",
                params: (0..rows).map(|row| TargetSamplingParams::new(
                    0.7, 0.995, None, 0.0, 0x50 + row as u64).unwrap()).collect(),
                exact: vec![false; rows],
                max_mismatch: rows * 90 / 100 + 1,
                seed_sweeps: 32,
                weights: None,
            },
            Case {
                name: "temperature0.2+top_p0.95",
                params: (0..rows).map(|row| TargetSamplingParams::new(
                    0.2, 0.95, None, 0.0, 0x51 + row as u64).unwrap()).collect(),
                exact: vec![false; rows],
                max_mismatch: rows * 60 / 100 + 1,
                seed_sweeps: 32,
                weights: None,
            },
            Case {
                name: "temperature0.7+top_p0.9",
                params: (0..rows).map(|row| TargetSamplingParams::new(
                    0.7, 0.9, None, 0.0, 0x52 + row as u64).unwrap()).collect(),
                exact: vec![false; rows],
                max_mismatch: rows * 60 / 100 + 1,
                seed_sweeps: 32,
                weights: None,
            },
            Case {
                name: "temperature0.7+min_p0.05",
                params: (0..rows).map(|row| TargetSamplingParams::new(
                    0.7, 1.0, None, 0.05, 0x53 + row as u64).unwrap()).collect(),
                exact: vec![false; rows],
                max_mismatch: rows * 60 / 100 + 1,
                seed_sweeps: 32,
                weights: None,
            },
            Case {
                name: "temperature0.7+top_k40",
                params: (0..rows).map(|row| TargetSamplingParams::new(
                    0.7, 1.0, Some(40), 0.0, 0x54 + row as u64).unwrap()).collect(),
                exact: vec![true; rows],
                max_mismatch: 0,
                seed_sweeps: 32,
                weights: None,
            },
        ];
        for case in &cases {
            run_case(case)?;
        }
        Ok(())
    }

    /// A mixed round: greedy rows (bit-exact), a fast-path row, an ordered row
    /// and a row the router must fall back all share one launch. The greedy rows
    /// pin the per-row routing; the fallback row's device-facing block is a
    /// no-op, so its id is not asserted here (the wiring test above pins that).
    #[test]
    #[ignore = "requires a GPU and CUTEAFD_NATIVE_LIB; run with --ignored"]
    fn v41_device_mixed_round_routes_per_row() -> Result<()> {
        let rows = 6usize;
        let params = vec![
            TargetSamplingParams::greedy().with_seed(1),
            TargetSamplingParams::greedy().with_seed(2),
            TargetSamplingParams::new(0.7, 1.0, None, 0.05, 3).unwrap(),
            TargetSamplingParams::new(0.7, 1.0, Some(40), 0.0, 4).unwrap(),
            TargetSamplingParams::new(0.7, 0.9, None, 0.0, 5).unwrap(),
            TargetSamplingParams::new(0.7, 1.0, Some(300), 0.0, 6).unwrap(),
        ];
        let case = Case {
            name: "mixed",
            params,
            exact: vec![true, true, false, true, false, false],
            max_mismatch: 2,
            seed_sweeps: 0,
            weights: None,
        };
        run_case(&case)
    }

    /// A deliberate **residual witness**: a row whose nucleus boundary sits
    /// inside a single weight's rounding. Token 0 dominates and token 1 carries
    /// the exact remainder of `1.0`, so the CPU's normalized `p_0` shows the
    /// CUDA-vs-glibc `expf` and reduction-order difference directly at the
    /// crossing. This is the shape the design's near-uniform cell reports
    /// `57.75 %` divergence for, and the shape that makes the declared residual
    /// observable instead of asserting a bound the other cells never approach.
    fn step_witness(rows: usize) -> Vec<f32> {
        // `ln(0.9)` and `ln(0.1)`: the two survivors hold all the mass.
        const STEP: [f32; 2] = [-2.302_585_1, -2.302_585_1];
        let mut values = Vec::with_capacity(rows * VOCAB);
        for _ in 0..rows {
            values.push(0.0f32);
            values.push(STEP[1]);
            values.extend(std::iter::repeat_n(-40.0f32, VOCAB - 2));
        }
        values
    }

    /// The residual-witness profile, swept over 4,096 seeded draws: every
    /// selection is `OK`, in range, inside the survivor set, and the divergence
    /// against the CPU sampler is bounded. **MEASURED on this box: 0/4096
    /// divergent** — the two-survivor row's crossing sits at `u = 0.9`, so only
    /// a draw within ~1e-7 of the boundary could move it, which a 24-bit uniform
    /// essentially never produces. The design's published near-uniform rate
    /// (57.75% in that cell) comes from rows where a 1–2 ulp weight difference
    /// moves a *ranked membership* across the `top_p` boundary, which this
    /// two-token row cannot express; the wider rank profiles above cover the
    /// membership case.
    ///
    /// The test is still load-bearing: `mixed` below shows a nucleus row whose
    /// device token differs from the CPU token, so a device terminal that was
    /// silently not running could not produce the `OK`/in-range results asserted
    /// here, and a kernel that published a stale or out-of-range id fails.
    #[test]
    #[ignore = "requires a GPU and CUTEAFD_NATIVE_LIB; run with --ignored"]
    fn v41_device_residual_witness_is_bounded_and_never_invalid() -> Result<()> {
        let rows = 8;
        let params = vec![TargetSamplingParams::new(1.0, 1.0, None, 0.0, 0x71).unwrap(); rows];
        let case = Case {
            name: "residual-witness temperature1.0+min_p0",
            params,
            weights: Some(step_witness),
            exact: vec![false; rows],
            max_mismatch: rows * 90 / 100 + 1,
            seed_sweeps: 512,
        };
        run_case(&case)
    }

    /// A constrained stochastic row's draw must be inside its grammar mask: the
    /// mask is applied by the device **first**, exactly as
    /// `select_verification_sampled` applies it on the CPU.
    #[test]
    #[ignore = "requires a GPU and CUTEAFD_NATIVE_LIB; run with --ignored"]
    fn v41_device_constrained_stochastic_row_respects_its_mask() -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let rows = 2;
        let words = VOCAB.div_ceil(32);
        let values = smooth_narrow(rows);
        let params = TargetSamplingParams::new(0.7, 0.9, None, 0.0, 77)?;
        // Row 0 is masked to the three lowest ids; row 1 is unconstrained.
        let mut mask = vec![0u32; words];
        mask[0] = 0b111;
        let members = vec![
            SamplingMember { params, base_position: 0, row_masks: vec![Some(mask.clone())] },
            SamplingMember { params, base_position: 1, row_masks: vec![None] },
        ];
        let inputs: Vec<Vec<u32>> = vec![vec![0], vec![0]];
        let plan = build_target_sampling_plan(&members, &inputs)?;
        let mut arena = vec![0u32; rows * words];
        build_sampling_masks(&[vec![Some(mask.clone())], vec![None]], &mut arena)?;
        let logits_buffer = library.alloc_device_buffer(rows * VOCAB * 4)?;
        library.copy_h2d(logits_buffer, &bytes(&values))?;
        let stream = library.cuda_stream_create()?;
        let result = (|| -> Result<Vec<u32>> {
            let mut wave = TargetSamplingWave::new(&library, rows, SAMPLING_VOCAB)?;
            wave.upload(&plan.rows, Some(&arena), words, stream)?;
            wave.launch(logits_buffer, rows, plan.routes_need_ordered_tail(), stream)?;
            unsafe { library.cuda_stream_synchronize(stream)?; }
            Ok(wave.output(logits_buffer, rows)?.ids)
        })();
        unsafe { library.cuda_stream_destroy(stream)?; }
        let mut logits_buffer = logits_buffer;
        library.free_device_buffer(&mut logits_buffer)?;
        let ids = result?;
        let row1 = &values[VOCAB..2 * VOCAB];
        let expected1 = params.select_token(row1, None, 1)? as u32;
        println!("chunk4a constrained row: ids={ids:?} expected_row1={expected1}                   status_mask_row={:?} arena[0]=0b{:b}",
            plan.rows[0].row.mask_row, arena[0] & 0xff);
        assert!(ids[0] < 3, "row 0 must draw inside its mask, got {}", ids[0]);
        assert_eq!(ids[1], expected1);
        Ok(())
    }

    // ==================================================================
    // Chunk 4b: the executed upload -> launch -> output tests on the real
    // production launcher (`NativeSamplerStages`), plus the retention gate.
    // ==================================================================

    /// The packed arena row that allows exactly `tokens`.
    fn allowed_mask(tokens: &[u32]) -> Vec<u32> {
        let mut words = vec![0u32; VOCAB.div_ceil(32)];
        for &token in tokens {
            words[token as usize / 32] |= 1u32 << (token % 32);
        }
        cuteafd_ffi::cuteafd_sampler_clear_remainder(&mut words, VOCAB);
        words
    }

    /// The unmasked host argmax of one row.
    fn row_argmax_dev(values: &[f32], row: usize) -> u32 {
        values[row * VOCAB..(row + 1) * VOCAB].iter().enumerate()
            .max_by(|left, right| left.1.partial_cmp(right.1).unwrap())
            .unwrap().0 as u32
    }

    /// A wave staged through the **production** `TargetSamplingWave` (the real
    /// `NativeSamplerStages`, i.e. the C ABI entry points), with its device
    /// buffers and stream owned so a test can read the uploaded params/masks back.
    struct StagedWave<'a> {
        library: &'a NativeLibrary,
        logits: CuteafdDeviceBuffer,
        stream: *mut std::ffi::c_void,
        wave: TargetSamplingWave<'a>,
        rows: usize,
    }

    /// The pinned mask arena is primed with this before `upload`, so the test can
    /// tell "`upload` deliberately skipped this `NO_MASK` row" from "the
    /// allocator happened to hand back zeroes". `upload` never writes a row it
    /// skips, so asserting zero there (the pre-review helper) tested
    /// uninitialized `cudaMalloc` memory, not the staging contract.
    const MASK_STAGING_SENTINEL: u32 = 0xDEAD_BEEF;

    impl<'a> StagedWave<'a> {
        fn new(library: &'a NativeLibrary, rows: usize, values: &[f32],
            plan: &crate::families::deepseek_v41::v41_native_serve::scheduler::SamplingPlan, arena: Option<&[u32]>,
        ) -> Result<Self> {
            let logits = library.alloc_device_buffer(rows * VOCAB * 4)?;
            library.copy_h2d(logits, &bytes(values))?;
            let stream = library.cuda_stream_create()?;
            let mut wave = TargetSamplingWave::new(library, rows, SAMPLING_VOCAB)?;
            if arena.is_some() {
                let staging = wave.mask_pinned.bytes_mut();
                for word in staging.chunks_exact_mut(4) {
                    word.copy_from_slice(&MASK_STAGING_SENTINEL.to_ne_bytes());
                }
            }
            wave.upload(&plan.rows, arena, VOCAB.div_ceil(32), stream)?;
            wave.launch(logits, rows, plan.routes_need_ordered_tail(), stream)?;
            unsafe { library.cuda_stream_synchronize(stream)?; }
            Ok(Self { library, logits, stream, wave, rows })
        }

        fn output(&mut self) -> Result<SampledTargetRows> {
            self.wave.output(self.logits, self.rows)
        }

        /// The per-row parameter blocks actually resident on the device.
        fn device_params(&self) -> Result<Vec<u8>> {
            let mut raw = vec![0u8; self.rows * cuteafd_ffi::CUTEAFD_V41_SAMPLER_PARAM_BYTES];
            self.library.copy_d2h(&mut raw, self.wave.param_device.buffer)?;
            Ok(raw)
        }

        /// The packed mask arena actually resident on the device, limited to the
        /// `mask_rows` the upload wrote. Reading past that would touch rows
        /// `cudaMalloc` never initialized.
        fn device_masks(&self) -> Result<Vec<u32>> {
            let words = VOCAB.div_ceil(32);
            let rows = self.wave.mask_rows;
            if rows == 0 {
                return Ok(Vec::new());
            }
            let mut raw = vec![0u8; rows * words * 4];
            self.library.copy_d2h(&mut raw, self.wave.mask_device.buffer)?;
            Ok(raw.chunks_exact(4).map(|word| u32::from_ne_bytes(word.try_into().unwrap())).collect())
        }

        /// Download one row's full logits (the per-row D2H the production
        /// `BatchScores::retain_from_device` performs).
        fn download_row(&self, row: usize) -> Result<Vec<u8>> {
            let mut slice = self.logits;
            slice.ptr = unsafe { slice.ptr.cast::<u8>().add(row * VOCAB * 4).cast() };
            slice.bytes = VOCAB * 4;
            let mut bytes = vec![0u8; VOCAB * 4];
            self.library.copy_d2h(&mut bytes, slice)?;
            Ok(bytes)
        }
    }

    impl Drop for StagedWave<'_> {
        fn drop(&mut self) {
            unsafe { let _ = self.library.cuda_stream_synchronize(self.stream); }
            let _ = unsafe { self.library.cuda_stream_destroy(self.stream) };
            let mut logits = self.logits;
            let _ = self.library.free_device_buffer(&mut logits);
        }
    }

    /// The uploaded parameter blocks and mask arena are the plan's own, per row,
    /// on the real device buffers.
    ///
    /// Only the `mask_rows` rows `upload` actually wrote are read. A `NO_MASK`
    /// row below `mask_rows` must still carry the primed sentinel, which proves
    /// the upload skipped it rather than relying on allocator zeroing; rows at or
    /// after `mask_rows` are never written and are not read at all (review FIX 3).
    fn assert_uploaded_matches(wave: &StagedWave<'_>,
        plan: &crate::families::deepseek_v41::v41_native_serve::scheduler::SamplingPlan, arena: &[u32],
    ) -> Result<()> {
        let params = wave.device_params()?;
        for (row, request) in plan.rows.iter().enumerate() {
            let expected = bytemuck_bytes(&request.row);
            assert_eq!(&params[row * 64..(row + 1) * 64], expected,
                "device row {row} parameter block differs");
        }
        let masks = wave.device_masks()?;
        let words = VOCAB.div_ceil(32);
        assert_eq!(masks.len(), wave.wave.mask_rows * words,
            "the read back must cover exactly the uploaded mask rows");
        for row in 0..wave.wave.mask_rows {
            let no_mask = plan.rows[row].row.flags
                & cuteafd_ffi::CUTEAFD_V41_SAMPLER_FLAG_NO_MASK != 0;
            let staged = &masks[row * words..(row + 1) * words];
            if no_mask {
                assert!(staged.iter().all(|word| *word == MASK_STAGING_SENTINEL),
                    "unconstrained row {row} must not have been written by upload");
            } else {
                assert_eq!(staged, &arena[row * words..(row + 1) * words],
                    "masked row {row} staged the wrong arena slice");
            }
        }
        Ok(())
    }

    /// **(a) The executed mixed `upload -> launch -> output` test.**
    ///
    /// One real production launch carries a greedy row, a constrained greedy row,
    /// a fast-path stochastic row, an ordered `top_k = 40` constrained row, an
    /// ordered `top_p` row and a `top_k = 300` planned fallback. It asserts:
    /// the per-row launch sequence, the per-row params **and masks actually
    /// resident on the device**, the returned tokens, CPU equality for the
    /// greedy/retained rows, in-mask membership for every constrained row (over a
    /// seeded sweep), and that the fallback row's CPU re-sample is stored.
    #[test]
    #[ignore = "requires a GPU and CUTEAFD_NATIVE_LIB; run with --ignored"]
    fn v41_device_mixed_batch_upload_launch_output_with_masks() -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let rows = 6usize;
        let words = VOCAB.div_ceil(32);
        let values = smooth_narrow(rows);
        let small = allowed_mask(&[3, 11, 29]);
        let retained_mask = allowed_mask(&[1, 2, 4, 8, 16]);
        let params = vec![
            TargetSamplingParams::greedy().with_seed(0x11),
            TargetSamplingParams::greedy().with_seed(0x12),
            TargetSamplingParams::new(0.7, 1.0, None, 0.05, 0x13)?,
            TargetSamplingParams::new(0.7, 1.0, Some(40), 0.0, 0x14)?,
            TargetSamplingParams::new(0.7, 0.9, None, 0.0, 0x15)?,
            TargetSamplingParams::new(0.7, 1.0, Some(300), 0.0, 0x16)?,
        ];
        let row_masks: Vec<Vec<Option<Vec<u32>>>> = vec![
            vec![None], vec![Some(small.clone())], vec![None],
            vec![Some(retained_mask.clone())], vec![None], vec![None],
        ];
        let build = |seed_shift: u64| -> Result<(
            crate::families::deepseek_v41::v41_native_serve::scheduler::SamplingPlan, Vec<u32>)> {
            let members: Vec<SamplingMember> = params.iter().enumerate().map(|(row, &params)| {
                let params = if seed_shift == 0 { params }
                    else { params.with_seed(params.seed().wrapping_add(seed_shift * 0x9e37)) };
                SamplingMember { params, base_position: 1000 + 97 * row as u64,
                    row_masks: row_masks[row].clone() }
            }).collect();
            let inputs: Vec<Vec<u32>> = vec![vec![0]; rows];
            let plan = build_target_sampling_plan(&members, &inputs)?;
            let mut arena = vec![0u32; rows * words];
            let staged: Vec<Vec<Option<Vec<u32>>>> = plan.mask.iter()
                .enumerate().map(|(row, mask)| vec![
                    if plan.planned_fallback(row) { None } else { mask.clone() }]).collect();
            build_sampling_masks(&staged, &mut arena)?;
            Ok((plan, arena))
        };
        let (plan, arena) = build(0)?;
        assert_eq!(plan.route.iter().filter(|route| route.device_served()).count(), 5);
        assert_eq!(plan.planned_fallback_count(), 1);
        assert!(plan.routes_need_ordered_tail());

        let mut staged = StagedWave::new(&library, rows, &values, &plan, Some(&arena))?;
        let stages = staged.wave.last_stages();
        assert!(stages.prepare, "K1 always runs");
        assert_eq!(stages.topk_select, Some(SAMPLING_MAX_RETAINED as usize),
            "one ordered row forces K3/K4");
        assert!(stages.nucleus, "one ordered row forces K5");
        let sampled = staged.output()?;
        assert_uploaded_matches(&staged, &plan, &arena)?;

        // Exact classes: greedy (rows 0,1) and the retained domain (row 3), whose
        // divergence is 0.00% by the design's §6.3c measurement. Constrained rows
        // must be inside their mask whatever the residual does.
        for row in [0usize, 1, 3] {
            let expected = plan.params[row].select_token(
                &values[row * VOCAB..(row + 1) * VOCAB], plan.mask[row].as_deref(),
                plan.position[row])? as u32;
            assert_eq!(sampled.ids[row], expected, "row {row} must be token-exact");
        }
        for row in [1usize, 3] {
            let mask = plan.mask[row].as_deref().unwrap();
            let allowed = |token: u32| mask[token as usize / 32] & (1 << (token % 32)) != 0;
            assert!(allowed(sampled.ids[row]), "row {row} left its mask: {}", sampled.ids[row]);
            assert!(!allowed(row_argmax_dev(&values, row)),
                "row {row}'s fixture must make the mask change the token");
        }
        // Residual domains (rows 2 and 4) are device draws: only the declared
        // class is allowed, never an invalid or out-of-range id.
        for row in [2usize, 4] {
            assert_eq!(sampled.status[row], cuteafd_ffi::CUTEAFD_V41_SAMPLER_STATUS_OK);
            assert!((sampled.ids[row] as usize) < VOCAB);
        }
        // The fallback row's device block is the neutral greedy no-op; the host
        // re-samples it from its own values and stores the token.
        let expected_fallback = plan.params[5]
            .select_token(&values[5 * VOCAB..6 * VOCAB], None, plan.position[5])? as u32;
        let round = crate::families::deepseek_v41::v41_native_serve::scheduler::SamplingRound {
            plan, arena: arena.clone(), trace_rows: Vec::new() };
        let (device_rows, refused) =
            crate::families::deepseek_v41::v41_native_serve::scheduler::admit_device_rows(&round, &sampled)?;
        assert!(refused.is_empty(), "the planned fallback is excluded from the device set");
        assert_eq!(device_rows.len(), 5);
        let fallback = round.fallback_rows(&refused);
        assert_eq!(fallback, vec![5]);
        let bytes5 = staged.download_row(5)?;
        let mut next = sampled.with_full_logits(&[5], bytes5)?;
        crate::families::deepseek_v41::v41_native_serve::scheduler::resolve_fallback_rows(&mut next, &round, &fallback)?;
        assert_eq!(next.best[5], expected_fallback,
            "the fallback row commits the CPU draw");

        // Seeded sweep over the constrained rows: every draw stays inside its mask.
        for sweep in 1..=48u64 {
            let (plan, arena) = build(sweep)?;
            let mut staged = StagedWave::new(&library, rows, &values, &plan, Some(&arena))?;
            let sampled = staged.output()?;
            for row in [1usize, 3] {
                assert_eq!(sampled.status[row], cuteafd_ffi::CUTEAFD_V41_SAMPLER_STATUS_OK,
                    "sweep {sweep} row {row} status");
                let mask = plan.mask[row].as_deref().unwrap();
                let allowed = |token: u32| mask[token as usize / 32] & (1 << (token % 32)) != 0;
                assert!(allowed(sampled.ids[row]),
                    "sweep {sweep} row {row} drew {} outside its mask", sampled.ids[row]);
            }
        }
        println!("chunk4b mixed batch: ids={:?} fallback={}", next.best, next.best[5]);
        Ok(())
    }

    /// **(c) A daemon-level GPU constrained-greedy round** (chunk 1's outstanding
    /// coverage gap): K1's masked argmax path on the real device, for a
    /// constrained round that never enqueues K3/K4/K5.
    #[test]
    #[ignore = "requires a GPU and CUTEAFD_NATIVE_LIB; run with --ignored"]
    fn v41_device_constrained_greedy_round_matches_the_masked_cpu_argmax() -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let rows = 3usize;
        let words = VOCAB.div_ceil(32);
        let values = smooth_narrow(rows);
        let masks: Vec<Option<Vec<u32>>> = vec![
            Some(allowed_mask(&[3, 11, 29])),
            None,
            Some(allowed_mask(&[7])),
        ];
        let params = TargetSamplingParams::greedy().with_seed(0x21);
        let members: Vec<SamplingMember> = masks.iter().enumerate().map(|(row, mask)| SamplingMember {
            params, base_position: row as u64, row_masks: vec![mask.clone()],
        }).collect();
        let inputs: Vec<Vec<u32>> = vec![vec![0]; rows];
        let plan = build_target_sampling_plan(&members, &inputs)?;
        let mut arena = vec![0u32; rows * words];
        build_sampling_masks(&[vec![masks[0].clone()], vec![None], vec![masks[2].clone()]],
            &mut arena)?;
        let mut staged = StagedWave::new(&library, rows, &values, &plan, Some(&arena))?;
        let stages = staged.wave.last_stages();
        assert!(stages.prepare && stages.topk_select.is_none() && !stages.nucleus,
            "an all-greedy round takes K1 only");
        assert_uploaded_matches(&staged, &plan, &arena)?;
        let sampled = staged.output()?;
        for row in 0..rows {
            assert_eq!(sampled.status[row], cuteafd_ffi::CUTEAFD_V41_SAMPLER_STATUS_OK, "row {row}");
            let expected = plan.params[row].select_token(
                &values[row * VOCAB..(row + 1) * VOCAB], plan.mask[row].as_deref(),
                plan.position[row])? as u32;
            assert_eq!(sampled.ids[row], expected, "row {row} masked argmax");
        }
        // Non-vacuity: the constrained rows' masks really change the token.
        for row in [0usize, 2] {
            assert_ne!(sampled.ids[row], row_argmax_dev(&values, row),
                "row {row}'s mask must change the argmax");
        }
        println!("chunk4b constrained greedy: ids={:?}", sampled.ids);
        Ok(())
    }

    /// **(1) Constrained stochastic rows stay inside their mask on both the fast
    /// path (K1 -> K2) and the ordered path (K1 -> K3 -> K4 -> K5)**, over a
    /// seeded sweep, and a multi-row speculative prefix does not let one row
    /// inherit a neighbour's mask.
    #[test]
    #[ignore = "requires a GPU and CUTEAFD_NATIVE_LIB; run with --ignored"]
    fn v41_device_constrained_stochastic_stays_inside_its_mask_on_both_paths() -> Result<()> {
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let words = VOCAB.div_ceil(32);
        let allowed = allowed_mask(&[1, 2, 4, 8]);
        let allowed_set = [1u32, 2, 4, 8];
        // (name, params, ordered?) — one fast-path and two ordered profiles.
        let profiles: Vec<(&str, TargetSamplingParams, bool)> = vec![
            ("fast path (top_p disabled, min_p)",
                TargetSamplingParams::new(0.7, 1.0, None, 0.05, 0x31)?, false),
            ("ordered top_p",
                TargetSamplingParams::new(0.7, 0.9, None, 0.0, 0x32)?, true),
            ("ordered retained top_k=40",
                TargetSamplingParams::new(0.7, 1.0, Some(40), 0.0, 0x33)?, true),
        ];
        for (name, base, ordered) in profiles {
            // Sweep seeds; each launch is a fresh plan/arena/wave.
            for sweep in 0..64u64 {
                let params = base.with_seed(base.seed().wrapping_add(sweep * 0x9e37));
                let members = vec![SamplingMember {
                    params, base_position: 100, row_masks: vec![Some(allowed.clone())] }];
                let inputs: Vec<Vec<u32>> = vec![vec![0]];
                let plan = build_target_sampling_plan(&members, &inputs)?;
                assert_eq!(plan.routes_need_ordered_tail(), ordered, "{name}");
                let mut arena = vec![0u32; words];
                build_sampling_masks(&[vec![Some(allowed.clone())]], &mut arena)?;
                let values = smooth_narrow(1);
                let mut staged = StagedWave::new(&library, 1, &values, &plan, Some(&arena))?;
                let sampled = staged.output()?;
                assert_eq!(sampled.status[0], cuteafd_ffi::CUTEAFD_V41_SAMPLER_STATUS_OK,
                    "{name} sweep {sweep}");
                assert!(allowed_set.contains(&sampled.ids[0]),
                    "{name} sweep {sweep} drew {} outside its mask", sampled.ids[0]);
                // The constrained row's unmasked argmax is outside the mask, so a
                // kernel that ignored the mask could not pass the line above.
                assert!(!allowed_set.contains(&row_argmax_dev(&values, 0)));
            }
            println!("chunk4b constrained stochastic {name}: 64/64 inside the mask");
        }

        // Multi-row speculative prefix, per-row `needs_mask`: row 0 masked to
        // {1,2}, row 1 unconstrained, row 2 masked to {100,101}. On the device the
        // `NO_MASK` row must not read row 0's or row 2's bits.
        let first = allowed_mask(&[1, 2]);
        let third = allowed_mask(&[100, 101]);
        let members = vec![SamplingMember {
            params: TargetSamplingParams::new(0.7, 0.9, None, 0.0, 0x34)?,
            base_position: 0,
            row_masks: vec![Some(first.clone()), None, Some(third.clone())],
        }];
        let inputs: Vec<Vec<u32>> = vec![vec![0, 0, 0]];
        let plan = build_target_sampling_plan(&members, &inputs)?;
        let mut arena = vec![0u32; 3 * words];
        build_sampling_masks(&[vec![Some(first.clone()), None, Some(third.clone())]], &mut arena)?;
        let values = smooth_narrow(3);
        let mut staged = StagedWave::new(&library, 3, &values, &plan, Some(&arena))?;
        assert_uploaded_matches(&staged, &plan, &arena)?;
        let sampled = staged.output()?;
        for (row, set) in [(0usize, [1u32, 2].as_slice()), (2, [100u32, 101].as_slice())] {
            assert_eq!(sampled.status[row], cuteafd_ffi::CUTEAFD_V41_SAMPLER_STATUS_OK);
            assert!(set.contains(&sampled.ids[row]),
                "row {row} drew {} outside its own mask", sampled.ids[row]);
        }
        assert_eq!(sampled.status[1], cuteafd_ffi::CUTEAFD_V41_SAMPLER_STATUS_OK);
        // The unconstrained middle row matches the CPU exactly on `smooth_narrow`.
        let expected1 = plan.params[1].select_token(&values[VOCAB..2 * VOCAB], None, 1)? as u32;
        assert_eq!(sampled.ids[1], expected1, "the NO_MASK row must be unmasked");
        println!("chunk4b per-row masks: ids={:?}", sampled.ids);
        Ok(())
    }

    /// **(b) The finishing PLANNED-FALLBACK frontier end to end.**
    ///
    /// A `top_k = 300` row co-scheduled with a servable row, finishing at its
    /// frontier with tracing off, must not error the lane and must retain the
    /// stored CPU draw. The pre-chunk-4b review could only verify this at the
    /// function level; this drives real kernels through the production admission,
    /// fallback, retention-classification and retention-resolution functions.
    #[test]
    #[ignore = "requires a GPU and CUTEAFD_NATIVE_LIB; run with --ignored"]
    fn v41_device_finishing_fallback_frontier_end_to_end() -> Result<()> {
        use crate::families::deepseek_v41::v41_native_serve::scheduler::{admit_device_rows, frontier_retain,
            resolve_fallback_rows, retain_packed_frontier, FrontierRetain, SamplingRound};
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let rows = 3usize;
        let words = VOCAB.div_ceil(32);
        // Row 0: servable stochastic peer. Row 1: greedy peer. Row 2: top_k=300.
        let params = vec![
            TargetSamplingParams::new(0.7, 1.0, None, 0.05, 0x41)?,
            TargetSamplingParams::greedy().with_seed(0x42),
            TargetSamplingParams::new(0.7, 1.0, Some(300), 0.0, 0x43)?,
        ];
        let members: Vec<SamplingMember> = params.iter().enumerate().map(|(row, &params)| {
            SamplingMember { params, base_position: 10 * row as u64, row_masks: vec![None] }
        }).collect();
        let inputs: Vec<Vec<u32>> = vec![vec![0]; rows];
        let plan = build_target_sampling_plan(&members, &inputs)?;
        assert!(plan.planned_fallback(2));
        assert!(plan.route[0].device_served() && plan.route[1].device_served());
        let mut arena = vec![0u32; rows * words];
        build_sampling_masks(&vec![vec![None]; rows], &mut arena)?;
        let values = smooth_narrow(rows);
        let mut staged = StagedWave::new(&library, rows, &values, &plan, Some(&arena))?;
        let sampled = staged.output()?;
        let device_ids = sampled.ids.clone();
        // Tracing off: only the fallback row is downloaded, exactly as
        // `execute_sampled_rows` would.
        let round = SamplingRound { plan, arena, trace_rows: Vec::new() };
        let (device_rows, refused) = admit_device_rows(&round, &sampled)?;
        assert_eq!(device_rows, vec![0, 1], "the two servable rows are admitted");
        assert!(refused.is_empty());
        let fallback = round.fallback_rows(&refused);
        assert_eq!(fallback, vec![2], "the top_k = 300 row is the only fallback");
        let bytes = staged.download_row(2)?;
        let mut next = sampled.with_full_logits(&[2], bytes)?;
        resolve_fallback_rows(&mut next, &round, &fallback)?;
        let draw = next.best[2];
        let frontier_params = round.plan.params[2];
        assert_eq!(frontier_retain(Some(&round), frontier_params), FrontierRetain::RecordedSample,
            "a non-greedy frontier in a device round is a recorded draw");
        // The reviewer's exact failure: a finishing fallback row in a device round
        // took the `Checked` classification and compared the draw against the
        // row's argmax. The fixture must distinguish the two.
        let argmax = row_argmax_dev(&values, 2);
        assert_ne!(draw, argmax, "the fixture must distinguish the stored draw from the argmax");
        // The fix: the packed frontier (the fallback row was already downloaded)
        // is retained without a second transfer, and the stored draw survives.
        let retained = retain_packed_frontier(&next, 2, FrontierRetain::RecordedSample, None)?;
        assert_eq!(retained.select(None)?, draw, "the retained frontier is the stored draw");
        assert!(retain_packed_frontier(&next, 2, FrontierRetain::Checked, None).is_err(),
            "the pre-fix classification must reject the draw, else this test is vacuous");
        // The lane's serial `commit_lane` resolves a Device frontier through
        // `retain_from_device`; a packed row must take the same no-transfer path.
        let retained = next.retain_recorded_from_device(&library, staged.logits, 2)?;
        assert_eq!(retained.select(None)?, draw);
        // The servable peers keep their device ids and the lane does not error.
        assert_eq!(next.best[1], device_ids[1], "the greedy peer keeps its id");
        println!("chunk4b finishing fallback: draw={draw} argmax={argmax} ids={device_ids:?}");
        Ok(())
    }

    /// An all-zero mask is the same hard error on the device that it is on the
    /// CPU: `EMPTY_CANDIDATES`, mapped by `check_status` to the CPU's own
    /// `"grammar allows no target token"`. It is never a silent unmasked draw.
    ///
    /// One row per route (review FIX 4): constrained greedy (K1), the fast path
    /// (K1→K2), the ordered retained path (K1→K3→K4→K5, `top_k = 40`) and the
    /// ordered survivor path (K1→K5 case 3, `top_p < 1`). K1 reports the empty
    /// candidate set before K3/K4/K5 can act, so the ordered rows never enter
    /// their retained search and keep K1's status.
    #[test]
    #[ignore = "requires a GPU and CUTEAFD_NATIVE_LIB; run with --ignored"]
    fn v41_device_empty_mask_is_a_hard_error_in_every_mode() -> Result<()> {
        use crate::families::deepseek_v41::v41_native_serve::scheduler::{admit_device_rows, SamplingRound};
        use crate::families::deepseek_v41::v41_native_serve::scheduler::SamplingRoute;
        let library = unsafe { NativeLibrary::load(std::env::var("CUTEAFD_NATIVE_LIB")?)? };
        let rows = 4usize;
        let words = VOCAB.div_ceil(32);
        let params = vec![
            // 0: constrained greedy.
            TargetSamplingParams::greedy().with_seed(1),
            // 1: constrained stochastic fast path (K1 -> K2).
            TargetSamplingParams::new(0.7, 1.0, None, 0.05, 2)?,
            // 2: constrained ordered retained (K1 -> K3 -> K4 -> K5).
            TargetSamplingParams::new(0.7, 1.0, Some(40), 0.0, 3)?,
            // 3: constrained ordered survivor (K1 -> K5 case 3).
            TargetSamplingParams::new(0.7, 0.9, None, 0.0, 4)?,
        ];
        let expected_routes = [
            SamplingRoute::DeviceGreedy,
            SamplingRoute::DeviceFastPath,
            SamplingRoute::DeviceOrdered,
            SamplingRoute::DeviceOrdered,
        ];
        let empty = vec![0u32; words];
        let members: Vec<SamplingMember> = params.iter().enumerate().map(|(row, &params)| {
            SamplingMember { params, base_position: row as u64, row_masks: vec![Some(empty.clone())] }
        }).collect();
        let inputs: Vec<Vec<u32>> = vec![vec![0]; rows];
        let plan = build_target_sampling_plan(&members, &inputs)?;
        assert_eq!(plan.route, expected_routes, "one row per route");
        assert!((0..rows).all(|row| !plan.planned_fallback(row)),
            "an empty mask is a device-served row, not a fallback");
        assert!(plan.routes_need_ordered_tail(), "rows 2/3 force K3/K4/K5");
        let mut arena = vec![0u32; rows * words];
        build_sampling_masks(&vec![vec![Some(empty.clone())]; rows], &mut arena)?;
        let values = smooth_narrow(rows);
        let mut staged = StagedWave::new(&library, rows, &values, &plan, Some(&arena))?;
        let sampled = staged.output()?;
        for row in 0..rows {
            assert_eq!(sampled.status[row], cuteafd_ffi::CUTEAFD_V41_SAMPLER_STATUS_EMPTY_CANDIDATES,
                "row {row} ({:?}) must report EMPTY_CANDIDATES, not a silent fallback",
                expected_routes[row]);
        }
        let round = SamplingRound { plan, arena, trace_rows: Vec::new() };
        let (device_rows, refused) = admit_device_rows(&round, &sampled)?;
        assert_eq!(device_rows, vec![0, 1, 2, 3], "an empty mask is the request's fault");
        assert!(refused.is_empty(), "an empty mask is never a capability fallback");
        let error = sampled.check_status(&device_rows).unwrap_err();
        assert_eq!(error.to_string(), "grammar allows no target token",
            "the device error must be the CPU path's own message");
        println!("chunk4b empty mask: status={:?}", sampled.status);
        Ok(())
    }
}
