//! Initial dense target executor over shared admission, token I/O and GDN state.
//! Qualification runs one sequence; serving and block-verify stay disabled until gated.
use super::{block::{Blocks, Planes, Step}, programs::DensePrograms, weights::DenseWeights};
use crate::shared::{memory::device::{Allocation, Device, Stream}, qwen::{GdnBank, GdnLayout, LayerStateMap},
    qwen::weights::WeightReader, target_head::TargetHead, token_io::{DeviceLogits, EmbedPlacement, EmbedSource, TokenEmbedding}};
use anyhow::{ensure, Context, Result};
use cuteafd_core::memory_layout::{Basis, Category};
use cuteafd_ffi::{programs::{Programs, Scalar, VocabularyHead, VOCABULARY_HEAD_WORKSPACE}, CuteafdDeviceBuffer, NativeLibrary};
use cuteafd_loader::{families::{qwen::HybridAttention, qwen3_5::Qwen35Config}, plan::checkpoint::Checkpoint,
    placement::{self, families::qwen3_5::{DenseInputs, LayerBytes}, Baseline, Demand, GpuBudget}};

const ROWS: usize = 4096;
const HEAD_ROWS: usize = 64;
const PREFIX: &str = "model.language_model.";

#[path = "verify.rs"]
mod verify;
pub(crate) use verify::DenseMark;
pub(crate) use placement::families::qwen3_5::DrafterBytes;

pub(crate) struct DenseEngine<'a> {
    // Drain queued work before fields retaining buffers/weights begin dropping.
    stream: Stream<'a>,
    pub(crate) cfg: Qwen35Config,
    programs: &'a Programs<'a>,
    admitted: DensePrograms,
    embedding: TokenEmbedding<'a>,
    weights: DenseWeights<'a>,
    homes: LayerStateMap,
    bank: GdnBank<'a>,
    kv: Vec<Option<Allocation<'a>>>,
    planes: Planes,
    _planes: Vec<Allocation<'a>>,
    ids: Allocation<'a>,
    logits: Allocation<'a>,
    head: VocabularyHead<'a>,
    _head_workspace: Allocation<'a>,
    pub(crate) pool_tokens: usize,
    pub(crate) position: usize,
    healthy: bool,
    verification: verify::Verification<'a>,
}

#[cfg(test)]
mod hook_borrow_tests {
    use super::*;

    #[test]
    fn target_head_workspace_owner_is_not_sync() {
        // Inference becomes ambiguous if a future refactor makes DenseEngine Sync.
        trait AmbiguousIfSync<A> { fn check() {} }
        impl<T: ?Sized> AmbiguousIfSync<()> for T {}
        struct IsSync;
        impl<T: ?Sized + Sync> AmbiguousIfSync<IsSync> for T {}
        let _ = <DenseEngine<'static> as AmbiguousIfSync<_>>::check;
    }

    #[test]
    fn embedding_stream_and_head_can_be_borrowed_together() {
        // Compile the actual binding pattern without requiring a CUDA fixture.
        fn bind(engine: &DenseEngine<'_>, hidden: CuteafdDeviceBuffer, output: CuteafdDeviceBuffer) -> Result<()> {
            let embedding = engine.token_embedding();
            let stream = engine.target_stream();
            let head = |x, logits, rows| engine.target_head(x, logits, rows);
            head(hidden, output, 1)?;
            embedding.check(&[0])?;
            stream.drain()
        }
        let _: fn(&DenseEngine<'_>, CuteafdDeviceBuffer, CuteafdDeviceBuffer) -> Result<()> = bind;
    }
}

fn alloc_bytes(bytes: usize) -> usize { bytes.max(256) }

fn weight_demand(group: &str, bytes: u64, category: Category) -> Demand {
    Demand::new(0, category, group, bytes, Basis::Exact)
}

impl<'a> DenseEngine<'a> {
    pub(crate) fn new(library: &'a NativeLibrary, programs: &'a Programs<'a>, checkpoint: &'a Checkpoint,
        cfg: Qwen35Config, requested_pool: Option<u64>, drafter: Option<DrafterBytes>) -> Result<Self> {
        let reader = WeightReader { library, checkpoint };
        let format = super::weights::DenseFormat::detect(&reader)?;
        let admitted = DensePrograms::new(&cfg, programs, format)?;
        let weight_bytes = reader.dense_bytes(&cfg, admitted.norm_dtype.clone(), format)?;
        programs.load_matching(|name| name.starts_with(&format!("{}_", admitted.prefix)))?;
        let device = Device { library, id: library.cuda_get_device()? };
        let stream = Stream::new(device)?;
        let layout = GdnLayout::new(cfg.gdn)?;
        let homes = LayerStateMap::new(&cfg.attention, &vec![0; cfg.attention.len()], 1)?;
        let mut layer_bytes = Vec::new();
        for (&kind, weights) in cfg.attention.iter().zip(weight_bytes.layers) {
            let (state, replay) = if kind == HybridAttention::Gdn {
                ((layout.state_slot_bytes + layout.conv_slot_bytes) as u64, layout.replay_layer_bytes as u64)
            } else { (0, 0) };
            layer_bytes.push(LayerBytes { weights, state, replay, prefix: 0 });
        }
        let hidden = ROWS * cfg.hidden * 2;
        let query = ROWS * cfg.heads * cfg.head_dim * 2;
        let table_bytes = cfg.max_context.div_ceil(64) * 4;
        let sizes = [hidden, hidden, hidden, query, query, query, admitted.scratch_bytes,
            ROWS * 4, ROWS * 4, ROWS * 12, ROWS * 8, ROWS * 8, table_bytes];
        let plane_bytes: usize = sizes.iter().map(|&bytes| alloc_bytes(bytes)).sum();
        let logit_bytes = HEAD_ROWS * cfg.vocab * 4;
        let mut fixed = vec![weight_demand("token embedding", weight_bytes.embedding, Category::Embedding),
            weight_demand("final norm", weight_bytes.norm, Category::Weights),
            weight_demand("vocabulary head", weight_bytes.head, Category::Weights),
            weight_demand("native step planes and scratch", plane_bytes as u64, Category::Workspace),
            weight_demand("vocabulary head workspace", VOCABULARY_HEAD_WORKSPACE as u64, Category::Workspace),
            weight_demand("logits", logit_bytes as u64, Category::Workspace),
            weight_demand("token ids", (ROWS * 4) as u64, Category::Workspace),
            weight_demand("GDN commit table", (3 * 64 * 4) as u64, Category::Workspace)];
        // cuBLAS handle/code and untracked runtime growth must not consume the pool.
        fixed.push(weight_demand("head and runtime growth", 512 << 20, Category::Runtime));
        fixed.push(weight_demand("block verify marks, logits and taps", verify::Verification::bytes(&cfg)? as u64,
            Category::Workspace));
        let (free, total) = library.cuda_memory_info()?;
        let request = placement::families::qwen3_5::request(DenseInputs { cfg: &cfg,
            gpu: GpuBudget { capacity_bytes: total as u64, headroom_bytes: 1 << 30,
                baseline: Baseline::Measured { free_bytes: free as u64 } }, layers: &layer_bytes,
            fixed, drafter, requested_pool, max_context: cfg.max_context as u64,
            record_bytes: (2 * cfg.kv_heads * cfg.head_dim * 2) as u64, pool_overhead: 0, pool_bounded: vec![] })?;
        let placement = placement::solve(&request)?;
        placement::families::qwen3_5::SINGLE.check(&placement)?;
        let pool_tokens = usize::try_from(placement.pool_tokens)?;
        tracing::info!(pool_tokens, layout = %placement.summary(), "dense Qwen admitted before weight/state allocation");
        let tensor = reader.tensor(&format!("{PREFIX}embed_tokens.weight"))?;
        let source = EmbedSource::new(&checkpoint.snapshot, &tensor.shard, &tensor.meta, cfg.hidden)?;
        let (embedding, weights) = TokenEmbedding::load(library, source, EmbedPlacement::Gpu,
            || reader.dense_native(&cfg, admitted.norm_dtype.clone(), format))?;
        let mut allocated = sizes.into_iter().map(|bytes| Allocation::new(device, alloc_bytes(bytes)))
            .collect::<Result<Vec<_>>>()?;
        let buffers: Vec<_> = allocated.iter().map(|allocation| allocation.buffer).collect();
        let planes = Planes { residual: buffers[0], norm: buffers[1], delta: buffers[2], query: buffers[3],
            gate: buffers[4], attention: buffers[5], scratch: buffers[6], slots: buffers[7], seq_first: buffers[8],
            rope_positions: buffers[9], kv_slots: buffers[10], positions: buffers[11], page_table: buffers[12] };
        let ids = Allocation::new(device, ROWS * 4)?;
        let logits = Allocation::new(device, logit_bytes)?;
        let head_workspace = Allocation::new(device, VOCABULARY_HEAD_WORKSPACE)?;
        // SAFETY: owned workspace is retained after the head and stream, including error cleanup.
        let head = unsafe { library.vocabulary_head_rows(head_workspace.buffer.ptr, cfg.hidden as u32,
            HEAD_ROWS as u32, cfg.vocab as u32)? };
        let commit = match admitted.prefix { "qwen35" => "qwen35_gdn_commit", "qwen38" => "qwen38_gdn_commit",
            _ => anyhow::bail!("unadmitted dense prefix") };
        let bank = GdnBank::new(library, programs, stream.raw, cfg.gdn, homes.gdn_layers[0], 1, commit)?;
        let kv = cfg.attention.iter().map(|&kind| -> Result<_> {
            if kind == HybridAttention::Gdn { return Ok(None); }
            Ok(Some(Allocation::new(device, pool_tokens * 2 * cfg.kv_heads * cfg.head_dim * 2)?))
        }).collect::<Result<_>>()?;
        let verification = verify::Verification::new(device, &cfg)?;
        Ok(Self { stream, cfg, programs, admitted, embedding, weights, homes, bank, kv, planes,
            _planes: std::mem::take(&mut allocated), ids, logits, head, _head_workspace: head_workspace,
            pool_tokens, position: 0, healthy: true, verification })
    }

    /// The one target-owned embedding table; the drafter must not clone its storage.
    pub(crate) fn token_embedding(&self) -> &TokenEmbedding<'a> { &self.embedding }

    /// Borrow the target stream, retaining its device scope and drain operations.
    pub(crate) fn target_stream(&self) -> &Stream<'a> { &self.stream }

    /// Queue the borrowed target head over already-normalized BF16 hidden rows.
    /// Source and destination storage must remain live until the target stream drains.
    /// The FFI-owned head workspace is reused in target-stream order, so borrowing
    /// the embedding concurrently needs no mutable engine borrow or storage copy.
    pub(crate) fn target_head(&self, hidden: CuteafdDeviceBuffer,
        output: CuteafdDeviceBuffer, rows: usize) -> Result<DeviceLogits> {
        ensure!((1..=HEAD_ROWS).contains(&rows), "target head requires 1..64 rows");
        ensure!(hidden.device_id == self.stream.device.id && output.device_id == self.stream.device.id,
            "target head buffers must reside on the target device");
        ensure!(!hidden.ptr.is_null() && hidden.bytes >= rows * self.cfg.hidden * 2,
            "target head hidden extent differs from dense geometry");
        ensure!(!output.ptr.is_null() && output.bytes >= rows * self.cfg.vocab * 4,
            "target head logits extent differs from dense vocabulary");
        self.stream.device.run(|| {
            // SAFETY: checked extents stay live through drain; !Sync engine serializes FFI workspace use on its one target stream.
            unsafe { crate::shared::target_head::launch(TargetHead::configured(), &self.head,
                hidden.ptr, self.weights.head.buffer.ptr, output.ptr.cast(), rows, self.stream.raw) }
        })?;
        Ok(DeviceLogits { ptr: output.ptr, rows, vocab: self.cfg.vocab, stride: self.cfg.vocab,
            stream: self.stream.raw, greedy: None })
    }

    fn put(&self, buffer: CuteafdDeviceBuffer, bytes: &[u8]) -> Result<()> {
        ensure!(bytes.len() <= buffer.bytes, "dense step table exceeds admitted capacity");
        // SAFETY: ordered upload owns pinned staging until this stream consumes it.
        unsafe { self.stream.device.library.copy_h2d_ordered(CuteafdDeviceBuffer { bytes: bytes.len(), ..buffer }, bytes, self.stream.raw) }
    }

    fn tables(&self, rows: usize) -> Result<()> {
        let first = self.position;
        let end = first.checked_add(rows).context("dense position overflow")?;
        ensure!(end <= self.cfg.max_context && end <= self.pool_tokens, "dense sequence exceeds admitted cache/context");
        self.put(self.planes.slots, &vec![0; rows * 4])?;
        self.put(self.planes.seq_first, &vec![0; rows * 4])?;
        let positions = (first..end).map(i64::try_from).collect::<std::result::Result<Vec<_>, _>>()?;
        let position_bytes: Vec<_> = positions.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.put(self.planes.positions, &position_bytes)?;
        self.put(self.planes.kv_slots, &position_bytes)?;
        let rope: Vec<_> = (first..end).map(i32::try_from).collect::<std::result::Result<Vec<_>, _>>()?
            .into_iter().flat_map(|p| [p; 3]).flat_map(i32::to_le_bytes).collect();
        self.put(self.planes.rope_positions, &rope)?;
        let pages: Vec<_> = (0..end.div_ceil(64)).map(i32::try_from).collect::<std::result::Result<Vec<_>, _>>()?
            .into_iter().flat_map(i32::to_le_bytes).collect();
        self.put(self.planes.page_table, &pages)
    }

    /// Evaluate causal input rows and return FP32 next-token distributions.
    /// Layer hook sees ordinary post-block BF16 hidden rows, before final norm.
    pub(crate) fn evaluate(&mut self, tokens: &[u32], mut hook: impl FnMut(usize, CuteafdDeviceBuffer) -> Result<()>)
        -> Result<Vec<f32>> {
        ensure!(!tokens.is_empty() && tokens.len() <= ROWS, "dense evaluation needs 1..4096 input rows");
        ensure!(self.healthy, "dense target cannot resume after a failed evaluation");
        ensure!(self.verification.pending.is_none(), "dense evaluation requires committed verify state");
        self.embedding.check(tokens)?;
        self.stream.drain()?;
        self.verification.invalidate()?;
        self.healthy = false;
        self.tables(tokens.len())?;
        self.embedding.embed(tokens, self.ids.buffer, 1, self.planes.residual, self.stream.raw)?;
        let step = Step { rows: tokens.len(), capacity: if tokens.len() <= 64 { 64 } else { ROWS },
            table_stride: 0, splits: 1, speculative: false };
        let blocks = Blocks { programs: self.programs, admitted: &self.admitted, cfg: &self.cfg,
            homes: &self.homes, bank: &self.bank, stream: self.stream.raw };
        for (index, layer) in self.weights.layers.iter().enumerate() {
            blocks.run(index, layer, self.planes, self.kv[index].as_ref().map(|v| v.buffer), &step)?;
            self.stream.drain()?;
            hook(index, CuteafdDeviceBuffer { bytes: tokens.len() * self.cfg.hidden * 2, ..self.planes.residual })?;
        }
        let norm = self.programs.program(&format!("{}_norm", self.admitted.prefix), &["x", "weight", "out"])?;
        // SAFETY: live norm, weight and output extents match the admitted native geometry.
        unsafe { norm.launch(&[self.planes.residual.ptr, self.weights.norm.buffer.ptr, self.planes.norm.ptr],
            &[Scalar::I32(i32::try_from(tokens.len())?)], self.stream.raw)?; }
        let mut output = Vec::with_capacity(tokens.len() * self.cfg.vocab);
        for first in (0..tokens.len()).step_by(HEAD_ROWS) {
            let rows = HEAD_ROWS.min(tokens.len() - first);
            let x = self.planes.norm.ptr.cast::<u8>().wrapping_add(first * self.cfg.hidden * 2);
            // SAFETY: head slices retain their normalized source and owned workspace until drain.
            unsafe { crate::shared::target_head::launch(TargetHead::configured(), &self.head, x.cast(),
                self.weights.head.buffer.ptr, self.logits.buffer.ptr.cast(), rows, self.stream.raw)?; }
            self.stream.drain()?;
            let mut bytes = vec![0; rows * self.cfg.vocab * 4];
            let byte_len = bytes.len();
            self.stream.device.library.copy_d2h(&mut bytes, CuteafdDeviceBuffer { bytes: byte_len, ..self.logits.buffer })?;
            output.extend(bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())));
        }
        self.position += tokens.len();
        self.healthy = true;
        Ok(output)
    }
}
