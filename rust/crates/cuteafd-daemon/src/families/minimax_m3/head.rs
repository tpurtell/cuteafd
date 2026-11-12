//! BF16 embedding and zero-centered final norm through the shared target head.
use super::weights::TargetWeights;
use crate::shared::{memory::{device::{Allocation, Stream}, receipts::Inventory}, target_head::{self, TargetHead}};
use anyhow::{ensure, Result};
use cuteafd_core::memory_layout::Tag;
use cuteafd_ffi::{minimax_m3::forward::{ForwardProducer, Kind, NormBatch},
    programs::{Programs, VocabularyHead}, CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::families::minimax_m3::MiniMaxM3Config;
use std::rc::Rc;

/// The handle drops before its borrowed BLAS workspace. All storage survives
/// queued embedding/norm/head operations through the retained stream's drain.
pub(crate) struct Head<'a> {
    stream: Rc<Stream<'a>>,
    head: VocabularyHead<'a>,
    norm: ForwardProducer<'a>,
    tokens: Allocation<'a>,
    normalized: Allocation<'a>,
    logits: Allocation<'a>,
    workspace: Allocation<'a>,
    rows: usize,
    vocab: usize,
}
impl Drop for Head<'_> {
    fn drop(&mut self) { self.stream.drain_or_abort("MiniMax head retirement"); }
}
impl<'a> Head<'a> {
    pub fn new(stream: Rc<Stream<'a>>, programs: &'a Programs<'a>, cfg: &MiniMaxM3Config, rows: usize) -> Result<Self> {
        ensure!((1..=4096).contains(&rows) && cfg.hidden == 6144, "MiniMax head geometry");
        let device = stream.device;
        let allocate = |bytes| Allocation::new(device, bytes, Tag::workspace("minimax/head"));
        let tokens = allocate(rows * 4)?;
        let normalized = allocate(rows * 6144 * 2)?;
        let logits = allocate(rows.checked_mul(cfg.vocab_size).and_then(|n| n.checked_mul(4))
            .ok_or_else(|| anyhow::anyhow!("MiniMax head extent overflow"))?)?;
        let workspace = allocate(cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE)?;
        let head = device.run(|| {
            // SAFETY: handle borrows this retained private workspace until drop;
            // row and checkpoint dimensions are checked before construction.
            unsafe { device.library.vocabulary_head_rows(workspace.buffer.ptr, 6144,
                rows as u32, cfg.vocab_size as u32, TargetHead::configured().native_mode()) }
        })?;
        let norm = device.run(|| Ok(ForwardProducer::bind(programs, Kind::ResidualNorm)?))?;
        Ok(Self { stream, head, norm, tokens, normalized, logits, workspace, rows, vocab: cfg.vocab_size })
    }
    pub fn inventory(&self, inventory: &mut Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, allocation) in [("tokens", &self.tokens), ("normalized", &self.normalized),
            ("logits", &self.logits), ("workspace", &self.workspace)] {
            allocation.inventory(inventory, &format!("{prefix}.{name}"))?;
        }
        Ok(())
    }
    pub fn embed(&self, weights: &TargetWeights<'_>, tokens: &[u32], residual: Buffer) -> Result<()> {
        ensure!(!tokens.is_empty() && tokens.len() <= self.rows
            && tokens.iter().all(|&token| (token as usize) < self.vocab), "MiniMax embedding tokens");
        let device = self.stream.device;
        self.stream.drain()?;
        let bytes = tokens.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        let embedding = weights.operand("language_model.model.embed_tokens.weight")?.buffer;
        device.run(|| {
            device.library.copy_h2d(self.tokens.buffer, &bytes)?;
            // SAFETY: target arena and all scratch remain live on this stream;
            // residual starts at zero so the existing BF16->FP32 add is a cast.
            unsafe {
                device.library.cuda_embedding_lookup_bf16_async(embedding, self.tokens.buffer,
                    self.normalized.buffer, tokens.len(), self.vocab, 6144, self.stream.raw)?;
                device.library.cuda_zero_bytes_async(residual, tokens.len() * 6144 * 4, self.stream.raw)?;
                device.library.cuda_accumulate_bf16_to_f32_async(self.normalized.buffer, residual,
                    tokens.len() * 6144, self.stream.raw)
            }
        })
    }
    pub fn project(&self, weights: &TargetWeights<'_>, residual: Buffer, rows: usize) -> Result<Vec<f32>> {
        ensure!(rows > 0 && rows <= self.rows, "MiniMax head live rows");
        let norm = weights.operand("language_model.model.norm.weight")?.buffer;
        let weight = weights.operand("language_model.lm_head.weight")?.buffer;
        let device = self.stream.device;
        device.run(|| {
            // SAFETY: all exact arena/scratch extents are retained; same ordered
            // stream as the shared prefill's final residual write.
            unsafe {
                self.norm.norm(NormBatch { rows, input: residual, weight: norm,
                    output: self.normalized.buffer }, self.stream.raw)?;
                target_head::launch(TargetHead::configured(), &self.head, self.normalized.buffer.ptr,
                    weight.ptr, self.logits.buffer.ptr.cast(), rows, self.stream.raw)?;
            }
            Ok(())
        })?;
        self.stream.drain()?;
        let mut bytes = vec![0; rows * self.vocab * 4];
        device.run(|| device.library.copy_d2h(&mut bytes, self.logits.buffer))?;
        let logits = bytes.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect::<Vec<_>>();
        ensure!(logits.iter().all(|value| value.is_finite()), "MiniMax nonfinite head logits");
        Ok(logits)
    }
}
