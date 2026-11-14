//! Trimmed vocabulary and final softcap use the shared target-head route.
use super::{layers::Kernels, owners::UnitOwner, weights::TargetWeights};
use crate::shared::{memory::device::{Allocation, Stream}, target_head::{self, TargetHead}};
use anyhow::{ensure, Result};
use cuteafd_core::memory_layout::Tag;
use cuteafd_ffi::{inkling2::producers::Precision, programs::VocabularyHead, CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::families::inkling2::Inkling2Config;
use std::rc::Rc;

pub(crate) struct Head<'a> {
    stream: Rc<Stream<'a>>,
    head: VocabularyHead<'a>,
    tokens: Allocation<'a>,
    normalized: Allocation<'a>,
    pub logits: Allocation<'a>,
    _workspace: Allocation<'a>,
    rows: usize,
}
impl<'a> Head<'a> {
    /// Caller admits tokens, normalized, logits and shared BLAS workspace first.
    pub fn new(stream: Rc<Stream<'a>>, cfg: &Inkling2Config, rows: usize) -> Result<Self> {
        ensure!(rows>0 && rows<=4096, "Inkling head rows exceed export");
        let d = stream.device;
        let alloc = |bytes, name| Allocation::new(d, bytes, Tag::workspace(name));
        let tokens = alloc(rows*4, "inkling2/head/tokens")?;
        let normalized = alloc(rows*cfg.hidden*2, "inkling2/head/normalized")?;
        let logits = alloc(rows*cfg.unpadded_vocab_size*4, "inkling2/head/logits")?;
        let workspace = alloc(Self::workspace_bytes(cfg, rows)?, "inkling2/head/blas")?;
        // SAFETY: admitted workspace survives the handle and stream drain;
        // passing unpadded vocabulary makes the stored tail rows unreachable.
        let head = d.run(|| unsafe { d.library.vocabulary_head_rows(workspace.buffer.ptr, cfg.hidden as u32,
            rows as u32, cfg.unpadded_vocab_size as u32, TargetHead::configured().native_mode()) })?;
        Ok(Self { stream, head, tokens, normalized, logits, _workspace: workspace, rows })
    }
    pub fn workspace_bytes(_cfg: &Inkling2Config, _rows: usize) -> Result<usize> {
        Ok(cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE)
    }
    pub fn inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, owner) in [("tokens", &self.tokens), ("normalized", &self.normalized),
            ("logits", &self.logits), ("workspace", &self._workspace)] {
            owner.inventory(inventory, &format!("{prefix}.{name}"))?;
        }
        Ok(())
    }
    pub fn embed(&self, k: &Kernels<'_>, owner: &UnitOwner<'_>, target: &TargetWeights<'_>,
        cfg: &Inkling2Config, tokens: &[u32]) -> Result<()> {
        ensure!(!tokens.is_empty() && tokens.len()<=self.rows && tokens.iter().all(|&t| (t as usize)<cfg.unpadded_vocab_size),
            "Inkling embedding token bounds");
        let d = self.stream.device;
        self.stream.drain()?;
        d.run(|| {
            d.library.copy_h2d(self.tokens.buffer, &tokens.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
            // SAFETY: shared gather reads native BF16 arena; zero+accumulate is
            // an exact widening into the retained F32 residual.
            unsafe {
                d.library.cuda_embedding_lookup_bf16_async(target.operand("model.llm.embed.weight")?.buffer,
                    self.tokens.buffer, self.normalized.buffer, tokens.len(), cfg.vocab_size, cfg.hidden, self.stream.raw)?;
                d.library.cuda_zero_bytes_async(owner.buffer("branch")?, tokens.len()*cfg.hidden*4, self.stream.raw)?;
                d.library.cuda_accumulate_bf16_to_f32_async(self.normalized.buffer, owner.buffer("branch")?,
                    tokens.len()*cfg.hidden, self.stream.raw)?;
            }
            k.norm(owner, owner.buffer("branch")?, target.operand("model.llm.embed_norm.weight")?.buffer,
                owner.buffer("partial")?, false, Precision::F32, tokens.len())
        })
    }
    pub fn project(&self, k: &Kernels<'_>, owner: &UnitOwner<'_>, target: &TargetWeights<'_>,
        cfg: &Inkling2Config, rows: usize, candidates: usize) -> Result<Buffer> {
        ensure!(rows>0 && rows<=self.rows, "Inkling head rows exceed admitted owner");
        k.residual(owner, target.operand("model.llm.attn_res.output_query")?.buffer,
            owner.buffer("partial")?, owner.buffer("mixed")?, rows, 1, candidates, false)?;
        k.norm(owner, owner.buffer("mixed")?, target.operand("model.llm.norm.weight")?.buffer,
            self.normalized.buffer, false, Precision::Bf16, rows)?;
        // SAFETY: BF16 normalized rows and trimmed weight matrix stay live;
        // shared head writes contiguous F32 logits, then shared softcap in place.
        unsafe {
            target_head::launch(TargetHead::configured(), &self.head, self.normalized.buffer.ptr,
                target.operand("model.llm.unembed.weight")?.buffer.ptr, self.logits.buffer.ptr.cast(), rows, self.stream.raw)?;
            self.stream.device.library.cuda_logits_softcap_f32_async(self.logits.buffer.ptr, rows,
                cfg.unpadded_vocab_size, cfg.unpadded_vocab_size, cfg.final_softcap as f32, self.stream.raw)?;
        }
        Ok(self.logits.buffer)
    }
}
impl Drop for Head<'_> {
    fn drop(&mut self) { self.stream.drain_or_abort("Inkling head handle/workspace retirement"); }
}
