//! Vocabulary embedding (with shared vision injection) and F32 softcap-30 head for Inkling 2 Mini.
use super::{
    layers::MiniKernels,
    owners::{UnitBuffers, UnitOwner},
    weights::MiniTargetWeights,
};
use crate::shared::{
    memory::device::{Allocation, Stream},
    target_head::{self, TargetHead},
};
use anyhow::{ensure, Result};
use cuteafd_core::memory_layout::Tag;
use cuteafd_ffi::{inkling2::producers::Precision, programs::VocabularyHead, CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::families::inkling2_mini::Inkling2MiniConfig;
use std::rc::Rc;

const HIDDEN: usize = 2304;

pub(crate) struct MiniHead<'a> {
    stream: Rc<Stream<'a>>,
    bf16_head: (VocabularyHead<'a>, Allocation<'a>),
    tokens: Allocation<'a>,
    normalized: Allocation<'a>,
    pub logits: Allocation<'a>,
    rows: usize,
}

impl<'a> MiniHead<'a> {
    pub fn new(stream: Rc<Stream<'a>>, cfg: &Inkling2MiniConfig, rows: usize) -> Result<Self> {
        let t = &cfg.text;
        ensure!(rows > 0 && rows <= 4096, "Inkling 2 Mini head rows exceed export");
        let d = stream.device;
        let alloc = |bytes, name| Allocation::new(d, bytes, Tag::workspace(name));
        let tokens = alloc(rows * 4, "inkling2_mini/head/tokens")?;
        let normalized = alloc(rows * t.hidden * 2, "inkling2_mini/head/normalized")?;
        let logits = alloc(rows * t.unpadded_vocab_size * 4, "inkling2_mini/head/logits")?;
        let workspace = Allocation::new(
            d,
            cuteafd_ffi::programs::VOCABULARY_HEAD_WORKSPACE,
            Tag::workspace("inkling2_mini/head/blas"),
        )?;
        // SAFETY: admitted local workspace outlives all launches.
        let head = d.run(|| unsafe {
            d.library.vocabulary_head_rows(
                workspace.buffer.ptr,
                t.hidden as u32,
                rows as u32,
                t.unpadded_vocab_size as u32,
                TargetHead::configured().native_mode(),
            )
        })?;
        Ok(Self {
            stream,
            bf16_head: (head, workspace),
            tokens,
            normalized,
            logits,
            rows,
        })
    }

    pub fn inventory(&self, inventory: &mut crate::shared::memory::receipts::Inventory<'_>, prefix: &str) -> Result<()> {
        for (name, owner) in [
            ("tokens", &self.tokens),
            ("normalized", &self.normalized),
            ("logits", &self.logits),
            ("workspace", &self.bf16_head.1),
        ] {
            owner.inventory(inventory, &format!("{prefix}.{name}"))?;
        }
        Ok(())
    }

    /// Cold golden embedding: BF16 embed -> embed_norm -> BF16 -> exact BF16-rounded F32 `partial`.
    pub fn embed(
        &self,
        k: &MiniKernels<'_>,
        owner: &UnitOwner<'_>,
        target: &MiniTargetWeights<'a>,
        cfg: &Inkling2MiniConfig,
        tokens: &[u32],
    ) -> Result<()> {
        let t = &cfg.text;
        ensure!(
            !tokens.is_empty()
                && tokens.len() <= self.rows
                && tokens.iter().all(|&tok| (tok as usize) < t.unpadded_vocab_size),
            "Inkling 2 Mini embedding token bounds"
        );
        let d = self.stream.device;
        self.stream.drain()?;
        d.run(|| {
            d.library.copy_h2d(
                self.tokens.buffer,
                &tokens.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
            )?;
            // SAFETY: gather BF16 embed, widen to F32, norm to BF16 (exact f1940f40eb), widen to F32 partial.
            unsafe {
                d.library.cuda_embedding_lookup_bf16_async(
                    target.operand("model.llm.embed.weight")?.buffer,
                    self.tokens.buffer,
                    self.normalized.buffer,
                    tokens.len(),
                    t.vocab_size,
                    t.hidden,
                    self.stream.raw,
                )?;
                d.library
                    .cuda_zero_bytes_async(owner.buffer("branch")?, tokens.len() * t.hidden * 4, self.stream.raw)?;
                d.library.cuda_accumulate_bf16_to_f32_async(
                    self.normalized.buffer,
                    owner.buffer("branch")?,
                    tokens.len() * t.hidden,
                    self.stream.raw,
                )?;
            }
            k.norm(
                owner,
                owner.buffer("branch")?,
                target.operand("model.llm.embed_norm.weight")?.buffer,
                owner.buffer("projection_input")?,
                false,
                Precision::Bf16,
                tokens.len(),
            )?;
            k.widen_zero(owner, owner.buffer("projection_input")?, owner.buffer("partial")?, tokens.len() * t.hidden)
        })
    }

    /// Serving embedding with optional PE-Lang vision feature injection (`still::inject_f32`).
    pub fn embed_unit(
        &self,
        k: &MiniKernels<'a>,
        owner: &UnitOwner<'a>,
        target: &MiniTargetWeights<'a>,
        cfg: &Inkling2MiniConfig,
        tokens: &[u32],
        media: &cuteafd_engine::media::MediaChunk,
        uploads: &mut crate::shared::token_table::RetainedUploads<'a>,
    ) -> Result<()> {
        let t = &cfg.text;
        ensure!(
            !tokens.is_empty()
                && tokens.len() <= owner.layout.rows
                && tokens.iter().all(|&tok| (tok as usize) < t.unpadded_vocab_size)
                && owner.stream.raw == self.stream.raw
                && owner.stream.device.id == self.stream.device.id,
            "Inkling 2 Mini prefill embedding exceeds admitted owner/stream"
        );
        let bytes = tokens.iter().flat_map(|tok| tok.to_le_bytes()).collect::<Vec<_>>();
        self.stream.device.run(|| {
            // SAFETY: completion-fenced upload source; disjoint unit scratch planes.
            unsafe {
                uploads.upload(owner.buffer("first")?, &bytes, self.stream.raw)?;
                self.stream.device.library.cuda_embedding_lookup_bf16_async(
                    target.operand("model.llm.embed.weight")?.buffer,
                    owner.buffer("first")?,
                    owner.buffer("projection_input")?,
                    tokens.len(),
                    t.vocab_size,
                    t.hidden,
                    self.stream.raw,
                )?;
                self.stream.device.library.cuda_zero_bytes_async(
                    owner.buffer("branch")?,
                    tokens.len() * t.hidden * 4,
                    self.stream.raw,
                )?;
                self.stream.device.library.cuda_accumulate_bf16_to_f32_async(
                    owner.buffer("projection_input")?,
                    owner.buffer("branch")?,
                    tokens.len() * t.hidden,
                    self.stream.raw,
                )?;
            }
            k.norm(
                owner,
                owner.buffer("branch")?,
                target.operand("model.llm.embed_norm.weight")?.buffer,
                owner.buffer("projection_input")?,
                false,
                Precision::Bf16,
                tokens.len(),
            )?;
            k.widen_zero(
                owner,
                owner.buffer("projection_input")?,
                owner.buffer("partial")?,
                tokens.len() * t.hidden,
            )?;
            crate::shared::vision::still::inject_f32(
                &owner.stream,
                media,
                owner.buffer("branch")?,
                owner.buffer("route_ids")?,
                owner.buffer("projection_input")?,
                owner.buffer("partial")?,
                tokens.len(),
                t.hidden,
            )
        })
    }

    pub fn normalize(
        &self,
        k: &MiniKernels<'a>,
        owner: &impl UnitBuffers<'a>,
        target: &MiniTargetWeights<'_>,
        rows: usize,
        candidates: usize,
    ) -> Result<Buffer> {
        ensure!((1..=self.rows).contains(&rows), "Inkling 2 Mini normalized tap exceeds head owner");
        k.residual_norm(
            owner,
            owner.buffer("residual_arena")?,
            target.operand("model.llm.attn_res.output_query")?.buffer,
            target.operand("model.llm.norm.weight")?.buffer,
            self.normalized.buffer,
            rows,
            1,
            candidates,
        )?;
        self.normalized(rows)
    }

    pub fn normalized(&self, rows: usize) -> Result<Buffer> {
        ensure!((1..=self.rows).contains(&rows), "Mini final norm tap rows outside head owner");
        super::cache::view(self.normalized.buffer, 0, rows * HIDDEN * 2)
    }

    pub fn project_normalized(
        &self,
        target: &MiniTargetWeights<'a>,
        cfg: &Inkling2MiniConfig,
        normalized_row_offset: usize,
        rows: usize,
        output_row: usize,
    ) -> Result<Buffer> {
        let t = &cfg.text;
        ensure!(
            rows > 0
                && normalized_row_offset
                    .checked_add(rows)
                    .is_some_and(|end| end <= self.rows)
                && output_row.checked_add(rows).is_some_and(|end| end <= self.rows),
            "Inkling 2 Mini normalized projection rows exceed head owner"
        );
        let input = super::cache::view(
            self.normalized.buffer,
            normalized_row_offset * HIDDEN * 2,
            rows * HIDDEN * 2,
        )?;
        let logits = super::cache::view(
            self.logits.buffer,
            output_row * t.unpadded_vocab_size * 4,
            rows * t.unpadded_vocab_size * 4,
        )?;
        let weight = target.operand("model.llm.unembed.weight")?.buffer;
        // SAFETY: BF16 normalized input x BF16 unembed -> F32 logits + F32 softcap 30.
        unsafe {
            target_head::launch(
                TargetHead::configured(),
                &self.bf16_head.0,
                input.ptr,
                weight.ptr,
                logits.ptr.cast(),
                rows,
                self.stream.raw,
            )?;
            self.stream.device.library.cuda_logits_softcap_f32_async(
                logits.ptr,
                rows,
                t.unpadded_vocab_size,
                t.unpadded_vocab_size,
                t.final_softcap as f32,
                self.stream.raw,
            )?;
        }
        Ok(logits)
    }

    pub fn project_at(
        &self,
        k: &MiniKernels<'a>,
        owner: &impl UnitBuffers<'a>,
        target: &MiniTargetWeights<'a>,
        cfg: &Inkling2MiniConfig,
        rows: usize,
        candidates: usize,
        output_row: usize,
    ) -> Result<Buffer> {
        ensure!(
            rows > 0
                && output_row.checked_add(rows).is_some_and(|end| end <= self.rows)
                && owner.stream().raw == self.stream.raw
                && owner.stream().device.id == self.stream.device.id,
            "Inkling 2 Mini head rows/stream differ from admitted owner"
        );
        self.normalize(k, owner, target, rows, candidates)?;
        self.project_normalized(target, cfg, 0, rows, output_row)
    }
}

impl Drop for MiniHead<'_> {
    fn drop(&mut self) {
        self.stream.drain_or_abort("Inkling 2 Mini head handle/workspace retirement");
    }
}
