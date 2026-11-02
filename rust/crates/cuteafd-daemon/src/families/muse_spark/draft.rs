//! Muse Spark's DFlash2 drafter on the shared executor (`DflashDrafter`),
//! beside the target head on the last coordinator rank.
//!
//! - Stream: the head rank's own stream, so taps, context updates, drafts and
//!   the target step are ordered without events (as GLM Flash).
//! - Taps: after each tapped layer the head rank's full-width `hidden` rows
//!   (identical on both ranks of a pair after the layer) are copied into the
//!   drafter's tap rows; a prefill call keeps only its tail (`TAP_ROWS`, the
//!   drafter's sliding window).
//! - Input rows: the target's embedding table lives on rank 0; the drafter
//!   gathers its block's rows (anchor, then the mask token) from it over peer
//!   access on its own stream (`draft_device_with`), as rank 1's step embed.
//! - Head: the drafter has no head of its own. Its `TargetHead::Launch` runs
//!   the target's exact head on the drafter's own scratch: the K6 head into
//!   stored-width F32 rows, trimmed to the vocabulary, then `muse_softcap`
//!   (output multiplier and softcap, the values the drafter was trained on;
//!   admission refuses a drafter with others).
use super::engine::RankResources;
use crate::shared::memory::device::{Allocation, Device};
use crate::shared::speculation::dflash::{ContextRow, DflashDrafter, Draft, DraftSeq, TargetHead, TAP_ROWS};
use anyhow::{ensure, Context, Result};
use cuteafd_ffi::{programs::Scalar, CuteafdDeviceBuffer as Buffer};
use cuteafd_loader::families::muse_spark::MuseSparkConfig;
use std::{ffi::c_void, path::Path};

pub(crate) struct MuseDraft<'a> {
    pub drafter: DflashDrafter<'a>,
    /// The K6 head's stored-width F32 rows for a full draft batch.
    stored: Allocation<'a>,
    /// The trimmed F32 rows the softcap reads.
    trimmed: Allocation<'a>,
    /// Token ids of a draft batch's input rows (on the drafter's device).
    ids: Allocation<'a>,
    pub stream: *mut c_void,
    device: Device<'a>,
    pub block: usize,
    rows: usize,
}

impl<'a> MuseDraft<'a> {
    /// Loads `snapshot` on the head rank's device and stream. `sequences`
    /// bounds a draft batch and the context slots (the admitted charge).
    pub(crate) fn load(cfg: &MuseSparkConfig, head: &RankResources<'a>, stream: *mut c_void, snapshot: &Path,
        table: &crate::shared::token_io::TokenEmbedding<'_>, admitted: &cuteafd_loader::placement::families::muse_spark::MuseDrafter,
        kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache) -> Result<Self> {
        let device = head.device;
        let config = crate::shared::speculation::dflash::DflashConfig::read(snapshot)?;
        ensure!(config.hidden == cfg.hidden && config.vocab == cfg.vocab_size && config.block == admitted.block
            && config.taps.iter().all(|&tap| tap < cfg.layers), "Muse drafter does not fit the admitted target");
        let file = crate::shared::speculation::dflash::prefetch(snapshot).join()
            .map_err(|_| anyhow::anyhow!("Muse drafter read panicked"))??;
        let mask = table.host_rows(&[config.mask_token])?;
        let _scope = cuteafd_ffi::memory_ledger::scope("drafter");
        let drafter = device.run(|| DflashDrafter::load(device.library, snapshot, file, stream, admitted.context_slots,
            admitted.sequences, mask, true,
            cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation::Bf16Only,
            crate::shared::fp8_linear::Fp8Scales::Amax, crate::shared::fp8_linear::Fp8Rows::W8a16, kv_cache))?;
        let stored_width = head.globals.head.as_ref().context("Muse drafter on a rank without the head")?.shape[0];
        let rows = admitted.sequences * admitted.block;
        ensure!(admitted.head_scratch == (rows * (stored_width + cfg.vocab_size) * 4) as u64,
            "Muse drafter head scratch differs from its admission");
        // The ids below and the backend's route probe ring (`RoundProbe::new` at HEAD_ROWS).
        let probe = cfg.layers * cuteafd_loader::families::muse_spark::lane::HEAD_ROWS * cfg.topk * 4;
        ensure!(admitted.runtime == (rows * 4 + probe) as u64, "Muse drafter runtime buffers differ from their admission");
        let _scope = cuteafd_ffi::memory_ledger::scope("drafter/head");
        Ok(Self { drafter, stored: Allocation::new(device, rows * stored_width * 4)?,
            trimmed: Allocation::new(device, rows * cfg.vocab_size * 4)?, ids: Allocation::new(device, rows * 4)?,
            stream, device, block: admitted.block, rows })
    }

    /// Copies the head rank's `hidden` rows `[first, first + n)` after `layer`
    /// into tap rows `to..to + n` when the layer is tapped.
    pub(crate) fn tap(&self, layer: usize, hidden: Buffer, first: usize, n: usize, to: usize) -> Result<()> {
        if n == 0 { return Ok(()); }
        self.device.run(|| self.drafter.tap_at(layer, hidden.ptr.cast_const(), first, n, to))
    }

    /// Commits tap rows as context: row `r` is position `first + r` of `slot`.
    pub(crate) fn update(&self, slot: usize, first: usize, rows: usize) -> Result<()> {
        if rows == 0 { return Ok(()); }
        ensure!(rows <= TAP_ROWS, "Muse drafter commits at most {TAP_ROWS} tap rows");
        self.device.run(|| self.drafter.update(&(0..rows).map(|r| ContextRow { tap_row: r, slot, position: first + r })
            .collect::<Vec<_>>()))
    }

    /// Commits verify-step tap rows `tap_row..tap_row + rows` of `slot` as
    /// positions `first..first + rows` (a sequence's kept rows at its step offset).
    pub(crate) fn update_rows(&self, slot: usize, tap_row: usize, first: usize, rows: usize) -> Result<()> {
        ensure!(tap_row + rows <= TAP_ROWS, "Muse drafter tap rows beyond {TAP_ROWS}");
        self.device.run(|| self.drafter.update(&(0..rows).map(|r| ContextRow { tap_row: tap_row + r, slot,
            position: first + r }).collect::<Vec<_>>()))
    }

    /// A block per sequence, through the target's embedding (peer gather) and head.
    pub(crate) fn draft(&self, cfg: &MuseSparkConfig, head: &RankResources<'a>,
        table: &crate::shared::token_io::TokenEmbedding<'_>, sequences: &[DraftSeq]) -> Result<Vec<Draft>> {
        ensure!(sequences.len() * self.block <= self.rows, "Muse draft batch beyond its admitted rows");
        let weights = head.globals.head.as_ref().context("Muse drafter head weights")?;
        let (vocab, stored) = (cfg.vocab_size, weights.shape[0]);
        let (stored_plane, trimmed_plane, ids) = (self.stored.buffer, self.trimmed.buffer, self.ids.buffer);
        let input = |tokens: &[u32], output: Buffer, stream: *mut c_void| -> Result<()> {
            ensure!(stream == self.stream, "Muse draft input on another stream");
            table.embed(tokens, ids, 1, output, stream)
        };
        let library = self.device.library;
        let programs = &head.programs;
        let launch = move |x: *const c_void, logits: *mut f32, rows: usize, stream: *mut c_void| -> Result<()> {
            ensure!(stream == self.stream && rows <= self.rows, "Muse draft head on another stream or too wide");
            let device = self.device.id;
            let input = Buffer { ptr: x.cast_mut(), bytes: rows * cfg.hidden * 2, device_id: device, flags: 0 };
            let staged = Buffer { bytes: rows * stored * 4, ..stored_plane };
            let trimmed = Buffer { bytes: rows * vocab * 4, ..trimmed_plane };
            let out = Buffer { ptr: logits.cast(), bytes: rows * vocab * 4, device_id: device, flags: 0 };
            // The target head path (`forward::head` after its final norm, which the
            // drafter's own final norm replaces): K6 head, trim, softcap.
            head.core.local(rows, input, weights, staged, stream)?;
            // SAFETY: the drafter's own planes, distinct, ordered on its stream.
            unsafe {
                library.copy_d2d_2d_async(trimmed, vocab * 4, staged, stored * 4, vocab * 4, rows, stream)?;
                programs.launch("softcap", &[trimmed, out], &[Scalar::I32(i32::try_from(rows)?)], stream)
            }
        };
        self.device.run(|| self.drafter.draft_device_with(sequences, &input, TargetHead::Launch(Box::new(launch))))
    }
}

#[cfg(test)]
mod tests {
    /// The drafter's head is the target's head path in order (K6 head, trim to
    /// the vocabulary, softcap) on its own scratch and stream; its input rows
    /// come from the target table by peer gather; taps and updates stay on the
    /// head rank's stream.
    #[test]
    fn drafter_reads_the_target_head_and_table_on_its_stream() {
        let source = include_str!("draft.rs").split("#[cfg(test)]").next().unwrap();
        let draft = source.split("pub(crate) fn draft(").nth(1).unwrap();
        let order = ["head.core.local(rows, input, weights, staged, stream)", "copy_d2d_2d_async(trimmed, vocab * 4, staged",
            "programs.launch(\"softcap\", &[trimmed, out]"];
        let at: Vec<usize> = order.iter().map(|needle| draft.find(needle).unwrap()).collect();
        assert!(at.windows(2).all(|pair| pair[0] < pair[1]), "{order:?}");
        assert!(draft.contains("table.embed(tokens, ids, 1, output, stream)") && draft.contains("draft_device_with("));
        assert!(draft.contains("stream == self.stream"));
        let load = source.split("pub(crate) fn load(").nth(1).unwrap().split("pub(crate) fn tap(").next().unwrap();
        assert!(load.contains("GlmDraftRepresentation::Bf16Only") && load.contains("admitted.head_scratch =="));
    }
}
