//! dSpark block drafter for GLM 5.3 Flash
//! (RedHatAI/GLM-5.3-Flash-speculator.dspark-preview: Speculators' DSparkDraftModel,
//! five Qwen3 layers of 64 heads x 64, a vanilla rank-256 Markov head and a
//! confidence head).
//!
//! Every target step taps the four-stream mean of the outputs of layers
//! `aux_hidden_state_layer_ids - 1` (vLLM's aux ids name a layer's input:
//! 20, 28, 32, 36, 40, 44 -> outputs of 19, 27, 31, 35, 39, 43) into
//! [`DsparkDrafter::taps`]; [`DsparkDrafter::update`] projects committed rows
//! (`hidden_norm(fc(taps))`) into every draft layer's context K/V, a ring of
//! the sequence's last 2048 positions (the checkpoint's sliding window: every
//! block row sees the same 2048 entries, as in training). A draft step runs
//! the layers over `[anchor, mask x 7]` at the anchor's position, causal
//! inside the block (sliding layers with `sliding_window_non_causal` off),
//! then the target's LM head; block row k predicts token `position + k + 1`
//! (`sample_from_anchor`): the Markov head adds `W2 . W1[previous token]` to
//! each row's logits and the drafts are taken greedily left to right, all on
//! the device. The confidence head predicts each row's acceptance
//! (`sigmoid(w . [h_k, W1[previous]] + b)`) for the draft policy.
//!
//! The embedding and LM head are the target's (the checkpoint's copies are
//! the official GLM 5.3 Flash tensors bit for bit, as in every GLM 5.3 Flash
//! quant), borrowed as the target keeps them ([`TargetHead`]: BF16, or its
//! FP8-only head). The GEMM weights are resident once, BF16 or (default) E4M3
//! packed at load, like the DFlash2 drafter ([`crate::shared::speculation::dflash`]).
//! Drafts only steer speculation: the
//! verify step keeps output identical to plain greedy decoding.
//! python/reference/families/glm5_flash/dspark/reference.py is the oracle.
use crate::shared::speculation::dflash::{ContextRow, Draft, DraftSeq, ReplayDrafter, TargetHead};
use crate::shared::fp8_linear;
use crate::shared::token_io::TokenEmbedding;
use cuteafd_loader::families::glm5::draft_representation::GlmDraftRepresentation;
use cuteafd_ffi::NativeLibrary;
use anyhow::{ensure, Result};
use std::path::Path;
use std::ffi::c_void;
use crate::shared::speculation::dflash::DraftHead;
pub(crate) use crate::shared::speculation::block_dspark::{
    is_dspark, check_checkpoint, prefetch, DsparkConfig, DsparkDrafter, MAX_SEQUENCES,
};

/// A drafter checkpoint read started ahead of the routed-expert load.
///
/// [`Drafter::load`] reads the drafter's snapshot inline, after the expert
/// packages: the 2.3 GB read then adds its whole duration to readiness. The
/// caller starts it here instead (before the expert load) and hands the handle
/// to [`Drafter::load`], which joins it, so the read runs under the exchange.
pub(crate) enum DraftRead {
    Dspark(std::thread::JoinHandle<Result<std::collections::HashMap<String, Vec<u8>>>>),
    Dflash(std::thread::JoinHandle<std::io::Result<Vec<u8>>>),
}

impl DraftRead {
    /// Starts reading the drafter snapshot `snapshot` names (the family's
    /// `prefetch` for whichever drafter it holds).
    pub(crate) fn start(snapshot: &Path) -> Result<Self> {
        Ok(if is_dspark(snapshot)? { Self::Dspark(prefetch(snapshot)) }
            else { Self::Dflash(crate::shared::speculation::dflash::prefetch(snapshot)) })
    }

    /// The DFlash2 file bytes, for a read started with [`Self::start`]; errors
    /// when the snapshot is a dSpark drafter.
    pub(crate) fn join_dflash(self) -> Result<Vec<u8>> {
        match self {
            Self::Dflash(handle) => Ok(handle.join().map_err(|_| anyhow::anyhow!("drafter read panicked"))??),
            Self::Dspark(_) => anyhow::bail!("the drafter read was started for dSpark, not DFlash2"),
        }
    }
}

/// The GLM 5.3 Flash drafter: DFlash2 or dSpark, behind one interface the
/// engine (taps), the serve loop and glmf-golden use.
pub(crate) enum Drafter<'a> {
    Dflash2(crate::shared::speculation::dflash::DflashDrafter<'a>),
    RemoteDflash(super::remote_dflash::RemoteDflash<'a>),
    Dspark(DsparkDrafter<'a>),
}

impl<'a> Drafter<'a> {
    pub fn require_live(&self) -> Result<()> {
        match self { Self::RemoteDflash(d) => d.require_live(), _ => Ok(()) }
    }

    pub fn publish_abort(&self) -> Result<()> {
        match self { Self::RemoteDflash(d) => d.publish_abort(), _ => Ok(()) }
    }

    pub fn synchronize_remote(&self) -> Result<()> {
        match self { Self::RemoteDflash(d) => d.synchronize(), _ => Ok(()) }
    }

    pub fn finish_request(&self) -> Result<()> {
        match self { Self::RemoteDflash(d) => d.finish_request(), _ => Ok(()) }
    }

    pub fn finish_terminal(&self, drained: bool) {
        if let Self::RemoteDflash(d) = self { d.finish_terminal(drained); }
    }

    /// The shared draft policy's prior: DFlash2's keyed selector fit, or the
    /// dSpark confidence head blended with the history rate.
    pub fn draft_prior(&self, fp8_head: bool) -> Result<super::draft_binding::Prior> {
        Ok(match self {
            Self::Dflash2(d) => super::draft_binding::Prior::Selector(d.selector_fit("glm5_flash", fp8_head)?),
            Self::RemoteDflash(d) => super::draft_binding::Prior::Selector(d.draft.selector_fit("glm5_flash", fp8_head)?),
            Self::Dspark(_) => super::draft_binding::Prior::Head,
        })
    }

    /// Loads the drafter `snapshot` names (dSpark when its config says so,
    /// else DFlash2) for a target of `hidden` x `vocab` with `layers` layers:
    /// `slots` ring contexts, draft steps of up to `sequences` sequences.
    #[allow(clippy::too_many_arguments)]
    pub fn load(library: &'a NativeLibrary, programs: &'a cuteafd_ffi::programs::Programs<'a>, snapshot: &Path, stream: *mut c_void, slots: usize, sequences: usize,
        embedding: &TokenEmbedding<'_>, hidden: usize, vocab: usize, layers: usize,
        representation: GlmDraftRepresentation, scales: fp8_linear::Fp8Scales, fp8_rows: fp8_linear::Fp8Rows, kv_cache: cuteafd_loader::families::qwen4::Qwen4KvCache, read: DraftRead)
        -> Result<Self> {
        if let DraftRead::Dspark(handle) = read {
            let cfg = DsparkConfig::read(snapshot)?;
            ensure!(cfg.hidden == hidden && cfg.vocab == vocab && cfg.taps.iter().all(|&l| l < layers),
                "the dSpark drafter does not fit this target");
            let mask = embedding.host_rows(&[cfg.mask_token])?;
            tracing::info!(mask_source = "target mask_token row", mask_token = cfg.mask_token, "drafter mask source");
            let tensors = handle.join().map_err(|_| anyhow::anyhow!("drafter read panicked"))??;
            return Ok(Self::Dspark(DsparkDrafter::load(library, programs, snapshot, tensors, stream, slots,
                sequences.min(MAX_SEQUENCES), mask, representation, scales, fp8_rows, kv_cache)?));
        }
        let cfg = crate::shared::speculation::dflash::DflashConfig::read(snapshot)?;
        ensure!(cfg.hidden == hidden && cfg.vocab == vocab && cfg.taps.iter().all(|&l| l < layers),
            "the DFlash2 drafter does not fit this target");
        let mask = embedding.host_rows(&[cfg.mask_token])?;
        tracing::info!(mask_source = "target mask_token row", mask_token = cfg.mask_token, "drafter mask source");
        let file = read.join_dflash()?;
        Ok(Self::Dflash2(crate::shared::speculation::dflash::DflashDrafter::load(library, programs, snapshot, file, stream, slots,
            sequences, mask, true, representation, scales, fp8_rows, kv_cache)?))
    }

    pub fn prepare_workspace(&self) -> Result<()> {
        match self {
            Self::Dflash2(d) => d.prepare_workspace(),
            Self::RemoteDflash(d) => d.draft.device.run(|| d.draft.prepare_workspace()),
            Self::Dspark(d) => d.prepare_workspace(),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Dflash2(_) | Self::RemoteDflash(_) => "DFlash2",
            Self::Dspark(_) => "dSpark",
        }
    }

    /// Most sequences one draft step takes.
    pub fn max_batch_sequences(&self) -> usize {
        match self {
            Self::Dflash2(d) => d.max_batch_sequences(),
            Self::RemoteDflash(d) => d.draft.max_batch_sequences(),
            Self::Dspark(d) => d.max_sequences,
        }
    }

    /// How draft steps run the target's BF16 head past one draft block (--draft-head).
    pub fn set_draft_head(&self, mode: DraftHead) {
        match self {
            Self::Dflash2(d) => d.set_draft_head(mode),
            Self::RemoteDflash(d) => d.draft.set_draft_head(mode),
            Self::Dspark(d) => d.set_draft_head(mode),
        }
    }

    /// How the FP8 GEMMs run from now on (--draft-linear): the load's mode or one before it.
    pub fn set_draft_linear(&self, mode: fp8_linear::Fp8Rows) -> Result<()> {
        match self {
            Self::Dflash2(d) => d.set_draft_linear(mode),
            Self::RemoteDflash(d) => d.draft.device.run(|| d.draft.set_draft_linear(mode)),
            Self::Dspark(d) => d.set_draft_linear(mode),
        }
    }

    /// Ring slots (sequences with a drafter context).
    pub fn slots(&self) -> usize {
        match self {
            Self::Dflash2(d) => d.slots,
            Self::RemoteDflash(d) => d.draft.slots,
            Self::Dspark(d) => d.slots,
        }
    }

    /// Tokens one draft proposes per sequence.
    pub fn drafts(&self) -> usize {
        match self {
            Self::Dflash2(d) => d.cfg.drafts(),
            Self::RemoteDflash(d) => d.draft.cfg.drafts(),
            Self::Dspark(d) => d.cfg.drafts(),
        }
    }

    /// Rows of one draft block.
    pub fn block(&self) -> usize {
        match self {
            Self::Dflash2(d) => d.cfg.block,
            Self::RemoteDflash(d) => d.draft.cfg.block,
            Self::Dspark(d) => d.cfg.block,
        }
    }

    /// Target layers whose outputs are tapped.
    pub fn taps(&self) -> &[usize] {
        match self {
            Self::Dflash2(d) => &d.cfg.taps,
            Self::RemoteDflash(d) => &d.draft.cfg.taps,
            Self::Dspark(d) => &d.cfg.taps,
        }
    }

    pub fn tap_streams(&self, layer: usize, streams: *const c_void, hc: usize, first: usize, n: usize) -> Result<()> {
        self.tap_streams_at(layer, streams, hc, first, n, 0)
    }

    pub fn tap_streams_at(&self, layer: usize, streams: *const c_void, hc: usize, first: usize, n: usize, to: usize)
        -> Result<()> {
        match self {
            Self::Dflash2(d) => d.tap_streams_at(layer, streams, hc, first, n, to),
            Self::RemoteDflash(d) => d.tap_streams_at(layer, streams, hc, first, n, to),
            Self::Dspark(d) => d.tap_streams_at(layer, streams, hc, first, n, to),
        }
    }

    pub fn update(&self, rows: &[ContextRow]) -> Result<()> {
        match self {
            Self::Dflash2(d) => d.update(rows),
            Self::RemoteDflash(d) => d.update(rows),
            Self::Dspark(d) => d.update(rows),
        }
    }

    pub fn draft_device(&self, sequences: &[DraftSeq], embedding: &TokenEmbedding<'_>, head: TargetHead<'_>)
        -> Result<Vec<Draft>> {
        match self {
            Self::Dflash2(d) => d.draft_device(sequences, embedding, head),
            Self::RemoteDflash(d) => d.draft_device(sequences, embedding, head),
            Self::Dspark(d) => d.draft_device(sequences, embedding, head),
        }
    }

    pub fn draft_device_diagnostic(&self, sequences: &[DraftSeq], embedding: &TokenEmbedding<'_>,
        head: TargetHead<'_>) -> Result<Vec<Draft>> {
        match self {
            Self::Dflash2(d) => d.draft_device_diagnostic(sequences, embedding, head),
            Self::RemoteDflash(d) => d.draft_device_diagnostic(sequences, embedding, head),
            Self::Dspark(d) => d.draft_device_diagnostic(sequences, embedding, head),
        }
    }

    pub fn put_taps(&self, rows: &[u8]) -> Result<()> {
        match self {
            Self::Dflash2(d) => d.put_taps(rows),
            Self::RemoteDflash(d) => d.put_taps(rows),
            Self::Dspark(d) => d.put_taps(rows),
        }
    }

    pub fn last_hidden(&self, sequences: usize) -> Result<Vec<u8>> {
        match self {
            Self::Dflash2(d) => d.last_hidden(sequences),
            Self::RemoteDflash(d) => d.draft.device.run(|| d.draft.last_hidden(sequences)),
            Self::Dspark(d) => d.last_hidden(sequences),
        }
    }

    /// The drafter as [`ReplayDrafter`] (teacher-forced replays).
    pub fn replay(&self) -> &dyn ReplayDrafter {
        match self {
            Self::Dflash2(d) => d,
            Self::RemoteDflash(d) => d,
            Self::Dspark(d) => d,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drafter_detection_reports_missing_and_corrupt_config() {
        let dir = tempfile::tempdir().unwrap();
        assert!(is_dspark(dir.path()).unwrap_err().to_string().contains("config.json"));
        std::fs::write(dir.path().join("config.json"), b"bad").unwrap();
        assert!(is_dspark(dir.path()).unwrap_err().to_string().contains("parsing drafter"));
    }

    #[test]
    fn reads_the_redhat_config() {
        let dir = std::env::temp_dir().join(format!("cuteafd-dspark-config-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), r#"{
          "aux_hidden_state_layer_ids": [20, 28, 32, 36, 40, 44], "block_size": 8,
          "confidence_head_with_markov": true, "draft_vocab_size": 154880, "enable_confidence_head": true,
          "markov_head_type": "vanilla", "markov_rank": 256, "mask_token_id": 154856, "sample_from_anchor": true,
          "sliding_window_non_causal": false, "speculators_model_type": "dspark",
          "transformer_layer_config": {"head_dim": 64, "hidden_act": "silu", "hidden_size": 4096,
            "intermediate_size": 12288, "layer_types": ["sliding_attention", "sliding_attention",
            "sliding_attention", "sliding_attention", "sliding_attention"], "num_attention_heads": 64,
            "num_hidden_layers": 5, "num_key_value_heads": 64, "rms_norm_eps": 1e-05,
            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}, "sliding_window": 2048,
            "vocab_size": 154880}}"#).unwrap();
        assert!(is_dspark(&dir).unwrap());
        let cfg = DsparkConfig::read(&dir).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(cfg.taps, vec![19, 27, 31, 35, 39, 43]);
        assert_eq!((cfg.block, cfg.drafts(), cfg.qkv_width(), cfg.kv_width()), (8, 8, 12288, 4096));
        assert!(cfg.causal);
        assert_eq!(cfg.mask_token, 154856);
    }
}
