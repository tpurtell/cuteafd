//! Inkling 2 Small model arguments from `config.json` (`Inkling2ForConditionalGeneration`,
//! `model_type=inkling2`), the reader shared by the planner and runtime.
//!
//! Semantics follow the pinned vendor integrations the #1915 reference was
//! verified against (vLLM `d664cd1e` `vllm/models/inkling2/`, SGLang `ec5e7ff5`):
//! - GQA on every layer; `local_layer_ids` attend a sliding window of
//!   `sliding_window_size`, the others the whole context (with per-token
//!   log-scaling of q and r above `log_scaling_n_floor` positions).
//! - A relative-position bias (`wr_du`, `rel_logits_proj`), per-token per-head
//!   attention sinks, a sigmoid output gate, L2 clips on q, k and r, and short
//!   causal K/V convolutions (static taps on local layers, static plus
//!   per-token dynamic taps on global layers).
//! - Residuals are softmax mixes over blocks of `n_layers_per_attn_res_block`
//!   layers (F32 queries); each layer also runs `attn_sconv` and `mlp_sconv`.
//! - The MoE routes to `num_experts_per_tok` of `n_routed_experts` (sigmoid
//!   plus a selection bias, log-sigmoid softmax weights times `route_scale`),
//!   no shared expert; experts run SiTU (`mlp_gate_softcap`, `mlp_up_softcap`).
//! - Logits are trimmed to `unpadded_vocab_size` and soft-capped.
use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;
use std::path::Path;

use crate::plan::checkpoint::read_json;

/// The text model.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2Config {
    /// Stored vocabulary rows (embedding and unembedding).
    pub vocab_size: usize,
    /// Logit rows the head emits (the rest of the stored rows are padding).
    pub unpadded_vocab_size: usize,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Relative-position projection width per head (`wr_du`).
    pub d_rel: usize,
    /// Relative-bias extent of global layers; local layers use the window.
    pub rel_extent: usize,
    /// `local[layer]`: sliding-window attention of `window` positions.
    pub local: Vec<bool>,
    pub window: usize,
    pub sconv_kernel: usize,
    pub rms_norm_eps: f64,
    pub qk_l2_clip: f64,
    pub rel_l2_clip: f64,
    /// Global-layer q/r log scaling: `1 + alpha * ln(max(1, (pos + 1) / floor))`.
    pub log_scaling: Option<(usize, f64)>,
    pub res_block: usize,
    pub experts: usize,
    pub topk: usize,
    pub intermediate: usize,
    pub gate_softcap: f64,
    pub up_softcap: f64,
    pub route_scale: f64,
    pub final_softcap: f64,
    pub max_position: usize,
    pub eos: Vec<u32>,
}

/// The bundled vision tower (`vision_config`): a pre-LN ViT with layer scale
/// and a 2x2 merge projector into the text width.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2VisionConfig {
    pub hidden: usize,
    pub heads: usize,
    pub layers: usize,
    pub intermediate: usize,
    pub patch: usize,
    pub temporal_patch: usize,
    pub merge: usize,
    pub posemb_grid: usize,
    pub projector_hidden: usize,
    pub projector_bias: bool,
    pub layer_norm_eps: f64,
}

/// The bundled dSpark drafter (`mtp_config`, weights in `mtp.safetensors`).
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2MtpConfig {
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub local: Vec<bool>,
    pub window: usize,
    /// Anchor plus `block - 1` drafted positions.
    pub block: usize,
    pub markov_rank: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    /// Target layers whose AttnRes partial-block input the drafter mixes; the
    /// last id equals the target layer count and names the final-norm output.
    pub target_layers: Vec<usize>,
    pub max_position: usize,
}

fn int(value: &Value, key: &str) -> Result<usize> {
    let n = value[key].as_u64().with_context(|| format!("Inkling 2 config lacks {key}"))?;
    ensure!(n > 0 && n <= 1 << 24, "invalid Inkling 2 {key}: {n}");
    Ok(n as usize)
}

fn float(value: &Value, key: &str) -> Result<f64> {
    let x = value[key].as_f64().filter(|x| x.is_finite()).with_context(|| format!("Inkling 2 config lacks {key}"))?;
    ensure!(x > 0.0, "Inkling 2 {key} must be positive, found {x}");
    Ok(x)
}

fn boolean(value: &Value, key: &str, expected: bool) -> Result<()> {
    ensure!(value[key].as_bool() == Some(expected), "Inkling 2 {key} must be {expected}, found {}", value[key]);
    Ok(())
}

fn layer_flags(value: &Value, key: &str, layers: usize) -> Result<Vec<bool>> {
    let ids = value[key].as_array().with_context(|| format!("Inkling 2 config lacks {key}"))?;
    let mut flags = vec![false; layers];
    for id in ids {
        let id = id.as_u64().filter(|id| (*id as usize) < layers)
            .with_context(|| format!("Inkling 2 {key} entry {id} is outside {layers} layers"))? as usize;
        ensure!(!flags[id], "Inkling 2 {key} repeats layer {id}");
        flags[id] = true;
    }
    Ok(flags)
}

fn token_ids(value: &Value, vocab: usize) -> Result<Vec<u32>> {
    let one = |id: &Value| -> Result<u32> {
        let id = id.as_u64().context("Inkling 2 eos_token_id must be a token id")?;
        ensure!(id < vocab as u64, "Inkling 2 eos_token_id {id} is outside vocabulary {vocab}");
        Ok(u32::try_from(id)?)
    };
    match value {
        Value::Array(ids) if !ids.is_empty() => ids.iter().map(one).collect(),
        Value::Array(_) => bail!("Inkling 2 eos_token_id must not be empty"),
        other => Ok(vec![one(other)?]),
    }
}

impl Inkling2Config {
    pub fn read(snapshot: &Path) -> Result<Self> {
        Self::from_hf(&read_json(&snapshot.join("config.json"))?)
    }

    pub fn from_hf(root: &Value) -> Result<Self> {
        ensure!(root["model_type"] == "inkling2", "Inkling 2 requires model_type=inkling2, found {}", root["model_type"]);
        let v = root.get("text_config").filter(|v| v.is_object()).context("Inkling 2 requires text_config")?;
        let layers = int(v, "num_hidden_layers")?;
        let (heads, kv_heads, head_dim) = (int(v, "num_attention_heads")?, int(v, "num_key_value_heads")?, int(v, "head_dim")?);
        ensure!(heads % kv_heads == 0, "Inkling 2 heads {heads} are not a multiple of KV heads {kv_heads}");
        // One attention geometry on every layer: the executor has no per-layer head shape.
        for (key, expected) in [("swa_num_attention_heads", heads), ("swa_num_key_value_heads", kv_heads), ("swa_head_dim", head_dim)] {
            let got = int(v, key)?;
            ensure!(got == expected, "Inkling 2 {key} {got} differs from the global attention geometry {expected}");
        }
        for key in ["q_bias", "o_bias", "pre_norm_pre_affine", "final_norm_pre_affine"] {
            ensure!(v.get(key).is_none_or(|value| value.as_bool() == Some(false)), "Inkling 2 {key} must be false or absent");
        }
        boolean(v, "use_value_norm_no_gamma", true)?;
        boolean(v, "use_gate_bias", true)?;
        boolean(v, "attn_res_normalize_values", true)?;
        ensure!(v["n_shared_experts"].as_u64() == Some(0), "Inkling 2 has no shared expert, found n_shared_experts={}", v["n_shared_experts"]);
        let hidden = int(v, "hidden_size")?;
        let intermediate = int(v, "intermediate_size")?;
        ensure!(hidden % 32 == 0 && intermediate % 32 == 0, "Inkling 2 hidden {hidden} and intermediate {intermediate} must be multiples of the MXFP8 group");
        let experts = int(v, "n_routed_experts")?;
        let topk = int(v, "num_experts_per_tok")?;
        ensure!(topk <= experts, "Inkling 2 top-k {topk} exceeds experts {experts}");
        let vocab_size = int(v, "vocab_size")?;
        let unpadded_vocab_size = int(v, "unpadded_vocab_size")?;
        ensure!(unpadded_vocab_size <= vocab_size, "Inkling 2 unpadded vocabulary {unpadded_vocab_size} exceeds stored {vocab_size}");
        let res_block = int(v, "n_layers_per_attn_res_block")?;
        ensure!(res_block <= layers, "Inkling 2 residual block {res_block} exceeds {layers} layers");
        let window = int(v, "sliding_window_size")?;
        let rel_extent = int(v, "rel_extent")?;
        let local = layer_flags(v, "local_layer_ids", layers)?;
        let log_scaling = match v.get("log_scaling_n_floor").filter(|f| !f.is_null()) {
            None => None,
            Some(_) => {
                let alpha = v["log_scaling_alpha"].as_f64().filter(|a| a.is_finite() && *a >= 0.0)
                    .context("Inkling 2 log_scaling_alpha must be a finite nonnegative number")?;
                Some((int(v, "log_scaling_n_floor")?, alpha))
            }
        };
        ensure!(v.get("tie_word_embeddings").is_none_or(|value| value.as_bool() == Some(false)),
            "Inkling 2 tie_word_embeddings must be false or absent");
        let eos = token_ids(root.get("eos_token_id").or_else(|| v.get("eos_token_id")).unwrap_or(&Value::from(200_006)), vocab_size)?;
        Ok(Self {
            vocab_size,
            unpadded_vocab_size,
            hidden,
            layers,
            heads,
            kv_heads,
            head_dim,
            d_rel: int(v, "d_rel")?,
            rel_extent,
            local,
            window,
            sconv_kernel: int(v, "sconv_kernel_size")?,
            rms_norm_eps: float(v, "rms_norm_eps")?,
            qk_l2_clip: float(v, "qk_l2_clip_threshold")?,
            rel_l2_clip: float(v, "rel_l2_clip_threshold")?,
            log_scaling,
            res_block,
            experts,
            topk,
            intermediate,
            gate_softcap: float(v, "mlp_gate_softcap")?,
            up_softcap: float(v, "mlp_up_softcap")?,
            route_scale: float(v, "route_scale")?,
            final_softcap: float(v, "final_logit_softcapping")?,
            max_position: int(v, "model_max_length")?,
            eos,
        })
    }

    /// The geometry this build's kernels are written for (Inkling-2-Small).
    pub fn release_geometry(&self) -> Result<()> {
        let attention = (self.hidden, self.layers, self.heads, self.kv_heads, self.head_dim, self.d_rel, self.rel_extent,
            self.window, self.sconv_kernel, self.res_block);
        ensure!(attention == (4608, 48, 48, 8, 128, 16, 1024, 1024, 4, 6),
            "Inkling 2 attention geometry (hidden, layers, heads, kv heads, head dim, d_rel, rel extent, window, conv \
             taps, residual block) {attention:?} is not the Inkling-2-Small release this build serves");
        let moe = (self.experts, self.topk, self.intermediate, self.vocab_size, self.unpadded_vocab_size);
        ensure!(moe == (384, 8, 2304, 201024, 200058),
            "Inkling 2 MoE/head geometry (experts, top-k, intermediate, vocab, unpadded vocab) {moe:?} is not the \
             Inkling-2-Small release this build serves");
        ensure!(self.global_layers() == (5..self.layers).step_by(6).collect::<Vec<_>>(),
            "Inkling-2-Small attends globally on layers 5, 11, ..., 47");
        Ok(())
    }

    pub fn activation(&self) -> cuteafd_core::FfnActivation {
        cuteafd_core::FfnActivation::Situ { beta: self.gate_softcap as f32, linear_beta: Some(self.up_softcap as f32) }
    }

    /// Layers attending the whole context, ascending.
    pub fn global_layers(&self) -> Vec<usize> {
        (0..self.layers).filter(|&layer| !self.local[layer]).collect()
    }

    pub fn local_layers(&self) -> usize { self.local.iter().filter(|l| **l).count() }

    /// KV heads' query group width.
    pub fn group(&self) -> usize { self.heads / self.kv_heads }

    /// Completed six-layer residual blocks the output mix reads (the partial
    /// block is a separate operand).
    pub fn residual_blocks(&self) -> usize { self.layers / self.res_block }
}

impl Inkling2VisionConfig {
    pub fn from_hf(root: &Value) -> Result<Option<Self>> {
        let Some(v) = root.get("vision_config") else { return Ok(None) };
        ensure!(v.is_object(), "Inkling 2 vision_config must be an object");
        let hidden = int(v, "hidden_size")?;
        let heads = int(v, "num_attention_heads")?;
        ensure!(hidden % heads == 0, "vision hidden {hidden} is not a multiple of heads {heads}");
        let ratio = v["mlp_ratio"].as_f64().filter(|r| r.is_finite() && *r > 0.0).context("Inkling 2 vision mlp_ratio")?;
        let intermediate = (hidden as f64 * ratio) as usize;
        ensure!((intermediate as f64 - hidden as f64 * ratio).abs() < 1e-9, "Inkling 2 vision mlp_ratio {ratio} is not a whole width");
        Ok(Some(Self {
            hidden, heads, intermediate,
            layers: int(v, "num_hidden_layers")?,
            patch: int(v, "native_patch_size")?,
            temporal_patch: int(v, "temporal_patch_size")?,
            merge: int(v, "adapter_merge_size")?,
            posemb_grid: int(v, "posemb_grid_size")?,
            projector_hidden: int(v, "projection_hidden_dim")?,
            projector_bias: v["projector_bias"].as_bool().context("Inkling 2 vision projector_bias")?,
            layer_norm_eps: float(v, "layer_norm_eps")?,
        }))
    }
}

impl Inkling2MtpConfig {
    pub fn from_hf(root: &Value, text: &Inkling2Config) -> Result<Option<Self>> {
        let Some(v) = root.get("mtp_config") else { return Ok(None) };
        ensure!(v.is_object(), "Inkling 2 mtp_config must be an object");
        ensure!(v["markov_head_type"] == "rnn_hidden", "Inkling 2 MTP head must be rnn_hidden, found {}", v["markov_head_type"]);
        for key in ["q_bias", "o_bias"] {
            ensure!(v.get(key).is_none_or(|value| value.as_bool() == Some(false)), "Inkling 2 MTP {key} must be false or absent");
        }
        let layers = int(v, "num_hidden_layers")?;
        let hidden = int(v, "hidden_size")?;
        ensure!(hidden == text.hidden, "Inkling 2 MTP hidden {hidden} differs from the target's {}", text.hidden);
        let targets = v["target_layer_ids"].as_array().context("Inkling 2 mtp_config lacks target_layer_ids")?
            .iter().map(|id| id.as_u64().map(|id| id as usize).context("Inkling 2 MTP target layer id")).collect::<Result<Vec<_>>>()?;
        ensure!(targets.len() >= 2 && targets.windows(2).all(|w| w[0] < w[1]) && targets.last() == Some(&text.layers),
            "Inkling 2 MTP target_layer_ids must be ascending and end at the target layer count ({})", text.layers);
        let block = int(v, "block_size")?;
        ensure!(block >= 2, "Inkling 2 MTP block {block} drafts nothing");
        let (heads, kv_heads) = (int(v, "num_attention_heads")?, int(v, "num_key_value_heads")?);
        ensure!(heads % kv_heads == 0, "Inkling 2 MTP heads {heads} are not a multiple of KV heads {kv_heads}");
        Ok(Some(Self {
            layers, hidden, heads, kv_heads,
            head_dim: int(v, "head_dim")?,
            intermediate: int(v, "intermediate_size")?,
            local: layer_flags(v, "local_layer_ids", layers)?,
            window: int(v, "sliding_window_size")?,
            block,
            markov_rank: int(v, "markov_rank")?,
            rope_theta: float(v, "rope_theta")?,
            rms_norm_eps: float(v, "rms_norm_eps")?,
            target_layers: targets,
            max_position: int(v, "max_position_embeddings")?,
        }))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn release_config() -> Value { serde_json::from_str(include_str!("fixtures/config.json")).unwrap() }
    pub(crate) fn release_quant_config() -> Value { serde_json::from_str(include_str!("fixtures/hf_quant_config.json")).unwrap() }

    #[test]
    fn release_text_config() {
        let cfg = Inkling2Config::from_hf(&release_config()).unwrap();
        cfg.release_geometry().unwrap();
        assert_eq!(cfg.global_layers(), vec![5, 11, 17, 23, 29, 35, 41, 47]);
        assert_eq!(cfg.local_layers(), 40);
        assert_eq!(cfg.log_scaling, Some((128_000, 0.1)));
        assert_eq!(cfg.residual_blocks(), 8);
        assert_eq!(cfg.eos, vec![200_006]);
        assert_eq!(cfg.activation(), cuteafd_core::FfnActivation::Situ { beta: 4.0, linear_beta: Some(25.0) });
        assert_eq!(cfg.final_softcap, 30.0);
    }

    #[test]
    fn release_vision_and_drafter() {
        let root = release_config();
        let text = Inkling2Config::from_hf(&root).unwrap();
        let vision = Inkling2VisionConfig::from_hf(&root).unwrap().unwrap();
        assert_eq!((vision.hidden, vision.heads, vision.layers, vision.intermediate, vision.patch, vision.merge),
            (1024, 16, 23, 4096, 20, 2));
        assert_eq!((vision.projector_hidden, vision.projector_bias, vision.posemb_grid), (4096, false, 32));
        let mtp = Inkling2MtpConfig::from_hf(&root, &text).unwrap().unwrap();
        assert_eq!((mtp.layers, mtp.block, mtp.markov_rank, mtp.intermediate, mtp.window), (5, 8, 256, 18432, 1024));
        assert_eq!(mtp.target_layers, vec![5, 11, 17, 23, 29, 35, 41, 48]);
        assert_eq!(mtp.local, vec![true, true, false, true, true]);
    }

    #[test]
    fn rejects_drifted_configs() {
        let mut root = release_config();
        root["text_config"]["n_shared_experts"] = 1.into();
        assert!(Inkling2Config::from_hf(&root).unwrap_err().to_string().contains("shared expert"));
        let mut root = release_config();
        root["text_config"]["swa_head_dim"] = 64.into();
        assert!(Inkling2Config::from_hf(&root).unwrap_err().to_string().contains("swa_head_dim"));
        let mut root = release_config();
        root["text_config"]["local_layer_ids"].as_array_mut().unwrap().push(3.into());
        assert!(Inkling2Config::from_hf(&root).unwrap_err().to_string().contains("repeats"));
        let mut root = release_config();
        root["text_config"]["num_hidden_layers"] = 46.into();
        assert!(Inkling2Config::from_hf(&root).unwrap_err().to_string().contains("outside 46 layers"),
            "local layer 46 is outside 46 layers");
        let mut root = release_config();
        root["mtp_config"]["markov_head_type"] = "rnn".into();
        let text = Inkling2Config::from_hf(&root).unwrap();
        assert!(Inkling2MtpConfig::from_hf(&root, &text).unwrap_err().to_string().contains("rnn_hidden"));
        let mut root = release_config();
        root["mtp_config"]["target_layer_ids"] = serde_json::json!([5, 11]);
        assert!(Inkling2MtpConfig::from_hf(&root, &text).unwrap_err().to_string().contains("target_layer_ids"));
    }

    #[test]
    fn quant_config_fixture_keeps_each_release_group() {
        let q = release_quant_config();
        assert_eq!(q["quant_method"], "inkling2");
        let groups = q["model"]["parameter_groups"].as_array().unwrap();
        assert_eq!(groups.iter().map(|g| g["compute_dtype"].as_str().unwrap()).collect::<Vec<_>>(),
            ["float32", "bfloat16", "mxfp8", "bfloat16"]);
    }
}
