//! Muse Spark text model arguments (`text_config` of
//! `MuseSparkForConditionalGeneration`, `model_type=muse_spark_text`), the one
//! reader shared by the planner, the runtime and the golden.
//!
//! Semantics follow the official `transformers` `models/muse_spark` code
//! (config transformers 5.18.0.dev0):
//! - Every layer's FFN is a MoE (`mlp_layer_types` all `sparse`) with no shared
//!   expert. The MoE projects the residual down to `moe_hidden_size` with
//!   `pre_expert_proj`, applies a centered RMSNorm (`pre_expert_norm`), runs the
//!   routed experts at that width, applies the input-scaled `post_expert_norm`
//!   to *each expert's* output before the routing weight, sums, and projects
//!   back up with `post_expert_proj`.
//! - Router: FP32 `gate.weight` logits times `output_multiplier`, sigmoid; the
//!   `e_score_correction_bias` only selects; weights renormalize with 1e-15.
//! - Attention: GQA with weightless per-head q/k RMSNorm (q times
//!   `qk_scale_factor`), scale `head_dim^-0.5`, weightless per-head output
//!   RMSNorm, then a sigmoid `gate_proj` output gate before `o_proj`.
//!   Full-attention layers are NoPE (`layer_rope_theta == 0`).
//! - Residual gates replace the residual adds (`residual_gate_temperature`).
//! - Head: `softcap * tanh(logits * output_multiplier / softcap)`.
use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;
use std::path::Path;

use crate::plan::checkpoint::read_json;

/// A layer's attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MuseAttention {
    /// Causal sliding window of `sliding_window` keys, RoPE.
    Sliding,
    /// Full causal attention without RoPE.
    Full,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MuseSparkConfig {
    pub vocab_size: usize,
    pub hidden: usize,
    pub layers: usize,
    pub attention: Vec<MuseAttention>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub window: usize,
    pub rope_theta: f64,
    pub qk_scale: f64,
    pub experts: usize,
    pub topk: usize,
    /// Expert input/output width (`moe_hidden_size`), after `pre_expert_proj`.
    pub expert_hidden: usize,
    pub moe_intermediate: usize,
    pub routed_scale: f64,
    pub output_multiplier: f64,
    pub softcap: f64,
    pub residual_gate_temperature: f64,
    pub rms_norm_eps: f64,
    pub post_norm_eps: f64,
    pub max_position: usize,
    pub eos: Vec<u32>,
    pub bos: u32,
    pub pad: u32,
}

impl MuseSparkConfig {
    pub fn read(snapshot: &Path) -> Result<Self> {
        Self::from_hf(&read_json(&snapshot.join("config.json"))?)
    }

    /// Recognizes the checkpoint by its own config (top-level or nested text).
    pub fn is_muse_spark(root: &Value) -> bool {
        root["model_type"] == "muse_spark"
            || root["text_config"]["model_type"] == "muse_spark_text"
            || root["architectures"].as_array().is_some_and(|names| names.iter()
                .any(|name| name == "MuseSparkForConditionalGeneration"))
    }

    pub fn from_hf(root: &Value) -> Result<Self> {
        let v = root.get("text_config").filter(|v| v.is_object()).unwrap_or(root);
        ensure!(v["model_type"] == "muse_spark_text",
            "Muse Spark requires text_config.model_type=muse_spark_text, found {}", v["model_type"]);
        let int = |key: &str| -> Result<usize> {
            let n = v[key].as_u64().with_context(|| format!("Muse Spark config lacks {key}"))?;
            ensure!(n > 0 && n <= 1 << 24, "invalid Muse Spark {key}: {n}");
            Ok(n as usize)
        };
        let float = |key: &str| -> Result<f64> {
            v[key].as_f64().filter(|x| x.is_finite()).with_context(|| format!("Muse Spark config lacks {key}"))
        };
        let layers = int("num_hidden_layers")?;
        let list = |key: &str| -> Result<&Vec<Value>> {
            let list = v[key].as_array().with_context(|| format!("Muse Spark config lacks {key}"))?;
            ensure!(list.len() == layers, "Muse Spark {key} has {} entries for {layers} layers", list.len());
            Ok(list)
        };
        let attention = list("layer_types")?.iter().enumerate().map(|(layer, kind)| match kind.as_str() {
            Some("sliding_attention") => Ok(MuseAttention::Sliding),
            Some("full_attention") => Ok(MuseAttention::Full),
            _ => bail!("layer_types[{layer}] {kind} is neither sliding_attention nor full_attention"),
        }).collect::<Result<Vec<_>>>()?;
        ensure!(list("mlp_layer_types")?.iter().all(|kind| kind == "sparse"),
            "Muse Spark dense MLP layers are not implemented: every mlp_layer_types entry must be sparse");
        let rope_theta = v["rope_parameters"]["rope_theta"].as_f64().filter(|x| x.is_finite() && *x > 0.0)
            .context("Muse Spark config lacks a positive rope_parameters.rope_theta")?;
        ensure!(v["rope_parameters"]["rope_type"].as_str().unwrap_or("default") == "default",
            "Muse Spark rope_type {} is not implemented (default RoPE only)", v["rope_parameters"]["rope_type"]);
        // NoPE exactly on the full layers; sliding layers rotate at the one base.
        for (layer, theta) in list("layer_rope_theta")?.iter().enumerate() {
            let theta = theta.as_f64().with_context(|| format!("layer_rope_theta[{layer}] is not a number"))?;
            match attention[layer] {
                MuseAttention::Full => ensure!(theta == 0.0,
                    "layer {layer} is full attention with RoPE theta {theta}; full layers must be NoPE (0)"),
                MuseAttention::Sliding => ensure!(theta == rope_theta,
                    "layer {layer} sliding RoPE theta {theta} differs from rope_parameters.rope_theta {rope_theta}"),
            }
        }
        ensure!(v["num_shared_experts"].as_u64().unwrap_or(0) == 0, "Muse Spark has no shared expert");
        ensure!(v["hidden_act"] == "silu", "Muse Spark experts must be SiLU-gated, found {}", v["hidden_act"]);
        ensure!(v["attention_bias"].as_bool() != Some(true), "Muse Spark attention has no biases");
        let token = |key: &str| v[key].as_u64().map(|id| id as u32).with_context(|| format!("Muse Spark config lacks {key}"));
        let eos = match &v["eos_token_id"] {
            Value::Array(ids) => ids.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect(),
            Value::Number(id) => id.as_u64().map(|x| x as u32).into_iter().collect(),
            _ => Vec::new(),
        };
        let cfg = Self {
            vocab_size: int("vocab_size")?,
            hidden: int("hidden_size")?,
            layers,
            attention,
            heads: int("num_attention_heads")?,
            kv_heads: int("num_key_value_heads")?,
            head_dim: int("head_dim")?,
            window: int("sliding_window")?,
            rope_theta,
            qk_scale: float("qk_scale_factor")?,
            experts: int("num_local_experts")?,
            topk: int("num_experts_per_tok")?,
            expert_hidden: int("moe_hidden_size")?,
            moe_intermediate: int("moe_intermediate_size")?,
            routed_scale: float("routed_scaling_factor")?,
            output_multiplier: float("output_multiplier")?,
            softcap: float("final_logit_softcapping")?,
            residual_gate_temperature: float("residual_gate_temperature")?,
            rms_norm_eps: float("rms_norm_eps")?,
            post_norm_eps: float("post_norm_eps")?,
            max_position: int("max_position_embeddings")?,
            eos,
            bos: token("bos_token_id")?,
            pad: token("pad_token_id")?,
        };
        ensure!(cfg.heads % cfg.kv_heads == 0, "Muse Spark query heads {} are not a multiple of KV heads {}",
            cfg.heads, cfg.kv_heads);
        ensure!(cfg.head_dim % 2 == 0, "Muse Spark RoPE needs an even head_dim");
        ensure!(cfg.topk <= cfg.experts, "Muse Spark top-k exceeds experts");
        ensure!(cfg.softcap > 0.0 && cfg.residual_gate_temperature > 0.0 && cfg.rms_norm_eps > 0.0
            && cfg.post_norm_eps > 0.0, "Muse Spark softcap, gate temperature and norm epsilons must be positive");
        if v["tie_word_embeddings"].as_bool() == Some(true) || root["tie_word_embeddings"].as_bool() == Some(true) {
            bail!("Muse Spark is untied (tie_word_embeddings=false)");
        }
        Ok(cfg)
    }

    /// The release's executable geometry (816B-A42B). Others parse but are not served.
    pub fn release_geometry(&self) -> Result<()> {
        let expected = [
            ("hidden_size", self.hidden, 8192), ("num_hidden_layers", self.layers, 62),
            ("vocab_size", self.vocab_size, 202_048), ("num_attention_heads", self.heads, 128),
            ("num_key_value_heads", self.kv_heads, 16), ("head_dim", self.head_dim, 64),
            ("sliding_window", self.window, 2048), ("num_local_experts", self.experts, 256),
            ("num_experts_per_tok", self.topk, 8), ("moe_hidden_size", self.expert_hidden, 4096),
            ("moe_intermediate_size", self.moe_intermediate, 4096),
            ("max_position_embeddings", self.max_position, 1 << 20),
        ];
        for (key, actual, wanted) in expected {
            ensure!(actual == wanted, "unsupported Muse Spark {key}: {actual}, the release geometry is {wanted}");
        }
        // Full attention every fourth layer counted back from the last (1, 5, ..., 61).
        ensure!(self.attention.iter().enumerate().all(|(layer, kind)|
            (*kind == MuseAttention::Full) == ((self.layers - 1 - layer) % 4 == 0)),
            "unsupported Muse Spark layer_types: full attention must be every fourth layer from the last");
        ensure!(self.routed_scale == 1.0, "unsupported routed_scaling_factor {}", self.routed_scale);
        Ok(())
    }

    pub fn full_layers(&self) -> usize { self.attention.iter().filter(|a| **a == MuseAttention::Full).count() }
    pub fn sliding_layers(&self) -> usize { self.layers - self.full_layers() }

    /// Routed expert kernel shape: experts see the `moe_hidden_size` stream.
    pub fn expert_geometry(&self) -> cuteafd_core::ExpertGeometry {
        cuteafd_core::ExpertGeometry { hidden: self.expert_hidden as u32, experts: self.experts as u32,
            topk: self.topk as u32, intermediate: self.moe_intermediate as u32, layers: self.layers as u32 }
    }

    /// Residual gate coefficients for one gate parameter `g` (FP32, as the reference).
    pub fn residual_gate(&self, g: f32) -> (f32, f32) {
        let w = g / self.residual_gate_temperature as f32;
        let beta = 1.0 / (1.0 + (-w).exp());
        let alpha = ((1.0 / (1.0 + w.exp())) * (1.0 + beta)).max(1e-3).sqrt();
        (alpha, beta)
    }

    /// BF16 K+V bytes of one token in one layer.
    pub fn kv_record_bytes(&self) -> usize { 2 * self.kv_heads * self.head_dim * 2 }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

    pub(crate) fn release_config() -> Value { crate::plan::testing::muse_spark_config() }

    #[test]
    fn release_config_parses_with_nope_full_layers() {
        let cfg = MuseSparkConfig::from_hf(&release_config()).unwrap();
        cfg.release_geometry().unwrap();
        assert_eq!((cfg.full_layers(), cfg.sliding_layers()), (16, 46));
        let full: Vec<_> = (0..cfg.layers).filter(|&l| cfg.attention[l] == MuseAttention::Full).collect();
        assert_eq!(&full[..3], &[1, 5, 9]);
        assert_eq!(full.last(), Some(&61));
        assert_eq!(cfg.expert_geometry(), cuteafd_core::ExpertGeometry::MUSE_SPARK);
        // 16 full layers x 16 KV heads x 64 x (K+V) BF16.
        assert_eq!(cfg.kv_record_bytes() * cfg.full_layers(), 64 << 10);
        assert!(MuseSparkConfig::is_muse_spark(&release_config()));
    }

    #[test]
    fn residual_gate_matches_the_reference_formula() {
        let cfg = MuseSparkConfig::from_hf(&release_config()).unwrap();
        let (alpha, beta) = cfg.residual_gate(0.0);
        assert_eq!(beta, 0.5);
        assert!((alpha - (0.5f32 * 1.5).sqrt()).abs() < 1e-7);
        // Saturated gate: alpha clamps at sqrt(1e-3).
        let (alpha, beta) = cfg.residual_gate(30.0);
        assert!((beta - 1.0).abs() < 1e-6 && (alpha - 1e-3f32.sqrt()).abs() < 1e-6, "{alpha} {beta}");
    }

    #[test]
    fn semantics_the_runtime_does_not_implement_are_refused() {
        let mut layer_rope = release_config();
        layer_rope["text_config"]["layer_rope_theta"][1] = json!(500000.0);
        let mut dense = release_config();
        dense["text_config"]["mlp_layer_types"][3] = json!("dense");
        let mut shared = release_config();
        shared["text_config"]["num_shared_experts"] = json!(1);
        let mut act = release_config();
        act["text_config"]["hidden_act"] = json!("gelu");
        let mut tied = release_config();
        tied["text_config"]["tie_word_embeddings"] = json!(true);
        let mut rope = release_config();
        rope["text_config"]["rope_parameters"]["rope_type"] = json!("yarn");
        for (config, needle) in [(layer_rope, "NoPE"), (dense, "dense MLP"), (shared, "shared expert"),
            (act, "SiLU"), (tied, "untied"), (rope, "rope_type")] {
            let error = MuseSparkConfig::from_hf(&config).unwrap_err().to_string();
            assert!(error.contains(needle), "{needle}: {error}");
        }
        let mut config = release_config();
        config["text_config"]["num_local_experts"] = json!(128);
        let cfg = MuseSparkConfig::from_hf(&config).unwrap();
        assert!(cfg.release_geometry().unwrap_err().to_string().contains("num_local_experts: 128"));
    }
}
