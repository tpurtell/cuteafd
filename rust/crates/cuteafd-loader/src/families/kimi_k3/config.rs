//! Kimi K3 text model arguments from `config.json` (`text_config` of
//! `KimiK3ForConditionalGeneration`), the reader shared by the planner and runtime.
//!
//! Semantics follow the checkpoint's bundled `modeling_kimi_linear.py`:
//! - `linear_attn_config.{kda_layers,full_attn_layers}` are 1-based.
//! - Full-attention layers are MLA without RoPE (`mla_use_nope`): the 64-wide
//!   `qk_rope` slot is concatenated unrotated, with a sigmoid output gate.
//! - Attention residuals (`attn_res_block_size`) replace the residual add.
//! - Routed experts see `routed_expert_down_proj` output (`routed_expert_hidden_size`),
//!   are combined, normalized (`latent_moe_use_norm`) and projected back up.
use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;
use std::path::Path;

use crate::plan::checkpoint::read_json;

/// A layer's attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KimiAttention {
    /// Kimi Delta Attention (linear, recurrent state).
    Kda,
    /// Full causal MLA without RoPE, sigmoid output gate.
    Mla,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KimiK3Config {
    pub vocab_size: usize,
    pub hidden: usize,
    pub layers: usize,
    pub attention: Vec<KimiAttention>,
    /// `dense[layer]`: a Situ MLP instead of the MoE.
    pub dense: Vec<bool>,
    pub dense_intermediate: usize,
    pub experts: usize,
    pub topk: usize,
    pub moe_intermediate: usize,
    /// Expert input/output width (`routed_expert_hidden_size`).
    pub latent: usize,
    pub shared_experts: usize,
    pub routed_scale: f64,
    pub situ_beta: f64,
    pub situ_linear_beta: f64,
    pub rms_norm_eps: f64,
    /// Attention-residual block size: a block boundary every this many layers.
    pub residual_block: usize,
    pub kda_heads: usize,
    pub kda_head_dim: usize,
    pub kda_conv: usize,
    pub kda_gate_lower_bound: f64,
    pub heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_dim: usize,
    /// Unrotated: the reference never applies RoPE.
    pub qk_rope_dim: usize,
    pub v_head_dim: usize,
    pub max_position: usize,
    pub tie_embeddings: bool,
    pub eos: Vec<u32>,
}

impl KimiK3Config {
    pub fn read(snapshot: &Path) -> Result<Self> {
        Self::from_hf(&read_json(&snapshot.join("config.json"))?)
    }

    pub fn from_hf(root: &Value) -> Result<Self> {
        let v = root.get("text_config").filter(|v| v.is_object()).unwrap_or(root);
        ensure!(v["model_type"] == "kimi_linear", "Kimi K3 requires text_config.model_type=kimi_linear, found {}", v["model_type"]);
        let int = |value: &Value, key: &str| -> Result<usize> {
            let n = value[key].as_u64().with_context(|| format!("Kimi K3 config lacks {key}"))?;
            ensure!(n > 0 && n <= 1 << 24, "invalid Kimi K3 {key}: {n}");
            Ok(n as usize)
        };
        let float = |value: &Value, key: &str| -> Result<f64> {
            value[key].as_f64().filter(|x| x.is_finite()).with_context(|| format!("Kimi K3 config lacks {key}"))
        };
        let flag = |key: &str| v[key].as_bool().unwrap_or(false);
        let layers = int(v, "num_hidden_layers")?;
        let linear = &v["linear_attn_config"];
        ensure!(linear.is_object(), "Kimi K3 requires linear_attn_config");
        let one_based = |key: &str| -> Result<Vec<usize>> {
            linear[key].as_array().with_context(|| format!("linear_attn_config lacks {key}"))?.iter().map(|x| {
                let layer = x.as_u64().with_context(|| format!("invalid {key} entry {x}"))? as usize;
                ensure!((1..=layers).contains(&layer), "{key} entry {layer} outside 1..={layers} (the list is 1-based)");
                Ok(layer - 1)
            }).collect()
        };
        let mut attention: Vec<Option<KimiAttention>> = vec![None; layers];
        for (key, kind) in [("kda_layers", KimiAttention::Kda), ("full_attn_layers", KimiAttention::Mla)] {
            for layer in one_based(key)? {
                ensure!(attention[layer].is_none(), "layer {} (1-based) is listed twice in linear_attn_config", layer + 1);
                attention[layer] = Some(kind);
            }
        }
        let attention = attention.into_iter().enumerate()
            .map(|(layer, kind)| kind.with_context(|| format!("layer {} (1-based) has no attention kind", layer + 1)))
            .collect::<Result<Vec<_>>>()?;

        let first_k = v["first_k_dense_replace"].as_u64().unwrap_or(0) as usize;
        let freq = v["moe_layer_freq"].as_u64().unwrap_or(1) as usize;
        ensure!(freq >= 1, "moe_layer_freq must be positive");
        let dense: Vec<bool> = (0..layers).map(|l| !(l >= first_k && l % freq == 0)).collect();

        ensure!(v["hidden_act"] == "situ", "Kimi K3 FFNs must be situ, found {}", v["hidden_act"]);
        ensure!(flag("mla_use_nope"), "Kimi K3 MLA must be NoPE (mla_use_nope)");
        ensure!(flag("mla_use_output_gate"), "Kimi K3 MLA requires the sigmoid output gate (mla_use_output_gate)");
        ensure!(linear["use_full_rank_gate"].as_bool() == Some(true), "Kimi K3 KDA requires the full-rank output gate");
        ensure!(v["moe_router_activation_func"] == "sigmoid" && v["moe_renormalize"].as_bool() == Some(true),
            "Kimi K3 routing must be renormalized sigmoid");
        ensure!(v["topk_method"].as_str().unwrap_or("noaux_tc") == "noaux_tc", "Kimi K3 routing must be noaux_tc");
        ensure!(v["num_expert_group"].as_u64().unwrap_or(1) == 1 && v["topk_group"].as_u64().unwrap_or(1) == 1,
            "Kimi K3 routing has one expert group");
        ensure!(flag("latent_moe_use_norm"), "Kimi K3 latent MoE requires routed_expert_norm (latent_moe_use_norm)");
        ensure!(v["num_nextn_predict_layers"].as_u64().unwrap_or(0) == 0, "Kimi K3 ships no native MTP layers");

        let kda_heads = int(linear, "num_heads")?;
        let kda_head_dim = int(linear, "head_dim")?;
        let heads = int(v, "num_attention_heads")?;
        ensure!(int(v, "num_key_value_heads")? == heads, "Kimi K3 MLA has one KV head per query head");
        let eos = match &v["eos_token_id"] {
            Value::Array(ids) => ids.iter().filter_map(|x| x.as_u64().map(|x| x as u32)).collect(),
            Value::Number(id) => id.as_u64().map(|x| x as u32).into_iter().collect(),
            _ => Vec::new(),
        };
        let cfg = Self {
            vocab_size: int(v, "vocab_size")?,
            hidden: int(v, "hidden_size")?,
            layers,
            attention,
            dense,
            dense_intermediate: int(v, "intermediate_size")?,
            experts: int(v, "num_experts")?,
            topk: int(v, "num_experts_per_token")?,
            moe_intermediate: int(v, "moe_intermediate_size")?,
            latent: int(v, "routed_expert_hidden_size")?,
            shared_experts: v["num_shared_experts"].as_u64().unwrap_or(0) as usize,
            routed_scale: float(v, "routed_scaling_factor")?,
            situ_beta: float(v, "activation_situ_beta")?,
            situ_linear_beta: float(v, "activation_situ_linear_beta")?,
            rms_norm_eps: float(v, "rms_norm_eps")?,
            residual_block: int(v, "attn_res_block_size")?,
            kda_heads,
            kda_head_dim,
            kda_conv: int(linear, "short_conv_kernel_size")?,
            kda_gate_lower_bound: float(linear, "gate_lower_bound")?,
            heads,
            q_lora_rank: int(v, "q_lora_rank")?,
            kv_lora_rank: int(v, "kv_lora_rank")?,
            qk_nope_dim: int(v, "qk_nope_head_dim")?,
            qk_rope_dim: int(v, "qk_rope_head_dim")?,
            v_head_dim: int(v, "v_head_dim")?,
            max_position: int(v, "max_position_embeddings")?,
            tie_embeddings: v["tie_word_embeddings"].as_bool().unwrap_or(false)
                || root["tie_word_embeddings"].as_bool().unwrap_or(false),
            eos,
        };
        ensure!(cfg.situ_beta > 0.0 && cfg.situ_linear_beta > 0.0, "Situ beta values must be positive");
        ensure!(cfg.kda_heads * cfg.kda_head_dim > 0 && cfg.latent < cfg.hidden,
            "Kimi K3 latent width {} must be narrower than hidden {}", cfg.latent, cfg.hidden);
        if cfg.tie_embeddings { bail!("Kimi K3 is untied (tie_word_embeddings=false)"); }
        Ok(cfg)
    }

    /// The release's executable geometry. Others parse but are not served.
    pub fn release_geometry(&self) -> Result<()> {
        let expected = [
            ("hidden_size", self.hidden, 7168), ("num_hidden_layers", self.layers, 93),
            ("vocab_size", self.vocab_size, 163840), ("intermediate_size", self.dense_intermediate, 33792),
            ("num_experts", self.experts, 896), ("num_experts_per_token", self.topk, 16),
            ("moe_intermediate_size", self.moe_intermediate, 3072), ("routed_expert_hidden_size", self.latent, 3584),
            ("num_shared_experts", self.shared_experts, 2), ("attn_res_block_size", self.residual_block, 12),
            ("kda num_heads", self.kda_heads, 96), ("kda head_dim", self.kda_head_dim, 128),
            ("short_conv_kernel_size", self.kda_conv, 4), ("num_attention_heads", self.heads, 96),
            ("q_lora_rank", self.q_lora_rank, 1536), ("kv_lora_rank", self.kv_lora_rank, 512),
            ("qk_nope_head_dim", self.qk_nope_dim, 128), ("qk_rope_head_dim", self.qk_rope_dim, 64),
            ("v_head_dim", self.v_head_dim, 128),
        ];
        for (key, actual, wanted) in expected {
            ensure!(actual == wanted, "unsupported Kimi K3 {key}: {actual}, the release geometry is {wanted}");
        }
        ensure!(self.dense.iter().enumerate().all(|(l, d)| *d == (l == 0)), "Kimi K3 release has exactly layer 0 dense");
        ensure!(self.kda_gate_lower_bound == -5.0, "unsupported KDA gate lower bound {}", self.kda_gate_lower_bound);
        ensure!(self.routed_scale == 1.0, "unsupported routed_scaling_factor {}", self.routed_scale);
        ensure!(self.situ_beta == 4.0 && self.situ_linear_beta == 25.0, "unsupported Situ betas {}/{}",
            self.situ_beta, self.situ_linear_beta);
        Ok(())
    }

    pub fn kda_layers(&self) -> usize { self.attention.iter().filter(|a| **a == KimiAttention::Kda).count() }
    pub fn mla_layers(&self) -> usize { self.attention.iter().filter(|a| **a == KimiAttention::Mla).count() }

    /// Layers that open a new attention-residual block (their input becomes a
    /// stored candidate). Layer 0's block is the embedding.
    pub fn block_starts(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.layers).step_by(self.residual_block)
    }

    /// Stored residual candidates live after the last block start (8 for the release).
    pub fn max_residual_blocks(&self) -> usize { self.layers.div_ceil(self.residual_block) }

    /// The Situ activation shared by routed, shared and dense FFNs.
    pub fn activation(&self) -> cuteafd_core::FfnActivation {
        cuteafd_core::FfnActivation::Situ { beta: self.situ_beta as f32, linear_beta: Some(self.situ_linear_beta as f32) }
    }

    pub fn expert_geometry(&self) -> cuteafd_core::ExpertGeometry {
        cuteafd_core::ExpertGeometry { hidden: self.latent as u32, experts: self.experts as u32, topk: self.topk as u32,
            intermediate: self.moe_intermediate as u32, layers: self.layers as u32 }
    }

    /// MLA latent record per token per layer (BF16 `kv_lora + rope` values).
    pub fn mla_record_values(&self) -> usize { self.kv_lora_rank + self.qk_rope_dim }

    /// KDA recurrent state per layer per sequence, FP32 `[H, V, K]`.
    pub fn kda_state_bytes(&self) -> u64 { (self.kda_heads * self.kda_head_dim * self.kda_head_dim * 4) as u64 }

    /// KDA conv window per layer per sequence: q, k, v channels × (taps − 1), BF16.
    pub fn kda_conv_bytes(&self) -> u64 { (3 * self.kda_heads * self.kda_head_dim * (self.kda_conv - 1) * 2) as u64 }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::json;

    pub(crate) fn release_config() -> Value { crate::plan::testing::kimi_k3_config() }

    #[test]
    fn release_config_parses_with_one_based_schedule() {
        let cfg = KimiK3Config::from_hf(&release_config()).unwrap();
        cfg.release_geometry().unwrap();
        assert_eq!((cfg.kda_layers(), cfg.mla_layers()), (69, 24));
        // 1-based 4, 8, ..., 92, 93 are 0-based 3, 7, ..., 91, 92.
        let mla: Vec<usize> = cfg.attention.iter().enumerate().filter(|(_, a)| **a == KimiAttention::Mla).map(|(l, _)| l).collect();
        assert_eq!(mla.first(), Some(&3));
        assert_eq!(&mla[mla.len() - 2..], &[91, 92]);
        assert_eq!(cfg.attention[0], KimiAttention::Kda);
        assert_eq!(cfg.dense.iter().filter(|d| **d).count(), 1);
        assert!(cfg.dense[0]);
        assert_eq!(cfg.block_starts().collect::<Vec<_>>(), vec![0, 12, 24, 36, 48, 60, 72, 84]);
        assert_eq!(cfg.max_residual_blocks(), 8);
        assert_eq!(cfg.expert_geometry(), cuteafd_core::ExpertGeometry::KIMI_K3);
        assert_eq!(cfg.activation(), cuteafd_core::ExpertGeometry::KIMI_K3.activation());
        assert_eq!(cfg.kda_state_bytes() * cfg.kda_layers() as u64, 414 << 20);
        assert_eq!(cfg.mla_record_values() * 2 * cfg.mla_layers(), 27 * 1024);
    }

    #[test]
    fn schedule_errors_name_the_one_based_layer() {
        let mut config = release_config();
        config["text_config"]["linear_attn_config"]["kda_layers"].as_array_mut().unwrap().push(json!(4));
        let error = KimiK3Config::from_hf(&config).unwrap_err().to_string();
        assert!(error.contains("layer 4 (1-based) is listed twice"), "{error}");
        let mut config = release_config();
        config["text_config"]["linear_attn_config"]["full_attn_layers"].as_array_mut().unwrap().push(json!(0));
        assert!(KimiK3Config::from_hf(&config).unwrap_err().to_string().contains("1-based"));
        let mut config = release_config();
        config["text_config"]["linear_attn_config"]["full_attn_layers"].as_array_mut().unwrap().pop();
        assert!(KimiK3Config::from_hf(&config).unwrap_err().to_string().contains("layer 93 (1-based) has no attention kind"));
    }

    #[test]
    fn semantics_the_runtime_does_not_implement_are_refused() {
        for (key, value, needle) in [
            ("hidden_act", json!("silu"), "situ"),
            ("mla_use_nope", json!(false), "NoPE"),
            ("mla_use_output_gate", json!(false), "output gate"),
            ("moe_router_activation_func", json!("softmax"), "sigmoid"),
            ("num_expert_group", json!(4), "one expert group"),
            ("latent_moe_use_norm", json!(false), "routed_expert_norm"),
            ("num_nextn_predict_layers", json!(1), "MTP"),
        ] {
            let mut config = release_config();
            config["text_config"][key] = value;
            let error = KimiK3Config::from_hf(&config).unwrap_err().to_string();
            assert!(error.contains(needle), "{key}: {error}");
        }
        let mut config = release_config();
        config["text_config"]["linear_attn_config"]["use_full_rank_gate"] = json!(false);
        assert!(KimiK3Config::from_hf(&config).unwrap_err().to_string().contains("full-rank"));
        let mut config = release_config();
        config["text_config"]["num_experts"] = json!(512);
        let cfg = KimiK3Config::from_hf(&config).unwrap();
        assert!(cfg.release_geometry().unwrap_err().to_string().contains("num_experts: 512"));
    }
}
