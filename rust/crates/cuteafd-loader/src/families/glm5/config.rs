//! GLM 5.x (glm_moe_dsa) model arguments from the Hugging Face `config.json`,
//! and the other MLA + DSA checkpoints the GLM engine runs (`GlmVariant`).
use anyhow::{ensure, Context, Result};
use serde_json::Value;
use std::path::Path;

use crate::plan::checkpoint::read_json;

/// Whether a layer runs its own DSA indexer or reuses the previous full
/// layer's top-k selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlmIndexer {
    Full,
    Shared,
}

/// How a GLM-engine checkpoint's MLA + DSA layer differs from GLM 5.x, read
/// from the checkpoint's own config (never a launch key). The engine runs one
/// layer structure; the variant selects producer and indexer stages and
/// per-layer records.
#[derive(Debug, Clone, PartialEq)]
pub enum GlmVariant {
    /// GLM 5.x: q-LoRA (`q_a_proj` -> `q_a_layernorm` -> `q_b_proj`), RoPE on
    /// every layer, the indexer query from the q-LoRA residual, a sigmoid
    /// noaux_tc router with a correction bias, no head soft cap.
    Glm,
    /// Mistral Large 4: `wq` straight from the normed hidden state with a
    /// per-head scalable-softmax factor, NoPE layers (latent-only 528-byte
    /// records, query nope = nope + rope), cross-layer latent sharing, the
    /// indexer query from `wq_b(LayerNorm(h))` with its own query scale, a
    /// softmax top-k router, interleaved YaRN with LeRoPE, a tanh-capped head.
    MistralLarge4(Box<crate::families::mistral_large4::Ml4Config>),
}

impl GlmVariant {
    pub fn is_glm(&self) -> bool { matches!(self, Self::Glm) }
    pub fn ml4(&self) -> Option<&crate::families::mistral_large4::Ml4Config> {
        match self { Self::MistralLarge4(cfg) => Some(cfg), Self::Glm => None }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GlmDsaConfig {
    pub vocab_size: usize,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub index_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub indexers: Vec<GlmIndexer>,
    /// Layers `0..first_moe_layer` are dense MLPs.
    pub first_moe_layer: usize,
    pub dense_intermediate: usize,
    pub experts: usize,
    pub topk: usize,
    pub moe_intermediate: usize,
    pub shared_experts: usize,
    pub routed_scale: f64,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub mtp_layers: usize,
    pub eos_tokens: Vec<u32>,
    pub variant: GlmVariant,
}

impl GlmDsaConfig {
    pub fn read(snapshot: &Path) -> Result<Self> {
        Self::from_hf(&read_json(&snapshot.join("config.json"))?)
    }

    pub fn from_hf(root: &Value) -> Result<Self> {
        if root["model_type"] == "mistral_large4" {
            return Self::mistral_large4(crate::families::mistral_large4::Ml4Config::from_hf(root)?);
        }
        let v = root.get("text_config").unwrap_or(root);
        ensure!(v["model_type"] == "glm_moe_dsa", "not a glm_moe_dsa config");
        let int = |key: &str| -> Result<usize> {
            v[key].as_u64().map(|x| x as usize).with_context(|| format!("glm_moe_dsa config lacks {key}"))
        };
        let float = |key: &str| -> Result<f64> {
            v[key].as_f64().with_context(|| format!("glm_moe_dsa config lacks {key}"))
        };
        let layers = int("num_hidden_layers")?;
        let indexers = match v.get("indexer_types").filter(|x| !x.is_null()) {
            Some(types) => types.as_array().context("indexer_types must be a list")?.iter().map(|t| match t.as_str() {
                Some("full") => Ok(GlmIndexer::Full),
                Some("shared") => Ok(GlmIndexer::Shared),
                other => anyhow::bail!("unknown indexer type {other:?}"),
            }).collect::<Result<Vec<_>>>()?,
            None => vec![GlmIndexer::Full; layers],
        };
        ensure!(indexers.len() == layers && indexers[0] == GlmIndexer::Full,
            "indexer_types must cover exactly every layer and start with a full indexer");
        // Every layer is MLA + DSA; a `layer_types` list, when present, must say so.
        if let Some(types) = v.get("layer_types").filter(|x| !x.is_null()) {
            let types = types.as_array().context("layer_types must be a list")?;
            ensure!(types.len() == layers && types.iter().all(|t| t == "deepseek_sparse_attention"),
                "glm_moe_dsa layer_types must name deepseek_sparse_attention for exactly every layer");
        }
        // Dense layers are a prefix: `mlp_layer_types` (dense then sparse) or
        // `first_k_dense_replace`; when both are present they agree.
        let first_k = v.get("first_k_dense_replace").filter(|x| !x.is_null())
            .map(|x| x.as_u64().map(|x| x as usize).context("first_k_dense_replace must be an unsigned integer"))
            .transpose()?;
        let first_moe_layer = match v.get("mlp_layer_types").filter(|x| !x.is_null()) {
            Some(types) => {
                let types = types.as_array().context("mlp_layer_types must be a list")?;
                ensure!(types.len() == layers, "mlp_layer_types has {} entries for {layers} layers", types.len());
                ensure!(types.iter().all(|t| t == "dense" || t == "sparse"), "mlp_layer_types entries are dense or sparse");
                let first = types.iter().position(|t| t == "sparse").context("no sparse layer in mlp_layer_types")?;
                ensure!(types[first..].iter().all(|t| t == "sparse"), "mlp_layer_types: dense layers must be a prefix");
                ensure!(first_k.is_none_or(|k| k == first),
                    "mlp_layer_types ({first} dense) and first_k_dense_replace ({}) disagree", first_k.unwrap_or(0));
                first
            }
            None => first_k.context("glm_moe_dsa config lacks first_k_dense_replace")?,
        };
        ensure!(v["scoring_func"] == "sigmoid" && v["topk_method"] == "noaux_tc"
            && v["n_group"].as_u64().unwrap_or(1) == 1, "GLM routing must be sigmoid noaux_tc without groups");
        ensure!(v["rope_interleave"].as_bool().unwrap_or(true), "GLM RoPE is interleaved");
        let rope_theta = v["rope_parameters"]["rope_theta"].as_f64()
            .or_else(|| v["rope_theta"].as_f64()).context("glm_moe_dsa config lacks rope_theta")?;
        ensure!(v["rope_parameters"]["rope_type"].as_str().unwrap_or("default") == "default",
            "scaled RoPE is not supported for GLM yet");
        let eos_tokens = match &v["eos_token_id"] {
            Value::Array(ids) => ids.iter().filter_map(|id| id.as_u64().map(|id| id as u32)).collect(),
            id => id.as_u64().map(|id| vec![id as u32]).unwrap_or_default(),
        };
        Ok(Self {
            vocab_size: int("vocab_size")?,
            hidden: int("hidden_size")?,
            layers,
            heads: int("num_attention_heads")?,
            q_lora_rank: int("q_lora_rank")?,
            kv_lora_rank: int("kv_lora_rank")?,
            qk_nope_head_dim: int("qk_nope_head_dim")?,
            qk_rope_head_dim: int("qk_rope_head_dim")?,
            v_head_dim: int("v_head_dim")?,
            index_heads: int("index_n_heads")?,
            index_head_dim: int("index_head_dim")?,
            index_topk: int("index_topk")?,
            indexers,
            first_moe_layer,
            dense_intermediate: int("intermediate_size")?,
            experts: int("n_routed_experts")?,
            topk: int("num_experts_per_tok")?,
            moe_intermediate: int("moe_intermediate_size")?,
            shared_experts: int("n_shared_experts")?,
            routed_scale: float("routed_scaling_factor")?,
            rope_theta,
            rms_norm_eps: float("rms_norm_eps")?,
            mtp_layers: v["num_nextn_predict_layers"].as_u64().unwrap_or(0) as usize,
            eos_tokens,
            variant: GlmVariant::Glm,
        })
    }

    /// Mistral Large 4 on the GLM engine. `qk_nope_head_dim` is the RoPE
    /// layers' (NoPE layers widen it by the rope width, see [`Self::nope_dim`]);
    /// `q_lora_rank` 0 selects the direct query projection.
    pub fn mistral_large4(cfg: crate::families::mistral_large4::Ml4Config) -> Result<Self> {
        ensure!(cfg.routed_scale > 0.0 && cfg.experts > 0, "Mistral Large 4 routing geometry");
        Ok(Self {
            vocab_size: cfg.vocab_size,
            hidden: cfg.hidden,
            layers: cfg.layers,
            heads: cfg.heads,
            q_lora_rank: 0,
            kv_lora_rank: cfg.kv_lora_rank,
            qk_nope_head_dim: cfg.qk_nope_head_dim,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
            v_head_dim: cfg.v_head_dim,
            index_heads: cfg.index_heads,
            index_head_dim: cfg.index_head_dim,
            index_topk: cfg.index_topk,
            indexers: cfg.indexers.iter().map(|kind| match kind {
                crate::families::mistral_large4::Ml4Indexer::Full => GlmIndexer::Full,
                crate::families::mistral_large4::Ml4Indexer::Shared => GlmIndexer::Shared,
            }).collect(),
            first_moe_layer: cfg.first_moe_layer,
            dense_intermediate: cfg.dense_intermediate,
            experts: cfg.experts,
            topk: cfg.topk,
            moe_intermediate: cfg.moe_intermediate,
            shared_experts: cfg.shared_experts,
            routed_scale: cfg.routed_scale,
            rope_theta: cfg.rope_scaling.theta,
            rms_norm_eps: cfg.rms_norm_eps,
            mtp_layers: 0,
            eos_tokens: cfg.eos_tokens.clone(),
            variant: GlmVariant::MistralLarge4(Box::new(cfg)),
        })
    }

    /// Whether layer `layer` applies RoPE (always on GLM 5.x).
    pub fn rope(&self, layer: usize) -> bool {
        self.variant.ml4().is_none_or(|cfg| cfg.rope[layer])
    }

    /// Nope width of layer `layer`'s query/key heads.
    pub fn nope_dim(&self, layer: usize) -> usize {
        if self.rope(layer) { self.qk_nope_head_dim } else { self.qk_nope_head_dim + self.qk_rope_head_dim }
    }

    /// The layer whose latent cache `layer` attends to (itself on GLM 5.x).
    pub fn kv_source(&self, layer: usize) -> usize {
        self.variant.ml4().map_or(layer, |cfg| cfg.kv_sources[layer])
    }

    /// Whether `layer` owns (computes and caches) its latent.
    pub fn owns_kv(&self, layer: usize) -> bool { self.kv_source(layer) == layer }

    /// FP8 latent record bytes of `layer`: 656 (512 E4M3 + 4 F32 scales + 64
    /// BF16 RoPE) or 528 without RoPE.
    pub fn record_bytes(&self, layer: usize) -> usize {
        self.kv_lora_rank + 4 * (self.kv_lora_rank / 128) + if self.rope(layer) { 2 * self.qk_rope_head_dim } else { 0 }
    }

    /// The head's tanh soft cap, if any.
    /// The plan family descriptor this checkpoint is admitted under.
    pub fn family_id(&self) -> &'static str {
        if self.variant.is_glm() { "glm5" } else { "mistral_large4" }
    }

    /// The coordinator program namespace of this variant's own exports
    /// (`glm_*`, or `ml4_*` from `export_b12x_dsv4_aot --geometry ml4`).
    pub fn program_family(&self) -> &'static str {
        if self.variant.is_glm() { "glm" } else { "ml4" }
    }

    /// The namespace of a head split's share programs (`glm2_*`); a variant
    /// without them runs whole attention on each GPU (heads on one GPU,
    /// whole-layer owners on two).
    pub fn split_program_family(&self) -> Option<&'static str> {
        self.variant.is_glm().then_some("glm2")
    }

    pub fn logit_softcap(&self) -> Option<f64> {
        self.variant.ml4().map(|cfg| cfg.logit_softcap)
    }

    /// The full layer whose top-k selection `layer` uses.
    pub fn index_source(&self, layer: usize) -> usize {
        (0..=layer).rev().find(|&l| self.indexers[l] == GlmIndexer::Full).unwrap_or(0)
    }

    /// Softmax scale of the MLA attention (qk_head_dim^-1/2).
    pub fn softmax_scale(&self) -> f64 {
        ((self.qk_nope_head_dim + self.qk_rope_head_dim) as f64).powf(-0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glm53_shape_and_index_sources() -> Result<()> {
        let mut types: Vec<&str> = vec!["full"; 3];
        for layer in 3..78 {
            types.push(if (layer - 2) % 4 == 0 { "full" } else { "shared" });
        }
        let v = serde_json::json!({
            "model_type": "glm_moe_dsa", "vocab_size": 154880, "hidden_size": 6144, "num_hidden_layers": 78,
            "num_attention_heads": 64, "q_lora_rank": 2048, "kv_lora_rank": 512, "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64, "v_head_dim": 256, "index_n_heads": 32, "index_head_dim": 128,
            "index_topk": 2048, "indexer_types": types, "first_k_dense_replace": 3, "intermediate_size": 12288,
            "n_routed_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
            "n_shared_experts": 1, "routed_scaling_factor": 2.5, "scoring_func": "sigmoid",
            "topk_method": "noaux_tc", "n_group": 1, "rope_interleave": true, "rms_norm_eps": 1e-5,
            "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
            "num_nextn_predict_layers": 1, "eos_token_id": [154820, 154827, 154829],
        });
        let cfg = GlmDsaConfig::from_hf(&v)?;
        assert_eq!((cfg.first_moe_layer, cfg.topk, cfg.eos_tokens.len()), (3, 8, 3));
        assert_eq!(cfg.index_source(5), 2);
        assert_eq!(cfg.index_source(6), 6);
        assert!((cfg.softmax_scale() - 1.0 / 16.0).abs() < 1e-12);
        Ok(())
    }
}
