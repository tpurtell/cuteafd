//! Mistral Large 4 (`mistral_large4`, text `mistral_large4_text`) model
//! arguments from the checkpoint's own `config.json`, checked against the
//! official Hugging Face modeling (transformers #50121) and Mistral's vLLM
//! port (vllm-project/vllm#64534):
//!
//! - MLA without q-LoRA: `wq` [heads x (nope + rope), hidden] straight from
//!   the normed hidden state; `wkv_a_with_mqa` -> latent (+ RoPE key on RoPE
//!   layers), `kv_a_norm`, `wkv_b`, `wo`. NoPE layers fold the rope width into
//!   the nope part (q/k head 128 = nope 128, no RoPE key).
//! - Scalable softmax: every query head is scaled by `s * log(pos + 1) + b`
//!   after RoPE (attention and indexer each have their own `ssmax_scaling`).
//! - DSA indexer on `full` layers only; `shared` layers reuse the previous full
//!   layer's top-k. The indexer query is `wq_b(LayerNorm(h))`, not a q-LoRA
//!   latent; its RoPE (interleaved, leading 64 dims) is skipped on NoPE layers.
//! - KV sharing: a layer whose `kv_source_layers` entry is an earlier layer
//!   attends to that layer's latent cache and has no `wkv_a`/`kv_a_norm`
//!   (its own `wkv_b` still expands the shared latent).
//! - Interleaved YaRN RoPE with LeRoPE (`inv_freq * exp(lerope.alpha)`), no
//!   magnitude scaling; final logits soft-capped by `tanh(x / c) * c`.
use anyhow::{bail, ensure, Context, Result};
use serde_json::Value;

/// Whether a layer runs its own DSA indexer or reuses the previous full
/// layer's top-k selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ml4Indexer {
    Full,
    Shared,
}

/// Interleaved YaRN over the pretraining window, LeRoPE alpha from the
/// checkpoint (`lerope.alpha`), no attention magnitude factor.
#[derive(Debug, Clone, PartialEq)]
pub struct Ml4Rope {
    pub theta: f64,
    pub factor: f64,
    pub original_max_position: usize,
    pub beta_fast: f64,
    pub beta_slow: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Ml4Config {
    pub vocab_size: usize,
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_lora_rank: usize,
    /// Nope width of a RoPE layer's q/k head; NoPE layers use nope + rope.
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub index_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub indexers: Vec<Ml4Indexer>,
    /// `rope[l]`: layer `l` applies RoPE (attention and indexer).
    pub rope: Vec<bool>,
    /// `kv_sources[l]`: the layer whose latent cache layer `l` attends to.
    pub kv_sources: Vec<usize>,
    /// Layers `0..first_moe_layer` are dense MLPs.
    pub first_moe_layer: usize,
    pub dense_intermediate: usize,
    pub experts: usize,
    pub topk: usize,
    pub moe_intermediate: usize,
    pub shared_experts: usize,
    pub routed_scale: f64,
    pub rope_scaling: Ml4Rope,
    pub rms_norm_eps: f64,
    pub logit_softcap: f64,
    pub max_position_embeddings: usize,
    pub eos_tokens: Vec<u32>,
}

/// The indexer's LayerNorms (`q_input_norm`, `k_norm`) use the reference's
/// fixed epsilon, not `rms_norm_eps`.
pub const INDEXER_NORM_EPS: f64 = 1e-6;

impl Ml4Config {
    pub fn from_hf(root: &Value) -> Result<Self> {
        let v = root.get("text_config").unwrap_or(root);
        ensure!(v["model_type"] == "mistral_large4_text", "not a mistral_large4_text config");
        let int = |key: &str| -> Result<usize> {
            v[key].as_u64().map(|x| x as usize).with_context(|| format!("mistral_large4 config lacks {key}"))
        };
        let float = |key: &str| -> Result<f64> {
            v[key].as_f64().with_context(|| format!("mistral_large4 config lacks {key}"))
        };
        let list = |key: &str| -> Result<&Vec<Value>> {
            v[key].as_array().with_context(|| format!("mistral_large4 config lacks the {key} list"))
        };
        let layers = int("num_hidden_layers")?;
        let heads = int("num_attention_heads")?;
        ensure!(v["num_key_value_heads"].as_u64() == Some(heads as u64),
            "MLA expands one latent per head: num_key_value_heads must equal num_attention_heads");
        ensure!(v["q_lora_rank"].is_null(), "Mistral Large 4 MLA has no q-LoRA");
        ensure!(v["hidden_act"] == "silu", "Mistral Large 4 FFNs are SiLU-gated");
        ensure!(!v["attention_bias"].as_bool().unwrap_or(false), "Mistral Large 4 attention has no bias");
        ensure!(v["tie_word_embeddings"].as_bool() != Some(true) && root["tie_word_embeddings"].as_bool() != Some(true),
            "Mistral Large 4 has an untied output head");

        let indexers = list("indexer_types")?.iter().map(|t| match t.as_str() {
            Some("full") => Ok(Ml4Indexer::Full),
            Some("shared") => Ok(Ml4Indexer::Shared),
            other => bail!("unknown indexer type {other:?}"),
        }).collect::<Result<Vec<_>>>()?;
        ensure!(indexers.len() == layers && indexers[0] == Ml4Indexer::Full,
            "indexer_types must cover every layer and start with a full indexer");
        if let Some(types) = v.get("layer_types").filter(|x| !x.is_null()) {
            let types = types.as_array().context("layer_types must be a list")?;
            ensure!(types.len() == layers && types.iter().all(|t| t == "indexed_attention"),
                "mistral_large4 layer_types must name indexed_attention for every layer");
        }
        // smollm3 convention: 1 = the layer uses RoPE, 0 = NoPE.
        let rope = list("no_rope_layers")?.iter().map(|flag| match flag.as_u64() {
            Some(1) => Ok(true),
            Some(0) => Ok(false),
            _ => bail!("no_rope_layers entries are 0 or 1"),
        }).collect::<Result<Vec<_>>>()?;
        ensure!(rope.len() == layers, "no_rope_layers has {} entries for {layers} layers", rope.len());
        let kv_sources = list("kv_source_layers")?.iter()
            .map(|s| s.as_u64().map(|s| s as usize).context("kv_source_layers entries are layer indices"))
            .collect::<Result<Vec<_>>>()?;
        ensure!(kv_sources.len() == layers, "kv_source_layers has {} entries for {layers} layers", kv_sources.len());
        for (layer, &source) in kv_sources.iter().enumerate() {
            // The reference's validate_architecture: a source owns its cache.
            ensure!(source <= layer && kv_sources[source] == source,
                "layer {layer} must share KV with an earlier layer that owns its cache, got {source}");
            // The reference keys the shared latent by RoPE kind: a reader takes
            // the latest source of its own kind. Equal kinds keep that the
            // named source; anything else would need the reference's keying.
            ensure!(rope[source] == rope[layer],
                "layer {layer} shares KV with layer {source} of the other RoPE kind");
            if source != layer {
                // A source stores the slot (`store_full_length_kv`) when any later layer names it.
                let stores = |l: usize| kv_sources[l + 1..].contains(&l);
                ensure!((source + 1..layer).all(|l| !stores(l) || rope[l] != rope[layer]),
                    "layer {layer} reads layer {source}'s latent but a later {} source owns its own",
                    if rope[layer] { "RoPE" } else { "NoPE" });
            }
        }

        let first_k = v.get("first_k_dense_replace").filter(|x| !x.is_null())
            .map(|x| x.as_u64().map(|x| x as usize).context("first_k_dense_replace must be an unsigned integer"))
            .transpose()?;
        let mlp = list("mlp_layer_types")?;
        ensure!(mlp.len() == layers && mlp.iter().all(|t| t == "dense" || t == "sparse"),
            "mlp_layer_types needs dense/sparse for every layer");
        let first_moe_layer = mlp.iter().position(|t| t == "sparse").context("no sparse layer in mlp_layer_types")?;
        ensure!(mlp[first_moe_layer..].iter().all(|t| t == "sparse"), "mlp_layer_types: dense layers must be a prefix");
        ensure!(first_k.is_none_or(|k| k == first_moe_layer),
            "mlp_layer_types ({first_moe_layer} dense) and first_k_dense_replace disagree");
        ensure!(v["n_group"].as_u64().unwrap_or(1) == 1 && v["topk_group"].as_u64().unwrap_or(1) == 1,
            "Mistral Large 4 routing is ungrouped softmax top-k");

        let r = &v["rope_parameters"];
        ensure!(r["rope_type"] == "yarn", "Mistral Large 4 RoPE is YaRN");
        // `apply_scale: false`: no cos/sin magnitude scaling (vLLM asserts the same).
        ensure!(r["attention_factor"].as_f64() == Some(1.0), "YaRN attention_factor must be 1.0 (no magnitude scaling)");
        ensure!(r.get("mscale").is_none_or(Value::is_null) && r.get("mscale_all_dim").is_none_or(Value::is_null),
            "YaRN mscale is not used by Mistral Large 4");
        let rope_scaling = Ml4Rope {
            theta: r["rope_theta"].as_f64().context("rope_parameters lacks rope_theta")?,
            factor: r["factor"].as_f64().context("rope_parameters lacks factor")?,
            original_max_position: r["original_max_position_embeddings"].as_u64()
                .context("rope_parameters lacks original_max_position_embeddings")? as usize,
            beta_fast: r["beta_fast"].as_f64().unwrap_or(32.0),
            beta_slow: r["beta_slow"].as_f64().unwrap_or(1.0),
        };
        ensure!(rope_scaling.factor >= 1.0 && rope_scaling.original_max_position > 0, "YaRN factor/window");

        let eos_tokens = match &v["eos_token_id"] {
            Value::Array(ids) => ids.iter().filter_map(|id| id.as_u64().map(|id| id as u32)).collect(),
            id => id.as_u64().map(|id| vec![id as u32]).unwrap_or_default(),
        };
        let cfg = Self {
            vocab_size: int("vocab_size")?,
            hidden: int("hidden_size")?,
            layers,
            heads,
            kv_lora_rank: int("kv_lora_rank")?,
            qk_nope_head_dim: int("qk_nope_head_dim")?,
            qk_rope_head_dim: int("qk_rope_head_dim")?,
            v_head_dim: int("v_head_dim")?,
            index_heads: int("index_n_heads")?,
            index_head_dim: int("index_head_dim")?,
            index_topk: int("index_topk")?,
            indexers,
            rope,
            kv_sources,
            first_moe_layer,
            dense_intermediate: int("intermediate_size")?,
            experts: int("n_routed_experts")?,
            topk: int("num_experts_per_tok")?,
            moe_intermediate: int("moe_intermediate_size")?,
            shared_experts: int("n_shared_experts")?,
            routed_scale: float("routed_scaling_factor")?,
            rope_scaling,
            rms_norm_eps: float("rms_norm_eps")?,
            logit_softcap: float("final_logit_softcapping")?,
            max_position_embeddings: int("max_position_embeddings")?,
            eos_tokens,
        };
        ensure!(cfg.index_head_dim >= cfg.qk_rope_head_dim, "indexer head narrower than its RoPE slice");
        ensure!(cfg.logit_softcap > 0.0, "final_logit_softcapping must be positive");
        Ok(cfg)
    }

    /// Q/K head width (nope + rope); NoPE layers spend it all on nope.
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Nope width of layer `layer`'s q/k heads.
    pub fn nope_dim(&self, layer: usize) -> usize {
        if self.rope[layer] { self.qk_nope_head_dim } else { self.qk_head_dim() }
    }

    /// RoPE key width carried in layer `layer`'s latent (0 on NoPE layers).
    pub fn rope_dim(&self, layer: usize) -> usize {
        if self.rope[layer] { self.qk_rope_head_dim } else { 0 }
    }

    /// Whether `layer` computes and caches its own latent KV.
    pub fn owns_kv(&self, layer: usize) -> bool {
        self.kv_sources[layer] == layer
    }

    /// Layers that own a latent cache (one cache per owner, shared readers included).
    pub fn kv_owners(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.layers).filter(|&layer| self.owns_kv(layer))
    }

    /// The full layer whose top-k selection `layer` uses.
    pub fn index_source(&self, layer: usize) -> usize {
        (0..=layer).rev().find(|&l| self.indexers[l] == Ml4Indexer::Full).unwrap_or(0)
    }

    /// Softmax scale of the MLA attention: qk_head_dim^-1/2 (no YaRN mscale).
    pub fn softmax_scale(&self) -> f64 {
        (self.qk_head_dim() as f64).powf(-0.5)
    }

    /// `[out, in]` of every dense (non-routed) text projection as the
    /// official checkpoint names it (Mistral consolidated naming).
    pub fn projection_shapes(&self) -> std::collections::BTreeMap<String, [usize; 2]> {
        let h = self.hidden;
        let mut shapes = std::collections::BTreeMap::from([("output".to_string(), [self.vocab_size, h])]);
        for layer in 0..self.layers {
            let a = format!("layers.{layer}.attention");
            shapes.insert(format!("{a}.wq"), [self.heads * self.qk_head_dim(), h]);
            shapes.insert(format!("{a}.wo"), [h, self.heads * self.v_head_dim]);
            shapes.insert(format!("{a}.wkv_b"), [self.heads * (self.nope_dim(layer) + self.v_head_dim), self.kv_lora_rank]);
            if self.owns_kv(layer) {
                shapes.insert(format!("{a}.wkv_a_with_mqa"), [self.kv_lora_rank + self.rope_dim(layer), h]);
            }
            if self.indexers[layer] == Ml4Indexer::Full {
                shapes.insert(format!("{a}.indexer.wq_b"), [self.index_heads * self.index_head_dim, h]);
                shapes.insert(format!("{a}.indexer.wk"), [self.index_head_dim, h]);
                shapes.insert(format!("{a}.indexer.weights_proj"), [self.index_heads, h]);
            }
            let (mlp, inter) = if layer < self.first_moe_layer { ("feed_forward", self.dense_intermediate) }
                else { ("shared_experts", self.moe_intermediate * self.shared_experts) };
            if layer >= self.first_moe_layer {
                shapes.insert(format!("layers.{layer}.gate"), [self.experts, h]);
            }
            for (w, n, k) in [("w1", inter, h), ("w3", inter, h), ("w2", h, inter)] {
                shapes.insert(format!("layers.{layer}.{mlp}.{w}"), [n, k]);
            }
        }
        shapes
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn official() -> Value {
        serde_json::from_str(include_str!("official-config.json")).unwrap()
    }

    #[test]
    fn official_config_matches_the_reference_schedule() -> Result<()> {
        let cfg = Ml4Config::from_hf(&official())?;
        assert_eq!((cfg.layers, cfg.hidden, cfg.heads, cfg.vocab_size), (72, 6144, 64, 204_800));
        assert_eq!((cfg.qk_head_dim(), cfg.v_head_dim, cfg.kv_lora_rank), (128, 128, 512));
        assert_eq!((cfg.experts, cfg.topk, cfg.moe_intermediate, cfg.shared_experts), (128, 4, 6144, 1));
        assert_eq!((cfg.first_moe_layer, cfg.dense_intermediate), (1, 24576));
        // NoPE every 4th layer (3, 7, ..., 71).
        let nope: Vec<_> = (0..72).filter(|&l| !cfg.rope[l]).collect();
        assert_eq!(nope, (0..18).map(|i| 4 * i + 3).collect::<Vec<_>>());
        assert_eq!((cfg.nope_dim(3), cfg.rope_dim(3), cfg.nope_dim(4), cfg.rope_dim(4)), (128, 0, 64, 64));
        // Layers 39, 43, ..., 71 read layer 35's latent: 63 owners.
        let readers: Vec<_> = (0..72).filter(|&l| !cfg.owns_kv(l)).collect();
        assert_eq!(readers, (0..9).map(|i| 39 + 4 * i).collect::<Vec<_>>());
        assert!(readers.iter().all(|&l| cfg.kv_sources[l] == 35));
        assert_eq!(cfg.kv_owners().count(), 63);
        // 44 full indexers; shared layers come in pairs from layer 17.
        assert_eq!(cfg.indexers.iter().filter(|&&i| i == Ml4Indexer::Full).count(), 44);
        assert_eq!((cfg.index_source(17), cfg.index_source(18), cfg.index_source(19)), (16, 16, 19));
        assert_eq!(cfg.index_source(70), 68);
        assert!((cfg.softmax_scale() - 128f64.powf(-0.5)).abs() < 1e-15);
        assert_eq!((cfg.logit_softcap, cfg.rms_norm_eps), (30.0, 1e-5));
        assert_eq!(cfg.rope_scaling, Ml4Rope { theta: 1e4, factor: 2048.0, original_max_position: 512,
            beta_fast: 32.0, beta_slow: 1.0 });
        assert_eq!((cfg.max_position_embeddings, cfg.eos_tokens.as_slice()), (1 << 20, &[2u32][..]));
        Ok(())
    }

    #[test]
    fn projection_shapes_follow_rope_and_kv_sharing() -> Result<()> {
        let cfg = Ml4Config::from_hf(&official())?;
        let shapes = cfg.projection_shapes();
        assert_eq!(shapes["layers.4.attention.wkv_a_with_mqa"], [576, 6144]);
        assert_eq!(shapes["layers.3.attention.wkv_a_with_mqa"], [512, 6144]);
        assert_eq!(shapes["layers.4.attention.wkv_b"], [12288, 512]);
        assert_eq!(shapes["layers.39.attention.wkv_b"], [16384, 512]);
        assert!(!shapes.contains_key("layers.39.attention.wkv_a_with_mqa"));
        assert!(!shapes.contains_key("layers.17.attention.indexer.wq_b"));
        assert_eq!(shapes["layers.0.feed_forward.w2"], [6144, 24576]);
        assert_eq!(shapes["layers.1.shared_experts.w1"], [6144, 6144]);
        assert_eq!(shapes["output"], [204_800, 6144]);
        Ok(())
    }

    #[test]
    fn refuses_what_the_reference_would_not_run() {
        let mut bad = official();
        bad["text_config"]["kv_source_layers"][39] = serde_json::json!(40);
        assert!(Ml4Config::from_hf(&bad).unwrap_err().to_string().contains("earlier layer"));
        let mut bad = official();
        bad["text_config"]["kv_source_layers"][39] = serde_json::json!(34);
        assert!(Ml4Config::from_hf(&bad).unwrap_err().to_string().contains("other RoPE kind"));
        let mut bad = official();
        bad["text_config"]["rope_parameters"]["attention_factor"] = serde_json::json!(1.2);
        assert!(Ml4Config::from_hf(&bad).is_err());
        let mut bad = official();
        bad["text_config"]["indexer_types"][0] = serde_json::json!("shared");
        assert!(Ml4Config::from_hf(&bad).is_err());
        let mut bad = official();
        bad["text_config"]["n_group"] = serde_json::json!(8);
        assert!(Ml4Config::from_hf(&bad).is_err());
    }
}
