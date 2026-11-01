//! Tensor names of the GLM engine's operands for each `GlmVariant`: the
//! checkpoint's own naming (GLM 5.x Hugging Face, Mistral consolidated), one
//! logical role per operand. Stems omit the `.weight` suffix; callers append
//! the member they read.
use super::config::{GlmDsaConfig, GlmVariant};

/// One attention or FFN operand of a decoder layer, by role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operand {
    InputNorm,
    PostAttentionNorm,
    /// GLM q-LoRA down projection (absent on direct-query variants).
    QA,
    QANorm,
    /// GLM q-LoRA up projection, or the direct query projection.
    QB,
    KvA,
    KvANorm,
    KvB,
    O,
    /// Per-head scalable-softmax vectors (`s`, `b`); query-scale variants only.
    QueryScaleS,
    QueryScaleB,
    IndexQ,
    IndexK,
    IndexWeights,
    IndexKNorm,
    /// The indexer query's input LayerNorm (direct-query variants).
    IndexQNorm,
    IndexQueryScaleS,
    IndexQueryScaleB,
    Router,
    RouterBias,
    /// Dense layers' MLP or the shared expert: gate, up, down.
    Gate,
    Up,
    Down,
}

impl GlmDsaConfig {
    /// The stem of `operand` in layer `layer` (`shared`: the shared expert's
    /// projections rather than the dense MLP's). `None` when this variant has
    /// no such operand.
    pub fn stem(&self, layer: usize, operand: Operand) -> Option<String> {
        use Operand::*;
        let dense = layer < self.first_moe_layer;
        match &self.variant {
            GlmVariant::Glm => {
                let p = format!("model.layers.{layer}");
                let mlp = if dense { format!("{p}.mlp") } else { format!("{p}.mlp.shared_experts") };
                Some(match operand {
                    InputNorm => format!("{p}.input_layernorm"),
                    PostAttentionNorm => format!("{p}.post_attention_layernorm"),
                    QA => format!("{p}.self_attn.q_a_proj"),
                    QANorm => format!("{p}.self_attn.q_a_layernorm"),
                    QB => format!("{p}.self_attn.q_b_proj"),
                    KvA => format!("{p}.self_attn.kv_a_proj_with_mqa"),
                    KvANorm => format!("{p}.self_attn.kv_a_layernorm"),
                    KvB => format!("{p}.self_attn.kv_b_proj"),
                    O => format!("{p}.self_attn.o_proj"),
                    IndexQ => format!("{p}.self_attn.indexer.wq_b"),
                    IndexK => format!("{p}.self_attn.indexer.wk"),
                    IndexWeights => format!("{p}.self_attn.indexer.weights_proj"),
                    IndexKNorm => format!("{p}.self_attn.indexer.k_norm"),
                    Router => format!("{p}.mlp.gate"),
                    RouterBias => format!("{p}.mlp.gate.e_score_correction_bias"),
                    Gate => format!("{mlp}.gate_proj"),
                    Up => format!("{mlp}.up_proj"),
                    Down => format!("{mlp}.down_proj"),
                    QueryScaleS | QueryScaleB | IndexQNorm | IndexQueryScaleS | IndexQueryScaleB => return None,
                })
            }
            GlmVariant::MistralLarge4(cfg) => {
                let p = format!("layers.{layer}");
                let a = format!("{p}.attention");
                let mlp = if dense { format!("{p}.feed_forward") } else { format!("{p}.shared_experts") };
                Some(match operand {
                    InputNorm => format!("{p}.attention_norm"),
                    PostAttentionNorm => format!("{p}.ffn_norm"),
                    QB => format!("{a}.wq"),
                    KvA if cfg.kv_sources[layer] == layer => format!("{a}.wkv_a_with_mqa"),
                    KvANorm if cfg.kv_sources[layer] == layer => format!("{a}.kv_a_norm"),
                    KvB => format!("{a}.wkv_b"),
                    O => format!("{a}.wo"),
                    QueryScaleS => format!("{a}.ssmax_scaling.s"),
                    QueryScaleB => format!("{a}.ssmax_scaling.b"),
                    IndexQ => format!("{a}.indexer.wq_b"),
                    IndexK => format!("{a}.indexer.wk"),
                    IndexWeights => format!("{a}.indexer.weights_proj"),
                    IndexKNorm => format!("{a}.indexer.k_norm"),
                    IndexQNorm => format!("{a}.indexer.q_input_norm"),
                    IndexQueryScaleS => format!("{a}.indexer.ssmax_scaling.s"),
                    IndexQueryScaleB => format!("{a}.indexer.ssmax_scaling.b"),
                    Router if !dense => format!("{p}.gate"),
                    Gate => format!("{mlp}.w1"),
                    Up => format!("{mlp}.w3"),
                    Down => format!("{mlp}.w2"),
                    QA | QANorm | KvA | KvANorm | Router | RouterBias => return None,
                })
            }
        }
    }

    /// The `.weight` tensor of `operand` (see [`Self::stem`]).
    pub fn weight(&self, layer: usize, operand: Operand) -> Option<String> {
        self.stem(layer, operand).map(|stem| match operand {
            Operand::QueryScaleS | Operand::QueryScaleB | Operand::IndexQueryScaleS | Operand::IndexQueryScaleB
                | Operand::RouterBias => stem,
            _ => format!("{stem}.weight"),
        })
    }

    pub fn embedding(&self) -> &'static str {
        if self.variant.is_glm() { "model.embed_tokens.weight" } else { "tok_embeddings.weight" }
    }

    pub fn final_norm(&self) -> &'static str {
        if self.variant.is_glm() { "model.norm.weight" } else { "norm.weight" }
    }

    /// The vocabulary head's stem (`lm_head` / `output`).
    pub fn head(&self) -> &'static str {
        if self.variant.is_glm() { "lm_head" } else { "output" }
    }

    /// A routed expert projection's stem (`gate`, `up` or `down` by role).
    pub fn expert(&self, layer: usize, expert: usize, operand: Operand) -> Option<String> {
        let leaf = match (&self.variant, operand) {
            (GlmVariant::Glm, Operand::Gate) => "gate_proj",
            (GlmVariant::Glm, Operand::Up) => "up_proj",
            (GlmVariant::Glm, Operand::Down) => "down_proj",
            (GlmVariant::MistralLarge4(_), Operand::Gate) => "w1",
            (GlmVariant::MistralLarge4(_), Operand::Up) => "w3",
            (GlmVariant::MistralLarge4(_), Operand::Down) => "w2",
            _ => return None,
        };
        Some(if self.variant.is_glm() { format!("model.layers.{layer}.mlp.experts.{expert}.{leaf}") }
            else { format!("layers.{layer}.experts.{expert}.{leaf}") })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mistral_large4_names_follow_consolidated_checkpoint() -> anyhow::Result<()> {
        let cfg = GlmDsaConfig::from_hf(&crate::families::mistral_large4::config::tests::official())?;
        assert_eq!(cfg.weight(4, Operand::QB).as_deref(), Some("layers.4.attention.wq.weight"));
        assert_eq!(cfg.weight(4, Operand::KvA).as_deref(), Some("layers.4.attention.wkv_a_with_mqa.weight"));
        assert_eq!(cfg.weight(39, Operand::KvA), None, "layer 39 reads layer 35's latent");
        assert_eq!(cfg.weight(39, Operand::KvB).as_deref(), Some("layers.39.attention.wkv_b.weight"));
        assert_eq!(cfg.weight(0, Operand::Gate).as_deref(), Some("layers.0.feed_forward.w1.weight"));
        assert_eq!(cfg.weight(1, Operand::Down).as_deref(), Some("layers.1.shared_experts.w2.weight"));
        assert_eq!(cfg.weight(0, Operand::Router), None);
        assert_eq!(cfg.weight(1, Operand::Router).as_deref(), Some("layers.1.gate.weight"));
        assert_eq!(cfg.weight(4, Operand::QueryScaleS).as_deref(), Some("layers.4.attention.ssmax_scaling.s"));
        assert_eq!(cfg.weight(4, Operand::QA), None);
        assert_eq!(cfg.expert(7, 3, Operand::Up).as_deref(), Some("layers.7.experts.3.w3"));
        assert_eq!((cfg.head(), cfg.embedding(), cfg.final_norm()), ("output", "tok_embeddings.weight", "norm.weight"));
        Ok(())
    }

    #[test]
    fn glm_names_are_unchanged() -> anyhow::Result<()> {
        let mut types: Vec<&str> = vec!["full"; 3];
        types.extend((3..78).map(|layer| if (layer - 2) % 4 == 0 { "full" } else { "shared" }));
        let cfg = GlmDsaConfig::from_hf(&serde_json::json!({
            "model_type": "glm_moe_dsa", "vocab_size": 154880, "hidden_size": 6144, "num_hidden_layers": 78,
            "num_attention_heads": 64, "q_lora_rank": 2048, "kv_lora_rank": 512, "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64, "v_head_dim": 256, "index_n_heads": 32, "index_head_dim": 128,
            "index_topk": 2048, "indexer_types": types, "first_k_dense_replace": 3, "intermediate_size": 12288,
            "n_routed_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
            "n_shared_experts": 1, "routed_scaling_factor": 2.5, "scoring_func": "sigmoid",
            "topk_method": "noaux_tc", "n_group": 1, "rope_interleave": true, "rms_norm_eps": 1e-5,
            "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"}, "eos_token_id": 154820}))?;
        assert!(cfg.variant.is_glm());
        assert_eq!(cfg.weight(5, Operand::QA).as_deref(), Some("model.layers.5.self_attn.q_a_proj.weight"));
        assert_eq!(cfg.weight(5, Operand::Up).as_deref(), Some("model.layers.5.mlp.shared_experts.up_proj.weight"));
        assert_eq!(cfg.weight(1, Operand::Up).as_deref(), Some("model.layers.1.mlp.up_proj.weight"));
        assert_eq!(cfg.weight(5, Operand::RouterBias).as_deref(), Some("model.layers.5.mlp.gate.e_score_correction_bias"));
        assert_eq!(cfg.weight(5, Operand::QueryScaleS), None);
        assert!((0..78).all(|layer| cfg.rope(layer) && cfg.owns_kv(layer) && cfg.record_bytes(layer) == 656));
        assert_eq!(cfg.logit_softcap(), None);
        Ok(())
    }
}
