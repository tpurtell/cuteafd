//! The bundled `mtp_dspark` drafter of Mistral Large 4: its description from
//! the checkpoint's own `params.json:mtp` (the Mistral-format config the
//! release ships beside `config.json`, which does not describe the drafter),
//! cross-checked against the `mtp_dspark.*` tensor headers. Semantics are
//! vLLM bd4fa263's `get_mistral_large_4_dspark_config`; the PyTorch oracle is
//! `python/reference/families/mistral_large4/dspark.py`.
use anyhow::{ensure, Context, Result};
use serde_json::Value;

/// Prefix of the drafter's tensors in the target checkpoint.
pub const PREFIX: &str = "mtp_dspark.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ml4DsparkConfig {
    /// Dense MLA draft layers (the target's attention geometry and dense FFN).
    pub layers: usize,
    /// Block depth: `[anchor, mask_1, ..]` rows, each predicting the next token
    /// (`sample_from_anchor`), so a proposal is `depth` drafts.
    pub depth: usize,
    /// 0-based target layer outputs the context reads, ascending.
    pub taps: Vec<usize>,
    /// Context records each block row attends (before its anchor).
    pub window: usize,
    pub markov_rank: usize,
}

impl Ml4DsparkConfig {
    /// `params.json:mtp`; only the trained variant the engine implements is
    /// accepted (RNN head, position-specific masks, one window for every layer).
    pub fn from_params(params: &Value, target_layers: usize) -> Result<Self> {
        let mtp = params.get("mtp").context("params.json has no mtp block (no bundled drafter)")?;
        let dspark = mtp.get("dspark").context("params.json mtp has no dspark block")?;
        let int = |v: &Value, key: &str| -> Result<usize> {
            v[key].as_u64().map(|x| x as usize).with_context(|| format!("mtp lacks {key}"))
        };
        let csv = |key: &str| -> Result<Vec<usize>> {
            mtp[key].as_str().with_context(|| format!("mtp lacks {key}"))?.split(',')
                .map(|v| v.trim().parse::<usize>().with_context(|| format!("mtp {key}: {v:?}"))).collect()
        };
        ensure!(dspark["head_type"] == "rnn" && dspark["draft_input"] == "position_specific",
            "mtp_dspark: the engine runs the RNN Markov head with position-specific masks, not {dspark}");
        let known = ["markov_rank", "head_type", "draft_input"];
        ensure!(dspark.as_object().is_some_and(|o| o.keys().all(|k| known.contains(&k.as_str()))),
            "mtp_dspark: unparsed dspark options {dspark}");
        let taps = csv("target_activation_layers")?;
        ensure!(!taps.is_empty() && taps.windows(2).all(|w| w[0] < w[1]) && taps.iter().all(|&l| l < target_layers),
            "mtp_dspark: target_activation_layers {taps:?} must be unique, ascending target layers");
        let windows = csv("ragged_attention")?;
        ensure!(windows.windows(2).all(|w| w[0] == w[1]) && windows.first().is_some_and(|&w| w > 0),
            "mtp_dspark: one positive sliding window for every draft layer, not {windows:?}");
        let cfg = Self { layers: int(mtp, "n_layers")?, depth: int(mtp, "depth")?, taps, window: windows[0],
            markov_rank: int(dspark, "markov_rank")? };
        ensure!(windows.len() == cfg.layers, "mtp_dspark: {} windows for {} layers", windows.len(), cfg.layers);
        ensure!(cfg.depth >= 2 && cfg.layers > 0, "mtp_dspark: position-specific masks need depth > 1");
        ensure!(mtp.get("norm_strategy").is_none_or(|n| n == "eagle"), "mtp_dspark: norm strategy {}", mtp["norm_strategy"]);
        Ok(cfg)
    }

    /// Selected record slots per block row: the window plus the block, padded to 64
    /// (the `ml4_draft_sparse_mla` width).
    pub fn slots(&self) -> usize {
        (self.window + self.depth).div_ceil(64) * 64
    }

    /// Every drafter tensor and its `[shape]`, given the target's geometry.
    pub fn tensors(&self, hidden: usize, heads: usize, qk_head: usize, latent: usize, rope: usize, nope: usize,
        v_head: usize, intermediate: usize, vocab: usize) -> Vec<(String, Vec<usize>)> {
        let p = |s: &str| format!("{PREFIX}{s}");
        let rank = self.markov_rank;
        let mut out = vec![
            (p("backbone.target_projection.weight"), vec![hidden, hidden * self.taps.len()]),
            (p("backbone.target_norm.weight"), vec![hidden]),
            (p("backbone.final_norm.weight"), vec![hidden]),
            (p("backbone.mask_embedding"), vec![self.depth - 1, hidden]),
            (p("autoregressive_head.embedding.weight"), vec![vocab, rank]),
            (p("autoregressive_head.output.weight"), vec![vocab, rank]),
            (p("autoregressive_head.joint_proj.weight"), vec![3 * rank, hidden + 2 * rank]),
            (p("autoregressive_head.joint_proj.bias"), vec![3 * rank]),
        ];
        for i in 0..self.layers {
            let l = |s: &str| p(&format!("backbone.layers.{i}.{s}"));
            out.extend([
                (l("attention_norm.weight"), vec![hidden]), (l("ffn_norm.weight"), vec![hidden]),
                (l("attention.wq.weight"), vec![heads * qk_head, hidden]),
                (l("attention.wkv_a_with_mqa.weight"), vec![latent + rope, hidden]),
                (l("attention.kv_a_norm.weight"), vec![latent]),
                (l("attention.wkv_b.weight"), vec![heads * (nope + v_head), latent]),
                (l("attention.wo.weight"), vec![hidden, heads * v_head]),
                (l("feed_forward.w1.weight"), vec![intermediate, hidden]),
                (l("feed_forward.w3.weight"), vec![intermediate, hidden]),
                (l("feed_forward.w2.weight"), vec![hidden, intermediate]),
            ]);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn official_params() -> Value {
        serde_json::json!({"mtp": {"n_layers": 3, "depth": 3, "ragged_attention": "2048,2048,2048",
            "target_activation_layers": "1,18,36,54,71", "norm_strategy": "eagle",
            "dspark": {"markov_rank": 512, "head_type": "rnn", "draft_input": "position_specific"}}})
    }

    #[test]
    fn official_mtp_block_is_the_engines_variant() -> Result<()> {
        let cfg = Ml4DsparkConfig::from_params(&official_params(), 72)?;
        assert_eq!(cfg, Ml4DsparkConfig { layers: 3, depth: 3, taps: vec![1, 18, 36, 54, 71], window: 2048,
            markov_rank: 512 });
        assert_eq!(cfg.slots(), 2112);
        // Every tensor the official checkpoint ships under mtp_dspark., with its shape.
        let template: Vec<Value> = serde_json::from_str(include_str!("official-tensors.json"))?;
        let mut shipped = std::collections::BTreeMap::new();
        for t in &template {
            let pattern = t["pattern"].as_str().unwrap();
            if !pattern.starts_with(PREFIX) { continue; }
            let shape: Vec<usize> = t["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
            let axes = t["axes"].as_array().unwrap();
            if axes.is_empty() { shipped.insert(pattern.to_string(), shape); continue; }
            for i in axes[0].as_array().unwrap() { shipped.insert(pattern.replacen("{}", &i.to_string(), 1), shape.clone()); }
        }
        let expected: std::collections::BTreeMap<_, _> =
            cfg.tensors(6144, 64, 128, 512, 64, 64, 128, 24576, 204800).into_iter().collect();
        assert_eq!(shipped, expected);
        Ok(())
    }

    #[test]
    fn the_shared_mla_draft_layout_admits_the_mtp_dspark_kind() -> Result<()> {
        use crate::families::glm5::draft_representation::{GlmDraftCapacity, GlmDraftRepresentation};
        use crate::formats::dflash::MlaDflashConfig;
        let target = crate::families::mistral_large4::Ml4Config::from_hf(&super::super::config::tests::official())?;
        let dspark = Ml4DsparkConfig::from_params(&official_params(), target.layers)?;
        let cfg = MlaDflashConfig::ml4(&dspark, &target)?;
        assert_eq!((cfg.drafts(), cfg.windows.clone()), (3, vec![Some(2048); 3]));
        // The draft's rotary is the target's at alpha 0 (no LeRoPE).
        assert_eq!(cfg.rotary.inv_freq, target.inv_freq(&[0.0; 32])?);
        assert_eq!(cfg.attention_scale(), (128f32).sqrt().recip());
        assert!(cfg.weight_layout(GlmDraftRepresentation::Fp8Only).is_err());
        let capacity = GlmDraftCapacity::new(20, 16, 3)?;
        let layout = cfg.runtime_layout(GlmDraftRepresentation::Bf16Only, capacity, 1 << 20, 1 << 21, 188, 0,
            cuteafd_core::MlaCacheFormat::Fp8)?;
        // Every draft layer is windowed: per-slot rings (window + block, page-rounded), no pooled layers.
        assert_eq!(layout.paged_cache().full_layers, 0);
        for layer in &layout.cache().layers {
            assert_eq!((layer.window(), layer.rows_per_slot()), (Some(2048), 2112));
        }
        let names: Vec<&str> = layout.buffers().iter().map(|b| b.name.as_str()).collect();
        for name in ["cos_sin", "unit_scale", "markov_norms", "markov_workspace", "logits"] {
            assert!(names.contains(&name), "{name}");
        }
        for absent in ["conv", "dynamic", "candidates", "projected", "q_residual"] {
            assert!(!names.contains(&absent), "{absent}");
        }
        let cos_sin = layout.buffers().iter().find(|b| b.name == "cos_sin").unwrap();
        assert_eq!(cos_sin.bytes, (1 << 20) * 64 * 4);
        assert!(cfg.runtime_layout(GlmDraftRepresentation::Bf16Only, capacity, 1 << 20, 1 << 21, 188, 0,
            cuteafd_core::MlaCacheFormat::Bf16).is_err(), "the ML4 draft records are the FP8 656-byte ones");
        Ok(())
    }

    #[test]
    fn other_drafter_variants_are_refused() {
        for (path, value) in [("/mtp/dspark/head_type", "vanilla"), ("/mtp/dspark/draft_input", "shared_mask"),
            ("/mtp/ragged_attention", "2048,1024,2048"), ("/mtp/target_activation_layers", "1,72")] {
            let mut params = official_params();
            *params.pointer_mut(path).unwrap() = Value::from(value);
            assert!(Ml4DsparkConfig::from_params(&params, 72).is_err(), "{path}={value}");
        }
        let mut params = official_params();
        params["mtp"]["dspark"]["qk_norm"] = Value::from(true);
        assert!(Ml4DsparkConfig::from_params(&params, 72).is_err());
        assert!(Ml4DsparkConfig::from_params(&serde_json::json!({}), 72).is_err());
    }
}
